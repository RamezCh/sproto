import os
import tempfile
import pytest
import torch
import numpy as np
from unittest.mock import patch

from transformers import BertConfig, BertModel

from hf.configuration_sproto import SprotoConfig
from hf.modeling_sproto import SprotoModel
from sproto.model.multi_proto import MultiProtoModule


@pytest.fixture(scope="session")
def dummy_bert_path(tmp_path_factory):
    """
    Creates a minimal local Hugging Face model
    so we don't need internet interactions or massive dataset loading.
    """
    config = BertConfig(
        vocab_size=1000,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=16,
        max_position_embeddings=128,
    )
    model = BertModel(config)
    path = tmp_path_factory.mktemp("dummy_bert")
    model.save_pretrained(str(path))
    return str(path)


@pytest.fixture
def temp_label_order_file(tmp_path):
    temp_path = tmp_path / "label_order.txt"
    temp_path.write_text("label1\tlabel2\tlabel3")
    return str(temp_path)


@pytest.fixture
def dummy_config(dummy_bert_path, temp_label_order_file):
    return SprotoConfig(
        pretrained_model=dummy_bert_path,
        num_classes=3,
        label_order_path=temp_label_order_file,
        use_cuda=False,
        reduce_hidden_size=None,
        num_prototypes_per_class=1,
        eval_buckets=None,
    )


@pytest.fixture
def sample_batch():
    batch_size = 4
    seq_len = 10
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_masks": torch.ones(batch_size, seq_len, dtype=torch.float),
        "token_type_ids": torch.zeros(batch_size, seq_len, dtype=torch.long),
        "targets": torch.zeros(batch_size, 3, dtype=torch.float),
        "tokens": [["token"] * seq_len for _ in range(batch_size)],
        "sample_ids": [f"id_{i}" for i in range(batch_size)]
    }


class TestMultiProtoModule:
    """Core Logic: Initialization constraints, logic dependencies, tensor shapes, and outputs"""

    def test_initialization_defaults(self, dummy_bert_path, temp_label_order_file):
        module = MultiProtoModule(
            pretrained_model=dummy_bert_path,
            num_classes=3,
            label_order_path=temp_label_order_file,
            use_cuda=False
        )
        assert module.hidden_size == 16
        assert not module.reduce_hidden_size
        assert module.prototype_vectors.shape == (3, 16)
        assert module.num_prototypes == 3
        assert module.attention_vectors.shape == (3, 16)

    def test_initialization_with_reduction(self, dummy_bert_path, temp_label_order_file):
        module = MultiProtoModule(
            pretrained_model=dummy_bert_path,
            num_classes=3,
            label_order_path=temp_label_order_file,
            use_cuda=False,
            reduce_hidden_size=8,
            num_prototypes_per_class=2
        )
        assert module.hidden_size == 8
        assert module.prototype_vectors.shape == (2, 3, 8)
        assert hasattr(module, "linear")

    @pytest.mark.parametrize("use_attention, use_global, dot_product", [
        (True, False, False),
        (True, False, True),
        (False, True, False),
    ])
    def test_forward_shapes_and_logic(
        self, dummy_bert_path, temp_label_order_file, sample_batch,
        use_attention, use_global, dot_product
    ):
        module = MultiProtoModule(
            pretrained_model=dummy_bert_path,
            num_classes=3,
            label_order_path=temp_label_order_file,
            use_cuda=False,
            use_attention=use_attention,
            use_global_attention=use_global,
            dot_product=dot_product,
            final_layer=False,
            num_prototypes_per_class=1
        )

        batch_size = sample_batch["input_ids"].shape[0]

        with patch("sproto.utils.utils.attention_mask_from_tokens", return_value=sample_batch["attention_masks"]):
            logits, max_indices, metadata = module(sample_batch)

        assert logits.shape == (batch_size, 3)

        if use_attention:
            assert metadata is not None
        else:
            assert metadata[2] is not None 

    def test_expected_numerical_outputs(self, dummy_bert_path, temp_label_order_file):
        module = MultiProtoModule(
            pretrained_model=dummy_bert_path,
            num_classes=2,
            label_order_path=temp_label_order_file,
            use_cuda=False,
            num_prototypes_per_class=2
        )

        prototype_vectors_data = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], requires_grad=True)

        module.prototype_vectors = torch.nn.Parameter(prototype_vectors_data)
        module.num_classes = 2 
        module.prototype_to_class_map = torch.tensor([0, 0, 1, 1])

        loss = module.calculate_prototype_loss()

        expected_val = 100.0 / (1.0 + np.sqrt(2))
        assert torch.isclose(loss, torch.tensor(expected_val, dtype=torch.float32))

    def test_training_step(self, dummy_bert_path, temp_label_order_file, sample_batch):
        module = MultiProtoModule(
            pretrained_model=dummy_bert_path,
            num_classes=3,
            label_order_path=temp_label_order_file,
            use_cuda=False,
            use_prototype_loss=True, 
            loss="BCE"
        )

        with patch("sproto.utils.utils.attention_mask_from_tokens", return_value=sample_batch["attention_masks"]):
            loss = module.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad


class TestSprotoConfig:
    """HF Wrapper: Defaults and Initialization properties logic behavior"""

    def test_config_defaults(self):
        config = SprotoConfig()
        assert config.model_type == "sproto"
        assert config.loss == "BCE"
        assert config.use_attention is True

    def test_config_custom_values(self, temp_label_order_file):
        config = SprotoConfig(
            pretrained_model="mypath",
            num_classes=10,
            label_order_path=temp_label_order_file,
            lr_features=1e-5
        )
        assert config.pretrained_model == "mypath"
        assert config.num_classes == 10
        assert config.lr_features == 1e-5


class TestSprotoModel:
    """HF Wrapper: Pretained consistency output formats compatibility and loading handling"""

    def test_model_initialization(self, dummy_config):
        model = SprotoModel(dummy_config)
        assert isinstance(model.module, MultiProtoModule)
        assert model.config.num_classes == 3

    def test_model_forward(self, dummy_config, sample_batch):
        model = SprotoModel(dummy_config)

        with patch("sproto.utils.utils.attention_mask_from_tokens", return_value=sample_batch["attention_masks"]):
            outputs = model(
                input_ids=sample_batch["input_ids"],
                attention_mask=sample_batch["attention_masks"],
                token_type_ids=sample_batch["token_type_ids"],
                targets=sample_batch["targets"],
                tokens=sample_batch["tokens"],
                sample_ids=sample_batch["sample_ids"]
            )

        assert "logits" in outputs
        assert "max_indices" in outputs
        assert "metadata" in outputs
        assert outputs["logits"].shape == (sample_batch["input_ids"].shape[0], 3)

    def test_from_pretrained_compatibility(self, dummy_config, tmp_path):
        model = SprotoModel(dummy_config)
        save_dir = tmp_path / "saved_sproto_model"
        
        # Inject config to avoid transformers save_pretrained attribute error on LightningModule
        model.module.config = dummy_config

        model.save_pretrained(str(save_dir))
        assert (save_dir / "config.json").exists()

        loaded_model = SprotoModel.from_pretrained(str(save_dir))

        assert isinstance(loaded_model, SprotoModel)
        assert loaded_model.config.num_classes == dummy_config.num_classes
        assert loaded_model.config.pretrained_model == dummy_config.pretrained_model

        batch_size = 2
        seq_len = 5
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float)

        with patch("sproto.utils.utils.attention_mask_from_tokens", return_value=attention_mask):
            outputs = loaded_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=torch.zeros((batch_size, seq_len), dtype=torch.long),
                targets=torch.zeros(batch_size, loaded_model.config.num_classes),
                tokens=[["t"] * seq_len] * batch_size,
                sample_ids=["id1", "id2"]
            )
        assert outputs["logits"].shape == (batch_size, dummy_config.num_classes)
