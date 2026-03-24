import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from sproto.model.multi_proto import MultiProtoModule


@pytest.fixture
def mock_bert():
    with patch('sproto.model.multi_proto.AutoModel.from_pretrained') as mock:
        mock_bert = MagicMock()
        mock_bert.config.hidden_size = 768
        mock.return_value = mock_bert
        yield mock_bert


@pytest.fixture
def minimal_kwargs():
    return {
        'pretrained_model': 'bert-base-uncased',
        'num_classes': 5,
        'label_order_path': None,
    }


class TestMultiProtoModuleInit:
    def test_init_basic(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        assert module.num_classes == 5
        assert module.hidden_size == 768
        assert module.prototype_vectors is not None
        assert module.attention_vectors is not None

    def test_init_with_custom_hidden_size(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'reduce_hidden_size': 256}
        module = MultiProtoModule(**kwargs)
        assert module.hidden_size == 256
        assert module.reduce_hidden_size is True
        assert hasattr(module, 'linear')

    def test_init_with_freeze_bert(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'lr_features': 0}
        module = MultiProtoModule(**kwargs)
        for param in module.bert.parameters():
            assert not param.requires_grad

    def test_init_with_sigmoid(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'use_sigmoid': True}
        module = MultiProtoModule(**kwargs)
        assert module.use_sigmoid is True

    def test_init_with_multiple_prototypes_per_class(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'num_prototypes_per_class': 3}
        module = MultiProtoModule(**kwargs)
        assert module.num_prototypes_per_class_tmp == 3


class TestMultiProtoModuleArchitecture:
    def test_prototype_to_class_mapping(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        mapping = module.prototype_to_class_map
        assert mapping.shape[0] == module.num_prototypes
        assert mapping.device == module.device

    def test_build_final_layer(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'final_layer': True}
        module = MultiProtoModule(**kwargs)
        assert hasattr(module, 'final_linear')
        assert module.final_linear.shape == (module.num_prototypes, module.num_classes)


class TestMultiProtoModuleMetrics:
    def test_setup_metrics(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        metrics = module.setup_metrics()
        assert 'auroc_micro' in metrics
        assert 'auroc_macro' in metrics
        assert 'average_precision' in metrics
        assert hasattr(module, 'f1')

    def test_metrics_have_correct_num_classes(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        assert module.auroc_micro.num_classes == 5
        assert module.auroc_macro.num_classes == 5
        assert module.average_precision.num_labels == 5


class TestMultiProtoModuleOptimizers:
    def test_configure_optimizers(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        optimizers, schedulers = module.configure_optimizers()
        assert len(optimizers) == 1
        assert len(schedulers) == 1
        assert isinstance(optimizers[0], torch.optim.AdamW)

    def test_optimizer_has_correct_parameter_groups(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        optimizers, _ = module.configure_optimizers()
        param_groups = optimizers[0].param_groups
        assert len(param_groups) >= 3

    def test_configure_optimizers_with_final_layer(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'final_layer': True}
        module = MultiProtoModule(**kwargs)
        optimizers, _ = module.configure_optimizers()
        assert len(optimizers[0].param_groups) >= 4


class TestMultiProtoModuleForward:
    def test_forward_returns_correct_shapes(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        module.eval()

        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_masks': torch.ones(2, 10),
            'token_type_ids': torch.zeros(2, 10),
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.zeros(2, 5),
            'sample_ids': [0, 1],
        }

        with torch.no_grad():
            logits, max_indices, metadata = module(batch)

        assert logits.shape == (2, 5)
        assert max_indices is None or max_indices.shape[0] == 2

    def test_forward_with_reduce_hidden_size(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'reduce_hidden_size': 256}
        module = MultiProtoModule(**kwargs)
        module.eval()

        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_masks': torch.ones(2, 10),
            'token_type_ids': torch.zeros(2, 10),
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.zeros(2, 5),
            'sample_ids': [0, 1],
        }

        with torch.no_grad():
            logits, max_indices, metadata = module(batch)

        assert logits.shape == (2, 5)

    def test_forward_normalize(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'normalize': 2}
        module = MultiProtoModule(**kwargs)
        module.eval()

        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_masks': torch.ones(2, 10),
            'token_type_ids': torch.zeros(2, 10),
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.zeros(2, 5),
            'sample_ids': [0, 1],
        }

        with torch.no_grad():
            logits, max_indices, metadata = module(batch)

        assert logits.shape == (2, 5)


class TestMultiProtoModuleComputeMaxIndices:
    def test_compute_max_indices_basic(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        module.eval()

        token_vectors = torch.randn(2, 10, 768)
        attention_mask = torch.ones(2, 10)
        batch = {
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.zeros(2, 5),
            'sample_ids': [0, 1],
        }

        with torch.no_grad():
            logits, max_indices, metadata = module.compute_max_indices(
                attention_mask, batch, token_vectors
            )

        assert logits.shape == (2, 5)

    def test_compute_max_indices_without_attention(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'use_attention': False, 'use_global_attention': False}
        module = MultiProtoModule(**kwargs)
        module.eval()

        token_vectors = torch.randn(2, 10, 768)
        attention_mask = torch.ones(2, 10)
        batch = {
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.zeros(2, 5),
            'sample_ids': [0, 1],
        }

        with torch.no_grad():
            logits, max_indices, metadata = module.compute_max_indices(
                attention_mask, batch, token_vectors
            )

        assert logits.shape == (2, 5)


class TestMultiProtoModuleTokenAttention:
    def test_calculate_token_class_attention(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        module.eval()

        batch_samples = torch.randn(2, 10, 768)
        class_attention_vectors = torch.randn(5, 768)
        mask = torch.ones(2, 10)

        with torch.no_grad():
            weighted, attention = module.calculate_token_class_attention(
                batch_samples, class_attention_vectors, 1, mask
            )

        assert weighted.shape == (2, 5, 768)
        assert attention.shape == (2, 5, 10)


class TestMultiProtoModuleGetLogits:
    def test_get_logits_per_class_without_final_layer(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        score = torch.randn(2, 5)
        logits, indices = module.get_logits_per_class(score)
        assert logits.shape == (2, 5)

    def test_get_logits_per_class_with_final_layer(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'final_layer': True}
        module = MultiProtoModule(**kwargs)
        score = torch.randn(2, module.num_prototypes)
        logits, indices = module.get_logits_per_class(score)
        assert logits.shape == (2, 5)


class TestMultiProtoModulePrototypeLoss:
    def test_calculate_prototype_loss(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        loss = module.calculate_prototype_loss()
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


class TestMultiProtoModuleTraining:
    def test_training_step(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        module.eval()

        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_masks': torch.ones(2, 10),
            'token_type_ids': torch.zeros(2, 10),
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.randint(0, 2, (2, 5)),
            'sample_ids': [0, 1],
        }

        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)

    def test_training_step_with_prototype_loss(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'use_prototype_loss': True}
        module = MultiProtoModule(**kwargs)
        module.eval()

        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_masks': torch.ones(2, 10),
            'token_type_ids': torch.zeros(2, 10),
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.randint(0, 2, (2, 5)),
            'sample_ids': [0, 1],
        }

        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)


class TestMultiProtoModuleValidation:
    def test_validation_step(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        module.eval()

        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_masks': torch.ones(2, 10),
            'token_type_ids': torch.zeros(2, 10),
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.randint(0, 2, (2, 5)),
            'sample_ids': [0, 1],
        }

        module.validation_step(batch, 0)


class TestMultiProtoModuleLoadVectors:
    def test_load_prototype_vectors_with_temp_file(self, mock_bert, minimal_kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            label_order_path = os.path.join(tmpdir, 'labels.txt')
            proto_path = os.path.join(tmpdir, 'protos.pt')

            with open(label_order_path, 'w') as f:
                f.write('class1\tclass2\tclass3\tclass4\tclass5')

            proto_dict = {
                'class1': [torch.randn(128)],
                'class2': [torch.randn(128)],
                'class3': [torch.randn(128)],
                'class4': [torch.randn(128)],
                'class5': [torch.randn(128)],
            }
            torch.save(proto_dict, proto_path)

            kwargs = {
                'pretrained_model': 'bert-base-uncased',
                'num_classes': 5,
                'label_order_path': label_order_path,
                'prototype_vector_path': proto_path,
            }

            module = MultiProtoModule(**kwargs)
            assert module.prototype_vectors is not None

    def test_build_prototype_to_class_mapping(self, mock_bert, minimal_kwargs):
        module = MultiProtoModule(**minimal_kwargs)
        num_prototypes = torch.tensor([2, 2, 1])
        mapping = module.build_prototype_to_class_mapping(num_prototypes)
        assert len(mapping) == 5
        assert mapping[0] == 0
        assert mapping[1] == 0
        assert mapping[2] == 1
        assert mapping[3] == 1
        assert mapping[4] == 2


class TestMultiProtoModuleEdgeCases:
    def test_attention_and_global_attention_mutual_exclusion(self, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'use_attention': True, 'use_global_attention': True}
        with pytest.raises(AssertionError):
            MultiProtoModule(**kwargs)

    def test_dot_product_mode(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'dot_product': True}
        module = MultiProtoModule(**kwargs)
        module.eval()

        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_masks': torch.ones(2, 10),
            'token_type_ids': torch.zeros(2, 10),
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.zeros(2, 5),
            'sample_ids': [0, 1],
        }

        with torch.no_grad():
            logits, max_indices, metadata = module(batch)

        assert logits.shape == (2, 5)

    def test_multiple_prototypes_per_class(self, mock_bert, minimal_kwargs):
        kwargs = {**minimal_kwargs, 'num_prototypes_per_class': 3}
        module = MultiProtoModule(**kwargs)
        module.eval()

        batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_masks': torch.ones(2, 10),
            'token_type_ids': torch.zeros(2, 10),
            'tokens': [['[CLS]', '[SEP]'] * 5] * 2,
            'targets': torch.zeros(2, 5),
            'sample_ids': [0, 1],
        }

        with torch.no_grad():
            logits, max_indices, metadata = module(batch)

        assert logits.shape == (2, 5)