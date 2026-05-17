import torch
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from sproto.model.multi_proto import MultiProtoModule

from .configuration_sproto import SprotoConfig


@dataclass
class SprotoOutput(ModelOutput):
    logits: torch.Tensor = None
    max_indices: torch.Tensor = None
    metadata: dict = None

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)


class SprotoModel(PreTrainedModel):
    config_class = SprotoConfig
    base_model_prefix = "sproto"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # HF's from_pretrained (on newer PyTorch / Transformers) can set a meta-device
        # context during weight loading. MultiProtoModule.__init__ internally calls
        # AutoModel.from_pretrained() for the BERT backbone, which conflicts with that
        # context. We escape it by forcing CPU initialization when on torch >= 2.0
        # (torch.device context manager was added in 2.0); on older torch the conflict
        # does not occur and we construct normally.
        _torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2] if x.isdigit())
        if _torch_version >= (2, 0):
            with torch.device("cpu"):
                self.module = MultiProtoModule(
                    pretrained_model=config.pretrained_model,
                    num_classes=config.num_classes,
                    label_order_path=config.label_order_path,
                    use_attention=config.use_attention,
                    use_global_attention=config.use_global_attention,
                    dot_product=config.dot_product,
                    normalize=config.normalize,
                    final_layer=config.final_layer,
                    reduce_hidden_size=config.reduce_hidden_size,
                    num_prototypes_per_class=config.num_prototypes_per_class,
                    loss=config.loss,
                    use_prototype_loss=config.use_prototype_loss,
                    use_sigmoid=config.use_sigmoid,
                    seed=config.seed,
                    # use_cuda=False: HF handles device placement via .to(device);
                    # manual .cuda() calls in the base model are bypassed here.
                    use_cuda=False,
                )
        else:
            self.module = MultiProtoModule(
                pretrained_model=config.pretrained_model,
                num_classes=config.num_classes,
                label_order_path=config.label_order_path,
                use_attention=config.use_attention,
                use_global_attention=config.use_global_attention,
                dot_product=config.dot_product,
                normalize=config.normalize,
                final_layer=config.final_layer,
                reduce_hidden_size=config.reduce_hidden_size,
                num_prototypes_per_class=config.num_prototypes_per_class,
                loss=config.loss,
                use_prototype_loss=config.use_prototype_loss,
                use_sigmoid=config.use_sigmoid,
                seed=config.seed,
                use_cuda=False,
            )

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        targets=None,
        tokens=None,
        sample_ids=None,
    ):
        # tokens MUST be provided when use_attention=True.
        # attention_mask_from_tokens() relies on real token strings to zero out clinical
        # section headers ([CLS], [SEP], "chief complaint :", etc.) before computing
        # token-prototype attention. A fake default would silently produce wrong logits.
        if tokens is None and self.config.use_attention:
            raise ValueError(
                "tokens (list-of-lists of token strings per sample) must be provided "
                "when use_attention=True. Obtain them via:\n"
                "  tokenizer.convert_ids_to_tokens(input_ids[i])\n"
                "for each sample i in the batch, or pass the full batch at once with:\n"
                "  [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]"
            )

        batch = {
            "input_ids": input_ids,
            "attention_masks": attention_mask,
            "token_type_ids": token_type_ids if token_type_ids is not None else torch.zeros_like(input_ids),
            "targets": targets if targets is not None else torch.zeros(input_ids.shape[0], self.config.num_classes),
            "tokens": tokens if tokens is not None else [["[PAD]"] * input_ids.shape[1]] * input_ids.shape[0],
            "sample_ids": sample_ids if sample_ids is not None else [f"sample_{i}" for i in range(input_ids.shape[0])],
        }

        logits, max_indices, metadata = self.module(batch)

        return SprotoOutput(
            logits=logits,
            max_indices=max_indices,
            metadata=metadata,
        )

    def get_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        bert_output = self.module.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids if token_type_ids is not None else torch.zeros_like(input_ids),
        )
        return bert_output.last_hidden_state

    def _init_weights(self, module):
        pass