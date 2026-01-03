import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .configuration_sproto import SprotoConfig
from sproto.model.sproto_core import SprotoCore


class SprotoModel(PreTrainedModel):
    config_class = SprotoConfig

    def __init__(self, config):
        super().__init__(config)
        self.core = SprotoCore(config)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        return_dict=True,
    ):
        logits = self.core(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.float()
            )

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
