import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from sproto.utils.utils import attention_mask_from_tokens


class SprotoCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = AutoModel.from_pretrained(
            config.pretrained_model_name
        )

        self.hidden_size = self.bert.config.hidden_size
        self.num_classes = config.num_classes
        self.num_prototypes_per_class = config.num_prototypes_per_class

        self.use_attention = config.use_attention
        self.use_global_attention = config.use_global_attention
        self.dot_product = config.dot_product
        self.normalize = config.normalize
        self.final_layer = config.final_layer

        if config.reduce_hidden_size is not None:
            self.linear = nn.Linear(
                self.hidden_size,
                config.reduce_hidden_size
            )
            self.hidden_size = config.reduce_hidden_size
        else:
            self.linear = None

        self.prototype_vectors = nn.Parameter(
            torch.randn(
                self.num_classes * self.num_prototypes_per_class,
                self.hidden_size
            )
        )

        self.attention_vectors = nn.Parameter(
            torch.randn(
                self.num_classes * self.num_prototypes_per_class,
                self.hidden_size
            )
        )

        self.pairwise_dist = nn.PairwiseDistance(p=2)

        if self.final_layer:
            self.final_linear = nn.Parameter(
                torch.eye(self.num_classes).repeat(
                    self.num_prototypes_per_class, 1
                )
            )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        token_vectors = outputs.last_hidden_state

        if self.linear is not None:
            token_vectors = self.linear(token_vectors)

        if self.normalize is not None:
            token_vectors = F.normalize(
                token_vectors, p=2, dim=self.normalize
            )

        pooled = token_vectors.mean(dim=1)

        scores = -torch.cdist(
            pooled, self.prototype_vectors
        )

        if self.final_layer:
            logits = torch.matmul(scores, self.final_linear)
        else:
            logits = scores.view(
                pooled.size(0),
                self.num_prototypes_per_class,
                self.num_classes
            ).max(dim=1).values

        return logits
