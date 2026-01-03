from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification

from .configuration_sproto import SprotoConfig
from .modeling_sproto import SprotoModel

AutoConfig.register("sproto", SprotoConfig)
AutoModel.register(SprotoConfig, SprotoModel)
AutoModelForSequenceClassification.register(SprotoConfig, SprotoModel)

__all__ = ["SprotoConfig", "SprotoModel"]
