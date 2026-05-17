from .configuration_sproto import SprotoConfig
from .modeling_sproto import SprotoModel, SprotoOutput
from transformers import AutoConfig, AutoModel

AutoConfig.register("sproto", SprotoConfig)
AutoModel.register(SprotoConfig, SprotoModel)

__all__ = ["SprotoConfig", "SprotoModel", "SprotoOutput"]