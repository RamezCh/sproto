from transformers import AutoConfig, AutoModel

from sproto.hf.configuration_sproto import SprotoConfig
from sproto.hf.modeling_sproto import SprotoModel

AutoConfig.register("sproto", SprotoConfig)
AutoModel.register(SprotoConfig, SprotoModel)

__all__ = ["SprotoConfig", "SprotoModel"]
