import os
import sys
import torch

# Go up one directory from current file
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sproto.model.multi_proto import MultiProtoModule
from modeling_sproto import SprotoModel
from configuration_sproto import SprotoConfig
from safetensors.torch import save_file

ckpt_path = "ckpt-epoch=573.ckpt"

# 1. Load Lightning model
lightning_model = MultiProtoModule.load_from_checkpoint(
    ckpt_path,
    strict=True,
    num_prototypes_per_class=5,
)
lightning_model.eval()

# 2. Load the checkpoint directly to get the state_dict
checkpoint = torch.load(ckpt_path, map_location='cpu')
state_dict = checkpoint['state_dict']

# 3. Remove the 'module.' prefix from the checkpoint
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_key = key[7:]  # Remove 'module.'
    else:
        new_key = key
    new_state_dict[new_key] = value

# 4. Build HF config from Lightning model
config = SprotoConfig(
    pretrained_model=lightning_model.bert.name_or_path,
    num_classes=lightning_model.num_classes,
    label_order_path=lightning_model.label_order_path,
    use_sigmoid=lightning_model.use_sigmoid,
    lr_prototypes=lightning_model.lr_prototypes,
    lr_features=lightning_model.lr_features,
    lr_others=lightning_model.lr_others,
    num_training_steps=lightning_model.num_training_steps,
    num_warmup_steps=lightning_model.num_warmup_steps,
    loss=lightning_model.loss,
    save_dir=lightning_model.save_dir,
    use_attention=lightning_model.use_attention,
    use_global_attention=lightning_model.use_global_attention,
    dot_product=lightning_model.dot_product,
    normalize=lightning_model.normalize,
    final_layer=lightning_model.final_layer,
    reduce_hidden_size=(
        lightning_model.hidden_size
        if lightning_model.reduce_hidden_size
        else None
    ),
    use_prototype_loss=lightning_model.use_prototype_loss,
    prototype_vector_path=None,
    attention_vector_path=None,
    eval_buckets=lightning_model.eval_buckets,
    seed=lightning_model.hparams.seed,
    num_prototypes_per_class=5,
    batch_size=lightning_model.batch_size,
)

config.auto_map = {
    "AutoConfig": "configuration_sproto.SprotoConfig",
    "AutoModel": "modeling_sproto.SprotoModel",
}

# 5. Instantiate HF model
hf_model = SprotoModel(config)

# 6. Load the cleaned state_dict into HF model
hf_model.module.load_state_dict(new_state_dict, strict=True)

# 7. Save in HF format - CUSTOM IMPLEMENTATION
save_directory = "output_dir"
os.makedirs(save_directory, exist_ok=True)

# Save the config
hf_model.config.save_pretrained(save_directory)

# Save the model weights as safetensors
model_weights = hf_model.module.state_dict()
save_file(model_weights, os.path.join(save_directory, "model.safetensors"), metadata={"format": "pt"})

print(f"Model successfully saved to {save_directory}")