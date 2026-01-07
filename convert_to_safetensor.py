from sproto.model.multi_proto import MultiProtoModule
from modeling_sproto import SprotoModel
from configuration_sproto import SprotoConfig
import os
from safetensors.torch import save_file

ckpt_path = "ckpt-epoch=573.ckpt"

# 1. Load Lightning model
lightning_model = MultiProtoModule.load_from_checkpoint(
    ckpt_path,
    strict=True,
    num_prototypes_per_class=5,
)
lightning_model.eval()

# 2. Build HF config from Lightning model
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

# 3. Instantiate HF model
hf_model = SprotoModel(config)

# 4. Copy weights
hf_model.load_state_dict(
    lightning_model.state_dict(),
    strict=True
)

# 5. Save in HF format - CUSTOM IMPLEMENTATION
save_directory = "output_dir"
os.makedirs(save_directory, exist_ok=True)

# Save the config
hf_model.config.save_pretrained(save_directory)

# Save the model weights as safetensors
model_weights = hf_model.state_dict()
save_file(model_weights, os.path.join(save_directory, "sproto.safetensors"))

print(f"Model successfully saved to {save_directory}")