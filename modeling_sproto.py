from transformers import PreTrainedModel
from sproto.model.multi_proto import MultiProtoModule
from configuration_sproto import SprotoConfig

class SprotoModel(PreTrainedModel):
    config_class = SprotoConfig
    base_model_prefix = "sproto"

    def __init__(self, config: SprotoConfig):
        super().__init__(config)

        self.module = MultiProtoModule(
            pretrained_model=config.pretrained_model,
            num_classes=config.num_classes,
            label_order_path=config.label_order_path,
            use_sigmoid=config.use_sigmoid,
            use_cuda=config.use_cuda,
            lr_prototypes=config.lr_prototypes,
            lr_features=config.lr_features,
            lr_others=config.lr_others,
            num_training_steps=config.num_training_steps,
            num_warmup_steps=config.num_warmup_steps,
            loss=config.loss,
            save_dir=config.save_dir,
            use_attention=config.use_attention,
            use_global_attention=config.use_global_attention,
            dot_product=config.dot_product,
            normalize=config.normalize,
            final_layer=config.final_layer,
            reduce_hidden_size=config.reduce_hidden_size,
            use_prototype_loss=config.use_prototype_loss,
            prototype_vector_path=config.prototype_vector_path,
            attention_vector_path=config.attention_vector_path,
            eval_buckets=config.eval_buckets,
            seed=config.seed,
            num_prototypes_per_class=config.num_prototypes_per_class,
            batch_size=config.batch_size,
        )
        
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (MultiProtoModule)):
            # MultiProtoModule handles its own initialization or is loaded from checkpoint
            return
        # Add other initializations if standard layers are used directly in SprotoModel
        pass

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        targets=None,
        tokens=None,
        sample_ids=None,
        **kwargs,
    ):
        if tokens is None and input_ids is not None:
             # Create dummy tokens to prevent crash in utils.attention_mask_from_tokens
             # The internal model expects a list of lists of strings
             tokens = [[] for _ in range(input_ids.shape[0])]

        batch = {
            "input_ids": input_ids,
            "attention_masks": attention_mask,
            "token_type_ids": token_type_ids,
            "targets": targets,
            "tokens": tokens,
            "sample_ids": sample_ids,
        }

        logits, max_indices, metadata = self.module(batch)

        return {
            "logits": logits,
            "max_indices": max_indices,
            "metadata": metadata,
        }