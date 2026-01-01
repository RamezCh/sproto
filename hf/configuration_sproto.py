from transformers import PretrainedConfig

class SprotoConfig(PretrainedConfig):
    model_type = "sproto"

    def __init__(
        self,
        pretrained_model_name="bert-base-uncased",
        num_classes=10,
        num_prototypes_per_class=1,
        use_attention=True,
        use_global_attention=False,
        dot_product=False,
        normalize=None,
        reduce_hidden_size=None,
        final_layer=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.pretrained_model_name = pretrained_model_name
        self.num_classes = num_classes
        self.num_prototypes_per_class = num_prototypes_per_class
        self.use_attention = use_attention
        self.use_global_attention = use_global_attention
        self.dot_product = dot_product
        self.normalize = normalize
        self.reduce_hidden_size = reduce_hidden_size
        self.final_layer = final_layer
