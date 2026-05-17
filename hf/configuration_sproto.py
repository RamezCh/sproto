from transformers import PretrainedConfig


class SprotoConfig(PretrainedConfig):
    model_type = "sproto"

    def __init__(
        self,
        pretrained_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        num_classes=55,
        label_order_path=None,
        use_attention=True,
        use_global_attention=False,
        dot_product=False,
        normalize=None,
        final_layer=False,
        reduce_hidden_size=None,
        num_prototypes_per_class=1,
        loss="BCE",
        use_prototype_loss=False,
        use_sigmoid=False,
        seed=7,
        vocab_size=28996,
        hidden_size=768,
        max_position_embeddings=512,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
        **kwargs,
    ):
        # Bug fix: the checkpoint serialises the num_prototypes_per_class buffer as a list
        # (one float per class). MultiProtoModule expects a scalar int for the uniform case.
        # Collapse the list → scalar, asserting all values are identical.
        if isinstance(num_prototypes_per_class, (list, tuple)):
            unique_vals = set(int(v) for v in num_prototypes_per_class)
            assert len(unique_vals) == 1, (
                "Non-uniform num_prototypes_per_class cannot be represented as a scalar "
                "config value. Got: {}".format(unique_vals)
            )
            num_prototypes_per_class = unique_vals.pop()

        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
        self.label_order_path = label_order_path
        self.use_attention = use_attention
        self.use_global_attention = use_global_attention
        self.dot_product = dot_product
        self.normalize = normalize
        self.final_layer = final_layer
        self.reduce_hidden_size = reduce_hidden_size
        self.num_prototypes_per_class = num_prototypes_per_class
        self.loss = loss
        self.use_prototype_loss = use_prototype_loss
        self.use_sigmoid = use_sigmoid
        self.seed = seed

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob

        super().__init__(**kwargs)