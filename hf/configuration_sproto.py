from transformers.configuration_utils import PretrainedConfig

class SprotoConfig(PretrainedConfig):
    model_type = "sproto"

    def __init__(
        self,
        pretrained_model=None,
        num_classes=None,
        label_order_path=None,
        use_sigmoid=False,
        use_cuda=True,
        lr_prototypes=5e-2,
        lr_features=2e-6,
        lr_others=2e-2,
        num_training_steps=5000,
        num_warmup_steps=1000,
        loss="BCE",
        save_dir="output",
        use_attention=True,
        use_global_attention=False,
        dot_product=False,
        normalize=None,
        final_layer=False,
        reduce_hidden_size=None,
        use_prototype_loss=False,
        prototype_vector_path=None,
        attention_vector_path=None,
        eval_buckets=None,
        seed=7,
        num_prototypes_per_class=1,
        batch_size=10,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.pretrained_model = pretrained_model
        self.num_classes = num_classes
        self.label_order_path = label_order_path
        self.use_sigmoid = use_sigmoid
        self.use_cuda = use_cuda
        self.lr_prototypes = lr_prototypes
        self.lr_features = lr_features
        self.lr_others = lr_others
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.loss = loss
        self.save_dir = save_dir
        self.use_attention = use_attention
        self.use_global_attention = use_global_attention
        self.dot_product = dot_product
        self.normalize = normalize
        self.final_layer = final_layer
        self.reduce_hidden_size = reduce_hidden_size
        self.use_prototype_loss = use_prototype_loss
        self.prototype_vector_path = prototype_vector_path
        self.attention_vector_path = attention_vector_path
        self.eval_buckets = eval_buckets
        self.seed = seed
        self.num_prototypes_per_class = num_prototypes_per_class
        self.batch_size = batch_size