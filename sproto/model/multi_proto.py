import logging
import pytorch_lightning as pl
import torch
import torchmetrics
import transformers

from pytorch_lightning.loggers import TensorBoardLogger

from sproto.hf import SprotoModel, SprotoConfig
from sproto.metrics import metrics

logger = logging.getLogger()


class MultiProtoModule(pl.LightningModule):
    logger: TensorBoardLogger

    def __init__(
        self,
        pretrained_model,
        num_classes,
        label_order_path=None,  # kept for compatibility, not used here
        use_sigmoid=False,
        lr_features=2e-6,
        lr_others=2e-4,
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
        eval_buckets=None,
        seed=7,
        num_prototypes_per_class=1,
        batch_size=10,
    ):
        super().__init__()

        assert use_attention != use_global_attention, (
            "use_attention and use_global_attention cannot both be True"
        )

        pl.seed_everything(seed)

        self.batch_size = batch_size
        self.loss_name = loss
        self.use_sigmoid = use_sigmoid

        self.lr_features = lr_features
        self.lr_others = lr_others
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps

        self.num_classes = num_classes
        self.eval_buckets = eval_buckets

        # -----------------------------
        # Hugging Face model
        # -----------------------------
        config = SprotoConfig(
            pretrained_model_name=pretrained_model,
            num_classes=num_classes,
            num_prototypes_per_class=num_prototypes_per_class,
            use_attention=use_attention,
            use_global_attention=use_global_attention,
            dot_product=dot_product,
            normalize=normalize,
            final_layer=final_layer,
            reduce_hidden_size=reduce_hidden_size,
        )

        self.model = SprotoModel(config)

        # Optional BERT freezing
        if lr_features == 0:
            for param in self.model.core.bert.parameters():
                param.requires_grad = False

        # -----------------------------
        # Metrics
        # -----------------------------
        self.train_metrics = self._setup_metrics()
        self.all_metrics = self.train_metrics

        self.save_hyperparameters()
        logger.info("MultiProtoModule initialized")

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    def _setup_metrics(self):
        f1 = torchmetrics.F1Score(task="multilabel", num_labels=self.num_classes, threshold=0.5)

        auroc_micro = metrics.FilteredAUROC(
            num_classes=self.num_classes,
            compute_on_step=False,
            average="micro",
        )

        auroc_macro = metrics.FilteredAUROC(
            num_classes=self.num_classes,
            compute_on_step=False,
            average="macro",
        )

        average_precision = torchmetrics.classification.MultilabelAveragePrecision(
            num_labels=self.num_classes,
            compute_on_step=False,
            average="macro",
        )

        return {
            "f1": f1,
            "auroc_micro": auroc_micro,
            "auroc_macro": auroc_macro,
            "average_precision": average_precision,
        }

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr_others,
        )

        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        return [optimizer], [scheduler]

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    def training_step(self, batch, batch_idx):
        targets = torch.tensor(batch["targets"], device=self.device)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_masks"],
            token_type_ids=batch.get("token_type_ids"),
            labels=targets,
        )

        loss = outputs.loss
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        return loss

    # -------------------------------------------------
    # Validation
    # -------------------------------------------------
    def validation_step(self, batch, batch_idx):
        targets = torch.tensor(batch["targets"], device=self.device)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_masks"],
            token_type_ids=batch.get("token_type_ids"),
        )

        preds = torch.sigmoid(outputs.logits)

        for metric in self.train_metrics.values():
            metric(preds, targets)

    def validation_epoch_end(self, outputs):
        for name, metric in self.train_metrics.items():
            self.log(
                f"val/{name}",
                metric.compute(),
                batch_size=self.batch_size,
            )
            metric.reset()

    # -------------------------------------------------
    # Testing
    # -------------------------------------------------
    def test_step(self, batch, batch_idx):
        targets = torch.tensor(batch["targets"], device=self.device)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_masks"],
            token_type_ids=batch.get("token_type_ids"),
        )

        preds = torch.sigmoid(outputs.logits)

        for metric in self.all_metrics.values():
            metric(preds, targets)

        return preds, targets

    def test_epoch_end(self, outputs):
        for name, metric in self.all_metrics.items():
            value = metric.compute()
            self.log(
                f"test/{name}",
                value,
                batch_size=self.batch_size,
            )
            metric.reset()
