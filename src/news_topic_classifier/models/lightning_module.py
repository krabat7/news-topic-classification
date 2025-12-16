from __future__ import annotations

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification import Accuracy, MulticlassF1Score

from .baseline import MeanPoolClassifier


class NewsClassifierModule(LightningModule):
    def __init__(
        self, vocab_size: int, embedding_dim: int, num_classes: int, dropout: float, lr: float
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = MeanPoolClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])
        preds = torch.argmax(logits, dim=-1)

        self.val_acc.update(preds, batch["labels"])
        self.val_f1_macro.update(preds, batch["labels"])

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        f1 = self.val_f1_macro.compute()
        self.log("val_accuracy", acc, prog_bar=True)
        self.log("val_macro_f1", f1, prog_bar=True)
        self.val_acc.reset()
        self.val_f1_macro.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
