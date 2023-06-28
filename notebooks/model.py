from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import torchmetrics
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F


class LitModel(L.LightningModule):
    def __init__(self, net: nn.Module, hparams: dict):
        super().__init__()
        self.net = net
        self.hparams.update(hparams)

        self.lr = self.hparams["learning_rate"]
        self.n_classes = self.hparams["n_classes"]
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.valid_top5_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes, top_k=5
        )

        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.test_top5_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes, top_k=5
        )

        self.early_stop_counter = 0
        self.patience = 5
        self.best_valid_acc = 0

        self.epoch_state = {"batch_scores": [], "batch_indices": [], "batch_losses": []}
        self.scores_df = pd.DataFrame(columns=["epoch", "sample_id", "score"])

    def forward(self, x):
        out = self.net(x)
        return out

    def step(self, batch, batch_idx):
        batch = self.drop_return_index(batch)

        logits = self(batch[0])
        loss_per_sample = F.cross_entropy(logits, batch[1], reduction="none")
        return loss_per_sample, logits, batch[1]

    def training_step(self, batch, batch_idx):
        loss_per_sample, logits, y = self.step(batch, batch_idx)
        loss_per_batch = loss_per_sample.mean()
        self.log(
            "train/loss",
            loss_per_batch,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        self.step_metrics(logits=logits, y=y, mode="train")
        return {
            "loss": loss_per_batch,
            "logits": logits,
            "y": y,
            "loss_per_sample": loss_per_sample,
        }

    def validation_step(self, batch, batch_idx):
        loss_per_sample, logits, y = self.step(batch, batch_idx)
        loss = loss_per_sample.mean()
        self.log("valid/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.step_metrics(logits=logits, y=y, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss_per_sample, logits, y = self.step(batch, batch_idx)
        loss = loss_per_sample.mean()

        self.log("test/loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.step_metrics(logits=logits, y=y, mode="val")
        return loss

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        from scores import compute_batch_scores

        x, y, indices = batch

        batch_scores = compute_batch_scores(outputs, (x, y), num_classes=self.n_classes)
        batch_losses = outputs["loss_per_sample"].cpu().detach().numpy()

        self.epoch_state["batch_scores"].extend(batch_scores.tolist())
        self.epoch_state["batch_losses"].extend(batch_losses)
        self.epoch_state["batch_indices"].extend(indices.tolist())

        return None

    def on_train_epoch_end(self) -> None:
        import pandas as pd

        el2n_scores = self.epoch_state["batch_scores"]
        sample_losses = self.epoch_state["batch_losses"]
        sample_indices = self.epoch_state["batch_indices"]
        epoch_column = self.current_epoch * np.ones(len(el2n_scores))

        df = pd.DataFrame(
            {
                "epoch": epoch_column,
                "sample_id": sample_indices,
                "score": el2n_scores,
                "loss": sample_losses,
            }
        )
        try:
            self.logger.log_table(
                "train/scores",
                dataframe=df,
                step=self.current_epoch,
            )
        except AttributeError as e:
            save_to = f"{self.logger.log_dir}/{self.logger.name}_train-scores.csv"
            self.scores_df = pd.concat([self.scores_df, df])
            self.scores_df.to_csv(save_to)

        self.epoch_state = {"batch_scores": [], "batch_indices": [], "batch_losses": []}

    def on_validation_end(self) -> None:
        current_valid_acc = self.valid_acc.compute()
        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter > self.patience:
                self.early_stop_counter = 0
                # self.trainer.should_stop = True
                log_fn = (
                    self.logger.experiment.add_scalar
                    if isinstance(self.logger, TensorBoardLogger)
                    else self.logger.experiment.log
                )

                log_fn({"early_stop": self.current_epoch})

    def step_metrics(self, logits, y, mode):
        if mode == "train":
            self.train_acc.update(logits, y)
            self.log(
                "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
            )

        elif mode == "val":
            self.valid_acc.update(logits, y)
            self.valid_top5_acc.update(logits, y)

            self.log(
                "valid/acc",
                self.valid_acc.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                "valid/top5_acc",
                self.valid_top5_acc.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        elif mode == "test":
            self.test_acc.update(logits, y)
            self.test_top5_acc.update(logits, y)
            self.log("test/acc", self.test_acc.compute(), on_step=False, on_epoch=True)
            self.log(
                "test/top5_acc",
                self.test_top5_acc.compute(),
                on_step=False,
                on_epoch=True,
            )

    def drop_return_index(self, batch):
        x, y, _ = batch

        return (x, y) if self.hparams["drop_return_index"] else batch

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
        return [optimizer], [scheduler]
