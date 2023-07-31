from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import torchmetrics
from generalization.utils import build_experiment, get_num_cpus
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

        self.epoch_state = {"batch_logits": [], "batch_indices": [], "batch_losses": []}
        self.scores_df = pd.DataFrame(columns=["epoch", "sample_id", "score"])

        self.save_hyperparameters(self.hparams)

    def forward(self, x):
        out = self.net(x)
        return out

    def step(self, batch, batch_idx):
        logits = self(batch[0])
        loss_per_sample = F.cross_entropy(logits, batch[1], reduction="none")
        if self.hparams["gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

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
        self.log(
            "valid/loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        self.step_metrics(logits=logits, y=y, mode="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss_per_sample, logits, y = self.step(batch, batch_idx)
        loss = loss_per_sample.mean()

        self.log(
            "test/loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=True
        )
        self.step_metrics(logits=logits, y=y, mode="val")
        return loss

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """
        Tracks per sample indices, logits and their losses for the current epoch.
        """
        _, _, indices = batch

        logits = outputs["logits"].tolist()
        batch_losses = outputs["loss_per_sample"].cpu().detach().numpy()

        self.epoch_state["batch_indices"].extend(indices.tolist())
        self.epoch_state["batch_logits"].extend(logits)
        self.epoch_state["batch_losses"].extend(batch_losses)

        return None

    def on_train_epoch_end(self) -> None:
        """
        Computes and logs per sample scores for the current epoch.
        """

        import pandas as pd
        from scores import compute_scores

        data = self.trainer.train_dataloader.dataset

        logits = torch.as_tensor(self.epoch_state["batch_logits"]).to(
            next(self.parameters()).device
        )
        y = torch.as_tensor(data.targets).to(next(self.parameters()).device)
        scores = compute_scores(logits, y).cpu().detach().numpy()

        sample_losses = self.epoch_state["batch_losses"]
        sample_indices = self.epoch_state["batch_indices"]
        epoch_column = self.current_epoch * np.ones(len(sample_indices))

        df = pd.DataFrame(
            {
                "epoch": epoch_column,
                "sample_id": sample_indices,
                "score": scores,
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

        # # reset epoch state
        self.epoch_state = {k: [] for k in self.epoch_state.keys()}

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
                "train/acc",
                self.train_acc,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
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
                logger=True,
            )
            self.log(
                "valid/top5_acc",
                self.valid_top5_acc.compute(),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
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
                logger=True,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.hparams["lr"],
            momentum=self.hparams["momentum"],
            weight_decay=self.hparams["weight_decay"],
        )

        if self.hparams["lr_scheduler"]:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
            return [optimizer], [scheduler]

        return [optimizer]


class LitDataModule(L.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

    def setup(self, stage=None):
        corrupt_prob = self.hparams["corrupt_prob"]
        corrupt_name = self.hparams["corrupt_name"]
        batch_size = self.hparams["batch_size"]

        experiment_data = build_experiment(
            corrupt_prob=corrupt_prob,
            corrupt_name=corrupt_name,
            batch_size=batch_size,
        )[corrupt_name]

        self.train_set = experiment_data["train_set"]
        self.val_set = experiment_data["val_set"]
        self.test_set = experiment_data["test_set"]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=get_num_cpus(),
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.hparams["batch_size"] * 2,
            shuffle=False,
            num_workers=get_num_cpus(),
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.hparams["batch_size"] * 2,
            shuffle=False,
            num_workers=get_num_cpus(),
            pin_memory=True,
        )

    def __repr__(self):
        return (
            "DataModule:\n"
            + str(self.train_set.__repr__())
            + "\n"
            + "Val: "
            + str(self.val_set.__repr__())
        )
