from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from generalization.utils.data import get_num_cpus
from generalization.utils.experiment import build_experiment
from torch import nn
from torch.nn import functional as F


class Classifier(pl.LightningModule):
    def __init__(self, net: nn.Module, hparams: dict):
        super().__init__()
        self.net = net
        self.hparams.update(hparams)
        self.save_hyperparameters(ignore=["net"])
        self.lr = self.hparams["learning_rate"]
        self.n_classes = self.hparams["n_classes"]

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes, average=None
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes, average=None
        )

        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes, average=None
        )

    def forward(self, x):
        out = self.net(x)
        return out

    def loss(self, y_hat, y, reduction="none"):
        return F.cross_entropy(y_hat, y, reduction=reduction)

    def step(self, batch, batch_idx, reduction="none"):
        logits = self(batch[0])
        loss = self.loss(logits, batch[1], reduction=reduction)
        if self.hparams["gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)

        return loss, logits, batch[1]

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch, batch_idx, reduction="mean")

        preds = logits.argmax(dim=1)
        self.train_acc.update(preds, y)
        acc = self.train_acc.compute()

        self.log("train/loss", loss, prog_bar=True, logger=True)
        self.log("train/acc", acc.mean(), prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch, batch_idx, reduction="mean")

        preds = logits.argmax(dim=1)
        self.valid_acc.update(preds, y)
        acc = self.valid_acc.compute()

        self.log("valid/loss", loss, prog_bar=True, logger=True)
        self.log("valid/acc", acc.mean(), prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.step(batch, batch_idx, reduction="mean")
        preds = logits.argmax(dim=1)
        self.test_acc.update(preds, y)
        self.log("test/loss", loss)
        self.log("test/acc", self.test_acc.compute().mean())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
            return [optimizer], [scheduler]

        return [optimizer]


class SampleClassifier(Classifier):
    def __init__(self, net: nn.Module, hparams: dict):
        super().__init__(net=net, hparams=hparams)
        self.valid_top5_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes, top_k=5
        )
        self.test_top5_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes, top_k=5
        )

        self.early_stop_counter = 0
        self.patience = 5
        self.best_valid_acc = 0

        self.train_sample_state = dict(
            batch_index=[],
            batch_loss=[],
            batch_label=[],
            batch_preds=[],
            corrupted=[],
        )
        self.valid_sample_state = dict(
            batch_index=[],
            batch_loss=[],
            batch_label=[],
            batch_preds=[],
            corrupted=[],
        )

        self.sample_metrics_df = pd.DataFrame()

    def on_train_epoch_end(self) -> None:
        """
        Computes and logs per sample scores for the current epoch.
        """
        dataset = self.trainer.train_dataloader.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        self.sample_metrics_df = self.update_sample_metrics(
            self.sample_metrics_df,
            self.train_sample_state,
            dataset=dataset,
            stage="train",
        )
        self.train_sample_state = {k: [] for k in self.train_sample_state.keys()}

        if self.hparams.debug:
            import matplotlib.pyplot as plt
            import seaborn as sns

            data = (
                self.sample_metrics_df[
                    self.sample_metrics_df.epoch == self.current_epoch
                ]
                .sort_values("sample_loss")
                .reset_index(drop=True, inplace=False)
            )

            ax = sns.scatterplot(
                x=np.arange(
                    0,
                    len(data),
                ),
                y="sample_loss",
                hue="corrupted",
                data=data,
                s=4,
            )
            plt.savefig(f"./{self.hparams.id}_loss-{self.current_epoch}.png")
            plt.show()

    def on_validation_epoch_end(self) -> None:
        """
        Computes and logs per sample scores for the current epoch.
        """
        dataset = self.trainer.val_dataloaders.dataset
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        self.sample_metrics_df = self.update_sample_metrics(
            self.sample_metrics_df,
            self.valid_sample_state,
            dataset=dataset,
            stage="valid",
        )
        self.valid_sample_state = {k: [] for k in self.valid_sample_state.keys()}

    def on_validation_end(self) -> None:
        current_valid_acc = self.valid_acc.compute().mean()

        if current_valid_acc > self.best_valid_acc:
            self.best_valid_acc = current_valid_acc
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter > self.patience:
                self.early_stop_counter = 0
                try:
                    self.logger.experiment.metrics({"early_stop": self.current_epoch})
                except Exception as e:
                    print("Cannot log early_stop:", e)
                    pass

    def on_train_end(self) -> None:
        print("Training completed.")
        try:
            save_path = Path(self.hparams["save_dir"])
        except KeyError:
            random_id = np.random.randint(0, 1000000)
            save_path = Path(f"logs/{random_id}")

        train_df = self.sample_metrics_df[self.sample_metrics_df["stage"] == "train"]
        valid_df = self.sample_metrics_df[self.sample_metrics_df["stage"] == "valid"]

        train_df.to_csv(save_path / "train-sample_metrics.csv", index=False)
        valid_df.to_csv(save_path / "valid-sample_metrics.csv", index=False)

        # save as table to wandb
        # if not self.hparams.offline:
        #     self.logger.experiment.log(
        #         {
        #             "train/sample_metrics": wandb.Table(data=train_df),
        #             "valid/sample_metrics": wandb.Table(data=valid_df),
        #         }
        #     )

        if self.hparams.debug:
            import matplotlib.pyplot as plt

            plt.clf()

        print("=== Sample metrics saved. ===")
        return None

    def update_sample_metrics(
        self, sample_metrics_df, sample_state, dataset, stage="train"
    ):
        sample_state["epoch"] = self.current_epoch * np.ones(
            len(sample_state["batch_index"])
        )
        sample_state["stage"] = [stage] * len(sample_state["batch_index"])

        sample_state["corrupted"] = [
            bool(dataset.corrupted[i]) for i in sample_state["batch_index"]
        ]

        sample_state["label"] = list(map(int, sample_state["batch_label"]))

        epoch_df = pd.DataFrame.from_dict(sample_state)
        # remove "batch_" prefix if present
        epoch_df.columns = epoch_df.columns.str.replace("batch_", "sample_")

        # concat to sample_metrics_df
        return pd.concat([sample_metrics_df, epoch_df])

    def step_metrics(self, sample_state, indices, losses, y, y_pred):
        sample_state["batch_index"].extend(indices.tolist())
        sample_state["batch_loss"].extend(losses.cpu().detach().tolist())
        sample_state["batch_label"].extend(y.cpu().detach().tolist())
        sample_state["batch_preds"].extend(y_pred.cpu().detach().tolist())

        return sample_state


class LitModel(SampleClassifier):
    def __init__(self, net: nn.Module, hparams: dict):
        super().__init__(net=net, hparams=hparams)
        self.net = net
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
        self.all_logits = []

    def training_step(self, batch, batch_idx):
        loss_per_sample, logits, y = self.step(batch, batch_idx, reduction="none")
        loss_per_batch = loss_per_sample.mean()

        ### LOGGING ###
        self.log(
            f"train/loss",
            loss_per_sample.mean(),
            prog_bar=True,
            logger=True,
        )
        self.train_acc.update(logits, y)
        self.log(
            "train/acc",
            self.train_acc.compute().mean(),
            prog_bar=True,
            logger=True,
        )

        self.train_sample_state = self.step_metrics(
            self.train_sample_state,
            indices=batch[2],
            losses=loss_per_sample,
            y=y,
            y_pred=logits.argmax(dim=1),
        )

        return loss_per_batch

    def validation_step(self, batch, batch_idx):
        loss_per_sample, logits, y = self.step(batch, batch_idx)
        loss_per_batch = loss_per_sample.mean()

        ### LOGGGING ###
        self.log(
            f"valid/loss",
            loss_per_sample.mean(),
            prog_bar=True,
            logger=True,
        )
        self.valid_acc.update(logits, y)
        self.valid_top5_acc.update(logits, y)
        self.log(
            "valid/acc",
            self.valid_acc.compute().mean(),
            prog_bar=True,
        )
        self.log(
            "valid/top5_acc",
            self.valid_top5_acc.compute(),
        )

        self.valid_sample_state = self.step_metrics(
            self.valid_sample_state,
            indices=batch[2],
            losses=loss_per_sample,
            y=y,
            y_pred=logits.argmax(dim=1),
        )

        return loss_per_batch

    def test_step(self, batch, batch_idx):
        loss_per_sample, logits, y = self.step(batch, batch_idx)
        loss_per_batch = loss_per_sample.mean()

        self.log(
            f"test/loss",
            loss_per_sample.mean(),
        )
        self.test_acc.update(logits, y)
        self.test_top5_acc.update(logits, y)
        self.log(
            "test/acc",
            self.test_acc.compute().mean(),
        )
        self.log(
            "test/top5_acc",
            self.test_top5_acc.compute(),
        )
        return loss_per_batch


class LitDataModule(pl.LightningDataModule):
    def __init__(self, hparams, corrupt_prob=0.0):
        super().__init__()
        self.hparams.update(hparams)
        self.corrupt_prob = corrupt_prob

    def setup(self, stage=None):
        experiment_data = build_experiment(
            corrupt_prob=self.corrupt_prob,
            corrupt_name=self.hparams["corrupt_name"],
            batch_size=self.hparams["batch_size"],
        )[self.hparams["corrupt_name"]]

        self.train_set = experiment_data["train_set"]
        self.val_set = experiment_data["val_set"]
        self.test_set = experiment_data["test_set"]

        if self.hparams.subset < 1.0:
            subset_size = int(len(self.train_set) * self.hparams.subset)
            self.train_set = torch.utils.data.Subset(
                self.train_set, torch.randperm(len(self.train_set))[:subset_size]
            )

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
