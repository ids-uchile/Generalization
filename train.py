import math
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchmetrics
from absl import app, flags, logging
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.nn.functional import cross_entropy as ce
from torch.utils import data
from torchinfo import summary
from torchvision.datasets import CIFAR10

from generalization import create_corrupted_dataset
from generalization.models import get_cifar_models
from generalization.randomization import available_corruptions
from generalization.randomization.transforms import get_cifar10_transforms
from generalization.utils.data import get_num_cpus

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", "./logs", "Directory to save logs")
flags.DEFINE_string("data_root", "/data", "Directory to save/load data")
flags.DEFINE_string("severity", "", "Specific severity of corruption to train on")
flags.DEFINE_boolean("attempt_load", False, "Attempt to load corrupted dataset")
flags.DEFINE_boolean("normal", False, "Train on normal data")
flags.DEFINE_boolean("augment", False, "Augment data")
flags.DEFINE_boolean("scheduler", False, "Use scheduler")

# DEBUG:
flags.DEFINE_boolean("test_pipeline", False, "Test pipeline")
flags.DEFINE_boolean("train_one_model", False, "Train one model")


class Classifier(pl.LightningModule):
    def __init__(self, model, hparams, train_corrupted=None, valid_corrupted=None):
        super().__init__()
        self.model = model
        self.train_corrupted = train_corrupted
        self.valid_corrupted = valid_corrupted
        self.hparams.update(hparams)
        self.save_hyperparameters(
            ignore=["model", "train_corrupted", "valid_corrupted"]
        )
        self.lr = self.hparams["learning_rate"]
        self.n_classes = self.hparams["n_classes"]
        self.train_acc_clean = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.train_acc_corrupted = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.valid_acc_clean = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        self.valid_acc_corrupted = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.n_classes
        )
        # self.automatic_optimization = False

    def forward(self, x):
        return self.model(x)

    def log_as_corrupted(self, loss, y_hat, y, corrupted, mode):
        acc_clean_fn = self.train_acc_clean if mode == "train" else self.valid_acc_clean
        acc_corrupted_fn = (
            self.train_acc_corrupted if mode == "train" else self.valid_acc_corrupted
        )

        if corrupted.sum() == corrupted.shape[0]:
            # All corrupted
            self.log(f"{mode}/loss_corrupted", loss.mean())
            self.log(f"{mode}/acc1_corrupted", acc_corrupted_fn(y_hat, y))
            return
        elif corrupted.sum() == 0:
            # All clean
            self.log(f"{mode}/loss_clean", loss.mean())
            self.log(f"{mode}/acc1_clean", acc_clean_fn(y_hat, y), prog_bar=True)
            return
        else:
            # Some clean, some corrupted
            self.log(f"{mode}/loss_clean", loss[corrupted == False].mean())
            self.log(
                f"{mode}/acc1_clean",
                acc_clean_fn(y_hat[corrupted == False], y[corrupted == False]),
                prog_bar=True,
            )

            self.log(f"{mode}/loss_corrupted", loss[corrupted == True].mean())
            self.log(
                f"{mode}/acc1_corrupted",
                acc_corrupted_fn(y_hat[corrupted == True], y[corrupted == True]),
            )

    def log_metrics(self, loss, y_hat, y, index=None, mode: str = "train"):
        # log current lr
        self.log(f"lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log(f"{mode}/loss", loss.mean(), prog_bar=True)

        corrupted = self.get_corrupted(index, mode)
        if corrupted is None and mode == "train":
            # No Train Corruption -> all are clean
            self.log(f"{mode}/acc1", self.train_acc_clean(y_hat, y))
        elif corrupted is None and mode == "valid":
            # No Valid Corruption -> all are clean
            self.log(f"{mode}/acc1", self.valid_acc_clean(y_hat, y), prog_bar=True)
        elif corrupted is not None:
            # Train/Valid Corruption -> log clean and corrupted separately
            self.log_as_corrupted(loss, y_hat, y, corrupted, mode)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        if len(batch) == 3:
            x, y, index = batch
        else:
            x, y = batch
            index = None

        y_hat = self.model(x)
        xe = ce(y_hat, y, reduction="none")
        if self.hparams["gradient_clipping"]:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.log_metrics(xe, y_hat, y, index, "train")
        return xe.mean()

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        if len(batch) == 3:
            x, y, index = batch
        else:
            x, y = batch
            index = None

        y_hat = self.model(x)
        xe = ce(y_hat, y, reduction="none")
        self.log_metrics(xe, y_hat, y, index, "valid")
        return xe.mean()

    def configure_optimizers(self):
        lr = self.lr
        nesterov = self.hparams.get("nesterov", False)
        momentum = self.hparams.get("momentum", 0.9)
        # optimizer = torch.optim.SGD(
        #     self.model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov
        # )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if self.hparams["use_scheduler"]:
            # 5 epoch of warmup, then cosine decay
            steps_per_epoch = math.ceil(
                self.hparams["size_train"] / self.hparams["batch_size"]
            )
            total_steps = self.hparams["max_epochs"] * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=steps_per_epoch * 5 / total_steps,
                final_div_factor=1e3,
                anneal_strategy="cos",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        return optimizer

    def get_corrupted(self, index=None, mode: str = "train"):
        if mode == "train":
            get_corrupted = index is not None and self.train_corrupted is not None
            corrupted = self.train_corrupted[index.to("cpu")] if get_corrupted else None
        elif mode == "valid":
            get_corrupted = index is not None and self.valid_corrupted is not None
            corrupted = self.valid_corrupted[index.to("cpu")] if get_corrupted else None
        return corrupted


def train_model(
    model,
    hparams,
    train_loader,
    valid_loader,
    train_corrupted=None,
    valid_corrupted=None,
    max_epochs=100,
):
    model = Classifier(
        model=model,
        hparams=hparams,
        train_corrupted=train_corrupted,
        valid_corrupted=valid_corrupted,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="valid/loss", save_top_k=1, mode="min"
    )
    augment_str = "aug" if FLAGS.augment else "noaug"
    sched_str = "sched" if FLAGS.scheduler else "nosched"
    version = f"{hparams['model_name']}-{augment_str}-{sched_str}"
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        callbacks=[checkpoint_callback],
        logger=[
            TensorBoardLogger(
                Path(FLAGS.workdir) / "cifar10",
                name=f"{hparams['corruption']}-{hparams['severity']}",
                version=version,
            ),
            CSVLogger(
                Path(FLAGS.workdir) / "cifar10",
                name=f"{hparams['corruption']}-{hparams['severity']}",
                version=version,
            ),
        ],
    )
    # this results in ./logs/cifar10/{corruption}-{severity}/{model_name}-{aug}-{nosched}

    trainer.fit(model, train_loader, valid_loader)
    return model


def run_training(models, dataset, test_dataset, severity, corruption):
    summaries = {}
    for model_name in models.keys():
        x = test_dataset[0][0]
        model = models[model_name]
        summaries[model_name] = str(
            summary(model, input_size=(1, *x.shape), verbose=0)
        ).split("\n")[-10:]

    common_hparams = {
        "n_classes": 10,
        "max_epochs": 30,
        "batch_size": 512,
        "learning_rate": 0.03,
        "momentum": 0.9,
        "nesterov": False,
        "gradient_clipping": True,
        "use_scheduler": True if FLAGS.scheduler else False,
    }

    for model_name in models.keys():
        model = models[model_name]
        hparams = common_hparams.copy()
        train_loader = data.DataLoader(
            dataset,
            batch_size=hparams["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=get_num_cpus() - 1,
            prefetch_factor=2,
        )
        valid_loader = data.DataLoader(
            test_dataset,
            batch_size=hparams["batch_size"] * 2,
            shuffle=False,
            num_workers=get_num_cpus() - 1,
            pin_memory=True,
            prefetch_factor=2,
        )

        hparams["model_name"] = model_name
        hparams["severity"] = severity
        hparams["corruption"] = corruption
        hparams["corruption_prob"] = getattr(dataset, "corruption_prob", 0.0)
        hparams["size_train"] = len(dataset)
        hparams["size_valid"] = len(test_dataset)

        logging.info(
            f"Training {model_name} on {corruption} with severity {severity} (prob = {hparams['corruption_prob']})"
        )
        has_corrupted = hasattr(dataset, "corrupted")
        if has_corrupted:
            logging.info(
                f"% of Corrupted: {sum(dataset.corrupted) / len(dataset.corrupted) * 100}",
            )

        logging.info(f"Training model with hparams: {hparams}")
        if FLAGS.test_pipeline:
            raise ValueError("Stopped before training")
        model = train_model(
            model,
            hparams,
            train_loader,
            valid_loader,
            train_corrupted=(dataset.corrupted if has_corrupted else None),
            valid_corrupted=(test_dataset.corrupted if has_corrupted else None),
            max_epochs=hparams["max_epochs"],
        )
        if FLAGS.train_one_model:
            raise ValueError("Stopped after training one model")
    return


def main(*args, **kwargs):
    SEED_A = 42
    SEED_B = 88
    SEED_C = 199

    seed = SEED_B
    models = get_cifar_models(
        use_batch_norm=True, use_dropout=True, in_size=32 * 32 * 3
    )
    root = Path(FLAGS.data_root) / "cifar10"
    input_severities = {
        "low": 0.15,
        "medium": 0.3,
        "high": 0.6,
    }
    label_severities = {
        "low": 0.05,
        "medium": 0.15,
        "high": 0.30,
    }
    if FLAGS.severity != "":
        input_severities = {FLAGS.severity: input_severities[FLAGS.severity]}
        label_severities = {FLAGS.severity: label_severities[FLAGS.severity]}

    CORRUPTIONS = available_corruptions()

    datasets = {}
    test_dataset = create_corrupted_dataset(
        dataset_name="cifar10",
        corruption_name="normal_labels",
        corruption_prob=0.0,
        root=root,
        transform=get_cifar10_transforms(),
        seed=seed,
        train=False,
        save_ds=False,
        attempt_load=False,
    )
    logging.info(
        f"Test x.shape={test_dataset[0][0].shape}, type(x)={type(test_dataset[0][0])}"
    )
    if FLAGS.normal:
        logging.info("Training on normal data only")
        dataset = create_corrupted_dataset(
            dataset_name="cifar10",
            corruption_name="normal_labels",
            corruption_prob=0.0,
            root=root,
            transform=get_cifar10_transforms(data_augmentations=FLAGS.augment),
            seed=seed,
            train=True,
            save_ds=True,
            attempt_load=FLAGS.attempt_load,
        )
        logging.info(
            f"Train x.shape={dataset[0][0].shape}, type(x)={type(dataset[0][0])}"
        )

        datasets[("none", "normal_labels")] = dataset
    elif not FLAGS.normal:
        print("Training on corrupted data:", input_severities, label_severities)
        for severity in input_severities.keys():
            for corruption in CORRUPTIONS:
                if "random" in corruption:
                    continue
                if not "pixels" in corruption:
                    continue

                logging.info(
                    f"Creating dataset for {corruption} with severity {severity}"
                )
                dataset = create_corrupted_dataset(
                    dataset_name="cifar10",
                    corruption_name=corruption,
                    corruption_prob=(
                        1.0 if "random" in corruption else input_severities[severity]
                    ),
                    root=root,
                    transform=get_cifar10_transforms(data_augmentations=FLAGS.augment),
                    seed=seed,
                    train=True,
                    save_ds=True,
                    attempt_load=FLAGS.attempt_load,
                )
                datasets[(severity, corruption)] = dataset

        for severity in label_severities.keys():
            for corruption in CORRUPTIONS:
                if "random" in corruption:
                    continue
                if not "labels" in corruption:
                    continue
                logging.info(
                    f"Creating dataset for {corruption} with severity {severity}"
                )
                dataset = create_corrupted_dataset(
                    dataset_name="cifar10",
                    corruption_name=corruption,
                    corruption_prob=(
                        1.0 if "random" in corruption else label_severities[severity]
                    ),
                    root=root,
                    transform=get_cifar10_transforms(data_augmentations=FLAGS.augment),
                    seed=seed,
                    train=True,
                    save_ds=True,
                    attempt_load=FLAGS.attempt_load,
                )

                datasets[(severity, corruption)] = dataset
    else:
        raise ValueError("Invalid configuration")
    logging.info("Datasets created")

    for severity, corruption in datasets.keys():
        dataset = datasets[(severity, corruption)]
        run_training(models, dataset, test_dataset, severity, corruption)


if __name__ == "__main__":
    app.run(main)
