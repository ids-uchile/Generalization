import time
from pathlib import Path

import ml_collections
import pytorch_lightning as pl
import torch
import wandb
import yaml
from generalization.models import get_cifar_models
from generalization.randomization import available_corruptions
from model import LitDataModule, LitModel
from pytorch_lightning.loggers import CSVLogger, WandbLogger

DEFAULT_PARAMS = {
    "seed": 88,
    "batch_size": 256,
    "learning_rate": 0.1,
    "epochs": 30,
    "val_every": 1,
    "log_dir": "logs",
}


def fit(trainer, model, datamodule):
    torch.set_float32_matmul_precision(precision="medium")
    start_time = time.time()
    trainer.fit(model, datamodule)
    print(f"Training took {time.time() - start_time:.2f} seconds")

    return trainer, model, datamodule


def run_one_experiment(model, datamodule, hparams, corrupt_prob=0.0):
    log_dir = Path("logs/dense")
    project_name = "generalization-dense"

    experiment_name = f"{hparams.model_name}-{hparams.corrupt_name}-{corrupt_prob}"
    timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
    hparams.name = experiment_name
    hparams.id = f"{experiment_name}-{timestamp}"
    hparams.save_dir = log_dir / project_name / experiment_name
    hparams.save_dir.mkdir(parents=True, exist_ok=True)

    # add datetime to avoid overwriting existing experiments
    logger = WandbLogger(
        name=experiment_name,
        save_dir=hparams.save_dir,
        offline=hparams.offline,
        project=project_name,
        id=hparams.id,
        tags=list(
            map(
                str,
                [hparams.model_name, hparams.corrupt_name, corrupt_prob],
            )
        ),
    )

    logger_csv = CSVLogger(save_dir=f"{log_dir}/{project_name}", name=experiment_name)

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        logger=[logger, logger_csv],
        default_root_dir=log_dir,
        check_val_every_n_epoch=hparams.val_every,
    )
    pl_model = LitModel(model, hparams=hparams)

    trainer, pl_model, datamodule = fit(trainer, pl_model, datamodule)

    trainer.test(pl_model, datamodule.test_dataloader())

    # assure that wb logger process has exited
    try:
        trainer.logger.experiment.finish()
        wandb.finish()
    except AttributeError:
        pass

    return trainer, pl_model


def main(hparams=DEFAULT_PARAMS) -> None:
    """
    Run all experiments for given corruption/model combination.

    Parameters
    ----------
    hparams : dict or ml_collections.ConfigDict
        Dict of hyperparameters (seed, batch_size, learning_rate, epochs, val_every, ...)
        See examples in the configs/ directory

    Examples
    --------
    """

    all_corruptions = available_corruptions()
    if hparams.corrupt_name != "all":
        all_corruptions = [hparams.corrupt_name]

    for corrupt_name in all_corruptions:
        print(f"Corruption: {corrupt_name}")

        if "normal_labels" in corrupt_name:
            corrupt_probs = [0.0]
        elif corrupt_name in available_corruptions():
            assert hparams.corrupt_prob != [""]
            corrupt_probs = hparams.corrupt_prob
        else:
            raise ValueError(f"Please specify valid corruption name: {corrupt_name}")

        dataset_size = -1
        for corrupt_prob in corrupt_probs:
            print(f"Corruption prob: {corrupt_prob}")
            dm = LitDataModule(hparams=hparams, corrupt_prob=corrupt_prob)
            if dataset_size == -1:
                dm.setup()
                dataset_size = len(dm.train_dataloader().dataset)

            if hparams.model_name != "all" and isinstance(hparams.model_name, str):
                get_cifar_models(in_size=32 * 32 * 3)
                models = get_cifar_models(
                    model_name=hparams.model_name,
                    use_batch_norm=True,
                    use_dropout=True,
                    lib="torch",
                    in_size=32 * 32 * 3,
                )
            elif hparams.model_name != "all" and isinstance(hparams.model_name, list):
                models = {
                    model_name: get_cifar_models(
                        model_name=model_name,
                        use_batch_norm=True,
                        use_dropout=True,
                        lib="torch",
                        in_size=32 * 32 * 3,
                    )
                    for model_name in hparams.model_name
                }
            else:
                models = get_cifar_models(
                    lib="torch",
                    use_batch_norm=True,
                    use_dropout=True,
                    in_size=32 * 32 * 3,
                )

            for model_name, model in models.items():
                print(f"Model: {model_name}")

                MODEL_NAME = model_name
                CORRUPT_NAME = corrupt_name
                CORRUPT_PROB = corrupt_prob

                hparams.update(
                    {
                        "model_name": MODEL_NAME,
                        "corrupt_name": CORRUPT_NAME,
                        "dataset_size": dataset_size,
                    }
                )
                trainer, pl_model = run_one_experiment(
                    model,
                    dm,
                    hparams,
                    corrupt_prob=CORRUPT_PROB,
                )


def check_args(args):
    if args.corrupt_name not in available_corruptions() + ["all"]:
        raise ValueError(
            "Please specify a corruption name or 'all' for all corruptions"
        )

    if args.model_name not in ["all", "alexnet", "inception", "mlp_1x512", "mlp_3x512"]:
        raise ValueError("Please specify a valid model name")

    return args.corrupt_name != "all" or args.model_name != "all"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--corrupt_name", type=str, default="gaussian_pixels")
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_PARAMS["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_PARAMS["epochs"])
    parser.add_argument("--seed", type=int, default=DEFAULT_PARAMS["seed"])
    parser.add_argument("--lr", type=float, default=DEFAULT_PARAMS["learning_rate"])
    parser.add_argument("--val_every", type=int, default=DEFAULT_PARAMS["val_every"])
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)

    args, _ = parser.parse_known_args()

    SEED = args.seed
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    VAL_EVERY = args.val_every

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(SEED)

    hparams = ml_collections.ConfigDict()
    hparams.update(
        {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "val_every": VAL_EVERY,
            "corrupt_name": args.corrupt_name,
            "model_name": args.model_name,
            "n_classes": args.nclasses,
            "offline": False,
            "debug": False,
        }
        if args.config == ""
        else yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    )
    hparams.update({"subset": args.subset, "offline": False, "debug": args.debug})

    if args.test:
        hparams.update(
            {
                "epochs": 5,
                "corrupt_prob": (
                    [0.5] if args.corrupt_name != "normal_labels" else [0.0]
                ),
                "debug": True,
                "offline": True,
            }
        )

    main(hparams)
