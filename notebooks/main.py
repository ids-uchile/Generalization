import os
import time

import lightning as L
import torch
from generalization.models import get_cifar_models
from generalization.randomization import available_corruptions, build_cifar10
from lightning.pytorch.loggers import WandbLogger
from model import LitModel


def get_num_cpus():
    return len(os.sched_getaffinity(0))


def collate_drop_return_index(batch):
    """
    Drops the return index from the batch

    Parameters:
    -----------
        batch (list): list of tuples (x, y, index)

    Returns:
    --------
        x, y (Tuple[torch.Tensor, torch.Tensor]): batch of data
    """
    x, y, _ = list(zip(*batch))

    return (torch.stack(x), torch.stack(y))


DEFAULT_PARAMS = {
    "seed": 88,
    "batch_size": 256,
    "learning_rate": 0.1,
    "epochs": 60,
    "val_every": 1,
    "log_dir": "logs",
}


def build_experiment(
    corrupt_prob, corrupt_name=None, batch_size=128, drop_return_index=False
):
    corruptions = available_corruptions()

    experiments = dict()

    if corrupt_name is not None:
        corruptions = [corrupt_name]

    for corrupt_name in corruptions:
        if corrupt_name == "gaussian_pixels":
            continue

        train_set, test_set = build_cifar10(
            corruption_name=corrupt_name,
            corruption_prob=corrupt_prob,
            show_images=False,
            verbose=False,
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=get_num_cpus(),
            pin_memory=True,
            collate_fn=collate_drop_return_index if drop_return_index else None,
        )

        val_set, test_set = torch.utils.data.random_split(
            test_set, [len(test_set) // 2, len(test_set) - len(test_set) // 2]
        )

        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=get_num_cpus(),
            pin_memory=True,
            collate_fn=collate_drop_return_index if drop_return_index else None,
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=get_num_cpus(),
            pin_memory=True,
            collate_fn=collate_drop_return_index if drop_return_index else None,
        )

        experiments[corrupt_name] = {
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
        }
    return experiments


def run_one_experiment(model, train_loader, val_loader, test_loader, hparams):
    logger = WandbLogger(
        name=f"{hparams['model_name']}-{hparams['corrupt_prob']}",
        project="generalization",
        log_model=True,
        save_dir="logs",
        version="4",
        id=f"{hparams['model_name']}-{hparams['corrupt_name']}-{hparams['corrupt_prob']}",
        group=f"{hparams['corrupt_name']}",
        tags=[hparams["model_name"], hparams["corrupt_name"]],
    )

    trainer = L.Trainer(
        max_epochs=hparams["epochs"],
        logger=logger,
        default_root_dir="logs",
        check_val_every_n_epoch=hparams["val_every"],
    )
    pl_model = LitModel(model, hparams=hparams)
    start_time = time.time()
    trainer.fit(
        pl_model,
        train_loader,
        val_loader,
    )
    print(f"Training took {time.time() - start_time:.2f} seconds")

    trainer.test(pl_model, test_loader)

    # assure that logger process has exited
    trainer.logger.experiment.finish()

    return trainer, pl_model


def main(
    corrupt_name: str = "all", model_name: str = "all", hparams=DEFAULT_PARAMS
) -> None:
    """
    Run all experiments for given corruption/model combination.

    Parameters
    ----------
    corrupt_name : str  (default: "all")
        Corruption name to run experiments for. If "all", run all corruptions.

    model_name : str  (default: "all")
        Model name to run experiments for. If "all", run all models.

    Examples
    --------
    >>> main(corrupt_name="all", model_name="all")
        # Run all experiments for all corruptions and all models

    >>> main(corrupt_name="random_labels", model_name="alexnet")
        # Run experiment for random labels and alexnet
    """

    all_corruptions = available_corruptions()
    if corrupt_name != "all":
        all_corruptions = [corrupt_name]

    for corrupt_name in all_corruptions:
        if corrupt_name == "gaussian_pixels":
            # raise ValueError("Gaussian pixels are not supported")
            continue

        print(f"Corruption: {corrupt_name}")

        corrup_probs = (
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
            if not "random" in corrupt_name
            else [1]
        )

        for corrupt_prob in corrup_probs:
            print(f"Corruption prob: {corrupt_prob}")
            experiment = build_experiment(
                corrupt_prob=corrupt_prob,
                corrupt_name=corrupt_name,
                batch_size=hparams["batch_size"],
            )

            models = get_cifar_models(lib="torch")
            if model_name != "all":
                models = {model_name: get_cifar_models(lib="torch")[model_name]}

            for model_name, model in models.items():
                print(f"Model: {model_name}")

                MODEL_NAME = model_name
                CORRUPT_NAME = corrupt_name
                CORRUPT_PROB = corrupt_prob
                hparams.update(
                    {
                        "model_name": MODEL_NAME,
                        "corrupt_name": CORRUPT_NAME,
                        "corrupt_prob": CORRUPT_PROB,
                    }
                )
                trainer, pl_model = run_one_experiment(
                    model,
                    experiment[corrupt_name]["train_loader"],
                    experiment[corrupt_name]["val_loader"],
                    experiment[corrupt_name]["test_loader"],
                    hparams,
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
    parser.add_argument("--corrupt_name", type=str, default="all")
    parser.add_argument("--model_name", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_PARAMS["batch_size"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_PARAMS["epochs"])
    parser.add_argument("--seed", type=int, default=DEFAULT_PARAMS["seed"])
    parser.add_argument("--lr", type=float, default=DEFAULT_PARAMS["learning_rate"])
    parser.add_argument("--val_every", type=int, default=DEFAULT_PARAMS["val_every"])

    args = parser.parse_args()

    SEED = args.seed
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    VAL_EVERY = args.val_every

    torch.set_float32_matmul_precision("medium")
    L.seed_everything(SEED)

    hparams = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "val_every": VAL_EVERY,
    }

    is_specific_run = check_args(args)
    if is_specific_run:
        main(args.corrupt_name, args.model_name, hparams)
    else:
        main()
