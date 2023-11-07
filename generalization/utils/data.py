import os

import torch
from generalization.randomization import available_corruptions, build_cifar10


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
    corrupt_prob,
    corrupt_name=None,
    batch_size=128,
    drop_return_index=False,
    *,
    build_dl=True
):
    corruptions = available_corruptions()

    experiments = dict()

    if corrupt_name is not None:
        corruptions = [corrupt_name]

    for corrupt_name in corruptions:
        train_set, test_set = build_cifar10(
            corruption_name=corrupt_name,
            corruption_prob=corrupt_prob,
            show_images=False,
            verbose=False,
        )

        val_set, test_set = torch.utils.data.random_split(
            test_set, [len(test_set) // 2, len(test_set) - len(test_set) // 2]
        )

        experiments[corrupt_name] = {
            "train_set": train_set,
            "val_set": val_set,
            "test_set": test_set,
        }

        if build_dl:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=get_num_cpus(),
                pin_memory=True,
                collate_fn=collate_drop_return_index if drop_return_index else None,
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

            experiments[corrupt_name].update(
                {
                    "train_loader": train_loader,
                    "val_loader": val_loader,
                    "test_loader": test_loader,
                }
            )
    return experiments
