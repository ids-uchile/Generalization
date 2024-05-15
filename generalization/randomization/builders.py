import logging
import os
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10, ImageNet

from .. import utils as utils
from .dataset import RandomizedDataset
from .transforms import get_cifar10_transforms
from .utils import image_grid


def create_corrupted_dataset(
    dataset_name="cifar10",
    dataset=None,
    corruption_name=None,
    corruption_prob=0.0,
    train=True,
    root="./data/cifar10",
    transform=None,
    target_transform=None,
    save_ds=False,
    attempt_load=True,
    seed=0,
):
    utils.seed_everything(seed)

    train_str = "train" if train else "test"
    if corruption_name is None:
        corruption_name = "normal_labels"
        corruption_prob = 0.0

    possible_path = (
        Path(root) / f"{seed}/{corruption_name}/{corruption_prob}/{train_str}"
    )

    if attempt_load:
        logging.info(f"Checking for dataset at {possible_path}")
    if os.path.exists(possible_path) and attempt_load:
        logging.info(f"Loading dataset from {possible_path}")
        return RandomizedDataset.load_dataset(
            root_path=root,
            filepath=f"/{seed}/{corruption_name}/{corruption_prob}/{train_str}",
            transform=transform,
            target_transform=target_transform,
        )

    # either pass the dataset or the dataset name
    # if both are passed, raise an error
    if dataset is not None and dataset_name is not None:
        raise ValueError("Either pass the dataset or the dataset name, not both.")
    if dataset is None:
        assert dataset_name is not None, "Dataset name must be provided."
        if dataset_name.lower() == "imagenet":
            dataset = ImageNet(root=root, download=True, train=train)
        elif dataset_name.lower() == "cifar10":
            dataset = CIFAR10(root=root, download=True, train=train)
        else:
            raise ValueError("Dataset name must be either 'imagenet' or 'cifar10'")

    dataset = RandomizedDataset(
        dataset=dataset,
        corruption_name=corruption_name,
        corruption_prob=corruption_prob,
        train=train,
        transform=transform,
        target_transform=target_transform,
        seed=seed,
    )
    if save_ds:
        dataset.save_dataset(
            root_path=root,
        )
        logging.info(f"Dataset saved at {possible_path}")
    return dataset


def build_cifar10(
    corruption_name,
    corruption_prob=0.0,
    root="./data/cifar10",
    seed=0,
    save_ds=False,
    attempt_load=True,
    show_images=False,
    verbose=False,
):
    base_transforms = get_cifar10_transforms()
    train_dset = create_corrupted_dataset(
        dataset_name="cifar10",
        corruption_name=corruption_name,
        corruption_prob=corruption_prob,
        train=True,
        root=root,
        transform=base_transforms,
        seed=seed,
        save_ds=save_ds,
        attempt_load=attempt_load,
    )

    test_dset = create_corrupted_dataset(
        dataset_name="cifar10",
        train=False,
        root=root,
        corruption_name="normal_labels",
        transform=base_transforms,
        seed=seed,
        save_ds=False,
        attempt_load=False,
    )
    random_idxs = np.random.choice(len(test_dset), 10)
    if verbose:
        print("Output Shape:", test_dset[random_idxs[0]][0].shape)
    if show_images:
        image_grid(train_dset, random_idxs, no_transform=True)
        image_grid(test_dset, random_idxs, no_transform=True)
    return (train_dset, test_dset)
