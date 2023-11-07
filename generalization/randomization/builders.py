import numpy as np
import torch
from torchvision.datasets import CIFAR10, ImageNet

from .dataset import RandomizedDataset
from .transforms import get_cifar10_transforms
from .utils import image_grid


def create_corrupted_dataset(
    dataset_name,
    corruption_name,
    corruption_prob=0.0,
    train=True,
    root="/data/cifar10",
    apply_corruption=False,
    return_corruption=False,
    transform=None,
    target_transform=None,
):
    if dataset_name.lower() == "imagenet":
        dataset = ImageNet(root=root, download=True, train=train)
    elif dataset_name.lower() == "cifar10":
        dataset = CIFAR10(root=root, download=True, train=train)
    else:
        raise ValueError("Dataset name must be either 'imagenet' or 'cifar10'")

    return RandomizedDataset(
        dataset=dataset,
        corruption_name=corruption_name,
        corruption_prob=corruption_prob,
        apply_corruption=apply_corruption,
        return_corruption=return_corruption,
        train=train,
        transform=transform,
        target_transform=target_transform,
    )


def build_cifar10(
    corruption_name,
    corruption_prob=0.0,
    root="/data/cifar10",
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
        apply_corruption=True,
        return_corruption=False,
        transform=base_transforms,
    )

    test_dset = create_corrupted_dataset(
        dataset_name="cifar10",
        train=False,
        root=root,
        corruption_name="normal_labels",
        apply_corruption=True,
        transform=base_transforms,
    )
    random_idxs = np.random.choice(len(test_dset), 10)
    if verbose:
        print("Output Shape:", test_dset[random_idxs[0]][0].shape)
    if show_images:
        image_grid(train_dset, random_idxs, no_transform=True)
        image_grid(test_dset, random_idxs, no_transform=True)
    return (train_dset, test_dset)
