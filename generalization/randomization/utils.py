from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageNet

from .dataset import RandomizedDataset
from .transforms import get_cifar10_transforms

# Mean and std for cifar dataset:
CIFAR10_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZE_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CHANNEL_MEAN = (CIFAR10_NORMALIZE_MEAN[c] * 255.0 for c in range(3))
CIFAR10_CHANNEL_STD = (CIFAR10_NORMALIZE_STD[c] * 255.0 for c in range(3))

# Mean and std for imagenet dataset:
IMAGENET_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORMALIZE_STD = (0.229, 0.224, 0.225)

IMAGENET_CHANNEL_MEAN = (IMAGENET_NORMALIZE_MEAN[c] * 255.0 for c in range(3))
IMAGENET_CHANNEL_STD = (IMAGENET_NORMALIZE_STD[c] * 255.0 for c in range(3))


def image_grid(dataset, idxs, no_transform=False):
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))

    if no_transform:
        dset_transform = dataset.transform
        # remove Normalize from dset_transform.transforms
        dataset.replace_transform(
            transforms.Compose(
                [
                    t
                    for t in dset_transform.transforms
                    if not isinstance(t, transforms.Normalize)
                ]
            )
        )

    for i, idx in enumerate(idxs):
        out = dataset[idx]

        if len(out) == 3:
            img, label, corrupted_label = out
        else:
            img, label = out
            corrupted_label = label

        axs[i // 5, i % 5].imshow(img.permute(1, 2, 0))
        axs[i // 5, i % 5].set_title(dataset.classes[corrupted_label])
        axs[i // 5, i % 5].axis("off")

    if no_transform:
        dataset.replace_transform(dset_transform)

    fig.show()


def image_grid_comparision(dataset, idxs, no_transform=False):
    fig, axs = plt.subplots(5, 2, figsize=(5, 10))

    if no_transform:
        dset_transform = dataset.transform
        dataset.replace_transform(transforms.ToTensor())

    for i, idx in enumerate(idxs[:5]):
        out = dataset[idx]

        if len(out) == 3:
            img, label, corrupted_perm = out
        else:
            img, label = out
            # corrupted_label = label

        permutation_as_img = corrupted_perm.repeat(3, 1).view(3, -1).long()
        permutated_img = img.view(3, -1).gather(1, permutation_as_img).view(3, 32, 32)

        # show original and permutated image side by side
        axs[i, 0].imshow(img.permute(1, 2, 0))
        axs[i, 0].set_title(dataset.classes[label])
        axs[i, 0].axis("off")

        axs[i, 1].imshow(permutated_img.permute(1, 2, 0))
        axs[i, 1].set_title(dataset.classes[label])
        axs[i, 1].axis("off")

    if no_transform:
        dataset.replace_transform(dset_transform)

    fig.show()


def create_corrupted_dataset(
    dataset_name,
    corruption_name,
    corruption_prob=0.3,
    train=True,
    root="./data/cifar10",
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
        dataset,
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
    corruption_prob=0.3,
    batch_size=128,
    root="./data/cifar10",
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
        corruption_name=None,
        transform=base_transforms,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=batch_size, shuffle=True, num_workers=6
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=batch_size * 2, shuffle=False, num_workers=6
    )
    random_idxs = np.random.choice(len(test_dset), 10)
    if verbose:
        print("Output Shape:", test_dset[random_idxs[0]][0].shape)
    if show_images:
        image_grid(train_dset, random_idxs, no_transform=True)
        image_grid(test_dset, random_idxs, no_transform=True)
    return (train_dset, test_dset), train_loader, test_loader


def open_data(data: Any):
    """
    Opens an unknown data type and returns a PIL image.
    """

    if isinstance(data, np.ndarray):
        return Image.fromarray(data)
    return Image.open(data)
