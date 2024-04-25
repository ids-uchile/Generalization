"""
Base Transforms for each dataset.
"""

from torchvision import transforms

from .utils import (
    CIFAR10_NORMALIZE_MEAN,
    CIFAR10_NORMALIZE_STD,
    IMAGENET_NORMALIZE_MEAN,
    IMAGENET_NORMALIZE_STD,
)


def get_cifar10_transforms(data_augmentations=False):
    da = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    tmfs = [
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_NORMALIZE_MEAN, std=CIFAR10_NORMALIZE_STD),
    ]
    return transforms.Compose(tmfs + da if data_augmentations else tmfs)


def get_imagenet_transforms(data_augmentations=False):
    da = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.Normalize(
                mean=IMAGENET_NORMALIZE_MEAN, std=IMAGENET_NORMALIZE_STD
            ),
        ]
        + da
        if data_augmentations
        else []
    )
