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


def get_cifar10_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(28),
            transforms.Normalize(
                mean=CIFAR10_NORMALIZE_MEAN, std=CIFAR10_NORMALIZE_STD
            ),
        ]
    )


def get_imagenet_transforms():
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_NORMALIZE_MEAN, std=IMAGENET_NORMALIZE_STD
            ),
        ]
    )
