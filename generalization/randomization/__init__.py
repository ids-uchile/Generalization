from .builders import build_cifar10, create_corrupted_dataset
from .corruptions import gaussian_pixels, random_labels, random_pixels, shuffled_pixels
from .dataset import RandomizedDataset


def available_corruptions():
    return [
        "gaussian_pixels",
        "random_labels",
        "random_pixels",
        "partial_labels",
        "shuffled_pixels",
    ]


__all__ = [
    "available_corruptions",
    "build_cifar10",
    "create_corrupted_dataset",
    "RandomizedDataset",
    "gaussian_pixels",
    "random_labels",
    "random_pixels",
    "shuffled_pixels",
]
