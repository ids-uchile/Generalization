from torchvision.datasets import CIFAR10

from .randomization import RandomizedDataset

"""
This file contains the functions to load the datasets.

corruption_name must be one of the following:

"random_labels"
"partial_labels"
"gaussian_pixels"
"random_pixels"
"shuffled_pixels"
"""

corruptions = [
    "random_labels",
    "partial_labels",
    "gaussian_pixels",
    "random_pixels",
    "shuffled_pixels",
]


def load_randomized_cifar10(corruption_name, corruption_prob=0.3, train=True):
    return RandomizedDataset(
        CIFAR10(root="./data/cifar10", download=True, train=train),
        corruption_name=corruption_name,
        corruption_prob=corruption_prob,
        apply_corruption=False,
        train=train,
    )
