from .randomization import (
    RandomizedDataset,
    create_corrupted_dataset,
    get_randomization,
)
from .utils import seed_everything

"""
This file contains the functions to load the datasets.

corruption_name must be one of the following:

"random_labels"
"partial_labels"
"gaussian_pixels"
"random_pixels"
"shuffled_pixels"
"""

CORRUPTIONS = [
    "random_labels",
    "partial_labels",
    "gaussian_pixels",
    "random_pixels",
    "shuffled_pixels",
]

RANDOMIZATIONS = CORRUPTIONS
