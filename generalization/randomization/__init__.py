from .builders import build_cifar10, create_corrupted_dataset
from .corruptions import RANDOMIZATIONS, get_randomization
from .dataset import RandomizedDataset


def available_corruptions():
    return list(RANDOMIZATIONS.keys())
