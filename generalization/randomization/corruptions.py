# CIFAR-10 datasets used in the paper
# We run our experiments with the following modifications of the labels and input images:
#   • True labels: the original dataset without modification.
#   • Partially corrupted labels: independently with probability p, the label of each image is corrupted as a uniform
#                                 random class.
#   • Random labels: all the labels are replaced with random ones.
#   • Shuffled pixels: a random permutation of the pixels is chosen and then the same permutation is applied to all the
#                      images in both training and test set.
#   • Random pixels: a different random permutation is applied to each image independently.
#   • Gaussian: A Gaussian distribution (with matching mean and variance to the original image dataset)
#               is used to generate random pixels for each imag
#
#
# Author Note:
#   - Implements the RandomizedDataset class
#   - Implements the TensorTransformDataset class
#   - If only tensors are provided, the class assumes that the tensors are (data, target) or (data, target, index)
#     and uses the TensorTransformDataset class to create the dataset
#   - If a dataset is provided, the class assumes that the dataset is a torch.utils.data.Dataset and uses it directly
#   - We make use of self.classes and self.class_to_idx to manage the label permutations [A MUST!]
#   - We make use of the self.train flag to determine if the dataset is used for training or testing
#   - We make use of the self.corruption_name to determine the corruption function to use
#   - We make use of the self.corruption_prob to determine the probability of corruption


import importlib
from typing import Callable, List

from absl import logging

RANDOMIZATIONS = {
    "random_labels": "generalization.randomization.labels.random_labels",
    "partial_labels": "generalization.randomization.labels.partial_labels",
    "random_pixels": "generalization.randomization.inputs.random_pixels",
    "shuffled_pixels": "generalization.randomization.inputs.shuffled_pixels",
    "gaussian_pixels": "generalization.randomization.inputs.gaussian_pixels",
}


class RandomizationRegistry(object):
    """Static class for keeping track of available datasets."""

    _REGISTRY = {}

    @classmethod
    def add(cls, name: str, builder_fn: Callable):
        """Add a randomization to the registry, i.e. register a randomization/corruption.

        Args:
          name: Randomization name (must be unique).
          builder_fn: Function to be called to corruption a sample. Must accept
            randomization-specific arguments and return a corrupted sample.

        Raises:
          KeyError: If the provided name is not unique.
        """
        if name in cls._REGISTRY:
            raise KeyError(
                f"Randomization with name ({name}) already registered."
            )
        cls._REGISTRY[name] = builder_fn

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a Randomization from the registry by its name.

        Args:
          name: Randomization name.

        Returns:
          Randomization function that accepts randomization-specific parameters and
          returns a corrupted sample.

        Raises:
          KeyError: If the randomization is not found.
        """
        if name not in cls._REGISTRY:
            if name in RANDOMIZATIONS:
                module = RANDOMIZATIONS[name]
                importlib.import_module(module)
                logging.info(
                    "On-demand import of randomization (%s) from module (%s).",
                    name,
                    module,
                )
                if name not in cls._REGISTRY:
                    raise KeyError(
                        f"Imported module ({module}) did not register randomization"
                        f"({name}). Please check that dataset names match."
                    )
            else:
                raise KeyError(
                    f"Unknown dataset ({name}). Did you import the randomization "
                    f"module explicitly?"
                )
        return cls._REGISTRY[name]

    @classmethod
    def list(cls) -> List[str]:
        """List registered randomizations."""
        return list(cls._REGISTRY.keys())


def add_randomization(name: str, *args, **kwargs):
    """Decorator for shorthand randomization registdation."""

    def inner(builder_fn: Callable) -> Callable:
        RandomizationRegistry.add(name, builder_fn)
        return builder_fn

    return inner


def get_randomization(randomization_name: str) -> Callable:
    """Maps dataset name to a dataset_builder.

    API kept for compatibility of existing code with the RandomizationRegistry.

    Args:
      randomization_name: Randomization name.

    Returns:
      A corruption func.
    """
    return RandomizationRegistry.get(randomization_name)
