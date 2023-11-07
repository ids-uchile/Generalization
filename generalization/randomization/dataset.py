# Author: @Stepp1
#
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
#   - If a dataset is provided, the class assumes that the dataset is a torch.utils.data.Dataset and uses it directly
#   - We make use of self.classes and self.class_to_idx to manage the label permutations [A MUST!]
#   - We make use of the self.train flag to apply corruptions (training) or not (testing)
#   - We make use of the self.corruption_name to determine the corruption function to use
#   - We make use of the self.corruption_prob to determine the probability of corruption
#
# All corruption functions are defined in generalization/data/corruptions.py

import warnings
from functools import partial

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from .corruptions import *
from .utils import get_dimensions, open_data


class RandomizedDataset(VisionDataset):
    """Dataset that applies Randomization Attacks as shown in https://arxiv.org/abs/1611.03530.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to be randomized
        data (torch.Tensor): Data tensor
        targets (torch.Tensor): Target tensor
        corruption_name (str): Name of the corruption to be applied
        corruption_probtarget_idx (float): Probability of corruption
        apply_corruption (bool): If True, the corruption is applied to the returned image
        return_corruption (bool): If True, the corruption is returned along with the image
        train (bool): If True, the dataset is used for training
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.


    Allowed corruption names:
        - "random_labels": all the labels are replaced with random ones
        - "partial_labels": independently with probability p, the label of each image is corrupted as a uniform random class
        - "gaussian_pixels": A Gaussian distribution (with matching mean and variance to the original image dataset) is used to generate random pixels for each image
        - "random_pixels": a different random permutation is applied to each image independently
        - "shuffled_pixels": a random permutation of the pixels is chosen and then the same permutation is applied to all the images in both training and test set

    All corruptions allow for a corruption_prob except for "random_labels" where corruption_prob is ignored and set to 1.0.
    """

    def __init__(
        self,
        data=None,
        targets=None,
        dataset=None,
        corruption_name=None,
        corruption_prob=0.0,
        apply_corruption=False,
        return_corruption=False,
        train=True,
        transform=None,
        target_transform=None,
        **kwargs,
    ):
        super().__init__(
            root=None, transform=transform, target_transform=target_transform
        )

        if data is not None and targets is not None:
            self.data = data
            self.targets = targets
            self.classes = None
            self.class_to_idx = None
            self.original_repr = "RandomizedDataset"

        if dataset is not None and isinstance(dataset, torch.utils.data.Dataset):
            self.data = dataset.data
            self.targets = dataset.targets
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
            self.original_repr = repr(dataset)

        else:
            raise ValueError(
                "Either dataset or data+targets must be provided as arguments"
            )

        self.indices = list(range(len(self.data)))
        self.train = train
        self.corruption_name = corruption_name
        self.corruption_prob = corruption_prob

        self.kwargs = kwargs if kwargs is not None else {}
        self.corrupted = []
        self.setup_corruption_func()
        self.apply_corruptions()

    def setup_corruption_func(self):
        c, w, h = get_dimensions(open_data(self.data[0]))
        permutation_size = h * w * c // c

        self.corruption_checks()

        if self.corruption_name in ["random_labels", "partial_labels"]:
            # choose a permutation of the labels
            self.label_permutation = torch.randperm(len(self.class_to_idx))

            # given a permutation and the true label, return a corrupted label
            self.get_random_label = lambda true_label: self.label_permutation[
                true_label
            ].item()

            self.corruption_func = partial(
                get_randomization(self.corruption_name),
                corruption_prob=self.corruption_prob,
                get_random_label=self.get_random_label,
            )

        elif self.corruption_name == "shuffled_pixels":
            # we cannot assume correct order [*,C,H,W] => we want to shuffle pixels in H,W
            self.pixel_permutation = torch.randperm(permutation_size)

            self.corruption_func = partial(
                get_randomization(self.corruption_name),
                corruption_prob=self.corruption_prob,
                permutation=self.pixel_permutation,
            )

        elif self.corruption_name == "random_pixels":
            self.corruption_func = partial(
                get_randomization(self.corruption_name),
                corruption_prob=self.corruption_prob,
                permutation_size=permutation_size,
            )
        elif self.corruption_name == "gaussian_pixels":
            self.corruption_func = partial(
                get_randomization(self.corruption_name),
                corruption_prob=self.corruption_prob,
                use_cifar=True,
            )

        else:
            self.corruption_func = lambda img, target, **kwargs: (img, target, False)

    def apply_corruptions(self):
        for index in tqdm(range(len(self.data))):
            x = transforms.functional.to_tensor(open_data(self.data[index]))
            y = torch.tensor(self.targets[index])

            x, y, is_corrupt = self.corruption_func(x, y)

            self.corrupted.append(is_corrupt)
            self.data[index] = transforms.functional.to_pil_image(x)
            self.targets[index] = y

    def __getitem__(self, index):
        x = transforms.functional.to_tensor(open_data(self.data[index]))
        y = torch.as_tensor(self.targets[index])

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return (x, y, index)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.original_repr + self.extra_repr()

    def extra_repr(self) -> str:
        corruption = self.corruption_name
        return f", Corruption: {corruption}"

    def replace_transform(self, transform, target_transform=None):
        self.transform = transform

        if target_transform is not None:
            self.target_transform = target_transform

    def corruption_checks(self):
        is_full_random = self.corruption_name in ["random_labels", "random_pixels"]
        if is_full_random:
            check_corrupt_prob = not self.corruption_prob in [0.0, 1.0]
            if check_corrupt_prob:
                warnings.warn(
                    "corruption_prob is ignored when corruption_name is 'random_*'"
                )
            self.corruption_prob = 1.0
        else:
            is_normal = self.corruption_name == "normal_labels"
            not_using_corruption_prob = self.corruption_prob == 0.0
            if not_using_corruption_prob and not is_normal:
                warnings.warn(
                    "corruption_prob is not provided, using default value of 0.0"
                )
