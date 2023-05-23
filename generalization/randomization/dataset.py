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

from .corruptions import *
from .utils import get_dimensions, open_data


class TensorTransformDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""

    def __init__(
        self, *tensors, data_idx=0, target_idx=1, transform=None, target_transform=None
    ):
        # check if *tensors is tensor, if not add ToTensor() to transform and target_transform
        if not all([isinstance(tensor, torch.Tensor) for tensor in tensors]):
            if transform is None:
                transform = transforms.ToTensor()
            # check if ToTensor is in transform, if not add it
            elif repr(transform).find("ToTensor") == -1:
                transform = transforms.Compose([transforms.ToTensor(), transform])

        # check if tensors are of same length
        if not all([len(tensors[0]) == len(tensor) for tensor in tensors]):
            raise ValueError("All tensors must be of same length")

        self.tensors = tensors

        self.data = tensors[data_idx]
        self.targets = tensors[target_idx]

        self.data_idx = data_idx
        self.target_idx = target_idx

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (*tensors) with transformed data and target
        """

        out_tuple = []
        x = open_data(self.data[index])
        if self.transform is not None:
            x = self.transform(x)
        out_tuple.append(x)

        y = self.targets[index]
        if self.target_transform is not None:
            y = self.target_transform(y)
        out_tuple.append(y)

        return tuple(out_tuple)


class RandomizedDataset(torch.utils.data.Dataset):
    """Dataset that applies Randomization Attacks as shown in https://arxiv.org/abs/1611.03530.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to be randomized
        data (torch.Tensor): Data tensor
        targets (torch.Tensor): Target tensor
        corruption_name (str): Name of the corruption to be applied
        corruption_prob (float): Probability of corruption
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
        dataset=None,
        data=None,
        targets=None,
        corruption_name=None,
        corruption_prob=0.0,
        apply_corruption=False,
        return_corruption=False,
        train=True,
        transform=None,
        target_transform=None,
    ):
        # apply_corruption and return_corruption cannot be both False
        if (
            not apply_corruption
            and not return_corruption
            and corruption_name is not None
        ):
            raise ValueError(
                "Either apply_corruption or return_corruption must be True"
            )

        if dataset is not None and isinstance(dataset, torch.utils.data.Dataset):
            data = dataset.data
            targets = dataset.targets
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
            self.original_repr = repr(dataset)

        if data is not None and targets is not None:
            self.dataset = TensorTransformDataset(
                data, targets, transform=transform, target_transform=target_transform
            )

        else:
            raise ValueError(
                "Either dataset or data+targets must be provided as arguments"
            )

        self.train = train
        self.corruption_name = corruption_name
        self.corruption_prob = corruption_prob
        self.apply_corruption = apply_corruption
        self.return_corruption = return_corruption
        self.transform = self.dataset.transform
        self.target_transform = self.dataset.target_transform

        if self.train:
            self.setup_corruption_func()
        else:
            self.corruption_func = None

    def setup_corruption_func(self):
        c, w, h = get_dimensions(open_data(self.dataset.data[0]))

        self.corruption_checks()

        if self.corruption_name in ["random_labels", "partial_labels"]:
            # choose a permutation of the labels
            self.label_permutation = torch.randperm(len(self.class_to_idx))

            # given a permutation and the true label, return a corrupted label
            self.get_random_label = lambda true_label: self.label_permutation[
                true_label
            ].item()

            self.corruption_func = partial(
                random_labels,
                train=self.train,
                corruption_prob=self.corruption_prob,
                get_random_label=self.get_random_label,
                apply_corruption=self.apply_corruption,
            )
            # TODO: maybe check https://discuss.pytorch.org/t/chose-random-element-from-tensor-excluding-certain-index/42285

        elif self.corruption_name == "shuffled_pixels":
            # we cannot assume correct order [*,C,H,W] => we want to shuffle pixels in H,W
            self.pixel_permutation = torch.randperm(h * w * c // c)
            self.corruption_func = partial(
                shuffled_pixels,
                train=self.train,
                corruption_prob=self.corruption_prob,
                permutation=self.pixel_permutation,
                apply_corruption=self.apply_corruption,
            )

        elif self.corruption_name == "random_pixels":
            # we cannot assume correct order [*,C,H,W] => we want to shuffle pixels in H,W
            permutation_size = h * w * c // c
            self.corruption_func = partial(
                random_pixels,
                train=self.train,
                corruption_prob=self.corruption_prob,
                permutation_size=permutation_size,
                apply_corruption=self.apply_corruption,
            )
        elif self.corruption_name == "gaussian_pixels":
            self.corruption_func = self.gaussian_image

        else:
            self.corruption_func = None

    def __getitem__(self, index):
        out = self.dataset[index]

        if self.corruption_func is not None:
            *out, corruption = self.corruption_func(*out)

        if self.return_corruption:
            return tuple((*out, corruption))

        return out

    def __len__(self):
        return len(self.dataset.tensors[0])

    def __repr__(self):
        return self.original_repr + self.extra_repr()

    def extra_repr(self) -> str:
        corruption = self.corruption_name
        return f", Corruption: {corruption}"

    def replace_transform(self, transform, target_transform=None):
        self.transform = transform
        self.dataset.transform = transform

        if target_transform is not None:
            self.target_transform = target_transform
            self.dataset.target_transform = target_transform

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
            not_using_corruption_prob = self.corruption_prob == 0.0
            if not_using_corruption_prob:
                warnings.warn(
                    "corruption_prob is not provided, using default value of 0.0"
                )
