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

import json
import os
import warnings
from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from safetensors.numpy import load_file, save_file
from torchvision import transforms
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from ..utils.data import seed_everything
from .corruptions import *
from .utils import get_dimensions, open_data


class RandomizedDataset(VisionDataset):
    """Dataset that applies Randomization Attacks as shown in https://arxiv.org/abs/1611.03530.

    Args:
        dataset (torch.utils.data.Dataset): Dataset to be randomized
        data (torch.Tensor): Data tensor
        targets (torch.Tensor): Target tensor
        corruption_name (str): Name of the corruption to be applied
        corruption_prob (float): Probability of corruption
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
        train=True,
        transform=None,
        target_transform=None,
        seed=0,
        **kwargs,
    ):
        super().__init__(
            root=None, transform=transform, target_transform=target_transform
        )

        seed_everything(seed)
        self.seed = seed
        self.train = train
        self.corruption_name = corruption_name
        self.corruption_prob = corruption_prob

        if data is not None and targets is not None:
            self.data = data
            self.targets = targets
            self.classes = kwargs.get("classes", None)
            self.class_to_idx = kwargs.get("class_to_idx", None)
            self.original_repr = f"RandomizedDataset(seed={seed}, corruption_name={corruption_name}, corruption_prob={corruption_prob})"

        elif dataset is not None and isinstance(dataset, torch.utils.data.Dataset):
            self.data = dataset.data
            self.targets = dataset.targets
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
            self.original_repr = (
                repr(dataset)
                + f", seed={seed}, corruption_name={corruption_name}, corruption_prob={corruption_prob})"
            )

        else:
            raise ValueError(
                "Either dataset or data+targets must be provided as arguments"
            )

        self.indices = list(range(len(self.data)))
        self.corrupted = kwargs.get("corrupted", [])
        if len(self.corrupted) > 0 and kwargs.get("applied_corruptions", False):
            self.applied_corruptions = True
        else:
            self.applied_corruptions = False

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
                true_label[None]
            ]

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
            )
        elif self.corruption_name == "gaussian_pixels":
            self.corruption_func = partial(
                get_randomization(self.corruption_name),
                corruption_prob=self.corruption_prob,
                shape=(c, w, h),
                use_cifar=True,
            )

        else:
            self.corruption_func = lambda img, target, **kwargs: (
                img,
                target,
                False,
            )

    def apply_corruptions(self):
        if self.applied_corruptions or len(self.corrupted) > 0:
            logging.info("Corruptions already applied, skipping .apply_corruptions()")
            return

        for index in tqdm(range(len(self.data))):
            x = transforms.functional.to_tensor(open_data(self.data[index]))
            y = torch.tensor(self.targets[index])

            x, y, is_corrupt = self.corruption_func(x, y)

            self.corrupted.append(is_corrupt)
            self.data[index] = transforms.functional.to_pil_image(x)
            self.targets[index] = y

        self.corrupted = torch.as_tensor(self.corrupted)
        self.applied_corruptions = True

    def __getitem__(self, index):
        x = transforms.functional.to_tensor(self.data[index])
        y = torch.as_tensor(self.targets[index])

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return (x, y, index)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return self.original_repr

    def replace_transform(self, transform, target_transform=None) -> None:
        self.transform = transform
        if target_transform is not None:
            self.target_transform = target_transform

    def corruption_checks(self) -> None:
        is_full_random = self.corruption_name in [
            "random_labels",
            "random_pixels",
        ]
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

    def sample_random_grid(self, n_samples: int = 2, **kwargs) -> tuple:
        """Make a grid of random samples from the dataset.

        Args:
            nrow (int, optional): Number of images per row. Defaults to 2.

        Returns:
            tuple: A tuple containing the grid, the labels and the corruption status
        """

        sample_idxs = torch.randint(0, len(self), (n_samples,))
        samples = [self[idx] for idx in sample_idxs]
        is_corrupt = [self.corrupted[idx] for idx in sample_idxs]

        return (
            [sample[0] for sample in samples],
            [sample[1] for sample in samples],
            is_corrupt,
        )

    def show_images(self, n_samples=5, **kwargs):
        """Display a grid of random samples from the dataset.
        Shows if the tensors are corrupted or not as titles.

        Args:
            nrow (int, optional): Number of images per row. Defaults to 2.
        """

        tensor_list, labels, is_corrupt = self.sample_random_grid(
            n_samples=n_samples, **kwargs
        )

        fig, axs = plt.subplots(tight_layout=True, ncols=n_samples)
        plt.axis("off")

        grid_titles = [
            (
                f"{self.classes[label.item()]}\n(corrupted)"
                if corrupt
                else f"{self.classes[label.item()]}"
            )
            for label, corrupt in zip(labels, is_corrupt)
        ]
        dataset_label = f"Randomized Dataset with {self.corruption_name} (prob = {self.corruption_prob})"
        plt.suptitle(dataset_label, y=0.72, fontsize=14)
        # given grid's width, set a text on top of each image
        # that text is the corresponding grid_titles element
        size_oneimg = tensor_list[0].shape[-1]
        for i, ax in enumerate(axs):
            ax.imshow(tensor_list[i].permute(1, 2, 0))
            ax.set_title(grid_titles[i], fontsize=12)
            ax.axis("off")

        if "save_path" in kwargs:
            plt.savefig(kwargs["save_path"], bbox_inches="tight", dpi=300)
        plt.show()

    @staticmethod
    def load_dataset(
        root_path,
        filepath,
        transform=None,
        target_transform=None,
    ) -> "RandomizedDataset":
        """Loads the Dataset from a file.

        filepath must follow the following structure:
        filepath = "seed/dataset_name/corruption_name/corruption_prob"
        e.g. "42/cifar10/random_labels/0.2"
        """
        path = f"{root_path}/{filepath}"

        filepath_split = filepath.split("/")
        filepath_split.remove("")
        seed, corruption_name, corruption_prob, train_str = filepath_split

        tensors = load_file(filename=f"{path}/dataset.safetensors")
        meta = json.load(open(f"{path}/metadata.json", "r"))

        # check what we can assert from the metadata
        assert meta["seed"] == int(seed)
        assert meta["corruption_name"] == corruption_name
        assert meta["corruption_prob"] == float(corruption_prob)
        assert meta["train"] == (train_str == "train")

        dataset = RandomizedDataset(
            data=tensors["data"],
            targets=tensors["targets"],
            corruption_name=corruption_name,
            corruption_prob=float(corruption_prob),
            transform=transform,
            target_transform=target_transform,
            ## Loaded from metadata:
            seed=int(seed),
            train=bool(meta["train"]),
            class_to_idx=meta["classes_to_idx"],
            classes=meta["classes"],
            corrupted=tensors["corrupted"],
            applied_corruptions=True,
        )
        dataset.original_repr = meta["__repr__"]
        return dataset

    @staticmethod
    def save(dataset, root_path, seed, corruption_name, corruption_prob, train):
        """Saves the dataset to a file using SafeTensors."""
        if corruption_name == None:
            corruption_name = "normal_labels"
            corruption_prob = 0.0
        train_str = "train" if train else "test"
        filepath = f"{seed}/{corruption_name}/{corruption_prob}/{train_str}"
        fullpath = f"{root_path}/{filepath}"
        os.makedirs(fullpath, exist_ok=True)
        tensors = {
            "data": np.asarray(dataset.data),
            "targets": np.asarray(torch.as_tensor(dataset.targets)),
            "corrupted": np.asarray(dataset.corrupted),
        }
        meta = {
            "seed": seed,
            "corruption_name": dataset.corruption_name,
            "corruption_prob": dataset.corruption_prob,
            "classes_to_idx": dataset.class_to_idx,
            "classes": dataset.classes,
            "train": dataset.train,
            "__repr__": repr(dataset),
        }
        save_file(tensor_dict=tensors, filename=f"{fullpath}/dataset.safetensors")
        json.dump(meta, open(f"{fullpath}/metadata.json", "w"))

        return filepath

    def save_dataset(self, root_path):
        return RandomizedDataset.save(
            dataset=self,
            root_path=root_path,
            seed=self.seed,
            corruption_name=self.corruption_name,
            corruption_prob=self.corruption_prob,
            train=self.train,
        )
