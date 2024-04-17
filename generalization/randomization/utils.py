from typing import Any, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor
from torchvision import transforms

# Mean and std for cifar dataset:
CIFAR10_NORMALIZE_MEAN = tuple((0.4914, 0.4822, 0.4465))
CIFAR10_NORMALIZE_STD = tuple((0.2470, 0.2435, 0.2616))

CIFAR10_CHANNEL_MEAN = tuple((CIFAR10_NORMALIZE_MEAN[c] * 255.0 for c in range(3)))
CIFAR10_CHANNEL_STD = tuple((CIFAR10_NORMALIZE_STD[c] * 255.0 for c in range(3)))

# Mean and std for imagenet dataset:
IMAGENET_NORMALIZE_MEAN = tuple((0.485, 0.456, 0.406))
IMAGENET_NORMALIZE_STD = tuple((0.229, 0.224, 0.225))

IMAGENET_CHANNEL_MEAN = tuple((IMAGENET_NORMALIZE_MEAN[c] * 255.0 for c in range(3)))
IMAGENET_CHANNEL_STD = tuple((IMAGENET_NORMALIZE_STD[c] * 255.0 for c in range(3)))


def image_grid(dataset, idxs, no_transform=False, save_fname=None):
    if save_fname:
        # increase font size
        plt.rcParams.update({"font.size": 16})
    assert len(idxs) == 10, "Only 10 images can be displayed at a time."

    fig, axs = plt.subplots(2, 5, figsize=(10, 5))

    if no_transform:
        dset_transform = dataset.transform
        # remove Normalize from dset_transform.transforms
        dataset.replace_transform(
            transforms.Compose(
                [
                    t
                    for t in dset_transform.transforms
                    if not isinstance(t, transforms.Normalize)
                ]
            )
        )

    for i, idx in enumerate(idxs):
        out = dataset[idx]

        if len(out) == 3:
            img, label, index = out
            corrupted_label = label
        elif len(out) == 4:
            img, label, index, corrupted_label = out
        else:
            raise ValueError(f"Invalid output from dataset: {len(out)}")

        axs[i // 5, i % 5].imshow(img.permute(1, 2, 0))
        axs[i // 5, i % 5].set_title(dataset.classes[corrupted_label])
        axs[i // 5, i % 5].axis("off")

    if no_transform:
        dataset.replace_transform(dset_transform)

    if save_fname:
        fig.savefig(f"{save_fname}-grid.png", bbox_inches="tight", dpi=300)

    plt.tight_layout()
    return fig


def image_grid_comparison(dataset, idxs, no_transform=False, save_fname=None):
    if save_fname:
        # increase font size
        plt.rcParams.update({"font.size": 16})

    fig, axs = plt.subplots(2, 5, figsize=(10, 5))

    if no_transform:
        dset_transform = dataset.transform
        dataset.replace_transform(transforms.ToTensor())

    for i, idx in enumerate(idxs[:5]):
        out = dataset[idx]

        if len(out) == 3:
            img, label, index = out
            corrupted_perm = None

        elif len(out) == 4:
            img, label, index, corrupted_perm = out
        else:
            raise ValueError(f"Invalid output from dataset: {len(out)}")

        if corrupted_perm is not None:
            permutation_as_img = corrupted_perm.repeat(3, 1).view(3, -1).long()
            permutated_img = (
                img.view(3, -1).gather(1, permutation_as_img).view(3, 32, 32)
            )
        else:
            permutated_img = img

        # show original and permutated image side by side
        axs[0, i].imshow(img.permute(1, 2, 0))
        axs[0, i].set_title(dataset.classes[label])
        axs[0, i].axis("off")

        axs[1, i].imshow(permutated_img.permute(1, 2, 0))
        axs[1, i].set_title(dataset.classes[label])
        axs[1, i].axis("off")

    if no_transform:
        dataset.replace_transform(dset_transform)

    if save_fname:
        fig.savefig(f"{save_fname}-comparison.png", bbox_inches="tight", dpi=300)

    plt.tight_layout()
    fig.show()


def open_data(data: Any):
    """
    Opens an unknown data type and returns a PIL image.
    """

    if isinstance(data, np.ndarray) or isinstance(data, Tensor):
        return Image.fromarray(data)
    return Image.open(data)


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")


def get_dimensions(img) -> List[int]:
    """Returns the dimensions of an image as [channels, height, width].

    Args:
        img (PIL Image or Tensor): The image to be checked.

    Returns:
        List[int]: The image dimensions.

    Taken from torchvision.transforms.functional ...
    """
    if isinstance(img, torch.Tensor):
        return tensor_dimensions(img)

    return pil_dimensions(img)


@torch.jit.unused
def _is_pil_image(img: Any) -> bool:
    return isinstance(img, Image.Image)


@torch.jit.unused
def pil_dimensions(img: Any) -> List[int]:
    if _is_pil_image(img):
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]


def tensor_dimensions(img: Tensor) -> List[int]:
    _assert_image_tensor(img)
    channels = 1 if img.ndim == 2 else img.shape[-3]
    height, width = img.shape[-2:]
    return [channels, height, width]
