from typing import Any, List

import torch
from PIL import Image
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

# Mean and std for cifar dataset:
CIFAR10_NORMALIZE_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_NORMALIZE_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CHANNEL_MEAN = (CIFAR10_NORMALIZE_MEAN[c] * 255.0 for c in range(3))
CIFAR10_CHANNEL_STD = (CIFAR10_NORMALIZE_STD[c] * 255.0 for c in range(3))

# Mean and std for imagenet dataset:
IMAGENET_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORMALIZE_STD = (0.229, 0.224, 0.225)

IMAGENET_CHANNEL_MEAN = (IMAGENET_NORMALIZE_MEAN[c] * 255.0 for c in range(3))
IMAGENET_CHANNEL_STD = (IMAGENET_NORMALIZE_STD[c] * 255.0 for c in range(3))


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    return x.ndim >= 2


def _assert_image_tensor(img: Tensor) -> None:
    if not _is_tensor_a_torch_image(img):
        raise TypeError("Tensor is not a torch image.")


def get_dimensions(img: Tensor) -> List[int]:
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
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
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
