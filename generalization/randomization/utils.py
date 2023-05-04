from typing import Any, List

import torch
from PIL import Image
from torch import Tensor

try:
    import accimage
except ImportError:
    accimage = None

# Mean and std for cifar dataset:
cifar10_mean = (125.3, 123.0, 113.9)
cifar10_std = (63.0, 62.1, 66.7)

cifar10_normalize_mean = (0.4914, 0.4822, 0.4465)
cifar10_normalize_std = (0.2470, 0.2435, 0.2616)

# Mean and std for imagenet dataset:
imagenet_normalize_mean = (0.485, 0.456, 0.406)
imagenet_normalize_std = (0.229, 0.224, 0.225)


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
