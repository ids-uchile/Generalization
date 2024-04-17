import torch

from ..corruptions import add_randomization
from .shuffled_pixels import apply_pixel_permutation


@add_randomization("random_pixels")
def random_pixels(img, target, corruption_prob):
    """
    Applies a random permutation to the pixels of the image.

    Args:
        img (torch.Tensor): Image tensor
        target (torch.Tensor): Target tensor
        corruption_prob (float): Probability of corruption
        apply_corruption (bool): If True, the corruption is applied to the returned image
    """
    # permutated idx are original indices
    permutation_pixels = torch.arange(img.size(1) * img.size(2))
    c, h, w = img.size()
    corrupted = False
    if torch.rand(1) <= corruption_prob:
        corrupted = True
        # choose different random permutation for each image
        permutation_pixels = torch.randperm(h * w)

        # apply it to the image
        img = apply_pixel_permutation(img, permutation_pixels)

    return img, target, torch.tensor(corrupted, dtype=torch.bool)
