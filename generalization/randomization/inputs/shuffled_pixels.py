import torch

from ..corruptions import add_randomization


@add_randomization("shuffled_pixels")
def shuffled_pixels(img, target, corruption_prob, permutation, generator=None):
    """
    Applies the given permutation to the pixels of the image.

    Args:
        img (torch.Tensor): Image tensor
        target (torch.Tensor): Target tensor
        corruption_prob (float): Probability of corruption
        permutation (torch.Tensor): Permutation of the pixels
        apply_corruption (bool): If True, the corruption is applied to the returned image
    """
    # permutated idx are original indices
    permutation_pixels = torch.arange(img.size(1) * img.size(2))
    c, h, w = img.size()

    # check if the permutation size matches the image size
    assert permutation.size(0) == h * w, "Permutation size does not match image size"

    corrupted = False
    if torch.rand(1, generator=generator) <= corruption_prob:
        corrupted = True
        permutation_pixels = permutation

        # apply it to the image
        img = apply_pixel_permutation(img, permutation_pixels)

    return img, target, torch.tensor(corrupted, dtype=torch.bool)


def apply_pixel_permutation(img, pixel_perm):
    """
    Applies the given permutation of the pixels to the image.
    """
    c, w, h = img.size()

    permutation_as_img = pixel_perm.repeat(c, 1).view(c, -1).long()
    permutated_img = img.view(c, -1).gather(1, permutation_as_img).view(c, h, w)
    return permutated_img


def undo_permutation(permuted_img, applied_pixel_perm):
    """
    Undoes the given permutation of the pixels to a permutated image.
    """
    c, w, h = permuted_img.size()

    true_order = torch.empty_like(applied_pixel_perm)
    true_order[applied_pixel_perm] = torch.arange(applied_pixel_perm.size(0))

    return permuted_img.view(c, -1)[:, true_order].view(c, w, h)
