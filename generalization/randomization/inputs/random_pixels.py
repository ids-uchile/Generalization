import torch
from generalization.randomization.corruptions import add_randomization
from generalization.randomization.labels.shuffled_pixels import apply_pixel_permutation


@add_randomization("random_pixels")
def random_pixels(
    img, target, corruption_prob, permutation_size, apply_corruption=False
):
    """
    Applies a random permutation to the pixels of the image.

    Args:
        img (torch.Tensor): Image tensor
        target (torch.Tensor): Target tensor
        corruption_prob (float): Probability of corruption
        permutation_size (int): Size of the permutation, e.g. 32x32 = 1024
        apply_corruption (bool): If True, the corruption is applied to the returned image
    """
    # permutated idx are original indices
    permutation_pixels = torch.arange(permutation_size)
    c, h, w = img.size()

    # check if the permutation size matches the image size
    assert permutation_size == h * w, "Permutation size does not match image size"

    corrupted = False
    if torch.rand(1) <= corruption_prob:
        corrupted = True
        # choose different random permutation for each image
        permutation_pixels = torch.randperm(permutation_size)

        # apply it to the image
        if apply_corruption:
            img = apply_pixel_permutation(img, permutation_pixels)

    return img, target, corrupted
