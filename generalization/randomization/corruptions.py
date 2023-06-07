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


import torch


def random_labels(
    img, target, corruption_prob, get_random_label, apply_corruption=False
):
    """
    Randomizes the labels of the dataset.

    Args:
        img (torch.Tensor): Image tensor
        target (torch.Tensor): Target tensor
        corruption_prob (float): Probability of corruption
        get_random_label (callable): Function that returns a random label
        apply_corruption (bool): If True, the corruption is applied to the returned image
    """
    random_label = target
    if torch.rand(1) <= corruption_prob:
        random_label = get_random_label(target)

        if apply_corruption:
            target = random_label

    return img, target, random_label


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

    if torch.rand(1) <= corruption_prob:
        # choose different random permutation for each image
        permutation_pixels = torch.randperm(permutation_size)

        # apply it to the image
        if apply_corruption:
            img = apply_pixel_permutation(img, permutation_pixels)

    return img, target, permutation_pixels


def shuffled_pixels(img, target, corruption_prob, permutation, apply_corruption=False):
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

    if torch.rand(1) <= corruption_prob:
        # choose different random permutation for each image
        permutation_pixels = permutation

        # apply it to the image
        if apply_corruption:
            img = apply_pixel_permutation(img, permutation_pixels)

    return img, target, permutation_pixels


def gaussian_pixels(img, target, corruption_prob, apply_corruption=False, cifar=False):
    c, w, h = img.size()

    sampled = None
    if torch.rand(1) <= corruption_prob:
        if cifar:
            from .utils import CIFAR_MEAN as mean
            from .utils import CIFAR_STD as std
        else:
            from .utils import IMAGENET_MEAN as mean
            from .utils import IMAGENET_STD as std

        sampled_channels = []
        for i in range(c):
            sampled_channels.append(torch.normal(mean[i], std[i], size=(w, h)))

        sampled = torch.cat(sampled_channels, dim=0).unsqueeze(0)
        if apply_corruption:
            img = sampled

    return img, target, sampled


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
