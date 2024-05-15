import torch

from .. import corruptions
from .. import utils as dataset_utils


@corruptions.add_randomization("gaussian_pixels")
def gaussian_pixels(
    img, target, corruption_prob, shape, use_cifar=False, generator=None
):
    w = shape[1]
    h = shape[2]

    sampled = None
    corrupted = False
    if torch.rand(1) <= corruption_prob:
        corrupted = True
        if use_cifar:
            mean = dataset_utils.CIFAR10_CHANNEL_MEAN
            std = dataset_utils.CIFAR10_CHANNEL_STD
        else:
            mean = dataset_utils.IMAGENET_CHANNEL_MEAN
            std = dataset_utils.IMAGENET_CHANNEL_STD

        normal_dist = torch.distributions.Normal(torch.tensor(mean), torch.tensor(std))
        sampled = normal_dist.sample((w, h)).round().clamp(0, 255).type(torch.uint8)
        img = sampled.numpy()

    return img, target, torch.tensor(corrupted, dtype=torch.bool)
