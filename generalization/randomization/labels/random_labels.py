import torch

from ..corruptions import add_randomization


@add_randomization("random_labels")
def random_labels(img, target, corruption_prob, get_random_label):
    """
    Randomizes the labels of the dataset.

    Args:
        img (torch.Tensor): Image tensor
        target (torch.Tensor): Target tensor
        corruption_prob (float): Probability of corruption
            Not used in this function, but maintained for consistency.
        get_random_label (callable): Function that returns a random label
        apply_corruption (bool): If True, the corruption is applied to the returned image
    """
    corrupted = True
    random_label = get_random_label(target)
    return img, random_label, torch.tensor(corrupted, dtype=torch.bool)
