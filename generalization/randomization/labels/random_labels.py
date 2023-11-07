import torch
from generalization.randomization.corruptions import add_randomization


@add_randomization("random_labels")
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
    corrupted = False
    if torch.rand(1) <= corruption_prob:
        corrupted = True
        random_label = get_random_label(target)

        if apply_corruption:
            target = random_label

    return img, target, corrupted
