import os
import random

import numpy as np
import torch
import torch.nn.functional as F


def get_num_cpus():
    return len(os.sched_getaffinity(0))


def seed_everything(seed) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def collate_drop_return_index(batch):
    """
    Drops the return index from the batch

    Parameters:
    -----------
        batch (list): list of tuples (x, y, index)

    Returns:
    --------
        x, y (Tuple[torch.Tensor, torch.Tensor]): batch of data
    """
    x, y, _ = list(zip(*batch))

    return (torch.stack(x), torch.stack(y))


def compute_el2n_scores(logits, y, num_classes=10):
    """Compute EL2N scores.

    Parameters:
        outputs: Torch.Tensor
            Logits from the model
        y: Torch.Tensor
            Corresponding labels
        num_classes: int
            Number of classes in the dataset. Default: 10.

    Returns:
        scores: Torch.Tensor
    """

    probs, targets = F.softmax(logits, dim=-1), F.one_hot(y, num_classes=num_classes)
    scores = torch.linalg.vector_norm(probs - targets, dim=-1)
    return scores.cpu()
