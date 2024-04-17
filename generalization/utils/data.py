import os
import random

import numpy as np
import torch


def get_num_cpus():
    return len(os.sched_getaffinity(0))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
