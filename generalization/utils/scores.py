import torch
import torch.nn.functional as F


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
