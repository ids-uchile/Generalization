import torch
import torch.nn.functional as F


def compute_batch_scores(outputs, batch, num_classes=10):
    """Compute EL2N scores for a batch of data.

    Parameters:
        outputs: Dict[str, torch.Tensor]
            Dictionary of model outputs. Must contain the following keys:
                `loss`, `logits`.
        batch: Torch.Tensor
            Batch of data. Must be a tuple of `(x, y)`.
        num_classes: int
            Number of classes in the dataset. Default: 10.

    Returns:
        batch_scores: Torch.Tensor
    """
    probs, targets = F.softmax(outputs["logits"], dim=-1), F.one_hot(
        batch[1], num_classes=num_classes
    )
    batch_scores = torch.linalg.vector_norm(probs - targets, dim=-1)
    return batch_scores.cpu()
