from generalization.randomization.dataset import RandomizedDataset
from torchvision.datasets import CIFAR10, ImageNet


def available_corruptions():
    return [
        "random_labels",
        "partial_labels",
        "gaussian_pixels",
        "random_pixels",
        "shuffled_pixels",
    ]


def create_corrupted_dataset(
    dataset_name,
    corruption_name,
    corruption_prob=0.3,
    train=True,
    root="./data/cifar10",
    apply_corruption=False,
    return_corruption=False,
    transform=None,
    target_transform=None,
):
    if dataset_name.lower() == "imagenet":
        dataset = ImageNet(root=root, download=True, train=train)
    elif dataset_name.lower() == "cifar10":
        dataset = CIFAR10(root=root, download=True, train=train)
    else:
        raise ValueError("Dataset name must be either 'imagenet' or 'cifar10'")

    return RandomizedDataset(
        dataset,
        corruption_name=corruption_name,
        corruption_prob=corruption_prob,
        apply_corruption=apply_corruption,
        return_corruption=return_corruption,
        train=train,
        transform=transform,
        target_transform=target_transform,
    )
