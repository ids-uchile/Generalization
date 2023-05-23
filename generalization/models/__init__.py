"""
From the original paper (adapted):
1. Imagenet: Inception V3 (Szegedy et al., 2016)
2. CIFAR10: A smaller version of Inception,
            Alexnet (Krizhevsky et al., 2012),
            MLPs with 1 and 3 hidden layers

Author: Stepp1
"""

import torch
import torchvision

from .inception import create_small as create_small_inception
from .mlp import create_mlp


def create_resnet(resnet_size=18, weights=None, cifar=False):
    resnet = "resnet"
    model = torchvision.models.get_model(resnet + resnet_size, weights=weights)

    if cifar:
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
    return model


def create_alexnet(weights=None, cifar=False):
    model = torchvision.models.alexnet(weights=weights)

    if cifar:
        model.features[0] = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
    return model


def create_inception(weights=None, cifar=False):
    if cifar:
        model = create_small_inception()
    else:
        model = torchvision.models.inception_v3(weights=weights)
    return model


def get_cifar_models():
    return {
        "inception_small": create_inception(cifar=True),
        "alexnet": create_alexnet(cifar=True),
        "mlp_1x512": create_mlp(32 * 32 * 3, [512], 10),
        "mlp_3x512": create_mlp(32 * 32 * 3, [512] * 3, 10),
    }


def get_imagenet_models():
    return {
        "inception": create_inception(),
    }
