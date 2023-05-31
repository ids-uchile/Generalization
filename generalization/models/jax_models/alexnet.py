from typing import Any

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.random as jrandom

from eqxvision.models.classification.alexnet import alexnet as imagenet_alexnet

NUM_CLASSES = 10


class SmallAlexNet(eqx.Module):
    """Small AlexNet for CIFAR."""

    features: eqx.Module
    avgpool: eqx.Module
    classifier: eqx.Module

    def __init__(self, dropout=0.5, num_classes=NUM_CLASSES, *, key=None):
        super().__init__()
        if key is None:
            key = jrandom.PRNGKey(0)
        keys = jrandom.split(key, 8)
        self.features = nn.Sequential(
            [
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, key=keys[0]),
                nn.Lambda(jnn.relu),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=1, key=keys[1]),
                nn.Lambda(jnn.relu),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1, key=keys[2]),
                nn.Lambda(jnn.relu),
                nn.Conv2d(384, 256, kernel_size=3, padding=1, key=keys[3]),
                nn.Lambda(jnn.relu),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, key=keys[4]),
                nn.Lambda(jnn.relu),
                nn.MaxPool2d(kernel_size=2),
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            [
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096, key=keys[5]),
                nn.Lambda(jnn.relu),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096, key=keys[6]),
                nn.Lambda(jnn.relu),
                nn.Linear(4096, num_classes, key=keys[7]),
            ]
        )

    def __call__(self, x, key=None):
        if key is not None:
            keys = jrandom.split(key, 2)
        x = self.features(x, key=keys[0])
        x = self.avgpool(x)
        x = jax.numpy.ravel(x)
        x = self.classifier(x, key=keys[1])
        return x

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


def alexnet(torch_weights: str = None, cifar=False, **kwargs: Any):
    if cifar:
        return SmallAlexNet(**kwargs)

    return imagenet_alexnet(torch_weights, **kwargs)
