from typing import Any

from flax import linen as nn
from jax import numpy as jnp


class AlexNet(nn.Module):
    num_classes: int = 1000
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = False):
        """Translate code from https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py"""
        x = jnp.array(x, dtype=self.dtype)
        x = nn.Sequential(
            [
                nn.Conv(
                    features=64,
                    kernel_size=(11, 11),
                    strides=(4, 4),
                    padding=(2, 2),
                    dtype=self.dtype,
                ),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
                nn.Conv(
                    features=192, kernel_size=(5, 5), padding=(2, 2), dtype=self.dtype
                ),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
                nn.Conv(
                    features=384, kernel_size=(3, 3), padding=(1, 1), dtype=self.dtype
                ),
                nn.relu,
                nn.Conv(
                    features=256, kernel_size=(3, 3), padding=(1, 1), dtype=self.dtype
                ),
                nn.relu,
                nn.Conv(
                    features=256, kernel_size=(3, 3), padding=(1, 1), dtype=self.dtype
                ),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(3, 3), strides=(2, 2)),
            ]
        )(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        logits = nn.Sequential(
            [
                nn.Dense(features=4096, dtype=self.dtype),
                nn.relu,
                nn.Dropout(0.5, deterministic=not training),
                nn.Dense(features=4096, dtype=self.dtype),
                nn.relu,
                nn.Dropout(0.5, deterministic=not training),
                nn.Dense(features=self.num_classes, dtype=self.dtype),
            ]
        )(x)


class SmallAlexNet(nn.Module):
    num_classes: int = 10
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = jnp.array(x, dtype=self.dtype)
        x = nn.Sequential(
            [
                nn.Conv(
                    features=64,
                    kernel_size=(3, 3),
                    strides=2,
                    padding=1,
                    dtype=self.dtype,
                ),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(2, 2)),
                nn.Conv(features=192, kernel_size=(5, 5), dtype=self.dtype),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(2, 2)),
                nn.Conv(features=384, kernel_size=(3, 3), dtype=self.dtype),
                nn.relu,
                nn.Conv(features=256, kernel_size=(3, 3), dtype=self.dtype),
                nn.relu,
                nn.Conv(features=256, kernel_size=(3, 3), dtype=self.dtype),
                nn.relu,
                lambda x: nn.max_pool(x, window_shape=(2, 2)),
            ]
        )(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        logits = nn.Sequential(
            [
                nn.Dense(features=4096, dtype=self.dtype),
                nn.relu,
                nn.Dropout(0.5, deterministic=not training),
                nn.Dense(features=4096, dtype=self.dtype),
                nn.relu,
                nn.Dropout(0.5, deterministic=not training),
                nn.Dense(features=self.num_classes, dtype=self.dtype),
            ]
        )(x)
        return logits


def alexnet(cifar=False, **kwargs: Any):
    if cifar:
        return SmallAlexNet(**kwargs)

    return AlexNet(**kwargs)
