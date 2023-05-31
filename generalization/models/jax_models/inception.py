from typing import Callable

import equinox as eqx
import jax
from jax import numpy as jnp

from eqxvision.layers import ConvNormActivation


class ConvModule(eqx.Module):
    module: eqx.nn.Sequential

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, key=None
    ):
        super().__init__()
        if key is None:
            key = jax.random.PRNGKey(0)

        self.module = ConvNormActivation(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=eqx.nn.BatchNorm,
            activation_layer=jax.nn.relu,
            key=key,
        )

    def __call__(self, x, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        x = self.module(x, key=key)
        return x


class InceptionModule(eqx.Module):
    conv1: ConvModule
    conv3: ConvModule

    def __init__(self, in_channels, out_1x1, out_3x3, key=None):
        super().__init__()
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 2)

        self.conv1 = ConvModule(
            in_channels, out_1x1, kernel_size=1, stride=1, padding=0, key=keys[0]
        )
        self.conv3 = ConvModule(
            in_channels, out_3x3, kernel_size=3, stride=1, padding=1, key=keys[1]
        )

    def __call__(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv3(x)
        return jnp.concatenate([out_1, out_2], 1)


class DownsampleModule(eqx.Module):
    conv: ConvModule
    maxpool: eqx.Module

    def __init__(self, in_channels, out_3x3, key=None):
        super(DownsampleModule, self).__init__()
        if key is None:
            key = jax.random.PRNGKey(0)

        self.conv = ConvModule(
            in_channels, out_3x3, kernel_size=3, stride=2, padding=0, key=key
        )
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2)

    def __call__(self, x):
        out_1 = self.conv(x)
        out_2 = self.maxpool(x)
        return jnp.concatenate([out_1, out_2], 1)


class InceptionSmall(eqx.Module):
    """
    Inception Small as shown in the Appendix A of the paper.

    The implementation follows the blocks from Figure 3.
    """

    conv1: ConvModule
    inception1: eqx.nn.Sequential
    inception2: eqx.nn.Sequential
    inception3: eqx.nn.Sequential
    mean_pool: eqx.Module
    fc: eqx.nn.Sequential

    def __init__(self, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 14)

        super(InceptionSmall, self).__init__()
        self.conv1 = ConvModule(3, 96, kernel_size=3, stride=1, padding=0, key=keys[0])
        self.inception1 = eqx.nn.Sequential(
            [
                InceptionModule(96, 32, 32, key=keys[1]),
                InceptionModule(64, 32, 48, key=keys[2]),
                DownsampleModule(80, 80, key=keys[3]),
            ]
        )
        self.inception2 = eqx.nn.Sequential(
            [
                InceptionModule(160, 112, 48, key=keys[4]),
                InceptionModule(160, 96, 64, key=keys[5]),
                InceptionModule(160, 80, 80, key=keys[6]),
                InceptionModule(160, 48, 96, key=keys[7]),
                DownsampleModule(144, 96, key=keys[8]),
            ]
        )
        self.inception3 = eqx.nn.Sequential(
            [
                InceptionModule(240, 176, 160, key=keys[9]),
                InceptionModule(336, 176, 160, key=keys[10]),
            ]
        )

        self.mean_pool = eqx.nn.AdaptiveAvgPool2d((7, 7))

        self.fc = eqx.nn.Sequential(
            [
                eqx.nn.Linear(16464, 384, key=keys[11]),
                eqx.nn.Linear(384, 192, key=keys[12]),
                eqx.nn.Linear(192, 10, key=keys[13]),
            ]
        )

    def __call__(self, x, key=None):
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 4)

        x = self.conv1(x, key=keys[0])
        x = self.inception1(x, key=keys[1])
        x = self.inception2(x, key=keys[2])
        x = self.inception3(x, key=keys[3])
        x = self.mean_pool(x)
        x = jnp.ravel(x)
        x = self.fc(x, key=keys[4])
        return x


def inception(cifar=False, key=None):
    if cifar:
        return InceptionSmall(key)

    raise NotImplementedError
