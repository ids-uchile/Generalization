"""
From the original paper (adapted):
1. Imagenet: Inception V3 (Szegedy et al., 2016)
2. CIFAR10: A smaller version of Inception,
            Alexnet (Krizhevsky et al., 2012),
            MLPs with 1 and 3 hidden layers

Author: Stepp1
"""


from functools import partial

import equinox as eqx
import torch
import torchvision

import eqxvision

from .torch_models.inception import InceptionSmall


class ModelFactory:
    def __init__(self):
        self.lib: str = None
        self.model_creators_torch = {
            "resnet18": partial(create_resnet, resnet_size=18, lib="torch"),
            "resnet34": partial(create_resnet, resnet_size=34, lib="torch"),
            "alexnet": partial(create_alexnet, resnet_size=34, lib="torch"),
            "mlp": partial(create_mlp, lib="torch"),
            "inception": partial(create_inception, lib="torch"),
        }

        self.model_creators_jax = {
            "resnet18": partial(create_resnet, resnet_size=18, lib="jax"),
            "resnet34": partial(create_resnet, resnet_size=34, lib="jax"),
            "alexnet": partial(create_alexnet, resnet_size=34, lib="jax"),
            "mlp_1x512": partial(
                create_mlp,
                in_size=32 * 32 * 3,
                hidden_sizes=[512],
                out_size=10,
                lib="jax",
            ),
            "mlp_3x512": partial(
                create_mlp,
                in_size=32 * 32 * 3,
                hidden_sizes=[512] * 3,
                out_size=10,
                lib="jax",
            ),
            "inception": partial(create_inception, lib="jax"),
        }

    def create_model(self, model_type: str, lib: str = None, **kwargs):
        if model_type not in self.model_creators:
            raise ValueError(f"Unknown model type: {model_type}")

        self.lib = lib or self.lib

        return self.get_model(model_type, **kwargs)

    def get_model(self, model_type, **kwargs):
        if self.lib == "jax":
            model = self.model_creators_jax[model_type](**kwargs)

        elif self.lib == "torch":
            model = self.model_creators_torch[model_type](**kwargs)

        else:
            raise ValueError(f"Unknown library: {self.lib}")

        return model

    def get_cifar_models(self, lib: str = "jax"):
        self.lib = lib or self.lib

        return {
            "alexnet": self.create_model("alexnet", cifar=True),
            # "small_inception": self.create_model("inception", cifar=True),
            "mlp_1x512": self.create_model("mlp_1x512"),
            "mlp_3x512": self.create_model("mlp_3x512"),
        }

    def get_imagenet_models(self, lib: str = "jax"):
        self.lib = lib or self.lib

        return {
            "alexnet": self.create_model("alexnet"),
            "inception": self.create_model("inception"),
            "resnet18": self.create_model("resnet18"),
            "resnet34": self.create_model("resnet34"),
        }


def create_mlp(in_size, hidden_sizes, out_size, lib="torch"):
    if lib == "jax":
        from .jax.mlp import create_mlp

        model = create_mlp(in_size, hidden_sizes, out_size)

    elif lib == "torch":
        from .torch.mlp import create_mlp

        model = create_mlp(in_size, hidden_sizes, out_size)

    else:
        raise ValueError(f"Unknown library: {lib}")

    return model


def create_resnet(resnet_size=18, weights=None, cifar=False, lib="torch"):
    resnet = "resnet"
    if lib == "jax":
        raise NotImplementedError

    elif lib == "torch":
        model = torchvision.models.get_model(resnet + resnet_size, weights=weights)
        cifar_conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

    else:
        raise ValueError(f"Unknown library: {lib}")

    if cifar:
        model.conv1 = cifar_conv1
    return model


def create_alexnet(weights=None, cifar=False, lib="torch"):
    if lib == "jax":
        model = eqxvision.models.alexnet(weights=weights)
        cifar_conv1 = eqx.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

    elif lib == "torch":
        model = torchvision.models.get_model("alexnet", weights=weights)
        cifar_conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

    else:
        raise ValueError(f"Unknown library: {lib}")

    if cifar:
        model.conv1 = cifar_conv1

    return model


def create_inception(weights=None, cifar=False, small="False", lib="torch"):
    cifar = small or cifar
    if lib == "jax":
        from .jax.inception import create_inception

        if cifar:
            model = create_inception(small=True)

        else:
            raise NotImplementedError

    elif lib == "torch":
        if cifar:
            model = InceptionSmall()
            cifar = False
        else:
            model = torchvision.models.get_model("inception_v3", weights=weights)
            cifar_conv1 = torch.nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )

    else:
        raise ValueError(f"Unknown library: {lib}")

    if cifar:
        model.conv1 = cifar_conv1
    return model
