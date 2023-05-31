"""
From the original paper (adapted):
1. Imagenet: Inception V3 (Szegedy et al., 2016)
2. CIFAR10: A smaller version of Inception,
            Alexnet (Krizhevsky et al., 2012),
            MLPs with 1 and 3 hidden layers

Author: Stepp1
"""


from functools import partial

import jax


class ModelFactory:
    def __init__(self):
        self.lib: str = None
        self.key = None
        self.model_creators_torch = {
            "resnet18": partial(create_resnet, resnet_size=18, lib="torch"),
            "resnet34": partial(create_resnet, resnet_size=34, lib="torch"),
            "alexnet": partial(create_alexnet, lib="torch"),
            "mlp_1x512": partial(
                create_mlp,
                in_size=28 * 28 * 3,
                hidden_sizes=[512],
                out_size=10,
                lib="torch",
            ),
            "mlp_3x512": partial(
                create_mlp,
                in_size=28 * 28 * 3,
                hidden_sizes=[512] * 3,
                out_size=10,
                lib="torch",
            ),
            "inception": partial(create_inception, lib="torch"),
        }

        self.model_creators_jax = {
            "resnet18": partial(create_resnet, resnet_size=18, lib="jax", key=self.key),
            "resnet34": partial(create_resnet, resnet_size=34, lib="jax", key=self.key),
            "alexnet": partial(create_alexnet, lib="jax", key=self.key),
            "mlp_1x512": partial(
                create_mlp,
                in_size=32 * 32 * 3,
                hidden_sizes=[512],
                out_size=10,
                lib="jax",
                key=self.key,
            ),
            "mlp_3x512": partial(
                create_mlp,
                in_size=32 * 32 * 3,
                hidden_sizes=[512] * 3,
                out_size=10,
                lib="jax",
                key=self.key,
            ),
            "inception": partial(create_inception, lib="jax", key=self.key),
        }

    def create_model(self, model_type: str, lib: str = None, **kwargs):
        self.lib = lib or self.lib

        return self.get_model(model_type, **kwargs)

    def get_model(self, model_type, **kwargs):
        if self.lib == "jax":
            model = self.model_creators_jax[model_type](key=self.key, **kwargs)

        elif self.lib == "torch":
            model = self.model_creators_torch[model_type](**kwargs)

        else:
            raise ValueError(f"Unknown library: {self.lib}")

        return model

    def get_cifar_models(self, model_name: str = None, lib: str = "jax", key=None):
        self.lib = lib or self.lib

        if self.lib == "jax" and key is None:
            self.key = jax.random.PRNGKey(42)
        elif self.lib == "jax" and key is not None:
            self.key = key

        models = {
            "alexnet": self.create_model("alexnet", cifar=True),
            "inception": self.create_model("inception", cifar=True),
            "mlp_1x512": self.create_model("mlp_1x512"),
            "mlp_3x512": self.create_model("mlp_3x512"),
        }
        return models if model_name is None else models[model_name]

    def get_imagenet_models(self, model_name: str = None, lib: str = "jax", key=None):
        self.lib = lib or self.lib
        self.key = key
        models = {
            "alexnet": self.create_model("alexnet"),
            "inception": self.create_model("inception"),
            "resnet18": self.create_model("resnet18"),
            "resnet34": self.create_model("resnet34"),
        }

        return models if model_name is None else models[model_name]


def create_mlp(in_size, hidden_sizes, out_size, lib="torch", key=None):
    if lib == "jax":
        from .jax_models.mlp import create_mlp

        model = create_mlp(in_size, hidden_sizes, out_size, key=key)

    elif lib == "torch":
        from .torch_models.mlp import create_mlp

        model = create_mlp(in_size, hidden_sizes, out_size)

    else:
        raise ValueError(f"Unknown library: {lib}")

    return model


def create_resnet(resnet_size=18, weights=None, cifar=False, lib="torch", key=None):
    if lib == "jax":
        raise NotImplementedError

    elif lib == "torch":
        from .torch_models.resnet import resnet

        model = resnet(resnet_size=resnet_size, weights=weights, cifar=cifar)

    else:
        raise ValueError(f"Unknown library: {lib}")

    return model


def create_alexnet(weights=None, cifar=False, lib="torch", key=None):
    if lib == "jax":
        from .jax_models.alexnet import alexnet

        model = alexnet(key=key, cifar=cifar)

    elif lib == "torch":
        from .torch_models.alexnet import alexnet

        model = alexnet(weights=weights, cifar=cifar)

    else:
        raise ValueError(f"Unknown library: {lib}")

    return model


def create_inception(weights=None, cifar=False, small="False", lib="torch", key=None):
    cifar = small or cifar
    if lib == "jax":
        from .jax_models.inception import inception

        model = inception(cifar=cifar, key=key)

    elif lib == "torch":
        from .torch_models.inception import inception

        model = inception(weights=weights, cifar=cifar)
    else:
        raise ValueError(f"Unknown library: {lib}")

    return model
