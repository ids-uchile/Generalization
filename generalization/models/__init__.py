from .factory import ModelFactory

get_imagenet_models = ModelFactory().get_imagenet_models
get_cifar_models = ModelFactory().get_cifar_models
create_model = ModelFactory().create_model

__all__ = [
    "get_imagenet_models",
    "get_cifar_models",
    "create_model",
]
