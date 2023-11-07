from torchvision.models import AlexNet

from .alexnet import SmallAlexNet, alexnet
from .inception import InceptionSmall, inception
from .mlp import MLP, mlp
from .resnet import resnet

__all__ = [
    "AlexNet",
    "SmallAlexNet",
    "InceptionSmall",
    "MLP",
    "alexnet",
    "inception",
    "mlp",
    "resnet",
]
