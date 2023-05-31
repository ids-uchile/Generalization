import torch
import torchvision


def resnet(resnet_size=18, weights=None, cifar=False):
    resnet = "resnet"
    model = torchvision.models.get_model(resnet + resnet_size, weights=weights)
    if cifar:
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
    return model
