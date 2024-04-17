import torch
import torchvision


def resnet(resnet_size=18, weights=None, cifar=False, **kwargs):
    resnet = "resnet"

    if "use_batch_norm" in kwargs and kwargs["use_batch_norm"]:
        norm_layer = torch.nn.BatchNorm2d
    else:
        norm_layer = torch.nn.Identity

    model = torchvision.models.get_model(
        resnet + str(resnet_size), weights=weights, norm_layer=norm_layer
    )
    if cifar:
        model.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.fc = torch.nn.Linear(512, 10)
    return model
