"""
Create a smaller version of the Inception network for CIFAR10 as proposed by the paper.

Author: Stepp1
"""

import torch
import torchvision
from torch import nn


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        use_batch_norm=False,
        activation_layer=nn.ReLU,
    ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        self.act = activation_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, **kwargs):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvModule(
            in_channels,
            out_1x1,
            kernel_size=1,
            stride=1,
            padding=0,
            **kwargs,
        )
        self.conv3 = ConvModule(
            in_channels,
            out_3x3,
            kernel_size=3,
            stride=1,
            padding=1,
            **kwargs,
        )

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv3(x)
        return torch.cat([out_1, out_2], 1)


class DownsampleModule(nn.Module):
    def __init__(self, in_channels, out_3x3, **kwargs):
        super(DownsampleModule, self).__init__()
        self.conv = ConvModule(
            in_channels, out_3x3, kernel_size=3, stride=2, padding=0, **kwargs
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out_1 = self.conv(x)
        out_2 = self.maxpool(x)
        return torch.cat([out_1, out_2], 1)


class InceptionSmall(nn.Module):
    """
    Inception Small as shown in the Appendix A of the paper.

    The implementation follows the blocks from Figure 3.
    """

    def __init__(
        self, num_outputs=10, use_batch_norm=False, use_dropout=False, dropout_rate=0.5
    ):
        super(InceptionSmall, self).__init__()
        self.num_outputs = num_outputs
        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout
        inception_module_kwargs = {
            "use_batch_norm": self.use_batch_norm,
            "activation_layer": nn.LeakyReLU,
        }

        self.conv1 = ConvModule(3, 96, kernel_size=3, stride=1, padding=0)
        self.inception1 = nn.Sequential(
            InceptionModule(96, 32, 32, **inception_module_kwargs),
            InceptionModule(64, 32, 48, **inception_module_kwargs),
            DownsampleModule(80, 80, **inception_module_kwargs),
        )
        self.inception2 = nn.Sequential(
            InceptionModule(160, 112, 48, **inception_module_kwargs),
            InceptionModule(160, 96, 64, **inception_module_kwargs),
            InceptionModule(160, 80, 80, **inception_module_kwargs),
            InceptionModule(160, 48, 96, **inception_module_kwargs),
            DownsampleModule(144, 96, **inception_module_kwargs),
        )
        self.inception3 = nn.Sequential(
            InceptionModule(240, 176, 160, **inception_module_kwargs),
            InceptionModule(336, 176, 160, **inception_module_kwargs),
        )

        self.mean_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate) if self.use_dropout else nn.Identity(),
            nn.Linear(16464, 384),
            nn.LeakyReLU(),
            nn.Linear(384, 192),
            nn.Dropout(dropout_rate) if self.use_dropout else nn.Identity(),
            nn.LeakyReLU(),
            nn.Linear(192, self.num_outputs),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.mean_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def inception(weights=None, cifar=False, **kwargs):
    if cifar:
        return InceptionSmall(**kwargs)
    return torchvision.models.get_model("inception_v3", weights=weights)


if __name__ == "__main__":
    model = inception(cifar=True)
    model.cpu()
    x = torch.randn(1, 3, 28, 28)
    x.cpu()
    _ = model(x)
