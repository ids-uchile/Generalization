import equinox as eqx
import jax


class ConvModule(eqx.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvModule, self).__init__()
        self.conv = eqx.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = eqx.nn.BatchNorm2d(out_channels)
        self.act = jax.nn.relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InceptionModule(eqx.Module):
    def __init__(self, in_channels, out_1x1, out_3x3):
        super(InceptionModule, self).__init__()
        self.conv1 = ConvModule(
            in_channels, out_1x1, kernel_size=1, stride=1, padding=0
        )
        self.conv3 = ConvModule(
            in_channels, out_3x3, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        out_1 = self.conv1(x)
        out_2 = self.conv3(x)
        # return torch.cat([out_1, out_2], 1)


class DownsampleModule(eqx.Module):
    def __init__(self, in_channels, out_3x3):
        super(DownsampleModule, self).__init__()
        self.conv = ConvModule(in_channels, out_3x3, kernel_size=3, stride=2, padding=0)
        self.maxpool = eqx.nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out_1 = self.conv(x)
        out_2 = self.maxpool(x)
        # return torch.cat([out_1, out_2], 1)


class InceptionSmall(eqx.Module):
    """
    Inception Small as shown in the Appendix A of the paper.

    The implementation follows the blocks from Figure 3.
    """

    def __init__(self):
        super(InceptionSmall, self).__init__()
        self.conv1 = ConvModule(3, 96, kernel_size=3, stride=1, padding=0)
        self.inception1 = eqx.nn.Sequential(
            InceptionModule(96, 32, 32),
            InceptionModule(64, 32, 48),
            DownsampleModule(80, 80),
        )
        self.inception2 = eqx.nn.Sequential(
            InceptionModule(160, 112, 48),
            InceptionModule(160, 96, 64),
            InceptionModule(160, 80, 80),
            InceptionModule(160, 48, 96),
            DownsampleModule(144, 96),
        )
        self.inception3 = eqx.nn.Sequential(
            InceptionModule(240, 176, 160),
            InceptionModule(336, 176, 160),
        )

        self.mean_pool = eqx.nn.AdaptiveAvgPool2d((7, 7))

        self.fc = eqx.nn.Sequential(
            eqx.nn.Linear(16464, 384),
            eqx.nn.Linear(384, 192),
            eqx.nn.Linear(192, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = self.mean_pool(x)
        # x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def create_small():
    return InceptionSmall()


def create_inception(small=False):
    raise NotImplementedError
