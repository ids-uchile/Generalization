import torch.nn as nn
import torchvision

NUM_CLASSES = 10


class SmallAlexNet(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        use_batch_norm=False,
        use_dropout=False,
        dropout_rate=0.5,
    ):
        super(SmallAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(4096, num_classes),
        )
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(weights=None, cifar=False, **kwargs):
    if cifar:
        return SmallAlexNet(**kwargs)
    return torchvision.models.get_model("alexnet", weights=weights)
