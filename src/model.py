import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        skip_connection=None,
        is_pooling=False,
        is_normalization=True,
        activation=nn.ReLU(inplace=True),
    ):
        super(ConvBlock, self).__init__()
        self.skip_connection = skip_connection  # output of skip connection

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.MaxPool2d(2, 2) if is_pooling else nn.Identity(),
            nn.BatchNorm2d(out_channels) if is_normalization else nn.Identity(),
            activation,
        )

    def forward(self, x):
        if self.skip_connection is not None:
            return self.conv(x) + self.skip_connection(x)
        else:
            return self.conv(x)


class ChessNet(nn.Module):
    def __init__(
        self,
        num_channels=6,  # number of channels in input image
        num_classes=64,  # prob distribution over 8x8 board
        activation=nn.ReLU(),
    ):
        super(ChessNet, self).__init__()

        self.conv1 = ConvBlock(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=3,
            activation=activation,
            is_pooling=True,
            is_normalization=False,
        )
        self.conv2 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            activation=activation,
            is_pooling=True,
            is_normalization=False,
        )
        self.conv3 = ConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            activation=activation,
            is_pooling=True,
            is_normalization=False,
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            activation,
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)
