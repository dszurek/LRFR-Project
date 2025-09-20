# edgeface.py
# Architecture from the official EdgeFace GitHub repository README.md
import torch
import torch.nn as nn


class Conv_BN_HSwish(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups=1,
        with_se=False,
    ):
        super(Conv_BN_HSwish, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.hswish = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.hswish(x)
        return x


class LDC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        groups,
        with_se=False,
    ):
        super(LDC, self).__init__()
        self.conv_1x1_in = Conv_BN_HSwish(in_channels, groups, 1, 1, 0)
        self.conv_dw = Conv_BN_HSwish(
            groups, groups, kernel_size, stride, padding, groups=groups
        )
        self.conv_1x1_out = nn.Sequential(
            nn.Conv2d(groups, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.conv_1x1_in(x)
        x = self.conv_dw(x)
        x = self.conv_1x1_out(x)
        return x


class EdgeFace(nn.Module):
    def __init__(self, embedding_size=512, back="edgeface_s"):
        super(EdgeFace, self).__init__()
        if back == "edgeface_s":
            self.features = nn.Sequential(
                Conv_BN_HSwish(3, 32, 3, 2, 1),
                LDC(32, 64, 3, 2, 1, 64),
                LDC(64, 64, 3, 1, 1, 128),
                LDC(64, 128, 3, 2, 1, 128),
                LDC(128, 128, 3, 1, 1, 256),
                LDC(128, 128, 3, 1, 1, 256),
                LDC(128, 256, 3, 2, 1, 256),
                LDC(256, 256, 3, 1, 1, 512),
                LDC(256, 256, 3, 1, 1, 512),
                LDC(256, 256, 3, 1, 1, 512),
                LDC(256, 256, 3, 1, 1, 512),
                Conv_BN_HSwish(256, 512, 1, 1, 0),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, embedding_size, bias=False),
                nn.BatchNorm1d(embedding_size),
            )
        # Add other backbones like 'edgeface_m' if needed

    def forward(self, x):
        x = self.features(x)
        return x
