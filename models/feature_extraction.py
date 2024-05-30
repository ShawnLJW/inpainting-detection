"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .DnCNN import make_net


class SRMFilter(nn.Module):
    def __init__(self):
        super().__init__()
        f1 = [
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0],
        ]

        f2 = [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ]

        f3 = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]

        q = torch.tensor([[4.0], [12.0], [2.0]]).unsqueeze(-1).unsqueeze(-1)
        filters = (
            torch.tensor([[f1, f1, f1], [f2, f2, f2], [f3, f3, f3]], dtype=torch.float)
            / q
        )
        self.register_buffer("filters", filters)
        self.truc = nn.Hardtanh(-2, 2)

    def forward(self, x):
        x = F.conv2d(x, self.filters, padding="same", stride=1)
        x = self.truc(x)
        return x


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = torch.ones(self.in_channels, self.out_channels, 1) * -1.000

        super().__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(
            torch.rand(self.in_channels, self.out_channels, kernel_size**2 - 1),
            requires_grad=True,
        )

    def bayarConstraint(self):
        self.kernel.data = self.kernel.data.div(self.kernel.data.sum(-1, keepdims=True))
        ctr = self.kernel_size**2 // 2
        real_kernel = torch.cat(
            (
                self.kernel[:, :, :ctr],
                self.minus1.to(self.kernel.device),
                self.kernel[:, :, ctr:],
            ),
            dim=2,
        )
        real_kernel = real_kernel.reshape(
            (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        )
        return real_kernel

    def forward(self, x):
        x = F.conv2d(
            x, self.bayarConstraint(), stride=self.stride, padding=self.padding
        )
        return x


class Noiseprint(nn.Module):
    def __init__(self):
        super().__init__()
        self.noiseprint = make_net(
            3,
            kernels=[3] * 17,
            features=[64] * (17 - 1) + [1],
            bns=[False] + [True] * (17 - 2) + [False],
            acts=["relu"] * (17 - 1) + ["linear"],
            dilats=[1] * 17,
            bn_momentum=0.1,
            padding=1,
        )

    def init_weights(self, pretrained_path):
        dat = torch.load(pretrained_path, map_location=torch.device("cpu"))
        self.noiseprint.load_state_dict(dat)
        self.noiseprint.eval()
        for param in self.noiseprint.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.noiseprint(x)