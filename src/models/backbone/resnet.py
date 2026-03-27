from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torchvision import models


class ResNetBackbone(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        if in_channels != 3:
            base.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        self.stem = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
        )
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

        self.out_channels = {
            "c2": 64,
            "c3": 128,
            "c4": 256,
            "c5": 512,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.stem(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return {
            "c2": c2,
            "c3": c3,
            "c4": c4,
            "c5": c5,
        }