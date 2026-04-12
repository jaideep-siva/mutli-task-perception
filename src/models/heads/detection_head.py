from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class DetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.pred_dim = 4 + 1 + num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        out_channels = self.num_anchors * self.pred_dim
        self.head_p3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.head_p4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.head_p5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def _reshape_level(self, pred: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = pred.shape
        pred = pred.view(batch_size, self.num_anchors, self.pred_dim, height, width)
        pred = pred.permute(0, 3, 4, 1, 2).contiguous()
        return pred.view(batch_size, height * width * self.num_anchors, self.pred_dim)

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3 = self.stem(feats["p3"])
        p4 = self.stem(feats["p4"])
        p5 = self.stem(feats["p5"])

        return {
            "p3": self._reshape_level(self.head_p3(p3)),
            "p4": self._reshape_level(self.head_p4(p4)),
            "p5": self._reshape_level(self.head_p5(p5)),
        }
