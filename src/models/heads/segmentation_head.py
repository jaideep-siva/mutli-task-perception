from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def get_seg_loss(name: str, num_classes: int) -> nn.Module:
    loss_name = name.lower()
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_name == "bce":
        if num_classes != 1:
            raise ValueError("BCE segmentation loss expects num_classes=1.")
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported segmentation loss '{name}'.")


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        loss_name: str = "cross_entropy",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.loss_fn = get_seg_loss(loss_name, num_classes)
        fused_channels = in_channels * 3

        self.fuse = nn.Sequential(
            nn.Conv2d(fused_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, kernel_size=1),
        )

    def forward(
        self,
        feats: Dict[str, torch.Tensor],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        p3, p4, p5 = feats["p3"], feats["p4"], feats["p5"]

        p4_up = F.interpolate(p4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        p5_up = F.interpolate(p5, size=p3.shape[-2:], mode="bilinear", align_corners=False)

        fused = torch.cat([p3, p4_up, p5_up], dim=1)
        logits = self.fuse(fused)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        expected_shape = (logits.shape[0], self.num_classes, output_size[0], output_size[1])
        assert logits.shape == expected_shape, (
            f"Segmentation logits must be {expected_shape}, got {tuple(logits.shape)}"
        )
        return logits

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
            if target.ndim == 3:
                target = target.unsqueeze(1)
            return self.loss_fn(logits, target.float())

        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        return self.loss_fn(logits, target.long())
