from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from src.models.backbone.resnet import ResNetBackbone
from src.models.neck.fpn import FPN
from src.models.heads.detection_head import DetectionHead
from src.models.heads.segmentation_head import SegmentationHead


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        model_cfg = cfg["model"]
        task_cfg = cfg["tasks"]

        backbone_name = model_cfg["backbone"]
        in_channels = model_cfg["in_channels"]
        pretrained = model_cfg.get("pretrained", False)
        neck_out_channels = model_cfg["neck"]["out_channels"]

        self.enable_detection = task_cfg["detection"]["enabled"]
        self.enable_segmentation = task_cfg["segmentation"]["enabled"]

        if backbone_name != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone = ResNetBackbone(
            in_channels=in_channels,
            pretrained=pretrained,
        )

        self.neck = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=neck_out_channels,
        )

        if self.enable_detection:
            self.detection_head = DetectionHead(
                in_channels=neck_out_channels,
                num_classes=task_cfg["detection"]["num_classes"],
            )
        else:
            self.detection_head = None

        if self.enable_segmentation:
            self.segmentation_head = SegmentationHead(
                in_channels=neck_out_channels,
                num_classes=task_cfg["segmentation"]["num_classes"],
            )
        else:
            self.segmentation_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        backbone_feats = self.backbone(x)
        neck_feats = self.neck(backbone_feats)

        outputs: Dict[str, Any] = {}

        if self.detection_head is not None:
            outputs["detection"] = self.detection_head(neck_feats)

        if self.segmentation_head is not None:
            outputs["segmentation"] = self.segmentation_head(
                neck_feats,
                output_size=(x.shape[-2], x.shape[-1]),
            )

        return outputs