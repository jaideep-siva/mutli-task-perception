from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn

from src.models.backbone.backbone import build_backbone
from src.models.detection_postprocess import postprocess_detections
from src.models.neck.fpn import FPN
from src.models.heads.detection_head import DetectionHead
from src.models.heads.segmentation_head import SegmentationHead
from src.models.loss import MultiTaskLossController


class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()

        model_cfg = cfg["model"]
        task_cfg = cfg["tasks"]
        loss_cfg = cfg.get("loss", {})

        neck_out_channels = model_cfg["neck"]["out_channels"]

        self.enable_detection = task_cfg["detection"]["enabled"]
        self.enable_segmentation = task_cfg["segmentation"]["enabled"]
        self.detection_cfg = dict(task_cfg.get("detection", {}))
        self.segmentation_cfg = dict(task_cfg.get("segmentation", {}))
        self.backbone = build_backbone(model_cfg)

        self.neck = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=neck_out_channels,
        )

        if self.enable_detection:
            self.detection_head = DetectionHead(
                in_channels=neck_out_channels,
                num_classes=task_cfg["detection"]["num_classes"],
                num_anchors=task_cfg["detection"].get("num_anchors", 1),
            )
        else:
            self.detection_head = None

        if self.enable_segmentation:
            self.segmentation_head = SegmentationHead(
                in_channels=neck_out_channels,
                num_classes=task_cfg["segmentation"]["num_classes"],
                loss_name=task_cfg["segmentation"].get("loss", "cross_entropy"),
            )
        else:
            self.segmentation_head = None

        self.loss_controller = MultiTaskLossController(
            weighting=str(loss_cfg.get("loss_weighting", "fixed")),
            detection_weight=float(loss_cfg.get("detection_weight", 1.0)),
            segmentation_weight=float(loss_cfg.get("segmentation_weight", 1.0)),
        )

    def forward(self, x: torch.Tensor, postprocess: bool = False) -> Dict[str, Any]:
        backbone_feats = self.backbone(x)
        neck_feats = self.neck(backbone_feats)

        outputs: Dict[str, Any] = {}

        if self.detection_head is not None:
            raw_detection = self.detection_head(neck_feats)
            if postprocess:
                feature_shapes = {
                    level: tuple(feat.shape[-2:]) for level, feat in neck_feats.items()
                }
                outputs["detection"] = postprocess_detections(
                    raw_outputs=raw_detection,
                    feature_shapes=feature_shapes,
                    image_size=(x.shape[-2], x.shape[-1]),
                    config=self.detection_cfg.get("postprocess"),
                )
            else:
                outputs["detection"] = raw_detection

        if self.segmentation_head is not None:
            outputs["segmentation"] = self.segmentation_head(
                neck_feats,
                output_size=(x.shape[-2], x.shape[-1]),
            )

        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, Any],
        segmentation_target: torch.Tensor | None = None,
        detection_loss: Any = None,
    ) -> Dict[str, torch.Tensor]:
        seg_loss = None
        if self.segmentation_head is not None and segmentation_target is not None:
            seg_loss = self.segmentation_head.compute_loss(
                outputs["segmentation"],
                segmentation_target,
            )
        return self.loss_controller(
            detection_loss=detection_loss,
            segmentation_loss=seg_loss,
        )
