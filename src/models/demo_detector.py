"""Torchvision Faster R-CNN baseline used by the detection demo."""

from __future__ import annotations

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_demo_detector(num_classes: int):
    """Builds a Faster R-CNN detector and swaps the ROI classifier head.

    Args:
        num_classes: Total number of classes including the background class.

    Returns:
        A ``torchvision`` ``FasterRCNN`` model ready for training or inference.
    """

    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
