"""Torchvision Faster R-CNN baseline used by the detection demo."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_demo_detector(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int | None = None,
) -> nn.Module:
    """Builds a Faster R-CNN detector and swaps the ROI classifier head.

    Args:
        num_classes: Total number of classes including the background class.
        pretrained: If True, loads pretrained weights from torchvision.
        trainable_backbone_layers: Number of backbone layers to keep trainable.
            If None, all layers are trainable. Only used when pretrained=True.

    Returns:
        A ``torchvision`` ``FasterRCNN`` model ready for training or inference.
    """

    weights = "DEFAULT" if pretrained else None
    weights_backbone = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=weights_backbone
    )

    # Replace ROI head classifier for dataset class count
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Configure trainable backbone layers if specified
    if trainable_backbone_layers is not None and pretrained:
        backbone = model.backbone
        if hasattr(backbone, "body"):  # ResNet backbone
            # Freeze earlier layers, keep later layers trainable
            # ResNet has: layer0 (stem), layer1, layer2, layer3, layer4
            layers_to_freeze = 4 - trainable_backbone_layers  # e.g., if trainable=3, freeze=1
            if hasattr(backbone.body, "layer1") and layers_to_freeze >= 1:
                for param in backbone.body.layer1.parameters():
                    param.requires_grad = False
            if hasattr(backbone.body, "layer2") and layers_to_freeze >= 2:
                for param in backbone.body.layer2.parameters():
                    param.requires_grad = False
            if hasattr(backbone.body, "layer3") and layers_to_freeze >= 3:
                for param in backbone.body.layer3.parameters():
                    param.requires_grad = False
            if hasattr(backbone.body, "layer4") and layers_to_freeze >= 4:
                for param in backbone.body.layer4.parameters():
                    param.requires_grad = False

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freezes all backbone parameters for the detector.

    Args:
        model: A Faster R-CNN model with a backbone.
    """
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreezes all backbone parameters for the detector.

    Args:
        model: A Faster R-CNN model with a backbone.
    """
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = True
