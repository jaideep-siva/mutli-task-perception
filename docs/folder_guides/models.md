# Models Folder Guide

## Purpose

`models/` contains the canonical multitask perception model and loss code.

## What Lives Here

- `models/multitask_model.py`: combines backbone, FPN, detection head, and segmentation head.
- `models/backbone.py`: registered backbone builders for ResNet, ConvNeXt, Swin, and lightweight HRNet-style fallback.
- `models/fpn.py`: feature pyramid neck.
- `models/detection_head.py`: dense detection prediction head.
- `models/detection_postprocess.py`: detection decoding and NMS.
- `models/segmentation_head.py`: segmentation decoder head.
- `models/segmentation_roi.py`: ROI-aware segmentation prediction helpers.
- `models/loss.py`: detection and segmentation loss composition.
- `models/__init__.py`: public package exports.

## How Files Interact

`MultiTaskPerceptionModel` calls `build_backbone`, feeds backbone features through `FPN`, and sends pyramid features to detection and segmentation heads. During training, `MultiTaskLoss` consumes raw model outputs and target dictionaries assembled by `engine/train.py`.

## Inputs And Outputs

Input:

- Image tensor `[B, 3, H, W]`.
- Config dictionary from `configs/multitask/*.yaml`.

Output:

- `detection`: raw multi-level detection logits.
- `segmentation`: segmentation logits `[B, C, H, W]`.
- `detections`: postprocessed boxes/scores/labels when `postprocess=True`.

## Entry Points

- `models.MultiTaskPerceptionModel`
- `models.MultiTaskLoss`
- `models.backbone.build_backbone`

## Safe To Edit

- Comments that explain tensor contracts.
- New registered backbones when tests are added.
- Export-friendly helper wrappers, once ONNX work begins.

## Change Carefully

- Output dictionary keys.
- Checkpoint state dict structure.
- Detection postprocess assumptions.
- Segmentation ROI handling.
- Loss scaling and reduction behavior.

## Typical Workflow

1. Select a backbone in config.
2. Build model in `engine.train.train`.
3. Train with raw outputs (`postprocess=False`).
4. Evaluate by decoding detections and computing segmentation metrics.

## Known Gaps

- ONNX export has not been validated.
- TensorRT conversion has not been validated.
- `hrnet_w32` is explicitly a lightweight fallback, not a full HRNet implementation.
