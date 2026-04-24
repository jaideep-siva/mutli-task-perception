# Configs Folder Guide

## Purpose

`configs/` stores YAML files that control datasets, model settings, training settings, and lane-label backend settings. A config is the preferred place to change experiment behavior before editing Python code.

## What Lives Here

- `configs/multitask/multitask_resnet18.yaml`: canonical multitask baseline.
- `configs/multitask/legacy_model_schema.yaml`: preserved schema reference from an older model/config style.
- `configs/train/detector_demo.yaml`: legacy Faster R-CNN detector demo training config.
- `configs/data/insta360_detection.yaml`: legacy detector dataset config.
- `configs/lane_backends/*.yaml`: OpenCV lane extractor config and templates for external lane adapters.
- `configs/config.yaml`: older/general config retained for compatibility.

## How Files Interact

`scripts/train.py` loads `configs/multitask/multitask_resnet18.yaml`, applies CLI overrides, and passes the resulting dictionary to `engine.train.train`. The training engine then constructs `data.MultitaskDataset`, `models.MultiTaskPerceptionModel`, and `models.MultiTaskLoss`.

Lane backend configs are consumed by `tools/equirect_lane_pipeline/generate_lane_masks.py`.

## Inputs And Outputs

Inputs are human-authored YAML values. Outputs are not written back to `configs/`; training writes checkpoints and metrics under `outputs/`, `artifacts/`, and MLflow directories.

## Entry Points

- `scripts/train.py --config configs/multitask/multitask_resnet18.yaml`
- `scripts/run_experiments.py --config configs/multitask/multitask_resnet18.yaml`
- `tools/equirect_lane_pipeline/generate_lane_masks.py --config configs/lane_backends/opencv_equirect_lane.yaml`

## Safe To Edit

- Learning rate, batch size, epochs, `num_workers`, and output-related fields.
- Dataset manifest paths.
- Augmentation probabilities.
- Lane backend threshold values.

## Change Carefully

- `detection.num_classes` and `segmentation.num_classes`; these must match labels and checkpoint expectations.
- `dataset.image_size`; this affects box scaling, masks, memory use, and export dimensions.
- Backbone names; only registered names in `models/backbone.py` are valid.
- Loss weights; they change training behavior.

## Typical Workflow

1. Copy or edit a YAML config.
2. Point the dataset manifest fields to train/validation manifests.
3. Run a smoke training job.
4. Run the full training or comparison job.
5. Save outputs outside source folders.

## Known Gaps

- No top-level locked environment file exists yet.
- Config schema validation is informal; invalid keys may fail late.
- Deployment/export-specific config fields still need to be defined.
