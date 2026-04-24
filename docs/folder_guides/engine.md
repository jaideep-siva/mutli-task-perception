# Engine Folder Guide

## Purpose

`engine/` contains reusable training, evaluation, metrics, checkpointing, and logging logic for the canonical multitask pipeline.

## What Lives Here

- `engine/train.py`: training loop, device resolution, YAML loading, dataloader construction, checkpoint writing, and MLflow logging.
- `engine/eval.py`: detection and segmentation evaluation.
- `engine/metrics.py`: metric helpers.
- `engine/__init__.py`: package marker.

## How Files Interact

`scripts/train.py` and `scripts/run_experiments.py` call into `engine.train`. The training loop builds datasets from `data/`, model/loss objects from `models/`, and evaluates through `engine.eval`.

## Inputs And Outputs

Inputs:

- Config dictionary.
- Train and validation manifests.
- Optional CLI overrides.

Outputs:

- Checkpoints under `checkpoint_root`.
- Per-class IoU artifacts under `artifacts/<run_name>/`.
- MLflow logs when MLflow is installed.
- Summary dictionary returned to the caller.

## Entry Points

- `engine.train.train`
- `engine.train.build_dataloader`
- `engine.eval.evaluate`
- `engine.eval.compute_seg_iou`

## Safe To Edit

- Logging messages.
- Additional summary fields.
- Validation checks that fail early with clear errors.

## Change Carefully

- Checkpoint keys.
- Metric names.
- Device fallback behavior.
- Scheduler stepping.
- Best checkpoint selection.

## Typical Workflow

1. CLI script loads config.
2. Engine builds dataloaders and model.
3. Engine trains for configured epochs.
4. Engine evaluates on validation data.
5. Engine writes checkpoints and metrics.

## Known Gaps

- Best model selection currently combines detection mAP and segmentation IoU; this should be revisited once final robot labels and acceptance criteria exist.
- Dependency behavior differs when MLflow is absent.
- No ONNX export engine exists yet.
