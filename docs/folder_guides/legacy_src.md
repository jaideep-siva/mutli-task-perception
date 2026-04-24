# Legacy Source Folder Guide

## Purpose

`src/` is the preserved legacy Faster R-CNN detector/demo implementation. It is not the canonical multitask stack, but it remains useful for historical reference, detection-only experiments, and compatibility with older scripts.

## What Lives Here

- `src/data/`: detection dataset, transforms, samplers, and split utilities.
- `src/models/`: legacy detector and older multitask-style modules.
- `src/utils/`: config loading, detection metrics, plots, and visualization utilities.

## How Files Interact

`scripts/train_detector_demo.py`, `scripts/infer_detector_demo.py`, and `evaluate_detector.py` import from `src/`. The canonical multitask path does not depend on `src/`.

## Inputs And Outputs

Inputs are legacy detector configs, LabelMe-style detection annotations, and image data. Outputs include detector checkpoints, TensorBoard logs, PNG plots, metrics JSON/CSV, and preview images.

## Entry Points

- `scripts/train_detector_demo.py`
- `scripts/infer_detector_demo.py`
- `evaluate_detector.py`

## Safe To Edit

- Documentation.
- Bug fixes needed to preserve the legacy path.
- Tests that prove the legacy path still imports.

## Change Carefully

- Anything that affects checkpoint loading for older detector runs.
- Dataset split behavior.
- Metric computation.

## Typical Workflow

Use only when you intentionally need the older detector baseline:

```bash
python scripts/train_detector_demo.py
python evaluate_detector.py outputs/checkpoints/best.pth --split val
```

## Known Gaps

- This path is separate from the main multitask model.
- Documentation has been moved to `docs/legacy/detection_baseline.md`.
- Future deployment work should target the canonical `models/` stack unless a detector-only deployment is explicitly chosen.
