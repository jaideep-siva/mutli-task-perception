# Multitask Perception Research Repository

This repository contains a multitask perception stack for robotics research. The current mainline model predicts object detections and lane/ground segmentation from equirectangular camera frames. The repository is in a strong development state: the core model and experimentation pipeline are mostly built, but deployment is not complete until the model is retrained on actual robot data and exported/integrated through ONNX, TensorRT, and ROS.

## Section A: How to Use This Repository

### Project Overview

This project is for training and evaluating a perception model that can support a robot operating from wide-field camera input. It separates model code, dataset loading, training/evaluation logic, experiment launchers, preprocessing tools, and future deployment-facing demos.

The current system can:

- Load manifest-based multitask datasets with RGB images, detection boxes, segmentation masks, confidence scores, and ROI masks.
- Train a PyTorch multitask model with a backbone, FPN, detection head, and segmentation head.
- Evaluate detection and segmentation metrics when labels are available.
- Generate or convert lane supervision artifacts for equirectangular frames.
- Run controlled backbone comparison experiments.
- Run a live OpenCV/PyTorch demo scaffold through `live_inference_demo/`.

Still pending:

- Retraining on data from the actual robot.
- ONNX export.
- TensorRT conversion.
- ROS wrapper or ROS2 integration.

### Repo Blueprint

Plain-language map first:

- `configs/`: settings files. Change these before changing Python code.
- `data/`: dataset loading code and lightweight data documentation. Large local data should stay ignored by Git.
- `models/`: the canonical multitask neural network.
- `engine/`: training, validation, metrics, checkpointing, and MLflow hooks.
- `scripts/`: command-line entry points that call the real code.
- `tools/`: data preparation and lane supervision utilities.
- `tests/`: smoke tests that check the main pipeline still runs.
- `docs/`: folder guides, learning material, reports, and legacy documentation.
- `live_inference_demo/`: OpenCV demo scaffold for live/video inference experiments.
- `src/`: legacy Faster R-CNN detector demo path. Keep it working, but do not treat it as the canonical multitask path.
- `outputs/`, `artifacts/`, `mlruns/`, `mlflow.db`: generated results, checkpoints, metrics, and experiment tracking state. These are ignored by Git.
- `archive/`: non-mainline preserved material.

Technical mainline:

```text
configs/multitask/multitask_resnet18.yaml
scripts/train.py
scripts/run_experiments.py
data/*
models/*
engine/*
tools/equirect_lane_pipeline/*
```

### Getting Started From Zero

Prerequisites:

- Python 3.10 or newer is recommended.
- A working PyTorch installation.
- CUDA is optional but strongly recommended for real training.
- Basic command-line familiarity.

Create and activate an environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the expected Python packages. The exact environment has not yet been frozen into a top-level lockfile, so use the imports in the repo as the current dependency source:

```bash
pip install torch torchvision torchmetrics pyyaml pillow numpy opencv-python mlflow albumentations pytest
```

Optional packages:

```bash
pip install wandb tensorboard matplotlib
```

Recommended reading order before running anything:

1. `README.md`
2. `docs/folder_guides/README.md`
3. `docs/learning/repo_orientation_for_beginners.md`
4. `configs/multitask/multitask_resnet18.yaml`
5. `scripts/train.py`
6. `data/dataset.py`
7. `models/multitask_model.py`
8. `engine/train.py`
9. `docs/report/project_report.md`

### Quickstart Workflows

Dataset preparation:

```bash
python tools/equirect_lane_pipeline/build_multitask_manifest.py \
  --images_dir path/to/images \
  --detection_annotations path/to/detections.json \
  --lane_records path/to/lane_records.jsonl \
  --train_manifest outputs/my_run/train_manifest.json \
  --val_manifest outputs/my_run/val_manifest.json \
  --val_ratio 0.2 \
  --require_segmentation \
  --shared_roi_mask path/to/roi_mask.png
```

Training:

```bash
python scripts/train.py \
  --config configs/multitask/multitask_resnet18.yaml \
  --train_manifest outputs/my_run/train_manifest.json \
  --val_manifest outputs/my_run/val_manifest.json \
  --output_dir outputs/my_run/resnet18 \
  --device auto
```

Fast smoke training:

```bash
python scripts/train.py \
  --config configs/multitask/multitask_resnet18.yaml \
  --train_manifest outputs/my_run/train_manifest.json \
  --val_manifest outputs/my_run/val_manifest.json \
  --output_dir outputs/smoke \
  --epochs 1 \
  --batch_size 1 \
  --max_steps_per_epoch 1 \
  --device cpu
```

Backbone comparison:

```bash
python scripts/run_experiments.py \
  --config configs/multitask/multitask_resnet18.yaml \
  --train_manifest outputs/my_run/train_manifest.json \
  --val_manifest outputs/my_run/val_manifest.json \
  --output_dir outputs/comparisons/my_backbone_sweep \
  --batch_size 1 \
  --device auto
```

Validation/evaluation happens automatically during training when `dataset.val_manifest` is set. The evaluation code lives in `engine/eval.py`.

Inference/demo:

```bash
cd live_inference_demo
python app.py --video path/to/demo.mp4
```

Configs live under `configs/`. Outputs, logs, checkpoints, generated masks, comparisons, and MLflow runs should go under `outputs/`, `artifacts/`, `mlruns/`, or another ignored experiment directory.

### If You Are Totally New

A config file is a YAML file that stores settings such as model type, image size, learning rate, batch size, and dataset paths. In this repo, configs let you change experiments without rewriting Python code.

A checkpoint is a saved copy of model weights plus training metadata. You use checkpoints to resume work, compare experiments, or run inference.

Training means showing labeled examples to the model so it can update its weights.

Inference means using a trained model to make predictions on new images or video.

Evaluation means measuring predictions against labels. For this repo, that can include detection mAP and segmentation IoU/Dice, depending on which labels exist.

Export means converting a trained PyTorch model into a deployment-friendly format such as ONNX and then TensorRT.

Scripts and modules relate like this: files in `scripts/` are thin command-line doors. They call reusable code in `data/`, `models/`, `engine/`, and `tools/`.

### Common Pitfalls And Troubleshooting

- Path issues: prefer paths relative to the repo root unless a script says otherwise.
- Empty manifests: `data.MultitaskDataset` expects a non-empty JSON list or an object with a non-empty `samples` list.
- Detection metrics unavailable: if validation samples have empty `boxes` and `labels`, detection mAP cannot support backbone selection.
- Segmentation masks not aligned: check resize settings and inspect lane QC previews before training.
- Config mismatch: make sure `segmentation.num_classes`, mask values, and `detection.num_classes` match the dataset.
- CUDA unavailable: `device: auto` falls back to CPU, but real training may be slow.
- Generated output clutter: `outputs/`, `artifacts/`, `mlruns/`, and `mlflow.db` are generated and ignored; do not treat them as source.
- Live demo checkpoint loading: `live_inference_demo/` is a demo scaffold, not the final ROS integration.

### Recommended Workflow For New Contributors

Read first:

- `docs/folder_guides/README.md`
- `docs/learning/repo_orientation_for_beginners.md`
- `docs/report/project_report.md`
- `configs/multitask/multitask_resnet18.yaml`

Safe to modify:

- New configs under `configs/`.
- New documentation under `docs/`.
- New experiment output folders under `outputs/`.
- Conservative additions to tests.

Change carefully:

- `models/loss.py`
- `models/detection_postprocess.py`
- `data/dataset.py`
- `engine/train.py`
- `engine/eval.py`
- Legacy detector code under `src/`, because it supports the older demo path.

Do not casually change:

- Manifest schema.
- Checkpoint format.
- Model output keys.
- ROI mask semantics.
- Class counts without updating configs and labels together.

## Section B: Project Status And Development Checklist

### Overall Status

- [x] Core model building mostly complete.
- [x] Canonical multitask training entry point exists.
- [x] Manifest-driven dataset loader exists.
- [x] Segmentation ROI masking and confidence-weighted supervision exist.
- [x] Backbone comparison tooling exists.
- [x] Legacy detector path preserved.
- [~] Evaluation flow exists, but available experiment evidence is limited by label coverage.
- [~] Live inference demo exists as a scaffold, not a deployment integration.
- [ ] Retraining on actual robot data.
- [ ] ONNX conversion.
- [ ] TensorRT conversion.
- [ ] ROS wrapper / integration.

### Data Pipeline Status

- [x] Manifest schema supports images, boxes, labels, segmentation masks, confidence, and ROI masks.
- [x] Equirectangular lane preprocessing tools exist.
- [x] Manual tape-label manifests exist in generated outputs.
- [~] Current observed manual tape-label split is small: 38 train samples and 9 validation samples.
- [ ] Actual robot data retraining dataset still needs to be collected/curated.
- [ ] Dataset versioning policy still needs to be formalized.

### Model Architecture Status

- [x] Canonical multitask model uses backbone + FPN + detection head + segmentation head.
- [x] ResNet-18 baseline is configured.
- [x] ConvNeXt-Base and Swin-B backbones are available for comparison.
- [~] `hrnet_w32` is a lightweight fallback, not a true HRNet implementation.
- [ ] Deployment-specific model constraints for ONNX/TensorRT still need validation.

### Training Status

- [x] `scripts/train.py` trains the canonical multitask stack.
- [x] Training writes checkpoints and metrics summaries.
- [x] MLflow hooks are present when MLflow is installed.
- [~] Current comparison evidence appears to come from small manual lane/tape data, not final robot data.
- [ ] Full retraining on actual robot data is pending.

### Evaluation Status

- [x] Segmentation IoU and Dice are implemented.
- [x] Detection mAP is wired through TorchMetrics.
- [~] Current manual tape-label comparison reports detection mAP as unavailable because the observed manifests contain empty detection boxes.
- [ ] Final detection evaluation requires robot data with ground-truth boxes.
- [ ] Final deployment acceptance metrics need to be defined.

### Deployment/Export Status

- [~] Live inference demo scaffold exists.
- [ ] ONNX conversion is pending.
- [ ] TensorRT conversion is pending.
- [ ] Runtime benchmarking on target hardware is pending.
- [ ] Model export tests are pending.

### Robotics Integration Status

- [~] Repository is organized with deployment in mind.
- [~] Live demo proves basic video/camera plumbing concepts.
- [ ] ROS wrapper / integration is pending.
- [ ] Robot sensor calibration and message contracts are pending.
- [ ] End-to-end robot validation is pending.

### Documentation Status

- [x] Top-level README rewritten for beginners and maintainers.
- [x] Folder-level technical guides added under `docs/folder_guides/`.
- [x] Learning documents added under `docs/learning/`.
- [x] Current-state technical report added under `docs/report/`.
- [~] Dependency/environment freeze remains pending.

### Recommended Next Steps

1. Freeze the Python environment into a top-level `requirements.txt` or equivalent lockfile.
2. Collect and curate actual robot data with detection and segmentation labels.
3. Retrain the ResNet-18 baseline on robot data.
4. Re-run backbone comparisons with complete labels.
5. Add ONNX export and export validation tests.
6. Add TensorRT conversion and latency benchmarking.
7. Build the ROS wrapper around the validated exported model.
8. Run robot-in-the-loop evaluation and document acceptance criteria.

### Risks, Blockers, And Technical Debt

- Current strongest experiment evidence is from a small manual tape-label dataset, not final robot data.
- Detection mAP is unavailable for observed manual tape-label comparisons because boxes are empty.
- ONNX/TensorRT compatibility has not been validated.
- Live inference is a demo scaffold and should not be confused with ROS deployment.
- Some generated output directories contain locked temp folders on Windows; they should remain ignored and can be cleaned outside active runs.
- There is not yet a top-level frozen dependency file.
