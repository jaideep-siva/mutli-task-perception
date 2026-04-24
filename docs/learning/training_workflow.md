# Training Workflow

## Overview

Training turns labeled examples into model weights. In this repo, training is controlled by a config file and a pair of train/validation manifests.

## Step 1: Prepare Manifests

Use tools or your own dataset process to create JSON manifests. Each sample can include:

- image path
- detection boxes
- detection labels
- segmentation mask path
- segmentation confidence
- segmentation ROI mask

## Step 2: Check The Config

Open `configs/multitask/multitask_resnet18.yaml` and verify:

- `dataset.train_manifest`
- `dataset.val_manifest`
- `dataset.image_size`
- `detection.num_classes`
- `segmentation.num_classes`
- `batch_size`
- `epochs`
- `device`

## Step 3: Run A Smoke Test

Run one epoch and one step before a full run:

```bash
python scripts/train.py \
  --config configs/multitask/multitask_resnet18.yaml \
  --train_manifest path/to/train_manifest.json \
  --val_manifest path/to/val_manifest.json \
  --epochs 1 \
  --batch_size 1 \
  --max_steps_per_epoch 1 \
  --device cpu
```

## Step 4: Run Full Training

```bash
python scripts/train.py \
  --config configs/multitask/multitask_resnet18.yaml \
  --train_manifest path/to/train_manifest.json \
  --val_manifest path/to/val_manifest.json \
  --output_dir outputs/my_run \
  --device auto
```

## Step 5: Inspect Outputs

Check:

- training logs
- checkpoint directory
- per-class IoU JSON files
- MLflow output if enabled
- comparison summaries if running sweeps

## How To Interpret Metrics

- Detection mAP measures object detection quality and requires ground-truth boxes.
- Segmentation IoU measures pixel overlap between predicted and true segmentation masks.
- Dice is another overlap metric that can be easier to interpret for small foreground regions.
- Latency and throughput are rough runtime indicators from evaluation, not final deployment benchmarks.

## Common Failure Modes

- Manifest paths point to missing files.
- Masks use unexpected class values.
- `num_classes` does not match labels.
- CPU training is slow.
- Validation set has no detection boxes, so detection mAP is unavailable.

## What Counts As A Good Training Run

A useful run has:

- a known dataset version
- a saved config
- clear train/validation split
- checkpoints
- validation metrics
- notes about missing labels or unavailable metrics
