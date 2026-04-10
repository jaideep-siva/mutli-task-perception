# Detection Baseline: Low-Data Training Pipeline

This directory contains an enhanced Faster R-CNN detection training pipeline optimized for low-data scenarios (~600 labeled images).

## Key Features

- **Pretrained model initialization** with backbone freezing for better low-data performance
- **Albumentations-based augmentation** with bbox-aware transforms (not destructive to small objects)
- **Detection-specific metrics**: mAP@[0.5:0.95], mAP@0.5, mAP@0.75, precision, recall
- **Best model selection** based on mAP, not just loss
- **Fixed validation previews** every epoch for qualitative monitoring
- **TensorBoard logging** and PNG plots for all metrics
- **Train/val/test splits** with optional group-aware splitting by sequence/video
- **Low-data sampling strategies**: weighted sampling and repeat-factor sampling
- **K-fold cross-validation** support for robustness evaluation
- **Optional bootstrap ensemble mode** for aggregating multiple trained models
- **CSV and JSON metrics logging** for easy post-hoc analysis

## Installation

Install required dependencies:

```bash
pip install torch torchvision albumentations torchmetrics tensorboard matplotlib numpy pillow pyyaml
```

## Quick Start

### Standard Training

```bash
python scripts/train_detector_demo.py
```

This runs a single training run with:
- Pretrained ResNet50-FPN backbone (first 3 epochs frozen)
- Moderate Albumentations augmentation
- 70% train / 15% val / 15% test split (deterministic seed)
- mAP-based best model selection
- TensorBoard logging to `outputs/tensorboard/`
- Checkpoint saved as `outputs/checkpoints/best.pth`

### Advanced Training: Weighted Sampling

For datasets with severe class imbalance, use inverse class frequency weighting:

```bash
# Edit configs/train/detector_demo.yaml:
#   use_weighted_sampling: true
#   weighted_sampling:
#     pow_factor: 0.5        # Smoothness factor
#     max_weight_cap: 10.0   # Prevent pathological oversampling

python scripts/train_detector_demo.py
```

This increases the sampling probability of images containing rare classes, allowing the model to see them more frequently during training without explicit over/undersampling.

### Advanced Training: Repeat-Factor Sampling

Alternative to weighted sampling, using Detectron2-style repeat factors:

```bash
# Edit configs/train/detector_demo.yaml:
#   use_repeat_factor_sampling: true
#   repeat_factor_sampling:
#     repeat_factor_threshold: 0.001

python scripts/train_detector_demo.py
```

This repeats entire images based on the rarity of classes they contain. Images with rare classes appear multiple times per epoch.

**Note**: Use only one of weighted_sampling or repeat_factor_sampling at a time.

### K-Fold Cross-Validation (Robustness Analysis)

Train and evaluate across all 5 folds:

```bash
# Train fold 0
python scripts/train_detector_demo.py

# Edit configs/train/detector_demo.yaml:
#   use_kfold: true
#   num_folds: 5
#   fold_index: 0

# Then increment fold_index and repeat for fold 1, 2, 3, 4
```

Collect and compare results across folds to assess model robustness on the limited data.

### Advanced: Group-Aware Splitting

If your dataset has Natural grouping (sequences, videos, sources), keep entire groups in the same split:

```bash
# Edit configs/train/detector_demo.yaml:
#   group_split_key: "video_id"  # or "sequence_id", "source_id", etc.

python scripts/train_detector_demo.py
```

The script will attempt to extract group IDs from dataset samples automatically. This prevents data leakage where the validation set contains frames from the same video as training data.

### Evaluation

Evaluate a trained model on validation or test set:

```bash
# Evaluate best checkpoint on val split
python evaluate_detector.py outputs/checkpoints/best.pth \
  --data-config configs/data/insta360_detection.yaml \
  --split val \
  --score-thresh 0.5 \
  --output-dir outputs/eval_best

# Evaluate on all splits (train, val, test)
python evaluate_detector.py outputs/checkpoints/best.pth \
  --split all \
  --output-dir outputs/eval_all
```

Output: JSON file with metrics + PNG visualizations of detections.

## Configuration

Key configuration in `configs/train/detector_demo.yaml`:

### Model

```yaml
pretrained: true                  # Use pretrained ResNet50-FPN weights
freeze_backbone_epochs: 3         # Freeze backbone for first N epochs
trainable_backbone_layers: 3      # Number of backbone layers to keep trainable overall
```

### Training

```yaml
epochs: 1000
lr: 0.0001
weight_decay: 0.0001
amp: true                         # Mixed precision training (faster on NVIDIA GPU)
gradient_clip_norm: 10.0          # Prevent exploding gradients
```

### Scheduling and Early Stopping

```yaml
scheduler_metric: val/loss        # or "val/map_50", "val/map"
early_stopping_metric: val/map_50 # Metric to monitor for early stopping (usually mAP)
early_stopping_patience: 10       # Epochs without improvement before stopping
```

### Split Configuration

```yaml
split_ratios:
  train: 0.7
  val: 0.15
  test: 0.15
group_split_key: null            # null for simple random split, or key name for group-aware
```

### Augmentation

```yaml
augmentation:
  p_flip: 0.5                    # Horizontal flip probability
  p_affine: 0.3                  # Affine transform probability (scale, translate, rotate, shear)
  p_brightness: 0.3              # Brightness/contrast adjustment probability
  p_blur: 0.1                    # Blur probability
  p_noise: 0.05                  # Gaussian noise probability
```

All augmentations are **bbox-aware** and preserve label information. Invalid boxes (e.g., those that fall outside the image after transforms) are dropped safely.

### Low-Data Strategies

```yaml
use_weighted_sampling: false      # Enable inverse class frequency weighting
use_repeat_factor_sampling: false # Enable Detectron2-style repeat factors
```

Only one can be enabled at a time.

## Output Structure

After training, the output directory contains:

```
outputs/
├── checkpoints/
│   ├── best.pth                    # Best model (by mAP)
│   ├── last.pth                    # Last epoch
│   ├── detector_epoch_001.pth
│   ├── detector_epoch_002.pth
│   └── metrics.csv                 # Per-epoch metrics
├── tensorboard/                    # TensorBoard logs
│   └── events.out.tfevents...
├── predictions/
│   └── epoch_001_preview.png       # Per-epoch validation preview
├── plots/
│   ├── loss_comparison.png         # Train loss vs val loss
│   ├── map_curves.png              # mAP50 and mAP50-95
│   └── training_curves_*.png
└── metrics.jsonl                   # Per-epoch metrics (JSON lines)
```

Monitor with TensorBoard:

```bash
tensorboard --logdir outputs/tensorboard --port 6006
```

Then open http://localhost:6006 in your browser.

## Metrics Explained

- **train/total_loss**: Sum of all loss terms during training
- **val/loss**: Validation loss (computed on val set in training mode)
- **val/map**: mAP@[0.5:0.95] (COCO standard)
- **val/map_50**: mAP@IoU=0.5
- **val/map_75**: mAP@IoU=0.75
- **val/mar_100**: Mean Average Recall (max 100 detections per image)
- **val/precision**: Per-image precision (simplified metric)
- **val/recall**: Per-image recall (simplified metric)

## Checkpoint Format

Each `.pth` file contains:

```python
{
    "model_state_dict": {...},        # Model weights
    "optimizer_state_dict": {...},    # Optimizer state
    "epoch": 42,                      # Epoch number
    "class_map": {...},               # Class ID to name mapping
    "image_size": [640, 1280],        # Resized image size
    "config": {...},                  # Training config
}
```

Load and use:

```python
import torch
from src.models.demo_detector import build_demo_detector

checkpoint = torch.load("outputs/checkpoints/best.pth", map_location="cpu")
class_map = checkpoint["class_map"]
image_size = checkpoint["image_size"]
config = checkpoint["config"]

model = build_demo_detector(num_classes=len(class_map), pretrained=True)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Architecture Notes

- Model: `torchvision.fasterrcnn_resnet50_fpn_v2`
- Backbone freezing: Supports trainable_backbone_layers config for fine-grained control
- Augmentation: Albumentations via `src/data/detection_transforms.py`
- Metrics: torchmetrics MeanAveragePrecision (COCO-style evaluation)
- Splits: Deterministic with seed for reproducibility, group-aware optional
- Sampling: Weighted or repeat-factor (mutually exclusive)

## Limitations and Future Work

1. **Ensemble mode** is currently optional/advanced. Integration into standard pipeline could be explored.
2. **Soft-NMS** is not currently used. Could be added to repetitive detection fusion.
3. **Per-class sampling** could be more sophisticated (e.g., focal loss, OHEM).
4. **Test-time augmentation** (TTA) is not implemented but could boost final metrics.

## Troubleshooting

### GPU Out of Memory

- Reduce `batch_size` in `configs/data/insta360_detection.yaml`
- Reduce `image_size` if dataset permits
- Disable `amp: false` (but this will be slower)

### Poor Validation Metrics

- Increase `freeze_backbone_epochs` for stronger regularization
- Reduce learning rate
- Enable weighted or repeat-factor sampling if class imbalance is severe
- Check that validation set is representative

### Slow Training

- Increase `num_workers` in data config (if available CPU cores)
- Enable `amp: true` (default) on CUDA
- Reduce `log_every` to print less frequently

## References

- Faster R-CNN: https://arxiv.org/abs/1506.01497
- Torchvision FPN: https://arxiv.org/abs/1612.03144
- Albumentations: https://albumentations.ai/
- Detectron2 Repeat-Factor Sampling: https://github.com/facebookresearch/detectron2
- COCO mAP Metric: https://cocodataset.org/

---

**Questions or issues?** Check the training logs and TensorBoard for debugging clues.
