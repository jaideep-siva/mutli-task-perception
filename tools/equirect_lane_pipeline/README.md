# Equirectangular Lane Supervision Pipeline

This path assumes the production model ingests equirectangular frames directly. Frame extraction, pseudo-label generation, ROI masking, training, validation, and evaluation all stay in equirectangular image space. No perspective unwarping is performed by these tools.

## Workflow

### 1. Extract Frames

```bash
python tools/equirect_lane_pipeline/extract_frames.py \
  --input_video data/raw/drive.mp4 \
  --output_dir work/equirect_run/frames \
  --fps 5 \
  --resize_h 640 \
  --resize_w 1280
```

Outputs `frames/` and `metadata.jsonl`.

### 2. Build ROI Mask

Manual equirectangular driving band:

```bash
python tools/equirect_lane_pipeline/build_roi_masks.py \
  --image_h 640 \
  --image_w 1280 \
  --mode manual_band \
  --top 320 \
  --bottom 640 \
  --left 0 \
  --right 1280 \
  --output_mask work/equirect_run/roi/roi_mask.png
```

Polygon ROI:

```bash
python tools/equirect_lane_pipeline/build_roi_masks.py \
  --image_h 640 \
  --image_w 1280 \
  --mode polygon \
  --polygon_json roi_points.json \
  --output_mask work/equirect_run/roi/roi_mask.png
```

ROI mask values are 255 for valid lane-supervision pixels and 0 elsewhere.

### 3. Generate Lane Pseudo-Labels

```bash
python tools/equirect_lane_pipeline/generate_lane_masks.py \
  --backend clrnet \
  --frames_dir work/equirect_run/frames/frames \
  --output_dir work/equirect_run/lane_labels \
  --config configs/lane_backends/opencv_equirect_lane.yaml \
  --device auto \
  --score_threshold 0.6 \
  --mask_thickness 6 \
  --roi_mask work/equirect_run/roi/roi_mask.png \
  --save_overlay
```

Backends are adapter-based: `dummy`, `clrnet`, and `laneatt`. `dummy` is only for plumbing. `configs/lane_backends/opencv_equirect_lane.yaml` wires a runnable OpenCV-based automated extractor through the existing `clrnet` backend slot. `clrnet` and `laneatt` can also load project-specific neural adapters from `--config`; see `configs/lane_backends/*_adapter_template.yaml`.

The imported adapter must expose `predict(image_bgr)` and return one of:

```python
{"mask": mask_array, "confidence": 0.91}
{"lanes": [[(x0, y0), (x1, y1)]], "confidence": 0.91}
(lanes, confidence)
```

Dense masks are converted to binary values, and lane point sequences are rasterized with `--mask_thickness`.

### 4. Filter Lane Masks

```bash
python tools/equirect_lane_pipeline/filter_lane_masks.py \
  --input_records work/equirect_run/lane_labels/records.jsonl \
  --output_records work/equirect_run/lane_labels/filtered_records.jsonl \
  --min_confidence 0.6 \
  --min_lane_pixels 1 \
  --drop_empty_masks
```

### 5. Build Multitask Manifests

```bash
python tools/equirect_lane_pipeline/build_multitask_manifest.py \
  --images_dir work/equirect_run/frames/frames \
  --detection_annotations data/detections.json \
  --lane_records work/equirect_run/lane_labels/filtered_records.jsonl \
  --train_manifest work/equirect_run/train_manifest.json \
  --val_manifest work/equirect_run/val_manifest.json \
  --val_ratio 0.2 \
  --require_segmentation \
  --shared_roi_mask work/equirect_run/roi/roi_mask.png
```

Manifest schema:

```json
[
  {
    "image": "path/to/image.jpg",
    "boxes": [[10, 20, 50, 80]],
    "labels": [1],
    "seg_mask": "path/to/mask.png",
    "seg_confidence": 0.92,
    "seg_roi_mask": "path/to/roi_mask.png"
  }
]
```

### 6. Train One Config

```bash
python scripts/train.py \
  --config configs/multitask/multitask_resnet18.yaml \
  --train_manifest work/equirect_run/train_manifest.json \
  --val_manifest work/equirect_run/val_manifest.json
```

### 7. Controlled Backbone Sweep

```bash
python scripts/run_experiments.py \
  --config configs/multitask/multitask_resnet18.yaml \
  --train_manifest work/equirect_run/train_manifest.json \
  --val_manifest work/equirect_run/val_manifest.json \
  --label_source clrnet \
  --roi_mode manual_band \
  --output_dir work/equirect_run/experiments
```

The launcher runs:

- Phase A: `resnet18` detection-only and detection+segmentation
- Phase A: `convnext_base` detection-only and detection+segmentation
- Phase A: `swin_b` detection-only and detection+segmentation
- Optional exploratory: `hrnet_w32` detection+segmentation when `--include_exploratory` is used with `scripts/run_experiments.py`

`hrnet_w32` is marked exploratory because the current implementation is a lightweight fallback, not a true upstream HRNet.

### 8. Full Orchestration

```bash
python scripts/run_phase_a_equirect_baselines.py \
  --input_video data/raw/drive.mp4 \
  --detection_annotations data/detections.json \
  --work_dir work/equirect_run \
  --lane_backend clrnet \
  --config configs/lane_backends/opencv_equirect_lane.yaml \
  --device auto \
  --batch_size 4 \
  --fps 5 \
  --min_confidence 0.6 \
  --val_ratio 0.2 \
  --roi_top 320 \
  --roi_bottom 640
```

## ROI-Masked Segmentation Loss

`models/loss.py` computes unreduced CE by default, then applies the binary ROI mask and per-sample pseudo-label confidence before reducing over valid pixels. Pixels outside the ROI do not contribute gradient. Dice loss is also supported with valid-region weighting via:

```yaml
segmentation:
  loss: dice
```

## Confidence-Weighted Pseudo-Labels

Each manifest row may include `seg_confidence`. Low-confidence masks still train if retained by filtering, but their segmentation loss is scaled down. This is per-sample pseudo-label confidence, not class weighting.

## Seam-Aware Augmentation

Equirectangular images are horizontally cyclic. `dataset.py` supports optional horizontal circular shift:

```yaml
augment:
  circular_shift:
    enabled: true
    max_fraction: 0.15
    shift_boxes: false
```

Images, segmentation masks, and ROI masks are shifted with wrap-around. Detection box shifting is disabled by default because wrapped boxes can split across the seam; enable `shift_boxes` only if your detection labels can tolerate seam wrapping behavior.
