# Tools Folder Guide

## Purpose

`tools/` contains data preparation utilities. The main toolset is `tools/equirect_lane_pipeline/`, which prepares equirectangular frames, ROI masks, lane pseudo-labels, filtered records, visual QC outputs, and multitask manifests.

## What Lives Here

- `extract_frames.py`: frame extraction from video.
- `build_roi_masks.py`: manual band or polygon ROI mask creation.
- `generate_lane_masks.py`: lane mask generation through dummy, OpenCV-backed, or external adapter backends.
- `filter_lane_masks.py`: filters lane records by confidence and mask content.
- `visualize_lane_labels.py`: QC preview generation.
- `build_multitask_manifest.py`: joins images, detections, lane labels, and ROI masks into train/val manifests.
- `labelme_tape_to_masks.py`: converts manual LabelMe tape annotations into masks.
- `adapters.py`: adapter interface for lane backends.
- `README.md`: pipeline-specific usage.

## How Files Interact

The pipeline typically extracts frames, builds an ROI mask, generates lane masks, filters them, visualizes QC samples, builds manifests, and then hands those manifests to `scripts/train.py`.

## Inputs And Outputs

Inputs:

- Raw video or frame directory.
- Detection annotations.
- Lane backend config.
- Optional manually annotated tape labels.
- ROI settings.

Outputs:

- Extracted frames.
- ROI mask PNG.
- Lane mask PNG files.
- `records.jsonl` and `filtered_records.jsonl`.
- QC overlay images.
- Train/validation manifest JSON files.

## Entry Points

- `tools/equirect_lane_pipeline/build_multitask_manifest.py`
- `tools/equirect_lane_pipeline/generate_lane_masks.py`
- `scripts/run_phase_a_equirect_baselines.py`

## Safe To Edit

- QC visualization options.
- Additional record validation.
- New backend adapters when isolated behind the adapter API.

## Change Carefully

- Manifest output schema.
- ROI mask value semantics.
- Coordinate conventions.
- Filtering thresholds used in published comparisons.

## Typical Workflow

1. Extract or select frames.
2. Build ROI mask.
3. Generate or convert lane labels.
4. Filter weak labels.
5. Inspect QC overlays.
6. Build manifests.
7. Train and evaluate.

## Known Gaps

- External neural lane adapters are templates, not complete integrations.
- Pseudo-label quality still requires human QC.
- Final robot data pipeline is pending.
