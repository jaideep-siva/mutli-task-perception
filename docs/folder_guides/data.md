# Data Folder Guide

## Purpose

`data/` contains the canonical manifest-driven dataset implementation for multitask training. It also acts as the local home for large image data when needed, but large files are ignored by Git.

## What Lives Here

- `data/dataset.py`: `MultitaskDataset`, image/mask loading, resizing, augmentation, ROI masks, and circular shift augmentation.
- `data/collate.py`: batching logic for variable-size detection targets.
- `data/transforms.py`: torchvision/albumentations transform helpers.
- `data/__init__.py`: public package exports.
- `data/README.md`: short package note.

## How Files Interact

`engine.train.build_dataloader` creates `MultitaskDataset` and passes `data.collate_fn` into PyTorch `DataLoader`. The dataset returns image tensors, detection boxes/labels, segmentation masks, ROI masks, confidence values, and `has_seg` flags.

## Inputs And Outputs

Input manifest schema:

```json
{
  "image": "path/to/image.jpg",
  "boxes": [[10, 20, 50, 80]],
  "labels": [1],
  "seg_mask": "path/to/mask.png",
  "seg_confidence": 0.93,
  "seg_roi_mask": "path/to/roi_mask.png"
}
```

Output sample keys:

- `image`: normalized tensor `[3, H, W]`.
- `boxes`: tensor `[N, 4]`.
- `labels`: tensor `[N]`.
- `seg_mask`: tensor `[H, W]`.
- `seg_confidence`: scalar tensor.
- `seg_roi_mask`: tensor `[H, W]`.
- `has_seg`: boolean tensor.

## Entry Points

The folder is usually entered through:

- `engine.train.build_dataloader`
- `tests/test_multitask_smoke.py`

## Safe To Edit

- Additional manifest validation.
- Clearer error messages.
- Non-behavioral comments.
- New tests for edge cases.

## Change Carefully

- Box scaling logic.
- Mask binarization.
- ROI mask semantics.
- Collate output keys.
- Circular shift behavior for detection boxes.

## Typical Workflow

1. Build manifests with `tools/equirect_lane_pipeline/build_multitask_manifest.py`.
2. Inspect masks and ROI coverage.
3. Load through `MultitaskDataset`.
4. Train with `scripts/train.py`.

## Known Gaps

- Dataset versioning is not formalized.
- Data validation could be stricter before long runs.
- Current observed manual tape-label manifests contain segmentation labels but empty detection boxes.
