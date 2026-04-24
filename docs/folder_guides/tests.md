# Tests Folder Guide

## Purpose

`tests/` contains lightweight checks that the canonical multitask stack still loads data, builds models, computes losses, and runs a minimal training step.

## What Lives Here

- `tests/test_multitask_smoke.py`: smoke tests for config loading, manifest loading, augmentation alignment, model forward pass, loss computation, ROI-aware segmentation metrics, class weights, and one training step.

## How Files Interact

The tests import the canonical `data`, `engine`, and `models` packages. They build tiny temporary images and masks rather than depending on the large local dataset.

## Inputs And Outputs

Inputs are temporary files created by pytest. Outputs are test pass/fail results and temporary pytest folders.

## Entry Points

```bash
python -m pytest tests/test_multitask_smoke.py -q
```

## Safe To Edit

- Additional tests for new behavior.
- More edge cases for manifest validation.
- Export tests once ONNX work begins.

## Change Carefully

- Existing tests that protect ROI masking, augmentation alignment, and loss behavior.

## Typical Workflow

Run tests before and after changing data/model/engine code.

## Known Gaps

- No ONNX/TensorRT export tests yet.
- No ROS integration tests yet.
- No full long-run training test, which is appropriate for local smoke coverage.
