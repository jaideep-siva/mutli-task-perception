# Multitask Perception Project Report

## Table Of Contents

1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Repository And System Overview](#repository-and-system-overview)
4. [Model Development Status](#model-development-status)
5. [Data And Experimentation Summary](#data-and-experimentation-summary)
6. [Software Architecture And Design Rationale](#software-architecture-and-design-rationale)
7. [Current Results And Progress Assessment](#current-results-and-progress-assessment)
8. [Completed Work vs Pending Deployment Work](#completed-work-vs-pending-deployment-work)
9. [Pending Work](#pending-work)
10. [Risks And Technical Debt](#risks-and-technical-debt)
11. [Recommended Roadmap](#recommended-roadmap)
12. [Conclusion](#conclusion)

## Abstract

This repository implements a research-oriented multitask perception stack for robotics using equirectangular camera imagery. The canonical system includes manifest-driven data loading, a PyTorch multitask model, ROI-aware segmentation supervision, detection and segmentation evaluation hooks, lane-supervision tooling, backbone comparison scripts, and a live inference demo scaffold. The model foundation and core training pipeline are mostly complete. Deployment remains incomplete: the system still needs retraining on actual robot data, ONNX export, TensorRT conversion, and ROS integration. The current experiment evidence in the repository supports pipeline viability and early backbone comparison, but it should not be interpreted as final robot performance.

## Introduction

Robotics perception systems require a careful bridge between research flexibility and deployment reliability. During development, engineers need configurable training pipelines, data preprocessing tools, experiment tracking, metrics, and reproducible comparisons. During deployment, the same system must eventually become stable enough for optimized inference, hardware benchmarking, and robot middleware integration.

This project targets that bridge. It focuses on a multitask perception model that can ingest equirectangular frames and produce both detection and segmentation outputs. Equirectangular input is treated as the native geometry of the system: preprocessing, lane masks, ROI masks, training, validation, and evaluation remain in equirectangular image space.

## Repository And System Overview

The current canonical workflow is:

```text
configs/multitask/multitask_resnet18.yaml
scripts/train.py
data/*
models/*
engine/*
tools/equirect_lane_pipeline/*
```

The repository is organized into clear responsibility boundaries:

- `configs/` stores experiment and backend settings.
- `data/` loads manifest-based multitask examples.
- `models/` defines the canonical multitask architecture.
- `engine/` trains, evaluates, checkpoints, and logs.
- `scripts/` provides command-line entry points.
- `tools/equirect_lane_pipeline/` prepares frames, ROI masks, lane labels, QC previews, and manifests.
- `docs/` now contains folder guides, learning material, legacy docs, and this report.
- `src/` preserves the older Faster R-CNN detector demo path.
- `live_inference_demo/` provides a prototype OpenCV/PyTorch inference scaffold.

Generated folders such as `outputs/`, `artifacts/`, `mlruns/`, and `mlflow.db` are local experiment state and are intentionally ignored by Git.

## Model Development Status

The canonical model is implemented in `models/multitask_model.py`. It combines:

- a configurable backbone from `models/backbone.py`
- an FPN neck from `models/fpn.py`
- a detection head from `models/detection_head.py`
- a segmentation head from `models/segmentation_head.py`
- postprocessing helpers from `models/detection_postprocess.py`
- ROI-aware segmentation helpers from `models/segmentation_roi.py`
- multitask loss composition from `models/loss.py`

Configured and available backbones include ResNet-18, ResNet-50, ConvNeXt-Base, Swin-B, and an `hrnet_w32` lightweight fallback. The HRNet entry should not be interpreted as a full upstream HRNet implementation; the code itself identifies it as a lightweight HRNet-style fallback.

Training is handled by `engine/train.py`, which builds dataloaders, constructs the model and loss, optimizes with AdamW, schedules learning rate with cosine annealing, evaluates after epochs, writes checkpoints, and logs MLflow metrics when MLflow is available.

## Data And Experimentation Summary

The data pipeline is manifest-driven. Each sample may contain an image path, detection boxes, labels, segmentation mask path, segmentation confidence, and segmentation ROI mask. This is a practical format for combining detection labels with segmentation pseudo-labels or manual masks.

The equirectangular lane pipeline supports:

- frame extraction
- ROI mask creation
- lane mask generation through adapter-style backends
- mask filtering
- QC visualization
- multitask manifest construction
- LabelMe tape annotation conversion

Observed generated evidence includes a manual tape-label dataset under `outputs/equirect_run/` with 47 mask records. The visible split is 38 training samples and 9 validation samples. The observed manifest rows contain segmentation masks and ROI masks, but detection `boxes` and `labels` are empty.

The generated backbone comparison under `outputs/comparisons/backbone_viability_comparison_cuda_full/` reports:

| Run | Backbone | Segmentation | Best Seg IoU | Best Seg Dice | Latency ms | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `phase_A_resnet18_det_seg` | ResNet-18 | yes | 0.7967 | 0.8711 | 150.03 | detection mAP unavailable |
| `phase_A_convnext_base_det_seg` | ConvNeXt-Base | yes | 0.7767 | 0.8489 | 158.30 | detection mAP unavailable |
| `phase_A_swin_b_det_seg` | Swin-B | yes | 0.7902 | 0.8665 | 1498.36 | detection mAP unavailable |
| `phase_C_hrnet_w32_det_seg_exploratory` | HRNet-style fallback | yes | 0.8298 | 0.8923 | 1007.63 | exploratory fallback |

The corresponding recommendation file selects `phase_A_resnet18_det_seg` as the best overall/practical choice, while explicitly warning that the comparison is rough and based on a small manual tape-label dataset. Because the observed validation manifests contain no ground-truth boxes, detection mAP is unavailable and should not be used for detection-backbone selection from this run.

## Software Architecture And Design Rationale

The repository follows a separation-of-concerns design:

- Configs are externalized into YAML for reproducibility.
- Data loading is isolated from model definition.
- Model components are modular enough to compare backbones and task heads.
- Training and evaluation are centralized in `engine/`.
- Scripts stay thin and workflow-oriented.
- Data preparation tools are separated from training code.
- Legacy code is preserved rather than mixed into the canonical path.

This layout supports a research workflow while keeping future deployment possible. In particular, stable model output keys, explicit ROI semantics, and isolated postprocessing will matter for ONNX/TensorRT export and ROS integration.

## Current Results And Progress Assessment

The repository demonstrates substantial engineering progress:

- The canonical model builds and runs.
- The dataset loader supports multitask labels and ROI masks.
- The loss supports ROI-masked and confidence-weighted segmentation supervision.
- The training loop writes checkpoints and evaluation artifacts.
- Backbone comparison tooling exists.
- Smoke tests cover the most important pipeline contracts.
- Documentation now explains the repository structure, workflows, and status.

The current metrics should be treated as development evidence, not final system validation. The visible comparison data is small and detection labels are absent. The segmentation metrics indicate that the pipeline can learn and evaluate the manual tape-label task, but robot-data retraining and full detection evaluation are still required.

## Completed Work vs Pending Deployment Work

Completed or mostly complete:

- Architecture development.
- Repository/model foundation.
- Canonical training pipeline.
- Manifest-driven data pipeline.
- Segmentation ROI and confidence weighting.
- Evaluation flow for detection and segmentation.
- Backbone comparison flow.
- Equirectangular lane-supervision tooling.
- Legacy detector baseline preservation.
- Documentation cleanup and project reporting.

Pending deployment work:

- Retraining on actual robot data.
- ONNX conversion.
- TensorRT conversion.
- ROS wrapper / ROS integration.
- Target-hardware benchmarking.
- End-to-end robot validation.

## Pending Work

The most important pending item is data: the model should be retrained and evaluated on actual robot data before deployment decisions are made. That data should include sufficient detection labels if detection performance is a deployment requirement.

Export work is also pending. ONNX conversion should be added first, with tests that compare PyTorch and ONNX outputs. TensorRT conversion should follow only after ONNX is stable. ROS integration should be built around the validated runtime, not around an unvalidated research checkpoint.

## Risks And Technical Debt

Key risks:

- Current experiment evidence is based on small manual tape-label data.
- Detection metrics are unavailable in the observed comparison because boxes are empty.
- ONNX/TensorRT compatibility is unknown.
- ROS message contracts are undefined.
- The live demo is useful but should not be treated as deployment integration.
- A top-level frozen dependency file is missing.
- Some generated Windows temp/cache folders are locked and should remain ignored or be cleaned outside active runs.

Technical debt:

- Two equirectangular orchestration scripts overlap.
- Config validation is informal.
- Deployment-specific tests do not exist yet.
- A curated, small results table should eventually replace reliance on local generated output folders.

## Recommended Roadmap

1. Freeze the current Python environment into a top-level dependency file.
2. Collect/curate actual robot data with documented train/validation/test splits.
3. Retrain the ResNet-18 baseline on robot data.
4. Re-run backbone comparisons with complete detection and segmentation labels.
5. Define final acceptance metrics for detection, segmentation, latency, and reliability.
6. Add ONNX export.
7. Add ONNX validation tests.
8. Add TensorRT conversion and target-hardware benchmarks.
9. Build ROS or ROS2 wrapper around the validated runtime.
10. Run robot-in-the-loop validation.

## Conclusion

This repository is now structured as a serious robotics/ML research codebase with a clear canonical path, preserved legacy code, folder-level documentation, beginner learning material, and an honest technical report. The model-building and training foundation are mostly complete. The project should be considered development-strong but not deployment-complete. The next decisive step is retraining and validating on actual robot data, followed by ONNX/TensorRT export and ROS integration.
