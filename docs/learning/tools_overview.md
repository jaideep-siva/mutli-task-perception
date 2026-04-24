# Tools Overview

## PyTorch

PyTorch is the deep learning framework used for model definition, training, loss computation, optimization, and checkpointing.

Where it appears:

- `models/`
- `engine/train.py`
- `engine/eval.py`
- `tests/test_multitask_smoke.py`

Beginner note: tensors are the main data structure. Model weights are learned tensors saved in checkpoints.

Future work depending on it: ONNX export starts from a trained PyTorch model.

## Torchvision

Torchvision provides pretrained or standard backbones such as ResNet, ConvNeXt, and Swin, plus vision utilities.

Where it appears:

- `models/backbone.py`
- legacy detector code under `src/`

Beginner note: a backbone extracts image features before task-specific heads make predictions.

## TorchMetrics

TorchMetrics is used for detection mAP evaluation.

Where it appears:

- `engine/eval.py`
- `src/utils/detection_metrics.py`

Beginner note: mAP measures detection quality, but it needs ground-truth boxes. If the validation manifest has no boxes, detection mAP is not meaningful.

## YAML Configs

YAML stores human-readable experiment settings.

Where it appears:

- `configs/`
- `engine/train.py`
- `src/utils/config.py`

Beginner note: change YAML values before editing code when you want to adjust an experiment.

## Albumentations

Albumentations is used for image augmentation with bounding-box and mask awareness.

Where it appears:

- `data/transforms.py`
- `src/data/detection_transforms.py`

Beginner note: augmentation creates realistic variations of training images so the model can generalize better.

## OpenCV

OpenCV is used for image/video processing, lane label tooling, and the live demo.

Where it appears:

- `tools/equirect_lane_pipeline/`
- `live_inference_demo/`

Beginner note: OpenCV usually stores images as BGR arrays, while many ML models expect RGB tensors.

## MLflow

MLflow is used optionally for experiment tracking.

Where it appears:

- `engine/train.py`
- `scripts/run_experiments.py`
- local `mlruns/` and `mlflow.db`

Beginner note: if MLflow is not installed, the canonical training path still runs and records a note.

## Weights And Biases

W&B is optional and belongs to the legacy detector demo path.

Where it appears:

- `scripts/train_detector_demo.py`
- `docs/legacy/detection_baseline.md`

Beginner note: this is not required for the canonical multitask path.

## ONNX

ONNX is a planned export format. It is not implemented yet.

Expected future location:

- likely a new export script under `scripts/`
- tests under `tests/`
- deployment docs under `docs/`

Future work depending on it: TensorRT conversion usually starts from ONNX.

## TensorRT

TensorRT is a planned NVIDIA inference optimization/runtime step. It is not implemented yet.

Future work depending on it: target-hardware latency benchmarking and robot deployment.

## ROS / ROS2

ROS or ROS2 integration is planned but not implemented.

Expected future role:

- subscribe to camera frames
- run preprocessing and inference
- publish detection/segmentation outputs
- expose runtime diagnostics

Beginner note: ROS is the robot software layer, not the neural network itself.
