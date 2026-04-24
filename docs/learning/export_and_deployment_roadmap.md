# Export And Deployment Roadmap

## Current Status

Export and deployment are pending. The model foundation is mostly complete, but the repo is not deployment-complete until robot-data retraining, ONNX export, TensorRT conversion, and ROS integration are done.

## Deployment Sequence

1. Retrain on actual robot data.
2. Freeze a candidate checkpoint.
3. Add ONNX export.
4. Validate ONNX numerical behavior against PyTorch.
5. Convert ONNX to TensorRT.
6. Benchmark TensorRT on target hardware.
7. Build ROS wrapper/integration.
8. Validate with robot camera input.

## ONNX Work

Expected tasks:

- Add an export script.
- Define fixed or dynamic input shape policy.
- Separate exportable model forward from Python-heavy postprocessing if needed.
- Add a test that exports a small model and runs ONNX Runtime, if available.
- Compare PyTorch and ONNX outputs on representative inputs.

Risks:

- Unsupported Torch/Torchvision operators.
- Dynamic postprocessing logic.
- Shape assumptions in detection decoding.

## TensorRT Work

Expected tasks:

- Convert validated ONNX model to TensorRT.
- Decide FP32, FP16, or INT8.
- Benchmark latency, throughput, memory, and warmup behavior.
- Test on the actual target hardware.

Risks:

- Unsupported ONNX operators.
- Accuracy changes from precision reduction.
- Memory pressure on embedded hardware.

## ROS / ROS2 Work

Expected tasks:

- Define subscribed image topic.
- Define published detection and segmentation messages.
- Add preprocessing that matches training.
- Load model runtime cleanly.
- Publish diagnostics such as FPS and dropped frames.
- Add launch/config files.

Risks:

- Camera calibration mismatch.
- Frame timing and synchronization issues.
- Different image encoding than training data.
- Runtime latency too high for robot behavior.

## Acceptance Criteria

Before calling deployment complete, require:

- robot-data checkpoint
- documented dataset split
- validated metrics
- ONNX export success
- TensorRT conversion success
- target-hardware benchmark
- ROS integration test
- documented failure modes
