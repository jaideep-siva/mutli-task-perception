# Design Philosophy

## Goals

The repository is designed to be a clean research-to-deployment bridge for robotics perception. It should be simple enough for a new student to run, but structured enough for future ONNX, TensorRT, and ROS work.

## Principles

### Modularity

Each part has a clear responsibility:

- `configs/` controls behavior.
- `data/` owns dataset loading.
- `models/` owns neural network structure.
- `engine/` owns training and evaluation.
- `scripts/` owns command-line entry points.
- `tools/` owns preprocessing and label generation.

This keeps changes local and makes failures easier to diagnose.

### Readability

The main workflow should be understandable from file names and folder boundaries. Scripts should be thin; reusable logic belongs in packages.

### Reproducibility

Experiments should record:

- Config path and settings.
- Train/validation manifest paths.
- Backbone and task settings.
- Metrics.
- Checkpoint paths.
- Notes about unavailable metrics or failed runs.

### Minimal Unnecessary Complexity

The repo should not add abstractions just to look sophisticated. New layers should solve real problems: reducing duplication, protecting contracts, or enabling future deployment.

### Separation Of Research And Deployment

Research code can support comparisons, diagnostics, and flexible configs. Deployment code must be stricter about stable shapes, supported operators, latency, memory, and runtime contracts. This repo is organized so deployment work can be added without rewriting the model from scratch.

## Future ONNX/TRT Compatibility

To prepare for export:

- Keep model input/output contracts stable.
- Avoid Python-only logic inside exported model paths.
- Separate postprocessing from forward pass when needed.
- Add export tests with representative input sizes.
- Track unsupported operators early.

## Future ROS Compatibility

To prepare for ROS or ROS2:

- Define image topic input assumptions.
- Define prediction message output contracts.
- Keep preprocessing deterministic and documented.
- Separate model runtime from visualization/debug UI.
- Benchmark on target hardware, not only a development machine.

## Practical Rule

When deciding where code belongs, ask: "Will a future maintainer know where to look?" If not, the boundary probably needs documentation or simplification.
