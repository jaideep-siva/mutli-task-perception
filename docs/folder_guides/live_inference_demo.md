# Live Inference Demo Folder Guide

## Purpose

`live_inference_demo/` is an OpenCV/PyTorch demo scaffold for running live or video inference. It is useful for prototyping display, frame capture, overlays, saved frames, and runtime metrics. It is not the final ROS integration.

## What Lives Here

- `app.py`: demo application entry point.
- `config.yaml`: demo defaults.
- `requirements.txt`: demo-specific requirements.
- `adapters/`: adapter interface and multitask adapter example.
- `utils/`: video source, timers, stats, saving, overlay, and logging helpers.
- `README.md`: demo usage.

## How Files Interact

`app.py` opens a camera/video/stream, passes frames through an adapter, overlays predictions, records metrics, and optionally saves rendered frames. The adapter isolates model-specific loading and postprocessing from UI/video logic.

## Inputs And Outputs

Inputs:

- Camera index, video path, or stream URL.
- Demo config.
- Optional checkpoint path.

Outputs:

- Display window.
- Saved frames.
- Per-frame metrics CSV.
- Summary JSON.

## Entry Points

```bash
cd live_inference_demo
python app.py --video path/to/demo.mp4
```

## Safe To Edit

- Adapter implementation for a known checkpoint format.
- Display and save cadence.
- Non-breaking metrics fields.

## Change Carefully

- Adapter output dictionary contract.
- Video source retry behavior.
- Runtime assumptions that may later affect ROS integration.

## Typical Workflow

1. Train or obtain a checkpoint.
2. Point `config.yaml` to that checkpoint.
3. Run video/camera demo.
4. Use findings to inform the eventual ROS wrapper.

## Known Gaps

- The demo is not yet wired as a production deployment path.
- ONNX/TensorRT are not integrated here.
- ROS message contracts and node lifecycle are pending.
