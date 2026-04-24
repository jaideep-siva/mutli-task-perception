# Repo Orientation For Beginners

## What This Repository Is

This is a machine-learning robotics repository. Its job is to train and evaluate a perception model that can look at camera frames and predict useful scene information for a robot.

The main model is multitask because it has more than one job:

- Object detection: find objects using bounding boxes and labels.
- Segmentation: mark pixels that belong to a lane/tape/ground class.

## The Big Idea

The repository is organized so each responsibility has a home:

- Configs say what experiment to run.
- Data code loads examples.
- Model code defines the neural network.
- Engine code trains and evaluates the model.
- Scripts are command-line entry points.
- Tools prepare labels and manifests.
- Docs explain the system.

## How Data Flows

1. Images and labels are prepared.
2. Tools build a manifest JSON file.
3. `data.MultitaskDataset` reads the manifest.
4. The training engine creates batches.
5. The model predicts detections and segmentation.
6. The loss compares predictions to labels.
7. The optimizer updates model weights.
8. Evaluation computes metrics.
9. Checkpoints and summaries are written to generated output folders.

## What To Read First

1. `README.md`
2. `configs/multitask/multitask_resnet18.yaml`
3. `scripts/train.py`
4. `data/dataset.py`
5. `models/multitask_model.py`
6. `engine/train.py`

Do not start by reading every file. Follow one workflow from config to script to engine to model.

## How To Run Safely

Start with a small run:

```bash
python scripts/train.py \
  --config configs/multitask/multitask_resnet18.yaml \
  --train_manifest path/to/train_manifest.json \
  --val_manifest path/to/val_manifest.json \
  --epochs 1 \
  --batch_size 1 \
  --max_steps_per_epoch 1 \
  --device cpu
```

This checks that paths, imports, data loading, model construction, and one optimization step work before spending time on a long run.

## What Not To Worry About At First

- ONNX and TensorRT: these are deployment export formats and are still pending.
- ROS: integration is still pending.
- `src/`: this is the older detector path, not the main multitask path.
- MLflow internals: useful for tracking, but not required to understand the model.

## Beginner Mental Model

Think of the repo as a lab notebook plus machine:

- The machine is the code that trains and evaluates.
- The lab notebook is the configs, outputs, reports, and docs.
- A good experiment changes one or two controlled things and records what happened.
