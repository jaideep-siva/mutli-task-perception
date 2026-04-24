# Outputs And Artifacts Folder Guide

## Purpose

`outputs/`, `artifacts/`, `mlruns/`, and `mlflow.db` are generated experiment state. They are valuable for local evidence and debugging, but they are not source code.

## What Lives Here

- `outputs/`: checkpoints, generated manifests, lane labels, QC images, comparison CSV/JSON files, and temporary test outputs.
- `artifacts/`: per-run metric artifacts such as per-class IoU JSON files.
- `mlruns/`: MLflow local experiment tracking files.
- `mlflow.db`: local MLflow database.

## How Files Interact

`engine/train.py` writes checkpoints and artifacts. `scripts/run_experiments.py` writes comparison summaries and recommendations. Lane tools write generated masks, records, and manifests.

## Inputs And Outputs

These folders are almost entirely outputs. Some generated manifests may be reused as inputs for training.

## Entry Points

- `scripts/train.py`
- `scripts/run_experiments.py`
- `scripts/run_phase_a_equirect_baselines.py`
- `tools/equirect_lane_pipeline/*`

## Safe To Edit

- New generated run folders.
- README or notes files, if intentionally tracked.

## Change Carefully

- Checkpoints referenced by comparison summaries.
- Manifests used by ongoing experiments.
- MLflow state if you still need local run history.

## Typical Workflow

1. Write every experiment to a unique subfolder.
2. Keep raw source code separate from generated outputs.
3. Promote only summarized, reviewed results into documentation.

## Known Gaps

- Some Windows temp folders under generated outputs may be locked by old runs or permissions.
- Large generated outputs are intentionally ignored by Git.
- A lightweight curated results table should eventually replace reliance on local generated folders.
