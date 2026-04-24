# Scripts Folder Guide

## Purpose

`scripts/` contains command-line entry points. Scripts should stay thin: parse arguments, apply overrides, and call reusable package code.

## What Lives Here

- `scripts/train.py`: canonical multitask training CLI.
- `scripts/run_experiments.py`: controlled backbone comparison runner.
- `scripts/run_phase_a_equirect_baselines.py`: end-to-end Phase A lane/data/training orchestration with QC previews.
- `scripts/run_equirect_lane_backbone_experiments.py`: older/leaner orchestration variant.
- `scripts/train_detector_demo.py`: legacy Faster R-CNN detector training.
- `scripts/infer_detector_demo.py`: legacy detector inference demo.
- `scripts/__init__.py`: package marker.

## How Files Interact

Canonical scripts import `data`, `engine`, and `models`. Legacy detector scripts import from `src/`.

## Inputs And Outputs

Inputs are command-line flags, configs, data paths, and manifests. Outputs are training summaries, checkpoints, comparison CSV/JSON files, lane labels, and QC images under generated output directories.

## Entry Points

Most users should start with:

```bash
python scripts/train.py --config configs/multitask/multitask_resnet18.yaml
```

For experiments:

```bash
python scripts/run_experiments.py --config configs/multitask/multitask_resnet18.yaml
```

For full equirectangular preprocessing plus experiments:

```bash
python scripts/run_phase_a_equirect_baselines.py --work_dir outputs/equirect_run ...
```

## Safe To Edit

- CLI help text.
- Additional non-breaking optional flags.
- Clearer validation for required paths.

## Change Carefully

- Default config paths.
- Output folder conventions.
- The set of experiments in `run_experiments.py`.
- Import paths, because scripts are intended to run from the repo root.

## Typical Workflow

1. Prepare data/manifests.
2. Run a smoke command.
3. Launch a full training run.
4. Launch comparisons only after smoke tests pass.

## Known Gaps

- Two orchestration scripts overlap. `run_phase_a_equirect_baselines.py` is the more complete one because it includes QC preview generation.
- Export and deployment scripts do not exist yet.
