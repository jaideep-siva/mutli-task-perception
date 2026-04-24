# Cleanup Log

## Summary

This cleanup pass focused on documentation structure, conservative organization, and removal of obvious generated clutter from the source-facing tree.

## Mainline Preserved

- `scripts/train.py`
- `scripts/run_experiments.py`
- `scripts/run_phase_a_equirect_baselines.py`
- `configs/multitask/multitask_resnet18.yaml`
- `data/`
- `models/`
- `engine/`
- `tools/equirect_lane_pipeline/`
- `tests/`
- legacy detector path under `src/`

## Moved

- `DETECTION_TRAINING_README.md` moved to `docs/legacy/detection_baseline.md`.
- `srcipts/` moved to `archive/legacy/srcipts/` because it was a typo-named folder containing a small duplicate utility and was not part of the mainline.

## Removed From Source Tree

- Tracked `__pycache__` bytecode files were removed where accessible.
- Some generated cache/temp folders under `.pytest_cache` and `outputs/` were locked by Windows permissions and were left in place. They are ignored by Git and should not affect source workflows.

## Documentation Added

- Rewritten top-level `README.md`.
- `docs/README.md`.
- Folder guides under `docs/folder_guides/`.
- Learning material under `docs/learning/`.
- Current-state report under `docs/report/project_report.md`.

## Ambiguities

- The repository contains both canonical multitask code and a legacy detector path. The cleanup preserves both and clearly marks `src/` as legacy.
- Generated experiment outputs contain useful evidence, so they were documented rather than deleted.
- Export/deployment work is planned but not implemented; docs describe it as pending rather than complete.
