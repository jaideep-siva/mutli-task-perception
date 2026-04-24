# Docs And Archive Folder Guide

## Purpose

`docs/` is the stable documentation home. `archive/` preserves non-mainline material that should not be active in normal workflows.

## What Lives Here

`docs/`:

- `folder_guides/`: technical folder documentation.
- `learning/`: beginner and maintainer education.
- `report/`: current-state technical report.
- `legacy/`: documentation for preserved legacy paths.

`archive/`:

- `archive/legacy/srcipts/`: preserved typo-folder shim moved out of the source root.

## How Files Interact

Docs should link to source files and workflows. Archive files should not be imported by active code.

## Inputs And Outputs

Inputs are repository state, configs, code, and generated experiment summaries. Outputs are Markdown documents.

## Entry Points

- `docs/README.md`
- `docs/folder_guides/README.md`
- `docs/report/project_report.md`

## Safe To Edit

- Documentation updates.
- New learning guides.
- Cleanup logs.

## Change Carefully

- Archived files, because they may be historical evidence.
- Legacy docs that describe non-mainline behavior.

## Typical Workflow

1. Update source or workflow.
2. Update the relevant folder guide.
3. Update the report/checklist if project status changed.
4. Move ambiguous old material to `archive/` rather than deleting it.

## Known Gaps

- Documentation should be revisited after robot-data retraining and export work.
