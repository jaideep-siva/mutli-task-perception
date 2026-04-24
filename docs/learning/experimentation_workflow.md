# Experimentation Workflow

## Purpose

Experiments compare controlled changes. In this repo, the most common experiment is a backbone comparison while holding the data pipeline and training settings mostly fixed.

## Main Experiment Script

Use:

```bash
python scripts/run_experiments.py
```

This schedules configured backbone/task combinations and writes comparison CSV/JSON files.

## Controlled Variables

Try to keep these stable during a comparison:

- train manifest
- validation manifest
- image size
- batch size
- epochs
- augmentation settings
- random seed
- ROI mask
- label source

## What To Change Deliberately

- backbone
- segmentation enabled/disabled
- loss weights
- augmentation strategy
- learning rate

Change one category at a time when possible.

## Current Evidence In The Repo

Generated comparison outputs show a small manual tape-label experiment. The observed manual split has 38 train samples and 9 validation samples. The comparison files indicate detection mAP was unavailable for that data because the manifests contain empty detection boxes.

This evidence is useful for pipeline viability, not final performance claims.

## How To Read Comparison Results

Look at:

- `comparison.csv`
- `comparison.json`
- `backbone_viability_recommendation.json`
- run-specific JSON summaries under `runs/`

Do not choose a detection backbone from rows where detection mAP is unavailable.

## Good Experiment Notes

Record:

- what changed
- why it changed
- exact command
- config path
- manifest paths
- dataset notes
- metric limitations
- hardware used

## When To Archive

Move old or ambiguous experiment helpers to `archive/` instead of deleting them when:

- the purpose is unclear
- it may be useful historically
- it is not part of the current mainline
- deleting it might break someone's notes

Generated outputs should normally remain ignored rather than promoted into source.
