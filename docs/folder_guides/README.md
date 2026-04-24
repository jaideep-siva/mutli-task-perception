# Folder Guides

This index explains where to look before editing a major part of the repository.

## Recommended Reading Order

1. [Configs](configs.md)
2. [Data](data.md)
3. [Models](models.md)
4. [Engine](engine.md)
5. [Scripts](scripts.md)
6. [Tools](tools.md)
7. [Outputs And Artifacts](outputs_and_artifacts.md)
8. [Live Inference Demo](live_inference_demo.md)
9. [Legacy Source](legacy_src.md)
10. [Tests](tests.md)
11. [Docs And Archive](docs_and_archive.md)

## Mainline Workflow Folders

- `configs/`: experiment settings.
- `data/`: manifest dataset and batching.
- `models/`: canonical multitask model.
- `engine/`: training and evaluation internals.
- `scripts/`: command-line entry points.
- `tools/equirect_lane_pipeline/`: data preparation and lane supervision.

## Non-Mainline But Preserved

- `src/`: legacy detector/demo implementation.
- `live_inference_demo/`: future-facing demo scaffold.
- `archive/`: preserved material that should not be imported by active code.
