"""Evaluation script for trained detection model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import detection_collate_fn
from src.data.detection_dataset import LabelMeDetectionDataset
from src.data.detection_transforms import build_validation_transforms
from src.utils.config import load_yaml
from src.utils.detection_metrics import compute_detection_metrics_epoch
from src.utils.visualization import draw_pred_boxes

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def load_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> dict[str, Any]:
    """Load checkpoint and extract model state, config, and metadata.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Torch device.

    Returns:
        Dict with 'state_dict', 'config', 'class_map', 'image_size'.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, dict, list]:
    """Build model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Torch device.

    Returns:
        Tuple of (model, model_state_dict, class_map, image_size).
    """
    from src.models.demo_detector import build_demo_detector

    checkpoint = load_checkpoint(checkpoint_path, device)
    class_map = checkpoint.get("class_map", {})
    image_size = checkpoint.get("image_size", [640, 1280])
    config = checkpoint.get("config", {})

    model = build_demo_detector(
        num_classes=len(class_map),
        pretrained=config.get("pretrained", True),
        trainable_backbone_layers=config.get("trainable_backbone_layers"),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, class_map, image_size


def evaluate_on_split(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    class_map: dict[str, int],
    score_thresh: float = 0.5,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate model on a data split.

    Args:
        model: Detection model.
        data_loader: DataLoader with (images, targets) batches.
        device: Torch device.
        num_classes: Total number of classes.
        class_map: Mapping from class name to class id.
        score_thresh: Score threshold for detections.
        output_dir: Optional directory to save visualizations.

    Returns:
        Dict with metrics.
    """
    model.eval()

    # Compute metrics
    metrics = compute_detection_metrics_epoch(
        model=model,
        data_loader=data_loader,
        device=device,
        num_classes=num_classes,
        score_thresh=score_thresh,
    )

    # Save visualizations if requested
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        id_to_class = {class_id: name for name, class_id in class_map.items()}

        model.eval()
        with torch.no_grad():
            for step, (images, targets) in enumerate(data_loader):
                if step >= 5:  # Save first 5 examples
                    break

                images = [img.to(device) for img in images]
                predictions = model(images)

                for batch_idx, (image, pred) in enumerate(zip(images, predictions)):
                    drawn = draw_pred_boxes(
                        image.cpu(),
                        pred,
                        id_to_class,
                        score_thresh,
                    )
                    array = drawn.permute(1, 2, 0).cpu().numpy()
                    Image.fromarray(array).save(
                        output_dir / f"evaluation_sample_{step}_{batch_idx}.png"
                    )

        logger.info(f"Saved visualization examples to {output_dir}")

    return metrics


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate a trained detection model")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to checkpoint file (best.pth or detector_epoch_*.pth)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/insta360_detection.yaml",
        help="Path to data config YAML",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="val",
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.5,
        help="Score threshold for detections",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results and visualizations",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load config
    data_config = load_yaml(ROOT / args.data_config)

    # Load checkpoint and build model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model, class_map, image_size = build_model_from_checkpoint(checkpoint_path, device)
    logger.info(f"Loaded model with {len(class_map)} classes: {class_map}")

    # Load dataset
    dataset = LabelMeDetectionDataset(
        root_dir=ROOT / data_config["root_dir"],
        image_size=image_size or data_config.get("image_size", [640, 1280]),
        class_map=class_map,
        keep_empty=bool(data_config.get("keep_empty", False)),
    )
    logger.info(f"Loaded dataset with {len(dataset)} samples")

    # Create validation split
    num_samples = len(dataset)
    val_fraction = data_config.get("val_fraction", 0.2)
    val_count = max(1, int(num_samples * val_fraction))

    indices = list(range(num_samples))
    import random

    random.seed(42)
    random.shuffle(indices)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    test_indices = train_indices[-len(train_indices) // 4 :]  # Last 25% as test
    train_indices = train_indices[: len(train_indices) - len(test_indices)]

    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    train_subset = Subset(dataset, train_indices)

    # Apply transforms
    val_transforms = build_validation_transforms(image_size=image_size)

    class TransformWrapper:
        def __init__(self, subset, transforms):
            self.subset = subset
            self.transforms = transforms

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            sample = self.subset[idx]
            return self.transforms(sample)

    train_subset_transformed = TransformWrapper(train_subset, val_transforms)
    val_subset_transformed = TransformWrapper(val_subset, val_transforms)
    test_subset_transformed = TransformWrapper(test_subset, val_transforms)

    # Create dataloaders
    splits_dict = {
        "train": train_subset_transformed,
        "val": val_subset_transformed,
        "test": test_subset_transformed,
    }

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    split_to_eval = [args.split] if args.split != "all" else ["train", "val", "test"]
    results = {}

    for split_name in split_to_eval:
        if split_name not in splits_dict:
            logger.warning(f"Split '{split_name}' not available")
            continue

        split_data = splits_dict[split_name]
        loader = DataLoader(
            split_data,
            batch_size=1,
            shuffle=False,
            collate_fn=detection_collate_fn,
            pin_memory=True,
            num_workers=0,
        )

        logger.info(
            f"Evaluating on '{split_name}' split ({len(split_data)} samples)..."
        )
        split_output_dir = (output_dir / split_name) if output_dir else None

        metrics = evaluate_on_split(
            model=model,
            data_loader=loader,
            device=device,
            num_classes=len(class_map),
            class_map=class_map,
            score_thresh=args.score_thresh,
            output_dir=split_output_dir,
        )

        results[split_name] = metrics
        logger.info(f"Metrics for '{split_name}':")
        for key, value in metrics.items():
            if not isinstance(value, list):
                logger.info(
                    f"  {key}: {value:.6f}"
                    if isinstance(value, float)
                    else f"  {key}: {value}"
                )

    # Save results
    if output_dir:
        results_file = output_dir / "evaluation_results.json"
        with results_file.open("w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved evaluation results to {results_file}")

        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 70)
        for split_name, metrics in results.items():
            logger.info(f"\n{split_name.upper()} SPLIT:")
            for key, value in metrics.items():
                if not isinstance(value, list):
                    logger.info(
                        f"  {key}: {value:.6f}"
                        if isinstance(value, float)
                        else f"  {key}: {value}"
                    )


if __name__ == "__main__":
    main()
