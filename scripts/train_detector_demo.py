"""Enhanced training script for Faster R-CNN detection with low-data robustness."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import detection_collate_fn
from src.data.detection_dataset import LabelMeDetectionDataset
from src.data.detection_transforms import (
    build_training_transforms,
    build_validation_transforms,
)
from src.data.samplers import RepeatFactorSampler, WeightedImageSampler
from src.data.split_utils import (
    create_kfold_splits,
    create_splits,
    extract_group_ids_from_dataset,
)
from src.models.demo_detector import (
    build_demo_detector,
    freeze_backbone,
    unfreeze_backbone,
)
from src.utils.config import load_yaml
from src.utils.detection_metrics import compute_detection_metrics_epoch
from src.utils.plot_training_curves import plot_metrics_summary
from src.utils.visualization import draw_gt_boxes, draw_pred_boxes

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


def resolve_device(device_name: str) -> torch.device:
    """Selects CUDA when available, otherwise falls back to CPU."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    dataset: Subset | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    sampler: Any = None,
) -> DataLoader | None:
    """Constructs a DataLoader for detection samples."""
    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    """Saves a drawn uint8 image tensor shaped [3, H, W] to disk."""
    array = image.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def log_ground_truth_example(
    writer: SummaryWriter,
    dataset: LabelMeDetectionDataset,
    id_to_class: dict[int, str],
) -> None:
    """Logs one ground-truth example to TensorBoard at training start."""
    sample = dataset[0]
    drawn = draw_gt_boxes(sample["image"], sample["target"], id_to_class)
    writer.add_image("ground_truth/example", drawn, 0)


def log_validation_examples(
    model: torch.nn.Module,
    writer: SummaryWriter,
    val_loader: DataLoader | None,
    device: torch.device,
    epoch: int,
    id_to_class: dict[int, str],
    score_thresh: float,
    num_examples: int = 5,
) -> None:
    """Logs validation predictions to TensorBoard and PNG files."""
    if val_loader is None:
        return

    model.eval()
    with torch.no_grad():
        for step, (images, _) in enumerate(val_loader):
            if step >= num_examples:
                break
            images = [image.to(device) for image in images]
            predictions = model(images)
            drawn = draw_pred_boxes(
                images[0].cpu(), predictions[0], id_to_class, score_thresh
            )
            writer.add_image(f"predictions/val_example_{step}", drawn, epoch)


def log_csv_metrics(
    csv_file: Path,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    val_metrics: dict[str, float] | None,
    lr: float,
) -> None:
    """Logs metrics to CSV file for easy analysis."""
    if not csv_file.parent.exists():
        csv_file.parent.mkdir(parents=True, exist_ok=True)

    # Header
    if not csv_file.exists():
        headers = ["epoch", "train_loss", "val_loss", "lr"]
        if val_metrics:
            headers.extend(val_metrics.keys())
        csv_file.write_text(",".join(headers) + "\n")

    # Row
    row = [
        str(epoch),
        f"{train_loss:.6f}",
        f"{val_loss:.6f}" if val_loss else "N/A",
        f"{lr:.8f}",
    ]
    if val_metrics:
        row.extend(
            f"{v:.6f}" if isinstance(v, float) else str(v) for v in val_metrics.values()
        )
    csv_file.write_text(csv_file.read_text() + ",".join(row) + "\n")


def log_json_metrics(
    json_file: Path,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    val_metrics: dict[str, float] | None,
    lr: float,
) -> None:
    """Logs metrics to JSON lines file."""
    if not json_file.parent.exists():
        json_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "epoch": epoch,
        "train/loss": train_loss,
        "train/lr": lr,
    }
    if val_loss is not None:
        entry["val/loss"] = val_loss
    if val_metrics:
        for key, value in val_metrics.items():
            entry[f"val/{key}"] = value

    with json_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def compute_validation_metrics(
    model: torch.nn.Module,
    val_loader: DataLoader | None,
    device: torch.device,
    num_classes: int,
    score_thresh: float = 0.0,
) -> dict[str, Any]:
    """Computes detection metrics on validation set."""
    if val_loader is None:
        return {}

    model.eval()
    metrics = {}
    try:
        metrics = compute_detection_metrics_epoch(
            model=model,
            data_loader=val_loader,
            device=device,
            num_classes=num_classes,
            score_thresh=score_thresh,
        )
    except Exception as e:
        logger.warning(f"Failed to compute metrics: {e}")

    return metrics


def compute_detector_loss(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Returns detector losses regardless of the model's current mode.

    Torchvision detection models return a loss dict only in training mode and
    predictions in eval mode. Validation still needs loss values, so we
    temporarily switch to training mode for the forward pass.
    """
    was_training = model.training
    model.train()
    try:
        loss_dict = model(images, targets)
    finally:
        if not was_training:
            model.eval()
    return loss_dict


def sanitize_detection_target(
    target: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Drops invalid boxes so the detector only sees finite, non-degenerate targets."""
    boxes = target["boxes"]
    labels = target["labels"]
    image_id = target["image_id"]
    original_count = boxes.shape[0] if boxes.ndim > 0 else 0

    if boxes.numel() == 0:
        empty_boxes = boxes.reshape(0, 4)
        return {
            "boxes": empty_boxes,
            "labels": labels[:0],
            "image_id": image_id,
            "area": torch.zeros((0,), dtype=torch.float32, device=boxes.device),
            "iscrowd": torch.zeros((0,), dtype=torch.int64, device=boxes.device),
        }

    finite_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(labels)
    valid_geom_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    valid_mask = finite_mask & valid_geom_mask

    iscrowd = target.get(
        "iscrowd",
        torch.zeros((original_count,), dtype=torch.int64, device=boxes.device),
    )
    if iscrowd.shape[0] != original_count:
        iscrowd = torch.zeros((original_count,), dtype=torch.int64, device=boxes.device)

    boxes = boxes[valid_mask]
    labels = labels[valid_mask]
    iscrowd = iscrowd[valid_mask]

    if boxes.numel() == 0:
        boxes = boxes.reshape(0, 4)
        area = torch.zeros((0,), dtype=torch.float32, device=boxes.device)
        iscrowd = torch.zeros((0,), dtype=torch.int64, device=boxes.device)
    else:
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return {
        "boxes": boxes,
        "labels": labels,
        "image_id": image_id,
        "area": area,
        "iscrowd": iscrowd,
    }


def sanitize_detection_targets(
    targets: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    """Sanitizes a batch of targets for torchvision detection models."""
    return [sanitize_detection_target(target) for target in targets]


def loss_dict_to_loggable(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    """Converts a loss dictionary into plain floats for diagnostics."""
    output: dict[str, float] = {}
    for name, value in loss_dict.items():
        try:
            output[name] = float(value.detach().item())
        except (TypeError, ValueError):
            output[name] = float("nan")
    return output


def flatten_wandb_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Converts nested config dictionaries into a flat mapping for W&B."""
    flattened: dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_wandb_config(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


def init_wandb_run(
    train_config: dict[str, Any],
    data_config: dict[str, Any],
) -> Any | None:
    """Initializes a Weights & Biases run when enabled in config."""
    wandb_config = train_config.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    if wandb is None:
        raise ImportError(
            "wandb is enabled in config but the package is not installed. "
            "Install it with `pip install wandb`."
        )

    run = wandb.init(
        project=wandb_config.get("project", "multitask-perception"),
        entity=wandb_config.get("entity") or None,
        name=wandb_config.get("run_name") or None,
        job_type=wandb_config.get("job_type", "train-detector"),
        tags=wandb_config.get("tags") or None,
        notes=wandb_config.get("notes") or None,
        mode=wandb_config.get("mode", os.environ.get("WANDB_MODE", "online")),
        config={
            **flatten_wandb_config({"train": train_config}),
            **flatten_wandb_config({"data": data_config}),
        },
    )
    return run


def log_wandb_metrics(
    run: Any | None,
    epoch: int,
    train_epoch_loss: float,
    val_loss: float | None,
    val_metrics: dict[str, float] | None,
    current_lr: float,
) -> None:
    """Logs scalar metrics to W&B when a run is active."""
    if run is None or wandb is None:
        return

    payload: dict[str, Any] = {
        "epoch": epoch,
        "train/epoch_loss": train_epoch_loss,
        "train/epoch_lr": current_lr,
    }
    if val_loss is not None:
        payload["val/loss"] = val_loss
    if val_metrics:
        for metric_name, metric_value in val_metrics.items():
            if isinstance(metric_value, (int, float)):
                payload[f"val/{metric_name}"] = metric_value

    wandb.log(payload, step=epoch)


def upload_wandb_artifact(
    run: Any | None,
    artifact_name: str,
    artifact_type: str,
    artifact_path: Path,
    aliases: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Uploads a file artifact to W&B when enabled."""
    if run is None or wandb is None or not artifact_path.exists():
        return

    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        metadata=metadata or {},
    )
    artifact.add_file(str(artifact_path), name=artifact_path.name)
    run.log_artifact(artifact, aliases=aliases or None)


def main() -> None:
    """Loads configs, trains the detector, logs metrics, and saves checkpoints."""

    data_config = load_yaml(ROOT / "configs" / "data" / "insta360_detection.yaml")
    train_config = load_yaml(ROOT / "configs" / "train" / "detector_demo.yaml")
    wandb_run = init_wandb_run(train_config, data_config)

    # Set seeds
    set_seed(int(train_config.get("split_seed", 42)))

    # Load dataset
    dataset = LabelMeDetectionDataset(
        root_dir=ROOT / data_config["root_dir"],
        image_size=data_config.get("image_size", [640, 1280]),
        class_map=data_config.get("class_map"),
        keep_empty=bool(data_config.get("keep_empty", False)),
    )
    logger.info(f"Loaded dataset with {len(dataset)} samples")

    class_map = data_config["class_map"]
    id_to_class = {class_id: name for name, class_id in class_map.items()}

    # Create splits
    split_ratios = train_config.get(
        "split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    group_split_key = train_config.get("group_split_key")

    # Try to extract group IDs if group-aware splitting is requested
    group_ids = None
    if group_split_key:
        group_ids = extract_group_ids_from_dataset(dataset, group_split_key)
        if group_ids:
            logger.info(f"Using group-aware split with key '{group_split_key}'")

    # Create train/val/test splits
    train_dataset, val_dataset, test_dataset = create_splits(
        dataset=dataset,
        train_ratio=split_ratios.get("train", 0.7),
        val_ratio=split_ratios.get("val", 0.15),
        test_ratio=split_ratios.get("test", 0.15),
        seed=int(train_config.get("split_seed", 42)),
        group_ids=group_ids,
        group_key=group_split_key,
    )

    # Optionally override with k-fold
    use_kfold = bool(train_config.get("use_kfold", False))
    if use_kfold:
        num_folds = int(train_config.get("num_folds", 5))
        fold_index = int(train_config.get("fold_index", 0))
        train_dataset, val_dataset = create_kfold_splits(
            dataset=dataset,
            num_folds=num_folds,
            fold_index=fold_index,
            seed=int(train_config.get("split_seed", 42)),
            group_ids=group_ids,
            group_key=group_split_key,
        )
        logger.info(f"Using k-fold CV: fold {fold_index}/{num_folds}")

    # Setup augmentations
    image_size = data_config.get("image_size", [640, 1280])
    aug_config = train_config.get("augmentation", {})

    train_transforms = build_training_transforms(
        image_size=image_size,
        p_flip=aug_config.get("p_flip", 0.5),
        p_affine=aug_config.get("p_affine", 0.3),
        p_brightness=aug_config.get("p_brightness", 0.3),
        p_blur=aug_config.get("p_blur", 0.1),
        p_noise=aug_config.get("p_noise", 0.05),
    )
    val_transforms = build_validation_transforms(image_size=image_size)

    # Wrap datasets with transforms
    class TransformWrapper:
        def __init__(self, subset, transforms):
            self.subset = subset
            self.transforms = transforms

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            sample = self.subset[idx]
            return self.transforms(sample)

    train_dataset = TransformWrapper(train_dataset, train_transforms)
    if val_dataset:
        val_dataset = TransformWrapper(val_dataset, val_transforms)
    if test_dataset:
        test_dataset = TransformWrapper(test_dataset, val_transforms)

    # Setup samplers
    train_sampler = None
    use_weighted_sampling = bool(train_config.get("use_weighted_sampling", False))
    use_repeat_factor_sampling = bool(
        train_config.get("use_repeat_factor_sampling", False)
    )

    if use_weighted_sampling and not use_repeat_factor_sampling:
        logger.info("Using weighted sampling for training")
        weighted_config = train_config.get("weighted_sampling", {})
        train_sampler = WeightedImageSampler.from_dataset(
            train_dataset,
            pow_factor=weighted_config.get("pow_factor", 0.5),
            max_weight_cap=weighted_config.get("max_weight_cap", 10.0),
        )
    elif use_repeat_factor_sampling and not use_weighted_sampling:
        logger.info("Using repeat-factor sampling for training")
        repeat_config = train_config.get("repeat_factor_sampling", {})
        train_sampler = RepeatFactorSampler(
            train_dataset,
            repeat_factor_threshold=repeat_config.get("repeat_factor_threshold", 0.001),
        )

    # Create dataloaders
    train_loader = make_loader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        shuffle=True,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=1,
        num_workers=int(data_config["num_workers"]),
        shuffle=False,
    )

    # Setup device and model
    device = resolve_device(str(train_config.get("device", "auto")))
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"

    model = build_demo_detector(
        num_classes=len(class_map),
        pretrained=bool(train_config.get("pretrained", True)),
        trainable_backbone_layers=train_config.get("trainable_backbone_layers"),
    ).to(device)

    # Setup training
    optimizer = AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Directories
    checkpoint_dir = ROOT / train_config["checkpoint_dir"]
    tb_dir = ROOT / train_config["tb_dir"]
    pred_dir = ROOT / train_config["pred_dir"]
    plots_dir = ROOT / train_config.get("plots_dir", "outputs/plots")
    metrics_file = ROOT / train_config.get("metrics_file", "outputs/metrics.jsonl")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))
    log_ground_truth_example(writer, dataset, id_to_class)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=str(train_config.get("scheduler_mode", "min")),
        factor=float(train_config.get("scheduler_factor", 0.5)),
        patience=int(train_config.get("scheduler_patience", 5)),
        threshold=float(train_config.get("scheduler_threshold", 1e-4)),
        min_lr=float(train_config.get("min_lr", 1e-6)),
    )

    # Training parameters
    epochs = int(train_config["epochs"])
    freeze_backbone_epochs = int(train_config.get("freeze_backbone_epochs", 0))
    log_every = int(train_config["log_every"])
    vis_every_epochs = int(train_config.get("vis_every_epochs", 1))
    score_thresh = float(train_config.get("score_thresh", 0.5))
    gradient_clip_norm = float(train_config.get("gradient_clip_norm", 10.0))
    backbone_unfreeze_lr_scale = float(
        train_config.get("backbone_unfreeze_lr_scale", 0.1)
    )
    max_consecutive_nonfinite_batches = int(
        train_config.get("max_consecutive_nonfinite_batches", 5)
    )

    early_stopping_metric = str(train_config.get("early_stopping_metric", "val/map_50"))
    early_stopping_patience = int(train_config.get("early_stopping_patience", 10))
    early_stopping_min_delta = float(train_config.get("early_stopping_min_delta", 1e-4))

    best_metric_value = float("inf") if "loss" in early_stopping_metric else 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    amp_enabled = use_amp
    stop_training_due_to_nonfinite = False

    # Main training loop
    for epoch in range(epochs):
        # Manage backbone freezing/unfreezing
        if epoch == freeze_backbone_epochs and freeze_backbone_epochs > 0:
            logger.info(f"Unfreezing backbone at epoch {epoch}")
            unfreeze_backbone(model)
            for param_group in optimizer.param_groups:
                param_group["lr"] *= backbone_unfreeze_lr_scale
            logger.info(
                "Reduced learning rate after backbone unfreeze to %.8f",
                optimizer.param_groups[0]["lr"],
            )
        elif epoch < freeze_backbone_epochs:
            freeze_backbone(model)

        # Training phase
        model.train()
        epoch_losses = []
        consecutive_nonfinite_batches = 0

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets = sanitize_detection_targets(targets)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                loss_dict = compute_detector_loss(model, images, targets)
                total_loss = sum(loss for loss in loss_dict.values())

            if not math.isfinite(float(total_loss.detach().item())):
                consecutive_nonfinite_batches += 1
                logger.warning(
                    "Skipping non-finite training loss at epoch=%03d step=%d (%d/%d): %s",
                    epoch + 1,
                    step,
                    consecutive_nonfinite_batches,
                    max_consecutive_nonfinite_batches,
                    loss_dict_to_loggable(loss_dict),
                )
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    amp_enabled = False
                    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
                    logger.warning(
                        "Disabled AMP after non-finite loss to improve stability."
                    )
                if consecutive_nonfinite_batches >= max_consecutive_nonfinite_batches:
                    logger.error(
                        "Stopping training after %d consecutive non-finite batches at epoch=%03d step=%d. "
                        "The current weights are no longer numerically stable.",
                        consecutive_nonfinite_batches,
                        epoch + 1,
                        step,
                    )
                    stop_training_due_to_nonfinite = True
                    break
                continue

            consecutive_nonfinite_batches = 0
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(total_loss.item())
            epoch_losses.append(loss_value)

            writer.add_scalar("train/total_loss", loss_value, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            for name, value in loss_dict.items():
                writer.add_scalar(f"train/{name}", value.item(), global_step)

            if global_step % log_every == 0:
                logger.info(
                    f"epoch={epoch + 1:03d}/{epochs} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={loss_value:.4f}"
                )
            global_step += 1

        if stop_training_due_to_nonfinite:
            break

        train_epoch_loss = (
            sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")
        )
        writer.add_scalar("train/epoch_loss", train_epoch_loss, epoch + 1)

        # Validation phase
        val_loss = None
        val_metrics = {}

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for images, targets in val_loader:
                    images = [image.to(device) for image in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    targets = sanitize_detection_targets(targets)
                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        loss_dict = compute_detector_loss(model, images, targets)
                    total_loss = sum(loss for loss in loss_dict.values())
                    total_loss_value = float(total_loss.item())
                    if math.isfinite(total_loss_value):
                        val_losses.append(total_loss_value)
                    else:
                        logger.warning(
                            "Skipping non-finite validation loss at epoch=%03d: %s",
                            epoch + 1,
                            loss_dict_to_loggable(loss_dict),
                        )

                val_loss = sum(val_losses) / len(val_losses) if val_losses else None

            if val_loss is not None:
                writer.add_scalar("val/loss", val_loss, epoch + 1)

            # Compute detection metrics
            val_metrics = compute_validation_metrics(
                model=model,
                val_loader=val_loader,
                device=device,
                num_classes=len(class_map),
                score_thresh=score_thresh,
            )

            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    writer.add_scalar(f"val/{metric_name}", metric_value, epoch + 1)

        # Learning rate scheduling
        scheduler_metric_key = train_config.get("scheduler_metric", "val/loss")
        if "loss" in scheduler_metric_key:
            scheduler_metric = val_loss if val_loss is not None else train_epoch_loss
        else:
            scheduler_metric = val_metrics.get(
                scheduler_metric_key.replace("val/", ""), 0.0
            )

        writer.add_scalar("train/monitored_metric", scheduler_metric, epoch + 1)
        scheduler.step(scheduler_metric)

        current_lr = float(optimizer.param_groups[0]["lr"])
        writer.add_scalar("train/epoch_lr", current_lr, epoch + 1)

        # Log metrics to CSV and JSON
        log_csv_metrics(
            checkpoint_dir / "metrics.csv",
            epoch + 1,
            train_epoch_loss,
            val_loss,
            val_metrics,
            current_lr,
        )
        log_json_metrics(
            metrics_file,
            epoch + 1,
            train_epoch_loss,
            val_loss,
            val_metrics,
            current_lr,
        )
        log_wandb_metrics(
            wandb_run,
            epoch + 1,
            train_epoch_loss,
            val_loss,
            val_metrics,
            current_lr,
        )

        # Visualization
        if val_loader is not None and (epoch + 1) % vis_every_epochs == 0:
            log_validation_examples(
                model=model,
                writer=writer,
                val_loader=val_loader,
                device=device,
                epoch=epoch + 1,
                id_to_class=id_to_class,
                score_thresh=score_thresh,
                num_examples=1,
            )

            model.eval()
            with torch.no_grad():
                for images, _ in val_loader:
                    images = [image.to(device) for image in images]
                    predictions = model(images)
                    preview = draw_pred_boxes(
                        images[0].cpu(),
                        predictions[0],
                        id_to_class,
                        score_thresh,
                    )
                    save_tensor_image(
                        preview, pred_dir / f"epoch_{epoch + 1:03d}_preview.png"
                    )
                    break

        # Save checkpoints
        checkpoint_path = checkpoint_dir / f"detector_epoch_{epoch + 1:03d}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "class_map": class_map,
                "image_size": list(image_size),
                "config": train_config,
            },
            checkpoint_path,
        )

        # Best model selection based on mAP or loss
        early_stopping_metric_key = early_stopping_metric.replace("val/", "")
        if "map" in early_stopping_metric_key:
            current_metric = val_metrics.get(early_stopping_metric_key, 0.0)
            is_improvement = current_metric > (
                best_metric_value + early_stopping_min_delta
            )
        else:
            current_metric = val_loss if val_loss is not None else train_epoch_loss
            is_improvement = current_metric < (
                best_metric_value - early_stopping_min_delta
            )

        if is_improvement:
            best_metric_value = current_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_path = checkpoint_dir / "best.pth"
            torch.save(torch.load(checkpoint_path), best_path)
            upload_wandb_artifact(
                wandb_run,
                artifact_name="detector-best-checkpoint",
                artifact_type="model",
                artifact_path=best_path,
                aliases=["best", f"epoch-{epoch + 1:03d}"],
                metadata={
                    "epoch": epoch + 1,
                    "metric_name": early_stopping_metric,
                    "metric_value": current_metric,
                },
            )
            logger.info(
                f"New best checkpoint at epoch {epoch + 1}: "
                f"{early_stopping_metric}={current_metric:.6f}"
            )
        else:
            epochs_without_improvement += 1

        val_loss_text = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        logger.info(
            f"epoch={epoch + 1:03d}/{epochs} "
            f"train_loss={train_epoch_loss:.4f} "
            f"val_loss={val_loss_text} "
            f"lr={current_lr:.6f} "
            f"no_improve={epochs_without_improvement}/{early_stopping_patience}"
        )

        # Save last.pth
        last_path = checkpoint_dir / "last.pth"
        torch.save(torch.load(checkpoint_path), last_path)

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch + 1}; "
                f"best epoch was {best_epoch} with {early_stopping_metric}={best_metric_value:.6f}"
            )
            break

    # Generate plots
    try:
        plot_metrics_summary(plots_dir, metrics_file)
    except Exception as e:
        logger.warning(f"Failed to generate plots: {e}")

    upload_wandb_artifact(
        wandb_run,
        artifact_name="detector-last-checkpoint",
        artifact_type="model",
        artifact_path=checkpoint_dir / "last.pth",
        aliases=["latest"],
        metadata={"best_epoch": best_epoch, "best_metric_value": best_metric_value},
    )
    upload_wandb_artifact(
        wandb_run,
        artifact_name="detector-training-metrics",
        artifact_type="metrics",
        artifact_path=checkpoint_dir / "metrics.csv",
        aliases=["latest"],
        metadata={"best_epoch": best_epoch, "best_metric_value": best_metric_value},
    )
    upload_wandb_artifact(
        wandb_run,
        artifact_name="detector-training-metrics-jsonl",
        artifact_type="metrics",
        artifact_path=metrics_file,
        aliases=["latest"],
        metadata={"best_epoch": best_epoch, "best_metric_value": best_metric_value},
    )

    writer.close()
    if wandb_run is not None and wandb is not None:
        wandb.finish()
    logger.info("Training complete")


if __name__ == "__main__":
    main()
