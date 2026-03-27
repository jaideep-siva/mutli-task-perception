"""Train a minimal torchvision Faster R-CNN demo on resized LabelMe data."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:  # pragma: no cover - explicit failure path requested
    raise ImportError(
        "TensorBoard support requires 'torch.utils.tensorboard'. "
        "Please install tensorboard before running this demo."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.collate import detection_collate_fn
from src.data.detection_dataset import LabelMeDetectionDataset
from src.models.demo_detector import build_demo_detector
from src.utils.config import load_yaml
from src.utils.visualization import draw_gt_boxes, draw_pred_boxes


def resolve_device(device_name: str) -> torch.device:
    """Selects CUDA when available, otherwise falls back to CPU.

    Args:
        device_name: Requested device string from config. ``auto`` uses CUDA if possible.

    Returns:
        A torch device instance.
    """

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_seed(seed: int) -> None:
    """Sets random seeds used by the simple train/validation split."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(
    dataset: LabelMeDetectionDataset, val_fraction: float, seed: int
) -> tuple[Subset[Any], Subset[Any] | None]:
    """Builds a deterministic train/validation split from one dataset instance."""

    num_samples = len(dataset)
    indices = list(range(num_samples))
    generator = random.Random(seed)
    generator.shuffle(indices)

    val_count = int(num_samples * val_fraction)
    if val_fraction > 0.0 and val_count == 0 and num_samples > 1:
        val_count = 1
    if val_count >= num_samples:
        val_count = max(0, num_samples - 1)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    if not train_indices:
        raise RuntimeError("Train split is empty. Reduce val_fraction or add more samples.")

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if val_indices else None
    return train_subset, val_subset


def make_loader(
    dataset: Subset[Any] | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[Any] | None:
    """Constructs a DataLoader for torchvision detection samples."""

    if dataset is None:
        return None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    """Saves a drawn ``uint8`` image tensor shaped ``[3, H, W]`` to disk."""

    array = image.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def log_ground_truth_example(
    writer: SummaryWriter,
    dataset: LabelMeDetectionDataset,
    id_to_class: dict[int, str],
) -> None:
    """Logs one resized ground-truth example to TensorBoard at training start."""

    sample = dataset[0]
    drawn = draw_gt_boxes(sample["image"], sample["target"], id_to_class)
    writer.add_image("ground_truth/example", drawn, 0)


def log_validation_predictions(
    model: torch.nn.Module,
    writer: SummaryWriter,
    val_loader: DataLoader[Any] | None,
    device: torch.device,
    epoch: int,
    id_to_class: dict[int, str],
    score_thresh: float,
) -> None:
    """Runs lightweight validation inference for visualization only."""

    if val_loader is None:
        return

    model.eval()
    with torch.no_grad():
        for step, (images, _) in enumerate(val_loader):
            images = [image.to(device) for image in images]
            predictions = model(images)
            drawn = draw_pred_boxes(images[0].cpu(), predictions[0], id_to_class, score_thresh)
            writer.add_image("predictions/example", drawn, epoch)
            if step == 0:
                break


def main() -> None:
    """Loads configs, trains the detector, logs TensorBoard summaries, and saves checkpoints."""

    data_config = load_yaml(ROOT / "configs" / "data" / "insta360_detection.yaml")
    train_config = load_yaml(ROOT / "configs" / "train" / "detector_demo.yaml")

    set_seed(int(train_config["seed"]))

    dataset = LabelMeDetectionDataset(
        root_dir=ROOT / data_config["root_dir"],
        image_size=data_config.get("image_size", [640, 1280]),
        class_map=data_config.get("class_map"),
        keep_empty=bool(data_config.get("keep_empty", False)),
    )
    train_dataset, val_dataset = split_dataset(
        dataset,
        val_fraction=float(data_config.get("val_fraction", 0.0)),
        seed=int(train_config["seed"]),
    )

    train_loader = make_loader(
        train_dataset,
        batch_size=int(data_config["batch_size"]),
        num_workers=int(data_config["num_workers"]),
        shuffle=True,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=1,
        num_workers=int(data_config["num_workers"]),
        shuffle=False,
    )

    device = resolve_device(str(train_config.get("device", "auto")))
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"

    class_map = data_config["class_map"]
    id_to_class = {class_id: name for name, class_id in class_map.items()}
    model = build_demo_detector(num_classes=len(class_map)).to(device)
    optimizer = AdamW(
        params=[parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint_dir = ROOT / train_config["checkpoint_dir"]
    tb_dir = ROOT / train_config["tb_dir"]
    pred_dir = ROOT / train_config["pred_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(tb_dir))
    log_ground_truth_example(writer, dataset, id_to_class)

    global_step = 0
    epochs = int(train_config["epochs"])
    log_every = int(train_config["log_every"])
    vis_every_epochs = int(train_config.get("vis_every_epochs", 1))
    score_thresh = float(train_config.get("score_thresh", 0.5))

    for epoch in range(epochs):
        model.train()
        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            for name, value in loss_dict.items():
                writer.add_scalar(f"train/{name}", value.item(), global_step)

            if global_step % log_every == 0:
                print(
                    f"epoch={epoch + 1}/{epochs} "
                    f"step={step}/{len(train_loader)} "
                    f"loss={total_loss.item():.4f}"
                )
            global_step += 1

        if val_loader is not None and (epoch + 1) % vis_every_epochs == 0:
            log_validation_predictions(
                model=model,
                writer=writer,
                val_loader=val_loader,
                device=device,
                epoch=epoch + 1,
                id_to_class=id_to_class,
                score_thresh=score_thresh,
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
                    save_tensor_image(preview, pred_dir / f"epoch_{epoch + 1:03d}_preview.png")
                    break

        checkpoint_path = checkpoint_dir / f"detector_epoch_{epoch + 1:03d}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "class_map": class_map,
                "image_size": list(data_config["image_size"]),
            },
            checkpoint_path,
        )
        print(f"saved checkpoint: {checkpoint_path}")

    writer.close()


if __name__ == "__main__":
    main()
