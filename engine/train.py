from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data import MultitaskDataset, collate_fn
from engine.eval import evaluate
from models import MultiTaskLoss, MultiTaskPerceptionModel

try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    mlflow = None


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(manifest_path: str | Path, cfg: dict[str, Any], shuffle: bool) -> DataLoader:
    device = resolve_device(str(cfg.get("device", "auto")))
    dataset = MultitaskDataset(
        manifest_path=manifest_path,
        image_size=cfg["dataset"]["image_size"],
        image_root=cfg["dataset"].get("image_root"),
        seg_root=cfg["dataset"].get("seg_root"),
        normalize=bool(cfg["dataset"].get("normalize", True)),
        augment=cfg.get("augment") if shuffle else None,
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=int(cfg["dataset"].get("num_workers", 4)),
        pin_memory=device.type == "cuda",
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: dict[str, Any],
    cfg: dict[str, Any],
    run_id: str,
    best_combined_so_far: float,
) -> tuple[float, str]:
    checkpoint = {
        "epoch": epoch,
        "run_id": run_id,
        "backbone": cfg["backbone"]["name"],
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics": metrics,
        "cfg": cfg,
    }
    path = Path(cfg.get("checkpoint_root", "checkpoints")) / str(cfg["backbone"]["name"]) / f"epoch_{epoch:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    combined = float(metrics["det_mAP"]) + float(metrics["seg_iou"])
    if combined > best_combined_so_far:
        best_path = path.parent / "best.pt"
        torch.save(checkpoint, best_path)
        return combined, str(best_path)
    return best_combined_so_far, str(path.parent / "best.pt")


def train(cfg: dict[str, Any], config_path: str | Path = "configs/multitask/multitask_resnet18.yaml") -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader = build_dataloader(cfg["dataset"]["train_manifest"], cfg, shuffle=True)
    val_manifest = cfg["dataset"].get("val_manifest")
    val_loader = build_dataloader(val_manifest, cfg, shuffle=False) if val_manifest else None

    model = MultiTaskPerceptionModel(cfg).to(device)
    criterion = MultiTaskLoss(cfg)
    optimizer = AdamW(model.parameters(), lr=float(cfg["optimizer"]["lr"]), weight_decay=1e-4)
    total_steps = max(1, int(cfg["epochs"]) * len(train_loader))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    log_every = int(cfg.get("log_every", 10))
    best_combined = float("-inf")
    best_epoch = 0
    best_metrics = {"det_mAP": 0.0, "seg_iou": 0.0, "seg_dice": 0.0, "latency_ms": 0.0, "throughput_fps": 0.0}
    best_checkpoint = ""
    run_name = str(cfg.get("run_name") or cfg["backbone"]["name"])
    training_start = time.perf_counter()
    notes: list[str] = []

    class _NullRun:
        class _Info:
            run_id = ""

        info = _Info()

        def __enter__(self) -> "_NullRun":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    if mlflow is not None:
        mlflow.set_experiment(str(cfg.get("experiment_name", "multitask_perception")))
        run_context = mlflow.start_run(run_name=run_name)
    else:
        notes.append("mlflow unavailable")
        run_context = _NullRun()

    with run_context as run:
        if mlflow is not None:
            mlflow.log_params(
            {
                "backbone": cfg["backbone"]["name"],
                "fpn_channels": cfg["fpn"]["out_channels"],
                "det_classes": cfg["detection"]["num_classes"],
                "seg_classes": cfg["segmentation"]["num_classes"],
                "segmentation_enabled": bool(cfg.get("segmentation", {}).get("enabled", True)),
                "label_source": cfg.get("label_source", "unknown"),
                "train_manifest": cfg["dataset"].get("train_manifest", ""),
                "val_manifest": cfg["dataset"].get("val_manifest", ""),
                "roi_mode": cfg.get("roi_mode", "none"),
                "roi_mask": cfg.get("roi_mask", cfg["dataset"].get("seg_roi_mask", "")),
                "loss_weighting": cfg.get("loss_weighting", "manual"),
                "epochs": cfg["epochs"],
                    "batch_size": cfg["batch_size"],
                    "lr": cfg["optimizer"]["lr"],
                    "seed": cfg.get("seed", 42),
                }
            )

        global_step = 0
        for epoch in range(int(cfg["epochs"])):
            start_time = time.perf_counter()
            model.train()
            det_running = seg_running = total_running = 0.0
            seg_batch_fraction = mean_seg_conf = mean_roi_fraction = 0.0
            num_batches = 0
            samples_seen = 0
            max_steps = int(cfg.get("max_steps_per_epoch", 0))

            for batch in train_loader:
                images = batch["image"].to(device)
                boxes = [tensor.to(device) for tensor in batch["boxes"]]
                labels = [tensor.to(device) for tensor in batch["labels"]]
                seg_mask = batch["seg_mask"].to(device)
                seg_confidence = batch["seg_confidence"].to(device)
                seg_roi_mask = batch["seg_roi_mask"].to(device)
                has_seg = batch["has_seg"].to(device)

                optimizer.zero_grad()
                preds = model(images, postprocess=False)
                targets = {
                    "detection": (boxes, labels),
                    "segmentation": seg_mask,
                    "seg_confidence": seg_confidence,
                    "seg_roi_mask": seg_roi_mask,
                    "has_seg": has_seg,
                }
                losses = criterion(preds, targets)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                scheduler.step()

                det_value = float(losses["det"].detach().item())
                seg_value = float(losses["seg"].detach().item())
                total_value = float(losses["total"].detach().item())
                det_running += det_value
                seg_running += seg_value
                total_running += total_value
                seg_batch_fraction += float(has_seg.float().mean().item())
                mean_seg_conf += float(seg_confidence[has_seg].mean().item()) if has_seg.any() else 0.0
                mean_roi_fraction += float(seg_roi_mask[has_seg].mean().item()) if has_seg.any() else 0.0
                num_batches += 1
                samples_seen += int(images.shape[0])

                if global_step % log_every == 0:
                    LOGGER.info(
                        "epoch=%03d step=%05d loss_det=%.4f loss_seg=%.4f loss_total=%.4f seg_batch_frac=%.3f seg_conf=%.3f roi_frac=%.3f",
                        epoch + 1, global_step, det_value, seg_value, total_value,
                        float(has_seg.float().mean().item()),
                        float(seg_confidence[has_seg].mean().item()) if has_seg.any() else 0.0,
                        float(seg_roi_mask[has_seg].mean().item()) if has_seg.any() else 0.0,
                    )
                global_step += 1
                if max_steps and num_batches >= max_steps:
                    break

            avg_det_loss = det_running / max(1, num_batches)
            avg_seg_loss = seg_running / max(1, num_batches)
            avg_total_loss = total_running / max(1, num_batches)
            avg_seg_batch_fraction = seg_batch_fraction / max(1, num_batches)
            avg_seg_conf = mean_seg_conf / max(1, num_batches)
            avg_roi_fraction = mean_roi_fraction / max(1, num_batches)
            throughput_fps = samples_seen / max(1e-6, time.perf_counter() - start_time)

            if val_loader is not None:
                metrics = evaluate(model, val_loader, cfg, device, epoch + 1, Path("artifacts") / run_name)
            else:
                metrics = {"det_mAP": 0.0, "det_mAP_50": 0.0, "seg_iou": 0.0, "per_class_iou": []}
                metrics["seg_dice"] = 0.0
                metrics["latency_ms"] = 0.0
                metrics["throughput_fps"] = 0.0

            if mlflow is not None:
                mlflow.log_metrics(
                    {
                        "train/loss_det": avg_det_loss,
                        "train/loss_seg": avg_seg_loss,
                        "train/loss_total": avg_total_loss,
                        "train/seg_batch_fraction": avg_seg_batch_fraction,
                        "train/mean_seg_confidence": avg_seg_conf,
                        "train/valid_roi_fraction": avg_roi_fraction,
                        "train/throughput_fps": throughput_fps,
                        "val/det_mAP": metrics["det_mAP"],
                        "val/seg_iou": metrics["seg_iou"],
                        "val/seg_dice": metrics["seg_dice"],
                        "val/latency_ms": metrics["latency_ms"],
                        "val/throughput_fps": metrics["throughput_fps"],
                    },
                    step=epoch,
                )
            per_class_path = Path("artifacts") / run_name / f"epoch_{epoch + 1:03d}_per_class_iou.json"
            per_class_path.parent.mkdir(parents=True, exist_ok=True)
            per_class_path.write_text(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "per_class_iou": metrics.get("per_class_iou", []),
                        "per_class_dice": metrics.get("per_class_dice", []),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if mlflow is not None:
                mlflow.log_artifact(str(per_class_path))

            previous_best = best_combined
            best_combined, best_checkpoint = save_checkpoint(model, optimizer, scheduler, epoch + 1, metrics, cfg, run.info.run_id, best_combined)
            if best_combined > previous_best:
                best_epoch = epoch + 1
                best_metrics = {
                    "det_mAP": metrics["det_mAP"],
                    "seg_iou": metrics["seg_iou"],
                    "seg_dice": metrics["seg_dice"],
                    "latency_ms": metrics["latency_ms"],
                    "throughput_fps": metrics["throughput_fps"],
                }

        if mlflow is not None:
            try:
                mlflow.pytorch.log_model(model, artifact_path="model")
            except Exception as exc:
                notes.append(f"mlflow model artifact skipped: {exc}")
                LOGGER.warning("mlflow model artifact skipped for %s: %s", run_name, exc)
            if Path(config_path).exists():
                try:
                    mlflow.log_artifact(str(config_path))
                except Exception as exc:
                    notes.append(f"mlflow config artifact skipped: {exc}")
                    LOGGER.warning("mlflow config artifact skipped for %s: %s", run_name, exc)

    total_training_time_sec = time.perf_counter() - training_start
    peak_memory_mb = ""
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    summary = {
        "run_name": run_name,
        "backbone": cfg["backbone"]["name"],
        "segmentation_enabled": bool(cfg.get("segmentation", {}).get("enabled", True)),
        "best_val_map": "" if float(best_metrics["det_mAP"]) < 0.0 else best_metrics["det_mAP"],
        "best_val_seg_iou": best_metrics["seg_iou"],
        "best_val_seg_dice": best_metrics["seg_dice"],
        "best_epoch": best_epoch,
        "checkpoint_path": best_checkpoint,
        "batch_size": int(cfg["batch_size"]),
        "latency_ms": best_metrics["latency_ms"],
        "throughput_fps": best_metrics["throughput_fps"],
        "total_training_time_sec": total_training_time_sec,
        "peak_memory_mb": peak_memory_mb,
        "notes": "; ".join(["det mAP unavailable"] + notes) if float(best_metrics["det_mAP"]) < 0.0 else "; ".join(notes),
    }
    return summary
