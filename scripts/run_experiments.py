from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import collate_fn
from engine.eval import evaluate
from engine.train import build_dataloader, load_yaml, resolve_device, set_seed, train
from models import MultiTaskLoss, MultiTaskPerceptionModel

try:
    import mlflow
except ImportError:
    mlflow = None


EXPERIMENTS = [
    {"phase": "A", "backbone": "resnet18", "segmentation_enabled": False, "exploratory": False},
    {"phase": "A", "backbone": "resnet18", "segmentation_enabled": True, "exploratory": False},
    {"phase": "A", "backbone": "convnext_base", "segmentation_enabled": False, "exploratory": False},
    {"phase": "A", "backbone": "convnext_base", "segmentation_enabled": True, "exploratory": False},
    {"phase": "A", "backbone": "swin_b", "segmentation_enabled": False, "exploratory": False},
    {"phase": "A", "backbone": "swin_b", "segmentation_enabled": True, "exploratory": False},
]

EXPLORATORY_EXPERIMENTS = [
    {"phase": "C", "backbone": "hrnet_w32", "segmentation_enabled": True, "exploratory": True},
]


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.train_manifest:
        cfg["dataset"]["train_manifest"] = args.train_manifest
    if args.val_manifest:
        cfg["dataset"]["val_manifest"] = args.val_manifest
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["dataset"]["num_workers"] = args.num_workers
    if args.device:
        cfg["device"] = args.device
    if args.max_steps_per_epoch:
        cfg["max_steps_per_epoch"] = args.max_steps_per_epoch
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["label_source"] = args.label_source
    cfg["roi_mode"] = args.roi_mode
    if args.roi_mask:
        cfg["roi_mask"] = args.roi_mask


def run_name_for(spec: dict[str, Any]) -> str:
    suffix = "det_seg" if spec["segmentation_enabled"] else "det_only"
    run_name = f"phase_{spec['phase']}_{spec['backbone']}_{suffix}"
    if spec["exploratory"]:
        run_name += "_exploratory"
    return run_name


def configure_run(base_config: str, args: argparse.Namespace, spec: dict[str, Any]) -> tuple[dict[str, Any], str]:
    cfg = load_yaml(base_config)
    apply_overrides(cfg, args)
    cfg["backbone"]["name"] = spec["backbone"]
    cfg["segmentation"]["enabled"] = bool(spec["segmentation_enabled"])
    run_name = run_name_for(spec)
    cfg["run_name"] = run_name
    cfg["experiment_name"] = "multitask_perception"
    cfg["checkpoint_root"] = str(Path(args.output_dir) / "checkpoints" / run_name)
    return cfg, run_name


def preflight_run(cfg: dict[str, Any], config_path: str, run_name: str) -> str:
    set_seed(int(cfg.get("seed", 42)))
    device = resolve_device(str(cfg.get("device", "auto")))
    train_loader = build_dataloader(cfg["dataset"]["train_manifest"], cfg, shuffle=True)
    val_loader = build_dataloader(cfg["dataset"]["val_manifest"], cfg, shuffle=False)
    model = MultiTaskPerceptionModel(cfg).to(device)
    criterion = MultiTaskLoss(cfg)
    optimizer = AdamW(model.parameters(), lr=float(cfg["optimizer"]["lr"]), weight_decay=1e-4)

    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    boxes = [tensor.to(device) for tensor in batch["boxes"]]
    labels = [tensor.to(device) for tensor in batch["labels"]]
    seg_mask = batch["seg_mask"].to(device)
    seg_confidence = batch["seg_confidence"].to(device)
    seg_roi_mask = batch["seg_roi_mask"].to(device)
    has_seg = batch["has_seg"].to(device)

    model.train()
    preds = model(images, postprocess=False)
    losses = criterion(
        preds,
        {
            "detection": (boxes, labels),
            "segmentation": seg_mask,
            "seg_confidence": seg_confidence,
            "seg_roi_mask": seg_roi_mask,
            "has_seg": has_seg,
        },
    )
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
    metrics = evaluate(model, val_loader, cfg, device, epoch=None, artifact_dir=None)

    if mlflow is not None:
        mlflow.set_experiment(str(cfg.get("experiment_name", "multitask_perception")))
        with mlflow.start_run(run_name=f"preflight_{run_name}", nested=False):
            mlflow.log_param("config", config_path)
            mlflow.log_param("backbone", cfg["backbone"]["name"])
            mlflow.log_param("segmentation_enabled", bool(cfg.get("segmentation", {}).get("enabled", True)))
            mlflow.log_metric("preflight_loss_total", float(losses["total"].detach().item()))
            mlflow.log_metric("preflight_val_map", float(metrics["det_mAP"]))
            mlflow.log_metric("preflight_val_seg_iou", float(metrics["seg_iou"]))
        return "preflight ok"
    return "preflight ok; mlflow unavailable"


def write_comparison(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    viability_json_path = output_dir / "backbone_viability_comparison.json"
    viability_csv_path = output_dir / "backbone_viability_comparison.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    viability_json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = [
        "run_name",
        "backbone",
        "segmentation_enabled",
        "best_val_map",
        "best_val_seg_iou",
        "best_val_seg_dice",
        "best_epoch",
        "checkpoint_path",
        "batch_size",
        "latency_ms",
        "throughput_fps",
        "total_training_time_sec",
        "peak_memory_mb",
        "notes",
    ]
    for path in (csv_path, viability_csv_path):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
    print(f"comparison_csv={csv_path}")
    print(f"viability_comparison_csv={viability_csv_path}")


def write_recommendation(rows: list[dict[str, Any]], output_dir: Path) -> None:
    completed = [row for row in rows if not row.get("notes", "").startswith("failed")]
    primary = [row for row in completed if "exploratory" not in row["run_name"] and row.get("segmentation_enabled")]
    det_only = [row for row in completed if "exploratory" not in row["run_name"] and not row.get("segmentation_enabled")]

    def sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(row.get("best_val_map") or 0.0),
            float(row.get("best_val_seg_iou") or 0.0),
            -float(row.get("latency_ms") or 0.0),
        )

    best_overall = max(primary, key=sort_key, default={})
    best_practical = min(primary, key=lambda row: float(row.get("latency_ms") or 1e12), default={})
    best_det_only = max(det_only, key=lambda row: float(row.get("best_val_map") or 0.0), default={})
    recommendation = {
        "best_overall": best_overall.get("run_name", ""),
        "best_practical_fastest": best_practical.get("run_name", ""),
        "best_detection_only": best_det_only.get("run_name", ""),
        "detection_note": "Detection mAP is unavailable when the selected manifests contain no ground-truth boxes; do not use detection-only rows to choose an OD backbone in that case.",
        "exploratory_note": "hrnet_w32 is a lightweight fallback in models/backbone.py, not a true HRNet implementation; do not use it for the final recommendation.",
        "caution": "This is a rough viability comparison on a small manual tape-label dataset. Treat tiny score differences as noise and prefer simpler/faster models when metrics are close.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backbone_viability_recommendation.json").write_text(json.dumps(recommendation, indent=2), encoding="utf-8")


def use_workspace_temp(output_dir: str | Path) -> None:
    temp_dir = Path(output_dir) / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled backbone sweep for equirectangular multitask perception.")
    parser.add_argument("--config", default="configs/multitask/multitask_resnet18.yaml")
    parser.add_argument("--output_dir", default="outputs/comparisons")
    parser.add_argument("--train_manifest", default="")
    parser.add_argument("--val_manifest", default="")
    parser.add_argument("--label_source", default="lane_pseudo_labels")
    parser.add_argument("--roi_mode", default="shared_roi")
    parser.add_argument("--roi_mask", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_exploratory", action="store_true")
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    use_workspace_temp(args.output_dir)

    summaries = []
    experiments = EXPERIMENTS + (EXPLORATORY_EXPERIMENTS if args.include_exploratory else [])
    for spec in experiments:
        cfg, run_name = configure_run(args.config, args, spec)
        print(f"scheduled {run_name}")
        notes = "exploratory fallback" if spec["exploratory"] else ""
        if args.dry_run:
            summaries.append({
                "run_name": run_name,
                "backbone": spec["backbone"],
                "segmentation_enabled": spec["segmentation_enabled"],
                "best_val_map": 0.0,
                "best_val_seg_iou": 0.0,
                "best_val_seg_dice": 0.0,
                "best_epoch": 0,
                "checkpoint_path": "",
                "batch_size": cfg.get("batch_size", ""),
                "latency_ms": "",
                "throughput_fps": "",
                "notes": notes,
            })
            continue
        try:
            if not args.skip_preflight:
                preflight_note = preflight_run(cfg, args.config, run_name)
                notes = f"{notes}; {preflight_note}".strip("; ")
            started = time.perf_counter()
            summary = train(cfg, config_path=args.config)
            summary.setdefault("total_training_time_sec", time.perf_counter() - started)
        except Exception as exc:
            summary = {
                "run_name": run_name,
                "backbone": spec["backbone"],
                "segmentation_enabled": spec["segmentation_enabled"],
                "best_val_map": "",
                "best_val_seg_iou": "",
                "best_val_seg_dice": "",
                "best_epoch": "",
                "checkpoint_path": "",
                "batch_size": cfg.get("batch_size", ""),
                "latency_ms": "",
                "throughput_fps": "",
                "total_training_time_sec": "",
                "peak_memory_mb": "",
                "notes": f"failed: {exc}",
            }
            summaries.append(summary)
            write_comparison(summaries, Path(args.output_dir))
            continue
        summary["notes"] = "; ".join(part for part in [summary.get("notes", ""), notes] if part)
        summaries.append(summary)
        run_summary_path = Path(args.output_dir) / "runs" / f"{run_name}.json"
        run_summary_path.parent.mkdir(parents=True, exist_ok=True)
        run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_comparison(summaries, Path(args.output_dir))
    write_recommendation(summaries, Path(args.output_dir))


if __name__ == "__main__":
    main()
