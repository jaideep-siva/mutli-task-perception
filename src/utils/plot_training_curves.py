"""Plotting utilities for training curves and metrics visualization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
logger = logging.getLogger(__name__)


def load_metrics_from_log(log_file: str | Path) -> dict[str, list[Any]]:
    """Load metrics from JSON log file.

    Args:
        log_file: Path to JSON log file with epoch metrics.

    Returns:
        Dict mapping metric names to lists of values, one per epoch.
    """
    log_file = Path(log_file)
    if not log_file.exists():
        logger.warning(f"Log file not found: {log_file}")
        return {}

    metrics: dict[str, list[Any]] = {}
    try:
        with log_file.open("r") as f:
            for line in f:
                entry = json.loads(line.strip())
                for key, value in entry.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading {log_file}: {e}")

    return metrics


def plot_training_curves(
    metrics: dict[str, list[float]],
    output_dir: str | Path | None = None,
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """Plot training and validation curves.

    Args:
        metrics: Dict with keys like 'train/loss', 'val/loss', 'val/map_50', etc.
        output_dir: Directory to save plots. If None, plots are not saved.
        figsize: Figure size (width, height) in inches.
    """
    output_dir = Path(output_dir) if output_dir else None

    if not metrics:
        logger.warning("No metrics to plot")
        return

    # Group metrics by category
    categories = {
        "loss": ["train/loss", "train/epoch_loss", "val/loss"],
        "map": ["val/map", "val/map_50", "val/map_75"],
        "precision_recall": ["val/precision", "val/recall"],
        "lr": ["train/lr", "train/epoch_lr"],
    }

    for category, keys in categories.items():
        available_keys = [k for k in keys if k in metrics]
        if not available_keys:
            continue

        fig, ax = plt.subplots(figsize=figsize)
        epochs = list(range(1, len(metrics[available_keys[0]]) + 1))

        for key in available_keys:
            values = metrics[key]
            label = key.replace("/", " ").title()
            ax.plot(epochs, values, marker="o", label=label, linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(
            f"Training Curves: {category.replace('_', ' ').title()}", fontsize=14
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"training_curves_{category}.png"
            fig.savefig(plot_path, dpi=100, bbox_inches="tight")
            logger.info(f"Saved plot: {plot_path}")

        plt.close(fig)


def plot_loss_comparison(
    metrics: dict[str, list[float]],
    output_dir: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot train vs validation loss.

    Args:
        metrics: Dict with 'train/loss' and 'val/loss' keys.
        output_dir: Directory to save plot.
        figsize: Figure size (width, height) in inches.
    """
    output_dir = Path(output_dir) if output_dir else None

    train_key = "train/epoch_loss" if "train/epoch_loss" in metrics else "train/loss"
    val_key = "val/loss"

    if train_key not in metrics or val_key not in metrics:
        logger.warning(f"Missing loss metrics. Available: {list(metrics.keys())}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    epochs = list(range(1, len(metrics[train_key]) + 1))

    ax.plot(
        epochs,
        metrics[train_key],
        marker="o",
        label="Train Loss",
        linewidth=2,
        color="blue",
    )
    ax.plot(
        epochs,
        metrics[val_key],
        marker="s",
        label="Validation Loss",
        linewidth=2,
        color="red",
    )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training vs Validation Loss", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "loss_comparison.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved plot: {plot_path}")

    plt.close(fig)


def plot_map_curves(
    metrics: dict[str, list[float]],
    output_dir: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Plot mAP curves (mAP50, mAP50-95).

    Args:
        metrics: Dict with 'val/map_50' and 'val/map' keys.
        output_dir: Directory to save plot.
        figsize: Figure size (width, height) in inches.
    """
    output_dir = Path(output_dir) if output_dir else None

    map_keys = [k for k in metrics.keys() if "map" in k.lower()]
    if not map_keys:
        logger.warning("No mAP metrics found in metrics dict")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Get the length from the first available metric
    first_key = map_keys[0]
    epochs = list(range(1, len(metrics[first_key]) + 1))

    colors = ["blue", "red", "green", "orange"]
    for idx, key in enumerate(sorted(map_keys)):
        values = metrics[key]
        label = key.replace("/", " ").title()
        ax.plot(
            epochs,
            values,
            marker="o",
            label=label,
            linewidth=2,
            color=colors[idx % len(colors)],
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP", fontsize=12)
    ax.set_title("Mean Average Precision", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "map_curves.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        logger.info(f"Saved plot: {plot_path}")

    plt.close(fig)


def plot_metrics_summary(
    output_dir: str | Path,
    metrics_file: str | Path | None = None,
) -> None:
    """Generate all training plots from metrics file or dict.

    Args:
        output_dir: Output directory for all plots.
        metrics_file: Optional JSON log file to load metrics from.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metrics_file is None:
        logger.warning("No metrics file provided. Skipping plot generation.")
        return

    metrics = load_metrics_from_log(metrics_file)
    if not metrics:
        logger.warning("Failed to load metrics. Skipping plot generation.")
        return

    plot_loss_comparison(metrics, output_dir)
    plot_map_curves(metrics, output_dir)
    plot_training_curves(metrics, output_dir)
    logger.info(f"Plot generation complete. Saved to {output_dir}")
