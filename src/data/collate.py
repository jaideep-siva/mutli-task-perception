"""Collate helpers for torchvision detection models."""

from __future__ import annotations

from typing import Any

from torch import Tensor


def detection_collate_fn(
    batch: list[dict[str, Any]],
) -> tuple[list[Tensor], list[dict[str, Tensor]]]:
    """Keeps images as a list because torchvision detectors do not expect stacked tensors.

    Args:
        batch: Dataset samples where each item contains ``image`` and ``target`` keys.

    Returns:
        Tuple of ``images`` and ``targets`` lists in the format expected by
        ``torchvision.models.detection``.
    """

    images = [sample["image"] for sample in batch]
    targets = [sample["target"] for sample in batch]
    return images, targets
