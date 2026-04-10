"""Detection metrics computation using torchmetrics and COCO-style evaluation."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics.detection import MeanAveragePrecision


class DetectionMetricsComputer:
    """Computes detection metrics using torchmetrics MeanAveragePrecision."""

    def __init__(self, num_classes: int, iou_type: str = "bbox"):
        """Initialize metrics computer.

        Args:
            num_classes: Total number of classes (including background, which is typically 0).
            iou_type: Type of IoU, "bbox" or "segm". Defaults to "bbox".
        """
        self.num_classes = num_classes
        self.metric = MeanAveragePrecision(
            iou_type=iou_type,
            box_format="xyxy",
            class_metrics=True,
        )

    def update(
        self,
        predictions: list[dict[str, Tensor]],
        targets: list[dict[str, Tensor]],
    ) -> None:
        """Update metrics with predictions and targets.

        Args:
            predictions: List of dicts with 'boxes', 'scores', 'labels' tensors.
                Shapes: boxes [N, 4], scores [N], labels [N] (int64).
            targets: List of dicts with 'boxes', 'labels' tensors.
                Shapes: boxes [M, 4], labels [M] (int64).
        """
        # Convert predictions to torchmetrics format
        preds = []
        for pred in predictions:
            preds.append(
                {
                    "boxes": pred["boxes"],
                    "scores": pred["scores"],
                    "labels": pred["labels"],
                }
            )

        # Convert targets to torchmetrics format
        targs = []
        for target in targets:
            targs.append(
                {
                    "boxes": target["boxes"],
                    "labels": target["labels"],
                }
            )

        self.metric.update(preds, targs)

    def compute(self) -> dict[str, Any]:
        """Compute and return metrics.

        Returns:
            Dict with keys:
                - 'map': mAP@[0.5:0.95]
                - 'map_50': mAP@0.5
                - 'map_75': mAP@0.75
                - 'map_per_class': per-class mAP
                - 'mar_100': mAR@100
                - Potentially class-specific metrics
        """
        results = self.metric.compute()

        # Convert tensors to Python floats for logging
        metrics_dict = {}
        for key, value in results.items():
            if isinstance(value, Tensor):
                if value.numel() == 1:
                    metrics_dict[key] = float(value.item())
                else:
                    metrics_dict[key] = (
                        value.tolist() if value.dim() > 0 else float(value.item())
                    )
            else:
                metrics_dict[key] = value

        # Extract commonly used metrics
        output = {
            "map": metrics_dict.get("map", 0.0),
            "map_50": metrics_dict.get("map_50", 0.0),
            "map_75": metrics_dict.get("map_75", 0.0),
            "mar_100": metrics_dict.get("mar_100", 0.0),
        }

        # Add per-class metrics if available
        map_per_class = metrics_dict.get("map_per_class")
        if isinstance(map_per_class, list):
            output["map_per_class"] = map_per_class
        elif map_per_class is not None:
            try:
                output["map_per_class"] = (
                    map_per_class.tolist()
                    if hasattr(map_per_class, "tolist")
                    else list(map_per_class)
                )
            except (ValueError, TypeError):
                pass

        return output

    def reset(self) -> None:
        """Reset the metric state."""
        self.metric.reset()


def compute_detection_metrics_epoch(
    model: torch.nn.Module,
    data_loader: Any,
    device: torch.device,
    num_classes: int,
    score_thresh: float = 0.0,
) -> dict[str, Any]:
    """Computes detection metrics for an entire validation epoch.

    Args:
        model: Detection model in eval mode.
        data_loader: DataLoader with (images, targets) batches.
        device: Torch device.
        num_classes: Total number of classes.
        score_thresh: Minimum score threshold for detections.

    Returns:
        Dict with mAP and other metrics.
    """
    model.eval()
    metric_computer = DetectionMetricsComputer(num_classes)

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            predictions = model(images)

            # Filter by score threshold
            filtered_predictions = []
            for pred in predictions:
                keep = pred["scores"] >= score_thresh
                filtered_predictions.append(
                    {
                        "boxes": pred["boxes"][keep],
                        "scores": pred["scores"][keep],
                        "labels": pred["labels"][keep],
                    }
                )

            metric_computer.update(filtered_predictions, targets)

    metrics = metric_computer.compute()
    return metrics


def compute_precision_recall(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute per-image precision and recall at a given IoU threshold.

    This is a simplified metric and does not match full COCO evaluation,
    but provides quick feedback during training.

    Args:
        predictions: List of dicts with detection outputs.
        targets: List of dicts with ground truth.
        iou_threshold: IoU threshold for matching predictions to targets.

    Returns:
        Dict with 'precision', 'recall' keys as float values.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, target in zip(predictions, targets):
        pred_boxes = pred.get("boxes", torch.empty((0, 4), device=pred["boxes"].device))
        target_boxes = target.get(
            "boxes", torch.empty((0, 4), device=target["boxes"].device)
        )

        if pred_boxes.numel() == 0 and target_boxes.numel() == 0:
            continue

        if pred_boxes.numel() == 0:
            false_negatives += target_boxes.shape[0]
            continue

        if target_boxes.numel() == 0:
            false_positives += pred_boxes.shape[0]
            continue

        # Simple greedy matching (not COCO-style)
        matched_target = set()
        for pred_box in pred_boxes:
            best_iou = 0
            best_target_idx = -1
            for target_idx, target_box in enumerate(target_boxes):
                if target_idx in matched_target:
                    continue
                iou = compute_iou(pred_box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_target_idx = target_idx

            if best_iou >= iou_threshold:
                true_positives += 1
                matched_target.add(best_target_idx)
            else:
                false_positives += 1

        false_negatives += target_boxes.shape[0] - len(matched_target)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )

    return {"precision": precision, "recall": recall}


def compute_iou(box1: Tensor, box2: Tensor) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format.

    Args:
        box1: Tensor of shape [4].
        box2: Tensor of shape [4].

    Returns:
        IoU as a float value.
    """
    x1_min, y1_min, x1_max, y1_max = box1.tolist()
    x2_min, y2_min, x2_max, y2_max = box2.tolist()

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area
