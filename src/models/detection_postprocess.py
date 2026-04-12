from __future__ import annotations

import math
from typing import Any, Dict

import torch
from torchvision.ops import batched_nms


def _normalize_anchor_sizes(
    levels: list[str],
    anchor_sizes: Dict[str, Any] | None,
) -> Dict[str, list[float]]:
    default_sizes: Dict[str, list[float]] = {
        "p3": [32.0],
        "p4": [64.0],
        "p5": [128.0],
    }
    if not anchor_sizes:
        return {level: list(default_sizes[level]) for level in levels}

    normalized: Dict[str, list[float]] = {}
    for level in levels:
        value = anchor_sizes.get(level, default_sizes.get(level, [64.0]))
        if isinstance(value, (list, tuple)):
            normalized[level] = [float(item) for item in value]
        else:
            normalized[level] = [float(value)]
    return normalized


def generate_anchors(
    feature_shapes: Dict[str, tuple[int, int]],
    image_size: tuple[int, int],
    anchor_sizes: Dict[str, Any] | None = None,
    aspect_ratios: list[float] | None = None,
) -> Dict[str, torch.Tensor]:
    device = torch.device("cpu")
    image_height, image_width = image_size
    aspect_ratios = aspect_ratios or [0.5, 1.0, 2.0]
    anchor_sizes = _normalize_anchor_sizes(list(feature_shapes.keys()), anchor_sizes)
    anchors: Dict[str, torch.Tensor] = {}

    for level, (feat_h, feat_w) in feature_shapes.items():
        stride_y = image_height / float(feat_h)
        stride_x = image_width / float(feat_w)
        centers_y = (torch.arange(feat_h, dtype=torch.float32, device=device) + 0.5) * stride_y
        centers_x = (torch.arange(feat_w, dtype=torch.float32, device=device) + 0.5) * stride_x
        grid_y, grid_x = torch.meshgrid(centers_y, centers_x, indexing="ij")
        centers = torch.stack((grid_x, grid_y, grid_x, grid_y), dim=-1).reshape(-1, 4)

        level_anchors = []
        sizes = anchor_sizes[level]
        for idx, size in enumerate(sizes):
            ratio = aspect_ratios[idx % len(aspect_ratios)]
            half_w = 0.5 * size * math.sqrt(ratio)
            half_h = 0.5 * size / math.sqrt(ratio)
            boxes = centers.clone()
            boxes[:, 0] -= half_w
            boxes[:, 1] -= half_h
            boxes[:, 2] += half_w
            boxes[:, 3] += half_h
            level_anchors.append(boxes)

        anchors[level] = torch.cat(level_anchors, dim=0)

    return anchors


def decode_boxes(
    box_deltas: torch.Tensor,
    anchors: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    anchors = anchors.to(box_deltas.device, dtype=box_deltas.dtype)
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0]
    dy = box_deltas[:, 1]
    dw = box_deltas[:, 2].clamp(max=4.0)
    dh = box_deltas[:, 3].clamp(max=4.0)

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = widths * torch.exp(dw)
    pred_h = heights * torch.exp(dh)

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    decoded = torch.stack((x1, y1, x2, y2), dim=-1)
    image_height, image_width = image_size
    decoded[:, 0::2] = decoded[:, 0::2].clamp(0, image_width)
    decoded[:, 1::2] = decoded[:, 1::2].clamp(0, image_height)
    return decoded


def postprocess_detections(
    raw_outputs: Dict[str, torch.Tensor],
    feature_shapes: Dict[str, tuple[int, int]],
    image_size: tuple[int, int],
    config: Dict[str, Any] | None = None,
) -> list[dict[str, torch.Tensor]]:
    config = config or {}
    score_threshold = float(config.get("score_threshold", 0.25))
    nms_threshold = float(config.get("nms_threshold", 0.5))
    topk_candidates = int(config.get("topk_candidates", 1000))
    max_detections = int(config.get("max_detections", 100))
    aspect_ratios = [float(ratio) for ratio in config.get("aspect_ratios", [0.5, 1.0, 2.0])]
    anchors_by_level = generate_anchors(
        feature_shapes=feature_shapes,
        image_size=image_size,
        anchor_sizes=config.get("anchor_sizes"),
        aspect_ratios=aspect_ratios,
    )

    ordered_levels = list(raw_outputs.keys())
    batch_size = next(iter(raw_outputs.values())).shape[0]
    predictions: list[dict[str, torch.Tensor]] = []

    for batch_idx in range(batch_size):
        per_level = [raw_outputs[level][batch_idx] for level in ordered_levels]
        raw_pred = torch.cat(per_level, dim=0)
        anchors = torch.cat(
            [anchors_by_level[level].to(raw_pred.device) for level in ordered_levels], dim=0
        )

        box_deltas = raw_pred[:, :4]
        objectness = raw_pred[:, 4].sigmoid()
        class_probs = raw_pred[:, 5:].softmax(dim=-1)
        class_scores, labels = class_probs.max(dim=-1)
        scores = objectness * class_scores

        keep = scores >= score_threshold
        if keep.any():
            scores = scores[keep]
            labels = labels[keep]
            box_deltas = box_deltas[keep]
            anchors = anchors[keep]
        else:
            predictions.append(
                {
                    "boxes": anchors.new_zeros((0, 4)),
                    "scores": anchors.new_zeros((0,)),
                    "labels": anchors.new_zeros((0,), dtype=torch.long),
                }
            )
            continue

        if topk_candidates > 0 and scores.numel() > topk_candidates:
            topk_scores, topk_indices = scores.topk(topk_candidates)
            scores = topk_scores
            labels = labels[topk_indices]
            box_deltas = box_deltas[topk_indices]
            anchors = anchors[topk_indices]

        boxes = decode_boxes(box_deltas, anchors, image_size=image_size)
        keep_indices = batched_nms(boxes, scores, labels, nms_threshold)
        keep_indices = keep_indices[:max_detections]

        predictions.append(
            {
                "boxes": boxes[keep_indices],
                "scores": scores[keep_indices],
                "labels": labels[keep_indices],
            }
        )

    return predictions
