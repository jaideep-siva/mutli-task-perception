"""Minimal visualization helpers for detection targets and predictions."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
from torchvision.utils import draw_bounding_boxes


def tensor_to_uint8_image(image: Tensor) -> Tensor:
    """Converts a float image tensor in ``[0, 1]`` to ``uint8`` ``[3, H, W]``.

    Args:
        image: Float tensor shaped ``[3, H, W]``.

    Returns:
        ``uint8`` tensor shaped ``[3, H, W]`` suitable for drawing utilities.
    """

    return image.detach().cpu().clamp(0.0, 1.0).mul(255).to(torch.uint8)


def draw_gt_boxes(
    image: Tensor,
    target: Mapping[str, Tensor],
    id_to_class: Mapping[int, str],
) -> Tensor:
    """Draws ground-truth boxes with class labels on an image tensor.

    Args:
        image: Float tensor shaped ``[3, H, W]`` in ``[0, 1]``.
        target: Detection target containing ``boxes`` and ``labels`` tensors.
        id_to_class: Mapping from integer class id to class name.

    Returns:
        ``uint8`` tensor shaped ``[3, H, W]`` with drawn boxes.
    """

    canvas = tensor_to_uint8_image(image)
    boxes = target["boxes"].detach().cpu()
    labels = [
        id_to_class.get(int(label), str(int(label)))
        for label in target["labels"].detach().cpu().tolist()
    ]
    if boxes.numel() == 0:
        return canvas
    return draw_bounding_boxes(canvas, boxes=boxes, labels=labels, colors="green", width=3)


def draw_pred_boxes(
    image: Tensor,
    prediction: Mapping[str, Tensor],
    id_to_class: Mapping[int, str],
    score_thresh: float,
) -> Tensor:
    """Draws predicted boxes above a score threshold as ``label:score`` strings.

    Args:
        image: Float tensor shaped ``[3, H, W]`` in ``[0, 1]``.
        prediction: Detection output with ``boxes``, ``labels``, and ``scores``.
        id_to_class: Mapping from integer class id to class name.
        score_thresh: Minimum score required to render a predicted box.

    Returns:
        ``uint8`` tensor shaped ``[3, H, W]`` with predicted boxes drawn.
    """

    canvas = tensor_to_uint8_image(image)
    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels_tensor = prediction["labels"].detach().cpu()
    keep = scores >= score_thresh
    if keep.sum().item() == 0:
        return canvas

    kept_boxes = boxes[keep]
    kept_scores = scores[keep]
    kept_labels = labels_tensor[keep]
    text = [
        f"{id_to_class.get(int(label), str(int(label)))}:{score:.2f}"
        for label, score in zip(kept_labels.tolist(), kept_scores.tolist())
    ]
    return draw_bounding_boxes(
        canvas,
        boxes=kept_boxes,
        labels=text,
        colors="red",
        width=3,
    )
