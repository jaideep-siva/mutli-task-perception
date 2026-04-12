from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _coerce_loss(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, dict):
        tensors = [item for item in value.values() if torch.is_tensor(item)]
        if not tensors:
            return None
        return torch.stack([tensor.float() for tensor in tensors]).sum()
    if torch.is_tensor(value):
        return value.float()
    return torch.as_tensor(value, dtype=torch.float32)


class MultiTaskLossController(nn.Module):
    def __init__(
        self,
        weighting: str = "fixed",
        detection_weight: float = 1.0,
        segmentation_weight: float = 1.0,
    ):
        super().__init__()
        self.weighting = weighting
        self.detection_weight = float(detection_weight)
        self.segmentation_weight = float(segmentation_weight)

        if weighting == "uncertainty":
            self.log_var_det = nn.Parameter(torch.zeros(()))
            self.log_var_seg = nn.Parameter(torch.zeros(()))
        elif weighting != "fixed":
            raise ValueError(
                f"Unsupported loss weighting '{weighting}'. Expected 'fixed' or 'uncertainty'."
            )

    def forward(
        self,
        detection_loss: Any = None,
        segmentation_loss: Any = None,
    ) -> dict[str, torch.Tensor]:
        det_loss = _coerce_loss(detection_loss)
        seg_loss = _coerce_loss(segmentation_loss)

        device = None
        for candidate in (det_loss, seg_loss):
            if candidate is not None:
                device = candidate.device
                break
        if device is None:
            device = torch.device("cpu")

        zero = torch.zeros((), device=device)
        det_term = det_loss if det_loss is not None else zero
        seg_term = seg_loss if seg_loss is not None else zero
        total = zero.clone()

        if self.weighting == "uncertainty":
            if det_loss is not None:
                total = total + torch.exp(-self.log_var_det) * det_loss + self.log_var_det
            if seg_loss is not None:
                total = total + torch.exp(-self.log_var_seg) * seg_loss + self.log_var_seg
        else:
            if det_loss is not None:
                total = total + self.detection_weight * det_loss
            if seg_loss is not None:
                total = total + self.segmentation_weight * seg_loss

        return {"det": det_term, "seg": seg_term, "total": total}
