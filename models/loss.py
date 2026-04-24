from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .segmentation_roi import roi_valid_mask


class MultiTaskLoss(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.det_weight = float(cfg.get("loss_weights", {}).get("detection", 1.0))
        self.seg_weight = float(cfg.get("loss_weights", {}).get("segmentation", 1.0))
        self.num_det_classes = int(cfg["detection"]["num_classes"])
        self.detection_enabled = bool(cfg.get("detection", {}).get("enabled", True))
        self.segmentation_enabled = bool(cfg.get("segmentation", {}).get("enabled", True))
        self.seg_loss_type = str(cfg.get("segmentation", {}).get("loss", "ce")).lower()
        self.ignore_index = int(cfg.get("segmentation", {}).get("ignore_index", -100))
        class_weights = cfg.get("segmentation", {}).get("class_weights")
        if class_weights is not None:
            self.register_buffer("seg_class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.seg_class_weights = None

    def forward(self, preds: dict[str, Any], targets: dict[str, Any]) -> dict[str, torch.Tensor]:
        zero = next(iter(preds["detection"].values())).sum() * 0.0
        if self.detection_enabled:
            det_loss = self._detection_loss(
                preds["detection"],
                boxes_list=targets["detection"][0],
                labels_list=targets["detection"][1],
                image_size=tuple(targets["segmentation"].shape[-2:]),
            )
        else:
            det_loss = zero

        if self.segmentation_enabled and targets.get("has_seg", torch.tensor(False)).any():
            seg_loss = self._segmentation_loss(
                logits=preds["segmentation"],
                masks=targets["segmentation"],
                confidence=targets.get("seg_confidence"),
                roi_mask=targets.get("seg_roi_mask"),
                has_seg=targets.get("has_seg"),
            )
        else:
            seg_loss = zero

        total = self.det_weight * det_loss + self.seg_weight * seg_loss
        return {"det": det_loss, "seg": seg_loss, "total": total}

    def _segmentation_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        confidence: torch.Tensor | None,
        roi_mask: torch.Tensor | None,
        has_seg: torch.Tensor | None,
    ) -> torch.Tensor:
        valid_bool = roi_valid_mask(masks.to(logits.device), roi_mask=roi_mask, has_seg=has_seg)
        valid_bool &= masks.to(logits.device) != self.ignore_index
        valid = valid_bool.to(logits.dtype)
        if confidence is None:
            confidence = torch.ones((logits.shape[0],), device=logits.device, dtype=logits.dtype)
        confidence = confidence.to(logits.device, dtype=logits.dtype).view(-1, 1, 1)
        safe_masks = masks.to(logits.device).clone()
        safe_masks = torch.where(valid_bool, safe_masks, torch.zeros_like(safe_masks))
        safe_masks = safe_masks.clamp(min=0)
        valid_weight = valid * confidence
        if self.seg_class_weights is not None:
            class_weights = self.seg_class_weights.to(logits.device, dtype=logits.dtype)
            if class_weights.numel() != logits.shape[1]:
                raise ValueError(
                    f"segmentation.class_weights has {class_weights.numel()} values, "
                    f"but the segmentation head has {logits.shape[1]} classes"
                )
            valid_weight = valid_weight * class_weights[safe_masks.long()]
        denom = valid_weight.sum().clamp(min=1.0)

        if self.seg_loss_type == "dice":
            return self._dice_loss(logits, safe_masks, valid_weight)
        ce = F.cross_entropy(logits, safe_masks.long(), reduction="none")
        return (ce * valid_weight).sum() / denom

    @staticmethod
    def _dice_loss(logits: torch.Tensor, masks: torch.Tensor, valid_weight: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        classes = logits.shape[1]
        one_hot = F.one_hot(masks.long().clamp(min=0, max=classes - 1), num_classes=classes).permute(0, 3, 1, 2).float()
        weight = valid_weight.unsqueeze(1)
        intersection = (probs * one_hot * weight).sum(dim=(0, 2, 3))
        union = ((probs + one_hot) * weight).sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return 1.0 - dice.mean()

    def _detection_loss(
        self,
        preds: dict[str, torch.Tensor],
        boxes_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        total_obj = preds["p3"].new_tensor(0.0)
        total_box = preds["p3"].new_tensor(0.0)
        total_cls = preds["p3"].new_tensor(0.0)
        for level_name, pred in preds.items():
            obj_target, box_target, cls_target, pos_mask = self._build_targets_for_level(pred, boxes_list, labels_list, image_size, level_name)
            obj_logits = pred[:, 4]
            box_logits = torch.relu(pred[:, :4].permute(0, 2, 3, 1))
            cls_logits = pred[:, 5:].permute(0, 2, 3, 1)
            total_obj = total_obj + F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction="mean")
            if pos_mask.any():
                total_box = total_box + F.l1_loss(box_logits[pos_mask], box_target[pos_mask], reduction="mean")
                total_cls = total_cls + F.cross_entropy(cls_logits[pos_mask], cls_target[pos_mask], reduction="mean")
        return total_obj + total_box + total_cls

    def _build_targets_for_level(
        self,
        pred: torch.Tensor,
        boxes_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
        image_size: tuple[int, int],
        level_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, _, feat_h, feat_w = pred.shape
        input_h, input_w = image_size
        stride_y = input_h / float(feat_h)
        stride_x = input_w / float(feat_w)
        obj_target = pred.new_zeros((batch_size, feat_h, feat_w))
        box_target = pred.new_zeros((batch_size, feat_h, feat_w, 4))
        cls_target = torch.zeros((batch_size, feat_h, feat_w), device=pred.device, dtype=torch.long)
        pos_mask = torch.zeros((batch_size, feat_h, feat_w), device=pred.device, dtype=torch.bool)
        for batch_idx, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            if boxes.numel() == 0:
                continue
            boxes = boxes.to(pred.device)
            labels = labels.to(pred.device).long()
            for box, label in zip(boxes, labels):
                if label < 0 or label >= self.num_det_classes:
                    continue
                width = float(box[2] - box[0])
                height = float(box[3] - box[1])
                if self._select_level(max(width, height)) != level_name:
                    continue
                center_x = float((box[0] + box[2]) * 0.5)
                center_y = float((box[1] + box[3]) * 0.5)
                grid_x = min(feat_w - 1, max(0, int(center_x / stride_x)))
                grid_y = min(feat_h - 1, max(0, int(center_y / stride_y)))
                cell_x = (grid_x + 0.5) * stride_x
                cell_y = (grid_y + 0.5) * stride_y
                obj_target[batch_idx, grid_y, grid_x] = 1.0
                box_target[batch_idx, grid_y, grid_x] = torch.tensor([
                    max((cell_x - float(box[0])) / stride_x, 0.0),
                    max((cell_y - float(box[1])) / stride_y, 0.0),
                    max((float(box[2]) - cell_x) / stride_x, 0.0),
                    max((float(box[3]) - cell_y) / stride_y, 0.0),
                ], device=pred.device, dtype=pred.dtype)
                cls_target[batch_idx, grid_y, grid_x] = label
                pos_mask[batch_idx, grid_y, grid_x] = True
        return obj_target, box_target, cls_target, pos_mask

    @staticmethod
    def _select_level(max_side: float) -> str:
        if max_side <= 64:
            return "p3"
        if max_side <= 128:
            return "p4"
        return "p5"
