"""Detection dataset for resized LabelMe rectangle annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


DEFAULT_CLASS_MAP: dict[str, int] = {
    "background": 0,
    "barrel": 1,
    "cone_large": 2,
    "cone_small": 3,
    "pedestrian": 4,
    "soup_sign": 5,
    "stop_sign": 6,
}


@dataclass(frozen=True)
class SampleRecord:
    """Index entry for a valid dataset sample."""

    json_path: Path
    image_path: Path
    boxes: list[list[float]]
    labels: list[int]


class LabelMeDetectionDataset(Dataset[dict[str, Any]]):
    """Loads LabelMe rectangle boxes and resizes every image before tensor conversion.

    Args:
        root_dir: Directory scanned recursively for LabelMe ``.json`` files.
        image_size: Target resized ``[height, width]`` used for training/inference.
        class_map: Mapping from normalized class name to integer id. Includes background.
        keep_empty: When ``True``, keep images with no valid boxes.

    Returns:
        A dict with ``image`` shaped ``[3, H, W]`` in ``[0, 1]``, a torchvision-style
        detection ``target`` dict, and the resolved ``image_path`` string.
    """

    def __init__(
        self,
        root_dir: str | Path,
        image_size: list[int] | tuple[int, int] = (640, 1280),
        class_map: dict[str, int] | None = None,
        keep_empty: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.class_map = class_map or DEFAULT_CLASS_MAP.copy()
        self.keep_empty = keep_empty
        self.samples = self._build_index()

        if not self.samples:
            raise RuntimeError(
                f"No valid detection samples were found under '{self.root_dir}'."
            )

    def __len__(self) -> int:
        """Returns the number of indexed samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Loads one image, resizes it, and rescales boxes to the resized frame.

        Args:
            index: Sample index in the precomputed dataset index.

        Returns:
            Dict with ``image``, ``target``, and ``image_path``. Boxes use
            ``[x1, y1, x2, y2]`` in resized pixel coordinates.
        """

        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            orig_width, orig_height = image.size
            resized = image.resize(
                (self.image_size[1], self.image_size[0]),
                resample=Image.BILINEAR,
            )

        image_tensor = F.to_tensor(resized)
        boxes = self._resize_boxes(sample.boxes, orig_width, orig_height)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(sample.labels, dtype=torch.int64)

        if boxes_tensor.numel() == 0:
            boxes_tensor = boxes_tensor.reshape(0, 4)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(sample.labels),), dtype=torch.int64),
        }
        return {
            "image": image_tensor,
            "target": target,
            "image_path": str(sample.image_path),
        }

    def _build_index(self) -> list[SampleRecord]:
        """Scans LabelMe JSON files and keeps only valid rectangle annotations."""

        if not self.root_dir.exists():
            raise RuntimeError(f"Dataset root directory does not exist: {self.root_dir}")

        samples: list[SampleRecord] = []
        for json_path in sorted(self.root_dir.rglob("*.json")):
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    annotation = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue

            image_path = self._resolve_image_path(json_path, annotation)
            if image_path is None or not image_path.exists():
                continue

            boxes, labels = self._parse_rectangles(annotation)
            if boxes or self.keep_empty:
                samples.append(
                    SampleRecord(
                        json_path=json_path,
                        image_path=image_path,
                        boxes=boxes,
                        labels=labels,
                    )
                )
        return samples

    def _resolve_image_path(
        self, json_path: Path, annotation: dict[str, Any]
    ) -> Path | None:
        """Resolves the corresponding image path for one LabelMe annotation file."""

        image_path_value = annotation.get("imagePath")
        if isinstance(image_path_value, str) and image_path_value.strip():
            candidate = Path(image_path_value)
            if not candidate.is_absolute():
                candidate = json_path.parent / candidate
            if candidate.exists():
                return candidate

        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = json_path.with_suffix(suffix)
            if candidate.exists():
                return candidate
        return None

    def _parse_rectangles(
        self, annotation: dict[str, Any]
    ) -> tuple[list[list[float]], list[int]]:
        """Parses valid rectangle boxes and class ids from LabelMe shapes."""

        boxes: list[list[float]] = []
        labels: list[int] = []
        for shape in annotation.get("shapes", []):
            if not isinstance(shape, dict):
                continue
            if shape.get("shape_type") != "rectangle":
                continue

            label = self._normalize_label(shape.get("label", ""))
            class_id = self.class_map.get(label)
            if class_id is None:
                continue

            box = self._parse_rectangle_points(shape.get("points"))
            if box is None:
                continue

            boxes.append(box)
            labels.append(class_id)
        return boxes, labels

    @staticmethod
    def _normalize_label(label: str) -> str:
        """Normalizes a raw label by removing BOMs, trimming whitespace, and lowercasing."""

        return str(label).replace("\ufeff", "").strip().lower()

    @staticmethod
    def _parse_rectangle_points(points: Any) -> list[float] | None:
        """Converts two LabelMe rectangle corner points into ``[x1, y1, x2, y2]``."""

        if not isinstance(points, list) or len(points) != 2:
            return None
        if not all(isinstance(point, list) and len(point) >= 2 for point in points):
            return None

        try:
            x1, y1 = float(points[0][0]), float(points[0][1])
            x2, y2 = float(points[1][0]), float(points[1][1])
        except (TypeError, ValueError):
            return None

        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right <= left or bottom <= top:
            return None
        return [left, top, right, bottom]

    def _resize_boxes(
        self, boxes: list[list[float]], orig_width: int, orig_height: int
    ) -> list[list[float]]:
        """Rescales boxes from original image coordinates into resized image coordinates."""

        if not boxes:
            return []

        scale_x = self.image_size[1] / float(orig_width)
        scale_y = self.image_size[0] / float(orig_height)
        resized_boxes: list[list[float]] = []
        max_x = float(self.image_size[1])
        max_y = float(self.image_size[0])

        for x1, y1, x2, y2 in boxes:
            resized_box = [
                min(max(x1 * scale_x, 0.0), max_x),
                min(max(y1 * scale_y, 0.0), max_y),
                min(max(x2 * scale_x, 0.0), max_x),
                min(max(y2 * scale_y, 0.0), max_y),
            ]
            resized_boxes.append(resized_box)
        return resized_boxes
