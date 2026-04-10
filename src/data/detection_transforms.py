"""Albumentations-based transforms for object detection with bbox-aware handling."""

from __future__ import annotations

from typing import Any

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch import Tensor


class DetectionTransforms:
    """Wraps Albumentations transforms for detection, maintaining torchvision compatibility.

    Handles detection samples with images and bounding boxes in Pascal VOC format [x1, y1, x2, y2].
    """

    def __init__(self, transforms: A.Compose | None = None):
        """Initialize with an Albumentations compose object.

        Args:
            transforms: An Albumentations Compose object for augmentation.
        """
        self.transforms = transforms

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply transforms to a detection sample.

        Args:
            sample: Dict with 'image', 'target' keys. Target has 'boxes' and 'labels'.

        Returns:
            Transformed sample with same structure and tensor dtypes.
        """
        if self.transforms is None:
            return sample

        image = sample["image"]
        target = sample["target"]

        # Convert to numpy for Albumentations (uint8 [H, W, C])
        image_np = self._tensor_to_image_np(image)
        boxes = (
            target["boxes"].cpu().numpy().tolist()
            if target["boxes"].numel() > 0
            else []
        )
        labels = (
            target["labels"].cpu().numpy().tolist()
            if target["labels"].numel() > 0
            else []
        )

        # Apply Albumentations
        transformed = self.transforms(image=image_np, bboxes=boxes, class_labels=labels)

        # Extract results
        image_transformed = transformed["image"]
        boxes_transformed = transformed.get("bboxes", [])
        labels_transformed = transformed.get("class_labels", [])

        # Convert back to tensors
        if isinstance(image_transformed, Tensor):
            image_tensor = image_transformed
        else:
            # If not already a tensor (should be from ToTensorV2), convert
            image_tensor = torch.from_numpy(image_transformed).permute(2, 0, 1).float()

        boxes_tensor = torch.tensor(boxes_transformed, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_transformed, dtype=torch.int64)

        # Rebuild target
        if boxes_tensor.numel() == 0:
            boxes_tensor = boxes_tensor.reshape(0, 4)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
                boxes_tensor[:, 3] - boxes_tensor[:, 1]
            )

        target_transformed = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": target["image_id"],
            "area": area,
            "iscrowd": target.get(
                "iscrowd", torch.zeros((len(labels_transformed),), dtype=torch.int64)
            ),
        }

        return {
            "image": image_tensor,
            "target": target_transformed,
            "image_path": sample.get("image_path", ""),
        }

    @staticmethod
    def _tensor_to_image_np(image: Tensor) -> Any:
        """Convert float [3, H, W] tensor in [0, 1] to uint8 [H, W, 3] numpy array."""
        image_np = image.permute(1, 2, 0).cpu().numpy()  # [3, H, W] -> [H, W, 3]
        image_np = (image_np * 255).astype("uint8")
        return image_np


def build_training_transforms(
    image_size: tuple[int, int] | list[int] = (640, 1280),
    p_flip: float = 0.5,
    p_affine: float = 0.3,
    p_brightness: float = 0.3,
    p_blur: float = 0.1,
    p_noise: float = 0.05,
) -> DetectionTransforms:
    """Builds training augmentation transforms with moderate augmentations.

    Args:
        image_size: Target (height, width) for resizing.
        p_flip: Probability of horizontal flip.
        p_affine: Probability of affine transforms (scale, translate, rotate, shear).
        p_brightness: Probability of brightness/contrast adjustment.
        p_blur: Probability of blur.
        p_noise: Probability of Gaussian noise.

    Returns:
        A DetectionTransforms object with training augmentations.
    """
    height, width = int(image_size[0]), int(image_size[1])

    transforms = A.Compose(
        [
            # Resize using longest-side strategy + pad
            A.LongestMaxSize(max_size=max(height, width), interpolation=1),
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0),
            # Horizontal flip
            A.HorizontalFlip(p=p_flip),
            # Mild affine transforms
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.1, 0.1),
                rotate=(-15, 15),
                shear=(-10, 10),
                p=p_affine,
                mode="constant",
                cval=0,
            ),
            # Brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=p_brightness
            ),
            # HSV color jitter
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=p_noise),
            # Mild blur
            A.Blur(blur_limit=3, p=p_blur),
            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Convert to tensor
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_visibility=0.1,
            label_fields=["class_labels"],
        ),
    )

    return DetectionTransforms(transforms)


def build_validation_transforms(
    image_size: tuple[int, int] | list[int] = (640, 1280),
) -> DetectionTransforms:
    """Builds validation augmentation transforms (deterministic only).

    Args:
        image_size: Target (height, width) for resizing.

    Returns:
        A DetectionTransforms object with validation transforms.
    """
    height, width = int(image_size[0]), int(image_size[1])

    transforms = A.Compose(
        [
            # Resize using longest-side strategy + pad
            A.LongestMaxSize(max_size=max(height, width), interpolation=1),
            A.PadIfNeeded(min_height=height, min_width=width, border_mode=0, value=0),
            # Normalize
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Convert to tensor
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_visibility=0.1,
            label_fields=["class_labels"],
        ),
    )

    return DetectionTransforms(transforms)
