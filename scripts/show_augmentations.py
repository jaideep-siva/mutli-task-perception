"""
Save individual images for each pipeline step:
  original.jpg, normalised.jpg, affine.jpg,
  brightness_contrast.jpg, hue_saturation.jpg, motion_blur.jpg

Usage:
  python scripts/show_augmentations.py
  python scripts/show_augmentations.py --image data/frame_00095.jpg
  python scripts/show_augmentations.py --image data/frame_00095.jpg --output_dir outputs/aug_demo
"""

import argparse
import os
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Exaggerated parameters so the effect is clear on a slide
AUGMENTATIONS = [
    ("affine",
     A.Affine(scale=(0.80, 1.20), translate_percent=(-0.12, 0.12),
              rotate=(-20, 20), fit_output=False, keep_ratio=True, p=1.0)),

    ("brightness_contrast",
     A.RandomBrightnessContrast(brightness_limit=0.50, contrast_limit=0.50, p=1.0)),

    ("hue_saturation",
     A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=60, val_shift_limit=40, p=1.0)),

    ("motion_blur",
     A.MotionBlur(blur_limit=(15, 25), p=1.0)),
]


def load(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        sys.exit(f"Cannot read: {path}")
    return bgr


def normalised_display(bgr: np.ndarray) -> np.ndarray:
    """ImageNet-normalise then rescale back to uint8 for display."""
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    normed = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    lo, hi = normed.min(), normed.max()
    display = (normed - lo) / (hi - lo + 1e-6)
    return (display[..., ::-1] * 255).clip(0, 255).astype(np.uint8)


def augment(bgr: np.ndarray, tfm: A.BasicTransform) -> np.ndarray:
    rgb = bgr[..., ::-1].copy()
    out = tfm(image=rgb)["image"]
    return out[..., ::-1].copy()


def save(path: Path, img: np.ndarray) -> None:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  {path.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",      type=Path, default=ROOT / "data/frame_00095.jpg")
    ap.add_argument("--output_dir", type=Path, default=ROOT / "outputs/aug_demo")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bgr = load(args.image)
    print(f"Source: {args.image.name}  ({bgr.shape[1]}×{bgr.shape[0]})")

    save(args.output_dir / "original.jpg",   bgr)
    save(args.output_dir / "normalised.jpg", normalised_display(bgr))

    for stem, tfm in AUGMENTATIONS:
        save(args.output_dir / f"{stem}.jpg", augment(bgr, tfm))


if __name__ == "__main__":
    main()
