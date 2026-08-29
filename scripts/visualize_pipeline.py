"""
Generate 4 pipeline visualization panels for the presentation.

  1. original.jpg      — raw frame + GT boxes + GT lane mask
  2. augmentations.jpg — 2×2 grid of albumentations variants (forced p=1)
  3. seg_head.jpg      — seg-logit probability heatmap  |  binary mask overlay
  4. detections.jpg    — model detection boxes, per-class coloured, scored
  5. combined.jpg      — 2×2 poster of all four panels

Usage:
  python scripts/visualize_pipeline.py
  python scripts/visualize_pipeline.py --image data/frame_01089.jpg
  python scripts/visualize_pipeline.py --checkpoint outputs/backbone_sweep/checkpoints/resnet18/best.pt
  python scripts/visualize_pipeline.py --no_model   # skip inference, show GT only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Constants ─────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["barrel", "pedestrian", "stop_sign", "unknown", "tire", "pothole"]

# BGR colours per class
CLASS_BGR = {
    0: (50,  60,  230),   # barrel     — red
    1: (50,  210,  50),   # pedestrian — green
    2: (0,   200, 240),   # stop_sign  — yellow
    3: (180, 180, 180),   # unknown    — silver
    4: (0,   140, 255),   # tire       — orange
    5: (200,  30, 200),   # pothole    — magenta
}

SEG_BGR    = (0, 200, 180)   # teal lane overlay
SEG_ALPHA  = 0.42
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2

# ── Helpers ───────────────────────────────────────────────────────────

def load_image(path: Path, hw: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.resize(bgr, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)


def label_bar(canvas: np.ndarray, text: str,
              color: tuple[int, int, int] = (220, 220, 220)) -> np.ndarray:
    h = 38
    bar = np.full((h, canvas.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(bar, text, (10, 26), FONT, 0.65, color, 1, cv2.LINE_AA)
    return np.concatenate([bar, canvas], axis=0)


def draw_box(canvas: np.ndarray, x1, y1, x2, y2,
             label: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
    ty = max(int(y1) - 4, th + 4)
    cv2.rectangle(canvas, (int(x1), ty - th - 3), (int(x1) + tw + 4, ty + 2), (20, 20, 20), -1)
    cv2.putText(canvas, label, (int(x1) + 2, ty), FONT, FONT_SCALE, color, 1, cv2.LINE_AA)


def overlay_seg(canvas: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = canvas.copy()
    overlay = canvas.copy()
    overlay[mask > 0] = SEG_BGR
    cv2.addWeighted(overlay, SEG_ALPHA, canvas, 1 - SEG_ALPHA, 0, out)
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, SEG_BGR, 2)
    return out


def load_seg_mask(path: str, hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(ROOT / path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(hw, dtype=np.uint8)
    return cv2.resize(mask, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)


# ── Panel 1: Original + GT ────────────────────────────────────────────

def panel_original(bgr: np.ndarray, entry: dict) -> np.ndarray:
    canvas = bgr.copy()
    H, W   = canvas.shape[:2]

    if entry.get("seg_mask"):
        mask = load_seg_mask(entry["seg_mask"], (H, W))
        canvas = overlay_seg(canvas, mask)

    boxes  = entry.get("boxes",  [])
    labels = entry.get("labels", [])
    for box, lbl in zip(boxes, labels):
        color = CLASS_BGR.get(int(lbl), (200, 200, 200))
        name  = CLASS_NAMES[int(lbl)] if int(lbl) < len(CLASS_NAMES) else str(lbl)
        draw_box(canvas, *box, name, color)

    return label_bar(canvas, "1 · Original frame  (GT boxes + GT lane mask)")


# ── Panel 2: Augmentations ────────────────────────────────────────────

def _aug_variant(bgr: np.ndarray, boxes: list, labels: list,
                 mask: np.ndarray, transforms) -> tuple[np.ndarray, list]:
    import albumentations as A
    H, W = bgr.shape[:2]
    result = transforms(
        image=bgr[..., ::-1].copy(),   # A expects RGB
        bboxes=[[b[0], b[1], b[2], b[3]] for b in boxes],
        class_labels=labels,
        mask=mask,
    )
    aug_rgb  = result["image"]
    aug_bgr  = aug_rgb[..., ::-1].copy()
    aug_mask = result["mask"]
    aug_boxes = result["bboxes"]
    aug_labels= result["class_labels"]
    return aug_bgr, aug_mask, aug_boxes, aug_labels


AUGMENTATION_VARIANTS = [
    ("aug_affine",
     "Affine  (scale ±8%, rotate ±4°, translate ±5%)",
     lambda A: A.Compose(
         [A.Affine(scale=(0.92, 1.08), translate_percent=(-0.05, 0.05),
                   rotate=(-4, 4), fit_output=False, keep_ratio=True, p=1.0)],
         bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"],
                                  min_visibility=0.05),
         additional_targets={"mask": "mask"})),

    ("aug_brightness_contrast",
     "Brightness / Contrast  (±25%)",
     lambda A: A.Compose(
         [A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0)],
         bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"],
                                  min_visibility=0.05),
         additional_targets={"mask": "mask"})),

    ("aug_hue_saturation",
     "Hue / Saturation / Value  (hue ±8, sat ±15, val ±15)",
     lambda A: A.Compose(
         [A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15,
                               val_shift_limit=15, p=1.0)],
         bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"],
                                  min_visibility=0.05),
         additional_targets={"mask": "mask"})),

    ("aug_motion_blur",
     "Motion Blur  (kernel 3–5 px)",
     lambda A: A.Compose(
         [A.MotionBlur(blur_limit=(3, 5), p=1.0)],
         bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"],
                                  min_visibility=0.05),
         additional_targets={"mask": "mask"})),
]


def augmentation_panels(bgr: np.ndarray, entry: dict) -> list[tuple[str, np.ndarray]]:
    """Return [(filename_stem, panel_image), ...] — one per augmentation variant."""
    import albumentations as A

    H, W   = bgr.shape[:2]
    boxes  = entry.get("boxes",  [])
    labels = entry.get("labels", [])
    mask   = (load_seg_mask(entry["seg_mask"], (H, W))
              if entry.get("seg_mask") else np.zeros((H, W), dtype=np.uint8))

    results = []
    for stem, title, build_tfm in AUGMENTATION_VARIANTS:
        tfm = build_tfm(A)
        try:
            aug_bgr, aug_mask, aug_boxes, aug_labels = _aug_variant(
                bgr, boxes, labels, mask, tfm)
        except Exception:
            aug_bgr, aug_mask, aug_boxes, aug_labels = bgr.copy(), mask.copy(), boxes, labels

        panel = aug_bgr.copy()
        if aug_mask.any():
            panel = overlay_seg(panel, aug_mask)
        for box, lbl in zip(aug_boxes, aug_labels):
            color = CLASS_BGR.get(int(lbl), (200, 200, 200))
            name  = CLASS_NAMES[int(lbl)] if int(lbl) < len(CLASS_NAMES) else str(lbl)
            draw_box(panel, *box, name, color)

        panel = label_bar(panel, f"Augmentation · {title}")
        results.append((stem, panel))

    return results


# ── Panel 3: Segmentation head output ────────────────────────────────

def panel_seg_head(bgr: np.ndarray, seg_logits: torch.Tensor) -> np.ndarray:
    H, W = bgr.shape[:2]

    # Probability map for lane class (class 1)
    probs = torch.softmax(seg_logits[0], dim=0)[1]   # (H, W) float
    prob_np = probs.cpu().numpy()

    # Colour probability heatmap — JET (blue=0, red=1)
    prob_u8   = (prob_np * 255).clip(0, 255).astype(np.uint8)
    heatmap   = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    heatmap   = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

    # Binary mask overlay
    pred_mask = (prob_np > 0.5).astype(np.uint8)
    seg_panel = overlay_seg(bgr, pred_mask)

    # Coverage stats
    cov = 100.0 * pred_mask.mean()
    cv2.putText(heatmap, "Lane prob  (softmax cls 1)", (8, 26),
                FONT, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(heatmap, "blue=0  red=1", (8, 50),
                FONT, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(seg_panel, f"Binary mask  (prob>0.5)  cov={cov:.1f}%", (8, 26),
                FONT, 0.58, (0, 240, 200), 1, cv2.LINE_AA)

    combined = np.concatenate([heatmap, seg_panel], axis=1)
    return label_bar(combined, "3 · Segmentation head output — raw probability map (left) vs binary mask (right)")


# ── Panel 4: Detection boxes ──────────────────────────────────────────

def panel_detections(bgr: np.ndarray, detections: dict, score_threshold: float) -> np.ndarray:
    canvas = bgr.copy()
    H, W   = canvas.shape[:2]

    boxes  = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    # Draw a score-gradient background strip for context
    n_drawn = 0
    for box, score, lbl in sorted(
            zip(boxes, scores, labels), key=lambda x: x[1]):   # low→high so high on top
        if score < score_threshold:
            continue
        color = CLASS_BGR.get(int(lbl), (200, 200, 200))
        name  = CLASS_NAMES[int(lbl)] if int(lbl) < len(CLASS_NAMES) else str(lbl)
        draw_box(canvas, *box, f"{name} {score:.2f}", color)
        n_drawn += 1

    # Legend strip at bottom
    legend_h = 40
    legend = np.full((legend_h, W, 3), 22, dtype=np.uint8)
    lx = 8
    for idx, cname in enumerate(CLASS_NAMES):
        col = CLASS_BGR[idx]
        cv2.rectangle(legend, (lx, 10), (lx + 14, 28), col, -1)
        cv2.putText(legend, cname, (lx + 18, 26), FONT, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
        lx += 115
    canvas = np.concatenate([canvas, legend], axis=0)

    return label_bar(canvas, f"4 · Detection head output — {n_drawn} boxes above score={score_threshold:.2f}")


# ── Model inference ───────────────────────────────────────────────────

def load_model(checkpoint: Path, config: Path, device: str):
    import yaml
    from models.multitask_model import MultiTaskPerceptionModel
    from models.detection_postprocess import decode_detections

    with open(config) as f:
        cfg = yaml.safe_load(f)

    model = MultiTaskPerceptionModel(cfg)
    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def run_inference(model, bgr: np.ndarray, device: str, score_threshold: float):
    from models.detection_postprocess import decode_detections

    inp = preprocess(bgr).to(device)
    out = model(inp, postprocess=False)

    seg_logits    = out["segmentation"]       # (1, 2, H, W)
    level_outputs = out["detection"]          # {"p3": ..., "p4": ..., "p5": ...}

    dets = decode_detections(
        level_outputs=level_outputs,
        image_size=(bgr.shape[0], bgr.shape[1]),
        score_threshold=score_threshold,
        nms_threshold=0.5,
        max_detections=100,
    )[0]

    return seg_logits.cpu(), {
        "boxes":  dets["boxes"].numpy(),
        "scores": dets["scores"].numpy(),
        "labels": dets["labels"].numpy(),
    }


# ── Main ──────────────────────────────────────────────────────────────

def pick_entry(manifest_path: Path, image_override: Path | None) -> tuple[dict, Path]:
    with open(manifest_path) as f:
        entries = json.load(f)

    if image_override:
        # find matching entry or build a minimal one
        stem = image_override.stem
        for e in entries:
            if Path(e["image"]).stem == stem:
                return e, ROOT / e["image"]
        return {"image": str(image_override), "boxes": [], "labels": []}, image_override

    # Pick entry that has both boxes and a seg_mask
    good = [e for e in entries if e.get("boxes") and e.get("seg_mask")] or entries
    # Take one in the middle (usually a diverse scene)
    entry = good[len(good) // 2]
    return entry, ROOT / entry["image"]


def save(path: Path, img: np.ndarray, quality: int = 93) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"  Saved: {path.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",      type=Path, default=None)
    ap.add_argument("--manifest",   type=Path, default=ROOT / "data/train_manifest.json")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "outputs/backbone_sweep/checkpoints/resnet18/best.pt")
    ap.add_argument("--config",     type=Path,
                    default=ROOT / "configs/multitask/multitask_resnet18.yaml")
    ap.add_argument("--output_dir", type=Path, default=ROOT / "outputs/pipeline_viz")
    ap.add_argument("--score_threshold", type=float, default=0.20)
    ap.add_argument("--no_model",   action="store_true",
                    help="Skip model inference (GT panels only)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    entry, img_path = pick_entry(args.manifest, args.image)
    print(f"Image: {img_path.name}")

    bgr = load_image(img_path, hw=(640, 1280))
    H, W = bgr.shape[:2]

    out = args.output_dir

    # Panel 1 — Original
    print("Building panel 1: original...")
    save(out / "original.jpg", panel_original(bgr, entry))

    # Panels 2a-2d — one file per augmentation
    print("Building augmentation panels (4 separate files)...")
    for stem, aug_panel in augmentation_panels(bgr, entry):
        save(out / f"{stem}.jpg", aug_panel)

    # Panels 3 & 4 need inference
    if not args.no_model and args.checkpoint.exists():
        print(f"Loading model from {args.checkpoint.name}...")
        try:
            model, _ = load_model(args.checkpoint, args.config, device)
            seg_logits, dets = run_inference(model, bgr, device, args.score_threshold)
            print(f"  Detected {(dets['scores'] >= args.score_threshold).sum()} objects")

            print("Building panel: seg head output...")
            seg_logits_resized = torch.nn.functional.interpolate(
                seg_logits, size=(H, W), mode="bilinear", align_corners=False)
            save(out / "seg_head.jpg",   panel_seg_head(bgr, seg_logits_resized))

            print("Building panel: detection boxes...")
            save(out / "detections.jpg", panel_detections(bgr, dets, args.score_threshold))

        except Exception as exc:
            print(f"  [WARN] Inference failed ({exc}) — skipping seg/det panels.")
    else:
        msg = "no_model flag set" if args.no_model else f"checkpoint not found: {args.checkpoint}"
        print(f"  Skipping inference ({msg})")

    print(f"\nDone. All outputs in: {out.relative_to(ROOT)}/")
    print("  original.jpg")
    for stem, *_ in AUGMENTATION_VARIANTS:
        print(f"  {stem}.jpg")
    print("  seg_head.jpg")
    print("  detections.jpg")


if __name__ == "__main__":
    main()
