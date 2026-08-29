"""
Generate a detailed annotated architecture diagram.
Saves: outputs/aug_demo/architecture_detail.jpg

Usage:
  python scripts/visualize_architecture.py
"""

import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]

# ── Colour palette ────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#1e293b"
BORDER = "#334155"
WHITE  = "#e2e8f0"
LIGHT  = "#94a3b8"
DIM    = "#475569"

BLUE   = "#60a5fa"
PURPLE = "#c084fc"
GREEN  = "#4ade80"
ORANGE = "#fb923c"
YELLOW = "#fbbf24"
RED    = "#f87171"
TEAL   = "#2dd4bf"


# ── Drawing helpers ───────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel="", color=BLUE,
        fontsize=12, subfontsize=9, alpha=0.18):
    """Draw a rounded component box."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.15",
                          facecolor=color, alpha=alpha,
                          edgecolor=color, linewidth=2.5,
                          zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.22, label,
            ha="center", va="top", color=color,
            fontsize=fontsize, fontweight="bold", zorder=3)
    if sublabel:
        ax.text(x + w/2, y + h - 0.52, sublabel,
                ha="center", va="top", color=LIGHT,
                fontsize=subfontsize, zorder=3)


def tensor_chip(ax, x, y, shape_str, desc="", color=YELLOW, fontsize=9):
    """Inline tensor shape tag."""
    w = len(shape_str) * 0.095 + 0.3
    rect = FancyBboxPatch((x - w/2, y - 0.16), w, 0.32,
                          boxstyle="round,pad=0.05",
                          facecolor=color, alpha=0.18,
                          edgecolor=color, linewidth=1.5, zorder=4)
    ax.add_patch(rect)
    ax.text(x, y, shape_str, ha="center", va="center",
            color=color, fontsize=fontsize,
            fontfamily="monospace", fontweight="bold", zorder=5)
    if desc:
        ax.text(x, y - 0.28, desc,
                ha="center", va="top", color=LIGHT,
                fontsize=fontsize - 1.5, zorder=5)


def arrow(ax, x1, y1, x2, y2, color=DIM):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.0, mutation_scale=16),
                zorder=3)


def hline(ax, x1, x2, y, color=DIM):
    ax.plot([x1, x2], [y, y], color=color, lw=1.5, zorder=3)


def vline(ax, x, y1, y2, color=DIM):
    ax.plot([x, x], [y1, y2], color=color, lw=1.5, zorder=3)


def note(ax, x, y, text, color=LIGHT, fontsize=8.5, ha="left"):
    ax.text(x, y, text, ha=ha, va="top", color=color,
            fontsize=fontsize, linespacing=1.55, zorder=5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL,
                      edgecolor=BORDER, linewidth=1, alpha=0.9))


def feat_row(ax, cx, y, shapes, colors, labels):
    """Draw a row of tensor chips with connecting lines."""
    n = len(shapes)
    spacing = 3.0
    xs = [cx + (i - (n-1)/2) * spacing for i in range(n)]
    for x, shape, color, lbl in zip(xs, shapes, colors, labels):
        tensor_chip(ax, x, y + 0.18, shape, lbl, color=color)
    return xs


# ── Main figure ───────────────────────────────────────────────────────

def build(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 30), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 30)
    ax.axis("off")

    CX = 9.0  # horizontal centre

    # ── TITLE ────────────────────────────────────────────────────────
    ax.text(CX, 29.5, "Multi-Task Perception — Architecture Detail",
            ha="center", va="top", color=WHITE, fontsize=18, fontweight="bold")
    ax.text(CX, 29.0, "ResNet-18 backbone  ·  FPN  ·  FCOS detection head  ·  Binary segmentation head",
            ha="center", va="top", color=LIGHT, fontsize=10)

    # ═══════════════════════════════════════════════════════════════
    # 1. INPUT
    # ═══════════════════════════════════════════════════════════════
    BX, BY, BW, BH = CX - 3.5, 27.0, 7.0, 1.3
    box(ax, BX, BY, BW, BH, "INPUT", "Equirectangular 360° camera frame", color=LIGHT)
    tensor_chip(ax, CX, BY + 0.42, "( 1,  3,  640, 1280 )",
                "batch · RGB · H · W", color=WHITE)

    note(ax, BX - 4.2, BY + BH,
         "Input is a single equirectangular frame.\n"
         "640 × 1280 px — wider than tall because 360° is\n"
         "unwrapped horizontally. Batch size is fixed at 1\n"
         "for robot deployment (no batching at inference).",
         color=LIGHT)

    arrow(ax, CX, BY, CX, BY - 0.25)

    # ═══════════════════════════════════════════════════════════════
    # 2. BACKBONE — ResNet-18
    # ═══════════════════════════════════════════════════════════════
    BBY = 24.2
    box(ax, CX - 4.5, BBY, 9.0, 2.5, "BACKBONE — ResNet-18",
        "11M params  ·  ImageNet pretrained  ·  extracts 3 feature maps", color=BLUE)

    feat_xs = feat_row(
        ax, CX, BBY + 0.55,
        shapes=["(1, 128, 80, 160)", "(1, 256, 40,  80)", "(1, 512, 20,  40)"],
        colors=[TEAL, TEAL, TEAL],
        labels=["C3  stride 8", "C4  stride 16", "C5  stride 32"],
    )

    note(ax, 0.3, BBY + 2.5,
         "WHY THREE SCALES?\n"
         "A single feature map can't find both small and\n"
         "large objects well. Deeper layers (C5) have large\n"
         "receptive fields — good for big objects / context.\n"
         "Shallow layers (C3) preserve spatial detail —\n"
         "essential for small or thin objects (potholes, lanes).",
         color=LIGHT)

    note(ax, 12.0, BBY + 2.5,
         "RESNET-18 CHANNEL PROGRESSION\n"
         "layer1 → 64 ch    (not exported)\n"
         "layer2 → 128 ch   ← C3\n"
         "layer3 → 256 ch   ← C4\n"
         "layer4 → 512 ch   ← C5\n"
         "Stride doubles each layer (8 → 16 → 32).",
         color=LIGHT)

    arrow(ax, CX, BBY, CX, BBY - 0.25)

    # ═══════════════════════════════════════════════════════════════
    # 3. FPN — Feature Pyramid Network
    # ═══════════════════════════════════════════════════════════════
    FY = 21.3
    box(ax, CX - 4.5, FY, 9.0, 2.5, "FPN — Feature Pyramid Network",
        "top-down fusion  ·  lateral 1×1 convs  ·  all levels → 256 channels", color=PURPLE)

    # Show fusion formula inside the box
    ax.text(CX, FY + 1.82,
            "P5 = conv(C5)          P4 = conv(C4) + upsample×2(P5)          P3 = conv(C3) + upsample×2(P4)",
            ha="center", va="top", color=PURPLE, fontsize=8.5,
            fontfamily="monospace")

    fpn_xs = feat_row(
        ax, CX, FY + 0.55,
        shapes=["(1, 256, 80, 160)", "(1, 256, 40,  80)", "(1, 256, 20,  40)"],
        colors=[GREEN, ORANGE, RED],
        labels=["P3  stride 8", "P4  stride 16", "P5  stride 32"],
    )

    note(ax, 0.3, FY + 2.5,
         "WHY FPN?\n"
         "C3/C4/C5 are one-directional — shallow layers\n"
         "never see high-level context. FPN adds top-down\n"
         "paths so every P level sees both fine texture\n"
         "(from the corresponding C) AND high-level scene\n"
         "understanding (propagated down from C5).\n"
         "All P levels are projected to 256 channels so\n"
         "the detection head can share weights across scales.",
         color=LIGHT)

    note(ax, 12.0, FY + 2.5,
         "SPATIAL SIZES\n"
         "P3: 80×160  = 12,800 cells  (small objects)\n"
         "P4: 40×80   =  3,200 cells  (medium objects)\n"
         "P5: 20×40   =    800 cells  (large objects)\n"
         "Total: 16,800 candidate positions fed\n"
         "to the detection head before filtering.",
         color=LIGHT)

    # Fork arrow down then split
    FORK_Y = FY - 0.3
    arrow(ax, CX, FY, CX, FORK_Y)
    hline(ax, CX - 3.5, CX + 3.5, FORK_Y)
    vline(ax, CX - 3.5, FORK_Y, FORK_Y - 0.4)
    vline(ax, CX + 3.5, FORK_Y, FORK_Y - 0.4)
    arrow(ax, CX - 3.5, FORK_Y - 0.4, CX - 3.5, FORK_Y - 0.65)
    arrow(ax, CX + 3.5, FORK_Y - 0.4, CX + 3.5, FORK_Y - 0.65)
    ax.text(CX - 3.5, FORK_Y + 0.12, "P3  P4  P5", ha="center",
            color=GREEN, fontsize=8, fontweight="bold")
    ax.text(CX + 3.5, FORK_Y + 0.12, "P3 + P4 + P5", ha="center",
            color=ORANGE, fontsize=8, fontweight="bold")

    # ═══════════════════════════════════════════════════════════════
    # 4a. DETECTION HEAD (left side)
    # ═══════════════════════════════════════════════════════════════
    DX, DY, DW, DH = 0.4, 16.8, 7.8, 3.6
    box(ax, DX, DY, DW, DH, "DETECTION HEAD", "FCOS — anchor-free", color=GREEN)

    # Per-level outputs
    levels = [
        ("P3 → (1, 11, 80, 160)", "12,800 cells", GREEN,  DX + 1.3, DY + 2.85),
        ("P4 → (1, 11, 40,  80)", " 3,200 cells", ORANGE, DX + 3.9, DY + 2.85),
        ("P5 → (1, 11, 20,  40)", "   800 cells", RED,    DX + 6.5, DY + 2.85),
    ]
    for shape, cells, col, tx, ty in levels:
        tensor_chip(ax, tx, ty, shape, cells, color=col, fontsize=8)

    # 11 = 4 + 1 + 6 breakdown
    ax.text(DX + DW/2, DY + 2.28,
            "11 output channels per cell:   4 LTRB offsets   +   1 objectness   +   6 class logits",
            ha="center", va="top", color=WHITE, fontsize=8.5)

    # LTRB diagram
    ltrb_x, ltrb_y = DX + 1.0, DY + 1.62
    ax.text(ltrb_x, ltrb_y,
            "LTRB = distances from cell centre\n"
            "to left / top / right / bottom\n"
            "edges of the ground-truth box",
            ha="left", va="top", color=TEAL, fontsize=8, linespacing=1.5)

    ax.text(DX + 4.2, DY + 1.62,
            "objectness = sigmoid score\n"
            "\"is there an object centred\n"
            "at this cell?\"  threshold=0.20",
            ha="left", va="top", color=YELLOW, fontsize=8, linespacing=1.5)

    note(ax, 0.3, DY + 3.6,
         "WHY ANCHOR-FREE (FCOS)?\n"
         "Traditional detectors pre-define hundreds of\n"
         "anchor boxes at each cell. Anchors need careful\n"
         "tuning for each dataset. FCOS predicts L/T/R/B\n"
         "offsets directly — no anchor hyperparameters,\n"
         "simpler training, fewer false positives near image\n"
         "borders (important for equirectangular frames).",
         color=LIGHT)

    # ═══════════════════════════════════════════════════════════════
    # 4b. SEGMENTATION HEAD (right side)
    # ═══════════════════════════════════════════════════════════════
    SX, SY, SW, SH = 9.8, 16.8, 7.8, 3.6
    box(ax, SX, SY, SW, SH, "SEGMENTATION HEAD", "binary lane mask", color=ORANGE)

    # Steps inside the seg head
    steps = [
        (SX + SW/2, SY + 3.1,  "(1, 256, 80, 160)  ← P3 direct",        TEAL),
        (SX + SW/2, SY + 2.65, "(1, 256, 80, 160)  ← P4 bilinear ×2",   ORANGE),
        (SX + SW/2, SY + 2.20, "(1, 256, 80, 160)  ← P5 bilinear ×4",   RED),
        (SX + SW/2, SY + 1.75, "concat → (1, 768, 80, 160)",             PURPLE),
        (SX + SW/2, SY + 1.30, "1×1 conv → (1, 2, 80, 160)",            PURPLE),
        (SX + SW/2, SY + 0.82, "bilinear upsample → (1, 2, 640, 1280)", ORANGE),
    ]
    for tx, ty, txt, col in steps:
        ax.text(tx, ty, txt, ha="center", va="top",
                color=col, fontsize=8.5, fontfamily="monospace")

    # Brace lines
    for ty in [SY + 3.1, SY + 2.65, SY + 2.20]:
        ax.plot([SX + 1.0, SX + 1.0 + 0.3], [ty - 0.07, ty - 0.07],
                color=PURPLE, lw=1.2, alpha=0.6)

    note(ax, 12.2, SY + 3.6,
         "WHY FUSE ALL THREE FPN LEVELS?\n"
         "P3 alone gives fine spatial detail but\n"
         "misses long-range context (where the lane\n"
         "goes). P5 alone gives context but is 8×\n"
         "too coarse for pixel-accurate masks.\n"
         "Fusing all three gives the head both\n"
         "sharp edges (P3) and scene understanding\n"
         "(P5) at the same time.",
         color=LIGHT)

    # Arrows into seg head
    arrow(ax, DX + DW/2, DY, DX + DW/2, DY - 0.3)
    arrow(ax, SX + SW/2, SY, SX + SW/2, SY - 0.3)

    # ═══════════════════════════════════════════════════════════════
    # 5a. NMS + DECODE (below det head)
    # ═══════════════════════════════════════════════════════════════
    NX, NY, NW, NH = 0.4, 13.5, 7.8, 2.9
    box(ax, NX, NY, NW, NH, "DECODE + NMS",
        "16,800 raw predictions → top-100 boxes", color=TEAL)

    steps_nms = [
        "① Sigmoid objectness  →  keep cells where score > 0.20",
        "② Decode LTRB from each surviving cell's centre (x,y) to (x1,y1,x2,y2)",
        "③ Final score  =  objectness  ×  softmax class confidence",
        "④ Concatenate predictions from P3 + P4 + P5",
        "⑤ Non-Maximum Suppression  IoU threshold = 0.50  →  keep top 100",
    ]
    for i, s in enumerate(steps_nms):
        col = [TEAL, YELLOW, GREEN, PURPLE, ORANGE][i]
        ax.text(NX + 0.25, NY + NH - 0.55 - i * 0.45, s,
                ha="left", va="top", color=col, fontsize=8.5)

    note(ax, 0.3, NY + NH,
         "WHY NMS?\n"
         "Multiple overlapping cells (and all three\n"
         "FPN scales) can fire for the same object.\n"
         "NMS keeps only the highest-scoring box\n"
         "for each object cluster and discards the\n"
         "rest (IoU>0.50 = too much overlap = same obj).",
         color=LIGHT)

    # ═══════════════════════════════════════════════════════════════
    # 5b. FINAL SEG OUTPUT (below seg head)
    # ═══════════════════════════════════════════════════════════════
    OX, OY, OW, OH = 9.8, 13.5, 7.8, 2.9
    box(ax, OX, OY, OW, OH, "SEGMENTATION OUTPUT",
        "binary lane mask at full resolution", color=ORANGE)

    seg_steps = [
        ("(1, 2, 640, 1280)  logits",
         "2 channels: background (0) and lane (1)",  ORANGE, OY + OH - 0.55),
        ("softmax  →  probability map",
         "P(lane) at every pixel, values 0.0 – 1.0",  TEAL,   OY + OH - 1.10),
        ("argmax(dim=1)  →  class map (640 × 1280)",
         "0 = background     1 = lane",               GREEN,  OY + OH - 1.65),
        ("ROI mask applied  →  zero-out sky pixels",
         "suppress false positives in upper 35% of frame", YELLOW, OY + OH - 2.20),
    ]
    for shape, desc, col, ty in seg_steps:
        ax.text(OX + OW/2, ty, shape, ha="center", va="top",
                color=col, fontsize=9, fontfamily="monospace")
        ax.text(OX + OW/2, ty - 0.30, desc, ha="center", va="top",
                color=LIGHT, fontsize=8)

    # ═══════════════════════════════════════════════════════════════
    # 6. FINAL OUTPUT BOXES
    # ═══════════════════════════════════════════════════════════════
    arrow(ax, NX + NW/2, NY, NX + NW/2, NY - 0.35)
    arrow(ax, OX + OW/2, OY, OX + OW/2, OY - 0.35)

    FY2 = 10.0
    # Det output
    box(ax, 0.4, FY2, 7.8, 1.6,
        "DETECTION OUTPUT",
        "up to 100 boxes  ·  (x1,y1,x2,y2)  ·  class  ·  score", color=GREEN)
    tensor_chip(ax, NX + NW/2, FY2 + 0.55, "N × [ x1, y1, x2, y2, score, class ]",
                "N ≤ 100, pixel coords, score ∈ [0,1]", color=GREEN, fontsize=8.5)

    # Seg output
    box(ax, 9.8, FY2, 7.8, 1.6,
        "LANE MASK OUTPUT",
        "full-resolution binary mask  ·  0=bg  ·  1=lane", color=ORANGE)
    tensor_chip(ax, OX + OW/2, FY2 + 0.55, "( 640, 1280 )  uint8",
                "0 = background     1 = lane pixel", color=ORANGE, fontsize=8.5)

    # ═══════════════════════════════════════════════════════════════
    # 7. LOSS / TRAINING ONLY ANNOTATION (bottom band)
    # ═══════════════════════════════════════════════════════════════
    ax.axhline(8.9, color=BORDER, lw=1, linestyle="--", alpha=0.5)
    ax.text(CX, 8.7,
            "TRAINING ONLY   (not present at inference)",
            ha="center", va="top", color=DIM, fontsize=9, fontstyle="italic")

    loss_info = [
        ("Detection Loss", GREEN,
         "Focal loss (objectness)  +  L1 (LTRB regression)  +  CrossEntropy (class)"),
        ("Segmentation Loss", ORANGE,
         "CrossEntropy (or Dice)  ·  ROI-masked  ·  scaled by pseudo-label confidence"),
        ("Total Loss", YELLOW,
         "L_total  =  w_det × L_det  +  w_seg × L_seg         w_det = w_seg = 1.0"),
    ]
    for i, (title, col, desc) in enumerate(loss_info):
        ty = 8.1 - i * 0.75
        ax.text(CX - 8.0, ty, f"{title}:", ha="left", va="top",
                color=col, fontsize=9, fontweight="bold")
        ax.text(CX - 5.4, ty, desc, ha="left", va="top",
                color=LIGHT, fontsize=9)

    # ═══════════════════════════════════════════════════════════════
    # 8. CLASS LEGEND (bottom)
    # ═══════════════════════════════════════════════════════════════
    ax.axhline(5.8, color=BORDER, lw=1, alpha=0.4)
    ax.text(CX, 5.65, "6 DETECTION CLASSES", ha="center", va="top",
            color=DIM, fontsize=9, fontstyle="italic")
    classes = [
        ("0  barrel",      "#e05555"),
        ("1  pedestrian",  "#55e055"),
        ("2  stop_sign",   "#e0d055"),
        ("3  unknown",     "#aaaaaa"),
        ("4  tire",        "#e08855"),
        ("5  pothole",     "#cc55cc"),
    ]
    cx_cls = 1.5
    for name, col in classes:
        ax.text(cx_cls, 5.15, name, ha="left", va="top",
                color=col, fontsize=9, fontfamily="monospace")
        cx_cls += 2.8

    # ── Save ─────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    build(ROOT / "outputs/aug_demo/architecture_detail.jpg")
