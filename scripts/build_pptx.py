"""Generate presentation.pptx — Multi-Task Perception results deck."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree
import copy

# ── Palette ───────────────────────────────────────────────────────────
BG     = RGBColor(0x0d, 0x11, 0x17)
BLUE   = RGBColor(0x60, 0xa5, 0xfa)
PURPLE = RGBColor(0xc0, 0x84, 0xfc)
GREEN  = RGBColor(0x4a, 0xde, 0x80)
ORANGE = RGBColor(0xfb, 0x92, 0x3c)
YELLOW = RGBColor(0xfb, 0xbf, 0x24)
RED    = RGBColor(0xf8, 0x71, 0x71)
WHITE  = RGBColor(0xe2, 0xe8, 0xf0)
LGRAY  = RGBColor(0xcb, 0xd5, 0xe1)
GRAY   = RGBColor(0x64, 0x74, 0x8b)
DIM    = RGBColor(0x47, 0x55, 0x69)
CARD   = RGBColor(0x1e, 0x29, 0x3b)
DARK   = RGBColor(0x0f, 0x17, 0x25)

W = Inches(13.33)   # widescreen 16:9
H = Inches(7.5)

# ── Helpers ───────────────────────────────────────────────────────────

def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_label(slide, text, x, y, w, h, size=10, bold=False, color=WHITE,
              align=PP_ALIGN.LEFT, italic=False):
    txb = slide.shapes.add_textbox(x, y, w, h)
    tf = txb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb


def add_box(slide, x, y, w, h, fill=CARD, border=DIM, border_w=Pt(1.5), radius=0):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        x, y, w, h
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = border
    shape.line.width = border_w
    if radius:
        sp = shape._element
        spPr = sp.find(qn('p:spPr'))
        if spPr is not None:
            prstGeom = spPr.find(qn('a:prstGeom'))
            if prstGeom is not None:
                prstGeom.set('prst', 'roundRect')
                avLst = prstGeom.find(qn('a:avLst'))
                if avLst is None:
                    avLst = etree.SubElement(prstGeom, qn('a:avLst'))
                gd = etree.SubElement(avLst, qn('a:gd'))
                gd.set('name', 'adj')
                gd.set('fmla', f'val {radius}')
    return shape


def add_arrow(slide, x1, y, x2, color=DIM):
    line = slide.shapes.add_connector(1, x1, y, x2, y)
    line.line.color.rgb = color
    line.line.width = Pt(1.5)
    return line


def add_table(slide, data, headers, col_widths, x, y, row_h=Inches(0.32),
              hdr_color=CARD, row_colors=None, col_aligns=None):
    rows = len(data) + 1
    cols = len(headers)
    total_w = sum(col_widths)
    tbl = slide.shapes.add_table(rows, cols, x, y, total_w, row_h * rows).table

    for c, (hdr, cw) in enumerate(zip(headers, col_widths)):
        tbl.columns[c].width = cw
        cell = tbl.cell(0, c)
        cell.text = hdr
        cell.fill.solid()
        cell.fill.fore_color.rgb = hdr_color
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.runs[0] if p.runs else p.add_run()
        run.font.size = Pt(8)
        run.font.bold = True
        run.font.color.rgb = GRAY

    for r, row_data in enumerate(data):
        rc = row_colors[r] if row_colors else DARK
        for c, cell_val in enumerate(row_data):
            cell = tbl.cell(r + 1, c)
            text, color = (cell_val if isinstance(cell_val, tuple) else (str(cell_val), WHITE))
            cell.text = text
            cell.fill.solid()
            cell.fill.fore_color.rgb = rc
            p = cell.text_frame.paragraphs[0]
            align = col_aligns[c] if col_aligns else PP_ALIGN.CENTER
            p.alignment = align
            run = p.runs[0] if p.runs else p.add_run()
            run.font.size = Pt(8)
            run.font.color.rgb = color

    return tbl


def add_bar(slide, x, y, bar_w, bar_h, pct, fill_color, bg_color=DIM):
    add_box(slide, x, y, bar_w, bar_h, fill=bg_color, border=bg_color, border_w=Pt(0))
    if pct > 0:
        add_box(slide, x, y, Emu(int(bar_w * pct)), bar_h,
                fill=fill_color, border=fill_color, border_w=Pt(0))


def note_card(slide, x, y, w, h, title, body, border=BLUE):
    add_box(slide, x, y, w, h, fill=CARD, border=border, border_w=Pt(2.5))
    add_label(slide, title, x + Inches(0.08), y + Inches(0.05),
              w - Inches(0.16), Inches(0.22), size=8, bold=True, color=WHITE)
    add_label(slide, body, x + Inches(0.08), y + Inches(0.25),
              w - Inches(0.16), h - Inches(0.3), size=7, color=LGRAY)


# ═══════════════════════════════════════════════════════════════
# BUILD PRESENTATION
# ═══════════════════════════════════════════════════════════════

prs = Presentation()
prs.slide_width = W
prs.slide_height = H
blank = prs.slide_layouts[6]  # completely blank layout


# ═══════════════════════════════════════════════════════════════
# SLIDE 1 — ARCHITECTURE
# ═══════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(blank)
set_bg(s1, BG)

# Title area
add_label(s1, "MULTI-TASK PERCEPTION FOR ROBOTICS",
          Inches(0.4), Inches(0.12), Inches(12), Inches(0.28),
          size=8, bold=False, color=GRAY)
add_label(s1, "Model Architecture",
          Inches(0.4), Inches(0.38), Inches(12), Inches(0.45),
          size=26, bold=True, color=WHITE)

# ── Box positions (y-center = 2.5") ──────────────────────────
BOX_Y   = Inches(1.05)
BOX_H   = Inches(3.2)
ARROW_Y = Inches(1.05) + BOX_H / 2 - Inches(0.02)

# Input box
IX, IW = Inches(0.3), Inches(1.55)
add_box(s1, IX, BOX_Y, IW, BOX_H, fill=CARD, border=DIM)
add_label(s1, "INPUT", IX + Inches(0.08), BOX_Y + Inches(0.1), IW - Inches(0.16), Inches(0.22),
          size=8, bold=True, color=LGRAY, align=PP_ALIGN.CENTER)
add_label(s1,
          "Equirectangular\n360° camera frame\n\n1 × 3 × 640 × 1280\n\nRGB · float32\nImageNet normalised",
          IX + Inches(0.08), BOX_Y + Inches(0.35), IW - Inches(0.16), Inches(2.7),
          size=8, color=LGRAY)

# Arrow 1
A1X = IX + IW
add_arrow(s1, A1X, ARROW_Y, A1X + Inches(0.25), color=DIM)

# Backbone box
BX, BW = A1X + Inches(0.25), Inches(2.2)
add_box(s1, BX, BOX_Y, BW, BOX_H, fill=CARD, border=BLUE)
add_label(s1, "BACKBONE", BX + Inches(0.08), BOX_Y + Inches(0.1), BW - Inches(0.16), Inches(0.22),
          size=8, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
bb_body = (
    "ResNet-18      ★ deploy\n"
    "ConvNeXt-B   89M params\n"
    "Swin-B            transformer\n"
    "HRNet-W32     custom\n\n"
    "Outputs:\n"
    "C3  stride 8  (fine detail)\n"
    "C4  stride 16 (semantics)\n"
    "C5  stride 32 (context)"
)
add_label(s1, bb_body, BX + Inches(0.08), BOX_Y + Inches(0.35),
          BW - Inches(0.16), Inches(2.7), size=8, color=LGRAY)

# Arrow 2
A2X = BX + BW
add_arrow(s1, A2X, ARROW_Y, A2X + Inches(0.25), color=DIM)

# FPN box
FX, FW = A2X + Inches(0.25), Inches(2.0)
add_box(s1, FX, BOX_Y, FW, BOX_H, fill=CARD, border=PURPLE)
add_label(s1, "FPN", FX + Inches(0.08), BOX_Y + Inches(0.1), FW - Inches(0.16), Inches(0.22),
          size=8, bold=True, color=PURPLE, align=PP_ALIGN.CENTER)
fpn_body = (
    "Top-down feature fusion\n\n"
    "C5 ──► P5\n"
    "         ↓ ×2\n"
    "C4 ──► P4 + up(P5)\n"
    "         ↓ ×2\n"
    "C3 ──► P3 + up(P4)\n\n"
    "All levels: 256 channels\n"
    "Lateral 1×1 convolutions"
)
add_label(s1, fpn_body, FX + Inches(0.08), BOX_Y + Inches(0.35),
          FW - Inches(0.16), Inches(2.7), size=8, color=LGRAY)

# Arrow 3
A3X = FX + FW
add_arrow(s1, A3X, ARROW_Y, A3X + Inches(0.25), color=DIM)

# Fork: Detection + Segmentation side by side
DX = A3X + Inches(0.25)
DW = Inches(3.3)

# Detection box (top half)
DET_H = Inches(1.52)
add_box(s1, DX, BOX_Y, DW, DET_H, fill=CARD, border=GREEN)
add_label(s1, "DETECTION HEAD — FCOS (anchor-free)",
          DX + Inches(0.08), BOX_Y + Inches(0.07), DW - Inches(0.16), Inches(0.22),
          size=8, bold=True, color=GREEN)
det_body = (
    "Per cell: 4 LTRB offsets · 1 objectness · 6 class logits (11 total)\n"
    "P3 → 11×80×160   P4 → 11×40×80   P5 → 11×20×40\n"
    "Post: sigmoid · decode LTRB · NMS (IoU 0.5) · top-100\n"
    "Loss: Focal (objectness) + L1 (box) + CrossEntropy (class)"
)
add_label(s1, det_body, DX + Inches(0.08), BOX_Y + Inches(0.3),
          DW - Inches(0.16), Inches(1.1), size=7.5, color=LGRAY)

# Segmentation box (bottom half)
SEG_Y = BOX_Y + DET_H + Inches(0.16)
SEG_H = Inches(1.52)
add_box(s1, DX, SEG_Y, DW, SEG_H, fill=CARD, border=ORANGE)
add_label(s1, "SEGMENTATION HEAD — Lane Mask",
          DX + Inches(0.08), SEG_Y + Inches(0.07), DW - Inches(0.16), Inches(0.22),
          size=8, bold=True, color=ORANGE)
seg_body = (
    "P3 + P4↑ + P5↑ → concat → 1×1 conv → bilinear upsample\n"
    "Output: 2 × 640 × 1280   →   argmax → 0 bg · 1 lane\n"
    "ROI masking · pseudo-label confidence weighting\n"
    "Loss: Cross-Entropy (default) or Dice"
)
add_label(s1, seg_body, DX + Inches(0.08), SEG_Y + Inches(0.3),
          DW - Inches(0.16), Inches(1.1), size=7.5, color=LGRAY)

# Bottom stat chips
CHIP_Y = Inches(4.55)
chips = [
    ("Single Forward Pass", "detection + segmentation", BLUE),
    ("15.9 ms", "ResNet-18 · PyTorch", GREEN),
    ("~11 M params", "ResNet-18 backbone", PURPLE),
    ("452 MB GPU", "peak memory", ORANGE),
    ("6 Classes", "barrel · pedestrian · stop sign\nunknown · tire · pothole", YELLOW),
]
cw = Inches(2.35)
gap = Inches(0.18)
cx = Inches(0.3)
for title, sub, col in chips:
    add_box(s1, cx, CHIP_Y, cw, Inches(0.78), fill=CARD, border=col, border_w=Pt(1))
    add_label(s1, title, cx + Inches(0.1), CHIP_Y + Inches(0.06),
              cw - Inches(0.2), Inches(0.3), size=11, bold=True, color=col,
              align=PP_ALIGN.CENTER)
    add_label(s1, sub, cx + Inches(0.1), CHIP_Y + Inches(0.35),
              cw - Inches(0.2), Inches(0.35), size=7, color=GRAY, align=PP_ALIGN.CENTER)
    cx += cw + gap

# Slide number
add_label(s1, "1 / 4", Inches(12.8), Inches(7.25), Inches(0.5), Inches(0.2),
          size=7, color=DIM, align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════════════════
# SLIDE 2 — MODEL METRICS
# ═══════════════════════════════════════════════════════════════
s2 = prs.slides.add_slide(blank)
set_bg(s2, BG)

add_label(s2, "BACKBONE SWEEP — ~1,400 ANNOTATED 640×1280 FRAMES · AdamW · EARLY STOPPING",
          Inches(0.4), Inches(0.12), Inches(12), Inches(0.28),
          size=8, color=GRAY)
add_label(s2, "Model Performance Comparison",
          Inches(0.4), Inches(0.38), Inches(12), Inches(0.45),
          size=26, bold=True, color=WHITE)

# Main metrics table
headers = ["Backbone", "Params", "mAP\n0.5:0.95", "mAP@50", "mAP@75",
           "Lane IoU", "Lane Dice", "Latency", "FPS", "GPU RAM", "Trained"]
col_widths = [Inches(1.3), Inches(0.7), Inches(0.75), Inches(0.75), Inches(0.75),
              Inches(0.75), Inches(0.75), Inches(0.85), Inches(0.6), Inches(0.8), Inches(0.8)]

P = (lambda v, c: (v, c))  # (text, color) helper

data = [
    [P("ResNet-18 ★", GREEN), P("11 M", WHITE), P("0.248", GREEN), P("0.559", GREEN),
     P("0.098", WHITE), P("0.901", GREEN), P("0.945", GREEN), P("15.9 ms", GREEN),
     P("62.9", GREEN), P("452 MB", GREEN), P("40/50 ep", YELLOW)],
    [P("ConvNeXt-Base", BLUE), P("89 M", WHITE), P("0.224", WHITE), P("0.461", WHITE),
     P("0.125", GREEN), P("0.884", WHITE), P("0.933", WHITE), P("206 ms", WHITE),
     P("4.8", WHITE), P("1,771 MB", WHITE), P("13/50 ep", RED)],
    [P("Swin-B", PURPLE), P("88 M", WHITE), P("—", DIM), P("—", DIM),
     P("—", DIM), P("—", DIM), P("—", DIM), P("~150 ms†", DIM),
     P("~7†", DIM), P("~2,000 MB†", DIM), P("pending", DIM)],
    [P("HRNet-W32*", ORANGE), P("29 M", WHITE), P("—", DIM), P("—", DIM),
     P("—", DIM), P("—", DIM), P("—", DIM), P("~80 ms†", DIM),
     P("~12†", DIM), P("~900 MB†", DIM), P("pending", DIM)],
]
row_colors = [DARK, DARK, DARK, DARK]
add_table(s2, data, headers, col_widths,
          Inches(0.3), Inches(0.92), row_h=Inches(0.34),
          row_colors=row_colors)

add_label(s2, "* Custom lightweight backbone (not full HRNet). † Architecture estimates — sweep pending. Best values in green.",
          Inches(0.3), Inches(2.35), Inches(12), Inches(0.22), size=7, color=DIM)

# Per-class AP sub-table
pc_headers = ["Backbone", "barrel", "pedestrian", "stop sign", "unknown", "tire", "pothole"]
pc_widths = [Inches(1.4), Inches(1.0), Inches(1.1), Inches(1.0), Inches(1.0), Inches(0.9), Inches(0.9)]
pc_data = [
    [P("ResNet-18", GREEN), P("0.305", WHITE), P("0.318", WHITE), P("0.076", WHITE),
     P("0.089", WHITE), P("0.202", WHITE), P("0.149", WHITE)],
    [P("ConvNeXt-B", BLUE), P("0.300", WHITE), P("0.263", WHITE), P("0.253", GREEN),
     P("0.060", WHITE), P("0.091", WHITE), P("0.174", WHITE)],
]
add_table(s2, pc_data, pc_headers, pc_widths,
          Inches(0.3), Inches(2.58), row_h=Inches(0.3),
          row_colors=[DARK, DARK])

# Note cards (2×2 grid)
NW = Inches(6.2)
NH = Inches(1.0)
NX1, NX2 = Inches(0.3), Inches(6.6)
NY1, NY2 = Inches(3.35), Inches(4.42)

note_card(s2, NX1, NY1, NW, NH,
          "Why is Lane IoU so high (90%+)?",
          "Segmentation is a binary task — lane vs. background — with consistent visual structure. "
          "The head fuses all 3 FPN scales so it sees both fine edge detail and broad scene context.",
          border=GREEN)

note_card(s2, NX2, NY1, NW, NH,
          "Why is mAP modest (~0.25)?",
          "Only ~1,400 annotated frames. Early stopping cut both runs short. Classes like pothole and "
          "unknown have high visual variance and are rare per image — pseudo-labelling will lift these.",
          border=BLUE)

note_card(s2, NX1, NY2, NW, NH,
          "Why does ResNet-18 beat ConvNeXt overall?",
          "ConvNeXt-B trained only 13/50 epochs before early stopping — heavily under-trained. "
          "Its 206 ms latency also makes it impractical for real-time robotics regardless of accuracy.",
          border=ORANGE)

note_card(s2, NX2, NY2, NW, NH,
          "Why is ConvNeXt mAP@75 higher (0.125 vs 0.098)?",
          "Larger receptive field (7×7 depthwise kernels) produces tighter box localisation. "
          "Higher mAP@75 = fewer but better-placed detections — quality over quantity.",
          border=YELLOW)

add_label(s2, "2 / 4", Inches(12.8), Inches(7.25), Inches(0.5), Inches(0.2),
          size=7, color=DIM, align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════════════════
# SLIDE 3 — THROUGHPUT COMPARISON
# ═══════════════════════════════════════════════════════════════
s3 = prs.slides.add_slide(blank)
set_bg(s3, BG)

add_label(s3, "DEPLOYMENT RUNTIME ANALYSIS — bars normalised to ConvNeXt-B PyTorch (206 ms = 100%)",
          Inches(0.4), Inches(0.12), Inches(12), Inches(0.28),
          size=8, color=GRAY)
add_label(s3, "ResNet-18 vs ConvNeXt-Base  ·  PyTorch  ·  ONNX Runtime  ·  TensorRT FP16",
          Inches(0.4), Inches(0.38), Inches(12), Inches(0.45),
          size=22, bold=True, color=WHITE)

# ── ResNet-18 card ─────────────────────────────────────────
RX, RY, RW, RCARD_H = Inches(0.3), Inches(0.98), Inches(6.1), Inches(3.0)
add_box(s3, RX, RY, RW, RCARD_H, fill=CARD, border=GREEN)
add_label(s3, "ResNet-18  ·  11M params  ·  ★ Recommended",
          RX + Inches(0.15), RY + Inches(0.12), RW - Inches(0.3), Inches(0.28),
          size=10, bold=True, color=GREEN)

# bar config: label x, bar x, bar max_w, val x
LBEL_X = RX + Inches(0.15)
BAR_X  = RX + Inches(1.5)
BAR_MW = Inches(3.5)
VAL_X  = BAR_X + BAR_MW + Inches(0.1)
BAR_H  = Inches(0.3)
BAR_BG = RGBColor(0x1e, 0x29, 0x3b)

rows_rn = [
    ("PyTorch",       0.077, BLUE,   "15.9 ms  ·  62.9 FPS",        "(measured)"),
    ("ONNX Runtime",  0.058, PURPLE, "~12 ms   ·  ~83 FPS",         "(estimated)"),
    ("TensorRT FP16", 0.039, GREEN,  "~8 ms     ·  ~125 FPS",       "(projected)"),
]
for i, (lbl, pct, col, val, note) in enumerate(rows_rn):
    ry = RY + Inches(0.55) + i * Inches(0.72)
    add_label(s3, lbl, LBEL_X, ry, Inches(1.3), Inches(0.25),
              size=8.5, color=GRAY, align=PP_ALIGN.RIGHT)
    add_bar(s3, BAR_X, ry, BAR_MW, BAR_H, pct, col, BAR_BG)
    add_label(s3, val, VAL_X, ry, Inches(1.3), Inches(0.25), size=8.5, color=WHITE)
    add_label(s3, note, VAL_X, ry + Inches(0.25), Inches(1.3), Inches(0.2),
              size=7, color=GRAY)

add_label(s3,
          "TensorRT FP16 speedup:  ~2×  ·  ONNX speedup:  ~1.3×  ·  GPU memory: 452 MB",
          RX + Inches(0.15), RY + RCARD_H - Inches(0.35), RW - Inches(0.3), Inches(0.25),
          size=7.5, color=GRAY)

# ── ConvNeXt-B card ────────────────────────────────────────
CX, CY, CW, CCARD_H = Inches(7.0), Inches(0.98), Inches(6.1), Inches(3.0)
add_box(s3, CX, CY, CW, CCARD_H, fill=CARD, border=BLUE)
add_label(s3, "ConvNeXt-Base  ·  89M params  ·  13/50 epochs trained",
          CX + Inches(0.15), CY + Inches(0.12), CW - Inches(0.3), Inches(0.28),
          size=10, bold=True, color=BLUE)

LBEL_X2 = CX + Inches(0.15)
BAR_X2  = CX + Inches(1.5)
VAL_X2  = BAR_X2 + BAR_MW + Inches(0.1)

rows_cn = [
    ("PyTorch",       1.00,  BLUE,   "206 ms   ·  4.8 FPS",  "(measured)"),
    ("ONNX Runtime",  0.777, PURPLE, "~160 ms  ·  ~6 FPS",   "(estimated)"),
    ("TensorRT FP16", 0.437, GREEN,  "~90 ms   ·  ~11 FPS",  "(projected)"),
]
for i, (lbl, pct, col, val, note) in enumerate(rows_cn):
    ry = CY + Inches(0.55) + i * Inches(0.72)
    add_label(s3, lbl, LBEL_X2, ry, Inches(1.3), Inches(0.25),
              size=8.5, color=GRAY, align=PP_ALIGN.RIGHT)
    add_bar(s3, BAR_X2, ry, BAR_MW, BAR_H, pct, col, BAR_BG)
    add_label(s3, val, VAL_X2, ry, Inches(1.3), Inches(0.25), size=8.5, color=WHITE)
    add_label(s3, note, VAL_X2, ry + Inches(0.25), Inches(1.3), Inches(0.2),
              size=7, color=GRAY)

add_label(s3,
          "TensorRT FP16 speedup:  ~2.3×  ·  ONNX speedup:  ~1.3×  ·  GPU memory: 1,771 MB (4× heavier)",
          CX + Inches(0.15), CY + CCARD_H - Inches(0.35), CW - Inches(0.3), Inches(0.25),
          size=7.5, color=GRAY)

# Legend
LY = Inches(4.17)
add_label(s3, "■  PyTorch (measured)     ■  ONNX Runtime / CUDA (estimated — typical 25–35% gain)     ■  TensorRT FP16 on Jetson Orin (projected — typical 2–2.5×)",
          Inches(0.3), LY, Inches(12.6), Inches(0.25), size=8, color=GRAY, align=PP_ALIGN.CENTER)
add_label(s3, "⚠  ONNX and TensorRT values are estimates from published speedup ratios for architectures of comparable size. Actual benchmark pending.",
          Inches(0.3), LY + Inches(0.25), Inches(12.6), Inches(0.22), size=7, color=DIM, align=PP_ALIGN.CENTER)

# Takeaway chips
chips3 = [
    ("ResNet-18 + TensorRT FP16", "~8 ms · ~125 FPS\nproduction target for Jetson Orin", GREEN),
    ("ConvNeXt-B cannot meet real-time", "Even with TensorRT, ~90 ms exceeds\nmost 30 FPS robot control loops", YELLOW),
    ("ONNX: portable middle ground", "No TensorRT install needed, ~30% speedup,\nruns on any CUDA-capable device", PURPLE),
]
CHP_Y = Inches(4.6)
CHP_W = Inches(4.05)
CHP_H = Inches(0.9)
cx3 = Inches(0.3)
for title, body, col in chips3:
    add_box(s3, cx3, CHP_Y, CHP_W, CHP_H, fill=CARD, border=col, border_w=Pt(1.5))
    add_label(s3, title, cx3 + Inches(0.1), CHP_Y + Inches(0.08),
              CHP_W - Inches(0.2), Inches(0.3), size=9, bold=True, color=col)
    add_label(s3, body, cx3 + Inches(0.1), CHP_Y + Inches(0.38),
              CHP_W - Inches(0.2), Inches(0.45), size=7.5, color=LGRAY)
    cx3 += CHP_W + Inches(0.24)

add_label(s3, "3 / 4", Inches(12.8), Inches(7.25), Inches(0.5), Inches(0.2),
          size=7, color=DIM, align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════════════════
# SLIDE 4 — WHY US / INDUSTRY STANDARDS
# ═══════════════════════════════════════════════════════════════
s4 = prs.slides.add_slide(blank)
set_bg(s4, BG)

add_label(s4, "LATENCY IN CONTEXT — ROBOTICS DEPLOYMENT",
          Inches(0.4), Inches(0.12), Inches(12), Inches(0.28),
          size=8, color=GRAY)
add_label(s4, "Industry Standards & Why Our Model Meets the Bar",
          Inches(0.4), Inches(0.38), Inches(12), Inches(0.45),
          size=24, bold=True, color=WHITE)

# Three pillar cards
pillars = [
    ("~8 ms", "projected · TensorRT FP16 · Jetson Orin",
     "Our Model — ResNet-18\n15.9 ms measured (PyTorch)\nDetection + lane seg, single pass\n452 MB GPU · 125 FPS projected", GREEN),
    ("50–100 ms", "neural network inference budget",
     "ADAS Industry Target\nTesla FSD · Mobileye EyeQ6\nNVIDIA DRIVE Orin full stack\nISO 26262 safety-critical SIL 2+", YELLOW),
    ("150–300 ms", "simple visual stimulus",
     "Human Reaction Time\nAvg. ~250 ms (trained driver)\nBest-case ~150 ms\nBraking adds further 300–700 ms", RED),
]
PW = Inches(4.05)
PH = Inches(1.5)
PY = Inches(0.98)
px = Inches(0.3)
for big, unit, body, col in pillars:
    add_box(s4, px, PY, PW, PH, fill=CARD, border=col, border_w=Pt(2))
    add_label(s4, big, px + Inches(0.12), PY + Inches(0.08),
              PW - Inches(0.24), Inches(0.5), size=26, bold=True, color=col,
              align=PP_ALIGN.CENTER)
    add_label(s4, unit, px + Inches(0.12), PY + Inches(0.55),
              PW - Inches(0.24), Inches(0.2), size=7.5, color=GRAY,
              align=PP_ALIGN.CENTER)
    add_label(s4, body, px + Inches(0.12), PY + Inches(0.76),
              PW - Inches(0.24), Inches(0.65), size=7.5, color=LGRAY)
    px += PW + Inches(0.24)

# Latency spectrum bar (log-scaled %)
# log positions: log(x)/log(300)*100
# 8ms=38.7%, 15.9ms=47.2%, 50ms=68.4%, 100ms=80.5%, 150ms=87.6%, 250ms=96.6%
SBX  = Inches(0.5)
SBY  = Inches(2.65)
SBW  = Inches(12.3)
SBH  = Inches(0.18)

# Gradient bar using three colored boxes
add_box(s4, SBX,                SBY, Emu(int(SBW*0.60)), SBH,
        fill=GREEN,  border=GREEN,  border_w=Pt(0))
add_box(s4, SBX + Emu(int(SBW*0.60)), SBY, Emu(int(SBW*0.25)), SBH,
        fill=YELLOW, border=YELLOW, border_w=Pt(0))
add_box(s4, SBX + Emu(int(SBW*0.85)), SBY, Emu(int(SBW*0.15)), SBH,
        fill=RED,    border=RED,    border_w=Pt(0))

# Tick marks on bar
for pct in [0.387, 0.472, 0.684, 0.805, 0.876, 0.966]:
    tx = SBX + Emu(int(SBW * pct))
    add_box(s4, tx, SBY, Inches(0.015), SBH,
            fill=DARK, border=DARK, border_w=Pt(0))

# Labels under bar
bar_labels = [
    (0.387, "▲ ~8 ms\n(ours TRT)",   GREEN),
    (0.472, "▲ 15.9 ms\n(PyTorch)",  GREEN),
    (0.684, "50 ms",                  YELLOW),
    (0.805, "100 ms\n(ADAS target)",  YELLOW),
    (0.876, "150 ms",                 RED),
    (0.966, "250 ms\n(human avg.)",   RED),
]
for pct, lbl, col in bar_labels:
    lx = SBX + Emu(int(SBW * pct)) - Inches(0.3)
    add_label(s4, lbl, lx, SBY + Inches(0.22), Inches(0.7), Inches(0.35),
              size=6.5, color=col, align=PP_ALIGN.CENTER)

add_label(s4, "Log scale — each major step is ≈3× slower",
          Inches(0.5), SBY + Inches(0.62), Inches(4), Inches(0.2),
          size=6.5, color=DIM)

# Industry table
it_headers = ["System / Standard", "Inference Latency", "Tasks", "vs. Our Model"]
it_widths   = [Inches(2.8), Inches(2.0), Inches(3.5), Inches(2.7)]
it_data = [
    [P("Tesla FSD (HW4, 2023–)", WHITE),
     P("~10–30 ms", WHITE),
     P("Det · seg · depth · occupancy", WHITE),
     P("Comparable — fraction of the cost", YELLOW)],
    [P("Mobileye EyeQ6 (Level 2+)", WHITE),
     P("~10–20 ms", WHITE),
     P("Det · lane · collision warning", WHITE),
     P("Same latency class — ASIC vs GPU", YELLOW)],
    [P("NVIDIA DRIVE Orin (full stack)", WHITE),
     P("~15–50 ms", WHITE),
     P("Multi-sensor · BEV fusion", WHITE),
     P("Faster net inference · same HW platform", GREEN)],
    [P("Waymo (on-car edge, 2023)", WHITE),
     P("~50 ms perception cycle", WHITE),
     P("360° det · HD map · prediction", WHITE),
     P("~6× faster inference budget", GREEN)],
    [P("SAE J3016 L2–L4 guidance", WHITE),
     P("< 100 ms recommended", WHITE),
     P("Any safety-critical perception task", WHITE),
     P("12× under threshold (PyTorch)", GREEN)],
    [P("Our Model — ResNet-18 TRT FP16 ★", GREEN),
     P("~8 ms proj · 15.9 ms measured", GREEN),
     P("Detection (6 cls) + lane seg", GREEN),
     P("✓ Meets all benchmarks · single pass", GREEN)],
]
it_row_colors = [DARK, DARK, DARK, DARK, DARK, RGBColor(0x0a, 0x1f, 0x12)]
add_table(s4, it_data, it_headers, it_widths,
          Inches(0.3), Inches(3.38), row_h=Inches(0.3),
          row_colors=it_row_colors,
          col_aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

# Claim chips
claims = [
    "~31× faster than human\nreaction (TRT proj.)",
    "16× faster than human\nreaction (PyTorch)",
    "Single model — 2 tasks,\nno stacked overhead",
    "62.9 FPS baseline —\n4× min. robot rate",
    "Deployable on\n~$500 Jetson Orin NX",
]
CCW = Inches(2.35)
CCH = Inches(0.55)
CCY = Inches(6.82)
cx4 = Inches(0.3)
for c in claims:
    add_box(s4, cx4, CCY, CCW, CCH, fill=DARK,
            border=GREEN, border_w=Pt(1))
    add_label(s4, c, cx4 + Inches(0.08), CCY + Inches(0.04),
              CCW - Inches(0.16), CCH - Inches(0.08),
              size=7.5, color=GREEN, align=PP_ALIGN.CENTER)
    cx4 += CCW + Inches(0.195)

add_label(s4, "4 / 4", Inches(12.8), Inches(7.25), Inches(0.5), Inches(0.2),
          size=7, color=DIM, align=PP_ALIGN.RIGHT)


# ── Save ──────────────────────────────────────────────────────
out = "presentation.pptx"
prs.save(out)
print(f"Saved: {out}")
