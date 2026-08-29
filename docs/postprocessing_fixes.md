# Post-processing Fixes — inference/infer_python.py

Two bugs were identified from visual inspection of TensorRT inference outputs and
fixed on 2026-05-26. Both are in `inference/infer_python.py`.

---

## Fix 1 — FPN-level ownership filter (cross-scale duplicate suppression)

### Problem

The FCOS head predicts at three FPN levels simultaneously:

| Level | Stride | Feature map | Intended target size |
|-------|--------|-------------|----------------------|
| p3 | 8 | 80 × 160 | Small objects (side ≤ ~64 px) |
| p4 | 16 | 40 × 80 | Medium objects (side ~64–128 px) |
| p5 | 32 | 20 × 40 | Large objects (side ≥ ~128 px) |

When a medium-sized object (e.g. a stop sign or pothole) falls near the boundary
between two levels, all three decode independently and each produces a valid box
at a slightly different pixel boundary (because the cell-centre offsets differ by
stride). These boxes have IoU in the range 0.10–0.30 — well below the NMS
threshold of 0.35 — so NMS never suppresses them, and the same physical object
appears 2–3 times in the output.

### Fix

Add `size_min` / `size_max` parameters (longest box side, in image pixels) to
`MultitaskEngine._decode_level`. Each call site receives the range that matches
its FPN stride:

```python
# p3 — small objects only
all_dets += self._decode_level(..., size_max=96)

# p4 — medium objects; overlaps intentionally with p3 and p5 border cases
all_dets += self._decode_level(..., size_min=48, size_max=192)

# p5 — large objects only
all_dets += self._decode_level(..., size_min=96)
```

Any box whose longest side falls outside the level's range is dropped before the
levels are merged and before NMS runs. The small overlaps (48–96 px for p3/p4,
96–192 px for p4/p5) mean border-size objects still pass through exactly one
level; intra-level NMS handles any remaining duplicates within that region.

### Why not just lower the NMS threshold?

A lower global NMS threshold (e.g. 0.1) would also suppress cross-scale boxes but
would simultaneously merge genuinely distinct nearby objects (two pedestrians
standing close together, two adjacent barrels). The ownership filter acts
*before* NMS and is conditioned on box size, not spatial proximity, so it has no
effect on same-scale, same-class, closely-spaced objects.

### Size range calibration

The ranges were chosen empirically for a 640 × 1280 equirectangular input:

| Level | size_min | size_max | Rationale |
|-------|----------|----------|-----------|
| p3 | 0 | 96 | 8 × 12 cells — objects up to ~96 px tall |
| p4 | 48 | 192 | 4 × 8 cells — centred on 96 px with ±50 % tolerance |
| p5 | 96 | ∞ | 2 × 4 cells — anything a full 3+ strides across |

If the camera or resolution changes, re-calibrate by looking at the expected
pixel size of each class at typical operating distances.

---

## Fix 2 — Label pill clipping at image edges

### Problem

`draw_detections` placed the class label at `lx = int(d["x1"])` without checking
whether the text pill overflowed the right or bottom edge. For boxes whose `x1`
was within ~100 px of `x = 1280`, the pill background rectangle and text were
drawn partly or entirely off-canvas, producing truncated labels like `"ire"` or
`"ure"` instead of `"tire"`.

### Fix

Clamp `lx` and `ly` before drawing so the pill always fits within the image
boundaries:

```python
lx = max(0, min(int(d["x1"]), img.shape[1] - ts[0] - 4))
ly = max(ts[1] + 2, min(int(d["y1"]) - baseline - 3, img.shape[0] - baseline - 2))
```

- Right-edge clamp: `img.shape[1] - ts[0] - 4` reserves the exact text width plus
  a 4-pixel margin before the image boundary.
- Top-edge clamp: `max(ts[1] + 2, ...)` prevents the pill from going above row 0
  when the box itself is at the very top of the frame.
- Bottom-edge clamp: prevents the pill baseline from going below the last row.

The bounding box rectangle is drawn at the original `d["x1"]` position; only the
label pill is repositioned.

---

---

## Fix 3 — Centre-proximity deduplication (cross-scale duplicate suppression)

### Problem

Even with FPN-level ownership filtering (Fix 1), some objects produce cross-scale
duplicate predictions that survive NMS. This happens when the model predicts the
same object at two different FPN levels and the resulting boxes differ significantly
in size — for example:

- p3 predicts a tight 45 × 24 px box around a pothole centre
- p4 predicts a looser 139 × 71 px box around the same centre

Box overlap (IoU) between these is ≈ 0.11 — well below the 0.35 NMS threshold —
so NMS never fires. Confirmed across multiple frames:

| Object | p3 box side | p4 box side | Centre distance | IoU |
|--------|-------------|-------------|-----------------|-----|
| pothole (frame_00066) | 45 px | 139 px | 27 px | 0.11 |
| stop_sign (frame_01575) | 42 px | 93 px | 16 px | ~0.15 |
| pothole (frame_01707) | 55 px | 99 px | 6 px | 0.27 |

### Fix

After NMS, apply a second deduplication pass based on centre distance.
Implemented in `MultitaskEngine._center_dedup`:

1. Sort detections descending by score (so the best prediction wins).
2. For each candidate, compare against every already-kept detection of the
   **same class**.
3. Compute the Euclidean distance between centres.
4. Suppress the candidate if:
   ```
   centre_distance < 0.55 × min(diag_a, diag_b)
   ```
   where `diag_a`, `diag_b` are the diagonal lengths of the two boxes.

Using the minimum diagonal as the reference scale means the threshold adapts to
the smaller of the two predictions. A factor of 0.55 was calibrated against real
frames: it suppresses cross-scale pairs whose centres are ≤ 28 px apart while
preserving genuinely distinct nearby objects (e.g. three pothole markers 40+ px
apart in the same frame).

### Why centre distance rather than IoU?

IoU rewards overlap, so it fails when one box is much larger than the other —
exactly the cross-scale case. Centre distance rewards co-location and is agnostic
to box size, making it the right tool for "same object, different scale" suppression.

### Pipeline order

```
decode p3 / p4 / p5  (with FPN size ranges)
        ↓
class-agnostic NMS (IoU ≥ 0.35)      ← handles same-scale duplicates
        ↓
centre-proximity dedup (dist < 0.5 × min_diag)  ← handles cross-scale duplicates
        ↓
spatial filters (robot body, equirectangular edges)
```

---

## Files changed

| File | Change |
|------|--------|
| `inference/infer_python.py` | `_decode_level`: added `size_min`/`size_max` params + size filter array op |
| `inference/infer_python.py` | `infer()`: added per-level size range arguments to the three `_decode_level` calls |
| `inference/infer_python.py` | `draw_detections`: clamped `lx`/`ly` to image bounds before drawing label pill |
