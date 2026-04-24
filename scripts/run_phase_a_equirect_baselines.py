from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase A equirectangular multitask baselines end to end.")
    parser.add_argument("--input_video", default="")
    parser.add_argument("--frames_dir", default="")
    parser.add_argument("--detection_annotations", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--lane_backend", default="dummy")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="configs/multitask/multitask_resnet18.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--min_confidence", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--shared_roi_mask", default="")
    parser.add_argument("--image_h", type=int, default=640)
    parser.add_argument("--image_w", type=int, default=1280)
    parser.add_argument("--roi_mode", choices=["manual_band", "polygon"], default="manual_band")
    parser.add_argument("--roi_top", type=int, default=320)
    parser.add_argument("--roi_bottom", type=int, default=-1)
    parser.add_argument("--roi_left", type=int, default=0)
    parser.add_argument("--roi_right", type=int, default=-1)
    parser.add_argument("--polygon_json", default="")
    parser.add_argument("--mask_thickness", type=int, default=6)
    parser.add_argument("--num_qc_samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    else:
        if not args.input_video:
            raise ValueError("Provide --frames_dir or --input_video")
        frames_root = work_dir / "frames"
        run([
            sys.executable, "tools/equirect_lane_pipeline/extract_frames.py",
            "--input_video", args.input_video,
            "--output_dir", str(frames_root),
            "--fps", str(args.fps),
            "--resize_h", str(args.image_h),
            "--resize_w", str(args.image_w),
        ])
        frames_dir = frames_root / "frames"

    if args.shared_roi_mask:
        roi_mask = Path(args.shared_roi_mask)
    else:
        roi_mask = work_dir / "roi" / "roi_mask.png"
        roi_cmd = [
            sys.executable, "tools/equirect_lane_pipeline/build_roi_masks.py",
            "--image_h", str(args.image_h),
            "--image_w", str(args.image_w),
            "--mode", args.roi_mode,
            "--top", str(args.roi_top),
            "--bottom", str(args.roi_bottom),
            "--left", str(args.roi_left),
            "--right", str(args.roi_right),
            "--output_mask", str(roi_mask),
        ]
        if args.polygon_json:
            roi_cmd.extend(["--polygon_json", args.polygon_json])
        run(roi_cmd)

    lane_dir = work_dir / "lane_labels"
    run([
        sys.executable, "tools/equirect_lane_pipeline/generate_lane_masks.py",
        "--backend", args.lane_backend,
        "--frames_dir", str(frames_dir),
        "--output_dir", str(lane_dir),
        "--checkpoint", args.checkpoint,
        "--config", args.config,
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--score_threshold", str(args.min_confidence),
        "--mask_thickness", str(args.mask_thickness),
        "--roi_mask", str(roi_mask),
        "--save_overlay",
    ])

    filtered_records = lane_dir / "filtered_records.jsonl"
    run([
        sys.executable, "tools/equirect_lane_pipeline/filter_lane_masks.py",
        "--input_records", str(lane_dir / "records.jsonl"),
        "--output_records", str(filtered_records),
        "--min_confidence", str(args.min_confidence),
        "--min_lane_pixels", "1",
        "--drop_empty_masks",
    ])

    qc_dir = work_dir / "qc"
    run([
        sys.executable, "tools/equirect_lane_pipeline/visualize_lane_labels.py",
        "--records", str(filtered_records),
        "--output_dir", str(qc_dir),
        "--num_samples", str(args.num_qc_samples),
        "--seed", "42",
        "--roi_mask", str(roi_mask),
    ])

    manifest_dir = work_dir / "manifests"
    train_manifest = manifest_dir / "train_manifest.json"
    val_manifest = manifest_dir / "val_manifest.json"
    run([
        sys.executable, "tools/equirect_lane_pipeline/build_multitask_manifest.py",
        "--images_dir", str(frames_dir),
        "--detection_annotations", args.detection_annotations,
        "--lane_records", str(filtered_records),
        "--train_manifest", str(train_manifest),
        "--val_manifest", str(val_manifest),
        "--val_ratio", str(args.val_ratio),
        "--seed", "42",
        "--require_segmentation",
        "--shared_roi_mask", str(roi_mask),
    ])

    experiments_dir = work_dir / "experiments"
    exp_cmd = [
        sys.executable, "scripts/run_experiments.py",
        "--config", args.config,
        "--output_dir", str(experiments_dir),
        "--train_manifest", str(train_manifest),
        "--val_manifest", str(val_manifest),
        "--label_source", args.lane_backend,
        "--roi_mode", args.roi_mode,
        "--roi_mask", str(roi_mask),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
    ]
    if args.epochs is not None:
        exp_cmd.extend(["--epochs", str(args.epochs)])
    if args.max_steps_per_epoch:
        exp_cmd.extend(["--max_steps_per_epoch", str(args.max_steps_per_epoch)])
    run(exp_cmd)

    comparison = experiments_dir / "comparison.csv"
    if comparison.exists():
        shutil.copy2(comparison, work_dir / "comparison.csv")


if __name__ == "__main__":
    main()
