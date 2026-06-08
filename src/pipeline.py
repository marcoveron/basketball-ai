"""Top-level CLI: run the full basketball analytics pipeline on a video.

Stages (run in order, each gated by --stages):
  detect    : YOLO + ByteTrack -> runs/detections.csv + runs/debug.mp4
  scene     : histogram scene-cut detection -> runs/camera_segments.csv
              + runs/detections_main.csv  (replays filtered out)
  homography: interactive 4-point picker -> runs/homography.npy (user input required)
  ocr       : per-tracklet jersey OCR -> runs/jersey_assignments.csv
  ball      : Kalman-smoothed ball trajectory -> runs/ball_trajectory.csv
  shots     : shot events + made/missed -> runs/shots.csv
  heatmap   : per-jersey shot heat maps -> runs/heatmaps/*.png

Usage:
  python -m src.pipeline --video video.mp4 --out runs/ --weights yolo11s.pt
  python -m src.pipeline --video video.mp4 --out runs/ --stages detect scene
  python -m src.pipeline --video video.mp4 --out runs/ --stages homography ocr ball shots heatmap
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = ["detect", "scene", "homography", "ocr", "ball", "shots", "heatmap"]


def total_frames(video: Path) -> int:
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def run_module(module: str, args: list[str]) -> None:
    cmd = [sys.executable, "-m", f"src.{module}", *args]
    print(f"\n=== {module} ===\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--rim-bbox", default=None,
                   help="optional fixed rim bbox 'x1,y1,x2,y2' for shots stage")
    p.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    p.add_argument("--sample-every", type=int, default=5, help="OCR every Nth frame")
    p.add_argument("--scene-threshold", type=float, default=0.75,
                   help="histogram correlation threshold for scene cuts (default 0.75)")
    p.add_argument("--min-main-frames", type=int, default=150,
                   help="min frames for a segment to be labeled 'main' (default 150)")
    p.add_argument("--homography-preset", default="paint",
                   choices=["default", "paint", "mixed"],
                   help="landmark preset for court homography (default: paint)")
    p.add_argument("--scene-preview", action="store_true",
                   help="save per-segment preview images during scene stage")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    det_csv      = args.out / "detections.csv"
    det_main_csv = args.out / "detections_main.csv"
    ball_csv     = args.out / "ball_trajectory.csv"
    homo_npy     = args.out / "homography.npy"
    jersey_csv   = args.out / "jersey_assignments.csv"
    shots_csv    = args.out / "shots.csv"
    seg_csv      = args.out / "camera_segments.csv"

    # Use filtered detections if available, otherwise fall back to full detections
    def effective_det_csv() -> Path:
        return det_main_csv if det_main_csv.exists() else det_csv

    if "detect" in args.stages:
        run_module("detection.detect_track", [
            "--video", str(args.video), "--out", str(args.out),
            "--weights", args.weights, "--tracker", args.tracker,
            "--imgsz", str(args.imgsz), "--conf", str(args.conf),
            "--device", args.device,
        ])

    if "scene" in args.stages:
        scene_args = [
            "--video", str(args.video), "--out", str(args.out),
            "--threshold", str(args.scene_threshold),
            "--min-main-frames", str(args.min_main_frames),
            "--detections", str(det_csv),
        ]
        if args.scene_preview:
            scene_args.append("--preview")
        run_module("scene_filter", scene_args)
        print(f"\n  Review {seg_csv} and edit any wrong labels before continuing.")
        print("  Then re-run with --stages homography ocr ball shots heatmap\n")

    if "homography" in args.stages:
        if homo_npy.exists():
            print(f"\n=== homography === skipped: {homo_npy} already exists")
        else:
            run_module("court_homography", [
                "--video", str(args.video), "--out", str(args.out),
                "--preset", args.homography_preset,
            ])

    if "ocr" in args.stages:
        run_module("jersey_ocr", [
            "--video", str(args.video), "--detections", str(effective_det_csv()),
            "--out", str(args.out), "--sample-every", str(args.sample_every),
        ])

    if "ball" in args.stages:
        run_module("ball_kalman", [
            "--detections", str(effective_det_csv()), "--out", str(ball_csv),
            "--total-frames", str(total_frames(args.video)),
        ])

    if "shots" in args.stages:
        shots_args = [
            "--detections", str(effective_det_csv()),
            "--ball-trajectory", str(ball_csv),
            "--homography", str(homo_npy),
            "--out", str(shots_csv),
            "--jersey-csv", str(jersey_csv),
        ]
        if args.rim_bbox:
            shots_args += ["--rim-bbox", args.rim_bbox]
        run_module("shot_detector", shots_args)

    if "heatmap" in args.stages:
        run_module("heatmap", [
            "--shots", str(shots_csv),
            "--out-dir", str(args.out / "heatmaps"),
        ])

    print("\nPipeline finished.")
    print(f"  Artifacts in: {args.out.resolve()}")


if __name__ == "__main__":
    main()
