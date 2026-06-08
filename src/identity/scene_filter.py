"""Detect camera cuts and label segments as 'main' or 'other'.

Compares grayscale histograms between consecutive frames to find scene cuts,
groups frames into segments, and auto-labels long segments as 'main'.

Outputs:
  <out>/camera_segments.csv       -- segment_id, start_frame, end_frame, n_frames, label
  <out>/detections_main.csv       -- detections.csv filtered to main-camera frames only
  <out>/scene_previews/           -- one sample JPEG per segment (if --preview)

Downstream stages (ocr, ball, shots, heatmap) use detections_main.csv instead
of detections.csv to avoid counting replays twice.

Usage:
  python -m src.scene_filter --video video.mp4 --out runs/
  python -m src.scene_filter --video video.mp4 --out runs/ --preview
  python -m src.scene_filter --video video.mp4 --out runs/ --threshold 0.80 --min-main-frames 90
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _hist(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(h, h)
    return h


def detect_cuts(
    video: Path,
    threshold: float = 0.75,
    sample_every: int = 1,
) -> list[int]:
    """Return list of frame indices where a scene cut starts (excluding frame 0)."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cuts: list[int] = []
    prev_hist: np.ndarray | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every == 0:
            h = _hist(frame)
            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, h, cv2.HISTCMP_CORREL)
                if corr < threshold:
                    cuts.append(frame_idx)
            prev_hist = h
        frame_idx += 1

    cap.release()
    print(f"  Scanned {frame_idx} frames, found {len(cuts)} cut(s) at threshold={threshold}")
    return cuts, frame_idx


def build_segments(
    cuts: list[int],
    total_frames: int,
    min_main_frames: int = 150,
) -> pd.DataFrame:
    """Build a DataFrame of segments from cut points."""
    boundaries = [0] + cuts + [total_frames]
    rows = []
    for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        n = end - start
        # Long segments = main camera; short ones = replay/cutaway
        label = "main" if n >= min_main_frames else "other"
        rows.append({
            "segment_id": i,
            "start_frame": start,
            "end_frame": end - 1,
            "n_frames": n,
            "label": label,
        })
    return pd.DataFrame(rows)


def filter_detections(
    detections_csv: Path,
    segments: pd.DataFrame,
    out_csv: Path,
) -> int:
    """Write a filtered CSV keeping only detections in 'main' segments."""
    det = pd.read_csv(detections_csv)
    main_segs = segments[segments["label"] == "main"]

    mask = pd.Series(False, index=det.index)
    for _, seg in main_segs.iterrows():
        mask |= (det["frame"] >= seg["start_frame"]) & (det["frame"] <= seg["end_frame"])

    filtered = det[mask]
    filtered.to_csv(out_csv, index=False)
    return len(filtered)


def save_previews(video: Path, segments: pd.DataFrame, out_dir: Path) -> None:
    """Save one sample frame per segment for manual review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))

    for _, seg in segments.iterrows():
        mid = int((seg["start_frame"] + seg["end_frame"]) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, frame = cap.read()
        if not ok:
            continue
        label = seg["label"]
        sid = seg["segment_id"]
        n = seg["n_frames"]
        cv2.putText(frame, f"seg {sid}  [{label}]  frames {seg['start_frame']}-{seg['end_frame']}  ({n}f)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        fname = out_dir / f"seg{sid:03d}_{label}_f{seg['start_frame']}-{seg['end_frame']}.jpg"
        cv2.imwrite(str(fname), frame)

    cap.release()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--threshold", type=float, default=0.75,
                   help="histogram correlation below this = scene cut (0-1, lower=more sensitive)")
    p.add_argument("--min-main-frames", type=int, default=150,
                   help="segments shorter than this are labeled 'other' (default 150 ≈ 5s at 30fps)")
    p.add_argument("--sample-every", type=int, default=1,
                   help="check every Nth frame for speed (1=every frame)")
    p.add_argument("--preview", action="store_true",
                   help="save a sample JPEG per segment to <out>/scene_previews/")
    p.add_argument("--detections", type=Path, default=None,
                   help="path to detections.csv to filter (default: <out>/detections.csv)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    det_csv = args.detections or (args.out / "detections.csv")

    print("=== scene_filter: scanning for camera cuts ===")
    cuts, total_frames = detect_cuts(args.video, args.threshold, args.sample_every)

    segments = build_segments(cuts, total_frames, args.min_main_frames)
    seg_csv = args.out / "camera_segments.csv"
    segments.to_csv(seg_csv, index=False)

    print(f"\nSegments ({len(segments)} total):")
    for _, seg in segments.iterrows():
        marker = "✓" if seg["label"] == "main" else "✗"
        print(f"  {marker} seg {seg['segment_id']:3d}  frames {seg['start_frame']:6d}-{seg['end_frame']:6d}"
              f"  ({seg['n_frames']:5d}f)  [{seg['label']}]")

    main_count = (segments["label"] == "main").sum()
    main_frames = segments.loc[segments["label"] == "main", "n_frames"].sum()
    print(f"\n  {main_count} main segment(s), {main_frames} frames"
          f"  ({main_frames/total_frames*100:.1f}% of video)")
    print(f"  Saved: {seg_csv}")
    print()
    print(f"  Tip: edit {seg_csv} to correct any wrong labels, then re-run with --detections to re-filter.")

    if det_csv.exists():
        out_det = args.out / "detections_main.csv"
        n_kept = filter_detections(det_csv, segments, out_det)
        n_total = sum(1 for _ in open(det_csv)) - 1
        print(f"  Filtered detections: {n_kept}/{n_total} rows kept → {out_det}")
    else:
        print(f"  (detections CSV not found at {det_csv} — run 'detect' stage first)")

    if args.preview:
        preview_dir = args.out / "scene_previews"
        save_previews(args.video, segments, preview_dir)
        print(f"  Previews saved to {preview_dir}/")


if __name__ == "__main__":
    main()
