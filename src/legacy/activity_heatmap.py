"""Player activity (occupancy) heat map in *pixel* space, overlaid on a real frame.

Why pixel space (not court space): the saved homography is calibrated from a single
frame with 4 hand-clicked points and is currently unreliable, so warping foot
positions to court metres would misplace them. This module sidesteps that entirely:
it accumulates player *foot* positions (bottom-center of each person bbox) directly
in image pixels and renders them as a smooth heat map blended over a reference frame.
It uses only the strongest signal in the pipeline (person detection + tracking), so
it is trustworthy enough to ship as a first deliverable.

Reads:  runs/detections_main.csv  (replay-filtered detections)
Writes: runs/activity_heatmap.png        (heat map blended on the court frame)
        runs/activity_heatmap_raw.png     (heat map alone, on black)

Close-up / replay leakage from the scene filter is suppressed two ways:
  * drop person boxes taller than --max-h px (face/torso close-ups), and
  * ignore frames with fewer than --min-persons people (cutaways, transitions).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def foot_points(det: pd.DataFrame, max_h: float, min_persons: int) -> np.ndarray:
    """Return an (N, 2) array of (u, v) foot points for plausible on-court players."""
    ppl = det[det["class_name"].str.lower().isin({"person", "player"})].copy()
    ppl["h"] = ppl["y2"] - ppl["y1"]
    # Drop egregious close-ups (a real player on this court is < ~220 px tall).
    ppl = ppl[ppl["h"] <= max_h]
    # Keep only frames that look like live action (enough people on screen).
    counts = ppl.groupby("frame")["h"].transform("size")
    ppl = ppl[counts >= min_persons]
    u = (ppl["x1"] + ppl["x2"]) / 2.0
    v = ppl["y2"]  # feet = bottom-center
    return np.column_stack([u.to_numpy(), v.to_numpy()])


def make_heat(points: np.ndarray, w: int, h: int, sigma: float) -> np.ndarray:
    """2D histogram of foot points, Gaussian-blurred and normalized to [0, 1]."""
    heat, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=[h, w], range=[[0, h], [0, w]]
    )
    k = int(sigma * 6) | 1  # odd kernel
    heat = cv2.GaussianBlur(heat, (k, k), sigma)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def colorize(heat: np.ndarray, gamma: float = 0.55) -> np.ndarray:
    """Map normalized heat to a BGR turbo image (gamma lifts mid-low densities)."""
    h8 = (np.power(heat, gamma) * 255).astype(np.uint8)
    return cv2.applyColorMap(h8, cv2.COLORMAP_TURBO)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path, default=Path("runs/detections_main.csv"))
    p.add_argument("--bg", type=Path, default=Path("runs/_frames/f00201.jpg"),
                   help="reference frame to blend the heat map over")
    p.add_argument("--out", type=Path, default=Path("runs/activity_heatmap.png"))
    p.add_argument("--max-h", type=float, default=230.0,
                   help="drop person boxes taller than this (close-up suppression)")
    p.add_argument("--min-persons", type=int, default=4,
                   help="ignore frames with fewer people than this")
    p.add_argument("--sigma", type=float, default=6.0, help="heat blur radius (px)")
    p.add_argument("--alpha", type=float, default=0.55, help="heat opacity over bg")
    args = p.parse_args()

    det = pd.read_csv(args.detections)
    bg = cv2.imread(str(args.bg))
    if bg is None:
        raise SystemExit(f"Could not read background frame: {args.bg}")
    h, w = bg.shape[:2]

    pts = foot_points(det, args.max_h, args.min_persons)
    print(f"Foot points kept: {len(pts)} "
          f"(from {(det['class_name'] == 'person').sum()} raw person detections)")

    heat = make_heat(pts, w, h, args.sigma)
    heat_bgr = colorize(heat)

    # Blend only where there is real heat, so empty court keeps the photo crisp.
    mask = (heat > 0.02)[..., None]
    blended = np.where(mask, cv2.addWeighted(bg, 1 - args.alpha, heat_bgr, args.alpha, 0), bg)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), blended)
    raw_path = args.out.with_name(args.out.stem + "_raw.png")
    cv2.imwrite(str(raw_path), heat_bgr)
    print(f"Wrote {args.out} and {raw_path}")


if __name__ == "__main__":
    main()
