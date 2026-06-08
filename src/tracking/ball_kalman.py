"""Extract ball-only series from detections and apply a 2D constant-acceleration Kalman filter.

Reads:  <out>/detections.csv  (per-frame raw detections, with class_name == 'sports ball'
                              OR 'ball' depending on weights provenance)
Writes: <out>/ball_trajectory.csv  with columns:
            frame, u_raw, v_raw, u, v, vx, vy, ax, ay, observed

`observed` is True for frames with a real detection; False for Kalman-predicted gaps.

State vector x = [u, v, vx, vy, ax, ay]^T (pixel coords + velocity + accel).
Measurement = [u, v]^T (raw detection center).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

BALL_CLASS_ALIASES = {"sports ball", "ball", "basketball"}


def build_kf(dt: float = 1.0, q_var: float = 5.0, r_var: float = 8.0) -> KalmanFilter:
    """6-state, 2-measurement constant-acceleration Kalman filter.

    dt   : seconds between frames (use 1.0 for "1 frame" units so velocities are px/frame)
    q_var: process noise variance (how erratic the ball's motion can be)
    r_var: measurement noise variance (detection center jitter)
    """
    kf = KalmanFilter(dim_x=6, dim_z=2)
    kf.F = np.array([
        [1, 0, dt, 0, 0.5 * dt * dt, 0],
        [0, 1, 0, dt, 0,             0.5 * dt * dt],
        [0, 0, 1, 0, dt,            0],
        [0, 0, 0, 1, 0,             dt],
        [0, 0, 0, 0, 1,             0],
        [0, 0, 0, 0, 0,             1],
    ], dtype=float)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
    ], dtype=float)
    kf.R = np.eye(2) * r_var
    kf.Q = np.eye(6) * q_var
    kf.P = np.eye(6) * 1000.0  # high initial uncertainty
    return kf


def extract_ball_series(det_csv: Path) -> pd.DataFrame:
    """Pick the highest-confidence ball detection per frame (no tracker needed here)."""
    df = pd.read_csv(det_csv)
    df = df[df["class_name"].str.lower().isin(BALL_CLASS_ALIASES)]
    if df.empty:
        return df
    df = df.sort_values(["frame", "conf"], ascending=[True, False]).drop_duplicates("frame")
    df["u"] = (df["x1"] + df["x2"]) / 2.0
    df["v"] = (df["y1"] + df["y2"]) / 2.0
    return df[["frame", "u", "v", "conf"]].reset_index(drop=True)


def smooth_trajectory(
    raw: pd.DataFrame,
    total_frames: int,
    max_gap: int = 30,
) -> pd.DataFrame:
    """Run KF across [0, total_frames). Predict every frame; update only when a detection exists.

    `max_gap` : if the ball has been unobserved for more than this many consecutive frames,
                reset filter uncertainty so we don't drift indefinitely.
    """
    kf = build_kf()
    initialized = False
    rows: list[dict] = []
    obs_lookup: dict[int, tuple[float, float]] = {
        int(r.frame): (float(r.u), float(r.v)) for r in raw.itertuples()
    }

    gap = 0
    for f in range(total_frames):
        if f in obs_lookup:
            u_obs, v_obs = obs_lookup[f]
            if not initialized:
                kf.x = np.array([u_obs, v_obs, 0, 0, 0, 0], dtype=float)
                initialized = True
            else:
                kf.predict()
                kf.update(np.array([u_obs, v_obs]))
            gap = 0
            observed = True
            u_raw, v_raw = u_obs, v_obs
        else:
            if not initialized:
                rows.append({"frame": f, "u_raw": np.nan, "v_raw": np.nan,
                             "u": np.nan, "v": np.nan, "vx": np.nan, "vy": np.nan,
                             "ax": np.nan, "ay": np.nan, "observed": False})
                continue
            kf.predict()
            gap += 1
            if gap > max_gap:
                kf.P *= 10.0  # spread uncertainty so a fresh detection re-anchors quickly
                gap = 0
            observed = False
            u_raw, v_raw = np.nan, np.nan

        u, v, vx, vy, ax, ay = kf.x[:6]
        rows.append({
            "frame": f, "u_raw": u_raw, "v_raw": v_raw,
            "u": float(u), "v": float(v),
            "vx": float(vx), "vy": float(vy),
            "ax": float(ax), "ay": float(ay),
            "observed": observed,
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/ball_trajectory.csv"))
    p.add_argument("--total-frames", type=int, required=True,
                   help="total frames in the video (so we predict across the full span)")
    p.add_argument("--max-gap", type=int, default=30)
    args = p.parse_args()

    raw = extract_ball_series(args.detections)
    print(f"Raw ball detections: {len(raw)} / {args.total_frames} frames "
          f"({len(raw)/args.total_frames*100:.1f}% recall)")
    traj = smooth_trajectory(raw, args.total_frames, max_gap=args.max_gap)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    traj.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
