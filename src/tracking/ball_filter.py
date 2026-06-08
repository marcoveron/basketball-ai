"""Filter best.pt 'ball' detections down to a usable per-frame ball track.

best.pt detects the ball in most frames but also fires ~2.3x/frame on circular
ad-board logos and the scorebug. The decisive signal: the real ball NEVER dwells
at one pixel, while a logo/overlay sits at (nearly) the same pixel for as long as
the camera frames it. So we reject any detection that has many neighbours within a
small radius over a time window ("static" => logo/overlay), then keep the single
most trajectory-consistent candidate per frame.

Pipeline:
  1. main-segment frames only.
  2. static rejection: drop dets with > --max-neighbours others within --radius px
     and +/- --window frames (kills scorebug + dwelling ad logos, no hardcoding).
  3. per frame keep the candidate closest to the local motion prediction (greedy
     forward/backward smoothing), tie-broken by confidence.

Output: <out> CSV (frame, u, v, conf) — feed to ball_kalman.smooth_trajectory.

Usage:
  python -m src.ball_filter --detections runs720/detections.csv \
      --segments runs720/camera_segments.csv --out runs720/ball_clean.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.tracking.rim_track import main_frames


def static_mask(u: np.ndarray, v: np.ndarray, frame: np.ndarray,
                radius: float, window: int, max_neighbours: int) -> np.ndarray:
    """True where a detection looks static (logo/overlay): many near-co-located
    detections within +/- window frames. O(N) via a radius-sized spatial grid."""
    n = len(u)
    cell = radius
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        buckets.setdefault((int(u[i] // cell), int(v[i] // cell)), []).append(i)

    static = np.zeros(n, dtype=bool)
    for i in range(n):
        cu, cv = int(u[i] // cell), int(v[i] // cell)
        cnt = 0
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                for j in buckets.get((cu + du, cv + dv), ()):
                    if j == i:
                        continue
                    if (abs(u[i] - u[j]) <= radius and abs(v[i] - v[j]) <= radius
                            and abs(frame[i] - frame[j]) <= window):
                        cnt += 1
                        if cnt > max_neighbours:
                            break
                if cnt > max_neighbours:
                    break
            if cnt > max_neighbours:
                break
        static[i] = cnt > max_neighbours
    return static


def greedy_track(df: pd.DataFrame, max_jump: float = 120.0) -> pd.DataFrame:
    """Keep one candidate per frame, preferring continuity with recent motion.

    Walk frames in order; predict next position from the last two kept points and
    pick the candidate nearest the prediction (within max_jump px); otherwise the
    highest-confidence candidate. Keeps a single coherent ball path."""
    by_frame = {f: g for f, g in df.groupby("frame")}
    frames = sorted(by_frame)
    kept: list[dict] = []
    prev = None      # (frame, u, v)
    prev2 = None
    for f in frames:
        g = by_frame[f]
        if prev is not None and prev2 is not None and (prev[0] - prev2[0]) > 0:
            # constant-velocity prediction
            dt = prev[0] - prev2[0]
            pu = prev[1] + (prev[1] - prev2[1]) / dt
            pv = prev[2] + (prev[2] - prev2[2]) / dt
        elif prev is not None:
            pu, pv = prev[1], prev[2]
        else:
            pu = pv = None

        if pu is not None:
            d = np.hypot(g["u"] - pu, g["v"] - pv)
            best = g.loc[d.idxmin()]
            if float(d.min()) > max_jump:
                # prediction too far — fall back to most confident candidate
                best = g.loc[g["conf"].idxmax()]
        else:
            best = g.loc[g["conf"].idxmax()]

        row = {"frame": int(f), "u": float(best.u), "v": float(best.v),
               "conf": float(best.conf)}
        kept.append(row)
        prev2 = prev
        prev = (int(f), float(best.u), float(best.v))
    return pd.DataFrame(kept)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs720/ball_clean.csv"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--window", type=int, default=45)
    p.add_argument("--max-neighbours", type=int, default=10)
    args = p.parse_args()

    df = pd.read_csv(args.detections)
    b = df[(df["class_name"] == "ball") & (df["conf"] >= args.conf)].copy()
    b["u"] = (b["x1"] + b["x2"]) / 2
    b["v"] = (b["y1"] + b["y2"]) / 2

    # main-segment frames only
    keep = np.zeros(len(b), dtype=bool)
    fr = b["frame"].to_numpy()
    for s, e in main_frames(args.segments):
        keep |= (fr >= s) & (fr <= e)
    b = b[keep].reset_index(drop=True)
    n0 = len(b)

    stat = static_mask(b["u"].to_numpy(), b["v"].to_numpy(), b["frame"].to_numpy(),
                       args.radius, args.window, args.max_neighbours)
    b = b[~stat].reset_index(drop=True)

    track = greedy_track(b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    track.to_csv(args.out, index=False)
    print(f"ball candidates (main, conf>={args.conf}): {n0}")
    print(f"  after static rejection: {len(b)}  (dropped {n0 - len(b)} static/logo)")
    print(f"  kept 1/frame -> {len(track)} frames  -> {args.out}")


if __name__ == "__main__":
    main()
