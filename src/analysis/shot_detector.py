"""Detect shot events, classify made/missed, and attribute the shooter.

Combines:
  - rim bbox (either from detections.csv class 'rim'/'hoop', or a fixed bbox supplied via --rim-bbox)
  - Kalman-smoothed ball trajectory (runs/ball_trajectory.csv)
  - per-frame tracked players from detections.csv (with track_id)
  - homography H (runs/homography.npy) to map shooter foot position to court coords

Algorithm (heuristic):
  1. Define up_region (above rim, rim diameter tall) and down_region (below rim).
  2. A shot trigger fires when the ball passes from up_region → near rim plane within W frames.
  3. Classify MADE iff: ball crossed rim x-extent moving downward, did not re-emerge above
     the rim within ~15 frames, and was eventually observed below the rim.
  4. Attribute shooter: walk backward from the trigger frame until a player bbox center is
     within K * mean_player_height of the ball. That track_id is the shooter.
  5. Shot court coords = homography-warped foot position of the shooter at the release frame.

Writes runs/shots.csv with columns:
  frame, shooter_track_id, jersey, court_x, court_y, result, conf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RIM_CLASS_ALIASES = {"rim", "hoop", "basket"}


def load_rim_bbox(detections: pd.DataFrame, fixed: tuple[float, float, float, float] | None) -> tuple[float, float, float, float]:
    """Return (x1, y1, x2, y2) for the rim. Either fixed (user-supplied) or median of detections."""
    if fixed is not None:
        return fixed
    rim_df = detections[detections["class_name"].str.lower().isin(RIM_CLASS_ALIASES)]
    if rim_df.empty:
        raise SystemExit("No rim detections found; please pass --rim-bbox x1,y1,x2,y2")
    return (
        float(rim_df["x1"].median()),
        float(rim_df["y1"].median()),
        float(rim_df["x2"].median()),
        float(rim_df["y2"].median()),
    )


def find_shot_events(ball: pd.DataFrame, rim: tuple[float, float, float, float],
                     window: int = 30, up_extra: float = 1.0, down_extra: float = 1.0,
                     dy_thresh: float = 0.5) -> list[dict]:
    """Find frames where the ball plausibly entered the rim region descending.

    Returns list of dicts: frame (trigger), result ('MADE'/'MISSED'), reason.
    """
    x1, y1, x2, y2 = rim
    rim_w = x2 - x1
    rim_h = y2 - y1
    rim_cx = (x1 + x2) / 2.0
    # Up/down regions: extend up_extra * rim_h above the rim, down_extra * rim_h below
    up_top    = y1 - up_extra * rim_h
    up_bottom = y1
    down_top    = y2
    down_bottom = y2 + down_extra * rim_h
    # Allow a horizontal slack equal to half the rim width on each side
    x_lo = x1 - 0.3 * rim_w
    x_hi = x2 + 0.3 * rim_w

    events: list[dict] = []
    in_cooldown_until = -1
    for i, row in ball.iterrows():
        if np.isnan(row.u) or np.isnan(row.v):
            continue
        if i < in_cooldown_until:
            continue
        # Ball above the rim within slack, with downward velocity → trigger candidate
        in_up = (up_top <= row.v <= up_bottom) and (x_lo <= row.u <= x_hi) and (row.vy > dy_thresh)
        if not in_up:
            continue

        # Look ahead `window` frames
        future = ball[(ball["frame"] > row.frame) & (ball["frame"] <= row.frame + window)]
        if future.empty:
            continue
        future = future.dropna(subset=["u", "v"])
        if future.empty:
            continue

        # Did it cross the rim x-extent moving downward?
        crossed_x = future[(future["u"] >= x1) & (future["u"] <= x2)]
        if crossed_x.empty:
            # ball missed the rim entirely — count as MISSED if it eventually drops far below
            below_far = future[future["v"] > down_bottom]
            if not below_far.empty:
                events.append({"frame": int(row.frame), "result": "MISSED", "reason": "no_rim_crossing"})
                in_cooldown_until = int(row.frame) + window
            continue

        # Did it re-emerge above the rim within the next ~15 frames after crossing?
        first_cross_frame = int(crossed_x.iloc[0]["frame"])
        post = future[(future["frame"] > first_cross_frame) & (future["frame"] <= first_cross_frame + 15)]
        re_emerged = (post["v"] < y1).any()

        # Did it eventually land clearly below the rim?
        below = future[future["v"] > down_bottom]

        if re_emerged or below.empty:
            events.append({"frame": int(row.frame), "result": "MISSED", "reason": "bounce_or_no_below"})
        else:
            events.append({"frame": int(row.frame), "result": "MADE", "reason": "clean_through"})
        in_cooldown_until = int(row.frame) + window
    return events


def attribute_shooter(
    trigger_frame: int,
    players: pd.DataFrame,
    ball: pd.DataFrame,
    lookback: int = 30,
    distance_factor: float = 1.2,
) -> tuple[int | None, int | None]:
    """Walk back from trigger_frame to find the player closest to the ball at release.

    Returns (shooter_track_id, release_frame) or (None, None) if no candidate.
    """
    bx_lookup = {int(r.frame): (r.u, r.v) for r in ball.itertuples() if not np.isnan(r.u)}
    # Use mean player bbox height as a distance scale
    mean_h = (players["y2"] - players["y1"]).mean() if not players.empty else 80.0

    for f in range(trigger_frame, max(0, trigger_frame - lookback) - 1, -1):
        if f not in bx_lookup:
            continue
        bx, by = bx_lookup[f]
        same_frame = players[players["frame"] == f]
        if same_frame.empty:
            continue
        cx = (same_frame["x1"] + same_frame["x2"]) / 2.0
        cy = (same_frame["y1"] + same_frame["y2"]) / 2.0
        d = np.hypot(cx - bx, cy - by)
        thresh = distance_factor * mean_h
        candidates = same_frame[d < thresh]
        if not candidates.empty:
            best_idx = d[candidates.index].idxmin()
            return int(same_frame.loc[best_idx, "track_id"]), int(f)
    return None, None


def foot_to_court(player_row: pd.Series, H: np.ndarray) -> tuple[float, float]:
    """Use bottom-center of bbox as the on-floor point, warp to court coords."""
    fu = (player_row["x1"] + player_row["x2"]) / 2.0
    fv = player_row["y2"]
    pt = H @ np.array([fu, fv, 1.0])
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--ball-trajectory", type=Path, required=True)
    p.add_argument("--homography", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/shots.csv"))
    p.add_argument("--rim-bbox", default=None,
                   help="fixed rim bbox 'x1,y1,x2,y2' (else median rim detection)")
    p.add_argument("--jersey-csv", type=Path, default=None,
                   help="optional: runs/jersey_assignments.csv to attach jersey numbers")
    p.add_argument("--person-classes", nargs="*", default=["person", "player"])
    args = p.parse_args()

    det = pd.read_csv(args.detections)
    ball = pd.read_csv(args.ball_trajectory)
    H = np.load(args.homography)
    fixed = None
    if args.rim_bbox:
        fixed = tuple(float(x) for x in args.rim_bbox.split(","))
    rim = load_rim_bbox(det, fixed)

    players = det[det["class_name"].str.lower().isin([c.lower() for c in args.person_classes])]
    players = players[players["track_id"] >= 0]

    jersey_map = {}
    if args.jersey_csv and args.jersey_csv.exists():
        jdf = pd.read_csv(args.jersey_csv)
        jersey_map = dict(zip(jdf["track_id"].astype(int), jdf["jersey"].astype(int)))

    events = find_shot_events(ball, rim)
    print(f"Detected {len(events)} shot events (using rim={rim})")

    rows = []
    for ev in events:
        tid, release_frame = attribute_shooter(ev["frame"], players, ball)
        if tid is None:
            rows.append({
                "frame": ev["frame"], "shooter_track_id": -1, "jersey": -1,
                "court_x": np.nan, "court_y": np.nan,
                "result": ev["result"], "release_frame": -1, "reason": ev["reason"],
            })
            continue
        prow = players[(players["frame"] == release_frame) & (players["track_id"] == tid)].iloc[0]
        cx, cy = foot_to_court(prow, H)
        rows.append({
            "frame": ev["frame"], "shooter_track_id": tid,
            "jersey": int(jersey_map.get(tid, -1)),
            "court_x": round(cx, 3), "court_y": round(cy, 3),
            "result": ev["result"], "release_frame": release_frame, "reason": ev["reason"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")
    if rows:
        made = sum(1 for r in rows if r["result"] == "MADE")
        print(f"  MADE: {made}, MISSED: {len(rows) - made}")


if __name__ == "__main__":
    main()
