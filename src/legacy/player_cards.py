"""Per-player activity cards: one occupancy heat map per identified jersey number.

Joins the jersey OCR output (track_id -> jersey) with the detections so every
player's foot positions can be pooled across all of their (fragmented) track IDs
and rendered as an individual heat map, blended on a reference frame. Like
`activity_heatmap`, this works in pixel space and needs neither the (currently
unreliable) homography nor the ball trajectory — only person detection + OCR.

Reads:  runs/detections_main.csv, runs/jersey_assignments.csv
Writes: runs/player_cards/jersey_<N>.png   (one per well-supported jersey)
        runs/player_cards/roster.csv        (summary table)

A jersey is only rendered if its tracks collected at least --min-support OCR
votes in total, which filters out low-confidence misreads (e.g. a stray "91").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.legacy.activity_heatmap import colorize, make_heat


def valid_action_frames(det: pd.DataFrame, min_persons: int) -> set[int]:
    """Frames that look like live play (>= min_persons people on screen)."""
    ppl = det[det["class_name"].str.lower().isin({"person", "player"})]
    counts = ppl.groupby("frame").size()
    return set(counts[counts >= min_persons].index.tolist())


def render_card(points: np.ndarray, bg: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    h, w = bg.shape[:2]
    if len(points) == 0:
        return bg.copy()
    heat = make_heat(points, w, h, sigma)
    heat_bgr = colorize(heat)
    mask = (heat > 0.02)[..., None]
    return np.where(mask, cv2.addWeighted(bg, 1 - alpha, heat_bgr, alpha, 0), bg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path, default=Path("runs/detections_main.csv"))
    p.add_argument("--jersey-csv", type=Path, default=Path("runs/jersey_assignments.csv"))
    p.add_argument("--bg", type=Path, default=Path("runs/_frames/f00201.jpg"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/player_cards"))
    p.add_argument("--min-support", type=int, default=10,
                   help="min total OCR votes across a jersey's tracks to render it")
    p.add_argument("--max-h", type=float, default=230.0, help="close-up suppression")
    p.add_argument("--min-persons", type=int, default=4, help="live-action frame filter")
    p.add_argument("--sigma", type=float, default=7.0)
    p.add_argument("--alpha", type=float, default=0.6)
    args = p.parse_args()

    det = pd.read_csv(args.detections)
    jdf = pd.read_csv(args.jersey_csv)
    bg = cv2.imread(str(args.bg))
    if bg is None:
        raise SystemExit(f"Could not read background frame: {args.bg}")

    # Jersey-level support: pool votes across all of a number's track IDs.
    support = jdf.groupby("jersey")["votes"].sum()
    keep = support[support >= args.min_support].sort_values(ascending=False)
    track2jersey = dict(zip(jdf["track_id"].astype(int), jdf["jersey"].astype(int)))

    ppl = det[det["class_name"].str.lower().isin({"person", "player"})].copy()
    ppl = ppl[(ppl["y2"] - ppl["y1"]) <= args.max_h]
    action = valid_action_frames(det, args.min_persons)
    ppl = ppl[ppl["frame"].isin(action)]
    ppl["jersey"] = ppl["track_id"].astype(int).map(track2jersey)
    ppl["foot_u"] = (ppl["x1"] + ppl["x2"]) / 2.0
    ppl["foot_v"] = ppl["y2"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    roster = []
    for jersey in keep.index:
        pts = ppl.loc[ppl["jersey"] == jersey, ["foot_u", "foot_v"]].to_numpy()
        card = render_card(pts, bg, args.sigma, args.alpha)
        n_tracks = jdf.loc[jdf["jersey"] == jersey, "track_id"].nunique()
        label = f"#{jersey}  -  {len(pts)} positions, {n_tracks} tracks, {int(keep[jersey])} votes"
        cv2.rectangle(card, (0, 0), (card.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(card, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        out = args.out_dir / f"jersey_{jersey}.png"
        cv2.imwrite(str(out), card)
        roster.append({"jersey": int(jersey), "positions": len(pts),
                       "n_tracks": int(n_tracks), "total_votes": int(keep[jersey])})
        print(f"  wrote {out}  ({len(pts)} foot points)")

    pd.DataFrame(roster).to_csv(args.out_dir / "roster.csv", index=False)
    print(f"Rendered {len(roster)} player cards (support >= {args.min_support} votes) to {args.out_dir}")


if __name__ == "__main__":
    main()
