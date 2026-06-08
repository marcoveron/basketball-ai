"""Attribute each scorebug-detected made shot to a shooter (jersey).

The scoreboard updates ~1-2s AFTER the basket, so for an increment at frame F we
look backwards: find where the ball passed nearest the rim (the make), then in the
~1s before that ("release window") take the person nearest the ball as the shooter,
and map their track -> jersey. This is inherently approximate (noisy ball, frag-
mented tracks, imperfect jersey OCR) so every row carries a confidence.

Output: <out>/shots_attributed.csv
  frame, clock, team, points, shooter_jersey, shooter_track, conf, f_rim

Usage:
  python -m src.attribute_shots --out runs720
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("FLAGS_use_mkldnn", "0")

import cv2
import numpy as np
import pandas as pd

from src.identity.jersey_ocr import chest_crop, run_paddle, upscale


def init_paddle():
    from paddleocr import PaddleOCR
    for kwargs in (
        {"use_textline_orientation": True, "lang": "en", "enable_mkldnn": False},
        {"use_textline_orientation": True, "lang": "en"},
        {"use_angle_cls": True, "lang": "en"},
        {"lang": "en"},
    ):
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError):
            continue
    raise SystemExit("Could not initialize PaddleOCR")


def nearest_person(persons: pd.DataFrame, bu: float, bv: float) -> tuple[int, float]:
    """Track id of the person whose box is nearest (bu,bv); distance to box."""
    if persons.empty:
        return -1, 1e9
    # distance from point to bbox (0 if inside), plus prefer the box whose upper
    # body / hands are near the ball
    x1, y1, x2, y2 = (persons.x1.to_numpy(), persons.y1.to_numpy(),
                      persons.x2.to_numpy(), persons.y2.to_numpy())
    z = np.zeros(len(persons))
    dx = np.maximum.reduce([x1 - bu, z, bu - x2])
    dy = np.maximum.reduce([y1 - bv, z, bv - y2])
    d = np.hypot(dx, dy)
    i = int(np.argmin(d))
    return int(persons.iloc[i].track_id), float(d[i])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("runs720"))
    p.add_argument("--video", type=Path, default=Path("2021 T3BA總冠軍賽-40s起.mp4"))
    p.add_argument("--made", type=Path, default=None)
    p.add_argument("--ball", type=Path, default=None)
    p.add_argument("--rim", type=Path, default=None)
    p.add_argument("--detections", type=Path, default=None)
    p.add_argument("--jersey", type=Path, default=None)
    p.add_argument("--rim-proximity", type=float, default=90.0,
                   help="max px from rim for the ball to count as 'at the rim'")
    p.add_argument("--lookback", type=int, default=150, help="frames before F to search for the make")
    p.add_argument("--release-window", type=int, default=45, help="frames before the make = release window")
    args = p.parse_args()

    o = args.out
    made = pd.read_csv(args.made or o / "made_shots.csv")
    ball = pd.read_csv(args.ball or o / "ball_clean.csv").set_index("frame")
    rim = pd.read_csv(args.rim or o / "rim_track.csv").set_index("frame")
    det = pd.read_csv(args.detections or o / "detections.csv")
    jersey = pd.read_csv(args.jersey or o / "jersey_assignments.csv")

    persons = det[det["class_name"] == "person"].copy()
    persons_by_frame = {f: g for f, g in persons.groupby("frame")}
    # best jersey per track (highest weighted_score)
    j_by_track = (jersey.sort_values("weighted_score", ascending=False)
                  .drop_duplicates("track_id").set_index("track_id")["jersey"].to_dict())

    rows = []
    for sh in made.itertuples():
        F = int(sh.frame)
        # 1) locate the make: ball closest to rim within the lookback window
        f_rim, best_d = None, 1e9
        for f in range(max(0, F - args.lookback), F + 1):
            if f in ball.index and f in rim.index:
                b, r = ball.loc[f], rim.loc[f]
                d = float(np.hypot(b.u - r.u, b.v - r.v))
                if d < best_d:
                    best_d, f_rim = d, f
        if f_rim is None or best_d > args.rim_proximity:
            f_rim = F - 30  # fallback: ~1s before the score update

        # 2) release window: who is nearest the ball just before the make
        votes: Counter = Counter()
        for f in range(max(0, f_rim - args.release_window), f_rim - 2):
            if f not in ball.index or f not in persons_by_frame:
                continue
            b = ball.loc[f]
            tid, d = nearest_person(persons_by_frame[f], float(b.u), float(b.v))
            if tid >= 0 and d < 60:           # ball within 60px of a player box
                votes[tid] += 1
        shooter_track, conf = -1, 0.0
        if votes:
            shooter_track, n = votes.most_common(1)[0]
            conf = n / max(1, sum(votes.values()))
        rows.append({
            "frame": F, "clock": sh.clock, "team": sh.team, "points": sh.points,
            "shooter_track": shooter_track, "conf": round(conf, 2),
            "f_rim": f_rim, "rim_dist": round(best_d, 1),
            "_rel_lo": max(0, f_rim - args.release_window), "_rel_hi": f_rim,
        })

    # ---- targeted jersey OCR on the shooter's box across the release window ----
    # Track ids are fragmented so a track->jersey lookup mostly misses; instead we
    # OCR the shooter's chest directly at shot time and vote the digit.
    reader = init_paddle()
    cap = cv2.VideoCapture(str(args.video))
    persons_idx = persons.set_index(["frame", "track_id"])
    for row in rows:
        tid = row["shooter_track"]
        votes: dict[int, float] = defaultdict(float)
        for f in range(int(row["_rel_lo"]), int(row["_rel_hi"]) + 1):
            try:
                pb = persons_idx.loc[(f, tid)]
            except KeyError:
                continue
            pb = pb.iloc[0] if isinstance(pb, pd.DataFrame) else pb
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, im = cap.read()
            if not ok:
                continue
            crop = chest_crop(im, pb.x1, pb.y1, pb.x2, pb.y2)
            if crop is None:
                continue
            for digit, sc in run_paddle(reader, upscale(crop)):
                if 0 <= digit <= 99:
                    votes[digit] += sc
        if votes:
            best = max(votes, key=votes.get)
            row["shooter_jersey"] = best
            row["ocr_score"] = round(votes[best], 2)
        else:
            row["shooter_jersey"] = None
            row["ocr_score"] = 0.0
    cap.release()

    out = pd.DataFrame(rows).drop(columns=["_rel_lo", "_rel_hi"])
    out = out[["frame", "clock", "team", "points", "shooter_jersey", "ocr_score",
               "shooter_track", "conf", "f_rim", "rim_dist"]]
    out.to_csv(o / "shots_attributed.csv", index=False)
    named = out["shooter_jersey"].notna().sum()
    print(f"Attributed {len(out)} shots; {named} got a jersey via targeted OCR.")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
