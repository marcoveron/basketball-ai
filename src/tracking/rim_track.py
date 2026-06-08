"""Per-frame rim (hoop) track from best.pt 'basket' detections.

The broadcast camera pans and zooms, so the rim moves in image space every
frame — there is no single fixed rim bbox. This module builds a smooth,
per-frame rim position that downstream shot detection and per-shot court
registration use as the court origin.

Strategy (the real rim dominates confidence — ~0.73 vs ~0.28 for the circular
ad-logo false positives, so a max-conf pick per frame is usually correct):
  1. Keep 'basket' detections with conf >= --conf.
  2. Per main-camera frame, take the highest-conf basket centre as a raw pick.
  3. Reject outliers (ad false positives) via a rolling-median Hampel filter.
  4. Linearly interpolate gaps *within* each main segment (never across cuts).

Output: <out>/rim_track.csv  (frame, u, v, w, h, conf, source)
  source = "detected" | "interp".  Only main-segment frames are emitted.

Usage:
  python -m src.rim_track --detections runs720/detections.csv \
      --segments runs720/camera_segments.csv --out runs720/rim_track.csv
  # add --video <v> --overlay to dump validation frames with the rim drawn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main_frames(segments_csv: Path) -> list[tuple[int, int]]:
    """Return list of (start, end) inclusive frame ranges labelled 'main'."""
    seg = pd.read_csv(segments_csv)
    return [(int(r.start_frame), int(r.end_frame))
            for r in seg.itertuples() if str(r.label) == "main"]


def hampel_mask(values: np.ndarray, window: int, n_sigma: float) -> np.ndarray:
    """True where a point is an outlier vs the rolling median (Hampel filter)."""
    s = pd.Series(values)
    med = s.rolling(window, center=True, min_periods=1).median()
    diff = (s - med).abs()
    mad = diff.rolling(window, center=True, min_periods=1).median()
    # 1.4826 scales MAD to std for a normal distribution; guard against mad==0
    thresh = n_sigma * 1.4826 * mad.replace(0, np.nan)
    return (diff > thresh).fillna(False).to_numpy()


def build_track(
    detections_csv: Path,
    segments_csv: Path,
    conf: float = 0.45,
    window: int = 15,
    n_sigma: float = 3.0,
) -> pd.DataFrame:
    df = pd.read_csv(detections_csv)
    baskets = df[(df["class_name"] == "basket") & (df["conf"] >= conf)].copy()
    baskets["u"] = (baskets["x1"] + baskets["x2"]) / 2
    baskets["v"] = (baskets["y1"] + baskets["y2"]) / 2
    baskets["w"] = baskets["x2"] - baskets["x1"]
    baskets["h"] = baskets["y2"] - baskets["y1"]

    # one pick per frame: highest confidence basket
    pick = (baskets.sort_values("conf", ascending=False)
            .groupby("frame", as_index=False).first())

    out_rows = []
    for start, end in main_frames(segments_csv):
        seg = pick[(pick["frame"] >= start) & (pick["frame"] <= end)].sort_values("frame")
        if len(seg) < 2:
            continue
        # drop spatial outliers (ad-logo false positives) on u and v
        out = hampel_mask(seg["u"].to_numpy(), window, n_sigma) | \
              hampel_mask(seg["v"].to_numpy(), window, n_sigma)
        seg = seg[~out]
        if len(seg) < 2:
            continue
        # interpolate every frame in the segment from the surviving detections
        frames = np.arange(start, end + 1)
        det_f = seg["frame"].to_numpy()
        rec = {"frame": frames}
        for col in ("u", "v", "w", "h", "conf"):
            rec[col] = np.interp(frames, det_f, seg[col].to_numpy())
        sub = pd.DataFrame(rec)
        sub["source"] = np.where(np.isin(frames, det_f), "detected", "interp")
        out_rows.append(sub)

    if not out_rows:
        return pd.DataFrame(columns=["frame", "u", "v", "w", "h", "conf", "source"])
    return pd.concat(out_rows, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs720/rim_track.csv"))
    p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--window", type=int, default=15)
    p.add_argument("--n-sigma", type=float, default=3.0)
    p.add_argument("--video", type=Path, default=None,
                   help="if given with --overlay, dump validation frames")
    p.add_argument("--overlay", action="store_true")
    args = p.parse_args()

    track = build_track(args.detections, args.segments, args.conf,
                        args.window, args.n_sigma)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    track.to_csv(args.out, index=False)

    n_det = int((track["source"] == "detected").sum()) if len(track) else 0
    print(f"Wrote {args.out}  ({len(track)} frames, {n_det} detected, "
          f"{len(track) - n_det} interpolated)")

    if args.overlay and args.video and len(track):
        import cv2
        cap = cv2.VideoCapture(str(args.video))
        outdir = args.out.parent / "_frames"
        outdir.mkdir(parents=True, exist_ok=True)
        idx = track.set_index("frame")
        for f in track["frame"].iloc[:: max(1, len(track) // 6)][:6]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, im = cap.read()
            if not ok:
                continue
            row = idx.loc[f]
            u, v, w, h = (float(row.u), float(row.v), float(row.w), float(row.h))
            cv2.rectangle(im, (int(u - w / 2), int(v - h / 2)),
                          (int(u + w / 2), int(v + h / 2)), (0, 255, 255), 2)
            cv2.circle(im, (int(u), int(v)), 4, (0, 0, 255), -1)
            cv2.putText(im, f"rim f{int(f)} ({row.source})", (int(u) - 40, int(v) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imwrite(str(outdir / f"rim_check_f{int(f)}.jpg"), im)
        cap.release()
        print(f"Saved rim overlay frames to {outdir}")


if __name__ == "__main__":
    main()
