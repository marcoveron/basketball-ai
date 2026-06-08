"""Jersey-number OCR per track ID, with tracklet-level majority voting.

For each tracked person, we sample N frames across the tracklet, crop the chest region
(35-65% of bbox height, with horizontal padding), upscale 3x with cubic, and feed it
to PaddleOCR (digit whitelist). We aggregate all OCR predictions per track ID by
weighted majority vote — high-confidence predictions count more.

Reads:  <out>/detections.csv (with track_id), and the video
Writes: <out>/jersey_assignments.csv  with columns:
            track_id, jersey, votes, top_conf, n_frames_seen

Usage:
  python -m src.jersey_ocr --video video.mp4 --detections runs/detections.csv \\
      --out runs/ --sample-every 5 --min-votes 3
"""
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

# PaddlePaddle 3.x's oneDNN backend is incompatible with the new PIR executor on
# CPU (raises "ConvertPirAttribute2RuntimeAttribute not support"). Disable MKLDNN
# before paddle is imported anywhere.
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

DIGIT_RE = re.compile(r"\d{1,2}")  # jerseys 0-99


def _safe_int(s: str) -> int | None:
    m = DIGIT_RE.search(s)
    return int(m.group(0)) if m else None


def chest_crop(frame: np.ndarray, x1: float, y1: float, x2: float, y2: float,
               h_lo: float = 0.20, h_hi: float = 0.65,
               w_pad: float = 0.05) -> np.ndarray | None:
    """Crop the chest region: vertically [h_lo, h_hi] of bbox, horizontally full + small pad."""
    H, W = frame.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    if bw < 20 or bh < 40:
        return None
    cx1 = max(0, int(x1 - w_pad * bw))
    cx2 = min(W, int(x2 + w_pad * bw))
    cy1 = max(0, int(y1 + h_lo * bh))
    cy2 = min(H, int(y1 + h_hi * bh))
    if cy2 <= cy1 or cx2 <= cx1:
        return None
    return frame[cy1:cy2, cx1:cx2]


def upscale(img: np.ndarray, factor: int = 3) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def run_paddle(reader, img: np.ndarray) -> list[tuple[int, float]]:
    """Return list of (digit, conf) found in the image. Handles PaddleOCR 3.x and 2.x APIs."""
    # PaddleOCR 3.x: reader.predict(img) returns list of dicts with 'rec_texts' & 'rec_scores'
    try:
        out = reader.predict(img)
        results: list[tuple[int, float]] = []
        for page in out:
            texts = page.get("rec_texts", []) if hasattr(page, "get") else getattr(page, "rec_texts", [])
            scores = page.get("rec_scores", []) if hasattr(page, "get") else getattr(page, "rec_scores", [])
            for t, s in zip(texts, scores):
                n = _safe_int(str(t))
                if n is not None:
                    results.append((n, float(s)))
        return results
    except (AttributeError, TypeError):
        pass
    # Fallback: 2.x API
    out = reader.ocr(img, cls=True)
    results = []
    for line in (out or []):
        for box, (t, s) in (line or []):
            n = _safe_int(str(t))
            if n is not None:
                results.append((n, float(s)))
    return results


def run(
    video: Path,
    det_csv: Path,
    out_dir: Path,
    sample_every: int = 5,
    person_class_names: tuple[str, ...] = ("person", "player"),
    min_votes: int = 3,
    conf_floor: float = 0.50,
) -> None:
    det = pd.read_csv(det_csv)
    det = det[det["class_name"].str.lower().isin([c.lower() for c in person_class_names])]
    det = det[det["track_id"] >= 0]
    if det.empty:
        raise SystemExit("No tracked persons in detections.csv")

    # Sample frames: keep every Nth frame index that has any person
    frames_to_sample = sorted(set(det["frame"].unique()[::sample_every].tolist()))
    print(f"Will sample OCR on {len(frames_to_sample)} frames out of {det['frame'].nunique()}")

    # PaddleOCR lazy import (heavy dependency). The constructor signature changed
    # across versions: 3.x dropped `show_log` and renamed `use_angle_cls` to
    # `use_textline_orientation`. Try newest-first and degrade gracefully.
    from paddleocr import PaddleOCR
    reader = None
    for kwargs in (
        {"use_textline_orientation": True, "lang": "en", "enable_mkldnn": False},  # 3.x
        {"use_textline_orientation": True, "lang": "en"},  # 3.x w/o mkldnn flag
        {"use_angle_cls": True, "lang": "en"},              # 2.x
        {"lang": "en"},
        {},
    ):
        try:
            reader = PaddleOCR(**kwargs)
            break
        except (ValueError, TypeError):
            continue
    if reader is None:
        raise SystemExit("Could not initialize PaddleOCR with any known argument set")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {video}")

    # votes[track_id][digit] = list of confidence values
    votes: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[int, int] = defaultdict(int)

    by_frame = det.groupby("frame")
    for f in tqdm(frames_to_sample, desc="OCR sampling"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        sub = by_frame.get_group(f)
        for r in sub.itertuples():
            tid = int(r.track_id)
            seen[tid] += 1
            crop = chest_crop(frame, r.x1, r.y1, r.x2, r.y2)
            if crop is None:
                continue
            up = upscale(crop, factor=3)
            preds = run_paddle(reader, up)
            for digit, conf in preds:
                if 0 <= digit <= 99 and conf >= conf_floor:
                    votes[tid][digit].append(conf)
    cap.release()

    # Aggregate per track: weighted majority (sum of confidences)
    out_rows = []
    for tid in sorted(votes.keys()):
        digit_scores = {d: sum(cs) for d, cs in votes[tid].items()}
        if not digit_scores:
            continue
        best_digit, best_score = max(digit_scores.items(), key=lambda x: x[1])
        n_votes = len(votes[tid][best_digit])
        if n_votes < min_votes:
            continue
        top_conf = max(votes[tid][best_digit])
        out_rows.append({
            "track_id": tid,
            "jersey": best_digit,
            "votes": n_votes,
            "weighted_score": round(best_score, 3),
            "top_conf": round(top_conf, 3),
            "n_frames_seen": seen[tid],
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "jersey_assignments.csv"
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out_rows)} assignments)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--sample-every", type=int, default=5, help="OCR every Nth frame")
    p.add_argument("--person-classes", nargs="*", default=["person", "player"],
                   help="class names treated as people")
    p.add_argument("--min-votes", type=int, default=3,
                   help="discard a track unless at least this many high-conf agreeing predictions")
    p.add_argument("--conf-floor", type=float, default=0.50)
    args = p.parse_args()

    run(
        video=args.video,
        det_csv=args.detections,
        out_dir=args.out,
        sample_every=args.sample_every,
        person_class_names=tuple(args.person_classes),
        min_votes=args.min_votes,
        conf_floor=args.conf_floor,
    )


if __name__ == "__main__":
    main()
