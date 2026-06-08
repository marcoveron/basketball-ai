"""Read the on-screen scorebug to get a score + game-clock timeline.

In 3x3 the scoreboard encodes shot outcomes directly: a +1 to a team's score is
a made shot from INSIDE the arc (1 point), a +2 is a made shot from OUTSIDE the
arc (2 points). The scorebug is a fixed overlay, so OCR on it is far more reliable
than jersey OCR. This gives made shots, their point value (= inside/outside arc),
and exact timing — without needing ball-trajectory made/missed classification.

Layout (bottom scorebug): [foul] HOME_SCORE | CLOCK m:ss | AWAY_SCORE [foul].
We OCR the strip, keep pure-digit tokens, and split the two SCORES from the small
foul indicators by box height (scores are the large digits); the clock is the
token containing ':'.

Outputs:
  <out>/score_timeline.csv : frame, clock, home, away   (sampled + forward-filled)
  <out>/made_shots.csv     : frame, clock, team, points, home, away  (one per increment)

Usage:
  python -m src.scorebug_ocr --video '2021 ....mp4' --out runs720 --sample-every 15
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("FLAGS_use_mkldnn", "0")

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# dedicated boxes in the 1280x720 frame (x1,y1,x2,y2). A tight crop per number
# avoids confusing the scores with the small foul indicators beside them.
HOME_BOX  = (135, 608, 238, 682)
CLOCK_BOX = (244, 608, 388, 682)
AWAY_BOX  = (372, 608, 442, 682)  # both digits of the zero-padded score, excludes foul


def init_paddle():
    from paddleocr import PaddleOCR
    for kwargs in (
        {"use_textline_orientation": False, "lang": "en", "enable_mkldnn": False},
        {"use_textline_orientation": False, "lang": "en"},
        {"use_angle_cls": False, "lang": "en"},
        {"lang": "en"},
    ):
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError):
            continue
    raise SystemExit("Could not initialize PaddleOCR")


def ocr_texts(reader, img: np.ndarray) -> list[tuple[str, float]]:
    """Return [(text, conf)] for a crop, across PaddleOCR 3.x / 2.x."""
    toks: list[tuple[str, float]] = []
    try:
        out = reader.predict(img)
        for page in out:
            texts = page.get("rec_texts", []) if hasattr(page, "get") else getattr(page, "rec_texts", [])
            scores = page.get("rec_scores", []) if hasattr(page, "get") else getattr(page, "rec_scores", [])
            for t, s in zip(texts, scores):
                toks.append((str(t), float(s)))
        return toks
    except (AttributeError, TypeError):
        pass
    out = reader.ocr(img, cls=False)
    for line in (out or []):
        for _box, (t, s) in (line or []):
            toks.append((str(t), float(s)))
    return toks


def _upscale(img: np.ndarray, f: int = 3) -> np.ndarray:
    return cv2.resize(img, (img.shape[1] * f, img.shape[0] * f), interpolation=cv2.INTER_CUBIC)


def read_number(reader, crop: np.ndarray) -> int | None:
    """Best plausible score (0..MAX_SCORE) from a tight single-number crop."""
    best_val, best_conf = None, -1.0
    for t, s in ocr_texts(reader, _upscale(crop)):
        m = re.search(r"\d{1,2}", t.strip())
        if m and int(m.group()) <= MAX_SCORE and s > best_conf:
            best_val, best_conf = int(m.group()), s
    return best_val


def read_clock(reader, crop: np.ndarray) -> str | None:
    for t, s in ocr_texts(reader, _upscale(crop)):
        m = re.search(r"(\d{1,2})[:.\s]?(\d{2})", t.strip())
        if m and int(m.group(1)) <= 9 and int(m.group(2)) < 60:
            return f"{int(m.group(1))}:{m.group(2)}"
    return None


MAX_SCORE = 30  # 3x3 ends at 21 or on the 10-min cap; anything higher is a misread


def clean_monotonic(values: list, debounce: int = 3) -> list:
    """Clean a noisy score series into a non-decreasing step function.

    A new value is accepted only if it is one possession above the confirmed score
    (+1 or +2 — the max single make in 3x3, so a +3 read is an OCR glitch) and
    observed on >= `debounce` consecutive samples. Returns the confirmed score at
    each sample (forward-held). A higher debounce stops a brief misread from
    locking the monotonic series at a wrong high value.
    """
    confirmed = None
    pending = None
    pending_n = 0
    out = []
    for v in values:
        if pd.notna(v):
            v = int(v)
            if confirmed is None:
                # accept the first plausible low value as the baseline (often 0)
                if v <= 3:
                    confirmed = v
            elif v == confirmed:
                pending, pending_n = None, 0
            elif confirmed < v <= confirmed + 2:
                if pending == v:
                    pending_n += 1
                else:
                    pending, pending_n = v, 1
                if pending_n >= debounce:
                    confirmed = v
                    pending, pending_n = None, 0
        out.append(confirmed)
    return out


def detect_made_shots(timeline: pd.DataFrame) -> pd.DataFrame:
    """Find score increments. A clean +1/+2 to one side = a made shot of that value.
    Ignores decreases / multi-jumps (OCR glitches or correction)."""
    rows = []
    prev_h = prev_a = None
    for r in timeline.itertuples():
        h, a = r.home, r.away
        if prev_h is not None and pd.notna(h) and pd.notna(prev_h):
            d = int(h) - int(prev_h)
            if d in (1, 2):
                rows.append({"frame": r.frame, "clock": r.clock, "team": "home",
                             "points": d, "home": int(h), "away": int(a) if pd.notna(a) else None})
        if prev_a is not None and pd.notna(a) and pd.notna(prev_a):
            d = int(a) - int(prev_a)
            if d in (1, 2):
                rows.append({"frame": r.frame, "clock": r.clock, "team": "away",
                             "points": d, "home": int(h) if pd.notna(h) else None, "away": int(a)})
        if pd.notna(h):
            prev_h = int(h)
        if pd.notna(a):
            prev_a = int(a)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs720"))
    p.add_argument("--sample-every", type=int, default=15)
    args = p.parse_args()

    def box(im, b):
        x1, y1, x2, y2 = b
        return im[y1:y2, x1:x2]

    reader = init_paddle()
    cap = cv2.VideoCapture(str(args.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows = []
    for f in tqdm(range(0, total, args.sample_every), desc="scorebug OCR"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            continue
        rows.append({
            "frame": f,
            "clock": read_clock(reader, box(im, CLOCK_BOX)),
            "home": read_number(reader, box(im, HOME_BOX)),
            "away": read_number(reader, box(im, AWAY_BOX)),
        })
    cap.release()

    tl = pd.DataFrame(rows)
    tl["home_raw"] = tl["home"]
    tl["away_raw"] = tl["away"]
    # clean each side into a debounced non-decreasing step function
    tl["home"] = clean_monotonic(tl["home_raw"].tolist())
    tl["away"] = clean_monotonic(tl["away_raw"].tolist())
    args.out.mkdir(parents=True, exist_ok=True)
    tl.to_csv(args.out / "score_timeline.csv", index=False)

    made = detect_made_shots(tl)
    made.to_csv(args.out / "made_shots.csv", index=False)
    fh = tl["home"].dropna()
    fa = tl["away"].dropna()
    print(f"\nscore_timeline.csv: {len(tl)} samples")
    print(f"final score (OCR): home {int(fh.iloc[-1]) if len(fh) else '?'} - "
          f"away {int(fa.iloc[-1]) if len(fa) else '?'}")
    print(f"made_shots.csv: {len(made)} made shots  "
          f"(1pt: {(made.points==1).sum()}, 2pt: {(made.points==2).sum()})")


if __name__ == "__main__":
    main()
