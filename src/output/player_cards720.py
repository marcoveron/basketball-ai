"""Digital player cards (720p) — export clean thumbnails + stats as data.

For each strongly-seen jersey we export:
  * a clean thumbnail cropped from the broadcast video (the player as actually tracked),
    saved as ``cards/thumb_<N>.png`` — no text baked in
  * its stats (tracked time, confident OCR reads, max confidence, tracks merged,
    confidence tier) collected into ``cards/cards.json``

The dashboard (``build_dashboard.py``) reads this JSON and renders interactive cards:
a compact tile shows the photo + number + confidence at a glance, and clicking it
opens a modal with the full stats. Keeping stats as data (not baked into the PNG)
is what makes that click-to-reveal interaction possible.

We deliberately do NOT attach per-player shot/rebound stats: on this footage those
can't be attributed reliably (noisy ball track + resolution-capped jersey OCR). The
dashboard adds ONE clearly-labelled "roadmap" card showing what higher-res + a fixed
court camera would unlock.

Track ids here come from detections_main.csv (the scene-filtered set the jersey OCR ran
on); detections.csv was re-tracked later with a different id scheme — don't use it here.

Usage:  python -m src.output.player_cards720 --runs runs720 --video "<video>.mp4"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

FPS = 30.0
IMG_W, IMG_H = 1280, 720          # 720p source frame size


def confidence(votes: float) -> tuple[str, str]:
    if votes >= 60:
        return "HIGH", "#1db954"
    if votes >= 15:
        return "MED", "#e0a800"
    return "LOW", "#d9534f"


def grab_crop(video: Path, frame_idx: int, box: tuple[int, int, int, int],
              pad: float = 0.12) -> np.ndarray | None:
    """Read one video frame and return an RGB crop of box=(x1,y1,x2,y2) with padding."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    H, W = fr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - pad * bw)); x2 = min(W, int(x2 + pad * bw))
    y1 = max(0, int(y1 - pad * bh)); y2 = min(H, int(y2 + pad * bh))
    crop = fr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def pick_thumb_frame(dm: pd.DataFrame, track_id: int) -> tuple[int, tuple] | None:
    """Pick the best frame to crop this player from: tall (clear), person-shaped (not a
    merged blob), centred in the frame and away from the image edges — edge crops tend to
    catch LED ad banners / courtside clutter, which make ugly thumbnails."""
    sub = dm[(dm.track_id == track_id) & (dm.class_name == "person")].copy()
    if sub.empty:
        return None
    sub["h"] = sub.y2 - sub.y1
    sub["w"] = (sub.x2 - sub.x1).clip(lower=1)
    sub["cx"] = (sub.x1 + sub.x2) / 2
    # sane height band + person-like aspect (avoid close-ups/replays and merged blobs)
    band = sub[(sub.h >= 110) & (sub.h <= 430) & (sub.h / sub.w >= 1.5)]
    cand = band if not band.empty else sub
    margin = 0.06 * IMG_W
    centred = 1 - (cand.cx - IMG_W / 2).abs() / (IMG_W / 2)        # 1 at centre, 0 at edge
    near_edge = ((cand.x1 < margin) | (cand.x2 > IMG_W - margin)).astype(float)
    score = cand.h * (0.6 + 0.4 * centred) - 60 * near_edge
    row = cand.loc[score.idxmax()]
    return int(row.frame), (row.x1, row.y1, row.x2, row.y2)


def save_thumb(thumb: np.ndarray | None, out: Path, target_w: int = 460) -> bool:
    """Save the RGB crop as a clean portrait-ish PNG (no text). Returns success."""
    if thumb is None:
        return False
    h, w = thumb.shape[:2]
    if w < 4 or h < 4:
        return False
    scale = target_w / w
    img = Image.fromarray(thumb).resize((target_w, max(1, int(h * scale))), Image.LANCZOS)
    img.save(out)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs720"))
    p.add_argument("--video", type=Path, default=Path("2021 T3BA總冠軍賽-40s起.mp4"))
    p.add_argument("--top", type=int, default=6, help="how many real player cards to render")
    args = p.parse_args()
    R = args.runs
    cards_dir = R / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    jdf = pd.read_csv(R / "jersey_assignments.csv")
    dm = pd.read_csv(R / "detections_main.csv")

    agg = jdf.groupby("jersey").agg(
        weighted=("weighted_score", "sum"), votes=("votes", "sum"),
        tracks=("track_id", "nunique"), maxconf=("top_conf", "max"),
    ).sort_values("weighted", ascending=False)
    chosen = agg.head(args.top)

    cards: list[dict] = []
    for jersey, r in chosen.iterrows():
        jersey = int(jersey)
        # best track for this jersey, then its clearest frame
        jt = jdf[jdf.jersey == jersey].sort_values("weighted_score", ascending=False)
        thumb = None
        for tr in jt.track_id.astype(int):
            pf = pick_thumb_frame(dm, tr)
            if pf:
                thumb = grab_crop(args.video, pf[0], pf[1])
                if thumb is not None:
                    break
        thumb_name = f"thumb_{jersey}.png"
        has_thumb = save_thumb(thumb, cards_dir / thumb_name)
        # tracked seconds = unique frames across this jersey's tracks
        frames = dm[dm.track_id.isin(jt.track_id) & (dm.class_name == "person")].frame.nunique()
        tag, col = confidence(float(r.weighted))
        cards.append({
            "jersey": jersey,
            "thumb": thumb_name if has_thumb else None,
            "seconds": round(frames / FPS),
            "votes": int(r.votes),
            "maxconf": round(float(r.maxconf), 2),
            "tracks": int(r.tracks),
            "weighted": round(float(r.weighted), 1),
            "tier": tag,
            "tier_color": col,
        })
        print(f"#{jersey:>2}: {frames/FPS:.0f}s · {int(r.votes)} reads · conf {float(r.maxconf):.2f} · "
              f"{tag}  {'(thumb)' if has_thumb else '(NO thumb)'}")

    (cards_dir / "cards.json").write_text(json.dumps(cards, indent=2), encoding="utf-8")
    print(f"Wrote {len(cards)} thumbnails + cards.json to {cards_dir}/")


if __name__ == "__main__":
    main()
