"""Cut highlight clips for every made shot, using the scorebug-derived timestamps.

The scoreboard increments are the most reliable shot ground-truth we have on this
panning-camera footage (see attribute_shots / scorebug_ocr notes): each row in
``made_shots.csv`` is a real made shot with an exact frame + game clock + point
value. The increment frame lands *just after* the make, so we use a generous
pre-roll and a short post-roll to capture the build-up and the bucket.

For every shot we cut one clip with a burnt-in caption (shot #, team, point
value, game clock), then optionally stitch them all into a single highlight reel.

Usage:
    python -m src.output.highlights --runs runs720 --video "2021 T3BA總冠軍賽-40s起.mp4"
    python -m src.output.highlights --runs runs720 --pre 6 --post 3 --no-reel
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# kept in sync with the dashboard palette (home 風格整形 / away 台寧天行者)
TEAM_COLOUR = {"home": "0xff5db1", "away": "0x4da3ff"}  # ffmpeg = 0xRRGGBB; pink & blue


def _ffprobe_fps(video: Path) -> float:
    """Read the video's frame rate (handles the '30/1' rational form)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    num, _, den = out.partition("/")
    return float(num) / float(den or 1)


def _ffprobe_duration(video: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def _caption(idx: int, n: int, row: pd.Series) -> str:
    """Build the burnt-in caption, escaped for ffmpeg's drawtext filter."""
    team = str(row["team"]).upper()
    pts = int(row["points"])
    clock = str(row["clock"]) if pd.notna(row["clock"]) and row["clock"] else "--:--"
    score = f"{int(row['home'])}-{int(row['away'])}"
    text = f"HIGHLIGHT {idx}/{n}   {team}  {pts}PT   {clock}   ({score})"
    # ':' must be escaped inside the drawtext value
    return text.replace(":", r"\:")


def cut_clip(video: Path, start: float, dur: float, caption: str,
             colour: str, dst: Path) -> None:
    """Cut a single re-encoded clip with a caption banner along the bottom."""
    drawtext = (
        f"drawtext=fontfile={FONT}:text='{caption}':"
        f"fontcolor={colour}:fontsize=28:box=1:boxcolor=black@0.55:boxborderw=12:"
        f"x=(w-text_w)/2:y=h-text_h-40"
    )
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video),
        "-t", f"{dur:.3f}", "-vf", drawtext,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def make_thumb(video: Path, t: float, dst: Path, width: int = 480) -> None:
    """Grab a single poster frame at time ``t`` (the make moment), scaled to width."""
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
         "-vf", f"scale={width}:-2", str(dst)],
        check=True, capture_output=True,
    )


def build_reel(clips: list[Path], dst: Path) -> None:
    """Concatenate clips (all share codec params) into one reel via the demuxer."""
    listfile = dst.with_suffix(".txt")
    listfile.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
         "-c", "copy", "-movflags", "+faststart", str(dst)],
        check=True, capture_output=True,
    )
    listfile.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=Path("runs720"))
    p.add_argument("--video", type=Path, default=Path("2021 T3BA總冠軍賽-40s起.mp4"))
    p.add_argument("--pre", type=float, default=6.0,
                   help="seconds of footage before the scorebug increment")
    p.add_argument("--post", type=float, default=3.0,
                   help="seconds of footage after the scorebug increment")
    p.add_argument("--no-reel", action="store_true", help="skip the stitched reel")
    args = p.parse_args()

    R = args.runs
    out_dir = R / "highlights"
    thumb_dir = out_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    made = pd.read_csv(R / "made_shots.csv")
    fps = _ffprobe_fps(args.video)
    vid_dur = _ffprobe_duration(args.video)
    n = len(made)
    print(f"{n} made shots | fps={fps:g} | video={vid_dur:.1f}s")

    clips: list[Path] = []
    manifest: list[dict] = []
    for i, (_, row) in enumerate(made.iterrows(), start=1):
        t = row["frame"] / fps
        start = max(0.0, t - args.pre)
        dur = min(args.pre + args.post, vid_dur - start)
        clock = str(row["clock"]).replace(":", "") if pd.notna(row["clock"]) and row["clock"] else "noclock"
        name = f"shot_{i:02d}_{row['team']}_{int(row['points'])}pt_{clock}.mp4"
        dst = out_dir / name
        colour = TEAM_COLOUR.get(str(row["team"]), "white")
        cut_clip(args.video, start, dur, _caption(i, n, row), colour, dst)
        # poster frame at the make moment (== `pre` seconds into the clip)
        thumb = thumb_dir / f"shot_{i:02d}.png"
        make_thumb(args.video, min(t, vid_dur - 0.1), thumb)
        clips.append(dst)
        manifest.append({
            "idx": i, "file": f"highlights/{name}",
            "thumb": f"highlights/thumbs/{thumb.name}",
            "team": row["team"], "points": int(row["points"]),
            "clock": "" if clock == "noclock" else str(row["clock"]),
            "home": int(row["home"]), "away": int(row["away"]),
            "start_s": round(start, 2), "dur_s": round(dur, 2),
        })
        print(f"  [{i:02d}/{n}] {name}  ({start:.1f}s +{dur:.1f}s)")

    pd.DataFrame(manifest).to_csv(out_dir / "manifest.csv", index=False)

    if not args.no_reel and clips:
        reel = out_dir / "highlights_reel.mp4"
        build_reel(clips, reel)
        size_mb = reel.stat().st_size / 1e6
        print(f"reel -> {reel}  ({size_mb:.1f} MB, {len(clips)} clips)")

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
