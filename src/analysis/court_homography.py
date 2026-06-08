"""Interactive 4-point court homography picker.

Opens a frame of the video, lets the user navigate frames and click 4 known
court points, maps them to real-world coordinates, computes H = cv2.findHomography.

Persists H to <out>/homography.npy and exposes a load_homography() helper
plus pixel_to_court(u, v) for downstream modules.

Real-world coordinates (meters) — FIBA half-court, origin at baseline center under hoop:
  LEFT_BASELINE_CORNER      = (-7.5,  0.0)
  RIGHT_BASELINE_CORNER     = ( 7.5,  0.0)
  FREE_THROW_LEFT           = (-2.45, 5.80)
  FREE_THROW_RIGHT          = ( 2.45, 5.80)
  PAINT_NEAR_LEFT (baseline) = (-2.45, 0.0)   -- left corner where paint meets baseline
  PAINT_NEAR_RIGHT(baseline) = ( 2.45, 0.0)   -- right corner where paint meets baseline

Controls in the picker window:
  Left/Right arrow  : previous / next frame  (hold for fast scrub)
  ,  /  .           : step -1 / +1 frame
  Space             : jump +30 frames
  Click             : register a landmark point
  r                 : undo last click
  q                 : save and quit  (only when all 4 points selected)
  ESC               : abort

Usage:
  python -m src.court_homography --video video.mp4 --out runs/
  python -m src.court_homography --video video.mp4 --out runs/ --preset paint
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ── landmark presets ───────────────────────────────────────────────────────────
# Use --preset to choose based on what's visible in your video.

PRESETS: dict[str, list[tuple[str, tuple[float, float]]]] = {
    # Full baseline corners + free-throw line: good when whole half-court is visible
    "default": [
        ("LEFT baseline corner",        (-7.5,  0.0)),
        ("RIGHT baseline corner",       ( 7.5,  0.0)),
        ("free-throw line LEFT end",    (-2.45, 5.80)),
        ("free-throw line RIGHT end",   ( 2.45, 5.80)),
    ],
    # Paint (key) corners only: visible even when camera shows just the near end
    "paint": [
        ("paint NEAR-LEFT  (baseline)", (-2.45, 0.0)),
        ("paint NEAR-RIGHT (baseline)", ( 2.45, 0.0)),
        ("free-throw line LEFT end",    (-2.45, 5.80)),
        ("free-throw line RIGHT end",   ( 2.45, 5.80)),
    ],
    # One baseline corner + three paint corners: useful for side-angle cameras
    "mixed": [
        ("LEFT baseline corner",        (-7.5,  0.0)),
        ("paint NEAR-LEFT  (baseline)", (-2.45, 0.0)),
        ("paint NEAR-RIGHT (baseline)", ( 2.45, 0.0)),
        ("free-throw line RIGHT end",   ( 2.45, 5.80)),
    ],
}

DEFAULT_LANDMARKS = PRESETS["default"]


def pick_points(
    video_path: Path,
    landmarks: list[tuple[str, tuple[float, float]]],
    start_frame: int = 0,
) -> tuple[np.ndarray, int]:
    """Navigate frames and collect one (u,v) per landmark via mouse clicks.

    Returns (Nx2 float32 array of pixel coords, chosen frame index).
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(idx: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, total - 1)))
        ok, f = cap.read()
        return f if ok else np.zeros((480, 640, 3), dtype=np.uint8)

    picks: list[tuple[int, int]] = []
    frame_idx = start_frame
    base_frame = read_frame(frame_idx)
    win = "court homography picker"

    def redraw(frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        for i, (u, v) in enumerate(picks):
            cv2.circle(display, (u, v), 6, (0, 255, 0), -1)
            cv2.putText(display, str(i + 1), (u + 8, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(picks) < len(landmarks):
            name, (rx, ry) = landmarks[len(picks)]
            cv2.putText(display,
                        f"Click [{len(picks)+1}/{len(landmarks)}]: {name}  ({rx:.2f}, {ry:.2f}m)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(display, "Arrows/Space: browse  |  r: undo  |  q: save",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        else:
            cv2.putText(display, "All points set. Press 'q' to save, 'r' to undo last.",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(display, f"frame {frame_idx}/{total-1}",
                    (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 180, 180), 1)
        return display

    current_display = [redraw(base_frame)]

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        nonlocal current_display
        if event == cv2.EVENT_LBUTTONDOWN and len(picks) < len(landmarks):
            picks.append((x, y))
            current_display[0] = redraw(base_frame)
            cv2.imshow(win, current_display[0])

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, current_display[0])

    while True:
        key = cv2.waitKey(20) & 0xFF

        step = 0
        if key == 81 or key == ord(","):   # left arrow or comma
            step = -1
        elif key == 83 or key == ord("."):  # right arrow or period
            step = 1
        elif key == 32:                     # space
            step = 30
        elif key == ord("f"):
            step = -30

        if step != 0 and not picks:
            # Only allow frame navigation before first click (keeps picks consistent)
            frame_idx = max(0, min(frame_idx + step, total - 1))
            base_frame = read_frame(frame_idx)
            current_display[0] = redraw(base_frame)
            cv2.imshow(win, current_display[0])

        if key == ord("r") and picks:
            picks.pop()
            current_display[0] = redraw(base_frame)
            cv2.imshow(win, current_display[0])

        if key == ord("q") and len(picks) == len(landmarks):
            break

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit("Picker aborted.")

    cap.release()
    cv2.destroyAllWindows()
    return np.array(picks, dtype=np.float32), frame_idx


def compute_homography(pixel_pts: np.ndarray, court_pts: np.ndarray) -> np.ndarray:
    H, _ = cv2.findHomography(pixel_pts.astype(np.float32), court_pts.astype(np.float32))
    return H


def pixel_to_court(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    pt = np.array([u, v, 1.0])
    out = H @ pt
    return float(out[0] / out[2]), float(out[1] / out[2])


def load_homography(path: Path) -> np.ndarray:
    return np.load(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--frame", type=int, default=0, help="starting frame index")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="default",
                   help="which set of landmarks to use (default/paint/mixed)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    landmarks = PRESETS[args.preset]

    print(f"Preset '{args.preset}': click these {len(landmarks)} points in order:")
    for i, (name, (x, y)) in enumerate(landmarks, 1):
        print(f"  {i}. {name}  →  ({x:.2f}, {y:.2f}) m")
    print()
    print("Controls: Left/Right arrows (or , / .) to browse frames, Space +30, r=undo, q=save")
    print()

    pixel_pts, chosen_frame = pick_points(args.video, landmarks, args.frame)
    court_pts = np.array([pt for _, pt in landmarks], dtype=np.float32)

    H = compute_homography(pixel_pts, court_pts)
    print("Homography H:")
    print(H)

    print("\nSanity (each click → court coords):")
    for (name, (rx, ry)), (u, v) in zip(landmarks, pixel_pts):
        x, y = pixel_to_court(H, float(u), float(v))
        err = np.hypot(x - rx, y - ry)
        print(f"  {name}: target=({rx:.2f},{ry:.2f}) got=({x:.2f},{y:.2f}) err={err:.3f}m")

    np.save(args.out / "homography.npy", H)
    meta = {
        "video": str(args.video),
        "frame": chosen_frame,
        "preset": args.preset,
        "landmarks": [
            {"name": n, "court_xy": list(pt), "pixel_uv": [float(u), float(v)]}
            for (n, pt), (u, v) in zip(landmarks, pixel_pts)
        ],
    }
    (args.out / "homography_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved {args.out / 'homography.npy'}")
    print(f"Saved {args.out / 'homography_meta.json'}")


if __name__ == "__main__":
    main()
