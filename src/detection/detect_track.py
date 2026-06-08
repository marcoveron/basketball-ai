"""YOLO11 detection + ByteTrack/BoT-SORT tracking over a basketball video.

Emits:
  - <out>/detections.csv : frame, track_id, class_id, class_name, x1, y1, x2, y2, conf
  - <out>/debug.mp4      : annotated video with bboxes, labels, and track IDs

Usage:
  python -m src.detect_track --video video.mp4 --out runs/
  python -m src.detect_track --video video.mp4 --out runs/ --max-frames 1500 --tag sanity
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

COCO_PERSON = 0
COCO_SPORTS_BALL = 32
DEFAULT_CLASSES = [COCO_PERSON, COCO_SPORTS_BALL]


def run(
    video: Path,
    out_dir: Path,
    weights: str = "yolo11s.pt",
    tracker: str = "bytetrack.yaml",
    classes: list[int] | None = None,
    imgsz: int = 640,
    conf: float = 0.20,
    device: str = "0",
    max_frames: int | None = None,
    tag: str = "",
    write_video: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    csv_path = out_dir / f"detections{suffix}.csv"
    video_path = out_dir / f"debug{suffix}.mp4"

    model = YOLO(weights)
    names = model.names

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if max_frames:
        total = min(total, max_frames)

    # Annotated debug video is optional — skipping it saves a lot of time/disk
    # on detection-only re-runs where only the CSV matters.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h)) if write_video else None

    box_ann = sv.BoxAnnotator(thickness=2)
    label_ann = sv.LabelAnnotator(text_thickness=1, text_scale=0.4)

    csv_f = csv_path.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["frame", "track_id", "class_id", "class_name", "x1", "y1", "x2", "y2", "conf"])

    results = model.track(
        source=str(video),
        stream=True,
        tracker=tracker,
        persist=True,
        classes=classes or DEFAULT_CLASSES,
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )

    frame_idx = 0
    try:
        for r in tqdm(results, total=total, desc="detect+track"):
            if max_frames is not None and frame_idx >= max_frames:
                break
            frame = r.orig_img
            det = sv.Detections.from_ultralytics(r)
            if det.tracker_id is None:
                tids = [-1] * len(det)
            else:
                tids = det.tracker_id.tolist()

            for i in range(len(det)):
                x1, y1, x2, y2 = det.xyxy[i].tolist()
                cid = int(det.class_id[i])
                cname = names.get(cid, str(cid))
                csv_w.writerow([
                    frame_idx, tids[i], cid, cname,
                    f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                    f"{float(det.confidence[i]):.3f}",
                ])

            if writer is not None:
                labels = [
                    f"#{tids[i]} {names.get(int(det.class_id[i]), str(int(det.class_id[i])))} {float(det.confidence[i]):.2f}"
                    for i in range(len(det))
                ]
                annotated = box_ann.annotate(scene=frame.copy(), detections=det)
                annotated = label_ann.annotate(scene=annotated, detections=det, labels=labels)
                writer.write(annotated)
            frame_idx += 1
    finally:
        if writer is not None:
            writer.release()
        csv_f.close()

    print(f"\nWrote {frame_idx} frames")
    print(f"  CSV:   {csv_path}")
    if writer is not None:
        print(f"  Video: {video_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs"))
    p.add_argument("--weights", default="yolo11s.pt")
    p.add_argument("--tracker", default="bytetrack.yaml", choices=["bytetrack.yaml", "botsort.yaml"])
    p.add_argument("--classes", type=int, nargs="*", default=DEFAULT_CLASSES,
                   help="COCO class IDs to keep (0=person, 32=sports ball)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--tag", default="", help="suffix for output files (e.g. 'sanity')")
    p.add_argument("--no-video", action="store_true",
                   help="skip writing the annotated debug.mp4 (CSV only; much faster)")
    args = p.parse_args()

    run(
        video=args.video,
        out_dir=args.out,
        weights=args.weights,
        tracker=args.tracker,
        classes=args.classes,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        max_frames=args.max_frames,
        tag=args.tag,
        write_video=not args.no_video,
    )


if __name__ == "__main__":
    main()
