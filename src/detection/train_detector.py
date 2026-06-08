"""Fine-tune yolo11s.pt on a Roboflow basketball dataset.

Usage:
  python -m src.train_detector \
      --data datasets/basketball/data.yaml \
      --epochs 30 --imgsz 512 --batch 8 \
      --name basketball_run1

Output: runs/detect/<name>/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="path to dataset data.yaml")
    p.add_argument("--weights", default="yolo11s.pt", help="starting weights")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=512, help="Lower (e.g. 512) saves RTX 4050 VRAM")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="basketball_run1")
    p.add_argument("--patience", type=int, default=10, help="early stop after N epochs without improvement")
    args = p.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        cache="ram",
        plots=True,
        verbose=True,
    )
    print(f"Best weights at: runs/detect/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
