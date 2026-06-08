"""Fetch a basketball detection dataset from Roboflow Universe.

Reads ROBOFLOW_API_KEY from the environment. We never persist the key.

Usage:
  ROBOFLOW_API_KEY=... python -m src.fetch_dataset \
      --workspace roboflow-universe-projects \
      --project basketball-players-fy4c2 \
      --version 12 \
      --out datasets/basketball
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    p.add_argument("--project", required=True, help="Roboflow project slug")
    p.add_argument("--version", type=int, required=True, help="Dataset version number")
    p.add_argument("--format", default="yolov8", help="Export format (yolov8 works for YOLO11)")
    p.add_argument("--out", type=Path, default=Path("datasets/basketball"))
    args = p.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY not set in environment")

    from roboflow import Roboflow  # imported here so script can be inspected without dep

    args.out.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    version = project.version(args.version)
    print(f"Downloading {args.workspace}/{args.project} v{args.version} as {args.format} -> {args.out}")
    dataset = version.download(args.format, location=str(args.out))
    print(f"Dataset at: {dataset.location}")
    # Print class names from data.yaml
    yaml_path = Path(dataset.location) / "data.yaml"
    if yaml_path.exists():
        print("--- data.yaml ---")
        print(yaml_path.read_text())


if __name__ == "__main__":
    main()
