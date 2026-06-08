"""Render per-player shot heat maps on a half-court diagram.

Reads runs/shots.csv (court_x, court_y, result, jersey) and produces:
  runs/heatmaps/jersey_<N>.png   per identified player
  runs/heatmaps/all.png          team-wide (all shots)

Court frame convention (matches src/court_homography.py):
  origin at center of the baseline under the hoop
  x ∈ [-7.5, 7.5] m  (sideline-to-sideline)
  y ∈ [0, 14.0] m    (baseline-to-half-court)
  hoop center = (0.0, 1.575)
  free-throw line at y = 5.8, width 2 * 2.45 = 4.9
  3-point arc radius ≈ 6.75 m from hoop, with corners at y < 0.9 m
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COURT_W = 15.0    # sideline-to-sideline
HALF_L = 14.0     # baseline-to-half-court
HOOP_XY = (0.0, 1.575)
THREE_RADIUS = 6.75
CORNER_3_X = 6.75   # corner threes are at the painted line, ~0.9 m inside the sideline
KEY_W = 4.9         # width of the painted lane (FIBA)
KEY_LEN = 5.80      # from baseline to free-throw line
FT_RADIUS = 1.80    # free-throw circle radius


def draw_court(ax: plt.Axes) -> None:
    # Court boundary (half court)
    ax.add_patch(mp.Rectangle((-COURT_W / 2, 0), COURT_W, HALF_L,
                              fill=False, lw=2, ec="black"))
    # Paint / key
    ax.add_patch(mp.Rectangle((-KEY_W / 2, 0), KEY_W, KEY_LEN,
                              fill=False, lw=1.5, ec="black"))
    # Free-throw circle
    ax.add_patch(mp.Circle((0, KEY_LEN), FT_RADIUS, fill=False, lw=1.5, ec="black"))
    # Backboard
    ax.plot([-0.9, 0.9], [1.20, 1.20], color="black", lw=2)
    # Hoop
    ax.add_patch(mp.Circle(HOOP_XY, 0.225, fill=False, lw=1.5, ec="orange"))

    # 3-point line: corners (straight) + arc
    hx, hy = HOOP_XY
    # corner segments
    corner_y_end = hy + math.sqrt(max(THREE_RADIUS**2 - CORNER_3_X**2, 0.0))
    ax.plot([-CORNER_3_X, -CORNER_3_X], [0, corner_y_end], color="black", lw=1.5)
    ax.plot([ CORNER_3_X,  CORNER_3_X], [0, corner_y_end], color="black", lw=1.5)
    # arc
    theta = np.linspace(0, np.pi, 200)
    arc_x = hx + THREE_RADIUS * np.cos(theta)
    arc_y = hy + THREE_RADIUS * np.sin(theta)
    mask = (np.abs(arc_x) <= CORNER_3_X) & (arc_y >= corner_y_end - 0.01)
    ax.plot(arc_x[mask], arc_y[mask], color="black", lw=1.5)

    ax.set_xlim(-COURT_W / 2 - 0.5, COURT_W / 2 + 0.5)
    ax.set_ylim(-0.5, HALF_L + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def render_heatmap(shots: pd.DataFrame, title: str, out_path: Path,
                   show_hexbin: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    draw_court(ax)

    pts = shots.dropna(subset=["court_x", "court_y"])
    if not pts.empty and show_hexbin:
        hb = ax.hexbin(pts["court_x"], pts["court_y"], gridsize=16,
                       extent=(-COURT_W / 2, COURT_W / 2, 0, HALF_L),
                       mincnt=1, cmap="Reds", alpha=0.55)
        cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("shots")

    made = pts[pts["result"] == "MADE"]
    miss = pts[pts["result"] == "MISSED"]
    ax.scatter(made["court_x"], made["court_y"], marker="o", s=44,
               facecolors="none", edgecolors="green", lw=1.5, label=f"MADE ({len(made)})")
    ax.scatter(miss["court_x"], miss["court_y"], marker="x", s=44,
               color="red", lw=1.5, label=f"MISSED ({len(miss)})")
    ax.legend(loc="upper right")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("runs/heatmaps"))
    args = p.parse_args()

    df = pd.read_csv(args.shots)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    render_heatmap(df, "All shots — full team", args.out_dir / "all.png")

    by_jersey = df[df["jersey"] >= 0].groupby("jersey")
    for jersey, sub in by_jersey:
        render_heatmap(sub, f"Jersey #{jersey} — {len(sub)} shots",
                       args.out_dir / f"jersey_{jersey}.png")
    print(f"Wrote {1 + by_jersey.ngroups} heatmap PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
