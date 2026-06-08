"""Assemble the self-contained HTML results dashboard (720p / team-level).

Single file, all images base64-embedded. Shows only what the pipeline supports
*honestly*:
  * KPIs (footage, detections, jerseys seen, made shots)
  * Game-flow chart: score progression with every made shot marked
  * Team shot mix: inside-arc (1pt) vs outside-arc (2pt) makes per team
  * Made-shots log (frame / clock / team / value)
  * Jersey-OCR roster with confidence tags
  * An explicit limitations section (panning camera -> no court-space map / no
    per-player shot attribution; OCR resolution cap)

Charts are generated with matplotlib in a matching dark theme.

Usage:  python -m src.build_dashboard --runs runs720
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle
import numpy as np
import pandas as pd

# palette (kept in sync with the page CSS)
BG, PANEL, INK, MUT = "#0f1115", "#181b22", "#e8eaed", "#9aa0aa"
HOME_C, AWAY_C = "#ff5db1", "#4da3ff"          # 風格整形 / 台寧天行者 kit colours
TEAMS = {"home": "風格整形 (Home)", "away": "台寧天行者 (Away)"}
FPS = 30.0
BASELINE = {"home": 3, "away": 3}              # score already on the board at footage start


def img_uri(path: Path) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{b64}"


def confidence(votes: float) -> tuple[str, str]:
    if votes >= 60:
        return "high", "#1db954"
    if votes >= 15:
        return "medium", "#e0a800"
    return "low", "#d9534f"


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color("#2a2f3a")
    ax.tick_params(colors=MUT, labelsize=9)
    ax.yaxis.label.set_color(MUT)
    ax.xaxis.label.set_color(MUT)
    ax.grid(True, color="#23262e", linewidth=0.8)


def chart_gameflow(tl: pd.DataFrame, made: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=150)
    fig.patch.set_facecolor(BG)
    t = tl["frame"] / FPS / 60.0          # minutes of footage
    # ASCII labels only — matplotlib's default font has no CJK glyphs (the HTML does)
    ax.step(t, tl["home"], where="post", color=HOME_C, lw=2.2, label="Home")
    ax.step(t, tl["away"], where="post", color=AWAY_C, lw=2.2, label="Away")
    for m in made.itertuples():
        c = HOME_C if m.team == "home" else AWAY_C
        y = m.home if m.team == "home" else m.away
        ax.scatter(m.frame / FPS / 60.0, y, s=90 if m.points == 2 else 42,
                   color=c, edgecolor="white", linewidth=0.8, zorder=5)
    ax.set_xlabel("footage time (min)")
    ax.set_ylabel("score")
    _style_ax(ax)
    leg = ax.legend(facecolor=PANEL, edgecolor="#2a2f3a", labelcolor=INK, fontsize=9, loc="upper left")
    ax.set_title("Score progression — large dots = 2-pointers, small = 1-pointers",
                 color=INK, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def chart_shotmix(made: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=150)
    fig.patch.set_facecolor(BG)
    teams = ["home", "away"]
    inside = [len(made[(made.team == t) & (made.points == 1)]) for t in teams]
    outside = [len(made[(made.team == t) & (made.points == 2)]) for t in teams]
    labels = ["Home", "Away"]
    x = range(len(teams))
    ax.bar(x, inside, color="#3a6ea5", label="inside arc (1pt)")
    ax.bar(x, outside, bottom=inside, color="#e0a800", label="outside arc (2pt)")
    for i, (ins, out_) in enumerate(zip(inside, outside)):
        ax.text(i, ins / 2, str(ins), ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        if out_:
            ax.text(i, ins + out_ / 2, str(out_), ha="center", va="center", color="#1a1500", fontsize=11, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("made shots")
    _style_ax(ax)
    ax.legend(facecolor=PANEL, edgecolor="#2a2f3a", labelcolor=INK, fontsize=8.5, loc="upper right")
    ax.set_title("Made-shot mix by team", color=INK, fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def _draw_halfcourt(ax, line="#5a6472"):
    """FIBA 3x3 half-court schematic. Origin: baseline center, y up the court (metres)."""
    W, L = 15.0, 11.0                 # half-court footprint
    basket = (0.0, 1.6)              # rim centre ~1.6 m off the baseline
    arc_r = 6.75                      # 3x3 two-point arc radius
    ax.add_patch(Rectangle((-W / 2, 0), W, L, fill=False, ec=line, lw=1.6))
    ax.plot([-0.9, 0.9], [1.2, 1.2], color=line, lw=2.2)            # backboard
    ax.add_patch(Circle(basket, 0.225, fill=False, ec=line, lw=1.4))  # rim
    ax.add_patch(Rectangle((-2.45, 0), 4.9, 5.8, fill=False, ec=line, lw=1.4))  # paint
    ax.add_patch(Arc((0, 5.8), 3.6, 3.6, theta1=0, theta2=180, ec=line, lw=1.4))  # FT circle
    th = np.linspace(0, np.pi, 240)                                  # 2pt arc
    ax.plot(basket[0] + arc_r * np.cos(th), basket[1] + arc_r * np.sin(th), color=line, lw=1.8)
    ax.plot([arc_r, arc_r], [0, basket[1]], color=line, lw=1.8)      # corner segments
    ax.plot([-arc_r, -arc_r], [0, basket[1]], color=line, lw=1.8)
    ax.set_xlim(-7.9, 7.9)
    ax.set_ylim(-0.5, 11.3)
    ax.set_aspect("equal")
    ax.axis("off")


def chart_shotchart(made: pd.DataFrame, out: Path) -> None:
    """Schematic shot-distribution map. Zone (inside/outside arc) is real scoreboard
    data; exact (x,y) within the zone is illustrative — labelled as such on the page."""
    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    _draw_halfcourt(ax)
    basket = np.array([0.0, 1.6])
    rng = np.random.default_rng(7)
    for m in made.itertuples():
        c = HOME_C if m.team == "home" else AWAY_C
        if m.points == 1:                       # inside the arc
            r = rng.uniform(0.9, 6.0); ang = rng.uniform(0.18 * np.pi, 0.82 * np.pi)
        else:                                   # outside the arc
            r = rng.uniform(6.95, 7.7); ang = rng.uniform(0.22 * np.pi, 0.78 * np.pi)
        x = basket[0] + r * np.cos(ang)
        y = max(0.5, basket[1] + r * np.sin(ang))
        x = float(np.clip(x, -7.3, 7.3)); y = float(np.clip(y, 0.5, 10.8))
        ax.scatter(x, y, s=210 if m.points == 2 else 150, color=c,
                   edgecolor="white", linewidth=1.3, zorder=6, alpha=0.95)
        ax.text(x, y, "2" if m.points == 2 else "1", ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold", zorder=7)
    # legend proxies
    ax.scatter([], [], s=150, color=HOME_C, edgecolor="white", label="Home make")
    ax.scatter([], [], s=150, color=AWAY_C, edgecolor="white", label="Away make")
    ax.legend(facecolor=PANEL, edgecolor="#2a2f3a", labelcolor=INK, fontsize=9,
              loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Made-shot distribution — inside vs outside the arc (from scoreboard)",
                 color=INK, fontsize=11, pad=22)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs720"))
    args = p.parse_args()
    R = args.runs

    det = pd.read_csv(R / "detections.csv")
    jdf = pd.read_csv(R / "jersey_assignments.csv")
    made = pd.read_csv(R / "made_shots.csv")
    tl = pd.read_csv(R / "score_timeline.csv")
    seg = pd.read_csv(R / "camera_segments.csv")

    n_main = int(seg.loc[seg.label == "main", "n_frames"].sum())
    total_frames = 10296
    n_person = int((det["class_name"] == "person").sum())
    n_basket = int((det["class_name"] == "basket").sum())

    # roster: aggregate per jersey (per-jersey, never per fragmented track)
    support = jdf.groupby("jersey")["weighted_score"].sum().sort_values(ascending=False)
    roster = support[support >= 2]

    # team scoring summary
    def team_line(t):
        mk = made[made.team == t]
        return len(mk), int(mk.points.sum()), int((mk.points == 1).sum()), int((mk.points == 2).sum())
    h = team_line("home"); a = team_line("away")
    final_home = BASELINE["home"] + h[1]
    final_away = BASELINE["away"] + a[1]

    # charts
    chart_gameflow(tl, made, R / "chart_gameflow.png")
    chart_shotmix(made, R / "chart_shotmix.png")
    chart_shotchart(made, R / "chart_shotchart.png")
    flow_uri = img_uri(R / "chart_gameflow.png")
    mix_uri = img_uri(R / "chart_shotmix.png")
    shot_uri = img_uri(R / "chart_shotchart.png")

    # interactive player cards — thumbnails + stats produced by src.output.player_cards720.
    # Each tile shows photo + number + confidence at a glance; clicking opens a modal with
    # the full stats (stats live in the JS CARDS object; the photo is read from the tile img).
    cards_dir = R / "cards"
    cards_json = cards_dir / "cards.json"
    pcards = json.loads(cards_json.read_text()) if cards_json.exists() else []
    tile_html = ""
    js_cards: dict[str, dict] = {}
    for c in pcards:
        j = str(c["jersey"])
        thumb_uri = img_uri(cards_dir / c["thumb"]) if c.get("thumb") else ""
        photo = (f'<img src="{thumb_uri}"/>' if thumb_uri
                 else '<div class="pcard-noimg">no crop</div>')
        tile_html += (
            f'<button class="pcard" id="pcard-{j}" onclick="openCard(\'{j}\')">'
            f'<div class="pcard-photo">{photo}'
            f'<span class="pcard-chip" style="background:{c["tier_color"]}">{c["tier"]}</span></div>'
            f'<div class="pcard-foot"><span class="pcard-num">#{j}</span>'
            f'<span class="pcard-cta">View stats &rarr;</span></div></button>'
        )
        js_cards[j] = {
            "num": f"#{j}", "tier": c["tier"], "color": c["tier_color"], "roadmap": False,
            "stats": [
                ["Court presence", f"{c['seconds']} s", "tracked on the main camera"],
                ["Confident OCR reads", f"{c['votes']}", "frames where the number was read"],
                ["OCR max confidence", f"{c['maxconf']:.2f}", "peak per-read certainty"],
                ["Tracks merged", f"{c['tracks']}", "fragments stitched into this identity"],
            ],
            "foot": "",
        }
    # one clearly-labelled roadmap card: what higher-res + a fixed court camera would unlock
    tile_html += (
        '<button class="pcard roadmap" id="pcard-roadmap" onclick="openCard(\'roadmap\')">'
        '<div class="pcard-photo"><div class="pcard-roadmap-art">&#128202;<span>Full card</span></div>'
        '<span class="pcard-chip" style="background:#1db954">ROADMAP</span></div>'
        '<div class="pcard-foot"><span class="pcard-num" style="font-size:15px">What\'s next</span>'
        '<span class="pcard-cta">Preview &rarr;</span></div></button>'
    )
    js_cards["roadmap"] = {
        "num": "Full player card", "tier": "ROADMAP", "color": "#1db954", "roadmap": True,
        "stats": [
            ["Points / game", "8.5", ""],
            ["Effective FG%", "61%", ""],
            ["Shot mix (in / mid / 3)", "40 / 35 / 25", ""],
            ["Rebounds (ORB / DRB)", "2 / 4", ""],
            ["Catch & Shoot vs Dribble", "58% / 41%", ""],
            ["Rebound control rate", "0.27", ""],
        ],
        "foot": "Illustrative values. Needs two things this footage lacks: a higher-resolution "
                "source (for reliable jersey numbers) and a fixed, calibrated court camera "
                "(for true shot positions & attribution).",
    }
    cards_js = json.dumps(js_cards)

    # KPI cards
    kpis = [
        ("Footage analyzed", f"{n_main:,} frames", f"{100*n_main/total_frames:.0f}% main camera · 720p"),
        ("Made shots detected", f"{len(made)}", f"{h[3]+a[3]}× 2pt · {h[2]+a[2]}× 1pt"),
        ("Final score (OCR)", f"{final_home}–{final_away}", "incl. 3–3 pre-footage baseline"),
        ("Jersey numbers seen", f"{len(roster)}", "by OCR (confidence-tagged)"),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="kpi-val">{v}</div>'
        f'<div class="kpi-lbl">{lbl}</div><div class="kpi-sub">{sub}</div></div>'
        for lbl, v, sub in kpis
    )

    # team scoring table
    team_rows = ""
    for t, (mk, pts, ins, outs) in (("home", h), ("away", a)):
        c = HOME_C if t == "home" else AWAY_C
        share = f"{100*outs/mk:.0f}%" if mk else "—"
        team_rows += (
            f"<tr><td><span class='dot' style='background:{c}'></span>{TEAMS[t]}</td>"
            f"<td>{mk}</td><td>{ins}</td><td>{outs}</td><td>{pts}</td><td>{share}</td></tr>"
        )

    # made-shots log
    log_rows = ""
    for m in made.itertuples():
        c = HOME_C if m.team == "home" else AWAY_C
        val = "2pt (outside arc)" if m.points == 2 else "1pt (inside arc)"
        clock = m.clock if isinstance(m.clock, str) else "—"
        log_rows += (
            f"<tr><td>{clock}</td><td><span class='dot' style='background:{c}'></span>{TEAMS[m.team].split(' ')[0]}</td>"
            f"<td>{val}</td><td>{m.home}–{m.away}</td></tr>"
        )

    # roster table
    ros_rows = ""
    for jersey, score in roster.items():
        n_tracks = jdf.loc[jdf["jersey"] == jersey, "track_id"].nunique()
        max_conf = jdf.loc[jdf["jersey"] == jersey, "top_conf"].max()
        tag, color = confidence(float(score))
        ros_rows += (
            f"<tr><td class='jno'>#{int(jersey)}</td><td>{n_tracks}</td>"
            f"<td>{score:.0f}</td><td>{max_conf:.2f}</td>"
            f"<td><span class='pill' style='background:{color}'>{tag}</span></td></tr>"
        )

    # highlights tab — built from the manifest written by src.output.highlights.
    # Thumbnails are embedded (self-contained); clicking a card opens the real clip
    # via a relative path (works when the dashboard is opened from inside the run dir).
    hl_dir = R / "highlights"
    hl_manifest = hl_dir / "manifest.csv"
    hl_tab_btn = ""
    hl_section = ""
    if hl_manifest.exists():
        hlm = pd.read_csv(hl_manifest)
        hl_tab_btn = f"<button class=\"tab-btn\" onclick=\"showTab('highlights',this)\">Highlights ({len(hlm)})</button>"
        reel_rel = "highlights/highlights_reel.mp4"
        reel_btn = (f'<a class="reel-btn" href="{reel_rel}">▶ Play full reel '
                    f'({len(hlm)} clips)</a>') if (hl_dir / "highlights_reel.mp4").exists() else ""
        cards = ""
        for m in hlm.itertuples():
            c = HOME_C if m.team == "home" else AWAY_C
            label = f"{m.points}PT" + (f" · {m.clock}" if isinstance(m.clock, str) and m.clock else "")
            team_short = TEAMS[m.team].split(" ")[0]
            thumb = img_uri(R / m.thumb)
            cards += (
                f'<a class="hl-card" href="{m.file}" title="Shot {m.idx} — {team_short} {label}">'
                f'<div class="hl-thumb"><img src="{thumb}"/>'
                f'<span class="hl-play">▶</span>'
                f'<span class="hl-badge" style="background:{c}">{m.points}PT</span></div>'
                f'<div class="hl-meta"><span class="hl-num">#{m.idx:02d}</span>'
                f'<span class="hl-team"><span class="dot" style="background:{c}"></span>{team_short}</span>'
                f'<span class="hl-clock">{m.clock if isinstance(m.clock,str) and m.clock else "—"}</span>'
                f'<span class="hl-score">{m.home}–{m.away}</span></div></a>'
            )
        hl_section = f"""
<section id="highlights" class="tab-panel">
<p class="sub" style="margin-top:6px">One clip per made shot, cut straight from the broadcast using the
scoreboard timestamps · <b>6&nbsp;s before → 3&nbsp;s after</b> each bucket, with a burnt-in caption</p>
<div class="hl-bar">{reel_btn}<span class="hl-hint">Click any thumbnail to play that clip</span></div>
<div class="hl-grid">{cards}</div>
<div class="note" style="margin-top:18px"><b>Note:</b> clips and the full reel live in
<code>{hl_dir.name}/</code> next to this file — open the dashboard from inside the run folder so the
links resolve. Each clip is centred on a real scoreboard increment, the most reliable shot signal
on this panning-camera footage.</div>
</section>"""

    css = """
    :root{--bg:#0b0d12;--panel:#161922;--ink:#eef0f4;--mut:#969cab;--acc:#22c55e;
      --line:#23262e;--radius:16px;}
    *{box-sizing:border-box}body{margin:0;color:var(--ink);line-height:1.55;
      font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      background:
        radial-gradient(1100px 520px at 78% -8%,rgba(34,197,94,.10),transparent 60%),
        radial-gradient(900px 480px at 10% -4%,rgba(77,163,255,.10),transparent 58%),
        var(--bg);
      background-attachment:fixed;-webkit-font-smoothing:antialiased}
    .wrap{max-width:1100px;margin:0 auto;padding:32px 28px 56px}
    h1{font-size:27px;margin:0 0 2px;letter-spacing:-.01em}
    h2{font-size:16px;margin:34px 0 12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mut)}
    .sub{color:var(--mut);margin:0 0 24px;font-size:14px}
    .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}
    .kpi{background:linear-gradient(160deg,#1c2029,#15181f);border:1px solid var(--line);
      border-radius:var(--radius);padding:18px 18px 16px;position:relative;overflow:hidden;
      transition:transform .15s,border-color .15s}
    .kpi::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
      background:linear-gradient(90deg,var(--acc),transparent 70%);opacity:.7}
    .kpi:hover{transform:translateY(-2px);border-color:#333a47}
    .kpi-val{font-size:26px;font-weight:800;letter-spacing:-.02em}.kpi-lbl{font-size:13px;margin-top:6px}
    .kpi-sub{color:var(--mut);font-size:11px;margin-top:3px}
    .cols{display:grid;grid-template-columns:1.7fr 1fr;gap:18px;align-items:start}
    /* interactive player cards */
    .pgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(184px,1fr));gap:14px}
    .pcard{position:relative;padding:0;border:1px solid var(--line);border-radius:16px;
      overflow:hidden;background:var(--panel);cursor:pointer;text-align:left;color:var(--ink);
      font:inherit;transition:transform .16s ease,border-color .16s,box-shadow .16s}
    .pcard:hover{transform:translateY(-4px);border-color:#39414f;
      box-shadow:0 16px 34px -16px rgba(0,0,0,.75)}
    .pcard:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
    .pcard-photo{position:relative;aspect-ratio:4/5;background:#0c0e13;overflow:hidden}
    .pcard-photo img{width:100%;height:100%;object-fit:cover;object-position:center 22%;
      display:block;transition:transform .3s ease}
    .pcard:hover .pcard-photo img{transform:scale(1.05)}
    .pcard-noimg{display:flex;align-items:center;justify-content:center;height:100%;
      color:var(--mut);font-size:12px}
    .pcard-chip{position:absolute;top:9px;right:9px;color:#06210f;font-size:9.5px;font-weight:800;
      letter-spacing:.05em;padding:2px 8px;border-radius:20px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
    .pcard-foot{display:flex;align-items:center;justify-content:space-between;padding:9px 12px;
      background:linear-gradient(180deg,#1b1f29,#14171e)}
    .pcard-num{font-size:18px;font-weight:850;letter-spacing:-.02em}
    .pcard-cta{font-size:11px;color:var(--mut);font-weight:600}
    .pcard:hover .pcard-cta{color:var(--acc)}
    .pcard.roadmap{border-color:#2f6b3f;background:linear-gradient(165deg,#15211a,#10160f)}
    .pcard-roadmap-art{display:flex;flex-direction:column;align-items:center;justify-content:center;
      gap:7px;height:100%;font-size:36px;color:#6fcf97}
    .pcard-roadmap-art span{font-size:12px;font-weight:700;letter-spacing:.03em;color:#9bd9b3}
    /* modal */
    .modal{position:fixed;inset:0;z-index:50;display:none;align-items:center;justify-content:center;
      background:rgba(5,7,11,.66);backdrop-filter:blur(7px);-webkit-backdrop-filter:blur(7px);padding:24px}
    .modal.open{display:flex;animation:mfade .16s ease}
    @keyframes mfade{from{opacity:0}to{opacity:1}}
    .modal-card{width:min(420px,100%);max-height:90vh;overflow:auto;background:var(--panel);
      border:1px solid #2a2f3a;border-radius:22px;position:relative;
      box-shadow:0 36px 90px -24px rgba(0,0,0,.85);animation:mpop .2s cubic-bezier(.2,.8,.3,1)}
    @keyframes mpop{from{transform:translateY(16px) scale(.97);opacity:0}to{transform:none;opacity:1}}
    .modal-x{position:absolute;top:12px;right:12px;z-index:2;width:34px;height:34px;border-radius:50%;
      border:none;background:rgba(0,0,0,.45);color:#fff;font-size:21px;line-height:1;cursor:pointer}
    .modal-x:hover{background:rgba(0,0,0,.72)}
    .modal-hero{position:relative;aspect-ratio:16/11;background:#0c0e13;overflow:hidden}
    .modal-hero img{width:100%;height:100%;object-fit:cover;display:block}
    .modal-hero-grad{position:absolute;inset:0;
      background:linear-gradient(180deg,rgba(0,0,0,0) 38%,rgba(15,12,18,.92))}
    .modal-hero-label{display:none}
    .modal.roadmap .modal-hero{background:linear-gradient(160deg,#1b3326,#10160f)}
    .modal.roadmap .modal-hero-label{display:flex;position:absolute;inset:0;z-index:1;
      align-items:center;justify-content:center;color:#bfe8c9;font-weight:800;font-size:17px}
    .modal-hero-head{position:absolute;left:18px;right:18px;bottom:14px;z-index:1;display:flex;
      align-items:center;justify-content:space-between;gap:10px}
    .modal-num{font-size:30px;font-weight:850;letter-spacing:-.02em;text-shadow:0 2px 10px rgba(0,0,0,.6)}
    .modal-body{padding:8px 20px 4px}
    .srow{padding:12px 0;border-bottom:1px solid var(--line)}
    .srow:last-child{border-bottom:none}
    .srow-top{display:flex;align-items:baseline;justify-content:space-between;gap:12px}
    .srow-lbl{color:var(--mut);font-size:13px}
    .srow-val{font-size:18px;font-weight:800;letter-spacing:-.01em}
    .srow-sub{color:#6b7280;font-size:11px;margin-top:2px}
    .modal-foot{padding:6px 20px 22px;color:#7fae8c;font-size:12px;font-style:italic;line-height:1.5}
    .modal-foot:empty{display:none}
    .pchip{padding:3px 11px;border-radius:20px;color:#06210f;font-size:11px;font-weight:800;
      letter-spacing:.04em;text-transform:uppercase}
    .panel{background:var(--panel);border:1px solid #23262e;border-radius:14px;padding:14px}
    .panel img{width:100%;display:block;border-radius:8px}
    table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid #23262e;border-radius:14px;overflow:hidden}
    th,td{padding:10px 14px;text-align:left;font-size:14px;border-bottom:1px solid #23262e}
    tr:last-child td{border-bottom:none}
    th{color:var(--mut);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
    .jno{font-weight:750}
    .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:8px;vertical-align:middle}
    .pill{padding:2px 10px;border-radius:20px;color:#06210f;font-size:11px;font-weight:700;text-transform:uppercase}
    .note{background:#15212b;border:1px solid #1f3a4a;border-radius:12px;padding:14px 18px;font-size:13px;color:#9fd0e6;margin:8px 0 0}
    .note b{color:#cfeeff}
    .limits{background:#241a12;border:1px solid #5a3a1a;border-radius:12px;padding:14px 18px;font-size:13px;color:#f0c79a}
    .limits b{color:#ffd9a8}.limits ul{margin:8px 0 0;padding-left:18px}.limits li{margin:3px 0}
    footer{color:var(--mut);font-size:12px;margin-top:34px;border-top:1px solid #23262e;padding-top:16px}
    code{background:#10131a;padding:1px 6px;border-radius:5px;font-size:12px}
    .appbar{position:sticky;top:0;z-index:10;background:rgba(15,17,21,.93);
      backdrop-filter:blur(8px);border-bottom:1px solid #23262e}
    .appbar-inner{max-width:1100px;margin:0 auto;padding:13px 28px 0;display:flex;
      align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
    .brand{font-size:18px;font-weight:800;letter-spacing:-.01em;white-space:nowrap}
    .score{display:flex;align-items:center;gap:10px;font-size:14px}
    .score .s{font-size:22px;font-weight:800}
    .score .home{color:#ff5db1}.score .away{color:#4da3ff}
    .badge{color:var(--mut);font-size:12px;border:1px solid #2a2f3a;border-radius:20px;
      padding:4px 12px;white-space:nowrap}
    .tabs{display:flex;gap:4px;max-width:1100px;margin:6px auto 0;padding:0 22px}
    .tab-btn{background:none;border:none;color:var(--mut);font:inherit;font-size:14px;
      font-weight:600;padding:13px 18px;cursor:pointer;border-bottom:2px solid transparent}
    .tab-btn:hover{color:var(--ink)}
    .tab-btn.active{color:var(--ink);border-bottom-color:var(--acc)}
    .tab-panel{display:none}.tab-panel.active{display:block}
    .hl-bar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:6px 0 18px}
    .reel-btn{background:var(--acc);color:#06210f;font-weight:700;font-size:14px;
      text-decoration:none;padding:10px 18px;border-radius:10px}
    .reel-btn:hover{filter:brightness(1.08)}
    .hl-hint{color:var(--mut);font-size:13px}
    .hl-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
    .hl-card{display:block;text-decoration:none;color:var(--ink);background:var(--panel);
      border:1px solid #23262e;border-radius:14px;overflow:hidden;transition:transform .12s,border-color .12s}
    .hl-card:hover{transform:translateY(-3px);border-color:#3a4150}
    .hl-thumb{position:relative;aspect-ratio:16/9;background:#000;overflow:hidden}
    .hl-thumb img{width:100%;height:100%;object-fit:cover;display:block}
    .hl-play{position:absolute;inset:0;margin:auto;width:52px;height:52px;border-radius:50%;
      background:rgba(0,0,0,.55);color:#fff;font-size:20px;display:flex;align-items:center;
      justify-content:center;padding-left:3px}
    .hl-card:hover .hl-play{background:var(--acc);color:#06210f}
    .hl-badge{position:absolute;top:8px;right:8px;color:#fff;font-size:11px;font-weight:800;
      padding:3px 9px;border-radius:20px;text-shadow:0 1px 2px rgba(0,0,0,.4)}
    .hl-meta{display:flex;align-items:center;gap:10px;padding:10px 12px;font-size:13px}
    .hl-num{font-weight:750;color:var(--mut)}.hl-team{font-weight:600}
    .hl-clock{margin-left:auto;color:var(--mut)}.hl-score{font-weight:700}
    @media(max-width:620px){.pgrid{grid-template-columns:repeat(2,1fr)}
      .kpis{grid-template-columns:repeat(2,1fr)}.cols{grid-template-columns:1fr!important}
      .hl-grid{grid-template-columns:repeat(2,1fr)}}
    """

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Basketball AI — Game Analytics</title><style>{css}</style></head>
<body>
<div class="appbar">
  <div class="appbar-inner">
    <div class="brand">🏀 Basketball AI</div>
    <div class="score">
      <span class="home">風格整形</span>
      <span class="s">{final_home}</span><span style="color:var(--mut)">–</span><span class="s">{final_away}</span>
      <span class="away">台寧天行者</span>
    </div>
    <div class="badge">T3BA 3×3 championship · 720p</div>
  </div>
  <div class="tabs">
    <button class="tab-btn active" onclick="showTab('game',this)">Game Overview</button>
    <button class="tab-btn" onclick="showTab('players',this)">Players</button>
    {hl_tab_btn}
  </div>
</div>

<div class="wrap">
<section id="game" class="tab-panel active">
<p class="sub" style="margin-top:6px">T3BA Taiwan 3×3 championship final · auto-generated from raw broadcast video</p>

<div class="kpis">{kpi_html}</div>

<div class="note"><b>How shots are measured:</b> in 3×3, the scoreboard encodes every make — a
<b>+1</b> is a basket from <b>inside</b> the arc, a <b>+2</b> from <b>outside</b>. We read the on-screen
score every 0.5&nbsp;s (OCR) and reconstruct each made shot with its value and game-clock time. This
reconciles exactly to the final score, so the team shot totals below are solid.</div>

<h2>Game flow</h2>
<div class="panel"><img src="{flow_uri}"/></div>

<h2>Shot distribution map</h2>
<div class="cols" style="grid-template-columns:1.25fr 1fr">
  <div class="panel"><img src="{shot_uri}"/></div>
  <div class="note" style="margin:0;align-self:start">
    <b>How to read this.</b> Each marker is a <b>made</b> shot, coloured by team and labelled
    with its point value. The <b>zone</b> — inside vs outside the two-point arc — is read directly
    from the scoreboard (a <b>+1</b> is an inside make, a <b>+2</b> is an outside make), so the
    inside/outside split and the per-team counts are exact.
    <br><br>
    The <b>exact position within a zone is illustrative</b>: the broadcast camera pans &amp; zooms,
    so we can't recover precise court coordinates from this footage. This delivers the PRD's shot
    distribution map at the fidelity the source supports — honest zones, not invented pin-point xy.
  </div>
</div>

<div class="cols" style="margin-top:18px">
  <div>
    <h2 style="margin-top:0">Team scoring</h2>
    <p class="sub" style="margin:-6px 0 12px">Points <b>scored during the analyzed footage</b> ({h[1]}&ndash;{a[1]}).
    The final score above ({final_home}&ndash;{final_away}) adds the 3&ndash;3 that was already on the
    board when the clip begins (&ldquo;40s&rdquo; in).</p>
    <table><thead><tr><th>Team</th><th>Makes</th><th>1pt</th><th>2pt</th><th>Pts</th><th>2pt share</th></tr></thead>
    <tbody>{team_rows}</tbody></table>
  </div>
  <div>
    <h2 style="margin-top:0">Shot mix</h2>
    <div class="panel"><img src="{mix_uri}"/></div>
  </div>
</div>

<h2>Made-shots log</h2>
<table><thead><tr><th>Clock</th><th>Team</th><th>Shot value</th><th>Score after</th></tr></thead>
<tbody>{log_rows}</tbody></table>

<h2>Honest limitations</h2>
<div class="limits"><b>What this footage does <i>not</i> support reliably:</b>
<ul>
<li><b>Pin-point court coordinates / heat maps</b> — the broadcast camera continuously pans &amp; zooms, so there is no single court-to-image mapping. We report shots by <b>zone</b> (inside / outside the arc, from the scoreboard) rather than exact (x,y); the shot-distribution map above is schematic within each zone.</li>
<li><b>Per-player shot attribution</b> — tying each make to a jersey needs both a clean ball track to the rim and a correct number read; validation showed neither holds (e.g. a courtside photographer got picked once; a #21 shooter was read as #12).</li>
<li><b>Jersey numbers</b> are <b>resolution-limited</b> (numbers ≈20&nbsp;px tall) — treat the roster as “numbers seen”, weighted by confidence, not a definitive lineup.</li>
<li><b>Field-goal % / eFG%</b> need missed attempts; misses aren't detected reliably here, so only <b>made</b>-shot distribution is reported.</li>
</ul></div>
</section>

<section id="players" class="tab-panel">
<p class="sub" style="margin-top:6px">Per-player identity &amp; court presence, attributed by jersey-number OCR + tracking</p>

<h2 style="margin-top:8px">Player cards</h2>
<p class="sub" style="margin:-4px 0 14px">Identity &amp; court-presence we can attribute reliably.
<b>Click a card</b> to reveal its stats. Per-player shot/rebound stats need higher-resolution footage —
the green <b>roadmap</b> card previews the complete card the system is built to produce.</p>
<div class="pgrid">{tile_html}</div>

<h2>Jersey numbers seen (OCR)</h2>
<table><thead><tr><th>Jersey</th><th>Tracks</th><th>OCR score</th><th>Max conf</th><th>Confidence</th></tr></thead>
<tbody>{ros_rows}</tbody></table>
</section>

{hl_section}

<footer>Generated by <code>src/output/build_dashboard.py</code> · shots from <code>scorebug_ocr.py</code> ·
cards from <code>player_cards720.py</code> · detection <code>best.pt</code> (ball/basket/person) ·
jersey OCR via PaddleOCR (GPU).</footer>
</div>

<div class="modal" id="cardModal" onclick="if(event.target===this)closeCard()">
  <div class="modal-card">
    <button class="modal-x" onclick="closeCard()" aria-label="Close">&times;</button>
    <div class="modal-hero">
      <img id="mPhoto" alt="player"/>
      <div id="mHeroLabel" class="modal-hero-label"></div>
      <div class="modal-hero-grad"></div>
      <div class="modal-hero-head">
        <span id="mNum" class="modal-num"></span>
        <span id="mTier" class="pchip"></span>
      </div>
    </div>
    <div id="mBody" class="modal-body"></div>
    <div id="mFoot" class="modal-foot"></div>
  </div>
</div>

<script>
var CARDS = {cards_js};
function showTab(id, btn){{
  document.querySelectorAll('.tab-panel').forEach(function(p){{p.classList.remove('active');}});
  document.querySelectorAll('.tab-btn').forEach(function(b){{b.classList.remove('active');}});
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
  window.scrollTo(0,0);
}}
function openCard(j){{
  var c = CARDS[j]; if(!c) return;
  var modal = document.getElementById('cardModal');
  var photo = document.getElementById('mPhoto');
  var label = document.getElementById('mHeroLabel');
  if(c.roadmap){{
    modal.classList.add('roadmap');
    photo.style.display='none';
    label.textContent='\\uD83D\\uDCC8 ' + c.num;
  }} else {{
    modal.classList.remove('roadmap');
    photo.style.display='';
    var tile = document.getElementById('pcard-'+j);
    var img = tile ? tile.querySelector('img') : null;
    photo.src = img ? img.src : '';
    label.textContent='';
  }}
  document.getElementById('mNum').textContent = c.num;
  var tier = document.getElementById('mTier');
  tier.textContent = c.tier; tier.style.background = c.color;
  document.getElementById('mBody').innerHTML = c.stats.map(function(s){{
    return '<div class="srow"><div class="srow-top"><span class="srow-lbl">'+s[0]+
      '</span><span class="srow-val">'+s[1]+'</span></div>'+
      (s[2] ? '<div class="srow-sub">'+s[2]+'</div>' : '')+'</div>';
  }}).join('');
  document.getElementById('mFoot').innerHTML = c.foot || '';
  modal.classList.add('open');
}}
function closeCard(){{ document.getElementById('cardModal').classList.remove('open'); }}
document.addEventListener('keydown', function(e){{ if(e.key==='Escape') closeCard(); }});
</script>
</body></html>"""

    out = R / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size/1024:.0f} KB, self-contained)")
    print(f"  Final {final_home}-{final_away} · home {h[0]} makes ({h[2]}+{h[3]}) · away {a[0]} makes ({a[2]}+{a[3]})")
    print(f"  Open: file://{out.resolve()}")


if __name__ == "__main__":
    main()
