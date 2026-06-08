# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Amateur Basketball Player Personal Data AI Intelligent Analysis System** — an AI platform that processes amateur basketball game footage to generate personalized, data-driven player cards and analytics. Input: raw video. Output: per-player stats, shot heat maps, rebound analysis, highlight clips, and growth curves.

This repository is in its **initial setup phase**. Currently only two reference assets exist:
- `Amateur-Basketball-Player-Personal-Data-AI-Intelligent-Analysis-System.pptx` — the full PRD and requirements spec; consult this as the source of truth for scope and metrics
- `video.mp4` — sample basketball footage for development and testing

## Environment

Python 3.12 virtual environment at `.venv/`. Activate with:
```bash
source .venv/bin/activate
```

Currently installed: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `pillow`. Deep learning / computer vision libraries (OpenCV, PyTorch, YOLO, etc.) are **not yet installed** and will need to be added as modules are built.

## Architecture & Module Plan

The system has four functional layers, each building on the previous:

### 1. Court Modeling & Object Detection
Calibrate spatial reference frame: detect hoop, three-point line, and backboard. All downstream coordinate tracking depends on this calibration.

### 2. Player Tracking & Identity
- **Jersey number OCR** — primary player identification mechanism
- **Facial recognition** — secondary/fallback identification
- Every stat must be attributed to a specific player via their jersey number

### 3. Behavior Recognition
- **Shooting**: detect shot events → record XY court coordinates → classify result (net-swish vs. backboard bounce) via ball trajectory tracking
- **Rebounding**: predict ball landing zone → classify rush behavior as offensive rebound (ORB) or defensive rebound (DRB)

### 4. Output Generation
- **Automatic video editing**: clip highlights (shot clips, rebound review) using data timestamps
- **Shot heat map**: XY court zone overlay showing hot/cold zones per player
- **Growth curves**: per-game trends for shooting and rebounding over time
- **Dashboard**: Shot Distribution Map, Effective Field Goal %, Rebound Control Rate, Shot Timing Analysis (Catch & Shoot vs. Off the Dribble)

## Key Metrics (from PRD)

| Metric | Definition |
|--------|------------|
| Shot Distribution Map | Three-point / mid-range / under-basket proportion, court zone overlay |
| Effective Field Goal % | Accounts for added value of 3PT shots vs. 2PT |
| Rebound Control Rate | Individual rebounds (ORB + DRB) / total available rebounds |
| Shot Timing Analysis | Efficiency delta between Catch & Shoot and Off the Dribble |
