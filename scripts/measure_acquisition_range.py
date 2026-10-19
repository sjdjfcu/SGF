#!/usr/bin/env python3
"""Measure how often a searching agent acquires the instance, versus distance.

Usage: measure_acquisition_range.py <batch-run-dir> [<batch-run-dir> ...]

Reads recorded episode artifacts only.
"""
from __future__ import annotations

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

BINS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, float("inf"))]


def episode_rows(episode_dir: str) -> list[tuple[float, bool, bool]]:
    try:
        result = json.load(open(os.path.join(episode_dir, "result.json")))
        steps = json.load(open(os.path.join(episode_dir, "trace.json")))
    except Exception:
        return []
    goal = np.asarray(result.get("gt_goal_position_m") or [np.nan] * 3, dtype=float)
    if goal.size != 3 or not np.all(np.isfinite(goal)):
        return []
    rows: list[tuple[float, bool, bool]] = []
    servo_live = False
    for step in steps:
        visual = step.get("visual_stop") or {}
        panorama = step.get("panorama_observation") or {}
        # Count steps the sweep owned even if its action lost arbitration
        in_sweep = bool(
            panorama.get("active") or panorama.get("started") or panorama.get("preempted")
        )
        if visual.get("attempted") and not servo_live:
            position = np.asarray(
                step.get("before_position_m") or [np.nan] * 3, dtype=float
            )
            if np.all(np.isfinite(position)):
                distance = float(np.linalg.norm(goal[[0, 2]] - position[[0, 2]]))
                rows.append((distance, bool(visual.get("target_updated")), in_sweep))
        servo_live = bool((step.get("visual_servo") or {}).get("active"))
    return rows


def turn_actions_per_revolution(run_dir: str) -> int:
    try:
        manifest = json.load(open(os.path.join(run_dir, "manifest.json")))
    except Exception:
        manifest = {}
    turn_angle = float(
        (manifest.get("nav_config") or {}).get("turn_angle", 30.0)
        if isinstance(manifest.get("nav_config"), dict)
        else 30.0
    )
    if not math.isfinite(turn_angle) or turn_angle <= 0.0:
        turn_angle = 30.0
    return max(1, int(round(360.0 / turn_angle)))


def report(run_dir: str) -> None:
    episodes_dir = os.path.join(run_dir, "episodes")
    if not os.path.isdir(episodes_dir):
        print(f"{run_dir}: no episodes/ directory")
        return
    paths = [
        os.path.join(episodes_dir, name) for name in sorted(os.listdir(episodes_dir))
    ]
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 8)) as pool:
        rows = [r for rs in pool.map(episode_rows, paths, chunksize=8) for r in rs]
    if not rows:
        print(f"{run_dir}: no searching frames found")
        return
    distance = np.asarray([r[0] for r in rows])
    acquired = np.asarray([r[1] for r in rows])
    sweep = np.asarray([r[2] for r in rows])
    turn = turn_actions_per_revolution(run_dir)
    print(f"\n=== {run_dir}")
    print(
        f"searching frames {len(rows)} | acquisitions {int(acquired.sum())} | "
        f"sweep share of frames {sweep.mean():.3f} | "
        f"sweep share of acquisitions {acquired[sweep].sum() / max(1, acquired.sum()):.3f}"
    )
    print(
        f"{'true distance':>16} {'n':>8} {'per frame':>11} {'sweep':>9} "
        f"{'non-sweep':>11} {'per sweep':>11}"
    )
    for low, high in BINS:
        mask = (distance >= low) & (distance < high)
        if int(mask.sum()) < 100:
            continue
        sweep_rate = acquired[mask & sweep].mean() if (mask & sweep).sum() else float("nan")
        per_sweep = (
            1.0 - (1.0 - sweep_rate) ** turn if math.isfinite(sweep_rate) else float("nan")
        )
        label = f"{low:4.1f}-{high:5.1f}" if math.isfinite(high) else f"{low:4.1f}+     "
        print(
            f"{label:>16} {int(mask.sum()):8d} {acquired[mask].mean():11.5f} "
            f"{sweep_rate:9.5f} "
            f"{acquired[mask & ~sweep].mean() if (mask & ~sweep).sum() else float('nan'):11.5f} "
            f"{per_sweep:11.3f}"
        )
    accepted = distance[acquired]
    if accepted.size:
        print(
            f"acquisition distance: median {np.median(accepted):.2f} m, "
            f"q25 {np.percentile(accepted, 25):.2f}, q75 {np.percentile(accepted, 75):.2f}"
        )


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for run_dir in sys.argv[1:]:
        report(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
