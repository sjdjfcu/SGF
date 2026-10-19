#!/usr/bin/env python3
"""Decompose the per-episode action budget of a finished run.

Usage: measure_action_budget.py <batch-run-dir> [<batch-run-dir> ...]

Reports translating, sweep, blocked-forward and turn-reversal shares.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

FORWARD_ACTION = 1
TURN_ACTIONS = (2, 3)
SWEEP_SOURCE = "panorama_scan"


def episode_stats(episode_dir: str) -> dict | None:
    try:
        result = json.load(open(os.path.join(episode_dir, "result.json")))
        steps = json.load(open(os.path.join(episode_dir, "trace.json")))
    except Exception:
        return None
    metrics = result.get("metrics", {}) or {}
    actions = [int(s.get("habitat_action", -1)) for s in steps]
    sources = [s.get("action_source") or s.get("nav_state") or "unknown" for s in steps]
    reversals: Counter[str] = Counter()
    blocked = 0
    translating = 0
    path = 0.0
    for index, step in enumerate(steps):
        before = step.get("before_position_m")
        after = step.get("after_position_m")
        moved = (
            math.dist((before[0], before[2]), (after[0], after[2]))
            if before and after
            else 0.0
        )
        path += moved
        if actions[index] == FORWARD_ACTION and moved <= 1e-6:
            blocked += 1
        elif moved > 1e-6:
            translating += 1
        if (
            index > 0
            and actions[index] in TURN_ACTIONS
            and actions[index - 1] in TURN_ACTIONS
            and actions[index] != actions[index - 1]
            and sources[index] != SWEEP_SOURCE
        ):
            reversals[sources[index]] += 1
    return {
        "episode": int(result.get("episode_id", -1)),
        "success": float(metrics.get("success", 0.0)),
        "spl": float(metrics.get("spl", 0.0)),
        "steps": len(steps),
        "path_m": path,
        "translating": translating,
        "blocked": blocked,
        "sweep": sum(1 for source in sources if source == SWEEP_SOURCE),
        "reversals": dict(reversals),
    }


def summarize(label: str, rows: list[dict]) -> None:
    if not rows:
        return
    total = sum(r["steps"] for r in rows)
    reversals: Counter[str] = Counter()
    for row in rows:
        reversals.update(row["reversals"])
    reversal_total = sum(reversals.values())
    print(
        f"{label:>10}  n={len(rows):4d}  actions={total:7d}  "
        f"translating={sum(r['translating'] for r in rows)/total:.3f}  "
        f"sweep={sum(r['sweep'] for r in rows)/total:.3f}  "
        f"blocked_forward={sum(r['blocked'] for r in rows)/total:.3f}  "
        f"turn_reversals={reversal_total/total:.3f}"
    )
    for source, count in reversals.most_common(5):
        print(
            f"{'':>12}reversals via {source:<22s} {count:7d}"
            f"  {count/max(1, reversal_total):.3f} of reversals"
        )


def main() -> int:
    run_dirs = sys.argv[1:]
    if not run_dirs:
        print(__doc__)
        return 2
    for run_dir in run_dirs:
        episodes_dir = os.path.join(run_dir, "episodes")
        if not os.path.isdir(episodes_dir):
            print(f"{run_dir}: no episodes/ directory")
            continue
        paths = [
            os.path.join(episodes_dir, name) for name in sorted(os.listdir(episodes_dir))
        ]
        with ProcessPoolExecutor(max_workers=min(32, (os.cpu_count() or 8))) as pool:
            rows = [r for r in pool.map(episode_stats, paths, chunksize=8) if r]
        print(f"\n=== {run_dir}  ({len(rows)} episodes with a trace)")
        summarize("all", rows)
        summarize("success", [r for r in rows if r["success"] > 0.5])
        summarize("failure", [r for r in rows if r["success"] <= 0.5])
        summarize("budget-bound", [r for r in rows if r["steps"] >= 800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
