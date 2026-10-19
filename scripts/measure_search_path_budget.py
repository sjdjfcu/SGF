#!/usr/bin/env python3
"""Decompose per-episode traveled path length by action source.

Usage: measure_search_path_budget.py <batch-run-dir>
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


def episode_stats(row):
    art = row["artifact_dir"]
    tp = os.path.join(art, "trace.json")
    out = {
        "episode_index": int(row["episode_index"]),
        "success": int(float(row["success"])) if row["success"] else 0,
        "spl": float(row["spl"]) if row["spl"] else 0.0,
        "steps": int(float(row["steps"])) if row["steps"] else 0,
        "goal_name": row["goal_name"],
        "scene": os.path.basename(row["scene_id"]).split(".")[0],
        "dist": float(row["distance_to_goal"]) if row["distance_to_goal"] else float("nan"),
        "by_source": {},
        "path": 0.0,
        "n_trans": 0,
        "ok": False,
    }
    if not os.path.exists(tp):
        return out
    try:
        steps = json.load(open(tp))
    except Exception:
        return out
    by = defaultdict(float)
    cnt = defaultdict(int)
    total = 0.0
    for s in steps:
        b = s.get("before_position_m")
        a = s.get("after_position_m")
        src = s.get("action_source") or s.get("nav_state") or "unknown"
        cnt[src] += 1
        if not b or not a:
            continue
        d = math.dist((b[0], b[2]), (a[0], a[2]))
        if d > 1e-6:
            by[src] += d
            total += d
            out["n_trans"] += 1
    out["by_source"] = dict(by)
    out["action_counts"] = dict(cnt)
    out["path"] = total
    out["ok"] = True
    # Shortest path length estimated from SPL
    if out["success"] and out["spl"] > 0:
        out["sp"] = out["spl"] * max(total, 1e-9)
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = sys.argv[1]
    rows = list(csv.DictReader(open(os.path.join(run_dir, "episodes.tsv")), delimiter="\t"))
    with ProcessPoolExecutor(max_workers=32) as ex:
        res = list(ex.map(episode_stats, rows, chunksize=8))
    json.dump(res, open("outputs/diagnostics/path_decompose.json", "w"))
    ok = [r for r in res if r["ok"]]
    print("episodes with trace:", len(ok), "of", len(res))
    succ = [r for r in ok if r["success"]]
    agg = defaultdict(float)
    aggc = defaultdict(int)
    for r in succ:
        for k, v in r["by_source"].items():
            agg[k] += v
        for k, v in r.get("action_counts", {}).items():
            aggc[k] += v
    tot = sum(agg.values())
    print(f"\n=== SUCCESSFUL episodes (n={len(succ)}) path length by action source, total={tot:.1f} m")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"  {k:32s} {v:10.1f} m  {100*v/tot:5.1f}%   actions={aggc.get(k,0)}")
    sp = sum(r.get("sp", 0.0) for r in succ)
    print(f"  shortest-path sum over successes: {sp:.1f} m  (ratio {sp/tot:.3f})")
    print(f"  excess path over successes: {tot-sp:.1f} m")

    fail = [r for r in ok if not r["success"]]
    agg2 = defaultdict(float)
    for r in fail:
        for k, v in r["by_source"].items():
            agg2[k] += v
    tot2 = sum(agg2.values())
    print(f"\n=== FAILED episodes (n={len(fail)}) path by source total={tot2:.1f} m")
    for k, v in sorted(agg2.items(), key=lambda x: -x[1])[:12]:
        print(f"  {k:32s} {v:10.1f} m  {100*v/tot2:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
