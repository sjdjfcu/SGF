#!/usr/bin/env python3
"""Paired factorial and exposure analysis of the representation-coupling suite.

Contrasts are computed per episode before bootstrap resampling.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from compare_nav_runs import (
    _flag,
    _validate_group_metadata,
    index_rows,
    read_rows,
    read_summary,
    require_summary_seed,
    resolve_failure_config,
)


FULL = "full"
DEPTH = "nav_map_depth_only"
EUCLIDEAN = "target_association_euclidean"
DECOUPLED = "decoupled_representation"
RANK = "target_association_reachable_euclidean"
DEPTH_RANK = "depth_only_reachable_euclidean"
CELL_SETTINGS = {
    FULL: (True, "navigation_map"),
    DEPTH: (False, "navigation_map"),
    EUCLIDEAN: (True, "euclidean"),
    DECOUPLED: (False, "euclidean"),
    RANK: (True, "reachable_euclidean"),
    DEPTH_RANK: (False, "reachable_euclidean"),
}
EXPOSURES = {
    "graph_mask_changed": (
        "navigation_graph_removed_free_cells", "navigation_graph_added_occupied_cells",
    ),
    "unreachable_candidate": ("text_association_unreachable_candidates",),
    "geometric_rank_disagreed": ("text_association_rank_disagreements",),
}


def contrast_weights(task: str, ranking_controls: bool) -> dict[str, dict[str, float]]:
    weights = {"occupancy_with_geodesic": {FULL: 1, DEPTH: -1}}
    if task == "instance-image":
        return weights
    weights.update({
        "occupancy_with_euclidean": {EUCLIDEAN: 1, DECOUPLED: -1},
        "geodesic_with_occupancy": {FULL: 1, EUCLIDEAN: -1},
        "geodesic_without_occupancy": {DEPTH: 1, DECOUPLED: -1},
        "occupancy_main": {FULL: .5, DEPTH: -.5, EUCLIDEAN: .5, DECOUPLED: -.5},
        "geodesic_main": {FULL: .5, DEPTH: .5, EUCLIDEAN: -.5, DECOUPLED: -.5},
        "occupancy_x_geodesic": {FULL: 1, DEPTH: -1, EUCLIDEAN: -1, DECOUPLED: 1},
    })
    if ranking_controls:
        weights.update({
            "ranking_with_occupancy": {FULL: 1, RANK: -1},
            "ranking_without_occupancy": {DEPTH: 1, DEPTH_RANK: -1},
            "reachability_with_occupancy": {RANK: 1, EUCLIDEAN: -1},
            "reachability_without_occupancy": {DEPTH_RANK: 1, DECOUPLED: -1},
            "occupancy_with_reachable_euclidean": {RANK: 1, DEPTH_RANK: -1},
            "occupancy_x_ranking": {FULL: 1, RANK: -1, DEPTH: -1, DEPTH_RANK: 1},
            "occupancy_x_reachability": {RANK: 1, EUCLIDEAN: -1, DEPTH_RANK: -1, DECOUPLED: 1},
        })
    return weights


def validate_design(manifest: dict, task: str, names: list[str]) -> None:
    """Require the recorded experiment to differ only in the intended factors."""
    baseline = manifest["resolved_full_config"][task]
    if baseline.get("navigation_graph_occupancy") is not True:
        raise ValueError("full must enable navigation_graph_occupancy")
    if baseline.get("navigation_floor_support") is not True:
        raise ValueError("coupling experiment must retain floor support")
    if task == "text" and (
        baseline.get("text_target_association") != "navigation_map"
        or baseline.get("text_detection_instances") is not True
    ):
        raise ValueError("text full must retain geodesic association and detection hypotheses")
    for name in names:
        config = dict(baseline)
        diff = manifest["profiles"][task][name]["config_diff"]
        allowed = {"navigation_graph_occupancy", "text_target_association"}
        if set(diff) - allowed:
            raise ValueError(f"{task}/{name}: unintended factor changes: {sorted(set(diff) - allowed)}")
        for key, change in diff.items():
            if change["before"] != baseline[key]:
                raise ValueError(f"{task}/{name}: inconsistent baseline for {key}")
            config[key] = change["after"]
        occupancy, association = CELL_SETTINGS[name]
        if config["navigation_graph_occupancy"] is not occupancy:
            raise ValueError(f"{task}/{name}: incorrect occupancy setting")
        if task == "text" and config["text_target_association"] != association:
            raise ValueError(f"{task}/{name}: incorrect association setting")


def load_paired_runs(root: Path, task: str, names: list[str], manifest: dict) -> dict:
    validate_design(manifest, task, names)
    runs = {}
    baseline_summary = None
    baseline_failure_config = None
    for name in names:
        directory = root / task / "runs" / name
        summary = read_summary(directory)
        rows = index_rows(read_rows(directory), ("split", "episode_index"), label=f"{task}/{name}")
        n = len(rows)
        if (summary.get("status") != "completed" or summary.get("episodes") != n
                or summary.get("completed") != n or summary.get("pending") != 0):
            raise ValueError(f"{task}/{name}: incomplete batch or episode count mismatch")
        if any(row["status"] not in {"success", "failure"} for row in rows.values()):
            raise ValueError(f"{task}/{name}: infrastructure failure or timeout")
        seed = require_summary_seed(summary, label=f"{task}/{name}")
        if seed != manifest["navigation_seed"]:
            raise ValueError(f"{task}/{name}: navigation seed mismatch")
        for row in rows.values():
            if str(row.get("seed", "")) != str(seed):
                raise ValueError(f"{task}/{name}: episode seed mismatch")
        failure_config = resolve_failure_config(summary)
        if any(value is None for value in failure_config.values()):
            raise ValueError(f"{task}/{name}: missing step budget or success radius")
        if not str(summary.get("commit", "")).strip():
            raise ValueError(f"{task}/{name}: missing source commit")
        if runs:
            if rows.keys() != runs[FULL].keys():
                raise ValueError(f"{task}/{name}: episode key mismatch")
            _validate_group_metadata(name, runs[FULL], rows, sorted(rows))
            if failure_config != baseline_failure_config:
                raise ValueError(f"{task}/{name}: step budget or success radius mismatch")
            if summary.get("commit") != baseline_summary.get("commit"):
                raise ValueError(f"{task}/{name}: source commit mismatch")
        else:
            baseline_summary = summary
            baseline_failure_config = failure_config
        runs[name] = rows
    return runs


def exposure_value(row: dict, counters: tuple[str, ...]) -> bool | None:
    """True if any counter fired, None if telemetry is missing or malformed."""
    try:
        payload = json.loads(row.get("telemetry_json", ""))
        activity = payload["feature_activity"]
        counts = [activity[key] for key in counters]
        for count in counts:
            if (isinstance(count, bool) or not isinstance(count, (int, float))
                    or not math.isfinite(count) or count < 0 or int(count) != count):
                return None
        return any(count > 0 for count in counts)
    except (TypeError, ValueError, KeyError, OverflowError):
        return None


def bootstrap_contrasts(
    values: np.ndarray, scenes: list[str], *, iterations: int, seed: int, unit: str,
) -> np.ndarray | None:
    """Percentile bootstrap 95% intervals, resampling episodes or whole scenes."""
    if iterations < 0 or seed < 0:
        raise ValueError("bootstrap iterations and seed must be non-negative")
    if unit not in {"episode", "scene"}:
        raise ValueError(f"unknown bootstrap unit: {unit}")
    if len(values) != len(scenes):
        raise ValueError("each episode must have one scene")
    if not iterations or len(values) < 2:
        return None
    if unit == "scene":
        labels = sorted(set(scenes))
        if len(labels) < 2:
            return None
        sums = np.array([values[np.array(scenes) == label].sum(axis=0) for label in labels])
        sizes = np.array([scenes.count(label) for label in labels])
    else:
        sums = values
        sizes = np.ones(len(values), dtype=int)
    rng = np.random.default_rng(seed)
    samples = np.empty((iterations, values.shape[1]))
    # Bounded blocks avoid allocating B x N x contrasts for a full benchmark.
    block = max(1, min(256, 1_000_000 // max(1, sums.size)))
    for start in range(0, iterations, block):
        stop = min(start + block, iterations)
        indices = rng.integers(len(sums), size=(stop - start, len(sums)))
        samples[start:stop] = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None]
    return np.quantile(samples, [.025, .975], axis=0)


def analyze_task(runs: dict, task: str, *, bootstrap: int, seed: int, units: list[str]) -> dict:
    weights = contrast_weights(task, RANK in runs)
    keys = sorted(runs[FULL])
    strata = {"all": keys}
    coverage = {}
    for label, counters in EXPOSURES.items():
        if task == "instance-image" and label != "graph_mask_changed":
            continue
        exposure = {key: exposure_value(runs[FULL][key], counters) for key in keys}
        coverage[label] = {
            "counters": list(counters),
            "exposed": sum(value is True for value in exposure.values()),
            "unexposed": sum(value is False for value in exposure.values()),
            "unknown": sum(value is None for value in exposure.values()),
        }
        for value, suffix in ((True, "exposed"), (False, "unexposed")):
            strata[f"{label}/{suffix}"] = [key for key in keys if exposure[key] is value]
    result_rows = []
    columns = [(contrast, metric) for contrast in weights for metric in ("success_rate", "spl")]
    for stratum, subset in strata.items():
        scenes = [runs[FULL][key]["scene_id"] for key in subset]
        values = np.zeros((len(subset), len(columns)))
        for i, key in enumerate(subset):
            for j, (contrast, metric) in enumerate(columns):
                values[i, j] = sum(
                    weight * (float(_flag(runs[name][key]["success"]))
                              if metric == "success_rate" else float(runs[name][key]["spl"]))
                    for name, weight in weights[contrast].items()
                )
        for unit in units:
            cis = bootstrap_contrasts(values, scenes, iterations=bootstrap, seed=seed, unit=unit)
            for j, (contrast, metric) in enumerate(columns):
                status = ("empty_stratum" if not subset else "bootstrap_disabled" if not bootstrap
                          else "insufficient_units" if cis is None else "estimated")
                result_rows.append({
                    "task_mode": task, "stratum": stratum, "contrast": contrast,
                    "metric": metric, "bootstrap_unit": unit,
                    "n_episodes": len(subset), "n_scenes": len(set(scenes)),
                    "effect": float(values[:, j].mean()) if subset else None,
                    "ci95_low": float(cis[0, j]) if cis is not None else None,
                    "ci95_high": float(cis[1, j]) if cis is not None else None,
                    "status": status,
                })
    return {"contrast_weights": weights, "baseline_exposure": coverage, "results": result_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--task-mode", choices=("text", "instance-image", "both"), default="both")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--statistics-seed", type=int, default=0)
    parser.add_argument("--bootstrap-unit", choices=("episode", "scene", "both"), default="both")
    args = parser.parse_args()
    if args.bootstrap < 0 or args.statistics_seed < 0:
        parser.error("bootstrap and statistics-seed must be non-negative")
    root = args.experiment_root.resolve()
    manifest_path = root / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    tasks = ("text", "instance-image") if args.task_mode == "both" else (args.task_mode,)
    units = ["episode", "scene"] if args.bootstrap_unit == "both" else [args.bootstrap_unit]
    output: dict[str, Any] = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "statistics": {"bootstrap": args.bootstrap, "seed": args.statistics_seed,
                       "units": units, "navigation_seed": manifest["navigation_seed"],
                       "ci": "pointwise paired percentile 95%; no multiplicity adjustment"},
        "interpretation": (
            "Positive simple/main effects favor the enabled mechanism; interaction is "
            "full - depth_only - euclidean + decoupled. Effects use the 0-1 scale. "
            "Exposure strata use only full-run telemetry and are exploratory trajectory "
            "diagnostics, not pre-treatment causal subgroups. A CI spanning zero does "
            "not establish equivalence. Cell transitions measure the current graph "
            "update, not a counterfactual depth-only replay."
        ),
        "tasks": {},
    }
    for task in tasks:
        names = [FULL, DEPTH] if task == "instance-image" else [FULL, DEPTH, EUCLIDEAN, DECOUPLED]
        selected = manifest["selected_ablations"][task]
        if task == "text" and ({RANK, DEPTH_RANK} & set(selected)):
            names += [RANK, DEPTH_RANK]
        if set(names) - set(selected):
            raise ValueError(f"{task}: required profiles not selected: {sorted(set(names) - set(selected))}")
        runs = load_paired_runs(root, task, names, manifest)
        output["tasks"][task] = analyze_task(
            runs, task, bootstrap=args.bootstrap, seed=args.statistics_seed, units=units,
        )
    rows = [row for task in output["tasks"].values() for row in task["results"]]
    output_path = root / "representation_coupling.json"
    output_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (root / "representation_coupling.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_path} and representation_coupling.tsv")
    for task, report in output["tasks"].items():
        for label, coverage in report["baseline_exposure"].items():
            print(f"{task}/{label}: {coverage['exposed']} exposed, "
                  f"{coverage['unexposed']} unexposed, {coverage['unknown']} unknown")


if __name__ == "__main__":
    main()
