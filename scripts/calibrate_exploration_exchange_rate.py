#!/usr/bin/env python3
"""Estimate the exchange rate for the difference frontier objective.

Pools the best value-per-action rates lambda* = max_i N_i / (S_i + S0)
recorded by rate-objective runs and reports per-task-mode quantiles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


SAMPLE_FIELDS = (
    ("rotation_priced", "exchange_rate_samples"),
    ("translation_only", "exchange_rate_samples_translation_only"),
)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
COMPLETED_STATUSES = frozenset({"success", "failure"})


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted sample."""

    if not sorted_values:
        raise ValueError("quantile of an empty sample")
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def episode_tables(path: Path, profile: str) -> list[Path]:
    if (path / "episodes.tsv").is_file():
        return [path / "episodes.tsv"]
    tables = sorted(path.glob(f"runs/{profile}/episodes.tsv"))
    if not tables:
        tables = sorted(path.glob(f"*/runs/{profile}/episodes.tsv"))
    if not tables:
        raise FileNotFoundError(
            f"no episodes.tsv for profile {profile!r} under {path}"
        )
    return tables


def read_samples(tables: Iterable[Path]) -> dict[str, dict[str, Any]]:
    by_mode: dict[str, dict[str, Any]] = {}
    for table in tables:
        with table.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if str(row.get("status", "")).strip().lower() not in (
                    COMPLETED_STATUSES
                ):
                    continue
                try:
                    telemetry = json.loads(row.get("telemetry_json") or "")
                except json.JSONDecodeError:
                    continue
                exploration = (
                    telemetry.get("exploration")
                    if isinstance(telemetry, dict)
                    else None
                )
                if not isinstance(exploration, dict):
                    continue
                mode = str(row.get("task_mode", "")).strip() or "unknown"
                record = by_mode.setdefault(
                    mode,
                    {
                        "episodes": 0,
                        "utility_forms": set(),
                        **{name: [] for name, _ in SAMPLE_FIELDS},
                    },
                )
                record["episodes"] += 1
                record["utility_forms"].add(
                    str(exploration.get("utility_form", ""))
                )
                for name, field in SAMPLE_FIELDS:
                    for value in exploration.get(field, ()) or ():
                        try:
                            number = float(value)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(number) and number > 0.0:
                            record[name].append(number)
    return by_mode


def summarize(by_mode: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for mode, record in sorted(by_mode.items()):
        forms = sorted(record["utility_forms"])
        if forms != ["rate"]:
            raise ValueError(
                f"{mode}: calibration requires runs of the rate objective, "
                f"found utility forms {forms}"
            )
        entry: dict[str, Any] = {"episodes": int(record["episodes"])}
        for name, _field in SAMPLE_FIELDS:
            values = sorted(record[name])
            if not values:
                raise ValueError(f"{mode}: no {name} exchange-rate samples")
            entry[name] = {
                "decisions": len(values),
                **{
                    f"q{int(round(q * 100)):02d}": quantile(values, q)
                    for q in QUANTILES
                },
            }
        rotation_median = entry["rotation_priced"]["q50"]
        translation_median = entry["translation_only"]["q50"]
        entry["recommended_exchange_rate"] = rotation_median
        entry["translation_only_median_ratio"] = (
            translation_median / rotation_median
        )
        summary[mode] = entry
    return summary


def yaml_snippet(summary: dict[str, Any]) -> str:
    lines = ["  task_mode_defaults:"]
    for mode, entry in summary.items():
        rotation = entry["rotation_priced"]
        lines.extend(
            [
                f"    {mode}:",
                "      # Median best rate of the rate objective at its own "
                f"decisions: {rotation['decisions']} decisions in "
                f"{entry['episodes']} episodes (IQR "
                f"{rotation['q25']:.6g}-{rotation['q75']:.6g}).",
                "      sgf_explore_exchange_rate: "
                f"{entry['recommended_exchange_rate']:.6g}",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate sgf_explore_exchange_rate from the best rates a full "
            "run's rate objective attained at its own decisions."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--profile",
        default="full",
        help="Ablation profile whose runs are pooled (default: full).",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    tables = [
        table
        for path in args.paths
        for table in episode_tables(path.expanduser(), args.profile)
    ]
    summary = summarize(read_samples(tables))
    if not summary:
        print("no exploration telemetry found", file=sys.stderr)
        return 1
    payload = {
        "sources": [str(table) for table in tables],
        "tasks": summary,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    print()
    print("# Navigation YAML (text_image_goal_nav section):")
    print(yaml_snippet(summary))
    for mode, entry in summary.items():
        ratio = entry["translation_only_median_ratio"]
        print(
            f"# {mode}: translation-only median / rotation-priced median = "
            f"{ratio:.3f}",
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
