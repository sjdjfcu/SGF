#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


SPL_TEST = "two-sided_paired_sign_flip_permutation"
DEFAULT_SPL_PERMUTATIONS = 1_000_000
# Sign assignments drawn per block, fixed for reproducible p-values
SIGN_FLIP_BLOCK = 2048
CRASH_STATUSES = {
    "crash",
    "missing_json",
    "interrupted",
    "invalid_metrics",
}
VALID_EPISODE_STATUSES = {
    "success",
    "failure",
    "timeout",
    *CRASH_STATUSES,
}
REQUIRED_EPISODE_COLUMNS = frozenset(
    {
        "dataset_episode_id",
        "split",
        "episode_index",
        "goal_key",
        "goal_name",
        "scene_id",
        "status",
        "timed_out",
        "elapsed_seconds",
        "success",
        "spl",
        "soft_spl",
        "distance_to_goal",
        "steps",
        "target_found",
        "stop_called",
    }
)
PAIR_METRICS: tuple[tuple[str, str, Callable[[Any], float]], ...] = (
    ("success_rate", "success", lambda value: float(_flag(value))),
    ("spl", "spl", lambda value: _float(value)),
    ("soft_spl", "soft_spl", lambda value: _float(value)),
    ("distance_to_goal", "distance_to_goal", lambda value: _float(value)),
    ("steps", "steps", lambda value: float(_int(value))),
    ("elapsed_seconds", "elapsed_seconds", lambda value: _float(value)),
    ("target_found_rate", "target_found", lambda value: float(_flag(value))),
    ("stop_called_rate", "stop_called", lambda value: float(_flag(value))),
)
GROUP_ABSOLUTE_NAMES = {
    "success_rate": "success_rate",
    "spl": "mean_spl",
    "soft_spl": "mean_soft_spl",
    "distance_to_goal": "mean_distance_to_goal",
    "steps": "mean_steps",
    "elapsed_seconds": "mean_elapsed_seconds",
    "target_found_rate": "target_found_rate",
    "stop_called_rate": "stop_called_rate",
}
GROUP_DELTA_NAMES = {
    "success_rate": "delta_success_rate",
    "spl": "delta_spl",
    "soft_spl": "delta_soft_spl",
    "distance_to_goal": "delta_distance_to_goal",
    "steps": "delta_steps",
    "elapsed_seconds": "delta_elapsed_seconds",
    "target_found_rate": "delta_target_found_rate",
    "stop_called_rate": "delta_stop_called_rate",
}
PAYLOAD_FILTERED_METRICS = {"distance_to_goal", "steps"}
TERMINAL_STOP_SOURCES = frozenset({"decision", "endgame", "budget", "none"})
FAILURE_TYPES = (
    "success",
    "false_stop",
    "near_without_stop",
    "max_steps_failure",
    "endgame_stop_failure",
    "budget_stop_failure",
    "stop_called_failure",
    "target_found_without_stop",
    "stopped_far_or_lost",
    "timeout",
    "crash_or_missing_json",
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    try:
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return None


def _flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _required_episode_value(
    row: Mapping[str, Any],
    field: str,
    *,
    label: str,
    row_number: int,
) -> Any:
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"invalid episode metrics for {label} at TSV row {row_number}: "
            f"missing {field!r}"
        )
    return value


def _strict_episode_float(
    row: Mapping[str, Any],
    field: str,
    *,
    label: str,
    row_number: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = _required_episode_value(
        row,
        field,
        label=label,
        row_number=row_number,
    )
    if isinstance(value, bool):
        number = None
    else:
        number = _optional_float(value)
    if (
        number is None
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        bounds = (
            f" in [{minimum}, {maximum}]"
            if minimum is not None and maximum is not None
            else f" >= {minimum}"
            if minimum is not None
            else f" <= {maximum}"
            if maximum is not None
            else ""
        )
        raise ValueError(
            f"invalid episode metrics for {label} at TSV row {row_number}: "
            f"{field!r} must be finite{bounds}, got {value!r}"
        )
    return number


def _strict_episode_int(
    row: Mapping[str, Any],
    field: str,
    *,
    label: str,
    row_number: int,
    minimum: int = 0,
) -> int:
    number = _strict_episode_float(
        row,
        field,
        label=label,
        row_number=row_number,
        minimum=float(minimum),
    )
    if number != int(number):
        raise ValueError(
            f"invalid episode metrics for {label} at TSV row {row_number}: "
            f"{field!r} must be an integer >= {minimum}, got {row.get(field)!r}"
        )
    return int(number)


def _strict_episode_flag(
    row: Mapping[str, Any],
    field: str,
    *,
    label: str,
    row_number: int,
) -> bool:
    value = _required_episode_value(
        row,
        field,
        label=label,
        row_number=row_number,
    )
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = _optional_float(value)
        if number in {0.0, 1.0}:
            return bool(number)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    try:
        number = float(text)
    except (TypeError, ValueError, OverflowError):
        number = None
    if number is not None and math.isfinite(number) and number in {0.0, 1.0}:
        return bool(number)
    raise ValueError(
        f"invalid episode metrics for {label} at TSV row {row_number}: "
        f"{field!r} must be a binary flag, got {value!r}"
    )


def validate_episode_row(
    row: Mapping[str, Any],
    *,
    label: str,
    row_number: int,
) -> None:
    """Reject malformed TSV rows before any converter can coerce them to zero."""

    status_value = _required_episode_value(
        row,
        "status",
        label=label,
        row_number=row_number,
    )
    status = str(status_value).strip().lower()
    if status not in VALID_EPISODE_STATUSES:
        raise ValueError(
            f"invalid episode metrics for {label} at TSV row {row_number}: "
            f"unrecognized status {status_value!r}"
        )

    _required_episode_value(
        row,
        "split",
        label=label,
        row_number=row_number,
    )
    _strict_episode_int(
        row,
        "episode_index",
        label=label,
        row_number=row_number,
    )
    for field in ("dataset_episode_id", "scene_id"):
        _required_episode_value(
            row,
            field,
            label=label,
            row_number=row_number,
        )

    timed_out = _strict_episode_flag(
        row,
        "timed_out",
        label=label,
        row_number=row_number,
    )
    success = _strict_episode_flag(
        row,
        "success",
        label=label,
        row_number=row_number,
    )
    spl = _strict_episode_float(
        row,
        "spl",
        label=label,
        row_number=row_number,
        minimum=0.0,
        maximum=1.0,
    )
    if spl > float(success):
        raise ValueError(
            f"invalid episode metrics for {label} at TSV row {row_number}: "
            f"spl={row.get('spl')!r} cannot exceed success={row.get('success')!r}"
        )
    _strict_episode_float(
        row,
        "soft_spl",
        label=label,
        row_number=row_number,
        minimum=0.0,
        maximum=1.0,
    )
    _strict_episode_float(
        row,
        "distance_to_goal",
        label=label,
        row_number=row_number,
        minimum=0.0,
    )
    _strict_episode_int(
        row,
        "steps",
        label=label,
        row_number=row_number,
    )
    _strict_episode_float(
        row,
        "elapsed_seconds",
        label=label,
        row_number=row_number,
        minimum=0.0,
    )
    _strict_episode_flag(
        row,
        "target_found",
        label=label,
        row_number=row_number,
    )
    _strict_episode_flag(
        row,
        "stop_called",
        label=label,
        row_number=row_number,
    )

    if success != (status == "success"):
        raise ValueError(
            f"invalid episode metrics for {label} at TSV row {row_number}: "
            f"status={status!r} is inconsistent with success={row.get('success')!r}"
        )
    if timed_out != (status == "timeout"):
        raise ValueError(
            f"invalid episode metrics for {label} at TSV row {row_number}: "
            f"status={status!r} is inconsistent with timed_out={row.get('timed_out')!r}"
        )


def read_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "episodes.tsv"
    if not path.exists():
        raise FileNotFoundError(f"missing episodes.tsv: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"episodes.tsv has no header: {path}")
        missing_columns = sorted(
            REQUIRED_EPISODE_COLUMNS - set(reader.fieldnames)
        )
        if missing_columns:
            raise ValueError(
                f"episodes.tsv is missing required metric column(s) at {path}: "
                + ", ".join(missing_columns)
            )
        return list(reader)


def read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary.json must contain an object: {path}")
    return payload


def read_navigation_rows(
    run_dir: Path, *, navigation_seed: int,
) -> dict[tuple[str, ...], dict[str, Any]]:
    """Read episode rows, scoring invalid episodes as failures."""
    path = run_dir / "episodes.tsv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"{path} has no episode rows")
    indexed = {}
    for row_number, row in enumerate(rows, start=2):
        label = str(path)
        for field in ("split", "dataset_episode_id", "scene_id"):
            _required_episode_value(row, field, label=label, row_number=row_number)
        episode_index = _strict_episode_int(
            row, "episode_index", label=label, row_number=row_number,
        )
        key = (str(row["split"]).strip(), str(episode_index))
        if key in indexed:
            raise ValueError(f"duplicate episode key for {path}: {key}")

        # Seed is missing if the episode exited before writing JSON
        seed = _optional_float(row.get("seed"))
        if seed is not None and seed >= 0 and seed.is_integer():
            if seed != navigation_seed:
                raise ValueError(f"seed mismatch for {path} at episode key {key}")
        try:
            validate_episode_row(row, label=label, row_number=row_number)
            valid = row["status"].strip().lower() in {"success", "failure"}
            valid = valid and seed == navigation_seed
            for field in ("max_steps", "success_radius"):
                number = _strict_episode_float(
                    row, field, label=label, row_number=row_number, minimum=0.0,
                )
                valid = valid and number > 0
                if field == "max_steps":
                    valid = valid and number.is_integer()
            if str(row.get("returncode", "")).strip():
                valid = valid and _optional_float(row["returncode"]) == 0.0
        except ValueError:
            valid = False
        observation = dict(row)
        observation["success"] = float(row["status"].strip().lower() == "success") if valid else 0.0
        observation["spl"] = float(row["spl"]) if valid else 0.0
        indexed[key] = observation
    return indexed


def compare_navigation_metrics(
    label: str,
    baseline: Mapping[tuple[str, ...], dict[str, Any]],
    candidate: Mapping[tuple[str, ...], dict[str, Any]],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Compute SR/SPL differences with episode-paired 95% intervals."""
    if not baseline or set(baseline) != set(candidate):
        raise ValueError(f"episode key mismatch for {label}")
    keys = sorted(baseline)
    _validate_group_metadata(label, baseline, candidate, keys)
    differences = {
        field: [candidate[key][field] - baseline[key][field] for key in keys]
        for field in ("success", "spl")
    }
    intervals = _paired_bootstrap_cis(differences, iterations=bootstrap, seed=seed)
    return {
        field: {"delta": mean(values), "ci95": intervals[field]}
        for field, values in differences.items()
    }


def key_for(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "") or "").strip() for field in fields)


def index_rows(
    rows: Sequence[dict[str, str]],
    fields: Sequence[str],
    *,
    label: str,
) -> dict[tuple[str, ...], dict[str, str]]:
    if not fields:
        raise ValueError("at least one episode key field is required")
    if len(set(fields)) != len(fields):
        raise ValueError(f"duplicate episode key fields: {list(fields)}")
    if not rows:
        raise ValueError(f"{label} has no episode rows")
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        validate_episode_row(
            row,
            label=label,
            row_number=row_number,
        )
        key = key_for(row, fields)
        empty_fields = [field for field, value in zip(fields, key) if not value]
        if empty_fields:
            raise ValueError(
                f"empty episode key for {label} at TSV row {row_number}; "
                f"fields={empty_fields}"
            )
        if key in indexed:
            raise ValueError(
                f"duplicate episode key for {label} at TSV row {row_number}: {key}"
            )
        indexed[key] = row
    return indexed


def _summary_setting(summary: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in summary and summary[name] not in (None, ""):
            return summary[name]
    for container_name in ("resolved_config", "nav_config", "navigation", "config"):
        container = summary.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for name in names:
            if name in container and container[name] not in (None, ""):
                return container[name]
    return None


def resolve_failure_config(
    summary: Mapping[str, Any],
    *,
    max_steps: int | None = None,
    success_radius: float | None = None,
) -> dict[str, int | float | None]:
    resolved_max_steps = max_steps
    if resolved_max_steps is None:
        resolved_max_steps = _optional_int(_summary_setting(summary, ("max_steps",)))
    if resolved_max_steps is not None and resolved_max_steps <= 0:
        resolved_max_steps = None

    resolved_success_radius = success_radius
    if resolved_success_radius is None:
        resolved_success_radius = _optional_float(
            _summary_setting(
                summary,
                ("success_radius", "success_radius_m", "task_success_radius_m"),
            )
        )
    if resolved_success_radius is not None and resolved_success_radius <= 0.0:
        resolved_success_radius = None
    return {
        "max_steps": resolved_max_steps,
        "success_radius": resolved_success_radius,
    }


def require_summary_seed(summary: Mapping[str, Any], *, label: str) -> int:
    raw_seed = summary.get("seed")
    if raw_seed is None or str(raw_seed).strip() == "":
        raise ValueError(f"missing seed in summary.json for {label}")
    if isinstance(raw_seed, bool):
        raise ValueError(f"invalid seed in summary.json for {label}: {raw_seed!r}")
    numeric_seed = _optional_float(raw_seed)
    if numeric_seed is None or not numeric_seed.is_integer():
        raise ValueError(f"invalid seed in summary.json for {label}: {raw_seed!r}")
    return int(numeric_seed)


def _effective_failure_config(
    row: Mapping[str, Any],
    *,
    max_steps: Any = None,
    success_radius: Any = None,
) -> dict[str, int | float | None]:
    effective_max_steps = _optional_int(max_steps)
    if effective_max_steps is None or effective_max_steps <= 0:
        effective_max_steps = _optional_int(row.get("max_steps"))
    if effective_max_steps is not None and effective_max_steps <= 0:
        effective_max_steps = None

    effective_success_radius = _optional_float(success_radius)
    if effective_success_radius is None or effective_success_radius <= 0.0:
        effective_success_radius = None
        for field in ("success_radius", "success_radius_m"):
            raw_value = row.get(field)
            if raw_value not in (None, ""):
                effective_success_radius = _optional_float(raw_value)
                if effective_success_radius is not None:
                    break
    if effective_success_radius is not None and effective_success_radius <= 0.0:
        effective_success_radius = None
    return {
        "max_steps": effective_max_steps,
        "success_radius": effective_success_radius,
    }


def terminal_stop_source(row: Mapping[str, Any]) -> str | None:
    """Source of the terminal STOP, or None if not recorded."""

    raw = row.get("telemetry_json")
    if isinstance(raw, Mapping):
        telemetry: Any = raw
    else:
        try:
            telemetry = json.loads(str(raw or ""))
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(telemetry, Mapping):
        return None
    termination = telemetry.get("termination")
    if not isinstance(termination, Mapping):
        return None
    source = termination.get("stop_source")
    return str(source) if source in TERMINAL_STOP_SOURCES else None


def failure_type(
    row: Mapping[str, Any],
    *,
    max_steps: int | None = None,
    success_radius: float | None = None,
) -> str:
    if _flag(row.get("success")):
        return "success"
    status = str(row.get("status", "") or "").strip().lower()
    if _flag(row.get("timed_out")) or status == "timeout":
        return "timeout"
    if status in CRASH_STATUSES:
        return "crash_or_missing_json"
    # Forced STOPs (endgame, budget failsafe) are not false stops
    stop_source = terminal_stop_source(row)
    if stop_source == "endgame":
        return "endgame_stop_failure"
    if stop_source == "budget":
        return "budget_stop_failure"

    effective_config = _effective_failure_config(
        row,
        max_steps=max_steps,
        success_radius=success_radius,
    )
    effective_max_steps = effective_config["max_steps"]
    effective_success_radius = effective_config["success_radius"]

    steps = _int(row.get("steps"))
    distance = _optional_float(row.get("distance_to_goal"))
    stop_called = _flag(row.get("stop_called"))
    if effective_max_steps is not None and steps >= effective_max_steps:
        return "max_steps_failure"
    if (
        stop_called
        and distance is not None
        and effective_success_radius is not None
        and distance > effective_success_radius
    ):
        return "false_stop"
    if (
        not stop_called
        and distance is not None
        and effective_success_radius is not None
        and distance <= effective_success_radius
    ):
        return "near_without_stop"
    if stop_called:
        return "stop_called_failure"
    if _flag(row.get("target_found")):
        return "target_found_without_stop"
    return "stopped_far_or_lost"


def _has_valid_distance_steps_payload(row: Mapping[str, Any]) -> bool:
    status = str(row.get("status", "") or "").strip().lower()
    if status not in {"success", "failure"}:
        return False
    distance = _optional_float(row.get("distance_to_goal"))
    steps = _optional_float(row.get("steps"))
    return bool(
        distance is not None
        and distance >= 0.0
        and steps is not None
        and steps >= 0.0
        and steps.is_integer()
    )


def run_metrics(
    label: str,
    run_dir: Path,
    rows: Sequence[dict[str, str]],
    *,
    summary: Mapping[str, Any] | None = None,
    failure_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = read_summary(run_dir) if summary is None else summary
    failure_config = failure_config or resolve_failure_config(summary)
    successes = sum(_flag(row.get("success")) for row in rows)
    target_found = sum(_flag(row.get("target_found")) for row in rows)
    stop_called = sum(_flag(row.get("stop_called")) for row in rows)
    n = len(rows)
    valid_metric_rows = [
        row for row in rows if _has_valid_distance_steps_payload(row)
    ]
    return {
        "label": label,
        "run_tag": summary.get("run_tag", run_dir.name),
        "task_mode": summary.get("task_mode", ""),
        "seed": summary.get("seed", ""),
        "commit": summary.get("commit", ""),
        "episodes": n,
        "completed": summary.get("completed", n),
        "pending": summary.get("pending", 0),
        "successes": successes,
        "success_rate": successes / n if n else 0.0,
        "mean_spl": mean(_float(row.get("spl")) for row in rows) if rows else 0.0,
        "mean_soft_spl": mean(_float(row.get("soft_spl")) for row in rows) if rows else 0.0,
        "mean_distance_to_goal": (
            mean(_float(row.get("distance_to_goal")) for row in valid_metric_rows)
            if valid_metric_rows
            else 0.0
        ),
        "mean_steps": (
            mean(_int(row.get("steps")) for row in valid_metric_rows)
            if valid_metric_rows
            else 0.0
        ),
        "mean_elapsed_seconds": (
            mean(_float(row.get("elapsed_seconds")) for row in rows) if rows else 0.0
        ),
        "target_found": target_found,
        "target_found_rate": target_found / n if n else 0.0,
        "stop_called": stop_called,
        "stop_called_rate": stop_called / n if n else 0.0,
        "timeouts": sum(
            _flag(row.get("timed_out"))
            or str(row.get("status", "")).strip().lower() == "timeout"
            for row in rows
        ),
        "crashes": sum(
            str(row.get("status", "")).strip().lower() in CRASH_STATUSES
            for row in rows
        ),
        "max_steps": failure_config.get("max_steps"),
        "success_radius": failure_config.get("success_radius"),
    }


def bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float] | None:
    if len(baseline) != len(candidate):
        raise ValueError("paired bootstrap inputs must have equal lengths")
    if iterations < 0:
        raise ValueError("bootstrap iterations cannot be negative")
    if iterations == 0 or not baseline:
        return None
    rng = random.Random(seed)
    deltas: list[float] = []
    n = len(baseline)
    paired_deltas = [
        float(candidate[index]) - float(baseline[index]) for index in range(n)
    ]
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += paired_deltas[rng.randrange(n)]
        deltas.append(total / n)
    deltas.sort()
    low = deltas[int(0.025 * (len(deltas) - 1))]
    high = deltas[int(0.975 * (len(deltas) - 1))]
    return low, high


def _paired_bootstrap_cis(
    paired_deltas: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, tuple[float, float] | None]:
    if iterations < 0:
        raise ValueError("bootstrap iterations cannot be negative")
    if not paired_deltas:
        return {}
    if iterations == 0:
        return {name: None for name in paired_deltas}

    names_by_length: dict[int, list[str]] = {}
    for name, values in paired_deltas.items():
        names_by_length.setdefault(len(values), []).append(name)

    output: dict[str, tuple[float, float] | None] = {}
    for group_index, (n, names) in enumerate(sorted(names_by_length.items())):
        if n == 0:
            output.update({name: None for name in names})
            continue
        distributions = {name: [] for name in names}
        rng = random.Random(seed + group_index)
        population = range(n)
        for _ in range(iterations):
            sampled_indices = rng.choices(population, k=n)
            for name in names:
                deltas = paired_deltas[name]
                distributions[name].append(
                    sum(float(deltas[index]) for index in sampled_indices) / n
                )
        for name in names:
            distribution = distributions[name]
            distribution.sort()
            output[name] = (
                distribution[int(0.025 * (iterations - 1))],
                distribution[int(0.975 * (iterations - 1))],
            )
    return output


def exact_mcnemar_p_value(fail_to_success: int, success_to_fail: int) -> float:
    if fail_to_success < 0 or success_to_fail < 0:
        raise ValueError("McNemar discordant counts cannot be negative")
    discordant = fail_to_success + success_to_fail
    if discordant == 0:
        return 1.0
    lower = min(fail_to_success, success_to_fail)
    log_probability_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(lower + 1)
    ]
    largest = max(log_probability_terms)
    lower_tail = math.exp(largest) * sum(
        math.exp(term - largest) for term in log_probability_terms
    )
    return min(1.0, 2.0 * lower_tail)


def paired_sign_flip_p_value(
    differences: Sequence[float],
    *,
    permutations: int,
    seed: int,
) -> float | None:
    """Two-sided paired sign-flip p-value (Phipson and Smyth, 2010)."""

    if permutations < 0:
        raise ValueError("sign-flip permutations cannot be negative")
    if permutations == 0:
        return None
    if seed < 0:
        raise ValueError("the sign-flip seed must be non-negative")
    values = np.asarray([float(value) for value in differences], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be finite")
    magnitudes = np.abs(values[values != 0.0])
    if magnitudes.size == 0:
        return 1.0
    total = float(magnitudes.sum())
    # Tolerance for summation error, ties count as extreme
    threshold = abs(float(values.sum())) - 1e-9 * total
    bit_generator = np.random.PCG64(int(seed))
    width = int(magnitudes.size)
    exceedances = 0
    remaining = int(permutations)
    while remaining:
        count = min(remaining, SIGN_FLIP_BLOCK)
        words = -(-(count * width) // 64)
        raw = bit_generator.random_raw(words).astype("<u8", copy=False)
        bits = np.unpackbits(raw.view(np.uint8))[: count * width]
        positive = bits.reshape(count, width).astype(np.float64)
        # Sum of s_i * |d_i| for signs s_i in {-1, +1}
        sums = 2.0 * (positive @ magnitudes) - total
        exceedances += int(np.count_nonzero(np.abs(sums) >= threshold))
        remaining -= count
    return (1 + exceedances) / (1 + int(permutations))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    if not p_values:
        return []
    indexed: list[tuple[float, int]] = []
    for index, value in enumerate(p_values):
        number = _optional_float(value)
        if number is None or not 0.0 <= number <= 1.0:
            raise ValueError(f"invalid p-value at index {index}: {value!r}")
        indexed.append((number, index))
    indexed.sort()
    adjusted = [1.0] * len(indexed)
    running_max = 0.0
    family_size = len(indexed)
    for rank, (value, original_index) in enumerate(indexed):
        corrected = min(1.0, value * (family_size - rank))
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max
    return adjusted


def _validate_indexed_rows(
    rows: Mapping[tuple[str, ...], dict[str, str]],
    *,
    label: str,
) -> None:
    if not rows:
        raise ValueError(f"{label} has no episode rows")
    for key in rows:
        if not key or any(not str(component).strip() for component in key):
            raise ValueError(f"empty episode key for {label}: {key}")


def _metric_values(
    rows: Mapping[tuple[str, ...], dict[str, str]],
    keys: Sequence[tuple[str, ...]],
    field: str,
    converter: Callable[[Any], float],
) -> list[float]:
    return [converter(rows[key].get(field)) for key in keys]


def _metric_pair_keys(
    metric_name: str,
    baseline_rows: Mapping[tuple[str, ...], dict[str, str]],
    candidate_rows: Mapping[tuple[str, ...], dict[str, str]],
    keys: Sequence[tuple[str, ...]],
) -> list[tuple[str, ...]]:
    if metric_name not in PAYLOAD_FILTERED_METRICS:
        return list(keys)
    return [
        key
        for key in keys
        if _has_valid_distance_steps_payload(baseline_rows[key])
        and _has_valid_distance_steps_payload(candidate_rows[key])
    ]


def _paired_metric_differences(
    metric_name: str,
    baseline_rows: Mapping[tuple[str, ...], dict[str, str]],
    candidate_rows: Mapping[tuple[str, ...], dict[str, str]],
    keys: Sequence[tuple[str, ...]],
) -> list[float]:
    field, converter = next(
        (field, converter)
        for name, field, converter in PAIR_METRICS
        if name == metric_name
    )
    metric_keys = _metric_pair_keys(metric_name, baseline_rows, candidate_rows, keys)
    return [
        candidate - baseline
        for baseline, candidate in zip(
            _metric_values(baseline_rows, metric_keys, field, converter),
            _metric_values(candidate_rows, metric_keys, field, converter),
        )
    ]


def _pair_metric_results(
    baseline_rows: Mapping[tuple[str, ...], dict[str, str]],
    candidate_rows: Mapping[tuple[str, ...], dict[str, str]],
    keys: Sequence[tuple[str, ...]],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    values: dict[str, tuple[list[float], list[float]]] = {}
    pair_counts: dict[str, int] = {}
    paired_deltas: dict[str, list[float]] = {}
    for name, field, converter in PAIR_METRICS:
        metric_keys = _metric_pair_keys(
            name, baseline_rows, candidate_rows, keys
        )
        baseline_values = _metric_values(
            baseline_rows, metric_keys, field, converter
        )
        candidate_values = _metric_values(
            candidate_rows, metric_keys, field, converter
        )
        values[name] = (baseline_values, candidate_values)
        pair_counts[name] = len(metric_keys)
        paired_deltas[name] = [
            candidate - baseline
            for baseline, candidate in zip(baseline_values, candidate_values)
        ]
    confidence_intervals = _paired_bootstrap_cis(
        paired_deltas, iterations=bootstrap, seed=seed
    )

    results: dict[str, dict[str, Any]] = {}
    for name, _, _ in PAIR_METRICS:
        baseline_values, candidate_values = values[name]
        baseline_mean = mean(baseline_values) if baseline_values else None
        candidate_mean = mean(candidate_values) if candidate_values else None
        results[name] = {
            "n_pairs": pair_counts[name],
            "baseline": baseline_mean,
            "candidate": candidate_mean,
            "delta": (
                candidate_mean - baseline_mean
                if baseline_mean is not None and candidate_mean is not None
                else None
            ),
            "ci95": confidence_intervals[name],
        }
    return results


def _validate_group_metadata(
    label: str,
    baseline_rows: Mapping[tuple[str, ...], dict[str, str]],
    candidate_rows: Mapping[tuple[str, ...], dict[str, str]],
    keys: Sequence[tuple[str, ...]],
) -> None:
    for key in keys:
        for field in ("dataset_episode_id", "scene_id"):
            baseline_value = str(baseline_rows[key].get(field, "") or "").strip()
            candidate_value = str(candidate_rows[key].get(field, "") or "").strip()
            if not baseline_value or not candidate_value:
                raise ValueError(
                    f"missing {field} metadata for {label} at episode key "
                    f"{key}: baseline={baseline_value!r}, "
                    f"candidate={candidate_value!r}"
                )
            if baseline_value != candidate_value:
                raise ValueError(
                    f"{field} metadata mismatch for {label} at episode key "
                    f"{key}: baseline={baseline_value!r}, "
                    f"candidate={candidate_value!r}"
                )
        for field in ("goal_key", "goal_name"):
            baseline_value = str(baseline_rows[key].get(field, "") or "").strip()
            candidate_value = str(candidate_rows[key].get(field, "") or "").strip()
            if (
                baseline_value
                and candidate_value
                and baseline_value != candidate_value
            ):
                raise ValueError(
                    f"{field} metadata mismatch for {label} at episode key "
                    f"{key}: baseline={baseline_value!r}, "
                    f"candidate={candidate_value!r}"
                )


def _group_rows(
    baseline_rows: Mapping[tuple[str, ...], dict[str, str]],
    candidate_rows: Mapping[tuple[str, ...], dict[str, str]],
    keys: Sequence[tuple[str, ...]],
    *,
    group_by: str,
) -> list[dict[str, Any]]:
    grouped_keys: dict[str, list[tuple[str, ...]]] = {}
    for key in keys:
        group_value = str(
            baseline_rows[key].get(group_by)
            or candidate_rows[key].get(group_by)
            or "(missing)"
        ).strip()
        grouped_keys.setdefault(group_value or "(missing)", []).append(key)

    output: list[dict[str, Any]] = []
    for group_value in sorted(grouped_keys):
        group_keys = grouped_keys[group_value]
        record: dict[str, Any] = {
            "group_by": group_by,
            "group_value": group_value,
            "n_common": len(group_keys),
        }
        for metric_name, field, converter in PAIR_METRICS:
            metric_keys = _metric_pair_keys(
                metric_name, baseline_rows, candidate_rows, group_keys
            )
            baseline_values = _metric_values(
                baseline_rows, metric_keys, field, converter
            )
            candidate_values = _metric_values(
                candidate_rows, metric_keys, field, converter
            )
            baseline_mean = mean(baseline_values) if baseline_values else None
            candidate_mean = mean(candidate_values) if candidate_values else None
            absolute_name = GROUP_ABSOLUTE_NAMES[metric_name]
            record[f"n_pairs_{metric_name}"] = len(metric_keys)
            record[f"baseline_{absolute_name}"] = baseline_mean
            record[f"candidate_{absolute_name}"] = candidate_mean
            record[GROUP_DELTA_NAMES[metric_name]] = (
                candidate_mean - baseline_mean
                if baseline_mean is not None and candidate_mean is not None
                else None
            )
        output.append(record)
    return output


def compare_pair(
    label: str,
    baseline_rows: Mapping[tuple[str, ...], dict[str, str]],
    candidate_rows: Mapping[tuple[str, ...], dict[str, str]],
    *,
    allow_subset: bool,
    bootstrap: int,
    seed: int,
    spl_permutations: int = 0,
    baseline_failure_config: Mapping[str, Any] | None = None,
    candidate_failure_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _validate_indexed_rows(baseline_rows, label="baseline")
    _validate_indexed_rows(candidate_rows, label=label)
    baseline_keys = set(baseline_rows)
    candidate_keys = set(candidate_rows)
    if baseline_keys != candidate_keys:
        if allow_subset and candidate_keys.issubset(baseline_keys):
            baseline_rows = {key: baseline_rows[key] for key in candidate_keys}
        else:
            missing = sorted(baseline_keys ^ candidate_keys)[:10]
            raise ValueError(f"episode key mismatch for {label}; examples={missing}")
    keys = sorted(candidate_rows)
    _validate_group_metadata(label, baseline_rows, candidate_rows, keys)
    metrics = _pair_metric_results(
        baseline_rows,
        candidate_rows,
        keys,
        bootstrap=bootstrap,
        seed=seed,
    )

    fail_to_success: list[
        tuple[tuple[str, ...], dict[str, str], dict[str, str]]
    ] = []
    success_to_fail: list[
        tuple[tuple[str, ...], dict[str, str], dict[str, str]]
    ] = []
    for key in keys:
        base = baseline_rows[key]
        cand = candidate_rows[key]
        if not _flag(base.get("success")) and _flag(cand.get("success")):
            fail_to_success.append((key, base, cand))
        elif _flag(base.get("success")) and not _flag(cand.get("success")):
            success_to_fail.append((key, base, cand))

    mcnemar_p_value = exact_mcnemar_p_value(
        len(fail_to_success), len(success_to_fail)
    )
    spl_differences = _paired_metric_differences(
        "spl", baseline_rows, candidate_rows, keys
    )
    spl_p_value = paired_sign_flip_p_value(
        spl_differences, permutations=spl_permutations, seed=seed
    )
    baseline_failure_config = dict(baseline_failure_config or {})
    candidate_failure_config = dict(candidate_failure_config or {})
    result: dict[str, Any] = {
        "label": label,
        "n_common": len(keys),
        "metrics": metrics,
        "delta_success_rate": metrics["success_rate"]["delta"],
        "delta_success_ci95": metrics["success_rate"]["ci95"],
        "delta_spl": metrics["spl"]["delta"],
        "delta_spl_ci95": metrics["spl"]["ci95"],
        "delta_soft_spl": metrics["soft_spl"]["delta"],
        "delta_soft_spl_ci95": metrics["soft_spl"]["ci95"],
        "delta_distance_to_goal": metrics["distance_to_goal"]["delta"],
        "delta_distance_to_goal_ci95": metrics["distance_to_goal"]["ci95"],
        "delta_steps": metrics["steps"]["delta"],
        "delta_steps_ci95": metrics["steps"]["ci95"],
        "delta_elapsed_seconds": metrics["elapsed_seconds"]["delta"],
        "delta_elapsed_seconds_ci95": metrics["elapsed_seconds"]["ci95"],
        "delta_target_found_rate": metrics["target_found_rate"]["delta"],
        "delta_stop_called_rate": metrics["stop_called_rate"]["delta"],
        "fail_to_success": len(fail_to_success),
        "success_to_fail": len(success_to_fail),
        "net_flips": len(fail_to_success) - len(success_to_fail),
        "mcnemar_p_value": mcnemar_p_value,
        "mcnemar_p_value_holm": None,
        "mcnemar": {
            "fail_to_success": len(fail_to_success),
            "success_to_fail": len(success_to_fail),
            "discordant": len(fail_to_success) + len(success_to_fail),
            "p_value_exact_two_sided": mcnemar_p_value,
            "p_value_holm": None,
        },
        "spl_p_value": spl_p_value,
        "spl_p_value_holm": None,
        "spl_test": (
            None
            if spl_p_value is None
            else {
                "n_pairs": len(spl_differences),
                "nonzero_differences": sum(
                    value != 0.0 for value in spl_differences
                ),
                "permutations": int(spl_permutations),
                "p_value_two_sided": spl_p_value,
                "p_value_holm": None,
            }
        ),
        "timeout_delta": sum(
            _flag(candidate_rows[key].get("timed_out"))
            or str(candidate_rows[key].get("status", "")).strip().lower()
            == "timeout"
            for key in keys
        )
        - sum(
            _flag(baseline_rows[key].get("timed_out"))
            or str(baseline_rows[key].get("status", "")).strip().lower()
            == "timeout"
            for key in keys
        ),
        "crash_delta": sum(
            str(candidate_rows[key].get("status", "")).strip().lower()
            in CRASH_STATUSES
            for key in keys
        )
        - sum(
            str(baseline_rows[key].get("status", "")).strip().lower()
            in CRASH_STATUSES
            for key in keys
        ),
        "failure_config": {
            "baseline": baseline_failure_config,
            "candidate": candidate_failure_config,
        },
        "termination_outcomes": termination_outcomes(
            baseline_rows,
            candidate_rows,
            keys,
            bootstrap=bootstrap,
            seed=seed,
            baseline_failure_config=baseline_failure_config,
            candidate_failure_config=candidate_failure_config,
        ),
        "groups": {
            "scene_id": _group_rows(
                baseline_rows, candidate_rows, keys, group_by="scene_id"
            ),
            "goal_name": _group_rows(
                baseline_rows, candidate_rows, keys, group_by="goal_name"
            ),
        },
        "flips": fail_to_success + success_to_fail,
    }
    return result


def termination_outcomes(
    baseline_rows: Mapping[tuple[str, ...], dict[str, str]],
    candidate_rows: Mapping[tuple[str, ...], dict[str, str]],
    keys: Sequence[tuple[str, ...]],
    *,
    bootstrap: int,
    seed: int,
    baseline_failure_config: Mapping[str, Any],
    candidate_failure_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Paired terminal-route and failure-type rates."""

    paired: dict[str, tuple[list[float], list[float]]] = {}
    source_keys = [
        key
        for key in keys
        if terminal_stop_source(baseline_rows[key]) is not None
        and terminal_stop_source(candidate_rows[key]) is not None
    ]
    for source in ("decision", "endgame", "budget"):
        paired[f"{source}_stop_rate"] = (
            [
                float(terminal_stop_source(baseline_rows[key]) == source)
                for key in source_keys
            ],
            [
                float(terminal_stop_source(candidate_rows[key]) == source)
                for key in source_keys
            ],
        )
    baseline_types = [
        failure_type(
            baseline_rows[key],
            max_steps=baseline_failure_config.get("max_steps"),
            success_radius=baseline_failure_config.get("success_radius"),
        )
        for key in keys
    ]
    candidate_types = [
        failure_type(
            candidate_rows[key],
            max_steps=candidate_failure_config.get("max_steps"),
            success_radius=candidate_failure_config.get("success_radius"),
        )
        for key in keys
    ]
    for kind in FAILURE_TYPES:
        paired[f"{kind}_rate"] = (
            [float(value == kind) for value in baseline_types],
            [float(value == kind) for value in candidate_types],
        )
    cis = _paired_bootstrap_cis(
        {
            name: [c - b for b, c in zip(base, cand)]
            for name, (base, cand) in paired.items()
        },
        iterations=bootstrap,
        seed=seed,
    )
    outcomes: dict[str, Any] = {}
    for name, (base, cand) in paired.items():
        n_pairs = len(base)
        ci = cis.get(name)
        outcomes[name] = {
            "n_pairs": n_pairs,
            "baseline": mean(base) if n_pairs else None,
            "candidate": mean(cand) if n_pairs else None,
            "delta": (mean(cand) - mean(base)) if n_pairs else None,
            "ci95": None if ci is None else [ci[0], ci[1]],
        }
    return outcomes


def apply_holm_correction(pairs: Sequence[dict[str, Any]]) -> None:
    adjusted = holm_adjust([pair["mcnemar_p_value"] for pair in pairs])
    for pair, corrected in zip(pairs, adjusted):
        pair["mcnemar_p_value_holm"] = corrected
        pair["mcnemar"]["p_value_holm"] = corrected
        pair["mcnemar"]["holm_family_size"] = len(pairs)
    tested = [pair for pair in pairs if pair.get("spl_p_value") is not None]
    adjusted = holm_adjust([pair["spl_p_value"] for pair in tested])
    for pair, corrected in zip(tested, adjusted):
        pair["spl_p_value_holm"] = corrected
        pair["spl_test"]["p_value_holm"] = corrected
        pair["spl_test"]["holm_family_size"] = len(tested)


def write_flips(path: Path, pair: Mapping[str, Any]) -> None:
    columns = [
        "variant",
        "flip_type",
        "split",
        "episode_index",
        "dataset_episode_id",
        "goal_key",
        "goal_name",
        "scene_id",
        "baseline_success",
        "candidate_success",
        "baseline_failure_type",
        "candidate_failure_type",
        "baseline_spl",
        "candidate_spl",
        "baseline_soft_spl",
        "candidate_soft_spl",
        "baseline_distance_to_goal",
        "candidate_distance_to_goal",
        "baseline_steps",
        "candidate_steps",
        "baseline_artifact_dir",
        "candidate_artifact_dir",
        "baseline_max_steps",
        "candidate_max_steps",
        "baseline_success_radius",
        "candidate_success_radius",
    ]
    failure_configs = pair.get("failure_config") or {}
    baseline_config = failure_configs.get("baseline") or {}
    candidate_config = failure_configs.get("candidate") or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for _, base, cand in pair["flips"]:
            baseline_episode_config = _effective_failure_config(
                base,
                max_steps=baseline_config.get("max_steps"),
                success_radius=baseline_config.get("success_radius"),
            )
            candidate_episode_config = _effective_failure_config(
                cand,
                max_steps=candidate_config.get("max_steps"),
                success_radius=candidate_config.get("success_radius"),
            )
            flip_type = (
                "fail_to_success"
                if not _flag(base.get("success"))
                else "success_to_fail"
            )
            writer.writerow(
                {
                    "variant": pair["label"],
                    "flip_type": flip_type,
                    "split": base.get("split", ""),
                    "episode_index": base.get("episode_index", ""),
                    "dataset_episode_id": base.get("dataset_episode_id", ""),
                    "goal_key": base.get("goal_key", ""),
                    "goal_name": base.get("goal_name", ""),
                    "scene_id": base.get("scene_id", ""),
                    "baseline_success": base.get("success", ""),
                    "candidate_success": cand.get("success", ""),
                    "baseline_failure_type": failure_type(
                        base,
                        max_steps=baseline_episode_config.get("max_steps"),
                        success_radius=baseline_episode_config.get(
                            "success_radius"
                        ),
                    ),
                    "candidate_failure_type": failure_type(
                        cand,
                        max_steps=candidate_episode_config.get("max_steps"),
                        success_radius=candidate_episode_config.get(
                            "success_radius"
                        ),
                    ),
                    "baseline_spl": base.get("spl", ""),
                    "candidate_spl": cand.get("spl", ""),
                    "baseline_soft_spl": base.get("soft_spl", ""),
                    "candidate_soft_spl": cand.get("soft_spl", ""),
                    "baseline_distance_to_goal": base.get(
                        "distance_to_goal", ""
                    ),
                    "candidate_distance_to_goal": cand.get(
                        "distance_to_goal", ""
                    ),
                    "baseline_steps": base.get("steps", ""),
                    "candidate_steps": cand.get("steps", ""),
                    "baseline_artifact_dir": base.get("artifact_dir", ""),
                    "candidate_artifact_dir": cand.get("artifact_dir", ""),
                    "baseline_max_steps": baseline_episode_config.get(
                        "max_steps"
                    ),
                    "candidate_max_steps": candidate_episode_config.get(
                        "max_steps"
                    ),
                    "baseline_success_radius": baseline_episode_config.get(
                        "success_radius"
                    ),
                    "candidate_success_radius": candidate_episode_config.get(
                        "success_radius"
                    ),
                }
            )


def _group_columns() -> list[str]:
    columns = ["variant", "group_by", "group_value", "n_common"]
    for metric_name, _, _ in PAIR_METRICS:
        absolute_name = GROUP_ABSOLUTE_NAMES[metric_name]
        columns.extend(
            [
                f"n_pairs_{metric_name}",
                f"baseline_{absolute_name}",
                f"candidate_{absolute_name}",
                GROUP_DELTA_NAMES[metric_name],
            ]
        )
    return columns


def write_groups(path: Path, pairs: Iterable[Mapping[str, Any]]) -> None:
    columns = _group_columns()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for pair in pairs:
            for group_by in ("scene_id", "goal_name"):
                for group in pair.get("groups", {}).get(group_by, []):
                    writer.writerow({"variant": pair["label"], **group})


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError("--run must be LABEL=RUN_DIR")
    label, path_text = spec.split("=", 1)
    label = label.strip()
    path_text = path_text.strip()
    if not label or not path_text:
        raise ValueError("--run must have non-empty LABEL and RUN_DIR")
    return label, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare SGF navigation batch runs by paired episodes."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--run", action="append", default=[], help="LABEL=RUN_DIR. Can be repeated."
    )
    parser.add_argument("--key", default="split,episode_index")
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10_000,
        help="Episode-paired percentile bootstrap iterations (default: 10000; 0 disables CIs).",
    )
    parser.add_argument(
        "--statistics-seed",
        type=int,
        default=0,
        help="Random seed used only for statistical resampling.",
    )
    parser.add_argument(
        "--spl-permutations",
        type=int,
        default=DEFAULT_SPL_PERMUTATIONS,
        help=(
            "Sign-flip permutations for the paired SPL test, seeded by "
            "--statistics-seed (default: 1000000; 0 disables)."
        ),
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow candidate keys to be a subset of baseline keys.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override summary.json max_steps for failure classification.",
    )
    parser.add_argument(
        "--success-radius",
        type=float,
        default=None,
        help="Override summary.json success_radius for failure classification.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--flips-tsv", type=Path, default=None)
    parser.add_argument("--groups-tsv", type=Path, default=None)
    args = parser.parse_args()

    if args.bootstrap < 0:
        parser.error("--bootstrap must be non-negative")
    if args.spl_permutations < 0:
        parser.error("--spl-permutations must be non-negative")
    if args.spl_permutations > 0 and args.statistics_seed < 0:
        parser.error("--statistics-seed must be non-negative for the SPL test")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.success_radius is not None and (
        not math.isfinite(args.success_radius) or args.success_radius <= 0.0
    ):
        parser.error("--success-radius must be finite and positive")

    key_fields = [field.strip() for field in args.key.split(",") if field.strip()]
    baseline_rows_raw = read_rows(args.baseline)
    baseline_rows = index_rows(
        baseline_rows_raw, key_fields, label=f"baseline {args.baseline}"
    )
    baseline_summary = read_summary(args.baseline)
    navigation_seed = require_summary_seed(
        baseline_summary, label=f"baseline {args.baseline}"
    )
    baseline_failure_config = resolve_failure_config(
        baseline_summary,
        max_steps=args.max_steps,
        success_radius=args.success_radius,
    )
    output: dict[str, Any] = {
        "statistics": {
            "bootstrap_iterations": args.bootstrap,
            "statistics_seed": args.statistics_seed,
            "navigation_seed": navigation_seed,
            "bootstrap_unit": "paired_episode",
            "success_test": "two-sided_exact_mcnemar",
            "spl_test": SPL_TEST,
            "spl_permutations": args.spl_permutations,
            "multiple_comparison_correction": "holm_within_invocation",
        },
        "baseline": run_metrics(
            "baseline",
            args.baseline,
            baseline_rows_raw,
            summary=baseline_summary,
            failure_config=baseline_failure_config,
        ),
        "runs": [],
        "pairs": [],
    }
    all_pairs: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for spec in args.run:
        label, run_dir = _parse_run_spec(spec)
        if label in seen_labels:
            raise ValueError(f"duplicate --run label: {label}")
        seen_labels.add(label)
        rows_raw = read_rows(run_dir)
        rows = index_rows(rows_raw, key_fields, label=f"{label} {run_dir}")
        summary = read_summary(run_dir)
        candidate_navigation_seed = require_summary_seed(
            summary, label=f"{label} {run_dir}"
        )
        if candidate_navigation_seed != navigation_seed:
            raise ValueError(
                f"seed mismatch for {label}: baseline={navigation_seed}, "
                f"candidate={candidate_navigation_seed}"
            )
        candidate_failure_config = resolve_failure_config(
            summary,
            max_steps=args.max_steps,
            success_radius=args.success_radius,
        )
        output["runs"].append(
            run_metrics(
                label,
                run_dir,
                rows_raw,
                summary=summary,
                failure_config=candidate_failure_config,
            )
        )
        all_pairs.append(
            compare_pair(
                label,
                baseline_rows,
                rows,
                allow_subset=bool(args.allow_subset),
                bootstrap=args.bootstrap,
                seed=args.statistics_seed,
                spl_permutations=args.spl_permutations,
                baseline_failure_config=baseline_failure_config,
                candidate_failure_config=candidate_failure_config,
            )
        )

    apply_holm_correction(all_pairs)
    for pair in all_pairs:
        pair_without_flips = dict(pair)
        pair_without_flips.pop("flips", None)
        output["pairs"].append(pair_without_flips)

    rendered_output = json.dumps(output, indent=2)
    print(rendered_output)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered_output, encoding="utf-8")
    if args.flips_tsv and all_pairs:
        if len(all_pairs) == 1:
            write_flips(args.flips_tsv, all_pairs[0])
        else:
            for pair in all_pairs:
                write_flips(
                    args.flips_tsv.with_name(
                        f"{args.flips_tsv.stem}_{pair['label']}.tsv"
                    ),
                    pair,
                )
    if args.groups_tsv:
        write_groups(args.groups_tsv, all_pairs)


if __name__ == "__main__":
    main()
