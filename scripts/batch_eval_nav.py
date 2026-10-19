#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_RUNNER_PATH = REPO_ROOT / "scripts" / "run_habitat_text_image_goal_nav.py"
DEFAULT_NAV_CONFIG_PATH = REPO_ROOT / "python" / "sgf_habitat" / "configs" / "config_habitat.yaml"
DEFAULT_TEXT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "textnav_batch"
DEFAULT_INSTANCE_IMAGE_OUTPUT_ROOT = REPO_ROOT / "outputs" / "instance_image_nav_batch"
DEFAULT_EPISODE_CPU_THREADS = 1
DEFAULT_MAX_GPU_WORKERS_PER_DEVICE = 15
DEFAULT_RETRY_CRASHES = 3
DEFAULT_RETRY_DELAY_SECONDS = 10.0
DEFAULT_MAX_STEPS = 1000
DEFAULT_TIMEOUT_SECONDS = 1500.0
GPU_VRAM_SAMPLE_DELAY_SECONDS = 10.0
GPU_VRAM_QUERY_TIMEOUT_SECONDS = 5.0
GPU_VRAM_COLUMNS = ["timestamp", "gpu_index", "memory_used_mib", "memory_total_mib"]
GPU_VRAM_WRITE_LOCK = threading.Lock()
EXECUTION_PLAN_VERSION = 1
EPISODE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)
RETRYABLE_STATUSES = {"crash", "missing_json", "invalid_metrics"}
VALID_EPISODE_STATUSES = {
    "success",
    "failure",
    "timeout",
    "crash",
    "missing_json",
    "interrupted",
    "invalid_metrics",
}
ACTIVE_PROCESSES: set[subprocess.Popen] = set()
ACTIVE_PROCESSES_LOCK = threading.Lock()
SHUTDOWN_REQUESTED = threading.Event()
SUMMARY_COLUMNS = [
    "timestamp",
    "run_tag",
    "task_mode",
    "branch",
    "commit",
    "split",
    "seed",
    "episodes",
    "completed",
    "pending",
    "successes",
    "success_rate",
    "mean_spl",
    "mean_soft_spl",
    "mean_distance_to_goal",
    "mean_steps",
    "distance_steps_valid_episodes",
    "mean_elapsed_seconds",
    "target_found_rate",
    "stop_called_rate",
    "failures",
    "timeouts",
    "crashes",
    "max_steps",
    "success_radius",
    "timeout_seconds",
    "telemetry_json",
    "status",
    "run_dir",
]
EPISODE_COLUMNS = [
    "timestamp",
    "run_tag",
    "task_mode",
    "commit",
    "branch",
    "split",
    "seed",
    "episode_index",
    "dataset_episode_id",
    "goal_key",
    "goal_name",
    "goal_image_source",
    "scene_id",
    "status",
    "returncode",
    "timed_out",
    "elapsed_seconds",
    "success",
    "spl",
    "soft_spl",
    "distance_to_goal",
    "max_steps",
    "success_radius",
    "steps",
    "target_found",
    "stop_called",
    "telemetry_json",
    "artifact_dir",
    "log_path",
    "json_path",
    "final_image_path",
    "final_image_match_path",
    "image_match_video_path",
]
FAIL_FAST_STATUSES = {
    "timeout",
    "crash",
    "missing_json",
    "invalid_metrics",
}


def repo_data_path(*parts: str) -> str:
    return str(REPO_DATA_ROOT.joinpath(*parts))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Habitat SGF navigation one episode per subprocess, "
            "with an explicit wall-clock timeout for every episode."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config-path",
        default="benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml",
        help="Habitat-Lab task config path.",
    )
    parser.add_argument(
        "--dataset-path",
        default=repo_data_path("datasets", "instance_imagenav", "hm3d", "v3", "{split}", "{split}.json.gz"),
        help="InstanceImageNav dataset path pattern.",
    )
    parser.add_argument(
        "--goal-source",
        default="textnav",
        choices=("textnav",),
        help="Source of the episode text instruction.",
    )
    parser.add_argument(
        "--textnav-dataset-path",
        default=repo_data_path("datasets", "textnav", "{split}", "{split}_text.json.gz"),
        help="Local textnav attribute dataset path pattern.",
    )
    parser.add_argument(
        "--scenes-dir",
        default=repo_data_path("scene_datasets"),
        help="Root folder that contains HM3D scene folders.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument(
        "--task-mode",
        default="text",
        choices=("text", "instance-image"),
        help="Task mode forwarded to run_habitat_text_image_goal_nav.py.",
    )
    parser.add_argument(
        "--goal-image-path",
        default="",
        help="Optional RGB goal image for --task-mode instance-image, forwarded to every episode.",
    )
    parser.add_argument(
        "--episodes",
        default="all",
        help="Episode selection, e.g. all, 0,1,2, 0-9, 0-99,200-299.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap after episode selection.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=30,
        help="Maximum number of episode subprocesses to run at the same time.",
    )
    parser.add_argument(
        "--episode-cpu-threads",
        type=int,
        default=DEFAULT_EPISODE_CPU_THREADS,
        help=(
            f"CPU library threads per episode subprocess (default: {DEFAULT_EPISODE_CPU_THREADS}; "
            "0 inherits the parent environment)."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        default="auto",
        help="Comma-separated GPU IDs for CUDA_VISIBLE_DEVICES; auto detects visible GPUs, none leaves it unchanged.",
    )
    parser.add_argument(
        "--max-gpu-workers-per-device",
        type=int,
        default=DEFAULT_MAX_GPU_WORKERS_PER_DEVICE,
        help="Maximum concurrent episode subprocesses per GPU (0: no cap).",
    )
    parser.add_argument(
        "--gpu-vram-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Sample GPU VRAM 10 seconds after each episode starts (default: off).",
    )
    parser.add_argument(
        "--retry-crashes",
        type=int,
        default=DEFAULT_RETRY_CRASHES,
        help="Retries per episode after a crash, missing result JSON or invalid metrics.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
        help="Seconds to wait before each retry.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"Maximum episode steps (default: {DEFAULT_MAX_STEPS}).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            f"Per-episode wall-clock timeout (default: {DEFAULT_TIMEOUT_SECONDS:g}s); "
            "timed-out episodes are killed and recorded as unsuccessful."
        ),
    )
    parser.add_argument(
        "--kill-grace-seconds",
        type=float,
        default=10.0,
        help="Grace period between SIGTERM and SIGKILL for timed-out episodes.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for the inner navigation command.",
    )
    parser.add_argument(
        "--runner-path",
        type=Path,
        default=DEFAULT_RUNNER_PATH,
        help="Path to scripts/run_habitat_text_image_goal_nav.py.",
    )
    parser.add_argument(
        "--runner-args",
        default="",
        help="Extra arguments forwarded to run_habitat_text_image_goal_nav.py (shell-quoted string).",
    )
    parser.add_argument(
        "--nav-config-path",
        default=None,
        help="YAML file containing the text_image_goal_nav strategy configuration.",
    )
    parser.add_argument(
        "--quiet-runner",
        action="store_true",
        help="Do not pass --verbose to run_habitat_text_image_goal_nav.py.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip episode logs and images and keep only batch metric files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root (default: outputs/textnav_batch or outputs/instance_image_nav_batch, by task mode).",
    )
    parser.add_argument("--run-tag", default="", help="Optional run folder name.")
    parser.add_argument(
        "--no-aggregate-summary",
        action="store_true",
        help="Keep run/episode data without writing the output-root summary.tsv.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing run_tag and skip completed episodes.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing run_tag directory before running.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop the batch on the first timeout or crash.")
    parser.add_argument(
        "--execution-plan-capabilities",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--execution-plan",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--execution-result",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if args.episode_cpu_threads < 0:
        parser.error("--episode-cpu-threads must be >= 0")
    if args.max_gpu_workers_per_device < 0:
        parser.error("--max-gpu-workers-per-device must be >= 0")
    if args.retry_crashes < 0:
        parser.error("--retry-crashes must be >= 0")
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds must be >= 0")
    return args


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def as_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


def as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _required_episode_row_value(
    row: dict,
    field: str,
    *,
    label: str,
    row_number: int,
):
    value = row.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(
            f"invalid episodes.tsv row {row_number} for {label}: "
            f"missing {field!r}; use --overwrite to rerun it"
        )
    return value


def _strict_episode_row_float(
    row: dict,
    field: str,
    *,
    label: str,
    row_number: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = _required_episode_row_value(
        row,
        field,
        label=label,
        row_number=row_number,
    )
    try:
        number = None if isinstance(value, bool) else float(value)
    except (TypeError, ValueError, OverflowError):
        number = None
    if (
        number is None
        or not math.isfinite(number)
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        raise ValueError(
            f"invalid episodes.tsv row {row_number} for {label}: "
            f"malformed {field!r}={value!r}; use --overwrite to rerun it"
        )
    return number


def _strict_episode_row_int(
    row: dict,
    field: str,
    *,
    label: str,
    row_number: int,
    minimum: int = 0,
) -> int:
    number = _strict_episode_row_float(
        row,
        field,
        label=label,
        row_number=row_number,
        minimum=float(minimum),
    )
    if number != int(number):
        raise ValueError(
            f"invalid episodes.tsv row {row_number} for {label}: "
            f"{field!r} must be an integer; use --overwrite to rerun it"
        )
    return int(number)


def _optional_episode_row_flag(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number in {0.0, 1.0}:
            return bool(number)
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    try:
        number = float(text)
    except (TypeError, ValueError, OverflowError):
        return None
    if math.isfinite(number) and number in {0.0, 1.0}:
        return bool(number)
    return None


def _strict_episode_row_flag(
    row: dict,
    field: str,
    *,
    label: str,
    row_number: int,
) -> bool:
    value = _required_episode_row_value(
        row,
        field,
        label=label,
        row_number=row_number,
    )
    parsed = _optional_episode_row_flag(value)
    if parsed is not None:
        return parsed
    raise ValueError(
        f"invalid episodes.tsv row {row_number} for {label}: "
        f"{field!r} must be a binary flag; use --overwrite to rerun it"
    )


def validate_episode_tsv_row(
    row: dict,
    *,
    label: str,
    row_number: int,
) -> None:
    status_value = _required_episode_row_value(
        row,
        "status",
        label=label,
        row_number=row_number,
    )
    status = str(status_value).strip().lower()
    if status not in VALID_EPISODE_STATUSES:
        raise ValueError(
            f"invalid episodes.tsv row {row_number} for {label}: "
            f"unrecognized status {status_value!r}; use --overwrite to rerun it"
        )
    timed_out = _strict_episode_row_flag(
        row,
        "timed_out",
        label=label,
        row_number=row_number,
    )
    success = _strict_episode_row_flag(
        row,
        "success",
        label=label,
        row_number=row_number,
    )
    spl = _strict_episode_row_float(
        row,
        "spl",
        label=label,
        row_number=row_number,
        minimum=0.0,
        maximum=1.0,
    )
    if spl > float(success):
        raise ValueError(
            f"invalid episodes.tsv row {row_number} for {label}: "
            "SPL cannot exceed success; use --overwrite to rerun it"
        )
    _strict_episode_row_float(
        row,
        "soft_spl",
        label=label,
        row_number=row_number,
        minimum=0.0,
        maximum=1.0,
    )
    _strict_episode_row_float(
        row,
        "distance_to_goal",
        label=label,
        row_number=row_number,
        minimum=0.0,
    )
    _strict_episode_row_int(
        row,
        "steps",
        label=label,
        row_number=row_number,
    )
    _strict_episode_row_int(
        row,
        "seed",
        label=label,
        row_number=row_number,
    )
    _strict_episode_row_int(
        row,
        "max_steps",
        label=label,
        row_number=row_number,
        minimum=1,
    )
    success_radius = _strict_episode_row_float(
        row,
        "success_radius",
        label=label,
        row_number=row_number,
        minimum=0.0,
    )
    if success_radius <= 0.0:
        raise ValueError(
            f"invalid episodes.tsv row {row_number} for {label}: "
            "'success_radius' must be positive; use --overwrite to rerun it"
        )
    _strict_episode_row_float(
        row,
        "elapsed_seconds",
        label=label,
        row_number=row_number,
        minimum=0.0,
    )
    _strict_episode_row_flag(
        row,
        "target_found",
        label=label,
        row_number=row_number,
    )
    _strict_episode_row_flag(
        row,
        "stop_called",
        label=label,
        row_number=row_number,
    )
    if success != (status == "success"):
        raise ValueError(
            f"invalid episodes.tsv row {row_number} for {label}: "
            "status and success disagree; use --overwrite to rerun it"
        )
    if timed_out != (status == "timeout"):
        raise ValueError(
            f"invalid episodes.tsv row {row_number} for {label}: "
            "status and timed_out disagree; use --overwrite to rerun it"
        )


def compact_json(value) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_json_mapping(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        payload = json.loads(str(value or ""))
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sum_numeric_mapping(target: dict, source: dict) -> None:
    for key, value in source.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, dict):
            child = target.setdefault(str(key), {})
            if isinstance(child, dict):
                _sum_numeric_mapping(child, value)
            continue
        if isinstance(value, (int, float)):
            target[str(key)] = target.get(str(key), 0) + value


RETAINED_ACTION_SOURCES = frozenset(
    {
        "block_recovery",
        "budget_stop",
        "metric_visual_closure",
        "navigation_stop",
        "panorama_scan",
        "sgf_depth_guard",
        "sgf_explore",
        "sgf_scan",
        "sgf_stop",
        "stop_suppressed",
        "stop_validation_rejected",
        "text_sgf_target",
        "visual_pose_navigation",
        "visual_stop",
    }
)
RETAINED_VALIDATION_COUNTERS = frozenset(
    {
        "detection_probe_count",
        "image_vlm_panoramic_query_attempts",
        "image_vlm_panoramic_query_hits",
        "image_vlm_stop_validation_attempts",
        "image_vlm_stop_validation_rejections",
        "metric_visual_closure_armed_count",
        "metric_visual_closure_commit_count",
        "metric_visual_closure_contextual_closeup_commit_count",
        "metric_visual_closure_intervention_count",
        "metric_visual_closure_recovery_monotonic_commit_count",
        "stop_validation_attempts",
        "stop_validation_rejections",
    }
)
RETAINED_RECOVERY_COUNTERS = frozenset(
    {
        "block_recovery_action_count",
    }
)

# Counters every episode must report, zeros included
REQUIRED_FEATURE_ACTIVITY_KEYS = frozenset(
    {
        "detection_probe",
        "endgame_return",
        "image_vlm_stop_validation",
        "metric_visual_closure",
        "panoramic_observation",
        "parsed_target_phrases",
        "sgf_exploration",
        "sgf_explore_evidence_verification",
        "stop_ledger",
        "stop_validation",
        "target_fatigue",
        "text_sgf_stop",
        "validated_stop_approach",
        "visual_memory",
        "visual_pose_servo",
    }
)

# Counters aggregated across episodes, other keys are ignored
RETAINED_FEATURE_ACTIVITY_KEYS = REQUIRED_FEATURE_ACTIVITY_KEYS | frozenset(
    {
        "llm_goal_parser",
        "first_sentence_query",
        "context_guided_exploration",
        # Capability counters shared by all profiles
        "decision_stop",
        "target_pursuit_actions",
        # Frontier objective branches
        "sgf_explore_difference_selection",
        "sgf_explore_nearest_selection",
        "sgf_explore_rate_selection",
        "sgf_explore_rotation_priced_selection",
        # Text termination factors
        "text_evidence_rule_override",
        "text_face_node_turn",
        "text_passive_release",
        "text_reobservation_maneuver",
        "text_same_pose_requery",
        "text_same_vantage_skip",
        "text_stop_object_top_detection",
        "text_stop_object_whole_frame",
        "text_withheld_press",
        # Instance-image factors
        "image_depth_cluster_target",
        "image_goal_camera_target_steps",
        "image_goal_view_alignment_turns",
        "image_goal_view_single_corroboration_commit",
        "image_goal_view_wide_band_commit",
        "image_object_target_steps",
        "image_pursuit_fmm_actions",
        "image_straight_servo_actions",
        "image_goal_view_terminal",
        "image_reachable_pursuit",
        "image_vlm_foveated_hypothesis",
        "image_vlm_foveated_query",
        "image_vlm_hypothesis_arrival_sweep",
        "image_vlm_panoramic_query",
        "panoramic_detector_scaled_interval",
        "panoramic_searched_extent_interval",
        "text_hypothesis_triage",
        "text_node_grounded_evidence",
        "text_sequential_evidence_stop",
        # Shared representation and evidence-weighted ranking
        "navigation_graph_occupancy",
        "navigation_graph_surfel_votes",
        "navigation_graph_removed_free_cells",
        "navigation_graph_added_occupied_cells",
        "text_association_unreachable_candidates",
        "text_association_rank_opportunities",
        "text_association_rank_disagreements",
        "text_reachable_euclidean_association",
        # Navigation representation (cell counts are summed over frames)
        "navigation_floor_support",
        "navigation_floor_support_veto_cells",
        "navigation_floor_support_potential_veto_cells",
        "navigation_floor_support_nonfloor_cells",
        "text_query_conditional_retrieval",
        "text_argmax_label_retrieval",
        "text_detection_instances",
        "text_detection_instance_candidates",
        "text_detection_instance_selected_steps",
        "text_euclidean_association",
        "text_evidence_weighted_ranking",
        "text_navigation_map_association",
    }
)


def _only_keys(source: dict, retained: frozenset[str]) -> dict:
    return {
        str(key): value
        for key, value in source.items()
        if str(key) in retained
    }


def valid_feature_activity_value(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return bool(
        math.isfinite(number)
        and number >= 0.0
        and number == int(number)
    )


def telemetry_payload_has_complete_feature_activity(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    activity = payload.get("feature_activity")
    if not isinstance(activity, dict):
        return False
    if not REQUIRED_FEATURE_ACTIVITY_KEYS.issubset(activity):
        return False
    for key in REQUIRED_FEATURE_ACTIVITY_KEYS:
        if not valid_feature_activity_value(activity.get(key)):
            return False
    return True


RUNTIME_TIMING_MODULES = (
    "sgf_update",
    "image_goal",
    "planning_control",
    "stop_verification",
    "environment_step",
    "diagnostics",
    "other",
)
RUNTIME_TIMING_SGF_STAGES = (
    "inseg",
    "connectivity",
    "detection_wait_and_fusion",
    "navigation_free",
    "navigation_occupied",
)


def _runtime_timing_number(value: object, *, count: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0.0:
        return None
    if count and not number.is_integer():
        return None
    return number


def _runtime_timing_mean_matches(value: object, total: float, count: int) -> bool:
    if not count:
        return value is None
    number = _runtime_timing_number(value)
    return number is not None and math.isclose(
        number, total / count, rel_tol=1e-9, abs_tol=1e-6
    )


def valid_episode_runtime_timing(row: dict) -> dict | None:
    if str(row.get("status", "")).strip().lower() not in {"success", "failure"}:
        return None
    timing = parse_json_mapping(row.get("telemetry_json")).get("runtime_timing")
    if not isinstance(timing, dict):
        return None
    if _runtime_timing_number(timing.get("schema_version"), count=True) != 1:
        return None
    initialization = _runtime_timing_number(timing.get("initialization_seconds"))
    navigation = _runtime_timing_number(timing.get("navigation_seconds"))
    steps = _runtime_timing_number(timing.get("steps"), count=True)
    try:
        row_steps = None if isinstance(row.get("steps"), bool) else float(row.get("steps"))
    except (TypeError, ValueError, OverflowError):
        row_steps = None
    if initialization is None or navigation is None or steps is None or row_steps != steps:
        return None
    if not _runtime_timing_mean_matches(timing.get("mean_step_seconds"), navigation, int(steps)):
        return None
    modules = timing.get("modules")
    breakdown = timing.get("sgf_breakdown")
    if not isinstance(modules, dict) or not isinstance(breakdown, dict):
        return None
    module_seconds = []
    for key in RUNTIME_TIMING_MODULES:
        record = modules.get(key)
        if not isinstance(record, dict):
            return None
        total = _runtime_timing_number(record.get("total_seconds"))
        calls = _runtime_timing_number(record.get("calls"), count=True)
        if total is None or calls is None or (not calls and total != 0.0):
            return None
        if not _runtime_timing_mean_matches(record.get("mean_call_seconds"), total, int(calls)):
            return None
        module_seconds.append(total)
    if not math.isclose(sum(module_seconds), navigation, rel_tol=1e-9, abs_tol=1e-6):
        return None
    if modules["environment_step"]["calls"] != steps:
        return None
    for key in RUNTIME_TIMING_SGF_STAGES:
        record = breakdown.get(key)
        if not isinstance(record, dict):
            return None
        total = _runtime_timing_number(record.get("total_seconds"))
        frames = _runtime_timing_number(record.get("frames"), count=True)
        if total is None or frames is None or (not frames and total != 0.0):
            return None
        if frames > modules["sgf_update"]["calls"]:
            return None
        if not _runtime_timing_mean_matches(record.get("mean_frame_seconds"), total, int(frames)):
            return None
    return timing


def aggregate_runtime_timing(rows: list[dict]) -> dict:
    payloads = [parse_json_mapping(row.get("telemetry_json")) for row in rows]
    eligible = [
        (row, payload)
        for row, payload in zip(rows, payloads)
        if str(row.get("status", "")).strip().lower() in {"success", "failure"}
    ]
    timings = [
        timing for row, _ in eligible
        if (timing := valid_episode_runtime_timing(row)) is not None
    ]
    timed_count = len(timings)
    missing_count = sum("runtime_timing" not in payload for _, payload in eligible)
    initialization = sum(float(timing["initialization_seconds"]) for timing in timings)
    navigation = sum(float(timing["navigation_seconds"]) for timing in timings)
    steps = sum(int(timing["steps"]) for timing in timings)
    modules = {}
    for key in RUNTIME_TIMING_MODULES:
        total = sum(float(timing["modules"][key]["total_seconds"]) for timing in timings)
        calls = sum(int(timing["modules"][key]["calls"]) for timing in timings)
        modules[key] = {
            "total_seconds": total,
            "calls": calls,
            "episode_count": timed_count,
            "mean_seconds_per_episode": total / timed_count if timed_count else None,
            "mean_call_seconds": total / calls if calls else None,
        }
    breakdown = {}
    for key in RUNTIME_TIMING_SGF_STAGES:
        records = [timing["sgf_breakdown"][key] for timing in timings]
        total = sum(float(record["total_seconds"]) for record in records)
        frames = sum(int(record["frames"]) for record in records)
        breakdown[key] = {
            "total_seconds": total,
            "frames": frames,
            "episode_count": sum(int(record["frames"]) > 0 for record in records),
            "mean_frame_seconds": total / frames if frames else None,
        }
    return {
        "schema_version": 1,
        "episode_count": timed_count,
        "completed_episode_count": len(rows),
        "eligible_episode_count": len(eligible),
        "timed_episode_count": timed_count,
        "missing_episode_count": missing_count,
        "invalid_episode_count": len(eligible) - missing_count - timed_count,
        "excluded_episode_count": len(rows) - len(eligible),
        "total_initialization_seconds": initialization,
        "total_navigation_seconds": navigation,
        "mean_initialization_seconds": initialization / timed_count if timed_count else None,
        "mean_navigation_seconds": navigation / timed_count if timed_count else None,
        "total_steps": steps,
        "mean_step_seconds": navigation / steps if steps else None,
        "modules": modules,
        "sgf_breakdown": breakdown,
    }


def aggregate_ablation_telemetry(rows: list[dict]) -> dict:
    payloads = [
        payload
        for payload in (
            parse_json_mapping(row.get("telemetry_json"))
            for row in rows
        )
        if payload
    ]
    action_source_counts: dict = {}
    feature_activity_counts: dict = {}
    feature_activity_episode_counts: dict = {}
    feature_activity_invalid_episode_counts: dict = {}
    validation_counts: dict = {}
    recovery_counts: dict = {}
    target_reacquire_count = 0
    endgame_enabled_episodes = 0
    endgame_activated_episodes = 0
    endgame_replan_failures = 0
    for payload in payloads:
        _sum_numeric_mapping(
            action_source_counts,
            _only_keys(
                parse_json_mapping(payload.get("action_source_counts")),
                RETAINED_ACTION_SOURCES,
            ),
        )
        feature_activity = parse_json_mapping(payload.get("feature_activity"))
        for key in RETAINED_FEATURE_ACTIVITY_KEYS:
            if key not in feature_activity:
                continue
            value = feature_activity.get(key)
            if not valid_feature_activity_value(value):
                feature_activity_invalid_episode_counts[key] = (
                    feature_activity_invalid_episode_counts.get(key, 0) + 1
                )
                continue
            feature_activity_counts[key] = (
                feature_activity_counts.get(key, 0) + int(float(value))
            )
            feature_activity_episode_counts[key] = (
                feature_activity_episode_counts.get(key, 0) + 1
            )
        target_tracking = parse_json_mapping(payload.get("target_tracking"))
        if not target_tracking:
            target_tracking = parse_json_mapping(payload.get("evidence"))
        target_reacquire_count += as_int(
            target_tracking.get("target_reacquire_count"),
            0,
        )
        _sum_numeric_mapping(
            validation_counts,
            _only_keys(
                parse_json_mapping(payload.get("validation")),
                RETAINED_VALIDATION_COUNTERS,
            ),
        )
        _sum_numeric_mapping(
            recovery_counts,
            _only_keys(
                parse_json_mapping(payload.get("recovery")),
                RETAINED_RECOVERY_COUNTERS,
            ),
        )
        endgame = parse_json_mapping(payload.get("endgame_return"))
        endgame_enabled_episodes += int(
            bool(endgame.get("enabled", False))
        )
        endgame_activated_episodes += int(
            as_int(endgame.get("activated_step"), -1) >= 0
        )
        endgame_replan_failures += as_int(
            endgame.get("replan_failures"),
            0,
        )
    episode_count = sum(
        telemetry_payload_has_complete_feature_activity(payload)
        for payload in payloads
    )
    runtime_timing = aggregate_runtime_timing(rows)
    return {
        "runtime_timing": runtime_timing,
        "episode_count": episode_count,
        "telemetry_payload_count": len(payloads),
        "invalid_schema_episode_count": len(payloads) - episode_count,
        "action_source_counts": action_source_counts,
        "feature_activity": feature_activity_counts,
        "feature_activity_episode_counts": feature_activity_episode_counts,
        "feature_activity_invalid_episode_counts": (
            feature_activity_invalid_episode_counts
        ),
        "target_tracking": {
            "target_reacquire_count": target_reacquire_count,
        },
        "validation": validation_counts,
        "recovery": recovery_counts,
        "endgame_return": {
            "enabled_episodes": endgame_enabled_episodes,
            "activated_episodes": endgame_activated_episodes,
            "replan_failures": endgame_replan_failures,
        },
    }


def git_value(*parts: str) -> str:
    try:
        return subprocess.check_output(["git", *parts], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def normalize_repo_path(path: str | Path) -> str:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def parse_csv_values(value: str) -> list[str]:
    values = []
    for raw_part in str(value or "").split(","):
        part = raw_part.strip()
        if part:
            values.append(part)
    return values


def visible_gpu_ids_from_env() -> list[str]:
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_value is None:
        return []
    value = raw_value.strip()
    if value in {"", "-1"}:
        return []
    return parse_csv_values(value)


def detect_gpu_ids() -> list[str]:
    env_gpu_ids = visible_gpu_ids_from_env()
    if env_gpu_ids:
        return env_gpu_ids

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def resolve_gpu_ids(spec: str) -> list[str]:
    value = str(spec or "auto").strip().lower()
    if value in {"", "auto"}:
        return detect_gpu_ids()
    if value in {"none", "cpu", "off", "disabled"}:
        return []
    return parse_csv_values(spec)


def ordered_attempt_gpu_ids(
    gpu_ids: Sequence[str],
    initial_gpu_id: str | None,
    episode_index: int,
) -> list[str | None]:
    """Return a deterministic, load-spread GPU order for crash retries."""

    if initial_gpu_id is None:
        return [None]

    alternate_gpu_ids = list(
        dict.fromkeys(gpu_id for gpu_id in gpu_ids if gpu_id != initial_gpu_id)
    )
    if not alternate_gpu_ids:
        return [initial_gpu_id]

    offset = int(episode_index) % len(alternate_gpu_ids)
    alternate_gpu_ids = alternate_gpu_ids[offset:] + alternate_gpu_ids[:offset]
    return [initial_gpu_id, *alternate_gpu_ids]


def normalize_args_paths(args: argparse.Namespace) -> None:
    config_candidate = Path(args.config_path).expanduser()
    if config_candidate.is_absolute():
        args.config_path = str(config_candidate)
    else:
        repo_candidate = REPO_ROOT / config_candidate
        if repo_candidate.exists():
            args.config_path = str(repo_candidate.resolve())
    args.dataset_path = normalize_repo_path(args.dataset_path)
    args.textnav_dataset_path = normalize_repo_path(args.textnav_dataset_path)
    args.scenes_dir = normalize_repo_path(args.scenes_dir)
    args.runner_path = Path(normalize_repo_path(args.runner_path))
    if not args.nav_config_path:
        args.nav_config_path = str(DEFAULT_NAV_CONFIG_PATH)
    args.nav_config_path = normalize_repo_path(args.nav_config_path)
    if args.goal_image_path:
        args.goal_image_path = normalize_repo_path(args.goal_image_path)
    if args.output_root is None:
        args.output_root = (
            DEFAULT_INSTANCE_IMAGE_OUTPUT_ROOT
            if args.task_mode == "instance-image"
            else DEFAULT_TEXT_OUTPUT_ROOT
        )
    args.output_root = Path(normalize_repo_path(args.output_root))


def ensure_tsv_header(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()


def append_tsv_row(path: Path, columns: list[str], row: dict) -> None:
    ensure_tsv_header(path, columns)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writerow({key: row.get(key, "") for key in columns})


def read_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def read_tsv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_resume_episode_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        if fieldnames != EPISODE_COLUMNS:
            missing = sorted(set(EPISODE_COLUMNS) - set(fieldnames))
            unexpected = sorted(set(fieldnames) - set(EPISODE_COLUMNS))
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if unexpected:
                details.append("unexpected=" + ",".join(unexpected))
            if not missing and not unexpected:
                details.append("column order differs from the current schema")
            raise ValueError(
                f"incompatible episodes.tsv header for resume file {path}: "
                f"{'; '.join(details)}; use --overwrite to rerun it"
            )
        return list(reader)


def load_episode_metadata_for_split(args: argparse.Namespace, split: str, dataset_path: str) -> list[dict]:
    from habitat import make_dataset
    from habitat.config.default import get_config
    from omegaconf import OmegaConf

    config = get_config(config_path=args.config_path)
    OmegaConf.set_readonly(config, False)
    config.habitat.dataset.split = split
    config.habitat.dataset.data_path = dataset_path
    config.habitat.dataset.scenes_dir = args.scenes_dir
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)

    metadata = []
    for index, episode in enumerate(dataset.episodes):
        metadata.append(
            {
                "episode_index": index,
                "dataset_episode_id": str(getattr(episode, "episode_id", index)),
                "goal_key": str(getattr(episode, "goal_key", "")),
                "goal_name": str(getattr(episode, "object_category", "object")),
                "scene_id": str(getattr(episode, "scene_id", "")),
                "split": split,
                "dataset_path": dataset_path,
            }
        )
    return metadata


def load_episode_metadata(args: argparse.Namespace) -> list[dict]:
    return load_episode_metadata_for_split(args, args.split, args.dataset_path)


def parse_episode_spec(spec: str, total_episodes: int) -> list[int]:
    spec = spec.strip().lower()
    if not spec or spec == "all":
        return list(range(total_episodes))

    selected: list[int] = []
    seen: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid episode range: {part}")
            values = range(start, end + 1)
        else:
            values = [int(part)]

        for value in values:
            if value < 0 or value >= total_episodes:
                raise ValueError(f"episode index out of range: {value} not in [0, {total_episodes - 1}]")
            if value not in seen:
                selected.append(value)
                seen.add(value)
    return selected


def episode_task_key(episode_meta: dict) -> tuple[str, int]:
    return str(episode_meta.get("split", "")), as_int(episode_meta.get("episode_index"), default=-1)


def row_task_key(row: dict, default_split: str) -> tuple[str, int]:
    return str(row.get("split") or default_split), as_int(row.get("episode_index"), default=-1)


def load_completed_episode_keys(episodes_tsv_path: Path, default_split: str) -> set[tuple[str, int]]:
    completed: set[tuple[str, int]] = set()
    label = f"resume file {episodes_tsv_path}"
    for row_number, row in enumerate(
        read_resume_episode_rows(episodes_tsv_path),
        start=2,
    ):
        # Failed attempts also count as completed on resume
        for field in ("split", "dataset_episode_id", "scene_id"):
            _required_episode_row_value(
                row, field, label=label, row_number=row_number,
            )
        _strict_episode_row_int(
            row,
            "episode_index",
            label=label,
            row_number=row_number,
        )
        key = row_task_key(row, default_split)
        if key in completed:
            raise ValueError(
                f"duplicate episode key in {episodes_tsv_path} at TSV row "
                f"{row_number}: {key}; use --overwrite to rerun it"
            )
        completed.add(key)
    return completed


def make_run_tag(user_run_tag: str, commit: str) -> str:
    if user_run_tag:
        return user_run_tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{commit}"


def prepare_run_directories(args: argparse.Namespace, run_tag: str) -> dict[str, Path]:
    run_dir = args.output_root / "runs" / run_tag
    if run_dir.exists():
        if args.overwrite:
            shutil.rmtree(run_dir)
        elif not args.resume:
            raise FileExistsError(f"run directory already exists: {run_dir}. Use --resume or --overwrite.")
        else:
            recorded_modes: set[bool] = set()
            for metadata_name in ("summary.json", "manifest.json"):
                metadata = read_json_object(run_dir / metadata_name)
                if "metrics_only" in metadata:
                    recorded_modes.add(bool(metadata["metrics_only"]))
            episodes_dir = run_dir / "episodes"
            episodes_tsv = run_dir / "episodes.tsv"
            if episodes_dir.exists():
                recorded_modes.add(False)
            elif episodes_tsv.exists() and not recorded_modes:
                # Interrupted metrics-only run (no summary, no episodes/ dir)
                recorded_modes.add(True)
            if len(recorded_modes) > 1:
                raise ValueError(
                    f"cannot resume mixed metrics-only and artifact output in {run_dir}; "
                    "use --overwrite to start a clean run"
                )
            if recorded_modes and bool(args.metrics_only) not in recorded_modes:
                previous_mode = "metrics-only" if True in recorded_modes else "artifact"
                requested_mode = "metrics-only" if args.metrics_only else "artifact"
                raise ValueError(
                    f"cannot resume {run_dir} in {requested_mode} mode because it was "
                    f"created in {previous_mode} mode; use --overwrite"
                )

    episodes_dir = run_dir / "episodes"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.metrics_only:
        if episodes_dir.exists():
            raise ValueError(
                f"metrics-only run cannot reuse an episode artifact directory: {episodes_dir}; "
                "use --overwrite"
            )
    else:
        episodes_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "run_dir": run_dir,
        "episodes_dir": episodes_dir,
        "episodes_tsv": run_dir / "episodes.tsv",
        "summary_json": run_dir / "summary.json",
        "manifest_json": run_dir / "manifest.json",
        "global_summary_tsv": args.output_root / "summary.tsv",
    }
    return row


def terminate_process_group(proc: subprocess.Popen, grace_seconds: float) -> int | None:
    if proc.poll() is not None:
        return proc.returncode

    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return proc.poll()

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.poll()

    deadline = time.monotonic() + max(0.0, grace_seconds)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return proc.returncode
        time.sleep(0.2)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return proc.poll()

    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        pass
    return proc.poll()


def register_process(proc: subprocess.Popen) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.add(proc)


def unregister_process(proc: subprocess.Popen) -> None:
    with ACTIVE_PROCESSES_LOCK:
        ACTIVE_PROCESSES.discard(proc)


def sample_gpu_vram(proc: subprocess.Popen, output_path: Path) -> None:
    """Record whole-device VRAM once while the episode is still running."""

    if SHUTDOWN_REQUESTED.is_set() or proc.poll() is not None:
        return
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=GPU_VRAM_QUERY_TIMEOUT_SECONDS,
        )
        rows = [row for row in csv.reader(result.stdout.splitlines()) if row]
        if not rows:
            return
        with GPU_VRAM_WRITE_LOCK:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not output_path.exists() or output_path.stat().st_size == 0
            with output_path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if write_header:
                    writer.writerow(GPU_VRAM_COLUMNS)
                writer.writerows([[field.strip() for field in row] for row in rows])
    except (OSError, subprocess.SubprocessError, csv.Error) as error:
        print(f"GPU VRAM sampling failed for {output_path}: {error}", file=sys.stderr, flush=True)


def terminate_active_processes(grace_seconds: float) -> None:
    SHUTDOWN_REQUESTED.set()
    with ACTIVE_PROCESSES_LOCK:
        processes = list(ACTIVE_PROCESSES)
    for proc in processes:
        terminate_process_group(proc, grace_seconds=grace_seconds)


def episode_artifact_dir(episodes_dir: Path, episode_meta: dict) -> Path:
    episode_index = int(episode_meta["episode_index"])
    return episodes_dir / f"episode_{episode_index:04d}"


def build_episode_command(args: argparse.Namespace, episode_meta: dict, artifact_dir: Path, json_path: Path) -> list[str]:
    runner_args = shlex.split(args.runner_args)
    episode_index = int(episode_meta["episode_index"])
    split = str(episode_meta.get("split", args.split))
    command = [
        args.python_executable,
        str(args.runner_path),
        "--nav-config-path",
        args.nav_config_path,
        "--config-path",
        args.config_path,
        "--scenes-dir",
        args.scenes_dir,
        "--episode-id",
        str(episode_index),
        "--max-steps",
        str(args.max_steps),
        "--artifact-dir",
        str(artifact_dir),
    ]
    if not args.metrics_only:
        command.extend(["--output-json", str(json_path)])
    command.extend(
        [
            "--dataset-path",
            str(episode_meta.get("dataset_path", args.dataset_path)),
            "--goal-source",
            args.goal_source,
            "--textnav-dataset-path",
            args.textnav_dataset_path,
            "--split",
            split,
            "--task-mode",
            args.task_mode,
        ]
    )
    if args.goal_image_path:
        command.extend(["--goal-image-path", args.goal_image_path])
    if not args.metrics_only and not args.quiet_runner:
        command.append("--verbose")
    command.extend(runner_args)
    if args.metrics_only:
        command.extend(["--metrics-only", "--output-json", str(json_path)])
    return command


def build_episode_environment(args: argparse.Namespace, gpu_id: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["SGF_ASSIGNED_GPU"] = str(gpu_id)
    if args.episode_cpu_threads <= 0:
        env["SGF_EPISODE_CPU_THREADS"] = "0"
        return env

    threads = str(args.episode_cpu_threads)
    env["SGF_EPISODE_CPU_THREADS"] = threads
    env.setdefault("SGF_EPISODE_TORCH_INTEROP_THREADS", "1")
    for name in EPISODE_THREAD_ENV_VARS:
        env[name] = threads
    env.setdefault("OMP_DYNAMIC", "FALSE")
    env.setdefault("MKL_DYNAMIC", "FALSE")
    env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    env.setdefault("KMP_BLOCKTIME", "0")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def load_result_payload(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def episode_payload_has_required_metrics(payload: object) -> bool:
    """Whether the payload has valid values for every summarized metric."""

    if not isinstance(payload, dict):
        return False
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return False

    def finite_number(
        value: object,
        *,
        allow_bool: bool = False,
    ) -> float | None:
        if isinstance(value, bool):
            return float(value) if allow_bool else None
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return parsed if math.isfinite(parsed) else None

    success = finite_number(metrics.get("success"), allow_bool=True)
    spl = finite_number(metrics.get("spl"))
    soft_spl = finite_number(metrics.get("soft_spl"))
    distance = finite_number(metrics.get("distance_to_goal"))
    if (
        success is None
        or success not in {0.0, 1.0}
        or spl is None
        or not 0.0 <= spl <= 1.0
        or spl > success
        or soft_spl is None
        or not 0.0 <= soft_spl <= 1.0
        or distance is None
        or distance < 0.0
    ):
        return False

    for key, minimum, strictly_positive in (
        ("seed", 0, False),
        ("steps", 0, False),
        ("max_steps", 0, True),
    ):
        number = finite_number(payload.get(key))
        if (
            number is None
            or number != int(number)
            or number < minimum
            or (strictly_positive and number <= minimum)
        ):
            return False

    success_radius = finite_number(payload.get("success_radius"))
    if success_radius is None or success_radius <= 0.0:
        return False
    for key in ("target_found", "stop_called"):
        value = payload.get(key)
        if not (
            isinstance(value, bool)
            or (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) in {0.0, 1.0}
            )
        ):
            return False
    return True


def result_artifact_path(
    payload: dict | None,
    key: str,
    *,
    fallback_paths: Sequence[Path],
    keep_missing_payload_value: bool = False,
) -> str:
    """Return an artifact path reported by the runner, with checked fallbacks."""

    if isinstance(payload, dict):
        value = str(payload.get(key) or "").strip()
        if value:
            payload_path = Path(value).expanduser()
            if payload_path.exists() or keep_missing_payload_value:
                return str(payload_path)
    for path in fallback_paths:
        if path.exists():
            return str(path)
    return ""


def _run_episode_attempt_in_dir(
    args: argparse.Namespace,
    run_tag: str,
    commit: str,
    branch: str,
    episode_meta: dict,
    artifact_dir: Path,
    *,
    attempt: int,
    gpu_id: str | None,
) -> dict:
    episode_index = episode_meta["episode_index"]
    if not args.metrics_only:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "diagnostics.log"
    subprocess_log_path = artifact_dir / ("subprocess.log" if attempt == 1 else f"subprocess_attempt_{attempt}.log")
    json_path = artifact_dir / "result.json"
    # Remove a stale result from an earlier attempt
    json_path.unlink(missing_ok=True)
    final_rgb_path = artifact_dir / "final_rgb.png"
    final_navigation_map_path = artifact_dir / "final_navigation_map.png"
    final_image_match_path = artifact_dir / "final_image_match.png"
    image_match_video_path = artifact_dir / "image_matches.mp4"
    command = build_episode_command(args, episode_meta, artifact_dir, json_path)
    env = build_episode_environment(args, gpu_id=gpu_id)

    started_at = utc_now_iso()
    wall_start = time.monotonic()
    timed_out = False
    elapsed_seconds = 0.0
    returncode: int | None

    def execute_command(stdout_target, stderr_target) -> int | None:
        nonlocal timed_out, elapsed_seconds
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
            start_new_session=True,
            env=env,
        )
        register_process(proc)
        sampler = None
        if args.gpu_vram_sampling:
            sampler = threading.Timer(
                GPU_VRAM_SAMPLE_DELAY_SECONDS,
                sample_gpu_vram,
                args=(proc, args.output_root / "runs" / run_tag / "gpu_vram_10s.csv"),
            )
            sampler.daemon = True
            sampler.start()
        try:
            try:
                returncode = proc.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                if sampler is not None:
                    sampler.cancel()
                returncode = terminate_process_group(proc, grace_seconds=args.kill_grace_seconds)
                if returncode is None:
                    returncode = 124
            except KeyboardInterrupt:
                SHUTDOWN_REQUESTED.set()
                terminate_process_group(proc, grace_seconds=args.kill_grace_seconds)
                raise
        finally:
            # Keep sampler cleanup out of the episode's elapsed time.
            elapsed_seconds = round(time.monotonic() - wall_start, 3)
            if sampler is not None:
                sampler.cancel()
                sampler.join()
            unregister_process(proc)
        return returncode

    if args.metrics_only:
        returncode = execute_command(subprocess.DEVNULL, subprocess.DEVNULL)
    else:
        with subprocess_log_path.open("w", encoding="utf-8") as subprocess_log:
            subprocess_log.write(f"attempt={attempt}\n")
            if gpu_id is not None:
                subprocess_log.write(f"assigned_gpu={gpu_id}\n")
            subprocess_log.write(f"command={shlex.join(command)}\n\n")
            subprocess_log.flush()
            returncode = execute_command(subprocess_log, subprocess.STDOUT)

    payload = load_result_payload(json_path)
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    metrics_payload_valid = episode_payload_has_required_metrics(payload)
    reported_success = bool(
        metrics_payload_valid
        and as_float(metrics.get("success"), 0.0) > 0.0
    )

    if timed_out:
        status = "timeout"
    elif SHUTDOWN_REQUESTED.is_set() and returncode not in (0, None) and payload is None:
        status = "interrupted"
    elif returncode not in (0, None):
        status = "crash"
    elif payload is None:
        status = "missing_json"
    elif not metrics_payload_valid:
        status = "invalid_metrics"
    elif reported_success:
        status = "success"
    else:
        status = "failure"
    metrics_are_valid = status in {"success", "failure"}
    success = int(status == "success")
    telemetry = (
        payload.get("ablation_telemetry", {})
        if isinstance(payload, dict)
        else {}
    )
    if not isinstance(telemetry, dict):
        telemetry = {}

    row = {
        "timestamp": started_at,
        "run_tag": run_tag,
        "task_mode": args.task_mode,
        "commit": commit,
        "branch": branch,
        "split": episode_meta.get("split", args.split),
        "seed": (
            as_int(payload.get("seed"), 0)
            if isinstance(payload, dict) and payload.get("seed") is not None
            else ""
        ),
        "episode_index": episode_index,
        "dataset_episode_id": episode_meta["dataset_episode_id"],
        "goal_key": episode_meta["goal_key"],
        "goal_name": episode_meta["goal_name"],
        "goal_image_source": payload.get("goal_image_source", "") if isinstance(payload, dict) else "",
        "scene_id": episode_meta["scene_id"],
        "status": status,
        "returncode": "" if returncode is None else returncode,
        "timed_out": int(timed_out),
        "elapsed_seconds": elapsed_seconds,
        "success": success,
        "spl": round(as_float(metrics.get("spl"), 0.0), 6) if metrics_are_valid else 0.0,
        "soft_spl": round(as_float(metrics.get("soft_spl"), 0.0), 6) if metrics_are_valid else 0.0,
        "distance_to_goal": (
            round(as_float(metrics.get("distance_to_goal"), 0.0), 6)
            if metrics_are_valid
            else 0.0
        ),
        "max_steps": (
            as_int(payload.get("max_steps"), args.max_steps)
            if isinstance(payload, dict)
            else args.max_steps
        ),
        "success_radius": (
            round(as_float(payload.get("success_radius"), 0.0), 6)
            if isinstance(payload, dict)
            and payload.get("success_radius") is not None
            else ""
        ),
        "steps": as_int(payload.get("steps"), 0) if isinstance(payload, dict) else 0,
        "target_found": int(bool(payload.get("target_found"))) if isinstance(payload, dict) else 0,
        "stop_called": int(bool(payload.get("stop_called"))) if isinstance(payload, dict) else 0,
        "telemetry_json": compact_json(telemetry) if telemetry else "",
        "artifact_dir": "" if args.metrics_only else str(artifact_dir),
        "log_path": (
            ""
            if args.metrics_only
            else str(log_path if log_path.exists() else subprocess_log_path)
        ),
        "json_path": "" if args.metrics_only else str(json_path),
        "final_image_path": (
            ""
            if args.metrics_only
            else result_artifact_path(
                payload,
                "final_navigation_map_path",
                fallback_paths=(final_navigation_map_path, final_rgb_path),
            )
        ),
        "final_image_match_path": (
            ""
            if args.metrics_only
            else result_artifact_path(
                payload,
                "final_image_match_path",
                fallback_paths=(final_image_match_path,),
            )
        ),
        "image_match_video_path": (
            ""
            if args.metrics_only
            else result_artifact_path(
                payload,
                "image_match_video_path",
                fallback_paths=(image_match_video_path,),
            )
        ),
    }
    return row


def run_episode_attempt(
    args: argparse.Namespace,
    run_tag: str,
    commit: str,
    branch: str,
    episode_meta: dict,
    directories: dict[str, Path],
    *,
    attempt: int,
    gpu_id: str | None,
) -> dict:
    if args.metrics_only:
        with tempfile.TemporaryDirectory(prefix="sgf_nav_metrics_") as temporary_dir:
            return _run_episode_attempt_in_dir(
                args=args,
                run_tag=run_tag,
                commit=commit,
                branch=branch,
                episode_meta=episode_meta,
                artifact_dir=Path(temporary_dir),
                attempt=attempt,
                gpu_id=gpu_id,
            )

    return _run_episode_attempt_in_dir(
        args=args,
        run_tag=run_tag,
        commit=commit,
        branch=branch,
        episode_meta=episode_meta,
        artifact_dir=episode_artifact_dir(directories["episodes_dir"], episode_meta),
        attempt=attempt,
        gpu_id=gpu_id,
    )


def run_episode(
    args: argparse.Namespace,
    run_tag: str,
    commit: str,
    branch: str,
    episode_meta: dict,
    directories: dict[str, Path],
    gpu_id: str | None = None,
    gpu_attempt_slots: dict[str, threading.BoundedSemaphore] | None = None,
) -> dict:
    max_attempts = max(1, int(args.retry_crashes) + 1)
    final_row: dict | None = None
    attempt_gpu_ids = ordered_attempt_gpu_ids(
        getattr(args, "resolved_gpu_ids", []),
        initial_gpu_id=gpu_id,
        episode_index=as_int(episode_meta.get("episode_index"), default=0),
    )
    for attempt in range(1, max_attempts + 1):
        attempt_gpu_id = attempt_gpu_ids[(attempt - 1) % len(attempt_gpu_ids)]
        attempt_slot = (
            (gpu_attempt_slots or {}).get(attempt_gpu_id)
            if attempt_gpu_id is not None
            else None
        )
        if attempt_slot is not None:
            attempt_slot.acquire()
        try:
            if SHUTDOWN_REQUESTED.is_set() and final_row is not None:
                return final_row
            row = run_episode_attempt(
                args=args,
                run_tag=run_tag,
                commit=commit,
                branch=branch,
                episode_meta=episode_meta,
                directories=directories,
                attempt=attempt,
                gpu_id=attempt_gpu_id,
            )
        finally:
            if attempt_slot is not None:
                attempt_slot.release()
        final_row = row
        if SHUTDOWN_REQUESTED.is_set():
            return row
        if row["status"] not in RETRYABLE_STATUSES or attempt >= max_attempts:
            return row

        delay_seconds = float(args.retry_delay_seconds)
        next_gpu_id = attempt_gpu_ids[attempt % len(attempt_gpu_ids)]
        if not args.metrics_only:
            print(
                f"retrying episode={row['episode_index']} after status={row['status']} "
                f"attempt={attempt}/{max_attempts} delay={delay_seconds:.1f}s "
                f"next_gpu={next_gpu_id if next_gpu_id is not None else 'none'}"
            )
            sys.stdout.flush()
        if delay_seconds > 0.0:
            if SHUTDOWN_REQUESTED.wait(delay_seconds):
                return row

    if final_row is None:
        raise RuntimeError("run_episode finished without producing a row")
    return final_row


def build_summary(rows: list[dict], selected_episode_count: int, run_tag: str, commit: str, branch: str, args: argparse.Namespace, run_dir: Path) -> dict:
    completed = len(rows)
    pending = max(0, selected_episode_count - completed)
    valid_metric_rows = []
    for row_number, row in enumerate(rows, start=2):
        if str(row.get("status", "")).strip().lower() not in {"success", "failure"}:
            continue
        if str(row.get("returncode", "")).strip() and as_float(row["returncode"], -1) != 0:
            continue
        try:
            validate_episode_tsv_row(row, label=run_tag, row_number=row_number)
        except ValueError:
            continue
        valid_metric_rows.append(row)
    successes = sum(
        _optional_episode_row_flag(row.get("success")) is True
        for row in valid_metric_rows
    )
    valid_metric_count = len(valid_metric_rows)
    mean_spl = sum(as_float(row.get("spl"), 0.0) for row in valid_metric_rows) / selected_episode_count if selected_episode_count else 0.0
    mean_soft_spl = (
        sum(as_float(row.get("soft_spl"), 0.0) for row in valid_metric_rows) / selected_episode_count if selected_episode_count else 0.0
    )
    mean_distance_to_goal = (
        sum(
            as_float(row.get("distance_to_goal"), 0.0)
            for row in valid_metric_rows
        )
        / valid_metric_count
        if valid_metric_count
        else None
    )
    mean_steps = (
        sum(as_int(row.get("steps"), 0) for row in valid_metric_rows)
        / valid_metric_count
        if valid_metric_count
        else None
    )
    mean_elapsed_seconds = (
        sum(as_float(row.get("elapsed_seconds"), 0.0) for row in rows)
        / selected_episode_count
        if selected_episode_count
        else 0.0
    )
    target_found = sum(
        _optional_episode_row_flag(row.get("target_found")) is True
        for row in rows
    )
    stop_called = sum(
        _optional_episode_row_flag(row.get("stop_called")) is True
        for row in rows
    )
    seeds = sorted(
        {
            int(float(row["seed"]))
            for row in rows
            if math.isfinite(as_float(row.get("seed"), math.nan))
            and as_float(row["seed"]) >= 0
            and as_float(row["seed"]).is_integer()
        }
    )
    seed: int | str = (
        seeds[0]
        if len(seeds) == 1
        else ("mixed" if seeds else "")
    )
    success_radii = sorted(
        {
            round(as_float(row.get("success_radius")), 6)
            for row in rows
            if str(row.get("success_radius", "")).strip()
        }
    )
    success_radius: float | str = (
        success_radii[0]
        if len(success_radii) == 1
        else ("mixed" if success_radii else "")
    )
    max_steps_values = sorted(
        {
            int(float(row["max_steps"]))
            for row in rows
            if math.isfinite(as_float(row.get("max_steps"), math.nan))
            and as_float(row["max_steps"]) > 0
            and as_float(row["max_steps"]).is_integer()
        }
    )
    max_steps: int | str = (
        max_steps_values[0]
        if len(max_steps_values) == 1
        else ("mixed" if max_steps_values else args.max_steps)
    )
    telemetry = aggregate_ablation_telemetry(rows)
    telemetry_json = compact_json(telemetry)
    timeouts = sum(
        1
        for row in rows
        if str(row.get("status", "")).strip().lower() == "timeout"
    )
    crashes = sum(
        1
        for row in rows
        if str(row.get("status", "")).strip().lower()
        in {"crash", "missing_json", "invalid_metrics"}
    )
    failures = sum(
        1
        for row in rows
        if str(row.get("status", "")).strip().lower() == "failure"
    )
    status = "completed" if pending == 0 else "partial"
    return {
        "timestamp": utc_now_iso(),
        "run_tag": run_tag,
        "task_mode": args.task_mode,
        "branch": branch,
        "commit": commit,
        "split": args.split,
        "seed": seed,
        "episodes": selected_episode_count,
        "completed": completed,
        "pending": pending,
        "successes": successes,
        "success_rate": round(successes / selected_episode_count, 6) if selected_episode_count else 0.0,
        "mean_spl": round(mean_spl, 6),
        "mean_soft_spl": round(mean_soft_spl, 6),
        "mean_distance_to_goal": (
            None
            if mean_distance_to_goal is None
            else round(mean_distance_to_goal, 6)
        ),
        "mean_steps": (
            None if mean_steps is None else round(mean_steps, 6)
        ),
        "distance_steps_valid_episodes": valid_metric_count,
        "mean_elapsed_seconds": round(mean_elapsed_seconds, 6),
        "target_found_rate": (
            round(target_found / selected_episode_count, 6)
            if selected_episode_count
            else 0.0
        ),
        "stop_called_rate": (
            round(stop_called / selected_episode_count, 6)
            if selected_episode_count
            else 0.0
        ),
        "failures": failures,
        "timeouts": timeouts,
        "crashes": crashes,
        "max_steps": max_steps,
        "success_radius": success_radius,
        "timeout_seconds": args.timeout_seconds,
        "metrics_only": bool(args.metrics_only),
        "telemetry_json": telemetry_json,
        "runtime_timing": telemetry["runtime_timing"],
        "status": status,
        "run_dir": str(run_dir),
    }


def print_progress(index: int, total: int, row: dict) -> None:
    timing = valid_episode_runtime_timing(row)
    timing_text = ""
    if timing is not None:
        timing_text = f" nav={timing['navigation_seconds']:.1f}s"
        if timing["mean_step_seconds"] is not None:
            timing_text += f" per_step={timing['mean_step_seconds']:.3f}s"
    print(
        f"[{index}/{total}] mode={row.get('task_mode', '')} split={row.get('split', '')} "
        f"episode={row['episode_index']} status={row['status']} "
        f"success={row['success']} spl={row['spl']:.3f} "
        f"steps={row['steps']} elapsed={row['elapsed_seconds']:.1f}s{timing_text}"
    )
    if row.get("log_path"):
        print(f"  log={row['log_path']}")
    if row["status"] in {"crash", "timeout", "missing_json"} and row.get("json_path"):
        print(f"  json={row['json_path']}")
    sys.stdout.flush()


@dataclass
class RunContext:
    """All mutable state for one batch invocation in a shared scheduler."""

    name: str
    mode: str
    args: argparse.Namespace
    run_tag: str
    commit: str
    branch: str
    metadata: list[dict]
    directories: dict[str, Path]
    selected_episode_ids: list[int]
    selected_episode_keys: list[tuple[str, int]]
    pending_episode_ids: list[int]
    next_position: int = 0
    completed_count: int = 0
    in_flight: int = 0
    scheduling_stopped: bool = False
    fail_fast_triggered: bool = False
    fail_fast_reported: bool = False
    finalization_attempted: bool = False
    summary: dict | None = None


@dataclass
class PlannedRun:
    """A lazily prepared run from an execution-plan entry."""

    mode: str
    name: str
    argv: list[str]
    args: argparse.Namespace
    context: RunContext | None = None
    setup_attempted: bool = False
    state: str = "not_started"
    returncode: int | None = None
    error: str | None = None


class ExecutionPlanInterrupted(KeyboardInterrupt):
    """Carry a resumable partial execution result across Ctrl-C cleanup."""

    def __init__(self, result: dict[str, Any]):
        super().__init__("execution plan interrupted")
        self.result = result


def normalize_batch_args(args: argparse.Namespace) -> argparse.Namespace:
    if not bool(getattr(args, "_batch_args_normalized", False)):
        normalize_args_paths(args)
        args.resolved_gpu_ids = resolve_gpu_ids(args.gpu_ids)
        args._batch_args_normalized = True
    return args


def scheduler_details(args: argparse.Namespace) -> dict[str, Any]:
    gpu_ids = list(getattr(args, "resolved_gpu_ids", []))
    requested_workers = int(args.num_workers)
    gpu_cap = int(args.max_gpu_workers_per_device)
    effective_max_workers = requested_workers
    if gpu_ids and gpu_cap > 0:
        effective_max_workers = min(
            effective_max_workers,
            len(gpu_ids) * gpu_cap,
        )
    return {
        "requested_workers": requested_workers,
        "effective_max_workers": effective_max_workers,
        "resolved_gpu_ids": gpu_ids,
        "max_gpu_workers_per_device": gpu_cap,
    }


def _manifest_for_context(
    *,
    args: argparse.Namespace,
    run_tag: str,
    commit: str,
    branch: str,
    selected_episode_ids: list[int],
    selected_episode_keys: list[tuple[str, int]],
) -> dict:
    return {
        "timestamp": utc_now_iso(),
        "run_tag": run_tag,
        "commit": commit,
        "branch": branch,
        "repo_root": str(REPO_ROOT),
        "runner_path": str(args.runner_path),
        "python_executable": args.python_executable,
        "task_mode": args.task_mode,
        "nav_config_path": args.nav_config_path,
        "config_path": args.config_path,
        "dataset_path": args.dataset_path,
        "goal_source": args.goal_source,
        "textnav_dataset_path": args.textnav_dataset_path,
        "goal_image_path": args.goal_image_path,
        "scenes_dir": args.scenes_dir,
        "split": args.split,
        "episodes_spec": args.episodes,
        "selected_episodes": selected_episode_ids,
        "selected_episode_keys": [
            {"split": split, "episode_index": episode_index}
            for split, episode_index in selected_episode_keys
        ],
        "num_workers": args.num_workers,
        "effective_max_workers": scheduler_details(args)["effective_max_workers"],
        "episode_cpu_threads": args.episode_cpu_threads,
        "gpu_ids": args.gpu_ids,
        "resolved_gpu_ids": args.resolved_gpu_ids,
        "max_gpu_workers_per_device": args.max_gpu_workers_per_device,
        "gpu_vram_sampling": args.gpu_vram_sampling,
        "retry_crashes": args.retry_crashes,
        "retry_delay_seconds": args.retry_delay_seconds,
        "max_steps": args.max_steps,
        "timeout_seconds": args.timeout_seconds,
        "runner_args": args.runner_args,
        "quiet_runner": args.quiet_runner,
        "metrics_only": args.metrics_only,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "fail_fast": args.fail_fast,
        "argv": list(getattr(args, "_invocation_argv", sys.argv)),
    }


def prepare_run_context(
    args: argparse.Namespace,
    mode: str | None = None,
    name: str | None = None,
    *,
    commit: str | None = None,
    branch: str | None = None,
) -> RunContext:
    normalize_batch_args(args)
    mode = str(mode or args.task_mode)
    name = str(name or args.run_tag or mode)
    if args.task_mode != mode:
        raise ValueError(
            f"execution-plan mode {mode!r} does not match run --task-mode "
            f"{args.task_mode!r} for {name!r}"
        )

    commit = commit or git_value("rev-parse", "--short", "HEAD")
    branch = branch if branch is not None else git_value("branch", "--show-current")
    run_tag = make_run_tag(args.run_tag, commit)
    directories = prepare_run_directories(args, run_tag)
    metadata = load_episode_metadata(args)
    selected_episode_ids = parse_episode_spec(args.episodes, len(metadata))
    if args.limit > 0:
        selected_episode_ids = selected_episode_ids[: args.limit]

    completed_episode_keys: set[tuple[str, int]] = set()
    if args.resume:
        completed_episode_keys = load_completed_episode_keys(
            directories["episodes_tsv"],
            default_split=args.split,
        )
    selected_episode_keys = [
        episode_task_key(metadata[episode_id])
        for episode_id in selected_episode_ids
    ]
    pending_episode_ids = [
        episode_id
        for episode_id in selected_episode_ids
        if episode_task_key(metadata[episode_id]) not in completed_episode_keys
    ]

    manifest = _manifest_for_context(
        args=args,
        run_tag=run_tag,
        commit=commit,
        branch=branch,
        selected_episode_ids=selected_episode_ids,
        selected_episode_keys=selected_episode_keys,
    )
    if not args.metrics_only:
        directories["manifest_json"].write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    ensure_tsv_header(directories["episodes_tsv"], EPISODE_COLUMNS)
    if not args.no_aggregate_summary:
        ensure_tsv_header(directories["global_summary_tsv"], SUMMARY_COLUMNS)

    if not args.metrics_only:
        details = scheduler_details(args)
        print(
            f"run_tag={run_tag} task_mode={args.task_mode} branch={branch} "
            f"commit={commit} split={args.split} selected={len(selected_episode_ids)} "
            f"pending={len(pending_episode_ids)} workers={args.num_workers} "
            f"effective_workers={details['effective_max_workers']} "
            f"episode_cpu_threads={args.episode_cpu_threads} "
            f"gpu_ids={','.join(args.resolved_gpu_ids) if args.resolved_gpu_ids else 'none'} "
            f"max_gpu_workers_per_device={args.max_gpu_workers_per_device} "
            f"retry_crashes={args.retry_crashes} output_dir={directories['run_dir']}"
        )
        sys.stdout.flush()

    return RunContext(
        name=name,
        mode=mode,
        args=args,
        run_tag=run_tag,
        commit=commit,
        branch=branch,
        metadata=metadata,
        directories=directories,
        selected_episode_ids=selected_episode_ids,
        selected_episode_keys=selected_episode_keys,
        pending_episode_ids=pending_episode_ids,
    )


def finalize_run_context(context: RunContext) -> dict:
    """Write a run's summary after every submitted episode has settled."""

    rows = read_tsv_rows(context.directories["episodes_tsv"])
    selected_row_set = set(context.selected_episode_keys)
    filtered_rows = [
        row
        for row in rows
        if row_task_key(row, default_split=context.args.split)
        in selected_row_set
    ]
    summary = build_summary(
        rows=filtered_rows,
        selected_episode_count=len(context.selected_episode_ids),
        run_tag=context.run_tag,
        commit=context.commit,
        branch=context.branch,
        args=context.args,
        run_dir=context.directories["run_dir"],
    )
    context.directories["summary_json"].write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if not context.args.no_aggregate_summary:
        append_tsv_row(
            context.directories["global_summary_tsv"],
            SUMMARY_COLUMNS,
            summary,
        )
    context.summary = summary
    print(json.dumps(summary, indent=2))
    return summary


def _validate_shared_scheduler_settings(specs: Sequence[PlannedRun]) -> dict:
    if not specs:
        raise ValueError("execution plan must contain at least one run")
    first = scheduler_details(specs[0].args)
    for spec in specs[1:]:
        current = scheduler_details(spec.args)
        if current != first:
            raise ValueError(
                "all execution-plan runs must use the same --num-workers, "
                "resolved --gpu-ids, and --max-gpu-workers-per-device; "
                f"{spec.name!r} has {current}, expected {first}"
            )
    return first


def _failure_text(error: BaseException | str) -> str:
    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"
    return str(error)


def _record_run_failure(spec: PlannedRun, error: BaseException | str) -> None:
    message = _failure_text(error)
    if spec.error:
        spec.error = f"{spec.error}; {message}"
    else:
        spec.error = message
    spec.state = "failed"
    spec.returncode = 1
    if spec.context is not None:
        spec.context.scheduling_stopped = True


def _scheduler_result(specs: Sequence[PlannedRun], details: dict) -> dict:
    return {
        "version": EXECUTION_PLAN_VERSION,
        "scheduler": details,
        "runs": [
            {
                "mode": spec.mode,
                "name": spec.name,
                "state": spec.state,
                "returncode": spec.returncode,
                "error": spec.error,
            }
            for spec in specs
        ],
    }


def _run_planned_specs(
    specs: list[PlannedRun],
    *,
    keep_going: bool,
    commit: str,
    branch: str,
) -> dict:
    details = _validate_shared_scheduler_settings(specs)
    worker_count = int(details["effective_max_workers"])
    gpu_ids = list(details["resolved_gpu_ids"])
    gpu_cap = int(details["max_gpu_workers_per_device"])
    gpu_assignment_enabled = bool(gpu_ids)
    gpu_slots_enabled = gpu_assignment_enabled and gpu_cap > 0
    gpu_inflight = {gpu_id: 0 for gpu_id in gpu_ids}
    gpu_attempt_slots = (
        {gpu_id: threading.BoundedSemaphore(gpu_cap) for gpu_id in gpu_ids}
        if gpu_slots_enabled
        else {}
    )
    gpu_cursor = 0
    global_stop = False

    def claim_gpu_slot() -> str | None | bool:
        nonlocal gpu_cursor
        if not gpu_assignment_enabled:
            return None
        if not gpu_slots_enabled:
            candidate = gpu_ids[gpu_cursor % len(gpu_ids)]
            gpu_cursor = (gpu_cursor + 1) % len(gpu_ids)
            return candidate
        for offset in range(len(gpu_ids)):
            index = (gpu_cursor + offset) % len(gpu_ids)
            candidate = gpu_ids[index]
            if gpu_inflight[candidate] < gpu_cap:
                gpu_inflight[candidate] += 1
                gpu_cursor = (index + 1) % len(gpu_ids)
                return candidate
        return False

    def release_gpu_slot(gpu_id: str | None) -> None:
        if gpu_id is not None and gpu_id in gpu_inflight:
            gpu_inflight[gpu_id] = max(0, gpu_inflight[gpu_id] - 1)

    def fail_spec(spec: PlannedRun, error: BaseException | str) -> None:
        nonlocal global_stop
        _record_run_failure(spec, error)
        if not keep_going:
            global_stop = True
            for planned in specs:
                if planned.context is not None:
                    planned.context.scheduling_stopped = True

    def finalize_if_ready(spec: PlannedRun) -> bool:
        context = spec.context
        if (
            context is None
            or context.finalization_attempted
            or context.in_flight > 0
            or (
                context.next_position < len(context.pending_episode_ids)
                and not context.scheduling_stopped
            )
        ):
            return False
        context.finalization_attempted = True
        try:
            finalize_run_context(context)
        except Exception as error:
            fail_spec(spec, error)
        else:
            if spec.state != "failed":
                never_started = (
                    bool(context.pending_episode_ids)
                    and context.next_position == 0
                    and context.scheduling_stopped
                )
                stopped_with_unsubmitted_work = (
                    context.scheduling_stopped
                    and not context.fail_fast_triggered
                    and context.next_position
                    < len(context.pending_episode_ids)
                )
                if stopped_with_unsubmitted_work and not never_started:
                    spec.state = "failed"
                    spec.returncode = 1
                    spec.error = "stopped after an earlier batch failure"
                elif not never_started:
                    spec.state = "completed"
                    spec.returncode = 0
                    spec.error = None
        return True

    def prepare_spec(spec: PlannedRun) -> None:
        if spec.setup_attempted:
            return
        spec.setup_attempted = True
        try:
            spec.context = prepare_run_context(
                spec.args,
                mode=spec.mode,
                name=spec.name,
                commit=commit,
                branch=branch,
            )
        except Exception as error:
            fail_spec(spec, error)

    def next_context_with_work() -> RunContext | None:
        if global_stop:
            return None
        for spec in specs:
            if not spec.setup_attempted:
                prepare_spec(spec)
                if global_stop:
                    return None
            context = spec.context
            if context is None:
                continue
            finalize_if_ready(spec)
            if global_stop:
                return None
            if context.finalization_attempted or context.scheduling_stopped:
                continue
            if context.next_position < len(context.pending_episode_ids):
                return context
        return None

    executor = ThreadPoolExecutor(max_workers=worker_count)
    shutdown_wait = True
    futures: dict[Any, tuple[PlannedRun, RunContext, int, str | None]] = {}
    try:
        while True:
            while len(futures) < worker_count and not global_stop:
                context = next_context_with_work()
                if context is None:
                    break
                spec = next(
                    item for item in specs if item.context is context
                )
                gpu_slot = claim_gpu_slot()
                if gpu_slot is False:
                    break
                episode_index = context.pending_episode_ids[context.next_position]
                context.next_position += 1
                context.in_flight += 1
                try:
                    future = executor.submit(
                        run_episode,
                        args=context.args,
                        run_tag=context.run_tag,
                        commit=context.commit,
                        branch=context.branch,
                        episode_meta=context.metadata[episode_index],
                        directories=context.directories,
                        gpu_id=gpu_slot,
                        gpu_attempt_slots=gpu_attempt_slots,
                    )
                except Exception as error:
                    context.in_flight = max(0, context.in_flight - 1)
                    release_gpu_slot(gpu_slot)
                    fail_spec(spec, error)
                    continue
                futures[future] = (spec, context, episode_index, gpu_slot)

            if not futures:
                for spec in specs:
                    finalize_if_ready(spec)
                break

            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                spec, context, _episode_index, gpu_slot = futures.pop(future)
                release_gpu_slot(gpu_slot)
                context.in_flight = max(0, context.in_flight - 1)
                try:
                    row = future.result()
                except Exception as error:
                    fail_spec(spec, error)
                    continue

                try:
                    context.completed_count += 1
                    append_tsv_row(
                        context.directories["episodes_tsv"],
                        EPISODE_COLUMNS,
                        row,
                    )
                    if not context.args.metrics_only:
                        print_progress(
                            context.completed_count,
                            len(context.pending_episode_ids),
                            row,
                        )
                    if (
                        context.args.fail_fast
                        and row["status"] in FAIL_FAST_STATUSES
                    ):
                        context.fail_fast_triggered = True
                        context.scheduling_stopped = True
                except Exception as error:
                    fail_spec(spec, error)

            for spec in specs:
                context = spec.context
                if (
                    context is not None
                    and context.fail_fast_triggered
                    and spec.state != "failed"
                    and not context.fail_fast_reported
                ):
                    context.fail_fast_reported = True
                    suffix = (
                        "; waiting for in-flight episodes to finish"
                        if context.in_flight
                        else ""
                    )
                    print(
                        "stopping current run early because --fail-fast was set"
                        f"{suffix}"
                    )
                    sys.stdout.flush()
                finalize_if_ready(spec)
    except KeyboardInterrupt as interrupt:
        grace_seconds = max(
            (float(spec.args.kill_grace_seconds) for spec in specs),
            default=10.0,
        )
        print(
            "received keyboard interrupt; terminating in-flight episode subprocesses"
        )
        sys.stdout.flush()
        SHUTDOWN_REQUESTED.set()
        for spec in specs:
            if spec.context is not None:
                spec.context.scheduling_stopped = True
        terminate_active_processes(grace_seconds=grace_seconds)
        shutdown_wait = False
        executor.shutdown(wait=True, cancel_futures=True)

        # Interrupted episodes are not recorded, so resume retries them
        for _future, (_spec, context, _episode_index, gpu_slot) in list(
            futures.items()
        ):
            release_gpu_slot(gpu_slot)
            context.in_flight = max(0, context.in_flight - 1)

        for spec in specs:
            context = spec.context
            if context is None or context.finalization_attempted:
                continue
            context.finalization_attempted = True
            try:
                finalize_run_context(context)
            except Exception as error:
                _record_run_failure(spec, error)
            if spec.state == "failed":
                continue
            if (
                isinstance(context.summary, dict)
                and context.summary.get("status") == "completed"
            ):
                spec.state = "completed"
                spec.returncode = 0
                spec.error = None
            elif context.next_position == 0:
                spec.state = "not_started"
                spec.returncode = None
                spec.error = None
            else:
                spec.state = "failed"
                spec.returncode = 130
                spec.error = "interrupted"

        raise ExecutionPlanInterrupted(
            _scheduler_result(specs, details)
        ) from interrupt
    finally:
        if shutdown_wait:
            executor.shutdown(wait=True)

    # Persist partial summaries so a failed plan can be resumed
    for spec in specs:
        finalize_if_ready(spec)
    return _scheduler_result(specs, details)


def run_execution_contexts(
    contexts: Sequence[RunContext],
    keep_going: bool = False,
) -> dict:
    """Run already prepared contexts, primarily for embedding and tests."""

    if not contexts:
        raise ValueError("at least one run context is required")
    specs = [
        PlannedRun(
            mode=context.mode,
            name=context.name,
            argv=list(getattr(context.args, "_invocation_argv", ())),
            args=context.args,
            context=context,
            setup_attempted=True,
        )
        for context in contexts
    ]
    return _run_planned_specs(
        specs,
        keep_going=keep_going,
        commit=contexts[0].commit,
        branch=contexts[0].branch,
    )


def run_pending_episodes(
    args: argparse.Namespace,
    run_tag: str,
    commit: str,
    branch: str,
    metadata: list[dict],
    directories: dict[str, Path],
    pending_episode_ids: list[int],
) -> None:
    total = len(pending_episode_ids)
    if total == 0:
        return

    gpu_ids = list(getattr(args, "resolved_gpu_ids", []))
    gpu_cap = int(args.max_gpu_workers_per_device)
    gpu_assignment_enabled = bool(gpu_ids)
    gpu_slots_enabled = bool(gpu_ids) and gpu_cap > 0
    total_gpu_slots = len(gpu_ids) * gpu_cap if gpu_slots_enabled else 0
    gpu_attempt_slots = (
        {gpu_id: threading.BoundedSemaphore(gpu_cap) for gpu_id in gpu_ids}
        if gpu_slots_enabled
        else {}
    )
    worker_count = min(args.num_workers, total)
    if gpu_slots_enabled:
        worker_count = min(worker_count, total_gpu_slots)
    if worker_count == 1:
        gpu_id = gpu_ids[0] if gpu_ids else None
        for progress_index, episode_index in enumerate(pending_episode_ids, start=1):
            row = run_episode(
                args=args,
                run_tag=run_tag,
                commit=commit,
                branch=branch,
                episode_meta=metadata[episode_index],
                directories=directories,
                gpu_id=gpu_id,
                gpu_attempt_slots=gpu_attempt_slots,
            )
            append_tsv_row(directories["episodes_tsv"], EPISODE_COLUMNS, row)
            if not args.metrics_only:
                print_progress(progress_index, total, row)

            if args.fail_fast and row["status"] in FAIL_FAST_STATUSES:
                print("stopping early because --fail-fast was set")
                break
        return

    next_position = 0
    completed_count = 0
    fail_fast_triggered = False
    fail_fast_reported = False
    gpu_inflight = {gpu_id: 0 for gpu_id in gpu_ids}
    gpu_cursor = 0

    def claim_gpu_slot() -> str | None | bool:
        nonlocal gpu_cursor
        if not gpu_assignment_enabled:
            return None
        if not gpu_slots_enabled:
            candidate = gpu_ids[gpu_cursor % len(gpu_ids)]
            gpu_cursor = (gpu_cursor + 1) % len(gpu_ids)
            return candidate
        for offset in range(len(gpu_ids)):
            index = (gpu_cursor + offset) % len(gpu_ids)
            candidate = gpu_ids[index]
            if gpu_inflight[candidate] < gpu_cap:
                gpu_inflight[candidate] += 1
                gpu_cursor = (index + 1) % len(gpu_ids)
                return candidate
        return False

    def release_gpu_slot(gpu_id: str | None) -> None:
        if gpu_id is not None and gpu_id in gpu_inflight:
            gpu_inflight[gpu_id] = max(0, gpu_inflight[gpu_id] - 1)

    def submit_next(executor: ThreadPoolExecutor, futures: dict) -> bool:
        nonlocal next_position
        if next_position >= total:
            return False
        gpu_slot = claim_gpu_slot()
        if gpu_slot is False:
            return False
        episode_index = pending_episode_ids[next_position]
        next_position += 1
        futures[
            executor.submit(
                run_episode,
                args=args,
                run_tag=run_tag,
                commit=commit,
                branch=branch,
                episode_meta=metadata[episode_index],
                directories=directories,
                gpu_id=gpu_slot,
                gpu_attempt_slots=gpu_attempt_slots,
            )
        ] = (episode_index, gpu_slot)
        return True

    executor = ThreadPoolExecutor(max_workers=worker_count)
    shutdown_wait = True
    try:
        futures = {}
        for _ in range(worker_count):
            submit_next(executor, futures)

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            completed_rows = []
            for future in done:
                _episode_index, gpu_slot = futures.pop(future)
                release_gpu_slot(gpu_slot)
                completed_rows.append(future.result())

            for row in completed_rows:
                completed_count += 1
                append_tsv_row(directories["episodes_tsv"], EPISODE_COLUMNS, row)
                if not args.metrics_only:
                    print_progress(completed_count, total, row)
                if args.fail_fast and row["status"] in FAIL_FAST_STATUSES:
                    fail_fast_triggered = True

            if fail_fast_triggered:
                if not fail_fast_reported:
                    if futures:
                        print("stopping early because --fail-fast was set; waiting for in-flight episodes to finish")
                    else:
                        print("stopping early because --fail-fast was set")
                    sys.stdout.flush()
                    fail_fast_reported = True
                continue

            for _ in completed_rows:
                submit_next(executor, futures)
    except KeyboardInterrupt:
        print("received keyboard interrupt; terminating in-flight episode subprocesses")
        sys.stdout.flush()
        SHUTDOWN_REQUESTED.set()
        terminate_active_processes(grace_seconds=args.kill_grace_seconds)
        shutdown_wait = False
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if shutdown_wait:
            executor.shutdown(wait=True)


def load_execution_plan(path: Path) -> tuple[bool, list[PlannedRun]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"cannot read execution plan {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid execution plan JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("execution plan must be a JSON object")
    version = payload.get("version")
    if isinstance(version, bool) or version != EXECUTION_PLAN_VERSION:
        raise ValueError(
            f"unsupported execution plan version {version!r}; "
            f"expected {EXECUTION_PLAN_VERSION}"
        )
    keep_going = payload.get("keep_going", False)
    if not isinstance(keep_going, bool):
        raise ValueError("execution plan keep_going must be a boolean")
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("execution plan runs must be a non-empty array")

    specs: list[PlannedRun] = []
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"execution plan run {index} must be an object")
        mode = raw_run.get("mode")
        name = raw_run.get("name")
        argv = raw_run.get("argv")
        if not isinstance(mode, str) or not mode:
            raise ValueError(f"execution plan run {index} has invalid mode")
        if not isinstance(name, str) or not name:
            raise ValueError(f"execution plan run {index} has invalid name")
        if not isinstance(argv, list) or not all(
            isinstance(value, str) for value in argv
        ):
            raise ValueError(
                f"execution plan run {index} argv must be an array of strings"
            )
        args = parse_args(argv)
        if (
            args.execution_plan_capabilities
            or args.execution_plan is not None
            or args.execution_result is not None
        ):
            raise ValueError(
                f"execution plan run {index} argv cannot contain execution-plan "
                "control options"
            )
        args._invocation_argv = [str(Path(__file__).resolve()), *argv]
        normalize_batch_args(args)
        if args.task_mode != mode:
            raise ValueError(
                f"execution plan run {index} mode {mode!r} does not match "
                f"--task-mode {args.task_mode!r}"
            )
        specs.append(
            PlannedRun(
                mode=mode,
                name=name,
                argv=list(argv),
                args=args,
            )
        )
    _validate_shared_scheduler_settings(specs)
    return keep_going, specs


def write_execution_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def execute_plan(plan_path: Path, result_path: Path) -> dict:
    keep_going, specs = load_execution_plan(plan_path)
    commit = git_value("rev-parse", "--short", "HEAD")
    branch = git_value("branch", "--show-current")
    try:
        result = _run_planned_specs(
            specs,
            keep_going=keep_going,
            commit=commit,
            branch=branch,
        )
    except ExecutionPlanInterrupted as error:
        write_execution_result(result_path, error.result)
        raise
    write_execution_result(result_path, result)
    return result


def main() -> None:
    SHUTDOWN_REQUESTED.clear()
    args = parse_args()
    if args.execution_plan_capabilities:
        print(
            json.dumps(
                {"execution_plan_version": EXECUTION_PLAN_VERSION},
                separators=(",", ":"),
            )
        )
        return
    if args.execution_plan is not None:
        if args.execution_result is None:
            raise ValueError(
                "--execution-result is required with --execution-plan"
            )
        result = execute_plan(args.execution_plan, args.execution_result)
        if any(run["state"] == "failed" for run in result["runs"]):
            raise SystemExit(1)
        return
    if args.execution_result is not None:
        raise ValueError("--execution-result requires --execution-plan")

    args._invocation_argv = list(sys.argv)
    normalize_batch_args(args)
    commit = git_value("rev-parse", "--short", "HEAD")
    branch = git_value("branch", "--show-current")
    spec = PlannedRun(
        mode=args.task_mode,
        name=args.run_tag or args.task_mode,
        argv=list(sys.argv[1:]),
        args=args,
    )
    result = _run_planned_specs(
        [spec],
        keep_going=False,
        commit=commit,
        branch=branch,
    )
    if result["runs"][0]["state"] == "failed":
        raise RuntimeError(result["runs"][0]["error"])


if __name__ == "__main__":
    try:
        main()
    except ExecutionPlanInterrupted:
        raise SystemExit(130) from None
