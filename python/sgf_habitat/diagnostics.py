from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions


def action_name(action: int) -> str:
    action_map = {
        int(HabitatSimActions.stop): "stop",
        int(HabitatSimActions.move_forward): "forward",
        int(HabitatSimActions.turn_left): "left",
        int(HabitatSimActions.turn_right): "right",
    }
    if hasattr(HabitatSimActions, "look_up"):
        action_map[int(HabitatSimActions.look_up)] = "look_up"
    if hasattr(HabitatSimActions, "look_down"):
        action_map[int(HabitatSimActions.look_down)] = "look_down"
    return action_map.get(int(action), str(int(action)))


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_path_component(text: str) -> str:
    text = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip().lower())
    text = "_".join(part for part in text.split("_") if part)
    return text or "episode"


def make_default_artifact_dir(artifact_root: str | Path, episode_id: int, goal_name: str) -> Path:
    root = Path(artifact_root).expanduser()
    stem = sanitize_path_component(goal_name)
    return root / f"{now_timestamp()}_episode_{episode_id:04d}_{stem}"


@dataclass
class EpisodeArtifacts:
    artifact_dir: Path
    metadata_path: Path
    diagnostics_log_path: Path
    trace_path: Path
    result_path: Path
    final_image_path: Path

    @classmethod
    def from_dir(cls, artifact_dir: str | Path) -> "EpisodeArtifacts":
        base = Path(artifact_dir).expanduser().resolve()
        return cls(
            artifact_dir=base,
            metadata_path=base / "metadata.json",
            diagnostics_log_path=base / "diagnostics.log",
            trace_path=base / "trace.jsonl",
            result_path=base / "result.json",
            final_image_path=base / "final_summary.png",
        )

    def ensure(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)


def _is_finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _round_float(value: Any, digits: int = 4) -> float | None:
    if not _is_finite_scalar(value):
        return None
    return round(float(value), digits)


def world_to_list(xyz: Any) -> list[float] | None:
    if xyz is None:
        return None
    arr = np.asarray(xyz, dtype=np.float32).reshape(-1)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return None
    return [round(float(arr[0]), 4), round(float(arr[1]), 4), round(float(arr[2]), 4)]


def cell_to_list(navigation_map, xyz: Any) -> list[int] | None:
    if xyz is None:
        return None
    world = np.asarray(xyz, dtype=np.float32).reshape(-1)
    if world.size != 3 or not np.all(np.isfinite(world)):
        return None
    row, col = navigation_map.world_to_cell(world)
    rows, cols = navigation_map.shape
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return None
    return [int(row), int(col)]


def summarize_target_node(target_node: dict | None, navigation_map=None) -> dict | None:
    if target_node is None:
        return None
    summary = {
        "id": int(target_node.get("id", -1)),
        "label": str(target_node.get("label", "")),
        "distance_m": _round_float(target_node.get("distance_m"), 4),
        "num_surfels": int(target_node.get("num_surfels", 0)),
        "semantic_confidence": _round_float(target_node.get("semantic_confidence"), 4),
        "centroid_m": world_to_list(target_node.get("centroid_m")),
        "goal_position_m": world_to_list(target_node.get("goal_position")),
        "bbox_min_m": world_to_list(target_node.get("bbox_min_m")),
        "bbox_max_m": world_to_list(target_node.get("bbox_max_m")),
    }
    if "match_score" in target_node:
        summary["match_score"] = _round_float(target_node.get("match_score"), 4)
    geometry_penalty = _round_float(target_node.get("geometry_penalty"), 4)
    if geometry_penalty is not None:
        summary["geometry_penalty"] = geometry_penalty
    hint_source = target_node.get("hint_source")
    if hint_source:
        summary["hint_source"] = str(hint_source)
    bbox_min = np.asarray(target_node.get("bbox_min_m", (np.nan, np.nan, np.nan)), dtype=np.float32).reshape(-1)
    bbox_max = np.asarray(target_node.get("bbox_max_m", (np.nan, np.nan, np.nan)), dtype=np.float32).reshape(-1)
    if bbox_min.size == 3 and bbox_max.size == 3 and np.all(np.isfinite(bbox_min)) and np.all(np.isfinite(bbox_max)):
        summary["bbox_extent_m"] = [_round_float(value, 4) for value in np.abs(bbox_max - bbox_min).tolist()]
    if navigation_map is not None:
        summary["goal_cell"] = cell_to_list(navigation_map, target_node.get("goal_position"))
    return summary


def summarize_runtime_stats(runtime_stats: dict) -> dict[str, float]:
    return {str(key): round(float(value), 4) for key, value in runtime_stats.items()}


def summarize_map_stats(
    navigation_map,
    *,
    visited_mask: np.ndarray | None = None,
    revisit_counts: np.ndarray | None = None,
) -> dict[str, Any]:
    observed_mask = np.asarray(navigation_map.observed_mask(), dtype=bool)
    free_mask = np.asarray(navigation_map.free_mask(), dtype=bool)
    occupied_mask = np.asarray(navigation_map.occupied_mask(), dtype=bool)
    frontier_mask = np.asarray(navigation_map.frontier_cells(), dtype=bool)
    stats: dict[str, Any] = {
        "shape": [int(observed_mask.shape[0]), int(observed_mask.shape[1])],
        "observed_cells": int(np.count_nonzero(observed_mask)),
        "free_cells": int(np.count_nonzero(free_mask)),
        "occupied_cells": int(np.count_nonzero(occupied_mask)),
        "unknown_cells": int(np.count_nonzero(~observed_mask)),
        "frontier_cells": int(np.count_nonzero(frontier_mask)),
    }
    if visited_mask is not None and visited_mask.shape == free_mask.shape:
        visited_reachable = visited_mask & free_mask
        stats["visited_free_cells"] = int(np.count_nonzero(visited_reachable))
    if revisit_counts is not None and revisit_counts.shape == free_mask.shape:
        stats["revisited_cells"] = int(np.count_nonzero(revisit_counts > 1))
        stats["max_revisit_count"] = int(revisit_counts.max(initial=0))
    return stats


def _cell_neighbors(row: int, col: int) -> tuple[tuple[int, int], ...]:
    return (
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    )


def nearest_true_cell(mask: np.ndarray, start_cell: tuple[int, int]) -> tuple[int, int] | None:
    rows, cols = mask.shape
    start_row = int(np.clip(start_cell[0], 0, max(0, rows - 1)))
    start_col = int(np.clip(start_cell[1], 0, max(0, cols - 1)))
    if mask[start_row, start_col]:
        return start_row, start_col

    visited = np.zeros_like(mask, dtype=bool)
    queue: list[tuple[int, int]] = [(start_row, start_col)]
    visited[start_row, start_col] = True
    head = 0
    while head < len(queue):
        row, col = queue[head]
        head += 1
        if mask[row, col]:
            return row, col
        for next_row, next_col in _cell_neighbors(row, col):
            if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                continue
            if visited[next_row, next_col]:
                continue
            visited[next_row, next_col] = True
            queue.append((next_row, next_col))
    return None


def reachable_mask_from_start(free_mask: np.ndarray, start_cell: tuple[int, int]) -> np.ndarray:
    reachable = np.zeros_like(free_mask, dtype=bool)
    start = nearest_true_cell(free_mask, start_cell)
    if start is None:
        return reachable

    queue: list[tuple[int, int]] = [start]
    reachable[start] = True
    head = 0
    while head < len(queue):
        row, col = queue[head]
        head += 1
        for next_row, next_col in _cell_neighbors(row, col):
            if next_row < 0 or next_row >= free_mask.shape[0] or next_col < 0 or next_col >= free_mask.shape[1]:
                continue
            if reachable[next_row, next_col] or not free_mask[next_row, next_col]:
                continue
            reachable[next_row, next_col] = True
            queue.append((next_row, next_col))
    return reachable


def rasterize_path(
    navigation_map,
    positions_m: list[np.ndarray],
    *,
    radius_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = navigation_map.shape
    visit_counts = np.zeros((rows, cols), dtype=np.int32)
    for position in positions_m:
        row, col = navigation_map.world_to_cell(np.asarray(position, dtype=np.float32))
        if row < 0 or row >= rows or col < 0 or col >= cols:
            continue
        visit_counts[row, col] += 1

    visited_mask = visit_counts > 0
    if radius_cells > 0 and np.any(visited_mask):
        kernel_size = radius_cells * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        visited_mask = cv2.dilate(visited_mask.astype(np.uint8), kernel, iterations=1) > 0
    return visit_counts, visited_mask


def _merge_segments(indices: list[int], *, gap_tolerance: int = 1, min_length: int = 1) -> list[dict[str, int]]:
    if not indices:
        return []
    segments: list[dict[str, int]] = []
    start = indices[0]
    prev = indices[0]
    for index in indices[1:]:
        if index - prev <= gap_tolerance:
            prev = index
            continue
        length = prev - start + 1
        if length >= min_length:
            segments.append({"start_step": start, "end_step": prev, "length": length})
        start = prev = index
    length = prev - start + 1
    if length >= min_length:
        segments.append({"start_step": start, "end_step": prev, "length": length})
    return segments


def detect_oscillation_segments(records: list[dict[str, Any]]) -> list[dict[str, int]]:
    if len(records) < 6:
        return []
    oscillating_steps: set[int] = set()
    for start in range(0, len(records) - 5):
        window = records[start : start + 6]
        turn_values: list[int] = []
        positions = []
        for record in window:
            executed_action = int(record.get("executed_action", -1))
            if executed_action == int(HabitatSimActions.turn_left):
                turn_values.append(-1)
            elif executed_action == int(HabitatSimActions.turn_right):
                turn_values.append(1)
            positions.append(np.asarray(record["agent_position_m"], dtype=np.float32))
        if len(turn_values) < 4:
            continue
        alternations = sum(1 for lhs, rhs in zip(turn_values[:-1], turn_values[1:]) if lhs != rhs)
        displacement = float(np.linalg.norm(positions[-1][[0, 2]] - positions[0][[0, 2]]))
        if alternations >= max(3, len(turn_values) - 2) and displacement <= 0.75:
            for offset in range(6):
                oscillating_steps.add(start + offset)
    return _merge_segments(sorted(oscillating_steps), min_length=4)


def navigation_trace_position(record: dict[str, Any]):
    """Read the navigation pose, avoiding simulator-world diagnostic poses."""
    position = record.get("navigation_before_position_m")
    return record.get("agent_position_m") if position is None else position


def _positions_from_records(records: list[dict[str, Any]], terminal_position_m: list[float] | None = None) -> list[np.ndarray]:
    positions = [
        np.asarray(position, dtype=np.float32)
        for record in records
        if (position := navigation_trace_position(record)) is not None
    ]
    if terminal_position_m is not None:
        terminal = np.asarray(terminal_position_m, dtype=np.float32)
        if terminal.size == 3 and np.all(np.isfinite(terminal)):
            if not positions or float(np.linalg.norm(positions[-1] - terminal)) > 1e-4:
                positions.append(terminal)
    return positions


@dataclass
class EpisodeTraceAnalysis:
    summary: dict[str, Any]
    nav_explored_mask: np.ndarray
    map_reachable_mask: np.ndarray
    revisit_counts: np.ndarray
    repeated_mask: np.ndarray
    timeline_lines: list[str]


def analyze_episode_trace(
    *,
    records: list[dict[str, Any]],
    navigation_map,
    result_payload: dict[str, Any],
    success_radius_m: float,
    visited_radius_cells: int,
) -> EpisodeTraceAnalysis:
    free_mask = np.asarray(navigation_map.free_mask(), dtype=bool)
    frontier_mask = np.asarray(navigation_map.frontier_cells(), dtype=bool)
    terminal = result_payload.get("diagnostics", {}).get("terminal_navigation_position_m")
    positions = _positions_from_records(records, terminal_position_m=terminal)
    start_cell = navigation_map.world_to_cell(positions[0]) if positions else (0, 0)
    map_reachable_mask = reachable_mask_from_start(free_mask, start_cell)
    revisit_counts, _ = rasterize_path(
        navigation_map,
        positions,
        radius_cells=max(0, int(visited_radius_cells)),
    )
    exact_counts, exact_visited = rasterize_path(navigation_map, positions, radius_cells=0)
    repeated_mask = exact_counts > 1

    visited_exact_cells = int(np.count_nonzero(exact_visited))
    repeated_cells = int(np.count_nonzero(repeated_mask))
    revisit_ratio = float(repeated_cells / visited_exact_cells) if visited_exact_cells > 0 else 0.0

    stagnation_steps = [
        int(record["step_index"])
        for record in records
        if bool(record.get("explorer", {}).get("stagnating"))
    ]
    planner_failed_steps = [
        int(record["step_index"])
        for record in records
        if bool(record.get("planner", {}).get("planner_replan_failed"))
    ]
    stalled_steps = [
        int(record["step_index"])
        for record in records
        if int(record.get("planner", {}).get("stalled_count", 0)) >= 2
    ]
    blocked_steps = [
        int(record["step_index"])
        for record in records
        if bool(record.get("planner", {}).get("goal_blocked"))
    ]
    recovery_steps = [
        int(record["step_index"])
        for record in records
        if bool(record.get("planner", {}).get("recovery_active"))
    ]
    stagnation_segments = _merge_segments(stagnation_steps, min_length=3)
    trapped_segments = _merge_segments(sorted(set(planner_failed_steps) | set(stalled_steps)), min_length=3)
    blocked_segments = _merge_segments(blocked_steps, min_length=1)
    oscillation_segments = detect_oscillation_segments(records)

    min_goal_distance_m = min(
        (
            float(record["planner"]["goal_distance_m"])
            for record in records
            if record.get("planner", {}).get("goal_distance_m") is not None
        ),
        default=float("inf"),
    )
    target_records = [record for record in records if record.get("selected_target")]
    min_target_record = min(
        (
            record
            for record in target_records
            if record["selected_target"].get("distance_m") is not None
        ),
        key=lambda record: float(record["selected_target"]["distance_m"]),
        default=None,
    )
    min_target_distance_m = (
        float(min_target_record["selected_target"]["distance_m"]) if min_target_record is not None else float("inf")
    )

    stop_suppression_counter: Counter[str] = Counter()
    stop_validator_rejections: list[dict[str, Any]] = []
    wrong_stop_steps: list[int] = []
    for record in records:
        stop_info = record.get("stop", {})
        for reason in stop_info.get("suppressed_reasons", []):
            stop_suppression_counter[str(reason)] += 1
        validator = stop_info.get("validator")
        if validator and not validator.get("approved", False):
            stop_validator_rejections.append(
                {
                    "step_index": int(record["step_index"]),
                    "reason": str(validator.get("reason", "")),
                    "validator_mode": str(validator.get("validator_mode", "")),
                }
            )
        if bool(stop_info.get("executed_stop")):
            wrong_stop_steps.append(int(record["step_index"]))

    success = float(result_payload.get("metrics", {}).get("success", 0.0)) > 0.0
    stop_called = bool(result_payload.get("stop_called"))
    target_found = bool(result_payload.get("target_found"))
    wrong_stop = stop_called and not success
    near_target_no_stop = (not stop_called) and math.isfinite(min_target_distance_m) and (
        min_target_distance_m <= max(success_radius_m * 1.35, success_radius_m + 0.25)
    )
    repeated_exploration = revisit_ratio >= 0.30
    stagnating = bool(stagnation_segments)
    trapped = bool(trapped_segments)
    oscillating = bool(oscillation_segments)
    unable_to_escape = (sum(segment["length"] for segment in trapped_segments) >= 12) or (
        sum(segment["length"] for segment in stagnation_segments) >= 25
    )

    issue_flags = {
        "stagnation": stagnating,
        "trapped": trapped,
        "oscillation": oscillating,
        "unable_to_escape": unable_to_escape,
        "repeated_exploration": repeated_exploration,
        "wrong_stop": wrong_stop,
        "near_target_no_stop": near_target_no_stop,
        "target_found_but_no_stop": target_found and not stop_called,
    }

    timeline_lines: list[str] = []
    for segment in stagnation_segments:
        timeline_lines.append(
            f"[step {segment['start_step']:03d}] entered stagnation window lasting {segment['length']} steps."
        )
    for segment in trapped_segments:
        timeline_lines.append(
            f"[step {segment['start_step']:03d}] planner repeatedly failed / stalled for {segment['length']} steps."
        )
    for segment in blocked_segments[:8]:
        timeline_lines.append(
            f"[step {segment['start_step']:03d}] blocked navigation goal reported for {segment['length']} step(s)."
        )
    for segment in oscillation_segments:
        timeline_lines.append(
            f"[step {segment['start_step']:03d}] repeated left-right oscillation lasted {segment['length']} steps."
        )
    for rejection in stop_validator_rejections[:8]:
        timeline_lines.append(
            f"[step {rejection['step_index']:03d}] stop validator rejected target: {rejection['reason']}"
        )
    for reason, count in stop_suppression_counter.items():
        timeline_lines.append(f"[summary] stop was suppressed by `{reason}` {count} time(s).")
    if wrong_stop:
        timeline_lines.append("[summary] a stop action was executed but the episode still failed.")
    if near_target_no_stop and min_target_record is not None:
        timeline_lines.append(
            f"[summary] closest target approach was {min_target_distance_m:.2f}m at step {min_target_record['step_index']:03d}, but no stop happened."
        )
    if repeated_exploration:
        timeline_lines.append("[summary] revisit density was high, indicating repeated exploration of the same area.")

    summary = {
        "revisit_ratio": round(revisit_ratio, 6),
        "visited_exact_cells": visited_exact_cells,
        "repeated_cells": repeated_cells,
        "final_frontier_cells": int(np.count_nonzero(frontier_mask)),
        "planner_replan_failed_steps": len(planner_failed_steps),
        "stalled_steps": len(stalled_steps),
        "goal_blocked_steps": len(blocked_steps),
        "recovery_active_steps": len(recovery_steps),
        "max_stalled_count": max(
            (int(record.get("planner", {}).get("stalled_count", 0)) for record in records),
            default=0,
        ),
        "max_replan_failed_count": max(
            (int(record.get("planner", {}).get("replan_failed_count", 0)) for record in records),
            default=0,
        ),
        "stagnation_segments": stagnation_segments,
        "trapped_segments": trapped_segments,
        "blocked_segments": blocked_segments,
        "oscillation_segments": oscillation_segments,
        "min_goal_distance_m": None if not math.isfinite(min_goal_distance_m) else round(min_goal_distance_m, 4),
        "min_target_distance_m": None if not math.isfinite(min_target_distance_m) else round(min_target_distance_m, 4),
        "closest_target_step": None if min_target_record is None else int(min_target_record["step_index"]),
        "stop_suppression_counts": dict(sorted(stop_suppression_counter.items())),
        "stop_validator_rejections": stop_validator_rejections,
        "issue_flags": issue_flags,
        "wrong_stop_steps": wrong_stop_steps if wrong_stop else [],
    }
    return EpisodeTraceAnalysis(
        summary=summary,
        nav_explored_mask=free_mask,
        map_reachable_mask=map_reachable_mask,
        revisit_counts=exact_counts,
        repeated_mask=repeated_mask,
        timeline_lines=timeline_lines,
    )


def build_text_report(
    *,
    episode_info,
    canonical_goal: str,
    goal_queries: list[str],
    result_payload: dict[str, Any],
    analysis: EpisodeTraceAnalysis,
    records: list[dict[str, Any]],
) -> str:
    label_diagnostics = result_payload.get("label_diagnostics") or analysis.summary.get("label_diagnostics") or {}

    def format_label_entries(entries: list[dict[str, Any]]) -> str:
        if not entries:
            return "-"
        parts: list[str] = []
        for entry in entries[:8]:
            fragment = (
                f"{entry.get('label')}[steps={entry.get('steps')}, "
                f"surfels={entry.get('max_surfels')}, conf={entry.get('max_confidence')}]"
            )
            parts.append(fragment)
        return "; ".join(parts)

    metrics = result_payload.get("metrics", {})
    lines = [
        "RUN",
        f"episode_id: {episode_info.episode_id}",
        f"scene_id: {episode_info.scene_id}",
        f"goal_name: {episode_info.goal_name}",
        f"canonical_goal: {canonical_goal}",
        f"text_goal_source: {episode_info.text_goal_source}",
        f"text_goal: {episode_info.text_goal}",
        f"goal_queries: {', '.join(goal_queries) if goal_queries else '-'}",
        "",
        "RESULT",
        f"success: {'yes' if float(metrics.get('success', 0.0)) > 0.0 else 'no'}",
        f"steps: {result_payload.get('steps', 0)}",
        f"distance_to_goal_m: {_round_float(metrics.get('distance_to_goal'), 4)}",
        f"target_found: {'yes' if result_payload.get('target_found') else 'no'}",
        f"stop_called: {'yes' if result_payload.get('stop_called') else 'no'}",
        f"stop_validation_attempts: {result_payload.get('stop_validation_attempts', 0)}",
        f"stop_validation_rejections: {result_payload.get('stop_validation_rejections', 0)}",
        "",
        "MAP",
        f"revisit_ratio: {analysis.summary['revisit_ratio']:.3f}",
        f"repeated_cells: {analysis.summary['repeated_cells']}",
        f"final_frontier_cells: {analysis.summary['final_frontier_cells']}",
        f"goal_blocked_steps: {analysis.summary.get('goal_blocked_steps', 0)}",
        f"max_replan_failed_count: {analysis.summary.get('max_replan_failed_count', 0)}",
    ]

    if label_diagnostics:
        lines.extend(
            [
                "",
                "SEMANTICS",
                f"observed_label_count: {label_diagnostics.get('observed_label_count', 0)}",
                f"goal_like_label_count: {label_diagnostics.get('goal_like_label_count', 0)}",
                f"goal_like_labels: {format_label_entries(label_diagnostics.get('goal_like_labels_top', []))}",
                f"top_labels: {format_label_entries(label_diagnostics.get('observed_labels_top', []))}",
                f"goal_candidate_diagnostics: {label_diagnostics.get('goal_candidate_diagnostics', {}) or '-'}",
            ]
        )

    lines.extend(["", "KEY FINDINGS"])

    issue_lines = []
    for key, enabled in analysis.summary["issue_flags"].items():
        if enabled:
            issue_lines.append(f"- {key}")
    if not issue_lines:
        issue_lines.append("- no major heuristic issue flagged")
    lines.extend(issue_lines)

    if analysis.timeline_lines:
        lines.append("")
        lines.append("TIMELINE")
        lines.extend(f"- {line}" for line in analysis.timeline_lines)

    if records:
        lines.append("")
        lines.append("STEP SNAPSHOT")
        for record in records[-min(len(records), 5) :]:
            planner = record.get("planner", {})
            lines.append(
                "- step={step:03d} mode={mode} executed={action} goal_dist={goal_dist} "
                "goal_viewpoint_dist={goal_viewpoint_dist} target_dist={target_dist} "
                "stalled={stalled} replan_failed={replan}".format(
                    step=int(record["step_index"]),
                    mode=record.get("mode", ""),
                    action=record.get("executed_action_name", ""),
                    goal_dist=planner.get("goal_distance_m"),
                    goal_viewpoint_dist=record.get("gt_goal_viewpoint_planar_distance_m"),
                    target_dist=(record.get("selected_target") or {}).get("distance_m"),
                    stalled=planner.get("stalled_count"),
                    replan=planner.get("planner_replan_failed"),
                )
            )
    return "\n".join(lines) + "\n"


class EpisodeDiagnosticsRecorder:
    def __init__(
        self,
        artifacts: EpisodeArtifacts,
        *,
        episode_info,
        metadata: dict[str, Any],
        verbose: bool,
    ) -> None:
        self.artifacts = artifacts
        self.episode_info = episode_info
        self.verbose = bool(verbose)
        self.records: list[dict[str, Any]] = []
        self.artifacts.ensure()
        self.artifacts.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        self._trace_handle = self.artifacts.trace_path.open("w", encoding="utf-8")
        self._log_handle = self.artifacts.diagnostics_log_path.open("w", encoding="utf-8")
        self._log_handle.write(
            f"Diagnostics log for episode {episode_info.episode_id} created at {datetime.now().isoformat()}\n\n"
        )

    def log(self, message: str = "") -> None:
        self._log_handle.write(message.rstrip() + "\n")
        self._log_handle.flush()

    def record_step(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self._trace_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._trace_handle.flush()
        planner = record.get("planner", {})
        target = record.get("selected_target")
        target_distance = None if target is None else target.get("distance_m")
        goal_viewpoint_distance = record.get("gt_goal_viewpoint_planar_distance_m")
        if not self.verbose:
            self.log(
                f"step={int(record['step_index']):03d} "
                f"gt_goal_viewpoint_planar_distance_m={goal_viewpoint_distance}"
            )
            return

        line = (
            f"step={int(record['step_index']):03d} mode={record.get('mode')} "
            f"proposed={record.get('proposed_action_name')} executed={record.get('executed_action_name')} "
            f"goal_distance={planner.get('goal_distance_m')} "
            f"gt_goal_viewpoint_planar_distance_m={goal_viewpoint_distance} "
            f"target_distance={target_distance} "
            f"stalled_count={planner.get('stalled_count')} "
            f"replan_failed_count={planner.get('replan_failed_count')} "
            f"planner_replan_failed={planner.get('planner_replan_failed')} "
            f"goal_blocked={planner.get('goal_blocked')} "
            f"recovery_reason={planner.get('recovery_reason')} "
            f"frontiers={record.get('map', {}).get('frontier_cells')}"
        )
        self.log(line)
        oracle_nav = planner.get("oracle_nav")
        if oracle_nav:
            self.log("oracle_nav=" + json.dumps(oracle_nav, ensure_ascii=False))
        validator = record.get("stop", {}).get("validator")
        if validator is not None:
            self.log("stop_validation=" + json.dumps(validator, ensure_ascii=False))

    def close(self) -> None:
        self._trace_handle.close()
        self._log_handle.close()
