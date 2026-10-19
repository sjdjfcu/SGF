"""Support functions shared by navigation and auxiliary tooling."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Any

import cv2
import numpy as np
import quaternion
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from .. import OnlineSGF, detector_prompt_for_goal, label_in_query_set
from ..decision.stop_policy import StopDecision
from ..exploration.utility_explorer import UtilityExplorer
from ..goal.grounding import dedupe_goal_phrases
from ..navigation.explorer import GoalEndReason
from ..navigation.fmm_navigator import FMMGoalNavigator, NavigationCommand
from ..navigation.sgf_map import SGFNavigationMap
from ..validation.stop import StopValidationResult


GRAPHICS_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
_ACTIVE_NAV_CONFIG: dict[str, Any] = {}
_SGF_EMA_ALPHA_MIN = 0.05
_SGF_EMA_ALPHA_MAX = 0.6


def configure_nav_rules(config: dict[str, Any]) -> None:
    """Set the config read by nav_rule()."""

    _ACTIVE_NAV_CONFIG.clear()
    _ACTIVE_NAV_CONFIG.update(config)


def nav_rule(name: str) -> Any:
    try:
        return _ACTIVE_NAV_CONFIG[name]
    except KeyError as exc:
        raise RuntimeError(f"navigation config rule is not loaded: {name}") from exc


def compute_intrinsics(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    hfov_rad = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def create_online_sgf(
    *,
    args: argparse.Namespace,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    scene_graph_queries: list[str],
    enable_semantics: bool = True,
) -> OnlineSGF:
    # OnlineSGF builds GraphSLAM in its constructor. Resetting here would
    # briefly hold two GraphSLAM and YOLOE instances in GPU memory
    return OnlineSGF(
        args.width,
        args.height,
        fx,
        fy,
        cx,
        cy,
        bool(enable_semantics),
        args.yolo_model,
        args.python_executable,
        scene_graph_queries,
        args.yolo_match_mode,
        args.sgf_min_pyr_level,
        args.segment_filter,
        args.sgf_depth_edge_threshold,
        args.yolo_conf_threshold,
        args.yolo_iou_threshold,
        args.yolo_match_iou_threshold,
        _SGF_EMA_ALPHA_MIN,
        _SGF_EMA_ALPHA_MAX,
        args.navigation_map_resolution,
        args.navigation_map_depth_stride,
        args.navigation_map_max_range,
        args.camera_height,
        args.surfel_min_confidence,
        args.navigation_obstacle_confirm_hits,
        args.navigation_free_confirm_hits,
        args.navigation_obstacle_free_margin,
    )


def sensor_state_to_world_from_camera_cv(sensor_state) -> np.ndarray:
    world_from_camera_gl = np.eye(4, dtype=np.float32)
    world_from_camera_gl[:3, :3] = quaternion.as_rotation_matrix(sensor_state.rotation).astype(np.float32)
    world_from_camera_gl[:3, 3] = np.asarray(sensor_state.position, dtype=np.float32)
    return world_from_camera_gl @ GRAPHICS_TO_CV


def depth_observation_to_2d(depth_observation: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_observation)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    return depth.astype(np.float32, copy=False)


def bgr_from_rgb_observation(rgb_observation: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_observation, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise RuntimeError("expected rgb observation with shape HxWx3")
    rgb = rgb[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def planar_distance_xz(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != 3 or b.size != 3:
        return float("inf")
    return float(np.linalg.norm(a[[0, 2]] - b[[0, 2]]))


def depth_forward_navigation_guard(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    forward_step_size_m: float,
    agent_radius_m: float,
    camera_height_m: float,
    max_depth_m: float,
    planner_heading_error_deg: float | None = None,
    forward_clearance_margin_m: float = 0.18,
    sample_stride: int = 6,
) -> dict[str, object]:
    depth_array = np.asarray(depth, dtype=np.float32)
    if depth_array.ndim != 2 or depth_array.size == 0:
        return {"checked": False, "blocked": False}
    stride = max(1, int(sample_stride))
    sampled = depth_array[::stride, ::stride]
    if sampled.size == 0:
        return {"checked": False, "blocked": False}

    rows, cols = np.indices(sampled.shape, dtype=np.float32)
    pixel_rows = rows * float(stride)
    pixel_cols = cols * float(stride)
    z = sampled
    known_depth = np.isfinite(z) & (z > 0.05)
    valid = known_depth & (z < float(max_depth_m))

    corridor_half_width_m = float(agent_radius_m) + 0.08
    forward_horizon_m = min(
        float(max_depth_m),
        float(forward_step_size_m)
        + float(agent_radius_m)
        + max(0.0, float(forward_clearance_margin_m)),
    )
    image_height, image_width = depth_array.shape[:2]
    central_half_width_px = max(
        24.0,
        min(
            0.35 * float(image_width),
            max(
                0.14 * float(image_width),
                corridor_half_width_m * float(fx) / max(1.0, forward_horizon_m),
            ),
        ),
    )
    central_window = (
        (pixel_cols >= float(cx) - central_half_width_px)
        & (pixel_cols <= float(cx) + central_half_width_px)
        & (pixel_rows >= 0.25 * float(image_height))
        & (pixel_rows <= 0.85 * float(image_height))
    )
    central_window_count = int(np.count_nonzero(central_window))
    central_known_count = int(np.count_nonzero(known_depth & central_window))
    central_known_ratio = (
        float(central_known_count) / float(central_window_count)
        if central_window_count > 0
        else 0.0
    )
    known_depth_ratio = float(np.count_nonzero(known_depth)) / float(known_depth.size)

    def preferred_turn_direction() -> str:
        if planner_heading_error_deg is not None and math.isfinite(float(planner_heading_error_deg)):
            return "left" if float(planner_heading_error_deg) < 0.0 else "right"
        return "left"

    def base_trace(*, blocked: bool, reason: str) -> dict[str, object]:
        return {
            "checked": True,
            "blocked": bool(blocked),
            "reason": reason,
            "central_known_depth_ratio": round(central_known_ratio, 4),
            "central_known_depth_count": central_known_count,
            "known_depth_ratio": round(known_depth_ratio, 4),
            "forward_horizon_m": round(forward_horizon_m, 4),
            "forward_clearance_margin_m": round(
                max(0.0, float(forward_clearance_margin_m)), 4
            ),
            "corridor_half_width_m": round(corridor_half_width_m, 4),
            "turn_direction": preferred_turn_direction(),
        }

    if not np.any(valid):
        if not np.any(known_depth):
            return base_trace(blocked=True, reason="no_known_depth")
        return base_trace(blocked=False, reason="no_depth_within_max_depth")

    x = (pixel_cols - float(cx)) * z / max(1e-6, float(fx))
    y_down = (pixel_rows - float(cy)) * z / max(1e-6, float(fy))
    height_above_floor = float(camera_height_m) - y_down
    obstacle_height = (height_above_floor >= 0.08) & (height_above_floor <= float(camera_height_m) + 0.35)
    obstacle = valid & obstacle_height
    if not np.any(obstacle):
        return base_trace(blocked=False, reason="no_obstacle_height_points")

    corridor = obstacle & (z <= forward_horizon_m) & (np.abs(x) <= corridor_half_width_m)
    corridor_z = z[corridor]
    corridor_count = int(corridor_z.size)
    blocked = corridor_count >= 4
    block_reason = "corridor_obstacle" if blocked else ""

    sector_horizon_m = min(float(max_depth_m), max(1.0, forward_horizon_m + 0.6))
    left_sector = obstacle & (z <= sector_horizon_m) & (x < -0.04) & (x >= -0.85)
    right_sector = obstacle & (z <= sector_horizon_m) & (x > 0.04) & (x <= 0.85)
    side_window = (pixel_rows >= 0.20 * float(image_height)) & (
        pixel_rows <= 0.90 * float(image_height)
    )
    left_known_window = (
        side_window
        & (pixel_cols < float(cx))
        & (pixel_cols >= float(cx) - 0.45 * float(image_width))
    )
    right_known_window = (
        side_window
        & (pixel_cols > float(cx))
        & (pixel_cols <= float(cx) + 0.45 * float(image_width))
    )

    def known_ratio(mask: np.ndarray) -> float:
        count = int(np.count_nonzero(mask))
        if count <= 0:
            return 0.0
        return float(np.count_nonzero(known_depth & mask)) / float(count)

    left_known_ratio = known_ratio(left_known_window)
    right_known_ratio = known_ratio(right_known_window)

    def sector_clearance(mask: np.ndarray) -> float:
        values = z[mask]
        if values.size == 0:
            return sector_horizon_m
        return float(np.percentile(values, 15))

    left_clearance = sector_clearance(left_sector)
    right_clearance = sector_clearance(right_sector)
    if planner_heading_error_deg is not None and math.isfinite(float(planner_heading_error_deg)):
        preferred = "left" if float(planner_heading_error_deg) < 0.0 else "right"
        preferred_clearance = left_clearance if preferred == "left" else right_clearance
        other_clearance = right_clearance if preferred == "left" else left_clearance
        turn_direction = (
            preferred
            if preferred_clearance + 0.08 >= other_clearance
            else ("left" if left_clearance >= right_clearance else "right")
        )
    else:
        turn_direction = "left" if left_clearance >= right_clearance else "right"

    expanded_window = (
        (pixel_cols >= float(cx) - 0.35 * float(image_width))
        & (pixel_cols <= float(cx) + 0.35 * float(image_width))
        & (pixel_rows >= 0.20 * float(image_height))
        & (pixel_rows <= 0.90 * float(image_height))
    )
    void_near_horizon_m = min(float(max_depth_m), max(0.9, forward_horizon_m + 0.35))
    near_structure_count = int(np.count_nonzero(obstacle & expanded_window & (z <= void_near_horizon_m)))
    central_depth_void = bool(central_known_ratio < 0.08 and near_structure_count >= 8)
    if central_depth_void:
        blocked = True
        block_reason = "central_depth_void"
        if left_known_ratio > right_known_ratio + 0.05:
            turn_direction = "left"
        elif right_known_ratio > left_known_ratio + 0.05:
            turn_direction = "right"

    min_corridor_z = float(np.min(corridor_z)) if corridor_count > 0 else None
    return {
        "checked": True,
        "blocked": bool(blocked),
        "reason": block_reason,
        "corridor_count": corridor_count,
        "corridor_min_depth_m": None if min_corridor_z is None else round(min_corridor_z, 4),
        "forward_horizon_m": round(forward_horizon_m, 4),
        "forward_clearance_margin_m": round(
            max(0.0, float(forward_clearance_margin_m)), 4
        ),
        "corridor_half_width_m": round(corridor_half_width_m, 4),
        "central_known_depth_ratio": round(central_known_ratio, 4),
        "central_known_depth_count": central_known_count,
        "known_depth_ratio": round(known_depth_ratio, 4),
        "central_depth_void": bool(central_depth_void),
        "near_structure_count": near_structure_count,
        "void_near_horizon_m": round(void_near_horizon_m, 4),
        "left_known_depth_ratio": round(left_known_ratio, 4),
        "right_known_depth_ratio": round(right_known_ratio, 4),
        "left_clearance_m": round(left_clearance, 4),
        "right_clearance_m": round(right_clearance, 4),
        "turn_direction": turn_direction,
    }


def build_query_sets(
    *,
    explicit_queries: list[str],
    parsed_goal,
    canonical_goal: str,
) -> tuple[list[str], list[str], list[str]]:
    def finalized_query_sets(target_queries: list[str]) -> tuple[list[str], list[str], list[str]]:
        targets = dedupe_goal_phrases(target_queries)
        if not targets:
            raise ValueError("text navigation requires at least one target detector query")
        return targets, list(targets), list(targets)

    if explicit_queries:
        target_detector_queries = dedupe_goal_phrases(detector_prompt_for_goal(query) for query in explicit_queries)
        return finalized_query_sets(target_detector_queries)

    target_detector_queries = dedupe_goal_phrases(parsed_goal.target_phrases)
    canonical_query = detector_prompt_for_goal(canonical_goal)
    if not target_detector_queries and canonical_query:
        target_detector_queries = [canonical_query]
    return finalized_query_sets(target_detector_queries)


def select_target_node(
    agent_position: np.ndarray,
    navigation_map: SGFNavigationMap,
    nodes: list[dict],
    goal_phrases: list[str],
    min_surfels: int,
    min_confidence: float,
    excluded_node_ids: set[int],
    excluded_positions: list[np.ndarray],
    exclusion_radius_m: float,
) -> tuple[dict | None, dict[str, int], list[dict]]:
    _ = nodes
    return navigation_map.select_target_node(
        agent_position=agent_position,
        goal_phrases=goal_phrases,
        min_surfels=min_surfels,
        min_confidence=min_confidence,
        excluded_node_ids=excluded_node_ids,
        excluded_positions=excluded_positions,
        exclusion_radius_m=exclusion_radius_m,
    )


def _rotate_xz(vector: np.ndarray, angle_rad: float) -> np.ndarray:
    cos_v = math.cos(float(angle_rad))
    sin_v = math.sin(float(angle_rad))
    return np.array(
        [
            float(vector[0]) * cos_v - float(vector[1]) * sin_v,
            float(vector[0]) * sin_v + float(vector[1]) * cos_v,
        ],
        dtype=np.float32,
    )


def select_image_approach_goal(
    *,
    agent_position: np.ndarray,
    centroid: np.ndarray,
    navigation_map: SGFNavigationMap,
    approach_radius_m: float,
    snap_radius_m: float,
) -> tuple[np.ndarray, float, float] | None:
    radius = max(0.0, float(approach_radius_m))
    if radius <= 0.0:
        return None

    target_floor = np.array([float(centroid[0]), float(agent_position[1]), float(centroid[2])], dtype=np.float32)
    target_to_agent = np.asarray(agent_position[[0, 2]] - target_floor[[0, 2]], dtype=np.float32)
    target_to_agent_norm = float(np.linalg.norm(target_to_agent))
    if target_to_agent_norm <= max(1e-6, radius * 0.35):
        return None
    target_to_agent /= target_to_agent_norm

    radii = (radius, radius * 0.75, radius * 1.25, radius * 1.5)
    angles_deg = (0.0, -35.0, 35.0, -70.0, 70.0, -110.0, 110.0, 180.0)
    candidates: list[np.ndarray] = []
    angle_penalties: list[float] = []
    for sample_radius in radii:
        if sample_radius <= 0.0:
            continue
        for angle_deg in angles_deg:
            direction = _rotate_xz(target_to_agent, math.radians(angle_deg))
            candidate = target_floor.copy()
            candidate[0] += direction[0] * float(sample_radius)
            candidate[2] += direction[1] * float(sample_radius)
            candidates.append(candidate)
            angle_penalties.append(abs(float(angle_deg)))
    if not candidates:
        return None

    snap_search_radius = max(0.15, min(float(snap_radius_m), radius * 0.4, 0.4))
    snapped, valid = navigation_map.snap_to_free_many(
        np.asarray(candidates, dtype=np.float32),
        max_radius_m=snap_search_radius,
    )
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if snapped.size == 0 or not np.any(valid):
        return None

    snapped = np.asarray(snapped, dtype=np.float32).reshape(-1, 3)
    valid_indices = np.flatnonzero(valid[: len(snapped)])
    snapped_valid = snapped[valid_indices]
    distances = navigation_map.geodesic_distances(agent_position, snapped_valid)
    best: tuple[float, float, float, np.ndarray, float] | None = None
    seen_cells: set[tuple[int, int]] = set()
    min_view_distance = max(0.25, radius * 0.4)
    max_view_distance = max(radius * 1.75, radius + 0.5)
    for snapped_index, snapped_goal in enumerate(snapped_valid):
        distance = float(distances[snapped_index]) if snapped_index < len(distances) else float("inf")
        if not math.isfinite(distance):
            continue
        cell = navigation_map.world_to_cell(snapped_goal)
        if cell in seen_cells:
            continue
        seen_cells.add(cell)
        view_distance = planar_distance_xz(snapped_goal, target_floor)
        if view_distance < min_view_distance or view_distance > max_view_distance:
            continue
        source_index = int(valid_indices[snapped_index])
        angle_penalty = angle_penalties[source_index] if source_index < len(angle_penalties) else 180.0
        score = distance + abs(view_distance - radius)
        key = (score, angle_penalty, distance, snapped_goal.copy(), view_distance)
        if best is None or key[:3] < best[:3]:
            best = key
    if best is None:
        return None
    return best[3], float(best[2]), float(best[4])


def finalize_image_target_node(
    *,
    candidate_node: dict,
    agent_position: np.ndarray,
    navigation_map: SGFNavigationMap,
    excluded_positions: list[np.ndarray],
    exclusion_radius_m: float,
    node_id: int,
    snap_radius_m: float = 2.0,
    approach_radius_m: float = 0.0,
) -> tuple[dict | None, str]:
    centroid = np.asarray(candidate_node.get("centroid_m", (np.nan, np.nan, np.nan)), dtype=np.float32).reshape(-1)
    if centroid.size != 3 or not np.all(np.isfinite(centroid)):
        return None, "invalid_centroid"
    snapped = navigation_map.snap_to_free(
        np.array([centroid[0], agent_position[1], centroid[2]], dtype=np.float32),
        max_radius_m=max(0.0, float(snap_radius_m)),
    )
    if snapped is None or not np.all(np.isfinite(snapped)):
        return None, "snap_failed"
    if any(planar_distance_xz(snapped, rejected_position) <= exclusion_radius_m for rejected_position in excluded_positions):
        return None, "excluded_position"
    distance = navigation_map.geodesic_distance(agent_position, snapped)
    if not math.isfinite(distance):
        return None, "geodesic_invalid"

    approach_result = select_image_approach_goal(
        agent_position=agent_position,
        centroid=centroid,
        navigation_map=navigation_map,
        approach_radius_m=approach_radius_m,
        snap_radius_m=snap_radius_m,
    )
    approach_selected = False
    approach_view_distance = None
    if approach_result is not None:
        approach_goal, approach_distance, approach_view_distance = approach_result
        if approach_distance <= distance + max(float(approach_radius_m), 0.5):
            snapped = np.asarray(approach_goal, dtype=np.float32)
            distance = float(approach_distance)
            approach_selected = True

    node = dict(candidate_node)
    node["id"] = int(node_id)
    node["goal_position"] = np.asarray(snapped, dtype=np.float32)
    node["distance_m"] = float(distance)
    if approach_selected:
        node["image_approach_goal"] = True
        node["image_approach_view_distance_m"] = float(approach_view_distance or 0.0)
    return node, ""


def refresh_image_target_node(
    *,
    target_node: dict | None,
    agent_position: np.ndarray,
    navigation_map: SGFNavigationMap,
    excluded_node_ids: set[int],
    excluded_positions: list[np.ndarray],
    exclusion_radius_m: float,
) -> dict | None:
    if target_node is None:
        return None
    if int(target_node.get("id", -1)) in excluded_node_ids:
        return None
    goal_position = np.asarray(target_node.get("goal_position", (np.nan, np.nan, np.nan)), dtype=np.float32).reshape(-1)
    if goal_position.size != 3 or not np.all(np.isfinite(goal_position)):
        return None
    if any(planar_distance_xz(goal_position, rejected_position) <= exclusion_radius_m for rejected_position in excluded_positions):
        return None
    refreshed = dict(target_node)
    distance = navigation_map.geodesic_distance(agent_position, goal_position)
    refreshed["distance_m"] = float(distance) if math.isfinite(distance) else planar_distance_xz(agent_position, goal_position)
    return refreshed


def select_hint_node(
    agent_position: np.ndarray,
    navigation_map: SGFNavigationMap,
    nodes: list[dict],
    goal_phrases: list[str],
    min_surfels: int,
    min_confidence: float,
    excluded_node_ids: set[int],
    excluded_positions: list[np.ndarray],
    exclusion_radius_m: float,
) -> dict | None:
    _ = nodes
    return navigation_map.select_hint_node(
        agent_position=agent_position,
        goal_phrases=goal_phrases,
        min_surfels=min_surfels,
        min_confidence=min_confidence,
        excluded_node_ids=excluded_node_ids,
        excluded_positions=excluded_positions,
        exclusion_radius_m=exclusion_radius_m,
    )


def update_episode_label_stats(
    *,
    nodes: list[dict],
    goal_phrases: list[str],
    observed_label_steps: Counter[str],
    observed_label_max_surfels: dict[str, int],
    observed_label_max_confidence: dict[str, float],
    goal_like_label_steps: Counter[str],
    goal_like_label_max_surfels: dict[str, int],
    goal_like_label_max_confidence: dict[str, float],
) -> None:
    step_stats: dict[str, dict[str, float | int | bool]] = {}
    for node in nodes:
        label = str(node.get("label", ""))
        if not label:
            continue

        surfels = int(node.get("num_surfels", 0))
        confidence = float(node.get("semantic_confidence", 0.0))
        goal_match = label_in_query_set(label, goal_phrases)
        label_stats = step_stats.setdefault(
            label,
            {
                "max_surfels": 0,
                "max_confidence": 0.0,
                "goal_like": False,
            },
        )
        label_stats["max_surfels"] = max(int(label_stats["max_surfels"]), surfels)
        label_stats["max_confidence"] = max(float(label_stats["max_confidence"]), confidence)
        label_stats["goal_like"] = bool(label_stats["goal_like"]) or goal_match

    for label, label_stats in step_stats.items():
        max_surfels = int(label_stats["max_surfels"])
        max_confidence = float(label_stats["max_confidence"])
        observed_label_steps[label] += 1
        observed_label_max_surfels[label] = max(observed_label_max_surfels.get(label, 0), max_surfels)
        observed_label_max_confidence[label] = max(observed_label_max_confidence.get(label, 0.0), max_confidence)
        if bool(label_stats["goal_like"]):
            goal_like_label_steps[label] += 1
            goal_like_label_max_surfels[label] = max(goal_like_label_max_surfels.get(label, 0), max_surfels)
            goal_like_label_max_confidence[label] = max(
                goal_like_label_max_confidence.get(label, 0.0),
                max_confidence,
            )


def update_episode_label_stats_from_summary(
    *,
    summary: dict,
    observed_label_steps: Counter[str],
    observed_label_max_surfels: dict[str, int],
    observed_label_max_confidence: dict[str, float],
    goal_like_label_steps: Counter[str],
    goal_like_label_max_surfels: dict[str, int],
    goal_like_label_max_confidence: dict[str, float],
) -> None:
    for entry in summary.get("labels", []) or []:
        label = str(entry.get("label", ""))
        if not label:
            continue
        max_surfels = int(entry.get("max_surfels", 0))
        max_confidence = float(entry.get("max_confidence", 0.0))
        observed_label_steps[label] += 1
        observed_label_max_surfels[label] = max(observed_label_max_surfels.get(label, 0), max_surfels)
        observed_label_max_confidence[label] = max(observed_label_max_confidence.get(label, 0.0), max_confidence)
        if bool(entry.get("goal_like", False)):
            goal_like_label_steps[label] += 1
            goal_like_label_max_surfels[label] = max(goal_like_label_max_surfels.get(label, 0), max_surfels)
            goal_like_label_max_confidence[label] = max(
                goal_like_label_max_confidence.get(label, 0.0),
                max_confidence,
            )


def summarize_stop_validation(stop_validation_result: StopValidationResult | None) -> dict | None:
    if stop_validation_result is None:
        return None
    bbox_xyxy = getattr(stop_validation_result, "detection_bbox_xyxy", None)
    bbox_center = getattr(stop_validation_result, "detection_bbox_center", None)
    return {
        "approved": bool(stop_validation_result.approved),
        "validator_mode": stop_validation_result.validator_mode,
        "reason": stop_validation_result.reason,
        "detection_found": bool(stop_validation_result.detection_found),
        "detection_confidence": _round_scalar(stop_validation_result.detection_confidence, 4),
        "detection_label": stop_validation_result.detection_label,
        "detection_bbox_xyxy": (
            None
            if bbox_xyxy is None
            else [int(value) for value in bbox_xyxy]
        ),
        "detection_bbox_area_fraction": _round_scalar(
            getattr(stop_validation_result, "detection_bbox_area_fraction", 0.0),
            4,
        ),
        "detection_bbox_center": (
            None
            if bbox_center is None
            else [_round_scalar(value, 4) for value in bbox_center]
        ),
        "p_yes": _round_scalar(stop_validation_result.p_yes, 4),
        "scene_p_yes": _round_scalar(stop_validation_result.scene_p_yes, 4),
        "evidence_source": str(
            getattr(stop_validation_result, "evidence_source", "detector_top")
        ),
        "hypothesis_overlap": _round_scalar(
            getattr(stop_validation_result, "hypothesis_overlap", 0.0), 4
        ),
        "hypothesis_depth_support": _round_scalar(
            getattr(stop_validation_result, "hypothesis_depth_support", 0.0), 4
        ),
        "appearance_p": _round_scalar(
            getattr(stop_validation_result, "appearance_p", None), 4
        ),
        "context_p": _round_scalar(
            getattr(stop_validation_result, "context_p", None), 4
        ),
        "context_observability": _round_scalar(
            getattr(stop_validation_result, "context_observability", 0.0), 4
        ),
        "factored": bool(getattr(stop_validation_result, "factored", False)),
    }


def summarize_stop_decision(decision: StopDecision | None) -> dict | None:
    if decision is None:
        return None
    return {
        "action": decision.action,
        "reason": decision.reason,
        "details": decision.details,
    }


def _round_scalar(value, digits: int = 4):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _agent_heading_rad(agent_state) -> float | None:
    rotation = getattr(agent_state, "rotation", None)
    if rotation is None:
        return None
    try:
        matrix = quaternion.as_rotation_matrix(rotation)
        forward = np.asarray(matrix, dtype=np.float64) @ np.array(
            [0.0, 0.0, -1.0], dtype=np.float64
        )
    except Exception:
        return None
    forward_xz = np.asarray([forward[0], forward[2]], dtype=np.float64)
    if not np.all(np.isfinite(forward_xz)):
        return None
    if float(np.linalg.norm(forward_xz)) < 1e-6:
        return None
    return float(math.atan2(-forward_xz[0], -forward_xz[1]))


def plan_exploration_action(
    *,
    agent_state,
    agent_position: np.ndarray,
    navigation_map: SGFNavigationMap,
    explorer: UtilityExplorer,
    navigator: FMMGoalNavigator,
    stop_radius_m: float,
    mode: str,
    local_blocked_retire_stall_count: int = 8,
    step_index: int | None = None,
) -> tuple[np.ndarray | None, NavigationCommand | None, int, str, bool, bool]:
    """Plan one online-only exploration action and retire blocked exploration goals."""
    explorer.set_heading_rad(_agent_heading_rad(agent_state))
    goal_position = explorer.get_goal(agent_position, step_index)
    if goal_position is None:
        explorer._clear_current_goal()
        return None, None, int(HabitatSimActions.turn_left), "explore:scan", False, False

    planner_command = navigator.next_action(
        agent_state=agent_state,
        goal_position=goal_position,
        navigation_map=navigation_map,
        allow_stop=False,
        stop_radius_m=stop_radius_m,
    )
    if planner_command.goal_blocked and planner_command.block_scope == "goal":
        explorer.mark_goal_failed(
            goal_position,
            reason=GoalEndReason.PLANNER_GOAL_BLOCKED,
        )
        return goal_position, planner_command, int(planner_command.action), "explore:blocked", False, True

    if (
        int(local_blocked_retire_stall_count) > 0
        and planner_command.local_blocked
        and int(planner_command.stalled_count) >= int(local_blocked_retire_stall_count)
    ):
        explorer.mark_goal_failed(
            goal_position,
            reason=GoalEndReason.PLANNER_LOCAL_BLOCKED,
        )
        return goal_position, planner_command, int(planner_command.action), "explore:local_blocked", False, True

    if planner_command.goal_reached:
        explorer.end_current_goal(GoalEndReason.REACHED)
        return goal_position, planner_command, int(HabitatSimActions.turn_left), "explore:reached", True, False

    return goal_position, planner_command, int(planner_command.action), mode, False, False
