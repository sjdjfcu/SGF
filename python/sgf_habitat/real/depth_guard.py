"""Depth safety checks for ZED/Kobuki planning and motion execution."""

from __future__ import annotations

import math

import numpy as np


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
    min_depth_m: float | None = None,
    base_from_camera_cv: np.ndarray | None = None,
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

    transform = (
        np.asarray(base_from_camera_cv, dtype=np.float64).reshape(4, 4)
        if base_from_camera_cv is not None
        else np.asarray([
            [1, 0, 0, 0], [0, -1, 0, camera_height_m],
            [0, 0, -1, 0], [0, 0, 0, 1],
        ], dtype=np.float64)
    )
    if not np.all(np.isfinite(transform)):
        return base_trace(blocked=True, reason="invalid_camera_extrinsics")

    def project_to_base(depth_values):
        points = np.stack([
            (pixel_cols - float(cx)) * depth_values / max(1e-6, float(fx)),
            (pixel_rows - float(cy)) * depth_values / max(1e-6, float(fy)),
            depth_values,
        ], axis=-1)
        return points @ transform[:3, :3].T + transform[:3, 3]

    points = project_to_base(np.where(valid, z, 0.0))
    x, height_above_floor, forward = points[..., 0], points[..., 1], -points[..., 2]
    obstacle_height = (height_above_floor >= 0.08) & (height_above_floor <= float(camera_height_m) + 0.35)
    obstacle = valid & obstacle_height & (forward > 0.0)
    if not np.any(obstacle):
        return base_trace(blocked=False, reason="no_obstacle_height_points")

    corridor = obstacle & (forward <= forward_horizon_m) & (np.abs(x) <= corridor_half_width_m)
    corridor_z = forward[corridor]
    corridor_count = int(corridor_z.size)
    blocked = corridor_count >= 4
    block_reason = "corridor_obstacle" if blocked else ""

    sector_horizon_m = min(float(max_depth_m), max(1.0, forward_horizon_m + 0.6))
    left_sector = obstacle & (forward <= sector_horizon_m) & (x < -0.04) & (x >= -0.85)
    right_sector = obstacle & (forward <= sector_horizon_m) & (x > 0.04) & (x <= 0.85)
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
        values = forward[mask]
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
    near_structure_count = int(np.count_nonzero(obstacle & expanded_window & (forward <= void_near_horizon_m)))
    central_depth_void = bool(central_known_ratio < 0.08 and near_structure_count >= 8)
    if central_depth_void:
        blocked = True
        block_reason = "central_depth_void"
        if left_known_ratio > right_known_ratio + 0.05:
            turn_direction = "left"
        elif right_known_ratio > left_known_ratio + 0.05:
            turn_direction = "right"

    # Jambs inside the stereo minimum range read as unknown depth, so block
    # when a side strip is mostly unknown beside measured near structure
    left_footprint_known_ratio = None
    right_footprint_known_ratio = None
    side_depth_void = False
    if min_depth_m is not None and 0.0 < float(min_depth_m) < float(max_depth_m):
        near_points = project_to_base(np.full_like(z, float(min_depth_m)))
        near_x, near_y = near_points[..., 0], near_points[..., 1]
        near_forward = -near_points[..., 2]
        near_limit = max(
            forward_horizon_m,
            float(min_depth_m) + float(np.linalg.norm(transform[[0, 2], 3])),
        )
        near_band = (
            (near_y >= 0.08) & (near_y <= float(camera_height_m) + 0.35)
            & (near_forward > 0.0) & (near_forward <= near_limit)
        )
        left_strip = near_band & (near_x >= -corridor_half_width_m) & (near_x <= -float(agent_radius_m))
        right_strip = near_band & (near_x >= float(agent_radius_m)) & (near_x <= corridor_half_width_m)
        left_footprint_known_ratio = known_ratio(left_strip) if np.any(left_strip) else None
        right_footprint_known_ratio = known_ratio(right_strip) if np.any(right_strip) else None
        left_void = left_footprint_known_ratio is not None and left_footprint_known_ratio < 0.5
        right_void = right_footprint_known_ratio is not None and right_footprint_known_ratio < 0.5
        side_depth_void = bool(near_structure_count >= 8 and (left_void or right_void))
        if side_depth_void:
            blocked = True
            if not block_reason:
                block_reason = "side_depth_void"
            if left_void and not right_void:
                turn_direction = "right"
            elif right_void and not left_void:
                turn_direction = "left"

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
        "side_depth_void": side_depth_void,
        "left_footprint_known_depth_ratio": left_footprint_known_ratio,
        "right_footprint_known_depth_ratio": right_footprint_known_ratio,
        "calibrated_base_frame": base_from_camera_cv is not None,
        "near_structure_count": near_structure_count,
        "void_near_horizon_m": round(void_near_horizon_m, 4),
        "left_known_depth_ratio": round(left_known_ratio, 4),
        "right_known_depth_ratio": round(right_known_ratio, 4),
        "left_clearance_m": round(left_clearance, 4),
        "right_clearance_m": round(right_clearance, 4),
        "turn_direction": turn_direction,
    }
