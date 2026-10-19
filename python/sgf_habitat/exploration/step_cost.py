"""Frontier travel cost in actions (forward steps plus in-place turns)."""

from __future__ import annotations

import math

import numpy as np


_TWO_PI = 2.0 * math.pi


def wrap_signed(angle_rad: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi)."""

    values = np.asarray(angle_rad, dtype=np.float64)
    return (values + math.pi) % _TWO_PI - math.pi


def frontier_bearings_rad(
    position_xz: np.ndarray,
    goals_xz: np.ndarray,
) -> np.ndarray:
    """Bearing to each goal in the atan2(-dx, -dz) heading convention."""

    origin = np.asarray(position_xz, dtype=np.float64).reshape(2)
    goals = np.asarray(goals_xz, dtype=np.float64).reshape(-1, 2)
    delta = goals - origin[None, :]
    return np.arctan2(-delta[:, 0], -delta[:, 1])


def kinematic_step_costs(
    distance_m: np.ndarray,
    bearing_rad: np.ndarray,
    heading_rad: float,
    *,
    forward_step_size_m: float,
    turn_angle_deg: float,
    rotation_weight: float = 1.0,
) -> np.ndarray:
    """Actions to reach each frontier, d / step + |bearing - heading| / turn."""

    distance = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    bearing = np.asarray(bearing_rad, dtype=np.float64).reshape(-1)
    if distance.size != bearing.size:
        raise ValueError("distance_m and bearing_rad must share one length")
    if distance.size == 0:
        raise ValueError("distance_m must be non-empty")

    step_size = float(forward_step_size_m)
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("forward_step_size_m must be finite and positive")
    turn_angle = float(turn_angle_deg)
    if not math.isfinite(turn_angle) or turn_angle <= 0.0:
        raise ValueError("turn_angle_deg must be finite and positive")
    heading = float(heading_rad)
    if not math.isfinite(heading):
        raise ValueError("heading_rad must be finite")
    weight = float(rotation_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("rotation_weight must be finite and non-negative")
    translation_steps = np.where(
        np.isfinite(distance),
        np.maximum(distance, 0.0) / step_size,
        np.inf,
    )
    turn_step_rad = math.radians(turn_angle)
    rotation_steps = np.abs(wrap_signed(bearing - heading)) / turn_step_rad
    if weight == 1.0:
        return translation_steps + rotation_steps
    return translation_steps + weight * rotation_steps


def action_rate_scaling(
    step_costs: np.ndarray,
    *,
    step_offset: float,
) -> np.ndarray:
    """Elementwise 1 / (steps + offset), zero for unreachable candidates."""

    costs = np.asarray(step_costs, dtype=np.float64).reshape(-1)
    if costs.size == 0:
        raise ValueError("step_costs must be non-empty")

    offset = float(step_offset)
    if not math.isfinite(offset) or offset <= 0.0:
        raise ValueError("step_offset must be finite and positive")

    safe_costs = np.where(np.isfinite(costs), np.maximum(costs, 0.0), np.inf)
    scale = 1.0 / (safe_costs + offset)
    return np.where(np.isfinite(scale), scale, 0.0)
