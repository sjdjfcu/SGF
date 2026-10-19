"""Pivot and straight-motion calibration in the navigator's Y-up frame."""

from __future__ import annotations

import math

import numpy as np


def fit_horizontal_camera_offset(
    world_from_camera_gl: np.ndarray,
    *,
    camera_base_y_m: float,
    camera_base_yaw_deg: float = 0.0,
) -> dict:
    """Fit the horizontal camera offset from in-place pivots."""

    poses = np.asarray(world_from_camera_gl, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 8:
        raise ValueError("at least 8 camera poses of shape 4x4 are required")
    if not np.all(np.isfinite(poses)):
        raise ValueError("camera poses must be finite")
    if not math.isfinite(camera_base_y_m) or camera_base_y_m <= 0.0:
        raise ValueError("camera_base_y_m must be finite and positive")
    if not math.isfinite(camera_base_yaw_deg):
        raise ValueError("camera_base_yaw_deg must be finite")
    rotations = poses[:, :3, :3]
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5)
        or not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-3)
        or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3)
    ):
        raise ValueError("camera poses must be rigid transforms")

    yaw = math.radians(camera_base_yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    base_from_camera_rotation = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    world_from_base_rotation = rotations @ base_from_camera_rotation.T
    adjusted = poses[:, [0, 2], 3] - world_from_base_rotation[:, [0, 2], 1] * camera_base_y_m
    design = np.zeros((len(poses), 2, 4), dtype=np.float64)
    design[:, :, :2] = world_from_base_rotation[:, [0, 2], :][:, :, [0, 2]]
    design[:, :, 2:] = np.eye(2)

    def solve(mask):
        matrix = design[mask].reshape(-1, 4)
        solution, _, rank, singular = np.linalg.lstsq(matrix, adjusted[mask].reshape(-1), rcond=None)
        condition = float(singular[0] / singular[-1]) if rank == 4 else float("inf")
        return solution, condition

    inliers = np.ones(len(poses), dtype=bool)
    for _ in range(8):
        solution, condition = solve(inliers)
        errors = np.linalg.norm(design @ solution - adjusted, axis=1)
        median = float(np.median(errors[inliers]))
        mad = float(np.median(np.abs(errors[inliers] - median)))
        proposed = errors <= max(0.015, median + 3.0 * 1.4826 * mad)
        if np.count_nonzero(proposed) < 8 or np.array_equal(inliers, proposed):
            break
        inliers = proposed
    solution, condition = solve(inliers)
    errors = np.linalg.norm(design @ solution - adjusted, axis=1)
    rms = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    before = adjusted[inliers] - np.mean(adjusted[inliers], axis=0)
    before_rms = float(np.sqrt(np.mean(np.sum(before ** 2, axis=1))))

    headings = np.mod(
        np.arctan2(-world_from_base_rotation[inliers, 0, 2], world_from_base_rotation[inliers, 2, 2]),
        2.0 * np.pi,
    )
    ordered = np.sort(headings)
    gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * np.pi])
    coverage = math.degrees(2.0 * np.pi - float(np.max(gaps)))
    indices = np.flatnonzero(inliers)
    half = len(indices) // 2
    first, first_condition = solve(indices[:half])
    second, second_condition = solve(indices[half:])
    split_difference = float(np.linalg.norm(first[:2] - second[:2]))

    reasons = []
    if coverage < 180.0 or condition > 20.0:
        reasons.append("insufficient_rotation_coverage")
    if np.mean(inliers) < 0.8:
        reasons.append("too_many_outliers")
    if rms > 0.01:
        reasons.append("pivot_residual_too_large")
    if max(first_condition, second_condition) > 20.0 or split_difference > 0.015:
        reasons.append("inconsistent_rotation_halves")

    return {
        "accepted": not reasons,
        "rejection_reasons": reasons,
        "camera_base_x_m": float(solution[0]),
        "camera_base_y_m": float(camera_base_y_m),
        "camera_base_z_m": float(solution[1]),
        "camera_base_yaw_deg": float(camera_base_yaw_deg),
        "estimated_parameters": ["camera_base_x_m", "camera_base_z_m"],
        "fixed_parameters": ["camera_base_y_m", "camera_base_yaw_deg"],
        "pivot_world_xz_m": solution[2:].tolist(),
        "horizontal_offset_norm_m": float(np.linalg.norm(solution[:2])),
        "pose_count": len(poses),
        "inlier_count": int(np.count_nonzero(inliers)),
        "outlier_indices": np.flatnonzero(~inliers).tolist(),
        "rotation_coverage_deg": coverage,
        "condition_number": condition if math.isfinite(condition) else None,
        "uncorrected_pivot_rms_m": before_rms,
        "corrected_pivot_rms_m": rms,
        "split_offset_difference_m": split_difference,
        "assumptions": [
            "The base rotates about a stationary physical origin without appreciable slip.",
            "Camera height and mounting yaw are supplied, not calibrated.",
            "Mounting roll and pitch are not estimated (yaw-only extrinsic model).",
        ],
    }


def fit_mounting_yaw_from_straights(
    segments: list[np.ndarray], *, camera_frame_offset_xyz: np.ndarray,
) -> dict:
    """Recover mounting yaw from straight forward segments."""

    offset = np.asarray(camera_frame_offset_xyz, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(offset)) or len(segments) < 2:
        raise ValueError("a finite pivot offset and two forward segments are required")
    fits = []
    for segment in segments:
        poses = np.asarray(segment, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 8 or not np.all(np.isfinite(poses)):
            raise ValueError("each straight segment requires at least 8 finite poses")
        rotations = poses[:, :3, :3]
        centers = poses[:, :3, 3] - rotations @ offset
        displacement = centers[-1] - centers[0]
        displacement[1] = 0.0
        distance = float(np.linalg.norm(displacement))
        if distance < 0.20:
            raise ValueError("straight segment is too short to estimate mounting yaw")
        direction = displacement / distance
        camera_directions = rotations.transpose(0, 2, 1) @ direction
        yaw_samples = -np.arctan2(camera_directions[:, 0], -camera_directions[:, 2])
        yaw = math.atan2(float(np.mean(np.sin(yaw_samples))), float(np.mean(np.cos(yaw_samples))))
        headings = np.unwrap(np.arctan2(-rotations[:, 0, 2], rotations[:, 2, 2]))
        heading_span = math.degrees(float(np.ptp(headings)))
        lateral = np.array([-direction[2], 0.0, direction[0]])
        line_rms = float(np.sqrt(np.mean(((centers - centers[0]) @ lateral) ** 2)))
        fits.append({
            "yaw_deg": math.degrees(yaw), "distance_m": distance,
            "heading_span_deg": heading_span, "line_rms_m": line_rms,
        })
    angles = np.radians([fit["yaw_deg"] for fit in fits])
    yaw = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    disagreement = math.degrees(float(np.max(np.abs(np.angle(np.exp(1j * (angles - yaw))))))) * 2.0
    reasons = []
    if any(fit["heading_span_deg"] > 3.0 or fit["line_rms_m"] > 0.015 for fit in fits):
        reasons.append("forward_motion_not_straight")
    if disagreement > 3.0:
        reasons.append("inconsistent_forward_yaw")
    c, s = math.cos(yaw), math.sin(yaw)
    actual_offset = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]) @ offset
    return {
        "accepted": not reasons, "rejection_reasons": reasons,
        "camera_base_x_m": float(actual_offset[0]),
        "camera_base_y_m": float(actual_offset[1]),
        "camera_base_z_m": float(actual_offset[2]),
        "camera_base_yaw_deg": math.degrees(yaw),
        "yaw_disagreement_deg": disagreement, "straight_fits": fits,
        "estimated_parameters": ["camera_base_x_m", "camera_base_z_m", "camera_base_yaw_deg"],
        "fixed_parameters": ["camera_base_y_m"],
    }


def load_camera_extrinsics(path) -> dict[str, float]:
    import json
    from pathlib import Path

    payload = json.loads(Path(path).expanduser().read_text())
    result = payload.get("selected", payload)
    if result.get("accepted") is not True:
        raise ValueError("camera calibration must have accepted=true")
    keys = ("camera_base_x_m", "camera_base_y_m", "camera_base_z_m", "camera_base_yaw_deg")
    values = {key: float(result[key]) for key in keys}
    if not all(math.isfinite(value) for value in values.values()) or values["camera_base_y_m"] <= 0.0:
        raise ValueError("camera extrinsics must be finite and height must be positive")
    return values
