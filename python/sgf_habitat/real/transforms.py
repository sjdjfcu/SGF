from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import quaternion


def transform_from_translation_yaw(
    translation_xyz: tuple[float, float, float] | list[float] | np.ndarray,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """Return an OpenGL-style Y-up transform from translation and yaw."""
    translation = np.asarray(translation_xyz, dtype=np.float32).reshape(3)
    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.asarray(
        [
            [c, 0.0, -s],
            [0.0, 1.0, 0.0],
            [s, 0.0, c],
        ],
        dtype=np.float32,
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def quaternion_from_rotation_matrix(rotation_matrix: np.ndarray):
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    return quaternion.from_rotation_matrix(rotation)


def state_from_transform(transform: np.ndarray):
    matrix = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    return SimpleNamespace(
        position=matrix[:3, 3].astype(np.float32).copy(),
        rotation=quaternion_from_rotation_matrix(matrix[:3, :3]),
    )


def world_from_base_from_camera_pose(
    world_from_camera: np.ndarray,
    base_from_camera: np.ndarray,
) -> np.ndarray:
    """Convert a tracked camera pose into the corresponding base pose."""
    world_from_camera = np.asarray(world_from_camera, dtype=np.float32).reshape(4, 4)
    base_from_camera = np.asarray(base_from_camera, dtype=np.float32).reshape(4, 4)
    return world_from_camera @ np.linalg.inv(base_from_camera).astype(np.float32)


def yaw_from_rotation(rotation) -> float:
    matrix = quaternion.as_rotation_matrix(rotation).astype(np.float32)
    forward = matrix @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return float(math.atan2(float(forward[0]), float(-forward[2])))


def angle_delta_rad(start: float, current: float) -> float:
    return float(math.atan2(math.sin(current - start), math.cos(current - start)))
