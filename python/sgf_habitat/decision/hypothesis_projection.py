"""Projection of 3D scene-graph hypotheses into the current frame."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ProjectedHypothesis:
    """Image-space footprint of one 3D scene-graph hypothesis."""

    bbox_xyxy: tuple[int, int, int, int]
    area_fraction: float
    center: tuple[float, float]
    near_m: float
    far_m: float
    containment: float
    depth_support: float
    depth_samples: int

    @property
    def truncated(self) -> bool:
        return self.containment < 1.0


def _finite_vec3(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        return None
    return vector


def box_corners(bbox_min_m: Any, bbox_max_m: Any) -> np.ndarray | None:
    """Return the eight world-space corners of an axis-aligned box."""

    low = _finite_vec3(bbox_min_m)
    high = _finite_vec3(bbox_max_m)
    if low is None or high is None:
        return None
    lower = np.minimum(low, high)
    upper = np.maximum(low, high)
    if float(np.max(upper - lower)) <= 0.0:
        return None
    return np.asarray(
        [
            [lower[0] if not x else upper[0],
             lower[1] if not y else upper[1],
             lower[2] if not z else upper[2]]
            for x in (0, 1)
            for y in (0, 1)
            for z in (0, 1)
        ],
        dtype=np.float64,
    )


def camera_from_world_transform(world_from_camera_cv: Any) -> np.ndarray | None:
    """Invert a rigid camera-to-world transform without a general solve."""

    try:
        transform = np.asarray(world_from_camera_cv, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        return None
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def _depth_2d(depth: Any) -> np.ndarray | None:
    if depth is None:
        return None
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return array if array.ndim == 2 else None


def project_hypothesis(
    *,
    bbox_min_m: Any,
    bbox_max_m: Any,
    world_from_camera_cv: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    depth: Any = None,
    depth_tolerance_m: float = 0.0,
) -> ProjectedHypothesis | None:
    """Return the image footprint of a node's 3D box, or None if not in view."""

    corners = box_corners(bbox_min_m, bbox_max_m)
    if corners is None:
        return None
    camera_from_world = camera_from_world_transform(world_from_camera_cv)
    if camera_from_world is None:
        return None
    if not (float(fx) > 0.0 and float(fy) > 0.0):
        return None
    image_width = int(width)
    image_height = int(height)
    if image_width <= 0 or image_height <= 0:
        return None

    homogeneous = np.concatenate(
        [corners, np.ones((corners.shape[0], 1), dtype=np.float64)], axis=1
    )
    camera_points = (camera_from_world @ homogeneous.T).T[:, :3]
    depths = camera_points[:, 2]
    in_front = depths > 1e-3
    if not bool(np.any(in_front)):
        return None
    visible = camera_points[in_front]
    u = float(fx) * visible[:, 0] / visible[:, 2] + float(cx)
    v = float(fy) * visible[:, 1] / visible[:, 2] + float(cy)
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return None

    raw_x1, raw_x2 = float(np.min(u)), float(np.max(u))
    raw_y1, raw_y2 = float(np.min(v)), float(np.max(v))
    raw_area = max(1e-9, (raw_x2 - raw_x1) * (raw_y2 - raw_y1))

    x1 = int(math.floor(max(0.0, min(raw_x1, image_width - 1.0))))
    y1 = int(math.floor(max(0.0, min(raw_y1, image_height - 1.0))))
    x2 = int(math.ceil(max(float(x1) + 1.0, min(raw_x2, float(image_width)))))
    y2 = int(math.ceil(max(float(y1) + 1.0, min(raw_y2, float(image_height)))))
    clipped_width = max(0.0, min(raw_x2, float(image_width)) - max(raw_x1, 0.0))
    clipped_height = max(0.0, min(raw_y2, float(image_height)) - max(raw_y1, 0.0))
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None
    containment = float(
        min(1.0, max(0.0, (clipped_width * clipped_height) / raw_area))
    )

    near_m = float(np.min(visible[:, 2]))
    far_m = float(np.max(visible[:, 2]))
    area_fraction = float((x2 - x1) * (y2 - y1)) / float(
        image_width * image_height
    )
    center = (
        (float(x1) + float(x2)) * 0.5 / float(image_width),
        (float(y1) + float(y2)) * 0.5 / float(image_height),
    )

    depth_support = 0.0
    depth_samples = 0
    depth_map = _depth_2d(depth)
    if depth_map is not None and depth_map.shape[:2] == (image_height, image_width):
        window = depth_map[y1:y2, x1:x2]
        finite = np.isfinite(window) & (window > 0.0)
        depth_samples = int(np.count_nonzero(finite))
        if depth_samples > 0:
            tolerance = max(0.0, float(depth_tolerance_m))
            agree = finite & (window >= near_m - tolerance) & (
                window <= far_m + tolerance
            )
            depth_support = float(np.count_nonzero(agree)) / float(depth_samples)

    return ProjectedHypothesis(
        bbox_xyxy=(x1, y1, x2, y2),
        area_fraction=area_fraction,
        center=center,
        near_m=near_m,
        far_m=far_m,
        containment=containment,
        depth_support=depth_support,
        depth_samples=depth_samples,
    )


def _voxel_keys(points: np.ndarray, resolution_m: float) -> np.ndarray:
    cells = np.floor(
        np.asarray(points, dtype=np.float64) / max(1e-6, float(resolution_m))
    ).astype(np.int64)
    # Spatial hash primes from Teschner et al. (2003)
    return (
        cells[:, 0] * np.int64(73856093)
        ^ cells[:, 1] * np.int64(19349663)
        ^ cells[:, 2] * np.int64(83492791)
    )


def project_hypothesis_points(
    points_world: Any,
    *,
    world_from_camera_cv: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    depth: Any,
    map_resolution_m: float,
    min_depth_m: float,
    max_depth_m: float,
    pixel_stride: int = 4,
    goal_position_m: Any = None,
) -> ProjectedHypothesis | None:
    """Return the image region where the depth frame observes a node's voxels."""

    try:
        points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    except (TypeError, ValueError):
        return None
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] == 0:
        return None
    camera_from_world = camera_from_world_transform(world_from_camera_cv)
    if camera_from_world is None:
        return None
    if not (float(fx) > 0.0 and float(fy) > 0.0):
        return None
    image_width, image_height = int(width), int(height)
    depth_map = _depth_2d(depth)
    if (
        image_width <= 0
        or image_height <= 0
        or depth_map is None
        or depth_map.shape[:2] != (image_height, image_width)
    ):
        return None

    resolution = max(1e-3, float(map_resolution_m))
    node_keys = np.unique(_voxel_keys(points, resolution))
    if node_keys.size == 0:
        return None

    stride = max(1, int(pixel_stride))
    rows = np.arange(0, image_height, stride)
    columns = np.arange(0, image_width, stride)
    if rows.size == 0 or columns.size == 0:
        return None
    grid_v, grid_u = np.meshgrid(rows, columns, indexing="ij")
    sampled_depth = depth_map[grid_v, grid_u].astype(np.float64)
    usable = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= float(min_depth_m))
        & (sampled_depth <= float(max_depth_m))
    )
    if not bool(np.any(usable)):
        return None
    z = sampled_depth[usable]
    u_px = grid_u[usable].astype(np.float64)
    v_px = grid_v[usable].astype(np.float64)
    camera_points = np.stack(
        [
            (u_px - float(cx)) / float(fx) * z,
            (v_px - float(cy)) / float(fy) * z,
            z,
        ],
        axis=1,
    )
    world_transform = np.asarray(world_from_camera_cv, dtype=np.float64)
    world_points = (
        world_transform[:3, :3] @ camera_points.T
    ).T + world_transform[:3, 3]
    hit = np.isin(_voxel_keys(world_points, resolution), node_keys)
    if not bool(np.any(hit)):
        return None

    grid_height, grid_width = rows.size, columns.size
    mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
    row_index = (grid_v[usable] // stride).astype(np.int64)
    column_index = (grid_u[usable] // stride).astype(np.int64)
    mask[row_index[hit], column_index[hit]] = 255
    dilated = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
    component_count, labels = cv2.connectedComponents(dilated)
    selected_mask = mask > 0
    if component_count > 2:
        preferred = 0
        goal = _finite_vec3(goal_position_m)
        if goal is not None:
            goal_camera = camera_from_world[:3, :3] @ goal + camera_from_world[:3, 3]
            if float(goal_camera[2]) > 1e-3:
                goal_u = int(
                    round(float(fx) * goal_camera[0] / goal_camera[2] + float(cx))
                )
                goal_v = int(
                    round(float(fy) * goal_camera[1] / goal_camera[2] + float(cy))
                )
                if 0 <= goal_u < image_width and 0 <= goal_v < image_height:
                    preferred = int(
                        labels[
                            min(grid_height - 1, goal_v // stride),
                            min(grid_width - 1, goal_u // stride),
                        ]
                    )
        if preferred <= 0:
            sizes = np.bincount(
                labels[selected_mask].reshape(-1), minlength=component_count
            )
            sizes[0] = 0
            preferred = int(np.argmax(sizes))
        component = selected_mask & (labels == preferred)
        if bool(np.any(component)):
            selected_mask = component

    component_rows, component_columns = np.nonzero(selected_mask)
    y1 = int(rows[int(component_rows.min())])
    y2 = int(min(image_height, rows[int(component_rows.max())] + stride))
    x1 = int(columns[int(component_columns.min())])
    x2 = int(min(image_width, columns[int(component_columns.max())] + stride))
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)

    box_rows = (rows >= y1) & (rows < y2)
    box_columns = (columns >= x1) & (columns < x2)
    box_cells = int(np.count_nonzero(box_rows) * np.count_nonzero(box_columns))
    node_cells = int(np.count_nonzero(selected_mask))
    depth_support = float(node_cells) / float(max(1, box_cells))

    hit_keys = _voxel_keys(world_points[hit], resolution)
    containment = float(np.unique(hit_keys).size) / float(node_keys.size)

    component_depths = z[
        np.isin(
            row_index * grid_width + column_index,
            np.nonzero(selected_mask.reshape(-1))[0],
        )
    ]
    near_m = float(component_depths.min()) if component_depths.size else float(z.min())
    far_m = float(component_depths.max()) if component_depths.size else float(z.max())

    area_fraction = float((x2 - x1) * (y2 - y1)) / float(image_width * image_height)
    center = (
        (float(x1) + float(x2)) * 0.5 / float(image_width),
        (float(y1) + float(y2)) * 0.5 / float(image_height),
    )
    return ProjectedHypothesis(
        bbox_xyxy=(x1, y1, x2, y2),
        area_fraction=area_fraction,
        center=center,
        near_m=near_m,
        far_m=far_m,
        containment=min(1.0, containment),
        depth_support=min(1.0, depth_support),
        depth_samples=node_cells,
    )


def _intersection_area(
    first: Sequence[float], second: Sequence[float]
) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def detection_share_inside(
    detection_bbox_xyxy: Sequence[float],
    footprint_bbox_xyxy: Sequence[float],
) -> float:
    """Return the share of a detection box that lies inside a footprint."""

    intersection = _intersection_area(detection_bbox_xyxy, footprint_bbox_xyxy)
    if intersection <= 0.0:
        return 0.0
    detection_area = max(
        1e-9,
        (float(detection_bbox_xyxy[2]) - float(detection_bbox_xyxy[0]))
        * (float(detection_bbox_xyxy[3]) - float(detection_bbox_xyxy[1])),
    )
    return float(min(1.0, max(0.0, intersection / detection_area)))


def overlap_with_hypothesis(
    detection_bbox_xyxy: Sequence[float],
    hypothesis_bbox_xyxy: Sequence[float],
) -> float:
    """Return intersection over the smaller of the two box areas."""

    intersection = _intersection_area(detection_bbox_xyxy, hypothesis_bbox_xyxy)
    if intersection <= 0.0:
        return 0.0
    hypothesis_area = max(
        1e-9,
        (float(hypothesis_bbox_xyxy[2]) - float(hypothesis_bbox_xyxy[0]))
        * (float(hypothesis_bbox_xyxy[3]) - float(hypothesis_bbox_xyxy[1])),
    )
    detection_area = max(
        1e-9,
        (float(detection_bbox_xyxy[2]) - float(detection_bbox_xyxy[0]))
        * (float(detection_bbox_xyxy[3]) - float(detection_bbox_xyxy[1])),
    )
    return float(
        min(
            1.0,
            max(intersection / hypothesis_area, intersection / detection_area),
        )
    )


def select_grounded_detection(
    detections: Iterable[dict[str, Any]],
    hypothesis: ProjectedHypothesis,
) -> tuple[dict[str, Any] | None, float]:
    """Return the most confident detection with overlap > 0.5, and its overlap."""

    best: dict[str, Any] | None = None
    best_key: tuple[float, float, tuple[int, int, int, int]] | None = None
    best_score = 0.0
    for detection in detections:
        bbox = detection.get("bbox_xyxy")
        if bbox is None or len(tuple(bbox)) != 4:
            continue
        score = overlap_with_hypothesis(bbox, hypothesis.bbox_xyxy)
        if score <= 0.5:
            continue
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        key = (
            confidence,
            score,
            tuple(int(value) for value in bbox),  # type: ignore[misc]
        )
        if best_key is None or key > best_key:
            best_key = key
            best = detection
            best_score = score
    return best, (0.0 if best is None else float(best_score))


__all__ = [
    "ProjectedHypothesis",
    "box_corners",
    "camera_from_world_transform",
    "detection_share_inside",
    "overlap_with_hypothesis",
    "project_hypothesis",
    "project_hypothesis_points",
    "select_grounded_detection",
]
