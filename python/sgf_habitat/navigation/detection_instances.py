"""Instance hypotheses from target detections without a scene-graph segment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DETECTION_ID_BASE = 1_000_000


@dataclass
class DetectionInstance:
    instance_id: int
    label: str
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    points: np.ndarray
    observations: int
    max_confidence: float
    first_step: int
    last_step: int
    matched_node_ids: set[int] = field(default_factory=set)


class DetectionInstanceRegistry:
    """Fuse confident target detections into position-keyed instances."""

    def __init__(
        self,
        *,
        merge_radius_m: float,
        min_observations: int,
        max_points: int = 96,
    ) -> None:
        self.merge_radius_m = float(merge_radius_m)
        self.min_observations = max(1, int(min_observations))
        self.max_points = max(8, int(max_points))
        self._instances: dict[int, DetectionInstance] = {}
        self._next_id = DETECTION_ID_BASE

    @staticmethod
    def is_detection_id(node_id: int) -> bool:
        try:
            return int(node_id) >= DETECTION_ID_BASE
        except (TypeError, ValueError):
            return False

    def size(self) -> int:
        return len(self._instances)

    def _nearest(self, centroid: np.ndarray) -> DetectionInstance | None:
        best = None
        best_distance = self.merge_radius_m
        for instance in self._instances.values():
            distance = math.hypot(
                float(centroid[0] - instance.centroid[0]),
                float(centroid[2] - instance.centroid[2]),
            )
            if distance <= best_distance:
                best = instance
                best_distance = distance
        return best

    def update(
        self,
        detections: list[dict[str, Any]],
        *,
        step: int,
        labels: set[str],
        min_confidence: float,
    ) -> int:
        """Add one frame's detections and return how many were used."""

        used = 0
        for detection in detections or ():
            label = str(detection.get("label", ""))
            if label not in labels:
                continue
            confidence = float(detection.get("confidence", 0.0) or 0.0)
            if confidence < float(min_confidence):
                continue
            points = np.asarray(detection.get("points_m"), dtype=np.float32).reshape(-1, 3)
            if int(detection.get("n_points", 0) or 0) <= 0 or points.shape[0] == 0:
                continue
            points = points[np.all(np.isfinite(points), axis=1)]
            if points.shape[0] == 0:
                continue
            # Masks of thin objects include background, so keep points near the median
            median = np.median(points, axis=0)
            keep = np.linalg.norm(points - median[None, :], axis=1) <= self.merge_radius_m
            points = points[keep]
            if points.shape[0] == 0:
                continue
            centroid = points.mean(axis=0).astype(np.float32)
            if not np.all(np.isfinite(centroid)):
                continue
            bbox_min = points.min(axis=0).astype(np.float32)
            bbox_max = points.max(axis=0).astype(np.float32)
            matched = int(detection.get("matched_node_id", 0) or 0)
            instance = self._nearest(centroid)
            if instance is None:
                instance = DetectionInstance(
                    instance_id=self._next_id,
                    label=label,
                    centroid=centroid.copy(),
                    bbox_min=bbox_min.copy(),
                    bbox_max=bbox_max.copy(),
                    points=points[: self.max_points].copy(),
                    observations=1,
                    max_confidence=confidence,
                    first_step=int(step),
                    last_step=int(step),
                )
                self._next_id += 1
                self._instances[instance.instance_id] = instance
            else:
                weight = float(instance.observations)
                instance.centroid = (instance.centroid * weight + centroid) / (weight + 1.0)
                instance.bbox_min = np.minimum(instance.bbox_min, bbox_min)
                instance.bbox_max = np.maximum(instance.bbox_max, bbox_max)
                merged = np.concatenate([instance.points, points], axis=0)
                if merged.shape[0] > self.max_points:
                    stride = int(math.ceil(merged.shape[0] / self.max_points))
                    merged = merged[::stride]
                instance.points = merged
                instance.observations += 1
                instance.last_step = int(step)
                if confidence > instance.max_confidence:
                    instance.max_confidence = confidence
                    instance.label = label
            if matched > 0:
                instance.matched_node_ids.add(matched)
            used += 1
        return used

    def candidates(
        self,
        *,
        navigation_map: Any,
        agent_position: np.ndarray,
        excluded_ids: set[int],
        excluded_positions: list[np.ndarray],
        exclusion_radius_m: float,
        graph_candidates: list[dict[str, Any]],
        association: str = "navigation_map",
        diagnostics: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Instances seen often enough and not already offered by the scene graph."""

        if association not in ("navigation_map", "reachable_euclidean", "euclidean"):
            raise ValueError(f"unknown target association {association!r}")
        euclidean = association != "navigation_map"
        require_reachable = association != "euclidean"
        if diagnostics is not None:
            diagnostics["association_unreachable_candidates"] = 0
        agent = np.asarray(agent_position, dtype=np.float32).reshape(3)
        graph_xz = []
        for candidate in graph_candidates or ():
            try:
                cxyz = np.asarray(candidate.get("centroid_m"), dtype=np.float32).reshape(3)
            except (TypeError, ValueError):
                continue
            graph_xz.append((float(cxyz[0]), float(cxyz[2])))
        out: list[dict[str, Any]] = []
        for instance in self._instances.values():
            if instance.observations < self.min_observations:
                continue
            if int(instance.instance_id) in excluded_ids:
                continue
            cx, cz = float(instance.centroid[0]), float(instance.centroid[2])
            if any(math.hypot(cx - gx, cz - gz) <= self.merge_radius_m for gx, gz in graph_xz):
                continue
            standpoint = navigation_map.snap_to_free(
                np.array([cx, float(agent[1]), cz], dtype=np.float32)
            )
            if standpoint is None or not np.all(np.isfinite(standpoint)):
                continue
            if any(
                math.hypot(float(standpoint[0]) - float(p[0]), float(standpoint[2]) - float(p[2]))
                <= float(exclusion_radius_m)
                for p in excluded_positions or ()
            ):
                continue
            try:
                distance = float(navigation_map.geodesic_distance(agent, standpoint))
            except Exception:
                if require_reachable:
                    continue
                distance = float("inf")
            if not math.isfinite(distance):
                if diagnostics is not None:
                    diagnostics["association_unreachable_candidates"] += 1
                if require_reachable:
                    continue
            association_distance = (
                math.hypot(
                    float(standpoint[0]) - float(agent[0]),
                    float(standpoint[2]) - float(agent[2]),
                )
                if euclidean
                else distance
            )
            out.append(
                {
                    "id": int(instance.instance_id),
                    "label": instance.label,
                    "argmax_label": instance.label,
                    "centroid_m": tuple(float(v) for v in instance.centroid),
                    "bbox_min_m": tuple(float(v) for v in instance.bbox_min),
                    "bbox_max_m": tuple(float(v) for v in instance.bbox_max),
                    "num_surfels": int(instance.points.shape[0]),
                    "semantic_confidence": float(instance.max_confidence),
                    "goal_position": tuple(float(v) for v in standpoint),
                    "distance_m": distance,
                    "association_distance_m": association_distance,
                    "association": association,
                    "source": "detection",
                    "observations": int(instance.observations),
                }
            )
        return out

    def summary(self, limit: int = 6) -> list[dict[str, Any]]:
        ranked = sorted(self._instances.values(), key=lambda item: (-item.observations, item.instance_id))
        return [
            {
                "id": int(item.instance_id),
                "label": item.label,
                "obs": int(item.observations),
                "c": [round(float(v), 2) for v in item.centroid],
                "conf": round(float(item.max_confidence), 2),
                "n": int(item.points.shape[0]),
            }
            for item in ranked[:limit]
        ]

    def points(self, instance_id: int) -> np.ndarray:
        instance = self._instances.get(int(instance_id))
        if instance is None:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(instance.points, dtype=np.float32).reshape(-1, 3)


__all__ = ["DETECTION_ID_BASE", "DetectionInstance", "DetectionInstanceRegistry"]
