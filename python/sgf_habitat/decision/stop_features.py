from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _node_label(target_node: dict[str, Any] | None) -> str:
    return "" if target_node is None else str(target_node.get("label", ""))


@dataclass(frozen=True)
class StopFeatures:
    """Measured evidence for one validation attempt."""

    validator_approved: bool
    validator_p_yes: float
    validator_scene_p_yes: float
    detection_found: bool
    detection_confidence: float
    target_distance_m: float
    # Navigation diagnostic only: the endpoint may be a re-observation waypoint.
    planner_goal_distance_m: float
    label_steps: int
    tracked_steps: int
    step_fraction: float
    hint_source: str = ""
    label: str = ""
    evidence_distance_m: float = float("inf")

    def node_distance_m(self) -> float:
        """Target proximity, independent of the current navigation endpoint."""
        distance = float(self.target_distance_m)
        return distance if math.isfinite(distance) and distance >= 0.0 else float("inf")

    def stop_distance_m(self) -> float:
        """Distance for stop gating, preferring the detection-projected estimate."""
        if math.isfinite(self.evidence_distance_m):
            return self.evidence_distance_m
        return self.node_distance_m()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("target_distance_m", "planner_goal_distance_m", "evidence_distance_m"):
            value = payload[key]
            payload[key] = None if not math.isfinite(value) else value
        return payload


def compute_stop_features(
    *,
    stop_validation_result: Any = None,
    target_node: dict[str, Any] | None = None,
    planner_command: Any = None,
    goal_like_label_steps: dict[str, int] | None = None,
    tracked_steps: int = 0,
    step_index: int = 0,
    max_steps: int = 1,
    evidence_distance_m: float = float("inf"),
) -> StopFeatures:
    validator_p_yes = _clamp(_finite_float(getattr(stop_validation_result, "p_yes", 0.0)))
    validator_scene_p_yes = _clamp(_finite_float(getattr(stop_validation_result, "scene_p_yes", 0.0)))
    detection_confidence = _clamp(
        _finite_float(getattr(stop_validation_result, "detection_confidence", 0.0))
    )
    target_distance_m = _finite_float(
        None if target_node is None else target_node.get("distance_m"),
        float("inf"),
    )
    planner_goal_distance_m = _finite_float(
        getattr(planner_command, "goal_distance_m", None),
        float("inf"),
    )
    evidence_distance = _finite_float(evidence_distance_m, float("inf"))
    if not math.isfinite(evidence_distance) or evidence_distance < 0.0:
        evidence_distance = float("inf")

    label = _node_label(target_node)
    label_steps = 0
    if label and goal_like_label_steps is not None:
        try:
            label_steps = int(goal_like_label_steps.get(label, 0))
        except (TypeError, ValueError):
            label_steps = 0
    step_fraction = _clamp(float(step_index) / max(float(max_steps), 1.0))
    return StopFeatures(
        validator_approved=bool(getattr(stop_validation_result, "approved", False)),
        validator_p_yes=validator_p_yes,
        validator_scene_p_yes=validator_scene_p_yes,
        detection_found=bool(getattr(stop_validation_result, "detection_found", False)),
        detection_confidence=detection_confidence,
        target_distance_m=target_distance_m,
        planner_goal_distance_m=planner_goal_distance_m,
        label_steps=max(0, label_steps),
        tracked_steps=max(0, int(tracked_steps)),
        step_fraction=step_fraction,
        hint_source="" if target_node is None else str(target_node.get("hint_source", "")),
        label=label,
        evidence_distance_m=evidence_distance,
    )


__all__ = ["StopFeatures", "compute_stop_features"]
