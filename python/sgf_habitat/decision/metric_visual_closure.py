from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from operator import index
from typing import Any

from .stop_policy import RECOVER_ACTION, STOP_ACTION


@dataclass(frozen=True)
class MetricVisualClosureConfig:
    """Thresholds for the contextual-view and grounded-near closure triggers."""

    grounded_detection_confidence_min: float
    grounded_target_distance_max_m: float
    contextual_bbox_area_max_fraction: float
    grounded_bbox_center_y_min_fraction: float


@dataclass(frozen=True)
class MetricVisualClosureDecision:
    arm: bool
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True)
class MetricVisualClosureApprovalSnapshot:
    """Online evidence retained across one action boundary."""

    step: int
    raw_validation_approved: bool
    policy_action: str
    selected_target_id: int | None
    bbox_area_fraction: float | None
    recovery_executed: bool = False


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(index(value))
    except TypeError:
        return None


def _valid_target_id(value: Any) -> int | None:
    target_id = _integer(value)
    if target_id is None or target_id < 0:
        return None
    return target_id


def _snapshot_details(
    snapshot: MetricVisualClosureApprovalSnapshot | None,
) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "step": _integer(snapshot.step),
        "raw_validation_approved": bool(snapshot.raw_validation_approved),
        "policy_action": str(snapshot.policy_action or ""),
        "selected_target_id": _valid_target_id(snapshot.selected_target_id),
        "bbox_area_fraction": _finite_float(snapshot.bbox_area_fraction),
        "recovery_executed": bool(snapshot.recovery_executed),
    }


def _checked_decision(
    *,
    success_reason: str,
    checks: dict[str, bool],
    failure_reasons: dict[str, str],
    details: dict[str, Any],
) -> MetricVisualClosureDecision:
    failed_checks = [name for name, passed in checks.items() if not passed]
    arm = not failed_checks
    details["checks"] = dict(checks)
    details["failed_checks"] = list(failed_checks)
    reason = (
        success_reason
        if arm
        else failure_reasons.get(failed_checks[0], "requirements_not_satisfied")
    )
    return MetricVisualClosureDecision(
        arm=bool(arm),
        reason=str(reason),
        details=details,
    )


def decide_recovery_monotonic_closure(
    *,
    enabled: bool,
    current_step: Any,
    raw_validation_approved: bool,
    policy_action: str,
    recovery_transaction_active: bool,
    recovery_action_count: Any,
    selected_target_id: Any,
    bbox_area_fraction: Any,
    previous_snapshot: MetricVisualClosureApprovalSnapshot | None,
) -> MetricVisualClosureDecision:
    """Close after an executed RECOVER if the target box area does not shrink."""

    step = _integer(current_step)
    action_count = _integer(recovery_action_count)
    current_target_id = _valid_target_id(selected_target_id)
    current_bbox_area = _finite_float(bbox_area_fraction)
    previous = _snapshot_details(previous_snapshot)
    previous_step = None if previous is None else previous["step"]
    previous_target_id = (
        None if previous is None else previous["selected_target_id"]
    )
    previous_bbox_area = (
        None if previous is None else previous["bbox_area_fraction"]
    )

    checks = {
        "enabled": bool(enabled),
        "current_raw_approved": bool(raw_validation_approved),
        "current_policy_recover": str(policy_action or "") == RECOVER_ACTION,
        "recovery_transaction_active": bool(recovery_transaction_active),
        "recovery_action_count_positive": (
            action_count is not None and action_count >= 1
        ),
        "previous_snapshot_present": previous is not None,
        "previous_snapshot_immediately_precedes": (
            step is not None
            and previous_step is not None
            and previous_step + 1 == step
        ),
        "previous_raw_approved": bool(
            previous is not None and previous["raw_validation_approved"]
        ),
        "previous_policy_recover": bool(
            previous is not None
            and previous["policy_action"] == RECOVER_ACTION
        ),
        "previous_recovery_executed": bool(
            previous is not None and previous["recovery_executed"]
        ),
        "current_target_id_valid": current_target_id is not None,
        "previous_target_id_valid": previous_target_id is not None,
        "same_selected_target": bool(
            current_target_id is not None
            and previous_target_id is not None
            and current_target_id == previous_target_id
        ),
        "current_bbox_finite": current_bbox_area is not None,
        "previous_bbox_finite": previous_bbox_area is not None,
        "bbox_monotonic": bool(
            current_bbox_area is not None
            and previous_bbox_area is not None
            and current_bbox_area >= previous_bbox_area
        ),
    }
    failure_reasons = {
        "enabled": "recovery_monotonic_disabled",
        "current_raw_approved": "current_validation_not_raw_approved",
        "current_policy_recover": "current_policy_not_recover",
        "recovery_transaction_active": "recovery_transaction_inactive",
        "recovery_action_count_positive": "recovery_action_not_yet_executed",
        "previous_snapshot_present": "previous_snapshot_missing",
        "previous_snapshot_immediately_precedes": (
            "previous_snapshot_not_immediately_preceding"
        ),
        "previous_raw_approved": "previous_validation_not_raw_approved",
        "previous_policy_recover": "previous_policy_not_recover",
        "previous_recovery_executed": "previous_recovery_not_executed",
        "current_target_id_valid": "current_selected_target_invalid",
        "previous_target_id_valid": "previous_selected_target_invalid",
        "same_selected_target": "selected_target_changed",
        "current_bbox_finite": "current_bbox_not_finite",
        "previous_bbox_finite": "previous_bbox_not_finite",
        "bbox_monotonic": "bbox_area_regressed",
    }
    return _checked_decision(
        success_reason="executed_recovery_monotonic_bbox_closure",
        checks=checks,
        failure_reasons=failure_reasons,
        details={
            "closure": "recovery_monotonic_bbox",
            "current": {
                "step": step,
                "raw_validation_approved": bool(raw_validation_approved),
                "policy_action": str(policy_action or ""),
                "selected_target_id": current_target_id,
                "bbox_area_fraction": current_bbox_area,
                "recovery_transaction_active": bool(
                    recovery_transaction_active
                ),
                "recovery_action_count": action_count,
            },
            "previous_snapshot": previous,
            "bbox_area_delta": (
                None
                if current_bbox_area is None or previous_bbox_area is None
                else current_bbox_area - previous_bbox_area
            ),
        },
    )


def decide_contextual_stop_persistence_closure(
    *,
    enabled: bool,
    current_step: Any,
    raw_validation_approved: bool,
    policy_action: str,
    selected_target_id: Any,
    bbox_area_fraction: Any,
    target_distance_m: Any,
    scene_confirmed: bool,
    previous_snapshot: MetricVisualClosureApprovalSnapshot | None,
    close_up_bbox_area_fraction: Any,
    grounded_target_distance_max_m: Any,
) -> MetricVisualClosureDecision:
    """Close a persistent, scene-confirmed STOP in the grounded close-up band."""

    step = _integer(current_step)
    current_target_id = _valid_target_id(selected_target_id)
    bbox_area = _finite_float(bbox_area_fraction)
    target_distance = _finite_float(target_distance_m)
    close_up_bound = _finite_float(close_up_bbox_area_fraction)
    grounded_distance_bound = _finite_float(grounded_target_distance_max_m)
    previous = _snapshot_details(previous_snapshot)
    previous_step = None if previous is None else previous["step"]
    previous_target_id = (
        None if previous is None else previous["selected_target_id"]
    )

    checks = {
        "enabled": bool(enabled),
        "current_raw_approved": bool(raw_validation_approved),
        "current_policy_stop": str(policy_action or "") == STOP_ACTION,
        "previous_snapshot_present": previous is not None,
        "previous_snapshot_immediately_precedes": (
            step is not None
            and previous_step is not None
            and previous_step + 1 == step
        ),
        "previous_raw_approved": bool(
            previous is not None and previous["raw_validation_approved"]
        ),
        "current_target_id_valid": current_target_id is not None,
        "previous_target_id_valid": previous_target_id is not None,
        "same_selected_target": bool(
            current_target_id is not None
            and previous_target_id is not None
            and current_target_id == previous_target_id
        ),
        "close_up_bound_finite": close_up_bound is not None,
        "current_bbox_finite": bbox_area is not None,
        "current_bbox_close_up": bool(
            bbox_area is not None
            and close_up_bound is not None
            and bbox_area >= close_up_bound
        ),
        "grounded_distance_bound_finite": grounded_distance_bound is not None,
        "target_distance_finite": target_distance is not None,
        "target_distance_grounded": bool(
            target_distance is not None
            and grounded_distance_bound is not None
            and 0.0 <= target_distance <= grounded_distance_bound
        ),
        "scene_confirmed": bool(scene_confirmed),
    }
    failure_reasons = {
        "enabled": "contextual_closeup_persistence_disabled",
        "current_raw_approved": "current_validation_not_raw_approved",
        "current_policy_stop": "current_policy_not_stop",
        "previous_snapshot_present": "previous_snapshot_missing",
        "previous_snapshot_immediately_precedes": (
            "previous_snapshot_not_immediately_preceding"
        ),
        "previous_raw_approved": "previous_validation_not_raw_approved",
        "current_target_id_valid": "current_selected_target_invalid",
        "previous_target_id_valid": "previous_selected_target_invalid",
        "same_selected_target": "selected_target_changed",
        "close_up_bound_finite": "close_up_bound_not_finite",
        "current_bbox_finite": "current_bbox_not_finite",
        "current_bbox_close_up": "current_bbox_below_close_up_bound",
        "grounded_distance_bound_finite": "grounded_distance_bound_not_finite",
        "target_distance_finite": "target_distance_not_finite",
        "target_distance_grounded": "target_distance_outside_grounded_band",
        "scene_confirmed": "scene_level_verdict_not_approved",
    }
    return _checked_decision(
        success_reason="contextual_closeup_persistence_stop_closure",
        checks=checks,
        failure_reasons=failure_reasons,
        details={
            "closure": "contextual_closeup_persistence",
            "current": {
                "step": step,
                "raw_validation_approved": bool(raw_validation_approved),
                "policy_action": str(policy_action or ""),
                "selected_target_id": current_target_id,
                "bbox_area_fraction": bbox_area,
                "target_distance_m": target_distance,
                "scene_confirmed": bool(scene_confirmed),
            },
            "previous_snapshot": previous,
            "bounds": {
                "close_up_bbox_area_fraction": close_up_bound,
                "grounded_target_distance_max_m": grounded_distance_bound,
            },
        },
    )


def decide_metric_visual_closure(
    *,
    raw_validation_approved: bool,
    detection_found: bool,
    ordinary_sgf_stop: bool,
    policy_action: str,
    detection_confidence: Any,
    bbox_area_fraction: Any,
    bbox_center_y: Any,
    target_distance_m: Any,
    ledger_present: bool,
    ledger_close_up_dissents: Any,
    config: MetricVisualClosureConfig,
) -> MetricVisualClosureDecision:
    """Decide whether to commit STOP after one more navigation primitive."""

    confidence = _finite_float(detection_confidence)
    bbox_area = _finite_float(bbox_area_fraction)
    center_y = _finite_float(bbox_center_y)
    target_distance = _finite_float(target_distance_m)
    close_up_dissents = _finite_float(ledger_close_up_dissents)
    contextual_bbox_limit = float(
        config.contextual_bbox_area_max_fraction
    )
    grounded_target_limit = float(
        config.grounded_target_distance_max_m
    )

    envelope = bool(
        raw_validation_approved
        and detection_found
        and ordinary_sgf_stop
        and str(policy_action) == "stop"
        and ledger_present
        and close_up_dissents is not None
        and close_up_dissents <= 0.0
    )
    contextual_view = bool(
        envelope
        and bbox_area is not None
        and 0.0 < bbox_area <= contextual_bbox_limit
    )
    grounded_near = bool(
        envelope
        and confidence is not None
        and confidence >= float(config.grounded_detection_confidence_min)
        and center_y is not None
        and center_y >= float(config.grounded_bbox_center_y_min_fraction)
        and target_distance is not None
        and 0.0 <= target_distance <= grounded_target_limit
    )
    arm = bool(contextual_view or grounded_near)
    if contextual_view and grounded_near:
        reason = "contextual_view_and_grounded_near"
    elif contextual_view:
        reason = "contextual_view"
    elif grounded_near:
        reason = "grounded_near"
    elif not envelope:
        reason = "evidence_envelope_not_satisfied"
    else:
        reason = "closure_geometry_not_satisfied"
    return MetricVisualClosureDecision(
        arm=arm,
        reason=reason,
        details={
            "contextual_view": bool(contextual_view),
            "grounded_near": bool(grounded_near),
            "evidence_envelope": bool(envelope),
            "detection_confidence": confidence,
            "bbox_area_fraction": bbox_area,
            "bbox_center_y": center_y,
            "target_distance_m": target_distance,
            "ledger_close_up_dissents": close_up_dissents,
            "config": asdict(config),
        },
    )


class OnePrimitiveClosureLatch:
    """Carry one trigger across one executed action boundary."""

    def __init__(self) -> None:
        self._pending: dict[str, Any] | None = None

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def arm(
        self,
        *,
        step: int,
        action: int,
        decision: MetricVisualClosureDecision,
    ) -> bool:
        if not decision.arm or self._pending is not None:
            return False
        self._pending = {
            "trigger_step": int(step),
            "refinement_action": int(action),
            "reason": str(decision.reason),
            "details": dict(decision.details),
        }
        return True

    def pop_commit(self) -> dict[str, Any] | None:
        pending = self._pending
        self._pending = None
        return None if pending is None else dict(pending)


__all__ = [
    "MetricVisualClosureApprovalSnapshot",
    "MetricVisualClosureConfig",
    "MetricVisualClosureDecision",
    "OnePrimitiveClosureLatch",
    "decide_contextual_stop_persistence_closure",
    "decide_metric_visual_closure",
    "decide_recovery_monotonic_closure",
]
