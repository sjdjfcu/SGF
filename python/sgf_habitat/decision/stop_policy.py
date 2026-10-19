"""Stop decision from a validator verdict and measured distances."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .stop_features import StopFeatures


STOP_ACTION = "stop"
RECOVER_ACTION = "recover"
EXPLORE_ACTION = "explore"


@dataclass(frozen=True)
class StopPolicyConfig:
    """Distance scales in meters, derived in policy_scales."""

    success_radius_m: float
    far_band_radius_m: float
    terminal_approach_distance_m: float


@dataclass
class StopDecision:
    action: str
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class StopDecisionPolicy:
    """Deterministic band rule over the validator verdict and evidence distance."""

    def __init__(self, config: StopPolicyConfig) -> None:
        self.config = config

    @staticmethod
    def belief(features: StopFeatures) -> float:
        value = float(features.validator_p_yes)
        return value if math.isfinite(value) else 0.0

    def decide(
        self,
        features: StopFeatures,
        *,
        recovery_available: bool = False,
    ) -> StopDecision:
        config = self.config
        node_distance = features.node_distance_m()
        evidence_distance = float(features.evidence_distance_m)
        node_backed = math.isfinite(node_distance)
        evidence_known = math.isfinite(evidence_distance)
        details: dict[str, Any] = {
            "p_stop": round(self.belief(features), 4),
            "features": features.to_dict(),
            "node_backed": bool(node_backed),
            "node_distance_m": None if not node_backed else round(node_distance, 4),
            "evidence_distance_m": (
                None if not evidence_known else round(evidence_distance, 4)
            ),
            "success_radius_m": float(config.success_radius_m),
            "far_band_radius_m": float(config.far_band_radius_m),
            "terminal_approach_distance_m": float(
                config.terminal_approach_distance_m
            ),
        }
        if not features.validator_approved:
            return StopDecision(
                action=EXPLORE_ACTION, reason="validator_rejected", details=details
            )
        if node_backed:
            node_within = node_distance <= float(config.success_radius_m)
            evidence_within = (
                not evidence_known
                or evidence_distance <= float(config.success_radius_m)
            )
            if node_within and evidence_within:
                return StopDecision(
                    action=STOP_ACTION,
                    reason="approved_within_success_radius",
                    details=details,
                )
            return StopDecision(
                action=EXPLORE_ACTION,
                reason=(
                    "approved_beyond_success_radius"
                    if not node_within
                    else "approved_evidence_beyond_success_radius"
                ),
                details=details,
            )
        # Location probe (no scene-graph node): never stops directly
        if (
            recovery_available
            and evidence_known
            and evidence_distance <= float(config.far_band_radius_m)
        ):
            return StopDecision(
                action=RECOVER_ACTION,
                reason="approved_within_reach",
                details=details,
            )
        return StopDecision(
            action=EXPLORE_ACTION, reason="approved_beyond_reach", details=details
        )


__all__ = [
    "EXPLORE_ACTION",
    "RECOVER_ACTION",
    "STOP_ACTION",
    "StopDecision",
    "StopDecisionPolicy",
    "StopPolicyConfig",
]
