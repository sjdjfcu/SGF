"""Shadow evaluation of alternative stop rules on the live verdict stream."""

from __future__ import annotations

import math
from typing import Any, Callable

from .stop_ledger import EVIDENCE_RULES, TargetStopLedger


class ShadowStopAdjudicator:
    def __init__(
        self,
        *,
        ledger_kwargs: dict[str, Any],
        commit_log_odds: float,
    ) -> None:
        bound = float(commit_log_odds)
        rules = (
            EVIDENCE_RULES
            if math.isfinite(bound) and bound > 0.0
            else tuple(rule for rule in EVIDENCE_RULES if rule != "sequential")
        )
        self._ledgers = {
            rule: TargetStopLedger(
                **ledger_kwargs,
                commit_log_odds=bound,
                rule=rule,
            )
            for rule in rules
        }
        self._first_commit: dict[str, dict[str, Any]] = {}
        self.opportunities = 0

    def record(self, **record_kwargs: Any) -> None:
        for ledger in self._ledgers.values():
            ledger.record(**record_kwargs)

    def observe_stop_opportunity(
        self,
        *,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
        approved: bool,
        p_yes: float,
        evidence_weight: float,
        agent_position_xz: tuple[float, float],
        agent_yaw_deg: float,
        step: int,
        detection_found: bool,
        distance_to_goal: Callable[[], float | None],
        path_length_m: Callable[[], float],
    ) -> None:
        """Note the first step at which each rule would commit a stop."""

        self.opportunities += 1
        for rule, ledger in self._ledgers.items():
            if rule in self._first_commit:
                continue
            sufficient, details = ledger.evidence_sufficient(
                node_id,
                node_position_xz,
                approved=approved,
                p_yes=p_yes,
                evidence_weight=evidence_weight,
                agent_position_xz=agent_position_xz,
                agent_yaw_deg=agent_yaw_deg,
                step=step,
                detection_found=detection_found,
            )
            if not sufficient:
                continue
            distance = distance_to_goal()
            self._first_commit[rule] = {
                "step": int(step),
                "distance_to_goal_m": (
                    None if distance is None else round(float(distance), 4)
                ),
                "path_length_m": round(float(path_length_m()), 4),
                "vantages": int(details["vantages"]),
            }

    def summary(self) -> dict[str, Any]:
        return {
            "stop_opportunities": int(self.opportunities),
            "rules": {
                rule: self._first_commit.get(rule) for rule in self._ledgers
            },
        }


__all__ = ["ShadowStopAdjudicator"]
