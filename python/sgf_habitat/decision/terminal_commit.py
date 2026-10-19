"""Endgame return to the best-evidence pose before the step budget runs out."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TerminalCandidate:
    position: np.ndarray  # world position (3,)
    score: float
    step: int
    anchored: bool = True  # derived from a detection box
    node_key: int = -1  # ledger node key, -1 if none
    # Validator p_yes before the corroboration bonus added to score
    raw_validator_confidence: float | None = None
    probe_owned: bool = False


@dataclass
class DormantApprovedCandidate:
    """Detection-approved bank kept as a last-resort return after invalidation."""

    position: np.ndarray  # world position (3,)
    raw_validator_confidence: float
    evidence_step: int
    invalidated_step: int
    node_key: int
    probe_owned: bool = False


class TerminalCommitPlanner:
    """Tracks the best stop-candidate pose and decides when to return to it."""

    def __init__(
        self,
        *,
        min_candidate_score: float,
        margin_steps: int = 40,
        fresh_evidence_window_steps: int = 60,
        arrival_radius_m: float = 0.5,
        forward_step_m: float = 0.25,
        score_advantage: float = 0.02,
        max_replan_failures: int = 12,
    ) -> None:
        self.min_candidate_score = float(min_candidate_score)
        self.margin_steps = int(margin_steps)
        self.fresh_evidence_window_steps = int(fresh_evidence_window_steps)
        self.arrival_radius_m = float(arrival_radius_m)
        self.forward_step_m = max(1e-3, float(forward_step_m))
        self.score_advantage = float(score_advantage)
        self.max_replan_failures = int(max_replan_failures)
        self.reset()

    def reset(self) -> None:
        self.candidate: TerminalCandidate | None = None
        self.dormant_approved_candidate: DormantApprovedCandidate | None = None
        self.active = False
        self.active_source = ""
        self.disabled = False
        self.activated_step = -1
        self.hard_reserve_committed = False
        self.last_cancellation_reason: str | None = None
        self.last_fresh_evidence_step = -(10**9)
        self.replan_failures = 0
        self._score_history: list[tuple[int, float]] = []

    # ----------------------------------------------------------------- inputs
    def note_observation(self, score: float, step: int) -> None:
        """Record a score for trend tracking without banking its pose."""
        score = float(score)
        if math.isfinite(score):
            self._score_history.append((int(step), score))
            horizon = int(step) - 2 * self.fresh_evidence_window_steps
            while self._score_history and self._score_history[0][0] < horizon:
                self._score_history.pop(0)

    def note_score(
        self,
        score: float,
        position: np.ndarray,
        step: int,
        *,
        anchored: bool = True,
        node_key: int = -1,
        raw_validator_confidence: float | None = None,
        probe_owned: bool = False,
    ) -> bool:
        """Bank a scored pose if it beats the current candidate."""
        score = float(score)
        if math.isfinite(score):
            self._score_history.append((int(step), score))
            horizon = int(step) - 2 * self.fresh_evidence_window_steps
            while self._score_history and self._score_history[0][0] < horizon:
                self._score_history.pop(0)
        if not math.isfinite(score) or score < self.min_candidate_score:
            return False
        if self.candidate is not None:
            if self.candidate.anchored and not anchored:
                return False
            if (
                self.candidate.anchored == bool(anchored)
                and score <= self.candidate.score
            ):
                return False
        pose = np.asarray(position, dtype=np.float32).reshape(-1)
        if pose.size < 3 or not np.all(np.isfinite(pose[:3])):
            return False
        raw_confidence: float | None = None
        if raw_validator_confidence is not None:
            try:
                raw_confidence_value = float(raw_validator_confidence)
            except (TypeError, ValueError):
                raw_confidence_value = float("nan")
            if math.isfinite(raw_confidence_value):
                raw_confidence = raw_confidence_value
        self.candidate = TerminalCandidate(
            position=pose[:3].copy(),
            score=score,
            step=int(step),
            anchored=bool(anchored),
            node_key=int(node_key),
            raw_validator_confidence=raw_confidence,
            probe_owned=bool(probe_owned),
        )
        return True

    def invalidate_source(
        self,
        node_key: int,
        *,
        step: int,
        probe_owned: bool | None = None,
    ) -> bool:
        """Drop a bank whose source node was refuted, retaining it as dormant."""
        if (
            self.candidate is not None
            and int(node_key) >= 0
            and bool(self.candidate.anchored)
            and int(self.candidate.node_key) >= 0
            and int(self.candidate.node_key) == int(node_key)
            and (
                probe_owned is None
                or bool(self.candidate.probe_owned) == bool(probe_owned)
            )
            and not self.active
        ):
            invalidated_candidate = self.candidate
            raw_confidence = invalidated_candidate.raw_validator_confidence
            if raw_confidence is not None and math.isfinite(raw_confidence):
                dormant_candidate = DormantApprovedCandidate(
                    position=invalidated_candidate.position.copy(),
                    raw_validator_confidence=float(raw_confidence),
                    evidence_step=int(invalidated_candidate.step),
                    invalidated_step=int(step),
                    node_key=int(invalidated_candidate.node_key),
                    probe_owned=bool(invalidated_candidate.probe_owned),
                )
                dormant_rank = (
                    float(dormant_candidate.raw_validator_confidence),
                    int(dormant_candidate.invalidated_step),
                )
                current_rank = (
                    (float("-inf"), -(10**9))
                    if self.dormant_approved_candidate is None
                    else (
                        float(
                            self.dormant_approved_candidate.raw_validator_confidence
                        ),
                        int(self.dormant_approved_candidate.invalidated_step),
                    )
                )
                if dormant_rank > current_rank:
                    self.dormant_approved_candidate = dormant_candidate
            self.candidate = None
            return True
        return False

    def cancel_dormant_activation(self, reason: str) -> bool:
        """Release a just-latched dormant return for fresh trusted evidence."""

        if not (self.active and self.active_source == "dormant_approved"):
            return False
        self.active = False
        self.active_source = ""
        self.activated_step = -1
        self.hard_reserve_committed = False
        self.last_cancellation_reason = str(reason)
        return True

    def note_fresh_evidence(self, step: int) -> None:
        """Record the latest step with fresh stop evidence."""
        self.last_fresh_evidence_step = max(self.last_fresh_evidence_step, int(step))

    def note_replan_failure(self) -> None:
        self.replan_failures += 1
        if self.replan_failures > self.max_replan_failures:
            self.disabled = True
            self.active = False
            self.active_source = ""
            self.hard_reserve_committed = False

    def note_replan_success(self) -> None:
        self.replan_failures = 0

    # -------------------------------------------------------------- decisions
    @property
    def return_candidate(
        self,
    ) -> TerminalCandidate | DormantApprovedCandidate | None:
        # Dormant bank is returned only once the hard reserve latches it
        if self.active_source == "dormant_approved":
            return self.dormant_approved_candidate
        return self.candidate

    @property
    def has_return_candidate(self) -> bool:
        return self.return_candidate is not None

    @property
    def return_arrival_radius_m(self) -> float:
        return self.arrival_radius_m

    def _planar_trip_m(self, current_position: np.ndarray) -> float:
        candidate = self.return_candidate
        if candidate is None:
            return float("inf")
        agent = np.asarray(current_position, dtype=np.float32).reshape(-1)
        if agent.size < 3 or not np.all(np.isfinite(agent[:3])):
            return float("inf")
        return math.hypot(
            float(agent[0] - candidate.position[0]),
            float(agent[2] - candidate.position[2]),
        )

    @staticmethod
    def travel_inflation(turn_fraction: float) -> float:
        """Actions per forward step given turn fraction f, 1 / (1 - f)."""

        fraction = float(turn_fraction)
        if not math.isfinite(fraction) or fraction < 0.0:
            fraction = 0.0
        if fraction >= 1.0:
            return float("inf")
        return 1.0 / (1.0 - fraction)

    def estimate_return_steps(
        self,
        current_position: np.ndarray,
        *,
        return_distance_m: float | None = None,
        turn_fraction: float = 0.0,
    ) -> float:
        """Travel-only step estimate (geodesic if given, else planar)."""

        distance = (
            float(return_distance_m)
            if return_distance_m is not None
            and math.isfinite(float(return_distance_m))
            and float(return_distance_m) >= 0.0
            else self._planar_trip_m(current_position)
        )
        return distance / self.forward_step_m * self.travel_inflation(turn_fraction)

    def should_activate(
        self,
        *,
        step: int,
        max_steps: int,
        current_position: np.ndarray,
        return_distance_m: float | None = None,
        turn_fraction: float = 0.0,
        overhead_steps: int = 0,
    ) -> tuple[bool, dict[str, Any]]:
        """Decide whether the return should start now."""

        details: dict[str, Any] = {}
        candidate = self.return_candidate
        overhead = max(0, int(overhead_steps))
        if self.disabled:
            return False, details
        if self.active:
            return candidate is not None, details
        # Dormant fallback, considered only when no trusted bank exists
        if self.candidate is None and self.dormant_approved_candidate is not None:
            dormant = self.dormant_approved_candidate
            remaining = int(max_steps) - 1 - int(step)
            agent = np.asarray(current_position, dtype=np.float32).reshape(-1)
            trip_m = float("inf")
            if agent.size >= 3 and np.all(np.isfinite(agent[:3])):
                trip_m = math.hypot(
                    float(agent[0] - dormant.position[0]),
                    float(agent[2] - dormant.position[2]),
                )
            minimum_actions = (
                math.ceil(trip_m / self.forward_step_m) + overhead
                if math.isfinite(trip_m)
                else math.inf
            )
            dormant_distance_m = (
                float(return_distance_m)
                if return_distance_m is not None
                and math.isfinite(float(return_distance_m))
                and float(return_distance_m) >= 0.0
                else trip_m
            )
            return_steps = (
                dormant_distance_m
                / self.forward_step_m
                * self.travel_inflation(turn_fraction)
                if math.isfinite(dormant_distance_m)
                else float("inf")
            )
            hard_reserve = return_steps + overhead
            feasible = bool(
                math.isfinite(trip_m) and minimum_actions <= remaining
            )
            in_window = bool(
                feasible
                and math.isfinite(hard_reserve)
                and remaining <= hard_reserve
            )
            details = {
                "candidate_source": "dormant_approved",
                "remaining_steps": int(remaining),
                "trip_m": round(float(trip_m), 4),
                "estimated_return_steps": round(float(return_steps), 1),
                "minimum_return_actions": (
                    None if not math.isfinite(minimum_actions) else int(minimum_actions)
                ),
                "hard_reserve_steps": round(float(hard_reserve), 1),
                "feasible": bool(feasible),
                "hard_reserve_window": bool(in_window),
                "candidate_score": round(
                    float(dormant.raw_validator_confidence), 4
                ),
                "candidate_step": int(dormant.evidence_step),
                "candidate_invalidated_step": int(dormant.invalidated_step),
            }
            if not in_window:
                return False, details
            self.active = True
            self.active_source = "dormant_approved"
            self.activated_step = int(step)
            self.hard_reserve_committed = True
            details["forced_by_hard_reserve"] = True
            details["hard_reserve_priority"] = True
            return True, details
        if candidate is None:
            return self.active and not self.disabled, details
        remaining = int(max_steps) - 1 - int(step)
        return_steps = self.estimate_return_steps(
            current_position,
            return_distance_m=return_distance_m,
            turn_fraction=turn_fraction,
        )
        fresh_evidence = (
            int(step) - self.last_fresh_evidence_step
            <= self.fresh_evidence_window_steps
        )
        # Defer while recent scores rival the bank or are still improving
        window = self.fresh_evidence_window_steps
        recent = [s for t, s in self._score_history if t > int(step) - window]
        previous = [
            s
            for t, s in self._score_history
            if int(step) - 2 * window < t <= int(step) - window
        ]
        recent_max = max(recent) if recent else float("-inf")
        previous_max = max(previous) if previous else float("-inf")
        current_rivals_candidate = (
            recent_max >= self.candidate.score - self.score_advantage
        )
        score_trend_improving = (
            recent
            and previous
            and recent_max > previous_max + self.score_advantage
        )
        details = {
            "candidate_source": "trusted",
            "remaining_steps": int(remaining),
            "estimated_return_steps": round(float(return_steps), 1),
            "fresh_evidence": bool(fresh_evidence),
            "current_rivals_candidate": bool(current_rivals_candidate),
            "score_trend_improving": bool(score_trend_improving),
            "candidate_score": round(float(self.candidate.score), 4),
            "candidate_step": int(self.candidate.step),
        }
        # Hard reserve: activate once the budget only covers trip plus overhead
        hard_reserve = return_steps + overhead
        details["hard_reserve_steps"] = round(float(hard_reserve), 1)
        if math.isfinite(return_steps) and remaining <= hard_reserve:
            self.active = True
            self.active_source = "trusted"
            self.activated_step = int(step)
            candidate_age = max(0, int(step) - int(self.candidate.step))
            candidate_is_recent = (
                candidate_age <= self.fresh_evidence_window_steps
            )
            # Only a recent bank gets non-preemptible priority
            self.hard_reserve_committed = bool(candidate_is_recent)
            details["candidate_age_steps"] = int(candidate_age)
            details["hard_reserve_candidate_recent"] = bool(candidate_is_recent)
            details["hard_reserve_priority"] = bool(
                self.hard_reserve_committed
            )
            details["forced_by_hard_reserve"] = True
            return True, details
        # Rival/trend signals defer only when the return is not cheap
        return_is_cheap = return_steps <= float(self.margin_steps)
        details["return_is_cheap"] = bool(return_is_cheap)
        if fresh_evidence or (
            not return_is_cheap
            and (current_rivals_candidate or score_trend_improving)
        ):
            return False, details
        if remaining > return_steps + self.margin_steps:
            return False, details
        self.active = True
        self.active_source = "trusted"
        self.activated_step = int(step)
        self.hard_reserve_committed = False
        return True, details

    def arrived(self, current_position: np.ndarray) -> bool:
        candidate = self.return_candidate
        if candidate is None:
            return False
        agent = np.asarray(current_position, dtype=np.float32).reshape(-1)
        if agent.size < 3:
            return False
        planar = math.hypot(
            float(agent[0] - candidate.position[0]),
            float(agent[2] - candidate.position[2]),
        )
        return planar <= self.return_arrival_radius_m

    def summary(self) -> dict[str, Any]:
        return {
            "active": bool(self.active),
            "active_source": self.active_source or None,
            "disabled": bool(self.disabled),
            "activated_step": int(self.activated_step),
            "hard_reserve_committed": bool(self.hard_reserve_committed),
            "replan_failures": int(self.replan_failures),
            "dormant_candidate_present": (
                self.dormant_approved_candidate is not None
            ),
            "last_cancellation_reason": self.last_cancellation_reason,
            "candidate": (
                None
                if self.candidate is None
                else {
                    "position": [round(float(v), 4) for v in self.candidate.position],
                    "score": round(float(self.candidate.score), 4),
                    "step": int(self.candidate.step),
                    "anchored": bool(self.candidate.anchored),
                    "node_key": int(self.candidate.node_key),
                    "probe_owned": bool(self.candidate.probe_owned),
                    "raw_validator_confidence": (
                        None
                        if self.candidate.raw_validator_confidence is None
                        else round(
                            float(self.candidate.raw_validator_confidence), 4
                        )
                    ),
                }
            ),
            "dormant_candidate": (
                None
                if self.dormant_approved_candidate is None
                else {
                    "position": [
                        round(float(v), 4)
                        for v in self.dormant_approved_candidate.position
                    ],
                    "raw_validator_confidence": round(
                        float(
                            self.dormant_approved_candidate.raw_validator_confidence
                        ),
                        4,
                    ),
                    "evidence_step": int(
                        self.dormant_approved_candidate.evidence_step
                    ),
                    "invalidated_step": int(
                        self.dormant_approved_candidate.invalidated_step
                    ),
                    "node_key": int(self.dormant_approved_candidate.node_key),
                    "probe_owned": bool(
                        self.dormant_approved_candidate.probe_owned
                    ),
                }
            ),
        }


__all__ = [
    "DormantApprovedCandidate",
    "TerminalCandidate",
    "TerminalCommitPlanner",
]
