"""Per-target evidence ledger for stop adjudication."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_PROBABILITY_FLOOR = 0.01
EVIDENCE_RULES = ("sequential", "count", "single_verdict")


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _verdict_log_odds(p_yes: float) -> float:
    try:
        probability = float(p_yes)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(probability):
        return 0.0
    probability = min(
        1.0 - _PROBABILITY_FLOOR,
        max(_PROBABILITY_FLOOR, probability),
    )
    return math.log(probability / (1.0 - probability))


@dataclass(frozen=True)
class LedgerObservation:
    step: int
    agent_x: float
    agent_z: float
    agent_yaw_deg: float
    kind: str  # approval | absence | close_up | informative
    p_yes: float
    detection_found: bool
    detection_confidence: float = 0.0
    # Share of the reviewed pixels belonging to this hypothesis, in [0, 1]
    evidence_weight: float = 1.0


class TargetStopLedger:
    """Accumulates stop-validation outcomes per scene-graph target node."""

    APPROVAL = "approval"
    ABSENCE = "absence"
    CLOSE_UP = "close_up"
    INFORMATIVE = "informative"

    def __init__(
        self,
        *,
        distinct_pose_distance_m: float = 0.5,
        distinct_pose_yaw_deg: float = 45.0,
        min_commit_approvals: int = 2,
        informative_negatives_for_exclusion: int = 2,
        absence_negatives_for_exclusion: int = 3,
        node_position_match_radius_m: float = 0.75,
        close_up_bbox_area_fraction: float = 0.18,
        min_approval_step_separation: int = 8,
        close_up_dissents_for_exclusion: int = 3,
        informative_min_detection_confidence: float = 0.0,
        commit_log_odds: float = float("inf"),
        stagnation_budget: int | None = None,
        rule: str = "sequential",
        pool_vantages: bool = True,
    ) -> None:
        if rule not in EVIDENCE_RULES:
            raise ValueError(f"unknown stop evidence rule: {rule!r}")
        self.rule = str(rule)
        self.pool_vantages = bool(pool_vantages)
        # SPRT-inspired threshold
        try:
            bound = float(commit_log_odds)
        except (TypeError, ValueError):
            bound = float("inf")
        if not math.isfinite(bound) or bound <= 0.0:
            bound = float("inf")
        if self.rule == "sequential" and not math.isfinite(bound):
            raise ValueError(
                "the sequential stop rule needs a finite commit bound; set "
                "text_stop_sequential_error_rate in (0, 0.5) or choose another "
                "text_stop_evidence_rule"
            )
        self.sequential_log_odds_bound = bound
        self.commit_log_odds = bound if self.rule == "sequential" else float("inf")
        self.min_approval_step_separation = max(1, int(min_approval_step_separation))
        self.close_up_dissents_for_exclusion = max(
            1, int(close_up_dissents_for_exclusion)
        )
        self.distinct_pose_distance_m = float(distinct_pose_distance_m)
        self.distinct_pose_yaw_deg = float(distinct_pose_yaw_deg)
        self.min_commit_approvals = max(1, int(min_commit_approvals))
        self.informative_negatives_for_exclusion = max(
            1, int(informative_negatives_for_exclusion)
        )
        self.absence_negatives_for_exclusion = max(
            1, int(absence_negatives_for_exclusion)
        )
        self.node_position_match_radius_m = float(node_position_match_radius_m)
        self.close_up_bbox_area_fraction = float(close_up_bbox_area_fraction)
        self.informative_min_detection_confidence = float(
            informative_min_detection_confidence
        )
        # Verdicts allowed without a new vantage before stagnation (0 disables)
        self.stagnation_budget = (
            self.unanimous_rejection_budget()
            if stagnation_budget is None
            else max(0, int(stagnation_budget))
        )
        if not self.pool_vantages:
            self.stagnation_budget = 0
        self._observations: dict[int, list[LedgerObservation]] = {}
        self._node_positions: dict[int, tuple[float, float]] = {}
        self._vantage_counts: dict[int, int] = {}
        self._attempts_since_new_vantage: dict[int, int] = {}

    # ------------------------------------------------------------------ keys
    def resolve_node_key(
        self,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
    ) -> int:
        """Stable hypothesis key, matched by position when the node id is new."""
        node_id = int(node_id)
        if node_id in self._observations:
            return node_id
        if node_position_xz is not None:
            px, pz = float(node_position_xz[0]), float(node_position_xz[1])
            best_key = -1
            best_distance = self.node_position_match_radius_m
            for key, (kx, kz) in self._node_positions.items():
                distance = math.hypot(px - kx, pz - kz)
                if distance <= best_distance:
                    best_distance = distance
                    best_key = key
            if best_key >= 0:
                return best_key
        return node_id

    # --------------------------------------------------------------- recording
    def classify(
        self,
        *,
        approved: bool,
        detection_found: bool,
        detection_bbox_area_fraction: float | None,
        context_observability: float | None = None,
        appearance_p: float | None = None,
    ) -> str:
        if approved:
            return self.APPROVAL
        if not detection_found:
            return self.ABSENCE
        if appearance_p is not None and float(appearance_p) < 0.5:
            return self.INFORMATIVE
        # Close-up when the frame could not show the surrounding context
        if context_observability is not None:
            return (
                self.INFORMATIVE
                if float(context_observability) > 0.5
                else self.CLOSE_UP
            )
        area = _finite(detection_bbox_area_fraction, 0.0)
        if area >= self.close_up_bbox_area_fraction:
            return self.CLOSE_UP
        return self.INFORMATIVE

    def record(
        self,
        *,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
        agent_position_xz: tuple[float, float],
        agent_yaw_deg: float,
        step: int,
        approved: bool,
        p_yes: float,
        detection_found: bool,
        detection_bbox_area_fraction: float | None,
        detection_confidence: float | None = None,
        context_observability: float | None = None,
        appearance_p: float | None = None,
        evidence_weight: float = 1.0,
    ) -> dict[str, Any]:
        key = self.resolve_node_key(
            node_id,
            node_position_xz,
        )
        kind = self.classify(
            approved=approved,
            detection_found=detection_found,
            detection_bbox_area_fraction=detection_bbox_area_fraction,
            context_observability=context_observability,
            appearance_p=appearance_p,
        )
        observation = LedgerObservation(
            step=int(step),
            agent_x=_finite(agent_position_xz[0]),
            agent_z=_finite(agent_position_xz[1]),
            agent_yaw_deg=_finite(agent_yaw_deg),
            kind=kind,
            p_yes=_finite(p_yes),
            detection_found=bool(detection_found),
            detection_confidence=_finite(detection_confidence, 0.0),
            evidence_weight=min(1.0, max(0.0, _finite(evidence_weight, 1.0))),
        )
        prior_observations = self._observations.get(key, [])
        prior_observation_count = len(prior_observations)
        prior_planar_distance_m = (
            None
            if not prior_observations
            else min(
                math.hypot(
                    observation.agent_x - prior.agent_x,
                    observation.agent_z - prior.agent_z,
                )
                for prior in prior_observations
            )
        )
        prior_approvals = self.distinct_pose_count(key, self.APPROVAL)
        self._observations.setdefault(key, []).append(observation)
        vantages = self.independent_observation_count(key)
        if vantages > int(self._vantage_counts.get(key, 0)):
            self._attempts_since_new_vantage[key] = 0
        else:
            self._attempts_since_new_vantage[key] = (
                int(self._attempts_since_new_vantage.get(key, 0)) + 1
            )
        self._vantage_counts[key] = int(vantages)
        if node_position_xz is not None and key not in self._node_positions:
            self._node_positions[key] = (
                _finite(node_position_xz[0]),
                _finite(node_position_xz[1]),
            )
        return {
            "node_key": int(key),
            "kind": kind,
            "prior_observation_count": int(prior_observation_count),
            "prior_planar_distance_m": (
                None
                if prior_planar_distance_m is None
                else float(prior_planar_distance_m)
            ),
            "prior_approvals": int(prior_approvals),
            "evidence_weight": round(float(observation.evidence_weight), 4),
            "vantages": int(vantages),
            "attempts_since_new_vantage": int(
                self._attempts_since_new_vantage.get(key, 0)
            ),
            "approvals": self.distinct_pose_count(key, self.APPROVAL),
            "informative_negatives": self.distinct_pose_count(key, self.INFORMATIVE),
            "absence_negatives": self.distinct_pose_count(key, self.ABSENCE),
            "close_up_dissents": self.distinct_pose_count(key, self.CLOSE_UP),
        }

    # ---------------------------------------------------------------- counting
    def _poses_of_kind(self, node_key: int, kind: str) -> list[LedgerObservation]:
        return [
            observation
            for observation in self._observations.get(int(node_key), [])
            if observation.kind == kind
        ]

    def _pose_distinct(self, a: LedgerObservation, b: LedgerObservation) -> bool:
        if (
            math.hypot(a.agent_x - b.agent_x, a.agent_z - b.agent_z)
            >= self.distinct_pose_distance_m
        ):
            return True
        yaw_delta = abs(a.agent_yaw_deg - b.agent_yaw_deg) % 360.0
        yaw_delta = min(yaw_delta, 360.0 - yaw_delta)
        return yaw_delta >= self.distinct_pose_yaw_deg

    def _pose_independent(self, a: LedgerObservation, b: LedgerObservation) -> bool:
        separated = abs(a.step - b.step) >= self.min_approval_step_separation
        if not separated:
            return False
        if not self.pool_vantages:
            return True
        displaced = (
            math.hypot(a.agent_x - b.agent_x, a.agent_z - b.agent_z)
            >= self.distinct_pose_distance_m
        )
        if displaced:
            return True
        if a.detection_found and b.detection_found:
            yaw_delta = abs(a.agent_yaw_deg - b.agent_yaw_deg) % 360.0
            yaw_delta = min(yaw_delta, 360.0 - yaw_delta)
            return yaw_delta >= self.distinct_pose_yaw_deg
        return False

    def distinct_pose_count(self, node_key: int, kind: str) -> int:
        """Greedy count of mutually distinct poses among observations of a kind."""
        predicate = (
            self._pose_independent if kind == self.APPROVAL else self._pose_distinct
        )
        chosen: list[LedgerObservation] = []
        for observation in self._poses_of_kind(node_key, kind):
            if all(predicate(observation, other) for other in chosen):
                chosen.append(observation)
        return len(chosen)

    def _same_vantage(self, a: LedgerObservation, b: LedgerObservation) -> bool:
        return not self._pose_distinct_detected(a, b)

    def _pose_distinct_detected(self, a: LedgerObservation, b: LedgerObservation) -> bool:
        if (
            math.hypot(a.agent_x - b.agent_x, a.agent_z - b.agent_z)
            >= self.distinct_pose_distance_m
        ):
            return True
        if a.detection_found and b.detection_found:
            yaw_delta = abs(a.agent_yaw_deg - b.agent_yaw_deg) % 360.0
            yaw_delta = min(yaw_delta, 360.0 - yaw_delta)
            return yaw_delta >= self.distinct_pose_yaw_deg
        return False

    def _vantage_clusters(
        self,
        node_key: int | None,
        *,
        current: LedgerObservation | None = None,
    ) -> list[list[LedgerObservation]]:
        """Build fresh clusters, optionally previewing an unrecorded verdict."""

        observations = (
            [] if node_key is None else list(self._observations.get(int(node_key), []))
        )
        if current is not None:
            observations.append(current)
        if not self.pool_vantages:
            return [[observation] for observation in observations]
        clusters: list[list[LedgerObservation]] = []
        for observation in observations:
            for cluster in clusters:
                if self._same_vantage(observation, cluster[0]):
                    cluster.append(observation)
                    break
            else:
                clusters.append([observation])
        return clusters

    @staticmethod
    def _cluster_evidence(cluster: list[LedgerObservation]) -> tuple[float, float]:
        """Return (association-scaled evidence, total weight) of one vantage."""

        total_weight = 0.0
        weighted = 0.0
        reliability = 0.0
        for observation in cluster:
            weight = float(observation.evidence_weight)
            if weight <= 0.0:
                continue
            total_weight += weight
            weighted += weight * _verdict_log_odds(observation.p_yes)
            reliability = max(reliability, weight)
        if total_weight <= 0.0:
            return 0.0, 0.0
        return reliability * weighted / total_weight, total_weight

    def _vantage_summary(
        self,
        node_key: int | None,
        *,
        current: LedgerObservation | None = None,
    ) -> tuple[float, int]:
        """Score stored or prospective vantages without changing ledger state."""

        evidence = 0.0
        vantages = 0
        for cluster in self._vantage_clusters(node_key, current=current):
            vantage_evidence, weight = self._cluster_evidence(cluster)
            if weight > 0.0:
                evidence += vantage_evidence
                vantages += 1
        return float(evidence), vantages

    def independent_observation_count(
        self, node_id: int, node_position_xz: tuple[float, float] | None = None
    ) -> int:
        """Number of vantages from which evidence about the hypothesis exists."""

        key = self.resolve_node_key(int(node_id), node_position_xz)
        return self._vantage_summary(key)[1]

    def stagnated(
        self, node_id: int, node_position_xz: tuple[float, float] | None = None
    ) -> bool:
        """Whether the hypothesis hit its stagnation budget without a new vantage."""

        if self.stagnation_budget <= 0:
            return False
        key = self.resolve_node_key(int(node_id), node_position_xz)
        return (
            int(self._attempts_since_new_vantage.get(key, 0))
            >= self.stagnation_budget
        )

    # ---------------------------------------------------------------- decisions
    def decide_commit(
        self,
        *,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
        current_kind: str,
        within_stop_radius: bool,
        current_position_xz: tuple[float, float] | None = None,
        within_success_radius: bool = True,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Return (commit, reason, details) under the configured evidence rule."""

        arguments = dict(
            node_id=node_id,
            node_position_xz=node_position_xz,
            current_kind=current_kind,
            within_stop_radius=within_stop_radius,
            current_position_xz=current_position_xz,
            within_success_radius=within_success_radius,
        )
        if self.rule == "sequential":
            return self._decide_commit_sequential(
                bound=self.commit_log_odds, **arguments
            )
        if self.rule == "count":
            commit, reason, details = self._decide_commit_count(**arguments)
        else:
            commit, reason, details = self._decide_commit_single(**arguments)
        if math.isfinite(self.sequential_log_odds_bound):
            details["sequential_decision"] = bool(
                self._decide_commit_sequential(
                    bound=self.sequential_log_odds_bound, **arguments
                )[0]
            )
        return commit, reason, details

    def _decide_commit_count(
        self,
        *,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
        current_kind: str,
        within_stop_radius: bool,
        current_position_xz: tuple[float, float] | None = None,
        within_success_radius: bool = True,
    ) -> tuple[bool, str, dict[str, Any]]:
        _ = current_position_xz
        key = self.resolve_node_key(node_id, node_position_xz)
        approvals = self.evidence_bearing_count(key, self.APPROVAL)
        informative = self.evidence_bearing_count(key, self.INFORMATIVE)
        details = {
            "rule": "count",
            "node_key": int(key),
            "approvals": int(approvals),
            "informative_negatives": int(informative),
            "current_kind": str(current_kind),
            "within_stop_radius": bool(within_stop_radius),
            "within_success_radius": bool(within_success_radius),
        }
        if not within_stop_radius:
            return False, "outside_stop_radius", details
        if current_kind != self.APPROVAL or not self._current_bears_evidence(key):
            return False, "current_frame_not_evidence_bearing_approval", details
        if self._count_rule_satisfied(approvals, informative):
            return True, "count_rule", details
        return False, "insufficient_approval_count", details

    def _decide_commit_single(
        self,
        *,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
        current_kind: str,
        within_stop_radius: bool,
        current_position_xz: tuple[float, float] | None = None,
        within_success_radius: bool = True,
    ) -> tuple[bool, str, dict[str, Any]]:
        _ = current_position_xz
        key = self.resolve_node_key(node_id, node_position_xz)
        details = {
            "rule": "single_verdict",
            "node_key": int(key),
            "current_kind": str(current_kind),
            "within_stop_radius": bool(within_stop_radius),
            "within_success_radius": bool(within_success_radius),
        }
        if not within_stop_radius:
            return False, "outside_stop_radius", details
        if current_kind == self.APPROVAL and self._current_bears_evidence(key):
            return True, "single_verdict", details
        return False, "current_frame_not_evidence_bearing_approval", details

    def _count_rule_satisfied(self, approvals: int, informative: int) -> bool:
        return bool(
            approvals >= self.min_commit_approvals
            and approvals > informative
            and (informative == 0 or approvals >= informative + 2)
        )

    def _current_bears_evidence(self, node_key: int) -> bool:
        observations = self._observations.get(int(node_key), [])
        return bool(observations and observations[-1].evidence_weight > 0.0)

    def _decide_commit_sequential(
        self,
        *,
        bound: float,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
        current_kind: str,
        within_stop_radius: bool,
        current_position_xz: tuple[float, float] | None = None,
        within_success_radius: bool = True,
    ) -> tuple[bool, str, dict[str, Any]]:
        key = self.resolve_node_key(node_id, node_position_xz)
        approvals = self.distinct_pose_count(key, self.APPROVAL)
        informative = self.distinct_pose_count(key, self.INFORMATIVE)
        details = {
            "node_key": int(key),
            "approvals": int(approvals),
            "informative_negatives": int(informative),
            "current_kind": str(current_kind),
            "within_stop_radius": bool(within_stop_radius),
            "within_success_radius": bool(within_success_radius),
        }
        if not within_stop_radius:
            return False, "outside_stop_radius", details
        if math.isfinite(bound) and self.stagnated(key):
            details["evidence_log_odds"] = round(
                float(self.independent_log_odds(key)), 4
            )
            details["stagnated"] = True
        if current_kind == self.INFORMATIVE:
            return False, "current_frame_contradiction", details
        # Commit when association-scaled vantage evidence reaches the threshold.
        if math.isfinite(bound):
            evidence_log_odds = self.independent_log_odds(key)
            details["evidence_log_odds"] = round(float(evidence_log_odds), 4)
            details["commit_log_odds"] = round(float(bound), 4)
            if (
                current_kind == self.APPROVAL
                and evidence_log_odds >= bound
            ):
                return True, "sequential_evidence", details
        # Count routes cannot commit below the sequential bound
        if math.isfinite(bound):
            evidence_log_odds = details.get("evidence_log_odds")
            if evidence_log_odds is None:
                evidence_log_odds = round(float(self.independent_log_odds(key)), 4)
                details["evidence_log_odds"] = evidence_log_odds
            if float(evidence_log_odds) < bound:
                return False, "insufficient_sequential_evidence", details
        if (
            approvals >= self.min_commit_approvals
            and approvals > informative
            and (informative == 0 or approvals >= informative + 2)
        ):
            return True, "multi_view_approvals", details
        if (
            approvals >= 1
            and informative == 0
            and current_kind == self.CLOSE_UP
            and self._travelled_since_approval(key, current_position_xz)
        ):
            return True, "close_up_credit", details
        return False, "insufficient_ledger_evidence", details

    def _travelled_since_approval(
        self,
        node_key: int,
        current_position_xz: tuple[float, float] | None,
    ) -> bool:
        if current_position_xz is None:
            return False
        cx, cz = float(current_position_xz[0]), float(current_position_xz[1])
        approvals = self._poses_of_kind(int(node_key), self.APPROVAL)
        if not approvals:
            return False
        return all(
            math.hypot(cx - observation.agent_x, cz - observation.agent_z)
            >= self.distinct_pose_distance_m
            for observation in approvals
        )

    def decide_exclusion(
        self,
        *,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Return (exclude, reason, details) under the configured evidence rule."""

        if self.rule == "sequential":
            exclude, reason, details = self._decide_exclusion_sequential(
                bound=self.commit_log_odds,
                node_id=node_id,
                node_position_xz=node_position_xz,
            )
        else:
            if self.rule == "count":
                exclude, reason, details = self._decide_exclusion_count(
                    node_id=node_id,
                    node_position_xz=node_position_xz,
                )
            else:
                exclude, reason, details = self._decide_exclusion_single(
                    node_id=node_id,
                    node_position_xz=node_position_xz,
                )
            if math.isfinite(self.sequential_log_odds_bound):
                details["sequential_decision"] = bool(
                    self._decide_exclusion_sequential(
                        bound=self.sequential_log_odds_bound,
                        node_id=node_id,
                        node_position_xz=node_position_xz,
                    )[0]
                )
        if (
            not exclude
            and not self.pool_vantages
            and self.independent_observation_count(node_id, node_position_xz)
            >= self.unanimous_rejection_budget()
        ):
            details["requery_budget"] = int(self.unanimous_rejection_budget())
            return True, "requery_budget_exhausted", details
        return exclude, reason, details

    def _decide_exclusion_count(
        self,
        *,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
    ) -> tuple[bool, str, dict[str, Any]]:
        key = self.resolve_node_key(node_id, node_position_xz)
        approvals = self.evidence_bearing_count(key, self.APPROVAL)
        informative = self.evidence_bearing_count(key, self.INFORMATIVE)
        close_up = self.evidence_bearing_count(key, self.CLOSE_UP)
        close_up_attempts = sum(
            1
            for observation in self._poses_of_kind(key, self.CLOSE_UP)
            if observation.evidence_weight > 0.0
        )
        details = {
            "rule": "count",
            "node_key": int(key),
            "approvals": int(approvals),
            "informative_negatives": int(informative),
            "close_up_dissents": int(close_up),
            "close_up_attempts": int(close_up_attempts),
        }
        if approvals == 0:
            if informative >= 1:
                return True, "informative_contradictions", details
            if (
                close_up >= self.close_up_dissents_for_exclusion
                or close_up_attempts >= 2 * self.close_up_dissents_for_exclusion
            ):
                return True, "persistent_close_up_dissent", details
            if self.stagnated(key):
                return True, "truncated_without_approval", details
            return False, "insufficient_negative_evidence", details
        if (
            informative >= self.informative_negatives_for_exclusion
            and informative > approvals
        ):
            return True, "informative_contradictions", details
        if close_up_attempts >= 2 * self.close_up_dissents_for_exclusion + 2 * approvals:
            return True, "persistent_close_up_dissent_despite_approval", details
        return False, "insufficient_negative_evidence", details

    def _decide_exclusion_single(
        self,
        *,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
    ) -> tuple[bool, str, dict[str, Any]]:
        key = self.resolve_node_key(node_id, node_position_xz)
        observations = self._observations.get(key, [])
        current = observations[-1] if observations else None
        details = {
            "rule": "single_verdict",
            "node_key": int(key),
            "current_kind": "" if current is None else str(current.kind),
        }
        if (
            current is not None
            and current.kind == self.INFORMATIVE
            and current.evidence_weight > 0.0
        ):
            return True, "current_frame_refutation", details
        if self.stagnated(key):
            return True, "truncated_without_approval", details
        return False, "no_current_refutation", details

    def _decide_exclusion_sequential(
        self,
        *,
        bound: float,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
    ) -> tuple[bool, str, dict[str, Any]]:
        key = self.resolve_node_key(node_id, node_position_xz)
        approvals = self.distinct_pose_count(key, self.APPROVAL)
        informative = self.distinct_pose_count(key, self.INFORMATIVE)
        absence = self.distinct_pose_count(key, self.ABSENCE)
        close_up = self.distinct_pose_count(key, self.CLOSE_UP)
        details = {
            "node_key": int(key),
            "approvals": int(approvals),
            "informative_negatives": int(informative),
            "absence_negatives": int(absence),
            "close_up_dissents": int(close_up),
        }
        if math.isfinite(bound):
            evidence_log_odds = self.independent_log_odds(key)
            details["evidence_log_odds"] = round(float(evidence_log_odds), 4)
            details["exclusion_log_odds"] = round(float(-bound), 4)
            if evidence_log_odds <= -bound:
                return True, "sequential_evidence", details
            # Truncated test: retire on negative evidence once out of budget
            observations = self.independent_observation_count(key)
            details["observations"] = int(observations)
            details["recorded_observations"] = int(self.observation_count(key))
            details["attempts_since_new_vantage"] = int(
                self._attempts_since_new_vantage.get(key, 0)
            )
            if self.stagnated(key) and evidence_log_odds <= 0.0:
                details["truncated"] = True
                return True, "truncated_negative_evidence", details
            if (
                observations >= self.unanimous_rejection_budget()
                and evidence_log_odds < 0.0
            ):
                return True, "interrogation_budget_exhausted", details
            return False, "insufficient_negative_evidence", details
        if approvals == 0:
            if informative >= 1:
                return True, "informative_contradictions", details
            if absence >= 1:
                return True, "target_absent", details
            close_up_attempts = len(self._poses_of_kind(key, self.CLOSE_UP))
            details["close_up_attempts"] = int(close_up_attempts)
            if (
                close_up >= self.close_up_dissents_for_exclusion
                or close_up_attempts >= 2 * self.close_up_dissents_for_exclusion
            ):
                return True, "persistent_close_up_dissent", details
            return False, "close_up_dissent_protected", details
        if (
            informative >= self.informative_negatives_for_exclusion
            and informative > approvals
        ):
            return True, "informative_contradictions", details
        if absence >= self.absence_negatives_for_exclusion + approvals:
            return True, "persistent_absence_despite_approval", details
        close_up_attempts = len(self._poses_of_kind(key, self.CLOSE_UP))
        details["close_up_attempts"] = int(close_up_attempts)
        if close_up_attempts >= 2 * self.close_up_dissents_for_exclusion + 2 * approvals:
            return True, "persistent_close_up_dissent_despite_approval", details
        return False, "insufficient_negative_evidence", details

    def distinct_detection_approval_count(self, node_key: int) -> int:
        """Independent approvals that carried a detector box."""
        chosen: list[LedgerObservation] = []
        for observation in self._poses_of_kind(int(node_key), self.APPROVAL):
            if not observation.detection_found:
                continue
            if all(self._pose_independent(observation, other) for other in chosen):
                chosen.append(observation)
        return len(chosen)

    def prospective_distinct_detection_approval_count(
        self,
        node_key: int,
        *,
        agent_position_xz: tuple[float, float],
        agent_yaw_deg: float,
        step: int,
    ) -> int:
        """Detector-backed approval count if the current view were also recorded."""

        candidate = LedgerObservation(
            step=int(step),
            agent_x=_finite(agent_position_xz[0]),
            agent_z=_finite(agent_position_xz[1]),
            agent_yaw_deg=_finite(agent_yaw_deg),
            kind=self.APPROVAL,
            p_yes=1.0,
            detection_found=True,
            detection_confidence=1.0,
        )
        chosen: list[LedgerObservation] = []
        observations = [
            observation
            for observation in self._poses_of_kind(
                int(node_key), self.APPROVAL
            )
            if observation.detection_found
        ]
        observations.append(candidate)
        for observation in observations:
            if all(
                self._pose_independent(observation, other)
                for other in chosen
            ):
                chosen.append(observation)
        return len(chosen)

    def evidence_bearing_count(self, node_key: int, kind: str) -> int:
        """Greedy count of mutually independent evidence-bearing verdicts."""

        predicate = (
            self._pose_independent if kind == self.APPROVAL else self._pose_distinct
        )
        chosen: list[LedgerObservation] = []
        for observation in self._poses_of_kind(int(node_key), kind):
            if observation.evidence_weight <= 0.0:
                continue
            if all(predicate(observation, other) for other in chosen):
                chosen.append(observation)
        return len(chosen)

    def _prospective_observation(
        self,
        *,
        approved: bool,
        p_yes: float,
        evidence_weight: float,
        agent_position_xz: tuple[float, float],
        agent_yaw_deg: float,
        step: int,
        detection_found: bool,
    ) -> LedgerObservation:
        return LedgerObservation(
            step=int(step),
            agent_x=_finite(agent_position_xz[0]),
            agent_z=_finite(agent_position_xz[1]),
            agent_yaw_deg=_finite(agent_yaw_deg),
            kind=self.APPROVAL if approved else self.INFORMATIVE,
            p_yes=_finite(p_yes),
            detection_found=bool(detection_found),
            evidence_weight=min(1.0, max(0.0, _finite(evidence_weight, 1.0))),
        )

    def evidence_sufficient(
        self,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
        *,
        approved: bool,
        p_yes: float,
        evidence_weight: float,
        agent_position_xz: tuple[float, float],
        agent_yaw_deg: float,
        step: int,
        detection_found: bool,
        rule: str | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Preview the evidence gate after pooling the current, unrecorded verdict."""

        rule = self.rule if rule is None else str(rule)
        if rule not in EVIDENCE_RULES:
            raise ValueError(f"unknown stop evidence rule: {rule!r}")
        current = self._prospective_observation(
            approved=approved,
            p_yes=p_yes,
            evidence_weight=evidence_weight,
            agent_position_xz=agent_position_xz,
            agent_yaw_deg=agent_yaw_deg,
            step=step,
            detection_found=detection_found,
        )
        key = (
            self.resolve_node_key(int(node_id), node_position_xz)
            if int(node_id) >= 0
            else None
        )
        evidence, vantages = self._vantage_summary(key, current=current)
        details: dict[str, Any] = {
            "rule": rule,
            "node_key": -1 if key is None else int(key),
            "vantages": int(vantages),
        }
        if rule == "sequential":
            details["evidence_log_odds"] = round(float(evidence), 4)
            details["commit_log_odds"] = round(
                float(self.sequential_log_odds_bound), 4
            )
            return bool(evidence >= self.sequential_log_odds_bound), details
        if rule == "single_verdict":
            return bool(approved and current.evidence_weight > 0.0), details
        observations = [] if key is None else list(self._observations.get(key, []))
        chosen: list[LedgerObservation] = []
        candidates = [
            observation
            for observation in observations
            if observation.kind == self.APPROVAL and observation.evidence_weight > 0.0
        ]
        if approved and current.evidence_weight > 0.0:
            candidates.append(current)
        for observation in candidates:
            if all(self._pose_independent(observation, other) for other in chosen):
                chosen.append(observation)
        informative = 0 if key is None else self.evidence_bearing_count(
            key, self.INFORMATIVE
        )
        details["approvals"] = int(len(chosen))
        details["informative_negatives"] = int(informative)
        return bool(
            approved
            and current.evidence_weight > 0.0
            and self._count_rule_satisfied(len(chosen), informative)
        ), details

    def commit_holds(
        self,
        node_id: int,
        node_position_xz: tuple[float, float] | None = None,
    ) -> bool:
        """Does a standing commitment still meet its rule's standard?"""

        key = self.resolve_node_key(int(node_id), node_position_xz)
        if self.rule == "sequential":
            return bool(self.independent_log_odds(key) >= self.commit_log_odds)
        if self.rule == "count":
            return self._count_rule_satisfied(
                self.evidence_bearing_count(key, self.APPROVAL),
                self.evidence_bearing_count(key, self.INFORMATIVE),
            )
        observations = self._observations.get(key, [])
        return not (
            observations
            and observations[-1].kind == self.INFORMATIVE
            and observations[-1].evidence_weight > 0.0
        )

    def last_observation_step(
        self,
        node_id: int,
        node_position_xz: tuple[float, float] | None = None,
    ) -> int | None:
        """Step of the latest evidence-bearing verdict on a hypothesis."""

        key = self.resolve_node_key(int(node_id), node_position_xz)
        steps = [
            int(observation.step)
            for observation in self._observations.get(key, [])
            if observation.evidence_weight > 0.0
        ]
        return max(steps) if steps else None

    def unanimous_rejection_budget(self) -> int:
        return (
            self.informative_negatives_for_exclusion
            + self.absence_negatives_for_exclusion
            + self.close_up_dissents_for_exclusion
        )

    def is_unanimously_rejected(
        self, node_id: int, node_position_xz: tuple[float, float] | None = None
    ) -> bool:
        """No approvals after unanimous_rejection_budget() evidence-bearing vantages."""
        key = self.resolve_node_key(int(node_id), node_position_xz)
        observations = self._observations.get(key, [])
        if any(o.kind == self.APPROVAL for o in observations):
            return False
        return self.independent_observation_count(key) >= self.unanimous_rejection_budget()

    def hypothesis_count(self) -> int:
        return len(self._observations)

    def independent_log_odds(
        self,
        node_id: int,
        node_position_xz: tuple[float, float] | None = None,
    ) -> float:
        """Sum association-scaled per-vantage evidence for the hypothesis."""

        key = self.resolve_node_key(int(node_id), node_position_xz)
        return self._vantage_summary(key)[0]

    def repeats_vantage(
        self,
        node_id: int,
        node_position_xz: tuple[float, float] | None,
        agent_position_xz: tuple[float, float],
        agent_yaw_deg: float,
        *,
        pooled: bool | None = None,
    ) -> bool:
        """Would a verdict from this pose repeat an evidence-bearing vantage?"""

        if not (self.pool_vantages if pooled is None else pooled):
            return False
        key = self.resolve_node_key(int(node_id), node_position_xz)
        ax = _finite(agent_position_xz[0])
        az = _finite(agent_position_xz[1])
        yaw = _finite(agent_yaw_deg)
        for observation in self._observations.get(key, []):
            if observation.evidence_weight <= 0.0:
                continue
            if (
                math.hypot(ax - observation.agent_x, az - observation.agent_z)
                >= self.distinct_pose_distance_m
            ):
                continue
            yaw_delta = abs(yaw - observation.agent_yaw_deg) % 360.0
            yaw_delta = min(yaw_delta, 360.0 - yaw_delta)
            if yaw_delta >= self.distinct_pose_yaw_deg:
                continue
            return True
        return False

    def observation_positions(
        self, node_id: int, node_position_xz: tuple[float, float] | None = None
    ) -> list[tuple[float, float]]:
        """Agent positions from which the hypothesis has been adjudicated."""

        key = self.resolve_node_key(int(node_id), node_position_xz)
        return [
            (float(observation.agent_x), float(observation.agent_z))
            for observation in self._observations.get(key, [])
        ]

    def node_position(self, node_id: int) -> tuple[float, float] | None:
        key = self.resolve_node_key(int(node_id), None)
        position = self._node_positions.get(key)
        return None if position is None else (float(position[0]), float(position[1]))

    def observation_count(self, node_id: int) -> int:
        """Total attempts recorded against a hypothesis."""
        key = self.resolve_node_key(int(node_id), None)
        return len(self._observations.get(key, []))

    def max_approval_pose_separation_m(self, node_id: int) -> float:
        key = self.resolve_node_key(int(node_id), None)
        approvals = self._poses_of_kind(key, self.APPROVAL)
        best = 0.0
        for i, a in enumerate(approvals):
            for b in approvals[i + 1 :]:
                best = max(
                    best,
                    math.hypot(a.agent_x - b.agent_x, a.agent_z - b.agent_z),
                )
        return best

__all__ = ["EVIDENCE_RULES", "LedgerObservation", "TargetStopLedger"]
