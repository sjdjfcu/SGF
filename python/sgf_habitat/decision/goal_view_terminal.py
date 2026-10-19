"""Goal-view terminal policy for instance-image navigation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _finite_vector(value: Any, size: int = 3) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size != size or not np.all(np.isfinite(array)):
        return None
    return array


def _unit_planar(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    planar = np.asarray([float(vector[0]), float(vector[2])], dtype=np.float64)
    norm = float(np.linalg.norm(planar))
    if not math.isfinite(norm) or norm <= 1e-9:
        return None
    return planar / norm


def signed_heading_error_deg(
    forward_xz: np.ndarray,
    target_xz: np.ndarray,
) -> float:
    """Signed heading error in degrees, negative when the target is to the left."""

    forward = np.asarray(forward_xz, dtype=np.float64).reshape(2)
    target = np.asarray(target_xz, dtype=np.float64).reshape(2)
    dot = float(np.clip(np.dot(forward, target), -1.0, 1.0))
    cross = float(forward[0] * target[1] - forward[1] * target[0])
    return math.degrees(math.atan2(cross, dot))


@dataclass
class GoalViewHypothesis:
    """One fused goal-camera estimate and its verification state."""

    identifier: int
    position: np.ndarray
    forward: np.ndarray | None
    anchor: np.ndarray | None
    support: float
    observations: int
    last_step: int
    misses: int = 0
    rejections: int = 0
    rejection_positions: list[tuple[float, float]] = field(default_factory=list)
    best_distance_m: float = float("inf")
    stalled_steps: int = 0
    suspended_at: tuple[float, float] | None = None
    dormant_positions: list[tuple[float, float]] = field(default_factory=list)
    refuted: bool = False
    blocked_commit: bool = False
    blocked_commit_at: tuple[float, float] | None = None
    member_positions: list[np.ndarray] = field(default_factory=list)
    member_forwards: list[np.ndarray | None] = field(default_factory=list)
    member_anchors: list[np.ndarray | None] = field(default_factory=list)
    member_weights: list[float] = field(default_factory=list)
    consensus: int = 1
    absence_refutations: int = 0
    refuted_by_absence: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "id": int(self.identifier),
            "position_m": [round(float(v), 4) for v in self.position],
            "anchor_m": (
                None
                if self.anchor is None
                else [round(float(v), 4) for v in self.anchor]
            ),
            "support": round(float(self.support), 2),
            "observations": int(self.observations),
            "consensus": int(self.consensus),
            "last_step": int(self.last_step),
            "misses": int(self.misses),
            "rejections": int(self.rejections),
            "stalled_steps": int(self.stalled_steps),
            "suspended": bool(self.suspended_at is not None),
            "refuted": bool(self.refuted),
            "blocked_commit": bool(self.blocked_commit),
        }


class GoalViewBelief:
    """Persistent, evidence-weighted goal-camera belief with a terminal rule."""

    def __init__(
        self,
        *,
        success_radius_m: float,
        forward_step_m: float,
        min_depth_m: float,
        max_depth_m: float,
        hfov_deg: float,
        turn_angle_deg: float,
        miss_budget_frames: int,
        refutation_rejections: int = 2,
        min_corroboration: int = 2,
        commit_band: bool = True,
    ) -> None:
        self.success_radius_m = float(success_radius_m)
        self.forward_step_m = float(forward_step_m)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.hfov_deg = float(hfov_deg)
        self.turn_angle_deg = float(turn_angle_deg)
        self.miss_budget_frames = max(1, int(miss_budget_frames))
        self.refutation_rejections = max(1, int(refutation_rejections))
        self.min_corroboration = max(1, int(min_corroboration))
        # Allow commits within R - s instead of at the depth floor
        self.commit_band = bool(commit_band)
        # Refuted hypotheses are kept so nearby estimates fuse into them
        self.hypotheses: list[GoalViewHypothesis] = []
        self._next_identifier = 1
        self.observation_count = 0
        self.expired_count = 0
        self.refuted_count = 0
        self.alignment_turns = 0
        self.band_commits = 0
        self.suspension_count = 0
        self.planner_handovers = 0
        self.revived_count = 0
        self.blocked_commit_count = 0
        self.uncorroborated_refutations = 0
        self.standpoint_observations = 0
        self.wide_band_commits = 0

    # ------------------------------------------------------------ scales
    @property
    def identity_radius_m(self) -> float:
        return self.success_radius_m

    @property
    def heading_tolerance_deg(self) -> float:
        return 0.5 * self.turn_angle_deg

    @property
    def fusion_radius_m(self) -> float:
        return self.forward_step_m

    def corroborated(self, hypothesis: GoalViewHypothesis | None) -> bool:
        return bool(
            hypothesis is not None
            and int(hypothesis.consensus) >= self.min_corroboration
        )

    def arrival_radius_m(
        self,
        hypothesis: GoalViewHypothesis | None,
        agent_position: Any = None,
    ) -> float:
        """Commitment band radius around the goal-camera estimate (meters)."""

        band = max(self.min_depth_m, self.success_radius_m - self.forward_step_m)
        if hypothesis is None:
            return float(band)
        agent = _finite_vector(agent_position)
        if hypothesis.blocked_commit:
            # Keep the standpoint where verification was granted inside the band
            if agent is not None and hypothesis.blocked_commit_at is not None:
                gap = math.hypot(
                    float(agent[0]) - hypothesis.blocked_commit_at[0],
                    float(agent[2]) - hypothesis.blocked_commit_at[1],
                )
                if gap <= self.forward_step_m:
                    camera_gap = math.hypot(
                        float(agent[0]) - float(hypothesis.position[0]),
                        float(agent[2]) - float(hypothesis.position[2]),
                    )
                    return float(
                        np.nextafter(
                            max(self.success_radius_m, camera_gap), np.inf
                        )
                    )
            return float(self.success_radius_m)
        if (
            not self.commit_band
            or int(hypothesis.rejections) > 0
            or not self.corroborated(hypothesis)
        ):
            return float(self.min_depth_m)
        if agent is not None and hypothesis.anchor is not None:
            camera_range = math.hypot(
                float(hypothesis.position[0]) - float(hypothesis.anchor[0]),
                float(hypothesis.position[2]) - float(hypothesis.anchor[2]),
            )
            agent_range = math.hypot(
                float(agent[0]) - float(hypothesis.anchor[0]),
                float(agent[2]) - float(hypothesis.anchor[2]),
            )
            if camera_range > self.success_radius_m and agent_range > camera_range:
                return float(self.min_depth_m)
        return float(band)

    # ------------------------------------------------------- bookkeeping
    def reset(self) -> None:
        self.hypotheses = []
        self._next_identifier = 1
        self.observation_count = 0
        self.expired_count = 0
        self.refuted_count = 0
        self.alignment_turns = 0
        self.band_commits = 0
        self.suspension_count = 0
        self.planner_handovers = 0
        self.revived_count = 0
        self.blocked_commit_count = 0
        self.uncorroborated_refutations = 0
        self.standpoint_observations = 0
        self.wide_band_commits = 0

    @property
    def active(self) -> GoalViewHypothesis | None:
        live = [item for item in self.hypotheses if not item.refuted]
        if not live:
            return None
        return max(
            live,
            key=lambda item: (float(item.support), int(item.last_step)),
        )

    def select(self, agent_position: Any) -> GoalViewHypothesis | None:
        """Best-supported hypothesis that may be pursued."""

        agent = _finite_vector(agent_position)
        for hypothesis in self.hypotheses:
            if (
                hypothesis.suspended_at is not None
                and not hypothesis.refuted
                and agent is not None
                and all(
                    math.hypot(float(agent[0]) - x, float(agent[2]) - z)
                    >= self.identity_radius_m
                    for x, z in hypothesis.dormant_positions
                )
            ):
                hypothesis.suspended_at = None
                hypothesis.best_distance_m = float("inf")
                hypothesis.stalled_steps = 0
                self.revived_count += 1
        candidates = [
            hypothesis
            for hypothesis in self.hypotheses
            if hypothesis.suspended_at is None and not hypothesis.refuted
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda item: (float(item.support), int(item.last_step)),
        )

    def note_progress(
        self,
        hypothesis: GoalViewHypothesis,
        distance_m: float,
        quantum_m: float,
    ) -> int:
        """Update approach progress and return the number of stalled steps."""

        distance = float(distance_m)
        if not math.isfinite(distance):
            return int(hypothesis.stalled_steps)
        if distance < float(hypothesis.best_distance_m) - float(quantum_m):
            hypothesis.best_distance_m = distance
            hypothesis.stalled_steps = 0
        else:
            hypothesis.stalled_steps += 1
        return int(hypothesis.stalled_steps)

    def within_reach(
        self,
        hypothesis: GoalViewHypothesis,
        agent_position: Any,
    ) -> bool:
        """Whether the agent is within R of the object anchor."""

        agent = _finite_vector(agent_position)
        if agent is None or hypothesis.anchor is None:
            return False
        object_gap = math.hypot(
            float(agent[0]) - float(hypothesis.anchor[0]),
            float(agent[2]) - float(hypothesis.anchor[2]),
        )
        return bool(object_gap <= self.success_radius_m)

    def note_blocked(
        self,
        hypothesis: GoalViewHypothesis | None,
        agent_position: Any,
    ) -> bool:
        """Grant verification after a blocked step within R of the object."""

        if (
            hypothesis is None
            or int(hypothesis.rejections) > 0
            or hypothesis.blocked_commit
            or not self.corroborated(hypothesis)
            or not self.within_reach(hypothesis, agent_position)
        ):
            return False
        agent = _finite_vector(agent_position)
        hypothesis.blocked_commit = True
        hypothesis.blocked_commit_at = (float(agent[0]), float(agent[2]))
        self.blocked_commit_count += 1
        return True

    def approach_exhausted(
        self,
        hypothesis: GoalViewHypothesis | None,
        agent_position: Any,
    ) -> bool:
        """Grant verification if within reach, otherwise suspend the hypothesis."""

        if hypothesis is None:
            return False
        if (
            int(hypothesis.rejections) == 0
            and self.corroborated(hypothesis)
            and self.within_reach(hypothesis, agent_position)
        ):
            agent = _finite_vector(agent_position)
            if not hypothesis.blocked_commit:
                self.blocked_commit_count += 1
            hypothesis.blocked_commit = True
            hypothesis.blocked_commit_at = (float(agent[0]), float(agent[2]))
            return True
        self.suspend(hypothesis, agent_position)
        return False

    def suspend(
        self,
        hypothesis: GoalViewHypothesis | None,
        agent_position: Any,
    ) -> None:
        """Stop pursuing a hypothesis from this vantage, keeping its evidence."""

        if hypothesis is None:
            return
        agent = _finite_vector(agent_position)
        if agent is None:
            self.remove(hypothesis)
            return
        hypothesis.suspended_at = (float(agent[0]), float(agent[2]))
        hypothesis.dormant_positions.append(hypothesis.suspended_at)
        hypothesis.best_distance_m = float("inf")
        hypothesis.stalled_steps = 0
        self.suspension_count += 1

    def _nearest(
        self,
        position: np.ndarray,
    ) -> GoalViewHypothesis | None:
        best = None
        best_distance = float("inf")
        for hypothesis in self.hypotheses:
            distance = float(
                np.linalg.norm((hypothesis.position - position)[[0, 2]])
            )
            if distance <= self.identity_radius_m and distance < best_distance:
                best = hypothesis
                best_distance = distance
        return best

    def observe(
        self,
        *,
        position: Any,
        forward: Any,
        anchor: Any,
        support: float,
        step: int,
        agent_position: Any = None,
    ) -> GoalViewHypothesis | None:
        """Fuse one accepted PnP goal-camera estimate."""

        centre = _finite_vector(position)
        if centre is None:
            return None
        weight = float(support)
        if not math.isfinite(weight) or weight <= 0.0:
            weight = 1.0
        axis = _finite_vector(forward)
        if axis is not None:
            norm = float(np.linalg.norm(axis))
            axis = axis / norm if norm > 1e-9 else None
        point = _finite_vector(anchor)
        self.observation_count += 1
        hypothesis = self._nearest(centre)
        if hypothesis is None:
            hypothesis = GoalViewHypothesis(
                identifier=int(self._next_identifier),
                position=centre.copy(),
                forward=None if axis is None else axis.copy(),
                anchor=None if point is None else point.copy(),
                support=weight,
                observations=1,
                last_step=int(step),
            )
            hypothesis.member_positions.append(centre.copy())
            hypothesis.member_forwards.append(None if axis is None else axis.copy())
            hypothesis.member_anchors.append(None if point is None else point.copy())
            hypothesis.member_weights.append(weight)
            self._next_identifier += 1
            self.hypotheses.append(hypothesis)
            return hypothesis
        hypothesis.member_positions.append(centre.copy())
        hypothesis.member_forwards.append(None if axis is None else axis.copy())
        hypothesis.member_anchors.append(None if point is None else point.copy())
        hypothesis.member_weights.append(weight)
        self._refuse(hypothesis)
        hypothesis.observations += 1
        hypothesis.last_step = int(step)
        hypothesis.misses = 0
        if hypothesis.refuted and hypothesis.refuted_by_absence:
            hypothesis.refuted = False
            hypothesis.refuted_by_absence = False
            self.revived_count += 1
        if hypothesis.refuted or hypothesis.suspended_at is not None:
            agent = _finite_vector(agent_position)
            if agent is not None and all(
                math.hypot(float(agent[0]) - x, float(agent[2]) - z)
                >= self.identity_radius_m
                for x, z in hypothesis.dormant_positions
            ):
                hypothesis.refuted = False
                hypothesis.suspended_at = None
                hypothesis.best_distance_m = float("inf")
                hypothesis.stalled_steps = 0
                self.revived_count += 1
        return hypothesis

    def _refuse(self, hypothesis: GoalViewHypothesis) -> None:
        """Re-fuse members around the support-weighted medoid."""

        positions = np.asarray(hypothesis.member_positions, dtype=np.float64)
        weights = np.asarray(hypothesis.member_weights, dtype=np.float64)
        planar = positions[:, [0, 2]]
        gaps = np.linalg.norm(planar[:, None, :] - planar[None, :, :], axis=2)
        medoid = int(np.argmin(gaps @ weights))
        members = gaps[medoid] <= self.fusion_radius_m
        member_weights = weights[members]
        total = float(np.sum(member_weights))
        hypothesis.position = (
            np.sum(positions[members] * member_weights[:, None], axis=0) / total
        )
        forwards = [
            (hypothesis.member_forwards[i], float(weights[i]))
            for i in np.flatnonzero(members)
            if hypothesis.member_forwards[i] is not None
        ]
        if forwards:
            fused = np.sum([vector * w for vector, w in forwards], axis=0)
            norm = float(np.linalg.norm(fused))
            if norm > 1e-9:
                hypothesis.forward = fused / norm
        anchors = [
            (hypothesis.member_anchors[i], float(weights[i]))
            for i in np.flatnonzero(members)
            if hypothesis.member_anchors[i] is not None
        ]
        if anchors:
            hypothesis.anchor = np.sum(
                [point * w for point, w in anchors], axis=0
            ) / float(sum(w for _, w in anchors))
        hypothesis.support = total
        hypothesis.consensus = int(np.count_nonzero(members))

    def remove(self, hypothesis: GoalViewHypothesis | None) -> None:
        if hypothesis is None:
            return
        self.hypotheses = [
            item for item in self.hypotheses if item.identifier != hypothesis.identifier
        ]

    # ------------------------------------------------------ visibility
    def predicted_visible(
        self,
        hypothesis: GoalViewHypothesis,
        *,
        world_from_camera_cv: np.ndarray,
        depth: np.ndarray | None,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> bool:
        """Whether this frame should have re-observed the hypothesis."""

        anchor = hypothesis.anchor
        if anchor is None:
            return False
        try:
            world_from_camera = np.asarray(world_from_camera_cv, dtype=np.float64)
            camera_from_world = np.linalg.inv(world_from_camera)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return False
        point = camera_from_world @ np.asarray(
            [anchor[0], anchor[1], anchor[2], 1.0], dtype=np.float64
        )
        z = float(point[2])
        if not math.isfinite(z) or z < self.min_depth_m or z > self.max_depth_m:
            return False
        u = float(fx) * float(point[0]) / z + float(cx)
        v = float(fy) * float(point[1]) / z + float(cy)
        if depth is None:
            return False
        height, width = int(depth.shape[0]), int(depth.shape[1])
        if not (0.0 <= u < width and 0.0 <= v < height):
            return False
        col = int(u)
        row = int(v)
        window = depth[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
        values = np.asarray(window, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values) & (values > 0.0)]
        if values.size == 0:
            return False
        measured = float(np.median(values))
        return bool(measured >= z - self.identity_radius_m)

    def note_frame(
        self,
        *,
        step: int,
        world_from_camera_cv: np.ndarray | None,
        depth: np.ndarray | None,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        confirmed: GoalViewHypothesis | None,
    ) -> list[GoalViewHypothesis]:
        """Charge a miss to every hypothesis this frame should have seen."""

        expired: list[GoalViewHypothesis] = []
        if world_from_camera_cv is None:
            return expired
        for hypothesis in list(self.hypotheses):
            if confirmed is not None and hypothesis.identifier == confirmed.identifier:
                continue
            if not self.predicted_visible(
                hypothesis,
                world_from_camera_cv=world_from_camera_cv,
                depth=depth,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            ):
                continue
            hypothesis.misses += 1
            if hypothesis.misses >= self.miss_budget_frames:
                expired.append(hypothesis)
        for hypothesis in expired:
            self.remove(hypothesis)
            self.expired_count += 1
        return expired

    # --------------------------------------------------------- terminal
    def alignment_target_xz(
        self,
        hypothesis: GoalViewHypothesis,
        agent_position: np.ndarray,
    ) -> np.ndarray | None:
        """Unit planar direction the agent should face before verifying."""

        agent = _finite_vector(agent_position)
        if agent is None:
            return None
        if hypothesis.anchor is not None:
            delta = np.asarray(
                [
                    float(hypothesis.anchor[0]) - float(agent[0]),
                    float(hypothesis.anchor[2]) - float(agent[2]),
                ],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(delta))
            # Within the depth floor, fall back to the goal camera axis
            if norm >= self.min_depth_m:
                return delta / norm
        return _unit_planar(hypothesis.forward)

    def note_uncorroborated_view(
        self,
        hypothesis: GoalViewHypothesis | None,
        agent_position: Any,
    ) -> None:
        """Refute an estimate the aligned band view failed to re-observe."""

        if hypothesis is None:
            return
        agent = _finite_vector(agent_position)
        hypothesis.refuted = True
        hypothesis.blocked_commit = False
        hypothesis.blocked_commit_at = None
        hypothesis.absence_refutations += 1
        hypothesis.refuted_by_absence = bool(
            hypothesis.absence_refutations < self.refutation_rejections
            and int(hypothesis.rejections) == 0
        )
        if agent is not None and not hypothesis.refuted_by_absence:
            hypothesis.dormant_positions.append((float(agent[0]), float(agent[2])))
        self.uncorroborated_refutations += 1

    def note_rejection(
        self,
        hypothesis: GoalViewHypothesis | None,
        agent_position: np.ndarray,
    ) -> bool:
        """Record an aligned verifier rejection and return True if refuted."""

        if hypothesis is None:
            return False
        agent = _finite_vector(agent_position)
        hypothesis.rejections += 1
        hypothesis.blocked_commit = False
        hypothesis.blocked_commit_at = None
        if agent is not None:
            hypothesis.rejection_positions.append((float(agent[0]), float(agent[2])))
            hypothesis.dormant_positions.append((float(agent[0]), float(agent[2])))
        within_floor = bool(
            agent is not None
            and float(np.linalg.norm((hypothesis.position - agent)[[0, 2]]))
            <= self.min_depth_m
        )
        if hypothesis.rejections >= self.refutation_rejections or within_floor:
            hypothesis.refuted = True
            self.refuted_count += 1
            return True
        return False

    def summary(self) -> dict[str, Any]:
        active = self.active
        return {
            "hypotheses": int(len(self.hypotheses)),
            "active": None if active is None else active.summary(),
            "arrival_radius_m": round(float(self.arrival_radius_m(active)), 4),
            "observations": int(self.observation_count),
            "expired": int(self.expired_count),
            "refuted": int(self.refuted_count),
            "alignment_turns": int(self.alignment_turns),
            "band_commits": int(self.band_commits),
            "wide_band_commits": int(self.wide_band_commits),
            "suspensions": int(self.suspension_count),
            "planner_handovers": int(self.planner_handovers),
            "revived": int(self.revived_count),
            "blocked_commits": int(self.blocked_commit_count),
            "uncorroborated_refutations": int(self.uncorroborated_refutations),
            "standpoint_observations": int(self.standpoint_observations),
        }


def goal_view_object_anchor(
    *,
    world_points: np.ndarray,
    goal_xy: np.ndarray,
    inlier_index: np.ndarray,
    rotation: np.ndarray,
    tvec: np.ndarray,
    camera_center: np.ndarray,
    camera_forward: np.ndarray,
    principal_point: tuple[float, float],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Point on the goal camera axis at the median depth of central PnP inliers."""

    diagnostics: dict[str, Any] = {}
    points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(goal_xy, dtype=np.float64).reshape(-1, 2)
    index = np.asarray(inlier_index, dtype=np.int64).reshape(-1)
    if index.size < 4:
        diagnostics["object_anchor_reject_reason"] = "too_few_inliers"
        return None, diagnostics
    index = index[(index >= 0) & (index < min(len(points), len(pixels)))]
    if index.size < 4:
        diagnostics["object_anchor_reject_reason"] = "too_few_inliers"
        return None, diagnostics
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trans = np.asarray(tvec, dtype=np.float64).reshape(3)
    camera_points = (rot @ points[index].T).T + trans
    depth = camera_points[:, 2]
    principal = np.asarray(principal_point, dtype=np.float64).reshape(2)
    radius = np.linalg.norm(pixels[index] - principal, axis=1)
    central = radius <= float(np.median(radius))
    valid = central & np.isfinite(depth) & (depth > 0.0)
    if int(np.count_nonzero(valid)) < 1:
        diagnostics["object_anchor_reject_reason"] = "no_central_depth"
        return None, diagnostics
    axis_depth = float(np.median(depth[valid]))
    centre = np.asarray(camera_center, dtype=np.float64).reshape(3)
    forward = np.asarray(camera_forward, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(axis_depth) or norm <= 1e-9:
        diagnostics["object_anchor_reject_reason"] = "degenerate_axis"
        return None, diagnostics
    anchor = centre + (forward / norm) * axis_depth
    if not np.all(np.isfinite(anchor)):
        diagnostics["object_anchor_reject_reason"] = "nonfinite_anchor"
        return None, diagnostics
    diagnostics["object_anchor_axis_depth_m"] = round(axis_depth, 4)
    diagnostics["object_anchor_central_inliers"] = int(np.count_nonzero(valid))
    return anchor, diagnostics


__all__ = [
    "GoalViewBelief",
    "GoalViewHypothesis",
    "goal_view_object_anchor",
    "signed_heading_error_deg",
]
