"""Frontier-based exploration driven by GraphSLAM's navigation map."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class GoalEndReason(str, Enum):
    FAILED = "failed"
    REACHED = "reached"
    PROGRESS_STALL = "progress_stall"
    POSITION_STAGNATION = "position_stagnation"
    PLANNER_GOAL_BLOCKED = "planner_goal_blocked"
    PLANNER_LOCAL_BLOCKED = "planner_local_blocked"
    DEPTH_BLOCKED = "depth_blocked"
    SUPPORT_INVALIDATED = "support_invalidated"
    POLICY_DISABLED = "policy_disabled"
    EVIDENCE_PREEMPTED = "evidence_preempted"


@dataclass(frozen=True)
class GoalEndEvent:
    """Event emitted once when a live goal ends."""

    sequence: int
    reason: GoalEndReason
    goal: tuple[float, float, float]
    owner: str | None
    token: int | None
    acknowledged_actions: int = 0


class NavigationMapExplorer:
    def __init__(
        self,
        navigation_map,
        visited_radius_cells: int = 6,
        min_distance_cells: int = 24,
        min_goal_distance_m: float = 1.5,
        min_frontier_size: int = 20,
        stagnation_window: int = 30,
        stagnation_threshold_m: float = 0.75,
        failed_goal_radius_m: float = 1.0,
        failed_goal_repeat_penalty_count: int = 3,
        failed_goal_repeat_penalty_scale: float = 4.0,
        goal_progress_window: int = 20,
        goal_progress_min_improvement_m: float = 0.20,
        goal_reached_radius_m: float = 0.75,
        same_goal_distance_m: float = 0.25,
        evidence_verification: bool = False,
        evidence_max_age_steps: int = 12,
    ) -> None:
        self.navigation_map = navigation_map
        self._goal_reached_radius_m = max(1e-3, float(goal_reached_radius_m))
        self._same_goal_distance_m = max(1e-3, float(same_goal_distance_m))
        self.visited_radius_cells = visited_radius_cells
        self.min_distance_cells = min_distance_cells
        self.min_goal_distance_m = min_goal_distance_m
        self.min_frontier_size = min_frontier_size
        self.stagnation_window = stagnation_window
        self.stagnation_threshold_m = stagnation_threshold_m
        self.visited: np.ndarray | None = None
        self.topdown_map: np.ndarray | None = None
        self.current_goal: np.ndarray | None = None
        self._visited_origin_xz: np.ndarray | None = None
        self._position_history: list[np.ndarray] = []
        self._failed_goals: list[np.ndarray] = []
        self._failed_goal_counts: list[int] = []
        self._failed_goal_penalty_scales: list[float] = []
        self._failed_goal_radius_m: float = float(failed_goal_radius_m)
        self._failed_goal_repeat_penalty_count = max(1, int(failed_goal_repeat_penalty_count))
        self._failed_goal_repeat_penalty_scale = max(1.0, float(failed_goal_repeat_penalty_scale))
        self._goal_progress_window = max(1, int(goal_progress_window))
        self._goal_progress_min_improvement_m = max(0.0, float(goal_progress_min_improvement_m))
        self._evidence_verification = bool(evidence_verification)
        self._hypothesis_max_age_steps = max(1, int(evidence_max_age_steps))
        # Verified hypotheses below this quality wait for the current epoch to end
        self.verified_preempt_min_quality = 0.0
        self._hypothesis_goal: np.ndarray | None = None
        self._hypothesis_completeness = 0.0
        self._hypothesis_step: int | None = None
        self._hypothesis_nav_goal: np.ndarray | None = None
        self._hypothesis_pending: np.ndarray | None = None
        self._hypothesis_pending_completeness = 0.0
        self._hypothesis_pending_step: int | None = None
        self._hypothesis_support = 0
        # Source of the live hypothesis ("geometry" or "verified")
        self._hypothesis_source: str | None = None
        self._hypothesis_arrival_point: np.ndarray | None = None
        self._verified_hypothesis_count = 0
        self._verified_hypothesis_examined_count = 0
        self._evidence_goal_epochs = 0
        self._current_goal_no_progress_steps = 0
        self._current_goal_best_distance = float("inf")
        self._stagnating: bool = False
        self._stagnation_steps: int = 0
        self._goal_end_sequence = 0
        self._goal_end_events: list[GoalEndEvent] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.visited = None
        self.topdown_map = None
        self.current_goal = None
        self._visited_origin_xz = None
        self._position_history = []
        self._failed_goals = []
        self._failed_goal_counts = []
        self._failed_goal_penalty_scales = []
        self._current_goal_no_progress_steps = 0
        self._current_goal_best_distance = float("inf")
        self._hypothesis_goal = None
        self._hypothesis_completeness = 0.0
        self._hypothesis_step = None
        self._hypothesis_nav_goal = None
        self._hypothesis_pending = None
        self._hypothesis_pending_completeness = 0.0
        self._hypothesis_pending_step = None
        self._hypothesis_support = 0
        self._hypothesis_source = None
        self._hypothesis_arrival_point = None
        self._verified_hypothesis_count = 0
        self._verified_hypothesis_examined_count = 0
        self._evidence_goal_epochs = 0
        self._stagnating = False
        self._stagnation_steps = 0
        self._goal_end_sequence = 0
        self._goal_end_events = []

    @property
    def stagnating(self) -> bool:
        return bool(self._stagnating)

    @property
    def stagnation_steps(self) -> int:
        return int(self._stagnation_steps)

    @property
    def failed_goal_count(self) -> int:
        return len(self._failed_goals)

    def clear_failed_goals(self) -> None:
        self._failed_goals = []
        self._failed_goal_counts = []
        self._failed_goal_penalty_scales = []

    def mark_goal_failed(
        self,
        goal: np.ndarray | None,
        *,
        reason: GoalEndReason = GoalEndReason.FAILED,
    ) -> GoalEndEvent | None:
        if goal is None:
            return None
        goal_arr = np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal_arr.size != 3 or not np.all(np.isfinite(goal_arr)):
            return None
        self._remember_failed_goal(goal_arr)
        if self._matches_current_goal(goal_arr):
            return self.end_current_goal(reason)
        return None

    def _remember_failed_goal(self, goal_arr: np.ndarray) -> None:
        failed_index = self._near_failed_goal_index(goal_arr)
        if failed_index is None:
            self._failed_goals.append(goal_arr.copy())
            self._failed_goal_counts.append(1)
            self._failed_goal_penalty_scales.append(1.0)
        else:
            while len(self._failed_goal_counts) <= failed_index:
                self._failed_goal_counts.append(1)
            while len(self._failed_goal_penalty_scales) <= failed_index:
                self._failed_goal_penalty_scales.append(1.0)
            self._failed_goal_counts[failed_index] += 1
            if self._failed_goal_counts[failed_index] >= self._failed_goal_repeat_penalty_count:
                self._failed_goal_penalty_scales[failed_index] = max(
                    float(self._failed_goal_penalty_scales[failed_index]),
                    self._failed_goal_repeat_penalty_scale,
                )

    def _sync_shape(self) -> tuple[int, int]:
        H, W = self.navigation_map.shape
        origin_xz = self._navigation_origin_xz()
        if self.visited is None:
            self.visited = np.zeros((H, W), dtype=np.uint8)
        elif self.visited.shape != (H, W):
            old = self.visited
            new = np.zeros((H, W), dtype=np.uint8)
            row_offset = 0
            col_offset = 0
            if self._visited_origin_xz is not None:
                resolution = max(1e-6, float(self.navigation_map.resolution_m))
                col_offset = int(round((float(self._visited_origin_xz[0]) - float(origin_xz[0])) / resolution))
                row_offset = int(round((float(self._visited_origin_xz[1]) - float(origin_xz[1])) / resolution))
            self._copy_with_offset(old, new, row_offset=row_offset, col_offset=col_offset)
            self.visited = new
        self._visited_origin_xz = origin_xz
        self.topdown_map = self.navigation_map.render()
        return H, W

    def _navigation_origin_xz(self) -> np.ndarray:
        origin = getattr(self.navigation_map, "origin_xz", None)
        if origin is None:
            return np.zeros(2, dtype=np.float32)
        value = origin() if callable(origin) else origin
        origin_xz = np.asarray(value, dtype=np.float32).reshape(-1)
        if origin_xz.size < 2 or not np.all(np.isfinite(origin_xz[:2])):
            return np.zeros(2, dtype=np.float32)
        return origin_xz[:2].copy()

    @staticmethod
    def _copy_with_offset(old: np.ndarray, new: np.ndarray, *, row_offset: int, col_offset: int) -> None:
        src_r0 = max(0, -row_offset)
        src_c0 = max(0, -col_offset)
        dst_r0 = max(0, row_offset)
        dst_c0 = max(0, col_offset)
        copy_h = min(old.shape[0] - src_r0, new.shape[0] - dst_r0)
        copy_w = min(old.shape[1] - src_c0, new.shape[1] - dst_c0)
        if copy_h <= 0 or copy_w <= 0:
            return
        new[dst_r0 : dst_r0 + copy_h, dst_c0 : dst_c0 + copy_w] = old[
            src_r0 : src_r0 + copy_h,
            src_c0 : src_c0 + copy_w,
        ]

    # ------------------------------------------------------------------
    # Stagnation / failed-goal helpers
    # ------------------------------------------------------------------

    def _is_stagnating(self) -> bool:
        if len(self._position_history) < self.stagnation_window:
            return False
        recent = self._position_history[-self.stagnation_window :]
        start = recent[0]
        max_displacement = max(float(np.linalg.norm(p[[0, 2]] - start[[0, 2]])) for p in recent)
        return max_displacement < self.stagnation_threshold_m

    def _is_near_failed_goal(self, position: np.ndarray) -> bool:
        return self._near_failed_goal_index(position) is not None

    def _near_failed_goal_index(self, position: np.ndarray) -> int | None:
        for index, failed_goal in enumerate(self._failed_goals):
            # 3-D distance, so a failed goal on another floor does not match
            distance = float(np.linalg.norm(position - failed_goal))
            if distance < self._failed_goal_radius_m:
                return int(index)
        return None

    def _is_repeated_failed_goal(self, position: np.ndarray) -> bool:
        for index, failed_goal in enumerate(self._failed_goals):
            distance = float(np.linalg.norm(position - failed_goal))
            if distance >= self._failed_goal_radius_m:
                continue
            count = (
                int(self._failed_goal_counts[index])
                if index < len(self._failed_goal_counts)
                else 1
            )
            if count >= self._failed_goal_repeat_penalty_count:
                return True
        return False

    def _mark_visited(self, position: np.ndarray) -> tuple[int, int]:
        H, W = self._sync_shape()
        assert self.visited is not None
        row, col = self.navigation_map.world_to_cell(position)
        radius = self.visited_radius_cells
        row_min = max(0, row - radius)
        row_max = min(H, row + radius + 1)
        col_min = max(0, col - radius)
        col_max = min(W, col + radius + 1)
        if row_min < row_max and col_min < col_max:
            self.visited[row_min:row_max, col_min:col_max] = 1
        return row, col

    def _goal_reached(self, position: np.ndarray, goal: np.ndarray) -> bool:
        return float(np.linalg.norm(position - goal)) <= self._goal_reached_radius_m

    def _matches_current_goal(self, goal: np.ndarray) -> bool:
        if self.current_goal is None:
            return False
        goal_arr = np.asarray(goal, dtype=np.float32).reshape(-1)
        return bool(
            goal_arr.size == 3
            and np.all(np.isfinite(goal_arr))
            and float(np.linalg.norm(goal_arr - self.current_goal))
            <= self._same_goal_distance_m
        )

    def _goal_end_context(self) -> tuple[str | None, int | None]:
        return None, None

    def _goal_end_acknowledged_actions(self) -> int:
        return 0

    def _on_goal_ended(self, event: GoalEndEvent) -> None:
        """Subclass hook invoked after a goal-end event has been queued."""

        _ = event

    def end_current_goal(
        self,
        reason: GoalEndReason,
    ) -> GoalEndEvent | None:
        """End the live goal and queue a GoalEndEvent."""

        if self.current_goal is None:
            return None
        parsed_reason = (
            reason
            if isinstance(reason, GoalEndReason)
            else GoalEndReason(reason)
        )
        goal = tuple(
            float(value)
            for value in np.asarray(self.current_goal, dtype=np.float32).reshape(3)
        )
        owner, token = self._goal_end_context()
        acknowledged_actions = self._goal_end_acknowledged_actions()
        if (
            self._hypothesis_nav_goal is not None
            and self._matches_current_goal(self._hypothesis_nav_goal)
            and parsed_reason is GoalEndReason.REACHED
        ):
            # Planner-reported arrival also counts as examining the hypothesis
            self._on_hypothesis_examined()
        self._goal_end_sequence += 1
        event = GoalEndEvent(
            sequence=int(self._goal_end_sequence),
            reason=parsed_reason,
            goal=goal,
            owner=None if owner is None else str(owner),
            token=None if token is None else int(token),
            acknowledged_actions=max(0, int(acknowledged_actions)),
        )
        self._clear_current_goal()
        self._goal_end_events.append(event)
        self._on_goal_ended(event)
        return event

    def drain_goal_end_events(self) -> tuple[GoalEndEvent, ...]:
        """Return queued goal-end events once, in sequence order."""

        events = tuple(self._goal_end_events)
        self._goal_end_events.clear()
        return events

    def _clear_current_goal(self) -> None:
        self.current_goal = None
        self._current_goal_no_progress_steps = 0
        self._current_goal_best_distance = float("inf")

    def _set_current_goal(self, goal: np.ndarray | None, position: np.ndarray) -> np.ndarray | None:
        if goal is None:
            self._clear_current_goal()
            return None
        goal_arr = np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal_arr.size != 3 or not np.all(np.isfinite(goal_arr)):
            self._clear_current_goal()
            return None

        same_goal = (
            self.current_goal is not None
            and float(np.linalg.norm(goal_arr - self.current_goal))
            <= self._same_goal_distance_m
        )
        self.current_goal = goal_arr.copy()
        if not same_goal or not math.isfinite(self._current_goal_best_distance):
            self._current_goal_no_progress_steps = 0
            self._current_goal_best_distance = float("inf")
        return self.current_goal

    @staticmethod
    def _distance_to_goal(position: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(position[[0, 2]] - goal[[0, 2]]))

    def _current_goal_has_stalled(self, position: np.ndarray) -> bool:
        if self.current_goal is None:
            return False
        distance = self._distance_to_goal(position, self.current_goal)
        if not math.isfinite(distance):
            return True
        if not math.isfinite(self._current_goal_best_distance):
            self._current_goal_best_distance = distance
            self._current_goal_no_progress_steps = 0
            return False
        if distance <= self._current_goal_best_distance - self._goal_progress_min_improvement_m:
            self._current_goal_best_distance = distance
            self._current_goal_no_progress_steps = 0
            return False
        self._current_goal_no_progress_steps += 1
        return self._current_goal_no_progress_steps >= self._goal_progress_window

    # ------------------------------------------------------------------
    # Frontier / fallback goal selection
    # ------------------------------------------------------------------

    def _select_frontier_goal(self, position: np.ndarray) -> np.ndarray | None:
        assert self.visited is not None
        frontier = self.navigation_map.frontier_cells()
        if not np.any(frontier):
            return None

        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier.astype(np.uint8), connectivity=8
        )
        candidates: list[tuple[np.ndarray, float, int, bool]] = []
        for label_index in range(1, num_labels):
            area = int(stats[label_index, cv2.CC_STAT_AREA])
            if area < self.min_frontier_size:
                continue

            centroid_col, centroid_row = centroids[label_index]
            candidate_world = self.navigation_map.cell_to_world(
                int(round(centroid_row)),
                int(round(centroid_col)),
                float(position[1]),
            )
            snapped = self.navigation_map.snap_to_free(candidate_world)
            if snapped is None or not np.all(np.isfinite(snapped)):
                continue

            planar_distance = float(np.linalg.norm(snapped[[0, 2]] - position[[0, 2]]))
            if planar_distance < self.min_goal_distance_m:
                continue

            distance = self.navigation_map.geodesic_distance(position, snapped)
            if not math.isfinite(distance):
                continue

            near_failed = self._is_near_failed_goal(snapped)
            candidates.append((snapped, distance, area, near_failed))

        if not candidates:
            return None

        pool = [c for c in candidates if not c[3]]
        if not pool:
            return None

        pool.sort(key=lambda c: c[1])
        return pool[0][0]

    def _select_unexplored_goal(self, position: np.ndarray, row: int, col: int) -> np.ndarray | None:
        assert self.visited is not None
        kernel = np.ones(
            (self.visited_radius_cells * 2 + 1, self.visited_radius_cells * 2 + 1),
            dtype=np.uint8,
        )
        visited_dilated = cv2.dilate(self.visited, kernel, iterations=1) > 0
        free_mask = self.navigation_map.free_mask()
        unexplored = free_mask & ~visited_dilated
        if not np.any(unexplored):
            unexplored = free_mask & ~(self.visited > 0)
        if not np.any(unexplored):
            self._clear_current_goal()
            return None

        candidates = np.argwhere(unexplored)
        if len(candidates) > 4000:
            stride = max(1, len(candidates) // 4000)
            candidates = candidates[::stride]

        deltas = candidates - np.array([row, col], dtype=np.int32)
        distances_sq = np.sum(deltas * deltas, axis=1)
        far_enough = distances_sq >= self.min_distance_cells * self.min_distance_cells
        if np.any(far_enough):
            candidates = candidates[far_enough]
            distances_sq = distances_sq[far_enough]

        order = np.argsort(distances_sq)
        for candidate_row, candidate_col in candidates[order]:
            world = self.navigation_map.cell_to_world(
                int(candidate_row), int(candidate_col), float(position[1])
            )
            snapped = self.navigation_map.snap_to_free(world)
            if snapped is None or not np.all(np.isfinite(snapped)):
                continue
            planar_distance = float(np.linalg.norm(snapped[[0, 2]] - position[[0, 2]]))
            if planar_distance < self.min_goal_distance_m:
                continue
            if self._is_near_failed_goal(np.asarray(snapped, dtype=np.float32)):
                continue
            geodesic_distance = self.navigation_map.geodesic_distance(position, snapped)
            if not math.isfinite(geodesic_distance):
                continue
            return np.asarray(snapped, dtype=np.float32)

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def note_instance_hypothesis(
        self,
        position: np.ndarray | None,
        *,
        completeness: float,
        step_index: int,
    ) -> bool:
        """Record an unverified instance hypothesis, activated after two looks."""

        if not self._evidence_verification:
            return False
        if position is None:
            return False
        candidate = np.asarray(position, dtype=np.float32).reshape(-1)
        if candidate.size != 3 or not np.all(np.isfinite(candidate)):
            return False
        quality = float(completeness)
        if not math.isfinite(quality) or quality <= 0.0:
            return False
        step = int(step_index)
        pending = self._hypothesis_pending
        recent = (
            pending is not None
            and self._hypothesis_pending_step is not None
            and step - int(self._hypothesis_pending_step)
            <= self._hypothesis_max_age_steps
        )
        same_surface = bool(
            recent
            and float(
                np.linalg.norm(candidate[[0, 2]] - pending[[0, 2]])
            )
            <= self._failed_goal_radius_m
        )
        if same_surface:
            self._hypothesis_support += 1
            if quality >= self._hypothesis_pending_completeness:
                self._hypothesis_pending = candidate.copy()
                self._hypothesis_pending_completeness = quality
        else:
            self._hypothesis_pending = candidate.copy()
            self._hypothesis_pending_completeness = quality
            self._hypothesis_support = 1
        self._hypothesis_pending_step = step
        if self._hypothesis_support < 2:
            return False
        return self._activate_instance_hypothesis(
            self._hypothesis_pending,
            float(self._hypothesis_pending_completeness),
            step,
            source="geometry",
        )

    def note_verified_instance_hypothesis(
        self,
        position: np.ndarray | None,
        *,
        quality: float,
        step_index: int,
        agent_position: np.ndarray | None = None,
    ) -> bool:
        """Install a verified instance hypothesis without the two-look requirement."""

        if not self._evidence_verification:
            return False
        if position is None:
            return False
        candidate = np.asarray(position, dtype=np.float32).reshape(-1)
        if candidate.size != 3 or not np.all(np.isfinite(candidate)):
            return False
        weight = float(quality)
        if not math.isfinite(weight) or weight <= 0.0:
            return False
        if agent_position is not None:
            planar = candidate.copy()
            planar[1] = float(np.asarray(agent_position, dtype=np.float32).reshape(3)[1])
            if self._is_near_failed_goal(planar):
                return False
        activated = self._activate_instance_hypothesis(
            candidate,
            weight,
            int(step_index),
            source="verified",
        )
        if activated:
            self._verified_hypothesis_count += 1
        return activated

    def pop_hypothesis_arrival(self) -> np.ndarray | None:
        """Return the object point of a verified hypothesis just examined, once."""

        point = self._hypothesis_arrival_point
        self._hypothesis_arrival_point = None
        return None if point is None else point.copy()

    @property
    def verified_hypothesis_count(self) -> int:
        return int(self._verified_hypothesis_count)

    @property
    def verified_hypothesis_examined_count(self) -> int:
        return int(self._verified_hypothesis_examined_count)

    def _activate_instance_hypothesis(
        self,
        candidate: np.ndarray,
        quality: float,
        step: int,
        *,
        source: str = "geometry",
    ) -> bool:
        if (
            self._hypothesis_is_live(step)
            and float(quality) < self._hypothesis_completeness
        ):
            return False
        moved = (
            self._hypothesis_goal is None
            or float(
                np.linalg.norm(candidate[[0, 2]] - self._hypothesis_goal[[0, 2]])
            )
            > self._failed_goal_radius_m
        )
        self._hypothesis_goal = (
            np.asarray(candidate, dtype=np.float32).reshape(3).copy()
        )
        self._hypothesis_completeness = float(quality)
        self._hypothesis_step = int(step)
        self._hypothesis_source = str(source)
        if moved:
            self._hypothesis_nav_goal = None
        return True

    def _hypothesis_is_live(self, step_index: int) -> bool:
        if self._hypothesis_goal is None or self._hypothesis_step is None:
            return False
        if (
            self._hypothesis_source == "verified"
            and self._hypothesis_nav_goal is not None
            and self._matches_current_goal(self._hypothesis_nav_goal)
        ):
            # Verified hypotheses do not expire while being pursued
            return True
        return int(step_index) - int(self._hypothesis_step) <= self._hypothesis_max_age_steps

    def _on_hypothesis_examined(self) -> None:
        if self._hypothesis_source == "verified" and self._hypothesis_goal is not None:
            nav_goal = self._hypothesis_nav_goal
            if nav_goal is not None:
                self._remember_failed_goal(
                    np.asarray(nav_goal, dtype=np.float32).reshape(3)
                )
            self._hypothesis_arrival_point = (
                np.asarray(self._hypothesis_goal, dtype=np.float32).reshape(3).copy()
            )
            self._verified_hypothesis_examined_count += 1
        self._clear_instance_hypothesis()

    def _clear_instance_hypothesis(self) -> None:
        self._hypothesis_goal = None
        self._hypothesis_completeness = 0.0
        self._hypothesis_step = None
        self._hypothesis_nav_goal = None
        self._hypothesis_pending = None
        self._hypothesis_pending_completeness = 0.0
        self._hypothesis_pending_step = None
        self._hypothesis_support = 0
        self._hypothesis_source = None

    @property
    def evidence_goal_epochs(self) -> int:
        """Exploration epochs that were owned by an instance hypothesis."""

        return int(self._evidence_goal_epochs)

    def get_goal(self, position: np.ndarray, step_index: int | None = None) -> np.ndarray | None:
        self._sync_shape()
        self._position_history.append(position.copy())
        row, col = self._mark_visited(position)

        was_stagnating = self._stagnating
        self._stagnating = self._is_stagnating()
        if self._stagnating:
            self._stagnation_steps += 1
        else:
            self._stagnation_steps = 0

        if (
            self._stagnating
            and not was_stagnating
            and self.current_goal is not None
        ):
            self.mark_goal_failed(
                self.current_goal,
                reason=GoalEndReason.POSITION_STAGNATION,
            )

        hypothesis = self._live_hypothesis_goal(position, step_index)
        defer_hypothesis = bool(
            hypothesis is not None
            and self.current_goal is not None
            and self._hypothesis_source == "verified"
            and float(self._hypothesis_completeness)
            < float(self.verified_preempt_min_quality)
        )
        if (
            hypothesis is not None
            and not defer_hypothesis
            and not self._matches_current_goal(hypothesis)
        ):
            if self.current_goal is not None:
                self.end_current_goal(GoalEndReason.EVIDENCE_PREEMPTED)
            self._evidence_goal_epochs += 1
            return self._set_current_goal(hypothesis, position)

        if self.current_goal is not None:
            hypothesis_is_current = bool(
                hypothesis is not None and self._matches_current_goal(hypothesis)
            )
            if self._goal_reached(position, self.current_goal):
                if hypothesis_is_current:
                    self._on_hypothesis_examined()
                self.end_current_goal(GoalEndReason.REACHED)
            elif self._current_goal_has_stalled(position):
                if hypothesis_is_current:
                    self._clear_instance_hypothesis()
                self.mark_goal_failed(
                    self.current_goal,
                    reason=GoalEndReason.PROGRESS_STALL,
                )
            else:
                return self.current_goal
            if defer_hypothesis:
                # The epoch the deferred hypothesis waited for has ended
                deferred = self._live_hypothesis_goal(position, step_index)
                if deferred is not None:
                    self._evidence_goal_epochs += 1
                    return self._set_current_goal(deferred, position)

        frontier_goal = self._select_frontier_goal(position)
        if frontier_goal is not None:
            return self._set_current_goal(frontier_goal, position)

        fallback_goal = self._select_unexplored_goal(position, row, col)
        return self._set_current_goal(fallback_goal, position)

    def _live_hypothesis_goal(
        self,
        position: np.ndarray,
        step_index: int | None,
    ) -> np.ndarray | None:
        if self._hypothesis_goal is None:
            return None
        if step_index is not None and not self._hypothesis_is_live(int(step_index)):
            self._clear_instance_hypothesis()
            return None
        cached = self._hypothesis_nav_goal
        if cached is not None:
            resolved = np.asarray(cached, dtype=np.float32).reshape(3).copy()
            if self._goal_reached(position, resolved):
                self._on_hypothesis_examined()
                return None
            if self._is_near_failed_goal(resolved):
                self._clear_instance_hypothesis()
                return None
            return resolved
        goal = np.asarray(self._hypothesis_goal, dtype=np.float32).reshape(3).copy()
        goal[1] = float(np.asarray(position, dtype=np.float32).reshape(3)[1])
        if self._goal_reached(position, goal) or self._is_near_failed_goal(goal):
            self._clear_instance_hypothesis()
            return None
        snapped = self.navigation_map.snap_to_free(goal)
        if snapped is None or not np.all(np.isfinite(snapped)):
            return None
        snapped = np.asarray(snapped, dtype=np.float32).reshape(3).copy()
        snapped[1] = goal[1]
        if self._goal_reached(position, snapped) or self._is_near_failed_goal(snapped):
            self._clear_instance_hypothesis()
            return None
        # Cache the snapped goal so map growth does not reset the epoch
        self._hypothesis_nav_goal = snapped.copy()
        return snapped
