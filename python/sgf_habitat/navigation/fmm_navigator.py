from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass

import numpy as np
import quaternion
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from ..planning.cost_map import CostMapBuilder, CostMapConfig, CostMapResult
from ..planning.escape_machine import EscapeMachine, EscapeMachineConfig
from ..planning.fmm_cost_planner import FMMCostPlanner


def _planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[[0, 2]] - b[[0, 2]]))


def _rotation_to_forward_xz(rotation) -> np.ndarray:
    rotation_matrix = quaternion.as_rotation_matrix(rotation).astype(np.float32)
    forward = rotation_matrix @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    forward_xz = np.asarray([forward[0], forward[2]], dtype=np.float32)
    norm = float(np.linalg.norm(forward_xz))
    if norm < 1e-6:
        return np.array([0.0, -1.0], dtype=np.float32)
    return forward_xz / norm


def compute_heading_error_deg(agent_rotation, waypoint: np.ndarray, agent_position: np.ndarray) -> float:
    forward_xz = _rotation_to_forward_xz(agent_rotation)
    target_vector_xz = np.asarray(
        [float(waypoint[0] - agent_position[0]), float(waypoint[2] - agent_position[2])],
        dtype=np.float32,
    )
    target_norm = float(np.linalg.norm(target_vector_xz))
    if target_norm < 1e-6:
        return 0.0
    target_xz = target_vector_xz / target_norm
    cross = float(forward_xz[0] * target_xz[1] - forward_xz[1] * target_xz[0])
    dot = float(np.clip(np.dot(forward_xz, target_xz), -1.0, 1.0))
    return math.degrees(math.atan2(cross, dot))


@dataclass
class NavigationCommand:
    action: int
    heading_error_deg: float | None
    goal_distance_m: float | None
    stalled_count: int
    replan_failed_count: int
    progress_stall_count: int
    planner_replan_failed: bool
    short_term_goal: np.ndarray | None = None
    goal_reached: bool = False
    goal_blocked: bool = False
    local_blocked: bool = False
    block_scope: str = "none"
    recovery_active: bool = False
    recovery_reason: str | None = None


def _readonly_array(value, *, dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class NavigationPlanningSnapshot:
    """Detached, immutable view of the navigator's current planning field."""

    resolution_m: float
    origin_xz: np.ndarray
    agent_position: np.ndarray
    agent_cell: tuple[int, int]
    known: np.ndarray
    occupied: np.ndarray
    traversible: np.ndarray
    cost: np.ndarray
    clearance_m: np.ndarray
    travel_cost: np.ndarray

    def __post_init__(self) -> None:
        resolution = float(self.resolution_m)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("resolution_m must be finite and positive")
        origin = np.asarray(self.origin_xz, dtype=np.float32).reshape(-1)
        if origin.size != 2 or not np.all(np.isfinite(origin)):
            raise ValueError("origin_xz must contain finite x/z coordinates")
        agent_position = np.asarray(self.agent_position, dtype=np.float32).reshape(-1)
        if agent_position.size != 3 or not np.all(np.isfinite(agent_position)):
            raise ValueError("agent_position must be a finite xyz coordinate")

        known = _readonly_array(self.known, dtype=bool)
        if known.ndim != 2:
            raise ValueError("planning snapshot masks must be 2D")
        arrays = {
            "occupied": _readonly_array(self.occupied, dtype=bool),
            "traversible": _readonly_array(self.traversible, dtype=bool),
            "cost": _readonly_array(self.cost, dtype=np.float32),
            "clearance_m": _readonly_array(self.clearance_m, dtype=np.float32),
            "travel_cost": _readonly_array(self.travel_cost, dtype=np.float32),
        }
        for name, array in arrays.items():
            if array.shape != known.shape:
                raise ValueError(f"{name} shape must match known shape")
        row, col = int(self.agent_cell[0]), int(self.agent_cell[1])
        if not (0 <= row < known.shape[0] and 0 <= col < known.shape[1]):
            raise ValueError("agent_cell must be inside the planning snapshot")

        origin = origin.copy()
        origin.setflags(write=False)
        agent_position = agent_position.copy()
        agent_position.setflags(write=False)
        object.__setattr__(self, "resolution_m", resolution)
        object.__setattr__(self, "origin_xz", origin)
        object.__setattr__(self, "agent_position", agent_position)
        object.__setattr__(self, "agent_cell", (row, col))
        object.__setattr__(self, "known", known)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.known.shape[0]), int(self.known.shape[1])

    def world_to_cell(self, xyz: np.ndarray) -> tuple[int, int]:
        point = np.asarray(xyz, dtype=np.float32).reshape(-1)
        if point.size != 3 or not np.all(np.isfinite(point)):
            raise ValueError("xyz must be a finite coordinate")
        row = math.floor((float(point[2]) - float(self.origin_xz[1])) / self.resolution_m)
        col = math.floor((float(point[0]) - float(self.origin_xz[0])) / self.resolution_m)
        return int(row), int(col)

    def cell_to_world(self, row: int, col: int, y_m: float | None = None) -> np.ndarray:
        height = float(self.agent_position[1]) if y_m is None else float(y_m)
        if not math.isfinite(height):
            raise ValueError("y_m must be finite")
        return np.asarray(
            [
                float(self.origin_xz[0]) + (int(col) + 0.5) * self.resolution_m,
                height,
                float(self.origin_xz[1]) + (int(row) + 0.5) * self.resolution_m,
            ],
            dtype=np.float32,
        )


@dataclass
class _ProposalState:
    escape_machine: EscapeMachine
    last_goal_position: np.ndarray | None
    local_recovery_queue: list[tuple[int, str, bool]]
    local_recovery_failures: int


@dataclass
class _PendingAction:
    proposal_id: int
    command: NavigationCommand
    agent_position: np.ndarray
    agent_rotation: object
    local_recovery_probe: bool
    prior_state: _ProposalState


class FMMGoalNavigator:
    """Cost-map navigator whose proposed actions are committed or cancelled."""

    _REPLAN_FAILURE_LIMIT = 8
    _STALL_STEP_LIMIT = 3

    def __init__(
        self,
        *,
        forward_step_size_m: float,
        turn_angle_deg: float,
        agent_radius_m: float,
        planner_cls=None,
        clearance_margin_m: float = 0.05,
    ) -> None:
        self.forward_step_size_m = float(forward_step_size_m)
        self.turn_angle_deg = float(turn_angle_deg)
        self.agent_radius_m = float(agent_radius_m)
        self.clearance_margin_m = float(clearance_margin_m)
        self._replan_failure_limit = self._REPLAN_FAILURE_LIMIT
        self._stall_step_limit = self._STALL_STEP_LIMIT
        self._planner_cls = planner_cls if planner_cls is not None else FMMCostPlanner
        self._escape_machine = EscapeMachine(
            EscapeMachineConfig(
                forward_action=int(HabitatSimActions.move_forward),
                turn_left_action=int(HabitatSimActions.turn_left),
                turn_right_action=int(HabitatSimActions.turn_right),
                turn_angle_deg=self.turn_angle_deg,
                stall_steps=self._stall_step_limit,
                replan_failures=self._replan_failure_limit,
                blocked_heading_deg=22.5,
            )
        )
        self.last_cost_map_diagnostics: dict[str, object] = {}
        self.reset()

    def reset(self) -> None:
        self._collision_penalty_mask: np.ndarray | None = None
        self._collision_origin_xz: np.ndarray | None = None
        self._motion_block_mask: np.ndarray | None = None
        self._motion_block_origin_xz: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_position: np.ndarray | None = None
        self._last_rotation = None
        self._last_action_was_local_probe = False
        self._last_action_result_recorded = True
        self._committed_action_generation = 0
        self._last_goal_position: np.ndarray | None = None
        self._local_recovery_queue: list[tuple[int, str, bool]] = []
        self._local_recovery_failures = 0
        self._local_recovery_failure_limit = 4
        self._proposal_sequence = 0
        self._active_proposal_state: _ProposalState | None = None
        self._pending_action: _PendingAction | None = None
        self._last_blocked_forward_event: tuple[object, ...] | None = None
        self._last_blocked_forward_fingerprint: tuple[float, ...] | None = None
        self.last_cost_map_diagnostics = {}
        self._escape_machine.reset()

    @property
    def has_pending_action(self) -> bool:
        return self._pending_action is not None

    @property
    def pending_action(self) -> int | None:
        if self._pending_action is None:
            return None
        return int(self._pending_action.command.action)

    def commit_pending_action(
        self,
        *,
        result_agent_state=None,
        navigation_map=None,
    ) -> bool:
        """Commit the pending proposal after its action was executed."""

        pending = self._pending_action
        if pending is None:
            return False
        if (
            result_agent_state is not None
            and int(pending.command.action)
            == int(HabitatSimActions.move_forward)
            and navigation_map is None
        ):
            raise ValueError(
                "navigation_map is required for immediate forward feedback"
            )
        self._last_action = int(pending.command.action)
        self._last_position = pending.agent_position.copy()
        self._last_rotation = pending.agent_rotation
        self._last_action_was_local_probe = bool(
            pending.local_recovery_probe
            and int(pending.command.action) == int(HabitatSimActions.move_forward)
        )
        self._last_action_result_recorded = False
        self._committed_action_generation += 1
        self._pending_action = None
        self._active_proposal_state = None
        if result_agent_state is not None:
            self._update_collision_penalty(
                np.asarray(result_agent_state.position, dtype=np.float32),
                navigation_map,
            )
        return True

    def cancel_pending_action(self) -> bool:
        """Cancel the pending proposal and restore the planner state."""

        pending = self._pending_action
        if pending is None:
            return False
        self._restore_proposal_state(pending.prior_state)
        self._pending_action = None
        self._active_proposal_state = None
        return True

    def planning_snapshot(
        self,
        *,
        agent_state,
        navigation_map,
    ) -> NavigationPlanningSnapshot:
        """Goal-free planning field and travel cost, without side effects."""

        agent_position = np.asarray(agent_state.position, dtype=np.float32).reshape(-1)
        if agent_position.size != 3 or not np.all(np.isfinite(agent_position)):
            raise ValueError("agent position must be a finite xyz coordinate")
        free_mask = np.asarray(navigation_map.free_mask(), dtype=bool)
        occupied_mask = np.asarray(navigation_map.occupied_mask(), dtype=bool)
        known_mask = np.asarray(navigation_map.observed_mask(), dtype=bool)
        if free_mask.ndim != 2:
            raise ValueError("navigation map masks must be 2D")
        if occupied_mask.shape != free_mask.shape or known_mask.shape != free_mask.shape:
            raise ValueError("navigation map masks must share one shape")

        rows, cols = free_mask.shape
        agent_row, agent_col = self._clamped_cell(
            navigation_map.world_to_cell(agent_position),
            rows,
            cols,
        )
        saved_collision_mask = self._collision_penalty_mask
        saved_collision_origin = self._collision_origin_xz
        saved_motion_mask = self._motion_block_mask
        saved_motion_origin = self._motion_block_origin_xz
        try:
            self._sync_collision_penalty(free_mask.shape, navigation_map)
            cost_map = self._build_cost_map(
                navigation_map=navigation_map,
                free_mask=free_mask,
                occupied_mask=occupied_mask,
                agent_cell=(agent_row, agent_col),
                goal_cell=None,
            )
            traversible = self._planner_traversible(
                cost_map,
                navigation_map=navigation_map,
                agent_cell=(agent_row, agent_col),
                apply_motion_blocks=True,
            )
            planner_cost = np.asarray(cost_map.cost, dtype=np.float32).copy()
            planner_cost[~traversible] = np.inf
            self._set_patch_value(
                planner_cost,
                agent_row,
                agent_col,
                radius_cells=1,
                value=1.0,
            )

            source_map = np.zeros_like(traversible, dtype=np.uint8)
            source_map[agent_row, agent_col] = 1
            planner = self._make_planner(
                traversible=traversible,
                cost=planner_cost,
                resolution_m=float(navigation_map.resolution_m),
            )
            self._set_multi_goal(planner, source_map, (agent_row, agent_col))
            travel_cost = np.asarray(planner.fmm_dist, dtype=np.float32).copy()
            if travel_cost.shape != traversible.shape:
                raise ValueError("planner travel-cost shape must match navigation map")
            travel_cost[~traversible] = np.inf

            return NavigationPlanningSnapshot(
                resolution_m=float(navigation_map.resolution_m),
                origin_xz=self._navigation_origin_xz(navigation_map),
                agent_position=agent_position,
                agent_cell=(agent_row, agent_col),
                known=known_mask,
                occupied=occupied_mask,
                traversible=traversible,
                cost=planner_cost,
                clearance_m=cost_map.clearance_m,
                travel_cost=travel_cost,
            )
        finally:
            self._collision_penalty_mask = saved_collision_mask
            self._collision_origin_xz = saved_collision_origin
            self._motion_block_mask = saved_motion_mask
            self._motion_block_origin_xz = saved_motion_origin

    def propose_face_position(
        self,
        *,
        agent_state,
        target_position: np.ndarray,
    ) -> NavigationCommand | None:
        """Stage the shortest turn toward a point, or return None if aligned."""

        if self._active_proposal_state is not None:
            self._restore_proposal_state(self._active_proposal_state)
            self._active_proposal_state = None
        if self._pending_action is not None:
            self.commit_pending_action()

        agent_position = np.asarray(agent_state.position, dtype=np.float32)
        target = np.asarray(target_position, dtype=np.float32)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_position must be a finite xyz coordinate")
        heading_error_deg = compute_heading_error_deg(
            agent_state.rotation,
            target,
            agent_position,
        )
        if abs(heading_error_deg) <= self.turn_angle_deg / 2.0:
            return None

        self._begin_action_proposal()
        action = (
            int(HabitatSimActions.turn_left)
            if heading_error_deg < 0.0
            else int(HabitatSimActions.turn_right)
        )
        return self._finalize_command(
            agent_state=agent_state,
            action=action,
            heading_error_deg=heading_error_deg,
            goal_distance_m=_planar_distance(agent_position, target),
            planner_replan_failed=False,
            short_term_goal=target,
            suppress_stall_block=True,
        )

    def next_action(
        self,
        *,
        agent_state,
        goal_position: np.ndarray,
        navigation_map,
        allow_stop: bool,
        stop_radius_m: float,
    ) -> NavigationCommand:
        # Undo a proposal interrupted by an exception
        if self._active_proposal_state is not None:
            self._restore_proposal_state(self._active_proposal_state)
            self._active_proposal_state = None
        # An unresolved previous command is treated as executed
        if self._pending_action is not None:
            self.commit_pending_action()

        agent_position = np.asarray(agent_state.position, dtype=np.float32)
        goal_world = np.asarray(goal_position, dtype=np.float32)
        if goal_world.shape != (3,) or not np.all(np.isfinite(goal_world)):
            raise ValueError("goal_position must be a finite xyz world coordinate")

        free_mask = np.asarray(navigation_map.free_mask(), dtype=bool)
        occupied_mask = np.asarray(navigation_map.occupied_mask(), dtype=bool)
        self._sync_collision_penalty(free_mask.shape, navigation_map)
        self._update_collision_penalty(agent_position, navigation_map)
        self._begin_action_proposal()

        goal_distance_m = _planar_distance(agent_position, goal_world)
        self._sync_goal_tracking(goal_world)
        if allow_stop and goal_distance_m <= float(stop_radius_m):
            return self._finalize_command(
                agent_state=agent_state,
                action=int(HabitatSimActions.stop),
                heading_error_deg=0.0,
                goal_distance_m=goal_distance_m,
                planner_replan_failed=False,
                goal_reached=True,
            )
        if not allow_stop and goal_distance_m <= float(stop_radius_m):
            return self._finalize_command(
                agent_state=agent_state,
                action=int(HabitatSimActions.turn_left),
                heading_error_deg=0.0,
                goal_distance_m=goal_distance_m,
                planner_replan_failed=False,
                goal_reached=True,
            )

        local_recovery_command = self._next_local_recovery_command()
        if local_recovery_command is not None:
            recovery_action, recovery_reason, recovery_probe = local_recovery_command
            return self._finalize_command(
                agent_state=agent_state,
                action=recovery_action,
                heading_error_deg=None,
                goal_distance_m=goal_distance_m,
                planner_replan_failed=False,
                local_blocked=True,
                block_scope="local",
                recovery_reason=recovery_reason,
                local_recovery_probe=recovery_probe,
                suppress_stall_block=True,
            )

        escape_command = self._next_escape_command()
        if escape_command is not None:
            escape_action, escape_reason = escape_command
            return self._finalize_command(
                agent_state=agent_state,
                action=escape_action,
                heading_error_deg=None,
                goal_distance_m=goal_distance_m,
                planner_replan_failed=False,
                local_blocked=True,
                block_scope="local",
                recovery_reason=escape_reason,
                local_recovery_probe=escape_action == int(HabitatSimActions.move_forward),
                suppress_stall_block=True,
            )

        rows, cols = free_mask.shape
        agent_row, agent_col = self._clamped_cell(navigation_map.world_to_cell(agent_position), rows, cols)
        cost_map = self._build_cost_map(
            navigation_map=navigation_map,
            free_mask=free_mask,
            occupied_mask=occupied_mask,
            agent_cell=(agent_row, agent_col),
            goal_cell=None,
        )
        traversible = self._planner_traversible(
            cost_map,
            navigation_map=navigation_map,
            agent_cell=(agent_row, agent_col),
            apply_motion_blocks=True,
        )

        goal_row, goal_col, goal_world = self._resolve_goal_cell(
            goal_world=goal_world,
            agent_position=agent_position,
            navigation_map=navigation_map,
            traversible=traversible,
        )
        if goal_row is None or goal_col is None:
            traversible = self._planner_traversible(
                cost_map,
                navigation_map=navigation_map,
                agent_cell=(agent_row, agent_col),
                apply_motion_blocks=False,
            )
            goal_row, goal_col, goal_world = self._resolve_goal_cell(
                goal_world=goal_world,
                agent_position=agent_position,
                navigation_map=navigation_map,
                traversible=traversible,
            )
        if goal_row is None or goal_col is None:
            return self._finalize_command(
                agent_state=agent_state,
                action=self._recovery_turn_action(),
                heading_error_deg=None,
                goal_distance_m=goal_distance_m,
                planner_replan_failed=True,
                goal_blocked=True,
                block_scope="goal",
                recovery_reason="goal_not_traversible",
            )
        cost_map = self._build_cost_map(
            navigation_map=navigation_map,
            free_mask=free_mask,
            occupied_mask=occupied_mask,
            agent_cell=(agent_row, agent_col),
            goal_cell=(goal_row, goal_col),
        )
        self.last_cost_map_diagnostics = dict(cost_map.diagnostics)
        traversible = self._planner_traversible(
            cost_map,
            navigation_map=navigation_map,
            agent_cell=(agent_row, agent_col),
            goal_cell=(goal_row, goal_col),
            apply_motion_blocks=True,
        )
        cost = cost_map.cost.copy()
        self.last_cost_map_diagnostics["motion_block_cells"] = int(
            0 if self._motion_block_mask is None else np.count_nonzero(self._motion_block_mask)
        )
        self.last_cost_map_diagnostics["motion_block_hard_applied"] = bool(self._has_motion_blocks(traversible.shape))
        self._set_patch_value(cost, agent_row, agent_col, radius_cells=1, value=1.0)
        traversible[goal_row, goal_col] = True
        cost[goal_row, goal_col] = 1.0

        goal_map = np.zeros_like(traversible, dtype=np.uint8)
        goal_map[goal_row, goal_col] = 1

        planner = self._make_planner(
            traversible=traversible,
            cost=cost,
            resolution_m=float(navigation_map.resolution_m),
        )
        self._set_multi_goal(planner, goal_map.astype(np.uint8), (agent_row, agent_col))
        fmm_dist = np.asarray(planner.fmm_dist, dtype=np.float32)
        current_distance = float(fmm_dist[agent_row, agent_col]) if fmm_dist.size else float("inf")
        if not math.isfinite(current_distance) and self._has_motion_blocks(traversible.shape):
            traversible = self._planner_traversible(
                cost_map,
                navigation_map=navigation_map,
                agent_cell=(agent_row, agent_col),
                goal_cell=(goal_row, goal_col),
                apply_motion_blocks=False,
            )
            traversible[goal_row, goal_col] = True
            planner = self._make_planner(
                traversible=traversible,
                cost=cost,
                resolution_m=float(navigation_map.resolution_m),
            )
            self._set_multi_goal(planner, goal_map.astype(np.uint8), (agent_row, agent_col))
            fmm_dist = np.asarray(planner.fmm_dist, dtype=np.float32)
            current_distance = float(fmm_dist[agent_row, agent_col]) if fmm_dist.size else float("inf")
            self.last_cost_map_diagnostics["motion_block_hard_fallback"] = True
        step_cells = max(1, int(round(self.forward_step_size_m / max(1e-6, float(navigation_map.resolution_m)))))
        if not math.isfinite(current_distance):
            # Unreachable goal: head for the nearest reachable cell instead
            reachable_goal = self._closest_reachable_cell(
                traversible=traversible,
                cost=cost,
                resolution_m=float(navigation_map.resolution_m),
                agent_cell=(agent_row, agent_col),
                goal_cell=(goal_row, goal_col),
                min_separation_cells=step_cells,
            )
            if reachable_goal is not None:
                goal_row, goal_col = reachable_goal
                goal_map = np.zeros_like(traversible, dtype=np.uint8)
                goal_map[goal_row, goal_col] = 1
                planner = self._make_planner(
                    traversible=traversible,
                    cost=cost,
                    resolution_m=float(navigation_map.resolution_m),
                )
                self._set_multi_goal(planner, goal_map, (agent_row, agent_col))
                fmm_dist = np.asarray(planner.fmm_dist, dtype=np.float32)
                current_distance = float(fmm_dist[agent_row, agent_col]) if fmm_dist.size else float("inf")
                self.last_cost_map_diagnostics["closest_reachable_goal"] = True
        if not math.isfinite(current_distance):
            return self._finalize_command(
                agent_state=agent_state,
                action=self._recovery_turn_action(),
                heading_error_deg=None,
                goal_distance_m=goal_distance_m,
                planner_replan_failed=True,
                goal_blocked=True,
                block_scope="goal",
                recovery_reason="no_fmm_path",
            )

        candidate = self._select_short_term_goal(
            agent_position=agent_position,
            agent_rotation=agent_state.rotation,
            agent_cell=(agent_row, agent_col),
            fmm_dist=fmm_dist,
            traversible=traversible,
            navigation_map=navigation_map,
            step_cells=step_cells,
        )
        if candidate is None and self._has_motion_blocks(traversible.shape):
            traversible = self._planner_traversible(
                cost_map,
                navigation_map=navigation_map,
                agent_cell=(agent_row, agent_col),
                goal_cell=(goal_row, goal_col),
                apply_motion_blocks=False,
            )
            traversible[goal_row, goal_col] = True
            planner = self._make_planner(
                traversible=traversible,
                cost=cost,
                resolution_m=float(navigation_map.resolution_m),
            )
            self._set_multi_goal(planner, goal_map.astype(np.uint8), (agent_row, agent_col))
            fmm_dist = np.asarray(planner.fmm_dist, dtype=np.float32)
            candidate = self._select_short_term_goal(
                agent_position=agent_position,
                agent_rotation=agent_state.rotation,
                agent_cell=(agent_row, agent_col),
                fmm_dist=fmm_dist,
                traversible=traversible,
                navigation_map=navigation_map,
                step_cells=step_cells,
            )
            self.last_cost_map_diagnostics["motion_block_stg_fallback"] = True
        if candidate is None:
            self._local_recovery_failures += 1
            return self._finalize_command(
                agent_state=agent_state,
                action=self._recovery_turn_action(),
                heading_error_deg=None,
                goal_distance_m=goal_distance_m,
                planner_replan_failed=False,
                local_blocked=True,
                block_scope="local",
                recovery_reason="no_descent_waypoint",
                suppress_stall_block=True,
            )

        heading_error_deg, waypoint = candidate
        if abs(heading_error_deg) > self.turn_angle_deg / 2.0:
            action = int(HabitatSimActions.turn_left) if heading_error_deg < 0.0 else int(HabitatSimActions.turn_right)
        else:
            action = int(HabitatSimActions.move_forward)

        if (
            action == int(HabitatSimActions.move_forward)
            and self._escape_machine.forward_stall_count > 0
            and self._current_heading_is_blocked(agent_state.rotation)
        ):
            local_recovery_command = self._start_local_bypass_recovery(
                agent_position=agent_position,
                agent_rotation=agent_state.rotation,
                agent_cell=(agent_row, agent_col),
                fmm_dist=fmm_dist,
                traversible=traversible,
                navigation_map=navigation_map,
            )
            if local_recovery_command is not None:
                recovery_action, recovery_reason, recovery_probe = local_recovery_command
                return self._finalize_command(
                    agent_state=agent_state,
                    action=recovery_action,
                    heading_error_deg=None,
                    goal_distance_m=goal_distance_m,
                    planner_replan_failed=False,
                    local_blocked=True,
                    block_scope="local",
                    recovery_reason=recovery_reason,
                    local_recovery_probe=recovery_probe,
                    suppress_stall_block=True,
                )

            self._local_recovery_failures += 1
            return self._finalize_command(
                agent_state=agent_state,
                action=self._recovery_turn_action(),
                heading_error_deg=None,
                goal_distance_m=goal_distance_m,
                planner_replan_failed=False,
                local_blocked=True,
                block_scope="local",
                recovery_reason="local_bypass_unavailable",
                suppress_stall_block=True,
            )

        return self._finalize_command(
            agent_state=agent_state,
            action=action,
            heading_error_deg=heading_error_deg,
            goal_distance_m=goal_distance_m,
            planner_replan_failed=False,
            short_term_goal=waypoint,
        )

    def _build_traversible(
        self,
        agent_position: np.ndarray,
        navigation_map,
        free_mask: np.ndarray,
        occupied_mask: np.ndarray,
    ) -> np.ndarray:
        rows, cols = free_mask.shape
        agent_row, agent_col = self._clamped_cell(navigation_map.world_to_cell(agent_position), rows, cols)
        cost_map = self._build_cost_map(
            navigation_map=navigation_map,
            free_mask=free_mask,
            occupied_mask=occupied_mask,
            agent_cell=(agent_row, agent_col),
            goal_cell=None,
        )
        traversible = self._planner_traversible(
            cost_map,
            navigation_map=navigation_map,
            agent_cell=(agent_row, agent_col),
            apply_motion_blocks=True,
        )
        return traversible

    def _build_cost_map(
        self,
        *,
        navigation_map,
        free_mask: np.ndarray,
        occupied_mask: np.ndarray,
        agent_cell: tuple[int, int] | None,
        goal_cell: tuple[int, int] | None,
    ) -> CostMapResult:
        builder = CostMapBuilder(
            CostMapConfig(
                agent_radius_m=self.agent_radius_m,
                clearance_margin_m=self.clearance_margin_m,
            )
        )
        return builder.build(
            free_mask=free_mask,
            occupied_mask=occupied_mask,
            resolution_m=float(navigation_map.resolution_m),
            collision_mask=self._collision_penalty_mask,
            agent_cell=agent_cell,
            goal_cell=goal_cell,
            goal_clearance_margin_m=None,
        )

    def _planner_traversible(
        self,
        cost_map: CostMapResult,
        *,
        navigation_map,
        agent_cell: tuple[int, int] | None,
        goal_cell: tuple[int, int] | None = None,
        apply_motion_blocks: bool = True,
    ) -> np.ndarray:
        traversible = np.asarray(cost_map.traversible, dtype=bool).copy()
        self._apply_motion_blocks(
            traversible,
            agent_cell=agent_cell,
            goal_cell=goal_cell,
            apply_motion_blocks=apply_motion_blocks,
        )
        return traversible

    def _make_planner(self, *, traversible: np.ndarray, cost: np.ndarray, resolution_m: float):
        try:
            parameters = inspect.signature(self._planner_cls).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "cost" in parameters:
            return self._planner_cls(
                traversible.astype(np.uint8),
                cost=cost.astype(np.float32),
                dx=max(1e-6, float(resolution_m)),
            )
        return self._planner_cls(traversible.astype(np.uint8))

    def _closest_reachable_cell(
        self,
        *,
        traversible: np.ndarray,
        cost: np.ndarray,
        resolution_m: float,
        agent_cell: tuple[int, int],
        goal_cell: tuple[int, int],
        min_separation_cells: int,
    ) -> tuple[int, int] | None:
        source = np.zeros_like(traversible, dtype=np.uint8)
        source[int(agent_cell[0]), int(agent_cell[1])] = 1
        planner = self._make_planner(
            traversible=traversible,
            cost=cost,
            resolution_m=resolution_m,
        )
        self._set_multi_goal(planner, source, agent_cell)
        travel = np.asarray(planner.fmm_dist, dtype=np.float32)
        if travel.shape != traversible.shape:
            return None
        rows, cols = np.nonzero(np.isfinite(travel) & traversible)
        if rows.size == 0:
            return None
        gap = np.hypot(rows - int(goal_cell[0]), cols - int(goal_cell[1]))
        # Sort by distance to goal, then travel cost
        best = int(np.lexsort((travel[rows, cols], gap))[0])
        row, col = int(rows[best]), int(cols[best])
        if math.hypot(row - int(agent_cell[0]), col - int(agent_cell[1])) < float(min_separation_cells):
            return None
        return row, col

    def _resolve_goal_cell(
        self,
        *,
        goal_world: np.ndarray,
        agent_position: np.ndarray,
        navigation_map,
        traversible: np.ndarray,
    ) -> tuple[int | None, int | None, np.ndarray]:
        rows, cols = traversible.shape
        row, col = navigation_map.world_to_cell(goal_world)
        if 0 <= row < rows and 0 <= col < cols and traversible[row, col]:
            return row, col, goal_world

        snapped = navigation_map.snap_to_free(goal_world)
        if snapped is None:
            return None, None, goal_world
        snapped_goal = np.asarray(snapped, dtype=np.float32)
        snapped_goal[1] = agent_position[1]
        row, col = navigation_map.world_to_cell(snapped_goal)
        if not (0 <= row < rows and 0 <= col < cols):
            return None, None, snapped_goal
        return row, col, snapped_goal

    def _select_short_term_goal(
        self,
        *,
        agent_position: np.ndarray,
        agent_rotation,
        agent_cell: tuple[int, int],
        fmm_dist: np.ndarray,
        traversible: np.ndarray,
        navigation_map,
        step_cells: int,
    ) -> tuple[float, np.ndarray] | None:
        cpp_result = navigation_map.select_short_term_goal(
            agent_position=agent_position,
            forward_xz=_rotation_to_forward_xz(agent_rotation),
            agent_cell=agent_cell,
            fmm_dist=fmm_dist,
            traversible=traversible,
            step_cells=step_cells,
        )
        if cpp_result is None:
            return None
        return float(cpp_result["heading_error_deg"]), np.asarray(cpp_result["waypoint"], dtype=np.float32)

    def _apply_motion_blocks(
        self,
        traversible: np.ndarray,
        *,
        agent_cell: tuple[int, int] | None = None,
        goal_cell: tuple[int, int] | None = None,
        apply_motion_blocks: bool = True,
    ) -> None:
        if apply_motion_blocks and self._has_motion_blocks(traversible.shape):
            traversible[self._motion_block_mask] = False
        if agent_cell is not None:
            self._set_patch(traversible, int(agent_cell[0]), int(agent_cell[1]), radius_cells=1, value=True)
        if goal_cell is not None:
            row, col = int(goal_cell[0]), int(goal_cell[1])
            if 0 <= row < traversible.shape[0] and 0 <= col < traversible.shape[1]:
                traversible[row, col] = True

    def _has_motion_blocks(self, shape: tuple[int, int]) -> bool:
        if self._motion_block_mask is None:
            return False
        if tuple(self._motion_block_mask.shape) != tuple(shape):
            return False
        return bool(np.any(self._motion_block_mask))

    def _next_local_recovery_command(self) -> tuple[int, str, bool] | None:
        if not self._local_recovery_queue:
            return None
        return self._local_recovery_queue.pop(0)

    def _start_local_bypass_recovery(
        self,
        *,
        agent_position: np.ndarray,
        agent_rotation,
        agent_cell: tuple[int, int],
        fmm_dist: np.ndarray,
        traversible: np.ndarray,
        navigation_map,
    ) -> tuple[int, str, bool] | None:
        sequence = self._build_local_bypass_sequence(
            agent_position=agent_position,
            agent_rotation=agent_rotation,
            agent_cell=agent_cell,
            fmm_dist=fmm_dist,
            traversible=traversible,
            navigation_map=navigation_map,
        )
        if not sequence:
            return None
        self._local_recovery_queue = sequence
        return self._next_local_recovery_command()

    def _build_local_bypass_sequence(
        self,
        *,
        agent_position: np.ndarray,
        agent_rotation,
        agent_cell: tuple[int, int],
        fmm_dist: np.ndarray,
        traversible: np.ndarray,
        navigation_map,
    ) -> list[tuple[int, str, bool]]:
        rows, cols = traversible.shape
        agent_row, agent_col = self._clamped_cell(agent_cell, rows, cols)
        current_distance = float(fmm_dist[agent_row, agent_col]) if fmm_dist.size else float("inf")
        if not math.isfinite(current_distance):
            return []

        forward_xz = _rotation_to_forward_xz(agent_rotation)
        resolution = max(1e-6, float(navigation_map.resolution_m))
        step_cells = max(1, int(round(self.forward_step_size_m / resolution)))
        candidates: list[tuple[float, int, float, bool]] = []
        for requested_angle in (
            15.0,
            -15.0,
            30.0,
            -30.0,
            45.0,
            -45.0,
            60.0,
            -60.0,
            90.0,
            -90.0,
            120.0,
            -120.0,
            150.0,
            -150.0,
        ):
            turn_steps = max(1, int(round(abs(requested_angle) / max(1e-6, self.turn_angle_deg))))
            actual_angle = math.copysign(turn_steps * self.turn_angle_deg, requested_angle)
            heading = self._rotate_xz(forward_xz, actual_angle)
            endpoint = np.asarray(agent_position, dtype=np.float32).copy()
            endpoint[0] += float(heading[0]) * self.forward_step_size_m
            endpoint[2] += float(heading[1]) * self.forward_step_size_m
            row, col = navigation_map.world_to_cell(endpoint)
            if not (0 <= row < rows and 0 <= col < cols):
                continue
            if not traversible[row, col]:
                continue
            if not self._line_is_traversible(traversible, agent_row, agent_col, row, col):
                continue
            candidate_distance = float(fmm_dist[row, col])
            if not math.isfinite(candidate_distance):
                continue
            distance_drop = current_distance - candidate_distance
            step_size_error = abs(
                float(np.linalg.norm(np.asarray([row - agent_row, col - agent_col], dtype=np.float32)))
                - float(step_cells)
            )
            blocked_heading_penalty = 0.12 if self._escape_machine.current_heading_is_blocked(heading) else 0.0
            turn_penalty = 0.02 * float(turn_steps)
            step_penalty = 0.01 * step_size_error
            blocked_heading = self._escape_machine.current_heading_is_blocked(heading)
            score = distance_drop - turn_penalty - step_penalty - blocked_heading_penalty
            candidates.append((score, turn_steps, actual_angle, blocked_heading))

        if not candidates:
            return []

        unblocked_candidates = [candidate for candidate in candidates if not candidate[3]]
        _score, _turn_steps, actual_angle, _blocked_heading = max(
            unblocked_candidates or candidates,
            key=lambda item: item[0],
        )
        preferred_sign = 1.0 if actual_angle > 0.0 else -1.0
        if self._local_recovery_failures % 2 == 1:
            preferred_sign *= -1.0

        sweep = [
            candidate
            for candidate in candidates
            if math.copysign(1.0, candidate[2]) == preferred_sign
        ]
        if not sweep:
            sweep = candidates
        sweep.sort(key=lambda item: (abs(float(item[2])), -float(item[0])))

        sequence: list[tuple[int, str, bool]] = []
        previous_turn_steps = 0
        probe_count = 0
        for _candidate_score, turn_steps, candidate_angle, _candidate_blocked in sweep:
            target_turn_steps = max(1, int(turn_steps))
            delta_turn_steps = max(0, target_turn_steps - previous_turn_steps)
            if delta_turn_steps <= 0:
                continue
            turn_action = (
                int(HabitatSimActions.turn_right)
                if candidate_angle > 0.0
                else int(HabitatSimActions.turn_left)
            )
            sequence.extend(
                (turn_action, "local_bypass_turn", False)
                for _ in range(delta_turn_steps)
            )
            sequence.append((int(HabitatSimActions.move_forward), "local_bypass_probe", True))
            previous_turn_steps = target_turn_steps
            probe_count += 1
            if probe_count >= 7:
                break

        if sequence:
            return sequence

        _score, turn_steps, actual_angle, _blocked_heading = max(
            unblocked_candidates or candidates,
            key=lambda item: item[0],
        )
        turn_action = (
            int(HabitatSimActions.turn_right)
            if actual_angle > 0.0
            else int(HabitatSimActions.turn_left)
        )
        sequence: list[tuple[int, str, bool]] = [
            (turn_action, "local_bypass_turn", False)
            for _ in range(max(0, int(turn_steps)))
        ]
        sequence.append((int(HabitatSimActions.move_forward), "local_bypass_probe", True))
        return sequence

    @staticmethod
    def _rotate_xz(vector: np.ndarray, angle_deg: float) -> np.ndarray:
        angle_rad = math.radians(float(angle_deg))
        cos_v = math.cos(angle_rad)
        sin_v = math.sin(angle_rad)
        x = float(vector[0])
        z = float(vector[1])
        return np.asarray([cos_v * x - sin_v * z, sin_v * x + cos_v * z], dtype=np.float32)

    @staticmethod
    def _line_is_traversible(
        traversible: np.ndarray,
        start_row: int,
        start_col: int,
        end_row: int,
        end_col: int,
    ) -> bool:
        row0, col0, row1, col1 = int(start_row), int(start_col), int(end_row), int(end_col)
        drow = abs(row1 - row0)
        dcol = abs(col1 - col0)
        steps = max(drow, dcol)
        if steps == 0:
            return bool(traversible[row0, col0])
        for index in range(steps + 1):
            t = float(index) / float(steps)
            row = int(round(row0 + (row1 - row0) * t))
            col = int(round(col0 + (col1 - col0) * t))
            if row < 0 or row >= traversible.shape[0] or col < 0 or col >= traversible.shape[1]:
                return False
            if not traversible[row, col]:
                return False
        return True

    def _update_collision_penalty(self, agent_position: np.ndarray, navigation_map) -> None:
        if self._last_action_result_recorded:
            return
        # Set before the early returns so each action is consumed once
        self._last_action_result_recorded = True
        if self._collision_penalty_mask is None:
            return
        if self._last_action is None or self._last_position is None or self._last_rotation is None:
            return
        if self._last_action != int(HabitatSimActions.move_forward):
            return

        displacement_m = _planar_distance(agent_position, self._last_position)
        forward_xz = _rotation_to_forward_xz(self._last_rotation)
        stalled = self._escape_machine.record_forward_result(
            displacement_m=displacement_m,
            forward_step_size_m=self.forward_step_size_m,
            forward_xz=forward_xz,
        )
        if not stalled:
            self._local_recovery_queue = []
            self._local_recovery_failures = 0
            return
        if self._last_action_was_local_probe:
            self._local_recovery_failures += 1

        self._paint_failed_forward_block(
            agent_position=np.asarray(self._last_position, dtype=np.float32),
            forward_xz=forward_xz,
            navigation_map=navigation_map,
        )

    def record_blocked_forward(self, *, agent_state, navigation_map) -> bool:
        """Record a blocked forward action once, cancelling it if pending."""

        pending = self._pending_action
        if (
            pending is not None
            and int(pending.command.action) == int(HabitatSimActions.move_forward)
        ):
            event: tuple[object, ...] = ("proposal", int(pending.proposal_id))
            forward_xz = _rotation_to_forward_xz(pending.agent_rotation)
            blocked_position = pending.agent_position.copy()
            self.cancel_pending_action()
        elif (
            self._last_action == int(HabitatSimActions.move_forward)
            and not self._last_action_result_recorded
        ):
            event = ("committed", int(self._committed_action_generation))
            forward_xz = _rotation_to_forward_xz(self._last_rotation)
            blocked_position = np.asarray(self._last_position, dtype=np.float32).copy()
            self._last_action_result_recorded = True
        else:
            forward_xz = _rotation_to_forward_xz(agent_state.rotation)
            blocked_position = np.asarray(agent_state.position, dtype=np.float32)
            # Unsolicited report, deduplicated by pose and heading
            event = (
                "external",
                *(round(float(value), 4) for value in blocked_position[[0, 2]]),
                *(round(float(value), 4) for value in forward_xz),
            )

        fingerprint = (
            *(round(float(value), 4) for value in blocked_position[[0, 2]]),
            *(round(float(value), 4) for value in forward_xz),
        )
        if event == self._last_blocked_forward_event or (
            event[0] == "external"
            and fingerprint == self._last_blocked_forward_fingerprint
        ):
            return False
        self._last_blocked_forward_event = event
        self._last_blocked_forward_fingerprint = fingerprint
        free_mask = np.asarray(navigation_map.free_mask(), dtype=bool)
        self._sync_collision_penalty(free_mask.shape, navigation_map)
        self._escape_machine.record_forward_result(
            displacement_m=0.0,
            forward_step_size_m=self.forward_step_size_m,
            forward_xz=forward_xz,
        )
        self._paint_failed_forward_block(
            agent_position=blocked_position,
            forward_xz=forward_xz,
            navigation_map=navigation_map,
        )
        return True

    def _begin_action_proposal(self) -> None:
        if self._active_proposal_state is not None or self._pending_action is not None:
            raise RuntimeError("an action proposal is already active")
        self._active_proposal_state = _ProposalState(
            escape_machine=copy.deepcopy(self._escape_machine),
            last_goal_position=(
                None
                if self._last_goal_position is None
                else self._last_goal_position.copy()
            ),
            local_recovery_queue=list(self._local_recovery_queue),
            local_recovery_failures=int(self._local_recovery_failures),
        )

    def _restore_proposal_state(self, state: _ProposalState) -> None:
        self._escape_machine = state.escape_machine
        self._last_goal_position = (
            None
            if state.last_goal_position is None
            else state.last_goal_position.copy()
        )
        self._local_recovery_queue = list(state.local_recovery_queue)
        self._local_recovery_failures = int(state.local_recovery_failures)

    def _paint_failed_forward_block(
        self,
        *,
        agent_position: np.ndarray,
        forward_xz: np.ndarray,
        navigation_map,
    ) -> None:
        if self._collision_penalty_mask is None:
            return
        rows, cols = self._collision_penalty_mask.shape
        start_row, start_col = self._clamped_cell(navigation_map.world_to_cell(agent_position), rows, cols)
        direction = np.asarray(
            [
                forward_xz[1] / max(1e-6, float(navigation_map.resolution_m)),
                forward_xz[0] / max(1e-6, float(navigation_map.resolution_m)),
            ],
            dtype=np.float32,
        )
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return
        direction /= norm
        lateral = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        step_cells = max(1, int(math.ceil(self.forward_step_size_m / max(1e-6, float(navigation_map.resolution_m)))))
        stall_limited = self._escape_machine.forward_stall_count >= self._stall_step_limit
        length_cells = step_cells + (2 if stall_limited else 0)
        half_width = (
            max(1, int(math.ceil((self.agent_radius_m + self.clearance_margin_m) / max(1e-6, float(navigation_map.resolution_m)) * 0.5)))
            if stall_limited
            else 0
        )
        penalty_increment = 0.5 if not stall_limited else 1.0
        for depth_index in range(1, length_cells + 1):
            center = np.asarray([start_row, start_col], dtype=np.float32) + direction * depth_index
            for width_index in range(-half_width, half_width + 1):
                cell = center + lateral * width_index
                row = int(round(float(cell[0])))
                col = int(round(float(cell[1])))
                if 0 <= row < rows and 0 <= col < cols:
                    self._collision_penalty_mask[row, col] = min(
                        1.0,
                        float(self._collision_penalty_mask[row, col]) + penalty_increment,
                    )
        self._paint_motion_block(
            start_row=start_row,
            start_col=start_col,
            direction=direction,
            lateral=lateral,
            length_cells=max(step_cells + 1, length_cells),
            half_width=max(
                1,
                int(
                    math.ceil(
                        (self.agent_radius_m + self.clearance_margin_m)
                        / max(1e-6, float(navigation_map.resolution_m))
                        * 0.5
                    )
                ),
            ),
        )

    def _paint_motion_block(
        self,
        *,
        start_row: int,
        start_col: int,
        direction: np.ndarray,
        lateral: np.ndarray,
        length_cells: int,
        half_width: int,
    ) -> None:
        if self._motion_block_mask is None:
            return
        rows, cols = self._motion_block_mask.shape
        for depth_index in range(1, max(1, int(length_cells)) + 1):
            center = np.asarray([start_row, start_col], dtype=np.float32) + direction * depth_index
            for width_index in range(-max(0, int(half_width)), max(0, int(half_width)) + 1):
                cell = center + lateral * width_index
                row = int(round(float(cell[0])))
                col = int(round(float(cell[1])))
                if 0 <= row < rows and 0 <= col < cols:
                    self._motion_block_mask[row, col] = True

    def _sync_collision_penalty(self, shape: tuple[int, int], navigation_map) -> None:
        origin_xz = self._navigation_origin_xz(navigation_map)
        if self._collision_penalty_mask is None:
            self._collision_penalty_mask = np.zeros(shape, dtype=np.float32)
            self._collision_origin_xz = origin_xz
        elif self._collision_penalty_mask.shape != shape:
            old = self._collision_penalty_mask
            new = np.zeros(shape, dtype=np.float32)
            row_offset = 0
            col_offset = 0
            if self._collision_origin_xz is not None:
                resolution = max(1e-6, float(navigation_map.resolution_m))
                col_offset = int(round((float(self._collision_origin_xz[0]) - float(origin_xz[0])) / resolution))
                row_offset = int(round((float(self._collision_origin_xz[1]) - float(origin_xz[1])) / resolution))
            self._copy_with_offset(old, new, row_offset=row_offset, col_offset=col_offset)
            self._collision_penalty_mask = new
        self._collision_origin_xz = origin_xz

        if self._motion_block_mask is None:
            self._motion_block_mask = np.zeros(shape, dtype=bool)
            self._motion_block_origin_xz = origin_xz
            return
        if self._motion_block_mask.shape != shape:
            old_motion = self._motion_block_mask
            new_motion = np.zeros(shape, dtype=bool)
            row_offset = 0
            col_offset = 0
            if self._motion_block_origin_xz is not None:
                resolution = max(1e-6, float(navigation_map.resolution_m))
                col_offset = int(round((float(self._motion_block_origin_xz[0]) - float(origin_xz[0])) / resolution))
                row_offset = int(round((float(self._motion_block_origin_xz[1]) - float(origin_xz[1])) / resolution))
            self._copy_with_offset(old_motion, new_motion, row_offset=row_offset, col_offset=col_offset)
            self._motion_block_mask = new_motion
        self._motion_block_origin_xz = origin_xz

    def _sync_goal_tracking(self, goal_world: np.ndarray) -> None:
        if self._last_goal_position is None:
            self._last_goal_position = goal_world.copy()
            return
        if _planar_distance(goal_world, self._last_goal_position) > max(0.5, self.forward_step_size_m * 2.0):
            self._escape_machine.replan_failed_count = 0
            self._local_recovery_queue = []
            self._local_recovery_failures = 0
            self._last_goal_position = goal_world.copy()

    @staticmethod
    def _navigation_origin_xz(navigation_map) -> np.ndarray:
        origin = getattr(navigation_map, "origin_xz", None)
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

    def _remember_blocked_forward_dir(self, forward_xz: np.ndarray) -> None:
        self._escape_machine.record_forward_result(
            displacement_m=0.0,
            forward_step_size_m=self.forward_step_size_m,
            forward_xz=forward_xz,
        )

    def _current_heading_is_blocked(self, rotation) -> bool:
        return self._escape_machine.current_heading_is_blocked(_rotation_to_forward_xz(rotation))

    def _recovery_turn_action(self) -> int:
        return self._escape_machine.recovery_turn_action()

    def _start_escape_recovery(self) -> int:
        return int(self._escape_machine.start_escape_recovery())

    def _next_escape_command(self) -> tuple[int, str] | None:
        escape = self._escape_machine.next_pending_action()
        if escape is None:
            return None
        return int(escape.action), escape.reason

    @staticmethod
    def _set_multi_goal(planner, goal_map: np.ndarray, agent_cell: tuple[int, int]) -> None:
        set_multi_goal = planner.set_multi_goal
        try:
            parameters = inspect.signature(set_multi_goal).parameters
        except (TypeError, ValueError):
            parameters = {}

        if "state" in parameters:
            set_multi_goal(goal_map, list(agent_cell))
            return
        if "visited" in parameters:
            set_multi_goal(goal_map, None)
            return
        try:
            set_multi_goal(goal_map, list(agent_cell))
        except TypeError:
            set_multi_goal(goal_map)

    @staticmethod
    def _clamped_cell(cell: tuple[int, int], rows: int, cols: int) -> tuple[int, int]:
        row, col = cell
        row = int(np.clip(row, 0, max(0, rows - 1)))
        col = int(np.clip(col, 0, max(0, cols - 1)))
        return row, col

    @staticmethod
    def _set_patch(mask: np.ndarray, row: int, col: int, radius_cells: int, value: bool) -> None:
        row_min = max(0, row - radius_cells)
        row_max = min(mask.shape[0], row + radius_cells + 1)
        col_min = max(0, col - radius_cells)
        col_max = min(mask.shape[1], col + radius_cells + 1)
        mask[row_min:row_max, col_min:col_max] = value

    @staticmethod
    def _set_patch_value(array: np.ndarray, row: int, col: int, radius_cells: int, value: float) -> None:
        row_min = max(0, row - radius_cells)
        row_max = min(array.shape[0], row + radius_cells + 1)
        col_min = max(0, col - radius_cells)
        col_max = min(array.shape[1], col + radius_cells + 1)
        array[row_min:row_max, col_min:col_max] = float(value)

    def _finalize_command(
        self,
        *,
        agent_state,
        action: int,
        heading_error_deg: float | None,
        goal_distance_m: float | None,
        planner_replan_failed: bool,
        short_term_goal: np.ndarray | None = None,
        goal_reached: bool = False,
        goal_blocked: bool = False,
        local_blocked: bool = False,
        block_scope: str = "none",
        recovery_reason: str | None = None,
        local_recovery_probe: bool = False,
        suppress_stall_block: bool = False,
    ) -> NavigationCommand:
        progress_stall_count = int(self._escape_machine.forward_stall_count)
        if not suppress_stall_block and not planner_replan_failed and not goal_reached:
            escape = self._escape_machine.maybe_escape_forward(
                action=int(action),
                forward_xz=_rotation_to_forward_xz(agent_state.rotation),
            )
        else:
            escape = None
        if escape is not None:
            action = int(escape.action)
            planner_replan_failed = False
            local_blocked = True
            block_scope = "local"
            recovery_reason = "forward_stall_limit"
            local_recovery_probe = int(action) == int(HabitatSimActions.move_forward)
        replan_failure_limit = self._escape_machine.record_plan_result(
            planner_replan_failed=bool(planner_replan_failed),
            goal_reached=bool(goal_reached),
        )
        if replan_failure_limit:
            if not goal_blocked:
                recovery_reason = "replan_failure_limit"
            goal_blocked = True
            block_scope = "goal"
        if goal_blocked and block_scope == "none":
            block_scope = "goal"
        if local_blocked and block_scope == "none":
            block_scope = "local"
        recovery_active = bool(planner_replan_failed or progress_stall_count > 0 or recovery_reason or local_blocked)
        command = NavigationCommand(
            action=int(action),
            heading_error_deg=heading_error_deg,
            goal_distance_m=goal_distance_m,
            stalled_count=int(self._escape_machine.forward_stall_count),
            replan_failed_count=int(self._escape_machine.replan_failed_count),
            progress_stall_count=progress_stall_count,
            planner_replan_failed=bool(planner_replan_failed),
            short_term_goal=None if short_term_goal is None else np.asarray(short_term_goal, dtype=np.float32),
            goal_reached=bool(goal_reached),
            goal_blocked=bool(goal_blocked),
            local_blocked=bool(local_blocked),
            block_scope=str(block_scope),
            recovery_active=recovery_active,
            recovery_reason=recovery_reason,
        )
        prior_state = self._active_proposal_state
        if prior_state is None:
            raise RuntimeError("navigation command finalized outside an active proposal")
        self._proposal_sequence += 1
        self._pending_action = _PendingAction(
            proposal_id=int(self._proposal_sequence),
            command=command,
            agent_position=np.asarray(agent_state.position, dtype=np.float32).copy(),
            agent_rotation=agent_state.rotation,
            local_recovery_probe=bool(local_recovery_probe),
            prior_state=prior_state,
        )
        self._active_proposal_state = None
        return command
