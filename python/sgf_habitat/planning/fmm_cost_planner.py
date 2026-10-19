from __future__ import annotations

import numpy as np
import skfmm
from numpy import ma


class FMMCostPlanner:
    """Cost-weighted FMM planner with the set_goal/set_multi_goal interface."""

    def __init__(
        self,
        traversible: np.ndarray,
        *,
        cost: np.ndarray | None = None,
        speed: np.ndarray | None = None,
        dx: float = 1.0,
        scale: float = 1.0,
        step_size: int = 5,
    ) -> None:
        self.scale = float(scale)
        self.step_size = int(step_size)
        self.dx = max(1e-6, float(dx))
        self.traversible = np.asarray(traversible, dtype=bool)
        if self.traversible.ndim != 2:
            raise ValueError("traversible must be a 2D array")
        if speed is not None:
            speed_array = np.asarray(speed, dtype=np.float32)
            if speed_array.shape != self.traversible.shape:
                raise ValueError("speed shape must match traversible shape")
            self.speed = np.where(self.traversible, np.maximum(speed_array, 1e-6), 0.0).astype(np.float32)
            self.cost = np.where(self.traversible, 1.0 / np.maximum(self.speed, 1e-6), np.inf).astype(np.float32)
        else:
            if cost is None:
                cost_array = np.ones(self.traversible.shape, dtype=np.float32)
            else:
                cost_array = np.asarray(cost, dtype=np.float32)
                if cost_array.shape != self.traversible.shape:
                    raise ValueError("cost shape must match traversible shape")
            self.cost = np.where(self.traversible, np.maximum(cost_array, 1.0), np.inf).astype(np.float32)
            self.speed = np.where(self.traversible, 1.0 / self.cost, 0.0).astype(np.float32)
        self.fmm_dist = np.full(self.traversible.shape, np.inf, dtype=np.float32)

    def set_goal(self, goal, auto_improve: bool = False) -> None:
        _ = auto_improve
        goal_map = np.zeros(self.traversible.shape, dtype=np.uint8)
        row, col = int(goal[0]), int(goal[1])
        if 0 <= row < goal_map.shape[0] and 0 <= col < goal_map.shape[1]:
            goal_map[row, col] = 1
        self.set_multi_goal(goal_map, None)

    def set_multi_goal(self, goal_map: np.ndarray, state=None) -> None:
        goals = np.asarray(goal_map) > 0
        if goals.shape != self.traversible.shape:
            raise ValueError("goal_map shape must match traversible shape")

        goal_mask = goals & self.traversible
        if not np.any(goal_mask):
            nearest = self._nearest_traversible_goal(goals, state)
            if nearest is None:
                self.fmm_dist = np.full(self.traversible.shape, np.inf, dtype=np.float32)
                return
            goal_mask[nearest[0], nearest[1]] = True

        phi = ma.array(np.ones(self.traversible.shape, dtype=np.float32), mask=~self.traversible)
        phi[goal_mask] = 0.0
        speed = ma.array(self.speed.astype(np.float32), mask=~self.traversible)
        try:
            distance = skfmm.travel_time(phi, speed, dx=self.dx)
        except Exception:
            self.fmm_dist = np.full(self.traversible.shape, np.inf, dtype=np.float32)
            return

        filled = ma.filled(distance, np.inf).astype(np.float32)
        filled[~self.traversible] = np.inf
        _ = state
        self.fmm_dist = filled

    def _nearest_traversible_goal(self, goals: np.ndarray, state) -> tuple[int, int] | None:
        traversible_cells = np.argwhere(self.traversible)
        if traversible_cells.size == 0:
            return None
        goal_cells = np.argwhere(goals)
        if goal_cells.size == 0:
            if state is None:
                return None
            goal_cells = np.asarray([[int(state[0]), int(state[1])]], dtype=np.int64)
        best_cell: tuple[int, int] | None = None
        best_score = float("inf")
        for goal in goal_cells:
            delta = traversible_cells - goal.reshape(1, 2)
            dist2 = np.sum(delta * delta, axis=1)
            index = int(np.argmin(dist2))
            score = float(dist2[index])
            if score < best_score:
                best_score = score
                best = traversible_cells[index]
                best_cell = (int(best[0]), int(best[1]))
        return best_cell
