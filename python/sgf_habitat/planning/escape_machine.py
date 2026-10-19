from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EscapeAction:
    action: int
    reason: str


@dataclass(frozen=True)
class EscapeMachineConfig:
    forward_action: int
    turn_left_action: int
    turn_right_action: int
    turn_angle_deg: float
    stall_steps: int = 3
    replan_failures: int = 8
    progress_min_m: float = 0.05
    progress_fraction: float = 0.1
    blocked_heading_deg: float = 45.0


class EscapeMachine:
    """Tracks local motion failures and emits bounded recovery actions."""

    def __init__(self, config: EscapeMachineConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.forward_stall_count = 0
        self.replan_failed_count = 0
        self.state = "committed_forward"
        self._blocked_forward_dirs: list[np.ndarray] = []
        self._turn_steps_remaining = 0
        self._probe_pending = False
        self._turn_action = int(self.config.turn_left_action)
        # Turn direction for the current run of failed plans, flipped per run
        self._streak_turn_left = True

    @property
    def blocked_confidence(self) -> float:
        stall_score = self.forward_stall_count / max(1.0, float(self.config.stall_steps))
        replan_score = self.replan_failed_count / max(1.0, float(self.config.replan_failures))
        return float(np.clip(max(stall_score, replan_score), 0.0, 1.0))

    def record_forward_result(
        self,
        *,
        displacement_m: float,
        forward_step_size_m: float,
        forward_xz: np.ndarray,
    ) -> bool:
        progress_threshold = max(
            float(self.config.progress_min_m),
            float(self.config.progress_fraction) * max(0.0, float(forward_step_size_m)),
        )
        if float(displacement_m) >= progress_threshold:
            self.forward_stall_count = 0
            self._blocked_forward_dirs = []
            self._clear_pending_recovery()
            return False

        self.forward_stall_count += 1
        self._remember_blocked_forward_dir(forward_xz)
        return True

    def next_pending_action(self) -> EscapeAction | None:
        if self._turn_steps_remaining > 0:
            self._turn_steps_remaining -= 1
            if self._turn_steps_remaining == 0:
                self._probe_pending = True
                self.state = "probing_turn"
            return EscapeAction(action=self._turn_action, reason="escape_turn")
        if self._probe_pending:
            self._probe_pending = False
            self.state = "committed_forward"
            return EscapeAction(action=int(self.config.forward_action), reason="escape_probe")
        return None

    def maybe_escape_forward(self, *, action: int, forward_xz: np.ndarray) -> EscapeAction | None:
        if int(action) != int(self.config.forward_action):
            return None
        if self.forward_stall_count < int(self.config.stall_steps):
            return None
        if not self.current_heading_is_blocked(forward_xz):
            return None
        return EscapeAction(action=self.start_escape_recovery(), reason="forward_stall_limit")

    def start_escape_recovery(self) -> int:
        turn_steps = max(3, int(round(90.0 / max(1e-6, float(self.config.turn_angle_deg)))))
        self._turn_action = self.recovery_turn_action()
        self._turn_steps_remaining = max(0, turn_steps - 1)
        self._probe_pending = self._turn_steps_remaining == 0
        self.state = "escape_long_turn"
        return self._turn_action

    def record_plan_result(self, *, planner_replan_failed: bool, goal_reached: bool) -> bool:
        if planner_replan_failed:
            self.replan_failed_count += 1
        else:
            if self.replan_failed_count > 0:
                self._streak_turn_left = not self._streak_turn_left
            self.replan_failed_count = 0
        return bool(
            planner_replan_failed
            and not goal_reached
            and self.replan_failed_count >= int(self.config.replan_failures)
        )

    def recovery_turn_action(self) -> int:
        if self._streak_turn_left:
            return int(self.config.turn_left_action)
        return int(self.config.turn_right_action)

    def current_heading_is_blocked(self, forward_xz: np.ndarray) -> bool:
        if not self._blocked_forward_dirs:
            return True
        direction = self._normalize_xz(forward_xz)
        if direction is None:
            return True
        blocked_cos = math.cos(math.radians(float(self.config.blocked_heading_deg)))
        return any(float(np.dot(blocked_dir, direction)) >= blocked_cos for blocked_dir in self._blocked_forward_dirs)

    def _clear_pending_recovery(self) -> None:
        self.state = "committed_forward"
        self._turn_steps_remaining = 0
        self._probe_pending = False

    def _remember_blocked_forward_dir(self, forward_xz: np.ndarray) -> None:
        direction = self._normalize_xz(forward_xz)
        if direction is None:
            return
        merge_cos = math.cos(math.radians(12.0))
        for index, existing in enumerate(self._blocked_forward_dirs):
            if float(np.dot(existing, direction)) >= merge_cos:
                self._blocked_forward_dirs[index] = direction
                return
        self._blocked_forward_dirs.append(direction)
        if len(self._blocked_forward_dirs) > 16:
            self._blocked_forward_dirs = self._blocked_forward_dirs[-16:]

    @staticmethod
    def _normalize_xz(forward_xz: np.ndarray) -> np.ndarray | None:
        direction = np.asarray(forward_xz, dtype=np.float32).reshape(-1)
        if direction.size < 2:
            return None
        norm = float(np.linalg.norm(direction[:2]))
        if norm < 1e-6:
            return None
        return direction[:2] / norm
