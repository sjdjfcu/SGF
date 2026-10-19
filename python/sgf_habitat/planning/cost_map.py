from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

_CPP_BACKEND = None
_CPP_BACKEND_CHECKED = False
_CPP_BACKEND_ERROR: Exception | None = None


def _sgf_cpp_backend():
    global _CPP_BACKEND, _CPP_BACKEND_CHECKED, _CPP_BACKEND_ERROR
    if not _CPP_BACKEND_CHECKED:
        _CPP_BACKEND_CHECKED = True
        try:
            from .. import _sgf as backend
        except Exception as exc:
            backend = None
            _CPP_BACKEND_ERROR = exc
        _CPP_BACKEND = backend
    if _CPP_BACKEND is None:
        raise RuntimeError("SGF C++ extension is required for cost map generation") from _CPP_BACKEND_ERROR
    return _CPP_BACKEND


@dataclass(frozen=True)
class CostMapConfig:
    agent_radius_m: float
    clearance_margin_m: float = 0.05
    clearance_penalty: float = 9.0
    collision_penalty: float = 12.0
    max_cost: float = 64.0
    min_speed: float = 1e-3
    goal_relaxation_radius_m: float = 0.55


@dataclass(frozen=True)
class CostMapResult:
    traversible: np.ndarray
    cost: np.ndarray
    speed: np.ndarray
    clearance_m: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


class CostMapBuilder:
    """Builds the cost/speed field consumed by the cost-weighted FMM planner."""

    def __init__(self, config: CostMapConfig) -> None:
        self.config = config

    def build(
        self,
        *,
        free_mask: np.ndarray,
        occupied_mask: np.ndarray,
        resolution_m: float,
        collision_mask: np.ndarray | None = None,
        agent_cell: tuple[int, int] | None = None,
        goal_cell: tuple[int, int] | None = None,
        goal_clearance_margin_m: float | None = None,
    ) -> CostMapResult:
        free = np.asarray(free_mask, dtype=bool)
        occupied = np.asarray(occupied_mask, dtype=bool)
        if free.shape != occupied.shape:
            raise ValueError("free_mask and occupied_mask must have the same shape")

        resolution = max(1e-6, float(resolution_m))
        backend = _sgf_cpp_backend()
        result = backend.build_cost_map(
            free,
            occupied,
            resolution,
            float(self.config.agent_radius_m),
            float(self.config.clearance_margin_m),
            float(self.config.clearance_penalty),
            float(self.config.collision_penalty),
            float(self.config.max_cost),
            float(self.config.min_speed),
            float(self.config.goal_relaxation_radius_m),
            None if collision_mask is None else np.asarray(collision_mask, dtype=np.float32),
            None if agent_cell is None else (int(agent_cell[0]), int(agent_cell[1])),
            None if goal_cell is None else (int(goal_cell[0]), int(goal_cell[1])),
            None if goal_clearance_margin_m is None else float(goal_clearance_margin_m),
        )
        return CostMapResult(
            traversible=np.asarray(result["traversible"], dtype=bool),
            cost=np.asarray(result["cost"], dtype=np.float32),
            speed=np.asarray(result["speed"], dtype=np.float32),
            clearance_m=np.asarray(result["clearance_m"], dtype=np.float32),
            diagnostics=dict(result["diagnostics"]),
        )
