"""Policy length and time scales derived from task and platform parameters."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite_positive(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class PolicyScales:
    """Task and platform quantities plus the policy scales they imply."""

    success_radius_m: float
    forward_step_m: float
    turn_angle_deg: float
    min_depth_m: float
    max_depth_m: float
    hfov_deg: float
    map_resolution_m: float

    def __post_init__(self) -> None:
        for name in (
            "success_radius_m",
            "forward_step_m",
            "turn_angle_deg",
            "min_depth_m",
            "max_depth_m",
            "hfov_deg",
            "map_resolution_m",
        ):
            object.__setattr__(
                self, name, _finite_positive(getattr(self, name), name=name)
            )
        if self.turn_angle_deg > 360.0:
            raise ValueError("turn_angle_deg must not exceed 360 degrees")
        if self.hfov_deg > 360.0:
            raise ValueError("hfov_deg must not exceed 360 degrees")

    # ------------------------------------------------------------ time scales
    @property
    def full_turn_actions(self) -> int:
        """Actions in one full in-place rotation, ceil(360 / theta)."""

        return max(1, int(math.ceil(360.0 / self.turn_angle_deg)))

    @property
    def evidence_max_age_steps(self) -> int:
        """Steps a localized visual target stays valid without re-observation."""

        return self.full_turn_actions

    # ---------------------------------------------------------- length scales
    @property
    def terminal_approach_distance_m(self) -> float:
        """Closest pursued standoff from a target: the depth sensor's floor."""

        return self.min_depth_m

    @property
    def attempt_radius_m(self) -> float:
        """Decisive validation band: within the task's own success radius."""

        return self.success_radius_m

    @property
    def far_band_radius_m(self) -> float:
        """Evidence-only validation band, one forward step outside R."""

        return self.success_radius_m + self.forward_step_m

    @property
    def hypothesis_exclusion_radius_m(self) -> float:
        """Stop hypotheses closer than R are treated as one."""

        return self.success_radius_m

    @property
    def frontier_reach_radius_m(self) -> float:
        """Frontier arrival radius (final step stays outside the depth blind zone)."""

        return self.min_depth_m + self.forward_step_m

    @property
    def progress_quantum_m(self) -> float:
        """Smallest progress the platform can make: one forward step."""

        return self.forward_step_m

    @property
    def planar_progress_quantum_m(self) -> float:
        """Minimum planar progress of one on-heading step, s * cos(theta / 2)."""

        return self.forward_step_m * math.cos(math.radians(0.5 * self.turn_angle_deg))

    @property
    def distinct_view_yaw_deg(self) -> float:
        """Yaw change at which two views overlap by at most half, hfov / 2."""

        return 0.5 * self.hfov_deg

    @property
    def distinct_view_distance_m(self) -> float:
        """Translation at which two views overlap by at most half at min_depth."""

        return self.min_depth_m * math.tan(math.radians(0.5 * self.hfov_deg))

    @property
    def blocked_displacement_m(self) -> float:
        """A forward action is blocked when it moves less than one map cell."""

        return self.map_resolution_m

    # --------------------------------------------------------- image scales
    @property
    def centering_tolerance_fraction(self) -> float:
        """Centering tolerance as a fraction of image width, theta / (2 hfov)."""

        return self.turn_angle_deg / (2.0 * self.hfov_deg)

    @property
    def heading_tolerance_deg(self) -> float:
        """Servo heading tolerance: half a turn increment."""

        return 0.5 * self.turn_angle_deg

    def steps_for_distance(self, distance_m: float) -> int:
        """Forward actions needed to cover distance_m (ceiling)."""

        value = float(distance_m)
        if not math.isfinite(value) or value <= 0.0:
            return 0
        return int(math.ceil(value / self.forward_step_m))

    @property
    def terminal_press_steps(self) -> int:
        """Forward actions spanning the depth floor, for press-to-contact."""

        return self.steps_for_distance(self.min_depth_m)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "success_radius_m": self.success_radius_m,
            "forward_step_m": self.forward_step_m,
            "turn_angle_deg": self.turn_angle_deg,
            "min_depth_m": self.min_depth_m,
            "max_depth_m": self.max_depth_m,
            "hfov_deg": self.hfov_deg,
            "map_resolution_m": self.map_resolution_m,
            "full_turn_actions": self.full_turn_actions,
            "evidence_max_age_steps": self.evidence_max_age_steps,
            "terminal_approach_distance_m": self.terminal_approach_distance_m,
            "attempt_radius_m": self.attempt_radius_m,
            "far_band_radius_m": self.far_band_radius_m,
            "hypothesis_exclusion_radius_m": self.hypothesis_exclusion_radius_m,
            "frontier_reach_radius_m": self.frontier_reach_radius_m,
            "progress_quantum_m": self.progress_quantum_m,
            "planar_progress_quantum_m": self.planar_progress_quantum_m,
            "distinct_view_distance_m": self.distinct_view_distance_m,
            "distinct_view_yaw_deg": self.distinct_view_yaw_deg,
            "blocked_displacement_m": self.blocked_displacement_m,
            "centering_tolerance_fraction": self.centering_tolerance_fraction,
            "heading_tolerance_deg": self.heading_tolerance_deg,
            "terminal_press_steps": self.terminal_press_steps,
        }


__all__ = ["PolicyScales"]
