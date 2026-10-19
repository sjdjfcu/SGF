from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..navigation.explorer import NavigationMapExplorer
from .step_cost import (
    action_rate_scaling,
    frontier_bearings_rad,
    kinematic_step_costs,
)


@dataclass(frozen=True)
class FrontierUtility:
    goal: np.ndarray
    utility: float
    information_gain: float
    goal_likelihood: float
    cost: float
    revisit_penalty: float
    area: int


def information_rate_utilities(
    information_gain: np.ndarray,
    goal_likelihood: np.ndarray,
    cost_m: np.ndarray,
    revisit_penalty: np.ndarray,
    *,
    info_gain_weight: float,
    goal_likelihood_weight: float,
    revisit_weight: float,
    cost_offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rate utilities (w_I I + w_G G - w_R R) / (C + c0) and their components."""

    gain = np.asarray(information_gain, dtype=np.float64).reshape(-1)
    likelihood = np.asarray(goal_likelihood, dtype=np.float64).reshape(-1)
    cost = np.asarray(cost_m, dtype=np.float64).reshape(-1)
    revisit = np.asarray(revisit_penalty, dtype=np.float64).reshape(-1)
    if not (gain.size == likelihood.size == cost.size == revisit.size):
        raise ValueError("frontier term arrays must share one length")
    if gain.size == 0:
        raise ValueError("frontier term arrays must be non-empty")

    offset = float(cost_offset_m)
    if not math.isfinite(offset) or offset <= 0.0:
        raise ValueError("cost_offset_m must be a positive finite length")

    safe_cost = np.where(np.isfinite(cost), np.maximum(cost, 0.0), np.inf)

    denominator = safe_cost + offset

    components = np.stack(
        [
            float(info_gain_weight) * gain / denominator,
            float(goal_likelihood_weight) * likelihood / denominator,
            np.zeros_like(denominator),
            -float(revisit_weight) * revisit / denominator,
        ],
        axis=1,
    )
    components = np.where(np.isfinite(components), components, 0.0)
    return np.sum(components, axis=1), components


def action_information_rate_utilities(
    information_gain: np.ndarray,
    goal_likelihood: np.ndarray,
    cost_m: np.ndarray,
    revisit_penalty: np.ndarray,
    bearing_rad: np.ndarray,
    heading_rad: float,
    *,
    info_gain_weight: float,
    goal_likelihood_weight: float,
    revisit_weight: float,
    cost_offset_m: float,
    forward_step_size_m: float,
    turn_angle_deg: float,
    rotation_cost_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rate utilities with travel cost measured in actions instead of meters."""

    gain = np.asarray(information_gain, dtype=np.float64).reshape(-1)
    likelihood = np.asarray(goal_likelihood, dtype=np.float64).reshape(-1)
    cost = np.asarray(cost_m, dtype=np.float64).reshape(-1)
    revisit = np.asarray(revisit_penalty, dtype=np.float64).reshape(-1)
    bearing = np.asarray(bearing_rad, dtype=np.float64).reshape(-1)
    if not (
        gain.size
        == likelihood.size
        == cost.size
        == revisit.size
        == bearing.size
    ):
        raise ValueError("frontier term and bearing arrays must share one length")
    if gain.size == 0:
        raise ValueError("frontier term arrays must be non-empty")

    step_size = float(forward_step_size_m)
    offset_m = float(cost_offset_m)
    if not math.isfinite(offset_m) or offset_m <= 0.0:
        raise ValueError("cost_offset_m must be a positive finite length")

    step_costs = kinematic_step_costs(
        cost,
        bearing,
        float(heading_rad),
        forward_step_size_m=step_size,
        turn_angle_deg=float(turn_angle_deg),
        rotation_weight=float(rotation_cost_weight),
    )
    scaling = action_rate_scaling(
        step_costs,
        step_offset=offset_m / step_size,
    )
    components = np.stack(
        [
            float(info_gain_weight) * gain * scaling,
            float(goal_likelihood_weight) * likelihood * scaling,
            np.zeros_like(scaling),
            -float(revisit_weight) * revisit * scaling,
        ],
        axis=1,
    )
    components = np.where(np.isfinite(components), components, 0.0)
    return np.sum(components, axis=1), components


UTILITY_FORMS = ("rate", "difference", "nearest")
MAX_EXCHANGE_RATE_SAMPLES = 64


def frontier_step_costs(
    cost_m: np.ndarray,
    *,
    forward_step_size_m: float,
    turn_angle_deg: float,
    bearing_rad: np.ndarray | None,
    heading_rad: float | None,
    rotation_cost_weight: float = 1.0,
) -> np.ndarray:
    """Actions to reach each frontier (translation only without a heading)."""

    cost = np.asarray(cost_m, dtype=np.float64).reshape(-1)
    if bearing_rad is not None and heading_rad is not None:
        return kinematic_step_costs(
            cost,
            np.asarray(bearing_rad, dtype=np.float64),
            float(heading_rad),
            forward_step_size_m=float(forward_step_size_m),
            turn_angle_deg=float(turn_angle_deg),
            rotation_weight=float(rotation_cost_weight),
        )
    step_size = float(forward_step_size_m)
    if not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("forward_step_size_m must be finite and positive")
    return np.where(
        np.isfinite(cost),
        np.maximum(cost, 0.0) / step_size,
        np.inf,
    )


def frontier_objective(
    information_gain: np.ndarray,
    goal_likelihood: np.ndarray,
    cost_m: np.ndarray,
    revisit_penalty: np.ndarray,
    *,
    form: str,
    info_gain_weight: float,
    goal_likelihood_weight: float,
    revisit_weight: float,
    cost_offset_m: float,
    forward_step_size_m: float,
    turn_angle_deg: float,
    bearing_rad: np.ndarray | None = None,
    heading_rad: float | None = None,
    rotation_cost_weight: float = 1.0,
    exchange_rate: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Utilities and components under the rate, difference or nearest form."""

    if form not in UTILITY_FORMS:
        raise ValueError(f"unknown frontier utility form: {form!r}")
    if form == "rate":
        if bearing_rad is not None and heading_rad is not None:
            return action_information_rate_utilities(
                information_gain,
                goal_likelihood,
                cost_m,
                revisit_penalty,
                bearing_rad,
                float(heading_rad),
                info_gain_weight=info_gain_weight,
                goal_likelihood_weight=goal_likelihood_weight,
                revisit_weight=revisit_weight,
                cost_offset_m=cost_offset_m,
                forward_step_size_m=forward_step_size_m,
                turn_angle_deg=turn_angle_deg,
                rotation_cost_weight=rotation_cost_weight,
            )
        return information_rate_utilities(
            information_gain,
            goal_likelihood,
            cost_m,
            revisit_penalty,
            info_gain_weight=info_gain_weight,
            goal_likelihood_weight=goal_likelihood_weight,
            revisit_weight=revisit_weight,
            cost_offset_m=cost_offset_m,
        )

    gain = np.asarray(information_gain, dtype=np.float64).reshape(-1)
    likelihood = np.asarray(goal_likelihood, dtype=np.float64).reshape(-1)
    cost = np.asarray(cost_m, dtype=np.float64).reshape(-1)
    revisit = np.asarray(revisit_penalty, dtype=np.float64).reshape(-1)
    if not (gain.size == likelihood.size == cost.size == revisit.size):
        raise ValueError("frontier term arrays must share one length")
    if gain.size == 0:
        raise ValueError("frontier term arrays must be non-empty")
    steps = frontier_step_costs(
        cost,
        forward_step_size_m=forward_step_size_m,
        turn_angle_deg=turn_angle_deg,
        bearing_rad=bearing_rad,
        heading_rad=heading_rad,
        rotation_cost_weight=rotation_cost_weight,
    )
    finite = np.isfinite(steps)
    if not bool(np.all(finite)):
        ceiling = (
            float(np.max(steps[finite])) if bool(np.any(finite)) else 0.0
        ) + float(cost_offset_m) / float(forward_step_size_m)
        steps = np.where(finite, steps, ceiling)
    if form == "difference":
        rate = float(exchange_rate)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("exchange_rate must be finite and non-negative")
        components = np.stack(
            [
                float(info_gain_weight) * gain,
                float(goal_likelihood_weight) * likelihood,
                -rate * steps,
                -float(revisit_weight) * revisit,
            ],
            axis=1,
        )
    else:
        zeros = np.zeros_like(steps)
        components = np.stack([zeros, zeros, -steps, zeros], axis=1)
    return np.sum(components, axis=1), components


@dataclass(frozen=True)
class ConfidenceConstrainedFrontierChoice:
    selected_index: int
    baseline_index: int
    coverage_best: float
    visual_top_index: int | None
    visual_scores: tuple[float, ...]
    effective_support: tuple[float, ...]
    corroboration: tuple[float, ...]
    selectivity: float
    trust: tuple[float, ...]
    trust_product: tuple[float, ...]
    trust_minimum: tuple[float, ...]
    trust_geometric: tuple[float, ...]
    coverage_regret: tuple[float, ...]
    component_contrast: tuple[float, ...]
    allowed_regret: tuple[float, ...]
    admissible: tuple[bool, ...]


def confidence_constrained_frontier_choice(
    coverage_utilities: np.ndarray,
    visual_support: np.ndarray,
    coverage_components: np.ndarray,
    *,
    baseline_index: int | None = None,
) -> ConfidenceConstrainedFrontierChoice:
    """Coverage argmax unless the visual top has regret <= trust * L1 contrast."""

    utilities = np.asarray(coverage_utilities, dtype=np.float64).reshape(-1)
    if utilities.size == 0:
        raise ValueError("coverage_utilities must contain at least one frontier")
    if not np.all(np.isfinite(utilities)):
        raise ValueError("coverage_utilities must be finite")

    components = np.asarray(coverage_components, dtype=np.float64)
    if (
        components.ndim != 2
        or components.shape[0] != utilities.size
        or components.shape[1] == 0
    ):
        raise ValueError(
            "coverage_components must have shape "
            "(number_of_frontiers, number_of_components)"
        )
    if not np.all(np.isfinite(components)):
        raise ValueError("coverage_components must be finite")
    component_sums = np.sum(components, axis=1)
    # Tolerance for float32 rounding in the native utility sum
    component_l1 = np.sum(np.abs(components), axis=1)
    float32_roundoff = (
        (2 * components.shape[1] + 1)
        * np.finfo(np.float32).eps
        * np.maximum(1.0, component_l1)
    )
    if np.any(np.abs(component_sums - utilities) > float32_roundoff):
        raise ValueError(
            "coverage_utilities must equal the sum of coverage_components"
        )

    support = np.asarray(visual_support, dtype=np.float64)
    if support.ndim != 2 or support.shape[0] != utilities.size:
        raise ValueError(
            "visual_support must have shape (number_of_frontiers, number_of_records)"
        )
    if not np.all(np.isfinite(support)):
        raise ValueError("visual_support must be finite")
    if np.any(support < 0.0) or np.any(support > 1.0):
        raise ValueError("visual_support must be in [0, 1]")

    best_utility = float(np.max(utilities))
    if baseline_index is None:
        baseline = int(np.argmax(utilities))
    else:
        baseline = int(baseline_index)
        if baseline < 0 or baseline >= utilities.size:
            raise ValueError("baseline_index is out of range")
        if float(utilities[baseline]) != best_utility:
            raise ValueError("baseline_index must identify a maximum coverage utility")

    if support.shape[1] == 0:
        visual_scores = np.zeros_like(utilities)
        support_sum = np.zeros_like(utilities)
        support_square_sum = np.zeros_like(utilities)
    else:
        visual_scores = 1.0 - np.prod(1.0 - support, axis=1)
        support_sum = np.sum(support, axis=1)
        support_square_sum = np.sum(support * support, axis=1)

    effective_support = np.zeros_like(utilities)
    nonzero_support = support_square_sum > 0.0
    effective_support[nonzero_support] = (
        support_sum[nonzero_support] * support_sum[nonzero_support]
    ) / support_square_sum[nonzero_support]

    # 1 - 1 / n_eff, written to give 0 for a single record
    corroboration = np.zeros_like(utilities)
    positive_mass = support_sum > 0.0
    support_sum_squared = support_sum[positive_mass] * support_sum[positive_mass]
    corroboration[positive_mass] = np.maximum(
        0.0,
        (support_sum_squared - support_square_sum[positive_mass])
        / support_sum_squared,
    )

    visual_top_index: int | None = None
    selectivity = 0.0
    if utilities.size > 1:
        top_visual = float(np.max(visual_scores))
        top_indices = np.flatnonzero(visual_scores == top_visual)
        if top_visual > 0.0 and top_indices.size == 1:
            visual_top_index = int(top_indices[0])
            second_visual = float(
                np.max(np.delete(visual_scores, visual_top_index))
            )
            selectivity = max(
                0.0,
                min(1.0, (top_visual - second_visual) / top_visual),
            )

    # Minimum and geometric-mean trust are logged only
    trust_product = np.zeros_like(utilities)
    trust_minimum = np.zeros_like(utilities)
    trust_geometric = np.zeros_like(utilities)
    if visual_top_index is not None:
        factors = np.asarray(
            [
                visual_scores[visual_top_index],
                corroboration[visual_top_index],
                selectivity,
            ],
            dtype=np.float64,
        )
        if bool(np.all(factors > 0.0)):
            product = float(np.prod(factors))
            trust_product[visual_top_index] = np.clip(product, 0.0, 1.0)
            trust_minimum[visual_top_index] = np.clip(
                float(np.min(factors)), 0.0, 1.0
            )
            trust_geometric[visual_top_index] = np.clip(
                float(product ** (1.0 / factors.size)), 0.0, 1.0
            )
    trust = trust_product
    coverage_regret = np.maximum(0.0, best_utility - utilities)
    component_contrast = np.sum(
        np.abs(components - components[baseline]),
        axis=1,
    )
    allowed_regret = trust * component_contrast
    admissible_array = (trust > 0.0) & (coverage_regret <= allowed_regret)
    admissible_array[baseline] = True

    selected = baseline
    if (
        visual_top_index is not None
        and bool(admissible_array[visual_top_index])
    ):
        selected = int(visual_top_index)

    trust_product_values = tuple(float(value) for value in trust_product)
    trust_minimum_values = tuple(float(value) for value in trust_minimum)
    trust_geometric_values = tuple(float(value) for value in trust_geometric)
    return ConfidenceConstrainedFrontierChoice(
        selected_index=int(selected),
        baseline_index=int(baseline),
        coverage_best=best_utility,
        visual_top_index=visual_top_index,
        visual_scores=tuple(float(value) for value in visual_scores),
        effective_support=tuple(float(value) for value in effective_support),
        corroboration=tuple(float(value) for value in corroboration),
        selectivity=float(selectivity),
        trust=trust_product_values,
        trust_product=trust_product_values,
        trust_minimum=trust_minimum_values,
        trust_geometric=trust_geometric_values,
        coverage_regret=tuple(float(value) for value in coverage_regret),
        component_contrast=tuple(float(value) for value in component_contrast),
        allowed_regret=tuple(float(value) for value in allowed_regret),
        admissible=tuple(bool(value) for value in admissible_array),
    )


class UtilityExplorer(NavigationMapExplorer):
    def __init__(
        self,
        navigation_map,
        *,
        hfov_deg: float,
        max_range_m: float,
        info_gain_weight: float,
        goal_likelihood_weight: float,
        revisit_weight: float,
        information_range_scale: float = 1.0,
        forward_step_size_m: float = 0.25,
        turn_angle_deg: float = 30.0,
        utility_form: str = "rate",
        exchange_rate: float = 0.0,
        exchange_rate_scale: float = 1.0,
        rotation_cost_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(navigation_map, **kwargs)
        if utility_form not in UTILITY_FORMS:
            raise ValueError(f"unknown frontier utility form: {utility_form!r}")
        self.utility_form = str(utility_form)
        self.reference_exchange_rate = float(exchange_rate)
        exchange_rate_scale = float(exchange_rate_scale)
        self.rotation_cost_weight = float(rotation_cost_weight)
        for name, value in (
            ("exchange_rate", self.reference_exchange_rate),
            ("exchange_rate_scale", exchange_rate_scale),
            ("rotation_cost_weight", self.rotation_cost_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            self.utility_form == "difference"
            and exchange_rate_scale > 0.0
            and self.reference_exchange_rate <= 0.0
        ):
            raise ValueError(
                "the difference objective needs a calibrated exchange rate "
                "(sgf_explore_exchange_rate > 0); see "
                "scripts/calibrate_exploration_exchange_rate.py"
            )
        self.exchange_rate = self.reference_exchange_rate * exchange_rate_scale
        self.forward_step_size_m = max(1e-3, float(forward_step_size_m))
        self.turn_angle_deg = min(180.0, max(1e-3, float(turn_angle_deg)))
        self._heading_rad: float | None = None
        self._frontier_bearing_rad: np.ndarray | None = None
        self._frontier_heading_rad: float | None = None
        self.hfov_deg = float(hfov_deg)
        self.max_range_m = float(max_range_m)
        self.info_gain_weight = float(info_gain_weight)
        self.goal_likelihood_weight = float(goal_likelihood_weight)
        self.revisit_weight = float(revisit_weight)
        self.cost_offset_m = max(1e-3, float(max_range_m))
        self.information_range_scale = max(1e-3, float(information_range_scale))
        self.information_range_m = max(
            float(navigation_map.resolution_m),
            float(max_range_m) * self.information_range_scale,
        )
        self._semantic_nodes: list[dict] = []
        self._goal_phrases: list[str] = []
        # Nodes rejected by the verifier, ignored for goal likelihood
        self._excluded_node_ids: set[int] = set()
        self.last_frontier_utility: FrontierUtility | None = None
        self.last_frontier_candidates: tuple[FrontierUtility, ...] = ()
        self.last_frontier_coverage_index: int | None = None
        self._reset_objective_telemetry()

    def reset(self) -> None:
        super().reset()
        self.last_frontier_utility = None
        self.last_frontier_candidates = ()
        self.last_frontier_coverage_index = None
        self._excluded_node_ids = set()
        self._heading_rad = None
        self._frontier_bearing_rad = None
        self._frontier_heading_rad = None
        self._reset_objective_telemetry()

    def _reset_objective_telemetry(self) -> None:
        self.objective_decisions = 0
        self.objective_multi_candidate_decisions = 0
        self.objective_selection_counts: Counter[str] = Counter()
        self.rotation_priced_decisions = 0
        self.heading_unavailable_decisions = 0
        self.shadow_disagreements: Counter[str] = Counter()
        self.exchange_rate_samples: list[float] = []
        self.exchange_rate_samples_translation_only: list[float] = []

    def objective_telemetry(self) -> dict[str, Any]:
        return {
            "utility_form": self.utility_form,
            "exchange_rate": round(float(self.exchange_rate), 6),
            "rotation_cost_weight": float(self.rotation_cost_weight),
            "decisions": int(self.objective_decisions),
            "multi_candidate_decisions": int(
                self.objective_multi_candidate_decisions
            ),
            "selection_counts": {
                key: int(value)
                for key, value in sorted(self.objective_selection_counts.items())
            },
            "rotation_priced_decisions": int(self.rotation_priced_decisions),
            "heading_unavailable_decisions": int(
                self.heading_unavailable_decisions
            ),
            "shadow_disagreements": {
                key: int(value)
                for key, value in sorted(self.shadow_disagreements.items())
            },
            "exchange_rate_samples": [
                round(float(value), 6) for value in self.exchange_rate_samples
            ],
            "exchange_rate_samples_translation_only": [
                round(float(value), 6)
                for value in self.exchange_rate_samples_translation_only
            ],
        }

    def set_heading_rad(self, heading_rad: float | None) -> None:
        """Set the heading (rad) used to price rotation, None for distance only."""

        if heading_rad is None:
            self._heading_rad = None
            return
        value = float(heading_rad)
        self._heading_rad = value if math.isfinite(value) else None

    @staticmethod
    def _candidate_bearings_rad(
        position: np.ndarray,
        candidates: tuple[FrontierUtility, ...],
    ) -> np.ndarray | None:
        if not candidates:
            return None
        goals_xz = np.asarray(
            [
                np.asarray(candidate.goal, dtype=np.float64).reshape(3)[[0, 2]]
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        origin_xz = np.asarray(position, dtype=np.float64).reshape(3)[[0, 2]]
        return frontier_bearings_rad(origin_xz, goals_xz)

    def update_semantic_nodes(self, nodes: list[dict]) -> None:
        self._semantic_nodes = list(nodes)

    def update_target_queries(self, goal_phrases: list[str]) -> None:
        self._goal_phrases = list(goal_phrases)

    def update_excluded_nodes(self, node_ids) -> None:
        self._excluded_node_ids = {int(value) for value in (node_ids or ())}

    def _select_frontier_goal(self, position: np.ndarray) -> np.ndarray | None:
        if self.visited is None:
            self.last_frontier_utility = None
            self.last_frontier_candidates = ()
            self.last_frontier_coverage_index = None
            self._clear_frontier_geometry_cache()
            return None
        cpp_result = self.navigation_map.utility_frontier_goal(
            position=position,
            visited_mask=self.visited,
            failed_goals=self._failed_goals,
            failed_goal_counts=self._failed_goal_counts,
            failed_goal_penalty_scales=self._failed_goal_penalty_scales,
            goal_phrases=self._goal_phrases,
            visited_radius_cells=self.visited_radius_cells,
            min_goal_distance_m=self.min_goal_distance_m,
            min_frontier_size=self.min_frontier_size,
            failed_goal_radius_m=self._failed_goal_radius_m,
            failed_goal_repeat_penalty_count=self._failed_goal_repeat_penalty_count,
            failed_goal_repeat_penalty_scale=self._failed_goal_repeat_penalty_scale,
            hfov_deg=self.hfov_deg,
            max_range_m=self.information_range_m,
            info_gain_weight=self.info_gain_weight,
            goal_likelihood_weight=self.goal_likelihood_weight,
            # Travel cost is applied in Python by frontier_objective
            cost_weight=0.0,
            revisit_weight=self.revisit_weight,
            excluded_node_ids=self._excluded_node_ids,
        )
        if cpp_result is None:
            self.last_frontier_utility = None
            self.last_frontier_candidates = ()
            self.last_frontier_coverage_index = None
            self._clear_frontier_geometry_cache()
            return None

        def as_frontier_utility(result: dict) -> FrontierUtility:
            return FrontierUtility(
                goal=np.asarray(result["goal"], dtype=np.float32),
                utility=float(result["utility"]),
                information_gain=float(result["information_gain"]),
                goal_likelihood=float(result["goal_likelihood"]),
                cost=float(result["cost"]),
                revisit_penalty=float(result["revisit_penalty"]),
                area=int(result["area"]),
            )

        native_winner = as_frontier_utility(cpp_result)
        self.last_frontier_utility = native_winner
        raw_candidates = cpp_result.get("candidates", ())
        candidates = tuple(as_frontier_utility(item) for item in raw_candidates)
        raw_best_index = cpp_result.get("best_index")
        if candidates:
            if raw_best_index is None:
                candidates = ()
            else:
                best_index = int(raw_best_index)
                if best_index < 0 or best_index >= len(candidates):
                    candidates = ()
        if not candidates:
            candidates = (native_winner,)

        self._clear_frontier_geometry_cache()

        def rescore(items: tuple[FrontierUtility, ...]) -> tuple[FrontierUtility, ...]:
            if self._heading_rad is not None:
                bearings = self._candidate_bearings_rad(position, items)
                if bearings is not None and bool(np.all(np.isfinite(bearings))):
                    self._frontier_bearing_rad = bearings
                    self._frontier_heading_rad = float(self._heading_rad)
            utilities, _components = self._objective(
                items,
                form=self.utility_form,
                rotation_cost_weight=self.rotation_cost_weight,
                exchange_rate=self.exchange_rate,
            )
            return tuple(
                FrontierUtility(
                    goal=candidate.goal,
                    utility=float(utility),
                    information_gain=candidate.information_gain,
                    goal_likelihood=candidate.goal_likelihood,
                    cost=candidate.cost,
                    revisit_penalty=candidate.revisit_penalty,
                    area=candidate.area,
                )
                for candidate, utility in zip(items, utilities)
            )

        candidates = rescore(candidates)
        coverage_index = self._candidate_argmax(candidates)
        self._record_objective_decision(candidates, coverage_index)

        self.last_frontier_candidates = candidates
        self.last_frontier_coverage_index = coverage_index
        self.last_frontier_utility = candidates[coverage_index]
        return np.asarray(self.last_frontier_utility.goal, dtype=np.float32)

    @staticmethod
    def _candidate_argmax(candidates: tuple[FrontierUtility, ...]) -> int:
        return int(np.argmax([candidate.utility for candidate in candidates]))

    def _clear_frontier_geometry_cache(self) -> None:
        self._frontier_bearing_rad = None
        self._frontier_heading_rad = None

    def _objective(
        self,
        candidates: tuple[FrontierUtility, ...],
        *,
        form: str,
        rotation_cost_weight: float,
        exchange_rate: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not candidates:
            return np.empty((0,), dtype=np.float64), np.empty(
                (0, 4), dtype=np.float64
            )
        bearings = self._matching_frontier_bearings(candidates)
        heading = self._frontier_heading_rad
        if bearings is None or heading is None:
            bearings = None
            heading = None
        return frontier_objective(
            np.asarray(
                [candidate.information_gain for candidate in candidates],
                dtype=np.float64,
            ),
            np.asarray(
                [candidate.goal_likelihood for candidate in candidates],
                dtype=np.float64,
            ),
            np.asarray([candidate.cost for candidate in candidates], dtype=np.float64),
            np.asarray(
                [candidate.revisit_penalty for candidate in candidates],
                dtype=np.float64,
            ),
            form=form,
            info_gain_weight=self.info_gain_weight,
            goal_likelihood_weight=self.goal_likelihood_weight,
            revisit_weight=self.revisit_weight,
            cost_offset_m=self.cost_offset_m,
            forward_step_size_m=self.forward_step_size_m,
            turn_angle_deg=self.turn_angle_deg,
            bearing_rad=bearings,
            heading_rad=heading,
            rotation_cost_weight=rotation_cost_weight,
            exchange_rate=exchange_rate,
        )

    def _record_objective_decision(
        self,
        candidates: tuple[FrontierUtility, ...],
        coverage_index: int,
    ) -> None:
        self.objective_decisions += 1
        self.objective_selection_counts[self.utility_form] += 1
        heading_known = (
            self._matching_frontier_bearings(candidates) is not None
            and self._frontier_heading_rad is not None
        )
        if not heading_known:
            self.heading_unavailable_decisions += 1
        elif self.rotation_cost_weight > 0.0 and (
            self.utility_form != "difference" or self.exchange_rate > 0.0
        ):
            self.rotation_priced_decisions += 1
        if heading_known:
            for kappa, samples in (
                (1.0, self.exchange_rate_samples),
                (0.0, self.exchange_rate_samples_translation_only),
            ):
                if len(samples) >= MAX_EXCHANGE_RATE_SAMPLES:
                    continue
                rates, _components = self._objective(
                    candidates,
                    form="rate",
                    rotation_cost_weight=kappa,
                    exchange_rate=0.0,
                )
                best_rate = float(np.max(rates))
                if math.isfinite(best_rate) and best_rate > 0.0:
                    samples.append(best_rate)
        if len(candidates) < 2:
            return
        self.objective_multi_candidate_decisions += 1
        shadows: list[tuple[str, str, float, float]] = [
            ("rate", "rate", 1.0, 0.0),
            ("rate_translation_only", "rate", 0.0, 0.0),
            ("nearest", "nearest", 1.0, 0.0),
        ]
        if self.reference_exchange_rate > 0.0:
            shadows.append(
                ("difference", "difference", 1.0, self.reference_exchange_rate)
            )
        for name, form, kappa, rate in shadows:
            utilities, _components = self._objective(
                candidates,
                form=form,
                rotation_cost_weight=kappa,
                exchange_rate=rate,
            )
            if int(np.argmax(utilities)) != int(coverage_index):
                self.shadow_disagreements[name] += 1

    def frontier_utility_components(
        self,
        candidates: tuple[FrontierUtility, ...],
    ) -> np.ndarray:
        """Signed components of the live utilities (rows sum to utilities)."""

        if not candidates:
            return np.empty((0, 4), dtype=np.float64)
        _utilities, components = self._objective(
            candidates,
            form=self.utility_form,
            rotation_cost_weight=self.rotation_cost_weight,
            exchange_rate=self.exchange_rate,
        )
        return components

    def _matching_frontier_bearings(
        self,
        candidates: tuple[FrontierUtility, ...],
    ) -> np.ndarray | None:
        cached = self._frontier_bearing_rad
        if cached is None:
            return None
        vector = np.asarray(cached, dtype=np.float64).reshape(-1)
        return vector if vector.size == len(candidates) else None
