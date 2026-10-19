from __future__ import annotations

import numpy as np


class SGFNavigationMap:
    """Adapter exposing GraphSLAM's C++ navigation map to Python."""

    def __init__(self, sgf) -> None:
        self._sgf = sgf

    def reset(self) -> None:
        self._sgf.reset_navigation_map()

    def begin_provisional(self) -> None:
        """Start a depth-only provisional branch of the navigation map."""

        self._sgf.begin_provisional_navigation_map()

    @property
    def has_provisional(self) -> bool:
        return bool(self._sgf.has_provisional_navigation_map())

    def drop_provisional(self) -> bool:
        """Discard the provisional branch without changing the active map."""

        return bool(self._sgf.drop_provisional_navigation_map())

    def commit_provisional(self) -> bool:
        """Make the provisional branch the active map."""

        return bool(self._sgf.commit_provisional_navigation_map())

    @property
    def graph_occupancy_enabled(self) -> bool:
        return bool(self._sgf.navigation_graph_occupancy_enabled())

    def graph_occupancy_stats(self) -> dict:
        """Surfel votes and mask transitions in the latest graph update."""
        try:
            return dict(self._sgf.navigation_graph_occupancy_stats())
        except AttributeError as exc:
            raise RuntimeError(
                "representation diagnostics require the _sgf extension built from "
                "this source tree (python3 tools/build_extension.py)"
            ) from exc

    def set_floor_support(self, enabled: bool) -> None:
        """Set the ground-support veto and check that the extension applied it."""
        try:
            self._sgf.set_navigation_floor_support(bool(enabled))
            actual = bool(self._sgf.navigation_floor_support_enabled())
        except AttributeError as exc:
            raise RuntimeError(
                "navigation_floor_support requires the _sgf extension built from "
                "this source tree (python3 tools/build_extension.py)"
            ) from exc
        if actual != bool(enabled):
            raise RuntimeError("native navigation_floor_support setting was not applied")

    def floor_support_stats(self) -> dict:
        """Support-veto cell counts for the current active navigation map."""
        return dict(self._sgf.navigation_floor_support_stats())

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self._sgf.navigation_shape()
        return int(rows), int(cols)

    @property
    def resolution_m(self) -> float:
        return float(self._sgf.navigation_resolution_m())

    @property
    def origin_xz(self) -> np.ndarray:
        return np.asarray(self._sgf.navigation_origin_xz(), dtype=np.float32)

    def world_to_cell(self, xyz: np.ndarray) -> tuple[int, int]:
        row, col = self._sgf.navigation_world_to_cell(np.asarray(xyz, dtype=np.float32))
        return int(row), int(col)

    def cell_to_world(self, row: int, col: int, y: float) -> np.ndarray:
        return np.asarray(self._sgf.navigation_cell_to_world(int(row), int(col), float(y)), dtype=np.float32)

    def observed_mask(self) -> np.ndarray:
        return np.asarray(self._sgf.navigation_observed_mask()) > 0

    def occupied_mask(self) -> np.ndarray:
        return np.asarray(self._sgf.navigation_occupied_mask()) > 0

    def free_mask(self) -> np.ndarray:
        return np.asarray(self._sgf.navigation_free_mask()) > 0

    def unknown_mask(self) -> np.ndarray:
        return ~self.observed_mask()

    def frontier_cells(self) -> np.ndarray:
        return np.asarray(self._sgf.navigation_frontier_cells()) > 0

    def snap_to_free(self, xyz: np.ndarray, max_radius_m: float = 2.0) -> np.ndarray | None:
        result = self._sgf.snap_to_free(np.asarray(xyz, dtype=np.float32), float(max_radius_m))
        if result is None:
            return None
        return np.asarray(result, dtype=np.float32)

    @staticmethod
    def _vec3_array(values) -> np.ndarray:
        array = np.asarray(list(values), dtype=np.float32)
        if array.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return array.reshape(-1, 3)

    def snap_to_free_many(self, xyzs: np.ndarray, max_radius_m: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        result = self._sgf.snap_to_free_many(self._vec3_array(xyzs), float(max_radius_m))
        return np.asarray(result["points_m"], dtype=np.float32), np.asarray(result["valid"], dtype=bool)

    def geodesic_distance(self, start_xyz: np.ndarray, goal_xyz: np.ndarray) -> float:
        return float(
            self._sgf.navigation_geodesic_distance(
                np.asarray(start_xyz, dtype=np.float32),
                np.asarray(goal_xyz, dtype=np.float32),
            )
        )

    def geodesic_distances(self, start_xyz: np.ndarray, goal_xyzs: np.ndarray) -> np.ndarray:
        goals = self._vec3_array(goal_xyzs)
        return np.asarray(
            self._sgf.navigation_geodesic_distances(np.asarray(start_xyz, dtype=np.float32), goals),
            dtype=np.float32,
        )

    def frontier_goal(
        self,
        position: np.ndarray,
        min_goal_distance_m: float = 1.5,
        min_frontier_size: int = 20,
        prefer_far: bool = False,
    ) -> np.ndarray | None:
        result = self._sgf.navigation_frontier_goal(
            np.asarray(position, dtype=np.float32),
            float(min_goal_distance_m),
            int(min_frontier_size),
            bool(prefer_far),
        )
        if result is None:
            return None
        return np.asarray(result, dtype=np.float32)

    def all_nodes_debug(self) -> list[dict]:
        """Every labelled node (diagnostic dump only)."""
        return [dict(item) for item in self._sgf.get_nodes(True, 0)]

    def set_query_conditional_labels(self, enabled: bool) -> None:
        """Match candidates by class posterior instead of the argmax label."""
        self._sgf.set_query_conditional_labels(bool(enabled))

    def set_target_association(self, mode: str) -> None:
        """Set "navigation_map", "reachable_euclidean" or "euclidean" association."""
        self._sgf.set_target_association(str(mode))

    @property
    def target_association(self) -> str:
        return str(self._sgf.target_association())

    def semantic_label_summary(
        self,
        goal_phrases: list[str],
    ) -> dict:
        return dict(
            self._sgf.get_semantic_label_summary(
                list(goal_phrases),
            )
        )

    def select_target_node(
        self,
        *,
        agent_position: np.ndarray,
        goal_phrases: list[str],
        min_surfels: int,
        min_confidence: float,
        excluded_node_ids: set[int],
        excluded_positions: list[np.ndarray],
        exclusion_radius_m: float,
    ) -> tuple[dict | None, dict[str, int], list[dict]] | None:
        result = self._sgf.select_target_node(
            np.asarray(agent_position, dtype=np.float32),
            list(goal_phrases),
            int(min_surfels),
            float(min_confidence),
            sorted(int(value) for value in excluded_node_ids),
            self._vec3_array(excluded_positions),
            float(exclusion_radius_m),
        )
        node, diagnostics = result[0], result[1]
        candidates = [dict(item) for item in result[2]] if len(result) > 2 else []
        return (
            (None if node is None else dict(node)),
            {str(key): int(value) for key, value in dict(diagnostics).items()},
            candidates,
        )

    def select_hint_node(
        self,
        *,
        agent_position: np.ndarray,
        goal_phrases: list[str],
        min_surfels: int,
        min_confidence: float,
        excluded_node_ids: set[int],
        excluded_positions: list[np.ndarray],
        exclusion_radius_m: float,
    ) -> dict | None:
        node = self._sgf.select_hint_node(
            np.asarray(agent_position, dtype=np.float32),
            list(goal_phrases),
            int(min_surfels),
            float(min_confidence),
            sorted(int(value) for value in excluded_node_ids),
            self._vec3_array(excluded_positions),
            float(exclusion_radius_m),
        )
        return None if node is None else dict(node)

    def utility_frontier_goal(
        self,
        *,
        position: np.ndarray,
        visited_mask: np.ndarray,
        failed_goals: list[np.ndarray],
        failed_goal_counts: list[int],
        failed_goal_penalty_scales: list[float],
        goal_phrases: list[str],
        visited_radius_cells: int,
        min_goal_distance_m: float,
        min_frontier_size: int,
        failed_goal_radius_m: float,
        failed_goal_repeat_penalty_count: int,
        failed_goal_repeat_penalty_scale: float,
        hfov_deg: float,
        max_range_m: float,
        info_gain_weight: float,
        goal_likelihood_weight: float,
        cost_weight: float,
        revisit_weight: float,
        excluded_node_ids: set[int] | None = None,
    ) -> dict | None:
        result = self._sgf.navigation_utility_frontier_goal(
            np.asarray(position, dtype=np.float32),
            np.asarray(visited_mask, dtype=np.uint8),
            self._vec3_array(failed_goals),
            [int(value) for value in failed_goal_counts],
            [float(value) for value in failed_goal_penalty_scales],
            list(goal_phrases),
            int(visited_radius_cells),
            float(min_goal_distance_m),
            int(min_frontier_size),
            float(failed_goal_radius_m),
            int(failed_goal_repeat_penalty_count),
            float(failed_goal_repeat_penalty_scale),
            float(hfov_deg),
            float(max_range_m),
            float(info_gain_weight),
            float(goal_likelihood_weight),
            float(cost_weight),
            float(revisit_weight),
            sorted(int(value) for value in (excluded_node_ids or ())),
        )
        if result is None:
            return None
        result = dict(result)
        result["goal"] = np.asarray(result["goal"], dtype=np.float32)
        candidates: list[dict] = []
        for raw_candidate in result.get("candidates", ()):
            candidate = dict(raw_candidate)
            candidate["goal"] = np.asarray(candidate["goal"], dtype=np.float32)
            candidates.append(candidate)
        if candidates:
            result["candidates"] = candidates
            result["best_index"] = int(result["best_index"])
        return result

    def select_short_term_goal(
        self,
        *,
        agent_position: np.ndarray,
        forward_xz: np.ndarray,
        agent_cell: tuple[int, int],
        fmm_dist: np.ndarray,
        traversible: np.ndarray,
        step_cells: int,
    ) -> dict | None:
        result = self._sgf.select_short_term_goal(
            np.asarray(agent_position, dtype=np.float32),
            np.asarray(forward_xz, dtype=np.float32).reshape(-1)[:2],
            int(agent_cell[0]),
            int(agent_cell[1]),
            np.asarray(fmm_dist, dtype=np.float32),
            np.asarray(traversible, dtype=np.uint8),
            int(step_cells),
        )
        if result is None:
            return None
        result = dict(result)
        result["waypoint"] = np.asarray(result["waypoint"], dtype=np.float32)
        return result

    def render(self, mode: str = "occupancy") -> np.ndarray:
        _ = mode
        return np.asarray(self._sgf.render_navigation_map(), dtype=np.uint8)
