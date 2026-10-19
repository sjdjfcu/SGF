"""Fixed-capacity SE(2) visual memory with leaky-max merging of nearby views."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable


_TWO_PI = 2.0 * math.pi


def _finite_positive(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _step_value(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError("step must be a non-negative integer")
    return int(value)


def _position_xz(value: Iterable[float]) -> tuple[float, float]:
    try:
        values = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("position_xz must contain two finite values") from exc
    if len(values) != 2 or not all(math.isfinite(component) for component in values):
        raise ValueError("position_xz must contain two finite values")
    return values[0], values[1]


def wrap_heading(heading_rad: float) -> float:
    """Wrap a heading to [-pi, pi)."""

    heading = float(heading_rad)
    if not math.isfinite(heading):
        raise ValueError("heading_rad must be finite")
    return (heading + math.pi) % _TWO_PI - math.pi


def heading_distance(first_rad: float, second_rad: float) -> float:
    """Return the shortest unsigned angular distance between two headings."""

    first = wrap_heading(first_rad)
    second = wrap_heading(second_rad)
    return abs((first - second + math.pi) % _TWO_PI - math.pi)


@dataclass(frozen=True, slots=True)
class VisualMemoryConfig:
    """Memory geometry and score thresholds (radius_m only sets the half-life)."""

    map_resolution_m: float
    forward_step_size_m: float
    turn_angle_deg: float
    radius_m: float
    min_similarity: float
    frontier_min_score: float
    max_records: int
    update_period_steps: int = 1

    def __post_init__(self) -> None:
        map_resolution = _finite_positive(
            self.map_resolution_m, name="map_resolution_m"
        )
        forward_step = _finite_positive(
            self.forward_step_size_m, name="forward_step_size_m"
        )
        turn_angle = _finite_positive(self.turn_angle_deg, name="turn_angle_deg")
        if turn_angle > 360.0:
            raise ValueError("turn_angle_deg must not exceed 360 degrees")
        radius = _finite_positive(self.radius_m, name="radius_m")

        min_similarity = float(self.min_similarity)
        if not math.isfinite(min_similarity) or min_similarity >= 1.0:
            raise ValueError("min_similarity must be finite and below 1")
        frontier_min_score = float(self.frontier_min_score)
        if (
            not math.isfinite(frontier_min_score)
            or frontier_min_score < 0.0
            or frontier_min_score > 1.0
        ):
            raise ValueError("frontier_min_score must be in [0, 1]")

        max_records = _positive_integer(self.max_records, name="max_records")
        update_period = _positive_integer(
            self.update_period_steps, name="update_period_steps"
        )

        object.__setattr__(self, "map_resolution_m", map_resolution)
        object.__setattr__(self, "forward_step_size_m", forward_step)
        object.__setattr__(self, "turn_angle_deg", turn_angle)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "min_similarity", min_similarity)
        object.__setattr__(self, "frontier_min_score", frontier_min_score)
        object.__setattr__(self, "max_records", max_records)
        object.__setattr__(self, "update_period_steps", update_period)

    @property
    def association_radius_m(self) -> float:
        """Metric duplicate radius derived from map and action resolution."""

        return max(2.0 * self.map_resolution_m, 0.5 * self.forward_step_size_m)

    @property
    def association_heading_rad(self) -> float:
        """Duplicate heading tolerance: half of one discrete turn."""

        return 0.5 * math.radians(self.turn_angle_deg)

    @property
    def half_life_steps(self) -> int:
        """Evidence half-life derived from refresh and traversal horizons."""

        refresh_horizon = self.max_records * self.update_period_steps
        traversal_horizon = math.ceil(
            2.0 * self.radius_m / self.forward_step_size_m
        ) + math.ceil(360.0 / self.turn_angle_deg)
        return max(refresh_horizon, traversal_horizon)

    def normalize_similarity(self, similarity: float) -> float:
        """Map a raw similarity to a quality in [0, 1]."""

        value = float(similarity)
        if not math.isfinite(value):
            raise ValueError("similarity must be finite")
        normalized = (value - self.min_similarity) / (1.0 - self.min_similarity)
        return min(1.0, max(0.0, normalized))


@dataclass(frozen=True, slots=True)
class VisualMemoryRecord:
    """Immutable effective record at the memory's current step."""

    record_id: int
    position_xz: tuple[float, float]
    heading_rad: float
    effective_quality: float
    stored_quality: float
    first_step: int
    last_step: int
    observation_count: int

    @property
    def x_m(self) -> float:
        return self.position_xz[0]

    @property
    def z_m(self) -> float:
        return self.position_xz[1]


@dataclass(frozen=True, slots=True)
class VisualMemoryStats:
    """Current occupancy, derived scales, and lifetime operation counters."""

    current_step: int | None
    active_records: int
    capacity: int
    association_radius_m: float
    association_heading_rad: float
    half_life_steps: int
    updates: int
    insertions: int
    merges: int
    representative_replacements: int
    rejected_observations: int
    pruned_records: int
    evicted_records: int
    best_effective_quality: float | None


@dataclass(frozen=True, slots=True)
class VisualMemoryUpdate:
    """Result of one visual observation update."""

    normalized_quality: float
    associated_record_id: int | None
    record: VisualMemoryRecord | None
    inserted: bool
    merged: bool
    representative_replaced: bool
    retained: bool
    pruned_record_ids: tuple[int, ...]
    evicted_record_id: int | None
    stats: VisualMemoryStats


@dataclass(slots=True)
class _StoredRecord:
    record_id: int
    position_xz: tuple[float, float]
    heading_rad: float
    stored_quality: float
    first_step: int
    last_step: int
    observation_count: int = 1


class LeakyMaxVisualMemory:
    """Fixed-capacity SE(2) visual memory updated at non-decreasing steps."""

    def __init__(self, config: VisualMemoryConfig) -> None:
        if not isinstance(config, VisualMemoryConfig):
            raise TypeError("config must be a VisualMemoryConfig")
        self.config = config
        self._records: dict[int, _StoredRecord] = {}
        self._current_step: int | None = None
        self._next_record_id = 0
        self._updates = 0
        self._insertions = 0
        self._merges = 0
        self._representative_replacements = 0
        self._rejected_observations = 0
        self._pruned_records = 0
        self._evicted_records = 0

    @property
    def current_step(self) -> int | None:
        return self._current_step

    @property
    def stats(self) -> VisualMemoryStats:
        effective = self.effective_records()
        best = effective[0].effective_quality if effective else None
        return VisualMemoryStats(
            current_step=self._current_step,
            active_records=len(self._records),
            capacity=self.config.max_records,
            association_radius_m=self.config.association_radius_m,
            association_heading_rad=self.config.association_heading_rad,
            half_life_steps=self.config.half_life_steps,
            updates=self._updates,
            insertions=self._insertions,
            merges=self._merges,
            representative_replacements=self._representative_replacements,
            rejected_observations=self._rejected_observations,
            pruned_records=self._pruned_records,
            evicted_records=self._evicted_records,
            best_effective_quality=best,
        )

    def advance(self, step: int) -> VisualMemoryStats:
        """Advance lazy decay and prune records below the frontier threshold."""

        validated_step = self._validate_monotonic_step(step)
        self._advance_validated(validated_step)
        return self.stats

    def update(
        self,
        *,
        position_xz: Iterable[float],
        heading_rad: float,
        similarity: float,
        step: int,
    ) -> VisualMemoryUpdate:
        """Merge the observation into a nearby record or insert a new one."""

        validated_step = self._validate_monotonic_step(step)
        validated_position = _position_xz(position_xz)
        validated_heading = wrap_heading(heading_rad)
        normalized_quality = self.config.normalize_similarity(similarity)

        pruned_ids = self._advance_validated(validated_step)
        self._updates += 1

        associated = self._association_candidate(
            position_xz=validated_position,
            heading_rad=validated_heading,
            step=validated_step,
        )
        inserted = False
        merged = associated is not None
        representative_replaced = False
        evicted_id: int | None = None

        if associated is not None:
            record = self._records[associated]
            previous_effective = self._effective_quality(record, validated_step)
            representative_replaced = normalized_quality > previous_effective
            if representative_replaced:
                record.position_xz = validated_position
                record.heading_rad = validated_heading
                self._representative_replacements += 1
            record.stored_quality = max(normalized_quality, previous_effective)
            record.last_step = validated_step
            record.observation_count += 1
            self._merges += 1
            retained_id: int | None = record.record_id
        elif normalized_quality >= self.config.frontier_min_score:
            retained_id = self._next_record_id
            self._next_record_id += 1
            self._records[retained_id] = _StoredRecord(
                record_id=retained_id,
                position_xz=validated_position,
                heading_rad=validated_heading,
                stored_quality=normalized_quality,
                first_step=validated_step,
                last_step=validated_step,
            )
            inserted = True
            self._insertions += 1
            if len(self._records) > self.config.max_records:
                evicted_id = self._eviction_candidate(validated_step)
                del self._records[evicted_id]
                self._evicted_records += 1
                if evicted_id == retained_id:
                    retained_id = None
        else:
            retained_id = None
            self._rejected_observations += 1

        snapshot = (
            self._snapshot(self._records[retained_id], validated_step)
            if retained_id is not None
            else None
        )
        return VisualMemoryUpdate(
            normalized_quality=normalized_quality,
            associated_record_id=associated,
            record=snapshot,
            inserted=inserted,
            merged=merged,
            representative_replaced=representative_replaced,
            retained=retained_id is not None,
            pruned_record_ids=pruned_ids,
            evicted_record_id=evicted_id,
            stats=self.stats,
        )

    def effective_records(self) -> tuple[VisualMemoryRecord, ...]:
        """Live records sorted by effective quality, decayed to current_step."""

        if self._current_step is None:
            return ()
        snapshots = [
            self._snapshot(record, self._current_step)
            for record in self._records.values()
        ]
        snapshots.sort(
            key=lambda record: (
                -record.effective_quality,
                -record.last_step,
                record.record_id,
            )
        )
        return tuple(snapshots)

    def _validate_monotonic_step(self, step: int) -> int:
        validated = _step_value(step)
        if self._current_step is not None and validated < self._current_step:
            raise ValueError(
                f"step must be non-decreasing (current={self._current_step}, "
                f"received={validated})"
            )
        return validated

    def _advance_validated(self, step: int) -> tuple[int, ...]:
        stale_ids = tuple(
            sorted(
                record.record_id
                for record in self._records.values()
                if self._effective_quality(record, step)
                < self.config.frontier_min_score
            )
        )
        for record_id in stale_ids:
            del self._records[record_id]
        self._pruned_records += len(stale_ids)
        self._current_step = step
        return stale_ids

    def _decay_factor(self, elapsed_steps: int) -> float:
        if elapsed_steps <= 0:
            return 1.0
        return math.exp2(-float(elapsed_steps) / self.config.half_life_steps)

    def _effective_quality(self, record: _StoredRecord, step: int) -> float:
        return record.stored_quality * self._decay_factor(step - record.last_step)

    def _association_candidate(
        self,
        *,
        position_xz: tuple[float, float],
        heading_rad: float,
        step: int,
    ) -> int | None:
        radius = self.config.association_radius_m
        heading_tolerance = self.config.association_heading_rad
        candidates: list[tuple[float, float, float, int]] = []
        for record in self._records.values():
            distance = math.hypot(
                position_xz[0] - record.position_xz[0],
                position_xz[1] - record.position_xz[1],
            )
            angular_distance = heading_distance(heading_rad, record.heading_rad)
            if distance <= radius and angular_distance <= heading_tolerance:
                candidates.append(
                    (
                        distance,
                        angular_distance,
                        -self._effective_quality(record, step),
                        record.record_id,
                    )
                )
        return min(candidates)[-1] if candidates else None

    def _eviction_candidate(self, step: int) -> int:
        return min(
            self._records.values(),
            key=lambda record: (
                self._effective_quality(record, step),
                record.last_step,
                record.record_id,
            ),
        ).record_id

    def _snapshot(self, record: _StoredRecord, step: int) -> VisualMemoryRecord:
        return VisualMemoryRecord(
            record_id=record.record_id,
            position_xz=record.position_xz,
            heading_rad=record.heading_rad,
            effective_quality=self._effective_quality(record, step),
            stored_quality=record.stored_quality,
            first_step=record.first_step,
            last_step=record.last_step,
            observation_count=record.observation_count,
        )


__all__ = [
    "LeakyMaxVisualMemory",
    "VisualMemoryConfig",
    "VisualMemoryRecord",
    "VisualMemoryStats",
    "VisualMemoryUpdate",
    "heading_distance",
    "wrap_heading",
]
