"""Weighted fair scheduling of goal epochs between frontier experts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral


class GoalExpert(str, Enum):
    SAFE = "safe"
    VISUAL = "visual"


@dataclass(frozen=True)
class GoalEpochLease:
    token: int
    owner: GoalExpert
    tied_at_nomination: bool
    counts_toward_service: bool
    acknowledged_actions: int
    last_action_id: int | None


@dataclass(frozen=True)
class FairGoalSchedulerSnapshot:
    safe_service: int
    visual_service: int
    safe_weight: int
    visual_weight: int
    next_token: int
    next_tie_owner: GoalExpert
    last_acknowledged_action_id: int | None
    last_resolved_action_id: int | None
    active: GoalEpochLease | None


class TransactionalFairGoalEpochScheduler:
    """Weighted least-normalized-service scheduler at epoch boundaries."""

    def __init__(self, *, safe_weight: int = 1, visual_weight: int = 1) -> None:
        self._default_safe_weight = self._positive_weight(
            safe_weight,
            name="safe_weight",
        )
        self._default_visual_weight = self._positive_weight(
            visual_weight,
            name="visual_weight",
        )
        # Not reset per episode, so a stale acknowledgement cannot match a new lease
        self._next_token = 1
        self.reset()

    @staticmethod
    def _positive_weight(value: int, *, name: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
        ):
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def reset(self) -> None:
        self._safe_weight = int(self._default_safe_weight)
        self._visual_weight = int(self._default_visual_weight)
        self._safe_service = 0
        self._visual_service = 0
        self._next_tie_owner = GoalExpert.SAFE
        self._last_acknowledged_action_id: int | None = None
        self._last_resolved_action_id: int | None = None
        self._active: GoalEpochLease | None = None

    @property
    def active(self) -> GoalEpochLease | None:
        return self._active

    @property
    def snapshot(self) -> FairGoalSchedulerSnapshot:
        return FairGoalSchedulerSnapshot(
            safe_service=int(self._safe_service),
            visual_service=int(self._visual_service),
            safe_weight=int(self._safe_weight),
            visual_weight=int(self._visual_weight),
            next_token=int(self._next_token),
            next_tie_owner=self._next_tie_owner,
            last_acknowledged_action_id=self._last_acknowledged_action_id,
            last_resolved_action_id=self._last_resolved_action_id,
            active=self._active,
        )

    def peek_disputed_owner(
        self,
        *,
        safe_weight: int | None = None,
        visual_weight: int | None = None,
    ) -> GoalExpert:
        """Owner the next disputed epoch would get, optionally under other weights."""

        if self._active is not None:
            raise ValueError("cannot peek disputed ownership with an active lease")
        safe = (
            int(self._safe_weight)
            if safe_weight is None
            else self._positive_weight(safe_weight, name="safe_weight")
        )
        visual = (
            int(self._visual_weight)
            if visual_weight is None
            else self._positive_weight(visual_weight, name="visual_weight")
        )
        safe_pressure = self._safe_service * visual
        visual_pressure = self._visual_service * safe
        if safe_pressure < visual_pressure:
            return GoalExpert.SAFE
        if visual_pressure < safe_pressure:
            return GoalExpert.VISUAL
        return self._next_tie_owner

    def set_weights(self, *, safe_weight: int, visual_weight: int) -> None:
        """Set the weights without resetting accumulated service."""

        if self._active is not None:
            raise ValueError("cannot change weights with an active lease")
        safe = self._positive_weight(safe_weight, name="safe_weight")
        visual = self._positive_weight(visual_weight, name="visual_weight")
        self._safe_weight = safe
        self._visual_weight = visual

    def open_disputed_epoch(self) -> GoalEpochLease:
        """Open a disputed epoch for the least-served owner (idempotent while open)."""

        if self._active is not None:
            return self._active

        safe_pressure = self._safe_service * self._visual_weight
        visual_pressure = self._visual_service * self._safe_weight
        tied = safe_pressure == visual_pressure
        owner = self.peek_disputed_owner()

        lease = GoalEpochLease(
            token=int(self._next_token),
            owner=owner,
            tied_at_nomination=bool(tied),
            counts_toward_service=True,
            acknowledged_actions=0,
            last_action_id=None,
        )
        self._next_token += 1
        # Next tie goes to the other owner, even if this epoch never executes
        self._next_tie_owner = (
            GoalExpert.VISUAL
            if owner is GoalExpert.SAFE
            else GoalExpert.SAFE
        )
        self._active = lease
        return lease

    def open_owned_epoch(
        self,
        owner: GoalExpert,
        *,
        counts_toward_service: bool = True,
    ) -> GoalEpochLease:
        """Open an epoch for an owner chosen by the caller."""

        requested = GoalExpert(owner)
        counted = bool(counts_toward_service)
        if self._active is not None:
            if (
                self._active.owner is requested
                and self._active.counts_toward_service == counted
            ):
                return self._active
            raise ValueError("active goal epoch has different semantics")

        lease = GoalEpochLease(
            token=int(self._next_token),
            owner=requested,
            tied_at_nomination=False,
            counts_toward_service=counted,
            acknowledged_actions=0,
            last_action_id=None,
        )
        self._next_token += 1
        self._active = lease
        return lease

    def acknowledge(
        self,
        token: int,
        action_id: int,
        *,
        executed: bool = True,
    ) -> bool:
        """Count one executed action for the live lease, ignoring stale action ids."""

        active = self._active
        action_sequence = int(action_id)
        if (
            active is None
            or int(token) != int(active.token)
            or (
                self._last_resolved_action_id is not None
                and action_sequence <= self._last_resolved_action_id
            )
        ):
            return False
        self._last_resolved_action_id = action_sequence
        if not bool(executed):
            return False

        if active.counts_toward_service:
            if active.owner is GoalExpert.SAFE:
                self._safe_service += 1
            else:
                self._visual_service += 1

        self._last_acknowledged_action_id = action_sequence
        self._active = GoalEpochLease(
            token=active.token,
            owner=active.owner,
            tied_at_nomination=active.tied_at_nomination,
            counts_toward_service=active.counts_toward_service,
            acknowledged_actions=active.acknowledged_actions + 1,
            last_action_id=action_sequence,
        )
        return True

    def close(self, token: int | None = None) -> bool:
        """Close a live epoch without changing committed service counters."""

        active = self._active
        if active is None:
            return False
        if token is not None and int(token) != int(active.token):
            return False
        self._active = None
        return True


__all__ = [
    "FairGoalSchedulerSnapshot",
    "GoalEpochLease",
    "GoalExpert",
    "TransactionalFairGoalEpochScheduler",
]
