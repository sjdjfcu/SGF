"""Exclusive wall-clock timing for one navigation episode."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from time import perf_counter
from typing import Any


MODULE_NAMES = (
    "sgf_update",
    "image_goal",
    "planning_control",
    "stop_verification",
    "environment_step",
    "diagnostics",
    "other",
)

SGF_STAGE_KEYS = {
    "inseg": "inseg_ms",
    "connectivity": "connectivity_ms",
    "detection_wait_and_fusion": "yolo_ms",
    "navigation_free": "navigation_free_ms",
    "navigation_occupied": "navigation_occupied_ms",
}

# nullcontext has no per-entry state, so nested disabled scopes can reuse it.
_DISABLED_SCOPE = nullcontext()


@dataclass
class _DetailStats:
    total_seconds: float = 0.0
    self_seconds: float = 0.0
    calls: int = 0
    children: dict[str, _DetailStats] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "total_seconds": self.total_seconds,
            "self_seconds": self.self_seconds,
            "calls": self.calls,
            "mean_call_seconds": self.total_seconds / self.calls if self.calls else None,
            "breakdown": {name: child.snapshot() for name, child in self.children.items()},
        }


def detail_scope(timing: RuntimeTiming | None, name: str) -> AbstractContextManager[None]:
    """Allow helpers to opt into the caller's timing without global state."""
    return _DISABLED_SCOPE if timing is None else timing.detail(name)


class RuntimeTiming:
    """Per-module timing that does not double-count nested scopes."""

    def __init__(
        self, clock: Callable[[], float] = perf_counter, *, enabled: bool = False
    ) -> None:
        self.enabled = enabled
        if not enabled:
            return
        self._clock = clock
        self._episode_started_at = clock()
        self._navigation_started_at: float | None = None
        self._navigation_finished_at: float | None = None
        self._last_at: float | None = None
        self._initialization_seconds: float | None = None
        self._navigation_seconds: float | None = None
        self._steps: int | None = None
        self._stack: list[tuple[object, str, tuple[_DetailStats, ...]]] = []
        self._step_token: object | None = None
        self._module_seconds = dict.fromkeys(MODULE_NAMES, 0.0)
        self._module_calls = dict.fromkeys(MODULE_NAMES, 0)
        self._module_self_seconds = dict.fromkeys(MODULE_NAMES, 0.0)
        self._details: dict[str, dict[str, _DetailStats]] = {
            name: {} for name in MODULE_NAMES
        }
        self._sgf_seconds = dict.fromkeys(SGF_STAGE_KEYS, 0.0)
        self._sgf_frames = dict.fromkeys(SGF_STAGE_KEYS, 0)

    @property
    def is_navigating(self) -> bool:
        return (
            self.enabled
            and self._navigation_started_at is not None
            and self._navigation_finished_at is None
        )

    def start_navigation(self) -> None:
        if not self.enabled:
            return
        if self._navigation_started_at is not None:
            raise RuntimeError("Navigation timing has already started")
        now = self._clock()
        self._initialization_seconds = now - self._episode_started_at
        self._navigation_started_at = now
        self._last_at = now

    def _account_until(self, now: float) -> None:
        assert self._last_at is not None
        module = self._stack[-1][1] if self._stack else "other"
        elapsed = now - self._last_at
        self._module_seconds[module] += elapsed
        ancestry = self._stack[-1][2] if self._stack else ()
        for detail in ancestry:
            detail.total_seconds += elapsed
        if ancestry:
            ancestry[-1].self_seconds += elapsed
        else:
            self._module_self_seconds[module] += elapsed
        self._last_at = now

    def _enter(self, module: str, detail: str | None = None) -> object:
        self._account_until(self._clock())
        token = object()
        ancestry: tuple[_DetailStats, ...] = ()
        if detail is not None:
            stats = self._details[module].setdefault(detail, _DetailStats())
            stats.calls += 1
            ancestry = (stats,)
        self._stack.append((token, module, ancestry))
        self._module_calls[module] += 1
        return token

    def _exit(self, token: object) -> None:
        # finish_navigation may have closed an enclosing scope already.
        if not self.is_navigating:
            return
        if not self._stack or self._stack[-1][0] is not token:
            raise RuntimeError("Timing scopes must close in reverse entry order")
        self._account_until(self._clock())
        self._stack.pop()

    def start_step(self) -> None:
        if not self.enabled:
            return
        if not self.is_navigating:
            raise RuntimeError("Navigation timing is not running")
        if self._step_token is not None or self._stack:
            raise RuntimeError("A step must start outside other timing scopes")
        self._step_token = self._enter("planning_control")

    def finish_step(self) -> None:
        if not self.enabled:
            return
        if self._step_token is not None:
            self._exit(self._step_token)
            self._step_token = None

    def measure(
        self, module: str, detail: str | None = None
    ) -> AbstractContextManager[None]:
        if not self.enabled:
            return _DISABLED_SCOPE
        return self._measure(module, detail)

    @contextmanager
    def _measure(self, module: str, detail: str | None = None) -> Iterator[None]:
        if module not in self._module_seconds or module == "other":
            raise ValueError(f"Unknown or reserved timing module: {module!r}")
        if not self.is_navigating:
            yield
            return
        token = self._enter(module, detail)
        try:
            yield
        finally:
            self._exit(token)

    def detail(self, name: str) -> AbstractContextManager[None]:
        if not self.enabled:
            return _DISABLED_SCOPE
        return self._detail(name)

    @contextmanager
    def _detail(self, name: str) -> Iterator[None]:
        if not self.is_navigating:
            yield
            return
        if not self._stack:
            raise RuntimeError("A detail must run inside a module timing scope")
        self._account_until(self._clock())
        _, module, ancestry = self._stack[-1]
        children = ancestry[-1].children if ancestry else self._details[module]
        stats = children.setdefault(name, _DetailStats())
        stats.calls += 1
        token = object()
        self._stack.append((token, module, (*ancestry, stats)))
        try:
            yield
        finally:
            self._exit(token)

    def record_sgf_stats(self, stats: Mapping[str, Any] | None) -> None:
        """Accumulate native SGF stage times (ms), which overlap sgf_update."""
        if not self.is_navigating or not isinstance(stats, Mapping):
            return
        for stage, key in SGF_STAGE_KEYS.items():
            value = stats.get(key)
            if isinstance(value, bool) or not isinstance(value, Real):
                continue
            try:
                seconds = float(value) / 1000.0
            except (OverflowError, ValueError):
                continue
            if not isfinite(seconds) or seconds < 0.0:
                continue
            self._sgf_seconds[stage] += seconds
            self._sgf_frames[stage] += 1

    def finish_navigation(self, steps: int) -> dict[str, Any] | None:
        """Stop the navigation clock and return a JSON-compatible snapshot."""
        if not self.enabled:
            return None
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if self._navigation_started_at is None:
            raise RuntimeError("Navigation timing has not started")
        if self._navigation_finished_at is None:
            now = self._clock()
            self._account_until(now)
            self._navigation_finished_at = now
            self._navigation_seconds = now - self._navigation_started_at
            self._steps = steps
            self._stack.clear()
            self._step_token = None
            self._module_calls["other"] = int(self._module_seconds["other"] > 0.0)
        elif self._steps != steps:
            raise ValueError("Cannot change steps after navigation timing finishes")

        assert self._navigation_seconds is not None
        return {
            "schema_version": 1,
            "initialization_seconds": self._initialization_seconds,
            "navigation_seconds": self._navigation_seconds,
            "steps": steps,
            "mean_step_seconds": self._navigation_seconds / steps if steps else None,
            "modules": {
                name: {
                    "total_seconds": self._module_seconds[name],
                    "self_seconds": self._module_self_seconds[name],
                    "calls": self._module_calls[name],
                    "mean_call_seconds": (
                        self._module_seconds[name] / self._module_calls[name]
                        if self._module_calls[name]
                        else None
                    ),
                    "breakdown": {
                        detail: stats.snapshot()
                        for detail, stats in self._details[name].items()
                    },
                }
                for name in MODULE_NAMES
            },
            "sgf_breakdown": {
                name: {
                    "total_seconds": self._sgf_seconds[name],
                    "frames": self._sgf_frames[name],
                    "mean_frame_seconds": (
                        self._sgf_seconds[name] / self._sgf_frames[name]
                        if self._sgf_frames[name]
                        else None
                    ),
                }
                for name in SGF_STAGE_KEYS
            },
        }
