"""Log-odds combination of appearance and context verdicts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

_PROBABILITY_FLOOR = 0.01


def _clamp_unit(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _probability(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return _clamp_unit(number)
    return None


def logit(probability: float) -> float:
    bounded = min(
        1.0 - _PROBABILITY_FLOOR, max(_PROBABILITY_FLOOR, float(probability))
    )
    return math.log(bounded / (1.0 - bounded))


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def context_observability(*, area_fraction: float, containment: float) -> float:
    """Return the share of the reviewed frame available to show surroundings."""

    try:
        area = _clamp_unit(float(area_fraction))
        covered = _clamp_unit(float(containment))
    except (TypeError, ValueError):
        return 0.0
    return _clamp_unit((1.0 - area) * covered)


@dataclass(frozen=True)
class FactoredVerdict:
    """Combined verdict and its intermediate quantities."""

    appearance_p: float | None
    context_p: float | None
    context_observability: float
    combined_log_odds: float
    p_yes: float
    approved: bool
    factored: bool


def combine_factored_verdict(
    payload: dict[str, Any],
    *,
    pooled_p_yes: float,
    pooled_approved: bool,
    area_fraction: float,
    containment: float,
) -> FactoredVerdict:
    """Combine appearance and context judgments into one verdict."""

    appearance = _probability(
        payload, "appearance_match", "appearance", "object_match", "p_appearance"
    )
    context = _probability(
        payload, "context_match", "context", "surroundings_match", "p_context"
    )
    weight = context_observability(
        area_fraction=area_fraction, containment=containment
    )
    if appearance is None:
        return FactoredVerdict(
            appearance_p=None,
            context_p=context,
            context_observability=weight,
            combined_log_odds=logit(pooled_p_yes),
            p_yes=_clamp_unit(pooled_p_yes),
            approved=bool(pooled_approved),
            factored=False,
        )
    combined = logit(appearance)
    if context is not None:
        combined += weight * logit(context)
    probability = sigmoid(combined)
    return FactoredVerdict(
        appearance_p=appearance,
        context_p=context,
        context_observability=weight,
        combined_log_odds=combined,
        p_yes=_clamp_unit(probability),
        approved=bool(combined > 0.0),
        factored=True,
    )


__all__ = [
    "FactoredVerdict",
    "combine_factored_verdict",
    "context_observability",
    "logit",
    "sigmoid",
]
