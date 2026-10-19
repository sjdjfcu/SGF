"""Read-only geometry diagnostics on the candidates available at one step."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def association_rank_diagnostics(
    candidates: Sequence[dict[str, Any]], agent_position: Sequence[float]
) -> dict[str, int]:
    """Whether geodesic and planar distance disagree on the nearest candidate."""
    distances = []
    for candidate in candidates:
        distance = float(candidate["distance_m"])
        if not math.isfinite(distance) or distance < 0:
            continue
        position = candidate["goal_position"]
        planar = math.hypot(
            float(position[0]) - float(agent_position[0]),
            float(position[2]) - float(agent_position[2]),
        )
        if math.isfinite(planar):
            distances.append((distance, planar))
    opportunity = len(distances) >= 2
    disagreement = opportunity and (
        min(range(len(distances)), key=lambda i: distances[i][0])
        != min(range(len(distances)), key=lambda i: distances[i][1])
    )
    return {
        "text_association_rank_opportunities": int(opportunity),
        "text_association_rank_disagreements": int(disagreement),
    }
