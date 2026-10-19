"""Public SGF Habitat API with dependency-lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FMMGoalNavigator",
    "GoalParserConfig",
    "NavigationCommand",
    "OnlineSGF",
    "ParsedGoal",
    "SGFNavigationMap",
    "canonicalize_text_goal",
    "compose_textnav_goal_text",
    "compute_heading_error_deg",
    "detector_prompt_for_goal",
    "label_in_query_set",
    "parse_text_goal",
]


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "FMMGoalNavigator": (".navigation.fmm_navigator", "FMMGoalNavigator"),
    "GoalParserConfig": (".goal.llm_parser", "GoalParserConfig"),
    "NavigationCommand": (".navigation.fmm_navigator", "NavigationCommand"),
    "OnlineSGF": ("._sgf", "OnlineSGF"),
    "ParsedGoal": (".goal.llm_parser", "ParsedGoal"),
    "SGFNavigationMap": (".navigation.sgf_map", "SGFNavigationMap"),
    "canonicalize_text_goal": (".goal.grounding", "canonicalize_text_goal"),
    "compose_textnav_goal_text": (".goal.grounding", "compose_textnav_goal_text"),
    "compute_heading_error_deg": (
        ".navigation.fmm_navigator",
        "compute_heading_error_deg",
    ),
    "detector_prompt_for_goal": (".goal.grounding", "detector_prompt_for_goal"),
    "label_in_query_set": (".goal.grounding", "label_in_query_set"),
    "parse_text_goal": (".goal.llm_parser", "parse_text_goal"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    try:
        value = getattr(import_module(module_name, __name__), attribute)
    except ImportError as exc:
        if name == "OnlineSGF":
            raise ImportError(
                "The SGF Python extension is not built. Run "
                "`python3 tools/build_extension.py` first."
            ) from exc
        raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
