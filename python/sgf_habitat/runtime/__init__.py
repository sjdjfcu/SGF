from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "NavigationResult",
    "main",
    "parse_args",
    "run_episode",
]


_LAZY_EXPORTS = {
    name: (".episode_loop", name)
    for name in __all__
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
