from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .grounding import (
    dedupe_goal_phrases,
    detector_prompt_for_goal,
    normalize_phrase,
)

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class GoalParserConfig:
    config_path: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 20.0
    max_phrases: int = 4
    mode: str = "llm"


@dataclass
class ParsedGoal:
    canonical_goal: str
    target_phrases: list[str]
    # Nearby objects from the instruction, used only as search cues
    context_phrases: list[str] = field(default_factory=list)


def _load_config_file(config_path: str) -> dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"} and yaml is not None:
        loaded = yaml.safe_load(text)
        return loaded if isinstance(loaded, dict) else {}
    loaded = json.loads(text)
    return loaded if isinstance(loaded, dict) else {}


LLM_API_KEY_ENV = "SGF_LLM_API_KEY"
LLM_BASE_URL_ENV = "SGF_LLM_BASE_URL"

_UNAUTHENTICATED_ENDPOINTS_WARNED: set[str] = set()


def resolve_endpoint_field(
    explicit: Any,
    file_value: Any,
    env_name: str,
) -> str:
    """Resolve a field from the caller, the environment, then the file."""

    for candidate in (explicit, os.environ.get(env_name, ""), file_value):
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def warn_once_if_unauthenticated(
    base_url: str,
    api_key: str,
    *,
    env_name: str = LLM_API_KEY_ENV,
) -> bool:
    """Warn once per endpoint that has no API key."""

    if not base_url or api_key:
        return False
    if base_url in _UNAUTHENTICATED_ENDPOINTS_WARNED:
        return False
    _UNAUTHENTICATED_ENDPOINTS_WARNED.add(base_url)
    print(
        f"warning: no API key resolved for {base_url}; "
        f"export {env_name} if that endpoint requires one",
        file=sys.stderr,
    )
    return True


def resolve_goal_parser_config(config: GoalParserConfig) -> GoalParserConfig:
    loaded = _load_config_file(config.config_path)
    base_url = resolve_endpoint_field(
        config.base_url, loaded.get("base_url", ""), LLM_BASE_URL_ENV
    )
    api_key = resolve_endpoint_field(
        config.api_key, loaded.get("api_key", ""), LLM_API_KEY_ENV
    )
    warn_once_if_unauthenticated(base_url, api_key)
    model = str(config.model or loaded.get("llm_model", "")).strip()
    return GoalParserConfig(
        config_path=config.config_path,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=config.timeout_seconds,
        max_phrases=config.max_phrases,
        mode=config.mode,
    )


def _extract_text_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return ""


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        loaded = json.loads(stripped)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        snippet = stripped[start : end + 1]
        loaded = json.loads(snippet)
        return loaded if isinstance(loaded, dict) else {}
    return {}


def _call_chat_completions(config: GoalParserConfig, prompt: str) -> str:
    url = config.base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract target detector queries for indoor navigation. "
                    "Return JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    response = requests.post(url, headers=headers, json=payload, timeout=config.timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    return _extract_text_content(content).strip()


_GOAL_PARSE_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "goal_parse_cache"
_GOAL_PARSE_CACHE_VERSION = "v1"


def _cache_key(text_goal: str, model: str, max_phrases: int) -> str:
    import hashlib

    raw = json.dumps(
        {
            "version": _GOAL_PARSE_CACHE_VERSION,
            "text_goal": text_goal,
            "model": model,
            "max_phrases": max_phrases,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _phrase_list(
    payload: dict[str, Any],
    field: str,
    *,
    required: bool = False,
    max_items: int | None = None,
) -> list[str]:
    if field not in payload and not required:
        return []
    value = payload.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RuntimeError(f"LLM field {field!r} must be a list of strings")
    if max_items is not None and len(value) > max_items:
        raise RuntimeError(f"LLM field {field!r} must contain at most {max_items} items")
    if any(not normalize_phrase(item) for item in value):
        raise RuntimeError(f"LLM field {field!r} must contain non-empty phrases")
    phrases = dedupe_goal_phrases(value)
    if required and not phrases:
        raise RuntimeError(f"LLM returned no {field.replace('_', ' ')}")
    return phrases


def _parsed_payload(
    payload: dict[str, Any],
    *,
    max_phrases: int,
) -> tuple[str, list[str], list[str]]:
    raw_canonical_goal = payload.get("canonical_goal")
    if not isinstance(raw_canonical_goal, str):
        raise RuntimeError("LLM field 'canonical_goal' must be a string")
    canonical_goal = detector_prompt_for_goal(raw_canonical_goal)
    if not canonical_goal:
        raise RuntimeError("LLM returned no canonical goal")

    target_phrases = _phrase_list(
        payload,
        "target_phrases",
        required=True,
        max_items=max_phrases,
    )
    context_phrases: list[str] = []
    if isinstance(payload.get("context_phrases"), list):
        try:
            context_phrases = _phrase_list(
                payload,
                "context_phrases",
                required=False,
                max_items=max_phrases,
            )
        except RuntimeError:
            # Context phrases are optional, drop a malformed list
            context_phrases = []
    target_set = set(target_phrases)
    target_set.add(canonical_goal)
    context_phrases = [phrase for phrase in context_phrases if phrase not in target_set]
    return canonical_goal, target_phrases, context_phrases


def _load_cached_goal(
    cache_file: Path,
    *,
    text_goal: str,
    config: GoalParserConfig,
) -> ParsedGoal | None:
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if not isinstance(cached, dict):
            return None
        if cached.get("cache_version") != _GOAL_PARSE_CACHE_VERSION:
            return None
        if cached.get("text_goal") != text_goal:
            return None
        if cached.get("model") != config.model:
            return None
        if cached.get("max_phrases") != config.max_phrases:
            return None
        canonical_goal, target_phrases, context_phrases = _parsed_payload(
            cached,
            max_phrases=config.max_phrases,
        )
    except (OSError, ValueError, TypeError, RuntimeError):
        return None

    return ParsedGoal(
        canonical_goal=canonical_goal,
        target_phrases=target_phrases,
        context_phrases=context_phrases,
    )


def _llm_goal_parse(text_goal: str, config: GoalParserConfig) -> ParsedGoal:
    # Check file-based cache for deterministic results
    cache_file = _GOAL_PARSE_CACHE_DIR / f"{_cache_key(text_goal, config.model, config.max_phrases)}.json"
    if cache_file.exists():
        cached_goal = _load_cached_goal(cache_file, text_goal=text_goal, config=config)
        if cached_goal is not None:
            return cached_goal

    prompt = (
        "Read the indoor navigation instruction and extract the main target object for an open-vocabulary detector.\n"
        "Return a JSON object with this schema:\n"
        '{'
        '"canonical_goal": "lowercase coarse object name", '
        '"target_phrases": ["short detector query phrases"], '
        '"context_phrases": ["short detector query phrases for distinctive objects placed around the target"]'
        '}\n'
        "Rules:\n"
        "- The target is the one object the instruction is ABOUT (its grammatical subject, the object whose own appearance is being described). "
        "Objects mentioned only as surroundings, neighbours, or things placed on, under, beside, or behind the target are context, never the target.\n"
        "- canonical_goal must name the category of that subject object.\n"
        "- target_phrases must refer to the same target object.\n"
        "- Keep useful color/material/shape/size words if present.\n"
        "- If the instruction identifies one specific target object, return exactly 1 target phrase.\n"
        "- Only return multiple target_phrases when the instruction is genuinely ambiguous and multiple alternative detector queries are necessary.\n"
        "- When returning multiple target_phrases, order them from most specific to least specific.\n"
        "- Use the shortest phrase that still uniquely captures the described target.\n"
        "- Do not include sentences, only short phrases.\n"
        "- Do not add a generic fallback phrase unless it is necessary because the instruction is ambiguous.\n"
        f"- target_phrases must contain 1 to {config.max_phrases} items.\n"
        "- context_phrases name physical objects the instruction places near the target (furniture, appliances, fixtures, decor), as short detector queries. "
        "Exclude walls, floors, ceilings, doors, rooms, lighting conditions, colours, and any object of the target's own category. "
        f"Return an empty list when the instruction describes no such objects; at most {config.max_phrases} items.\n"
        "Examples:\n"
        '- Unique target: {"canonical_goal":"chair","target_phrases":["brown wooden chair"],"context_phrases":["dining table"]}\n'
        '- Ambiguous target: {"canonical_goal":"chair","target_phrases":["brown wooden chair","wooden chair"],"context_phrases":[]}\n'
        f"Instruction:\n{text_goal}\n"
    )
    raw_response = _call_chat_completions(config, prompt)
    payload = _extract_json_payload(raw_response)
    if not payload:
        raise RuntimeError("LLM did not return a JSON object")

    canonical_goal, target_phrases, context_phrases = _parsed_payload(
        payload,
        max_phrases=config.max_phrases,
    )

    result = ParsedGoal(
        canonical_goal=canonical_goal,
        target_phrases=target_phrases,
        context_phrases=context_phrases,
    )

    # Write cache
    try:
        _GOAL_PARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {
                    "cache_version": _GOAL_PARSE_CACHE_VERSION,
                    "canonical_goal": result.canonical_goal,
                    "target_phrases": result.target_phrases,
                    "context_phrases": result.context_phrases,
                    "text_goal": text_goal,
                    "model": config.model,
                    "max_phrases": config.max_phrases,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    return result


def parse_text_goal(text_goal: str, config: GoalParserConfig) -> ParsedGoal:
    if not str(text_goal or "").strip():
        raise ValueError("text_goal must be non-empty")
    if config.mode == "first_sentence":
        # Use the first sentence as the detector query, without the LLM
        first_sentence = re.split(
            r"""[.!?]+["'”’]*(?:\s+|$)|[。！？]+""",
            text_goal.strip(),
            maxsplit=1,
        )[0]
        query = detector_prompt_for_goal(first_sentence)
        if not query:
            raise ValueError("text_goal first sentence must contain a detector query")
        return ParsedGoal(canonical_goal=query, target_phrases=[query])
    if config.mode != "llm":
        raise ValueError("goal parser mode must be 'llm' or 'first_sentence'")
    resolved_config = resolve_goal_parser_config(config)
    if not resolved_config.base_url:
        raise ValueError("goal parser base_url is required")
    if not resolved_config.model:
        raise ValueError("goal parser model is required")
    if resolved_config.max_phrases < 1:
        raise ValueError("goal parser max_phrases must be at least 1")
    return _llm_goal_parse(text_goal, resolved_config)
