"""Navigation config loading and resolution, free of Habitat/Torch imports."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
DEFAULT_NAV_CONFIG_PATH = (
    PYTHON_ROOT / "sgf_habitat" / "configs" / "config_habitat.yaml"
)
NAV_CONFIG_SECTION = "text_image_goal_nav"
TASK_MODE_CONFIG_KEY = "task_mode_defaults"
REAL_ENV_TASK_MODE_CONFIG_KEY = "real_env_task_mode_defaults"
TASK_MODES = ("text", "instance-image")
CONFIG_TEMPLATE_PATTERN = re.compile(r"\$\{([^}]+)\}")
SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|secret|password|token|credential|authorization)",
    flags=re.IGNORECASE,
)


NAV_ARG_CONFIG_DEFAULTS: dict[str, Any] = {
    "max_steps": 1000,
    "runtime_timing": False,
    "width": "auto",
    "height": "auto",
    "hfov": 79.0,
    "camera_height": 0.88,
    "min_depth": 0.5,
    "max_depth": 5.0,
    "turn_angle": 30.0,
    "forward_step_size": 0.25,
    "agent_radius": 0.18,
    "success_radius": 1.0,
    "device": "auto",
    "seed": 0,
    "final_image_match": False,
    "final_image_match_path": "",
    "image_match_every_n_steps": 1,
    "image_match_video": False,
    "image_match_video_fps": 5.0,
    "image_match_video_path": "",
    "image_match_visual_every_n_steps": 0,
    "image_match_visual_max_lines": 300,
    "no_image_match_visuals": False,
    "visual_pose_clear_stale_goal": False,
    "panoramic_observation_trigger_scale": 1.0,
    "panoramic_observation_searched_extent_interval": False,
    "image_vlm_panoramic_query": False,
    "image_vlm_panoramic_query_panels": 4,
    "image_vlm_hypothesis_trust": False,
    "visual_pose_servo": False,
    "visual_pose_min_matches": 24,
    "visual_pose_min_inliers": 14,
    "visual_pose_min_inlier_fraction": 0.25,
    "visual_pose_min_inlier_coverage": 0.0,
    "visual_pose_reproj_threshold_px": 10.0,
    "visual_pose_min_confidence": 0.25,
    "visual_pose_min_agent_distance": 0.25,
    "visual_pose_max_agent_distance": 4.0,
    "visual_pose_max_height_delta": 1.0,
    "visual_pose_allow_stop": False,
    "sgf_stop": False,
    "text_goal_parser_mode": "llm",
    "text_sgf_stop_min_target_persistence": 0,
    "stop_validation": False,
    "text_metric_visual_closure": False,
    "text_metric_visual_closure_recovery_monotonic_bbox": False,
    "text_metric_visual_closure_contextual_closeup_persistence": False,
    "text_metric_visual_closure_grounded_detection_confidence_min": 1.01,
    "text_metric_visual_closure_contextual_bbox_area_max_fraction": 0.0,
    "text_metric_visual_closure_grounded_bbox_center_y_min_fraction": 1.01,
    "text_metric_visual_closure_grounded_success_radius_fraction": 0.0,
    "text_validated_stop_approach": False,
    "text_validated_stop_approach_max_forces": 8,
    "text_detection_probe": False,
    "text_stop_verification_object": "node_footprint",
    "text_hypothesis_triage": False,
    "text_context_guided_exploration": False,
    "text_evidence_weighted_target_ranking": False,
    "text_target_association": "navigation_map",
    "text_query_conditional_labels": False,
    "text_detection_instances": False,
    "text_detection_instance_min_observations": 2,
    # Minimum actions between two verifier queries on one hypothesis
    "text_stop_requery_interval_steps": 8,
    "text_detection_probe_min_confidence": 0.2,
    "text_target_fatigue": False,
    "text_target_fatigue_patience_steps": 60,
    "sgf_explore": False,
    "sgf_explore_min_goal_distance": 1.5,
    "sgf_explore_frontier_min_size": 20,
    "sgf_explore_info_gain_weight": 1.0,
    "sgf_explore_revisit_weight": 25.0,
    "sgf_explore_evidence_verification": False,
    "sgf_explore_information_range_scale": 1.0,
    "sgf_explore_utility_form": "rate",
    # Exchange rate (value per action) for the difference form
    "sgf_explore_exchange_rate": 0.0,
    "sgf_explore_exchange_rate_scale": 1.0,
    # Turn cost relative to one forward step (0 charges translation only)
    "sgf_explore_rotation_cost_weight": 1.0,
    "sgf_depth_guard": False,
    "sgf_visual_memory": False,
    "sgf_visual_memory_max_records": 80,
    "sgf_visual_memory_min_similarity": 0.18,
    "sgf_visual_memory_frontier_min_score": 0.02,
    "sgf_visual_memory_radius": 5.0,
    "sgf_visual_memory_cone_deg": 90.0,
    "sgf_visual_memory_weight": 8.0,
    "sgf_visual_memory_area_weight": 0.35,
    "sgf_visual_memory_cost_weight": 0.25,
    "panoramic_observation": True,
    # Add stable scene-graph surfels as occupied votes in the navigation map
    "navigation_graph_occupancy": True,
    "navigation_floor_support": True,
    "image_vlm_stop_validation": False,
    "image_vlm_foveated_hypothesis": False,
    "image_vlm_hypothesis_arrival_sweep": False,
    "image_reachable_pursuit": False,
    "image_goal_view_terminal": False,
    "image_target_localization": "pnp_goal_view",
    "image_navigation_target": "goal_camera",
    "image_pursuit_controller": "servo",
    "image_goal_view_min_corroboration": 2,
    "image_goal_view_commit_band": True,
    "image_goal_view_alignment": "active",
    "video_fps": 15,
    "image_format": "composite",
    "image_every_n_steps": 1,
    "overwrite_image_dir": False,
    "dump_sgf_inputs": False,
    "sgf_input_dump_dir": "",
    "sgf_input_dump_every_n_steps": 1,
    "overwrite_sgf_input_dump_dir": False,
    "sgf_input_replay_dir": "",
    "sgf_input_replay_stop_ends_episode": False,
    "endgame_return": False,
    "endgame_return_margin_steps": 40,
    "endgame_return_fresh_evidence_window_steps": 60,
    "endgame_return_score_advantage": 0.02,
    "text_stop_ledger": False,
    "text_stop_ledger_min_commit_approvals": 2,
    "text_stop_ledger_informative_negatives_for_exclusion": 2,
    "text_stop_ledger_absence_negatives_for_exclusion": 3,
    "text_stop_ledger_close_up_bbox_area_fraction": 0.18,
    # SPRT-inspired threshold parameter in (0, 0.5)
    "text_stop_sequential_error_rate": 0.0,
    "text_stop_evidence_rule": "sequential",
    "text_stop_reobservation": "active",
}


# Allowed values for mode switches, checked at load time
NAV_ENUM_CHOICES: dict[str, tuple[str, ...]] = {
    "text_goal_parser_mode": ("llm", "first_sentence"),
    "sgf_explore_utility_form": ("rate", "difference", "nearest"),
    "text_stop_verification_object": (
        "node_footprint",
        "top_detection",
        "whole_frame",
    ),
    "text_stop_evidence_rule": ("sequential", "count", "single_verdict"),
    "text_stop_reobservation": ("active", "passive", "same_pose"),
    "image_target_localization": ("pnp_goal_view", "depth_cluster"),
    "image_navigation_target": ("goal_camera", "object", "nearer"),
    "image_pursuit_controller": ("servo", "fmm"),
    "image_goal_view_alignment": ("active", "none"),
    "text_target_association": ("navigation_map", "reachable_euclidean", "euclidean"),
}

# Lower bounds for scalar fields whose meaning breaks below them
NAV_MINIMUM_VALUES: dict[str, float] = {
    "sgf_explore_exchange_rate": 0.0,
    "sgf_explore_exchange_rate_scale": 0.0,
    "sgf_explore_rotation_cost_weight": 0.0,
    "image_goal_view_min_corroboration": 1,
}


# Fields valid only at the root of the YAML section
NAV_CONFIG_BASE_ONLY_KEYS: tuple[str, ...] = (
    "explore_failed_goal_radius",
    "explore_min_distance_cells",
    "goal_query_limit",
    "image_fine_conf_threshold",
    "image_goal_min_y_frac",
    "image_min_coarse_sim",
    "image_min_confidence_sum",
    "image_min_inliers",
    "image_min_matches",
    "image_min_mean_confidence",
    "llm_api_key",
    "llm_base_url",
    "llm_config_path",
    "llm_model",
    "llm_timeout_seconds",
    "min_target_confidence",
    "min_target_surfels",
    "navigation_free_confirm_hits",
    "navigation_map_depth_stride",
    "navigation_map_max_range",
    "navigation_map_resolution",
    "navigation_obstacle_confirm_hits",
    "navigation_obstacle_free_margin",
    "planner_clearance_margin_m",
    "python_executable",
    "segment_filter",
    "sgf_depth_edge_threshold",
    "sgf_min_pyr_level",
    "stop_verify_api_key",
    "stop_verify_base_url",
    "stop_verify_config_path",
    "stop_verify_model",
    "stop_verify_timeout_seconds",
    "stop_verify_yolo_device",
    "surfel_min_confidence",
    "utility_explorer_goal_likelihood_weight",
    "visited_radius_cells",
    "xfeat_root",
    "xfeat_top_k",
    "yolo_class",
    "yolo_conf_threshold",
    "yolo_iou_threshold",
    "yolo_match_iou_threshold",
    "yolo_match_mode",
    "yolo_model",
)


def _config_template_context() -> dict[str, str]:
    return {
        "repo_root": str(REPO_ROOT),
        "package_root": str(PYTHON_ROOT / "sgf_habitat"),
        "repo_data_root": str(REPO_ROOT / "data"),
    }


def expand_nav_config_value(value: Any) -> Any:
    """Expand repository path templates recursively in one config value."""

    if isinstance(value, str):
        context = _config_template_context()
        return CONFIG_TEMPLATE_PATTERN.sub(
            lambda match: context.get(match.group(1), match.group(0)),
            value,
        )
    if isinstance(value, list):
        return [expand_nav_config_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): expand_nav_config_value(item)
            for key, item in value.items()
        }
    return value


def expand_repo_config_path(value: Any) -> str:
    return str(Path(str(expand_nav_config_value(value) or "")).expanduser())


def resolve_config_path(path: str | Path) -> Path:
    if not str(path or "").strip():
        raise ValueError("navigation config path is empty")
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    return config_path


def is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float, str))


def is_sensitive_key(key: str) -> bool:
    return bool(SENSITIVE_KEY_RE.search(str(key)))


def normalize_scalar_type(value: Any, baseline: Any) -> Any:
    if (
        isinstance(baseline, float)
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return float(value)
    return value


def validate_nav_override_type(
    key: str,
    value: Any,
    default: Any,
    *,
    option_label: str = "--nav-override",
) -> None:
    """Validate an already-parsed scalar against a navigation field."""

    if key in {"width", "height"}:
        if isinstance(value, str) and value.strip().lower() == "auto":
            return
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{option_label} {key} must be a positive integer or 'auto'"
            )
        return
    if isinstance(default, bool):
        valid = isinstance(value, bool)
        expected = "true or false"
    elif isinstance(default, int) and not isinstance(default, bool):
        valid = isinstance(value, int) and not isinstance(value, bool)
        expected = "an integer"
    elif isinstance(default, float):
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        expected = "a finite number"
        if valid:
            valid = math.isfinite(float(value))
    elif isinstance(default, str):
        valid = isinstance(value, str)
        expected = "a string"
    else:
        valid = type(value) is type(default)
        expected = f"a {type(default).__name__}"
    if not valid:
        raise ValueError(
            f"{option_label} {key} must be {expected}, got {value!r}"
        )
    choices = NAV_ENUM_CHOICES.get(key)
    if choices is not None and value not in choices:
        raise ValueError(
            f"{option_label} {key} must be one of {', '.join(choices)}, "
            f"got {value!r}"
        )
    minimum = NAV_MINIMUM_VALUES.get(key)
    if minimum is not None and float(value) < float(minimum):
        raise ValueError(
            f"{option_label} {key} must be at least {minimum}, got {value!r}"
        )


def _validate_nav_config_value(
    key: str,
    value: Any,
    default: Any,
    *,
    config_path: Path,
) -> None:
    try:
        validate_nav_override_type(
            key,
            value,
            default,
            option_label=f"{NAV_CONFIG_SECTION}",
        )
    except ValueError as exc:
        raise TypeError(f"{exc} in {config_path}") from exc


def _validate_task_mode_defaults(
    config: Mapping[str, Any],
    *,
    config_key: str,
    config_path: Path,
) -> None:
    overlays = config.get(config_key, {})
    if not isinstance(overlays, dict):
        raise TypeError(
            f"{NAV_CONFIG_SECTION}.{config_key} in {config_path} must be a mapping"
        )
    unknown_modes = sorted(set(overlays) - set(TASK_MODES))
    if unknown_modes:
        raise ValueError(
            f"unknown task mode in {NAV_CONFIG_SECTION}.{config_key} config "
            f"{config_path}: {', '.join(unknown_modes)}"
        )
    for task_mode, overlay in overlays.items():
        if not isinstance(overlay, dict):
            raise TypeError(
                f"{NAV_CONFIG_SECTION}.{config_key}.{task_mode} in "
                f"{config_path} must be a mapping"
            )
        unknown = sorted(set(overlay) - set(NAV_ARG_CONFIG_DEFAULTS))
        if unknown:
            raise ValueError(
                f"unknown keys in {NAV_CONFIG_SECTION}.{config_key}.{task_mode} "
                f"config {config_path}: {', '.join(unknown)}"
            )
        for key, value in overlay.items():
            _validate_nav_config_value(
                key,
                value,
                NAV_ARG_CONFIG_DEFAULTS[key],
                config_path=config_path,
            )


def _validate_nav_config(
    config: Mapping[str, Any],
    *,
    config_path: Path,
) -> None:
    allowed_root_keys = {
        *NAV_ARG_CONFIG_DEFAULTS,
        *NAV_CONFIG_BASE_ONLY_KEYS,
        TASK_MODE_CONFIG_KEY,
        REAL_ENV_TASK_MODE_CONFIG_KEY,
    }
    unknown = sorted(set(config) - allowed_root_keys)
    if unknown:
        raise ValueError(
            f"unknown keys in {NAV_CONFIG_SECTION} config {config_path}: "
            f"{', '.join(unknown)}"
        )
    for key, default in NAV_ARG_CONFIG_DEFAULTS.items():
        _validate_nav_config_value(
            key,
            config[key],
            default,
            config_path=config_path,
        )
    _validate_task_mode_defaults(
        config,
        config_key=TASK_MODE_CONFIG_KEY,
        config_path=config_path,
    )
    _validate_task_mode_defaults(
        config,
        config_key=REAL_ENV_TASK_MODE_CONFIG_KEY,
        config_path=config_path,
    )


def load_nav_config(path: str | Path) -> dict[str, Any]:
    """Load, expand, and validate the navigation YAML section."""

    config_path = resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"navigation config file does not exist: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"navigation config must be a YAML mapping: {config_path}")
    section = payload.get(NAV_CONFIG_SECTION, {})
    if not isinstance(section, dict):
        raise ValueError(
            f"navigation config missing mapping section "
            f"{NAV_CONFIG_SECTION!r}: {config_path}"
        )
    expanded = expand_nav_config_value(section)
    config = {
        **NAV_ARG_CONFIG_DEFAULTS,
        **{str(key): value for key, value in expanded.items()},
    }
    _validate_nav_config(config, config_path=config_path)
    return config


def _task_mode_overlay(
    nav_config: Mapping[str, Any],
    config_key: str,
    task_mode: str,
) -> dict[str, Any]:
    overlays = nav_config.get(config_key, {})
    if not isinstance(overlays, Mapping):
        return {}
    overlay = overlays.get(str(task_mode), {})
    return dict(overlay) if isinstance(overlay, Mapping) else {}


def effective_nav_config_for_task(
    nav_config: Mapping[str, Any],
    *,
    task_mode: str,
    real_env: bool,
) -> dict[str, Any]:
    """Apply task and optional real-environment overlays to a loaded config."""

    if task_mode not in TASK_MODES:
        raise ValueError(f"unknown task mode: {task_mode!r}")
    config = {
        **nav_config,
        **_task_mode_overlay(nav_config, TASK_MODE_CONFIG_KEY, task_mode),
    }
    if real_env:
        config.update(
            _task_mode_overlay(
                nav_config,
                REAL_ENV_TASK_MODE_CONFIG_KEY,
                task_mode,
            )
        )
    return config


def _flat_effective_config(
    nav_config: Mapping[str, Any],
    *,
    task_mode: str,
    real_env: bool,
) -> dict[str, Any]:
    effective = effective_nav_config_for_task(
        nav_config,
        task_mode=task_mode,
        real_env=real_env,
    )
    effective.pop(TASK_MODE_CONFIG_KEY, None)
    effective.pop(REAL_ENV_TASK_MODE_CONFIG_KEY, None)
    return effective


def parse_nav_override_value(raw_value: str) -> Any:
    value = str(raw_value).strip()
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return ""
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(
        r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?",
        value,
    ):
        return float(value)
    return value


def _parse_typed_nav_override_value(
    *,
    key: str,
    raw_value: str,
    baseline: Any,
) -> Any:
    value = str(raw_value).strip()
    lower = value.lower()
    if key in {"width", "height"}:
        if lower == "auto":
            return "auto"
        if re.fullmatch(r"[-+]?\d+", value):
            parsed_dimension = int(value)
            if parsed_dimension > 0:
                return parsed_dimension
        raise TypeError(
            f"--nav-override {key} must be a positive integer or 'auto', "
            f"got {raw_value!r}"
        )
    if isinstance(baseline, bool):
        if lower not in {"true", "false"}:
            raise TypeError(
                f"--nav-override {key} must be a boolean, got {raw_value!r}"
            )
        return lower == "true"
    if isinstance(baseline, int) and not isinstance(baseline, bool):
        if not re.fullmatch(r"[-+]?\d+", value):
            raise TypeError(
                f"--nav-override {key} must be an integer, got {raw_value!r}"
            )
        return int(value)
    if isinstance(baseline, float):
        if not re.fullmatch(
            r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?",
            value,
        ):
            raise TypeError(
                f"--nav-override {key} must be numeric, got {raw_value!r}"
            )
        parsed = float(value)
        if not math.isfinite(parsed):
            raise TypeError(
                f"--nav-override {key} must be finite, got {raw_value!r}"
            )
        return parsed
    if isinstance(baseline, str):
        parsed = parse_nav_override_value(raw_value)
        if not isinstance(parsed, str):
            raise TypeError(
                f"--nav-override {key} must be a string, got {raw_value!r}"
            )
        return parsed
    raise TypeError(
        f"--nav-override {key} is not a scalar configuration field"
    )


def parse_nav_overrides(
    values: Sequence[str] | None,
    effective_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse repeatable KEY=VALUE options against the effective schema."""

    schema = NAV_ARG_CONFIG_DEFAULTS if effective_config is None else effective_config
    overrides: dict[str, Any] = {}
    for item in values or ():
        if "=" not in str(item):
            raise ValueError(f"--nav-override must be KEY=VALUE, got {item!r}")
        key, raw_value = str(item).split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--nav-override has empty key: {item!r}")
        if any(marker in key for marker in (".", "[", "]")):
            raise ValueError(
                f"--nav-override only supports top-level keys, got {key!r}"
            )
        if is_sensitive_key(key):
            raise ValueError(
                f"--nav-override cannot modify sensitive field: {key}"
            )
        if key not in schema:
            raise ValueError(f"unknown --nav-override key: {key}")
        overrides[key] = _parse_typed_nav_override_value(
            key=key,
            raw_value=raw_value,
            baseline=schema[key],
        )
    return overrides


def _validated_typed_overrides(
    values: Mapping[str, Any] | None,
    *,
    effective_config: Mapping[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw_key, value in (values or {}).items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("navigation override has an empty key")
        if any(marker in key for marker in (".", "[", "]")):
            raise ValueError(
                f"navigation override only supports top-level keys, got {key!r}"
            )
        if is_sensitive_key(key):
            raise ValueError(f"navigation override cannot modify sensitive field: {key}")
        if key not in effective_config:
            raise ValueError(f"unknown navigation override key: {key}")
        validate_nav_override_type(key, value, effective_config[key])
        overrides[key] = normalize_scalar_type(value, effective_config[key])
    return overrides


@dataclass(frozen=True)
class ResolvedNavConfig(Mapping[str, Any]):
    """Read-only, flat navigation configuration for one runtime mode."""

    config_path: Path
    task_mode: str
    real_env: bool
    config: Mapping[str, Any] = field(repr=False)
    overrides: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "config_path", Path(self.config_path))
        object.__setattr__(self, "task_mode", str(self.task_mode))
        object.__setattr__(self, "real_env", bool(self.real_env))
        object.__setattr__(
            self,
            "config",
            MappingProxyType(dict(self.config)),
        )
        object.__setattr__(
            self,
            "overrides",
            MappingProxyType(dict(self.overrides)),
        )

    def __getitem__(self, key: str) -> Any:
        return self.config[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.config)

    def __len__(self) -> int:
        return len(self.config)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.config)


def _resolve_loaded_nav_config(
    loaded: Mapping[str, Any],
    *,
    config_path: Path,
    task_mode: str,
    real_env: bool,
    nav_overrides: Sequence[str] | None,
    overrides: Mapping[str, Any] | None,
) -> ResolvedNavConfig:
    task_config = _flat_effective_config(
        loaded,
        task_mode=task_mode,
        real_env=False,
    )
    parsed_overrides = parse_nav_overrides(
        nav_overrides,
        effective_config=task_config,
    )
    typed_overrides = _validated_typed_overrides(
        overrides,
        effective_config={**task_config, **parsed_overrides},
    )
    merged_overrides = {**parsed_overrides, **typed_overrides}
    preliminary = {**task_config, **merged_overrides}
    effective_real_env = bool(
        real_env
        or str(preliminary.get("sgf_input_replay_dir", "") or "").strip()
    )
    effective = _flat_effective_config(
        loaded,
        task_mode=task_mode,
        real_env=effective_real_env,
    )
    # Overrides apply last, after any real-env overlay
    parsed_overrides = parse_nav_overrides(
        nav_overrides,
        effective_config=effective,
    )
    typed_overrides = _validated_typed_overrides(
        overrides,
        effective_config={**effective, **parsed_overrides},
    )
    merged_overrides = {**parsed_overrides, **typed_overrides}
    return ResolvedNavConfig(
        config_path=config_path,
        task_mode=task_mode,
        real_env=effective_real_env,
        config={**effective, **merged_overrides},
        overrides=merged_overrides,
    )


def resolve_nav_config(
    path: str | Path = DEFAULT_NAV_CONFIG_PATH,
    *,
    task_mode: str,
    real_env: bool = False,
    nav_overrides: Sequence[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ResolvedNavConfig:
    """Resolve one flat, typed task configuration from YAML and overrides."""

    config_path = resolve_config_path(path)
    loaded = load_nav_config(config_path)
    return _resolve_loaded_nav_config(
        loaded,
        config_path=config_path,
        task_mode=task_mode,
        real_env=real_env,
        nav_overrides=tuple(nav_overrides or ()),
        overrides=overrides,
    )


def resolve_nav_configs(
    path: str | Path,
    task_modes: Sequence[str],
    *,
    real_env: bool = False,
) -> dict[str, ResolvedNavConfig]:
    """Resolve several task modes while loading and validating YAML once."""

    config_path = resolve_config_path(path)
    loaded = load_nav_config(config_path)
    return {
        str(task_mode): _resolve_loaded_nav_config(
            loaded,
            config_path=config_path,
            task_mode=str(task_mode),
            real_env=real_env,
            nav_overrides=(),
            overrides=None,
        )
        for task_mode in task_modes
    }


__all__ = [
    "DEFAULT_NAV_CONFIG_PATH",
    "NAV_ARG_CONFIG_DEFAULTS",
    "NAV_CONFIG_BASE_ONLY_KEYS",
    "NAV_CONFIG_SECTION",
    "REAL_ENV_TASK_MODE_CONFIG_KEY",
    "ResolvedNavConfig",
    "TASK_MODES",
    "TASK_MODE_CONFIG_KEY",
    "effective_nav_config_for_task",
    "expand_nav_config_value",
    "expand_repo_config_path",
    "is_scalar",
    "is_sensitive_key",
    "load_nav_config",
    "normalize_scalar_type",
    "parse_nav_override_value",
    "parse_nav_overrides",
    "resolve_config_path",
    "resolve_nav_config",
    "resolve_nav_configs",
    "validate_nav_override_type",
]
