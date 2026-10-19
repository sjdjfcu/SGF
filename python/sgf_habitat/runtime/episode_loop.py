#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import sys
import types
from collections import Counter
from dataclasses import replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Sequence


THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
)
DEFAULT_EPISODE_CPU_THREADS = 1


def _parse_positive_int(value: str | None, default: int = 0) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _configured_cpu_threads() -> int:
    explicit = os.environ.get("SGF_EPISODE_CPU_THREADS")
    if explicit is not None:
        return _parse_positive_int(explicit, default=0)
    inherited = os.environ.get("OMP_NUM_THREADS")
    if inherited is not None:
        return _parse_positive_int(inherited, default=0)
    return DEFAULT_EPISODE_CPU_THREADS


def _apply_thread_env_defaults() -> None:
    threads = _configured_cpu_threads()
    if threads > 0:
        value = str(threads)
        os.environ.setdefault("SGF_EPISODE_CPU_THREADS", value)
        for name in THREAD_ENV_VARS:
            os.environ.setdefault(name, value)
        os.environ.setdefault("OMP_DYNAMIC", "FALSE")
        os.environ.setdefault("MKL_DYNAMIC", "FALSE")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("KMP_BLOCKTIME", "0")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Must be set before torch and CUDA are imported
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", "0")


_apply_thread_env_defaults()

import cv2
import numpy as np
import quaternion
import requests
import torch
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from PIL import Image


def _apply_runtime_thread_limits() -> None:
    threads = _configured_cpu_threads()
    if threads <= 0:
        return
    try:
        cv2.setNumThreads(threads)
    except Exception:
        pass
    try:
        torch.set_num_threads(threads)
    except Exception:
        pass
    interop_threads = _parse_positive_int(
        os.environ.get("SGF_EPISODE_TORCH_INTEROP_THREADS"),
        default=1,
    )
    try:
        torch.set_num_interop_threads(interop_threads)
    except Exception:
        pass


_apply_runtime_thread_limits()


def _configure_determinism(seed: int) -> None:
    import random as _random
    seed = int(seed) if int(seed) >= 0 else 0
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    matmul_backend = getattr(torch.backends.cuda, "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
XFEAT_MATCHER_NAME = "xfeat"
SGF_DEPTH_GUARD_RETIRE_BLOCK_COUNT = 2
# Synthetic ids for detection-probe hypotheses, above any real node id
DETECTION_PROBE_NODE_ID_BASE = 10**6
NavigationResult = dict[str, Any]


def should_run_endgame_return_bookkeeping(
    *,
    planner_disabled: bool,
    proposed_stop: bool,
    fallback_return_allowed: bool = False,
) -> bool:
    return bool(
        not planner_disabled
        and (
            not proposed_stop
            or fallback_return_allowed
        )
    )


if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
from sgf_habitat.runtime.timing import RuntimeTiming, detail_scope  # noqa: E402
from sgf_habitat.runtime.nav_config import (  # noqa: E402
    DEFAULT_NAV_CONFIG_PATH,
    NAV_ARG_CONFIG_DEFAULTS,
    NAV_CONFIG_BASE_ONLY_KEYS,
    NAV_CONFIG_SECTION,
    expand_repo_config_path,
    resolve_nav_config,
)
from sgf_habitat import (  # noqa: E402
    GoalParserConfig,
    parse_text_goal,
)
from sgf_habitat.decision.stop_features import compute_stop_features  # noqa: E402
from sgf_habitat.decision.instance_image_arrival import (  # noqa: E402
    vlm_unavailable_strong_geometry_allows_stop,
    weak_visual_stop_rejection_reason,
)
from sgf_habitat.decision.metric_visual_closure import (  # noqa: E402
    MetricVisualClosureApprovalSnapshot,
    MetricVisualClosureConfig,
    OnePrimitiveClosureLatch,
    decide_contextual_stop_persistence_closure,
    decide_metric_visual_closure,
    decide_recovery_monotonic_closure,
)
from sgf_habitat.decision.policy_scales import PolicyScales  # noqa: E402
from sgf_habitat.decision.goal_view_terminal import (  # noqa: E402
    GoalViewBelief,
    signed_heading_error_deg,
)
from sgf_habitat.decision.stop_policy import (  # noqa: E402
    EXPLORE_ACTION,
    RECOVER_ACTION,
    STOP_ACTION,
    StopDecisionPolicy,
    StopPolicyConfig,
)
from sgf_habitat.decision.hypothesis_projection import (  # noqa: E402
    ProjectedHypothesis,
    detection_share_inside,
    project_hypothesis,
    project_hypothesis_points,
)
from sgf_habitat.navigation.detection_instances import DetectionInstanceRegistry
from sgf_habitat.navigation.representation_diagnostics import association_rank_diagnostics
from sgf_habitat.decision.shadow_stop import ShadowStopAdjudicator  # noqa: E402
from sgf_habitat.decision.stop_ledger import (  # noqa: E402
    TargetStopLedger,
)
from sgf_habitat.decision.terminal_commit import TerminalCommitPlanner  # noqa: E402
from sgf_habitat.env import TextGoalEpisodeEnv  # noqa: E402
from sgf_habitat.goal.grounding import dedupe_goal_phrases  # noqa: E402
from sgf_habitat.goal.instance_image import (  # noqa: E402
    CameraIntrinsics,
    ImageSpaceMatchEvidence,
    ImageTargetConfig,
    XFeatStarMatcher,
    build_frame_geometry_evidence,
    estimate_visual_pose_servo_goal,
    geometric_evidence_completeness,
    localize_instance_image_evidence,
    match_instance_image_evidence,
    weak_instance_hypothesis,
)
from sgf_habitat.goal.vlm_reid import (  # noqa: E402
    lift_box_to_world,
    query_instance_box,
)
from sgf_habitat.exploration.utility_explorer import (  # noqa: E402
    ConfidenceConstrainedFrontierChoice,
    FrontierUtility,
    UtilityExplorer,
    confidence_constrained_frontier_choice,
)
from sgf_habitat.exploration.visual_memory import (  # noqa: E402
    LeakyMaxVisualMemory,
    VisualMemoryConfig,
)
from sgf_habitat.navigation.fmm_navigator import (  # noqa: E402
    FMMGoalNavigator,
    NavigationCommand,
    compute_heading_error_deg,
)
from sgf_habitat.navigation.explorer import (  # noqa: E402
    GoalEndEvent,
    GoalEndReason,
)
from sgf_habitat.navigation.sgf_map import SGFNavigationMap  # noqa: E402
from sgf_habitat.planning.fair_goal_scheduler import (  # noqa: E402
    GoalExpert,
    TransactionalFairGoalEpochScheduler,
)
import sgf_habitat.runtime.episode_support as _episode_support  # noqa: E402
from sgf_habitat.runtime.episode_support import (  # noqa: E402,F401
    bgr_from_rgb_observation,
    build_query_sets,
    compute_intrinsics,
    create_online_sgf,
    depth_forward_navigation_guard,
    depth_observation_to_2d,
    finalize_image_target_node,
    planar_distance_xz,
    plan_exploration_action,
    refresh_image_target_node,
    select_hint_node,
    select_target_node,
    sensor_state_to_world_from_camera_cv,
    summarize_stop_decision,
    summarize_stop_validation,
    update_episode_label_stats,
    update_episode_label_stats_from_summary,
)
from sgf_habitat.validation.stop import (  # noqa: E402
    StopValidationResult,
    StopValidator,
    VisionLanguageConfig,
    resolve_vlm_config,
)
from sgf_habitat.visualization import (  # noqa: E402
    EpisodeImageDumper,
    EpisodeVisualizer,
    ImageMatchVideoWriter,
    render_image_match_canvas,
    render_episode_canvas,
    save_image_match_visualization,
)


def repo_data_path(*parts: str) -> str:
    return str((REPO_ROOT / "data").joinpath(*parts))


class ImageGoalEncoder:
    """CLIP image embeddings used only by instance-image navigation."""

    model_name = "RN50"

    def __init__(self, device: torch.device) -> None:
        import clip

        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(
            self.model_name,
            device=self.device,
            jit=False,
        )
        self.model.eval()

    @torch.no_grad()
    def encode_image_embeddings(
        self,
        images_rgb: list[np.ndarray],
        device: torch.device,
    ) -> torch.Tensor:
        if not images_rgb:
            output_dim = int(getattr(self.model.visual, "output_dim", 1024))
            return torch.zeros(
                (0, output_dim),
                dtype=torch.float32,
                device=device,
            )
        tensors = []
        for image_rgb in images_rgb:
            image = np.asarray(image_rgb, dtype=np.uint8)[:, :, :3]
            tensors.append(self.preprocess(Image.fromarray(image)))
        batch = torch.stack(tensors, dim=0).to(device=self.device)
        embeddings = self.model.encode_image(batch).float()
        embeddings /= embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return embeddings.to(device=device)

    @torch.no_grad()
    def encode_image_embedding(
        self,
        image_rgb: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        return self.encode_image_embeddings([image_rgb], device)

def resolve_xfeat_root(value: Any) -> str:
    configured = Path(expand_repo_config_path(value)).expanduser()
    candidates: list[Path] = []
    env_root = os.environ.get("SGF_XFEAT_ROOT", "").strip()
    if env_root:
        candidates.append(Path(expand_repo_config_path(env_root)).expanduser())
    candidates.append(configured)
    for parent in (REPO_ROOT, *REPO_ROOT.parents):
        candidates.append(parent / "accelerated_features")
    seen: set[str] = set()
    for candidate in candidates:
        root = candidate.resolve()
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if (root / "modules" / "xfeat.py").is_file():
            return str(root)
    return str(configured.resolve())


def load_goal_image(path: str) -> np.ndarray:
    image = cv2.imread(str(Path(path).expanduser()), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read goal image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray | None) -> str:
    if image_rgb is None:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8)[:, :, :3], cv2.COLOR_RGB2BGR))
    return str(path)


class SGFInputDumper:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        every_n_steps: int = 1,
        overwrite: bool = False,
    ) -> None:
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be >= 1")
        self.output_dir = Path(output_dir).expanduser()
        if self.output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"SGF input dump directory already exists: {self.output_dir}")
            if self.output_dir.is_dir():
                shutil.rmtree(self.output_dir)
            else:
                self.output_dir.unlink()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.rgb_dir = self.output_dir / "rgb"
        self.depth_m_dir = self.output_dir / "depth_m"
        self.depth_mm_dir = self.output_dir / "depth_mm"
        self.pose_dir = self.output_dir / "pose"
        for directory in (self.rgb_dir, self.depth_m_dir, self.depth_mm_dir, self.pose_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.every_n_steps = int(every_n_steps)
        self.frames_jsonl_path = self.output_dir / "frames.jsonl"
        self.metadata_path = self.output_dir / "metadata.json"

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        payload = {
            **metadata,
            "format": {
                "rgb": "rgb/step_XXXX.png, saved as RGB observation rendered through PNG",
                "depth_m": "depth_m/step_XXXX.npy, float32 depth in meters exactly as passed to SGF",
                "depth_mm": "depth_mm/step_XXXX.png, uint16 depth in millimeters after SGF-style conversion",
                "pose": "pose/step_XXXX.json, world_from_camera_cv 4x4 matrix in meters",
                "frames": "frames.jsonl contains one relative-path record per dumped frame",
            },
            "sgf_process_frame_color_arg": "color_bgr",
            "rgb_png_note": "SGF receives BGR; the PNG is converted back to RGB for inspection.",
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _relative(self, path: Path) -> str:
        return str(path.relative_to(self.output_dir))

    @staticmethod
    def _depth_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32)
        depth_mm = np.zeros(depth.shape, dtype=np.uint16)
        valid = np.isfinite(depth) & (depth > 0.0)
        if np.any(valid):
            depth_mm[valid] = np.clip(
                np.rint(depth[valid].astype(np.float64) * 1000.0),
                0.0,
                float(np.iinfo(np.uint16).max),
            ).astype(np.uint16)
        return depth_mm

    @staticmethod
    def _sensor_state_payload(sensor_state: Any | None) -> dict[str, Any]:
        if sensor_state is None:
            return {}
        payload: dict[str, Any] = {}
        position = getattr(sensor_state, "position", None)
        if position is not None:
            position_array = np.asarray(position, dtype=np.float64).reshape(-1)
            if position_array.size == 3 and np.all(np.isfinite(position_array)):
                payload["sensor_position_m"] = [float(value) for value in position_array]
        rotation = getattr(sensor_state, "rotation", None)
        if rotation is not None and all(hasattr(rotation, name) for name in ("w", "x", "y", "z")):
            payload["sensor_rotation_wxyz"] = [
                float(rotation.w),
                float(rotation.x),
                float(rotation.y),
                float(rotation.z),
            ]
        return payload

    def dump_frame(
        self,
        *,
        step_index: int,
        rgb_observation: np.ndarray,
        depth_m: np.ndarray,
        world_from_camera_cv: np.ndarray,
        processed_by: Sequence[str],
        sensor_state: Any | None = None,
    ) -> dict[str, Any] | None:
        if int(step_index) % self.every_n_steps != 0:
            return None

        stem = f"step_{int(step_index):04d}"
        rgb = np.asarray(rgb_observation, dtype=np.uint8)[:, :, :3]
        depth = np.asarray(depth_m, dtype=np.float32)
        pose = np.asarray(world_from_camera_cv, dtype=np.float32)
        if depth.ndim != 2:
            raise RuntimeError(f"expected SGF depth input to be HxW, got shape {depth.shape}")
        if pose.shape != (4, 4):
            raise RuntimeError(f"expected SGF camera pose input to be 4x4, got shape {pose.shape}")

        rgb_path = self.rgb_dir / f"{stem}.png"
        depth_npy_path = self.depth_m_dir / f"{stem}.npy"
        depth_png_path = self.depth_mm_dir / f"{stem}.png"
        pose_json_path = self.pose_dir / f"{stem}.json"

        if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write SGF input RGB image {rgb_path}")
        np.save(str(depth_npy_path), depth)
        if not cv2.imwrite(str(depth_png_path), self._depth_to_uint16_mm(depth)):
            raise RuntimeError(f"failed to write SGF input depth image {depth_png_path}")

        pose_payload: dict[str, Any] = {
            "step": int(step_index),
            "world_from_camera_cv": pose.astype(np.float64).tolist(),
            "camera_position_m": pose[:3, 3].astype(np.float64).tolist(),
            "pose_convention": "OpenCV camera coordinates in Habitat world frame; translation is meters.",
            "processed_by": list(processed_by),
        }
        pose_payload.update(self._sensor_state_payload(sensor_state))
        pose_json_path.write_text(json.dumps(pose_payload, indent=2), encoding="utf-8")

        record = {
            "step": int(step_index),
            "rgb_path": self._relative(rgb_path),
            "depth_m_npy_path": self._relative(depth_npy_path),
            "depth_mm_png_path": self._relative(depth_png_path),
            "pose_json_path": self._relative(pose_json_path),
            "processed_by": list(processed_by),
            "rgb_shape": [int(value) for value in rgb.shape],
            "depth_shape": [int(value) for value in depth.shape],
            "depth_dtype": str(depth.dtype),
        }
        with self.frames_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def close(self) -> None:
        return None


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode(
        ".png",
        np.asarray(image_bgr, dtype=np.uint8)[:, :, :3],
    )
    if not success:
        raise RuntimeError("failed to encode VLM stop validation image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def _extract_json_object(text: Any) -> dict[str, Any]:
    if isinstance(text, list):
        text = " ".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in text
        )
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match is None:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def create_image_pair_vlm_config(
    nav_config: dict[str, Any],
) -> VisionLanguageConfig:
    return resolve_vlm_config(
        VisionLanguageConfig(
            config_path=(
                expand_repo_config_path(
                    nav_config.get("stop_verify_config_path", "")
                )
                if str(
                    nav_config.get("stop_verify_config_path", "") or ""
                ).strip()
                else ""
            ),
            base_url=str(
                nav_config.get("stop_verify_base_url", "") or ""
            ),
            api_key=str(nav_config.get("stop_verify_api_key", "") or ""),
            model=str(nav_config.get("stop_verify_model", "") or ""),
            timeout_seconds=float(
                nav_config.get("stop_verify_timeout_seconds", 20.0)
            ),
        )
    )


def validate_image_stop_with_vlm(
    *,
    vlm_config: VisionLanguageConfig,
    goal_image_rgb: np.ndarray,
    current_rgb: np.ndarray,
    timing: RuntimeTiming | None = None,
) -> dict[str, Any]:
    if not (vlm_config.base_url and vlm_config.model):
        return {
            "attempted": False,
            "approved": False,
            "error": "incomplete_vlm_config",
        }
    goal_bgr = cv2.cvtColor(
        np.asarray(goal_image_rgb, dtype=np.uint8)[:, :, :3],
        cv2.COLOR_RGB2BGR,
    )
    current_bgr = cv2.cvtColor(
        np.asarray(current_rgb, dtype=np.uint8)[:, :, :3],
        cv2.COLOR_RGB2BGR,
    )
    prompt = (
        "You compare two images for an instance-image navigation stop. "
        "Image A is the goal target image. Image B is the robot's current "
        "camera view. Return JSON only: "
        '{"same_target_visible": true/false, "confidence": 0.0-1.0, '
        '"object_evidence": "strong|weak|none", "context_only": '
        'true/false, "reason": "short"}. '
        "Note: confidence measures your confidence in your verdict itself (whether same_target_visible is true or false). "
        "The target is the main subject of Image A, the object the "
        "photograph is centred on, not the room or the other objects "
        "around it. "
        "Answer true only if Image B clearly shows the same physical object "
        "instance as Image A, not just the same category, room, layout, or "
        "background context. Set context_only true when the match is based "
        "mainly on room layout, furniture arrangement, walls, windows, "
        "lighting, or style instead of the target object itself. Partial "
        "close views are OK only if distinctive object evidence matches. "
        "If uncertain, answer false."
    )
    headers = {"Content-Type": "application/json"}
    if vlm_config.api_key:
        headers["Authorization"] = f"Bearer {vlm_config.api_key}"
    with detail_scope(timing, "image_encoding"):
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + _encode_png_base64(goal_bgr)
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/png;base64," + _encode_png_base64(current_bgr)
                    )
                },
            },
        ]
        payload = {
            "model": vlm_config.model,
            "messages": [{"role": "user", "content": content}],
        }
    with detail_scope(timing, "vlm_request"):
        response = requests.post(
            vlm_config.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=float(vlm_config.timeout_seconds),
        )
        response.raise_for_status()
    with detail_scope(timing, "response_parsing"):
        message_content = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json_object(message_content)
        normalized_same_target = _vlm_bool(
            parsed,
            "same_target_visible",
            "same_target",
        )
        confidence = parsed.get("confidence")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = None
        approved = bool(normalized_same_target)
        object_evidence = str(
            parsed.get("object_evidence", "")
        ).strip().lower()
        context_only = bool(_vlm_bool(parsed, "context_only"))
        if object_evidence == "none" or (
            object_evidence == "weak" and context_only
        ):
            approved = False
        trace = {
            "attempted": True,
            "approved": bool(approved),
            "same_target_visible": normalized_same_target,
            "object_evidence": object_evidence or None,
            "context_only": bool(context_only),
            "confidence": (
                None
                if confidence_value is None
                else round(float(confidence_value), 4)
            ),
            "reason": str(parsed.get("reason", "")),
            "raw_response": str(message_content)[:1000],
        }
        if normalized_same_target is None:
            trace["parse_error"] = True
        return trace


def compose_panel_grid(
    views_rgb: list[np.ndarray],
    *,
    panel_width: int = 320,
) -> np.ndarray:
    """Tile views into a labelled 2xN grid in reading order."""

    if not views_rgb:
        raise ValueError("at least one view is required")
    panels: list[np.ndarray] = []
    for index, view in enumerate(views_rgb):
        image = np.asarray(view, dtype=np.uint8)[:, :, :3]
        scale = float(panel_width) / max(1.0, float(image.shape[1]))
        resized = cv2.resize(
            image,
            (panel_width, max(1, int(round(image.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        labelled = cv2.copyMakeBorder(
            resized, 28, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        cv2.putText(
            labelled,
            str(index + 1),
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(labelled)
    columns = 2
    rows = int(math.ceil(len(panels) / columns))
    height = max(panel.shape[0] for panel in panels)
    width = max(panel.shape[1] for panel in panels)
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        canvas[
            row * height : row * height + panel.shape[0],
            column * width : column * width + panel.shape[1],
        ] = panel
    return canvas


def query_panoramic_instance_panel(
    *,
    vlm_config: VisionLanguageConfig,
    goal_image_rgb: np.ndarray,
    panel_grid_rgb: np.ndarray,
    panel_count: int,
    timing: RuntimeTiming | None = None,
) -> dict[str, Any]:
    """Ask the VLM which sweep panel, if any, shows the reference instance."""

    if not (vlm_config.base_url and vlm_config.model):
        return {"attempted": False, "error": "incomplete_vlm_config"}
    prompt = (
        "Image A is a reference photograph of one specific object instance. "
        f"Image B is a {panel_count}-panel grid of views captured by a robot "
        "turning in place; each panel is numbered in its top-left corner. "
        'Return JSON only: {"panel": <0-'
        f"{panel_count}"
        '>, "confidence": 0.0-1.0, "reason": "short"}. '
        "Give the number of the single panel that shows the same physical "
        "object instance as image A, even if it is small, partially visible, "
        "or seen from a different angle. Answer 0 when no panel shows that "
        "object, when several panels are equally plausible, or when the only "
        "resemblance is room layout, furniture style, or background."
    )
    headers = {"Content-Type": "application/json"}
    if vlm_config.api_key:
        headers["Authorization"] = f"Bearer {vlm_config.api_key}"
    with detail_scope(timing, "image_encoding"):
        goal_bgr = cv2.cvtColor(
            np.asarray(goal_image_rgb, dtype=np.uint8)[:, :, :3], cv2.COLOR_RGB2BGR
        )
        panel_bgr = cv2.cvtColor(
            np.asarray(panel_grid_rgb, dtype=np.uint8)[:, :, :3], cv2.COLOR_RGB2BGR
        )
        payload = {
            "model": vlm_config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + _encode_png_base64(goal_bgr)
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + _encode_png_base64(panel_bgr)
                            },
                        },
                    ],
                }
            ],
        }
    with detail_scope(timing, "vlm_request"):
        response = requests.post(
            vlm_config.base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=float(vlm_config.timeout_seconds),
        )
        response.raise_for_status()
    with detail_scope(timing, "response_parsing"):
        message_content = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json_object(message_content)
        try:
            panel = int(parsed.get("panel"))
        except (TypeError, ValueError):
            panel = 0
        if panel < 1 or panel > int(panel_count):
            panel = 0
        confidence = parsed.get("confidence")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = None
        return {
            "attempted": True,
            "panel": int(panel),
            "confidence": (
                None if confidence_value is None else round(confidence_value, 4)
            ),
            "reason": str(parsed.get("reason", ""))[:200],
        }


def _vlm_bool(payload: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"true", "yes", "1", "visible"}:
            return True
        if normalized in {"false", "no", "0", "none", "not_visible"}:
            return False
    return None


def _finite_trace_float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in payload:
            continue
        try:
            value = float(payload.get(key))
        except (AttributeError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            return float(value)
    return None


def planar_path_length_m(positions: Sequence[np.ndarray]) -> float:
    """Planar length of a recorded agent trail."""

    return float(
        sum(
            float(np.linalg.norm((np.asarray(later) - np.asarray(earlier))[[0, 2]]))
            for earlier, later in zip(positions[:-1], positions[1:])
        )
    )


def ledger_evidence_weight(
    stop_validation_result: Any,
    *,
    node: dict[str, Any] | None,
    probe_owned: bool,
    world_from_camera_cv: Any,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    evidence_standpoint_m: Any = None,
    consistency_radius_m: float = float("inf"),
    verification_object: str = "node_footprint",
) -> float:
    """Fraction of a verifier verdict credited to the node hypothesis, in [0, 1]."""

    if stop_validation_result is None:
        return 0.0
    if not bool(getattr(stop_validation_result, "detection_found", False)):
        return 0.0
    if probe_owned:
        return 1.0
    if verification_object == "whole_frame":
        return 1.0
    if verification_object == "top_detection":
        if node is None or evidence_standpoint_m is None:
            return 0.0
        try:
            standpoint = np.asarray(evidence_standpoint_m, dtype=np.float64).reshape(3)
            node_goal = np.asarray(node.get("goal_position"), dtype=np.float64).reshape(3)
        except (TypeError, ValueError):
            return 0.0
        if not (np.all(np.isfinite(standpoint)) and np.all(np.isfinite(node_goal))):
            return 0.0
        return float(
            math.hypot(
                float(standpoint[0] - node_goal[0]),
                float(standpoint[2] - node_goal[2]),
            )
            <= float(consistency_radius_m)
        )
    if (
        node is not None
        and evidence_standpoint_m is not None
        and math.isfinite(float(consistency_radius_m))
    ):
        try:
            standpoint = np.asarray(evidence_standpoint_m, dtype=np.float64).reshape(3)
            node_goal = np.asarray(node.get("goal_position"), dtype=np.float64).reshape(3)
            if np.all(np.isfinite(standpoint)) and np.all(np.isfinite(node_goal)):
                if math.hypot(
                    float(standpoint[0] - node_goal[0]),
                    float(standpoint[2] - node_goal[2]),
                ) > float(consistency_radius_m):
                    return 0.0
        except (TypeError, ValueError):
            pass
    source = str(getattr(stop_validation_result, "evidence_source", "") or "")
    if source in {"detector_grounded", "hypothesis_projection"}:
        return 1.0
    bbox = getattr(stop_validation_result, "detection_bbox_xyxy", None)
    if node is None or bbox is None or world_from_camera_cv is None:
        return 0.0
    try:
        footprint = project_hypothesis(
            bbox_min_m=node.get("bbox_min_m"),
            bbox_max_m=node.get("bbox_max_m"),
            world_from_camera_cv=world_from_camera_cv,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=int(width),
            height=int(height),
        )
    except Exception:
        return 0.0
    if footprint is None:
        return 0.0
    return float(detection_share_inside(tuple(bbox), footprint.bbox_xyxy))


def median_depth_in_window(
    depth: np.ndarray,
    window: dict[str, float],
    *,
    min_depth_m: float,
    max_depth_m: float,
    center_fraction: float = 0.35,
) -> float | None:
    depth_2d = np.asarray(depth, dtype=np.float32)
    if depth_2d.ndim == 3 and depth_2d.shape[2] == 1:
        depth_2d = depth_2d[:, :, 0]
    if depth_2d.ndim != 2:
        return None
    height, width = depth_2d.shape[:2]
    cx = float(window["cx"])
    cy = float(window["cy"])
    crop_w = max(1.0, float(window["x1"]) - float(window["x0"]))
    crop_h = max(1.0, float(window["y1"]) - float(window["y0"]))
    half_w = max(1, int(round(crop_w * max(0.05, min(1.0, center_fraction)) * 0.5)))
    half_h = max(1, int(round(crop_h * max(0.05, min(1.0, center_fraction)) * 0.5)))
    x0 = max(0, min(width - 1, int(round(cx)) - half_w))
    x1 = max(x0 + 1, min(width, int(round(cx)) + half_w + 1))
    y0 = max(0, min(height - 1, int(round(cy)) - half_h))
    y1 = max(y0 + 1, min(height, int(round(cy)) + half_h + 1))
    values = depth_2d[y0:y1, x0:x1].reshape(-1)
    valid = values[
        np.isfinite(values)
        & (values >= float(min_depth_m))
        & (values <= float(max_depth_m))
    ]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def choose_validated_text_stop_approach_action(
    *,
    stop_validation_result: StopValidationResult | None,
    depth: np.ndarray | None,
    min_depth_m: float,
    max_depth_m: float,
    approach_min_depth_m: float,
    centered_x_tolerance: float,
    contact_press_remaining: int = 0,
    forward_blocked: bool = False,
) -> tuple[int | None, dict[str, Any]]:
    trace: dict[str, Any] = {
        "enabled": True,
        "checked": False,
        "forced": False,
    }
    if stop_validation_result is None or not bool(stop_validation_result.detection_found):
        trace["reject_reason"] = "no_detection"
        return None, trace
    bbox_xyxy = getattr(stop_validation_result, "detection_bbox_xyxy", None)
    bbox_center = getattr(stop_validation_result, "detection_bbox_center", None)
    if bbox_xyxy is None or bbox_center is None:
        trace["reject_reason"] = "missing_bbox"
        return None, trace
    try:
        x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
        center_x = float(bbox_center[0])
        center_y = float(bbox_center[1])
        area_fraction = float(getattr(stop_validation_result, "detection_bbox_area_fraction", 0.0) or 0.0)
    except (TypeError, ValueError, IndexError):
        trace["reject_reason"] = "invalid_bbox"
        return None, trace
    trace.update(
        {
            "checked": True,
            "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_center": [round(float(center_x), 4), round(float(center_y), 4)],
            "bbox_area_fraction": round(float(area_fraction), 4),
            "centered_x_tolerance": float(centered_x_tolerance),
            "approach_min_depth_m": float(approach_min_depth_m),
        }
    )
    x_offset = center_x - 0.5
    if abs(x_offset) > max(0.0, float(centered_x_tolerance)):
        action = int(HabitatSimActions.turn_left) if x_offset < 0.0 else int(HabitatSimActions.turn_right)
        trace.update(
            {
                "forced": True,
                "reason": "center_detection",
                "x_offset": round(float(x_offset), 4),
                "action": int(action),
            }
        )
        return action, trace
    if depth is None:
        trace["reject_reason"] = "no_depth"
        return None, trace
    depth_m = median_depth_in_window(
        depth,
        {
            "x0": float(x1),
            "y0": float(y1),
            "x1": float(x2),
            "y1": float(y2),
            "cx": 0.5 * float(x1 + x2),
            "cy": 0.5 * float(y1 + y2),
        },
        min_depth_m=float(min_depth_m),
        max_depth_m=float(max_depth_m),
        center_fraction=0.45,
    )
    trace["depth_m"] = None if depth_m is None else round(float(depth_m), 4)
    if depth_m is not None and float(depth_m) > float(approach_min_depth_m):
        action = int(HabitatSimActions.move_forward)
        trace.update(
            {
                "forced": True,
                "reason": "approach_detection",
                "action": int(action),
            }
        )
        return action, trace
    # Below the depth floor, press on while the contact budget lasts
    if int(contact_press_remaining) > 0 and not bool(forward_blocked):
        action = int(HabitatSimActions.move_forward)
        trace.update(
            {
                "forced": True,
                "reason": "press_to_contact",
                "contact_press_remaining": int(contact_press_remaining),
                "action": int(action),
            }
        )
        return action, trace
    if depth_m is None:
        trace["reject_reason"] = "no_valid_depth"
        return None, trace
    trace["reject_reason"] = "close_enough"
    return None, trace


def detection_projected_standpoint(
    agent_state: Any,
    stop_validation_result: Any,
    depth: np.ndarray | None,
    *,
    hfov_deg: float,
    min_depth_m: float,
    max_depth_m: float,
    standoff_m: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return a standpoint standoff_m short of the detection along its camera ray."""
    trace: dict[str, Any] = {"projected": False}
    if stop_validation_result is None or depth is None:
        trace["reject_reason"] = "no_detection_or_depth"
        return None, trace
    if not bool(getattr(stop_validation_result, "detection_found", False)):
        trace["reject_reason"] = "no_detection"
        return None, trace
    bbox_xyxy = getattr(stop_validation_result, "detection_bbox_xyxy", None)
    bbox_center = getattr(stop_validation_result, "detection_bbox_center", None)
    if bbox_xyxy is None or bbox_center is None:
        trace["reject_reason"] = "missing_bbox"
        return None, trace
    try:
        x1, y1, x2, y2 = [int(value) for value in bbox_xyxy]
        center_x = float(bbox_center[0])
    except (TypeError, ValueError, IndexError):
        trace["reject_reason"] = "invalid_bbox"
        return None, trace
    depth_m = median_depth_in_window(
        depth,
        {
            "x0": float(x1),
            "y0": float(y1),
            "x1": float(x2),
            "y1": float(y2),
            "cx": 0.5 * float(x1 + x2),
            "cy": 0.5 * float(y1 + y2),
        },
        min_depth_m=float(min_depth_m),
        max_depth_m=float(max_depth_m),
        center_fraction=0.45,
    )
    if depth_m is None or not math.isfinite(float(depth_m)):
        trace["reject_reason"] = "no_valid_depth"
        return None, trace
    position = np.asarray(agent_state.position, dtype=np.float32).reshape(-1)
    if position.size < 3 or not np.all(np.isfinite(position[:3])):
        trace["reject_reason"] = "invalid_agent_position"
        return None, trace
    try:
        rotation = quaternion.as_rotation_matrix(agent_state.rotation).astype(
            np.float32
        )
    except Exception:
        trace["reject_reason"] = "invalid_agent_rotation"
        return None, trace
    forward = rotation @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    right = rotation @ np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    forward_xz = np.asarray([forward[0], forward[2]], dtype=np.float32)
    right_xz = np.asarray([right[0], right[2]], dtype=np.float32)
    forward_norm = float(np.linalg.norm(forward_xz))
    right_norm = float(np.linalg.norm(right_xz))
    if forward_norm < 1e-6 or right_norm < 1e-6:
        trace["reject_reason"] = "degenerate_camera_axes"
        return None, trace
    forward_xz /= forward_norm
    right_xz /= right_norm
    # Planar bearing of the box-center column relative to the optical axis
    half_tan = math.tan(math.radians(max(1.0, float(hfov_deg))) * 0.5)
    bearing_rad = math.atan((2.0 * center_x - 1.0) * half_tan)
    direction_xz = (
        forward_xz * math.cos(bearing_rad) + right_xz * math.sin(bearing_rad)
    )
    direction_norm = float(np.linalg.norm(direction_xz))
    if direction_norm < 1e-6:
        trace["reject_reason"] = "degenerate_ray"
        return None, trace
    direction_xz /= direction_norm
    # Z-buffer depth, so the planar range along the ray is depth / cos(bearing)
    range_m = float(depth_m) / max(0.2, math.cos(bearing_rad))
    advance_m = max(0.0, range_m - max(0.0, float(standoff_m)))
    standpoint = position[:3].copy()
    standpoint[0] += float(direction_xz[0]) * advance_m
    standpoint[2] += float(direction_xz[1]) * advance_m
    trace.update(
        {
            "projected": True,
            "depth_m": round(float(depth_m), 4),
            "bearing_deg": round(math.degrees(bearing_rad), 2),
            "advance_m": round(float(advance_m), 4),
            "standoff_m": round(float(standoff_m), 4),
            "standpoint_m": [round(float(v), 4) for v in standpoint],
        }
    )
    return standpoint, trace


def encode_image_goal_similarity(
    goal_encoder: ImageGoalEncoder,
    image_rgb: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return goal_encoder.encode_image_embedding(image_rgb, device).reshape(-1)


def world_position_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        return None
    return [round(float(array[0]), 4), round(float(array[1]), 4), round(float(array[2]), 4)]


def agent_yaw_deg(agent_state: Any) -> float | None:
    rotation = getattr(agent_state, "rotation", None)
    if rotation is None:
        return None
    try:
        matrix = quaternion.as_rotation_matrix(rotation).astype(np.float32)
    except Exception:
        return None
    forward = matrix @ np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    yaw_rad = math.atan2(float(forward[0]), float(-forward[2]))
    yaw_deg = math.degrees(float(yaw_rad))
    if not math.isfinite(yaw_deg):
        return None
    return round(yaw_deg, 3)


def diagnostic_agent_state(env: Any) -> Any:
    """Ground-truth agent state, for diagnostics and rendering only."""

    getter = getattr(env, "get_ground_truth_agent_state", None)
    if callable(getter):
        return getter()
    return env.get_agent_state()


def rounded_planar_distance_xz(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    distance = planar_distance_xz(
        np.asarray(a, dtype=np.float32),
        np.asarray(b, dtype=np.float32),
    )
    if not math.isfinite(distance):
        return None
    return round(float(distance), 4)


def same_planar_goal(
    left: np.ndarray | None,
    right: np.ndarray | None,
    *,
    tolerance_m: float = 0.35,
) -> bool:
    if left is None or right is None:
        return False
    left_array = np.asarray(left, dtype=np.float32).reshape(-1)
    right_array = np.asarray(right, dtype=np.float32).reshape(-1)
    if left_array.size != 3 or right_array.size != 3:
        return False
    if not (np.all(np.isfinite(left_array)) and np.all(np.isfinite(right_array))):
        return False
    distance = float(np.linalg.norm(left_array[[0, 2]] - right_array[[0, 2]]))
    return distance <= float(tolerance_m)


def advance_sgf_blocked_goal_provenance(
    previous_goal: np.ndarray | None,
    previous_token: int | None,
    previous_count: int,
    *,
    forward_blocked: bool,
    sgf_owned: bool,
    current_goal: np.ndarray | None,
    current_token: int | None,
) -> tuple[np.ndarray | None, int | None, int]:
    """Count consecutive blocked forwards toward the same SGF goal."""

    if not forward_blocked or not sgf_owned or current_goal is None:
        return None, None, 0
    goal = np.asarray(current_goal, dtype=np.float32).reshape(-1)
    if goal.size != 3 or not np.all(np.isfinite(goal)):
        return None, None, 0
    previous = (
        np.asarray(previous_goal, dtype=np.float32).reshape(-1)
        if previous_goal is not None
        else np.empty((0,), dtype=np.float32)
    )
    same_generation = bool(
        previous.size == 3
        and np.all(np.isfinite(previous))
        and previous_token == current_token
        and float(np.linalg.norm(previous - goal)) <= 0.25
    )
    return (
        goal.copy(),
        None if current_token is None else int(current_token),
        int(previous_count) + 1 if same_generation else 1,
    )


def sgf_blocked_goal_failure_is_attributable(
    blocked_goal: np.ndarray | None,
    blocked_token: int | None,
    blocked_count: int,
    *,
    current_goal: np.ndarray | None,
    current_token: int | None,
    minimum_blocks: int,
) -> bool:
    """Whether recovery may retire the SGF epoch that caused its blocks."""

    if (
        blocked_goal is None
        or current_goal is None
        or int(blocked_count) < max(1, int(minimum_blocks))
        or blocked_token != current_token
    ):
        return False
    blocked = np.asarray(blocked_goal, dtype=np.float32).reshape(-1)
    current = np.asarray(current_goal, dtype=np.float32).reshape(-1)
    return bool(
        blocked.size == 3
        and current.size == 3
        and np.all(np.isfinite(blocked))
        and np.all(np.isfinite(current))
        and float(np.linalg.norm(blocked - current)) <= 0.25
    )


def full_turn_observation_actions(*, turn_angle_deg: float) -> int:
    """Number of turn actions needed to cover 360 degrees."""

    turn_angle = float(turn_angle_deg)
    if not math.isfinite(turn_angle) or turn_angle <= 0.0:
        raise ValueError("turn_angle_deg must be finite and positive")
    actions = 360.0 / turn_angle
    if not math.isfinite(actions):
        raise ValueError("turn_angle_deg is too small to represent a full turn")
    return max(1, int(math.ceil(actions)))


def panoramic_observation_interval_mode(*, searched_extent: bool) -> str:
    return "searched_extent" if bool(searched_extent) else "linear"


def accumulate_panoramic_path(
    *,
    previous_position: np.ndarray | None,
    current_position: np.ndarray,
    path_since_start_m: float,
) -> tuple[np.ndarray, float]:
    current = np.asarray(current_position, dtype=np.float32).reshape(-1)
    if current.size != 3 or not np.all(np.isfinite(current)):
        raise ValueError("current panoramic position must be a finite xyz vector")
    segment_m = 0.0
    if previous_position is not None:
        previous = np.asarray(previous_position, dtype=np.float32).reshape(-1)
        if previous.size != 3 or not np.all(np.isfinite(previous)):
            raise ValueError(
                "previous panoramic position must be a finite xyz vector"
            )
        segment_m = float(np.linalg.norm((current - previous)[[0, 2]]))
    return current.copy(), float(path_since_start_m) + segment_m


def panoramic_observation_trigger_due(
    *,
    remaining: int,
    anchor_position: np.ndarray | None,
    current_position: np.ndarray,
    trigger_distance_m: float,
) -> bool:
    if int(remaining) > 0:
        return False
    if anchor_position is None:
        return True
    anchor = np.asarray(anchor_position, dtype=np.float32).reshape(-1)
    current = np.asarray(current_position, dtype=np.float32).reshape(-1)
    if anchor.size != 3 or current.size != 3:
        raise ValueError("panoramic observation positions must contain x, y, and z")
    distance = float(np.linalg.norm((current - anchor)[[0, 2]]))
    return bool(distance >= float(trigger_distance_m))


def panoramic_observation_unknown_gate(
    *,
    unknown_cells_in_range: int | None,
    min_unknown_cells: int,
) -> bool:
    return bool(
        unknown_cells_in_range is None
        or int(unknown_cells_in_range) >= int(min_unknown_cells)
    )


def reconcile_panoramic_observation_action(
    *,
    proposal_action: int | None,
    proposal_selected: bool,
    final_action: int | None,
    sweep_turn_action: int,
    remaining: int,
) -> tuple[bool, int]:
    """Count a sweep turn against the budget only if it was executed."""

    remaining_actions = max(0, int(remaining))
    executed = bool(
        remaining_actions > 0
        and proposal_selected
        and proposal_action is not None
        and final_action is not None
        and int(proposal_action) == int(sweep_turn_action)
        and int(final_action) == int(sweep_turn_action)
    )
    return executed, remaining_actions - int(executed)


def earned_visual_bonus_is_eligible(
    *,
    enabled: bool,
    base_owner: GoalExpert | str,
    bonus_owner: GoalExpert | str,
    goal_end_reason: GoalEndReason | str | None,
    goal_end_sequence: int | None,
    goal_end_token: int | None,
    goal_end_acknowledged_actions: int,
    last_considered_sequence: int,
    target_unlocalized: bool,
    panoramic_corroborated: bool,
    nominee_failed_penalty: float,
) -> bool:
    """Whether the episode has earned the extra visual frontier share."""

    if not bool(enabled):
        return False
    if not isinstance(target_unlocalized, bool) or not target_unlocalized:
        return False
    if not isinstance(panoramic_corroborated, bool) or not panoramic_corroborated:
        return False
    if (
        isinstance(goal_end_sequence, bool)
        or not isinstance(goal_end_sequence, Integral)
        or int(goal_end_sequence) <= int(last_considered_sequence)
    ):
        return False
    if (
        isinstance(goal_end_token, bool)
        or not isinstance(goal_end_token, Integral)
    ):
        return False
    if (
        isinstance(goal_end_acknowledged_actions, bool)
        or not isinstance(goal_end_acknowledged_actions, Integral)
        or int(goal_end_acknowledged_actions) <= 0
    ):
        return False
    if (
        isinstance(nominee_failed_penalty, bool)
        or not isinstance(nominee_failed_penalty, Real)
    ):
        return False
    try:
        ordinary = GoalExpert(base_owner)
        bonus = GoalExpert(bonus_owner)
        reason = GoalEndReason(goal_end_reason)
        failed_penalty = float(nominee_failed_penalty)
    except (TypeError, ValueError):
        return False
    return bool(
        ordinary is GoalExpert.SAFE
        and bonus is GoalExpert.VISUAL
        and reason is GoalEndReason.REACHED
        and math.isfinite(failed_penalty)
        and failed_penalty == 0.0
    )


def node_surfel_positions(
    sgf: Any,
    *,
    node_id: int,
    min_confidence: float,
) -> np.ndarray:
    if sgf is None or int(node_id) < 0:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        # Unfiltered, so freshly observed nodes still have a footprint
        surfels = sgf.get_surfels(0.0, False)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    positions = np.asarray(surfels.get("positions_m"), dtype=np.float32)
    node_ids = np.asarray(surfels.get("node_ids"), dtype=np.int64)
    if positions.ndim != 2 or positions.shape[0] != node_ids.shape[0]:
        return np.zeros((0, 3), dtype=np.float32)
    return positions[node_ids == int(node_id)]


def summarize_hypothesis_projection(
    projection: ProjectedHypothesis | None,
    *,
    node_id: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "node_id": int(node_id),
        "projected": projection is not None,
    }
    if projection is not None:
        payload.update(
            {
                "bbox_xyxy": [int(value) for value in projection.bbox_xyxy],
                "area_fraction": round(float(projection.area_fraction), 5),
                "containment": round(float(projection.containment), 4),
                "depth_support": round(float(projection.depth_support), 4),
                "depth_samples": int(projection.depth_samples),
                "near_m": round(float(projection.near_m), 4),
                "far_m": round(float(projection.far_m), 4),
            }
        )
    return payload


def default_resolution_for_task_mode(task_mode: str) -> tuple[int, int]:
    if task_mode not in {"text", "instance-image"}:
        raise ValueError(f"Unsupported task mode: {task_mode!r}")
    return 640, 480


def target_found_for_task_mode(
    task_mode: str,
    *,
    text_target_found: bool,
    instance_image_target_found: bool,
) -> bool:
    if task_mode == "text":
        return bool(text_target_found)
    if task_mode == "instance-image":
        return bool(instance_image_target_found)
    raise ValueError(f"Unsupported task mode: {task_mode!r}")


def depth_cluster_pose_goal(
    target_node: dict[str, Any] | None,
    *,
    localizer_reject_reason: str | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Pursuit goal from the matched depth cluster (no goal-camera pose)."""

    trace: dict[str, Any] = {"localization": "depth_cluster", "accepted": False}
    if not isinstance(target_node, dict):
        trace["reject_reason"] = (
            "too_few_inliers"
            if localizer_reject_reason == "not_enough_ransac_inliers"
            else "no_reliable_cluster"
        )
        return None, trace
    for field in ("target_centroid_m", "centroid_m", "goal_position"):
        value = target_node.get(field)
        if value is None:
            continue
        try:
            point = np.asarray(value, dtype=np.float32).reshape(3)
        except (TypeError, ValueError):
            continue
        if np.all(np.isfinite(point)):
            trace.update({"accepted": True, "source_field": field})
            return point.copy(), trace
    trace["reject_reason"] = "invalid_centroid"
    return None, trace


def reliable_instance_image_target_localized(
    *,
    visual_match_navigation_enabled: bool,
    target_node: dict[str, Any] | None,
) -> bool:
    return bool(visual_match_navigation_enabled and target_node is not None)


def optional_output_path(value: str | Path | None, default_path: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return str(default_path)
    return str(Path(text).expanduser())


METRICS_ONLY_RESULT_KEYS = (
    "task_mode",
    "navigation_backend",
    "episode_id",
    "goal_key",
    "goal_name",
    "goal_image_source",
    "seed",
    "max_steps",
    "success_radius",
    "steps",
    "done",
    "stop_called",
    "target_found",
    "ablation_telemetry",
    "metrics",
)


def compact_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in METRICS_ONLY_RESULT_KEYS}


def _resolve_config_dimension(value: Any, *, default: int, name: str) -> int:
    if isinstance(value, str) and value.strip().lower() == "auto":
        return int(default)
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{NAV_CONFIG_SECTION}.{name} must be an integer or 'auto'") from exc
    if parsed <= 0:
        raise ValueError(f"{NAV_CONFIG_SECTION}.{name} must be positive or 'auto'")
    return parsed


def apply_nav_config(args: argparse.Namespace) -> argparse.Namespace:
    cli_overrides: dict[str, Any] = {}
    if args.max_steps is not None:
        cli_overrides["max_steps"] = args.max_steps
    for key in (
        "runtime_timing",
        "dump_sgf_inputs",
        "final_image_match",
        "final_image_match_path",
        "image_match_every_n_steps",
        "image_match_video",
        "image_match_video_fps",
        "image_match_video_path",
        "image_match_visual_every_n_steps",
        "image_match_visual_max_lines",
        "no_image_match_visuals",
        "sgf_input_dump_dir",
        "sgf_input_dump_every_n_steps",
        "overwrite_sgf_input_dump_dir",
        "sgf_input_replay_dir",
        "sgf_input_replay_stop_ends_episode",
    ):
        value = getattr(args, key, None)
        if value is not None:
            cli_overrides[key] = value

    resolved = resolve_nav_config(
        path=args.nav_config_path,
        task_mode=str(args.task_mode),
        real_env=bool(getattr(args, "real_env", False)),
        nav_overrides=getattr(args, "nav_override", ()),
        overrides=cli_overrides,
    )
    nav_config = resolved.as_dict()
    default_width, default_height = default_resolution_for_task_mode(
        str(args.task_mode)
    )
    for key in NAV_ARG_CONFIG_DEFAULTS:
        value = nav_config[key]
        if key == "width":
            value = _resolve_config_dimension(
                value, default=default_width, name=key
            )
        elif key == "height":
            value = _resolve_config_dimension(
                value, default=default_height, name=key
            )
        elif key == "device" and str(value).strip().lower() == "auto":
            value = "cuda" if torch.cuda.is_available() else "cpu"
        setattr(args, key, value)
    _episode_support.configure_nav_rules(nav_config)
    args.effective_nav_config = resolved
    args.nav_config_path = str(resolved.config_path)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SGF/FMM text or instance-image navigation on one Habitat episode."
    )
    parser.add_argument(
        "--nav-config-path",
        dest="nav_config_path",
        default=str(DEFAULT_NAV_CONFIG_PATH),
        help="YAML file containing the text_image_goal_nav strategy configuration.",
    )
    parser.add_argument(
        "--nav-override",
        action="append",
        default=[],
        help="Override a navigation config key as KEY=VALUE (repeatable).",
    )
    parser.add_argument("--config-path", default="benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml")
    parser.add_argument("--dataset-path", default=repo_data_path("datasets", "instance_imagenav", "hm3d", "v3", "{split}", "{split}.json.gz"))
    parser.add_argument("--textnav-dataset-path", default=repo_data_path("datasets", "textnav", "{split}", "{split}_text.json.gz"))
    parser.add_argument("--scenes-dir", default=repo_data_path("scene_datasets"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--task-mode", choices=("text", "instance-image"), default="instance-image")
    parser.add_argument("--goal-source", choices=("textnav",), default="textnav")
    parser.add_argument("--goal-text", default="")
    parser.add_argument("--goal-image-path", default="")
    parser.add_argument(
        "--real-env",
        action="store_true",
        help="Run against a real Kobuki base and ZED camera instead of Habitat.",
    )
    parser.add_argument("--real-goal-name", default="", help="Optional target category/name for real-env metadata.")
    parser.add_argument("--kobuki-device", default="/dev/kobuki")
    parser.add_argument("--kobuki-install", default="~/kobuki/install")
    parser.add_argument(
        "--kobuki-bridge-binary",
        default=str(REPO_ROOT / "bin" / "sgf-kobuki-json-bridge"),
        help="Path to the direct Kobuki JSON-lines bridge binary.",
    )
    parser.add_argument("--kobuki-linear-speed", type=float, default=0.15)
    parser.add_argument("--kobuki-angular-speed", type=float, default=0.6)
    parser.add_argument(
        "--real-forward-step-size",
        type=float,
        default=0.0,
        help="Real-robot forward distance per move action in meters (0: use forward_step_size from the navigation config).",
    )
    parser.add_argument(
        "--real-turn-angle",
        type=float,
        default=0.0,
        help="Real-robot turn angle per turn action in degrees (0: use turn_angle from the navigation config).",
    )
    parser.add_argument("--real-action-timeout", type=float, default=8.0)
    parser.add_argument("--real-settle-time", type=float, default=0.15)
    parser.add_argument("--real-control-period", type=float, default=0.03)
    parser.add_argument("--zed-resolution", default="HD720")
    parser.add_argument("--zed-fps", type=int, default=30)
    parser.add_argument("--zed-depth-mode", default="NEURAL")
    parser.add_argument("--zed-output-width", type=int, default=0)
    parser.add_argument("--zed-output-height", type=int, default=0)
    parser.add_argument("--zed-color-order", choices=("bgra", "rgba", "bgr", "rgb"), default="bgra")
    parser.add_argument("--zed-require-tracking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zed-camera-base-x", type=float, default=0.0)
    parser.add_argument("--zed-camera-base-y", type=float, default=0.0)
    parser.add_argument("--zed-camera-base-z", type=float, default=0.0)
    parser.add_argument("--zed-camera-base-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--zed-extrinsics-path", default="",
        help=(
            "Accepted camera-to-base calibration JSON for --real-env; "
            "overrides --zed-camera-base-* (ignored during SGF input replay)."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum episode steps (overrides text_image_goal_nav.max_steps from the navigation config).",
    )
    parser.add_argument(
        "--video-path",
        default="",
        help="Optional mp4 episode visualization using the SGF navigation map and agent path.",
    )
    parser.add_argument(
        "--final-navigation-map-path",
        default="",
        help="Optional output path for the final SGF navigation map visualization.",
    )
    parser.add_argument(
        "--final-image-match",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save the final instance-image feature-match visualization.",
    )
    parser.add_argument(
        "--final-image-match-path",
        default=None,
        help="Optional output path for the final instance-image feature-match visualization.",
    )
    parser.add_argument(
        "--image-match-video",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save an mp4 of instance-image feature-match visualizations.",
    )
    parser.add_argument(
        "--image-match-video-path",
        default=None,
        help="Optional output path for the instance-image feature-match video.",
    )
    parser.add_argument(
        "--image-match-video-fps",
        type=float,
        default=None,
        help="Frames per second for --image-match-video.",
    )
    parser.add_argument(
        "--image-match-every-n-steps",
        type=int,
        default=None,
        help="Run extra image matching every N steps when match artifacts are requested without visual navigation.",
    )
    parser.add_argument(
        "--image-match-visual-every-n-steps",
        type=int,
        default=None,
        help="Save image match PNGs every N steps under artifact_dir/image_matches (0 disables).",
    )
    parser.add_argument(
        "--image-match-visual-max-lines",
        type=int,
        default=None,
        help="Maximum match lines drawn in image match visualizations.",
    )
    parser.add_argument(
        "--no-image-match-visuals",
        dest="no_image_match_visuals",
        action="store_true",
        default=None,
        help="Disable final/per-step/video image match visualizations.",
    )
    parser.add_argument(
        "--image-match-visuals",
        dest="no_image_match_visuals",
        action="store_false",
        default=None,
        help="Allow image match visualizations (overrides no_image_match_visuals from the navigation config).",
    )
    parser.add_argument(
        "--image-dir",
        default="",
        help="Optional directory that saves per-step visualization PNGs.",
    )
    parser.add_argument(
        "--dump-sgf-inputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Dump the RGB-D inputs passed to SGF under the episode artifact directory.",
    )
    parser.add_argument(
        "--sgf-input-dump-dir",
        default=None,
        help="Directory for SGF input dumps; setting it enables dumping.",
    )
    parser.add_argument(
        "--sgf-input-dump-every-n-steps",
        type=int,
        default=None,
        help="Dump SGF inputs every N navigation steps (default: from the navigation config).",
    )
    parser.add_argument(
        "--overwrite-sgf-input-dump-dir",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow replacing an existing SGF input dump directory.",
    )
    parser.add_argument(
        "--sgf-input-replay-dir",
        default=None,
        help=(
            "Replay an SGF input directory recorded by --dump-sgf-inputs; "
            "actions are logged, but observations follow the recorded trajectory."
        ),
    )
    parser.add_argument(
        "--sgf-input-replay-stop-ends-episode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="End SGF input replay at the first STOP (default: replay all recorded frames).",
    )
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument(
        "--artifact-root",
        default=str(REPO_ROOT / "outputs" / "sgf_text_instance_image_nav_runs"),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--runtime-timing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Collect navigation module and detailed wall-clock timing (default: off).",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help=(
            "Write only compact metrics (to --output-json, else stdout); "
            "skip logs, traces, images, videos, and SGF dumps."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return apply_nav_config(parser.parse_args())


def config_value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any) -> Any:
    _ = config
    value = getattr(args, name.replace("-", "_"), None)
    if value is not None:
        return value
    return default


def bool_config_value(config: dict[str, Any], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def numeric_option(
    args: argparse.Namespace,
    nav_config: dict[str, Any],
    arg_name: str,
    config_name: str,
    default: int | float,
    cast,
) -> Any:
    value = getattr(args, arg_name, None)
    if value is None:
        value = nav_config.get(config_name, default)
    try:
        return cast(value)
    except (TypeError, ValueError):
        return cast(default)


def list_config_value(nav_config: dict[str, Any], name: str) -> list[str]:
    value = nav_config.get(name, [])
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def json_safe_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [json_safe_value(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_value(item) for item in value]
    return value


def candidate_association_distance(candidate: dict[str, Any]) -> float:
    """Geodesic distance, or planar distance under euclidean association."""

    value = candidate.get("association_distance_m", candidate.get("distance_m"))
    try:
        return float("inf") if value is None else float(value)
    except (TypeError, ValueError):
        return float("inf")


def summarize_text_sgf_target_node(node: dict[str, Any] | None) -> dict[str, Any] | None:
    if node is None:
        return None
    summary: dict[str, Any] = {}
    for key in (
        "id",
        "label",
        "hint_source",
        "num_surfels",
        "semantic_confidence",
        "distance_m",
        "association_distance_m",
        "association",
    ):
        if key not in node:
            continue
        value = node[key]
        if isinstance(value, (np.integer, int)):
            summary[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            summary[key] = (
                round(float(value), 4) if math.isfinite(float(value)) else None
            )
        else:
            summary[key] = value
    summary["goal_position_m"] = world_position_list(node.get("goal_position"))
    summary["centroid_m"] = world_position_list(node.get("centroid_m"))
    return summary


def summarize_navigation_command(command: Any | None) -> dict[str, Any] | None:
    if command is None:
        return None
    return {
        "action": int(command.action),
        "heading_error_deg": command.heading_error_deg,
        "goal_distance_m": command.goal_distance_m,
        "stalled_count": int(command.stalled_count),
        "replan_failed_count": int(command.replan_failed_count),
        "progress_stall_count": int(command.progress_stall_count),
        "planner_replan_failed": bool(command.planner_replan_failed),
        "goal_reached": bool(command.goal_reached),
        "goal_blocked": bool(command.goal_blocked),
        "local_blocked": bool(command.local_blocked),
        "block_scope": str(command.block_scope),
        "recovery_active": bool(command.recovery_active),
        "recovery_reason": command.recovery_reason,
        "short_term_goal_m": world_position_list(command.short_term_goal),
    }


def safe_non_stop_navigation_action(
    preferred_action: int | None = None,
) -> int:
    """Fall back to a non-STOP locomotion action after a rejected stop."""

    if preferred_action is not None:
        candidate = int(preferred_action)
        if candidate in {
            int(HabitatSimActions.move_forward),
            int(HabitatSimActions.turn_left),
            int(HabitatSimActions.turn_right),
        }:
            return candidate
    return int(HabitatSimActions.turn_left)


def unobserved_cells_in_range(
    navigation_map: Any,
    *,
    position: np.ndarray,
    range_m: float,
) -> int | None:
    """Unknown cells within range_m of position, or None without a map."""

    if navigation_map is None:
        return None
    try:
        unknown = np.asarray(navigation_map.unknown_mask())
        resolution_m = float(navigation_map.resolution_m)
        row, col = navigation_map.world_to_cell(
            np.asarray(position, dtype=np.float32).reshape(3)
        )
    except Exception:
        return None
    if unknown.ndim != 2 or unknown.size == 0 or resolution_m <= 0.0:
        return None
    radius_cells = max(1, int(round(float(range_m) / resolution_m)))
    rows, cols = unknown.shape
    row_min = max(0, int(row) - radius_cells)
    row_max = min(rows, int(row) + radius_cells + 1)
    col_min = max(0, int(col) - radius_cells)
    col_max = min(cols, int(col) + radius_cells + 1)
    if row_min >= row_max or col_min >= col_max:
        return 0
    window = unknown[row_min:row_max, col_min:col_max]
    grid_rows = np.arange(row_min, row_max, dtype=np.float32) - float(row)
    grid_cols = np.arange(col_min, col_max, dtype=np.float32) - float(col)
    inside = (
        grid_rows[:, None] ** 2 + grid_cols[None, :] ** 2
    ) <= float(radius_cells) ** 2
    return int(np.count_nonzero(np.asarray(window, dtype=bool) & inside))


def rotation_to_forward_xz(rotation) -> np.ndarray:
    rotation_matrix = quaternion.as_rotation_matrix(rotation).astype(np.float32)
    forward = rotation_matrix @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    forward_xz = np.asarray([forward[0], forward[2]], dtype=np.float32)
    norm = float(np.linalg.norm(forward_xz))
    if norm < 1e-6:
        return np.array([0.0, -1.0], dtype=np.float32)
    return forward_xz / norm


def weighted_match_center_x(result: Any) -> tuple[float | None, int]:
    """Confidence-weighted mean x of the inlier matches, and the inlier count."""

    try:
        matches = result.visualization.matches
        xy = np.asarray(matches.current_xy, dtype=np.float64).reshape(-1, 2)
        confidence = np.asarray(
            matches.fine_conf, dtype=np.float64
        ).reshape(-1)
        inliers = np.asarray(
            result.visualization.inlier_mask, dtype=bool
        ).reshape(-1)
        count = min(len(xy), len(confidence), len(inliers))
        accepted_xy = xy[:count][inliers[:count]]
        accepted_confidence = np.maximum(
            confidence[:count][inliers[:count]], 0.0
        )
        accepted_count = int(len(accepted_xy))
        if accepted_count <= 0:
            raise ValueError("no inlier matches")
        confidence_sum = float(np.sum(accepted_confidence))
        center_x = float(
            np.average(
                accepted_xy[:, 0],
                weights=(
                    accepted_confidence if confidence_sum > 1e-9 else None
                ),
            )
        )
    except (AttributeError, TypeError, ValueError):
        return None, 0
    return center_x, accepted_count


class VisualMemoryUtilityExplorer(UtilityExplorer):
    """Nonsemantic frontier utility that biases exploration toward image-similar views."""

    def __init__(
        self,
        navigation_map,
        *,
        visual_memory_min_similarity: float,
        visual_memory_frontier_min_score: float,
        visual_memory_radius_m: float,
        visual_memory_cone_deg: float,
        visual_memory_weight: float,
        visual_memory_area_weight: float,
        visual_memory_cost_weight: float,
        visual_memory_max_records: int,
        forward_step_size_m: float,
        turn_angle_deg: float,
        **kwargs,
    ) -> None:
        super().__init__(
            navigation_map,
            forward_step_size_m=float(forward_step_size_m),
            turn_angle_deg=float(turn_angle_deg),
            **kwargs,
        )
        self.visual_memory_min_similarity = float(visual_memory_min_similarity)
        self.visual_memory_frontier_min_score = float(visual_memory_frontier_min_score)
        self.visual_memory_radius_m = max(0.25, float(visual_memory_radius_m))
        self.visual_memory_cone_deg = max(1.0, min(180.0, float(visual_memory_cone_deg)))
        self.visual_memory_weight = float(visual_memory_weight)
        self.visual_memory_area_weight = float(visual_memory_area_weight)
        self.visual_memory_cost_weight = float(visual_memory_cost_weight)
        self.visual_memory_max_records = max(1, int(visual_memory_max_records))
        self._fairness_fixed_disputed_epoch_count = 0
        self._fairness_bonus_target_unlocalized = True
        self._fairness_bonus_panoramic_corroborated = False
        self._fairness_bonus_last_considered_sequence = 0
        self._fairness_bonus_latched = False
        self._fairness_bonus_activation_count = 0
        self._fairness_bonus_token: int | None = None
        self._fairness_bonus_epoch_count = 0
        self._fairness_bonus_action_count = 0
        self._fairness_bonus_trace: dict[str, Any] = {
            "enabled": True,
            "eligible": False,
            "applied": False,
            "reason": "episode_reset",
            "rejections": [],
        }
        self.visual_memory_config = VisualMemoryConfig(
            map_resolution_m=float(navigation_map.resolution_m),
            forward_step_size_m=float(forward_step_size_m),
            turn_angle_deg=float(turn_angle_deg),
            radius_m=self.visual_memory_radius_m,
            min_similarity=self.visual_memory_min_similarity,
            frontier_min_score=self.visual_memory_frontier_min_score,
            max_records=self.visual_memory_max_records,
            update_period_steps=1,
        )
        self.visual_memory = LeakyMaxVisualMemory(self.visual_memory_config)
        self.last_visual_frontier: dict[str, Any] | None = None
        self._goal_scheduler = TransactionalFairGoalEpochScheduler()
        self._goal_scheduler.set_weights(safe_weight=1, visual_weight=2)
        self._last_goal_end_event: GoalEndEvent | None = None
        self._goal_epoch_owner: str | None = None
        self._goal_epoch_token: int | None = None
        self.goal_epoch_owner_counts: Counter[str] = Counter()

    def reset(self) -> None:
        super().reset()
        self.goal_epoch_owner_counts = Counter()
        self.visual_memory = LeakyMaxVisualMemory(self.visual_memory_config)
        self.last_visual_frontier = None
        self._goal_scheduler.reset()
        self._goal_scheduler.set_weights(safe_weight=1, visual_weight=2)
        self._last_goal_end_event = None
        self._goal_epoch_owner = None
        self._goal_epoch_token = None
        self._fairness_fixed_disputed_epoch_count = 0
        self._fairness_bonus_target_unlocalized = True
        self._fairness_bonus_panoramic_corroborated = False
        self._fairness_bonus_last_considered_sequence = 0
        self._fairness_bonus_latched = False
        self._fairness_bonus_activation_count = 0
        self._fairness_bonus_token = None
        self._fairness_bonus_epoch_count = 0
        self._fairness_bonus_action_count = 0
        self._fairness_bonus_trace = {
            "enabled": True,
            "eligible": False,
            "applied": False,
            "reason": "episode_reset",
            "rejections": [],
        }

    def set_earned_visual_bonus_context(
        self,
        *,
        target_unlocalized: bool,
        panoramic_corroborated: bool,
    ) -> None:
        self._fairness_bonus_target_unlocalized = bool(target_unlocalized)
        self._fairness_bonus_panoramic_corroborated = bool(
            panoramic_corroborated
        )

    def _unconsumed_bonus_goal_end(self) -> GoalEndEvent | None:
        event = self._last_goal_end_event
        if (
            event is None
            or int(event.sequence)
            <= int(self._fairness_bonus_last_considered_sequence)
        ):
            return None
        return event

    def _consume_bonus_goal_end(self, event: GoalEndEvent | None) -> None:
        if event is None:
            return
        self._fairness_bonus_last_considered_sequence = max(
            int(self._fairness_bonus_last_considered_sequence),
            int(event.sequence),
        )

    def _goal_end_context(self) -> tuple[str | None, int | None]:
        return self._goal_epoch_owner, self._goal_epoch_token

    def _goal_end_acknowledged_actions(self) -> int:
        active = self._goal_scheduler.active
        return 0 if active is None else int(active.acknowledged_actions)

    def _on_goal_ended(self, event: GoalEndEvent) -> None:
        self._last_goal_end_event = event

    def _clear_current_goal(self) -> None:
        scheduler = getattr(self, "_goal_scheduler", None)
        if scheduler is not None:
            scheduler.close()
        super()._clear_current_goal()
        self._goal_epoch_owner = None
        self._goal_epoch_token = None

    def _set_current_goal(
        self,
        goal: np.ndarray | None,
        position: np.ndarray,
    ) -> np.ndarray | None:
        selected = super()._set_current_goal(goal, position)
        if selected is not None and self._goal_epoch_owner is None:
            # Free-space fallback goal (no frontier candidate) is owned by SAFE
            self._goal_scheduler.close()
            self._consume_bonus_goal_end(self._unconsumed_bonus_goal_end())
            lease = self._goal_scheduler.open_owned_epoch(
                GoalExpert.SAFE,
                counts_toward_service=False,
            )
            self._goal_epoch_owner = GoalExpert.SAFE.value
            self._goal_epoch_token = int(lease.token)
        return selected

    def update_visual_memory(
        self,
        *,
        position: np.ndarray,
        rotation,
        similarity: float,
        step_index: int,
    ) -> bool:
        forward_xz = rotation_to_forward_xz(rotation)
        return self.update_visual_memory_heading(
            position=position,
            heading_rad=math.atan2(
                -float(forward_xz[0]),
                -float(forward_xz[1]),
            ),
            similarity=similarity,
            step_index=step_index,
        )

    def update_visual_memory_heading(
        self,
        *,
        position: np.ndarray,
        heading_rad: float,
        similarity: float,
        step_index: int,
    ) -> bool:
        sim = float(similarity)
        if not math.isfinite(sim):
            return False
        position_arr = np.asarray(position, dtype=np.float32).reshape(-1)
        if position_arr.size != 3 or not np.all(np.isfinite(position_arr)):
            return False
        heading_rad = float(heading_rad)
        if not math.isfinite(heading_rad):
            return False
        update = self.visual_memory.update(
            position_xz=position_arr[[0, 2]],
            heading_rad=heading_rad,
            similarity=sim,
            step=int(step_index),
        )
        # Goal epochs are nonpreemptive unless a VISUAL goal loses its support
        if self._cached_visual_selection_changed():
            self.end_current_goal(GoalEndReason.SUPPORT_INVALIDATED)
        return bool(update.retained)

    def advance_visual_memory(self, step_index: int) -> None:
        self.visual_memory.advance(int(step_index))
        # Continuous decay may invalidate a VISUAL lease even without pruning
        if self._cached_visual_selection_changed():
            self.end_current_goal(GoalEndReason.SUPPORT_INVALIDATED)

    def _coverage_component_matrix(
        self,
        candidates: tuple[FrontierUtility, ...],
    ) -> np.ndarray:
        return self.frontier_utility_components(candidates)

    def _arbitrate_frontier_candidates(
        self,
        candidates: tuple[FrontierUtility, ...],
        coverage_index: int,
        records: tuple[Any, ...],
    ) -> ConfidenceConstrainedFrontierChoice:
        visual_support = np.stack(
            [
                self._frontier_visual_supports(candidate.goal, records)
                for candidate in candidates
            ],
            axis=0,
        )
        return confidence_constrained_frontier_choice(
            np.asarray([candidate.utility for candidate in candidates]),
            visual_support,
            self._coverage_component_matrix(candidates),
            baseline_index=int(coverage_index),
        )

    def _cached_visual_selection_changed(self) -> bool:
        return bool(
            self.current_goal is not None
            and self._goal_epoch_owner == GoalExpert.VISUAL.value
            and not self._visual_goal_has_qualifying_live_support()
        )

    def _visual_goal_has_qualifying_live_support(self) -> bool:
        if self.current_goal is None:
            return False
        records = self.visual_memory.effective_records()
        if not records:
            return False
        return bool(
            self._frontier_visual_score(self.current_goal, records=records)
            >= self.visual_memory_frontier_min_score
        )

    def _fair_scheduler_trace(
        self,
        *,
        drain_goal_events: bool = False,
    ) -> dict[str, Any]:
        snapshot = self._goal_scheduler.snapshot
        active = snapshot.active
        goal_end_events = (
            self.drain_goal_end_events() if drain_goal_events else ()
        )
        return {
            "allocation_policy": "fixed_transactional_fair",
            "services": {
                "safe": int(snapshot.safe_service),
                "visual": int(snapshot.visual_service),
            },
            "weights": {
                "safe": int(snapshot.safe_weight),
                "visual": int(snapshot.visual_weight),
            },
            "token": (
                None if self._goal_epoch_token is None else int(self._goal_epoch_token)
            ),
            "active_token": None if active is None else int(active.token),
            "active_owner": None if active is None else active.owner.value,
            "counts_toward_service": (
                False
                if active is None
                else bool(active.counts_toward_service)
            ),
            "acknowledged_actions": (
                0 if active is None else int(active.acknowledged_actions)
            ),
            "last_action_id": (
                None if active is None else active.last_action_id
            ),
            "last_resolved_action_id": snapshot.last_resolved_action_id,
            "next_token": int(snapshot.next_token),
            "next_tie_owner": snapshot.next_tie_owner.value,
            "earned_visual_bonus": {
                **dict(self._fairness_bonus_trace),
                "latched": bool(self._fairness_bonus_latched),
                "activation_count": int(
                    self._fairness_bonus_activation_count
                ),
                "epoch_count": int(self._fairness_bonus_epoch_count),
                "action_count": int(self._fairness_bonus_action_count),
                "target_unlocalized": bool(
                    self._fairness_bonus_target_unlocalized
                ),
                "panoramic_corroborated": bool(
                    self._fairness_bonus_panoramic_corroborated
                ),
                "last_considered_goal_end_sequence": int(
                    self._fairness_bonus_last_considered_sequence
                ),
            },
            "last_goal_end": (
                None
                if self._last_goal_end_event is None
                else {
                    "sequence": int(self._last_goal_end_event.sequence),
                    "reason": self._last_goal_end_event.reason.value,
                    "goal_m": list(self._last_goal_end_event.goal),
                    "owner": self._last_goal_end_event.owner,
                    "token": self._last_goal_end_event.token,
                    "acknowledged_actions": int(
                        self._last_goal_end_event.acknowledged_actions
                    ),
                }
            ),
            "goal_end_events": [
                {
                    "sequence": int(event.sequence),
                    "reason": event.reason.value,
                    "goal_m": list(event.goal),
                    "owner": event.owner,
                    "token": event.token,
                    "acknowledged_actions": int(
                        event.acknowledged_actions
                    ),
                }
                for event in goal_end_events
            ],
        }

    @property
    def goal_epoch_service_token(self) -> int | None:
        return (
            None if self._goal_epoch_token is None else int(self._goal_epoch_token)
        )

    def acknowledge_executed_sgf_proposal(
        self,
        token: int | None,
        action_id: int,
    ) -> bool:
        if token is None:
            return False
        charged = self._goal_scheduler.acknowledge(
            int(token),
            int(action_id),
            executed=True,
        )
        if charged and self._fairness_bonus_token == int(token):
            self._fairness_bonus_action_count += 1
        if self.last_visual_frontier is not None:
            self.last_visual_frontier["fairness"] = self._fair_scheduler_trace()
        return bool(charged)

    def _visual_memory_nominee(
        self,
        candidates: tuple[FrontierUtility, ...],
        records: tuple[Any, ...],
    ) -> tuple[int | None, dict[str, Any]]:
        eligible: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            visual_score = self._frontier_visual_score(
                candidate.goal,
                records=records,
            )
            if visual_score < self.visual_memory_frontier_min_score:
                continue

            revisit_penalty = 0.0
            if self.visited is not None:
                row, col = self.navigation_map.world_to_cell(candidate.goal)
                if (
                    0 <= row < self.visited.shape[0]
                    and 0 <= col < self.visited.shape[1]
                    and int(self.visited[row, col]) > 0
                ):
                    revisit_penalty = 1.0
            failed_penalty = (
                1.0 if self._is_near_failed_goal(candidate.goal) else 0.0
            )
            utility = (
                self.visual_memory_weight * visual_score
                + self.visual_memory_area_weight * math.log1p(float(candidate.area))
                - self.visual_memory_cost_weight * float(candidate.cost)
                - self.revisit_weight * revisit_penalty
                - self._failed_goal_repeat_penalty_scale * failed_penalty
            )
            eligible.append(
                {
                    "index": int(index),
                    "utility": float(utility),
                    "visual_score": float(visual_score),
                    "cost": float(candidate.cost),
                    "area": int(candidate.area),
                    "revisit_penalty": float(revisit_penalty),
                    "failed_penalty": float(failed_penalty),
                }
            )

        if not eligible:
            return None, {"eligible_count": 0, "nominee": None}
        best = max(eligible, key=lambda item: float(item["utility"]))
        return int(best["index"]), {
            "eligible_count": int(len(eligible)),
            "nominee": {
                key: (round(value, 6) if isinstance(value, float) else value)
                for key, value in best.items()
            },
        }

    def _schedule_frontier_nominees(
        self,
        safe_index: int,
        visual_index: int | None,
        *,
        visual_failed_penalty: float | None = None,
    ) -> tuple[int, str]:
        self._goal_scheduler.close()
        self._fairness_bonus_token = None
        bonus_goal_end = self._unconsumed_bonus_goal_end()
        if visual_index is None:
            lease = self._goal_scheduler.open_owned_epoch(
                GoalExpert.SAFE,
                counts_toward_service=False,
            )
            owner = GoalExpert.SAFE.value
            selected_index = int(safe_index)
            self._fairness_bonus_trace = {
                "enabled": True,
                "eligible": False,
                "applied": False,
                "reason": "no_visual_nominee",
                "rejections": ["no_visual_nominee"],
            }
        elif int(visual_index) == int(safe_index):
            lease = self._goal_scheduler.open_owned_epoch(
                GoalExpert.SAFE,
                counts_toward_service=False,
            )
            owner = "common"
            selected_index = int(safe_index)
            self._fairness_bonus_trace = {
                "enabled": True,
                "eligible": False,
                "applied": False,
                "reason": "common_nominee",
                "rejections": ["common_nominee"],
            }
        else:
            snapshot = self._goal_scheduler.snapshot
            base_safe_weight = int(snapshot.safe_weight)
            base_visual_weight = int(snapshot.visual_weight)
            if self._fairness_bonus_latched:
                lease = self._goal_scheduler.open_disputed_epoch()
                self._fairness_bonus_trace = {
                    "enabled": True,
                    "eligible": True,
                    "applied": True,
                    "latched": True,
                    "reason": "earned_visual_allocation_active",
                    "rejections": [],
                    "selected_owner": lease.owner.value,
                    "weights": {
                        "safe": base_safe_weight,
                        "visual": base_visual_weight,
                    },
                }
            else:
                base_owner = self._goal_scheduler.peek_disputed_owner()
                bonus_owner = self._goal_scheduler.peek_disputed_owner(
                    safe_weight=base_safe_weight,
                    visual_weight=base_visual_weight + 1,
                )
                failed_penalty = (
                    float(visual_failed_penalty)
                    if isinstance(visual_failed_penalty, Real)
                    and not isinstance(visual_failed_penalty, bool)
                    else float("nan")
                )
                bonus_eligible = earned_visual_bonus_is_eligible(
                    enabled=True,
                    base_owner=base_owner,
                    bonus_owner=bonus_owner,
                    goal_end_reason=(
                        None if bonus_goal_end is None else bonus_goal_end.reason
                    ),
                    goal_end_sequence=(
                        None if bonus_goal_end is None else bonus_goal_end.sequence
                    ),
                    goal_end_token=(
                        None if bonus_goal_end is None else bonus_goal_end.token
                    ),
                    goal_end_acknowledged_actions=(
                        0
                        if bonus_goal_end is None
                        else int(bonus_goal_end.acknowledged_actions)
                    ),
                    last_considered_sequence=int(
                        self._fairness_bonus_last_considered_sequence
                    ),
                    target_unlocalized=bool(
                        self._fairness_bonus_target_unlocalized
                    ),
                    panoramic_corroborated=bool(
                        self._fairness_bonus_panoramic_corroborated
                    ),
                    nominee_failed_penalty=failed_penalty,
                )
                rejections: list[str] = []
                if not (
                    base_owner is GoalExpert.SAFE
                    and bonus_owner is GoalExpert.VISUAL
                ):
                    rejections.append("extra_share_not_decisive")
                if bonus_goal_end is None:
                    rejections.append("goal_end_credit_unavailable")
                elif bonus_goal_end.reason is not GoalEndReason.REACHED:
                    rejections.append("previous_goal_not_reached")
                elif (
                    bonus_goal_end.token is None
                    or int(bonus_goal_end.acknowledged_actions) <= 0
                ):
                    rejections.append("previous_goal_not_action_backed")
                if not self._fairness_bonus_target_unlocalized:
                    rejections.append("target_already_localized")
                if not self._fairness_bonus_panoramic_corroborated:
                    rejections.append("panorama_not_pose_corroborated")
                if not math.isfinite(failed_penalty) or failed_penalty != 0.0:
                    rejections.append("failed_visual_nominee")
                if bonus_eligible:
                    self._goal_scheduler.set_weights(
                        safe_weight=base_safe_weight,
                        visual_weight=base_visual_weight + 1,
                    )
                    self._fairness_bonus_latched = True
                    self._fairness_bonus_activation_count += 1
                lease = self._goal_scheduler.open_disputed_epoch()
                self._fairness_bonus_trace = {
                    "enabled": True,
                    "eligible": bool(bonus_eligible),
                    "applied": bool(bonus_eligible),
                    "latched": bool(self._fairness_bonus_latched),
                    "reason": (
                        "earned_visual_allocation_activated"
                        if bonus_eligible
                        else (rejections[0] if rejections else "ineligible")
                    ),
                    "rejections": rejections,
                    "base_owner": base_owner.value,
                    "bonus_owner": bonus_owner.value,
                    "selected_owner": lease.owner.value,
                    "base_weights": {
                        "safe": base_safe_weight,
                        "visual": base_visual_weight,
                    },
                    "earned_weights": {
                        "safe": base_safe_weight,
                        "visual": base_visual_weight + 1,
                    },
                    "goal_end_sequence": (
                        None if bonus_goal_end is None else bonus_goal_end.sequence
                    ),
                    "goal_end_reason": (
                        None
                        if bonus_goal_end is None
                        else bonus_goal_end.reason.value
                    ),
                    "goal_end_acknowledged_actions": (
                        0
                        if bonus_goal_end is None
                        else int(bonus_goal_end.acknowledged_actions)
                    ),
                }
            if lease.owner is GoalExpert.VISUAL and self._fairness_bonus_latched:
                self._fairness_bonus_token = int(lease.token)
                self._fairness_bonus_epoch_count += 1
            self._fairness_fixed_disputed_epoch_count += 1
            owner = lease.owner.value
            selected_index = (
                int(safe_index)
                if lease.owner is GoalExpert.SAFE
                else int(visual_index)
            )
        self._consume_bonus_goal_end(bonus_goal_end)
        self._goal_epoch_token = int(lease.token)
        self._goal_epoch_owner = owner
        return int(selected_index), owner

    def visual_memory_stats(self) -> dict[str, Any]:
        stats = self.visual_memory.stats
        records = self.visual_memory.effective_records()
        return {
            "records": int(stats.active_records),
            "best_effective_quality": (
                None
                if stats.best_effective_quality is None
                else round(float(stats.best_effective_quality), 6)
            ),
            "half_life_steps": int(stats.half_life_steps),
            "insertions": int(stats.insertions),
            "merges": int(stats.merges),
            "pruned_records": int(stats.pruned_records),
            "evicted_records": int(stats.evicted_records),
            "median_age_steps": (
                None
                if not records or stats.current_step is None
                else round(
                    float(
                        np.median(
                            [
                                int(stats.current_step) - record.last_step
                                for record in records
                            ]
                        )
                    ),
                    2,
                )
            ),
            # Snapshot, since the live dict is mutated after acknowledgement
            "last_frontier": json_safe_value(self.last_visual_frontier),
            "fairness": self._fair_scheduler_trace(),
            "fairness_fixed_disputed_epoch_count": int(
                self._fairness_fixed_disputed_epoch_count
            ),
            "earned_visual_bonus": {
                "enabled": True,
                "latched": bool(self._fairness_bonus_latched),
                "activation_count": int(
                    self._fairness_bonus_activation_count
                ),
                "epoch_count": int(self._fairness_bonus_epoch_count),
                "action_count": int(self._fairness_bonus_action_count),
                "last_decision": dict(self._fairness_bonus_trace),
            },
        }

    @staticmethod
    def _frontier_choice_trace(
        choice: ConfidenceConstrainedFrontierChoice,
        candidates: tuple[FrontierUtility, ...],
        *,
        selected_index: int | None = None,
        safe_index: int | None = None,
        visual_index: int | None = None,
        owner: str | None = None,
        visual_nominee_trace: dict[str, Any] | None = None,
        fairness_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_nominee_index = (
            int(choice.selected_index) if safe_index is None else int(safe_index)
        )
        final_selected_index = (
            safe_nominee_index if selected_index is None else int(selected_index)
        )
        selected = candidates[final_selected_index]
        visual_top_index = choice.visual_top_index

        operator_decisions: dict[str, dict[str, Any]] = {}
        for operator_name, operator_trust in (
            ("product", choice.trust_product),
            ("minimum", choice.trust_minimum),
            ("geometric", choice.trust_geometric),
        ):
            if visual_top_index is None:
                top_trust = None
                top_allowed_regret = None
                top_admissible = None
                operator_selected_index = int(choice.baseline_index)
                structural_veto = True
            else:
                top_index = int(visual_top_index)
                top_trust = float(operator_trust[top_index])
                top_allowed_regret = (
                    top_trust * float(choice.component_contrast[top_index])
                )
                structural_veto = top_trust <= 0.0
                top_admissible = bool(
                    top_index == int(choice.baseline_index)
                    or (
                        not structural_veto
                        and float(choice.coverage_regret[top_index])
                        <= top_allowed_regret
                    )
                )
                operator_selected_index = (
                    top_index
                    if top_admissible
                    else int(choice.baseline_index)
                )
            operator_decisions[operator_name] = {
                "visual_top_trust": (
                    None if top_trust is None else round(top_trust, 6)
                ),
                "visual_top_allowed_regret": (
                    None
                    if top_allowed_regret is None
                    else round(top_allowed_regret, 4)
                ),
                "visual_top_admissible": top_admissible,
                "structural_veto": bool(structural_veto),
                "selected_index": int(operator_selected_index),
                "visual_override": bool(
                    operator_selected_index != int(choice.baseline_index)
                ),
            }

        visual_top: dict[str, Any] | None = None
        if visual_top_index is not None:
            top_index = int(visual_top_index)
            visual_top = {
                "index": top_index,
                "utility": round(float(candidates[top_index].utility), 4),
                "visual_score": round(
                    float(choice.visual_scores[top_index]), 6
                ),
                "effective_support": round(
                    float(choice.effective_support[top_index]), 4
                ),
                "corroboration": round(
                    float(choice.corroboration[top_index]), 6
                ),
                "selectivity": round(float(choice.selectivity), 6),
                "coverage_regret": round(
                    float(choice.coverage_regret[top_index]), 4
                ),
                "component_contrast": round(
                    float(choice.component_contrast[top_index]), 4
                ),
                "active_trust": round(float(choice.trust[top_index]), 6),
                "active_allowed_regret": round(
                    float(choice.allowed_regret[top_index]), 4
                ),
                "active_admissible": bool(choice.admissible[top_index]),
                "trust_product": round(
                    float(choice.trust_product[top_index]), 6
                ),
                "trust_minimum": round(
                    float(choice.trust_minimum[top_index]), 6
                ),
                "trust_geometric": round(
                    float(choice.trust_geometric[top_index]), 6
                ),
            }

        return {
            "utility": round(float(selected.utility), 4),
            "coverage_best": round(float(choice.coverage_best), 4),
            "visual_top_index": visual_top_index,
            "visual_top": visual_top,
            "visual_score": round(
                float(choice.visual_scores[final_selected_index]), 6
            ),
            "effective_support": round(
                float(choice.effective_support[final_selected_index]), 4
            ),
            "corroboration": round(
                float(choice.corroboration[final_selected_index]), 6
            ),
            "selectivity": round(float(choice.selectivity), 6),
            "trust": round(float(choice.trust[final_selected_index]), 6),
            "coverage_regret": round(
                float(choice.coverage_regret[final_selected_index]), 4
            ),
            "component_contrast": round(
                float(choice.component_contrast[final_selected_index]), 4
            ),
            "allowed_regret": round(
                float(choice.allowed_regret[final_selected_index]), 4
            ),
            "safe_visual_override": bool(
                choice.selected_index != choice.baseline_index
            ),
            "final_visual_override": bool(
                final_selected_index != choice.baseline_index
            ),
            "final_admissible": bool(choice.admissible[final_selected_index]),
            "coverage_index": int(choice.baseline_index),
            "safe_index": safe_nominee_index,
            "nominees": {
                "safe": safe_nominee_index,
                "visual": visual_index,
            },
            "owner": "safe" if owner is None else str(owner),
            "selected_index": final_selected_index,
            "visual_nominee": visual_nominee_trace,
            "fairness": fairness_trace,
            "admissible_count": int(sum(choice.admissible)),
            "candidate_count": int(len(candidates)),
            "active_trust_operator": "product",
            "operator_decisions": operator_decisions,
            "selection_mode": "episode_adaptive_transactional_fair_goal_epoch",
        }

    def _select_frontier_goal(self, position: np.ndarray) -> np.ndarray | None:
        # Called only at a goal boundary (new, reached, stalled, or failed)
        coverage_goal = super()._select_frontier_goal(position)
        candidates = self.last_frontier_candidates
        coverage_index = self.last_frontier_coverage_index
        if (
            coverage_goal is None
            or coverage_index is None
            or not candidates
        ):
            self.last_visual_frontier = None
            return coverage_goal

        records = self.visual_memory.effective_records()
        choice = self._arbitrate_frontier_candidates(
            candidates,
            int(coverage_index),
            records,
        )
        safe_index = int(choice.selected_index)
        visual_index: int | None = None
        visual_nominee_trace: dict[str, Any] = {
            "eligible_count": 0,
            "nominee": None,
        }
        if records:
            visual_index, visual_nominee_trace = (
                self._visual_memory_nominee(candidates, records)
            )
        visual_nominee_record = visual_nominee_trace.get("nominee")
        visual_failed_penalty = (
            visual_nominee_record.get("failed_penalty")
            if isinstance(visual_nominee_record, dict)
            else None
        )
        visual_nominee_trace["ccfa_admissible"] = (
            None
            if visual_index is None
            else bool(choice.admissible[int(visual_index)])
        )

        selected_index, owner = self._schedule_frontier_nominees(
            safe_index,
            visual_index,
            visual_failed_penalty=visual_failed_penalty,
        )
        self.goal_epoch_owner_counts[str(owner)] += 1
        visual_nominee_trace["authority"] = "transactional_fair_lease"

        selected = candidates[int(selected_index)]
        self.last_frontier_utility = selected
        self.last_visual_frontier = self._frontier_choice_trace(
            choice,
            candidates,
            selected_index=int(selected_index),
            safe_index=safe_index,
            visual_index=visual_index,
            owner=owner,
            visual_nominee_trace=visual_nominee_trace,
            fairness_trace=self._fair_scheduler_trace(),
        )
        return np.asarray(selected.goal, dtype=np.float32)

    def _frontier_visual_supports(
        self,
        candidate_world: np.ndarray,
        records: tuple[Any, ...] | None = None,
    ) -> np.ndarray:
        active_records = (
            self.visual_memory.effective_records()
            if records is None
            else records
        )
        candidate_xz = np.asarray(candidate_world, dtype=np.float32).reshape(3)[
            [0, 2]
        ]
        cos_half_cone = math.cos(
            math.radians(self.visual_memory_cone_deg * 0.5)
        )
        support = np.zeros((len(active_records),), dtype=np.float64)
        for index, record in enumerate(active_records):
            origin = np.asarray(record.position_xz, dtype=np.float32)
            delta = candidate_xz - origin
            distance = float(np.linalg.norm(delta))
            if distance < 1e-4 or distance > self.visual_memory_radius_m:
                continue
            direction = delta / distance
            forward = np.asarray(
                [
                    -math.sin(float(record.heading_rad)),
                    -math.cos(float(record.heading_rad)),
                ],
                dtype=np.float32,
            )
            dot = float(np.clip(np.dot(direction, forward), -1.0, 1.0))
            if dot < cos_half_cone:
                continue
            cone_score = (dot - cos_half_cone) / max(
                1e-6, 1.0 - cos_half_cone
            )
            distance_score = max(
                0.0, 1.0 - distance / self.visual_memory_radius_m
            )
            support[index] = float(record.effective_quality) * (
                0.7 * cone_score + 0.3 * distance_score
            )
        return np.clip(support, 0.0, 1.0)

    def _frontier_visual_score(
        self,
        candidate_world: np.ndarray,
        records: tuple[Any, ...] | None = None,
    ) -> float:
        support = self._frontier_visual_supports(candidate_world, records)
        return 0.0 if support.size == 0 else float(np.max(support))


def sgf_constructor_args(
    *,
    nav_config: dict[str, Any],
    width: int,
    height: int,
    camera_height: float,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        width=int(width),
        height=int(height),
        yolo_model=expand_repo_config_path(nav_config.get("yolo_model", str(REPO_ROOT / "models" / "yoloe-26s-seg.pt"))),
        python_executable=str(nav_config.get("python_executable", "python3") or "python3"),
        yolo_match_mode=str(nav_config.get("yolo_match_mode", "accurate_mask") or "accurate_mask"),
        sgf_min_pyr_level=int(nav_config.get("sgf_min_pyr_level", 2)),
        segment_filter=int(nav_config.get("segment_filter", 512)),
        sgf_depth_edge_threshold=float(nav_config.get("sgf_depth_edge_threshold", -1.0)),
        yolo_conf_threshold=float(nav_config.get("yolo_conf_threshold", 0.05)),
        yolo_iou_threshold=float(nav_config.get("yolo_iou_threshold", 0.5)),
        yolo_match_iou_threshold=float(nav_config.get("yolo_match_iou_threshold", 0.15)),
        navigation_map_resolution=float(nav_config.get("navigation_map_resolution", 0.05)),
        navigation_map_depth_stride=int(nav_config.get("navigation_map_depth_stride", 4)),
        navigation_map_max_range=float(nav_config.get("navigation_map_max_range", 5.0)),
        camera_height=float(camera_height),
        surfel_min_confidence=float(nav_config.get("surfel_min_confidence", 0.3)),
        navigation_obstacle_confirm_hits=int(nav_config.get("navigation_obstacle_confirm_hits", 2)),
        navigation_free_confirm_hits=int(nav_config.get("navigation_free_confirm_hits", 1)),
        navigation_obstacle_free_margin=int(nav_config.get("navigation_obstacle_free_margin", 0)),
    )


def _create_sgf_explorer(
    *,
    args: argparse.Namespace,
    nav_config: dict[str, Any],
    width: int,
    height: int,
    hfov: float,
    camera_height: float,
    env: TextGoalEpisodeEnv,
    semantic: bool,
    scene_graph_queries: list[str],
    goal_phrases: list[str],
    scales: PolicyScales,
) -> tuple[Any, SGFNavigationMap, UtilityExplorer, FMMGoalNavigator]:
    fx, fy, cx, cy = compute_intrinsics(width, height, hfov)
    sgf = create_online_sgf(
        args=sgf_constructor_args(
            nav_config=nav_config,
            width=width,
            height=height,
            camera_height=camera_height,
        ),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        scene_graph_queries=scene_graph_queries,
        enable_semantics=semantic,
    )
    navigation_map = SGFNavigationMap(sgf)
    navigation_map.graph_occupancy_stats()  # Fail early if the extension lacks this API
    navigator = FMMGoalNavigator(
        forward_step_size_m=env.forward_step_size_m,
        turn_angle_deg=env.turn_angle_deg,
        agent_radius_m=env.agent_radius_m,
        clearance_margin_m=float(
            nav_config.get("planner_clearance_margin_m", 0.05)
        ),
    )
    use_visual_memory = bool(not semantic and args.sgf_visual_memory)
    explorer_cls = VisualMemoryUtilityExplorer if use_visual_memory else UtilityExplorer
    explorer_kwargs: dict[str, Any] = {}
    if use_visual_memory:
        explorer_kwargs.update(
            visual_memory_min_similarity=float(
                args.sgf_visual_memory_min_similarity
            ),
            visual_memory_frontier_min_score=float(
                args.sgf_visual_memory_frontier_min_score
            ),
            visual_memory_radius_m=float(args.sgf_visual_memory_radius),
            visual_memory_cone_deg=float(args.sgf_visual_memory_cone_deg),
            visual_memory_weight=float(args.sgf_visual_memory_weight),
            visual_memory_area_weight=float(args.sgf_visual_memory_area_weight),
            visual_memory_cost_weight=float(args.sgf_visual_memory_cost_weight),
            visual_memory_max_records=int(args.sgf_visual_memory_max_records),
        )
    explorer = explorer_cls(
        navigation_map,
        visited_radius_cells=int(nav_config.get("visited_radius_cells", 6)),
        min_distance_cells=int(
            nav_config.get("explore_min_distance_cells", 24)
        ),
        min_goal_distance_m=float(args.sgf_explore_min_goal_distance),
        min_frontier_size=int(args.sgf_explore_frontier_min_size),
        failed_goal_radius_m=float(
            nav_config.get("explore_failed_goal_radius", 1.0)
        ),
        hfov_deg=hfov,
        max_range_m=float(nav_config.get("navigation_map_max_range", 5.0)),
        info_gain_weight=float(args.sgf_explore_info_gain_weight),
        goal_likelihood_weight=(
            float(
                nav_config.get(
                    "utility_explorer_goal_likelihood_weight", 150.0
                )
            )
            if semantic
            else 0.0
        ),
        revisit_weight=float(args.sgf_explore_revisit_weight),
        information_range_scale=float(args.sgf_explore_information_range_scale),
        forward_step_size_m=float(env.forward_step_size_m),
        turn_angle_deg=float(env.turn_angle_deg),
        utility_form=str(args.sgf_explore_utility_form),
        exchange_rate=float(args.sgf_explore_exchange_rate),
        exchange_rate_scale=float(args.sgf_explore_exchange_rate_scale),
        rotation_cost_weight=float(args.sgf_explore_rotation_cost_weight),
        goal_progress_min_improvement_m=float(scales.planar_progress_quantum_m),
        goal_reached_radius_m=float(scales.frontier_reach_radius_m),
        same_goal_distance_m=float(scales.progress_quantum_m),
        evidence_verification=bool(args.sgf_explore_evidence_verification),
        evidence_max_age_steps=int(scales.evidence_max_age_steps),
        **explorer_kwargs,
    )
    navigation_map.reset()
    navigation_map.set_floor_support(bool(args.navigation_floor_support))
    if not bool(args.navigation_graph_occupancy):
        # Depth-only map ablation (must follow reset())
        navigation_map.begin_provisional()
        if (
            not navigation_map.commit_provisional()
            or navigation_map.graph_occupancy_enabled
        ):
            raise RuntimeError(
                "navigation_graph_occupancy=false: could not establish a "
                "depth-only navigation map"
            )
    explorer.reset()
    explorer.update_target_queries(goal_phrases)
    navigator.reset()
    return sgf, navigation_map, explorer, navigator


def create_nonsemantic_sgf_explorer(
    *, args, nav_config, width, height, hfov, camera_height, env, scales
):
    return _create_sgf_explorer(
        args=args,
        nav_config=nav_config,
        width=width,
        height=height,
        hfov=hfov,
        camera_height=camera_height,
        env=env,
        semantic=False,
        scene_graph_queries=[],
        goal_phrases=[],
        scales=scales,
    )


def create_text_semantic_sgf_explorer(
    *,
    args,
    nav_config,
    width,
    height,
    hfov,
    camera_height,
    env,
    scene_graph_queries,
    goal_match_phrases,
    scales,
):
    return _create_sgf_explorer(
        args=args,
        nav_config=nav_config,
        width=width,
        height=height,
        hfov=hfov,
        camera_height=camera_height,
        env=env,
        semantic=True,
        scene_graph_queries=scene_graph_queries,
        goal_phrases=goal_match_phrases,
        scales=scales,
    )

def create_text_stop_validator(
    *,
    nav_config: dict[str, Any],
    detector_queries: list[str],
    canonical_goal: str,
    timing: RuntimeTiming | None = None,
) -> StopValidator:
    return StopValidator(
        timing=timing,
        yolo_model_path=expand_repo_config_path(
            nav_config.get("yolo_model", str(REPO_ROOT / "models" / "yoloe-26s-seg.pt"))
        ),
        detector_queries=detector_queries,
        fallback_queries=[canonical_goal] if canonical_goal else None,
        yolo_conf_threshold=float(nav_config.get("yolo_conf_threshold", 0.05)),
        yolo_iou_threshold=float(nav_config.get("yolo_iou_threshold", 0.5)),
        detector_device=str(nav_config.get("stop_verify_yolo_device", "auto") or "auto"),
        vlm_config=VisionLanguageConfig(
            config_path=(
                expand_repo_config_path(nav_config.get("stop_verify_config_path", ""))
                if str(nav_config.get("stop_verify_config_path", "") or "").strip()
                else ""
            ),
            base_url=str(nav_config.get("stop_verify_base_url", "") or ""),
            api_key=str(nav_config.get("stop_verify_api_key", "") or ""),
            model=str(nav_config.get("stop_verify_model", "") or ""),
            timeout_seconds=float(nav_config.get("stop_verify_timeout_seconds", 20.0)),
        ),
    )


def image_target_config_from_nav(nav_config: dict[str, Any], *, min_depth: float, max_depth: float) -> ImageTargetConfig:
    return ImageTargetConfig(
        min_matches=int(nav_config.get("image_min_matches", 40)),
        min_inliers=int(nav_config.get("image_min_inliers", 14)),
        fine_conf_threshold=float(nav_config.get("image_fine_conf_threshold", 0.25)),
        min_mean_confidence=float(nav_config.get("image_min_mean_confidence", 0.35)),
        min_confidence_sum=float(nav_config.get("image_min_confidence_sum", 12.0)),
        min_coarse_sim=float(nav_config.get("image_min_coarse_sim", 0.0)),
        goal_min_y_frac=float(nav_config.get("image_goal_min_y_frac", 0.0)),
        min_depth_m=float(min_depth),
        max_depth_m=float(max_depth),
    )


def create_xfeat_matcher(nav_config: dict[str, Any]) -> XFeatStarMatcher:
    return XFeatStarMatcher(
        resolve_xfeat_root(
            nav_config.get("xfeat_root", str(REPO_ROOT / ".." / "accelerated_features"))
        ),
        top_k=int(nav_config.get("xfeat_top_k", 8000)),
    )


def run_episode(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "effective_nav_config", None) is None:
        apply_nav_config(args)
    timing = RuntimeTiming(enabled=bool(args.runtime_timing))
    resolved_nav_config = args.effective_nav_config
    nav_config = (
        resolved_nav_config.as_dict()
        if hasattr(resolved_nav_config, "as_dict")
        else dict(resolved_nav_config)
    )
    _configure_determinism(int(args.seed))
    metrics_only = bool(getattr(args, "metrics_only", False))
    if metrics_only:
        args.verbose = False
        args.video_path = ""
        args.final_navigation_map_path = ""
        args.final_image_match = False
        args.final_image_match_path = ""
        args.image_match_video = False
        args.image_match_video_path = ""
        args.image_match_visual_every_n_steps = 0
        args.no_image_match_visuals = True
        args.image_dir = ""
        args.dump_sgf_inputs = False
        args.sgf_input_dump_dir = ""
    replay_input_dir = str(args.sgf_input_replay_dir or "").strip()
    replay_env = bool(replay_input_dir)
    physical_real_env = bool(args.real_env and not replay_env)
    real_env = bool(physical_real_env or replay_env)
    extrinsics_path = str(getattr(args, "zed_extrinsics_path", "") or "").strip()
    if extrinsics_path and replay_env:
        if args.verbose:
            print(
                "[sgf-input-replay] Ignoring --zed-extrinsics-path; using recorded camera poses.",
                file=sys.stderr,
                flush=True,
            )
    elif extrinsics_path:
        if not physical_real_env:
            raise ValueError("--zed-extrinsics-path requires a physical --real-env run")
        from sgf_habitat.real.calibration import load_camera_extrinsics

        extrinsics = load_camera_extrinsics(extrinsics_path)
        args.zed_camera_base_x = extrinsics["camera_base_x_m"]
        args.zed_camera_base_y = extrinsics["camera_base_y_m"]
        args.zed_camera_base_z = extrinsics["camera_base_z_m"]
        args.zed_camera_base_yaw_deg = extrinsics["camera_base_yaw_deg"]
    _episode_support.configure_nav_rules(nav_config)
    replay_metadata: dict[str, Any] = {}
    if replay_env:
        from sgf_habitat.real.replay import load_sgf_input_dump_metadata

        replay_metadata = load_sgf_input_dump_metadata(replay_input_dir)
    default_width, default_height = default_resolution_for_task_mode(str(args.task_mode))
    width = int(config_value(args, nav_config, "width", default_width))
    height = int(config_value(args, nav_config, "height", default_height))
    if physical_real_env:
        if int(args.zed_output_width) > 0:
            width = int(args.zed_output_width)
        if int(args.zed_output_height) > 0:
            height = int(args.zed_output_height)
    if replay_env:
        replay_width = int(replay_metadata.get("width") or 0)
        replay_height = int(replay_metadata.get("height") or 0)
        if replay_width > 0:
            width = replay_width
        if replay_height > 0:
            height = replay_height
    hfov = float(config_value(args, nav_config, "hfov", 79.0))
    if replay_env and replay_metadata.get("hfov") is not None:
        try:
            hfov = float(replay_metadata["hfov"])
        except (TypeError, ValueError):
            pass
    camera_height = float(config_value(args, nav_config, "camera_height", 0.88))
    min_depth = float(config_value(args, nav_config, "min_depth", 0.5))
    max_depth = float(config_value(args, nav_config, "max_depth", 5.0))
    if replay_env:
        try:
            min_depth = float(replay_metadata.get("min_depth_m", min_depth))
        except (TypeError, ValueError):
            pass
        try:
            max_depth = float(replay_metadata.get("max_depth_m", max_depth))
        except (TypeError, ValueError):
            pass
    turn_angle = float(config_value(args, nav_config, "turn_angle", 30.0))
    forward_step_size = float(config_value(args, nav_config, "forward_step_size", 0.25))
    if forward_step_size <= 0.0:
        forward_step_size = 0.25
    agent_radius = float(config_value(args, nav_config, "agent_radius", 0.18))
    if agent_radius <= 0.0:
        agent_radius = 0.18
    args.agent_radius = agent_radius
    if replay_env:
        replay_goal_name = str(replay_metadata.get("goal_name") or "").strip()
        if replay_goal_name and not str(args.real_goal_name or "").strip():
            args.real_goal_name = replay_goal_name
        if args.task_mode == "text" and not str(args.goal_text or "").strip():
            metadata_text_goal = str(replay_metadata.get("text_goal") or "").strip()
            if metadata_text_goal:
                args.goal_text = metadata_text_goal
            else:
                raise RuntimeError(
                    "text SGF input replay requires --goal-text or metadata.text_goal; "
                    "goal_name cannot be used as navigation text"
                )
        if args.task_mode == "instance-image" and not str(args.goal_image_path or "").strip():
            candidate_goal_image = Path(replay_input_dir).expanduser().parent / "goal_image.png"
            if candidate_goal_image.exists():
                args.goal_image_path = str(candidate_goal_image)
    if real_env:
        if args.task_mode == "text" and not str(args.goal_text or "").strip():
            raise RuntimeError("real-env text mode requires --goal-text")
        if args.task_mode == "instance-image" and not str(args.goal_image_path or "").strip():
            raise RuntimeError("real-env instance-image mode requires --goal-image-path")
        if float(args.real_forward_step_size) <= 0.0:
            args.real_forward_step_size = float(forward_step_size)
        if float(args.real_turn_angle) <= 0.0:
            args.real_turn_angle = float(turn_angle)
    success_radius = float(config_value(args, nav_config, "success_radius", 1.0))
    scales = PolicyScales(
        success_radius_m=success_radius,
        forward_step_m=forward_step_size,
        turn_angle_deg=turn_angle,
        min_depth_m=min_depth,
        max_depth_m=max_depth,
        hfov_deg=hfov,
        map_resolution_m=float(nav_config.get("navigation_map_resolution", 0.05)),
    )
    requery_interval_steps = max(1, int(args.text_stop_requery_interval_steps))
    video_fps = int(args.video_fps if args.video_fps is not None else nav_config.get("video_fps", 15))
    if video_fps <= 0:
        video_fps = 15
    frame_fx, frame_fy, frame_cx, frame_cy = compute_intrinsics(width, height, hfov)

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else Path(args.artifact_root) / f"episode_{args.episode_id:04d}"
    if os.environ.get("SGF_DEBUG_YOLO_MATCH"):  # diagnostics only: map this process to its episode
        try:
            Path(os.environ["SGF_DEBUG_YOLO_MATCH"]).mkdir(parents=True, exist_ok=True)
            (Path(os.environ["SGF_DEBUG_YOLO_MATCH"]) / f"{os.getpid()}.episode").write_text(str(int(args.episode_id)))
        except Exception:
            pass
    if not metrics_only:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_log_path = artifact_dir / "diagnostics.log"
    final_navigation_map_path = (
        Path(args.final_navigation_map_path).expanduser()
        if str(args.final_navigation_map_path or "").strip()
        else artifact_dir / "final_navigation_map.png"
    )
    image_match_artifacts_allowed = bool(
        not metrics_only
        and args.task_mode == "instance-image"
        and not bool(args.no_image_match_visuals)
    )
    final_image_match_requested = bool(
        image_match_artifacts_allowed
        and (
            bool(args.final_image_match)
            or bool(str(args.final_image_match_path or "").strip())
        )
    )
    configured_final_image_match_path = optional_output_path(
        args.final_image_match_path,
        artifact_dir / "final_image_match.png",
    )
    configured_image_match_video_path = ""
    if image_match_artifacts_allowed and (
        bool(args.image_match_video)
        or bool(str(args.image_match_video_path or "").strip())
    ):
        configured_image_match_video_path = optional_output_path(
            args.image_match_video_path,
            artifact_dir / "image_matches.mp4",
        )
    image_match_visual_interval = max(0, int(args.image_match_visual_every_n_steps))
    image_match_outputs_requested = bool(
        image_match_artifacts_allowed
        and (
            final_image_match_requested
            or bool(configured_image_match_video_path)
            or image_match_visual_interval > 0
        )
    )
    image_match_visual_dir = artifact_dir / "image_matches"
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    if replay_env:
        from sgf_habitat.real.replay import ReplaySGFInputEnv

        replay_goal_name = str(args.real_goal_name or replay_metadata.get("goal_name") or "").strip()
        if not replay_goal_name:
            replay_goal_name = "target" if args.task_mode == "instance-image" else str(args.goal_text or "target")
        env = ReplaySGFInputEnv(
            replay_input_dir,
            task_mode=str(args.task_mode),
            goal_text=str(args.goal_text or ""),
            goal_name=replay_goal_name,
            episode_id=int(args.episode_id),
            max_episode_steps=int(args.max_steps),
            forward_step_size_m=float(forward_step_size),
            turn_angle_deg=float(turn_angle),
            agent_radius_m=float(agent_radius),
            stop_ends_episode=bool(args.sgf_input_replay_stop_ends_episode),
        )
    elif physical_real_env:
        from sgf_habitat.real.env import create_real_zed_kobuki_env_from_args

        env = create_real_zed_kobuki_env_from_args(
            args,
            width=width,
            height=height,
            min_depth=min_depth,
            max_depth=max_depth,
        )
    else:
        env = TextGoalEpisodeEnv(
            config_path=args.config_path,
            dataset_path=args.dataset_path,
            scenes_dir=args.scenes_dir,
            split=args.split,
            image_width=width,
            image_height=height,
            hfov=hfov,
            camera_height=camera_height,
            min_depth=min_depth,
            max_depth=max_depth,
            turn_angle=turn_angle,
            forward_step_size=forward_step_size,
            agent_radius=agent_radius,
            max_episode_steps=args.max_steps,
            success_radius=success_radius,
            goal_text=args.goal_text,
            goal_source=args.goal_source,
            task_mode=args.task_mode,
            textnav_dataset_path=args.textnav_dataset_path,
            episode_id=args.episode_id,
            seed=int(args.seed),
        )
    navigation_backend = "sgf"
    image_goal_encoder = (
        ImageGoalEncoder(device)
        if args.task_mode == "instance-image"
        else None
    )

    trace: list[dict[str, Any]] = []
    action_source_counts: Counter[str] = Counter()
    action_kind_counts: dict[str, Counter[str]] = {}
    metrics: dict[str, Any] = {}
    stop_called = False
    # Which mechanism ended the episode (stop_called is also set by failsafes)
    terminal_stop_source = "none"
    terminal_stop_step = -1
    terminal_stop_vantages = -1
    last_ledger_record_vantages = -1
    last_ledger_record_step = -1
    sgf_explore_planning_errors = 0
    endgame_stop_kind = ""
    first_target_found_step = -1
    done = False
    steps = 0
    final_rgb: np.ndarray | None = None
    blocked_forward_steps = 0
    visual_pose_goal_position: np.ndarray | None = None
    visual_pose_goal_is_weak = False
    visual_pose_last_update_step = -1
    visual_pose_goal_metadata: dict[str, Any] = {}
    sgf_depth_guard_block_goal: np.ndarray | None = None
    sgf_depth_guard_block_token: int | None = None
    sgf_depth_guard_block_count = 0
    sgf_blocked_forward_goal: np.ndarray | None = None
    sgf_blocked_forward_token: int | None = None
    sgf_blocked_forward_count = 0
    path_positions: list[np.ndarray] = []
    target_reacquire_count = 0
    panorama_anchor_position: np.ndarray | None = None
    panorama_previous_position: np.ndarray | None = None
    panorama_sweep_views: list[tuple[float, np.ndarray, np.ndarray, Any, Any]] = []
    panorama_query_attempts = 0
    panorama_query_hits = 0
    foveated_query_attempts = 0
    foveated_query_visible = 0
    foveated_hypothesis_count = 0
    # VLM hypothesis trust: posterior mean under a uniform Beta prior
    vlm_trust_confirmed = 0
    vlm_trust_disconfirmed = 0
    vlm_trust_point: np.ndarray | None = None
    vlm_trust_arrival_deadline: int | None = None
    reachable_pursuit_action_count = 0
    goal_view_active_step_count = 0
    visual_pose_coverage_rejections = 0
    hypothesis_arrival_sweep_count = 0
    panorama_positive_origins: list[np.ndarray] = []
    panorama_pose_corroborated = False
    panorama_remaining = 0
    panorama_count = 0
    panorama_action_count = 0
    panoramic_detector_scaled_interval_application_count = 0
    panoramic_searched_extent_interval_application_count = 0
    panorama_path_since_start_m = 0.0
    visual_pose_goal_update_count = 0
    sgf_explore_evidence_hypothesis_count = 0
    # Goal bearing shift per servo turn, measured online to size the dead-band
    servo_prev_goal_xz: np.ndarray | None = None
    servo_prev_action_was_turn = False
    servo_turn_shift_samples: list[float] = []
    diagnostic_log_handle = None
    visualizer: EpisodeVisualizer | None = None
    image_dumper: EpisodeImageDumper | None = None
    image_match_video_writer: ImageMatchVideoWriter | None = None
    sgf_input_dumper: SGFInputDumper | None = None
    sgf_input_dump_dir_path: Path | None = None
    diagnostic_state_enabled = not bool(metrics_only)

    try:
        observations, episode_info = env.reset()
        if real_env:
            frame_fx, frame_fy, frame_cx, frame_cy = env.camera_intrinsics
            if physical_real_env and float(args.zed_camera_base_y) > 0.0:
                camera_height = float(args.zed_camera_base_y)
            hfov = math.degrees(2.0 * math.atan(float(width) / max(1e-6, 2.0 * float(frame_fx))))
        if args.video_path:
            visualizer = EpisodeVisualizer(
                args.video_path,
                fps=video_fps,
            )
        if args.image_dir:
            image_dumper = EpisodeImageDumper(
                args.image_dir,
                image_format=str(args.image_format),
                every_n_steps=max(1, int(args.image_every_n_steps)),
                overwrite=bool(args.overwrite_image_dir),
            )
            image_metadata = {
                "episode_id": episode_info.episode_id,
                "scene_id": episode_info.scene_id,
                "goal_key": episode_info.goal_key,
                "goal_name": episode_info.goal_name,
                "task_mode": args.task_mode,
                "map_source": "sgf_navigation_map",
                "navigation_map_resolution": float(nav_config.get("navigation_map_resolution", 0.05)),
                "map_frame": "observation_odometry_frame" if real_env else "episode_local_observation_frame",
                "image_format": str(args.image_format),
                "image_every_n_steps": max(1, int(args.image_every_n_steps)),
            }
            (Path(args.image_dir) / "metadata.json").write_text(
                json.dumps(image_metadata, indent=2),
                encoding="utf-8",
            )
        if configured_image_match_video_path:
            image_match_video_writer = ImageMatchVideoWriter(
                configured_image_match_video_path,
                fps=float(args.image_match_video_fps),
            )
        configured_sgf_input_dump_dir = str(args.sgf_input_dump_dir or "").strip()
        sgf_input_dump_enabled = bool(args.dump_sgf_inputs) or bool(configured_sgf_input_dump_dir)
        if sgf_input_dump_enabled:
            sgf_input_dump_dir_path = (
                Path(configured_sgf_input_dump_dir).expanduser()
                if configured_sgf_input_dump_dir
                else artifact_dir / "sgf_inputs"
            )
            sgf_input_dumper = SGFInputDumper(
                sgf_input_dump_dir_path,
                every_n_steps=max(1, int(args.sgf_input_dump_every_n_steps)),
                overwrite=bool(args.overwrite_sgf_input_dump_dir),
            )
            sgf_input_dumper.write_metadata(
                {
                    "episode_id": episode_info.episode_id,
                    "scene_id": episode_info.scene_id,
                    "goal_key": episode_info.goal_key,
                    "goal_name": episode_info.goal_name,
                    "text_goal": episode_info.text_goal,
                    "text_goal_source": episode_info.text_goal_source,
                    "task_mode": args.task_mode,
                    "width": width,
                    "height": height,
                    "hfov": hfov,
                    "fx": frame_fx,
                    "fy": frame_fy,
                    "cx": frame_cx,
                    "cy": frame_cy,
                    "min_depth_m": min_depth,
                    "max_depth_m": max_depth,
                    "every_n_steps": max(1, int(args.sgf_input_dump_every_n_steps)),
                }
            )
        gt_goal_viewpoint = (
            np.asarray(episode_info.gt_goal_viewpoint_m, dtype=np.float32)
            if episode_info.gt_goal_viewpoint_m is not None
            else None
        )
        if not metrics_only:
            diagnostic_log_handle = diagnostic_log_path.open("w", encoding="utf-8")
            diagnostic_log_handle.write(f"episode_id={episode_info.episode_id}\n")
            diagnostic_log_handle.write(f"scene_id={episode_info.scene_id}\n")
            diagnostic_log_handle.write(f"goal_name={episode_info.goal_name}\n")
            diagnostic_log_handle.write(f"gt_goal_position_m={episode_info.gt_goal_position_m}\n")
            diagnostic_log_handle.write(f"gt_goal_viewpoint_m={episode_info.gt_goal_viewpoint_m}\n")
            diagnostic_log_handle.write(
                f"map_source=sgf_navigation_map "
                f"resolution_m={float(nav_config.get('navigation_map_resolution', 0.05))}\n"
            )
            diagnostic_log_handle.flush()
        goal_image_source = ""
        goal_image_rgb: np.ndarray | None = None
        goal_image_vector: torch.Tensor | None = None
        xfeat_matcher: XFeatStarMatcher | None = None
        visual_target_config: ImageTargetConfig | None = None
        visual_intrinsics: CameraIntrinsics | None = None
        visual_goal_intrinsics: CameraIntrinsics | None = None
        sgf = None
        sgf_navigation_map: SGFNavigationMap | None = None
        sgf_explorer: UtilityExplorer | None = None
        sgf_navigator: FMMGoalNavigator | None = None
        visual_pursuit_navigator: FMMGoalNavigator | None = None
        visual_pursuit_planned = False
        visual_pursuit_standpoint: np.ndarray | None = None
        visual_pursuit_best_distance_m = float("inf")
        visual_pursuit_no_progress_steps = 0
        visual_pursuit_standpoint_reached_once = False
        servo_forward_blocked = False
        visual_pursuit_command: NavigationCommand | None = None
        text_sgf_enabled = False
        text_sgf_target_found = False
        instance_image_target_found = False
        parsed_text_goal = None
        parsed_text_goal_trace: dict[str, Any] = {}
        canonical_text_goal = ""
        detector_queries: list[str] = []
        scene_graph_queries: list[str] = []
        goal_match_phrases: list[str] = []
        stop_validator: StopValidator | None = None
        stop_decision_policy = StopDecisionPolicy(
            StopPolicyConfig(
                success_radius_m=float(scales.attempt_radius_m),
                far_band_radius_m=float(scales.far_band_radius_m),
                terminal_approach_distance_m=float(
                    scales.terminal_approach_distance_m
                ),
            )
        )
        stop_validation_attempts = 0
        stop_validation_rejections = 0
        hypothesis_map_resolution_m = float(
            nav_config.get("navigation_map_resolution", 0.05)
        )
        hypothesis_surfel_min_confidence = float(
            nav_config.get("surfel_min_confidence", 0.3)
        )
        hypothesis_pixel_stride = int(
            nav_config.get("navigation_map_depth_stride", 4)
        )
        text_hypothesis_projection_count = 0
        text_grounded_evidence_query_count = 0
        text_projection_evidence_query_count = 0
        text_triage_last_step: dict[int, int] = {}
        text_triage_queries = 0
        text_triage_exclusions = 0
        text_evidence_ranking_overrides = 0
        text_evidence_weighted_ranking_steps = 0
        text_navigation_map_association_steps = 0
        text_euclidean_association_steps = 0
        text_reachable_euclidean_association_steps = 0
        representation_activity = Counter({
            "navigation_graph_surfel_votes": 0,
            "navigation_graph_removed_free_cells": 0,
            "navigation_graph_added_occupied_cells": 0,
            "text_association_unreachable_candidates": 0,
            "text_association_rank_opportunities": 0,
            "text_association_rank_disagreements": 0,
        })
        text_query_conditional_labels = bool_config_value(
            nav_config, "text_query_conditional_labels", False
        )
        text_query_conditional_retrieval_steps = 0
        text_argmax_label_retrieval_steps = 0
        navigation_graph_occupancy_steps = 0
        navigation_floor_support_steps = 0
        navigation_floor_support_veto_cells = 0
        navigation_floor_support_potential_veto_cells = 0
        navigation_floor_support_nonfloor_cells = 0
        text_detection_instance_observations = 0
        text_detection_instance_candidate_count = 0
        text_detection_instance_selected_steps = 0
        text_target_association = str(args.text_target_association)
        if text_target_association not in ("navigation_map", "reachable_euclidean", "euclidean"):
            raise ValueError(
                "text_target_association must be 'navigation_map', 'reachable_euclidean' or "
                f"'euclidean', got {text_target_association!r}"
            )
        # Re-observation from a new vantage when the sequential test stalls
        text_reobservation: dict[str, Any] | None = None
        text_reobservation_counts: dict[int, int] = {}
        text_reobservation_launches = 0
        text_readmission_counts: dict[int, int] = {}
        text_readmissions = 0
        text_same_vantage_skips = 0
        text_reobservation_retirements = 0

        metric_visual_closure_armed_count = 0
        metric_visual_closure_commit_count = 0
        metric_visual_closure_withheld_count = 0
        direct_stop_withheld_count = 0
        metric_visual_closure_intervention_count = 0
        metric_visual_closure_recovery_monotonic_commit_count = 0
        metric_visual_closure_contextual_closeup_commit_count = 0
        last_stop_validation: dict[str, Any] = {}
        last_stop_policy_decision: dict[str, Any] = {}
        metric_visual_closure_enabled = bool(
            str(args.task_mode) == "text"
            and bool(args.text_metric_visual_closure)
        )
        metric_visual_closure_config = MetricVisualClosureConfig(
            grounded_detection_confidence_min=float(
                args.text_metric_visual_closure_grounded_detection_confidence_min
            ),
            contextual_bbox_area_max_fraction=float(
                args.text_metric_visual_closure_contextual_bbox_area_max_fraction
            ),
            grounded_bbox_center_y_min_fraction=float(
                args.text_metric_visual_closure_grounded_bbox_center_y_min_fraction
            ),
            grounded_target_distance_max_m=(
                float(
                    args.text_metric_visual_closure_grounded_success_radius_fraction
                )
                * float(success_radius)
            ),
        )
        metric_visual_closure_latch = (
            OnePrimitiveClosureLatch()
            if metric_visual_closure_enabled
            else None
        )
        image_pair_vlm_config = create_image_pair_vlm_config(nav_config)
        image_vlm_stop_validation_attempts = 0
        image_vlm_stop_validation_rejections = 0
        excluded_text_node_ids: set[int] = set()
        excluded_text_positions: list[np.ndarray] = []
        text_detection_registry: DetectionInstanceRegistry | None = None

        def retire_text_hypothesis(standpoint: np.ndarray | None) -> None:
            if standpoint is not None:
                excluded_text_positions.append(
                    np.asarray(standpoint, dtype=np.float32).copy()
                )

        text_rejection_near_success_counts: Counter[int] = Counter()
        tracked_text_target_node_id: int | None = None
        tracked_text_target_steps = 0
        text_target_navigation_active = False
        text_label_rejected_node_ids: dict[str, set[int]] = {}
        text_target_progress_node_id: int | None = None
        text_target_progress_best_distance = float("inf")
        text_target_progress_last_improvement_step = -1
        text_target_fatigue_retirements = 0
        relaxed_sgf_stop_cooldown_node_id: int | None = None
        relaxed_sgf_stop_cooldown_until_step = -1
        validated_text_stop_approach_forces = 0
        # Approach commitment: stop at this node unless informatively rejected
        text_stop_approach_commit_node_id = -1
        text_stop_approach_commit_probe_owned = False
        text_stop_approach_commit_step = -(10**9)
        # 'approval' commits may be cancelled, 'ledger' commits always stop
        text_stop_approach_commit_source = ""
        text_stop_recovery_action_count = 0
        text_stop_recovery_first_action: int | None = None
        recovery_approval_snapshot: MetricVisualClosureApprovalSnapshot | None = None
        last_raw_approval_snapshot: MetricVisualClosureApprovalSnapshot | None = None
        # Detection probe: validate a confident nearby detection directly
        detection_probe_last_step = -(10**9)
        detection_probe_last_check_step = -(10**9)
        detection_probe_check_interval_steps = max(
            1, int(scales.steps_for_distance(success_radius))
        )
        detection_probe_counter = 0
        text_stop_contact_press_steps = 0
        text_stop_contact_press_cap = int(scales.terminal_press_steps)
        observed_label_steps: Counter[str] = Counter()
        observed_label_max_surfels: dict[str, int] = {}
        observed_label_max_confidence: dict[str, float] = {}
        goal_like_label_steps: Counter[str] = Counter()
        goal_like_label_max_surfels: dict[str, int] = {}
        goal_like_label_max_confidence: dict[str, float] = {}
        goal_candidate_rejection_counts: Counter[str] = Counter()
        # Endgame return: bank the best-evidence pose, return when budget runs low
        endgame_return_planner: TerminalCommitPlanner | None = None
        endgame_return_navigator: FMMGoalNavigator | None = None
        if bool(args.endgame_return):
            endgame_candidate_floor = (
                float(args.sgf_visual_memory_min_similarity)
                if args.task_mode == "instance-image"
                else 0.0
            )
            endgame_return_planner = TerminalCommitPlanner(
                min_candidate_score=endgame_candidate_floor,
                margin_steps=int(args.endgame_return_margin_steps),
                fresh_evidence_window_steps=int(
                    args.endgame_return_fresh_evidence_window_steps
                ),
                arrival_radius_m=float(scales.terminal_approach_distance_m),
                forward_step_m=float(args.forward_step_size),
                score_advantage=float(args.endgame_return_score_advantage),
            )
            endgame_return_navigator = FMMGoalNavigator(
                forward_step_size_m=float(args.forward_step_size),
                turn_angle_deg=float(args.turn_angle),
                agent_radius_m=float(args.agent_radius),
                clearance_margin_m=float(
                    nav_config.get("planner_clearance_margin_m", 0.05)
                ),
            )
        endgame_return_stop_forced = False
        endgame_residual_walk_steps = 0
        endgame_nav_turn_actions = 0
        endgame_nav_forward_actions = 0
        endgame_arrival_overhead_steps = int(
            scales.full_turn_actions + scales.terminal_press_steps + 1
        )
        text_stop_ledger: TargetStopLedger | None = None
        # SPRT threshold log((1 - alpha) / alpha)
        sequential_error_rate = float(
            getattr(args, "text_stop_sequential_error_rate", 0.0) or 0.0
        )
        sequential_commit_log_odds = (
            math.log((1.0 - sequential_error_rate) / sequential_error_rate)
            if 0.0 < sequential_error_rate < 0.5
            else float("inf")
        )
        text_sequential_evidence_commit_count = 0
        ledger_press_steps = 0
        text_verification_object = str(args.text_stop_verification_object)
        text_evidence_rule = str(args.text_stop_evidence_rule)
        text_reobservation_mode = str(args.text_stop_reobservation)
        text_top_detection_queries = 0
        text_whole_frame_queries = 0
        text_evidence_rule_overrides = 0
        text_withheld_presses = 0
        text_face_node_turns = 0
        text_passive_releases = 0
        text_same_pose_requeries = 0
        text_shadow_stop: ShadowStopAdjudicator | None = None
        if args.task_mode == "text" and bool(args.text_stop_ledger):
            text_stop_ledger_kwargs: dict[str, Any] = dict(
                distinct_pose_distance_m=float(scales.distinct_view_distance_m),
                distinct_pose_yaw_deg=float(scales.distinct_view_yaw_deg),
                node_position_match_radius_m=float(
                    scales.hypothesis_exclusion_radius_m
                ),
                min_approval_step_separation=int(requery_interval_steps),
                min_commit_approvals=int(args.text_stop_ledger_min_commit_approvals),
                informative_negatives_for_exclusion=int(
                    args.text_stop_ledger_informative_negatives_for_exclusion
                ),
                absence_negatives_for_exclusion=int(
                    args.text_stop_ledger_absence_negatives_for_exclusion
                ),
                close_up_bbox_area_fraction=float(
                    args.text_stop_ledger_close_up_bbox_area_fraction
                ),
            )
            text_stop_ledger = TargetStopLedger(
                **text_stop_ledger_kwargs,
                commit_log_odds=float(sequential_commit_log_odds),
                rule=text_evidence_rule,
                pool_vantages=text_reobservation_mode != "same_pose",
            )
            text_shadow_stop = ShadowStopAdjudicator(
                ledger_kwargs=text_stop_ledger_kwargs,
                commit_log_odds=float(sequential_commit_log_odds),
            )
        if (
            args.task_mode == "instance-image"
            and bool(args.sgf_explore)
        ):
            sgf, sgf_navigation_map, sgf_explorer, sgf_navigator = create_nonsemantic_sgf_explorer(
                args=args,
                nav_config=nav_config,
                width=width,
                height=height,
                hfov=hfov,
                camera_height=camera_height,
                env=env,
                scales=scales,
            )
            visual_pursuit_navigator = FMMGoalNavigator(
                forward_step_size_m=env.forward_step_size_m,
                turn_angle_deg=env.turn_angle_deg,
                agent_radius_m=env.agent_radius_m,
                clearance_margin_m=float(
                    nav_config.get("planner_clearance_margin_m", 0.05)
                ),
            )
            if bool(getattr(args, "image_vlm_hypothesis_trust", False)) and hasattr(
                sgf_explorer, "verified_preempt_min_quality"
            ):
                # Preempt only if more likely right than wrong
                sgf_explorer.verified_preempt_min_quality = 0.5
        image_target_localization = str(args.image_target_localization)
        image_navigation_target = str(args.image_navigation_target)
        image_pursuit_controller = str(args.image_pursuit_controller)
        image_goal_view_alignment = str(args.image_goal_view_alignment)
        image_object_target_steps = 0
        image_goal_camera_target_steps = 0
        image_object_anchor_missing_steps = 0
        image_pursuit_fmm_actions = 0
        image_straight_servo_actions = 0
        image_depth_cluster_target_updates = 0
        image_depth_cluster_alignment_turns = 0
        image_single_corroboration_commits = 0
        goal_view_belief: GoalViewBelief | None = None
        if (
            args.task_mode == "instance-image"
            and bool(getattr(args, "image_goal_view_terminal", False))
            # The belief needs goal-camera PnP solutions
            and image_target_localization == "pnp_goal_view"
        ):
            goal_view_belief = GoalViewBelief(
                success_radius_m=float(success_radius),
                forward_step_m=float(forward_step_size),
                min_depth_m=float(min_depth),
                max_depth_m=float(max_depth),
                hfov_deg=float(hfov),
                turn_angle_deg=float(turn_angle),
                miss_budget_frames=int(scales.evidence_max_age_steps),
                min_corroboration=int(args.image_goal_view_min_corroboration),
                commit_band=bool(args.image_goal_view_commit_band),
            )
        visual_match_navigation_enabled = False
        if args.task_mode == "instance-image":
            if image_goal_encoder is None:
                raise RuntimeError("instance-image mode requires the CLIP image encoder")
            goal_image_rgb = load_goal_image(args.goal_image_path) if args.goal_image_path else episode_info.goal_image_rgb
            if goal_image_rgb is None:
                raise RuntimeError("instance-image mode requires a dataset goal image or --goal-image-path")
            goal_image_source = f"file:{Path(args.goal_image_path).expanduser()}" if args.goal_image_path else episode_info.goal_image_source
            if not metrics_only:
                save_rgb(artifact_dir / "goal_image.png", goal_image_rgb)
            goal_image_vector = encode_image_goal_similarity(
                image_goal_encoder,
                goal_image_rgb,
                device,
            )
            visual_match_navigation_enabled = bool(args.visual_pose_servo)
            xfeat_matcher_needed = bool(
                visual_match_navigation_enabled
                or image_match_outputs_requested
            )
            if xfeat_matcher_needed:
                xfeat_matcher = create_xfeat_matcher(nav_config)
                visual_intrinsics = CameraIntrinsics(fx=frame_fx, fy=frame_fy, cx=frame_cx, cy=frame_cy)
                goal_hfov = episode_info.goal_image_hfov_deg or hfov
                goal_fx, goal_fy, goal_cx, goal_cy = compute_intrinsics(
                    int(goal_image_rgb.shape[1]),
                    int(goal_image_rgb.shape[0]),
                    float(goal_hfov),
                )
                visual_goal_intrinsics = CameraIntrinsics(
                    fx=goal_fx,
                    fy=goal_fy,
                    cx=goal_cx,
                    cy=goal_cy,
                )
                visual_target_config = image_target_config_from_nav(
                    nav_config,
                    min_depth=min_depth,
                    max_depth=max_depth,
                )
        if args.task_mode == "text":
            text_sgf_enabled = True
            goal_parser_config = GoalParserConfig(
                config_path=(
                    expand_repo_config_path(nav_config.get("llm_config_path", ""))
                    if str(nav_config.get("llm_config_path", "") or "").strip()
                    else ""
                ),
                base_url=str(nav_config.get("llm_base_url", "") or ""),
                api_key=str(nav_config.get("llm_api_key", "") or ""),
                model=str(nav_config.get("llm_model", "") or ""),
                timeout_seconds=float(nav_config.get("llm_timeout_seconds", 20.0)),
                max_phrases=int(nav_config.get("goal_query_limit", 4)),
                mode=str(nav_config.get("text_goal_parser_mode", "llm")),
            )
            parsed_text_goal = parse_text_goal(
                episode_info.text_goal,
                config=goal_parser_config,
            )
            canonical_text_goal = parsed_text_goal.canonical_goal
            detector_queries, scene_graph_queries, goal_match_phrases = build_query_sets(
                explicit_queries=(
                    list_config_value(nav_config, "yolo_class")
                    if goal_parser_config.mode == "llm"
                    else []
                ),
                parsed_goal=parsed_text_goal,
                canonical_goal=canonical_text_goal,
            )
            # Context phrases guide search only, not target selection or stop checks
            context_guided_exploration = (
                goal_parser_config.mode == "llm"
                and bool_config_value(
                    nav_config, "text_context_guided_exploration", False
                )
            )
            context_phrases: list[str] = []
            if context_guided_exploration:
                context_phrases = [
                    phrase
                    for phrase in dedupe_goal_phrases(
                        getattr(parsed_text_goal, "context_phrases", []) or []
                    )
                    if phrase not in set(scene_graph_queries)
                ]
                if context_phrases:
                    scene_graph_queries = list(scene_graph_queries) + list(
                        context_phrases
                    )
            exploration_cue_phrases = list(goal_match_phrases) + list(context_phrases)
            parsed_text_goal_trace = {
                "enabled": True,
                "parser_mode": goal_parser_config.mode,
                "canonical_goal": canonical_text_goal,
                "parsed_target_phrases": (
                    list(parsed_text_goal.target_phrases)
                    if goal_parser_config.mode == "llm"
                    else []
                ),
                "target_phrases": list(parsed_text_goal.target_phrases),
                "detector_queries": list(detector_queries),
                "scene_graph_queries": list(scene_graph_queries),
                "goal_match_phrases": list(goal_match_phrases),
                "context_guided_exploration": bool(context_guided_exploration),
                "context_phrases": list(context_phrases),
            }
            sgf, sgf_navigation_map, sgf_explorer, sgf_navigator = (
                create_text_semantic_sgf_explorer(
                    args=args,
                    nav_config=nav_config,
                    width=width,
                    height=height,
                    hfov=hfov,
                    camera_height=camera_height,
                    env=env,
                    scene_graph_queries=scene_graph_queries,
                    goal_match_phrases=goal_match_phrases,
                    scales=scales,
                )
            )
            if context_phrases:
                sgf_explorer.update_target_queries(exploration_cue_phrases)
            try:
                sgf_navigation_map.set_query_conditional_labels(
                    text_query_conditional_labels
                )
            except AttributeError as exc:
                raise RuntimeError(
                    "text_query_conditional_labels requires the _sgf extension "
                    "built from this source tree (python3 tools/build_extension.py)"
                ) from exc
            try:
                sgf_navigation_map.set_target_association(text_target_association)
            except AttributeError as exc:
                if text_target_association != "navigation_map":
                    raise RuntimeError(
                        "text_target_association="
                        f"{text_target_association!r} requires the _sgf extension "
                        "built from this source tree (python3 tools/build_extension.py)"
                    ) from exc
            if bool_config_value(nav_config, "text_detection_instances", False):
                text_detection_registry = DetectionInstanceRegistry(
                    merge_radius_m=float(scales.hypothesis_exclusion_radius_m),
                    min_observations=int(
                        nav_config.get("text_detection_instance_min_observations", 2)
                    ),
                )
            if bool(args.stop_validation):
                stop_validator = create_text_stop_validator(
                    nav_config=nav_config,
                    detector_queries=detector_queries,
                    canonical_goal=canonical_text_goal,
                    timing=timing,
                )
        else:
            parsed_text_goal_trace = {
                "enabled": False,
                "canonical_goal": "",
            }
        last_visualization_mode = "sgf_scan"
        last_visualization_action = int(HabitatSimActions.stop)
        last_visualization_step_record: dict[str, Any] = {}
        last_image_match_visualization = None
        last_image_match_current_rgb: np.ndarray | None = None
        last_image_match_diagnostics: dict[str, Any] | None = None
        last_image_match_step = -1
        last_image_match_target_node: dict[str, Any] | None = None
        best_image_match_visualization = None
        best_image_match_current_rgb: np.ndarray | None = None
        best_image_match_diagnostics: dict[str, Any] | None = None
        best_image_match_step = -1
        best_image_match_target_node: dict[str, Any] | None = None
        best_image_match_quality: tuple[float, int] | None = None

        timing.start_navigation()
        for step_index in range(max(1, int(args.max_steps))):
            timing.start_step()
            # Initialized here because target selection reads it before arbitration
            forced_budget_stop = False
            metric_visual_closure_commit_due = (
                None
                if metric_visual_closure_latch is None
                else metric_visual_closure_latch.pop_commit()
            )
            metric_visual_closure_decision = None
            metric_visual_direct_closure_decision = None
            metric_visual_direct_closure_kind = ""
            closure_evidence_ok = True
            closure_evidence_log_odds = None
            recovery_monotonic_closure_decision = None
            contextual_closeup_closure_decision = None
            # Step-local until the RECOVER action has executed
            recovery_approval_snapshot_candidate: (
                MetricVisualClosureApprovalSnapshot | None
            ) = None
            recovery_approval_proposed_action: int | None = None
            metric_visual_closure_commit_forced = False
            metric_visual_closure_intervened = False
            metric_visual_closure_trace: dict[str, Any] = {
                "enabled": bool(metric_visual_closure_enabled),
                "pending_before": bool(
                    metric_visual_closure_commit_due is not None
                ),
            }
            if metric_visual_closure_commit_due is not None:
                metric_visual_closure_trace["pending"] = dict(
                    metric_visual_closure_commit_due
                )
            loop_agent_state = env.get_agent_state()
            loop_agent_position = np.asarray(
                loop_agent_state.position, dtype=np.float32
            ).reshape(3)
            diagnostic_loop_agent_position = (
                np.asarray(
                    diagnostic_agent_state(env).position, dtype=np.float32
                ).reshape(3)
                if diagnostic_state_enabled
                else loop_agent_position.copy()
            )
            gt_goal_viewpoint_planar_distance_m = (
                rounded_planar_distance_xz(
                    diagnostic_loop_agent_position,
                    gt_goal_viewpoint,
                )
                if diagnostic_state_enabled
                else None
            )
            path_positions.append(loop_agent_position.copy())
            rgb_observation = np.asarray(observations["rgb"], dtype=np.uint8)[:, :, :3]
            final_rgb = rgb_observation
            current_image_vector_cache: torch.Tensor | None = None
            current_goal_image_similarity_cache: float | None = None

            def current_goal_image_similarity_for_frame() -> float | None:
                nonlocal current_image_vector_cache
                nonlocal current_goal_image_similarity_cache
                if goal_image_vector is None:
                    return None
                if current_goal_image_similarity_cache is None:
                    with timing.measure("image_goal", "clip_encoding_similarity"):
                        if image_goal_encoder is None:
                            raise RuntimeError("instance-image mode requires the CLIP image encoder")
                        current_image_vector_cache = image_goal_encoder.encode_image_embedding(
                            rgb_observation, device
                        ).reshape(-1)
                        current_goal_image_similarity_cache = float(
                            torch.dot(
                                current_image_vector_cache,
                                goal_image_vector,
                            )
                            .detach()
                            .cpu()
                        )
                return current_goal_image_similarity_cache
            visual_stop_trace: dict[str, Any] = {"attempted": False, "target_updated": False}
            image_space_match_evidence: ImageSpaceMatchEvidence | None = None
            image_match_visualization_trace: dict[str, Any] = {
                "enabled": bool(image_match_outputs_requested),
                "attempted": False,
            }
            visual_pose_trace: dict[str, Any] = {
                "enabled": bool(args.visual_pose_servo),
                "attempted": False,
            }
            visual_pose_navigation_trace: dict[str, Any] = {"active": False}

            visual_stop_ready = False
            goal_view_confirmed = None
            goal_view_trace: dict[str, Any] = {
                "enabled": bool(goal_view_belief is not None),
            }
            visual_match_due = (
                bool(visual_match_navigation_enabled)
                and xfeat_matcher is not None
                and visual_target_config is not None
                and visual_intrinsics is not None
            )
            image_match_save_due = (
                image_match_outputs_requested
                and not bool(visual_match_navigation_enabled)
                and xfeat_matcher is not None
                and visual_target_config is not None
                and visual_intrinsics is not None
                and step_index % max(1, int(args.image_match_every_n_steps)) == 0
            )
            sgf_schedule_active = bool(args.sgf_explore)
            depth = None
            world_from_camera_cv = None
            sensor_state = None
            world_projection_required = bool(
                sgf is not None
                or visual_match_due
                or image_match_save_due
                or (bool(args.sgf_depth_guard) and sgf_schedule_active)
            )
            if world_projection_required:
                depth = depth_observation_to_2d(observations["depth"])
            if world_projection_required:
                sensor_state = env.get_sensor_state("depth")
                world_from_camera_cv = sensor_state_to_world_from_camera_cv(sensor_state)
            sgf_explore_trace: dict[str, Any] = {
                "enabled": bool(args.sgf_explore and sgf_explorer is not None),
                "updated": False,
                "active": False,
                "planned": False,
            }
            sgf_plan_goal: np.ndarray | None = None
            sgf_plan_command: Any | None = None
            sgf_plan_action: int | None = None
            sgf_plan_mode = ""
            sgf_plan_goal_reached = False
            sgf_plan_goal_retired = False
            text_selected_target_node: dict[str, Any] | None = None
            text_hypothesis_projection: ProjectedHypothesis | None = None
            text_hypothesis_trace: dict[str, Any] = {
                "verification_object": (
                    text_verification_object
                    if args.task_mode == "text"
                    else None
                ),
            }
            text_goal_candidate_diagnostics: dict[str, int] = {}
            text_semantic_summary: dict[str, Any] = {}
            text_target_plan_command: Any | None = None
            text_target_plan_action: int | None = None
            text_target_goal_position: np.ndarray | None = None
            text_sgf_trace: dict[str, Any] = {
                "enabled": bool(text_sgf_enabled),
                "target_control": "sgf_fmm",
                "parsed_goal": parsed_text_goal_trace,
                "selected_target": None,
                "target_plan": None,
                "goal_candidate_diagnostics": {},
            }
            text_navigation_map = sgf_navigation_map
            text_navigator = sgf_navigator
            text_target_plan_navigator = text_navigator
            sgf_visual_memory_trace: dict[str, Any] = {
                "enabled": bool(isinstance(sgf_explorer, VisualMemoryUtilityExplorer)),
                "attempted": False,
                "updated": False,
            }
            sgf_input_dump_trace: dict[str, Any] = {
                "enabled": bool(sgf_input_dumper is not None),
                "dumped": False,
            }
            sgf_frame_index = int(getattr(env, "current_replay_step", step_index)) if replay_env else int(step_index)
            replay_trace: dict[str, Any] = {
                "enabled": bool(replay_env),
                "dump_step": int(sgf_frame_index),
                "input_dir": replay_input_dir if replay_env else "",
            }
            image_vlm_stop_validation_trace: dict[str, Any] = {
                "enabled": bool(args.image_vlm_stop_validation),
                "attempted": False,
                "approved": None,
            }
            sgf_input_consumers: list[str] = []
            if sgf is not None and depth is not None and world_from_camera_cv is not None:
                sgf_input_consumers.append("sgf")
            if sgf_input_dumper is not None:
                with timing.measure("diagnostics"):
                    if sgf_input_consumers:
                        sgf_input_record = sgf_input_dumper.dump_frame(
                            step_index=sgf_frame_index,
                            rgb_observation=rgb_observation,
                            depth_m=depth,
                            world_from_camera_cv=world_from_camera_cv,
                            processed_by=sgf_input_consumers,
                            sensor_state=sensor_state,
                        )
                        if sgf_input_record is not None:
                            sgf_input_dump_trace.update(
                                {
                                    "dumped": True,
                                    "record": sgf_input_record,
                                }
                            )
                        else:
                            sgf_input_dump_trace["skip_reason"] = "step_stride"
                    else:
                        sgf_input_dump_trace["skip_reason"] = "sgf_not_updated"
            if (
                sgf is not None
                and depth is not None
                and world_from_camera_cv is not None
            ):
                with timing.measure("sgf_update"):
                    sgf.process_frame(
                        bgr_from_rgb_observation(rgb_observation),
                        depth,
                        world_from_camera_cv,
                        frame_index=sgf_frame_index,
                    )
                if timing.enabled:
                    get_runtime_stats = getattr(sgf, "get_runtime_stats", None)
                    if callable(get_runtime_stats):
                        timing.record_sgf_stats(get_runtime_stats())
                sgf_explore_trace["updated"] = True
                if sgf_navigation_map is not None:
                    floor_support = sgf_navigation_map.floor_support_stats()
                    sgf_explore_trace["floor_support"] = floor_support
                    navigation_floor_support_steps += int(floor_support["enabled"])
                    navigation_floor_support_veto_cells += int(
                        floor_support["vetoed_free_cells"]
                    )
                    navigation_floor_support_potential_veto_cells += int(
                        floor_support["potential_vetoed_free_cells"]
                    )
                    navigation_floor_support_nonfloor_cells += int(
                        floor_support["nonfloor_cells"]
                    )
                if (
                    sgf_navigation_map is not None
                    and sgf_navigation_map.graph_occupancy_enabled
                    and not sgf_navigation_map.has_provisional
                ):
                    graph_stats = sgf_navigation_map.graph_occupancy_stats()
                    navigation_graph_occupancy_steps += int(graph_stats["surfel_votes"] > 0)
                    sgf_explore_trace["graph_occupancy"] = graph_stats
                    for key in ("surfel_votes", "removed_free_cells", "added_occupied_cells"):
                        representation_activity[f"navigation_graph_{key}"] += int(graph_stats[key])
            if (
                text_sgf_enabled
                and text_navigation_map is not None
                and text_navigator is not None
            ):
                agent_state_for_text_sgf = env.get_agent_state()
                agent_position_for_text_sgf = np.asarray(
                    agent_state_for_text_sgf.position,
                    dtype=np.float32,
                ).reshape(3)
                try:
                    text_semantic_summary = text_navigation_map.semantic_label_summary(
                        goal_match_phrases,
                    )
                    text_summary_labels = list(text_semantic_summary.get("labels", []) or [])
                    text_sgf_trace["semantic_summary"] = {
                        "label_count": int(len(text_summary_labels)),
                        "labels": json_safe_value(text_summary_labels[:8]),
                    }
                    if os.environ.get("SGF_DEBUG_NODE_DUMP"):
                        try:
                            text_sgf_trace["node_dump"] = [
                                {
                                    "id": int(item.get("id", -1)),
                                    "label": str(item.get("label", "")),
                                    "c": [round(float(v), 3) for v in item.get("centroid_m", (0, 0, 0))],
                                    "n": int(item.get("num_surfels", 0)),
                                    "p": round(float(item.get("semantic_confidence", 0.0)), 3),
                                }
                                for item in text_navigation_map.all_nodes_debug()
                                if str(item.get("label", "")) not in ("", "unknown")
                            ]
                        except Exception as exc:  # diagnostics only
                            text_sgf_trace["node_dump_error"] = str(exc)
                    update_episode_label_stats_from_summary(
                        summary=text_semantic_summary,
                        observed_label_steps=observed_label_steps,
                        observed_label_max_surfels=observed_label_max_surfels,
                        observed_label_max_confidence=observed_label_max_confidence,
                        goal_like_label_steps=goal_like_label_steps,
                        goal_like_label_max_surfels=goal_like_label_max_surfels,
                        goal_like_label_max_confidence=goal_like_label_max_confidence,
                    )
                    text_exclusion_radius_m = float(
                        scales.hypothesis_exclusion_radius_m
                    )
                    # Retired hypotheses keep their evidence and can be re-admitted
                    def readmissible_text_hypotheses() -> list[tuple[int, tuple[float, float]]]:
                        if (
                            text_stop_ledger is None
                            or not excluded_text_node_ids
                            or agent_position_for_text_sgf is None
                        ):
                            return []
                        readmit_agent_xz = (
                            float(agent_position_for_text_sgf[0]),
                            float(agent_position_for_text_sgf[2]),
                        )
                        readmit_band_m = float(scales.far_band_radius_m)
                        readmit_vantage_m = float(scales.distinct_view_distance_m)
                        readmit_budget = int(text_stop_ledger.min_commit_approvals)
                        found: list[tuple[int, tuple[float, float]]] = []
                        for readmit_id in sorted(excluded_text_node_ids):
                            if int(readmit_id) >= DETECTION_PROBE_NODE_ID_BASE:
                                continue
                            readmit_xz = text_stop_ledger.node_position(int(readmit_id))
                            if readmit_xz is None:
                                continue
                            if (
                                math.hypot(
                                    readmit_agent_xz[0] - readmit_xz[0],
                                    readmit_agent_xz[1] - readmit_xz[1],
                                )
                                > readmit_band_m
                            ):
                                continue
                            readmit_key = int(
                                text_stop_ledger.resolve_node_key(int(readmit_id), readmit_xz)
                            )
                            if int(text_readmission_counts.get(readmit_key, 0)) >= readmit_budget:
                                continue
                            if (
                                text_stop_ledger.distinct_pose_count(
                                    readmit_key, text_stop_ledger.APPROVAL
                                )
                                < 1
                            ):
                                continue
                            if any(
                                math.hypot(
                                    readmit_agent_xz[0] - used_xz[0],
                                    readmit_agent_xz[1] - used_xz[1],
                                )
                                < readmit_vantage_m
                                for used_xz in text_stop_ledger.observation_positions(
                                    int(readmit_id), readmit_xz
                                )
                            ):
                                continue
                            found.append((int(readmit_id), readmit_xz))
                        return found

                    def select_text_target():
                        return text_navigation_map.select_target_node(
                            agent_position=agent_position_for_text_sgf,
                            goal_phrases=goal_match_phrases,
                            min_surfels=int(nav_config.get("min_target_surfels", 64)),
                            min_confidence=float(nav_config.get("min_target_confidence", 0.2)),
                            excluded_node_ids=excluded_text_node_ids,
                            excluded_positions=excluded_text_positions,
                            exclusion_radius_m=text_exclusion_radius_m,
                        )

                    selected_result = select_text_target()
                    if selected_result is None or selected_result[0] is None:
                        readmit_candidates = readmissible_text_hypotheses()
                        if readmit_candidates:
                            readmitted_positions: list[tuple[float, float]] = []
                            for readmit_id, readmit_xz in readmit_candidates:
                                readmit_key = int(
                                    text_stop_ledger.resolve_node_key(int(readmit_id), readmit_xz)
                                )
                                excluded_text_node_ids.discard(int(readmit_id))
                                readmitted_positions.append(readmit_xz)
                                text_readmission_counts[readmit_key] = (
                                    int(text_readmission_counts.get(readmit_key, 0)) + 1
                                )
                                text_readmissions += 1
                                text_sgf_trace.setdefault("readmitted", []).append(
                                    {
                                        "node_id": int(readmit_id),
                                        "node_key": int(readmit_key),
                                        "trip": int(text_readmission_counts[readmit_key]),
                                        "evidence_log_odds": round(
                                            float(
                                                text_stop_ledger.independent_log_odds(
                                                    int(readmit_id), readmit_xz
                                                )
                                            ),
                                            4,
                                        ),
                                    }
                                )
                            excluded_text_positions[:] = [
                                standpoint
                                for standpoint in excluded_text_positions
                                if all(
                                    math.hypot(
                                        float(standpoint[0]) - readmit_xz[0],
                                        float(standpoint[2]) - readmit_xz[1],
                                    )
                                    > text_exclusion_radius_m
                                    for readmit_xz in readmitted_positions
                                )
                            ]
                            selected_result = select_text_target()
                    text_target_candidates: list[dict] = []
                    if selected_result is not None:
                        if text_query_conditional_labels:
                            text_query_conditional_retrieval_steps += 1
                        else:
                            text_argmax_label_retrieval_steps += 1
                        text_selected_target_node = selected_result[0]
                        text_goal_candidate_diagnostics = selected_result[1]
                        representation_activity["text_association_unreachable_candidates"] += int(
                            text_goal_candidate_diagnostics["association_unreachable_candidates"]
                        )
                        if len(selected_result) > 2:
                            text_target_candidates = list(selected_result[2] or [])
                    if text_detection_registry is not None:
                        # Detections never segmented into a node
                        # (detection_instances.py)
                        try:
                            frame_detections = sgf.get_last_detections()
                        except Exception:
                            frame_detections = []
                        used_detections = text_detection_registry.update(
                            frame_detections,
                            step=int(step_index),
                            labels=set(goal_match_phrases),
                            min_confidence=float(nav_config.get("min_target_confidence", 0.2)),
                        )
                        text_detection_instance_observations += int(used_detections)
                        detection_association_diagnostics: dict[str, int] = {}
                        detection_candidates = text_detection_registry.candidates(
                            navigation_map=text_navigation_map,
                            agent_position=agent_position_for_text_sgf,
                            excluded_ids=excluded_text_node_ids,
                            excluded_positions=excluded_text_positions,
                            exclusion_radius_m=text_exclusion_radius_m,
                            graph_candidates=text_target_candidates,
                            association=text_target_association,
                            diagnostics=detection_association_diagnostics,
                        )
                        representation_activity["text_association_unreachable_candidates"] += (
                            detection_association_diagnostics["association_unreachable_candidates"]
                        )
                        text_detection_instance_candidate_count += len(detection_candidates)
                        if detection_candidates:
                            text_target_candidates = list(text_target_candidates) + detection_candidates
                            nearest_detection = min(
                                detection_candidates, key=candidate_association_distance
                            )
                            if text_selected_target_node is None or candidate_association_distance(
                                nearest_detection
                            ) < candidate_association_distance(text_selected_target_node):
                                text_selected_target_node = dict(nearest_detection)
                            text_goal_candidate_diagnostics = dict(text_goal_candidate_diagnostics or {})
                            text_goal_candidate_diagnostics["detection_candidates"] = int(len(detection_candidates))
                        text_sgf_trace["detection_instances"] = {
                            "instances": int(text_detection_registry.size()),
                            "candidates": int(len(detection_candidates)),
                            "frame_detections": int(len(frame_detections)),
                            "accepted_detections": int(used_detections),
                            "top": text_detection_registry.summary(4),
                            "frame_targets": [
                                {
                                    "label": str(d.get("label", "")),
                                    "conf": round(float(d.get("confidence", 0.0)), 2),
                                    "n": int(d.get("n_points", 0)),
                                    "c": [round(float(v), 2) for v in (d.get("centroid_m") or (0, 0, 0))],
                                    "node": int(d.get("matched_node_id", 0)),
                                }
                                for d in frame_detections
                                if str(d.get("label", "")) in set(goal_match_phrases)
                            ][:4],
                        }
                    representation_activity.update(association_rank_diagnostics(
                        text_target_candidates, agent_position_for_text_sgf
                    ))
                    if text_selected_target_node is not None:
                        selection_modes = {
                            str(candidate.get("association", ""))
                            for candidate in [text_selected_target_node, *text_target_candidates]
                        }
                        if "euclidean" in selection_modes:
                            text_euclidean_association_steps += 1
                        elif "reachable_euclidean" in selection_modes:
                            text_reachable_euclidean_association_steps += 1
                        elif text_target_association == "navigation_map":
                            text_navigation_map_association_steps += 1
                    # Rank by expected travel: distance / posterior from ledger log-odds
                    if (
                        text_stop_ledger is not None
                        and bool_config_value(
                            nav_config,
                            "text_evidence_weighted_target_ranking",
                            False,
                        )
                        and len(text_target_candidates) >= 1
                    ):
                        ranked: list[tuple[float, float, float, dict]] = []
                        for candidate in text_target_candidates:
                            try:
                                candidate_distance = candidate_association_distance(
                                    candidate
                                )
                                goal_xyz = np.asarray(
                                    candidate.get("goal_position"), dtype=np.float64
                                ).reshape(3)
                            except (TypeError, ValueError):
                                continue
                            if not (
                                math.isfinite(candidate_distance)
                                and np.all(np.isfinite(goal_xyz))
                            ):
                                continue
                            candidate_key = text_stop_ledger.resolve_node_key(
                                int(candidate.get("id", -1)),
                                (float(goal_xyz[0]), float(goal_xyz[2])),
                            )
                            evidence = float(
                                text_stop_ledger.independent_log_odds(candidate_key)
                            )
                            posterior = 1.0 / (1.0 + math.exp(-evidence))
                            if posterior <= 0.0:
                                continue
                            ranked.append(
                                (
                                    candidate_distance / posterior,
                                    candidate_distance,
                                    evidence,
                                    candidate,
                                )
                            )
                        if ranked:
                            if any(float(item[2]) != 0.0 for item in ranked):
                                text_evidence_weighted_ranking_steps += 1
                            ranked.sort(key=lambda item: item[0])
                            chosen = ranked[0][3]
                            text_sgf_trace["candidate_ranking"] = {
                                "count": int(len(ranked)),
                                "selected_id": int(chosen.get("id", -1)),
                                "nearest_id": int(
                                    text_selected_target_node.get("id", -1)
                                    if text_selected_target_node is not None
                                    else -1
                                ),
                                "top": [
                                    {
                                        "id": int(item[3].get("id", -1)),
                                        "distance_m": round(float(item[1]), 3),
                                        "evidence_log_odds": round(float(item[2]), 3),
                                        "expected_travel_m": round(float(item[0]), 3),
                                    }
                                    for item in ranked[:4]
                                ],
                            }
                            if (
                                text_selected_target_node is None
                                or int(chosen.get("id", -1))
                                != int(text_selected_target_node.get("id", -1))
                            ):
                                text_evidence_ranking_overrides += 1
                                text_sgf_trace["candidate_ranking"]["override"] = True
                            text_selected_target_node = dict(chosen)
                    for reason, count in text_goal_candidate_diagnostics.items():
                        goal_candidate_rejection_counts[str(reason)] += int(count)
                    text_sgf_trace["goal_candidate_diagnostics"] = dict(
                        sorted((str(key), int(value)) for key, value in text_goal_candidate_diagnostics.items())
                    )
                    if text_selected_target_node is not None:
                        text_sgf_target_found = True
                        selected_node_id = int(text_selected_target_node.get("id", -1))
                        if (
                            text_detection_registry is not None
                            and text_detection_registry.is_detection_id(selected_node_id)
                        ):
                            text_detection_instance_selected_steps += 1
                        if (
                            depth is not None
                            and world_from_camera_cv is not None
                        ):
                            text_hypothesis_projection = project_hypothesis_points(
                                (
                                    text_detection_registry.points(selected_node_id)
                                    if text_detection_registry is not None
                                    and text_detection_registry.is_detection_id(selected_node_id)
                                    else node_surfel_positions(
                                        sgf,
                                        node_id=selected_node_id,
                                        min_confidence=hypothesis_surfel_min_confidence,
                                    )
                                ),
                                world_from_camera_cv=world_from_camera_cv,
                                fx=frame_fx,
                                fy=frame_fy,
                                cx=frame_cx,
                                cy=frame_cy,
                                width=int(width),
                                height=int(height),
                                depth=depth,
                                map_resolution_m=hypothesis_map_resolution_m,
                                min_depth_m=float(min_depth),
                                max_depth_m=float(max_depth),
                                pixel_stride=hypothesis_pixel_stride,
                                goal_position_m=text_selected_target_node.get(
                                    "goal_position"
                                ),
                            )
                            text_hypothesis_trace.update(
                                summarize_hypothesis_projection(
                                    text_hypothesis_projection,
                                    node_id=selected_node_id,
                                )
                            )
                            if text_hypothesis_projection is not None:
                                text_hypothesis_projection_count += 1

                            # Query the verifier on a visible far candidate before
                            # travelling to it
                            triage_trace: dict[str, Any] = {}
                            if (
                                bool(args.text_hypothesis_triage)
                                and stop_validator is not None
                                and text_stop_ledger is not None
                                and text_hypothesis_projection is not None
                                and text_hypothesis_projection.depth_support > 0.5
                                and text_hypothesis_projection.containment > 0.5
                                and not bool(forced_budget_stop)
                            ):
                                triage_goal_position = np.asarray(
                                    text_selected_target_node.get(
                                        "goal_position", (np.nan, np.nan, np.nan)
                                    ),
                                    dtype=np.float32,
                                ).reshape(-1)
                                triage_goal_valid = bool(
                                    triage_goal_position.size == 3
                                    and np.all(np.isfinite(triage_goal_position))
                                )
                                triage_node_position = (
                                    (
                                        float(triage_goal_position[0]),
                                        float(triage_goal_position[2]),
                                    )
                                    if triage_goal_valid
                                    else None
                                )
                                # +inf (unreachable) counts as far, zero or missing is
                                # skipped
                                triage_raw_distance = text_selected_target_node.get(
                                    "distance_m"
                                )
                                triage_distance_m = (
                                    float("nan")
                                    if triage_raw_distance is None
                                    or float(triage_raw_distance) == 0.0
                                    else float(triage_raw_distance)
                                )
                                triage_key = int(
                                    text_stop_ledger.resolve_node_key(
                                        selected_node_id,
                                        triage_node_position,
                                    )
                                )
                                triage_steps_since = int(step_index) - int(
                                    text_triage_last_step.get(triage_key, -(10**9))
                                )
                                triage_due = triage_steps_since >= int(
                                    requery_interval_steps
                                )
                                if (
                                    triage_due
                                    and not math.isnan(triage_distance_m)
                                    and triage_distance_m
                                    > float(scales.far_band_radius_m)
                                ):
                                    text_triage_last_step[triage_key] = int(step_index)
                                    text_triage_queries += 1
                                    try:
                                        if text_verification_object == "top_detection":
                                            text_top_detection_queries += 1
                                        elif text_verification_object == "whole_frame":
                                            text_whole_frame_queries += 1
                                        with timing.measure("stop_verification", "text_triage"):
                                            triage_result = stop_validator.validate(
                                                text_goal=episode_info.text_goal,
                                                canonical_goal=canonical_text_goal,
                                                rgb_bgr=bgr_from_rgb_observation(
                                                    rgb_observation
                                                ),
                                                hypothesis=text_hypothesis_projection,
                                                verification_object=text_verification_object,
                                            )
                                    except Exception as exc:
                                        triage_result = None
                                        triage_trace["error"] = str(exc)[:200]
                                    if triage_result is not None:
                                        triage_state = env.get_agent_state()
                                        triage_position = np.asarray(
                                            triage_state.position, dtype=np.float32
                                        ).reshape(3)
                                        triage_yaw = agent_yaw_deg(triage_state)
                                        triage_standpoint = None
                                        if text_verification_object == "top_detection":
                                            triage_standpoint, _triage_bank = (
                                                detection_projected_standpoint(
                                                    triage_state,
                                                    triage_result,
                                                    depth,
                                                    hfov_deg=float(args.hfov),
                                                    min_depth_m=min_depth,
                                                    max_depth_m=max_depth,
                                                    standoff_m=float(
                                                        scales.terminal_approach_distance_m
                                                    ),
                                                )
                                            )
                                        triage_evidence_weight = ledger_evidence_weight(
                                            triage_result,
                                            node=text_selected_target_node,
                                            probe_owned=False,
                                            world_from_camera_cv=world_from_camera_cv,
                                            fx=frame_fx,
                                            fy=frame_fy,
                                            cx=frame_cx,
                                            cy=frame_cy,
                                            width=int(width),
                                            height=int(height),
                                            evidence_standpoint_m=triage_standpoint,
                                            consistency_radius_m=(
                                                float(scales.hypothesis_exclusion_radius_m)
                                                if triage_standpoint is not None
                                                else float("inf")
                                            ),
                                            verification_object=text_verification_object,
                                        )
                                        triage_record_kwargs: dict[str, Any] = dict(
                                            node_id=selected_node_id,
                                            node_position_xz=triage_node_position,
                                            agent_position_xz=(
                                                float(triage_position[0]),
                                                float(triage_position[2]),
                                            ),
                                            agent_yaw_deg=float(
                                                triage_yaw
                                                if triage_yaw is not None
                                                else 0.0
                                            ),
                                            step=int(step_index),
                                            approved=bool(triage_result.approved),
                                            p_yes=float(triage_result.p_yes or 0.0),
                                            detection_found=bool(
                                                triage_result.detection_found
                                            ),
                                            detection_bbox_area_fraction=float(
                                                triage_result.detection_bbox_area_fraction
                                            ),
                                            detection_confidence=float(
                                                triage_result.detection_confidence
                                            ),
                                            context_observability=(
                                                triage_result.context_observability
                                                if triage_result.factored
                                                else None
                                            ),
                                            appearance_p=triage_result.appearance_p,
                                            evidence_weight=triage_evidence_weight,
                                        )
                                        triage_record = text_stop_ledger.record(
                                            **triage_record_kwargs
                                        )
                                        if text_shadow_stop is not None:
                                            text_shadow_stop.record(
                                                **triage_record_kwargs
                                            )
                                        appearance_p = triage_result.appearance_p
                                        appearance_refuted = bool(
                                            appearance_p is not None
                                            and float(appearance_p) < 0.5
                                        )
                                        triage_trace.update(
                                            {
                                                "attempted": True,
                                                "distance_m": (
                                                    round(triage_distance_m, 3)
                                                    if math.isfinite(triage_distance_m)
                                                    else None
                                                ),
                                                "approved": bool(
                                                    triage_result.approved
                                                ),
                                                "appearance_p": appearance_p,
                                                "context_p": triage_result.context_p,
                                                "appearance_refuted": appearance_refuted,
                                                "ledger": dict(triage_record),
                                            }
                                        )
                                        if triage_result is not None:
                                            (
                                                triage_exclude,
                                                triage_reason,
                                                triage_details,
                                            ) = text_stop_ledger.decide_exclusion(
                                                node_id=selected_node_id,
                                                node_position_xz=triage_node_position,
                                            )
                                            triage_trace["exclusion"] = {
                                                "excluded": bool(triage_exclude),
                                                "reason": str(triage_reason),
                                                "evidence_log_odds": triage_details.get(
                                                    "evidence_log_odds"
                                                ),
                                            }
                                            if triage_exclude:
                                                excluded_text_node_ids.add(
                                                    int(selected_node_id)
                                                )
                                                retire_text_hypothesis(
                                                    triage_goal_position.copy()
                                                    if triage_goal_valid
                                                    else None,
                                                )
                                                text_triage_exclusions += 1
                            if triage_trace:
                                text_hypothesis_trace["triage"] = triage_trace
                        if tracked_text_target_node_id == selected_node_id:
                            tracked_text_target_steps += 1
                        else:
                            tracked_text_target_node_id = selected_node_id
                            tracked_text_target_steps = 1
                        raw_goal_position = np.asarray(
                            text_selected_target_node.get("goal_position", (np.nan, np.nan, np.nan)),
                            dtype=np.float32,
                        ).reshape(-1)
                        if (
                            raw_goal_position.size == 3
                            and np.all(np.isfinite(raw_goal_position))
                        ):
                            text_target_goal_position = raw_goal_position.copy()
                            target_transaction_trace = text_sgf_trace.setdefault(
                                "target_transaction", {}
                            )
                            if text_navigator.has_pending_action:
                                target_transaction_trace[
                                    "orphan_cancelled_before_plan"
                                ] = bool(
                                    text_navigator.cancel_pending_action()
                                )
                            text_reobservation_active = False
                            if text_reobservation is not None:
                                reobs = text_reobservation
                                reobs_same_node = int(reobs["node_id"]) == int(
                                    selected_node_id
                                ) or (
                                    text_stop_ledger is not None
                                    and int(
                                        text_stop_ledger.resolve_node_key(
                                            selected_node_id,
                                            (
                                                float(raw_goal_position[0]),
                                                float(raw_goal_position[2]),
                                            ),
                                        )
                                    )
                                    == int(reobs["key"])
                                )
                                reobs_displacement_m = float(
                                    planar_distance_xz(
                                        agent_position_for_text_sgf, reobs["origin"]
                                    )
                                )
                                reobs_displaced = (
                                    reobs_displacement_m
                                    >= float(scales.distinct_view_distance_m)
                                )
                                reobs_expired = int(step_index) - int(
                                    reobs["start_step"]
                                ) > int(reobs["budget_steps"])
                                if not reobs_same_node or reobs_displaced or reobs_expired:
                                    reobs_end = (
                                        "node_changed"
                                        if not reobs_same_node
                                        else ("displaced" if reobs_displaced else "expired")
                                    )
                                    text_sgf_trace["reobservation_end"] = {
                                        "node_key": int(reobs["key"]),
                                        "reason": reobs_end,
                                        "displacement_m": round(reobs_displacement_m, 3),
                                        "steps": int(step_index) - int(reobs["start_step"]),
                                    }
                                    if reobs_end == "expired":
                                        # No new vantage reached within budget: retire
                                        # the hypothesis
                                        excluded_text_node_ids.add(int(selected_node_id))
                                        retire_text_hypothesis(raw_goal_position.copy())
                                        text_reobservation_retirements += 1
                                    text_reobservation = None
                                else:
                                    text_reobservation_active = True
                            if text_reobservation_active:
                                text_sgf_trace["reobservation"] = {
                                    "node_key": int(text_reobservation["key"]),
                                    "goal_m": world_position_list(text_reobservation["goal"]),
                                    "steps": int(step_index) - int(text_reobservation["start_step"]),
                                    "budget_steps": int(text_reobservation["budget_steps"]),
                                }
                            text_target_plan_command = text_navigator.next_action(
                                agent_state=agent_state_for_text_sgf,
                                goal_position=(
                                    text_reobservation["goal"]
                                    if text_reobservation_active
                                    else text_target_goal_position
                                ),
                                navigation_map=text_navigation_map,
                                allow_stop=False,
                                stop_radius_m=(
                                    float(scales.distinct_view_distance_m)
                                    if text_reobservation_active
                                    else float(success_radius)
                                ),
                            )
                            text_target_plan_action = int(text_target_plan_command.action)
                            target_transaction_trace["proposed_action"] = int(
                                text_target_plan_action
                            )
                            target_plan_goal_distance_m = float("nan")
                            try:
                                target_plan_goal_distance_m = float(
                                    text_target_plan_command.goal_distance_m
                                )
                            except Exception:
                                target_plan_goal_distance_m = float("nan")
                            fatigue_min_progress_m = float(
                                scales.planar_progress_quantum_m
                            )
                            if text_target_progress_node_id != selected_node_id:
                                text_target_progress_node_id = selected_node_id
                                text_target_progress_best_distance = target_plan_goal_distance_m
                                text_target_progress_last_improvement_step = int(step_index)
                            else:
                                if math.isfinite(target_plan_goal_distance_m) and (
                                    not math.isfinite(text_target_progress_best_distance)
                                    or target_plan_goal_distance_m
                                    < text_target_progress_best_distance
                                    - fatigue_min_progress_m
                                ):
                                    text_target_progress_best_distance = target_plan_goal_distance_m
                                    text_target_progress_last_improvement_step = int(step_index)
                            target_no_progress_steps = (
                                int(step_index)
                                - int(text_target_progress_last_improvement_step)
                                if text_target_progress_last_improvement_step >= 0
                                else 0
                            )
                            # Fatigue applies only outside the validation bands
                            fatigue_min_goal_distance = float(
                                scales.far_band_radius_m
                            )
                            fatigue_distance_ok = bool(
                                math.isfinite(target_plan_goal_distance_m)
                                and target_plan_goal_distance_m
                                > fatigue_min_goal_distance
                            )
                            target_plan_local_blocked = bool(
                                getattr(
                                    text_target_plan_command,
                                    "local_blocked",
                                    False,
                                )
                            )
                            target_plan_recovery_active = bool(
                                getattr(
                                    text_target_plan_command,
                                    "recovery_active",
                                    False,
                                )
                            )
                            target_plan_goal_blocked = bool(
                                getattr(
                                    text_target_plan_command,
                                    "goal_blocked",
                                    False,
                                )
                            )
                            target_plan_replan_failed = bool(
                                getattr(
                                    text_target_plan_command,
                                    "planner_replan_failed",
                                    False,
                                )
                            )
                            try:
                                target_plan_progress_stall_count = int(
                                    text_target_plan_command.progress_stall_count
                                )
                            except (AttributeError, TypeError, ValueError):
                                target_plan_progress_stall_count = 0
                            patience_steps = int(
                                args.text_target_fatigue_patience_steps
                            )
                            target_fatigue_trace = {
                                "enabled": bool(args.text_target_fatigue),
                                "retired": False,
                                "node_id": int(selected_node_id),
                                "tracked_steps": int(tracked_text_target_steps),
                                "patience_steps": int(patience_steps),
                                "progress_quantum_m": round(
                                    float(fatigue_min_progress_m), 4
                                ),
                                "goal_distance_m": (
                                    None
                                    if not math.isfinite(target_plan_goal_distance_m)
                                    else round(float(target_plan_goal_distance_m), 4)
                                ),
                                "best_goal_distance_m": (
                                    None
                                    if not math.isfinite(
                                        text_target_progress_best_distance
                                    )
                                    else round(
                                        float(text_target_progress_best_distance), 4
                                    )
                                ),
                                "no_progress_steps": int(target_no_progress_steps),
                                "min_goal_distance_m": round(
                                    float(fatigue_min_goal_distance), 4
                                ),
                                "distance_ok": bool(fatigue_distance_ok),
                                "local_blocked": bool(target_plan_local_blocked),
                                "recovery_active": bool(target_plan_recovery_active),
                                "goal_blocked": bool(target_plan_goal_blocked),
                                "planner_replan_failed": bool(
                                    target_plan_replan_failed
                                ),
                                "progress_stall_count": int(
                                    target_plan_progress_stall_count
                                ),
                            }
                            fatigue_should_retire = bool(
                                args.text_target_fatigue
                                and selected_node_id >= 0
                                and patience_steps >= 0
                                and target_no_progress_steps >= patience_steps
                                and fatigue_distance_ok
                            )
                            if fatigue_should_retire:
                                fatigue_cancelled_pending = False
                                if text_navigator.has_pending_action:
                                    fatigue_cancelled_pending = bool(
                                        text_navigator.cancel_pending_action()
                                    )
                                target_transaction_trace.update(
                                    {
                                        "fatigue_cancelled_pending": bool(
                                            fatigue_cancelled_pending
                                        ),
                                        "resolution": "fatigue_cancelled",
                                    }
                                )
                                excluded_text_node_ids.add(int(selected_node_id))
                                retire_text_hypothesis(
                                    raw_goal_position.copy()
                                )
                                text_target_fatigue_retirements += 1
                                text_selected_target_node = None
                                text_target_goal_position = None
                                text_target_plan_command = None
                                text_target_plan_action = None
                                tracked_text_target_node_id = None
                                tracked_text_target_steps = 0
                                text_target_progress_node_id = None
                                text_target_progress_best_distance = float("inf")
                                text_target_progress_last_improvement_step = -1
                                target_fatigue_trace.update(
                                    {
                                        "retired": True,
                                        "retire_reason": "no_progress",
                                        "retire_count": int(
                                            text_target_fatigue_retirements
                                        ),
                                    }
                                )
                            text_sgf_trace["target_fatigue"] = target_fatigue_trace
                        elif (
                            raw_goal_position.size == 3
                            and np.all(np.isfinite(raw_goal_position))
                        ):
                            text_sgf_trace["target_plan_suppressed"] = "min_step"
                        else:
                            text_sgf_trace["target_plan_error"] = "invalid_goal_position"
                    else:
                        tracked_text_target_node_id = None
                        tracked_text_target_steps = 0
                        text_target_progress_node_id = None
                        text_target_progress_best_distance = float("inf")
                        text_target_progress_last_improvement_step = -1
                    text_sgf_trace["selected_target"] = summarize_text_sgf_target_node(
                        text_selected_target_node
                    )
                    text_sgf_trace["hypothesis"] = dict(text_hypothesis_trace)
                    text_sgf_trace["target_plan"] = summarize_navigation_command(
                        text_target_plan_command
                    )
                    text_sgf_trace["tracked_target_steps"] = int(tracked_text_target_steps)
                    text_target_navigation_active = True
                    text_sgf_trace["target_navigation_active"] = bool(
                        text_target_navigation_active
                    )
                except Exception as exc:
                    text_sgf_trace["error"] = str(exc)
            if (
                isinstance(sgf_explorer, VisualMemoryUtilityExplorer)
                and goal_image_vector is not None
            ):
                memory_agent_state = env.get_agent_state()
                image_similarity = float(
                    current_goal_image_similarity_for_frame()
                )
                if endgame_return_planner is not None:
                    # Appearance trend only, never a return anchor on its own
                    endgame_return_planner.note_observation(
                        float(image_similarity),
                        int(step_index),
                    )
                with timing.measure("image_goal", "visual_memory_update"):
                    memory_updated = sgf_explorer.update_visual_memory(
                        position=np.asarray(memory_agent_state.position, dtype=np.float32).reshape(3),
                        rotation=memory_agent_state.rotation,
                        similarity=image_similarity,
                        step_index=step_index,
                    )
                sgf_visual_memory_trace.update(
                    {
                        "attempted": True,
                        "updated": bool(memory_updated),
                        "similarity": round(float(image_similarity), 6),
                        "stats": sgf_explorer.visual_memory_stats(),
                    }
                )
            elif isinstance(sgf_explorer, VisualMemoryUtilityExplorer):
                with timing.measure("image_goal", "visual_memory_advance"):
                    sgf_explorer.advance_visual_memory(step_index)
                sgf_visual_memory_trace["stats"] = sgf_explorer.visual_memory_stats()
            if visual_match_due or image_match_save_due:
                assert xfeat_matcher is not None
                with timing.measure(
                    "image_goal" if visual_match_due else "diagnostics",
                    "xfeat_matching",
                ):
                    image_space_match_evidence = match_instance_image_evidence(
                        xfeat_matcher,
                        goal_image_rgb=goal_image_rgb,
                        current_image_rgb=rgb_observation,
                        config=visual_target_config,
                        timing=timing,
                    )
            if (
                (visual_match_due or image_match_save_due)
                and depth is not None
                and world_from_camera_cv is not None
                and image_space_match_evidence is not None
            ):
                with timing.measure(
                    "image_goal" if visual_match_due else "diagnostics",
                    "depth_geometry",
                ):
                    frame_geometry_evidence = build_frame_geometry_evidence(
                        image_space_match_evidence,
                        depth=depth,
                        intrinsics=visual_intrinsics,
                        world_from_camera_cv=world_from_camera_cv,
                        min_depth_m=min_depth,
                        max_depth_m=max_depth,
                    )
                with timing.measure(
                    "image_goal" if visual_match_due else "diagnostics",
                    "depth_localization",
                ):
                    visual_result = localize_instance_image_evidence(
                        frame_geometry_evidence,
                        goal_image_shape=tuple(
                            np.asarray(goal_image_rgb).shape
                        ),
                        config=visual_target_config,
                    )
                localized_visual_stop_trace = {
                    "attempted": True,
                    "matcher": XFEAT_MATCHER_NAME,
                    "target_updated": bool(
                        visual_result.diagnostics.get(
                            "target_updated", False
                        )
                    ),
                    "reject_reason": visual_result.diagnostics.get("reject_reason"),
                    "ransac_inliers": int(
                        visual_result.diagnostics.get("ransac_inliers") or 0
                    ),
                    "confidence_sum": visual_result.diagnostics.get("confidence_sum"),
                    "cluster_size": int(
                        visual_result.diagnostics.get("cluster_size") or 0
                    ),
                }
                center_x_px, center_inlier_count = weighted_match_center_x(
                    visual_result
                )
                localized_visual_stop_trace["match_center_x_px"] = center_x_px
                localized_visual_stop_trace[
                    "match_center_inliers"
                ] = center_inlier_count
                if (
                    bool(args.sgf_explore_evidence_verification)
                    and sgf_explorer is not None
                ):
                    # Search-side evidence only, no stop, servo, or target use
                    geometric_completeness = geometric_evidence_completeness(
                        ransac_inliers=int(
                            visual_result.diagnostics.get(
                                "ransac_inliers"
                            )
                            or 0
                        ),
                        confidence_sum=visual_result.diagnostics.get(
                            "confidence_sum"
                        ),
                        mean_confidence=visual_result.diagnostics.get(
                            "mean_confidence"
                        ),
                        min_inliers=int(visual_target_config.min_inliers),
                        min_confidence_sum=float(
                            visual_target_config.min_confidence_sum
                        ),
                        min_mean_confidence=float(
                            visual_target_config.min_mean_confidence
                        ),
                    )
                    localized_visual_stop_trace["geometric_completeness"] = round(
                        float(geometric_completeness), 6
                    )
                    localized_visual_stop_trace[
                        "evidence_ransac_inliers"
                    ] = int(
                        visual_result.diagnostics.get(
                            "ransac_inliers"
                        )
                        or 0
                    )
                    unverified_geometry = bool(
                        geometric_completeness > 0.0
                        and not bool(
                            visual_result.diagnostics.get("target_updated", False)
                        )
                    )
                    if unverified_geometry:
                        hypothesis_position = weak_instance_hypothesis(
                            frame_geometry_evidence,
                            config=visual_target_config,
                        )
                        if sgf_explorer.note_instance_hypothesis(
                            hypothesis_position,
                            completeness=float(geometric_completeness),
                            step_index=int(step_index),
                        ):
                            sgf_explore_evidence_hypothesis_count += 1
                            sgf_explore_trace["instance_hypothesis"] = {
                                "position_m": world_position_list(
                                    hypothesis_position
                                ),
                                "completeness": round(
                                    float(geometric_completeness), 6
                                ),
                                "episode_count": int(
                                    sgf_explore_evidence_hypothesis_count
                                ),
                            }
                visual_stop_ready = reliable_instance_image_target_localized(
                    visual_match_navigation_enabled=(
                        visual_match_navigation_enabled
                    ),
                    target_node=visual_result.node,
                )
                if visual_stop_ready:
                    instance_image_target_found = True
                if (
                    bool(args.visual_pose_servo)
                    and visual_result.visualization is not None
                    and visual_goal_intrinsics is not None
                    and image_target_localization == "depth_cluster"
                ):
                    # Target is the matched object point (no goal-camera pose)
                    pose_goal, visual_pose_trace = depth_cluster_pose_goal(
                        visual_result.node if visual_stop_ready else None,
                        localizer_reject_reason=(
                            visual_result.diagnostics.get("reject_reason")
                            if isinstance(visual_result.diagnostics, dict)
                            else None
                        ),
                    )
                    if pose_goal is not None:
                        image_depth_cluster_target_updates += 1
                    if pose_goal is not None and endgame_return_planner is not None:
                        grounded_similarity = _finite_trace_float(
                            sgf_visual_memory_trace, "similarity"
                        )
                        if grounded_similarity is not None:
                            endgame_return_planner.note_score(
                                float(grounded_similarity),
                                np.asarray(pose_goal, dtype=np.float32),
                                int(step_index),
                                anchored=True,
                            )
                    if pose_goal is not None:
                        if visual_pose_goal_position is None:
                            target_reacquire_count += 1
                        visual_pose_goal_position = np.asarray(
                            pose_goal, dtype=np.float32
                        ).copy()
                        visual_pose_goal_is_weak = not bool(
                            args.visual_pose_allow_stop
                        )
                        visual_pose_last_update_step = int(step_index)
                        visual_pose_goal_metadata = {
                            "source": "depth_cluster",
                            "matcher": XFEAT_MATCHER_NAME,
                            "update_step": int(step_index),
                        }
                        visual_pose_trace["goal_m"] = world_position_list(
                            visual_pose_goal_position
                        )
                elif (
                    bool(args.visual_pose_servo)
                    and visual_result.visualization is not None
                    and visual_goal_intrinsics is not None
                ):
                    with timing.measure(
                        "image_goal" if visual_match_due else "diagnostics",
                        "pnp_pose_estimation",
                    ):
                        pose_goal, visual_pose_trace = estimate_visual_pose_servo_goal(
                            evidence=frame_geometry_evidence,
                            goal_intrinsics=visual_goal_intrinsics,
                            world_from_camera_cv=world_from_camera_cv,
                            agent_position=np.asarray(env.get_agent_state().position, dtype=np.float32).reshape(3),
                            min_matches=int(args.visual_pose_min_matches),
                            min_inliers=int(args.visual_pose_min_inliers),
                            min_inlier_fraction=float(args.visual_pose_min_inlier_fraction),
                            reproj_threshold_px=float(args.visual_pose_reproj_threshold_px),
                            min_confidence=float(args.visual_pose_min_confidence),
                            min_agent_distance=float(args.visual_pose_min_agent_distance),
                            max_agent_distance=float(args.visual_pose_max_agent_distance),
                            max_height_delta=float(args.visual_pose_max_height_delta),
                            min_inlier_coverage=float(
                                getattr(args, "visual_pose_min_inlier_coverage", 0.0)
                                or 0.0
                            ),
                        )
                    if pose_goal is not None:
                        visual_pose_goal_update_count += 1
                        visual_pose_trace["raw_goal_m"] = world_position_list(pose_goal)
                        trust_anchor = visual_pose_trace.get(
                            "viewpoint_pnp_object_anchor_m"
                        )
                        if (
                            vlm_trust_point is not None
                            and trust_anchor is not None
                            and float(
                                np.hypot(
                                    float(trust_anchor[0]) - float(vlm_trust_point[0]),
                                    float(trust_anchor[2]) - float(vlm_trust_point[2]),
                                )
                            )
                            <= float(success_radius)
                        ):
                            vlm_trust_confirmed += 1
                            vlm_trust_point = None
                            vlm_trust_arrival_deadline = None
                    elif (
                        visual_pose_trace.get("reject_reason")
                        == "inlier_coverage_too_small"
                    ):
                        visual_pose_coverage_rejections += 1
                    if (
                        pose_goal is None
                        and goal_view_belief is not None
                        and visual_pose_trace.get("reject_reason")
                        == "pose_goal_too_close"
                        and bool(visual_pose_trace.get("measurement_valid"))
                        and visual_pose_trace.get("measurement_m") is not None
                    ):
                        # Too close for the servo, but still fuse the solve into the
                        # belief
                        goal_view_belief.standpoint_observations += 1
                        goal_view_confirmed = goal_view_belief.observe(
                            position=visual_pose_trace["measurement_m"],
                            forward=visual_pose_trace.get(
                                "viewpoint_pnp_forward_world"
                            ),
                            anchor=visual_pose_trace.get(
                                "viewpoint_pnp_object_anchor_m"
                            ),
                            support=float(
                                visual_pose_trace.get("viewpoint_pnp_inliers")
                                or 0.0
                            ),
                            step=int(step_index),
                            agent_position=np.asarray(
                                env.get_agent_state().position,
                                dtype=np.float32,
                            ),
                        )
                        visual_pose_trace["standpoint_observation"] = True
                        visual_pose_last_update_step = int(step_index)
                    if pose_goal is not None and endgame_return_planner is not None:
                        grounded_similarity = _finite_trace_float(
                            sgf_visual_memory_trace, "similarity"
                        )
                        if grounded_similarity is not None:
                            endgame_return_planner.note_score(
                                float(grounded_similarity),
                                np.asarray(pose_goal, dtype=np.float32),
                                int(step_index),
                                anchored=True,
                            )
                    if pose_goal is not None:
                        if visual_pose_goal_position is None:
                            target_reacquire_count += 1
                        visual_pose_goal_position = np.asarray(
                            pose_goal, dtype=np.float32
                        ).copy()
                        if goal_view_belief is not None:
                            goal_view_confirmed = goal_view_belief.observe(
                                position=pose_goal,
                                forward=visual_pose_trace.get(
                                    "viewpoint_pnp_forward_world"
                                ),
                                anchor=visual_pose_trace.get(
                                    "viewpoint_pnp_object_anchor_m"
                                ),
                                support=float(
                                    visual_pose_trace.get("viewpoint_pnp_inliers")
                                    or 0.0
                                ),
                                step=int(step_index),
                                agent_position=np.asarray(
                                    env.get_agent_state().position,
                                    dtype=np.float32,
                                ),
                            )
                            goal_view_active = goal_view_belief.select(
                                np.asarray(
                                    env.get_agent_state().position,
                                    dtype=np.float32,
                                )
                            )
                            if goal_view_active is not None:
                                visual_pose_goal_position = np.asarray(
                                    goal_view_active.position, dtype=np.float32
                                ).reshape(3).copy()
                        visual_pose_goal_is_weak = not bool(
                            args.visual_pose_allow_stop
                        )
                        visual_pose_last_update_step = int(step_index)
                        visual_pose_goal_metadata = {
                            "source": "visual_pose",
                            "matcher": XFEAT_MATCHER_NAME,
                            "update_step": int(step_index),
                        }
                        visual_pose_trace["goal_m"] = world_position_list(
                            visual_pose_goal_position
                        )
                        visual_pose_trace["weak_evidence"] = bool(
                            visual_pose_goal_is_weak
                        )
                if visual_result.visualization is not None:
                    with timing.measure("diagnostics"):
                        image_match_target_node = (
                            dict(visual_result.node)
                            if isinstance(visual_result.node, dict)
                            else None
                        )
                        image_match_diagnostics = dict(localized_visual_stop_trace)
                        image_match_diagnostics["matcher"] = XFEAT_MATCHER_NAME
                        image_match_visualization_trace.update(
                            {
                                "enabled": bool(image_match_outputs_requested),
                                "attempted": bool(image_match_outputs_requested),
                                "match_available": True,
                                "matcher": XFEAT_MATCHER_NAME,
                                "step": int(step_index),
                                "ransac_inliers": int(
                                    visual_result.diagnostics.get("ransac_inliers") or 0
                                ),
                                "confidence_sum": visual_result.diagnostics.get(
                                    "confidence_sum"
                                ),
                            }
                        )
                        last_image_match_visualization = visual_result.visualization
                        last_image_match_current_rgb = rgb_observation.copy()
                        last_image_match_diagnostics = image_match_diagnostics
                        last_image_match_step = int(step_index)
                        last_image_match_target_node = image_match_target_node
                        if image_match_target_node is not None:
                            current_quality = (
                                float(visual_result.diagnostics.get("confidence_sum") or 0.0),
                                int(visual_result.diagnostics.get("ransac_inliers") or 0),
                            )
                            if (
                                best_image_match_quality is None
                                or current_quality > best_image_match_quality
                            ):
                                best_image_match_quality = current_quality
                                best_image_match_visualization = visual_result.visualization
                                best_image_match_current_rgb = rgb_observation.copy()
                                best_image_match_diagnostics = image_match_diagnostics
                                best_image_match_step = int(step_index)
                                best_image_match_target_node = image_match_target_node
                        save_image_match_step_visual = bool(
                            image_match_outputs_requested
                            and image_match_visual_interval > 0
                            and step_index % image_match_visual_interval == 0
                        )
                        add_image_match_video_frame = bool(
                            image_match_video_writer is not None
                            and step_index % max(1, image_match_visual_interval) == 0
                        )
                        if save_image_match_step_visual or add_image_match_video_frame:
                            image_match_canvas_path = image_match_visual_dir / f"match_{step_index:04d}.png"
                            if save_image_match_step_visual:
                                image_match_canvas = save_image_match_visualization(
                                    output_path=image_match_canvas_path,
                                    goal_image_rgb=goal_image_rgb,
                                    current_image_rgb=rgb_observation,
                                    match_debug=visual_result.visualization,
                                    diagnostics=image_match_diagnostics,
                                    step_index=step_index,
                                    target_node=image_match_target_node,
                                    max_lines=int(args.image_match_visual_max_lines),
                                )
                                image_match_visualization_trace["visualization_path"] = str(
                                    image_match_canvas_path
                                )
                            else:
                                image_match_canvas = render_image_match_canvas(
                                    goal_image_rgb=goal_image_rgb,
                                    current_image_rgb=rgb_observation,
                                    match_debug=visual_result.visualization,
                                    diagnostics=image_match_diagnostics,
                                    step_index=step_index,
                                    target_node=image_match_target_node,
                                    max_lines=int(args.image_match_visual_max_lines),
                                )
                            if add_image_match_video_frame:
                                image_match_video_writer.add_frame(image_match_canvas)
                                image_match_visualization_trace["video_path"] = str(
                                    configured_image_match_video_path
                                )
                if visual_match_due:
                    visual_stop_trace = localized_visual_stop_trace
            image_evidence_max_age_steps = int(scales.evidence_max_age_steps)
            visual_pose_evidence_age_steps = int(
                step_index - visual_pose_last_update_step
            )
            goal_view_active = None
            if goal_view_belief is not None:
                # Only frames that should have re-observed a hypothesis count against it
                goal_view_expired = goal_view_belief.note_frame(
                    step=int(step_index),
                    world_from_camera_cv=world_from_camera_cv,
                    depth=depth,
                    fx=float(frame_fx),
                    fy=float(frame_fy),
                    cx=float(frame_cx),
                    cy=float(frame_cy),
                    confirmed=goal_view_confirmed,
                )
                goal_view_active = goal_view_belief.select(
                    np.asarray(env.get_agent_state().position, dtype=np.float32)
                )
                if goal_view_active is None:
                    visual_pose_goal_position = None
                    visual_pose_goal_is_weak = False
                    visual_pose_goal_metadata = {}
                else:
                    visual_pose_goal_position = np.asarray(
                        goal_view_active.position, dtype=np.float32
                    ).reshape(3).copy()
                    goal_view_trace["pursued_point"] = "goal_camera"
                    if (
                        image_navigation_target == "object"
                        and goal_view_active.anchor is None
                    ):
                        image_object_anchor_missing_steps += 1
                        goal_view_trace["pursued_point"] = None
                        visual_pose_goal_position = None
                    elif (
                        image_navigation_target != "goal_camera"
                        and goal_view_active.anchor is not None
                    ):
                        # 'object' pursues the anchor, 'nearer' whichever point is
                        # closer
                        agent_location = np.asarray(
                            env.get_agent_state().position, dtype=np.float32
                        ).reshape(3)
                        camera_gap_m = float(
                            np.linalg.norm(
                                (visual_pose_goal_position - agent_location)[[0, 2]]
                            )
                        )
                        object_gap_m = float(
                            np.hypot(
                                float(goal_view_active.anchor[0])
                                - float(agent_location[0]),
                                float(goal_view_active.anchor[2])
                                - float(agent_location[2]),
                            )
                        )
                        if (
                            image_navigation_target == "object"
                            or object_gap_m < camera_gap_m
                        ):
                            visual_pose_goal_position = np.asarray(
                                [
                                    float(goal_view_active.anchor[0]),
                                    float(visual_pose_goal_position[1]),
                                    float(goal_view_active.anchor[2]),
                                ],
                                dtype=np.float32,
                            )
                            goal_view_trace["pursued_point"] = "object_anchor"
                    if goal_view_trace.get("pursued_point") == "object_anchor":
                        image_object_target_steps += 1
                    elif goal_view_trace.get("pursued_point") == "goal_camera":
                        image_goal_camera_target_steps += 1
                    visual_pose_goal_is_weak = not bool(args.visual_pose_allow_stop)
                    if not visual_pose_goal_metadata:
                        visual_pose_goal_metadata = {
                            "source": "visual_pose",
                            "matcher": XFEAT_MATCHER_NAME,
                            "update_step": int(goal_view_active.last_step),
                        }
                if goal_view_active is not None:
                    goal_view_active_step_count += 1
                goal_view_trace.update(goal_view_belief.summary())
                goal_view_trace["pursued"] = (
                    None if goal_view_active is None else goal_view_active.summary()
                )
                if goal_view_expired:
                    goal_view_trace["expired_now"] = [
                        item.summary() for item in goal_view_expired
                    ]
            visual_pose_goal_is_fresh = (
                visual_pose_goal_position is not None
                and (
                    goal_view_belief is not None
                    or image_evidence_max_age_steps <= 0
                    or visual_pose_evidence_age_steps <= image_evidence_max_age_steps
                )
            )
            if (
                goal_view_belief is None
                and bool(args.visual_pose_clear_stale_goal)
                and visual_pose_goal_position is not None
                and not bool(visual_pose_goal_is_fresh)
                and int(image_evidence_max_age_steps) > 0
                and visual_pose_evidence_age_steps
                > int(image_evidence_max_age_steps)
            ):
                stale_visual_goal_m = world_position_list(visual_pose_goal_position)
                stale_visual_goal_age = int(step_index - visual_pose_last_update_step)
                visual_pose_goal_position = None
                visual_pose_goal_is_weak = False
                visual_pose_goal_metadata = {}
                visual_pose_trace.update(
                    {
                        "active": False,
                        "cleared_stale_goal": True,
                        "cleared_stale_goal_m": stale_visual_goal_m,
                        "cleared_stale_goal_age_steps": int(stale_visual_goal_age),
                        "clear_stale_goal_max_age_steps": int(image_evidence_max_age_steps),
                    }
                )
            visual_pose_distance_m: float | None = None
            if visual_pose_goal_is_fresh:
                agent_state_for_visual_preempt = env.get_agent_state()
                agent_position_for_visual_preempt = np.asarray(
                    agent_state_for_visual_preempt.position,
                    dtype=np.float32,
                ).reshape(3)
                visual_delta = (
                    np.asarray(visual_pose_goal_position, dtype=np.float32).reshape(3)
                    - agent_position_for_visual_preempt
                )
                visual_pose_distance_m = float(np.linalg.norm(visual_delta[[0, 2]]))
            # Stop at the depth floor, not R, since the goal is only an estimate
            goal_view_force_plan = False
            if (
                goal_view_belief is not None
                and goal_view_active is not None
                and servo_forward_blocked
                and not visual_pursuit_planned
            ):
                # Servo blocked: verify here if already within R of the object
                if goal_view_belief.note_blocked(
                    goal_view_active,
                    np.asarray(env.get_agent_state().position, dtype=np.float32),
                ):
                    goal_view_trace["verify_from_obstruction"] = True
            visual_pose_stop_distance_m = float(
                scales.terminal_approach_distance_m
            )
            if image_target_localization == "depth_cluster":
                # Centroid lies on the object surface, so the depth floor may be
                # unreachable
                visual_pose_stop_distance_m = max(
                    float(min_depth),
                    float(success_radius) - float(forward_step_size),
                )
            if goal_view_belief is not None and goal_view_active is not None:
                visual_pose_stop_distance_m = float(
                    goal_view_belief.arrival_radius_m(
                        goal_view_active,
                        np.asarray(env.get_agent_state().position, dtype=np.float32),
                    )
                )
                if (
                    visual_pose_distance_m is not None
                    and visual_pose_distance_m > visual_pose_stop_distance_m
                    and not visual_pursuit_planned
                ):
                    # Hand over to the map planner after full_turn_actions steps without
                    # progress
                    goal_view_stalled = goal_view_belief.note_progress(
                        goal_view_active,
                        float(visual_pose_distance_m),
                        float(scales.planar_progress_quantum_m),
                    )
                    if goal_view_stalled >= int(scales.full_turn_actions):
                        goal_view_force_plan = True
                        goal_view_belief.planner_handovers += 1
                        goal_view_active.stalled_steps = 0
                        goal_view_trace["planner_handover"] = True
            if (
                goal_view_belief is not None
                and goal_view_active is not None
                and visual_pose_distance_m is not None
                and visual_pose_distance_m <= visual_pose_stop_distance_m
                and visual_pose_stop_distance_m > float(min_depth)
                and sgf_navigation_map is not None
                and not goal_view_active.blocked_commit
            ):
                # Use the depth floor if the goal is more than R away by geodesic
                try:
                    band_agent = np.asarray(
                        env.get_agent_state().position, dtype=np.float32
                    ).reshape(3)
                    band_goal = np.asarray(
                        visual_pose_goal_position, dtype=np.float32
                    ).reshape(3)
                    band_snapped = sgf_navigation_map.snap_to_free(
                        band_goal, max_radius_m=float(success_radius)
                    )
                    band_geodesic_m = float("inf")
                    if band_snapped is not None:
                        band_geodesic_m = float(
                            sgf_navigation_map.geodesic_distance(
                                band_agent, band_snapped
                            )
                        ) + float(
                            np.linalg.norm((band_snapped - band_goal)[[0, 2]])
                        )
                except Exception:
                    band_geodesic_m = float("inf")
                if math.isfinite(band_geodesic_m):
                    goal_view_trace["band_geodesic_m"] = round(band_geodesic_m, 4)
                    if band_geodesic_m > float(success_radius):
                        visual_pose_stop_distance_m = float(min_depth)
                        goal_view_trace["band_geodesic_floor"] = True
            visual_pose_stop_distance_trace = {
                "distance_m": round(float(visual_pose_stop_distance_m), 4),
            }
            visual_pose_effective_stop_trace: dict[str, Any] = {
                "distance_m": round(float(visual_pose_stop_distance_m), 4),
            }
            close_visual_pose_stop_preempts_navigation = (
                args.visual_pose_servo
                and visual_pose_goal_is_fresh
                and not bool(visual_pose_goal_is_weak)
                and visual_pose_distance_m is not None
                and visual_pose_distance_m
                <= float(visual_pose_stop_distance_m)
            )
            visual_pursuit_goal_position: np.ndarray | None = None
            if (
                visual_pose_goal_is_fresh
                and not bool(visual_pose_goal_is_weak)
                and visual_pose_distance_m is not None
            ):
                visual_pursuit_goal_position = np.asarray(
                    visual_pose_goal_position, dtype=np.float32
                ).reshape(3)
            visual_pursuit_stop_distance_m = float(
                visual_pose_stop_distance_m
            )
            visual_pose_preempts_sgf = bool(
                args.visual_pose_servo
                and visual_pursuit_goal_position is not None
            )
            # Reachable-viewpoint pursuit: once a servo step is blocked, plan to the
            # closest free standpoint to the reference viewpoint
            visual_pursuit_command = None
            visual_pursuit_trace: dict[str, Any] = {"planned": False}
            pursuit_abandoned = False
            pursuit_agent_position = None
            pursuit_standpoint_distance_m = None
            # FMM controller: the planner pursues from the start, never the servo
            pursuit_controller_fmm = image_pursuit_controller == "fmm"
            if pursuit_controller_fmm:
                visual_pursuit_trace["controller"] = "fmm"
            if visual_pursuit_goal_position is None:
                visual_pursuit_planned = False
                visual_pursuit_standpoint = None
                visual_pursuit_standpoint_reached_once = False
            elif (
                not pursuit_controller_fmm
                and not bool(getattr(args, "image_reachable_pursuit", False))
            ):
                visual_pursuit_planned = False
                visual_pursuit_standpoint = None
                visual_pursuit_standpoint_reached_once = False
            elif (
                not pursuit_controller_fmm
                and (servo_forward_blocked or goal_view_force_plan)
                and visual_pursuit_standpoint_reached_once
                and not close_visual_pose_stop_preempts_navigation
            ):
                # Blocked again at the closest free standpoint: give up from this side
                pursuit_abandoned = True
                pursuit_agent_position = np.asarray(
                    env.get_agent_state().position, dtype=np.float32
                ).reshape(3)
                visual_pursuit_trace["blocked_at_standpoint"] = True
            elif (
                visual_pursuit_navigator is not None
                and sgf_navigation_map is not None
                and (
                    visual_pursuit_planned
                    or servo_forward_blocked
                    or goal_view_force_plan
                    or pursuit_controller_fmm
                )
                and not close_visual_pose_stop_preempts_navigation
            ):
                if not visual_pursuit_planned:
                    visual_pursuit_navigator.reset()
                    visual_pursuit_planned = True
                    visual_pursuit_best_distance_m = float("inf")
                    visual_pursuit_no_progress_steps = 0
                pursuit_agent_state = env.get_agent_state()
                pursuit_agent_position = np.asarray(
                    pursuit_agent_state.position, dtype=np.float32
                ).reshape(3)
                pursuit_standpoint: np.ndarray | None = None
                try:
                    pursuit_snapped = sgf_navigation_map.snap_to_free(
                        visual_pursuit_goal_position
                    )
                except Exception:
                    pursuit_snapped = None
                if pursuit_snapped is not None and np.all(
                    np.isfinite(pursuit_snapped)
                ):
                    pursuit_standpoint = (
                        np.asarray(pursuit_snapped, dtype=np.float32)
                        .reshape(3)
                        .copy()
                    )
                    pursuit_standpoint[1] = float(pursuit_agent_position[1])
                visual_pursuit_standpoint = pursuit_standpoint
                pursuit_standpoint_distance_m = (
                    None
                    if pursuit_standpoint is None
                    else float(
                        np.linalg.norm(
                            (pursuit_standpoint - pursuit_agent_position)[[0, 2]]
                        )
                    )
                )
                pursuit_reach_radius_m = float(scales.frontier_reach_radius_m)
                if pursuit_controller_fmm:
                    # Without a servo, the planner must reach the commitment band itself
                    pursuit_reach_radius_m = min(
                        pursuit_reach_radius_m,
                        float(visual_pursuit_stop_distance_m),
                    )
                visual_pursuit_trace.update(
                    {
                        "planned": True,
                        "standpoint_m": world_position_list(pursuit_standpoint),
                        "standpoint_distance_m": (
                            None
                            if pursuit_standpoint_distance_m is None
                            else round(float(pursuit_standpoint_distance_m), 4)
                        ),
                        "reach_radius_m": round(pursuit_reach_radius_m, 4),
                    }
                )
                # Standpoint reached: hand back to the servo, which may still enter the
                # band
                pursuit_standpoint_reached = bool(
                    pursuit_standpoint_distance_m is not None
                    and pursuit_standpoint_distance_m <= pursuit_reach_radius_m
                )
                if pursuit_standpoint_reached:
                    visual_pursuit_planned = False
                    visual_pursuit_standpoint_reached_once = True
                    visual_pursuit_trace["standpoint_reached"] = True
                    if pursuit_controller_fmm:
                        pursuit_abandoned = True
                if pursuit_standpoint_distance_m is not None:
                    if (
                        pursuit_standpoint_distance_m
                        < visual_pursuit_best_distance_m
                        - float(scales.planar_progress_quantum_m)
                    ):
                        visual_pursuit_best_distance_m = float(
                            pursuit_standpoint_distance_m
                        )
                        visual_pursuit_no_progress_steps = 0
                    else:
                        visual_pursuit_no_progress_steps += 1
                    visual_pursuit_trace["no_progress_steps"] = int(
                        visual_pursuit_no_progress_steps
                    )
                    if (
                        not pursuit_abandoned
                        and visual_pursuit_no_progress_steps
                        >= int(scales.full_turn_actions)
                    ):
                        visual_pursuit_trace["stalled"] = True
                        pursuit_abandoned = True
                if not pursuit_abandoned and not pursuit_standpoint_reached and pursuit_standpoint is not None:
                    try:
                        visual_pursuit_command = visual_pursuit_navigator.next_action(
                            agent_state=pursuit_agent_state,
                            goal_position=pursuit_standpoint,
                            navigation_map=sgf_navigation_map,
                            allow_stop=False,
                            stop_radius_m=pursuit_reach_radius_m,
                        )
                    except Exception as exc:
                        visual_pursuit_command = None
                        visual_pursuit_trace["error"] = str(exc)
                    if visual_pursuit_command is not None:
                        visual_pursuit_trace.update(
                            {
                                "action": int(visual_pursuit_command.action),
                                "goal_reached": bool(
                                    visual_pursuit_command.goal_reached
                                ),
                                "goal_blocked": bool(
                                    visual_pursuit_command.goal_blocked
                                ),
                                "planner_replan_failed": bool(
                                    visual_pursuit_command.planner_replan_failed
                                ),
                                "recovery_reason": (
                                    visual_pursuit_command.recovery_reason
                                ),
                            }
                        )
                        if visual_pursuit_command.goal_reached:
                            visual_pursuit_navigator.cancel_pending_action()
                            visual_pursuit_command = None
                            visual_pursuit_planned = False
                            visual_pursuit_standpoint_reached_once = True
                            visual_pursuit_trace["standpoint_reached"] = True
                            if pursuit_controller_fmm:
                                pursuit_abandoned = True
                        elif (
                            visual_pursuit_command.goal_blocked
                            or visual_pursuit_command.planner_replan_failed
                        ):
                            visual_pursuit_navigator.cancel_pending_action()
                            visual_pursuit_command = None
                            pursuit_abandoned = True
                elif pursuit_standpoint is None:
                    pursuit_abandoned = True
            if visual_pursuit_planned or pursuit_abandoned:
                if pursuit_controller_fmm:
                    image_pursuit_fmm_actions += 1
                else:
                    reachable_pursuit_action_count += 1
            if pursuit_abandoned:
                visual_pursuit_trace["abandoned"] = True
                goal_view_verify_here = False
                if goal_view_belief is not None and goal_view_active is not None:
                    # Verify here if within R of the object, else wait for a new vantage
                    goal_view_verify_here = goal_view_belief.approach_exhausted(
                        goal_view_active,
                        np.asarray(env.get_agent_state().position, dtype=np.float32),
                    )
                    goal_view_trace["abandoned"] = True
                    goal_view_trace["verify_from_standpoint"] = bool(
                        goal_view_verify_here
                    )
                    if not goal_view_verify_here:
                        # Not reachable from this side: its bearing becomes a frontier
                        # search cue
                        cue_agent = np.asarray(
                            env.get_agent_state().position, dtype=np.float32
                        ).reshape(3)
                        cue_target = (
                            goal_view_active.anchor
                            if goal_view_active.anchor is not None
                            else goal_view_active.position
                        )
                        cue_dx = float(cue_target[0]) - float(cue_agent[0])
                        cue_dz = float(cue_target[2]) - float(cue_agent[2])
                        if (
                            isinstance(sgf_explorer, VisualMemoryUtilityExplorer)
                            and math.hypot(cue_dx, cue_dz) > 1e-6
                        ):
                            with timing.measure("image_goal", "visual_memory_heading"):
                                goal_view_trace["search_cue"] = bool(
                                    sgf_explorer.update_visual_memory_heading(
                                        position=cue_agent,
                                        heading_rad=math.atan2(-cue_dx, -cue_dz),
                                        similarity=1.0,
                                        step_index=int(step_index),
                                    )
                                )
                        goal_view_active = None
            if pursuit_abandoned and not goal_view_verify_here:
                visual_pose_goal_position = None
                visual_pose_goal_is_weak = False
                visual_pose_goal_metadata = {}
                visual_pose_goal_is_fresh = False
                visual_pose_distance_m = None
                visual_pursuit_goal_position = None
                visual_pose_preempts_sgf = False
                visual_pursuit_planned = False
                visual_pursuit_standpoint = None
                visual_pursuit_standpoint_reached_once = False
            if pursuit_controller_fmm:
                visual_pose_preempts_sgf = bool(
                    visual_pose_preempts_sgf
                    and (
                        visual_pursuit_command is not None
                        or close_visual_pose_stop_preempts_navigation
                    )
                )
            visual_pose_trace["reachable_pursuit"] = visual_pursuit_trace
            # Panoramic sweep: a full in-place turn per sensor range travelled,
            # gated on a frontier's worth of unknown cells within range
            panorama_trace: dict[str, Any] = {}
            panorama_action: int | None = None
            if bool(args.panoramic_observation):
                panorama_agent_state = env.get_agent_state()
                panorama_position = np.asarray(
                    panorama_agent_state.position, dtype=np.float32
                ).reshape(3)
                panorama_budget_fraction = min(
                    1.0, float(step_index) / max(1.0, float(args.max_steps))
                )
                panorama_trigger_scale = float(
                    args.panoramic_observation_trigger_scale
                )
                if not math.isclose(
                    panorama_trigger_scale,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    panoramic_detector_scaled_interval_application_count += 1
                panorama_interval_mode = panoramic_observation_interval_mode(
                    searched_extent=bool(
                        args.panoramic_observation_searched_extent_interval
                    ),
                )
                if panorama_interval_mode == "searched_extent":
                    panoramic_searched_extent_interval_application_count += 1
                    # Interval grows with searched path length, in sensor ranges
                    panorama_searched_extent = max(
                        1.0,
                        float(panorama_path_since_start_m)
                        / max(1e-6, float(max_depth)),
                    )
                    panorama_trigger_distance_m = (
                        float(max_depth)
                        * panorama_trigger_scale
                        * panorama_searched_extent
                    )
                    panorama_trace["searched_extent"] = round(
                        panorama_searched_extent, 3
                    )
                else:
                    panorama_trigger_distance_m = float(max_depth) * (
                        panorama_trigger_scale
                        + (1.0 - panorama_trigger_scale)
                        * panorama_budget_fraction
                    )
                panorama_trace["trigger_distance_m"] = (
                    round(panorama_trigger_distance_m, 4)
                    if math.isfinite(panorama_trigger_distance_m)
                    else None
                )
                # Add this step's path segment only after computing the interval
                (
                    panorama_previous_position,
                    panorama_path_since_start_m,
                ) = accumulate_panoramic_path(
                    previous_position=panorama_previous_position,
                    current_position=panorama_position,
                    path_since_start_m=panorama_path_since_start_m,
                )
                panorama_due = panoramic_observation_trigger_due(
                    remaining=int(panorama_remaining),
                    anchor_position=panorama_anchor_position,
                    current_position=panorama_position,
                    trigger_distance_m=panorama_trigger_distance_m,
                )
                panorama_arrival_forced = False
                hypothesis_arrival_point = (
                    sgf_explorer.pop_hypothesis_arrival()
                    if sgf_explorer is not None
                    and hasattr(sgf_explorer, "pop_hypothesis_arrival")
                    else None
                )
                if (
                    vlm_trust_arrival_deadline is not None
                    and int(step_index) >= int(vlm_trust_arrival_deadline)
                ):
                    vlm_trust_disconfirmed += 1
                    vlm_trust_point = None
                    vlm_trust_arrival_deadline = None
                if (
                    hypothesis_arrival_point is not None
                    and vlm_trust_point is not None
                    and vlm_trust_arrival_deadline is None
                ):
                    vlm_trust_arrival_deadline = int(step_index) + int(
                        scales.full_turn_actions
                    ) + 1
                if (
                    hypothesis_arrival_point is not None
                    and bool(
                        getattr(args, "image_vlm_hypothesis_arrival_sweep", False)
                    )
                    and int(panorama_remaining) <= 0
                    and not instance_image_target_found
                ):
                    panorama_due = True
                    panorama_arrival_forced = True
                    panorama_trace["hypothesis_arrival_m"] = world_position_list(
                        hypothesis_arrival_point
                    )
                if panorama_due:
                    panorama_unknown_cells = unobserved_cells_in_range(
                        sgf_navigation_map,
                        position=panorama_position,
                        range_m=float(max_depth),
                    )
                    panorama_trace["unknown_cells_in_range"] = (
                        None
                        if panorama_unknown_cells is None
                        else int(panorama_unknown_cells)
                    )
                    panorama_gate_open = bool(
                        panorama_arrival_forced
                        or panoramic_observation_unknown_gate(
                            unknown_cells_in_range=panorama_unknown_cells,
                            min_unknown_cells=int(
                                args.sgf_explore_frontier_min_size
                            ),
                        )
                    )
                    if panorama_arrival_forced:
                        hypothesis_arrival_sweep_count += 1
                        panorama_trace["hypothesis_arrival_sweep"] = True
                    if panorama_gate_open:
                        panorama_remaining = int(
                            full_turn_observation_actions(
                                turn_angle_deg=float(env.turn_angle_deg)
                            )
                        )
                        panorama_count += 1
                        panorama_trace["started"] = True
                    else:
                        panorama_trace["skipped_fully_observed"] = True
                    panorama_anchor_position = panorama_position.copy()
                if panorama_remaining > 0:
                    panorama_action = int(HabitatSimActions.turn_left)
                    panorama_trace.update(
                        {
                            "active": True,
                            "remaining": int(panorama_remaining),
                            "episode_count": int(panorama_count),
                        }
                    )
            sgf_preplan_allowed = (
                sgf_navigation_map is not None
                and sgf_explorer is not None
                and sgf_navigator is not None
                and sgf_schedule_active
                and not (
                    text_sgf_enabled
                    and text_selected_target_node is not None
                    and text_target_plan_command is not None
                )
                and not bool(close_visual_pose_stop_preempts_navigation)
                and not bool(visual_pose_preempts_sgf)
                and panorama_action is None
            )
            if sgf_preplan_allowed:
                if (
                    text_navigator is sgf_navigator
                    and sgf_navigator.has_pending_action
                ):
                    sgf_explore_trace[
                        "orphan_pending_cancelled_before_plan"
                    ] = bool(sgf_navigator.cancel_pending_action())
                agent_state = env.get_agent_state()
                agent_position = np.asarray(agent_state.position, dtype=np.float32)
                if text_sgf_enabled and hasattr(sgf_explorer, "update_excluded_nodes"):
                    sgf_explorer.update_excluded_nodes(excluded_text_node_ids)
                if isinstance(sgf_explorer, VisualMemoryUtilityExplorer):
                    sgf_explorer.set_earned_visual_bonus_context(
                        target_unlocalized=bool(
                            str(args.task_mode) == "instance-image"
                            and not instance_image_target_found
                        ),
                        panoramic_corroborated=bool(
                            panorama_pose_corroborated
                        ),
                    )
                try:
                    (
                        sgf_plan_goal,
                        sgf_plan_command,
                        sgf_plan_action,
                        sgf_plan_mode,
                        sgf_plan_goal_reached,
                        sgf_plan_goal_retired,
                    ) = plan_exploration_action(
                        navigation_map=sgf_navigation_map,
                        agent_position=agent_position,
                        agent_state=agent_state,
                        explorer=sgf_explorer,
                        navigator=sgf_navigator,
                        stop_radius_m=float(scales.frontier_reach_radius_m),
                        mode="sgf_explore",
                        local_blocked_retire_stall_count=int(
                            scales.full_turn_actions
                        ),
                        step_index=int(step_index),
                    )
                    sgf_explore_trace.update(
                        {
                            "planned": True,
                            "mode": sgf_plan_mode,
                            "goal_m": world_position_list(sgf_plan_goal),
                            "goal_reached": bool(sgf_plan_goal_reached),
                            "goal_retired": bool(sgf_plan_goal_retired),
                            "planned_action": int(sgf_plan_action),
                            "failed_goal_count": int(sgf_explorer.failed_goal_count),
                            "stagnating": bool(sgf_explorer.stagnating),
                            "stagnation_steps": int(sgf_explorer.stagnation_steps),
                        }
                    )
                    if sgf_plan_command is not None:
                        sgf_explore_trace["planner"] = {
                            "heading_error_deg": sgf_plan_command.heading_error_deg,
                            "goal_distance_m": sgf_plan_command.goal_distance_m,
                            "stalled_count": int(sgf_plan_command.stalled_count),
                            "replan_failed_count": int(sgf_plan_command.replan_failed_count),
                            "progress_stall_count": int(sgf_plan_command.progress_stall_count),
                            "planner_replan_failed": bool(sgf_plan_command.planner_replan_failed),
                            "goal_blocked": bool(sgf_plan_command.goal_blocked),
                            "local_blocked": bool(sgf_plan_command.local_blocked),
                            "block_scope": str(sgf_plan_command.block_scope),
                            "recovery_active": bool(sgf_plan_command.recovery_active),
                            "recovery_reason": sgf_plan_command.recovery_reason,
                            "short_term_goal_m": world_position_list(sgf_plan_command.short_term_goal),
                        }
                except Exception as exc:
                    sgf_explore_planning_errors += 1
                    sgf_explore_trace.update({"planned": False, "active": False, "error": str(exc)})

            navigation_action = int(HabitatSimActions.turn_left)
            habitat_action = int(navigation_action)
            stop_suppressed = False
            visual_pose_navigation_forced = False
            sgf_explore_forced = False
            panorama_forced = False
            sgf_proposal_passthrough_forced = False
            sgf_depth_guard_forced = False
            forced_budget_stop = False
            text_target_navigation_forced = False
            sgf_stop_forced = False
            sgf_stop_relaxed_forced = False
            sgf_stop_trace: dict[str, Any] = {
                "enabled": bool(args.sgf_stop and text_sgf_enabled),
                "requested": False,
                "distance_m": None,
                "success_radius_m": round(float(success_radius), 4),
            }
            stop_validation_trace: dict[str, Any] = {
                "enabled": bool(args.stop_validation and args.task_mode == "text"),
                "attempted": False,
                "approved": None,
                "source": None,
            }
            navigation_guard_trace: dict[str, Any] = {"checked": False, "blocked": False}
            visual_stop_candidate_radius_m = float(
                visual_pose_stop_distance_m
            )
            close_visual_pose_stop_preempts_navigation_for_stop = (
                args.visual_pose_servo
                and visual_pose_goal_is_fresh
                and not bool(visual_pose_goal_is_weak)
                and visual_pose_distance_m is not None
                and visual_pose_distance_m
                <= float(visual_pose_stop_distance_m)
            )
            visual_stop_guard_trace = {
                "dynamic_stop_distance": visual_pose_stop_distance_trace,
                "effective_stop_distance": visual_pose_effective_stop_trace,
                "suppressed": False,
            }
            # Endgame return bookkeeping
            endgame_return_act = False
            endgame_hard_reserve = False
            endgame_return_priority = False
            endgame_return_activation_details: dict[str, Any] = {}
            step_endgame_return_trace: dict[str, Any] = {}
            endgame_return_agent_state = None
            endgame_navigation_map = None
            endgame_pending_command: Any | None = None
            endgame_navigator_action_selected = False
            if endgame_return_planner is not None:
                preliminary_endgame_stop = bool(
                    habitat_action == int(HabitatSimActions.stop)
                )
                fallback_return_check_allowed = bool(
                    endgame_return_planner.active_source == "dormant_approved"
                    or (
                        endgame_return_planner.candidate is None
                        and endgame_return_planner.dormant_approved_candidate
                        is not None
                    )
                )
                if (
                    visual_pose_goal_position is not None
                    and not bool(visual_pose_goal_is_weak)
                    and int(step_index - visual_pose_last_update_step)
                    <= max(1, int(image_evidence_max_age_steps))
                ):
                    endgame_return_planner.note_fresh_evidence(int(step_index))
                if should_run_endgame_return_bookkeeping(
                    planner_disabled=bool(endgame_return_planner.disabled),
                    proposed_stop=bool(
                        habitat_action == int(HabitatSimActions.stop)
                    ),
                    fallback_return_allowed=fallback_return_check_allowed,
                ):
                    endgame_navigation_map = (
                        sgf_navigation_map
                        if args.task_mode == "instance-image"
                        else text_navigation_map
                    )
                    if (
                        endgame_navigation_map is not None
                        and endgame_return_navigator is not None
                    ):
                        endgame_return_agent_state = env.get_agent_state()
                        endgame_turn_fraction = float(
                            endgame_nav_turn_actions
                        ) / float(
                            max(
                                1,
                                endgame_nav_turn_actions
                                + endgame_nav_forward_actions,
                            )
                        )
                        endgame_travel_inflation = float(
                            TerminalCommitPlanner.travel_inflation(
                                endgame_turn_fraction
                            )
                        )
                        if not math.isfinite(endgame_travel_inflation):
                            endgame_travel_inflation = float(
                                max(1, endgame_nav_turn_actions + 1)
                            )
                        endgame_return_distance_m: float | None = None
                        endgame_geodesic_candidate = (
                            endgame_return_planner.return_candidate
                            or endgame_return_planner.dormant_approved_candidate
                        )
                        if endgame_geodesic_candidate is not None:
                            try:
                                endgame_geodesic = float(
                                    endgame_navigation_map.geodesic_distance(
                                        np.asarray(
                                            endgame_return_agent_state.position,
                                            dtype=np.float32,
                                        ),
                                        np.asarray(
                                            endgame_geodesic_candidate.position,
                                            dtype=np.float32,
                                        ),
                                    )
                                )
                            except Exception:
                                endgame_geodesic = float("nan")
                            if math.isfinite(endgame_geodesic) and endgame_geodesic >= 0.0:
                                endgame_return_distance_m = endgame_geodesic
                        (
                            endgame_return_act,
                            endgame_return_activation_details,
                        ) = endgame_return_planner.should_activate(
                            step=int(step_index),
                            max_steps=int(args.max_steps),
                            current_position=np.asarray(
                                endgame_return_agent_state.position,
                                dtype=np.float32,
                            ),
                            return_distance_m=endgame_return_distance_m,
                            turn_fraction=endgame_turn_fraction,
                            overhead_steps=int(endgame_arrival_overhead_steps),
                        )
                        endgame_return_activation_details.update(
                            {
                                "turn_fraction": round(float(endgame_turn_fraction), 4),
                                "travel_inflation": round(
                                    float(endgame_travel_inflation), 4
                                ),
                                "return_distance_m": (
                                    None
                                    if endgame_return_distance_m is None
                                    else round(float(endgame_return_distance_m), 4)
                                ),
                                "arrival_overhead_steps": int(
                                    endgame_arrival_overhead_steps
                                ),
                            }
                        )
                        if (
                            preliminary_endgame_stop
                            and endgame_return_act
                            and endgame_return_planner.active_source
                            == "dormant_approved"
                            and endgame_return_planner.activated_step
                            == int(step_index)
                        ):
                            # Validate this STOP once before the return latches
                            endgame_return_activation_details[
                                "deferred_for_stop_validation"
                            ] = True
                            text_sgf_trace["fallback_return_latched"] = dict(
                                endgame_return_activation_details
                            )
                            endgame_return_act = False
                        endgame_hard_reserve = bool(
                            endgame_return_act
                            and endgame_return_planner.hard_reserve_committed
                        )
                        endgame_return_priority = bool(endgame_hard_reserve)
            goal_view_alignment_action: int | None = None
            depth_cluster_alignment = False
            if (
                close_visual_pose_stop_preempts_navigation_for_stop
                and image_target_localization == "depth_cluster"
                and visual_pose_goal_position is not None
            ):
                alignment_state = env.get_agent_state()
                alignment_delta = np.asarray(
                    [
                        float(visual_pose_goal_position[0])
                        - float(alignment_state.position[0]),
                        float(visual_pose_goal_position[2])
                        - float(alignment_state.position[2]),
                    ],
                    dtype=np.float64,
                )
                alignment_norm = float(np.linalg.norm(alignment_delta))
                if alignment_norm >= float(min_depth):
                    alignment_error_deg = signed_heading_error_deg(
                        np.asarray(
                            rotation_to_forward_xz(alignment_state.rotation),
                            dtype=np.float64,
                        ),
                        alignment_delta / alignment_norm,
                    )
                    if abs(alignment_error_deg) > 0.5 * float(args.turn_angle):
                        goal_view_alignment_action = (
                            int(HabitatSimActions.turn_left)
                            if alignment_error_deg < 0.0
                            else int(HabitatSimActions.turn_right)
                        )
                        depth_cluster_alignment = True
            if (
                close_visual_pose_stop_preempts_navigation_for_stop
                and goal_view_belief is not None
                and goal_view_active is not None
                and image_goal_view_alignment == "active"
            ):
                alignment_state = env.get_agent_state()
                alignment_target_xz = goal_view_belief.alignment_target_xz(
                    goal_view_active,
                    np.asarray(alignment_state.position, dtype=np.float32),
                )
                alignment_forward_xz = rotation_to_forward_xz(
                    alignment_state.rotation
                )
                if alignment_target_xz is not None:
                    alignment_error_deg = signed_heading_error_deg(
                        np.asarray(alignment_forward_xz, dtype=np.float64),
                        alignment_target_xz,
                    )
                    goal_view_trace["alignment_error_deg"] = round(
                        float(alignment_error_deg), 2
                    )
                    if abs(alignment_error_deg) > float(
                        goal_view_belief.heading_tolerance_deg
                    ):
                        goal_view_alignment_action = (
                            int(HabitatSimActions.turn_left)
                            if alignment_error_deg < 0.0
                            else int(HabitatSimActions.turn_right)
                        )
            if (
                goal_view_alignment_action is None
                and close_visual_pose_stop_preempts_navigation_for_stop
                and goal_view_belief is not None
                and goal_view_active is not None
                and not goal_view_belief.corroborated(goal_view_active)
                and int(goal_view_active.last_step) >= int(step_index)
            ):
                # The frame that produced the estimate cannot test it
                goal_view_trace["uncorroborated_deferred"] = True
                close_visual_pose_stop_preempts_navigation_for_stop = False
                close_visual_pose_stop_preempts_navigation = False
                visual_pose_preempts_sgf = False
                visual_pursuit_goal_position = None
                visual_pose_goal_position = None
                visual_pose_goal_is_weak = False
                visual_pose_goal_metadata = {}
                visual_pose_goal_is_fresh = False
            elif (
                goal_view_alignment_action is None
                and close_visual_pose_stop_preempts_navigation_for_stop
                and goal_view_belief is not None
                and goal_view_active is not None
                and not goal_view_belief.corroborated(goal_view_active)
            ):
                goal_view_belief.note_uncorroborated_view(
                    goal_view_active,
                    np.asarray(env.get_agent_state().position, dtype=np.float32),
                )
                goal_view_trace["uncorroborated_refuted"] = True
                goal_view_active = None
                close_visual_pose_stop_preempts_navigation_for_stop = False
                close_visual_pose_stop_preempts_navigation = False
                visual_pose_preempts_sgf = False
                visual_pursuit_goal_position = None
                visual_pose_goal_position = None
                visual_pose_goal_is_weak = False
                visual_pose_goal_metadata = {}
                visual_pose_goal_is_fresh = False
            if goal_view_alignment_action is not None:
                if depth_cluster_alignment:
                    image_depth_cluster_alignment_turns += 1
                else:
                    goal_view_belief.alignment_turns += 1
                habitat_action = int(goal_view_alignment_action)
                navigation_action = int(goal_view_alignment_action)
                visual_pose_navigation_forced = True
                goal_view_trace["aligning"] = True
                visual_pose_navigation_trace = {
                    "active": True,
                    "goal_m": world_position_list(visual_pose_goal_position),
                    "weak_evidence": bool(visual_pose_goal_is_weak),
                    "weak_stop_suppressed": False,
                    "age_steps": int(step_index - visual_pose_last_update_step),
                    "distance_m": round(float(visual_pose_distance_m or 0.0), 4),
                    "stop_distance_m": round(
                        float(visual_pose_stop_distance_m),
                        4,
                    ),
                    "action": int(goal_view_alignment_action),
                    "close_stop": False,
                    "aligning": True,
                    "goal_source": str(visual_pose_goal_metadata.get("source", "")),
                    "stop_guard": visual_stop_guard_trace,
                }
            elif close_visual_pose_stop_preempts_navigation_for_stop:
                if goal_view_belief is not None and goal_view_active is not None:
                    goal_view_belief.band_commits += 1
                    goal_view_trace["committed"] = True
                    if (
                        float(visual_pose_stop_distance_m) > float(min_depth)
                        and not goal_view_active.blocked_commit
                    ):
                        goal_view_belief.wide_band_commits += 1
                    if int(goal_view_active.consensus) < 2 and (
                        goal_view_belief.corroborated(goal_view_active)
                    ):
                        image_single_corroboration_commits += 1
                visual_stop_candidate_radius_m = float(
                    visual_pose_stop_distance_m
                )
                habitat_action = int(HabitatSimActions.stop)
                navigation_action = int(HabitatSimActions.stop)
                visual_pose_navigation_forced = True
                visual_pose_navigation_trace = {
                    "active": True,
                    "goal_m": world_position_list(visual_pose_goal_position),
                    "weak_evidence": bool(visual_pose_goal_is_weak),
                    "weak_stop_suppressed": False,
                    "age_steps": int(step_index - visual_pose_last_update_step),
                    "distance_m": round(float(visual_pose_distance_m or 0.0), 4),
                    "stop_distance_m": round(
                        float(visual_pose_stop_distance_m),
                        4,
                    ),
                    "action": int(HabitatSimActions.stop),
                    "close_stop": True,
                    "goal_source": str(visual_pose_goal_metadata.get("source", "")),
                    "stop_guard": visual_stop_guard_trace,
                }
            elif visual_pose_preempts_sgf and not endgame_return_priority:
                agent_state = env.get_agent_state()
                agent_position = np.asarray(
                    agent_state.position, dtype=np.float32
                ).reshape(3)
                delta = np.asarray(
                    visual_pursuit_goal_position, dtype=np.float32
                ).reshape(3) - agent_position
                delta_xz = np.asarray([delta[0], delta[2]], dtype=np.float32)
                distance_m = float(np.linalg.norm(delta_xz))
                visual_stop_candidate_radius_m = float(
                    visual_pursuit_stop_distance_m
                )
                servo_action = int(HabitatSimActions.stop)
                heading_deg = 0.0
                if visual_pursuit_command is not None:
                    servo_action = int(visual_pursuit_command.action)
                    heading_deg = float(
                        visual_pursuit_command.heading_error_deg or 0.0
                    )
                    servo_prev_action_was_turn = False
                elif distance_m > float(visual_pursuit_stop_distance_m):
                    rotation = quaternion.as_rotation_matrix(
                        agent_state.rotation
                    ).astype(np.float32)
                    forward = rotation @ np.array(
                        [0.0, 0.0, -1.0], dtype=np.float32
                    )
                    forward_xz = np.asarray(
                        [forward[0], forward[2]], dtype=np.float32
                    )
                    forward_norm = float(np.linalg.norm(forward_xz))
                    if forward_norm > 1e-6 and distance_m > 1e-6:
                        forward_xz /= forward_norm
                        target_xz = delta_xz / distance_m
                        dot = float(
                            np.clip(np.dot(forward_xz, target_xz), -1.0, 1.0)
                        )
                        cross = float(
                            forward_xz[0] * target_xz[1]
                            - forward_xz[1] * target_xz[0]
                        )
                        heading_deg = math.degrees(math.atan2(cross, dot))
                    # Goal bearing shift over the last servo turn, both from this
                    # position
                    goal_xz_now = np.asarray(
                        [delta[0] + agent_position[0], delta[2] + agent_position[2]],
                        dtype=np.float64,
                    )
                    if servo_prev_action_was_turn and servo_prev_goal_xz is not None:
                        bearing_now = math.atan2(
                            float(goal_xz_now[0] - agent_position[0]),
                            -float(goal_xz_now[1] - agent_position[2]),
                        )
                        bearing_prev = math.atan2(
                            float(servo_prev_goal_xz[0] - agent_position[0]),
                            -float(servo_prev_goal_xz[1] - agent_position[2]),
                        )
                        shift = abs(
                            math.degrees(
                                (bearing_now - bearing_prev + math.pi) % (2.0 * math.pi)
                                - math.pi
                            )
                        )
                        servo_turn_shift_samples.append(float(shift))
                        del servo_turn_shift_samples[: -int(scales.full_turn_actions)]
                    view_shift_deg = (
                        float(np.mean(servo_turn_shift_samples))
                        if servo_turn_shift_samples
                        else 0.0
                    )
                    # Dead-band, no tighter than the arrival-disc bearing
                    arrival_disc_deg = math.degrees(
                        math.asin(
                            min(1.0, float(visual_pursuit_stop_distance_m) / max(1e-3, distance_m))
                        )
                    )
                    threshold = max(
                        0.5 * (float(args.turn_angle) + view_shift_deg),
                        arrival_disc_deg,
                    )
                    if heading_deg < -threshold:
                        servo_action = int(HabitatSimActions.turn_left)
                    elif heading_deg > threshold:
                        servo_action = int(HabitatSimActions.turn_right)
                    else:
                        servo_action = int(HabitatSimActions.move_forward)
                    image_straight_servo_actions += 1
                    servo_prev_goal_xz = goal_xz_now.copy()
                    servo_prev_action_was_turn = servo_action in {
                        int(HabitatSimActions.turn_left),
                        int(HabitatSimActions.turn_right),
                    }
                    visual_pose_trace["servo_dead_band"] = {
                        "threshold_deg": round(float(threshold), 2),
                        "view_shift_deg": round(float(view_shift_deg), 2),
                        "arrival_disc_deg": round(float(arrival_disc_deg), 2),
                        "samples": int(len(servo_turn_shift_samples)),
                    }
                else:
                    servo_prev_action_was_turn = False
                habitat_action = int(servo_action)
                navigation_action = int(servo_action)
                visual_pose_navigation_forced = True
                weak_stop_suppressed = False
                visual_pose_navigation_trace = {
                    "active": not bool(weak_stop_suppressed),
                    "goal_m": world_position_list(visual_pursuit_goal_position),
                    "weak_evidence": bool(visual_pose_goal_is_weak),
                    "weak_stop_suppressed": bool(weak_stop_suppressed),
                    "age_steps": int(step_index - visual_pose_last_update_step),
                    "distance_m": round(float(distance_m), 4),
                    "stop_distance_m": round(
                        float(visual_pursuit_stop_distance_m), 4
                    ),
                    "heading_deg": round(float(heading_deg), 2),
                    "action": int(servo_action),
                    "planned": bool(visual_pursuit_command is not None),
                    "goal_source": str(
                        visual_pose_goal_metadata.get("source", "")
                    ),
                    "stop_guard": visual_stop_guard_trace,
                }
            elif (
                text_sgf_enabled
                and bool(text_target_navigation_active)
                and text_target_plan_action is not None
                # Not preempted: the budget only just covers the return trip
                and not endgame_return_act
            ):
                habitat_action = int(text_target_plan_action)
                if habitat_action == int(HabitatSimActions.stop):
                    habitat_action = int(HabitatSimActions.turn_left)
                navigation_action = int(habitat_action)
                text_target_navigation_forced = True
                text_sgf_trace["target_navigation_action"] = int(habitat_action)
            elif endgame_return_act and endgame_return_planner is not None:
                # Return to the banked best-evidence pose and stop there
                endgame_candidate = endgame_return_planner.return_candidate
                endgame_current_position = np.asarray(
                    endgame_return_agent_state.position, dtype=np.float32
                )
                endgame_arrived = bool(
                    endgame_return_planner.arrived(endgame_current_position)
                )
                endgame_action: int | None = None
                if not endgame_arrived and endgame_candidate is not None:
                    try:
                        endgame_command = endgame_return_navigator.next_action(
                            agent_state=endgame_return_agent_state,
                            goal_position=endgame_candidate.position,
                            navigation_map=endgame_navigation_map,
                            allow_stop=True,
                            stop_radius_m=float(
                                endgame_return_planner.return_arrival_radius_m
                            ),
                        )
                    except Exception:
                        endgame_command = None
                    endgame_pending_command = endgame_command
                    if endgame_command is None or bool(
                        getattr(
                            endgame_command,
                            "planner_replan_failed",
                            False,
                        )
                    ):
                        endgame_return_planner.note_replan_failure()
                    else:
                        endgame_return_planner.note_replan_success()
                        endgame_action = int(endgame_command.action)
                        if bool(
                            getattr(
                                endgame_command, "goal_reached", False
                            )
                        ):
                            endgame_arrived = True
                if endgame_arrived or (
                    endgame_action is not None
                    and endgame_action == int(HabitatSimActions.stop)
                ):
                    endgame_press_cap = int(scales.terminal_press_steps) + 1
                    endgame_residual_walk_taken = False
                    if (
                        endgame_candidate is not None
                        and endgame_residual_walk_steps
                        < endgame_press_cap
                    ):
                        endgame_residual_m = float(
                            planar_distance_xz(
                                np.asarray(
                                    endgame_return_agent_state.position,
                                    dtype=np.float32,
                                ),
                                endgame_candidate.position,
                            )
                        )
                        if endgame_residual_m > 0.8 * float(
                            args.forward_step_size
                        ):
                            if endgame_return_navigator.has_pending_action:
                                endgame_return_navigator.cancel_pending_action()
                            endgame_pending_command = None
                            try:
                                endgame_residual_command = (
                                    endgame_return_navigator.next_action(
                                        agent_state=endgame_return_agent_state,
                                        goal_position=endgame_candidate.position,
                                        navigation_map=endgame_navigation_map,
                                        allow_stop=True,
                                        stop_radius_m=max(
                                            0.15,
                                            0.6
                                            * float(args.forward_step_size),
                                        ),
                                    )
                                )
                            except Exception:
                                endgame_residual_command = None
                            endgame_pending_command = endgame_residual_command
                            if (
                                endgame_residual_command is not None
                                and not bool(
                                    getattr(
                                        endgame_residual_command,
                                        "planner_replan_failed",
                                        False,
                                    )
                                )
                                and int(endgame_residual_command.action)
                                != int(HabitatSimActions.stop)
                            ):
                                endgame_residual_walk_steps += 1
                                habitat_action = int(
                                    endgame_residual_command.action
                                )
                                navigation_action = int(habitat_action)
                                sgf_explore_forced = False
                                endgame_residual_walk_taken = True
                                endgame_navigator_action_selected = True
                    if not endgame_residual_walk_taken:
                        navigation_action = int(HabitatSimActions.stop)
                        habitat_action = int(HabitatSimActions.stop)
                        endgame_return_stop_forced = True
                        endgame_stop_kind = "arrival"
                        # Budget-driven stop, exempt from stop validation
                        forced_budget_stop = True
                        endgame_navigator_action_selected = bool(
                            endgame_pending_command is not None
                            and int(endgame_pending_command.action)
                            == int(HabitatSimActions.stop)
                        )
                elif endgame_action is not None:
                    habitat_action = int(endgame_action)
                    navigation_action = int(endgame_action)
                    sgf_explore_forced = False
                    endgame_navigator_action_selected = True
                if (
                    habitat_action != int(HabitatSimActions.stop)
                    and step_index >= int(args.max_steps) - 1
                ):
                    navigation_action = int(HabitatSimActions.stop)
                    habitat_action = int(HabitatSimActions.stop)
                    endgame_return_stop_forced = True
                    endgame_stop_kind = "budget_cap"
                    forced_budget_stop = True
                step_endgame_return_trace = dict(endgame_return_planner.summary())
                step_endgame_return_trace.update(
                    {
                        "activation": dict(endgame_return_activation_details),
                        "hard_reserve": bool(endgame_hard_reserve),
                        "return_priority": bool(endgame_return_priority),
                        "arrived": bool(endgame_arrived),
                        "action": (
                            None if habitat_action is None else int(habitat_action)
                        ),
                        "stop_forced": bool(endgame_return_stop_forced),
                    }
                )
            elif panorama_action is not None:
                habitat_action = int(panorama_action)
                navigation_action = int(habitat_action)
                panorama_forced = True
            elif sgf_plan_action is not None:
                habitat_action = int(sgf_plan_action)
                if habitat_action == int(HabitatSimActions.stop):
                    habitat_action = int(HabitatSimActions.turn_left)
                navigation_action = int(habitat_action)
                sgf_explore_forced = True
                sgf_explore_trace.update(
                    {
                        "active": True,
                        "action": int(habitat_action),
                    }
                )
            elif (
                step_index >= int(args.max_steps) - 1
                and habitat_action != int(HabitatSimActions.stop)
            ):
                navigation_action = int(HabitatSimActions.stop)
                habitat_action = int(HabitatSimActions.stop)
                forced_budget_stop = True

            image_vlm_stop_candidate = bool(
                args.task_mode == "instance-image"
                and bool(args.image_vlm_stop_validation)
                and habitat_action == int(HabitatSimActions.stop)
                and not bool(forced_budget_stop)
                and goal_image_rgb is not None
            )
            image_vlm_stop_validation_trace.update(
                {
                    "attempt_count": int(
                        image_vlm_stop_validation_attempts
                    ),
                    "rejection_count": int(
                        image_vlm_stop_validation_rejections
                    ),
                    "visual_pose_navigation_forced": bool(visual_pose_navigation_forced),
                    "candidate": bool(image_vlm_stop_candidate),
                }
            )
            if image_vlm_stop_candidate:
                image_vlm_stop_validation_attempts += 1
                try:
                    with timing.measure("stop_verification", "image_vlm_stop_validation"):
                        image_vlm_stop_validation_trace.update(
                            validate_image_stop_with_vlm(
                                vlm_config=image_pair_vlm_config,
                                goal_image_rgb=goal_image_rgb,
                                current_rgb=rgb_observation,
                                timing=timing,
                            )
                        )
                except Exception as exc:
                    image_vlm_stop_validation_trace.update(
                        {
                            "attempted": True,
                            "approved": False,
                            "error": str(exc),
                        }
                    )
                validation_error_present = bool(
                    (
                        image_vlm_stop_validation_trace.get("error")
                        or image_vlm_stop_validation_trace.get(
                            "parse_error", False
                        )
                        or image_vlm_stop_validation_trace.get(
                            "same_target_visible"
                        )
                        is None
                    )
                    and image_vlm_stop_validation_trace.get("context_only")
                    is not True
                    and str(
                        image_vlm_stop_validation_trace.get(
                            "object_evidence", ""
                        )
                        or ""
                    ).strip().lower()
                    != "none"
                )
                unavailable_geometry_fallback = bool(
                    vlm_unavailable_strong_geometry_allows_stop(
                        validation_error_present=validation_error_present,
                        geometric_stop_candidate=bool(visual_pose_navigation_forced),
                        visual_target_updated=bool(
                            visual_stop_trace.get(
                                "target_updated", False
                            )
                        ),
                        visual_goal_source=str(
                            visual_pose_goal_metadata.get("source", "")
                        ),
                        visual_goal_updated_this_frame=bool(
                            visual_pose_goal_metadata.get("update_step")
                            == int(step_index)
                        ),
                        visual_goal_is_fresh=bool(
                            visual_pose_goal_is_fresh
                        ),
                        visual_goal_is_weak=bool(
                            visual_pose_goal_is_weak
                        ),
                        visual_goal_distance_m=(
                            float("nan")
                            if visual_pose_distance_m is None
                            else float(visual_pose_distance_m)
                        ),
                        stop_radius_m=float(
                            visual_stop_candidate_radius_m
                        ),
                        match_inliers=int(
                            visual_stop_trace.get("ransac_inliers", 0)
                            or 0
                        ),
                        min_match_inliers=int(
                            args.visual_pose_min_inliers
                        ),
                        match_confidence_sum=visual_stop_trace.get(
                            "confidence_sum", float("nan")
                        ),
                        min_match_confidence_sum=(
                            float(visual_target_config.min_confidence_sum)
                            if visual_target_config is not None
                            else float("inf")
                        ),
                        match_center_x_px=visual_stop_trace.get(
                            "match_center_x_px", float("nan")
                        ),
                        image_width_px=float(rgb_observation.shape[1]),
                        pose_accepted=bool(
                            visual_pose_trace.get("accepted", False)
                        ),
                        viewpoint_pnp_selected=bool(
                            visual_pose_trace.get("accepted", False)
                        ),
                        pnp_inlier_fraction=visual_pose_trace.get(
                            "viewpoint_pnp_inlier_fraction",
                            float("nan"),
                        ),
                        pose_confirms_visual_goal=bool(
                            visual_pose_trace.get(
                                "confirmed_existing_servo_goal", False
                            )
                        ),
                        pose_agent_distance_m=visual_pose_trace.get(
                            "agent_distance_m", float("nan")
                        ),
                        pose_alignment_m=visual_pose_trace.get(
                            "confirm_alignment_m", float("nan")
                        ),
                        max_alignment_m=float(args.forward_step_size),
                    )
                )
                if unavailable_geometry_fallback:
                    image_vlm_stop_validation_trace["approved"] = None
                    image_vlm_stop_validation_trace[
                        "unavailable_geometry_fallback"
                    ] = {
                        "applied": True,
                        "reason": "same_frame_xfeat_pnp_consensus",
                    }

                weak_visual_guard_reason = (
                    weak_visual_stop_rejection_reason(
                        vlm_approved=bool(
                            image_vlm_stop_validation_trace.get(
                                "approved", False
                            )
                        ),
                        visual_pose_reject_reason=(
                            visual_pose_trace.get("reject_reason")
                        ),
                        current_target_updated=bool(
                            visual_stop_trace.get(
                                "target_updated", False
                            )
                        ),
                    )
                )
                if (
                    weak_visual_guard_reason is not None
                    and goal_view_belief is not None
                    and goal_view_active is not None
                    and bool(visual_pose_navigation_forced)
                    and goal_view_belief.corroborated(goal_view_active)
                ):
                    image_vlm_stop_validation_trace[
                        "weak_visual_stop_guard_scoped_out"
                    ] = str(weak_visual_guard_reason)
                    weak_visual_guard_reason = None
                image_vlm_stop_validation_trace[
                    "weak_visual_stop_guard"
                ] = {
                    "scoped": bool(
                        image_vlm_stop_validation_trace.get(
                            "approved", False
                        )
                        and visual_pose_trace.get("reject_reason")
                        == "too_few_inliers"
                    ),
                    "accepted": bool(
                        weak_visual_guard_reason is None
                    ),
                    "reject_reason": weak_visual_guard_reason,
                }
                if weak_visual_guard_reason is not None:
                    image_vlm_stop_validation_trace[
                        "approved_before_weak_visual_stop_guard"
                    ] = bool(
                        image_vlm_stop_validation_trace.get(
                            "approved", False
                        )
                    )
                    image_vlm_stop_validation_trace["approved"] = False

                ordinary_image_vlm_rejected = bool(
                    image_vlm_stop_validation_trace.get("approved") is False
                )
                image_vlm_stop_validation_trace.update(
                    {
                        "rejected": bool(ordinary_image_vlm_rejected),
                    }
                )
                goal_view_hypothesis_survives = False
                if (
                    ordinary_image_vlm_rejected
                    and goal_view_belief is not None
                    and goal_view_active is not None
                    and bool(visual_pose_navigation_forced)
                ):
                    rejection_state = env.get_agent_state()
                    goal_view_refuted = goal_view_belief.note_rejection(
                        goal_view_active,
                        np.asarray(rejection_state.position, dtype=np.float32),
                    )
                    goal_view_trace["rejected"] = True
                    goal_view_trace["refuted"] = bool(goal_view_refuted)
                    goal_view_hypothesis_survives = not bool(goal_view_refuted)
                    if goal_view_refuted:
                        goal_view_active = None
                if ordinary_image_vlm_rejected:
                    image_vlm_stop_validation_rejections += 1
                    cleared_rejected_visual_goal = bool(
                        visual_pose_goal_position is not None
                        and visual_pose_goal_is_fresh
                        and not goal_view_hypothesis_survives
                    )
                    cleared_rejected_visual_goal_m = (
                        world_position_list(visual_pose_goal_position)
                        if cleared_rejected_visual_goal
                        else None
                    )
                    if cleared_rejected_visual_goal:
                        visual_pose_goal_position = None
                        visual_pose_goal_is_weak = False
                        visual_pose_goal_metadata = {}
                    replacement_action = (
                        safe_non_stop_navigation_action(sgf_plan_action)
                    )
                    navigation_action = int(replacement_action)
                    habitat_action = int(replacement_action)
                    sgf_proposal_passthrough_forced = bool(
                        sgf_plan_command is not None
                        and int(replacement_action)
                        == int(sgf_plan_command.action)
                    )
                    visual_pose_navigation_forced = False
                    stop_suppressed = True
                    image_vlm_stop_validation_trace.update(
                        {
                            "replacement_action": int(
                                replacement_action
                            ),
                            "replacement_source": (
                                "sgf_proposal"
                                if sgf_plan_action is not None
                                and int(replacement_action)
                                == int(sgf_plan_action)
                                else "fallback_scan"
                            ),
                            "cleared_visual_pose_goal": bool(
                                cleared_rejected_visual_goal
                            ),
                            "cleared_visual_pose_goal_m": (
                                cleared_rejected_visual_goal_m
                            ),
                        }
                    )
            image_vlm_stop_validation_trace.update(
                {
                    "attempt_count": int(
                        image_vlm_stop_validation_attempts
                    ),
                    "rejection_count": int(
                        image_vlm_stop_validation_rejections
                    ),
                }
            )

            # Past its activation step, a dormant return owns arbitration
            fallback_return_owns_late_action = bool(
                endgame_return_planner is not None
                and endgame_return_planner.active
                and endgame_return_planner.active_source
                == "dormant_approved"
                and endgame_return_planner.activated_step < int(step_index)
            )

            if fallback_return_owns_late_action:
                sgf_stop_trace["blocked_by_dormant_return"] = True

            if (
                bool(args.sgf_stop)
                and text_sgf_enabled
                and text_target_goal_position is not None
                and not fallback_return_owns_late_action
                and text_reobservation is None
            ):
                sgf_stop_distance = planar_distance_xz(
                    np.asarray(env.get_agent_state().position, dtype=np.float32),
                    text_target_goal_position,
                )
                # Far band (one step past success_radius): verdicts never stop
                sgf_stop_radius_m = float(scales.far_band_radius_m)
                sgf_stop_trace["distance_m"] = (
                    None
                    if not math.isfinite(sgf_stop_distance)
                    else round(float(sgf_stop_distance), 4)
                )
                sgf_relaxed_stop = (
                    math.isfinite(sgf_stop_distance)
                    and sgf_stop_distance > float(success_radius)
                    and sgf_stop_distance <= sgf_stop_radius_m
                )
                if sgf_relaxed_stop and text_selected_target_node is not None:
                    try:
                        sgf_relaxed_node_id = int(text_selected_target_node.get("id", -1))
                    except (TypeError, ValueError):
                        sgf_relaxed_node_id = -1
                    sgf_relaxed_cooling_down = (
                        sgf_relaxed_node_id >= 0
                        and relaxed_sgf_stop_cooldown_node_id == sgf_relaxed_node_id
                        and step_index < int(relaxed_sgf_stop_cooldown_until_step)
                    )
                    sgf_relaxed_persistence_ok = bool(
                        int(tracked_text_target_steps)
                        >= int(scales.full_turn_actions)
                    )
                    sgf_relaxed_stop = bool(
                        sgf_relaxed_persistence_ok
                        and not bool(sgf_relaxed_cooling_down)
                    )
                    sgf_stop_trace["relaxed_persistence_ok"] = bool(
                        sgf_relaxed_persistence_ok
                    )
                    sgf_stop_trace["relaxed_node_id"] = int(sgf_relaxed_node_id)
                    sgf_stop_trace["relaxed_tracked_steps"] = int(tracked_text_target_steps)
                    sgf_stop_trace["relaxed_cooldown_until_step"] = int(
                        relaxed_sgf_stop_cooldown_until_step
                    )
                    sgf_stop_trace["relaxed_cooling_down"] = bool(sgf_relaxed_cooling_down)
                sgf_stop_min_persistence = int(args.text_sgf_stop_min_target_persistence)
                sgf_stop_persistence_ok = (
                    sgf_stop_min_persistence <= 0
                    or int(tracked_text_target_steps) >= sgf_stop_min_persistence
                )
                sgf_stop_trace["min_target_persistence"] = int(sgf_stop_min_persistence)
                sgf_stop_trace["tracked_target_steps"] = int(tracked_text_target_steps)
                sgf_stop_trace["persistence_ok"] = bool(sgf_stop_persistence_ok)
                sgf_stop_trace["stop_radius_m"] = round(float(sgf_stop_radius_m), 4)
                sgf_stop_trace["relaxed_requested"] = bool(sgf_relaxed_stop)
                if (
                    math.isfinite(sgf_stop_distance)
                    and bool(sgf_stop_persistence_ok)
                    and (
                        sgf_stop_distance <= float(success_radius)
                        or bool(sgf_relaxed_stop)
                    )
                ):
                    navigation_action = int(HabitatSimActions.stop)
                    habitat_action = int(HabitatSimActions.stop)
                    sgf_stop_forced = True
                    sgf_stop_relaxed_forced = bool(sgf_relaxed_stop)
                    sgf_stop_trace["requested"] = True

            early_recovery_first_valid = (
                text_stop_recovery_first_action
                in {
                    int(HabitatSimActions.move_forward),
                    int(HabitatSimActions.turn_left),
                    int(HabitatSimActions.turn_right),
                }
            )
            early_recovery_sequence_valid = bool(
                text_stop_recovery_action_count == 1
                or (
                    text_stop_recovery_action_count == 2
                    and text_stop_recovery_first_action
                    in {
                        int(HabitatSimActions.turn_left),
                        int(HabitatSimActions.turn_right),
                    }
                )
            )
            early_recovery_identity_valid = bool(
                text_stop_approach_commit_probe_owned
                and text_stop_approach_commit_node_id >= 0
            )
            early_recovery_state_trace = {
                "node_id": int(text_stop_approach_commit_node_id),
                "probe_owned": bool(text_stop_approach_commit_probe_owned),
                "source": str(text_stop_approach_commit_source),
                "action_count": int(text_stop_recovery_action_count),
                "first_action": (
                    None
                    if text_stop_recovery_first_action is None
                    else int(text_stop_recovery_first_action)
                ),
            }
            early_recovery_full_cleanup_reason = ""
            early_recovery_counters_only_reason = ""
            if text_stop_approach_commit_source == "recovery" and not (
                early_recovery_identity_valid
                and early_recovery_first_valid
                and early_recovery_sequence_valid
            ):
                early_recovery_full_cleanup_reason = (
                    "invalid_recovery_identity_or_sequence"
                )
            elif (
                text_stop_approach_commit_source == ""
                and early_recovery_identity_valid
            ):
                early_recovery_full_cleanup_reason = (
                    "orphan_probe_commitment_without_source"
                )
            elif (
                text_stop_approach_commit_source in {"approval", "ledger", ""}
                and (
                    text_stop_recovery_action_count != 0
                    or text_stop_recovery_first_action is not None
                )
            ):
                early_recovery_counters_only_reason = (
                    "stray_recovery_state_under_nonrecovery_source"
                )
            if early_recovery_full_cleanup_reason:
                text_stop_approach_commit_node_id = -1
                text_stop_approach_commit_probe_owned = False
                text_stop_approach_commit_step = -(10**9)
                text_stop_approach_commit_source = ""
                text_stop_contact_press_steps = 0
                text_stop_recovery_action_count = 0
                text_stop_recovery_first_action = None
                recovery_approval_snapshot = None
                stop_validation_trace["recovery_transaction_early_cleanup"] = {
                    **early_recovery_state_trace,
                    "cleanup": "full_latch",
                    "reason": str(early_recovery_full_cleanup_reason),
                }
            elif early_recovery_counters_only_reason:
                text_stop_recovery_action_count = 0
                text_stop_recovery_first_action = None
                recovery_approval_snapshot = None
                stop_validation_trace["recovery_transaction_early_cleanup"] = {
                    **early_recovery_state_trace,
                    "cleanup": "recovery_counters_only",
                    "reason": str(early_recovery_counters_only_reason),
                    "commitment_preserved": True,
                }

            # Commitment pursuit: request STOP every step while the window is open
            # (RECOVER is instead validated passively after the depth guard)
            approach_commit_window_steps = 2 * max(
                1, int(args.text_validated_stop_approach_max_forces)
            )
            if (
                text_stop_approach_commit_node_id >= 0
                and int(step_index) - int(text_stop_approach_commit_step)
                > approach_commit_window_steps
            ):
                stop_validation_trace["approach_commitment_expired"] = {
                    "node_id": int(text_stop_approach_commit_node_id),
                    "probe_owned": bool(
                        text_stop_approach_commit_probe_owned
                    ),
                    "source": str(text_stop_approach_commit_source),
                    "commit_step": int(text_stop_approach_commit_step),
                    "window_steps": int(approach_commit_window_steps),
                }
                if (
                    text_stop_approach_commit_source == "recovery"
                    or text_stop_recovery_action_count > 0
                    or text_stop_recovery_first_action is not None
                ):
                    stop_validation_trace[
                        "approach_commitment_expired"
                    ].update(
                        {
                            "recovery_action_count": int(
                                text_stop_recovery_action_count
                            ),
                            "recovery_first_action": (
                                None
                                if text_stop_recovery_first_action is None
                                else int(text_stop_recovery_first_action)
                            ),
                        }
                    )
                text_stop_approach_commit_node_id = -1
                text_stop_approach_commit_probe_owned = False
                text_stop_approach_commit_step = -(10**9)
                text_stop_approach_commit_source = ""
                text_stop_contact_press_steps = 0
                text_stop_recovery_action_count = 0
                text_stop_recovery_first_action = None
                recovery_approval_snapshot = None
            if (
                args.task_mode == "text"
                and bool(args.stop_validation)
                and bool(args.sgf_stop)
                and text_sgf_enabled
                and not bool(forced_budget_stop)
                and habitat_action != int(HabitatSimActions.stop)
                and not fallback_return_owns_late_action
                and text_stop_approach_commit_node_id >= 0
                and text_stop_approach_commit_source != "recovery"
                and (
                    int(step_index) - int(text_stop_approach_commit_step)
                    <= approach_commit_window_steps
                )
            ):
                navigation_action = int(HabitatSimActions.stop)
                habitat_action = int(HabitatSimActions.stop)
                sgf_stop_forced = True
                sgf_stop_trace["requested"] = True
                sgf_stop_trace["commitment_pursuit"] = True

            # Detection probe when no selected target is within the far band
            detection_probe_attempt = False
            detection_probe_node_id = -1
            detection_probe_node_xz: tuple[float, float] | None = None
            detection_probe_prev_navigation_action: int | None = None
            detection_probe_prev_habitat_action: int | None = None
            if (
                args.task_mode == "text"
                and bool(args.text_detection_probe)
                and bool(args.stop_validation)
                and bool(args.sgf_stop)
                and text_sgf_enabled
                and stop_validator is not None
                and depth is not None
                and not bool(forced_budget_stop)
                and habitat_action != int(HabitatSimActions.stop)
                and not endgame_return_act
                and text_stop_approach_commit_node_id < 0
                and int(step_index) - int(detection_probe_last_check_step)
                >= int(detection_probe_check_interval_steps)
                and int(step_index) - int(detection_probe_last_step)
                >= int(requery_interval_steps)
            ):
                detection_probe_last_check_step = int(step_index)
                probe_node_far = True
                if text_target_goal_position is not None:
                    probe_node_distance = planar_distance_xz(
                        np.asarray(
                            env.get_agent_state().position, dtype=np.float32
                        ),
                        text_target_goal_position,
                    )
                    probe_node_far = not (
                        math.isfinite(probe_node_distance)
                        and probe_node_distance
                        <= float(scales.far_band_radius_m)
                    )
                if probe_node_far:
                    try:
                        with timing.measure("stop_verification", "text_detection_probe"):
                            probe_detection = stop_validator._top_detection(
                                bgr_from_rgb_observation(rgb_observation)
                            )
                    except Exception:
                        probe_detection = None
                    probe_confidence = float(
                        (probe_detection or {}).get("confidence", 0.0) or 0.0
                    )
                    if probe_confidence >= float(
                        args.text_detection_probe_min_confidence
                    ):
                        probe_result = StopValidationResult(
                            approved=False,
                            validator_mode="detection_probe",
                            reason="detection probe",
                            detection_found=True,
                            detection_confidence=probe_confidence,
                            detection_label=str(
                                probe_detection.get("label", "")
                            ),
                            detection_bbox_xyxy=probe_detection.get(
                                "bbox_xyxy"
                            ),
                        )
                        (
                            probe_bbox,
                            probe_area,
                            probe_center,
                        ) = StopValidator._detection_geometry(
                            bgr_from_rgb_observation(rgb_observation),
                            probe_detection,
                        )
                        probe_result.detection_bbox_xyxy = probe_bbox
                        probe_result.detection_bbox_area_fraction = float(
                            probe_area
                        )
                        probe_result.detection_bbox_center = probe_center
                        probe_agent_state = env.get_agent_state()
                        probe_standpoint, _ = (
                            detection_projected_standpoint(
                                probe_agent_state,
                                probe_result,
                                depth,
                                hfov_deg=float(args.hfov),
                                min_depth_m=min_depth,
                                max_depth_m=max_depth,
                                standoff_m=float(scales.terminal_approach_distance_m),
                            )
                        )
                        if probe_standpoint is not None:
                            probe_standpoint_distance = float(
                                planar_distance_xz(
                                    np.asarray(
                                        probe_agent_state.position,
                                        dtype=np.float32,
                                    ),
                                    probe_standpoint,
                                )
                            )
                            if probe_standpoint_distance <= float(
                                scales.far_band_radius_m
                            ):
                                detection_probe_attempt = True
                                text_stop_recovery_action_count = 0
                                text_stop_recovery_first_action = None
                                recovery_approval_snapshot = None
                                recovery_approval_snapshot_candidate = None
                                recovery_approval_proposed_action = None
                                detection_probe_counter += 1
                                detection_probe_node_id = (
                                    DETECTION_PROBE_NODE_ID_BASE
                                    + detection_probe_counter
                                )
                                detection_probe_node_xz = (
                                    float(probe_standpoint[0]),
                                    float(probe_standpoint[2]),
                                )
                                detection_probe_last_step = int(step_index)
                                # Restored if the probe is rejected
                                detection_probe_prev_navigation_action = int(navigation_action)
                                detection_probe_prev_habitat_action = int(
                                    habitat_action
                                )
                                navigation_action = int(HabitatSimActions.stop)
                                habitat_action = int(HabitatSimActions.stop)
                                sgf_stop_forced = True
                                sgf_stop_trace["requested"] = True
                                sgf_stop_trace["detection_probe"] = {
                                    "confidence": round(probe_confidence, 4),
                                    "label": str(
                                        probe_detection.get("label", "")
                                    ),
                                    "standpoint_distance_m": round(
                                        probe_standpoint_distance, 4
                                    ),
                                }

            if (
                bool(args.sgf_depth_guard)
                and (
                    sgf_explore_forced
                    or text_target_navigation_forced
                )
                and habitat_action == int(HabitatSimActions.move_forward)
                and depth is not None
            ):
                planner_heading = None
                planner_trace = sgf_explore_trace.get("planner")
                if isinstance(planner_trace, dict):
                    planner_heading = planner_trace.get("heading_error_deg")
                # Base geometry and stereo min-range checks are robot-only
                navigation_depth_guard = depth_forward_navigation_guard
                real_depth_guard_kwargs = {}
                if physical_real_env:
                    from sgf_habitat.real.depth_guard import (
                        depth_forward_navigation_guard as navigation_depth_guard,
                    )

                    real_depth_guard_kwargs = {
                        "min_depth_m": min_depth,
                        "base_from_camera_cv": env.base_from_camera_cv,
                    }
                navigation_guard_trace = navigation_depth_guard(
                    depth,
                    fx=frame_fx,
                    fy=frame_fy,
                    cx=frame_cx,
                    cy=frame_cy,
                    forward_step_size_m=env.forward_step_size_m,
                    agent_radius_m=env.agent_radius_m,
                    camera_height_m=camera_height,
                    max_depth_m=max_depth,
                    planner_heading_error_deg=planner_heading,
                    forward_clearance_margin_m=float(env.agent_radius_m),
                    **real_depth_guard_kwargs,
                )
                if bool(navigation_guard_trace.get("blocked", False)):
                    if (
                        sgf_explore_forced
                        and sgf_navigator is not None
                        and sgf_navigation_map is not None
                    ):
                        sgf_navigator.record_blocked_forward(
                            agent_state=env.get_agent_state(),
                            navigation_map=sgf_navigation_map,
                        )
                    cleared_weak_visual_goal = False
                    cleared_weak_visual_goal_m = None
                    if (
                        visual_pose_goal_position is not None
                        and bool(visual_pose_goal_is_weak)
                    ):
                        cleared_weak_visual_goal = True
                        cleared_weak_visual_goal_m = world_position_list(
                            visual_pose_goal_position
                        )
                        visual_pose_goal_position = None
                        visual_pose_goal_is_weak = False
                        visual_pose_goal_metadata = {}
                        visual_pose_navigation_trace.update(
                            {
                                "active": False,
                                "invalidated_by_depth_guard": True,
                                "invalidated_goal_m": cleared_weak_visual_goal_m,
                            }
                        )

                    retired_sgf_goal = False
                    retired_sgf_goal_m = None
                    if (
                        bool(sgf_explore_forced)
                        and sgf_explorer is not None
                        and sgf_plan_goal is not None
                    ):
                        sgf_plan_goal_array = np.asarray(
                            sgf_plan_goal,
                            dtype=np.float32,
                        ).reshape(-1)
                        if (
                            sgf_plan_goal_array.size == 3
                            and np.all(np.isfinite(sgf_plan_goal_array))
                        ):
                            current_depth_guard_token = (
                                sgf_explorer.goal_epoch_service_token
                                if isinstance(
                                    sgf_explorer,
                                    VisualMemoryUtilityExplorer,
                                )
                                else None
                            )
                            if same_planar_goal(
                                sgf_depth_guard_block_goal,
                                sgf_plan_goal_array,
                            ) and (
                                sgf_depth_guard_block_token
                                == current_depth_guard_token
                            ):
                                sgf_depth_guard_block_count += 1
                            else:
                                sgf_depth_guard_block_goal = sgf_plan_goal_array.copy()
                                sgf_depth_guard_block_token = (
                                    current_depth_guard_token
                                )
                                sgf_depth_guard_block_count = 1
                            retired_sgf_goal_m = world_position_list(sgf_plan_goal_array)
                            if (
                                sgf_depth_guard_block_count
                                >= SGF_DEPTH_GUARD_RETIRE_BLOCK_COUNT
                            ):
                                retirement_event = sgf_explorer.mark_goal_failed(
                                    sgf_plan_goal_array,
                                    reason=GoalEndReason.DEPTH_BLOCKED,
                                )
                                retired_sgf_goal = retirement_event is not None
                                sgf_depth_guard_block_goal = None
                                sgf_depth_guard_block_token = None
                                sgf_depth_guard_block_count = 0
                                if retired_sgf_goal:
                                    sgf_explore_trace["goal_retired"] = True
                                    sgf_explore_trace[
                                        "depth_guard_retired_goal_m"
                                    ] = retired_sgf_goal_m
                                sgf_explore_trace["failed_goal_count"] = int(
                                    sgf_explorer.failed_goal_count
                                )
                    sgf_explore_trace["depth_guard_block_count"] = int(
                        sgf_depth_guard_block_count
                    )
                    sgf_explore_trace["depth_guard_block_token"] = (
                        sgf_depth_guard_block_token
                    )
                    sgf_explore_trace["depth_guard_retire_threshold"] = int(
                        SGF_DEPTH_GUARD_RETIRE_BLOCK_COUNT
                    )
                    navigation_guard_trace.update(
                        {
                            "cleared_weak_visual_pose_goal": bool(cleared_weak_visual_goal),
                            "cleared_weak_visual_pose_goal_m": cleared_weak_visual_goal_m,
                            "sgf_goal_block_count": int(sgf_depth_guard_block_count),
                            "sgf_goal_retire_threshold": int(SGF_DEPTH_GUARD_RETIRE_BLOCK_COUNT),
                            "retired_sgf_goal": bool(retired_sgf_goal),
                            "retired_sgf_goal_m": retired_sgf_goal_m if retired_sgf_goal else None,
                        }
                    )
                    habitat_action = (
                        int(HabitatSimActions.turn_left)
                        if navigation_guard_trace.get("turn_direction") == "left"
                        else int(HabitatSimActions.turn_right)
                    )
                    navigation_action = int(habitat_action)
                    navigation_guard_trace["vetoed_action"] = int(
                        HabitatSimActions.move_forward
                    )
                    navigation_guard_trace["replacement_action"] = int(
                        habitat_action
                    )
                    sgf_depth_guard_forced = True

            # RECOVER follow-up: one passive re-observation on the next frame
            recovery_followup_transaction_live = bool(
                args.task_mode == "text"
                and bool(args.stop_validation)
                and bool(args.sgf_stop)
                and text_sgf_enabled
                and bool(args.text_validated_stop_approach)
                and not bool(forced_budget_stop)
                and not fallback_return_owns_late_action
                and text_stop_approach_commit_source == "recovery"
                and text_stop_approach_commit_probe_owned
                and text_stop_approach_commit_node_id >= 0
                and text_stop_recovery_first_action
                in {
                    int(HabitatSimActions.move_forward),
                    int(HabitatSimActions.turn_left),
                    int(HabitatSimActions.turn_right),
                }
                and (
                    text_stop_recovery_action_count == 1
                    or (
                        text_stop_recovery_action_count == 2
                        and text_stop_recovery_first_action
                        in {
                            int(HabitatSimActions.turn_left),
                            int(HabitatSimActions.turn_right),
                        }
                    )
                )
                and (
                    int(step_index) - int(text_stop_approach_commit_step)
                    <= approach_commit_window_steps
                )
            )
            recovery_followup_conflicting_stop_replaced = bool(
                recovery_followup_transaction_live
                and habitat_action == int(HabitatSimActions.stop)
                and not bool(sgf_stop_forced)
            )
            if recovery_followup_conflicting_stop_replaced:
                recovery_conflict_replacement = (
                    safe_non_stop_navigation_action()
                )
                navigation_action = int(recovery_conflict_replacement)
                habitat_action = int(recovery_conflict_replacement)
                stop_validation_trace[
                    "recovery_followup_conflicting_stop"
                ] = {
                    "replaced": True,
                    "replacement_action": int(recovery_conflict_replacement),
                    "reason": "reconfirmation_required",
                }
            recovery_followup_observation_probe = bool(
                recovery_followup_transaction_live
                and habitat_action != int(HabitatSimActions.stop)
            )
            recovery_followup_prev_navigation_action: int | None = None
            recovery_followup_prev_habitat_action: int | None = None
            if recovery_followup_observation_probe:
                recovery_followup_prev_navigation_action = int(
                    navigation_action
                )
                recovery_followup_prev_habitat_action = int(habitat_action)
                stop_validation_trace["recovery_followup_observation"] = {
                    "passive": True,
                    "planned_navigation_action": int(navigation_action),
                    "planned_habitat_action": int(habitat_action),
                    "post_depth_guard": True,
                }

            stop_validation_result: StopValidationResult | None = None
            stop_policy_decision = None
            stop_features = None
            validated_stop_approach_action: int | None = None
            text_stop_deferred_to_target_plan = False
            validate_stop_request = (
                bool(args.stop_validation)
                and args.task_mode == "text"
                and not bool(forced_budget_stop)
                and (
                    (
                        habitat_action == int(HabitatSimActions.stop)
                        and bool(sgf_stop_forced)
                    )
                    or recovery_followup_observation_probe
                )
            )
            if validate_stop_request:
                stop_validation_attempts += 1
                stop_source = "sgf_stop"
                stop_validation_trace["attempted"] = True
                stop_validation_trace["source"] = stop_source
                stop_validation_trace["relaxed_sgf_stop"] = bool(sgf_stop_relaxed_forced)
                if stop_validator is None:
                    stop_validation_result = StopValidationResult(
                        approved=False,
                        validator_mode="unavailable_retry",
                        reason="stop validation unavailable: validator is not configured",
                        detection_found=False,
                        p_yes=0.05,
                        scene_p_yes=0.0,
                    )
                else:
                    validation_probe_owned = bool(
                        detection_probe_attempt
                        or text_stop_approach_commit_probe_owned
                    )
                    validation_node_footprint = None
                    if (
                        not validation_probe_owned
                        and text_selected_target_node is not None
                        and world_from_camera_cv is not None
                    ):
                        try:
                            validation_node_extent = project_hypothesis(
                                bbox_min_m=text_selected_target_node.get("bbox_min_m"),
                                bbox_max_m=text_selected_target_node.get("bbox_max_m"),
                                world_from_camera_cv=world_from_camera_cv,
                                fx=float(frame_fx),
                                fy=float(frame_fy),
                                cx=float(frame_cx),
                                cy=float(frame_cy),
                                width=int(width),
                                height=int(height),
                            )
                        except Exception:
                            validation_node_extent = None
                        if validation_node_extent is not None:
                            validation_node_footprint = tuple(
                                int(value) for value in validation_node_extent.bbox_xyxy
                            )
                    validation_repeats_vantage = False
                    if (
                        not validation_probe_owned
                        and text_selected_target_node is not None
                        and validation_node_footprint is not None
                        and text_stop_ledger is not None
                        and text_target_goal_position is not None
                        and text_stop_approach_commit_node_id < 0
                    ):
                        try:
                            repeat_state = env.get_agent_state()
                            repeat_arguments = (
                                int(text_selected_target_node.get("id", -1)),
                                (
                                    float(text_target_goal_position[0]),
                                    float(text_target_goal_position[2]),
                                ),
                                (
                                    float(repeat_state.position[0]),
                                    float(repeat_state.position[2]),
                                ),
                                float(agent_yaw_deg(repeat_state)),
                            )
                            if text_reobservation_mode == "same_pose":
                                last_verdict_step = (
                                    text_stop_ledger.last_observation_step(
                                        repeat_arguments[0],
                                        repeat_arguments[1],
                                    )
                                )
                                validation_repeats_vantage = bool(
                                    last_verdict_step is not None
                                    and int(step_index) - int(last_verdict_step)
                                    < int(requery_interval_steps)
                                )
                                if (
                                    not validation_repeats_vantage
                                    and text_stop_ledger.repeats_vantage(
                                        *repeat_arguments, pooled=True
                                    )
                                ):
                                    text_same_pose_requeries += 1
                            else:
                                validation_repeats_vantage = bool(
                                    text_stop_ledger.repeats_vantage(
                                        *repeat_arguments
                                    )
                                )
                        except Exception:
                            validation_repeats_vantage = False
                    if (
                        not validation_probe_owned
                        and text_selected_target_node is not None
                        and validation_node_footprint is None
                    ):
                        stop_validation_result = StopValidationResult(
                            approved=False,
                            validator_mode="hypothesis_unobservable",
                            reason="The pursued node projects outside the frame; the verifier was not asked.",
                            detection_found=False,
                            p_yes=0.5,
                            scene_p_yes=0.0,
                            evidence_source="none",
                        )
                    elif (
                        validation_repeats_vantage
                        and text_reobservation_mode == "same_pose"
                    ):
                        stop_validation_result = StopValidationResult(
                            approved=False,
                            validator_mode="same_pose_wait",
                            reason="The re-query interval since the last verdict on this hypothesis has not elapsed; the verifier was not re-asked.",
                            detection_found=False,
                            p_yes=0.5,
                            scene_p_yes=0.0,
                            evidence_source="none",
                        )
                    elif validation_repeats_vantage:
                        text_same_vantage_skips += 1
                        stop_validation_result = StopValidationResult(
                            approved=False,
                            validator_mode="same_vantage_repeat",
                            reason="An evidence-bearing verdict from this vantage is already on the ledger; the verifier was not re-asked.",
                            detection_found=False,
                            p_yes=0.5,
                            scene_p_yes=0.0,
                            evidence_source="none",
                        )
                    else:
                        try:
                            if text_verification_object == "top_detection":
                                text_top_detection_queries += 1
                            elif text_verification_object == "whole_frame":
                                text_whole_frame_queries += 1
                            with timing.measure("stop_verification", "text_stop_validation"):
                                stop_validation_result = stop_validator.validate(
                                    text_goal=episode_info.text_goal,
                                    canonical_goal=canonical_text_goal,
                                    rgb_bgr=bgr_from_rgb_observation(rgb_observation),
                                    hypothesis=text_hypothesis_projection,
                                    footprint_bbox_xyxy=validation_node_footprint,
                                    verification_object=text_verification_object,
                                )
                        except Exception as exc:
                            stop_validation_result = StopValidationResult(
                                approved=False,
                                validator_mode="unavailable_retry",
                                reason=f"stop validation unavailable: {exc}",
                                detection_found=False,
                                p_yes=0.05,
                                scene_p_yes=0.0,
                            )
                evidence_source = str(
                    getattr(stop_validation_result, "evidence_source", "")
                )
                if evidence_source == "detector_grounded":
                    text_grounded_evidence_query_count += 1
                elif evidence_source == "hypothesis_projection":
                    text_projection_evidence_query_count += 1
                stop_validation_trace["hypothesis"] = dict(text_hypothesis_trace)
                last_stop_validation = summarize_stop_validation(stop_validation_result) or {}
                validation_unobservable = str(
                    getattr(stop_validation_result, "validator_mode", "")
                ) in ("hypothesis_unobservable", "same_vantage_repeat", "same_pose_wait")
                validation_unavailable_retry = (
                    str(getattr(stop_validation_result, "validator_mode", ""))
                    == "unavailable_retry"
                )
                recovery_entry_action_count = int(
                    text_stop_recovery_action_count
                )
                recovery_entry_first_action = (
                    text_stop_recovery_first_action
                )
                probe_owned_stop_transaction = bool(
                    detection_probe_attempt
                    or text_stop_approach_commit_probe_owned
                )
                probe_recovery_isolation_active = bool(
                    args.text_validated_stop_approach
                    and probe_owned_stop_transaction
                )
                probe_recovery_policy_available = bool(
                    probe_recovery_isolation_active
                    and text_stop_approach_commit_source in {"", "recovery"}
                )
                recovery_first_action_valid = (
                    text_stop_recovery_first_action
                    in {
                        int(HabitatSimActions.move_forward),
                        int(HabitatSimActions.turn_left),
                        int(HabitatSimActions.turn_right),
                    }
                )
                recovery_sequence_state_valid = bool(
                    text_stop_recovery_action_count == 1
                    or (
                        text_stop_recovery_action_count == 2
                        and text_stop_recovery_first_action
                        in {
                            int(HabitatSimActions.turn_left),
                            int(HabitatSimActions.turn_right),
                        }
                    )
                )
                probe_recovery_commit_identity_valid = bool(
                    text_stop_approach_commit_probe_owned
                    and text_stop_approach_commit_node_id >= 0
                )
                probe_recovery_maneuver_active = bool(
                    args.text_validated_stop_approach
                    and probe_recovery_commit_identity_valid
                    and text_stop_approach_commit_source == "recovery"
                    and recovery_first_action_valid
                    and recovery_sequence_state_valid
                )
                probe_recovery_state_present = bool(
                    text_stop_approach_commit_source == "recovery"
                    or text_stop_recovery_action_count != 0
                    or text_stop_recovery_first_action is not None
                )
                probe_recovery_state_corrupt = bool(
                    args.text_validated_stop_approach
                    and (
                        probe_recovery_state_present
                        or (
                            probe_recovery_commit_identity_valid
                            and text_stop_approach_commit_source == ""
                        )
                    )
                    and not probe_recovery_maneuver_active
                )
                # Anchor the evidence at the detection-projected standpoint, not the
                # agent
                evidence_bank_agent_state = env.get_agent_state()
                evidence_standpoint_position, endgame_bank_trace = (
                    detection_projected_standpoint(
                        evidence_bank_agent_state,
                        stop_validation_result,
                        depth,
                        hfov_deg=float(args.hfov),
                        min_depth_m=min_depth,
                        max_depth_m=max_depth,
                        standoff_m=float(scales.terminal_approach_distance_m),
                    )
                )
                evidence_standpoint_distance_m = float("inf")
                if evidence_standpoint_position is not None:
                    evidence_standpoint_distance_m = float(
                        planar_distance_xz(
                            np.asarray(
                                evidence_bank_agent_state.position,
                                dtype=np.float32,
                            ),
                            evidence_standpoint_position,
                        )
                    )
                    endgame_bank_trace["standpoint_distance_m"] = round(
                        float(evidence_standpoint_distance_m), 4
                    )
                stop_validation_trace["endgame_bank"] = endgame_bank_trace
                current_verdict_evidence_weight = ledger_evidence_weight(
                    stop_validation_result,
                    node=text_selected_target_node,
                    probe_owned=bool(
                        detection_probe_attempt
                        or text_stop_approach_commit_probe_owned
                    ),
                    world_from_camera_cv=world_from_camera_cv,
                    fx=frame_fx,
                    fy=frame_fy,
                    cx=frame_cx,
                    cy=frame_cy,
                    width=int(width),
                    height=int(height),
                    evidence_standpoint_m=evidence_standpoint_position,
                    consistency_radius_m=float(scales.hypothesis_exclusion_radius_m),
                    verification_object=text_verification_object,
                )
                stop_validation_trace["evidence_weight"] = round(
                    float(current_verdict_evidence_weight), 4
                )
                if endgame_return_planner is not None:
                    # Bank approvals as return candidates (boxless ones as unanchored)
                    bank_key: int | None = None
                    if bool(getattr(stop_validation_result, "approved", False)):
                        bank_p_yes = float(
                            getattr(stop_validation_result, "p_yes", 0.0) or 0.0
                        )
                        if bool(
                            getattr(
                                stop_validation_result, "detection_found", False
                            )
                        ):
                            bank_pose = evidence_standpoint_position
                            if bank_pose is None:
                                bank_pose = np.asarray(
                                    evidence_bank_agent_state.position,
                                    dtype=np.float32,
                                )
                            # Corroborated nodes outrank any single approval
                            bank_prior_approvals = 0
                            if text_stop_ledger is not None:
                                if (
                                    detection_probe_attempt
                                    and detection_probe_node_xz is not None
                                ):
                                    bank_key = (
                                        text_stop_ledger.resolve_node_key(
                                            int(detection_probe_node_id),
                                            detection_probe_node_xz,
                                        )
                                    )
                                elif text_stop_approach_commit_probe_owned:
                                    bank_key = text_stop_ledger.resolve_node_key(
                                        int(text_stop_approach_commit_node_id),
                                        text_stop_ledger.node_position(
                                            int(
                                                text_stop_approach_commit_node_id
                                            )
                                        ),
                                    )
                                elif text_selected_target_node is not None:
                                    try:
                                        bank_node_id = int(
                                            text_selected_target_node.get(
                                                "id", -1
                                            )
                                        )
                                    except (TypeError, ValueError):
                                        bank_node_id = -1
                                    if bank_node_id >= 0:
                                        bank_node_xz = None
                                        if (
                                            text_target_goal_position
                                            is not None
                                        ):
                                            bank_node_xz = (
                                                float(
                                                    text_target_goal_position[
                                                        0
                                                    ]
                                                ),
                                                float(
                                                    text_target_goal_position[
                                                        2
                                                    ]
                                                ),
                                            )
                                        bank_key = (
                                            text_stop_ledger.resolve_node_key(
                                                bank_node_id, bank_node_xz
                                            )
                                        )
                                if bank_key is not None:
                                    bank_prior_approvals = text_stop_ledger.distinct_detection_approval_count(
                                        bank_key
                                    )
                            bank_probe_owned = bool(
                                detection_probe_attempt
                                or text_stop_approach_commit_probe_owned
                            )
                            endgame_return_planner.note_score(
                                bank_p_yes
                                + (1.0 if bank_prior_approvals >= 1 else 0.0),
                                bank_pose,
                                int(step_index),
                                anchored=True,
                                node_key=(
                                    -1 if bank_key is None else int(bank_key)
                                ),
                                raw_validator_confidence=bank_p_yes,
                                probe_owned=bank_probe_owned,
                            )
                        else:
                            endgame_return_planner.note_score(
                                bank_p_yes,
                                np.asarray(
                                    evidence_bank_agent_state.position,
                                    dtype=np.float32,
                                ),
                                int(step_index),
                                anchored=False,
                            )
                    endgame_fresh_radius = float(scales.far_band_radius_m)
                    endgame_attempt_basis = float(evidence_standpoint_distance_m)
                    if (
                        not math.isfinite(endgame_attempt_basis)
                        and not probe_recovery_isolation_active
                        and text_selected_target_node is not None
                    ):
                        try:
                            endgame_attempt_basis = float(
                                text_selected_target_node.get(
                                    "distance_m", float("inf")
                                )
                            )
                        except (TypeError, ValueError):
                            endgame_attempt_basis = float("inf")
                    endgame_attempt_near = (
                        math.isfinite(endgame_attempt_basis)
                        and endgame_attempt_basis <= endgame_fresh_radius
                    )
                    endgame_near_pursuit_live = False
                    if endgame_attempt_near:
                        if bool(
                            getattr(stop_validation_result, "approved", False)
                        ):
                            endgame_near_pursuit_live = True
                        elif (
                            text_stop_ledger is not None
                            and not probe_recovery_isolation_active
                            and text_selected_target_node is not None
                        ):
                            try:
                                fresh_node_id = int(
                                    text_selected_target_node.get("id", -1)
                                )
                            except (TypeError, ValueError):
                                fresh_node_id = -1
                            if fresh_node_id >= 0:
                                fresh_node_xz = None
                                if text_target_goal_position is not None:
                                    fresh_node_xz = (
                                        float(text_target_goal_position[0]),
                                        float(text_target_goal_position[2]),
                                    )
                                fresh_key = text_stop_ledger.resolve_node_key(
                                    fresh_node_id, fresh_node_xz
                                )
                                endgame_near_pursuit_live = (
                                    text_stop_ledger.distinct_pose_count(
                                        fresh_key, TargetStopLedger.APPROVAL
                                    )
                                    >= 1
                                )
                    if (
                        bool(getattr(stop_validation_result, "approved", False))
                        and bool(
                            getattr(
                                stop_validation_result, "detection_found", False
                            )
                        )
                    ) or endgame_near_pursuit_live:
                        endgame_return_planner.note_fresh_evidence(
                            int(step_index)
                        )
                raw_validation_approved = bool(
                    getattr(stop_validation_result, "approved", False)
                )
                validation_approved = bool(raw_validation_approved)
                # Preview pooled evidence before recording the current verdict;
                # ordinary text stop proposals must pass the configured rule.
                closure_evidence_ok = True
                closure_evidence_log_odds = None
                closure_evidence_gate_active = False
                closure_gate_kwargs: dict[str, Any] | None = None
                if (
                    text_stop_ledger is not None
                    and stop_validation_result is not None
                ):
                    closure_key_id = -1
                    closure_key_position = None
                    if detection_probe_attempt and detection_probe_node_xz is not None:
                        closure_key_id = int(detection_probe_node_id)
                        closure_key_position = detection_probe_node_xz
                    elif text_stop_approach_commit_probe_owned:
                        closure_key_id = int(text_stop_approach_commit_node_id)
                        closure_key_position = text_stop_ledger.node_position(
                            closure_key_id
                        )
                    elif text_selected_target_node is not None:
                        try:
                            closure_key_id = int(
                                text_selected_target_node.get("id", -1)
                            )
                        except (TypeError, ValueError):
                            closure_key_id = -1
                        if text_target_goal_position is not None:
                            closure_key_position = (
                                float(text_target_goal_position[0]),
                                float(text_target_goal_position[2]),
                            )
                    gate_state = env.get_agent_state()
                    gate_yaw = agent_yaw_deg(gate_state)
                    closure_gate_kwargs = dict(
                        approved=bool(raw_validation_approved),
                        p_yes=float(
                            getattr(stop_validation_result, "p_yes", 0.0) or 0.0
                        ),
                        evidence_weight=float(current_verdict_evidence_weight),
                        agent_position_xz=(
                            float(gate_state.position[0]),
                            float(gate_state.position[2]),
                        ),
                        agent_yaw_deg=float(
                            gate_yaw if gate_yaw is not None else 0.0
                        ),
                        step=int(step_index),
                        detection_found=bool(
                            getattr(stop_validation_result, "detection_found", False)
                        ),
                    )
                    closure_evidence_ok, closure_gate_details = (
                        text_stop_ledger.evidence_sufficient(
                            closure_key_id,
                            closure_key_position,
                            **closure_gate_kwargs,
                        )
                    )
                    closure_evidence_gate_active = True
                    sequential_gate_ok: bool | None = None
                    if math.isfinite(text_stop_ledger.sequential_log_odds_bound):
                        sequential_gate_ok, sequential_gate_details = (
                            text_stop_ledger.evidence_sufficient(
                                closure_key_id,
                                closure_key_position,
                                rule="sequential",
                                **closure_gate_kwargs,
                            )
                        )
                        closure_evidence_log_odds = float(
                            sequential_gate_details["evidence_log_odds"]
                        )
                    if (
                        text_evidence_rule != "sequential"
                        and raw_validation_approved
                        and sequential_gate_ok is not None
                        and bool(sequential_gate_ok) != bool(closure_evidence_ok)
                    ):
                        text_evidence_rule_overrides += 1
                    stop_validation_trace["sequential_evidence_gate"] = {
                        "node_key": int(closure_key_id),
                        "rule": str(text_evidence_rule),
                        "evidence_log_odds": (
                            None
                            if closure_evidence_log_odds is None
                            else round(float(closure_evidence_log_odds), 4)
                        ),
                        "commit_log_odds": round(
                            float(text_stop_ledger.sequential_log_odds_bound), 4
                        ),
                        "rule_details": dict(closure_gate_details),
                        "satisfied": bool(closure_evidence_ok),
                    }
                probe_recovery_requested = False
                probe_recovery_reconfirmed_stop = False
                probe_recovery_continuation_executed = False
                recovery_followup_requires_resolution = False
                recovery_followup_nonrecover_frame = False
                recovery_followup_ledger_record_only = False
                recovery_transaction_cancel_requested = False
                recovery_cancel_reason = ""
                recovery_commitment_rejected = bool(
                    (
                        probe_recovery_maneuver_active
                        or probe_recovery_state_corrupt
                    )
                    and not raw_validation_approved
                    and not validation_unavailable_retry
                )
                policy_target_node = text_selected_target_node
                policy_planner_command = text_target_plan_command
                policy_tracked_steps = int(tracked_text_target_steps)
                if probe_owned_stop_transaction:
                    policy_target_node = None
                    policy_planner_command = None
                    policy_tracked_steps = 0
                if (
                    text_selected_target_node is not None
                    or detection_probe_attempt
                    or text_stop_approach_commit_probe_owned
                ):
                    # Verdicts about the node are measured to the node itself
                    policy_evidence_distance_m = float(evidence_standpoint_distance_m)
                    if (
                        float(current_verdict_evidence_weight) > 0.0
                        and not probe_owned_stop_transaction
                        and policy_target_node is not None
                    ):
                        policy_evidence_distance_m = float("inf")
                    features = compute_stop_features(
                        stop_validation_result=stop_validation_result,
                        target_node=policy_target_node,
                        planner_command=policy_planner_command,
                        goal_like_label_steps=goal_like_label_steps,
                        tracked_steps=policy_tracked_steps,
                        step_index=step_index,
                        max_steps=args.max_steps,
                        evidence_distance_m=policy_evidence_distance_m,
                    )
                    stop_features = features
                    stop_policy_decision = stop_decision_policy.decide(
                        features,
                        recovery_available=probe_recovery_policy_available,
                    )
                    last_stop_policy_decision = summarize_stop_decision(stop_policy_decision) or {}
                    try:
                        endgame_attempt_score = float(
                            stop_policy_decision.details.get("p_stop", 0.0)
                        )
                    except (AttributeError, TypeError, ValueError):
                        endgame_attempt_score = 0.0
                    if (
                        endgame_return_planner is not None
                        and not (
                            probe_recovery_isolation_active
                            and validation_unavailable_retry
                        )
                    ):
                        endgame_return_planner.note_observation(
                            endgame_attempt_score,
                            int(step_index),
                        )
                    policy_would_stop = (
                        stop_policy_decision.action == STOP_ACTION
                    )
                    if (
                        policy_would_stop
                        and text_shadow_stop is not None
                        and closure_gate_kwargs is not None
                    ):
                        text_shadow_stop.observe_stop_opportunity(
                            node_id=int(closure_key_id),
                            node_position_xz=closure_key_position,
                            distance_to_goal=getattr(
                                env, "diagnostic_distance_to_goal", lambda: None
                            ),
                            path_length_m=lambda: planar_path_length_m(
                                path_positions
                            ),
                            **closure_gate_kwargs,
                        )
                    if (
                        policy_would_stop
                        and closure_evidence_gate_active
                        and not closure_evidence_ok
                    ):
                        policy_would_stop = False
                        stop_validation_trace["direct_stop_withheld"] = {
                            "rule": str(text_evidence_rule),
                            "evidence_log_odds": (
                                None
                                if closure_evidence_log_odds is None
                                else round(float(closure_evidence_log_odds), 4)
                            ),
                            "bound": round(
                                float(text_stop_ledger.sequential_log_odds_bound),
                                4,
                            ),
                        }
                    probe_recovery_requested = bool(
                        raw_validation_approved
                        and probe_owned_stop_transaction
                        and stop_policy_decision.action == RECOVER_ACTION
                    )
                    validation_approved = bool(policy_would_stop)
                    probe_recovery_reconfirmed_stop = bool(
                        probe_recovery_maneuver_active
                        and raw_validation_approved
                        and validation_approved
                    )
                    current_selected_target_id = None
                    current_selected_target_distance_m = None
                    if text_selected_target_node is not None:
                        current_selected_target_id = (
                            text_selected_target_node.get("id")
                        )
                        current_selected_target_distance_m = (
                            text_selected_target_node.get("distance_m")
                        )
                    current_validation_bbox_area = getattr(
                        stop_validation_result,
                        "detection_bbox_area_fraction",
                        None,
                    )
                    recovery_monotonic_closure_decision = (
                        decide_recovery_monotonic_closure(
                            enabled=bool(
                                metric_visual_closure_enabled
                                and args.text_metric_visual_closure_recovery_monotonic_bbox
                            ),
                            current_step=int(step_index),
                            raw_validation_approved=bool(
                                raw_validation_approved
                            ),
                            policy_action=str(stop_policy_decision.action),
                            recovery_transaction_active=bool(
                                probe_recovery_maneuver_active
                            ),
                            recovery_action_count=int(
                                recovery_entry_action_count
                            ),
                            selected_target_id=current_selected_target_id,
                            bbox_area_fraction=current_validation_bbox_area,
                            previous_snapshot=recovery_approval_snapshot,
                        )
                    )
                    contextual_closeup_closure_decision = (
                        decide_contextual_stop_persistence_closure(
                            enabled=bool(
                                metric_visual_closure_enabled
                                and args.text_metric_visual_closure_contextual_closeup_persistence
                                and stop_source == "sgf_stop"
                            ),
                            current_step=int(step_index),
                            raw_validation_approved=bool(
                                raw_validation_approved
                            ),
                            policy_action=str(stop_policy_decision.action),
                            selected_target_id=current_selected_target_id,
                            bbox_area_fraction=current_validation_bbox_area,
                            target_distance_m=(
                                current_selected_target_distance_m
                            ),
                            scene_confirmed=bool(
                                raw_validation_approved
                                and float(
                                    getattr(stop_validation_result, "scene_p_yes", 0.0)
                                    or 0.0
                                )
                                > 0.0
                            ),
                            previous_snapshot=last_raw_approval_snapshot,
                            close_up_bbox_area_fraction=float(
                                args.text_stop_ledger_close_up_bbox_area_fraction
                            ),
                            grounded_target_distance_max_m=float(
                                metric_visual_closure_config.grounded_target_distance_max_m
                            ),
                        )
                    )
                    if not closure_evidence_ok and (
                        recovery_monotonic_closure_decision.arm
                        or contextual_closeup_closure_decision.arm
                    ):
                        metric_visual_closure_withheld_count += 1
                    if recovery_monotonic_closure_decision.arm and closure_evidence_ok:
                        metric_visual_direct_closure_decision = (
                            recovery_monotonic_closure_decision
                        )
                        metric_visual_direct_closure_kind = (
                            "recovery_monotonic_bbox"
                        )
                        validation_approved = True
                        probe_recovery_reconfirmed_stop = True
                    elif contextual_closeup_closure_decision.arm and closure_evidence_ok:
                        metric_visual_direct_closure_decision = (
                            contextual_closeup_closure_decision
                        )
                        metric_visual_direct_closure_kind = (
                            "contextual_closeup_persistence"
                        )
                    if metric_visual_direct_closure_decision is not None:
                        metric_visual_closure_trace.update(
                            {
                                "triggered": True,
                                "direct": True,
                                "kind": str(
                                    metric_visual_direct_closure_kind
                                ),
                                "reason": str(
                                    metric_visual_direct_closure_decision.reason
                                ),
                                "details": dict(
                                    metric_visual_direct_closure_decision.details
                                ),
                            }
                        )
                        stop_validation_trace[
                            "metric_visual_closure"
                        ] = dict(metric_visual_closure_trace)
                    if raw_validation_approved:
                        last_raw_approval_snapshot = (
                            MetricVisualClosureApprovalSnapshot(
                                step=int(step_index),
                                raw_validation_approved=True,
                                policy_action=str(
                                    stop_policy_decision.action
                                ),
                                selected_target_id=(
                                    current_selected_target_id
                                ),
                                bbox_area_fraction=(
                                    current_validation_bbox_area
                                ),
                                recovery_executed=False,
                            )
                        )
                    else:
                        last_raw_approval_snapshot = None
                    recovery_followup_requires_resolution = bool(
                        (
                            probe_recovery_maneuver_active
                            or probe_recovery_state_corrupt
                        )
                        and not validation_unavailable_retry
                        and not probe_recovery_reconfirmed_stop
                    )
                    recovery_followup_nonrecover_frame = bool(
                        recovery_followup_requires_resolution
                        and not probe_recovery_requested
                    )
                    recovery_followup_ledger_record_only = bool(
                        probe_recovery_maneuver_active
                        and not validation_unavailable_retry
                        and not probe_recovery_reconfirmed_stop
                    )
                    if probe_recovery_isolation_active:
                        stop_validation_trace["probe_recovery"] = {
                            "available": bool(
                                probe_recovery_policy_available
                            ),
                            "requested": bool(probe_recovery_requested),
                            "reconfirmed_stop": bool(
                                probe_recovery_reconfirmed_stop
                            ),
                            "raw_validator_approved": bool(
                                raw_validation_approved
                            ),
                            "policy_action": str(
                                stop_policy_decision.action
                            ),
                            "p_stop": stop_policy_decision.details.get(
                                "p_stop"
                            ),
                            "recovery_radius_m": (
                                stop_policy_decision.details.get(
                                    "recovery_radius_m"
                                )
                            ),
                            "transaction_active": bool(
                                probe_recovery_maneuver_active
                            ),
                            "identity_valid": bool(
                                probe_recovery_commit_identity_valid
                            ),
                            "state_corrupt": bool(
                                probe_recovery_state_corrupt
                            ),
                            "ledger_record_only": bool(
                                recovery_followup_ledger_record_only
                            ),
                            "action_count": int(
                                text_stop_recovery_action_count
                            ),
                            "first_action": (
                                None
                                if text_stop_recovery_first_action is None
                                else int(text_stop_recovery_first_action)
                            ),
                            "max_actions": 2,
                            "owner": "probe",
                        }
                        if recovery_monotonic_closure_decision is not None:
                            stop_validation_trace["probe_recovery"][
                                "monotonic_bbox_closure"
                            ] = {
                                "triggered": bool(
                                    recovery_monotonic_closure_decision.arm
                                ),
                                "reason": str(
                                    recovery_monotonic_closure_decision.reason
                                ),
                                "details": dict(
                                    recovery_monotonic_closure_decision.details
                                ),
                            }
                    if contextual_closeup_closure_decision is not None:
                        stop_validation_trace[
                            "contextual_closeup_persistence"
                        ] = {
                            "triggered": bool(
                                contextual_closeup_closure_decision.arm
                            ),
                            "reason": str(
                                contextual_closeup_closure_decision.reason
                            ),
                            "details": dict(
                                contextual_closeup_closure_decision.details
                            ),
                        }
                else:
                    last_stop_policy_decision = {
                        "action": "stop" if validation_approved else EXPLORE_ACTION,
                        "reason": "current_view_validation",
                    }
                base_text_stop_approach_max_forces = int(
                    args.text_validated_stop_approach_max_forces
                )
                text_stop_approach_max_forces = int(
                    base_text_stop_approach_max_forces
                )
                text_stop_approach_min_goal_distance_m: float | None = None
                if stop_features is not None:
                    approach_distance = stop_features.node_distance_m()
                    if math.isfinite(approach_distance):
                        text_stop_approach_min_goal_distance_m = approach_distance
                text_stop_approach_sgf_close_ok = (
                    text_stop_approach_min_goal_distance_m is None
                    or text_stop_approach_min_goal_distance_m > float(success_radius)
                )
                text_stop_approach_force_budget_ok = (
                    text_stop_approach_max_forces <= 0
                    or int(validated_text_stop_approach_forces)
                    < text_stop_approach_max_forces
                )
                target_plan_available = (
                    text_selected_target_node is not None
                    and text_target_plan_action is not None
                )
                target_plan_stop_distance_m = float(scales.terminal_approach_distance_m)
                target_plan_goal_reached = bool(
                    getattr(text_target_plan_command, "goal_reached", False)
                )
                target_plan_not_converged = (
                    not bool(target_plan_goal_reached)
                    and (
                        text_stop_approach_min_goal_distance_m is None
                        or float(text_stop_approach_min_goal_distance_m)
                        > max(0.0, target_plan_stop_distance_m, float(success_radius))
                    )
                )
                policy_approved_beyond_band = bool(
                    stop_policy_decision is not None
                    and raw_validation_approved
                    and stop_policy_decision.action != STOP_ACTION
                    and str(stop_policy_decision.reason)
                    in {
                        "approved_beyond_success_radius",
                        "approved_evidence_beyond_success_radius",
                    }
                )
                sgf_target_plan_deferral = (
                    policy_approved_beyond_band
                    and metric_visual_direct_closure_decision is None
                    and stop_source == "sgf_stop"
                    and not probe_recovery_isolation_active
                    and target_plan_available
                )
                if sgf_target_plan_deferral:
                    validation_approved = False
                    text_stop_deferred_to_target_plan = True
                    stop_validation_trace["sgf_stop_deferred_to_target_plan"] = {
                        "min_goal_distance_m": (
                            None
                            if text_stop_approach_min_goal_distance_m is None
                            else round(float(text_stop_approach_min_goal_distance_m), 4)
                        ),
                        "target_plan_stop_distance_m": round(
                            float(target_plan_stop_distance_m),
                            4,
                        ),
                        "target_plan_goal_reached": bool(target_plan_goal_reached),
                        "tracked_steps": int(tracked_text_target_steps),
                    }
                    last_stop_policy_decision = {
                        "action": EXPLORE_ACTION,
                        "reason": "sgf_stop_target_plan_not_converged",
                        "details": {
                            "min_goal_distance_m": (
                                None
                                if text_stop_approach_min_goal_distance_m is None
                                else round(float(text_stop_approach_min_goal_distance_m), 4)
                            ),
                            "target_plan_stop_distance_m": round(
                                float(target_plan_stop_distance_m),
                                4,
                            ),
                            "target_plan_goal_reached": bool(target_plan_goal_reached),
                            "validator": summarize_stop_validation(stop_validation_result),
                        },
                    }
                # A withheld approval still presses toward its evidence for a new view
                withheld_approval_press = bool(
                    text_reobservation_mode == "active"
                    and stop_validation_trace.get("direct_stop_withheld")
                    is not None
                )
                if (
                    (
                        validation_approved
                        or probe_recovery_requested
                        or withheld_approval_press
                    )
                    and raw_validation_approved
                    and bool(args.text_validated_stop_approach)
                    and metric_visual_direct_closure_decision is None
                    and not probe_recovery_reconfirmed_stop
                    and not recovery_followup_nonrecover_frame
                    and bool(text_stop_approach_force_budget_ok)
                ):
                    if (
                        withheld_approval_press
                        and not validation_approved
                        and not probe_recovery_requested
                    ):
                        text_withheld_presses += 1
                    approach_action, approach_trace = choose_validated_text_stop_approach_action(
                        stop_validation_result=stop_validation_result,
                        depth=depth,
                        min_depth_m=min_depth,
                        max_depth_m=max_depth,
                        approach_min_depth_m=float(scales.terminal_approach_distance_m),
                        centered_x_tolerance=float(
                            scales.centering_tolerance_fraction
                        ),
                        contact_press_remaining=max(
                            0,
                            int(text_stop_contact_press_cap)
                            - int(text_stop_contact_press_steps),
                        ),
                        forward_blocked=blocked_forward_steps > 0,
                    )
                    recovery_action_bound_allowed = True
                    recovery_action_bound_reason = "not_recovery"
                    if probe_recovery_requested:
                        proposed_recovery_action = (
                            None
                            if approach_action is None
                            else int(approach_action)
                        )
                        first_recovery_action_is_turn = (
                            text_stop_recovery_first_action
                            in {
                                int(HabitatSimActions.turn_left),
                                int(HabitatSimActions.turn_right),
                            }
                        )
                        if proposed_recovery_action is None:
                            recovery_action_bound_allowed = False
                            recovery_action_bound_reason = "no_primitive"
                        elif probe_recovery_state_corrupt:
                            recovery_action_bound_allowed = False
                            recovery_action_bound_reason = (
                                "corrupt_transaction_state"
                            )
                        elif not probe_recovery_maneuver_active:
                            recovery_action_bound_allowed = (
                                text_stop_recovery_action_count == 0
                            )
                            recovery_action_bound_reason = (
                                "first_primitive"
                                if recovery_action_bound_allowed
                                else "invalid_new_transaction_state"
                            )
                        else:
                            recovery_action_bound_allowed = bool(
                                text_stop_recovery_action_count == 1
                                and first_recovery_action_is_turn
                                and proposed_recovery_action
                                == int(HabitatSimActions.move_forward)
                                # Never re-add a forward step the guard vetoed
                                and not sgf_depth_guard_forced
                            )
                            if recovery_action_bound_allowed:
                                recovery_action_bound_reason = (
                                    "turn_then_forward"
                                )
                            elif text_stop_recovery_action_count >= 2:
                                recovery_action_bound_reason = "max_actions_reached"
                            elif not first_recovery_action_is_turn:
                                recovery_action_bound_reason = (
                                    "forward_first_has_no_continuation"
                                )
                            elif sgf_depth_guard_forced:
                                recovery_action_bound_reason = (
                                    "current_depth_guard_veto"
                                )
                            else:
                                recovery_action_bound_reason = (
                                    "continuation_must_move_forward"
                                )
                        approach_trace["recovery_bound"] = {
                            "transaction_active_before": bool(
                                probe_recovery_maneuver_active
                            ),
                            "action_count_before": int(
                                text_stop_recovery_action_count
                            ),
                            "first_action": (
                                None
                                if text_stop_recovery_first_action is None
                                else int(text_stop_recovery_first_action)
                            ),
                            "proposed_action": proposed_recovery_action,
                            "allowed": bool(recovery_action_bound_allowed),
                            "reason": str(recovery_action_bound_reason),
                            "max_actions": 2,
                        }
                    if (
                        approach_action is not None
                        and str(approach_trace.get("reason", ""))
                        == "press_to_contact"
                        and not probe_recovery_requested
                    ):
                        text_stop_contact_press_steps += 1
                    approach_trace["force_count"] = int(validated_text_stop_approach_forces)
                    approach_trace["max_forces"] = int(text_stop_approach_max_forces)
                    if probe_recovery_isolation_active:
                        approach_trace["policy_action"] = (
                            None
                            if stop_policy_decision is None
                            else str(stop_policy_decision.action)
                        )
                        approach_trace["probe_owned"] = True
                        approach_trace["recovery_requested"] = bool(
                            probe_recovery_requested
                        )
                    approach_trace["min_goal_distance_m"] = (
                        None
                        if text_stop_approach_min_goal_distance_m is None
                        else round(float(text_stop_approach_min_goal_distance_m), 4)
                    )
                    approach_trace["success_radius_m"] = round(float(success_radius), 4)
                    approach_trace["sgf_close"] = not bool(text_stop_approach_sgf_close_ok)
                    stop_validation_trace["approach"] = approach_trace
                    if approach_action is not None:
                        try:
                            approach_detection_confidence = float(
                                getattr(stop_validation_result, "detection_confidence", 0.0) or 0.0
                            )
                        except (TypeError, ValueError):
                            approach_detection_confidence = 0.0
                        close_sgf_action_ok = True
                        if (
                            probe_recovery_requested
                            and not recovery_action_bound_allowed
                        ):
                            close_sgf_action_ok = False
                        approach_trace.update(
                            {
                                "detection_confidence": round(
                                    float(approach_detection_confidence),
                                    4,
                                ),
                                "close_sgf_action_ok": bool(close_sgf_action_ok),
                            }
                        )
                        if close_sgf_action_ok:
                            if (
                                probe_recovery_requested
                                and str(approach_trace.get("reason", ""))
                                == "press_to_contact"
                            ):
                                text_stop_contact_press_steps += 1
                            validation_approved = False
                            validated_stop_approach_action = int(approach_action)
                            validated_text_stop_approach_forces += 1
                            if (
                                text_selected_target_node is not None
                                or detection_probe_attempt
                                or probe_recovery_isolation_active
                            ):
                                previous_commit_node_id = int(
                                    text_stop_approach_commit_node_id
                                )
                                if (
                                    detection_probe_attempt
                                    and detection_probe_node_xz is not None
                                ):
                                    text_stop_approach_commit_node_id = int(
                                        detection_probe_node_id
                                    )
                                    text_stop_approach_commit_probe_owned = True
                                elif not text_stop_approach_commit_probe_owned:
                                    try:
                                        text_stop_approach_commit_node_id = int(
                                            text_selected_target_node.get(
                                                "id", -1
                                            )
                                        )
                                    except (TypeError, ValueError):
                                        text_stop_approach_commit_node_id = -1
                                    text_stop_approach_commit_probe_owned = False
                                if (
                                    text_stop_approach_commit_node_id
                                    != previous_commit_node_id
                                ):
                                    text_stop_contact_press_steps = 0
                                if text_stop_approach_commit_node_id >= 0:
                                    text_stop_approach_commit_step = int(step_index)
                                    # An existing approval or ledger commitment is never
                                    # downgraded
                                    if text_stop_approach_commit_source not in {
                                        "approval",
                                        "ledger",
                                    }:
                                        text_stop_approach_commit_source = (
                                            "recovery"
                                            if probe_recovery_requested
                                            else "approval"
                                        )
                                    if probe_recovery_isolation_active:
                                        approach_trace[
                                            "commitment_source"
                                        ] = str(text_stop_approach_commit_source)
                                    if probe_recovery_requested:
                                        if probe_recovery_maneuver_active:
                                            text_stop_recovery_action_count += 1
                                            probe_recovery_continuation_executed = True
                                        else:
                                            text_stop_recovery_first_action = int(
                                                approach_action
                                            )
                                            text_stop_recovery_action_count = 1
                                        recovery_approval_snapshot_candidate = (
                                            MetricVisualClosureApprovalSnapshot(
                                                step=int(step_index),
                                                raw_validation_approved=bool(
                                                    raw_validation_approved
                                                ),
                                                policy_action=str(
                                                    stop_policy_decision.action
                                                ),
                                                selected_target_id=(
                                                    current_selected_target_id
                                                ),
                                                bbox_area_fraction=(
                                                    current_validation_bbox_area
                                                ),
                                                recovery_executed=False,
                                            )
                                        )
                                        recovery_approval_proposed_action = int(
                                            approach_action
                                        )
                                        recovery_bound_trace = approach_trace.get(
                                            "recovery_bound"
                                        )
                                        if isinstance(recovery_bound_trace, dict):
                                            recovery_bound_trace.update(
                                                {
                                                    "selected_for_execution": True,
                                                    "executed": False,
                                                    "action_count_after": int(
                                                        text_stop_recovery_action_count
                                                    ),
                                                }
                                            )
                                        probe_recovery_trace = (
                                            stop_validation_trace.get(
                                                "probe_recovery"
                                            )
                                        )
                                        if isinstance(probe_recovery_trace, dict):
                                            probe_recovery_trace.update(
                                                {
                                                    "action_proposed": int(
                                                        approach_action
                                                    ),
                                                    "action_executed": None,
                                                    "action_count_after": int(
                                                        text_stop_recovery_action_count
                                                    ),
                                                    "continuation_selected": bool(
                                                        probe_recovery_continuation_executed
                                                    ),
                                                    "continuation_executed": False,
                                                }
                                            )
                            last_stop_policy_decision = {
                                "action": EXPLORE_ACTION,
                                "reason": (
                                    "validated_stop_recovery"
                                    if probe_recovery_requested
                                    else "validated_stop_approach"
                                ),
                                "details": dict(approach_trace),
                            }
                        else:
                            approach_trace["forced"] = False
                            approach_trace["suppressed"] = True
                            approach_trace["proposed_action"] = int(approach_action)
                            approach_trace["reject_reason"] = str(
                                recovery_action_bound_reason
                            )
                            recovery_bound_trace = approach_trace.get(
                                "recovery_bound"
                            )
                            if isinstance(recovery_bound_trace, dict):
                                recovery_bound_trace["executed"] = False
                elif (
                    validation_approved
                    and bool(args.text_validated_stop_approach)
                    and not probe_recovery_reconfirmed_stop
                    and not recovery_followup_nonrecover_frame
                ):
                    if not bool(text_stop_approach_force_budget_ok):
                        approach_reject_reason = "force_limit_reached"
                    else:
                        approach_reject_reason = "disabled"
                    # Out of presses: needs a prior independent approval
                    budget_corroboration_ok = True
                    budget_corroboration_count = None
                    if (
                        approach_reject_reason == "force_limit_reached"
                        and text_stop_ledger is not None
                    ):
                        budget_key: int | None = None
                        if (
                            detection_probe_attempt
                            and detection_probe_node_xz is not None
                        ):
                            budget_key = text_stop_ledger.resolve_node_key(
                                int(detection_probe_node_id),
                                detection_probe_node_xz,
                            )
                        elif (
                            text_stop_approach_commit_probe_owned
                            and text_stop_approach_commit_node_id >= 0
                        ):
                            budget_key = text_stop_ledger.resolve_node_key(
                                int(text_stop_approach_commit_node_id),
                                text_stop_ledger.node_position(
                                    int(text_stop_approach_commit_node_id)
                                ),
                            )
                        elif text_selected_target_node is not None:
                            try:
                                budget_node_id = int(
                                    text_selected_target_node.get("id", -1)
                                )
                            except (TypeError, ValueError):
                                budget_node_id = -1
                            if budget_node_id >= 0:
                                budget_node_xz = None
                                if text_target_goal_position is not None:
                                    budget_node_xz = (
                                        float(text_target_goal_position[0]),
                                        float(text_target_goal_position[2]),
                                    )
                                budget_key = text_stop_ledger.resolve_node_key(
                                    budget_node_id, budget_node_xz
                                )
                        elif text_stop_approach_commit_node_id >= 0:
                            budget_key = text_stop_ledger.resolve_node_key(
                                int(text_stop_approach_commit_node_id),
                                text_stop_ledger.node_position(
                                    int(text_stop_approach_commit_node_id)
                                ),
                            )
                        if budget_key is None or not bool(
                            getattr(stop_validation_result, "detection_found", False)
                        ):
                            budget_corroboration_count = 0
                        else:
                            budget_state = env.get_agent_state()
                            budget_yaw = agent_yaw_deg(budget_state)
                            budget_position = np.asarray(
                                budget_state.position, dtype=np.float32
                            ).reshape(3)
                            budget_corroboration_count = int(
                                text_stop_ledger.prospective_distinct_detection_approval_count(
                                    int(budget_key),
                                    agent_position_xz=(
                                        float(budget_position[0]),
                                        float(budget_position[2]),
                                    ),
                                    agent_yaw_deg=float(
                                        budget_yaw if budget_yaw is not None else 0.0
                                    ),
                                    step=int(step_index),
                                )
                            )
                        budget_corroboration_ok = budget_corroboration_count >= int(
                            text_stop_ledger.min_commit_approvals
                        )
                    stop_validation_trace["approach"] = {
                        "enabled": True,
                        "checked": False,
                        "forced": False,
                        "reject_reason": approach_reject_reason,
                        "min_goal_distance_m": (
                            None
                            if text_stop_approach_min_goal_distance_m is None
                            else round(float(text_stop_approach_min_goal_distance_m), 4)
                        ),
                        "success_radius_m": round(float(success_radius), 4),
                        "force_count": int(validated_text_stop_approach_forces),
                        "max_forces": int(text_stop_approach_max_forces),
                        "budget_corroboration_count": budget_corroboration_count,
                        "budget_corroboration_ok": bool(budget_corroboration_ok),
                    }
                    if not budget_corroboration_ok:
                        validation_approved = False
                        last_stop_policy_decision = {
                            "action": EXPLORE_ACTION,
                            "reason": "budget_exhausted_without_corroboration",
                            "details": {
                                "independent_approvals": budget_corroboration_count,
                                "validator": summarize_stop_validation(
                                    stop_validation_result
                                ),
                            },
                        }
                recovery_transaction_cancel_requested = bool(
                    recovery_followup_requires_resolution
                    and not probe_recovery_continuation_executed
                )
                if recovery_transaction_cancel_requested:
                    if probe_recovery_state_corrupt:
                        recovery_cancel_reason = "recovery_state_corrupt"
                    elif recovery_commitment_rejected:
                        recovery_cancel_reason = "recovery_not_reconfirmed"
                    elif not probe_recovery_requested:
                        recovery_cancel_reason = "recovery_followup_not_stop"
                    else:
                        recovery_cancel_reason = (
                            "recovery_continuation_rejected"
                        )
                    probe_recovery_trace = stop_validation_trace.get(
                        "probe_recovery"
                    )
                    if isinstance(probe_recovery_trace, dict):
                        probe_recovery_trace.update(
                            {
                                "continuation_executed": False,
                                "cancel_requested": True,
                                "cancel_reason": str(recovery_cancel_reason),
                            }
                        )
                # Per-target evidence ledger (decision/stop_ledger.py)
                ledger_should_exclude_target: bool | None = None
                ledger_unanimously_rejected = False
                ledger_exclusion_details: dict[str, Any] = {}
                ledger_record: dict[str, Any] | None = None
                ledger_commit = False
                ledger_commit_anchor_waived = False
                ledger_node_id = -1
                ledger_node_position: tuple[float, float] | None = None
                ledger_hypothesis_probe_owned = bool(
                    detection_probe_attempt
                    or text_stop_approach_commit_probe_owned
                )
                ledger_hypothesis_requires_probe_diversity = bool(
                    ledger_hypothesis_probe_owned
                )
                if (
                    text_stop_ledger is not None
                    and stop_validation_result is not None
                    and (
                        text_selected_target_node is not None
                        or detection_probe_attempt
                        or text_stop_approach_commit_probe_owned
                    )
                    and str(getattr(stop_validation_result, "validator_mode", ""))
                    != "unavailable_retry"
                    # Rejected probes are not recorded: their position key may alias a
                    # node
                    and not (
                        detection_probe_attempt
                        and not bool(
                            getattr(stop_validation_result, "approved", False)
                        )
                    )
                ):
                    if (
                        detection_probe_attempt
                        and detection_probe_node_xz is not None
                    ):
                        ledger_node_id = int(detection_probe_node_id)
                        ledger_node_position = detection_probe_node_xz
                    elif text_stop_approach_commit_probe_owned:
                        ledger_node_id = int(text_stop_approach_commit_node_id)
                        ledger_node_position = text_stop_ledger.node_position(
                            ledger_node_id
                        )
                    elif text_selected_target_node is not None:
                        try:
                            ledger_node_id = int(
                                text_selected_target_node.get("id", -1)
                            )
                        except (TypeError, ValueError):
                            ledger_node_id = -1
                        ledger_node_position = None
                        if text_target_goal_position is not None:
                            ledger_node_position = (
                                float(text_target_goal_position[0]),
                                float(text_target_goal_position[2]),
                            )
                    else:
                        ledger_node_id = -1
                        ledger_node_position = None
                    if ledger_node_id >= 0:
                        ledger_agent_state = env.get_agent_state()
                        ledger_agent_position = np.asarray(
                            ledger_agent_state.position, dtype=np.float32
                        ).reshape(3)
                        ledger_yaw = agent_yaw_deg(ledger_agent_state)
                        ledger_raw_approved = bool(
                            getattr(stop_validation_result, "approved", False)
                        )
                        ledger_record_kwargs: dict[str, Any] = dict(
                            node_id=ledger_node_id,
                            node_position_xz=ledger_node_position,
                            agent_position_xz=(
                                float(ledger_agent_position[0]),
                                float(ledger_agent_position[2]),
                            ),
                            agent_yaw_deg=float(
                                ledger_yaw if ledger_yaw is not None else 0.0
                            ),
                            step=int(step_index),
                            approved=ledger_raw_approved,
                            p_yes=float(
                                getattr(stop_validation_result, "p_yes", 0.0) or 0.0
                            ),
                            detection_found=bool(
                                getattr(stop_validation_result, "detection_found", False)
                            ),
                            detection_bbox_area_fraction=getattr(
                                stop_validation_result,
                                "detection_bbox_area_fraction",
                                None,
                            ),
                            detection_confidence=getattr(
                                stop_validation_result,
                                "detection_confidence",
                                None,
                            ),
                            context_observability=(
                                getattr(
                                    stop_validation_result,
                                    "context_observability",
                                    None,
                                )
                                if bool(
                                    getattr(stop_validation_result, "factored", False)
                                )
                                else None
                            ),
                            appearance_p=getattr(
                                stop_validation_result, "appearance_p", None
                            ),
                            evidence_weight=float(current_verdict_evidence_weight),
                        )
                        ledger_record = text_stop_ledger.record(
                            **ledger_record_kwargs
                        )
                        last_ledger_record_vantages = int(
                            ledger_record.get("vantages", 0)
                        )
                        last_ledger_record_step = int(step_index)
                        if text_shadow_stop is not None:
                            text_shadow_stop.record(**ledger_record_kwargs)
                        stop_validation_trace["ledger"] = dict(ledger_record)
                        if ledger_hypothesis_probe_owned:
                            # Probe ids may resolve onto an existing node key
                            ledger_raw_node_id = int(ledger_node_id)
                            ledger_node_id = int(ledger_record["node_key"])
                            if (
                                text_stop_approach_commit_probe_owned
                                and text_stop_approach_commit_node_id >= 0
                            ):
                                text_stop_approach_commit_node_id = int(
                                    ledger_node_id
                                )
                            stop_validation_trace["probe_ledger_identity"] = {
                                "raw_node_id": int(ledger_raw_node_id),
                                "resolved_node_key": int(ledger_node_id),
                            }
                        (
                            ledger_should_exclude_target,
                            ledger_exclusion_reason,
                            ledger_exclusion_details,
                        ) = text_stop_ledger.decide_exclusion(
                            node_id=ledger_node_id,
                            node_position_xz=ledger_node_position,
                        )
                        ledger_exclusion_details["reason"] = str(
                            ledger_exclusion_reason
                        )
                        if (
                            "sequential_decision" in ledger_exclusion_details
                            and bool(ledger_exclusion_details["sequential_decision"])
                            != bool(ledger_should_exclude_target)
                        ):
                            text_evidence_rule_overrides += 1
                        ledger_unanimously_rejected = (
                            text_stop_ledger.is_unanimously_rejected(
                                ledger_node_id, ledger_node_position
                            )
                        )
                        if ledger_unanimously_rejected:
                            ledger_retirement_alternative = False
                            if text_navigation_map is not None:
                                try:
                                    probe_excluded = set(excluded_text_node_ids)
                                    probe_excluded.add(int(ledger_node_id))
                                    probe_result = (
                                        text_navigation_map.select_target_node(
                                            agent_position=agent_position_for_text_sgf,
                                            goal_phrases=goal_match_phrases,
                                            min_surfels=int(
                                                nav_config.get(
                                                    "min_target_surfels", 64
                                                )
                                            ),
                                            min_confidence=float(
                                                nav_config.get(
                                                    "min_target_confidence", 0.2
                                                )
                                            ),
                                            excluded_node_ids=probe_excluded,
                                            excluded_positions=excluded_text_positions,
                                            exclusion_radius_m=text_exclusion_radius_m,
                                        )
                                    )
                                    ledger_retirement_alternative = bool(
                                        probe_result is not None
                                        and probe_result[0] is not None
                                    )
                                except Exception:
                                    ledger_retirement_alternative = False
                            stop_validation_trace[
                                "ledger_unanimous_rejection"
                            ] = {
                                "budget": int(
                                    text_stop_ledger.unanimous_rejection_budget()
                                ),
                                "alternative_available": bool(
                                    ledger_retirement_alternative
                                ),
                                "retire": bool(ledger_unanimously_rejected),
                            }
                        if not validation_approved:
                            ledger_distance_basis = float("inf")
                            if stop_features is not None:
                                ledger_distance_basis = float(
                                    stop_features.stop_distance_m()
                                )
                                if (
                                    float(current_verdict_evidence_weight) > 0.0
                                    and not ledger_hypothesis_probe_owned
                                ):
                                    node_basis = stop_features.node_distance_m()
                                    if math.isfinite(node_basis):
                                        ledger_distance_basis = node_basis
                            if not math.isfinite(ledger_distance_basis):
                                # Lost track: measure to the ledger position
                                ledger_node_xz = text_stop_ledger.node_position(
                                    ledger_node_id
                                )
                                if ledger_node_xz is not None:
                                    ledger_distance_basis = float(
                                        math.hypot(
                                            float(ledger_agent_position[0])
                                            - ledger_node_xz[0],
                                            float(ledger_agent_position[2])
                                            - ledger_node_xz[1],
                                        )
                                    )
                            ledger_stop_radius = float(scales.far_band_radius_m)
                            ledger_within_stop_radius = bool(
                                math.isfinite(ledger_distance_basis)
                                and ledger_distance_basis <= ledger_stop_radius
                            )
                            ledger_within_success_radius = bool(
                                math.isfinite(ledger_distance_basis)
                                and ledger_distance_basis
                                <= float(scales.attempt_radius_m)
                            )
                            (
                                ledger_commit,
                                ledger_commit_reason,
                                ledger_commit_details,
                            ) = text_stop_ledger.decide_commit(
                                node_id=ledger_node_id,
                                node_position_xz=ledger_node_position,
                                current_kind=str(
                                    ledger_record.get("kind", "")
                                ),
                                within_stop_radius=bool(
                                    ledger_within_stop_radius
                                ),
                                current_position_xz=(
                                    float(ledger_agent_position[0]),
                                    float(ledger_agent_position[2]),
                                ),
                                within_success_radius=bool(
                                    ledger_within_success_radius
                                ),
                            )
                            if (
                                ledger_commit
                                and str(ledger_commit_reason)
                                == "multi_view_approvals"
                                and ledger_hypothesis_requires_probe_diversity
                            ):
                                probe_commit_separation = (
                                    text_stop_ledger.max_approval_pose_separation_m(
                                        ledger_node_id
                                    )
                                )
                                ledger_commit_details[
                                    "probe_approval_separation_m"
                                ] = round(float(probe_commit_separation), 4)
                                ledger_commit_details[
                                    "probe_diversity_required"
                                ] = True
                                if (
                                    probe_commit_separation
                                    < text_stop_ledger.node_position_match_radius_m
                                ):
                                    ledger_commit = False
                                    ledger_commit_reason = (
                                        "probe_approvals_not_diverse"
                                    )
                            if (
                                (
                                    recovery_followup_ledger_record_only
                                    or recovery_transaction_cancel_requested
                                )
                                and ledger_commit
                            ):
                                ledger_commit_details[
                                    "suppressed_by_recovery_followup"
                                ] = True
                                ledger_commit_details[
                                    "recovery_followup_cancel_requested"
                                ] = bool(recovery_transaction_cancel_requested)
                                ledger_commit_details[
                                    "recovery_continuation_executed"
                                ] = bool(
                                    probe_recovery_continuation_executed
                                )
                                ledger_commit = False
                                ledger_commit_reason = str(
                                    recovery_cancel_reason
                                    if recovery_transaction_cancel_requested
                                    else "recovery_followup_record_only"
                                )
                            ledger_commit_details["reason"] = str(ledger_commit_reason)
                            if (
                                "sequential_decision" in ledger_commit_details
                                and bool(ledger_commit_details["sequential_decision"])
                                != bool(ledger_commit)
                            ):
                                text_evidence_rule_overrides += 1
                            if (
                                ledger_commit
                                and not raw_validation_approved
                                and bool(ledger_commit_details.get("stagnated"))
                                and bool(ledger_within_success_radius)
                            ):
                                ledger_commit_anchor_waived = True
                                ledger_commit_details["anchor_waived"] = True
                            stop_validation_trace["ledger_commit"] = (
                                ledger_commit_details
                            )
                            if (
                                ledger_commit
                                and str(ledger_commit_reason)
                                == "sequential_evidence"
                            ):
                                text_sequential_evidence_commit_count += 1
                            if ledger_commit:
                                if (
                                    text_stop_approach_commit_source == "recovery"
                                    or text_stop_recovery_action_count > 0
                                    or text_stop_recovery_first_action is not None
                                ):
                                    ledger_commit_details[
                                        "recovery_transaction_reset"
                                    ] = True
                                text_stop_recovery_action_count = 0
                                text_stop_recovery_first_action = None
                                recovery_approval_snapshot = None
                                recovery_approval_snapshot_candidate = None
                                recovery_approval_proposed_action = None
                                # Ledger commit: press toward a live detection
                                # first, without leaving the decision band
                                ledger_commit_press_action: int | None = None
                                ledger_press_band_ok = True
                                if stop_features is not None:
                                    press_step_m = max(
                                        0.0, float(args.forward_step_size)
                                    )
                                    press_target_limit = float(
                                        scales.far_band_radius_m
                                    )
                                    press_evidence_d = float(
                                        stop_features.evidence_distance_m
                                    )
                                    if math.isfinite(press_evidence_d):
                                        ledger_press_band_ok = (
                                            press_target_limit <= 0.0
                                            or (
                                                press_evidence_d
                                                <= press_target_limit
                                                and int(ledger_press_steps)
                                                < max(
                                                    1,
                                                    int(
                                                        math.ceil(
                                                            press_evidence_d
                                                            / max(1e-3, press_step_m)
                                                        )
                                                    ),
                                                )
                                            )
                                        )
                                    else:
                                        press_target_d = stop_features.node_distance_m()
                                        ledger_press_band_ok = (
                                            math.isfinite(press_target_d)
                                            and (
                                                press_target_limit <= 0.0
                                                or press_target_d
                                                <= press_target_limit - press_step_m
                                            )
                                        )
                                if (
                                    bool(args.text_validated_stop_approach)
                                    and raw_validation_approved
                                    and bool(text_stop_approach_force_budget_ok)
                                    and ledger_press_band_ok
                                    and bool(
                                        getattr(
                                            stop_validation_result,
                                            "detection_found",
                                            False,
                                        )
                                    )
                                ):
                                    (
                                        ledger_commit_press_action,
                                        ledger_commit_press_trace,
                                    ) = choose_validated_text_stop_approach_action(
                                        stop_validation_result=stop_validation_result,
                                        depth=depth,
                                        min_depth_m=min_depth,
                                        max_depth_m=max_depth,
                                        approach_min_depth_m=float(
                                            scales.terminal_approach_distance_m
                                        ),
                                        centered_x_tolerance=float(
                                            scales.centering_tolerance_fraction
                                        ),
                                        contact_press_remaining=max(
                                            0,
                                            int(text_stop_contact_press_cap)
                                            - int(text_stop_contact_press_steps),
                                        ),
                                        forward_blocked=blocked_forward_steps > 0,
                                    )
                                    if (
                                        ledger_commit_press_action is not None
                                        and str(
                                            ledger_commit_press_trace.get(
                                                "reason", ""
                                            )
                                        )
                                        == "press_to_contact"
                                    ):
                                        text_stop_contact_press_steps += 1
                                    stop_validation_trace[
                                        "ledger_commit_press"
                                    ] = dict(ledger_commit_press_trace)
                                if ledger_commit_press_action is not None:
                                    validation_approved = False
                                    validated_stop_approach_action = int(
                                        ledger_commit_press_action
                                    )
                                    validated_text_stop_approach_forces += 1
                                    if int(text_stop_approach_commit_node_id) != int(
                                        ledger_node_id
                                    ):
                                        text_stop_contact_press_steps = 0
                                        ledger_press_steps = 0
                                    ledger_press_steps += 1
                                    text_stop_approach_commit_node_id = int(
                                        ledger_node_id
                                    )
                                    text_stop_approach_commit_probe_owned = bool(
                                        ledger_hypothesis_probe_owned
                                    )
                                    text_stop_approach_commit_step = int(
                                        step_index
                                    )
                                    text_stop_approach_commit_source = "ledger"
                                    last_stop_policy_decision = {
                                        "action": EXPLORE_ACTION,
                                        "reason": "ledger_commit_press",
                                        "details": dict(ledger_commit_details),
                                    }
                                else:
                                    validation_approved = True
                                    validated_stop_approach_action = None
                                    text_stop_deferred_to_target_plan = False
                                    last_stop_policy_decision = {
                                        "action": STOP_ACTION,
                                        "reason": f"ledger_{ledger_commit_reason}",
                                        "details": dict(ledger_commit_details),
                                    }
                            # Stalled test: move to a new bearing at the same range and
                            # re-query
                            if (
                                text_stop_ledger is not None
                                and text_reobservation is None
                                and (
                                    not ledger_commit
                                    or (
                                        not raw_validation_approved
                                        and not ledger_commit_anchor_waived
                                    )
                                )
                                and not bool(ledger_should_exclude_target)
                                and not ledger_unanimously_rejected
                                and not ledger_hypothesis_probe_owned
                                and text_target_goal_position is not None
                                and text_navigation_map is not None
                                and text_stop_ledger.stagnated(
                                    ledger_node_id, ledger_node_position
                                )
                            ):
                                reobs_key = int(
                                    text_stop_ledger.resolve_node_key(
                                        ledger_node_id, ledger_node_position
                                    )
                                )
                                reobs_count = int(text_reobservation_counts.get(reobs_key, 0))
                                reobs_trace: dict[str, Any] = {
                                    "node_key": int(reobs_key),
                                    "prior_trips": int(reobs_count),
                                    "evidence_log_odds": round(
                                        float(
                                            text_stop_ledger.independent_log_odds(
                                                ledger_node_id, ledger_node_position
                                            )
                                        ),
                                        4,
                                    ),
                                }
                                reobs_goal: np.ndarray | None = None
                                reobs_geodesic = float("inf")
                                if (
                                    text_reobservation_mode == "active"
                                    and reobs_count
                                    < int(text_stop_ledger.min_commit_approvals)
                                ):
                                    reobs_agent = np.asarray(
                                        ledger_agent_position, dtype=np.float64
                                    ).reshape(3)
                                    reobs_node = np.asarray(
                                        text_target_goal_position, dtype=np.float64
                                    ).reshape(3)
                                    reobs_offset = reobs_agent[[0, 2]] - reobs_node[[0, 2]]
                                    reobs_range = max(
                                        float(np.linalg.norm(reobs_offset)),
                                        float(scales.distinct_view_distance_m),
                                    )
                                    reobs_bearing = math.atan2(
                                        float(reobs_offset[1]), float(reobs_offset[0])
                                    )
                                    reobs_used = text_stop_ledger.observation_positions(
                                        reobs_key
                                    )
                                    reobs_min_sep = float(scales.distinct_view_distance_m)
                                    for reobs_turn in (0.5 * math.pi, -0.5 * math.pi, math.pi):
                                        angle = reobs_bearing + reobs_turn
                                        candidate = np.asarray(
                                            [
                                                reobs_node[0] + reobs_range * math.cos(angle),
                                                float(reobs_agent[1]),
                                                reobs_node[2] + reobs_range * math.sin(angle),
                                            ],
                                            dtype=np.float32,
                                        )
                                        try:
                                            snapped = text_navigation_map.snap_to_free(candidate)
                                        except Exception:
                                            snapped = None
                                        if snapped is None or not np.all(np.isfinite(snapped)):
                                            continue
                                        snapped = np.asarray(snapped, dtype=np.float32).reshape(3)
                                        snapped[1] = float(reobs_agent[1])
                                        if float(planar_distance_xz(snapped, reobs_agent)) < reobs_min_sep:
                                            continue
                                        if any(
                                            math.hypot(float(snapped[0]) - ux, float(snapped[2]) - uz)
                                            < reobs_min_sep
                                            for ux, uz in reobs_used
                                        ):
                                            continue
                                        try:
                                            geodesic = float(
                                                text_navigation_map.geodesic_distance(
                                                    reobs_agent.astype(np.float32), snapped
                                                )
                                            )
                                        except Exception:
                                            geodesic = float("nan")
                                        if not (math.isfinite(geodesic) and geodesic >= 0.0):
                                            continue
                                        if geodesic < reobs_geodesic:
                                            reobs_geodesic = geodesic
                                            reobs_goal = snapped.copy()
                                if reobs_goal is not None:
                                    text_reobservation = {
                                        "key": int(reobs_key),
                                        "node_id": int(ledger_node_id),
                                        "goal": reobs_goal,
                                        "origin": np.asarray(
                                            ledger_agent_position, dtype=np.float32
                                        ).reshape(3).copy(),
                                        "start_step": int(step_index),
                                        "budget_steps": int(
                                            scales.steps_for_distance(reobs_geodesic)
                                            + scales.full_turn_actions
                                        ),
                                    }
                                    text_reobservation_counts[reobs_key] = reobs_count + 1
                                    text_reobservation_launches += 1
                                    reobs_trace.update(
                                        {
                                            "launched": True,
                                            "goal_m": world_position_list(reobs_goal),
                                            "geodesic_m": round(reobs_geodesic, 3),
                                            "budget_steps": int(
                                                text_reobservation["budget_steps"]
                                            ),
                                        }
                                    )
                                else:
                                    reobs_trace["launched"] = False
                                    reobs_trace["retired"] = True
                                    excluded_text_node_ids.add(int(ledger_node_id))
                                    retire_text_hypothesis(
                                        np.asarray(
                                            text_target_goal_position, dtype=np.float32
                                        ).copy()
                                    )
                                    if text_reobservation_mode == "active":
                                        text_reobservation_retirements += 1
                                    else:
                                        reobs_trace["passive_release"] = True
                                        text_passive_releases += 1
                                    tracked_text_target_node_id = None
                                    tracked_text_target_steps = 0
                                    stop_validation_trace["target_excluded"] = True
                                stop_validation_trace["reobservation"] = reobs_trace
                            # Dissent kinds that conclude an in-flight approach
                            approach_commit_window_steps = 2 * max(
                                1, int(text_stop_approach_max_forces)
                            )
                            approach_commit_same_node = (
                                text_stop_approach_commit_node_id >= 0
                                and text_stop_ledger.resolve_node_key(
                                    text_stop_approach_commit_node_id, None
                                )
                                == int(ledger_record.get("node_key", -2))
                            )
                            approach_commit_kind = str(
                                ledger_record.get("kind", "")
                            )
                            approach_commit_window_ok = (
                                int(step_index)
                                - int(text_stop_approach_commit_step)
                                <= approach_commit_window_steps
                            )
                            if text_stop_approach_commit_source == "ledger":
                                approach_commit_conclude_kinds = (
                                    TargetStopLedger.ABSENCE,
                                    TargetStopLedger.CLOSE_UP,
                                    TargetStopLedger.INFORMATIVE,
                                    TargetStopLedger.APPROVAL,
                                )
                            elif text_stop_approach_commit_source == "approval":
                                approach_commit_conclude_kinds = (
                                    TargetStopLedger.ABSENCE,
                                    TargetStopLedger.CLOSE_UP,
                                )
                            else:
                                # RECOVER concludes only on a later approval
                                approach_commit_conclude_kinds = ()
                            if (
                                approach_commit_same_node
                                and text_stop_approach_commit_source
                                != "ledger"
                                and approach_commit_kind
                                == TargetStopLedger.INFORMATIVE
                            ):
                                informative_recovery_reset = bool(
                                    text_stop_approach_commit_source == "recovery"
                                    or text_stop_recovery_action_count > 0
                                    or text_stop_recovery_first_action is not None
                                )
                                text_stop_approach_commit_node_id = -1
                                text_stop_approach_commit_probe_owned = False
                                text_stop_approach_commit_step = -(10**9)
                                text_stop_approach_commit_source = ""
                                text_stop_contact_press_steps = 0
                                text_stop_recovery_action_count = 0
                                text_stop_recovery_first_action = None
                                recovery_approval_snapshot = None
                                recovery_approval_snapshot_candidate = None
                                recovery_approval_proposed_action = None
                                stop_validation_trace["approach_commitment"] = {
                                    "cancelled": True,
                                    "reason": "informative_contradiction",
                                }
                                if informative_recovery_reset:
                                    stop_validation_trace[
                                        "approach_commitment"
                                    ]["recovery_transaction_reset"] = True
                            elif (
                                approach_commit_same_node
                                and text_stop_approach_commit_source == "ledger"
                                and not text_stop_ledger.commit_holds(
                                    ledger_node_id, ledger_node_position
                                )
                            ):
                                text_stop_approach_commit_node_id = -1
                                text_stop_approach_commit_probe_owned = False
                                text_stop_approach_commit_step = -(10**9)
                                text_stop_approach_commit_source = ""
                                text_stop_contact_press_steps = 0
                                stop_validation_trace["approach_commitment"] = {
                                    "cancelled": True,
                                    "reason": "evidence_below_standard",
                                    "rule": str(text_evidence_rule),
                                    "evidence_log_odds": round(
                                        float(
                                            text_stop_ledger.independent_log_odds(
                                                ledger_node_id, ledger_node_position
                                            )
                                        ),
                                        4,
                                    ),
                                }
                            elif (
                                not validation_approved
                                and approach_commit_same_node
                                and approach_commit_window_ok
                                and approach_commit_kind
                                in approach_commit_conclude_kinds
                                and math.isfinite(ledger_distance_basis)
                                and ledger_distance_basis <= ledger_stop_radius
                                # Not on the ledger commit's launch step: its press runs
                                # first
                                and not (
                                    text_stop_approach_commit_source == "ledger"
                                    and int(text_stop_approach_commit_step)
                                    >= int(step_index)
                                )
                            ):
                                validation_approved = True
                                validated_stop_approach_action = None
                                text_stop_deferred_to_target_plan = False
                                stop_validation_trace["approach_commitment"] = {
                                    "committed": True,
                                    "source": str(
                                        text_stop_approach_commit_source
                                    ),
                                    "node_id": int(
                                        text_stop_approach_commit_node_id
                                    ),
                                    "commit_step": int(
                                        text_stop_approach_commit_step
                                    ),
                                    "window_steps": int(
                                        approach_commit_window_steps
                                    ),
                                    "current_kind": approach_commit_kind,
                                    "distance_basis_m": round(
                                        float(ledger_distance_basis), 4
                                    ),
                                }
                                last_stop_policy_decision = {
                                    "action": STOP_ACTION,
                                    "reason": "approach_commitment",
                                    "details": dict(
                                        stop_validation_trace[
                                            "approach_commitment"
                                        ]
                                    ),
                                }
                if probe_recovery_reconfirmed_stop:
                    recovery_node_id = int(
                        text_stop_approach_commit_node_id
                    )
                    recovery_commit_step = int(
                        text_stop_approach_commit_step
                    )
                    recovery_action_count = int(recovery_entry_action_count)
                    recovery_first_action = recovery_entry_first_action
                    recovery_confirm_reason = (
                        "recovery_monotonic_bbox_closure"
                        if metric_visual_direct_closure_kind
                        == "recovery_monotonic_bbox"
                        else "recovery_reconfirmed_stop"
                    )
                    validation_approved = True
                    validated_stop_approach_action = None
                    text_stop_deferred_to_target_plan = False
                    navigation_action = int(HabitatSimActions.stop)
                    habitat_action = int(HabitatSimActions.stop)
                    sgf_stop_forced = True
                    sgf_stop_trace["requested"] = True
                    sgf_stop_trace["recovery_reconfirmed"] = True
                    sgf_stop_trace["recovery_confirmation_reason"] = str(
                        recovery_confirm_reason
                    )
                    text_stop_approach_commit_node_id = -1
                    text_stop_approach_commit_probe_owned = False
                    text_stop_approach_commit_step = -(10**9)
                    text_stop_approach_commit_source = ""
                    text_stop_contact_press_steps = 0
                    text_stop_recovery_action_count = 0
                    text_stop_recovery_first_action = None
                    recovery_approval_snapshot = None
                    recovery_approval_snapshot_candidate = None
                    recovery_approval_proposed_action = None
                    recovery_confirm_trace = {
                        "committed": True,
                        "source": "recovery",
                        "reason": str(recovery_confirm_reason),
                        "node_id": recovery_node_id,
                        "commit_step": recovery_commit_step,
                        "confirmation_step": int(step_index),
                        "action_count": recovery_action_count,
                        "first_action": (
                            None
                            if recovery_first_action is None
                            else int(recovery_first_action)
                        ),
                    }
                    stop_validation_trace[
                        "approach_commitment"
                    ] = recovery_confirm_trace
                    probe_recovery_trace = stop_validation_trace.get(
                        "probe_recovery"
                    )
                    if isinstance(probe_recovery_trace, dict):
                        probe_recovery_trace.update(
                            {
                                "committed": True,
                                "commit_reason": str(
                                    recovery_confirm_reason
                                ),
                            }
                        )
                    last_stop_policy_decision = (
                        summarize_stop_decision(stop_policy_decision)
                        or {
                            "action": STOP_ACTION,
                            "reason": str(recovery_confirm_reason),
                            "details": dict(recovery_confirm_trace),
                        }
                    )
                if recovery_transaction_cancel_requested:
                    recovery_action_count = int(recovery_entry_action_count)
                    recovery_first_action = recovery_entry_first_action
                    validation_approved = False
                    validated_stop_approach_action = None
                    text_stop_deferred_to_target_plan = False
                    text_stop_approach_commit_node_id = -1
                    text_stop_approach_commit_probe_owned = False
                    text_stop_approach_commit_step = -(10**9)
                    text_stop_approach_commit_source = ""
                    text_stop_contact_press_steps = 0
                    text_stop_recovery_action_count = 0
                    text_stop_recovery_first_action = None
                    recovery_approval_snapshot = None
                    recovery_approval_snapshot_candidate = None
                    recovery_approval_proposed_action = None
                    recovery_kind = ""
                    ledger_trace = stop_validation_trace.get("ledger")
                    if isinstance(ledger_trace, dict):
                        recovery_kind = str(ledger_trace.get("kind", ""))
                    recovery_cancel_trace = {
                        "cancelled": True,
                        "source": "recovery",
                        "reason": str(recovery_cancel_reason),
                        "current_kind": recovery_kind or None,
                        "action_count": recovery_action_count,
                        "first_action": (
                            None
                            if recovery_first_action is None
                            else int(recovery_first_action)
                        ),
                        "continuation_executed": False,
                    }
                    stop_validation_trace[
                        "approach_commitment"
                    ] = recovery_cancel_trace
                    probe_recovery_trace = stop_validation_trace.get(
                        "probe_recovery"
                    )
                    if isinstance(probe_recovery_trace, dict):
                        probe_recovery_trace.update(
                            {
                                "cancelled": True,
                                "cancel_reason": str(recovery_cancel_reason),
                            }
                        )
                    last_stop_policy_decision = {
                        "action": EXPLORE_ACTION,
                        "reason": str(recovery_cancel_reason),
                        "details": dict(recovery_cancel_trace),
                    }
                if (
                    metric_visual_closure_enabled
                    and metric_visual_closure_commit_due is None
                    and metric_visual_direct_closure_decision is None
                    and stop_policy_decision is not None
                    and stop_features is not None
                ):
                    metric_bbox_center = getattr(
                        stop_validation_result,
                        "detection_bbox_center",
                        None,
                    )
                    metric_bbox_center_y = None
                    if metric_bbox_center is not None:
                        try:
                            metric_bbox_center_y = float(
                                metric_bbox_center[1]
                            )
                        except (IndexError, TypeError, ValueError):
                            metric_bbox_center_y = None
                    metric_ledger_trace = stop_validation_trace.get(
                        "ledger"
                    )
                    metric_ledger_present = isinstance(
                        metric_ledger_trace,
                        dict,
                    )
                    metric_visual_closure_decision = (
                        decide_metric_visual_closure(
                            raw_validation_approved=bool(
                                raw_validation_approved
                            ),
                            detection_found=bool(
                                getattr(
                                    stop_validation_result,
                                    "detection_found",
                                    False,
                                )
                            ),
                            ordinary_sgf_stop=bool(
                                stop_source == "sgf_stop"
                                and not probe_recovery_isolation_active
                            ),
                            policy_action=str(
                                stop_policy_decision.action
                            ),
                            detection_confidence=getattr(
                                stop_validation_result,
                                "detection_confidence",
                                None,
                            ),
                            bbox_area_fraction=getattr(
                                stop_validation_result,
                                "detection_bbox_area_fraction",
                                None,
                            ),
                            bbox_center_y=metric_bbox_center_y,
                            target_distance_m=(
                                stop_features.target_distance_m
                            ),
                            ledger_present=bool(metric_ledger_present),
                            ledger_close_up_dissents=(
                                metric_ledger_trace.get(
                                    "close_up_dissents"
                                )
                                if metric_ledger_present
                                else None
                            ),
                            config=metric_visual_closure_config,
                        )
                    )
                    metric_visual_closure_trace.update(
                        {
                            "triggered": bool(
                                metric_visual_closure_decision.arm
                            ),
                            "reason": str(
                                metric_visual_closure_decision.reason
                            ),
                            "details": dict(
                                metric_visual_closure_decision.details
                            ),
                        }
                    )
                    stop_validation_trace[
                        "metric_visual_closure"
                    ] = dict(metric_visual_closure_trace)
                if metric_visual_direct_closure_decision is not None:
                    validation_approved = True
                    validated_stop_approach_action = None
                    text_stop_deferred_to_target_plan = False
                    stop_validation_trace["direct_closure_finalized"] = {
                        "kind": str(metric_visual_direct_closure_kind),
                        "reason": str(
                            metric_visual_direct_closure_decision.reason
                        ),
                    }
                # Require a fresh approval unless the anchor was waived
                if (
                    validation_approved
                    and not raw_validation_approved
                    and not ledger_commit_anchor_waived
                ):
                    validation_approved = False
                    validated_stop_approach_action = None
                    stop_validation_trace[
                        "current_validation_reacquisition_required"
                    ] = {
                        "required": True,
                        "raw_validator_approved": False,
                        "detection_found": bool(
                            getattr(
                                stop_validation_result,
                                "detection_found",
                                False,
                            )
                        ),
                        "preserved_candidate_evidence": True,
                    }
                    last_stop_policy_decision = {
                        "action": EXPLORE_ACTION,
                        "reason": "current_validation_reacquisition_required",
                        "details": dict(
                            stop_validation_trace[
                                "current_validation_reacquisition_required"
                            ]
                        ),
                    }
                stop_validation_trace.update(
                    {
                        "approved": bool(validation_approved),
                        "validator": summarize_stop_validation(stop_validation_result),
                        "policy": (
                            summarize_stop_decision(stop_policy_decision)
                            if stop_policy_decision is not None
                            else dict(last_stop_policy_decision)
                        ),
                    }
                )
                if validation_approved:
                    text_sgf_target_found = True
                    if (
                        not probe_recovery_isolation_active
                        and text_selected_target_node is not None
                    ):
                        try:
                            approved_node_id = int(text_selected_target_node["id"])
                            text_rejection_near_success_counts[approved_node_id] = 0
                        except (KeyError, TypeError, ValueError):
                            pass
                elif validation_unavailable_retry:
                    # An unavailable validator is not negative evidence
                    if (
                        recovery_followup_observation_probe
                        and recovery_followup_prev_navigation_action is not None
                        and recovery_followup_prev_habitat_action is not None
                    ):
                        navigation_action = int(
                            recovery_followup_prev_navigation_action
                        )
                        habitat_action = int(
                            recovery_followup_prev_habitat_action
                        )
                        replacement_source = (
                            "probe_recovery_followup_passthrough"
                        )
                    elif (
                        detection_probe_attempt
                        and detection_probe_prev_navigation_action is not None
                        and detection_probe_prev_habitat_action is not None
                    ):
                        navigation_action = int(
                            detection_probe_prev_navigation_action
                        )
                        habitat_action = int(detection_probe_prev_habitat_action)
                        replacement_source = "detection_probe_passthrough"
                    else:
                        replacement_action = safe_non_stop_navigation_action()
                        navigation_action = int(replacement_action)
                        habitat_action = int(replacement_action)
                        replacement_source = (
                            "stop_validation_unavailable_retry"
                        )
                    stop_suppressed = True
                    stop_validation_trace["retry_deferred"] = True
                    stop_validation_trace["replacement_source"] = replacement_source
                    stop_validation_trace["replacement_action"] = int(
                        habitat_action
                    )
                else:
                    if validated_stop_approach_action is None:
                        stop_validation_rejections += 1
                        if (
                            not probe_recovery_isolation_active
                            and text_selected_target_node is not None
                        ):
                            rejected_target_label = str(
                                text_selected_target_node.get("label", "")
                            )
                            try:
                                rejected_target_node_id = int(
                                    text_selected_target_node.get("id", -1)
                                )
                            except (TypeError, ValueError):
                                rejected_target_node_id = -1
                            # A rejection without a detection blames the label, not the
                            # instance
                            rejected_without_detection = not bool(
                                getattr(stop_validation_result, "detection_found", False)
                            ) and not validation_unobservable
                            if (
                                rejected_target_label
                                and rejected_target_node_id >= 0
                                and rejected_without_detection
                            ):
                                text_label_rejected_node_ids.setdefault(
                                    rejected_target_label, set()
                                ).add(rejected_target_node_id)
                        relaxed_rejection_keeps_target = (
                            not probe_recovery_isolation_active
                            and stop_source == "sgf_stop"
                            and bool(sgf_stop_relaxed_forced)
                        )
                        target_plan_deferral_keeps_target = bool(
                            text_stop_deferred_to_target_plan
                        )
                        near_success_rejection_keeps_target = False
                        near_success_rejection_scan_action: int | None = None
                        near_success_rejection_action_override_ok = False
                        policy_veto_of_approved_stop = bool(
                            stop_validation_result is not None
                            and getattr(stop_validation_result, "approved", False)
                            and stop_validation_trace.get("direct_stop_withheld") is None
                        )
                        if ledger_should_exclude_target is False:
                            stop_validation_trace["ledger_retained_target"] = dict(
                                ledger_exclusion_details
                            )
                        probe_bank_invalidated = False
                        if (
                            ledger_hypothesis_probe_owned
                            and not policy_veto_of_approved_stop
                            and bool(ledger_should_exclude_target)
                            and ledger_node_id >= 0
                            and endgame_return_planner is not None
                        ):
                            probe_bank_invalidated = (
                                endgame_return_planner.invalidate_source(
                                    int(ledger_node_id),
                                    step=int(step_index),
                                    probe_owned=True,
                                )
                            )
                        if probe_bank_invalidated:
                            stop_validation_trace[
                                "endgame_probe_bank_invalidated"
                            ] = True
                            dormant_summary = (
                                endgame_return_planner.summary().get(
                                    "dormant_candidate"
                                )
                            )
                            if dormant_summary is not None:
                                stop_validation_trace[
                                    "endgame_dormant_candidate"
                                ] = dict(dormant_summary)
                        if (
                            text_selected_target_node is not None
                            and not policy_veto_of_approved_stop
                            and not ledger_hypothesis_probe_owned
                            and (
                                # Ledger decisions bypass the retention guards
                                bool(ledger_unanimously_rejected)
                                or bool(ledger_should_exclude_target)
                                or (
                                    not relaxed_rejection_keeps_target
                                    and not target_plan_deferral_keeps_target
                                    and not near_success_rejection_keeps_target
                                    and ledger_should_exclude_target is None
                                )
                            )
                        ):
                            excluded_text_target = False
                            excluded_text_node_id = -1
                            try:
                                excluded_text_node_id = int(
                                    text_selected_target_node["id"]
                                )
                                excluded_text_node_ids.add(excluded_text_node_id)
                                excluded_text_target = True
                            except (KeyError, TypeError, ValueError):
                                pass
                            if ledger_exclusion_details:
                                stop_validation_trace["ledger_exclusion"] = dict(
                                    ledger_exclusion_details
                                )
                            excluded_text_node_position = None
                            if text_target_goal_position is not None:
                                excluded_text_node_position = (
                                    float(text_target_goal_position[0]),
                                    float(text_target_goal_position[2]),
                                )
                                excluded_text_target = True
                            retire_text_hypothesis(
                                None
                                if text_target_goal_position is None
                                else text_target_goal_position.copy(),
                            )
                            if excluded_text_target:
                                selected_bank_invalidated = False
                                if (
                                    endgame_return_planner is not None
                                    and text_stop_ledger is not None
                                    and excluded_text_node_id >= 0
                                ):
                                    excluded_ledger_key = (
                                        text_stop_ledger.resolve_node_key(
                                            excluded_text_node_id,
                                            excluded_text_node_position,
                                        )
                                    )
                                    selected_bank_invalidated = (
                                        endgame_return_planner.invalidate_source(
                                            int(excluded_ledger_key),
                                            step=int(step_index),
                                            probe_owned=False,
                                        )
                                    )
                                if selected_bank_invalidated:
                                    stop_validation_trace[
                                        "endgame_bank_invalidated"
                                    ] = True
                                    dormant_summary = (
                                        endgame_return_planner.summary().get(
                                            "dormant_candidate"
                                        )
                                    )
                                    if dormant_summary is not None:
                                        stop_validation_trace[
                                            "endgame_dormant_candidate"
                                        ] = dict(dormant_summary)
                                stop_validation_trace["target_excluded"] = True
                            tracked_text_target_node_id = None
                            tracked_text_target_steps = 0
                        elif target_plan_deferral_keeps_target:
                            stop_validation_trace["target_retained_after_scene_only_deferral"] = True
                        elif relaxed_rejection_keeps_target:
                            stop_validation_trace["target_retained_after_relaxed_rejection"] = True
                            if (
                                not probe_recovery_isolation_active
                                and text_selected_target_node is not None
                            ):
                                try:
                                    relaxed_sgf_stop_cooldown_node_id = int(text_selected_target_node["id"])
                                    relaxed_sgf_stop_cooldown_until_step = (
                                        step_index + int(requery_interval_steps)
                                    )
                                except (KeyError, TypeError, ValueError):
                                    pass
                        if probe_recovery_isolation_active:
                            if (
                                recovery_transaction_cancel_requested
                                and recovery_followup_observation_probe
                                and recovery_followup_prev_habitat_action
                                is not None
                            ):
                                replacement_action = int(
                                    recovery_followup_prev_habitat_action
                                )
                                stop_validation_trace[
                                    "replacement_source"
                                ] = "probe_recovery_followup_passthrough"
                            else:
                                replacement_action = (
                                    safe_non_stop_navigation_action()
                                )
                                stop_validation_trace[
                                    "replacement_source"
                                ] = "probe_followup_scan"
                        elif (
                            target_plan_deferral_keeps_target
                            or (
                                near_success_rejection_keeps_target
                                and near_success_rejection_action_override_ok
                            )
                        ):
                            if (
                                text_target_plan_action is not None
                                and int(text_target_plan_action)
                                != int(HabitatSimActions.stop)
                            ):
                                replacement_action = int(text_target_plan_action)
                            elif (
                                near_success_rejection_keeps_target
                                and near_success_rejection_action_override_ok
                            ):
                                replacement_action = int(
                                    near_success_rejection_scan_action
                                    if near_success_rejection_scan_action is not None
                                    else int(HabitatSimActions.turn_left)
                                )
                            else:
                                replacement_action = safe_non_stop_navigation_action()
                            if (
                                near_success_rejection_keeps_target
                                and near_success_rejection_action_override_ok
                            ):
                                replacement_source = (
                                    "text_sgf_near_success_rejection_scan"
                                    if (
                                        text_target_plan_action is None
                                        or int(text_target_plan_action)
                                        == int(HabitatSimActions.stop)
                                    )
                                    else "text_sgf_target_plan_after_near_success_rejection"
                                )
                            elif target_plan_deferral_keeps_target:
                                replacement_source = "text_sgf_target_plan_deferral"
                            else:
                                replacement_source = "text_sgf_target_plan"
                            stop_validation_trace["replacement_source"] = replacement_source
                        else:
                            replacement_action = safe_non_stop_navigation_action()
                            if near_success_rejection_keeps_target:
                                stop_validation_trace["replacement_source"] = (
                                    "navigation_scan_retained_near_success_target"
                                )
                            else:
                                stop_validation_trace["replacement_source"] = "navigation_scan"
                            if (
                                text_selected_target_node is not None
                                and bool(sgf_stop_forced)
                                and text_reobservation_mode != "passive"
                            ):
                                # Keep the node in frame for the next verdict
                                try:
                                    face_state = env.get_agent_state()
                                    face_heading = float(
                                        compute_heading_error_deg(
                                            face_state.rotation,
                                            np.asarray(
                                                text_selected_target_node.get(
                                                    "centroid_m"
                                                ),
                                                dtype=np.float32,
                                            ).reshape(3),
                                            np.asarray(
                                                face_state.position, dtype=np.float32
                                            ).reshape(3),
                                        )
                                    )
                                    if abs(face_heading) > 0.5 * float(args.turn_angle):
                                        replacement_action = int(
                                            HabitatSimActions.turn_right
                                            if face_heading > 0.0
                                            else HabitatSimActions.turn_left
                                        )
                                        text_face_node_turns += 1
                                        stop_validation_trace["replacement_source"] = (
                                            "hypothesis_unobservable_face_node"
                                            if validation_unobservable
                                            else "navigation_scan_face_node"
                                        )
                                    stop_validation_trace[
                                        "face_node_heading_error_deg"
                                    ] = round(face_heading, 2)
                                except Exception:
                                    pass
                    else:
                        replacement_action = int(validated_stop_approach_action)
                        stop_validation_trace[
                            "target_retained_after_approach"
                        ] = True
                        stop_validation_trace["replacement_source"] = (
                            "validated_stop_recovery"
                            if probe_recovery_requested
                            else "validated_stop_approach"
                        )
                    navigation_action = int(replacement_action)
                    habitat_action = int(replacement_action)
                    stop_validation_trace["replacement_action"] = int(replacement_action)
                    if (
                        recovery_transaction_cancel_requested
                        and recovery_followup_observation_probe
                        and recovery_followup_prev_navigation_action is not None
                        and recovery_followup_prev_habitat_action is not None
                    ):
                        navigation_action = int(
                            recovery_followup_prev_navigation_action
                        )
                        habitat_action = int(
                            recovery_followup_prev_habitat_action
                        )
                        stop_validation_trace["replacement_source"] = (
                            "probe_recovery_followup_passthrough"
                        )
                        stop_validation_trace["replacement_action"] = int(
                            habitat_action
                        )
                    if (
                        detection_probe_attempt
                        and validated_stop_approach_action is None
                        and detection_probe_prev_habitat_action is not None
                        and habitat_action != int(HabitatSimActions.stop)
                    ):
                        navigation_action = int(detection_probe_prev_navigation_action)
                        habitat_action = int(detection_probe_prev_habitat_action)
                        stop_validation_trace["replacement_source"] = (
                            "detection_probe_passthrough"
                        )
                        stop_validation_trace["replacement_action"] = int(
                            habitat_action
                        )

            fresh_trusted_fallback_source = (
                ""
                if endgame_return_planner is None
                else str(endgame_return_planner.active_source)
            )
            fresh_trusted_cancels_fallback = bool(
                endgame_return_planner is not None
                and habitat_action != int(HabitatSimActions.stop)
                and fresh_trusted_fallback_source == "dormant_approved"
                and endgame_return_planner.activated_step == int(step_index)
                and endgame_return_planner.candidate is not None
                and endgame_return_planner.candidate.step == int(step_index)
            )
            if fresh_trusted_cancels_fallback:
                fallback_cancelled = (
                    endgame_return_planner.cancel_dormant_activation(
                        "fresh_trusted_stop_evidence"
                    )
                )
                stop_validation_trace[
                    "fallback_return_cancelled_for_fresh_trusted_evidence"
                ] = {
                    "cancelled": bool(fallback_cancelled),
                    "source": str(fresh_trusted_fallback_source),
                }

            if (
                endgame_return_planner is not None
                and step_endgame_return_trace
            ):
                step_endgame_return_trace.update(
                    endgame_return_planner.summary()
                )

            if (
                step_index >= int(args.max_steps) - 1
                and habitat_action != int(HabitatSimActions.stop)
            ):
                navigation_action = int(HabitatSimActions.stop)
                habitat_action = int(HabitatSimActions.stop)
                forced_budget_stop = True

            if metric_visual_direct_closure_decision is not None:
                metric_visual_closure_precommit_action = int(habitat_action)
                metric_visual_closure_intervened = True
                navigation_action = int(HabitatSimActions.stop)
                habitat_action = int(HabitatSimActions.stop)
                metric_visual_closure_commit_forced = True
                metric_visual_closure_commit_count += 1
                if metric_visual_closure_intervened:
                    metric_visual_closure_intervention_count += 1
                if (
                    metric_visual_direct_closure_kind
                    == "recovery_monotonic_bbox"
                ):
                    metric_visual_closure_recovery_monotonic_commit_count += 1
                elif (
                    metric_visual_direct_closure_kind
                    == "contextual_closeup_persistence"
                ):
                    metric_visual_closure_contextual_closeup_commit_count += 1
                text_sgf_target_found = True
                metric_visual_closure_trace.update(
                    {
                        "committed": True,
                        "direct": True,
                        "kind": str(metric_visual_direct_closure_kind),
                        "commit_step": int(step_index),
                        "precommit_action": int(
                            metric_visual_closure_precommit_action
                        ),
                        "intervened": bool(
                            metric_visual_closure_intervened
                        ),
                        "executed_action": int(HabitatSimActions.stop),
                    }
                )
                metric_stop_trace = stop_validation_trace.get(
                    "metric_visual_closure"
                )
                if isinstance(metric_stop_trace, dict):
                    metric_stop_trace.update(metric_visual_closure_trace)
            elif metric_visual_closure_commit_due is not None:
                metric_visual_closure_precommit_action = int(habitat_action)
                metric_visual_closure_intervened = bool(
                    metric_visual_closure_precommit_action
                    != int(HabitatSimActions.stop)
                )
                navigation_action = int(HabitatSimActions.stop)
                habitat_action = int(HabitatSimActions.stop)
                metric_visual_closure_commit_forced = bool(
                    metric_visual_closure_intervened
                )
                metric_visual_closure_commit_count += 1
                if metric_visual_closure_intervened:
                    metric_visual_closure_intervention_count += 1
                text_sgf_target_found = True
                metric_visual_closure_trace.update(
                    {
                        "committed": True,
                        "commit_step": int(step_index),
                        "precommit_action": int(
                            metric_visual_closure_precommit_action
                        ),
                        "intervened": bool(
                            metric_visual_closure_intervened
                        ),
                        "executed_action": int(HabitatSimActions.stop),
                    }
                )
            elif (
                metric_visual_closure_latch is not None
                and metric_visual_closure_decision is not None
                and metric_visual_closure_decision.arm
                and habitat_action != int(HabitatSimActions.stop)
                and not closure_evidence_ok
            ):
                metric_visual_closure_withheld_count += 1
                metric_visual_closure_trace.update(
                    {
                        "withheld": True,
                        "evidence_log_odds": (
                            None
                            if closure_evidence_log_odds is None
                            else round(float(closure_evidence_log_odds), 4)
                        ),
                    }
                )
            elif (
                metric_visual_closure_latch is not None
                and metric_visual_closure_decision is not None
                and metric_visual_closure_decision.arm
                and habitat_action != int(HabitatSimActions.stop)
            ):
                metric_visual_closure_armed = (
                    metric_visual_closure_latch.arm(
                        step=int(step_index),
                        action=int(habitat_action),
                        decision=metric_visual_closure_decision,
                    )
                )
                if metric_visual_closure_armed:
                    metric_visual_closure_armed_count += 1
                    metric_visual_closure_trace.update(
                        {
                            "armed": True,
                            "trigger_step": int(step_index),
                            "refinement_action": int(habitat_action),
                            "commit_step": int(step_index) + 1,
                        }
                    )
                    metric_stop_trace = stop_validation_trace.get(
                        "metric_visual_closure"
                    )
                    if isinstance(metric_stop_trace, dict):
                        metric_stop_trace.update(
                            {
                                "armed": True,
                                "refinement_action": int(habitat_action),
                                "commit_step": int(step_index) + 1,
                            }
                        )

            if panorama_action is not None:
                panorama_proposal_selected = bool(panorama_forced)
                (
                    panorama_executed,
                    panorama_remaining,
                ) = reconcile_panoramic_observation_action(
                    proposal_action=panorama_action,
                    proposal_selected=panorama_proposal_selected,
                    final_action=habitat_action,
                    sweep_turn_action=int(HabitatSimActions.turn_left),
                    remaining=int(panorama_remaining),
                )
                panorama_trace["proposal_selected"] = bool(
                    panorama_proposal_selected
                )
                if panorama_executed:
                    panorama_action_count += 1
                    panorama_trace["action"] = int(habitat_action)
                    panorama_trace["remaining_after"] = int(
                        panorama_remaining
                    )
                    if bool(args.image_vlm_panoramic_query):
                        # Views belong to the pre-action loop pose
                        panorama_sweep_forward_xz = rotation_to_forward_xz(
                            loop_agent_state.rotation
                        )
                        panorama_sweep_views.append(
                            (
                                float(
                                    math.atan2(
                                        -float(panorama_sweep_forward_xz[0]),
                                        -float(panorama_sweep_forward_xz[1]),
                                    )
                                ),
                                rgb_observation.copy(),
                                np.asarray(
                                    loop_agent_position, dtype=np.float32
                                ).reshape(3).copy(),
                                None if depth is None else np.asarray(depth, dtype=np.float32).copy(),
                                (
                                    None
                                    if world_from_camera_cv is None
                                    else np.asarray(world_from_camera_cv, dtype=np.float32).copy()
                                ),
                            )
                        )
                else:
                    panorama_trace.update(
                        {
                            "preempted": True,
                            "final_action": int(habitat_action),
                            "remaining_after": int(panorama_remaining),
                        }
                    )
                panorama_forced = bool(panorama_executed)
                if (
                    bool(args.image_vlm_panoramic_query)
                    and int(panorama_remaining) <= 0
                    and panorama_sweep_views
                    and goal_image_rgb is not None
                ):
                    panorama_query_trace: dict[str, Any] = {}
                    panel_count = max(
                        2, int(args.image_vlm_panoramic_query_panels)
                    )
                    # A preempted sweep may resume elsewhere: keep views from this pose
                    panorama_query_origin = np.asarray(
                        loop_agent_position, dtype=np.float32
                    ).reshape(3)
                    panorama_sweep_views = [
                        view
                        for view in panorama_sweep_views
                        if float(
                            np.linalg.norm(
                                (view[2] - panorama_query_origin)[[0, 2]]
                            )
                        )
                        <= float(args.forward_step_size)
                    ]
                    if len(panorama_sweep_views) >= panel_count:
                        stride = len(panorama_sweep_views) / float(panel_count)
                        selected = [
                            panorama_sweep_views[
                                min(
                                    len(panorama_sweep_views) - 1,
                                    int(index * stride),
                                )
                            ]
                            for index in range(panel_count)
                        ]
                        panorama_query_attempts += 1
                        try:
                            with timing.measure("stop_verification", "image_vlm_panorama"):
                                with timing.detail("panel_composition"):
                                    panel_grid_rgb = compose_panel_grid(
                                        [view[1] for view in selected]
                                    )
                                panorama_query_trace = (
                                    query_panoramic_instance_panel(
                                        vlm_config=image_pair_vlm_config,
                                        goal_image_rgb=goal_image_rgb,
                                        panel_grid_rgb=panel_grid_rgb,
                                        panel_count=panel_count,
                                        timing=timing,
                                    )
                                )
                        except Exception as exc:
                            panorama_query_trace = {
                                "attempted": True,
                                "error": str(exc),
                            }
                        panel_index = int(
                            panorama_query_trace.get("panel", 0) or 0
                        )
                        if (
                            1 <= panel_index <= panel_count
                            and isinstance(
                                sgf_explorer, VisualMemoryUtilityExplorer
                            )
                        ):
                            panorama_query_hits += 1
                            positive_origin = np.asarray(
                                panorama_query_origin, dtype=np.float32
                            ).reshape(3)
                            if np.all(np.isfinite(positive_origin)):
                                if not panorama_pose_corroborated:
                                    panorama_pose_corroborated = any(
                                        float(
                                            np.linalg.norm(
                                                (
                                                    positive_origin
                                                    - prior_origin
                                                )[[0, 2]]
                                            )
                                        )
                                        >= float(args.forward_step_size)
                                        for prior_origin in panorama_positive_origins
                                    )
                                panorama_positive_origins.append(
                                    positive_origin.copy()
                                )
                            panorama_query_trace[
                                "pose_corroborated"
                            ] = bool(panorama_pose_corroborated)
                            panorama_query_trace[
                                "origin_separation_required_m"
                            ] = round(float(args.forward_step_size), 4)
                            panorama_query_quality = 1.0
                            vlm_trust = float(
                                (1 + vlm_trust_confirmed)
                                / (2 + vlm_trust_confirmed + vlm_trust_disconfirmed)
                            )
                            panorama_query_trace[
                                "memory_quality"
                            ] = round(float(panorama_query_quality), 4)
                            panorama_query_heading_rad = float(
                                selected[panel_index - 1][0]
                            )
                            panorama_query_trace["heading_deg"] = round(
                                math.degrees(panorama_query_heading_rad), 2
                            )
                            if bool(
                                getattr(args, "image_vlm_foveated_hypothesis", False)
                            ) and hasattr(
                                sgf_explorer, "note_verified_instance_hypothesis"
                            ):
                                # Foveated query on the selected view, lifted
                                # to 3D through its depth and pose
                                foveated_view = selected[panel_index - 1]
                                foveated_trace: dict[str, Any] = {
                                    "attempted": False
                                }
                                foveated_depth = (
                                    foveated_view[3]
                                    if len(foveated_view) > 4
                                    else None
                                )
                                foveated_pose = (
                                    foveated_view[4]
                                    if len(foveated_view) > 4
                                    else None
                                )
                                if (
                                    foveated_depth is not None
                                    and foveated_pose is not None
                                ):
                                    foveated_query_attempts += 1
                                    try:
                                        with timing.measure("stop_verification", "image_vlm_foveated"):
                                            foveated_trace = query_instance_box(
                                                vlm_config=image_pair_vlm_config,
                                                goal_image_rgb=goal_image_rgb,
                                                view_rgb=foveated_view[1],
                                                timing=timing,
                                            )
                                    except Exception as exc:
                                        foveated_trace = {
                                            "attempted": True,
                                            "error": str(exc),
                                        }
                                    if bool(foveated_trace.get("visible")):
                                        foveated_query_visible += 1
                                    foveated_box = foveated_trace.get("box_xyxy")
                                    if (
                                        bool(foveated_trace.get("visible"))
                                        and foveated_box is not None
                                    ):
                                        (
                                            foveated_point,
                                            foveated_lift,
                                        ) = lift_box_to_world(
                                            foveated_box,
                                            depth=foveated_depth,
                                            fx=float(frame_fx),
                                            fy=float(frame_fy),
                                            cx=float(frame_cx),
                                            cy=float(frame_cy),
                                            world_from_camera_cv=foveated_pose,
                                            min_depth_m=float(min_depth),
                                            max_depth_m=float(max_depth),
                                        )
                                        foveated_trace["lift"] = foveated_lift
                                        if foveated_point is not None:
                                            foveated_confidence = (
                                                foveated_trace.get("confidence")
                                            )
                                            foveated_quality = (
                                                1.0
                                                if foveated_confidence is None
                                                else float(foveated_confidence)
                                            )
                                            if bool(getattr(args, "image_vlm_hypothesis_trust", False)):
                                                foveated_quality = float(vlm_trust)
                                            foveated_accepted = bool(
                                                sgf_explorer.note_verified_instance_hypothesis(
                                                    foveated_point,
                                                    quality=max(
                                                        1e-3, foveated_quality
                                                    ),
                                                    step_index=int(step_index),
                                                    agent_position=np.asarray(
                                                        loop_agent_position,
                                                        dtype=np.float32,
                                                    ).reshape(3),
                                                )
                                            )
                                            foveated_trace[
                                                "hypothesis_accepted"
                                            ] = foveated_accepted
                                            if foveated_accepted:
                                                foveated_hypothesis_count += 1
                                                vlm_trust_point = np.asarray(
                                                    foveated_point, dtype=np.float32
                                                ).reshape(3).copy()
                                                vlm_trust_arrival_deadline = None
                                                foveated_trace["trust"] = round(
                                                    float(vlm_trust), 4
                                                )
                                panorama_query_trace["foveated"] = foveated_trace
                            with timing.measure("image_goal", "visual_memory_heading"):
                                panorama_query_trace["memory_updated"] = bool(
                                    sgf_explorer.update_visual_memory_heading(
                                        position=np.asarray(
                                            loop_agent_position, dtype=np.float32
                                        ).reshape(3),
                                        heading_rad=panorama_query_heading_rad,
                                        similarity=float(panorama_query_quality),
                                        step_index=int(step_index),
                                    )
                                )
                    panorama_trace["instance_query"] = panorama_query_trace
                    panorama_sweep_views = []

            visual_pose_effective_stop_trace[
                "candidate_radius_m"
            ] = round(float(visual_stop_candidate_radius_m), 4)

            if habitat_action == int(HabitatSimActions.stop):
                stop_called = True
                terminal_stop_step = int(step_index)
                if metric_visual_closure_commit_forced or not forced_budget_stop:
                    terminal_stop_source = "decision"
                    if last_ledger_record_step == int(step_index):
                        terminal_stop_vantages = int(last_ledger_record_vantages)
                elif endgame_return_stop_forced:
                    terminal_stop_source = "endgame"
                else:
                    terminal_stop_source = "budget"
            if first_target_found_step < 0 and target_found_for_task_mode(
                args.task_mode,
                text_target_found=text_sgf_target_found,
                instance_image_target_found=instance_image_target_found,
            ):
                first_target_found_step = int(step_index)

            visualization_mode = "sgf_scan"
            if metric_visual_closure_commit_forced:
                visualization_mode = "metric_visual_closure"
            elif forced_budget_stop:
                visualization_mode = "budget_stop"
            elif bool(stop_validation_trace.get("attempted")) and not bool(stop_validation_trace.get("approved")):
                visualization_mode = "stop_validation_rejected"
            elif sgf_stop_forced:
                visualization_mode = "sgf_stop"
            elif sgf_depth_guard_forced:
                visualization_mode = "sgf_depth_guard"
            elif visual_pose_navigation_forced:
                visualization_mode = "visual_pose_navigation"
            elif text_target_navigation_forced:
                visualization_mode = "text_sgf_target"
            elif panorama_forced:
                visualization_mode = "panorama_scan"
            elif sgf_explore_forced:
                visualization_mode = "sgf_explore"
            elif stop_suppressed:
                visualization_mode = "stop_suppressed"
            elif habitat_action == int(HabitatSimActions.stop):
                visualization_mode = "navigation_stop"
            action_source_counts[visualization_mode] += 1
            if habitat_action == int(HabitatSimActions.move_forward):
                executed_action_kind = "forward"
            elif habitat_action == int(HabitatSimActions.stop):
                executed_action_kind = "stop"
            else:
                executed_action_kind = "turn"
            action_kind_counts.setdefault(visualization_mode, Counter())[
                executed_action_kind
            ] += 1
            if visualization_mode in {
                "text_sgf_target",
                "sgf_explore",
                "visual_pose_navigation",
                "sgf_depth_guard",
            }:
                if habitat_action == int(HabitatSimActions.move_forward):
                    endgame_nav_forward_actions += 1
                elif habitat_action in {
                    int(HabitatSimActions.turn_left),
                    int(HabitatSimActions.turn_right),
                }:
                    endgame_nav_turn_actions += 1

            with timing.measure("diagnostics"):
                visualization_current_goal = visual_pose_goal_position
                visualization_goal_label = "visual_pose_goal"
                if visualization_current_goal is None and text_target_goal_position is not None:
                    visualization_current_goal = text_target_goal_position
                    visualization_goal_label = str(
                        "text_sgf_target"
                        if text_selected_target_node is None
                        else text_selected_target_node.get("label", "text_sgf_target")
                    )
                if visualization_current_goal is None and sgf_plan_goal is not None:
                    visualization_current_goal = sgf_plan_goal
                    visualization_goal_label = "frontier"
                visualization_selected_target: dict[str, Any] = {}
                if visualization_current_goal is not None:
                    visualization_selected_target = {
                        "label": visualization_goal_label,
                        "distance_m": rounded_planar_distance_xz(
                            env.get_agent_state().position,
                            visualization_current_goal,
                        ),
                        "goal_position_m": world_position_list(visualization_current_goal),
                    }
                suppressed_reasons: list[str] = []
                if stop_suppressed:
                    suppressed_reasons.append("stop_rejected")
                if bool(stop_validation_trace.get("attempted")) and not bool(stop_validation_trace.get("approved")):
                    suppressed_reasons.append("stop_validation_rejected")
                planner_record = sgf_explore_trace.get("planner")
                if not isinstance(planner_record, dict):
                    planner_record = {}
                visualization_step_record = {
                    "planner": planner_record,
                    "selected_target": visualization_selected_target,
                    "stop": {
                        "executed_stop": bool(habitat_action == int(HabitatSimActions.stop)),
                        "suppressed_reasons": suppressed_reasons,
                    },
                    "map": {"frontier_cells": 0},
                    "live_metrics": {},
                }
                last_visualization_mode = visualization_mode
                last_visualization_action = int(habitat_action)
                last_visualization_step_record = visualization_step_record
                # SGF map, goals and path must share the observation pose frame
                visualization_agent_state = env.get_agent_state()
                if visualizer is not None:
                    visualizer.add_frame(
                        episode_info=episode_info,
                        rgb_bgr=bgr_from_rgb_observation(rgb_observation),
                        agent_state=visualization_agent_state,
                        nodes=[],
                        canonical_goal=canonical_text_goal if args.task_mode == "text" else str(episode_info.goal_name or "target"),
                        goal_queries=goal_match_phrases or [str(episode_info.goal_name or "target")],
                        target_node=text_selected_target_node,
                        current_goal=visualization_current_goal,
                        step_index=step_index,
                        mode=visualization_mode,
                        action=habitat_action,
                        runtime_stats={},
                        stop_called=stop_called,
                        target_found=target_found_for_task_mode(
                            args.task_mode,
                            text_target_found=text_sgf_target_found,
                            instance_image_target_found=instance_image_target_found,
                        ),
                        navigation_map=sgf_navigation_map,
                        step_record=visualization_step_record,
                        rejected_positions=excluded_text_positions,
                        task_mode=args.task_mode,
                    )
                if image_dumper is not None:
                    image_dumper.add_frame(
                        episode_info=episode_info,
                        rgb_bgr=bgr_from_rgb_observation(rgb_observation),
                        agent_state=visualization_agent_state,
                        nodes=[],
                        canonical_goal=canonical_text_goal if args.task_mode == "text" else str(episode_info.goal_name or "target"),
                        goal_queries=goal_match_phrases or [str(episode_info.goal_name or "target")],
                        target_node=text_selected_target_node,
                        current_goal=visualization_current_goal,
                        step_index=step_index,
                        mode=visualization_mode,
                        action=habitat_action,
                        runtime_stats={},
                        stop_called=stop_called,
                        target_found=target_found_for_task_mode(
                            args.task_mode,
                            text_target_found=text_sgf_target_found,
                            instance_image_target_found=instance_image_target_found,
                        ),
                        navigation_map=sgf_navigation_map,
                        step_record=visualization_step_record,
                        rejected_positions=excluded_text_positions,
                        task_mode=args.task_mode,
                    )

            safety_vetoed_forward = bool(
                navigation_guard_trace.get("blocked", False)
                and int(navigation_guard_trace.get("vetoed_action", -1))
                == int(HabitatSimActions.move_forward)
            )
            text_target_proposal_executes = False
            sgf_proposal_executes = False
            endgame_proposal_executes = False
            sgf_goal_service_token: int | None = None
            text_target_owns_pending = bool(
                text_target_plan_navigator is not None
                and text_target_plan_navigator.has_pending_action
                and text_target_plan_command is not None
            )
            if text_target_owns_pending:
                text_target_proposal_executes = bool(
                    int(habitat_action) == int(text_target_plan_command.action)
                )
                target_transaction_trace = text_sgf_trace.setdefault(
                    "target_transaction", {}
                )
                target_transaction_trace.update(
                    {
                        "final_action": int(habitat_action),
                        "executed": bool(text_target_proposal_executes),
                    }
                )
                if not text_target_proposal_executes and (
                    safety_vetoed_forward
                    and int(text_target_plan_command.action)
                    == int(HabitatSimActions.move_forward)
                    and text_navigation_map is not None
                ):
                    target_transaction_trace["blocked_forward_recorded"] = bool(
                        text_target_plan_navigator.record_blocked_forward(
                            agent_state=env.get_agent_state(),
                            navigation_map=text_navigation_map,
                        )
                    )
                    target_transaction_trace["resolution"] = "safety_veto"
                elif not text_target_proposal_executes:
                    text_target_plan_navigator.cancel_pending_action()
                    target_transaction_trace["resolution"] = "preempted"
                else:
                    target_transaction_trace["resolution"] = "awaiting_commit"
            if (
                sgf_navigator is not None
                and sgf_navigator.has_pending_action
                and sgf_plan_command is not None
                and not (
                    text_target_owns_pending
                    and text_target_plan_navigator is sgf_navigator
                )
            ):
                sgf_proposal_executes = bool(
                    (
                        sgf_explore_forced
                        or sgf_proposal_passthrough_forced
                    )
                    and int(habitat_action) == int(sgf_plan_command.action)
                )
                if not sgf_proposal_executes and (
                    safety_vetoed_forward
                    and int(sgf_plan_command.action)
                    == int(HabitatSimActions.move_forward)
                    and sgf_navigation_map is not None
                ):
                    sgf_navigator.record_blocked_forward(
                        agent_state=env.get_agent_state(),
                        navigation_map=sgf_navigation_map,
                    )
                elif not sgf_proposal_executes:
                    sgf_navigator.cancel_pending_action()
                if (
                    sgf_proposal_executes
                    and isinstance(sgf_explorer, VisualMemoryUtilityExplorer)
                    and hasattr(sgf_explorer, "goal_epoch_service_token")
                ):
                    sgf_goal_service_token = (
                        sgf_explorer.goal_epoch_service_token
                    )
            if (
                endgame_return_navigator is not None
                and endgame_return_navigator.has_pending_action
                and endgame_pending_command is not None
            ):
                endgame_proposal_executes = bool(
                    endgame_navigator_action_selected
                    and int(habitat_action)
                    == int(endgame_pending_command.action)
                )
                if not endgame_proposal_executes and (
                    safety_vetoed_forward
                    and int(endgame_pending_command.action)
                    == int(HabitatSimActions.move_forward)
                    and endgame_navigation_map is not None
                ):
                    endgame_return_navigator.record_blocked_forward(
                        agent_state=env.get_agent_state(),
                        navigation_map=endgame_navigation_map,
                    )
                elif not endgame_proposal_executes:
                    endgame_return_navigator.cancel_pending_action()
            before_navigation_agent_state = env.get_agent_state()
            before_navigation_position = world_position_list(
                before_navigation_agent_state.position
            )
            before_agent_state = (
                diagnostic_agent_state(env)
                if diagnostic_state_enabled
                else before_navigation_agent_state
            )
            before_position = world_position_list(before_agent_state.position)
            before_yaw_deg = agent_yaw_deg(before_agent_state)
            with timing.measure("environment_step"):
                observations, done, metrics = env.step(habitat_action)
            if recovery_approval_snapshot_candidate is not None:
                recovery_primitive_executed = bool(
                    recovery_approval_proposed_action is not None
                    and int(habitat_action)
                    == int(recovery_approval_proposed_action)
                )
                recovery_bound_trace = (
                    stop_validation_trace.get("approach", {}).get(
                        "recovery_bound"
                    )
                    if isinstance(stop_validation_trace.get("approach"), dict)
                    else None
                )
                probe_recovery_trace = stop_validation_trace.get(
                    "probe_recovery"
                )
                if recovery_primitive_executed:
                    recovery_approval_snapshot = replace(
                        recovery_approval_snapshot_candidate,
                        recovery_executed=True,
                    )
                    if isinstance(recovery_bound_trace, dict):
                        recovery_bound_trace.update(
                            {
                                "executed": True,
                                "executed_action": int(habitat_action),
                                "action_count_after": int(
                                    text_stop_recovery_action_count
                                ),
                            }
                        )
                    if isinstance(probe_recovery_trace, dict):
                        probe_recovery_trace.update(
                            {
                                "action_executed": int(habitat_action),
                                "action_count_after": int(
                                    text_stop_recovery_action_count
                                ),
                                "continuation_executed": bool(
                                    probe_recovery_continuation_executed
                                ),
                            }
                        )
                else:
                    text_stop_approach_commit_node_id = -1
                    text_stop_approach_commit_probe_owned = False
                    text_stop_approach_commit_step = -(10**9)
                    text_stop_approach_commit_source = ""
                    text_stop_contact_press_steps = 0
                    text_stop_recovery_action_count = 0
                    text_stop_recovery_first_action = None
                    recovery_approval_snapshot = None
                    probe_recovery_continuation_executed = False
                    if isinstance(recovery_bound_trace, dict):
                        recovery_bound_trace.update(
                            {
                                "executed": False,
                                "preempted": True,
                                "preempted_action": int(habitat_action),
                                "action_count_after": 0,
                            }
                        )
                    if isinstance(probe_recovery_trace, dict):
                        probe_recovery_trace.update(
                            {
                                "action_executed": None,
                                "action_count_after": 0,
                                "continuation_executed": False,
                                "transaction_preempted": True,
                                "preempted_action": int(habitat_action),
                            }
                        )
            after_navigation_agent_state = env.get_agent_state()
            text_target_proposal_committed = False
            sgf_proposal_committed = False
            # Resolve executed proposals at the action boundary
            if (
                text_target_proposal_executes
                and text_target_plan_navigator is not None
                and text_target_plan_navigator.has_pending_action
                and text_navigation_map is not None
            ):
                text_target_proposal_committed = (
                    text_target_plan_navigator.commit_pending_action(
                        result_agent_state=after_navigation_agent_state,
                        navigation_map=text_navigation_map,
                    )
                )
                text_sgf_trace.setdefault("target_transaction", {}).update(
                    {
                        "committed": bool(text_target_proposal_committed),
                        "resolution": (
                            "committed"
                            if text_target_proposal_committed
                            else "commit_missing"
                        ),
                    }
                )
            if (
                sgf_proposal_executes
                and sgf_navigator is not None
                and sgf_navigator.has_pending_action
                and sgf_navigation_map is not None
                and not (
                    text_target_owns_pending
                    and text_target_plan_navigator is sgf_navigator
                )
            ):
                sgf_proposal_committed = sgf_navigator.commit_pending_action(
                    result_agent_state=after_navigation_agent_state,
                    navigation_map=sgf_navigation_map,
                )
                if (
                    sgf_proposal_committed
                    and isinstance(sgf_explorer, VisualMemoryUtilityExplorer)
                    and hasattr(
                        sgf_explorer,
                        "acknowledge_executed_sgf_proposal",
                    )
                ):
                    sgf_explorer.acknowledge_executed_sgf_proposal(
                        sgf_goal_service_token,
                        int(step_index),
                    )
            if (
                endgame_proposal_executes
                and endgame_return_navigator is not None
                and endgame_return_navigator.has_pending_action
                and endgame_navigation_map is not None
            ):
                endgame_return_navigator.commit_pending_action(
                    result_agent_state=after_navigation_agent_state,
                    navigation_map=endgame_navigation_map,
                )
            after_navigation_position = world_position_list(
                after_navigation_agent_state.position
            )
            after_agent_state = (
                diagnostic_agent_state(env)
                if diagnostic_state_enabled
                else after_navigation_agent_state
            )
            after_position = world_position_list(after_agent_state.position)
            after_yaw_deg = agent_yaw_deg(after_agent_state)
            real_step_diagnostics = {}
            if real_env:
                try:
                    real_step_diagnostics = dict(getattr(env, "last_step_recovery", {}))
                except Exception:
                    real_step_diagnostics = {}
                if "tracking_state" not in real_step_diagnostics:
                    tracking_state = getattr(env, "last_tracking_state", "")
                    if tracking_state:
                        real_step_diagnostics["tracking_state"] = str(tracking_state)
            after_gt_goal_viewpoint_planar_distance_m = (
                rounded_planar_distance_xz(
                    after_position,
                    gt_goal_viewpoint,
                )
                if diagnostic_state_enabled
                else None
            )
            step_displacement_m = 0.0
            if (
                before_navigation_position is not None
                and after_navigation_position is not None
            ):
                step_displacement_m = float(
                    np.hypot(
                        float(after_navigation_position[0])
                        - float(before_navigation_position[0]),
                        float(after_navigation_position[2])
                        - float(before_navigation_position[2]),
                    )
                )
            forward_blocked_this_step = bool(
                habitat_action == int(HabitatSimActions.move_forward)
                and before_navigation_position is not None
                and after_navigation_position is not None
                and step_displacement_m
                < float(scales.blocked_displacement_m)
            )
            if (
                habitat_action == int(HabitatSimActions.move_forward)
                and before_navigation_position is not None
                and after_navigation_position is not None
            ):
                if forward_blocked_this_step:
                    blocked_forward_steps += 1
                else:
                    blocked_forward_steps = 0
                    sgf_depth_guard_block_goal = None
                    sgf_depth_guard_block_token = None
                    sgf_depth_guard_block_count = 0
            elif habitat_action != int(HabitatSimActions.stop):
                blocked_forward_steps = 0
            servo_forward_blocked = bool(
                visual_pose_navigation_forced
                and habitat_action == int(HabitatSimActions.move_forward)
                and forward_blocked_this_step
            )
            if (
                visual_pursuit_navigator is not None
                and visual_pursuit_navigator.has_pending_action
            ):
                pursuit_executed = bool(
                    visual_pursuit_command is not None
                    and visual_pose_navigation_forced
                    and int(habitat_action) == int(visual_pursuit_command.action)
                )
                if (
                    pursuit_executed
                    and forward_blocked_this_step
                    and sgf_navigation_map is not None
                ):
                    visual_pursuit_navigator.record_blocked_forward(
                        agent_state=env.get_agent_state(),
                        navigation_map=sgf_navigation_map,
                    )
                elif not pursuit_executed:
                    visual_pursuit_navigator.cancel_pending_action()
            (
                sgf_blocked_forward_goal,
                sgf_blocked_forward_token,
                sgf_blocked_forward_count,
            ) = advance_sgf_blocked_goal_provenance(
                sgf_blocked_forward_goal,
                sgf_blocked_forward_token,
                sgf_blocked_forward_count,
                forward_blocked=forward_blocked_this_step,
                sgf_owned=bool(sgf_proposal_executes),
                current_goal=sgf_plan_goal,
                current_token=sgf_goal_service_token,
            )
            steps = step_index + 1
            with timing.measure("diagnostics"):
                action_diagnostic = f"navigation_action={int(navigation_action)} "
                diagnostic_line = (
                    f"step={step_index:03d} "
                    f"current_position_m={world_position_list(diagnostic_loop_agent_position)} "
                    f"current_yaw_deg={before_yaw_deg} "
                    f"gt_goal_viewpoint_planar_distance_m={gt_goal_viewpoint_planar_distance_m} "
                    f"after_position_m={after_position} "
                    f"after_yaw_deg={after_yaw_deg} "
                    f"after_gt_goal_viewpoint_planar_distance_m={after_gt_goal_viewpoint_planar_distance_m} "
                    f"{action_diagnostic}habitat_action={int(habitat_action)}"
                )
                if real_step_diagnostics:
                    diagnostic_line += (
                        " real_step="
                        + json.dumps(real_step_diagnostics, sort_keys=True, separators=(",", ":"))
                    )
                if diagnostic_log_handle is not None:
                    diagnostic_log_handle.write(diagnostic_line + "\n")
                    diagnostic_log_handle.flush()
                if bool(args.verbose):
                    print(diagnostic_line, flush=True)
                if isinstance(sgf_explorer, VisualMemoryUtilityExplorer):
                    sgf_visual_memory_trace["goal_lifecycle"] = (
                        sgf_explorer._fair_scheduler_trace(
                            drain_goal_events=True,
                        )
                    )
                trace.append(
                    {
                        "step": step_index,
                        "nav_state": visualization_mode,
                        "action_source": visualization_mode,
                        "navigation_action": navigation_action,
                        "habitat_action": habitat_action,
                        "gt_goal_viewpoint_m": world_position_list(gt_goal_viewpoint),
                        "gt_goal_viewpoint_planar_distance_m": gt_goal_viewpoint_planar_distance_m,
                        "before_position_m": before_position,
                        "after_position_m": after_position,
                        "navigation_before_position_m": before_navigation_position,
                        "navigation_after_position_m": after_navigation_position,
                        "navigation_before_yaw_deg": agent_yaw_deg(
                            before_navigation_agent_state
                        ),
                        "navigation_after_yaw_deg": agent_yaw_deg(
                            after_navigation_agent_state
                        ),
                        "before_yaw_deg": before_yaw_deg,
                        "after_yaw_deg": after_yaw_deg,
                        "real_step": real_step_diagnostics,
                        "after_gt_goal_viewpoint_planar_distance_m": after_gt_goal_viewpoint_planar_distance_m,
                        "visual_stop": visual_stop_trace,
                        "image_match_visualization": image_match_visualization_trace,
                        "visual_pose": visual_pose_trace,
                        "image_vlm_stop_validation": (
                            image_vlm_stop_validation_trace
                        ),
                        "visual_pose_stop_distance": visual_pose_stop_distance_trace,
                        "visual_pose_effective_stop_distance": visual_pose_effective_stop_trace,
                        "visual_pose_navigation": visual_pose_navigation_trace,
                        "visual_pose_navigation_forced": bool(visual_pose_navigation_forced),
                        "goal_view": goal_view_trace,
                        "text_sgf": text_sgf_trace,
                        "text_target_navigation_forced": bool(
                            text_target_navigation_forced
                        ),
                        "sgf_explore": sgf_explore_trace,
                        "panorama_observation": panorama_trace,
                        "sgf_visual_memory": sgf_visual_memory_trace,
                        "sgf_input_dump": sgf_input_dump_trace,
                        "sgf_input_replay": replay_trace,
                        "sgf_explore_forced": bool(sgf_explore_forced),
                        "sgf_proposal_passthrough_forced": bool(
                            sgf_proposal_passthrough_forced
                        ),
                        "sgf_stop": sgf_stop_trace,
                        "sgf_stop_forced": bool(sgf_stop_forced),
                        "stop_validation": stop_validation_trace,
                        "metric_visual_closure": metric_visual_closure_trace,
                        "metric_visual_closure_commit_forced": bool(
                            metric_visual_closure_commit_forced
                        ),
                        "navigation_guard": navigation_guard_trace,
                        "sgf_depth_guard_forced": bool(sgf_depth_guard_forced),
                        "stop_suppressed": bool(stop_suppressed),
                        "blocked_forward_steps": int(blocked_forward_steps),
                        "forced_budget_stop": bool(forced_budget_stop),
                        "endgame_return_stop_forced": bool(endgame_return_stop_forced),
                    }
                )
                if step_endgame_return_trace:
                    trace[-1]["endgame_return"] = step_endgame_return_trace
                if metrics_only:
                    trace.clear()
            timing.finish_step()
            if done:
                final_rgb = np.asarray(observations["rgb"], dtype=np.uint8)[:, :, :3]
                break

        runtime_timing = timing.finish_navigation(steps)

        final_navigation_agent_state = env.get_agent_state()
        final_navigation_position_array = np.asarray(
            final_navigation_agent_state.position, dtype=np.float32
        ).reshape(3)
        if not path_positions or float(
            np.linalg.norm(path_positions[-1] - final_navigation_position_array)
        ) > 1e-4:
            path_positions.append(final_navigation_position_array.copy())
        final_agent_state = (
            diagnostic_agent_state(env)
            if diagnostic_state_enabled
            else final_navigation_agent_state
        )
        if final_rgb is None:
            final_rgb = np.asarray(observations["rgb"], dtype=np.uint8)[:, :, :3]
        final_visualization_goal = text_target_goal_position
        if final_visualization_goal is None:
            final_visualization_goal = visual_pose_goal_position
        if final_visualization_goal is None and sgf_explorer is not None:
            final_visualization_goal = sgf_explorer.current_goal
        if visualizer is not None:
            visualizer.add_frame(
                episode_info=episode_info,
                rgb_bgr=bgr_from_rgb_observation(final_rgb),
                agent_state=final_navigation_agent_state,
                nodes=[],
                canonical_goal=canonical_text_goal if args.task_mode == "text" else str(episode_info.goal_name or "target"),
                goal_queries=goal_match_phrases or [str(episode_info.goal_name or "target")],
                target_node=text_selected_target_node,
                current_goal=final_visualization_goal,
                step_index=steps,
                mode="final",
                action=last_visualization_action,
                runtime_stats={},
                stop_called=stop_called,
                target_found=target_found_for_task_mode(
                    args.task_mode,
                    text_target_found=text_sgf_target_found,
                    instance_image_target_found=instance_image_target_found,
                ),
                navigation_map=sgf_navigation_map,
                step_record=last_visualization_step_record,
                rejected_positions=excluded_text_positions,
                task_mode=args.task_mode,
            )
        if image_dumper is not None:
            image_dumper.add_frame(
                episode_info=episode_info,
                rgb_bgr=bgr_from_rgb_observation(final_rgb),
                agent_state=final_navigation_agent_state,
                nodes=[],
                canonical_goal=canonical_text_goal if args.task_mode == "text" else str(episode_info.goal_name or "target"),
                goal_queries=goal_match_phrases or [str(episode_info.goal_name or "target")],
                target_node=text_selected_target_node,
                current_goal=final_visualization_goal,
                step_index=steps,
                mode="final",
                action=last_visualization_action,
                runtime_stats={},
                stop_called=stop_called,
                target_found=target_found_for_task_mode(
                    args.task_mode,
                    text_target_found=text_sgf_target_found,
                    instance_image_target_found=instance_image_target_found,
                ),
                navigation_map=sgf_navigation_map,
                step_record=last_visualization_step_record,
                rejected_positions=excluded_text_positions,
                task_mode=args.task_mode,
            )
        final_image_path = ""
        if not metrics_only:
            final_image_path = save_rgb(artifact_dir / "final_rgb.png", final_rgb)
            final_navigation_map_path.parent.mkdir(parents=True, exist_ok=True)
            final_navigation_canvas = render_episode_canvas(
                episode_info=episode_info,
                rgb_bgr=bgr_from_rgb_observation(final_rgb),
                agent_state=final_navigation_agent_state,
                path_history=path_positions,
                nodes=[],
                canonical_goal=(
                    canonical_text_goal
                    if args.task_mode == "text"
                    else str(episode_info.goal_name or "target")
                ),
                goal_queries=goal_match_phrases or [str(episode_info.goal_name or "target")],
                target_node=text_selected_target_node,
                current_goal=final_visualization_goal,
                step_index=max(0, steps - 1),
                mode=last_visualization_mode,
                action=last_visualization_action,
                runtime_stats={},
                stop_called=stop_called,
                target_found=target_found_for_task_mode(
                    args.task_mode,
                    text_target_found=text_sgf_target_found,
                    instance_image_target_found=instance_image_target_found,
                ),
                navigation_map=sgf_navigation_map,
                step_record=last_visualization_step_record,
                rejected_positions=excluded_text_positions,
                task_mode=args.task_mode,
            )
            if not cv2.imwrite(str(final_navigation_map_path), final_navigation_canvas):
                raise RuntimeError(
                    f"failed to write final navigation map visualization {final_navigation_map_path}"
                )
        final_image_match_path = ""
        if best_image_match_visualization is not None:
            final_match_visualization = best_image_match_visualization
            final_match_current_rgb = best_image_match_current_rgb
            final_match_diagnostics = best_image_match_diagnostics
            final_match_step = best_image_match_step
            final_match_target_node = best_image_match_target_node
        else:
            final_match_visualization = last_image_match_visualization
            final_match_current_rgb = last_image_match_current_rgb
            final_match_diagnostics = last_image_match_diagnostics
            final_match_step = last_image_match_step
            final_match_target_node = last_image_match_target_node
        if (
            not metrics_only
            and final_image_match_requested
            and goal_image_rgb is not None
            and final_match_visualization is not None
            and final_match_current_rgb is not None
            and final_match_diagnostics is not None
        ):
            final_image_match_path = str(configured_final_image_match_path)
            save_image_match_visualization(
                output_path=final_image_match_path,
                goal_image_rgb=goal_image_rgb,
                current_image_rgb=final_match_current_rgb,
                match_debug=final_match_visualization,
                diagnostics=final_match_diagnostics,
                step_index=final_match_step,
                target_node=final_match_target_node,
                max_lines=int(args.image_match_visual_max_lines),
            )
        image_match_video_result_path = (
            str(configured_image_match_video_path)
            if not metrics_only
            and image_match_video_writer is not None
            and image_match_video_writer.frame_count > 0
            else ""
        )
        terminal_position_m = (
            world_position_list(final_agent_state.position)
            if diagnostic_state_enabled
            else None
        )
        terminal_navigation_position_m = world_position_list(
            final_navigation_agent_state.position
        )
        endgame_summary = (
            {}
            if endgame_return_planner is None
            else endgame_return_planner.summary()
        )
        final_visual_memory_stats: dict[str, Any] = {}
        if isinstance(sgf_explorer, VisualMemoryUtilityExplorer):
            final_visual_memory_stats = sgf_explorer.visual_memory_stats()
        ledger_observations = (
            {}
            if text_stop_ledger is None
            else getattr(text_stop_ledger, "_observations", {})
        )
        text_stop_ledger_observation_count = (
            sum(len(items) for items in ledger_observations.values())
            if isinstance(ledger_observations, dict)
            else 0
        )
        exploration_objective: dict[str, Any] = (
            sgf_explorer.objective_telemetry()
            if isinstance(sgf_explorer, UtilityExplorer)
            else {}
        )
        exploration_selection_counts = dict(
            exploration_objective.get("selection_counts", {}) or {}
        )
        feature_activity = {
            "llm_goal_parser": int(
                parsed_text_goal_trace.get("parser_mode") == "llm"
            ),
            "first_sentence_query": int(
                parsed_text_goal_trace.get("parser_mode") == "first_sentence"
            ),
            "parsed_target_phrases": int(
                len(parsed_text_goal_trace.get("parsed_target_phrases", []))
            ),
            "context_guided_exploration": int(
                len(parsed_text_goal_trace.get("context_phrases", []))
            ),
            "text_sgf_stop": int(action_source_counts["sgf_stop"]),
            "sgf_exploration": int(action_source_counts["sgf_explore"]),
            "text_sequential_evidence_stop": int(
                text_sequential_evidence_commit_count
            ),
            "sgf_explore_evidence_verification": int(
                0
                if sgf_explorer is None
                else getattr(sgf_explorer, "evidence_goal_epochs", 0)
            ),
            "stop_validation": int(stop_validation_attempts),
            "metric_visual_closure": int(
                metric_visual_closure_armed_count
                + metric_visual_closure_recovery_monotonic_commit_count
                + metric_visual_closure_contextual_closeup_commit_count
            ),
            "image_vlm_stop_validation": int(
                image_vlm_stop_validation_attempts
            ),
            "detection_probe": int(detection_probe_counter),
            "text_node_grounded_evidence": int(
                text_grounded_evidence_query_count
                + text_projection_evidence_query_count
            ),
            "text_hypothesis_triage": int(text_triage_queries),
            "validated_stop_approach": int(
                validated_text_stop_approach_forces
            ),
            "stop_ledger": int(text_stop_ledger_observation_count),
            "target_fatigue": int(text_target_fatigue_retirements),
            "endgame_return": int(
                endgame_summary.get("activated_step", -1) >= 0
            ),
            "visual_memory": int(
                final_visual_memory_stats.get("insertions", 0)
                + final_visual_memory_stats.get("merges", 0)
            ),
            "visual_pose_servo": int(visual_pose_goal_update_count),
            "panoramic_observation": int(panorama_action_count),
            "panoramic_detector_scaled_interval": int(
                panoramic_detector_scaled_interval_application_count
            ),
            "panoramic_searched_extent_interval": int(
                panoramic_searched_extent_interval_application_count
            ),
            "image_vlm_panoramic_query": int(panorama_query_attempts),
            "image_vlm_foveated_query": int(foveated_query_attempts),
            "image_vlm_foveated_hypothesis": int(foveated_hypothesis_count),
            "image_vlm_hypothesis_arrival_sweep": int(
                hypothesis_arrival_sweep_count
            ),
            "image_reachable_pursuit": int(reachable_pursuit_action_count),
            "image_goal_view_terminal": int(goal_view_active_step_count),
            "image_object_target_steps": int(image_object_target_steps),
            "image_goal_camera_target_steps": int(image_goal_camera_target_steps),
            "image_pursuit_fmm_actions": int(image_pursuit_fmm_actions),
            "image_straight_servo_actions": int(image_straight_servo_actions),
            "image_depth_cluster_target": int(image_depth_cluster_target_updates),
            "image_goal_view_alignment_turns": int(
                0 if goal_view_belief is None else goal_view_belief.alignment_turns
            ),
            "image_goal_view_wide_band_commit": int(
                0
                if goal_view_belief is None
                else goal_view_belief.wide_band_commits
            ),
            "image_goal_view_single_corroboration_commit": int(
                image_single_corroboration_commits
            ),
            "sgf_explore_rate_selection": int(
                exploration_selection_counts.get("rate", 0)
            ),
            "sgf_explore_difference_selection": int(
                exploration_selection_counts.get("difference", 0)
            ),
            "sgf_explore_nearest_selection": int(
                exploration_selection_counts.get("nearest", 0)
            ),
            "sgf_explore_rotation_priced_selection": int(
                exploration_objective.get("rotation_priced_decisions", 0)
            ),
            "text_stop_object_top_detection": int(text_top_detection_queries),
            "text_stop_object_whole_frame": int(text_whole_frame_queries),
            "text_evidence_rule_override": int(text_evidence_rule_overrides),
            "text_reobservation_maneuver": int(text_reobservation_launches),
            "text_withheld_press": int(text_withheld_presses),
            "text_face_node_turn": int(text_face_node_turns),
            "text_passive_release": int(text_passive_releases),
            "text_same_vantage_skip": int(text_same_vantage_skips),
            "text_same_pose_requery": int(text_same_pose_requeries),
            "navigation_graph_occupancy": int(navigation_graph_occupancy_steps),
            "navigation_floor_support": int(navigation_floor_support_steps),
            "navigation_floor_support_veto_cells": int(
                navigation_floor_support_veto_cells
            ),
            "navigation_floor_support_potential_veto_cells": int(
                navigation_floor_support_potential_veto_cells
            ),
            "navigation_floor_support_nonfloor_cells": int(
                navigation_floor_support_nonfloor_cells
            ),
            "text_query_conditional_retrieval": int(
                text_query_conditional_retrieval_steps
            ),
            "text_argmax_label_retrieval": int(text_argmax_label_retrieval_steps),
            "text_detection_instances": int(text_detection_instance_observations),
            "text_detection_instance_candidates": int(
                text_detection_instance_candidate_count
            ),
            "text_detection_instance_selected_steps": int(
                text_detection_instance_selected_steps
            ),
            "text_evidence_weighted_ranking": int(
                text_evidence_weighted_ranking_steps
            ),
            "text_navigation_map_association": int(
                text_navigation_map_association_steps
            ),
            "text_euclidean_association": int(text_euclidean_association_steps),
            "text_reachable_euclidean_association": int(text_reachable_euclidean_association_steps),
            **dict(representation_activity),
            "decision_stop": int(terminal_stop_source == "decision"),
            "target_pursuit_actions": int(
                action_source_counts["text_sgf_target"]
                + action_source_counts["visual_pose_navigation"]
            ),
        }
        ablation_telemetry = {
            **({"runtime_timing": runtime_timing} if runtime_timing is not None else {}),
            "action_source_counts": {
                key: int(value)
                for key, value in sorted(action_source_counts.items())
            },
            "feature_activity": feature_activity,
            "exploration": {
                **exploration_objective,
                "planning_errors": int(sgf_explore_planning_errors),
                "goal_epoch_owners": (
                    {
                        key: int(value)
                        for key, value in sorted(
                            sgf_explorer.goal_epoch_owner_counts.items()
                        )
                    }
                    if isinstance(sgf_explorer, VisualMemoryUtilityExplorer)
                    else {}
                ),
                "hypothesis_goal_epochs": int(
                    0
                    if sgf_explorer is None
                    else getattr(sgf_explorer, "evidence_goal_epochs", 0)
                ),
                "path_length_m": round(planar_path_length_m(path_positions), 4),
            },
            "termination": {
                "stop_source": str(terminal_stop_source),
                "endgame_stop_kind": str(endgame_stop_kind),
                "stop_step": int(terminal_stop_step),
                "stop_vantages": int(terminal_stop_vantages),
            },
            "instance_image": (
                {}
                if args.task_mode != "instance-image"
                else {
                    "target_localization": str(image_target_localization),
                    "navigation_target": str(image_navigation_target),
                    "pursuit_controller": str(image_pursuit_controller),
                    "goal_view_alignment": str(image_goal_view_alignment),
                    "pnp_goal_updates": int(visual_pose_goal_update_count),
                    "depth_cluster_target_updates": int(
                        image_depth_cluster_target_updates
                    ),
                    "depth_cluster_alignment_turns": int(
                        image_depth_cluster_alignment_turns
                    ),
                    "object_anchor_missing_steps": int(
                        image_object_anchor_missing_steps
                    ),
                    "straight_servo_actions": int(image_straight_servo_actions),
                    "fmm_pursuit_steps": int(image_pursuit_fmm_actions),
                    "reachable_pursuit_steps": int(reachable_pursuit_action_count),
                    "goal_view": (
                        {}
                        if goal_view_belief is None
                        else goal_view_belief.summary()
                    ),
                }
            ),
            "stop_ledger": (
                {}
                if text_stop_ledger is None
                else {
                    "verification_object": str(text_verification_object),
                    "evidence_rule": str(text_evidence_rule),
                    "reobservation": str(text_reobservation_mode),
                    "hypotheses": int(text_stop_ledger.hypothesis_count()),
                    "verdicts": int(text_stop_ledger_observation_count),
                    "rule_overrides": int(text_evidence_rule_overrides),
                    "reobservation_launches": int(text_reobservation_launches),
                    "reobservation_retirements": int(
                        text_reobservation_retirements
                    ),
                    "passive_releases": int(text_passive_releases),
                    "withheld_presses": int(text_withheld_presses),
                    "face_node_turns": int(text_face_node_turns),
                    "same_vantage_skips": int(text_same_vantage_skips),
                    "same_pose_requeries": int(text_same_pose_requeries),
                    "readmissions": int(text_readmissions),
                }
            ),
            "shadow_stop": (
                {} if text_shadow_stop is None else text_shadow_stop.summary()
            ),
            "actions": {
                "by_source": {
                    source: {
                        kind: int(count)
                        for kind, count in sorted(counts.items())
                    }
                    for source, counts in sorted(action_kind_counts.items())
                },
                "acquisition_step": int(first_target_found_step),
            },
            "target_tracking": {
                "target_reacquire_count": int(target_reacquire_count),
            },
            "panoramic_observation": {
                "panorama_count": int(panorama_count),
                "action_count": int(panorama_action_count),
                "positive_origin_recorded": bool(
                    panorama_positive_origins
                ),
                "positive_origin_count": int(
                    len(panorama_positive_origins)
                ),
                "pose_corroborated": bool(
                    panorama_pose_corroborated
                ),
                "origin_separation_required_m": float(
                    args.forward_step_size
                ),
            },
            "validation": {
                "stop_validation_attempts": int(stop_validation_attempts),
                "stop_validation_rejections": int(stop_validation_rejections),
                "detection_probe_count": int(detection_probe_counter),
                "hypothesis_projection_step_count": int(
                    text_hypothesis_projection_count
                ),
                "grounded_detection_evidence_queries": int(
                    text_grounded_evidence_query_count
                ),
                "projected_hypothesis_evidence_queries": int(
                    text_projection_evidence_query_count
                ),
                "triage_queries": int(text_triage_queries),
                "triage_exclusions": int(text_triage_exclusions),
                "sequential_evidence_commit_count": int(
                    text_sequential_evidence_commit_count
                ),
                "validated_stop_approach_action_count": int(
                    validated_text_stop_approach_forces
                ),
                "metric_visual_closure_armed_count": int(
                    metric_visual_closure_armed_count
                ),
                "metric_visual_closure_commit_count": int(
                    metric_visual_closure_commit_count
                ),
                "metric_visual_closure_withheld_count": int(
                    metric_visual_closure_withheld_count
                ),
                "evidence_ranking_overrides": int(
                    text_evidence_ranking_overrides
                ),
                "direct_stop_withheld_count": int(
                    direct_stop_withheld_count
                ),
                "metric_visual_closure_recovery_monotonic_commit_count": int(
                    metric_visual_closure_recovery_monotonic_commit_count
                ),
                "metric_visual_closure_contextual_closeup_commit_count": int(
                    metric_visual_closure_contextual_closeup_commit_count
                ),
                "metric_visual_closure_intervention_count": int(
                    metric_visual_closure_intervention_count
                ),
                "image_vlm_stop_validation_attempts": int(
                    image_vlm_stop_validation_attempts
                ),
                "image_vlm_stop_validation_rejections": int(
                    image_vlm_stop_validation_rejections
                ),
                "image_vlm_panoramic_query_attempts": int(
                    panorama_query_attempts
                ),
                "image_vlm_panoramic_query_hits": int(panorama_query_hits),
                "image_vlm_hypothesis_confirmed": int(vlm_trust_confirmed),
                "image_vlm_hypothesis_disconfirmed": int(vlm_trust_disconfirmed),
                "image_vlm_foveated_query_attempts": int(
                    foveated_query_attempts
                ),
                "image_vlm_foveated_query_visible": int(
                    foveated_query_visible
                ),
                "image_vlm_foveated_hypotheses": int(foveated_hypothesis_count),
                "image_vlm_foveated_hypotheses_examined": int(
                    0
                    if sgf_explorer is None
                    else getattr(
                        sgf_explorer, "verified_hypothesis_examined_count", 0
                    )
                ),
                "image_vlm_hypothesis_arrival_sweeps": int(
                    hypothesis_arrival_sweep_count
                ),
            },
            "endgame_return": {
                "enabled": bool(endgame_return_planner is not None),
                "active": bool(endgame_summary.get("active", False)),
                "active_source": endgame_summary.get("active_source"),
                "activated_step": int(endgame_summary.get("activated_step", -1)),
                "disabled": bool(endgame_summary.get("disabled", False)),
                "replan_failures": int(endgame_summary.get("replan_failures", 0)),
                "candidate_present": bool(
                    endgame_summary.get("candidate") is not None
                ),
            },
        }
        payload = {
            "task_mode": args.task_mode,
            "navigation_backend": navigation_backend,
            "episode_id": episode_info.episode_id,
            "goal_key": episode_info.goal_key,
            "goal_name": episode_info.goal_name,
            "gt_goal_position_m": episode_info.gt_goal_position_m,
            "gt_goal_viewpoint_m": episode_info.gt_goal_viewpoint_m,
            "text_goal": episode_info.text_goal,
            "text_goal_source": episode_info.text_goal_source,
            "goal_image_source": goal_image_source,
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "success_radius": float(success_radius),
            "steps": steps,
            "done": done,
            "stop_called": stop_called,
            "target_found": target_found_for_task_mode(
                args.task_mode,
                text_target_found=text_sgf_target_found,
                instance_image_target_found=instance_image_target_found,
            ),
            "ablation_telemetry": ablation_telemetry,
            "target_reacquire_count": int(target_reacquire_count),
            "artifact_dir": str(artifact_dir),
            "metadata_path": "",
            "diagnostic_log_path": str(diagnostic_log_path),
            "trace_path": str(artifact_dir / "trace.json"),
            "final_image_path": final_image_path,
            "final_navigation_map_path": str(final_navigation_map_path),
            "final_image_match_path": final_image_match_path,
            "image_match_video_path": image_match_video_result_path,
            "video_path": str(args.video_path or ""),
            "image_dir": str(args.image_dir or ""),
            "sgf_input_dump_dir": "" if sgf_input_dump_dir_path is None else str(sgf_input_dump_dir_path),
            "sgf_input_replay_dir": replay_input_dir if replay_env else "",
            "export_sequence_dir": "",
            "terminal_position_m": terminal_position_m,
            "diagnostics": {
                "navigation_backend": navigation_backend,
                "navigation_pose_source": (
                    "episodic_gps_compass"
                    if not real_env
                else "observation_odometry"
                ),
                "terminal_navigation_position_m": terminal_navigation_position_m,
                "terminal_navigation_frame": (
                    "episode_local_observation_frame"
                    if not real_env
                    else "observation_odometry_frame"
                ),
                "embedding_model": (
                    "CLIP RN50"
                    if args.task_mode == "instance-image"
                    else None
                ),
                "device": str(device),
                "real_env": bool(real_env),
                "physical_real_env": bool(physical_real_env),
                "sgf_input_replay": bool(replay_env),
                "nav_config_path": str(args.nav_config_path),
                "nav_overrides": list(getattr(args, "nav_override", []) or []),
                "dataset_path": str(args.dataset_path),
                "split": str(args.split),
                "goal_source": str(args.goal_source),
                "width": width,
                "height": height,
                "hfov": hfov,
                "camera_height": camera_height,
                "text_sgf": bool(text_sgf_enabled),
                "text_sgf_goal": parsed_text_goal_trace,
                "text_sgf_target_found": bool(text_sgf_target_found),
                "instance_image_target_found": bool(
                    instance_image_target_found
                ),
                "goal_view": (
                    None
                    if goal_view_belief is None
                    else goal_view_belief.summary()
                ),
                "text_sgf_excluded_node_count": int(len(excluded_text_node_ids)),
                "text_sgf_excluded_position_count": int(len(excluded_text_positions)),
                "text_sgf_readmissions": int(text_readmissions),
                "text_sgf_same_vantage_skips": int(text_same_vantage_skips),
                "text_sgf_reobservation_launches": int(text_reobservation_launches),
                "text_sgf_goal_candidate_rejections": dict(
                    sorted(goal_candidate_rejection_counts.items())
                ),
                "stop_validation_attempts": int(stop_validation_attempts),
                "stop_validation_rejections": int(stop_validation_rejections),
                "last_stop_validation": last_stop_validation,
                "last_stop_policy_decision": last_stop_policy_decision,
                "image_vlm_stop_validation_attempts": int(
                    image_vlm_stop_validation_attempts
                ),
                "image_vlm_stop_validation_rejections": int(
                    image_vlm_stop_validation_rejections
                ),
                "image_vlm_panoramic_query_attempts": int(
                    panorama_query_attempts
                ),
                "image_vlm_panoramic_query_hits": int(panorama_query_hits),
                "image_vlm_hypothesis_confirmed": int(vlm_trust_confirmed),
                "image_vlm_hypothesis_disconfirmed": int(vlm_trust_disconfirmed),
                "navigation_config": {
                    key: json_safe_value(getattr(args, key))
                    for key in NAV_ARG_CONFIG_DEFAULTS
                },
            },
            "metrics": metrics,
        }
        if metrics_only:
            payload = compact_metrics_payload(payload)
        else:
            (artifact_dir / "trace.json").write_text(
                json.dumps(trace, indent=2),
                encoding="utf-8",
            )
        output_path: Path | None = None
        if args.output_json:
            output_path = Path(args.output_json)
        elif not metrics_only:
            output_path = artifact_dir / "result.json"
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    finally:
        if diagnostic_log_handle is not None:
            diagnostic_log_handle.close()
        if visualizer is not None:
            visualizer.close()
        if image_dumper is not None:
            image_dumper.close()
        if image_match_video_writer is not None:
            image_match_video_writer.close()
        if sgf_input_dumper is not None:
            sgf_input_dumper.close()
        env.close()


def main() -> None:
    args = parse_args()
    payload = run_episode(args)
    if bool(args.metrics_only) and not str(args.output_json or "").strip():
        print(json.dumps(payload, sort_keys=True))


__all__ = [
    "NAV_ARG_CONFIG_DEFAULTS",
    "NAV_CONFIG_BASE_ONLY_KEYS",
    "NavigationResult",
    "accumulate_panoramic_path",
    "bgr_from_rgb_observation",
    "build_query_sets",
    "compute_intrinsics",
    "create_online_sgf",
    "depth_forward_navigation_guard",
    "depth_observation_to_2d",
    "finalize_image_target_node",
    "main",
    "parse_args",
    "panoramic_observation_interval_mode",
    "panoramic_observation_trigger_due",
    "panoramic_observation_unknown_gate",
    "planar_distance_xz",
    "plan_exploration_action",
    "reconcile_panoramic_observation_action",
    "refresh_image_target_node",
    "run_episode",
    "select_hint_node",
    "select_target_node",
    "sensor_state_to_world_from_camera_cv",
    "summarize_stop_decision",
    "summarize_stop_validation",
    "update_episode_label_stats",
    "update_episode_label_stats_from_summary",
]
