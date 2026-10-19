from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import quaternion
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from ..env import EpisodeInfo
from .transforms import quaternion_from_rotation_matrix, state_from_transform


GRAPHICS_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def load_sgf_input_dump_metadata(dump_dir: str | Path) -> dict[str, Any]:
    metadata_path = Path(dump_dir).expanduser() / "metadata.json"
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"SGF input dump metadata must be a JSON object: {metadata_path}")
    return payload


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return payload


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to read replay RGB image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _read_depth(record: dict[str, Any], dump_dir: Path) -> np.ndarray:
    npy_rel = str(record.get("depth_m_npy_path") or "").strip()
    if npy_rel:
        path = dump_dir / npy_rel
        depth = np.load(str(path)).astype(np.float32, copy=False)
    else:
        png_rel = str(record.get("depth_mm_png_path") or "").strip()
        if not png_rel:
            raise RuntimeError("SGF input replay record has no depth path")
        path = dump_dir / png_rel
        depth_mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise RuntimeError(f"failed to read replay depth image: {path}")
        depth = depth_mm.astype(np.float32) / 1000.0
    if depth.ndim != 2:
        raise RuntimeError(f"expected replay depth HxW, got {depth.shape}")
    return np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _matrix_from_pose_payload(payload: dict[str, Any], path: Path) -> np.ndarray:
    matrix = np.asarray(payload.get("world_from_camera_cv"), dtype=np.float32)
    if matrix.shape != (4, 4):
        raise RuntimeError(f"expected world_from_camera_cv 4x4 in {path}")
    return matrix


def _sensor_state_from_pose_payload(payload: dict[str, Any], world_from_camera_cv: np.ndarray):
    position = payload.get("sensor_position_m")
    rotation = payload.get("sensor_rotation_wxyz")
    if position is not None and rotation is not None:
        position_array = np.asarray(position, dtype=np.float32).reshape(-1)
        rotation_array = np.asarray(rotation, dtype=np.float64).reshape(-1)
        if position_array.size == 3 and rotation_array.size == 4:
            return SimpleNamespace(
                position=position_array.astype(np.float32).copy(),
                rotation=np.quaternion(
                    float(rotation_array[0]),
                    float(rotation_array[1]),
                    float(rotation_array[2]),
                    float(rotation_array[3]),
                ).normalized(),
            )

    world_from_camera_gl = np.asarray(world_from_camera_cv, dtype=np.float32).reshape(4, 4) @ GRAPHICS_TO_CV
    return SimpleNamespace(
        position=world_from_camera_gl[:3, 3].astype(np.float32).copy(),
        rotation=quaternion_from_rotation_matrix(world_from_camera_gl[:3, :3]),
    )


class ReplaySGFInputEnv:
    """Offline env that replays frames dumped by SGFInputDumper (open loop)."""

    def __init__(
        self,
        dump_dir: str | Path,
        *,
        task_mode: str,
        goal_text: str,
        goal_name: str,
        episode_id: int,
        max_episode_steps: int,
        forward_step_size_m: float,
        turn_angle_deg: float,
        agent_radius_m: float,
        stop_ends_episode: bool = False,
        goal_image_rgb: np.ndarray | None = None,
        goal_image_source: str = "",
    ) -> None:
        self.dump_dir = Path(dump_dir).expanduser()
        if not self.dump_dir.is_dir():
            raise FileNotFoundError(f"SGF input replay directory does not exist: {self.dump_dir}")
        self.metadata = load_sgf_input_dump_metadata(self.dump_dir)
        self._task_mode = str(task_mode)
        self._goal_text = str(goal_text or "").strip()
        self._goal_name = str(goal_name or "").strip() or str(
            self.metadata.get("goal_name") or "target"
        )
        if not self._goal_text:
            self._goal_text = str(self.metadata.get("text_goal") or "").strip()
        if self._task_mode == "text" and not self._goal_text:
            raise RuntimeError(
                "text SGF input replay requires an explicit goal_text or non-empty "
                "metadata text_goal; goal_name cannot be used as navigation text"
            )

        self.frames = self._load_frame_records()
        if not self.frames:
            raise RuntimeError(f"SGF input replay has no frames: {self.dump_dir / 'frames.jsonl'}")

        first_shape = self.frames[0].get("rgb_shape") or []
        self.width = _safe_int(self.metadata.get("width"), 0)
        self.height = _safe_int(self.metadata.get("height"), 0)
        if (self.width <= 0 or self.height <= 0) and len(first_shape) >= 2:
            self.height = _safe_int(first_shape[0], self.height)
            self.width = _safe_int(first_shape[1], self.width)
        self.fx = _safe_float(self.metadata.get("fx"), 0.0)
        self.fy = _safe_float(self.metadata.get("fy"), self.fx)
        self.cx = _safe_float(self.metadata.get("cx"), max(0.0, float(self.width) / 2.0))
        self.cy = _safe_float(self.metadata.get("cy"), max(0.0, float(self.height) / 2.0))
        if self.fx <= 0.0 and self.width > 0:
            hfov = _safe_float(self.metadata.get("hfov"), 79.0)
            self.fx = float(self.width) / (2.0 * math.tan(math.radians(hfov) / 2.0))
        if self.fy <= 0.0:
            self.fy = self.fx

        self.forward_step_size_m = float(forward_step_size_m)
        self.turn_angle_deg = float(turn_angle_deg)
        self.agent_radius_m = float(agent_radius_m)
        self._episode_id = int(episode_id)
        self._max_episode_steps = int(max_episode_steps)
        self._stop_ends_episode = bool(stop_ends_episode)
        self._goal_image_rgb = goal_image_rgb
        self._goal_image_source = str(goal_image_source or "")
        self._cursor = 0
        self._step_count = 0
        self._last_observations: dict[str, np.ndarray] | None = None
        self._last_sensor_state = SimpleNamespace(
            position=np.zeros(3, dtype=np.float32),
            rotation=np.quaternion(1.0, 0.0, 0.0, 0.0),
        )
        self._last_agent_state = state_from_transform(np.eye(4, dtype=np.float32))
        self._last_step_recovery: dict[str, object] = {}
        self.current_info: EpisodeInfo | None = None

    @property
    def camera_intrinsics(self) -> tuple[float, float, float, float]:
        return float(self.fx), float(self.fy), float(self.cx), float(self.cy)

    @property
    def camera_height_m(self) -> float:
        return float(np.asarray(self._last_agent_state.position, dtype=np.float32).reshape(3)[1])

    @property
    def last_step_recovery(self) -> dict[str, object]:
        return dict(self._last_step_recovery)

    @property
    def last_tracking_state(self) -> str:
        return "replay"

    @property
    def current_replay_step(self) -> int:
        if not self.frames:
            return int(self._cursor)
        return int(self.frames[min(self._cursor, len(self.frames) - 1)].get("step", self._cursor))

    def reset(self):
        self._cursor = 0
        self._step_count = 0
        observations = self._load_current_frame()
        self.current_info = EpisodeInfo(
            episode_id=_safe_int(self.metadata.get("episode_id"), self._episode_id),
            scene_id=str(self.metadata.get("scene_id") or f"replay:{self.dump_dir}"),
            goal_name=self._goal_name,
            goal_object_id=-1,
            goal_key=str(self.metadata.get("goal_key") or ""),
            text_goal=self._goal_text,
            text_goal_source="sgf-input-replay",
            gt_goal_position_m=None,
            gt_goal_viewpoint_m=None,
            goal_image_rgb=self._goal_image_rgb,
            goal_image_source=self._goal_image_source,
            goal_image_hfov_deg=None,
        )
        self._last_step_recovery = {
            "type": "sgf_input_replay_reset",
            "dump_dir": str(self.dump_dir),
            "frame_count": int(len(self.frames)),
            "dump_step": int(self.current_replay_step),
        }
        return observations, self.current_info

    def step(self, action: int):
        self._step_count += 1
        previous_dump_step = int(self.current_replay_step)
        if int(action) == int(HabitatSimActions.stop) and self._stop_ends_episode:
            observations = self._last_observations or self._load_current_frame()
            self._last_step_recovery = {
                "type": "sgf_input_replay",
                "action": int(action),
                "previous_dump_step": previous_dump_step,
                "dump_step": previous_dump_step,
                "advanced": False,
                "stop_ended_episode": True,
            }
            return observations, True, {"replay_stop": True}

        done = False
        if self._cursor + 1 < len(self.frames):
            self._cursor += 1
            observations = self._load_current_frame()
            advanced = True
        else:
            observations = self._last_observations or self._load_current_frame()
            done = True
            advanced = False
        if self._step_count >= self._max_episode_steps:
            done = True
        self._last_step_recovery = {
            "type": "sgf_input_replay",
            "action": int(action),
            "previous_dump_step": previous_dump_step,
            "dump_step": int(self.current_replay_step),
            "advanced": bool(advanced),
            "cursor": int(self._cursor),
            "frame_count": int(len(self.frames)),
        }
        metrics: dict[str, object] = {}
        if not advanced:
            metrics["replay_exhausted"] = True
        if self._step_count >= self._max_episode_steps:
            metrics["max_steps"] = True
        return observations, done, metrics

    def get_sensor_state(self, sensor_uuid: str = "rgb"):
        _ = sensor_uuid
        return self._last_sensor_state

    def get_agent_state(self):
        return self._last_agent_state


    def close(self) -> None:
        return None

    def _load_frame_records(self) -> list[dict[str, Any]]:
        frames_path = self.dump_dir / "frames.jsonl"
        if not frames_path.exists():
            raise FileNotFoundError(f"SGF input replay frames file does not exist: {frames_path}")
        records: list[dict[str, Any]] = []
        with frames_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise RuntimeError(f"expected frame record object at {frames_path}:{line_number}")
                records.append(payload)
        records.sort(key=lambda item: int(item.get("step", 0)))
        return records

    def _load_current_frame(self) -> dict[str, np.ndarray]:
        record = self.frames[self._cursor]
        rgb_rel = str(record.get("rgb_path") or "").strip()
        pose_rel = str(record.get("pose_json_path") or "").strip()
        if not rgb_rel:
            raise RuntimeError("SGF input replay record has no rgb_path")
        if not pose_rel:
            raise RuntimeError("SGF input replay record has no pose_json_path")
        rgb = _read_rgb(self.dump_dir / rgb_rel)
        depth = _read_depth(record, self.dump_dir)
        pose_payload = _load_json(self.dump_dir / pose_rel)
        world_from_camera_cv = _matrix_from_pose_payload(pose_payload, self.dump_dir / pose_rel)
        self._last_sensor_state = _sensor_state_from_pose_payload(pose_payload, world_from_camera_cv)
        world_from_camera_gl = np.asarray(world_from_camera_cv, dtype=np.float32).reshape(4, 4) @ GRAPHICS_TO_CV
        self._last_agent_state = state_from_transform(world_from_camera_gl)
        self._last_observations = {
            "rgb": rgb,
            "depth": depth[:, :, None],
        }
        return self._last_observations
