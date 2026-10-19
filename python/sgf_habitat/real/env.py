from __future__ import annotations

import math
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions

from ..env import EpisodeInfo
from .depth_guard import depth_forward_navigation_guard
from .kobuki import KobukiBaseController
from .transforms import (
    angle_delta_rad,
    state_from_transform,
    transform_from_translation_yaw,
    world_from_base_from_camera_pose,
    yaw_from_rotation,
)
from .zed import ZEDConfig, ZEDFrame, ZEDFrameProvider, frame_as_observations, make_sensor_state


@dataclass
class RealControlConfig:
    forward_step_size_m: float = 0.25
    turn_angle_deg: float = 30.0
    agent_radius_m: float = 0.18
    linear_speed_mps: float = 0.15
    angular_speed_radps: float = 0.6
    action_timeout_s: float = 8.0
    settle_time_s: float = 0.15
    control_period_s: float = 0.03
    forward_tolerance_m: float = 0.03
    turn_tolerance_deg: float = 2.0


def real_turn_progress_rad(start_yaw: float, current_yaw: float, *, left: bool) -> float:
    delta = angle_delta_rad(start_yaw, current_yaw)
    # In the ZED Y-up frame, yaw decreases on a Kobuki left turn
    return -delta if left else delta


class RealZedKobukiEnv:
    def __init__(
        self,
        *,
        zed: ZEDFrameProvider,
        kobuki: KobukiBaseController,
        control: RealControlConfig,
        task_mode: str,
        goal_text: str,
        goal_name: str,
        episode_id: int,
        max_episode_steps: int,
        camera_base_xyz: tuple[float, float, float],
        camera_base_yaw_deg: float,
    ) -> None:
        self.zed = zed
        self.kobuki = kobuki
        self.control = control
        self.forward_step_size_m = float(control.forward_step_size_m)
        self.turn_angle_deg = float(control.turn_angle_deg)
        self.agent_radius_m = float(control.agent_radius_m)
        self._task_mode = str(task_mode)
        self._goal_text = str(goal_text or "").strip()
        self._goal_name = str(goal_name or "").strip() or "target"
        if self._task_mode == "text" and not self._goal_text:
            raise RuntimeError(
                "real text navigation requires a non-empty goal_text; "
                "goal_name cannot be used as navigation text"
            )
        self._episode_id = int(episode_id)
        self._max_episode_steps = int(max_episode_steps)
        self._base_from_camera = transform_from_translation_yaw(camera_base_xyz, camera_base_yaw_deg)
        self._last_frame: ZEDFrame | None = None
        self._last_observations: dict[str, np.ndarray] | None = None
        self._last_agent_state = state_from_transform(np.eye(4, dtype=np.float32))
        self._last_sensor_state = make_sensor_state(
            ZEDFrame(
                rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                depth=np.zeros((1, 1), dtype=np.float32),
                intrinsics=(1.0, 1.0, 0.0, 0.0),
                world_from_camera_gl=np.eye(4, dtype=np.float32),
                tracking_state="initial",
            )
        )
        self.current_info: EpisodeInfo | None = None
        self._started = False
        self._step_count = 0
        self._last_step_recovery: dict[str, object] = {}

    @property
    def camera_intrinsics(self) -> tuple[float, float, float, float]:
        return self.zed.intrinsics

    @property
    def base_from_camera_cv(self) -> np.ndarray:
        return self._base_from_camera @ np.diag([1.0, -1.0, -1.0, 1.0])

    @property
    def camera_height_m(self) -> float:
        return float(self._last_agent_state.position[1])

    @property
    def last_step_recovery(self) -> dict[str, object]:
        return dict(self._last_step_recovery)

    @property
    def last_tracking_state(self) -> str:
        if self._last_frame is None:
            return ""
        return str(self._last_frame.tracking_state)

    def reset(self):
        if not self._started:
            try:
                self.zed.open()
                self.kobuki.start()
            except Exception:
                try:
                    self.kobuki.close()
                finally:
                    self.zed.close()
                raise
            self._started = True
        self._step_count = 0
        observations = self._capture_observations()
        self.current_info = EpisodeInfo(
            episode_id=self._episode_id,
            scene_id="real:kobuki_zed2i",
            goal_name=self._goal_name,
            goal_object_id=-1,
            goal_key="",
            text_goal=self._goal_text,
            text_goal_source="real-cli",
            gt_goal_position_m=None,
            gt_goal_viewpoint_m=None,
            goal_image_rgb=None,
            goal_image_source="",
            goal_image_hfov_deg=None,
        )
        return observations, self.current_info

    def step(self, action: int):
        action = int(action)
        self._last_step_recovery = {}
        self._step_count += 1
        done = False
        metrics: dict[str, object] = {}
        if action == int(HabitatSimActions.stop):
            self.kobuki.stop()
            observations = self._capture_observations()
            done = True
            metrics["real_stop"] = True
            return observations, done, metrics

        if action == int(HabitatSimActions.move_forward):
            self._execute_forward()
        elif action == int(HabitatSimActions.turn_left):
            self._execute_turn(left=True)
        elif action == int(HabitatSimActions.turn_right):
            self._execute_turn(left=False)
        else:
            self.kobuki.stop()
            self._last_step_recovery = {"type": "unsupported_real_action", "action": action}

        observations = self._capture_observations()
        status = self.kobuki.latest_status
        if status.safety_stop:
            done = True
            metrics["safety_stop"] = status.raw or {}
        if self._step_count >= self._max_episode_steps:
            done = True
            metrics["max_steps"] = True
        return observations, done, metrics

    def set_forward_step_size(self, amount_m: float) -> bool:
        amount = float(amount_m)
        if amount <= 0.0:
            return False
        self.forward_step_size_m = amount
        self.control.forward_step_size_m = amount
        return True

    def get_sensor_state(self, sensor_uuid: str = "rgb"):
        _ = sensor_uuid
        return self._last_sensor_state

    def get_agent_state(self):
        return self._last_agent_state


    def close(self) -> None:
        try:
            self.kobuki.close()
        finally:
            self.zed.close()

    def _capture_observations(self) -> dict[str, np.ndarray]:
        frame = self.zed.grab()
        self._last_frame = frame
        self._last_observations = frame_as_observations(frame)
        world_from_base = world_from_base_from_camera_pose(frame.world_from_camera_gl, self._base_from_camera)
        self._last_agent_state = state_from_transform(world_from_base)
        self._last_sensor_state = make_sensor_state(frame)
        return self._last_observations

    def _execute_forward(self) -> None:
        start = np.asarray(self.get_agent_state().position, dtype=np.float32).copy()
        start_yaw = yaw_from_rotation(self.get_agent_state().rotation)
        started_at = time.monotonic()
        deadline = time.monotonic() + float(self.control.action_timeout_s)
        target_m = float(self.forward_step_size_m)
        tolerance_m = float(self.control.forward_tolerance_m)
        displacement = 0.0
        iterations = 0
        reason = "timeout"
        depth_guard = None
        try:
            while time.monotonic() < deadline:
                iterations += 1
                self._capture_observations()
                current = np.asarray(self.get_agent_state().position, dtype=np.float32)
                displacement = float(np.linalg.norm(current[[0, 2]] - start[[0, 2]]))
                if displacement >= max(0.0, target_m - tolerance_m):
                    reason = "target"
                    break
                if self.kobuki.latest_status.safety_stop:
                    reason = "safety_stop"
                    break
                # Re-check depth clearance during the motion
                fx, fy, cx, cy = self._last_frame.intrinsics
                depth_guard = depth_forward_navigation_guard(
                    self._last_frame.depth,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    forward_step_size_m=max(0.0, target_m - displacement),
                    agent_radius_m=self.agent_radius_m,
                    camera_height_m=float(self._base_from_camera[1, 3]),
                    min_depth_m=float(self.zed.config.min_depth_m),
                    max_depth_m=float(self.zed.config.max_depth_m),
                    base_from_camera_cv=self.base_from_camera_cv,
                    forward_clearance_margin_m=self.agent_radius_m,
                )
                if depth_guard.get("blocked", False):
                    reason = "depth_guard"
                    break
                self.kobuki.set_velocity(abs(float(self.control.linear_speed_mps)), 0.0)
                time.sleep(float(self.control.control_period_s))
        finally:
            self.kobuki.stop()
            time.sleep(max(0.0, float(self.control.settle_time_s)))
            end_position = np.asarray(self.get_agent_state().position, dtype=np.float32)
            end_yaw = yaw_from_rotation(self.get_agent_state().rotation)
            self._last_step_recovery = {
                "type": "real_forward",
                "reason": reason,
                "target_m": round(target_m, 4),
                "tolerance_m": round(tolerance_m, 4),
                "progress_m": round(displacement, 4),
                "elapsed_s": round(time.monotonic() - started_at, 3),
                "iterations": int(iterations),
                "start_yaw_deg": round(math.degrees(start_yaw), 3),
                "end_yaw_deg": round(math.degrees(end_yaw), 3),
                "delta_yaw_deg": round(math.degrees(angle_delta_rad(start_yaw, end_yaw)), 3),
                "start_position_m": np.round(start, 4).tolist(),
                "end_position_m": np.round(end_position, 4).tolist(),
                "tracking_state": self.last_tracking_state,
                "depth_guard": depth_guard,
            }

    def _execute_turn(self, *, left: bool) -> None:
        start_yaw = yaw_from_rotation(self.get_agent_state().rotation)
        target_rad = math.radians(float(self.turn_angle_deg))
        angular = abs(float(self.control.angular_speed_radps)) * (1.0 if left else -1.0)
        started_at = time.monotonic()
        deadline = time.monotonic() + float(self.control.action_timeout_s)
        current_yaw = start_yaw
        delta = 0.0
        progress = 0.0
        iterations = 0
        reason = "timeout"
        self.kobuki.set_velocity(0.0, angular)
        try:
            while time.monotonic() < deadline:
                iterations += 1
                self._capture_observations()
                current_yaw = yaw_from_rotation(self.get_agent_state().rotation)
                delta = angle_delta_rad(start_yaw, current_yaw)
                progress = real_turn_progress_rad(start_yaw, current_yaw, left=left)
                if progress >= max(0.0, target_rad - math.radians(float(self.control.turn_tolerance_deg))):
                    reason = "target"
                    break
                if self.kobuki.latest_status.safety_stop:
                    reason = "safety_stop"
                    break
                time.sleep(float(self.control.control_period_s))
        finally:
            self.kobuki.stop()
            time.sleep(max(0.0, float(self.control.settle_time_s)))
            self._last_step_recovery = {
                "type": "real_turn_left" if left else "real_turn_right",
                "reason": reason,
                "target_deg": round(math.degrees(target_rad), 3),
                "tolerance_deg": round(float(self.control.turn_tolerance_deg), 3),
                "progress_deg": round(math.degrees(progress), 3),
                "raw_delta_deg": round(math.degrees(delta), 3),
                "start_yaw_deg": round(math.degrees(start_yaw), 3),
                "end_yaw_deg": round(math.degrees(current_yaw), 3),
                "commanded_angular_radps": round(float(angular), 4),
                "elapsed_s": round(time.monotonic() - started_at, 3),
                "iterations": int(iterations),
                "tracking_state": self.last_tracking_state,
            }


def create_real_zed_kobuki_env_from_args(args, *, width: int, height: int, min_depth: float, max_depth: float):
    task_mode = str(args.task_mode)
    goal_text = str(args.goal_text or "").strip()
    if task_mode == "text" and not goal_text:
        raise RuntimeError("--goal-text is required for real text navigation")

    zed_config = ZEDConfig(
        resolution=str(args.zed_resolution),
        fps=int(args.zed_fps),
        depth_mode=str(args.zed_depth_mode),
        min_depth_m=float(min_depth),
        max_depth_m=float(max_depth),
        output_width=width,
        output_height=height,
        color_order=str(args.zed_color_order),
        require_tracking=bool(args.zed_require_tracking),
    )
    control = RealControlConfig(
        forward_step_size_m=float(args.real_forward_step_size),
        turn_angle_deg=float(args.real_turn_angle),
        agent_radius_m=float(args.agent_radius),
        linear_speed_mps=float(args.kobuki_linear_speed),
        angular_speed_radps=float(args.kobuki_angular_speed),
        action_timeout_s=float(args.real_action_timeout),
        settle_time_s=float(args.real_settle_time),
        control_period_s=float(args.real_control_period),
    )
    goal_name = str(args.real_goal_name or "").strip()
    if not goal_name:
        goal_name = "target" if task_mode == "instance-image" else goal_text
    return RealZedKobukiEnv(
        zed=ZEDFrameProvider(zed_config),
        kobuki=KobukiBaseController(
            device=str(args.kobuki_device),
            kobuki_install=str(args.kobuki_install),
            bridge_binary=str(args.kobuki_bridge_binary),
        ),
        control=control,
        task_mode=task_mode,
        goal_text=goal_text,
        goal_name=goal_name,
        episode_id=int(args.episode_id),
        max_episode_steps=int(args.max_steps),
        camera_base_xyz=(
            float(args.zed_camera_base_x),
            float(args.zed_camera_base_y),
            float(args.zed_camera_base_z),
        ),
        camera_base_yaw_deg=float(args.zed_camera_base_yaw_deg),
    )
