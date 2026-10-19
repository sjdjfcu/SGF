from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import quaternion
from habitat import Env, make_dataset
from habitat.config.default import get_config
from habitat.config.default_structured_configs import (
    CompassSensorConfig,
    GPSSensorConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from omegaconf import OmegaConf

from .goal.grounding import compose_textnav_goal_text


@dataclass
class EpisodeInfo:
    episode_id: int
    scene_id: str
    goal_name: str
    goal_object_id: int
    goal_key: str
    text_goal: str
    text_goal_source: str
    gt_goal_position_m: list[float] | None = None
    gt_goal_viewpoint_m: list[float] | None = None
    goal_image_rgb: np.ndarray | None = field(default=None, repr=False)
    goal_image_source: str = ""
    goal_image_hfov_deg: float | None = None


def set_habitat_seed(config: Any, seed: int) -> None:
    config.habitat.seed = int(seed)
    config.habitat.simulator.seed = int(seed)


class TextGoalEpisodeEnv:
    def __init__(
        self,
        config_path: str,
        dataset_path: str,
        scenes_dir: str,
        split: str,
        image_width: int,
        image_height: int,
        hfov: float,
        camera_height: float,
        min_depth: float,
        max_depth: float,
        turn_angle: float,
        forward_step_size: float,
        agent_radius: float,
        max_episode_steps: int,
        success_radius: float,
        goal_text: str = "",
        goal_source: str = "textnav",
        task_mode: str = "text",
        textnav_dataset_path: str = "",
        episode_id: int = -1,
        seed: int = 0,
    ) -> None:
        self._task_mode = task_mode.strip().lower()
        if self._task_mode not in {"text", "instance-image"}:
            raise ValueError(f"unsupported task_mode: {task_mode}")
        self._goal_text_override = goal_text.strip()
        self._goal_source = goal_source.strip().lower() if goal_source else "textnav"
        if self._goal_source != "textnav":
            raise ValueError(f"unsupported goal_source: {goal_source}")
        if self._task_mode == "instance-image":
            self._patch_instance_image_goal_sensor_cleanup()

        config = get_config(config_path=config_path)
        OmegaConf.set_readonly(config, False)
        set_habitat_seed(config, seed)
        config.habitat.dataset.split = split
        config.habitat.dataset.data_path = dataset_path
        config.habitat.dataset.scenes_dir = scenes_dir
        config.habitat.environment.max_episode_steps = max_episode_steps
        config.habitat.simulator.habitat_sim_v0.allow_sliding = True

        for sensor_name in ("rgb_sensor", "depth_sensor"):
            sensor = config.habitat.simulator.agents.main_agent.sim_sensors[sensor_name]
            sensor.width = image_width
            sensor.height = image_height
            sensor.hfov = int(round(hfov))
            sensor.position = [0.0, camera_height, 0.0]

        depth_sensor = config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
        depth_sensor.min_depth = min_depth
        depth_sensor.max_depth = max_depth
        depth_sensor.normalize_depth = False

        # Both task modes use episodic GPS/compass as the pose observation
        gps_sensor_config = GPSSensorConfig(dimensionality=2)
        compass_sensor_config = CompassSensorConfig()
        if self._task_mode == "instance-image":
            config.habitat.task.type = "InstanceImageNav-v1"
            config.habitat.task.goal_sensor_uuid = "instance_imagegoal"
            config.habitat.task.lab_sensors.gps_sensor = gps_sensor_config
            config.habitat.task.lab_sensors.compass_sensor = compass_sensor_config
        else:
            config.habitat.task.type = "Nav-v0"
            config.habitat.task.lab_sensors = {
                "gps_sensor": gps_sensor_config,
                "compass_sensor": compass_sensor_config,
            }
            config.habitat.task.goal_sensor_uuid = "rgb"
        config.habitat.task.measurements.num_steps.type = "NumStepsMeasure"
        config.habitat.task.measurements.distance_to_goal.type = "DistanceToGoal"
        config.habitat.task.measurements.distance_to_goal.distance_to = "VIEW_POINTS"
        config.habitat.task.measurements.success.type = "Success"
        config.habitat.task.measurements.success.success_distance = float(success_radius)
        config.habitat.task.measurements.spl.type = "SPL"
        config.habitat.task.measurements.soft_spl.type = "SoftSPL"

        config.habitat.simulator.agents.main_agent.height = camera_height
        if float(agent_radius) > 0.0:
            config.habitat.simulator.agents.main_agent.radius = float(agent_radius)
        if float(forward_step_size) > 0.0:
            config.habitat.simulator.forward_step_size = float(forward_step_size)
        config.habitat.simulator.turn_angle = int(round(turn_angle))
        config.habitat.environment.iterator_options.shuffle = False

        self.forward_step_size_m = float(config.habitat.simulator.forward_step_size)
        self.turn_angle_deg = float(config.habitat.simulator.turn_angle)
        self.agent_radius_m = float(config.habitat.simulator.agents.main_agent.radius)
        self.camera_height_m = float(camera_height)

        dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
        OmegaConf.set_readonly(config, True)

        self._env = Env(config=config, dataset=dataset)
        self._textnav_dataset_path = textnav_dataset_path.format(split=split).strip() if textnav_dataset_path else ""
        needs_textnav_attributes = self._task_mode == "text" and not self._goal_text_override
        self._textnav_attributes = (
            self._load_textnav_attributes(self._textnav_dataset_path)
            if needs_textnav_attributes
            else {}
        )
        self._fixed_episode_id = episode_id
        self._episode_counter = -1
        self._last_step_recovery: dict[str, object] = {}
        self._observed_agent_state = None
        self._observed_yaw_rad = 0.0
        self.current_info: EpisodeInfo | None = None

    @property
    def sim(self):
        return self._env.sim

    @property
    def habitat_env(self) -> Env:
        return self._env

    @staticmethod
    def _patch_instance_image_goal_sensor_cleanup() -> None:
        try:
            from habitat.tasks.nav import instance_image_nav_task as image_nav_task
        except Exception:
            return
        sensor_cls = getattr(image_nav_task, "InstanceImageGoalSensor", None)
        if sensor_cls is None or getattr(sensor_cls, "_sgf_safe_cleanup_patch", False):
            return

        def _safe_remove_sensor(self, sensor_uuid: str) -> None:
            agent = self._sim.get_agent(0)
            sensors = getattr(self._sim, "_sensors", None)
            if isinstance(sensors, dict):
                sensors.pop(sensor_uuid, None)
            try:
                image_nav_task.hsim.SensorFactory.delete_subtree_sensor(agent.scene_node, sensor_uuid)
            except Exception:
                pass
            agent_sensors = getattr(agent, "_sensors", None)
            if isinstance(agent_sensors, dict):
                agent_sensors.pop(sensor_uuid, None)
            agent.agent_config.sensor_specifications = [
                spec
                for spec in agent.agent_config.sensor_specifications
                if getattr(spec, "uuid", None) != sensor_uuid
            ]

        sensor_cls._remove_sensor = _safe_remove_sensor
        sensor_cls._sgf_safe_cleanup_patch = True

    def _load_textnav_attributes(self, dataset_path: str) -> dict[str, dict]:
        if not dataset_path:
            raise ValueError("TextNav goal source requires --textnav-dataset-path")
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"TextNav attribute dataset not found: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        attribute_data = payload.get("attribute_data", {})
        if isinstance(attribute_data, dict):
            return attribute_data
        raise ValueError(f"TextNav attribute dataset has invalid attribute_data: {path}")

    def _resolve_text_goal(self, episode) -> tuple[str, str]:
        if self._task_mode == "instance-image":
            return "", ""

        if self._goal_text_override:
            return self._goal_text_override, "override"

        goal_key = getattr(episode, "goal_key", "")
        if not goal_key:
            raise ValueError(
                f"TextNav episode {getattr(episode, 'episode_id', '')} is missing goal_key"
            )
        if goal_key not in self._textnav_attributes:
            raise ValueError(f"TextNav attributes are missing goal_key {goal_key!r}")
        return compose_textnav_goal_text(self._textnav_attributes[goal_key]), "textnav"

    def _episode_goal_name(self, episode) -> str:
        return str(getattr(episode, "object_category", "object"))

    def _episode_goal_key(self, episode) -> str:
        return str(getattr(episode, "goal_key", ""))

    def _episode_goal_object_id(self, episode) -> int:
        if hasattr(episode, "goal_object_id"):
            try:
                return int(getattr(episode, "goal_object_id"))
            except (TypeError, ValueError):
                pass
        return 0

    @staticmethod
    def _world_position_list(value) -> list[float] | None:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != 3 or not np.all(np.isfinite(array)):
            return None
        return [round(float(array[0]), 4), round(float(array[1]), 4), round(float(array[2]), 4)]

    def _extract_goal_position(self, episode) -> tuple[list[float] | None, list[float] | None]:
        goals = getattr(episode, "goals", None) or []
        if not goals:
            return None, None

        goal = goals[0]
        goal_position = self._world_position_list(getattr(goal, "position", None))

        best_viewpoint_position = None
        best_viewpoint_iou = float("-inf")
        for viewpoint in getattr(goal, "view_points", None) or []:
            agent_state = getattr(viewpoint, "agent_state", None)
            position = None
            if agent_state is not None:
                position = getattr(agent_state, "position", None)
            if position is None:
                position = getattr(viewpoint, "position", None)
            position_list = self._world_position_list(position)
            if position_list is None:
                continue
            iou = float(getattr(viewpoint, "iou", 0.0))
            if iou > best_viewpoint_iou:
                best_viewpoint_iou = iou
                best_viewpoint_position = position_list

        return goal_position, best_viewpoint_position

    @staticmethod
    def _extract_goal_image(observations: dict) -> tuple[np.ndarray | None, float | None]:
        image = observations.get("instance_imagegoal")
        if image is None:
            return None, None
        goal_image = np.asarray(image, dtype=np.uint8)
        if goal_image.ndim != 3 or goal_image.shape[2] < 3:
            return None, None
        goal_image = goal_image[:, :, :3].copy()

        hfov = None
        hfov_observation = observations.get("instance_imagegoal_hfov")
        if hfov_observation is not None:
            hfov_array = np.asarray(hfov_observation, dtype=np.float32).reshape(-1)
            if hfov_array.size > 0 and np.isfinite(hfov_array[0]):
                hfov = round(float(hfov_array[0]), 4)
        return goal_image, hfov

    def reset(self):
        self._episode_counter += 1
        if self._fixed_episode_id >= 0:
            self._env.current_episode = self._env.episodes[self._fixed_episode_id]
            self._episode_counter = self._fixed_episode_id

        observations = self._env.reset()
        self._update_observed_pose(observations)
        episode = self._env.current_episode
        text_goal, text_goal_source = self._resolve_text_goal(episode)
        gt_goal_position_m, gt_goal_viewpoint_m = self._extract_goal_position(episode)
        goal_image_rgb, goal_image_hfov_deg = (
            self._extract_goal_image(observations)
            if self._task_mode == "instance-image"
            else (None, None)
        )
        self.current_info = EpisodeInfo(
            episode_id=self._episode_counter,
            scene_id=episode.scene_id,
            goal_name=self._episode_goal_name(episode),
            goal_object_id=self._episode_goal_object_id(episode),
            goal_key=self._episode_goal_key(episode),
            text_goal=text_goal,
            text_goal_source=text_goal_source,
            gt_goal_position_m=gt_goal_position_m,
            gt_goal_viewpoint_m=gt_goal_viewpoint_m,
            goal_image_rgb=goal_image_rgb,
            goal_image_source="dataset:instance_imagegoal" if goal_image_rgb is not None else "",
            goal_image_hfov_deg=goal_image_hfov_deg,
        )
        self._last_step_recovery = {}
        return observations, self.current_info

    def step(self, action: int):
        action = int(action)
        self._last_step_recovery = {}

        observations = self._env.step({"action": action})
        self._update_observed_pose(observations)
        done = self._env.episode_over
        metrics = dict(self._env.get_metrics()) if done else {}
        return observations, done, metrics

    @staticmethod
    def _finite_observation_vector(observations: dict, key: str) -> np.ndarray | None:
        value = observations.get(key)
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            return None
        return array

    def _update_observed_pose(self, observations: dict) -> None:
        gps = self._finite_observation_vector(observations, "gps")
        compass = self._finite_observation_vector(observations, "compass")
        if gps is None or gps.size != 2 or compass is None or compass.size != 1:
            missing = []
            if gps is None or gps.size != 2:
                missing.append("gps")
            if compass is None or compass.size != 1:
                missing.append("compass")
            raise RuntimeError(
                "Habitat navigation requires valid episodic GPS/compass "
                f"observations; missing or invalid: {', '.join(missing)}"
            )

        # Habitat's 2-D EpisodicGPSSensor returns [-z, x] in the reset-pose frame
        position = np.asarray(
            [float(gps[1]), 0.0, -float(gps[0])], dtype=np.float32
        )
        self._observed_yaw_rad = float(compass[0])

        self._observed_yaw_rad = math.atan2(
            math.sin(self._observed_yaw_rad), math.cos(self._observed_yaw_rad)
        )
        rotation = quaternion.from_rotation_vector(
            np.asarray([0.0, self._observed_yaw_rad, 0.0], dtype=np.float64)
        )
        sensor_position = position.copy()
        sensor_position[1] += self.camera_height_m
        sensor_state = SimpleNamespace(
            position=sensor_position,
            rotation=rotation,
        )
        self._observed_agent_state = SimpleNamespace(
            position=position,
            rotation=rotation,
            sensor_states={"rgb": sensor_state, "depth": sensor_state},
        )

    def set_forward_step_size(self, amount_m: float) -> bool:
        amount = float(amount_m)
        if amount <= 0.0:
            return False
        try:
            agent = self._env.sim.get_agent(0)
            action_space = agent.agent_config.action_space
            action_spec = action_space.get("move_forward")
            if action_spec is None:
                action_spec = action_space.get(int(HabitatSimActions.move_forward))
            if action_spec is None:
                return False
            action_spec.actuation.amount = amount
        except Exception:
            return False
        self.forward_step_size_m = amount
        return True

    @property
    def last_step_recovery(self) -> dict[str, object]:
        return dict(self._last_step_recovery)

    def get_sensor_state(self, sensor_uuid: str = "rgb"):
        state = self.get_agent_state()
        if sensor_uuid in state.sensor_states:
            return state.sensor_states[sensor_uuid]
        raise KeyError(f"sensor state not found: {sensor_uuid}")

    def get_agent_state(self):
        if self._observed_agent_state is None:
            raise RuntimeError("navigation pose is unavailable before reset")
        return self._observed_agent_state

    def get_ground_truth_agent_state(self):
        """Simulator pose for evaluation diagnostics only."""

        return self._env.sim.get_agent_state(0)

    def diagnostic_distance_to_goal(self) -> float | None:
        """Geodesic distance to goal, for diagnostics only."""

        value = self._env.get_metrics().get("distance_to_goal")
        try:
            distance = float(value)
        except (TypeError, ValueError):
            return None
        return distance if np.isfinite(distance) else None

    def get_ground_truth_sensor_state(self, sensor_uuid: str = "rgb"):
        """Simulator sensor pose for diagnostics only."""

        state = self.get_ground_truth_agent_state()
        if sensor_uuid in state.sensor_states:
            return state.sensor_states[sensor_uuid]
        raise KeyError(f"ground-truth sensor state not found: {sensor_uuid}")

    def get_agent_position(self) -> np.ndarray:
        return np.asarray(self.get_agent_state().position, dtype=np.float32)


    def close(self) -> None:
        self._env.close()
