from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import quaternion

from .transforms import quaternion_from_rotation_matrix, state_from_transform


@dataclass
class ZEDConfig:
    resolution: str = "HD720"
    fps: int = 30
    depth_mode: str = "NEURAL"
    min_depth_m: float = 0.2
    max_depth_m: float = 5.0
    output_width: int = 0
    output_height: int = 0
    color_order: str = "bgra"
    require_tracking: bool = True


@dataclass
class ZEDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: tuple[float, float, float, float]
    world_from_camera_gl: np.ndarray
    tracking_state: str


class ZEDFrameProvider:
    def __init__(self, config: ZEDConfig) -> None:
        self.config = config
        self._sl: Any | None = None
        self._zed: Any | None = None
        self._runtime_params: Any | None = None
        self._image_mat: Any | None = None
        self._depth_mat: Any | None = None
        self._pose: Any | None = None
        self._resolution: Any | None = None
        self._intrinsics: tuple[float, float, float, float] | None = None

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        if self._intrinsics is None:
            raise RuntimeError("ZED camera is not open")
        return self._intrinsics

    @property
    def image_size(self) -> tuple[int, int]:
        if self._resolution is None:
            raise RuntimeError("ZED camera is not open")
        return int(self._resolution.width), int(self._resolution.height)

    def open(self) -> None:
        import pyzed.sl as sl  # type: ignore

        self._sl = sl
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = self._enum_value(sl.RESOLUTION, self.config.resolution)
        if int(self.config.fps) > 0:
            init_params.camera_fps = int(self.config.fps)
        init_params.depth_mode = self._enum_value(sl.DEPTH_MODE, self.config.depth_mode)
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.coordinate_units = sl.UNIT.METER
        if hasattr(init_params, "depth_minimum_distance"):
            init_params.depth_minimum_distance = float(self.config.min_depth_m)
        if hasattr(init_params, "depth_maximum_distance"):
            init_params.depth_maximum_distance = float(self.config.max_depth_m)

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"failed to open ZED camera: {err}")

        tracking_params = sl.PositionalTrackingParameters()
        if hasattr(sl.POSITIONAL_TRACKING_MODE, "GEN_3") and hasattr(tracking_params, "mode"):
            tracking_params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_3
        err = zed.enable_positional_tracking(tracking_params)
        if err != sl.ERROR_CODE.SUCCESS:
            zed.close()
            raise RuntimeError(f"failed to enable ZED positional tracking: {err}")

        camera_info = zed.get_camera_information()
        native_resolution = self._camera_information_resolution(camera_info)
        width = int(self.config.output_width) if int(self.config.output_width) > 0 else int(native_resolution[0])
        height = int(self.config.output_height) if int(self.config.output_height) > 0 else int(native_resolution[1])
        self._resolution = sl.Resolution(width, height)
        resized_info = zed.get_camera_information(self._resolution)
        self._intrinsics = self._camera_information_intrinsics(resized_info)

        self._zed = zed
        self._runtime_params = sl.RuntimeParameters()
        self._image_mat = sl.Mat()
        self._depth_mat = sl.Mat()
        self._pose = sl.Pose()

    def grab(self) -> ZEDFrame:
        if self._zed is None or self._sl is None:
            raise RuntimeError("ZED camera is not open")
        sl = self._sl
        err = self._zed.grab(self._runtime_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"failed to grab ZED frame: {err}")

        self._zed.retrieve_image(self._image_mat, sl.VIEW.LEFT, sl.MEM.CPU, self._resolution)
        self._zed.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, self._resolution)
        tracking_state = self._zed.get_position(self._pose, sl.REFERENCE_FRAME.WORLD)
        if self.config.require_tracking and tracking_state != sl.POSITIONAL_TRACKING_STATE.OK:
            raise RuntimeError(f"ZED positional tracking is not OK: {tracking_state}")

        image = np.asarray(self._image_mat.get_data())
        depth = np.asarray(self._depth_mat.get_data(), dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        return ZEDFrame(
            rgb=self._convert_image_to_rgb(image),
            depth=depth,
            intrinsics=self.intrinsics,
            world_from_camera_gl=self._pose_to_world_from_camera_gl(),
            tracking_state=str(tracking_state),
        )

    def sensor_state_from_pose(self, world_from_camera_gl: np.ndarray):
        return state_from_transform(world_from_camera_gl)

    def close(self) -> None:
        if self._zed is not None:
            try:
                self._zed.disable_positional_tracking()
            except Exception:
                pass
            self._zed.close()
        self._zed = None

    def _enum_value(self, enum_cls, name: str):
        enum_name = str(name or "").strip().upper()
        if not enum_name:
            raise ValueError("empty ZED enum name")
        try:
            return getattr(enum_cls, enum_name)
        except AttributeError as exc:
            raise ValueError(f"unsupported ZED enum value {enum_cls.__name__}.{enum_name}") from exc

    def _camera_information_resolution(self, camera_info: Any) -> tuple[int, int]:
        candidates = [
            getattr(camera_info, "camera_resolution", None),
            getattr(getattr(camera_info, "camera_configuration", None), "resolution", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            width = int(getattr(candidate, "width", 0) or 0)
            height = int(getattr(candidate, "height", 0) or 0)
            if width > 0 and height > 0:
                return width, height

        left = self._left_camera_parameters(camera_info)
        image_size = getattr(left, "image_size", None)
        width = int(getattr(image_size, "width", 0) or 0)
        height = int(getattr(image_size, "height", 0) or 0)
        if width > 0 and height > 0:
            return width, height

        if int(self.config.output_width) > 0 and int(self.config.output_height) > 0:
            return int(self.config.output_width), int(self.config.output_height)

        raise RuntimeError(
            "failed to determine ZED image resolution; pass --zed-output-width and --zed-output-height"
        )

    def _camera_information_intrinsics(self, camera_info: Any) -> tuple[float, float, float, float]:
        left = self._left_camera_parameters(camera_info)
        return (float(left.fx), float(left.fy), float(left.cx), float(left.cy))

    def _left_camera_parameters(self, camera_info: Any) -> Any:
        calibration = getattr(getattr(camera_info, "camera_configuration", None), "calibration_parameters", None)
        left = getattr(calibration, "left_cam", None)
        if left is None:
            raise RuntimeError("failed to read ZED left-camera calibration parameters")
        return left

    def _convert_image_to_rgb(self, image: np.ndarray) -> np.ndarray:
        color_order = str(self.config.color_order or "bgra").strip().lower()
        if image.ndim != 3 or image.shape[2] < 3:
            raise RuntimeError(f"expected ZED color image with shape HxWx3/4, got {image.shape}")
        if image.shape[2] == 4:
            if color_order == "rgba":
                return np.ascontiguousarray(image[:, :, :3].astype(np.uint8, copy=False))
            if color_order == "bgra":
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if color_order == "rgb":
            return np.ascontiguousarray(image[:, :, :3].astype(np.uint8, copy=False))
        if color_order == "bgr":
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
        raise ValueError(f"unsupported ZED color order: {self.config.color_order}")

    def _pose_to_world_from_camera_gl(self) -> np.ndarray:
        if self._sl is None or self._pose is None:
            raise RuntimeError("ZED camera is not open")
        sl = self._sl
        translation = sl.Translation()
        orientation = sl.Orientation()
        t = np.asarray(self._pose.get_translation(translation).get(), dtype=np.float32).reshape(3)
        q_values = np.asarray(self._pose.get_orientation(orientation).get(), dtype=np.float64).reshape(4)
        # ZED returns orientation as x, y, z, w.
        q = np.quaternion(float(q_values[3]), float(q_values[0]), float(q_values[1]), float(q_values[2]))
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = quaternion.as_rotation_matrix(q.normalized()).astype(np.float32)
        matrix[:3, 3] = t
        return matrix


def frame_as_observations(frame: ZEDFrame) -> dict[str, np.ndarray]:
    return {
        "rgb": frame.rgb,
        "depth": frame.depth[:, :, None],
    }


def make_sensor_state(frame: ZEDFrame):
    return SimpleNamespace(
        position=frame.world_from_camera_gl[:3, 3].astype(np.float32).copy(),
        rotation=quaternion_from_rotation_matrix(frame.world_from_camera_gl[:3, :3]),
    )
