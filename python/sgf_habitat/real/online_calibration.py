"""Live, bounded Kobuki motion for ZED horizontal extrinsic calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .calibration import fit_horizontal_camera_offset, fit_mounting_yaw_from_straights


@dataclass(frozen=True)
class OnlineCalibrationConfig:
    camera_height_m: float
    angular_speed_radps: float = 0.3
    linear_speed_mps: float = 0.1
    straight_distance_m: float = 0.3
    agent_radius_m: float = 0.18
    max_mount_yaw_deg: float = 30.0
    velocity_timeout_s: float = 0.75
    settle_time_s: float = 0.5
    control_period_s: float = 0.03
    heading_gain_s_inv: float = 1.5
    heading_max_angular_speed_radps: float = 0.08

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.angular_speed_radps > 0.4 or self.linear_speed_mps > 0.15:
            raise ValueError("calibration speeds must be at most 0.4 rad/s and 0.15 m/s")
        if not 0.25 <= self.straight_distance_m <= 0.75:
            raise ValueError("each straight segment must be between 0.25 and 0.75 m")
        if self.max_mount_yaw_deg > 30.0:
            raise ValueError("live calibration requires a forward-facing camera within 30 degrees")
        if self.heading_max_angular_speed_radps > 0.15:
            raise ValueError("forward heading corrections must be at most 0.15 rad/s")


def forward_calibration_clear(frame, offset, config: OnlineCalibrationConfig) -> bool:
    """Check the forward corridor for every candidate mounting yaw."""

    depth = np.asarray(frame.depth, dtype=np.float32)[::6, ::6]
    if depth.ndim != 2 or depth.size == 0:
        return False
    rows, cols = np.indices(depth.shape)
    fx, fy, cx, cy = frame.intrinsics
    known = np.isfinite(depth) & (depth > 0.05)
    central = known[depth.shape[0] // 4:3 * depth.shape[0] // 4,
                    depth.shape[1] // 4:3 * depth.shape[1] // 4]
    if central.size == 0 or float(np.mean(central)) < 0.5:
        return False
    x = (cols * 6 - cx) * depth / fx + offset[0]
    y = -(rows * 6 - cy) * depth / fy + offset[1]
    z = -depth + offset[2]
    obstacles = known & (y >= 0.08) & (y <= config.camera_height_m + 0.35)
    horizon = config.agent_radius_m + 0.22 + config.linear_speed_mps * config.velocity_timeout_s
    for yaw in np.radians(np.linspace(-config.max_mount_yaw_deg, config.max_mount_yaw_deg, 13)):
        base_x = math.cos(yaw) * x - math.sin(yaw) * z
        forward = -(math.sin(yaw) * x + math.cos(yaw) * z)
        corridor = obstacles & (forward > 0.0) & (forward < horizon)
        corridor &= np.abs(base_x) < config.agent_radius_m + 0.1
        if np.count_nonzero(corridor) >= 4:
            return False
    return True


class OnlineExtrinsicsCalibrator:
    """Online camera-to-base extrinsic calibration by driving the Kobuki."""

    def __init__(self, camera, base, config, *, record=None, progress=None,
                 monotonic=time.monotonic, sleep=time.sleep):
        self.camera, self.base, self.config = camera, base, config
        self.record = record or (lambda record: None)
        self.progress = progress or (lambda message: None)
        self.now, self.sleep = monotonic, sleep
        self.phase = "initializing"
        self._frame_time = float("-inf")
        self._previous_yaw = None
        self._last_command = {"linear_mps": 0.0, "angular_radps": 0.0}
        self.diagnostics = {"rotation_fits": [], "forward_segments": []}

    def _check_base(self):
        status = self.base.latest_status
        if not status.alive or not status.enabled or self.base.status_age_s > 1.0:
            raise RuntimeError("Kobuki status is unavailable or stale")
        if status.safety_stop or status.bumper or status.cliff or status.wheel_drop:
            raise RuntimeError("Kobuki bumper/cliff/wheel-drop safety stop")

    @staticmethod
    def _yaw(pose):
        return math.atan2(-float(pose[0, 2]), float(pose[2, 2]))

    @staticmethod
    def _angle_error(yaw, target):
        return math.atan2(math.sin(yaw - target), math.cos(yaw - target))

    def _read(self):
        self._check_base()
        started = self.now()
        frame = self.camera.grab()
        self._frame_time = self.now()
        self._check_base()
        if self._frame_time - started >= self.config.velocity_timeout_s:
            raise RuntimeError("ZED frame exceeded the velocity watchdog interval")
        if str(frame.tracking_state) != "OK":
            raise RuntimeError(f"ZED tracking is not OK: {frame.tracking_state}")
        pose = np.asarray(frame.world_from_camera_gl, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise RuntimeError("invalid ZED pose")
        yaw = self._yaw(pose)
        if self._previous_yaw is not None:
            jump = math.atan2(math.sin(yaw - self._previous_yaw), math.cos(yaw - self._previous_yaw))
            if abs(jump) > math.radians(30.0):
                raise RuntimeError("ZED pose jumped during calibration")
        self._previous_yaw = yaw
        status = self.base.latest_status
        self.record({
            "time_s": self._frame_time, "phase": self.phase,
            "world_from_camera_gl": pose.tolist(),
            "commanded_velocity": dict(self._last_command),
            "base_odometry": {"x_m": status.x, "y_m": status.y, "heading_rad": status.heading},
            "base_status_age_s": self.base.status_age_s,
        })
        return frame

    def _command(self, linear, angular):
        self._check_base()
        if self.now() - self._frame_time >= self.config.velocity_timeout_s:
            raise RuntimeError("cannot move using a stale camera frame")
        self.base.set_velocity(linear, angular, timeout_s=self.config.velocity_timeout_s)
        self._last_command = {"linear_mps": float(linear), "angular_radps": float(angular)}

    def _stop(self):
        self.base.stop()
        self._last_command = {"linear_mps": 0.0, "angular_radps": 0.0}

    def _settle(self):
        self._stop()
        deadline = self.now() + self.config.settle_time_s
        frame = self._read()
        while self.now() < deadline:
            self.sleep(self.config.control_period_s)
            frame = self._read()
        return frame

    def _rotate(self, left):
        self.phase = "rotate_left" if left else "rotate_right"
        self.progress(f"{self.phase}: 360 degrees")
        frame = self._settle()
        poses = [np.asarray(frame.world_from_camera_gl).copy()]
        previous = self._yaw(poses[-1])
        progress = last_sample = last_progress = 0.0
        direction = -1.0 if left else 1.0
        start = progressed_at = self.now()
        timeout = 2.0 * (2.0 * math.pi) / self.config.angular_speed_radps + 5.0
        while progress < 2.0 * math.pi:
            remaining = 2.0 * math.pi - progress
            speed = min(self.config.angular_speed_radps, max(0.08, remaining * 0.8))
            self._command(0.0, -direction * speed)
            self.sleep(self.config.control_period_s)
            frame = self._read()
            pose = np.asarray(frame.world_from_camera_gl)
            yaw = self._yaw(pose)
            progress += direction * math.atan2(math.sin(yaw - previous), math.cos(yaw - previous))
            previous = yaw
            if progress - last_sample >= math.radians(5.0):
                poses.append(pose.copy())
                last_sample = progress
            if progress - last_progress > math.radians(1.0):
                progressed_at, last_progress = self.now(), progress
            if progress < -math.radians(5.0) or self.now() - progressed_at > 4.0 or self.now() - start > timeout:
                raise RuntimeError("rotation failed to make the expected progress")
        frame = self._settle()
        poses.append(np.asarray(frame.world_from_camera_gl).copy())
        return np.asarray(poses)

    def _straight(self, index, offset):
        self.phase = f"forward_{index}"
        self.progress(f"{self.phase}: {self.config.straight_distance_m:.2f} m")
        frame = self._settle()
        poses = [np.asarray(frame.world_from_camera_gl).copy()]
        origin = poses[0][:3, 3] - poses[0][:3, :3] @ offset
        target_yaw = self._yaw(poses[0])
        heading_error = 0.0
        start = progressed_at = self.now()
        progress = last_progress = 0.0
        diagnostic = {
            "phase": self.phase, "target_distance_m": self.config.straight_distance_m,
            "distance_m": 0.0, "heading_error_deg": 0.0, "max_heading_error_deg": 0.0,
            "heading_limit_deg": 3.0, "completed": False,
        }
        self.diagnostics["forward_segments"].append(diagnostic)

        def observe(pose):
            center = pose[:3, 3] - pose[:3, :3] @ offset
            distance = float(np.linalg.norm((center - origin)[[0, 2]]))
            error = self._angle_error(self._yaw(pose), target_yaw)
            diagnostic.update({
                "distance_m": distance, "heading_error_deg": math.degrees(error),
                "max_heading_error_deg": max(diagnostic["max_heading_error_deg"], abs(math.degrees(error))),
                "elapsed_s": self.now() - start,
            })
            if abs(error) > math.radians(diagnostic["heading_limit_deg"]):
                raise RuntimeError(
                    f"forward calibration motion is not straight ({self.phase}: "
                    f"heading error {math.degrees(error):+.3f} deg exceeds 3.0 deg "
                    f"after {distance:.3f} m despite heading feedback)"
                )
            return distance, error

        timeout = 2.0 * self.config.straight_distance_m / self.config.linear_speed_mps + 5.0
        while progress < self.config.straight_distance_m:
            if not forward_calibration_clear(frame, offset, self.config):
                raise RuntimeError("forward calibration corridor is obstructed or depth is unavailable")
            remaining = self.config.straight_distance_m - progress
            speed = min(self.config.linear_speed_mps, max(0.03, remaining * 0.8))
            # GL yaw increases to the right and positive Kobuki angular velocity
            # turns left, so a positive yaw error needs a positive command
            angular_limit = min(self.config.heading_max_angular_speed_radps, 0.8 * speed)
            angular = float(np.clip(self.config.heading_gain_s_inv * heading_error,
                                    -angular_limit, angular_limit))
            self._command(speed, angular)
            self.sleep(self.config.control_period_s)
            frame = self._read()
            pose = np.asarray(frame.world_from_camera_gl)
            poses.append(pose.copy())
            progress, heading_error = observe(pose)
            if progress - last_progress > 0.01:
                progressed_at, last_progress = self.now(), progress
            if self.now() - progressed_at > 4.0 or self.now() - start > timeout:
                raise RuntimeError("forward motion failed to make progress")
        frame = self._settle()
        poses.append(np.asarray(frame.world_from_camera_gl).copy())
        observe(poses[-1])
        diagnostic["completed"] = True
        self.progress(
            f"{self.phase}: completed {diagnostic['distance_m']:.3f} m; "
            f"max heading error {diagnostic['max_heading_error_deg']:.3f} deg"
        )
        return np.asarray(poses)

    def run(self):
        try:
            self._stop()
            deadline = self.now() + 3.0
            while self.base.status_age_s > 1.0 or not self.base.latest_status.alive:
                if self.now() >= deadline:
                    raise RuntimeError("timed out waiting for fresh Kobuki status")
                self.sleep(0.05)
            if not self.base.latest_status.velocity_timeout_supported:
                raise RuntimeError(
                    "the Kobuki bridge does not report velocity watchdog support; "
                    "build it with python3 tools/build_kobuki_bridge.py"
                )
            fits = self.diagnostics["rotation_fits"]
            for left in (True, False):
                poses = self._rotate(left)
                fit = fit_horizontal_camera_offset(poses, camera_base_y_m=self.config.camera_height_m)
                fits.append(fit)
                if not fit["accepted"]:
                    raise RuntimeError(f"rotation calibration rejected: {fit['rejection_reasons']}")
            offsets = np.array([[fit["camera_base_x_m"], self.config.camera_height_m, fit["camera_base_z_m"]] for fit in fits])
            difference = float(np.linalg.norm(offsets[0] - offsets[1]))
            self.diagnostics["rotation_offset_disagreement_m"] = difference
            if difference > 0.015:
                raise RuntimeError("clockwise/counterclockwise pivot offsets disagree by more than 15 mm")
            offset = np.mean(offsets, axis=0)
            self.progress("rotation fits agree; estimating mounting yaw with two forward segments")
            segments = [self._straight(index, offset) for index in (1, 2)]
            result = fit_mounting_yaw_from_straights(segments, camera_frame_offset_xyz=offset)
            if abs(result["camera_base_yaw_deg"]) > self.config.max_mount_yaw_deg:
                result["accepted"] = False
                result["rejection_reasons"].append("mounting_yaw_outside_forward_camera_assumption")
            result.update({
                "method": "online_bidirectional_pivot_and_forward_motion",
                **self.diagnostics,
            })
            return result
        finally:
            self._stop()
