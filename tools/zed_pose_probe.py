#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sgf_habitat.real.transforms import (  # noqa: E402
    angle_delta_rad,
    state_from_transform,
    transform_from_translation_yaw,
    world_from_base_from_camera_pose,
    yaw_from_rotation,
)
from sgf_habitat.real.zed import ZEDConfig, ZEDFrameProvider  # noqa: E402


def _position_list(value: np.ndarray) -> list[float]:
    array = np.asarray(value, dtype=np.float32).reshape(3)
    return [round(float(array[0]), 4), round(float(array[1]), 4), round(float(array[2]), 4)]


def _record(
    *,
    index: int,
    started_at: float,
    last_at: float | None,
    frame,
    base_from_camera: np.ndarray,
    start_position: np.ndarray,
    start_yaw: float,
) -> tuple[dict[str, object], float]:
    now = time.monotonic()
    world_from_base = world_from_base_from_camera_pose(frame.world_from_camera_gl, base_from_camera)
    base_state = state_from_transform(world_from_base)
    base_position = np.asarray(base_state.position, dtype=np.float32).reshape(3)
    yaw = yaw_from_rotation(base_state.rotation)
    camera_position = np.asarray(frame.world_from_camera_gl[:3, 3], dtype=np.float32).reshape(3)
    planar_delta = float(np.linalg.norm(base_position[[0, 2]] - start_position[[0, 2]]))
    hz = 0.0 if last_at is None else 1.0 / max(1e-6, now - last_at)
    return (
        {
            "i": int(index),
            "t_s": round(now - started_at, 3),
            "grab_hz": round(hz, 2),
            "tracking_state": str(frame.tracking_state),
            "camera_position_m": _position_list(camera_position),
            "base_position_m": _position_list(base_position),
            "yaw_deg": round(math.degrees(yaw), 3),
            "delta_yaw_deg": round(math.degrees(angle_delta_rad(start_yaw, yaw)), 3),
            "planar_delta_m": round(planar_delta, 4),
        },
        now,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print ZED 2i positional tracking pose without driving Kobuki.")
    parser.add_argument("--zed-resolution", default="HD720")
    parser.add_argument("--zed-fps", type=int, default=30)
    parser.add_argument("--zed-depth-mode", default="NEURAL")
    parser.add_argument("--zed-output-width", type=int, default=640)
    parser.add_argument("--zed-output-height", type=int, default=480)
    parser.add_argument("--zed-color-order", default="bgra")
    parser.add_argument("--zed-require-tracking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zed-camera-base-x", type=float, default=0.0)
    parser.add_argument("--zed-camera-base-y", type=float, default=0.88)
    parser.add_argument("--zed-camera-base-z", type=float, default=0.0)
    parser.add_argument("--zed-camera-base-yaw-deg", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--period", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = ZEDFrameProvider(
        ZEDConfig(
            resolution=str(args.zed_resolution),
            fps=int(args.zed_fps),
            depth_mode=str(args.zed_depth_mode),
            output_width=int(args.zed_output_width),
            output_height=int(args.zed_output_height),
            color_order=str(args.zed_color_order),
            require_tracking=bool(args.zed_require_tracking),
        )
    )
    base_from_camera = transform_from_translation_yaw(
        (
            float(args.zed_camera_base_x),
            float(args.zed_camera_base_y),
            float(args.zed_camera_base_z),
        ),
        float(args.zed_camera_base_yaw_deg),
    )
    provider.open()
    try:
        first = provider.grab()
        first_base = world_from_base_from_camera_pose(first.world_from_camera_gl, base_from_camera)
        first_state = state_from_transform(first_base)
        start_position = np.asarray(first_state.position, dtype=np.float32).reshape(3)
        start_yaw = yaw_from_rotation(first_state.rotation)
        started_at = time.monotonic()
        last_at: float | None = None
        index = 0
        deadline = started_at + max(0.0, float(args.duration))
        print(
            json.dumps(
                {
                    "event": "start",
                    "tracking_state": str(first.tracking_state),
                    "start_base_position_m": _position_list(start_position),
                    "start_yaw_deg": round(math.degrees(start_yaw), 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        while time.monotonic() < deadline:
            frame = provider.grab()
            record, last_at = _record(
                index=index,
                started_at=started_at,
                last_at=last_at,
                frame=frame,
                base_from_camera=base_from_camera,
                start_position=start_position,
                start_yaw=start_yaw,
            )
            print(json.dumps(record, sort_keys=True), flush=True)
            index += 1
            time.sleep(max(0.0, float(args.period)))
    finally:
        provider.close()


if __name__ == "__main__":
    main()
