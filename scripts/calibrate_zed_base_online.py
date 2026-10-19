#!/usr/bin/env python3
"""Drive Kobuki through live ZED pivot and forward calibration motions."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from sgf_habitat.real.kobuki import KobukiBaseController
from sgf_habitat.real.online_calibration import OnlineCalibrationConfig, OnlineExtrinsicsCalibrator
from sgf_habitat.real.zed import ZEDConfig, ZEDFrameProvider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kobuki-device", default="/dev/kobuki")
    parser.add_argument("--kobuki-install", default="~/kobuki/install")
    parser.add_argument("--zed-resolution", default="HD720")
    parser.add_argument("--zed-output-width", type=int, default=1280)
    parser.add_argument("--zed-output-height", type=int, default=720)
    parser.add_argument("--zed-camera-base-y", type=float, required=True)
    parser.add_argument("--angular-speed", type=float, default=0.3)
    parser.add_argument("--linear-speed", type=float, default=0.1)
    parser.add_argument("--straight-distance", type=float, default=0.3, help="Meters per segment; two forward segments are executed.")
    parser.add_argument("--agent-radius", type=float, default=0.18)
    parser.add_argument("--output", type=Path, default=Path("outputs/real/zed_base_calibration.json"))
    args = parser.parse_args()
    config = OnlineCalibrationConfig(
        camera_height_m=args.zed_camera_base_y,
        angular_speed_radps=args.angular_speed,
        linear_speed_mps=args.linear_speed,
        straight_distance_m=args.straight_distance,
        agent_radius_m=args.agent_radius,
    )
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    samples_path = output.with_suffix(".samples.jsonl")
    camera = ZEDFrameProvider(ZEDConfig(
        resolution=args.zed_resolution, output_width=args.zed_output_width,
        output_height=args.zed_output_height, require_tracking=True,
    ))
    base = KobukiBaseController(device=args.kobuki_device, kobuki_install=args.kobuki_install)
    result = {"accepted": False, "rejection_reasons": ["calibration_incomplete"]}
    runner = None

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, interrupt)
    print(
        f"Live calibration: left 360 deg, right 360 deg, then two {args.straight_distance:.2f} m forward segments. "
        "Forward motion uses ZED heading feedback. "
        "Use a level, clear area with the camera facing within 30 deg of forward. Ctrl+C stops motion.",
        flush=True,
    )
    try:
        with samples_path.open("w", encoding="utf-8") as handle:
            camera.open()
            base.start()
            runner = OnlineExtrinsicsCalibrator(
                camera, base, config,
                record=lambda row: handle.write(json.dumps(row, separators=(",", ":")) + "\n"),
                progress=lambda message: print(message, flush=True),
            )
            result = runner.run()
    except KeyboardInterrupt:
        result = {"accepted": False, "rejection_reasons": ["interrupted"]}
    except Exception as exc:
        result = {"accepted": False, "rejection_reasons": [str(exc)]}
    finally:
        try:
            base.close()
        finally:
            try:
                camera.close()
            finally:
                signal.signal(signal.SIGTERM, previous_sigterm)
                if runner is not None:
                    result.update(runner.diagnostics)
                result.update({
                    "motion_config": asdict(config), "samples_path": str(samples_path.resolve()),
                    "last_phase": None if runner is None else runner.phase,
                })
                output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if result["accepted"]:
        for key in ("camera_base_x_m", "camera_base_y_m", "camera_base_z_m", "camera_base_yaw_deg"):
            print(f"{key}={result[key]:.6f}")
        print(f"Navigation option: --zed-extrinsics-path {output}")
        return 0
    print(f"Calibration rejected: {result['rejection_reasons']}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
