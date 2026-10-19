#!/usr/bin/env python3
"""Estimate horizontal ZED/base offsets offline from saved in-place turns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from sgf_habitat.real.calibration import fit_horizontal_camera_offset


def rotation_runs(diagnostics_path: Path) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for line in diagnostics_path.read_text().splitlines():
        match = re.match(r"step=(\d+)\b", line)
        if match is None or " real_step=" not in line:
            continue
        step = int(match.group(1))
        motion = json.loads(line.split(" real_step=", 1)[1])
        is_turn = (
            motion.get("type") in ("real_turn_left", "real_turn_right")
            and motion.get("reason") == "target"
            and motion.get("tracking_state") == "OK"
        )
        if current and (not is_turn or step != current[-1] + 1):
            runs.append(current + [current[-1] + 1])
            current = []
        if is_turn:
            current.append(step)
    if current:
        runs.append(current + [current[-1] + 1])
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgf-input-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-log", type=Path)
    parser.add_argument("--camera-base-y", type=float, required=True, help="Known mounting height in meters.")
    parser.add_argument("--camera-base-yaw-deg", type=float, default=0.0, help="Known mounting yaw; cannot be solved from pure turns alone.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dump = args.sgf_input_dir.expanduser().resolve()
    diagnostics = (args.diagnostics_log or dump.parent / "diagnostics.log").expanduser()
    candidates = []
    graphics_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    for steps in rotation_runs(diagnostics):
        paths = [dump / "pose" / f"step_{step:04d}.json" for step in steps]
        available = [(step, path) for step, path in zip(steps, paths) if path.exists()]
        if len(available) < 8:
            continue
        poses = [json.loads(path.read_text())["world_from_camera_cv"] for _, path in available]
        result = fit_horizontal_camera_offset(
            np.asarray(poses) @ graphics_to_cv,
            camera_base_y_m=args.camera_base_y,
            camera_base_yaw_deg=args.camera_base_yaw_deg,
        )
        result["pose_steps"] = [step for step, _ in available]
        candidates.append(result)
    if not candidates:
        parser.error("no logged pure-turn interval has at least 8 saved poses")
    result = max(candidates, key=lambda item: (item["accepted"], item["rotation_coverage_deg"], item["inlier_count"]))
    report = {
        "method": "stationary_pivot_fit_with_known_height_and_yaw",
        "sgf_input_dir": str(dump),
        "diagnostics_log": str(diagnostics.resolve()),
        "selected": result,
        "candidate_runs": candidates,
    }
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if not result["accepted"]:
        print("Calibration rejected; see rejection_reasons. No navigation settings were changed.", file=sys.stderr)
        return 2
    print(
        f"--zed-camera-base-x {result['camera_base_x_m']:.5f} "
        f"--zed-camera-base-y {result['camera_base_y_m']:.5f} "
        f"--zed-camera-base-z {result['camera_base_z_m']:.5f} "
        f"--zed-camera-base-yaw-deg {result['camera_base_yaw_deg']:.5f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
