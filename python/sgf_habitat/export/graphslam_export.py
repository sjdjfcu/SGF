from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import quaternion
from PIL import Image


def save_matrix_txt(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in matrix:
            handle.write(" ".join(f"{float(value):.8f}" for value in row))
            handle.write("\n")


def save_u16_pgm(path: Path, depth_mm: np.ndarray) -> None:
    depth_mm = np.asarray(depth_mm, dtype=np.uint16)
    with path.open("wb") as handle:
        handle.write(f"P5\n{depth_mm.shape[1]} {depth_mm.shape[0]}\n65535\n".encode("ascii"))
        handle.write(depth_mm.byteswap().tobytes())


class GraphSLAMSequenceExporter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        max_depth_m: float,
        metadata: dict,
        overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"export directory already exists: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.max_depth_m = float(max_depth_m)
        self.frame_count = 0
        self.metadata = dict(metadata)

        with (self.output_dir / "intrinsics.txt").open("w", encoding="utf-8") as handle:
            handle.write(f"{self.fx:.8f} {self.fy:.8f} {self.cx:.8f} {self.cy:.8f} {self.width} {self.height}\n")
        (self.output_dir / "graphslam_sequence.txt").write_text("habitat_hm3d_v1\n", encoding="utf-8")

        self._trajectory_handle = (self.output_dir / "trajectory.csv").open("w", encoding="utf-8", newline="")
        self._trajectory_writer = csv.writer(self._trajectory_handle)
        self._trajectory_writer.writerow(["frame", "tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz"])

    def export_frame(
        self,
        frame_index: int,
        rgb_rgb: np.ndarray,
        depth_m: np.ndarray,
        world_from_camera_cv: np.ndarray,
        rotation_gl,
    ) -> None:
        rgb_rgb = np.asarray(rgb_rgb, dtype=np.uint8)
        if rgb_rgb.ndim != 3 or rgb_rgb.shape[2] < 3:
            raise RuntimeError("expected rgb observation with shape HxWx3")
        rgb_rgb = rgb_rgb[:, :, :3]

        depth_m = np.asarray(depth_m, dtype=np.float32)
        depth_mm = np.clip(depth_m, 0.0, self.max_depth_m) * 1000.0
        depth_mm = depth_mm.astype(np.uint16)
        depth_mm[depth_m <= 0.0] = 0
        depth_mm[depth_m >= self.max_depth_m] = 0

        pose_cv = np.asarray(world_from_camera_cv, dtype=np.float32)
        if pose_cv.shape != (4, 4):
            raise RuntimeError("world_from_camera_cv must be a 4x4 matrix")

        frame_name = f"frame-{int(frame_index):06d}"
        Image.fromarray(rgb_rgb, mode="RGB").save(self.output_dir / f"{frame_name}.color.png")
        save_u16_pgm(self.output_dir / f"{frame_name}.depth.pgm", depth_mm)
        save_matrix_txt(self.output_dir / f"{frame_name}.pose.txt", pose_cv)

        q = quaternion.as_float_array(rotation_gl)
        self._trajectory_writer.writerow(
            [
                int(frame_index),
                float(pose_cv[0, 3]),
                float(pose_cv[1, 3]),
                float(pose_cv[2, 3]),
                float(q[0]),
                float(q[1]),
                float(q[2]),
                float(q[3]),
            ]
        )
        self.frame_count = max(self.frame_count, int(frame_index) + 1)

    def close(self) -> None:
        if not self._trajectory_handle.closed:
            self._trajectory_handle.close()

        metadata = dict(self.metadata)
        metadata["generator"] = "run_habitat_text_image_goal_nav.py"
        metadata["num_frames"] = self.frame_count
        metadata["max_depth_m"] = self.max_depth_m
        metadata["intrinsics"] = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
        }
        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
