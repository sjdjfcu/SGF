#!/usr/bin/env python3
"""Diagnostic: instance matcher precision at goal-facing and random poses."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

CORRECT_RADIUS_M = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="0-19")
    parser.add_argument("--split", default="val")
    parser.add_argument("--poses-per-episode", type=int, default=12)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def parse_episode_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    args = parse_args()
    import quaternion

    from sgf_habitat.env import TextGoalEpisodeEnv
    from sgf_habitat.goal.instance_image import (
        CameraIntrinsics,
        ImageSpaceMatchEvidence,
        localize_instance_image_evidence,
        ransac_inlier_mask,
    )
    from sgf_habitat.runtime.episode_loop import (
        DEFAULT_NAV_CONFIG_PATH,
        compute_intrinsics,
        default_resolution_for_task_mode,
        image_target_config_from_nav,
        load_nav_config,
        repo_data_path,
    )
    from sgf_habitat.goal.instance_image import XFeatStarMatcher
    from sgf_habitat.runtime.episode_loop import resolve_xfeat_root, REPO_ROOT

    nav_config = load_nav_config(str(DEFAULT_NAV_CONFIG_PATH))
    overlay = (nav_config.get("task_mode_defaults") or {}).get("instance-image") or {}
    nav_config = {**nav_config, **overlay}
    width, height = default_resolution_for_task_mode("instance-image")
    hfov = float(nav_config.get("hfov", 79.0))
    min_depth = float(nav_config.get("min_depth", 0.5))
    max_depth = float(nav_config.get("max_depth", 5.0))
    config = image_target_config_from_nav(nav_config, min_depth=min_depth, max_depth=max_depth)
    matcher = XFeatStarMatcher(
        resolve_xfeat_root(nav_config.get("xfeat_root", str(REPO_ROOT / ".." / "accelerated_features"))),
        top_k=int(nav_config.get("xfeat_top_k", 8000)),
    )
    frame_fx, frame_fy, frame_cx, frame_cy = compute_intrinsics(width, height, hfov)
    frame_intrinsics = CameraIntrinsics(fx=frame_fx, fy=frame_fy, cx=frame_cx, cy=frame_cy)

    records: list[dict] = []
    for episode_id in parse_episode_list(args.episodes):
        env = TextGoalEpisodeEnv(
            config_path="benchmark/nav/instance_imagenav/instance_imagenav_hm3d_v2.yaml",
            dataset_path=repo_data_path("datasets", "instance_imagenav", "hm3d", "v3", "{split}", "{split}.json.gz"),
            scenes_dir=repo_data_path("scene_datasets"),
            split=args.split,
            image_width=width,
            image_height=height,
            hfov=hfov,
            camera_height=float(nav_config.get("camera_height", 0.88)),
            min_depth=min_depth,
            max_depth=max_depth,
            turn_angle=float(nav_config.get("turn_angle", 30.0)),
            forward_step_size=float(nav_config.get("forward_step_size", 0.25)),
            agent_radius=float(nav_config.get("agent_radius", 0.18)),
            max_episode_steps=10,
            success_radius=float(nav_config.get("success_radius", 1.0)),
            task_mode="instance-image",
            episode_id=int(episode_id),
            seed=0,
        )
        try:
            env.reset()
            info = env.current_info
            goal_image = info.goal_image_rgb
            goal_position = np.asarray(info.gt_goal_position_m, dtype=np.float64)
            if goal_image is None or goal_position.size != 3:
                continue
            sim = env.sim
            pathfinder = sim.pathfinder
            rng = np.random.default_rng(1000 + episode_id)
            poses: list[tuple[str, np.ndarray, float]] = []
            attempts = 0
            while len(poses) < 2 * args.poses_per_episode and attempts < 8000:
                attempts += 1
                point = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float64)
                if abs(float(point[1]) - float(goal_position[1])) > 1.0:
                    continue
                distance = float(np.linalg.norm(point[[0, 2]] - goal_position[[0, 2]]))
                if not 0.75 <= distance <= 6.0:
                    continue
                direction = goal_position[[0, 2]] - point[[0, 2]]
                yaw = math.atan2(-float(direction[0]), -float(direction[1]))
                facing_count = sum(1 for c, _, _ in poses if c == "facing")
                random_count = len(poses) - facing_count
                if facing_count <= random_count and facing_count < args.poses_per_episode:
                    poses.append(("facing", point, yaw))
                elif random_count < args.poses_per_episode:
                    poses.append(("random", point, float(rng.uniform(-math.pi, math.pi))))
            for condition, point, yaw in poses:
                rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
                state = sim.get_agent_state()
                state.position = np.asarray(point, dtype=np.float32)
                state.rotation = rotation
                sim.get_agent(0).set_state(state)
                observations = sim.get_sensor_observations()
                rgb = np.asarray(observations["rgb"], dtype=np.uint8)[:, :, :3]
                depth = np.asarray(observations["depth"], dtype=np.float32)
                if depth.ndim == 3:
                    depth = depth[:, :, 0]
                sensor_state = sim.get_agent_state().sensor_states["rgb"]
                world_from_camera = np.eye(4, dtype=np.float32)
                rot = quaternion.as_rotation_matrix(sensor_state.rotation).astype(np.float32)
                world_from_camera[:3, :3] = rot @ np.diag([1.0, -1.0, -1.0]).astype(np.float32)
                world_from_camera[:3, 3] = np.asarray(sensor_state.position, dtype=np.float32)
                row = {
                    "episode": int(episode_id),
                    "condition": condition,
                    "distance_m": float(np.linalg.norm(np.asarray(point)[[0, 2]] - goal_position[[0, 2]])),
                }
                matches = matcher.match(
                    goal_image,
                    rgb,
                    fine_conf_threshold=float(config.fine_conf_threshold),
                    min_coarse_sim=float(config.min_coarse_sim),
                )
                attempted = bool(len(matches.current_xy) >= int(config.min_matches))
                mask = (
                    ransac_inlier_mask(
                        matches.goal_xy,
                        matches.current_xy,
                        reproj_threshold_px=float(config.ransac_reproj_threshold_px),
                    )
                    if attempted
                    else np.zeros((len(matches.current_xy),), dtype=bool)
                )
                result = localize_instance_image_evidence(
                    ImageSpaceMatchEvidence(
                        matches=matches, inlier_mask=mask, ransac_attempted=attempted
                    ),
                    goal_image_shape=tuple(goal_image.shape),
                    depth=depth,
                    intrinsics=frame_intrinsics,
                    world_from_camera_cv=world_from_camera,
                    config=config,
                )
                accepted = bool(result.diagnostics.get("target_updated", False))
                error = None
                if accepted and result.node is not None:
                    landmark = np.asarray(
                        result.node.get("target_centroid_m"), dtype=np.float64
                    ).reshape(-1)
                    if landmark.size == 3 and np.all(np.isfinite(landmark)):
                        error = float(
                            np.linalg.norm(landmark[[0, 2]] - goal_position[[0, 2]])
                        )
                row["single"] = {
                    "accepted": accepted,
                    "inliers": int(result.diagnostics.get("ransac_inliers") or 0),
                    "landmark_error_m": error,
                }
                records.append(row)
        finally:
            env.close()
        print(f"episode {episode_id}: {len(records)} samples", flush=True)

    summarize(records)
    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(records, handle)
    return 0


def summarize(records: list[dict]) -> None:
    print()
    for condition in ("facing", "random"):
        group = [r for r in records if r["condition"] == condition]
        if not group:
            continue
        n = len(group)
        print(f"condition={condition} n={n}")
        accepted = [r for r in group if r["single"]["accepted"]]
        correct = [
            r
            for r in accepted
            if r["single"]["landmark_error_m"] is not None
            and r["single"]["landmark_error_m"] <= CORRECT_RADIUS_M
        ]
        print(
            f"   {'single':8s} accept={len(accepted)/n:.4f} "
            f"correct={len(correct)/n:.4f} "
            f"precision={(len(correct)/len(accepted)) if accepted else float('nan'):.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
