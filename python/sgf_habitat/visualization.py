from __future__ import annotations

import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import quaternion

from .diagnostics import (
    action_name,
    navigation_trace_position,
    reachable_mask_from_start,
    summarize_map_stats,
)
from .goal.grounding import label_in_query_set


STEP_CANVAS_WIDTH = 1520
STEP_CANVAS_HEIGHT = 860
FINAL_CANVAS_WIDTH = 1600
FINAL_CANVAS_HEIGHT = 920
BACKGROUND_BGR = (241, 238, 233)
PANEL_BGR = (250, 247, 243)
TEXT_BGR = (39, 43, 47)
MUTED_TEXT_BGR = (99, 104, 111)
BORDER_BGR = (207, 197, 186)
FREE_BGR = np.array([238, 234, 225], dtype=np.float32)
UNREACHABLE_FREE_BGR = np.array([214, 210, 203], dtype=np.float32)
UNKNOWN_BGR = np.array([225, 232, 238], dtype=np.float32)
OCCUPIED_BGR = np.array([58, 64, 70], dtype=np.float32)
VISITED_BGR = np.array([126, 186, 179], dtype=np.float32)
FRONTIER_BGR = np.array([53, 170, 245], dtype=np.float32)
UNCOVERED_BGR = np.array([92, 109, 126], dtype=np.float32)
TARGET_BGR = (0, 147, 255)
GOAL_BGR = (255, 188, 86)
REJECTED_BGR = (83, 83, 214)
START_BGR = (88, 179, 92)
END_BGR = (70, 60, 220)
STOP_BGR = (57, 64, 220)
PATH_START_BGR = np.array([222, 139, 61], dtype=np.float32)
PATH_END_BGR = np.array([62, 85, 227], dtype=np.float32)


def _panel(canvas: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> np.ndarray:
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_BGR, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), PANEL_BGR, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BORDER_BGR, 2, cv2.LINE_AA)
    cv2.putText(canvas, title, (x + 18, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_BGR, 2, cv2.LINE_AA)
    return canvas[y + 48 : y + h - 12, x + 12 : x + w - 12]


def _draw_text_block(
    image: np.ndarray,
    lines: list[str],
    *,
    origin: tuple[int, int] = (0, 0),
    line_height: int = 24,
    color: tuple[int, int, int] = TEXT_BGR,
    font_scale: float = 0.58,
    thickness: int = 1,
) -> None:
    x, y = origin
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _wrap_text(text: str, width: int = 34) -> list[str]:
    words = text.split()
    if not words:
        return ["-"]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) <= width:
            current += " " + word
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def _rgb_to_bgr_image(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image = image[:, :, :3]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.dtype != np.uint8:
            image_f = image.astype(np.float32)
            if image_f.size and float(np.nanmax(image_f)) <= 1.0:
                image_f *= 255.0
            image = np.clip(image_f, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(image)


def _resize_with_scale(
    image_bgr: np.ndarray,
    *,
    target_height: int,
    max_width: int,
) -> tuple[np.ndarray, float, float]:
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0:
        return np.zeros((1, 1, 3), dtype=np.uint8), 1.0, 1.0
    scale = min(float(target_height) / float(height), float(max_width) / float(width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, new_width / float(width), new_height / float(height)


def _match_mask(mask: np.ndarray | None, count: int) -> np.ndarray:
    result = np.zeros((count,), dtype=bool)
    if mask is None:
        return result
    values = np.asarray(mask, dtype=bool).reshape(-1)
    size = min(count, int(values.size))
    if size > 0:
        result[:size] = values[:size]
    return result


def _diagnostic_text(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def render_image_match_canvas(
    *,
    goal_image_rgb: np.ndarray,
    current_image_rgb: np.ndarray,
    match_debug,
    diagnostics: dict,
    step_index: int,
    target_node: dict | None = None,
    max_lines: int = 300,
) -> np.ndarray:
    goal_bgr = _rgb_to_bgr_image(goal_image_rgb)
    current_bgr = _rgb_to_bgr_image(current_image_rgb)
    display_height = 560
    max_panel_width = 740
    goal_view, goal_sx, goal_sy = _resize_with_scale(
        goal_bgr,
        target_height=display_height,
        max_width=max_panel_width,
    )
    current_view, current_sx, current_sy = _resize_with_scale(
        current_bgr,
        target_height=display_height,
        max_width=max_panel_width,
    )

    margin = 24
    gap = 34
    header_h = 154
    footer_h = 64
    match_h = max(goal_view.shape[0], current_view.shape[0])
    canvas_w = max(900, margin * 2 + goal_view.shape[1] + gap + current_view.shape[1])
    canvas_h = header_h + match_h + footer_h
    canvas = np.full((canvas_h, canvas_w, 3), BACKGROUND_BGR, dtype=np.uint8)

    left_x = margin
    right_x = margin + goal_view.shape[1] + gap
    image_y = header_h
    canvas[image_y : image_y + goal_view.shape[0], left_x : left_x + goal_view.shape[1]] = goal_view
    canvas[image_y : image_y + current_view.shape[0], right_x : right_x + current_view.shape[1]] = current_view
    cv2.rectangle(
        canvas,
        (left_x, image_y),
        (left_x + goal_view.shape[1], image_y + goal_view.shape[0]),
        BORDER_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        (right_x, image_y),
        (right_x + current_view.shape[1], image_y + current_view.shape[0]),
        BORDER_BGR,
        1,
        cv2.LINE_AA,
    )

    raw = diagnostics.get("raw_matches", 0)
    fine = diagnostics.get("fine_conf_filtered_matches", 0)
    inliers = diagnostics.get("ransac_inliers", 0)
    projected = diagnostics.get("projected_point_count", 0)
    cluster_size = diagnostics.get("cluster_size", 0)
    accepted = bool(diagnostics.get("target_updated", False))
    reject_reason = diagnostics.get("reject_reason") or "-"
    title = f"XFeat* feature matches | step {step_index:04d}"
    status = "accepted target" if accepted else f"rejected: {reject_reason}"
    metric_line = (
        f"raw={raw} fine={fine} ransac={inliers} projected={projected} cluster={cluster_size} "
        f"conf_sum={_diagnostic_text(diagnostics.get('cluster_confidence_sum', diagnostics.get('confidence_sum')))} "
        f"mean_conf={_diagnostic_text(diagnostics.get('cluster_mean_confidence', diagnostics.get('mean_confidence')))} "
        f"mean_coarse={_diagnostic_text(diagnostics.get('cluster_mean_coarse_sim', diagnostics.get('mean_coarse_sim')))}"
    )
    target_line = ""
    if target_node is not None:
        target_line = (
            f"target id={target_node.get('id', '-')} score={_diagnostic_text(target_node.get('match_score'))} "
            f"surfels={target_node.get('num_surfels', '-')}"
        )

    cv2.putText(canvas, title, (margin, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_BGR, 2, cv2.LINE_AA)
    _draw_text_block(
        canvas,
        [status, metric_line] + ([target_line] if target_line else []),
        origin=(margin, 72),
        line_height=20,
        color=MUTED_TEXT_BGR,
        font_scale=0.52,
    )
    cv2.putText(
        canvas,
        "goal image",
        (left_x, image_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_BGR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "current frame",
        (right_x, image_y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_BGR,
        1,
        cv2.LINE_AA,
    )

    matches = match_debug.matches
    goal_xy = np.asarray(matches.goal_xy, dtype=np.float32).reshape(-1, 2)
    current_xy = np.asarray(matches.current_xy, dtype=np.float32).reshape(-1, 2)
    count = min(len(goal_xy), len(current_xy))
    if count > 0:
        goal_xy = goal_xy[:count]
        current_xy = current_xy[:count]
        fine_conf = np.asarray(matches.fine_conf, dtype=np.float32).reshape(-1)
        confidence = np.zeros((count,), dtype=np.float32)
        confidence[: min(count, int(fine_conf.size))] = fine_conf[: min(count, int(fine_conf.size))]
        inlier_mask = _match_mask(getattr(match_debug, "inlier_mask", None), count)
        valid_depth_mask = _match_mask(getattr(match_debug, "valid_depth_mask", None), count)
        cluster_mask = _match_mask(getattr(match_debug, "cluster_mask", None), count)
        priority = (
            cluster_mask.astype(np.int32) * 4
            + inlier_mask.astype(np.int32) * 3
            + valid_depth_mask.astype(np.int32) * 2
        )
        order = np.lexsort((-confidence, -priority))
        limit = max(1, int(max_lines))
        selected = order[: min(count, limit)]

        color_outlier = (138, 141, 147)
        color_valid_depth = (52, 142, 232)
        color_inlier = (74, 190, 214)
        color_cluster = (78, 184, 96)
        match_line_thickness = 1
        point_radius = 2
        for index in selected[::-1]:
            left_point = (
                int(round(left_x + goal_xy[index, 0] * goal_sx)),
                int(round(image_y + goal_xy[index, 1] * goal_sy)),
            )
            right_point = (
                int(round(right_x + current_xy[index, 0] * current_sx)),
                int(round(image_y + current_xy[index, 1] * current_sy)),
            )
            if cluster_mask[index]:
                color = color_cluster
                thickness = match_line_thickness
            elif valid_depth_mask[index]:
                color = color_valid_depth
                thickness = match_line_thickness
            elif inlier_mask[index]:
                color = color_inlier
                thickness = match_line_thickness
            else:
                color = color_outlier
                thickness = match_line_thickness
            cv2.line(canvas, left_point, right_point, color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, left_point, point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, right_point, point_radius, color, -1, cv2.LINE_AA)

        if count > limit:
            cv2.putText(
                canvas,
                f"showing top {limit} of {count} filtered matches by confidence and target support",
                (margin, canvas_h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                MUTED_TEXT_BGR,
                1,
                cv2.LINE_AA,
            )

    legend_y = canvas_h - 42
    legend_items = [
        ("cluster target", (78, 184, 96)),
        ("valid depth", (52, 142, 232)),
        ("RANSAC inlier", (74, 190, 214)),
        ("filtered/outlier", (138, 141, 147)),
    ]
    legend_x = margin
    for label, color in legend_items:
        cv2.line(canvas, (legend_x, legend_y), (legend_x + 32, legend_y), color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (legend_x + 40, legend_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            MUTED_TEXT_BGR,
            1,
            cv2.LINE_AA,
        )
        legend_x += 190

    return canvas


def save_image_match_visualization(
    *,
    output_path: str | Path,
    goal_image_rgb: np.ndarray,
    current_image_rgb: np.ndarray,
    match_debug,
    diagnostics: dict,
    step_index: int,
    target_node: dict | None = None,
    max_lines: int = 300,
) -> np.ndarray:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = render_image_match_canvas(
        goal_image_rgb=goal_image_rgb,
        current_image_rgb=current_image_rgb,
        match_debug=match_debug,
        diagnostics=diagnostics,
        step_index=step_index,
        target_node=target_node,
        max_lines=max_lines,
    )
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write image match visualization {output_path}")
    return canvas


class ImageMatchVideoWriter:
    def __init__(self, video_path: str | Path, *, fps: float = 5.0) -> None:
        self.video_path = Path(video_path).expanduser()
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(0.1, float(fps))
        self.writer: cv2.VideoWriter | None = None
        self.frame_size: tuple[int, int] | None = None
        self.frame_count = 0

    def _init_writer(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            return
        height, width = frame.shape[:2]
        self.frame_size = (int(width), int(height))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.video_path), fourcc, self.fps, self.frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"failed to open image match video writer for {self.video_path}")

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        frame = np.asarray(frame_bgr, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("image match video frame must be a BGR uint8 image")
        self._init_writer(frame)
        if self.frame_size is None:
            return
        width, height = self.frame_size
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        assert self.writer is not None
        self.writer.write(frame)
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if image.shape[:2] != mask.shape:
        return
    image_f = image.astype(np.float32)
    image_f[mask] = image_f[mask] * (1.0 - alpha) + color * alpha
    image[:] = np.clip(image_f, 0, 255).astype(np.uint8)


def _render_map_base(
    navigation_map,
    *,
    reachable_mask: np.ndarray | None = None,
    visited_mask: np.ndarray | None = None,
    frontier_mask: np.ndarray | None = None,
    uncovered_mask: np.ndarray | None = None,
) -> np.ndarray:
    observed_mask = np.asarray(navigation_map.observed_mask(), dtype=bool)
    free_mask = np.asarray(navigation_map.free_mask(), dtype=bool)
    occupied_mask = np.asarray(navigation_map.occupied_mask(), dtype=bool)
    unknown_mask = ~observed_mask

    image = np.zeros((*free_mask.shape, 3), dtype=np.uint8)
    image[:] = UNKNOWN_BGR.astype(np.uint8)
    _blend_mask(image, free_mask, FREE_BGR, 1.0)
    if reachable_mask is not None and reachable_mask.shape == free_mask.shape:
        _blend_mask(image, free_mask & ~reachable_mask, UNREACHABLE_FREE_BGR, 1.0)
    _blend_mask(image, unknown_mask, UNKNOWN_BGR, 1.0)
    _blend_mask(image, occupied_mask, OCCUPIED_BGR, 1.0)

    if uncovered_mask is not None and uncovered_mask.shape == free_mask.shape:
        _blend_mask(image, uncovered_mask, UNCOVERED_BGR, 0.22)
    if visited_mask is not None and visited_mask.shape == free_mask.shape:
        _blend_mask(image, visited_mask & free_mask, VISITED_BGR, 0.50)
    if frontier_mask is not None and frontier_mask.shape == free_mask.shape and np.any(frontier_mask):
        frontier_vis = cv2.dilate(frontier_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
        _blend_mask(image, frontier_vis, FRONTIER_BGR, 0.55)
    return image


def _cell_to_panel(row: int, col: int, map_shape: tuple[int, int], panel_size: tuple[int, int]) -> tuple[int, int] | None:
    rows, cols = map_shape
    panel_w, panel_h = panel_size
    if rows <= 0 or cols <= 0:
        return None
    if row < 0 or row >= rows or col < 0 or col >= cols:
        return None
    x = int(round(col / max(1, cols - 1) * (panel_w - 1)))
    y = int(round(row / max(1, rows - 1) * (panel_h - 1)))
    return x, y


def _world_to_panel(navigation_map, world_xyz: np.ndarray | list[float] | None, panel_size: tuple[int, int]) -> tuple[int, int] | None:
    if world_xyz is None:
        return None
    xyz = np.asarray(world_xyz, dtype=np.float32).reshape(-1)
    if xyz.size != 3 or not np.all(np.isfinite(xyz)):
        return None
    row, col = navigation_map.world_to_cell(xyz)
    return _cell_to_panel(row, col, navigation_map.shape, panel_size)


def _draw_path(panel: np.ndarray, navigation_map, positions: list[np.ndarray]) -> None:
    if len(positions) < 2:
        return
    panel_size = (panel.shape[1], panel.shape[0])
    for index, (prev, current) in enumerate(zip(positions[:-1], positions[1:]), start=1):
        prev_xy = _world_to_panel(navigation_map, prev, panel_size)
        curr_xy = _world_to_panel(navigation_map, current, panel_size)
        if prev_xy is None or curr_xy is None:
            continue
        t = index / max(1, len(positions) - 1)
        color = PATH_START_BGR * (1.0 - t) + PATH_END_BGR * t
        cv2.line(panel, prev_xy, curr_xy, tuple(int(round(v)) for v in color), 3, cv2.LINE_AA)


def _draw_rejected(panel: np.ndarray, navigation_map, rejected_positions: list[np.ndarray]) -> None:
    panel_size = (panel.shape[1], panel.shape[0])
    for position in rejected_positions:
        xy = _world_to_panel(navigation_map, position, panel_size)
        if xy is None:
            continue
        cv2.line(panel, (xy[0] - 7, xy[1] - 7), (xy[0] + 7, xy[1] + 7), REJECTED_BGR, 2, cv2.LINE_AA)
        cv2.line(panel, (xy[0] - 7, xy[1] + 7), (xy[0] + 7, xy[1] - 7), REJECTED_BGR, 2, cv2.LINE_AA)


def _draw_nodes(panel: np.ndarray, navigation_map, nodes: list[dict], goal_queries: list[str]) -> None:
    panel_size = (panel.shape[1], panel.shape[0])
    for node in nodes:
        centroid = np.asarray(node.get("centroid_m", (np.nan, np.nan, np.nan)), dtype=np.float32)
        xy = _world_to_panel(navigation_map, centroid, panel_size)
        if xy is None:
            continue
        label = str(node.get("label", ""))
        if label_in_query_set(label, goal_queries):
            cv2.circle(panel, xy, 5, TARGET_BGR, -1, cv2.LINE_AA)
            cv2.circle(panel, xy, 8, (240, 240, 240), 1, cv2.LINE_AA)
        else:
            cv2.circle(panel, xy, 3, (120, 126, 132), -1, cv2.LINE_AA)


def _draw_goal_marker(panel: np.ndarray, navigation_map, goal_position: np.ndarray | None, color: tuple[int, int, int], radius: int) -> None:
    xy = _world_to_panel(navigation_map, goal_position, (panel.shape[1], panel.shape[0]))
    if xy is None:
        return
    cv2.circle(panel, xy, radius, color, 2, cv2.LINE_AA)


def _draw_agent(panel: np.ndarray, navigation_map, agent_state, color: tuple[int, int, int]) -> None:
    panel_size = (panel.shape[1], panel.shape[0])
    agent_position = np.asarray(agent_state.position, dtype=np.float32)
    xy = _world_to_panel(navigation_map, agent_position, panel_size)
    if xy is None:
        return
    rotation_matrix = quaternion.as_rotation_matrix(agent_state.rotation).astype(np.float32)
    forward = rotation_matrix @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    tip = agent_position + forward * 0.65
    tip_xy = _world_to_panel(navigation_map, tip, panel_size)
    if tip_xy is None:
        cv2.circle(panel, xy, 8, color, -1, cv2.LINE_AA)
        return
    dx = float(tip_xy[0] - xy[0])
    dy = float(tip_xy[1] - xy[1])
    angle = math.atan2(dy, dx)
    size = 16
    points = np.array(
        [
            (
                int(xy[0] + size * math.cos(angle)),
                int(xy[1] + size * math.sin(angle)),
            ),
            (
                int(xy[0] + size * 0.70 * math.cos(angle + 2.5)),
                int(xy[1] + size * 0.70 * math.sin(angle + 2.5)),
            ),
            (
                int(xy[0] + size * 0.55 * math.cos(angle + math.pi)),
                int(xy[1] + size * 0.55 * math.sin(angle + math.pi)),
            ),
            (
                int(xy[0] + size * 0.70 * math.cos(angle - 2.5)),
                int(xy[1] + size * 0.70 * math.sin(angle - 2.5)),
            ),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(panel, points, color, cv2.LINE_AA)
    cv2.polylines(panel, [points], True, (245, 245, 245), 1, cv2.LINE_AA)


def _render_map_panel(
    *,
    navigation_map,
    positions: list[np.ndarray],
    agent_state,
    nodes: list[dict],
    goal_queries: list[str],
    canonical_goal: str,
    target_node: dict | None,
    current_goal: np.ndarray | None,
    rejected_positions: list[np.ndarray],
    panel_size: tuple[int, int],
    coverage_mask: np.ndarray | None = None,
    reachable_mask: np.ndarray | None = None,
) -> np.ndarray:
    if navigation_map is None or not all(navigation_map.shape):
        panel = np.full((panel_size[1], panel_size[0], 3), PANEL_BGR, dtype=np.uint8)
        _draw_text_block(panel, ["Navigation map unavailable"], origin=(24, 40))
        return panel
    if reachable_mask is None:
        free_mask = np.asarray(navigation_map.free_mask(), dtype=bool)
        start_cell = navigation_map.world_to_cell(positions[0]) if positions else (0, 0)
        reachable_mask = reachable_mask_from_start(free_mask, start_cell)
    frontier_mask = np.asarray(navigation_map.frontier_cells(), dtype=bool)
    map_image = _render_map_base(
        navigation_map,
        reachable_mask=reachable_mask,
        visited_mask=coverage_mask,
        frontier_mask=frontier_mask,
        uncovered_mask=None,
    )
    panel = cv2.resize(map_image, panel_size, interpolation=cv2.INTER_NEAREST)
    _draw_path(panel, navigation_map, positions)
    _draw_nodes(panel, navigation_map, nodes, goal_queries)
    _draw_rejected(panel, navigation_map, rejected_positions)
    if current_goal is not None:
        _draw_goal_marker(panel, navigation_map, current_goal, GOAL_BGR, 11)
    if target_node is not None:
        _draw_goal_marker(panel, navigation_map, target_node.get("goal_position"), TARGET_BGR, 13)
    if positions:
        _draw_goal_marker(panel, navigation_map, positions[0], START_BGR, 10)
    _draw_agent(panel, navigation_map, agent_state, END_BGR)
    return panel


def _extract_trace_positions(trace_records: list[dict]) -> list[np.ndarray]:
    return [
        np.asarray(position, dtype=np.float32)
        for record in trace_records
        if (position := navigation_trace_position(record)) is not None
    ]


def _stop_positions(trace_records: list[dict]) -> list[np.ndarray]:
    positions = []
    for record in trace_records:
        if not record.get("stop", {}).get("executed_stop"):
            continue
        position = navigation_trace_position(record)
        if position is None:
            continue
        positions.append(np.asarray(position, dtype=np.float32))
    return positions


def _closest_target_position(trace_records: list[dict]) -> np.ndarray | None:
    best_distance = float("inf")
    best_position = None
    for record in trace_records:
        target = record.get("selected_target")
        if not target or target.get("distance_m") is None:
            continue
        distance = float(target["distance_m"])
        if distance < best_distance:
            best_distance = distance
            best_position = target.get("goal_position_m")
    if best_position is None:
        return None
    return np.asarray(best_position, dtype=np.float32)


def render_episode_canvas(
    *,
    episode_info,
    rgb_bgr: np.ndarray,
    agent_state,
    path_history: list[np.ndarray],
    nodes: list[dict],
    canonical_goal: str,
    goal_queries: list[str],
    target_node: dict | None,
    current_goal: np.ndarray | None,
    step_index: int,
    mode: str,
    action: int,
    runtime_stats: dict,
    stop_called: bool,
    target_found: bool,
    navigation_map,
    step_record: dict | None = None,
    rejected_positions: list[np.ndarray] | None = None,
    task_mode: str = "text",
) -> np.ndarray:
    rejected_positions = rejected_positions or []
    positions = [np.asarray(position, dtype=np.float32) for position in path_history]
    canvas = np.full((STEP_CANVAS_HEIGHT, STEP_CANVAS_WIDTH, 3), BACKGROUND_BGR, dtype=np.uint8)

    map_panel = _panel(canvas, 24, 24, 980, 812, "SGF Navigation Map")
    side_top = _panel(canvas, 1030, 24, 466, 328, "Observation")
    side_mid = _panel(canvas, 1030, 376, 466, 212, "Status")
    side_bot = _panel(canvas, 1030, 612, 466, 224, "Goal / Target")

    if positions:
        map_panel[:] = _render_map_panel(
            navigation_map=navigation_map,
            positions=positions,
            agent_state=agent_state,
            nodes=nodes,
            goal_queries=goal_queries,
            canonical_goal=canonical_goal,
            target_node=target_node,
            current_goal=current_goal,
            rejected_positions=rejected_positions,
            panel_size=(map_panel.shape[1], map_panel.shape[0]),
        )

    observation = cv2.resize(rgb_bgr, (side_top.shape[1], side_top.shape[0]), interpolation=cv2.INTER_AREA)
    side_top[:] = observation

    map_stats = (
        summarize_map_stats(navigation_map)
        if navigation_map is not None and all(navigation_map.shape)
        else {}
    )
    planner = {} if step_record is None else step_record.get("planner", {})
    target_summary = {} if step_record is None else (step_record.get("selected_target") or {})
    stop_info = {} if step_record is None else step_record.get("stop", {})
    status_lines = [
        f"step {step_index:03d}   mode {mode}",
        f"action {action_name(action)}   target_found {'yes' if target_found else 'no'}",
        f"goal_dist {planner.get('goal_distance_m')}   target_dist {target_summary.get('distance_m')}",
        f"free {map_stats.get('free_cells', 0)}   occupied {map_stats.get('occupied_cells', 0)}",
        f"frontiers {map_stats.get('frontier_cells', 0)}   nodes {len(nodes)}",
        f"stalled {planner.get('stalled_count', 0)}   replan_fail {planner.get('planner_replan_failed', False)}",
        f"stop_called {'yes' if stop_called else 'no'}   suppressed {', '.join(stop_info.get('suppressed_reasons', [])) or '-'}",
        f"yolo_ms {runtime_stats.get('yolo_ms', 0.0):.1f}   nav_free_ms {runtime_stats.get('navigation_free_ms', 0.0):.1f}",
    ]
    _draw_text_block(side_mid, status_lines, origin=(8, 22), line_height=24)

    goal_lines = []
    if task_mode != "instance-image":
        goal_lines.extend([f"goal: {canonical_goal}", f"query: {', '.join(goal_queries) if goal_queries else '-'}", ""])
    goal_lines.extend(_wrap_text(episode_info.text_goal, width=34)[:5])
    if target_summary:
        goal_lines.extend(
            [
                "",
                f"target label: {target_summary.get('label', '-')}",
                f"target conf: {target_summary.get('semantic_confidence', '-')}",
                f"target surfels: {target_summary.get('num_surfels', '-')}",
            ]
        )
    _draw_text_block(side_bot, goal_lines, origin=(8, 20), line_height=23)

    cv2.putText(
        canvas,
        f"Episode {episode_info.episode_id} | {episode_info.goal_name}",
        (28, STEP_CANVAS_HEIGHT - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        MUTED_TEXT_BGR,
        2,
        cv2.LINE_AA,
    )
    return canvas


def render_final_summary_canvas(
    *,
    episode_info,
    final_rgb_bgr: np.ndarray,
    final_agent_state,
    navigation_map,
    trace_records: list[dict],
    canonical_goal: str,
    goal_queries: list[str],
    analysis,
    result_payload: dict,
    rejected_positions: list[np.ndarray],
    success_radius_m: float,
) -> np.ndarray:
    _ = success_radius_m
    issue_aliases = {
        "stagnation": "stagnate",
        "trapped": "trapped",
        "oscillation": "oscillate",
        "unable_to_escape": "no_escape",
        "repeated_exploration": "repeat",
        "wrong_stop": "wrong_stop",
        "near_target_no_stop": "miss_stop",
        "target_found_but_no_stop": "no_stop",
    }
    canvas = np.full((FINAL_CANVAS_HEIGHT, FINAL_CANVAS_WIDTH, 3), BACKGROUND_BGR, dtype=np.uint8)
    map_panel = _panel(canvas, 24, 24, 1026, 872, "Final SGF Navigation Map")
    rgb_panel = _panel(canvas, 1076, 24, 500, 318, "Final Observation")
    summary_panel = _panel(canvas, 1076, 366, 500, 172, "Outcome Summary")
    map_stats_panel = _panel(canvas, 1076, 562, 500, 334, "Map Statistics")

    positions = _extract_trace_positions(trace_records)
    # The caller supplies the final pose in the same frame as the SGF map.
    terminal_position = final_agent_state.position
    if terminal_position is not None:
        terminal = np.asarray(terminal_position, dtype=np.float32)
        if terminal.size == 3 and np.all(np.isfinite(terminal)):
            if not positions or float(np.linalg.norm(positions[-1] - terminal)) > 1e-4:
                positions.append(terminal)

    if positions:
        map_panel[:] = _render_map_panel(
            navigation_map=navigation_map,
            positions=positions,
            agent_state=final_agent_state,
            nodes=[],
            goal_queries=goal_queries,
            canonical_goal=canonical_goal,
            target_node=None,
            current_goal=_closest_target_position(trace_records),
            rejected_positions=rejected_positions,
            panel_size=(map_panel.shape[1], map_panel.shape[0]),
            coverage_mask=analysis.nav_explored_mask,
            reachable_mask=analysis.map_reachable_mask,
        )
        for stop_position in (_stop_positions(trace_records) if navigation_map is not None and all(navigation_map.shape) else []):
            xy = _world_to_panel(navigation_map, stop_position, (map_panel.shape[1], map_panel.shape[0]))
            if xy is None:
                continue
            cv2.drawMarker(map_panel, xy, STOP_BGR, cv2.MARKER_STAR, 18, 2, cv2.LINE_AA)

    resized_rgb = cv2.resize(final_rgb_bgr, (rgb_panel.shape[1], rgb_panel.shape[0]), interpolation=cv2.INTER_AREA)
    rgb_panel[:] = resized_rgb

    metrics = result_payload.get("metrics", {})
    issue_flags = analysis.summary.get("issue_flags", {})
    enabled_issues = [name for name, enabled in issue_flags.items() if enabled]
    issue_summary = "none"
    if enabled_issues:
        visible_issues = [issue_aliases.get(name, name) for name in enabled_issues[:2]]
        hidden_count = max(0, len(enabled_issues) - len(visible_issues))
        issue_summary = ", ".join(visible_issues)
        if hidden_count > 0:
            issue_summary += f" +{hidden_count}"
    summary_lines = [
        f"ep {episode_info.episode_id}   goal {episode_info.goal_name}",
        f"{'success' if float(metrics.get('success', 0.0)) > 0.0 else 'fail'}   steps {result_payload.get('steps', 0)}   dist {float(metrics.get('distance_to_goal', 0.0)):.2f}m",
        f"revisit {analysis.summary.get('revisit_ratio', 0.0):.2f}   frontier {analysis.summary.get('final_frontier_cells', 0)}",
        f"target {'yes' if result_payload.get('target_found') else 'no'}   stop {'yes' if result_payload.get('stop_called') else 'no'}   min_t {analysis.summary.get('min_target_distance_m')}",
        f"issues: {issue_summary}",
    ]
    _draw_text_block(summary_panel, summary_lines, origin=(8, 20), line_height=22, font_scale=0.56, thickness=2)

    if navigation_map is not None and all(navigation_map.shape):
        map_stats = summarize_map_stats(navigation_map)
        _draw_text_block(
            map_stats_panel,
            [
                "Source: SGF navigation map",
                f"Observed cells: {map_stats['observed_cells']}",
                f"Free cells: {map_stats['free_cells']}",
                f"Occupied cells: {map_stats['occupied_cells']}",
                f"Frontier cells: {map_stats['frontier_cells']}",
            ],
            origin=(8, 24), line_height=30,
        )

    cv2.putText(
        canvas,
        f"Goal: {canonical_goal} | Queries: {', '.join(goal_queries) if goal_queries else '-'}",
        (28, FINAL_CANVAS_HEIGHT - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        MUTED_TEXT_BGR,
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_final_summary_image(
    *,
    output_path: str | Path,
    episode_info,
    final_rgb_bgr: np.ndarray,
    final_agent_state,
    navigation_map,
    trace_records: list[dict],
    canonical_goal: str,
    goal_queries: list[str],
    analysis,
    result_payload: dict,
    rejected_positions: list[np.ndarray],
    success_radius_m: float,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = render_final_summary_canvas(
        episode_info=episode_info,
        final_rgb_bgr=final_rgb_bgr,
        final_agent_state=final_agent_state,
        navigation_map=navigation_map,
        trace_records=trace_records,
        canonical_goal=canonical_goal,
        goal_queries=goal_queries,
        analysis=analysis,
        result_payload=result_payload,
        rejected_positions=rejected_positions,
        success_radius_m=success_radius_m,
    )
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"failed to write final summary image {output_path}")


class EpisodeVisualizer:
    def __init__(self, video_path: str, fps: int = 15) -> None:
        self.video_path = Path(video_path)
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.path_history: list[np.ndarray] = []

    def _init_writer(self) -> None:
        if self.writer is not None:
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            fourcc,
            float(self.fps),
            (STEP_CANVAS_WIDTH, STEP_CANVAS_HEIGHT),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"failed to open video writer for {self.video_path}")

    def add_frame(
        self,
        *,
        episode_info,
        rgb_bgr: np.ndarray,
        agent_state,
        nodes: list[dict],
        canonical_goal: str,
        goal_queries: list[str],
        target_node: dict | None,
        current_goal: np.ndarray | None,
        step_index: int,
        mode: str,
        action: int,
        runtime_stats: dict,
        stop_called: bool,
        target_found: bool,
        navigation_map,
        step_record: dict | None = None,
        rejected_positions: list[np.ndarray] | None = None,
        task_mode: str = "text",
    ) -> None:
        self._init_writer()
        self.path_history.append(np.asarray(agent_state.position, dtype=np.float32).copy())
        canvas = render_episode_canvas(
            episode_info=episode_info,
            rgb_bgr=rgb_bgr,
            agent_state=agent_state,
            path_history=self.path_history,
            nodes=nodes,
            canonical_goal=canonical_goal,
            goal_queries=goal_queries,
            target_node=target_node,
            current_goal=current_goal,
            step_index=step_index,
            mode=mode,
            action=action,
            runtime_stats=runtime_stats,
            stop_called=stop_called,
            target_found=target_found,
            navigation_map=navigation_map,
            step_record=step_record,
            rejected_positions=rejected_positions,
            task_mode=task_mode,
        )
        assert self.writer is not None
        self.writer.write(canvas)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class EpisodeImageDumper:
    def __init__(
        self,
        output_dir: str,
        *,
        image_format: str = "composite",
        every_n_steps: int = 1,
        overwrite: bool = False,
    ) -> None:
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be >= 1")
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            if not overwrite:
                raise FileExistsError(f"image directory already exists: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.image_format = image_format
        self.every_n_steps = every_n_steps
        self.path_history: list[np.ndarray] = []

    def add_frame(
        self,
        *,
        episode_info,
        rgb_bgr: np.ndarray,
        agent_state,
        nodes: list[dict],
        canonical_goal: str,
        goal_queries: list[str],
        target_node: dict | None,
        current_goal: np.ndarray | None,
        step_index: int,
        mode: str,
        action: int,
        runtime_stats: dict,
        stop_called: bool,
        target_found: bool,
        navigation_map,
        step_record: dict | None = None,
        rejected_positions: list[np.ndarray] | None = None,
        task_mode: str = "text",
    ) -> None:
        self.path_history.append(np.asarray(agent_state.position, dtype=np.float32).copy())
        if step_index % self.every_n_steps != 0:
            return

        if self.image_format in {"composite", "both"}:
            canvas = render_episode_canvas(
                episode_info=episode_info,
                rgb_bgr=rgb_bgr,
                agent_state=agent_state,
                path_history=self.path_history,
                nodes=nodes,
                canonical_goal=canonical_goal,
                goal_queries=goal_queries,
                target_node=target_node,
                current_goal=current_goal,
                step_index=step_index,
                mode=mode,
                action=action,
                runtime_stats=runtime_stats,
                stop_called=stop_called,
                target_found=target_found,
                navigation_map=navigation_map,
                step_record=step_record,
                rejected_positions=rejected_positions,
                task_mode=task_mode,
            )
            composite_path = self.output_dir / f"frame_{step_index:04d}.png"
            if not cv2.imwrite(str(composite_path), canvas):
                raise RuntimeError(f"failed to write image {composite_path}")

        if self.image_format in {"rgb", "both"}:
            rgb_path = self.output_dir / f"rgb_{step_index:04d}.png"
            if not cv2.imwrite(str(rgb_path), rgb_bgr):
                raise RuntimeError(f"failed to write image {rgb_path}")

    def close(self) -> None:
        return None
