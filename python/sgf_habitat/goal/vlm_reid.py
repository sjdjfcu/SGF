"""VLM re-identification of the goal instance in one full-resolution view."""

from __future__ import annotations

import base64
import json
import math
import re
from typing import Any

import cv2
import numpy as np
import requests

from ..runtime.timing import RuntimeTiming, detail_scope


FOVEATED_REID_PROMPT = (
    "Image A is a reference photograph of one specific object instance. "
    "Image B is the robot's current camera view. Return JSON only: "
    '{"visible": true/false, "confidence": 0.0-1.0, '
    '"box_2d": [ymin, xmin, ymax, xmax], "reason": "short"}. '
    "Set visible true only if Image B shows the same physical object "
    "instance as Image A, not merely the same category, a similar object, "
    "or the same room, even if it is small, partially visible, or seen from "
    "a different angle. box_2d is the tight bounding box of that object in "
    "Image B with coordinates normalized to 0-1000 as [ymin, xmin, ymax, "
    "xmax]. If it is not visible, set box_2d to null."
)


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", np.ascontiguousarray(image_bgr))
    if not ok:
        raise RuntimeError("failed to encode image as PNG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", str(text), re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "visible"}:
        return True
    if normalized in {"false", "no", "0", "none", "not_visible", "null"}:
        return False
    return None


def parse_normalized_box(
    payload: Any,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    """Convert a 0-1000 [ymin, xmin, ymax, xmax] box to pixel xyxy."""

    if not isinstance(payload, (list, tuple)) or len(payload) != 4:
        return None
    try:
        values = [float(v) for v in payload]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    ymin, xmin, ymax, xmax = values
    scale_x = float(width) / 1000.0
    scale_y = float(height) / 1000.0
    x0 = min(max(0.0, xmin * scale_x), float(width))
    x1 = min(max(0.0, xmax * scale_x), float(width))
    y0 = min(max(0.0, ymin * scale_y), float(height))
    y1 = min(max(0.0, ymax * scale_y), float(height))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def query_instance_box(
    *,
    vlm_config: Any,
    goal_image_rgb: np.ndarray,
    view_rgb: np.ndarray,
    timing: RuntimeTiming | None = None,
) -> dict[str, Any]:
    """Ask whether one full-resolution view shows the goal instance and where."""

    base_url = str(getattr(vlm_config, "base_url", "") or "")
    model = str(getattr(vlm_config, "model", "") or "")
    if not (base_url and model):
        return {"attempted": False, "error": "incomplete_vlm_config"}
    goal = np.asarray(goal_image_rgb, dtype=np.uint8)[:, :, :3]
    view = np.asarray(view_rgb, dtype=np.uint8)[:, :, :3]
    headers = {"Content-Type": "application/json"}
    api_key = str(getattr(vlm_config, "api_key", "") or "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    with detail_scope(timing, "image_encoding"):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": FOVEATED_REID_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + _encode_png_base64(
                                    cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
                                )
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + _encode_png_base64(
                                    cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                                )
                            },
                        },
                    ],
                }
            ],
        }
    with detail_scope(timing, "vlm_request"):
        response = requests.post(
            base_url.rstrip("/") + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=float(getattr(vlm_config, "timeout_seconds", 20.0)),
        )
        response.raise_for_status()
    with detail_scope(timing, "response_parsing"):
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_json_object(content)
        visible = _parse_bool(parsed.get("visible"))
        confidence = parsed.get("confidence")
        try:
            confidence_value: float | None = float(confidence)
            if not math.isfinite(confidence_value):
                confidence_value = None
        except (TypeError, ValueError):
            confidence_value = None
        height, width = view.shape[:2]
        box = parse_normalized_box(parsed.get("box_2d"), width=width, height=height)
        return {
            "attempted": True,
            "visible": visible,
            "confidence": (
                None if confidence_value is None else round(confidence_value, 4)
            ),
            "box_xyxy": None if box is None else [round(float(v), 1) for v in box],
            "reason": str(parsed.get("reason", ""))[:200],
            "parse_error": bool(visible is None),
        }


def lift_box_to_world(
    box_xyxy: tuple[float, float, float, float] | list[float],
    *,
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    world_from_camera_cv: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
    core_fraction: float = 0.5,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Lift a pixel box to a world point from its central median depth."""

    diagnostics: dict[str, Any] = {}
    depth_2d = np.asarray(depth, dtype=np.float32)
    if depth_2d.ndim == 3 and depth_2d.shape[2] == 1:
        depth_2d = depth_2d[:, :, 0]
    if depth_2d.ndim != 2:
        diagnostics["reject_reason"] = "invalid_depth"
        return None, diagnostics
    height, width = depth_2d.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box_xyxy]
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w < 2.0 or box_h < 2.0:
        diagnostics["reject_reason"] = "box_too_small"
        return None, diagnostics
    fraction = min(1.0, max(0.05, float(core_fraction)))
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    half_w = max(1.0, 0.5 * box_w * fraction)
    half_h = max(1.0, 0.5 * box_h * fraction)
    c0 = max(0, int(math.floor(center_x - half_w)))
    c1 = min(width, int(math.ceil(center_x + half_w)))
    r0 = max(0, int(math.floor(center_y - half_h)))
    r1 = min(height, int(math.ceil(center_y + half_h)))
    if c1 <= c0 or r1 <= r0:
        diagnostics["reject_reason"] = "box_outside_image"
        return None, diagnostics
    core = depth_2d[r0:r1, c0:c1].reshape(-1)
    lower = float(min_depth_m)
    upper = float(max_depth_m)
    # Values at the clamp limits are not measurements.
    margin = 0.02 * max(1e-6, upper - lower)
    valid = core[
        np.isfinite(core) & (core > lower + margin) & (core < upper - margin)
    ]
    diagnostics["core_valid_fraction"] = round(
        float(valid.size) / float(max(1, core.size)), 4
    )
    if valid.size < max(4, int(0.25 * core.size)):
        diagnostics["reject_reason"] = "range_not_measured"
        return None, diagnostics
    z = float(np.median(valid))
    diagnostics["range_m"] = round(z, 4)
    if not (fx > 0.0 and fy > 0.0):
        diagnostics["reject_reason"] = "invalid_intrinsics"
        return None, diagnostics
    point_camera = np.array(
        [
            (center_x - float(cx)) / float(fx) * z,
            (center_y - float(cy)) / float(fy) * z,
            z,
            1.0,
        ],
        dtype=np.float64,
    )
    pose = np.asarray(world_from_camera_cv, dtype=np.float64).reshape(4, 4)
    point_world = (pose @ point_camera)[:3]
    if not np.all(np.isfinite(point_world)):
        diagnostics["reject_reason"] = "nonfinite_point"
        return None, diagnostics
    diagnostics["point_m"] = [round(float(v), 4) for v in point_world]
    return point_world.astype(np.float32), diagnostics
