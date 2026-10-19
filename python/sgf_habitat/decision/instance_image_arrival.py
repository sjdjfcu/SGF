"""Close-stop decisions for image navigation."""

from __future__ import annotations

import math


def vlm_unavailable_strong_geometry_allows_stop(
    *,
    validation_error_present: bool,
    geometric_stop_candidate: bool,
    visual_target_updated: bool,
    visual_goal_source: str,
    visual_goal_updated_this_frame: bool,
    visual_goal_is_fresh: bool,
    visual_goal_is_weak: bool,
    visual_goal_distance_m: float,
    stop_radius_m: float,
    match_inliers: int,
    min_match_inliers: int,
    match_confidence_sum: float,
    min_match_confidence_sum: float,
    match_center_x_px: float,
    image_width_px: float,
    pose_accepted: bool,
    viewpoint_pnp_selected: bool,
    pnp_inlier_fraction: float,
    pose_confirms_visual_goal: bool,
    pose_agent_distance_m: float,
    pose_alignment_m: float,
    max_alignment_m: float,
) -> bool:
    """Permit a grounded stop when the semantic validator is unavailable."""

    try:
        goal_distance = float(visual_goal_distance_m)
        stop_radius = float(stop_radius_m)
        inliers = int(match_inliers)
        min_inliers = int(min_match_inliers)
        confidence_sum = float(match_confidence_sum)
        min_confidence_sum = float(min_match_confidence_sum)
        match_center_x = float(match_center_x_px)
        image_width = float(image_width_px)
        inlier_fraction = float(pnp_inlier_fraction)
        pose_distance = float(pose_agent_distance_m)
        alignment = float(pose_alignment_m)
        max_alignment = float(max_alignment_m)
    except (TypeError, ValueError):
        return False
    scalars = (
        goal_distance,
        stop_radius,
        confidence_sum,
        min_confidence_sum,
        match_center_x,
        image_width,
        inlier_fraction,
        pose_distance,
        alignment,
        max_alignment,
    )
    if not all(math.isfinite(value) for value in scalars):
        return False
    if (
        goal_distance < 0.0
        or stop_radius <= 0.0
        or inliers < 0
        or min_inliers <= 0
        or confidence_sum < 0.0
        or min_confidence_sum < 0.0
        or image_width <= 0.0
        or not 0.0 <= inlier_fraction <= 1.0
        or pose_distance < 0.0
        or alignment < 0.0
        or max_alignment < 0.0
    ):
        return False
    centered = abs(
        match_center_x - 0.5 * image_width
    ) <= 0.1 * image_width
    return bool(
        bool(validation_error_present)
        and bool(geometric_stop_candidate)
        and bool(visual_target_updated)
        and str(visual_goal_source) == "xfeat"
        and bool(visual_goal_updated_this_frame)
        and bool(visual_goal_is_fresh)
        and not bool(visual_goal_is_weak)
        and goal_distance <= stop_radius
        and inliers >= min_inliers
        and confidence_sum >= min_confidence_sum
        and centered
        and bool(pose_accepted)
        and bool(viewpoint_pnp_selected)
        and inlier_fraction >= 0.5
        and bool(pose_confirms_visual_goal)
        and pose_distance <= stop_radius
        and alignment <= max_alignment
    )


def weak_visual_stop_rejection_reason(
    *,
    vlm_approved: bool,
    visual_pose_reject_reason: str | None,
    current_target_updated: bool,
) -> str | None:
    """Rejection reason for a VLM approval with too few PnP inliers, else None."""

    if not bool(vlm_approved):
        return None
    if str(visual_pose_reject_reason or "") != "too_few_inliers":
        return None
    if not bool(current_target_updated):
        return "weak_current_pnp_without_current_target_update"
    return "weak_current_pnp_without_retained_visual_support"


__all__ = [
    "vlm_unavailable_strong_geometry_allows_stop",
    "weak_visual_stop_rejection_reason",
]
