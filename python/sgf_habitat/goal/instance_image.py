from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..runtime.timing import RuntimeTiming, detail_scope

_CPP_BACKEND = None
_CPP_BACKEND_CHECKED = False
_CPP_BACKEND_ERROR: Exception | None = None


def _sgf_cpp_backend():
    global _CPP_BACKEND, _CPP_BACKEND_CHECKED, _CPP_BACKEND_ERROR
    if not _CPP_BACKEND_CHECKED:
        _CPP_BACKEND_CHECKED = True
        try:
            from .. import _sgf as backend
        except Exception as exc:
            backend = None
            _CPP_BACKEND_ERROR = exc
        _CPP_BACKEND = backend
    if _CPP_BACKEND is None:
        raise RuntimeError("SGF C++ extension is required for instance-image hot paths") from _CPP_BACKEND_ERROR
    return _CPP_BACKEND


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ImageTargetConfig:
    min_matches: int = 40
    min_inliers: int = 20
    fine_conf_threshold: float = 0.25
    min_mean_confidence: float = 0.35
    min_confidence_sum: float = 12.0
    min_coarse_sim: float = 0.0
    ransac_reproj_threshold_px: float = 5.0
    dbscan_eps_m: float = 0.35
    dbscan_min_samples: int = 4
    min_depth_m: float = 0.0
    max_depth_m: float = 10.0
    goal_min_y_frac: float = 0.0


@dataclass
class XFeatStarMatches:
    goal_xy: np.ndarray
    current_xy: np.ndarray
    fine_conf: np.ndarray
    coarse_sim: np.ndarray
    raw_match_count: int = 0
    coarse_filtered_count: int = 0
    fine_conf_filtered_count: int = 0

    @classmethod
    def empty(cls, *, raw_match_count: int = 0, coarse_filtered_count: int = 0) -> "XFeatStarMatches":
        xy = np.zeros((0, 2), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        return cls(
            goal_xy=xy,
            current_xy=xy.copy(),
            fine_conf=scores,
            coarse_sim=scores.copy(),
            raw_match_count=int(raw_match_count),
            coarse_filtered_count=int(coarse_filtered_count),
            fine_conf_filtered_count=0,
        )


@dataclass(frozen=True)
class ImageSpaceMatchEvidence:
    """Target-specific XFeat geometry before depth or world projection."""

    matches: XFeatStarMatches
    inlier_mask: np.ndarray
    ransac_attempted: bool

    def __post_init__(self) -> None:
        mask = np.asarray(self.inlier_mask, dtype=bool).reshape(-1).copy()
        if len(mask) != len(self.matches.current_xy):
            raise ValueError("inlier_mask must align with filtered matches")
        mask.setflags(write=False)
        object.__setattr__(self, "inlier_mask", mask)

    @property
    def match_count(self) -> int:
        return int(len(self.matches.current_xy))

    @property
    def ransac_inliers(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))


@dataclass(frozen=True)
class FrameGeometryEvidence:
    """Read-only RGB-D projection of one frame's matches."""

    image_evidence: ImageSpaceMatchEvidence
    world_points: np.ndarray
    valid_depth_mask: np.ndarray
    projected_match_indices: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.image_evidence, ImageSpaceMatchEvidence):
            raise TypeError(
                "image_evidence must be an ImageSpaceMatchEvidence"
            )

        matches = self.image_evidence.matches
        match_count = self.image_evidence.match_count
        aligned_lengths = {
            len(np.asarray(matches.goal_xy).reshape(-1, 2)),
            len(np.asarray(matches.current_xy).reshape(-1, 2)),
            len(np.asarray(matches.fine_conf).reshape(-1)),
            len(np.asarray(matches.coarse_sim).reshape(-1)),
        }
        if aligned_lengths != {match_count}:
            raise ValueError(
                "image-space match arrays must have identical lengths"
            )

        try:
            points = np.array(
                self.world_points,
                dtype=np.float32,
                copy=True,
                order="C",
            ).reshape(-1, 3)
        except ValueError as exc:
            raise ValueError("world_points must be an Nx3 array") from exc
        valid = np.array(
            self.valid_depth_mask,
            dtype=bool,
            copy=True,
            order="C",
        ).reshape(-1)
        indices = np.array(
            self.projected_match_indices,
            dtype=np.int64,
            copy=True,
            order="C",
        ).reshape(-1)

        if len(valid) != match_count:
            raise ValueError(
                "valid_depth_mask must align with image-space matches"
            )
        expected_indices = np.flatnonzero(valid).astype(
            np.int64,
            copy=False,
        )
        if not np.array_equal(indices, expected_indices):
            raise ValueError(
                "projected_match_indices must identify every valid match in order"
            )
        if len(points) != len(indices):
            raise ValueError(
                "world_points must align with projected_match_indices"
            )

        points.setflags(write=False)
        valid.setflags(write=False)
        indices.setflags(write=False)
        object.__setattr__(self, "world_points", points)
        object.__setattr__(self, "valid_depth_mask", valid)
        object.__setattr__(self, "projected_match_indices", indices)

    @property
    def matches(self) -> XFeatStarMatches:
        return self.image_evidence.matches

    @property
    def inlier_mask(self) -> np.ndarray:
        return self.image_evidence.inlier_mask

    @property
    def match_count(self) -> int:
        return self.image_evidence.match_count


@dataclass(frozen=True)
class ClusterSelection:
    mask: np.ndarray
    confidence_sum: float
    mean_confidence: float
    mean_coarse_sim: float
    size: int


@dataclass
class ImageMatchDebugData:
    matches: XFeatStarMatches
    inlier_mask: np.ndarray
    valid_depth_mask: np.ndarray
    cluster_mask: np.ndarray


@dataclass
class ImageTargetResult:
    node: dict[str, Any] | None
    diagnostics: dict[str, Any]
    visualization: ImageMatchDebugData | None = None


def _as_xy(array: np.ndarray) -> np.ndarray:
    xy = np.asarray(array, dtype=np.float32)
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return xy.reshape(-1, 2)


def _as_score_vector(array: np.ndarray) -> np.ndarray:
    scores = np.asarray(array, dtype=np.float32)
    if scores.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return scores.reshape(-1)


def _matches_from_cpp_result(result: Any) -> XFeatStarMatches:
    return XFeatStarMatches(
        goal_xy=_as_xy(result["goal_xy"]),
        current_xy=_as_xy(result["current_xy"]),
        fine_conf=_as_score_vector(result["fine_conf"]),
        coarse_sim=_as_score_vector(result["coarse_sim"]),
        raw_match_count=int(result["raw_match_count"]),
        coarse_filtered_count=int(result["coarse_filtered_count"]),
        fine_conf_filtered_count=int(result["fine_conf_filtered_count"]),
    )


def filter_match_scores(
    goal_xy: np.ndarray,
    current_xy: np.ndarray,
    fine_conf: np.ndarray,
    coarse_sim: np.ndarray,
    *,
    fine_conf_threshold: float,
    min_coarse_sim: float,
) -> XFeatStarMatches:
    goal_xy = _as_xy(goal_xy)
    current_xy = _as_xy(current_xy)
    fine_conf = _as_score_vector(fine_conf)
    coarse_sim = _as_score_vector(coarse_sim)
    count = min(len(goal_xy), len(current_xy), len(fine_conf), len(coarse_sim))
    goal_xy = goal_xy[:count]
    current_xy = current_xy[:count]
    fine_conf = fine_conf[:count]
    coarse_sim = coarse_sim[:count]

    backend = _sgf_cpp_backend()
    return _matches_from_cpp_result(
        backend.filter_match_scores(
            goal_xy,
            current_xy,
            fine_conf,
            coarse_sim,
            float(fine_conf_threshold),
            float(min_coarse_sim),
        )
    )


def filter_goal_image_region(
    matches: XFeatStarMatches,
    goal_image_shape: tuple[int, ...],
    *,
    min_x_frac: float,
    max_x_frac: float,
    min_y_frac: float,
    max_y_frac: float,
) -> tuple[XFeatStarMatches, int]:
    height = int(goal_image_shape[0]) if len(goal_image_shape) >= 1 else 0
    width = int(goal_image_shape[1]) if len(goal_image_shape) >= 2 else 0
    if width <= 0 or height <= 0 or len(matches.goal_xy) == 0:
        return matches, 0

    backend = _sgf_cpp_backend()
    result, removed = backend.filter_goal_image_region(
        matches.goal_xy,
        matches.current_xy,
        matches.fine_conf,
        matches.coarse_sim,
        int(matches.raw_match_count),
        int(matches.coarse_filtered_count),
        height,
        width,
        float(min_x_frac),
        float(max_x_frac),
        float(min_y_frac),
        float(max_y_frac),
    )
    return _matches_from_cpp_result(result), int(removed)


def _candidate_xfeat_roots(xfeat_root: str | Path) -> list[Path]:
    configured = Path(xfeat_root).expanduser()
    candidates = [configured]
    if not configured.is_absolute():
        candidates.append(Path.cwd() / configured)
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "accelerated_features")

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        root = candidate.resolve()
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _load_xfeat_class(xfeat_root: str | Path):
    checked_roots = _candidate_xfeat_roots(xfeat_root)
    root = next(
        (
            candidate
            for candidate in checked_roots
            if (candidate / "modules" / "xfeat.py").is_file()
        ),
        None,
    )
    if root is None:
        checked = ", ".join(
            str(candidate / "modules" / "xfeat.py") for candidate in checked_roots
        )
        raise FileNotFoundError(f"XFeat module not found; checked: {checked}")
    xfeat_py = root / "modules" / "xfeat.py"

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    module = importlib.import_module("modules.xfeat")
    module_file = Path(getattr(module, "__file__", "")).resolve()
    if module_file != xfeat_py:
        raise ImportError(f"loaded modules.xfeat from {module_file}, expected {xfeat_py}")
    return module.XFeat


class XFeatStarMatcher:
    def __init__(self, xfeat_root: str | Path, *, top_k: int = 8000) -> None:
        xfeat_cls = _load_xfeat_class(xfeat_root)
        self.top_k = int(top_k)
        self.xfeat = xfeat_cls(top_k=self.top_k)
        self._cached_goal_image_rgb: np.ndarray | None = None
        self._cached_goal_features: Any | None = None

    def _goal_features(self, goal_image_rgb: np.ndarray) -> Any:
        import torch

        goal_array = np.asarray(goal_image_rgb, dtype=np.uint8)
        cache_hit = (
            self._cached_goal_image_rgb is not None
            and self._cached_goal_features is not None
            and np.array_equal(goal_array, self._cached_goal_image_rgb)
        )
        if not cache_hit:
            with torch.inference_mode():
                goal_tensor = self.xfeat.parse_input(goal_array)
                self._cached_goal_features = self.xfeat.detectAndComputeDense(
                    goal_tensor,
                    top_k=self.top_k,
                )
            # Copy so in-place changes by the caller invalidate the cache
            self._cached_goal_image_rgb = goal_array.copy()
        return self._cached_goal_features

    def match(
        self,
        goal_image_rgb: np.ndarray,
        current_image_rgb: np.ndarray,
        *,
        fine_conf_threshold: float = 0.25,
        min_coarse_sim: float = 0.0,
    ) -> XFeatStarMatches:
        import torch

        with torch.inference_mode():
            goal_features = self._goal_features(goal_image_rgb)
            current_tensor = self.xfeat.parse_input(
                np.asarray(current_image_rgb, dtype=np.uint8)
            )
            current_features = self.xfeat.detectAndComputeDense(
                current_tensor,
                top_k=self.top_k,
            )
        return _match_xfeat_star_features_with_scores(
            self.xfeat,
            goal_features,
            current_features,
            fine_conf_threshold=fine_conf_threshold,
            min_coarse_sim=min_coarse_sim,
        )



def match_xfeat_star_with_scores(
    xfeat,
    goal_image_rgb: np.ndarray,
    current_image_rgb: np.ndarray,
    *,
    top_k: int = 8000,
    fine_conf_threshold: float = 0.25,
    min_coarse_sim: float = 0.0,
) -> XFeatStarMatches:
    import torch

    with torch.inference_mode():
        goal_tensor = xfeat.parse_input(np.asarray(goal_image_rgb, dtype=np.uint8))
        current_tensor = xfeat.parse_input(np.asarray(current_image_rgb, dtype=np.uint8))

        goal_features = xfeat.detectAndComputeDense(goal_tensor, top_k=int(top_k))
        current_features = xfeat.detectAndComputeDense(current_tensor, top_k=int(top_k))
    return _match_xfeat_star_features_with_scores(
        xfeat,
        goal_features,
        current_features,
        fine_conf_threshold=fine_conf_threshold,
        min_coarse_sim=min_coarse_sim,
    )


def _match_xfeat_star_features_with_scores(
    xfeat,
    goal_features: Any,
    current_features: Any,
    *,
    fine_conf_threshold: float,
    min_coarse_sim: float,
) -> XFeatStarMatches:
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        match_indices = xfeat.batch_match(goal_features["descriptors"], current_features["descriptors"])
        idx0, idx1 = match_indices[0]
        raw_match_count = int(idx0.numel())
        if raw_match_count == 0:
            return XFeatStarMatches.empty(raw_match_count=0)

        goal_descriptors = goal_features["descriptors"][0][idx0]
        current_descriptors = current_features["descriptors"][0][idx1]
        coarse_sim = (goal_descriptors * current_descriptors).sum(dim=-1)
        coarse_mask = coarse_sim >= float(min_coarse_sim)
        if int(coarse_mask.count_nonzero().item()) == 0:
            return XFeatStarMatches.empty(raw_match_count=raw_match_count, coarse_filtered_count=0)

        idx0 = idx0[coarse_mask]
        idx1 = idx1[coarse_mask]
        coarse_sim = coarse_sim[coarse_mask]
        goal_descriptors = goal_features["descriptors"][0][idx0]
        current_descriptors = current_features["descriptors"][0][idx1]

        offsets = xfeat.net.fine_matcher(torch.cat([goal_descriptors, current_descriptors], dim=-1))
        fine_conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]
        subpixel_offsets = xfeat.subpix_softmax2d(offsets.view(-1, 8, 8))

        goal_xy = goal_features["keypoints"][0][idx0]
        current_xy = current_features["keypoints"][0][idx1]
        scales = goal_features["scales"][0][idx0]
        refined_goal_xy = goal_xy + subpixel_offsets * scales[:, None]

        goal_np = refined_goal_xy.detach().cpu().numpy().astype(np.float32, copy=False)
        current_np = current_xy.detach().cpu().numpy().astype(np.float32, copy=False)
        fine_np = fine_conf.detach().cpu().numpy().astype(np.float32, copy=False)
        coarse_np = coarse_sim.detach().cpu().numpy().astype(np.float32, copy=False)

    filtered = filter_match_scores(
        goal_np,
        current_np,
        fine_np,
        coarse_np,
        fine_conf_threshold=fine_conf_threshold,
        min_coarse_sim=-float("inf"),
    )
    filtered.raw_match_count = raw_match_count
    filtered.coarse_filtered_count = int(len(coarse_np))
    return filtered


def ransac_inlier_mask(
    goal_xy: np.ndarray,
    current_xy: np.ndarray,
    *,
    reproj_threshold_px: float = 5.0,
) -> np.ndarray:
    goal_xy = _as_xy(goal_xy)
    current_xy = _as_xy(current_xy)
    count = min(len(goal_xy), len(current_xy))
    mask = np.zeros((count,), dtype=bool)
    if count < 4:
        return mask

    src = goal_xy[:count].astype(np.float32, copy=False)
    dst = current_xy[:count].astype(np.float32, copy=False)
    try:
        _, homography_mask = cv2.findHomography(
            src,
            dst,
            cv2.RANSAC,
            ransacReprojThreshold=float(reproj_threshold_px),
        )
    except cv2.error:
        homography_mask = None
    homography_result = None
    if homography_mask is not None:
        candidate = homography_mask.reshape(-1).astype(bool)
        if candidate.size == count and np.count_nonzero(candidate) >= 4:
            homography_result = candidate

    if homography_result is not None:
        return homography_result

    if count < 8:
        return mask
    try:
        _, fundamental_mask = cv2.findFundamentalMat(
            src,
            dst,
            cv2.FM_RANSAC,
            ransacReprojThreshold=max(0.5, float(reproj_threshold_px) * 0.5),
            confidence=0.995,
        )
    except cv2.error:
        fundamental_mask = None
    if fundamental_mask is None:
        return mask
    result = fundamental_mask.reshape(-1).astype(bool)
    return result if result.size == count else mask


def geometric_evidence_completeness(
    *,
    ransac_inliers: int,
    confidence_sum: float | None,
    mean_confidence: float | None,
    min_inliers: int,
    min_confidence_sum: float,
    min_mean_confidence: float,
) -> float:
    """Return in [0, 1] how completely a match meets the localization gate."""

    inliers = int(ransac_inliers)
    if inliers < int(min_inliers) or inliers <= 0:
        return 0.0
    if confidence_sum is None or mean_confidence is None:
        return 0.0
    total = float(confidence_sum)
    mean = float(mean_confidence)
    if not math.isfinite(total) or not math.isfinite(mean):
        return 0.0
    sum_floor = float(min_confidence_sum)
    mean_floor = float(min_mean_confidence)
    total_ratio = 1.0 if sum_floor <= 0.0 else total / sum_floor
    mean_ratio = 1.0 if mean_floor <= 0.0 else mean / mean_floor
    total_ratio = min(1.0, max(0.0, total_ratio))
    mean_ratio = min(1.0, max(0.0, mean_ratio))
    return float(total_ratio * mean_ratio)


def weak_instance_hypothesis(
    matches: XFeatStarMatches | FrameGeometryEvidence,
    inlier_mask: np.ndarray | None = None,
    *,
    depth: np.ndarray | None = None,
    intrinsics: CameraIntrinsics | None = None,
    world_from_camera_cv: np.ndarray | None = None,
    config: ImageTargetConfig,
) -> np.ndarray | None:
    """Return a world point for sub-threshold instance evidence, or None."""

    frame_evidence = (
        matches if isinstance(matches, FrameGeometryEvidence) else None
    )
    image_matches = (
        frame_evidence.matches if frame_evidence is not None else matches
    )
    stored_mask = (
        frame_evidence.inlier_mask
        if frame_evidence is not None and inlier_mask is None
        else inlier_mask
    )
    if stored_mask is None:
        raise ValueError("inlier_mask is required for image-space matches")
    mask = np.asarray(stored_mask, dtype=bool).reshape(-1)
    count = min(len(mask), len(image_matches.current_xy))
    if count <= 0:
        return None
    mask = mask[:count]
    if int(np.count_nonzero(mask)) < int(config.dbscan_min_samples):
        return None
    inlier_conf = image_matches.fine_conf[:count][mask]
    inlier_coarse = image_matches.coarse_sim[:count][mask]
    if frame_evidence is not None:
        full_inlier_mask = np.zeros(
            (frame_evidence.match_count,),
            dtype=bool,
        )
        full_inlier_mask[:count] = mask
        world_points, projected_match_indices = (
            _projected_geometry_for_match_mask(
                frame_evidence,
                full_inlier_mask,
            )
        )
        projected_conf = image_matches.fine_conf[
            projected_match_indices
        ]
        projected_coarse = image_matches.coarse_sim[
            projected_match_indices
        ]
    else:
        if depth is None or intrinsics is None or world_from_camera_cv is None:
            raise ValueError(
                "depth, intrinsics, and world_from_camera_cv are required "
                "for image-space matches"
            )
        inlier_current_xy = image_matches.current_xy[:count][mask]
        world_points, valid_depth_mask = project_pixels_to_world(
            inlier_current_xy,
            depth,
            intrinsics,
            world_from_camera_cv,
            min_depth_m=float(config.min_depth_m),
            max_depth_m=float(config.max_depth_m),
        )
        valid = np.asarray(valid_depth_mask, dtype=bool).reshape(-1)
        projected_conf = inlier_conf[valid]
        projected_coarse = inlier_coarse[valid]
    if len(world_points) < int(config.dbscan_min_samples):
        return None
    cluster = select_confidence_cluster(
        world_points,
        projected_conf,
        projected_coarse,
        eps_m=float(config.dbscan_eps_m),
        min_samples=int(config.dbscan_min_samples),
    )
    if cluster is None or int(cluster.size) < int(config.dbscan_min_samples):
        return None
    cluster_points = world_points[cluster.mask]
    cluster_conf = np.asarray(
        projected_conf,
        dtype=np.float64,
    )[cluster.mask]
    weight_sum = float(np.sum(cluster_conf))
    if weight_sum > 1e-6:
        centroid = np.average(cluster_points, axis=0, weights=cluster_conf)
    else:
        centroid = np.mean(cluster_points, axis=0)
    centroid = np.asarray(centroid, dtype=np.float32).reshape(-1)
    if centroid.size != 3 or not np.all(np.isfinite(centroid)):
        return None
    return centroid


def match_instance_image_evidence(
    matcher: XFeatStarMatcher,
    *,
    goal_image_rgb: np.ndarray,
    current_image_rgb: np.ndarray,
    config: ImageTargetConfig,
    timing: RuntimeTiming | None = None,
) -> ImageSpaceMatchEvidence:
    """Run one RGB-only target match and its configured 2-D RANSAC gate."""

    with detail_scope(timing, "xfeat"):
        matches = matcher.match(
            goal_image_rgb,
            current_image_rgb,
            fine_conf_threshold=float(config.fine_conf_threshold),
            min_coarse_sim=float(config.min_coarse_sim),
        )
    match_count = int(len(matches.current_xy))
    attempted = bool(match_count >= int(config.min_matches))
    if attempted:
        with detail_scope(timing, "ransac_2d"):
            inlier_mask = ransac_inlier_mask(
                matches.goal_xy,
                matches.current_xy,
                reproj_threshold_px=float(config.ransac_reproj_threshold_px),
            )
    else:
        inlier_mask = np.zeros((match_count,), dtype=bool)
    return ImageSpaceMatchEvidence(
        matches=matches,
        inlier_mask=inlier_mask,
        ransac_attempted=attempted,
    )


def project_pixels_to_world(
    xy: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    world_from_camera_cv: np.ndarray,
    *,
    min_depth_m: float = 0.0,
    max_depth_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    xy = _as_xy(xy)
    depth_2d = np.asarray(depth, dtype=np.float32)
    if depth_2d.ndim == 3 and depth_2d.shape[2] == 1:
        depth_2d = depth_2d[:, :, 0]
    if depth_2d.ndim != 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((len(xy),), dtype=bool)

    transform = np.asarray(world_from_camera_cv, dtype=np.float32)
    if transform.shape != (4, 4):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((len(xy),), dtype=bool)

    backend = _sgf_cpp_backend()
    points, valid = backend.project_pixels_to_world(
        xy.astype(np.float32, copy=False),
        depth_2d.astype(np.float32, copy=False),
        float(intrinsics.fx),
        float(intrinsics.fy),
        float(intrinsics.cx),
        float(intrinsics.cy),
        transform,
        float(min_depth_m),
        float(max_depth_m),
    )
    return np.asarray(points, dtype=np.float32), np.asarray(valid, dtype=bool)


def build_frame_geometry_evidence(
    image_evidence: ImageSpaceMatchEvidence,
    *,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    world_from_camera_cv: np.ndarray,
    min_depth_m: float = 0.0,
    max_depth_m: float = 10.0,
) -> FrameGeometryEvidence:
    """Project all current-frame matches into compact world geometry."""

    if not isinstance(image_evidence, ImageSpaceMatchEvidence):
        raise TypeError(
            "image_evidence must be an ImageSpaceMatchEvidence"
        )
    world_points, valid_depth_mask = project_pixels_to_world(
        image_evidence.matches.current_xy,
        depth,
        intrinsics,
        world_from_camera_cv,
        min_depth_m=float(min_depth_m),
        max_depth_m=float(max_depth_m),
    )
    valid = np.asarray(valid_depth_mask, dtype=bool).reshape(-1)
    return FrameGeometryEvidence(
        image_evidence=image_evidence,
        world_points=world_points,
        valid_depth_mask=valid,
        projected_match_indices=np.flatnonzero(valid),
    )


def _projected_geometry_for_match_mask(
    evidence: FrameGeometryEvidence,
    match_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(match_mask, dtype=bool).reshape(-1)
    if len(selected) != evidence.match_count:
        raise ValueError("match_mask must align with frame geometry")
    projected_selection = selected[evidence.projected_match_indices]
    selected_indices = evidence.projected_match_indices[projected_selection]
    return evidence.world_points[projected_selection], selected_indices


def dbscan_labels_xz(points_world: np.ndarray, *, eps_m: float, min_samples: int) -> np.ndarray:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    count = len(points)
    labels = np.full((count,), -1, dtype=np.int32)
    if count == 0:
        return labels

    backend = _sgf_cpp_backend()
    return np.asarray(
        backend.dbscan_labels_xz(points, float(eps_m), int(min_samples)),
        dtype=np.int32,
    ).reshape(-1)[:count]


def select_confidence_cluster(
    points_world: np.ndarray,
    fine_conf: np.ndarray,
    coarse_sim: np.ndarray,
    *,
    eps_m: float = 0.35,
    min_samples: int = 4,
) -> ClusterSelection | None:
    points = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    fine_conf = _as_score_vector(fine_conf)[: len(points)]
    coarse_sim = _as_score_vector(coarse_sim)[: len(points)]
    if len(points) == 0 or len(fine_conf) != len(points) or len(coarse_sim) != len(points):
        return None

    backend = _sgf_cpp_backend()
    result = backend.select_confidence_cluster(
        points,
        fine_conf,
        coarse_sim,
        float(eps_m),
        max(1, int(min_samples)),
    )
    if result is None:
        return None
    return ClusterSelection(
        mask=np.asarray(result["mask"], dtype=bool),
        confidence_sum=float(result["confidence_sum"]),
        mean_confidence=float(result["mean_confidence"]),
        mean_coarse_sim=float(result["mean_coarse_sim"]),
        size=int(result["size"]),
    )


def _round_float(value: Any, digits: int = 4) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _points_to_list(points: np.ndarray, *, limit: int = 64) -> list[list[float]]:
    rows = []
    for point in np.asarray(points, dtype=np.float32).reshape(-1, 3)[:limit]:
        rows.append([round(float(point[0]), 4), round(float(point[1]), 4), round(float(point[2]), 4)])
    return rows


def _world_position_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 3 or not np.all(np.isfinite(array)):
        return None
    return [
        round(float(array[0]), 4),
        round(float(array[1]), 4),
        round(float(array[2]), 4),
    ]


INLIER_COVERAGE_GRID = 8


def estimate_goal_viewpoint_pnp(
    world_points: np.ndarray,
    goal_xy: np.ndarray,
    goal_intrinsics: CameraIntrinsics,
    *,
    min_points: int,
    reproj_threshold_px: float,
    min_inlier_fraction: float,
    min_distance_from_cluster_m: float,
    max_distance_from_cluster_m: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Solve the goal camera pose from frame geometry matched to goal pixels."""

    points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    image_points = _as_xy(goal_xy)
    count = min(len(points), len(image_points))
    diagnostics: dict[str, Any] = {"viewpoint_pnp_points": int(count)}
    if count < max(4, int(min_points)):
        diagnostics["viewpoint_pnp_reject_reason"] = "not_enough_points"
        return None, diagnostics

    camera_matrix = np.array(
        [
            [float(goal_intrinsics.fx), 0.0, float(goal_intrinsics.cx)],
            [0.0, float(goal_intrinsics.fy), float(goal_intrinsics.cy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(camera_matrix)) or camera_matrix[0, 0] <= 0.0 or camera_matrix[1, 1] <= 0.0:
        diagnostics["viewpoint_pnp_reject_reason"] = "invalid_intrinsics"
        return None, diagnostics

    try:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            points[:count],
            image_points[:count],
            camera_matrix,
            None,
            iterationsCount=100,
            reprojectionError=float(reproj_threshold_px),
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error as exc:
        diagnostics["viewpoint_pnp_reject_reason"] = "solvepnp_error"
        diagnostics["viewpoint_pnp_error"] = str(exc)
        return None, diagnostics
    if not ok or rvec is None or tvec is None:
        diagnostics["viewpoint_pnp_reject_reason"] = "solvepnp_failed"
        return None, diagnostics

    inlier_count = 0 if inliers is None else int(len(np.asarray(inliers).reshape(-1)))
    inlier_fraction = float(inlier_count) / float(count)
    diagnostics["viewpoint_pnp_inliers"] = int(inlier_count)
    diagnostics["viewpoint_pnp_inlier_fraction"] = round(inlier_fraction, 4)
    if (
        inlier_count < max(4, int(min_points))
        or inlier_fraction < float(min_inlier_fraction)
    ):
        diagnostics["viewpoint_pnp_reject_reason"] = "too_few_inliers"
        return None, diagnostics
    inlier_index = np.asarray(inliers, dtype=np.int64).reshape(-1)
    # Fraction of goal-image grid cells that contain a PnP inlier
    goal_width = 2.0 * float(goal_intrinsics.cx)
    goal_height = 2.0 * float(goal_intrinsics.cy)
    if goal_width > 0.0 and goal_height > 0.0:
        grid = INLIER_COVERAGE_GRID
        inlier_pixels = image_points[:count][inlier_index]
        cells = {
            (
                min(grid - 1, max(0, int(float(x) * grid / goal_width))),
                min(grid - 1, max(0, int(float(y) * grid / goal_height))),
            )
            for x, y in inlier_pixels
            if math.isfinite(float(x)) and math.isfinite(float(y))
        }
        diagnostics["viewpoint_pnp_inlier_coverage"] = round(
            float(len(cells)) / float(grid * grid), 4
        )
    inlier_center = np.mean(points[:count][inlier_index], axis=0)
    if np.all(np.isfinite(inlier_center)):
        diagnostics["viewpoint_pnp_inlier_center_m"] = [
            round(float(inlier_center[0]), 4),
            round(float(inlier_center[1]), 4),
            round(float(inlier_center[2]), 4),
        ]

    try:
        rotation, _ = cv2.Rodrigues(rvec)
    except cv2.error as exc:
        diagnostics["viewpoint_pnp_reject_reason"] = "rodrigues_error"
        diagnostics["viewpoint_pnp_error"] = str(exc)
        return None, diagnostics
    camera_center = (-rotation.T @ np.asarray(tvec, dtype=np.float32).reshape(3, 1)).reshape(3)
    if not np.all(np.isfinite(camera_center)):
        diagnostics["viewpoint_pnp_reject_reason"] = "nonfinite_camera_center"
        return None, diagnostics
    # Optical axis (+Z in the OpenCV camera frame) in world coordinates
    camera_forward = np.asarray(rotation.T, dtype=np.float32) @ np.asarray(
        [0.0, 0.0, 1.0], dtype=np.float32
    )
    forward_norm = float(np.linalg.norm(camera_forward))
    if np.all(np.isfinite(camera_forward)) and forward_norm > 1.0e-6:
        camera_forward = camera_forward / forward_norm
        diagnostics["viewpoint_pnp_forward_world"] = [
            round(float(camera_forward[0]), 6),
            round(float(camera_forward[1]), 6),
            round(float(camera_forward[2]), 6),
        ]
        # Object anchor on the recovered optical axis
        from ..decision.goal_view_terminal import goal_view_object_anchor

        object_anchor, anchor_diagnostics = goal_view_object_anchor(
            world_points=points[:count],
            goal_xy=image_points[:count],
            inlier_index=inlier_index,
            rotation=rotation,
            tvec=tvec,
            camera_center=camera_center,
            camera_forward=camera_forward,
            principal_point=(float(goal_intrinsics.cx), float(goal_intrinsics.cy)),
        )
        diagnostics.update(anchor_diagnostics)
        if object_anchor is not None:
            diagnostics["viewpoint_pnp_object_anchor_m"] = [
                round(float(object_anchor[0]), 4),
                round(float(object_anchor[1]), 4),
                round(float(object_anchor[2]), 4),
            ]
    cluster_center = np.mean(points[:count], axis=0)
    distance_from_cluster = float(np.linalg.norm(camera_center[[0, 2]] - cluster_center[[0, 2]]))
    diagnostics["viewpoint_pnp_center_m"] = [
        round(float(camera_center[0]), 4),
        round(float(camera_center[1]), 4),
        round(float(camera_center[2]), 4),
    ]
    diagnostics["viewpoint_pnp_distance_from_cluster_m"] = round(distance_from_cluster, 4)
    if distance_from_cluster < float(min_distance_from_cluster_m):
        diagnostics["viewpoint_pnp_reject_reason"] = "camera_center_too_close"
        return None, diagnostics
    if distance_from_cluster > float(max_distance_from_cluster_m):
        diagnostics["viewpoint_pnp_reject_reason"] = "camera_center_too_far"
        return None, diagnostics
    diagnostics["viewpoint_pnp_selected"] = True
    return camera_center.astype(np.float32), diagnostics


def estimate_visual_pose_servo_goal(
    *,
    evidence: FrameGeometryEvidence,
    goal_intrinsics: CameraIntrinsics,
    world_from_camera_cv: np.ndarray,
    agent_position: np.ndarray,
    min_matches: int,
    min_inliers: int,
    min_inlier_fraction: float,
    reproj_threshold_px: float,
    min_confidence: float,
    min_agent_distance: float,
    max_agent_distance: float,
    max_height_delta: float,
    min_inlier_coverage: float = 0.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate the goal camera center from one shared frame projection."""

    if not isinstance(evidence, FrameGeometryEvidence):
        raise TypeError("evidence must be a FrameGeometryEvidence")
    matches = evidence.matches
    match_count = evidence.match_count
    trace: dict[str, Any] = {
        "enabled": True,
        "attempted": True,
        "match_count": match_count,
        "accepted": False,
        "reject_reason": None,
    }
    if match_count < max(4, int(min_matches)):
        trace["reject_reason"] = "not_enough_matches"
        return None, trace

    confidence = np.asarray(
        matches.fine_conf,
        dtype=np.float32,
    ).reshape(-1)
    confidence_mask = confidence >= float(min_confidence)
    confidence_filtered_count = int(np.count_nonzero(confidence_mask))
    trace["confidence_filtered_count"] = confidence_filtered_count
    if confidence_filtered_count < max(4, int(min_matches)):
        trace["reject_reason"] = "not_enough_confident_matches"
        return None, trace

    points_world, projected_match_indices = (
        _projected_geometry_for_match_mask(
            evidence,
            confidence_mask,
        )
    )
    goal_xy = np.asarray(
        matches.goal_xy,
        dtype=np.float32,
    ).reshape(-1, 2)[projected_match_indices]
    trace["projected_count"] = int(len(points_world))
    if len(points_world) < max(4, int(min_matches)):
        trace["reject_reason"] = "not_enough_projected_points"
        return None, trace

    viewpoint_center, pnp_trace = estimate_goal_viewpoint_pnp(
        points_world,
        goal_xy,
        goal_intrinsics,
        min_points=int(min_inliers),
        reproj_threshold_px=float(reproj_threshold_px),
        min_inlier_fraction=float(min_inlier_fraction),
        min_distance_from_cluster_m=0.0,
        max_distance_from_cluster_m=100.0,
    )
    trace.update(pnp_trace)
    if viewpoint_center is None:
        trace["reject_reason"] = str(
            pnp_trace.get("viewpoint_pnp_reject_reason")
            or "pose_pnp_rejected"
        )
        return None, trace

    goal = np.asarray(viewpoint_center, dtype=np.float32).reshape(-1)
    if goal.size != 3 or not np.all(np.isfinite(goal)):
        trace["reject_reason"] = "nonfinite_pose_goal"
        return None, trace
    if float(min_inlier_coverage) > 0.0 and float(
        pnp_trace.get("viewpoint_pnp_inlier_coverage", 0.0) or 0.0
    ) < float(min_inlier_coverage):
        trace["reject_reason"] = "inlier_coverage_too_small"
        return None, trace
    camera_center = np.asarray(
        world_from_camera_cv,
        dtype=np.float32,
    )[:3, 3]
    height_delta: float | None = None
    if (
        np.all(np.isfinite(camera_center))
        and float(max_height_delta) >= 0.0
    ):
        height_delta = abs(float(goal[1]) - float(camera_center[1]))
    trace["measurement_m"] = _world_position_list(goal)
    trace["measurement_valid"] = bool(
        height_delta is None or height_delta <= float(max_height_delta)
    )
    agent = np.asarray(agent_position, dtype=np.float32).reshape(-1)
    if agent.size != 3 or not np.all(np.isfinite(agent)):
        trace["reject_reason"] = "invalid_agent_position"
        return None, trace
    agent_distance = float(
        np.linalg.norm(goal[[0, 2]] - agent[[0, 2]])
    )
    trace["agent_distance_m_unrounded"] = float(agent_distance)
    trace["agent_distance_m"] = round(float(agent_distance), 4)
    if agent_distance < float(min_agent_distance):
        trace["reject_reason"] = "pose_goal_too_close"
        return None, trace
    if agent_distance > float(max_agent_distance):
        trace["reject_reason"] = "pose_goal_too_far"
        return None, trace

    if height_delta is not None:
        trace["height_delta_m_unrounded"] = float(height_delta)
        trace["height_delta_m"] = round(float(height_delta), 4)
        if height_delta > float(max_height_delta):
            trace["reject_reason"] = "pose_goal_height_delta_too_large"
            return None, trace

    trace["accepted"] = True
    trace["goal_m"] = _world_position_list(goal)
    return goal.astype(np.float32), trace


def _localize_instance_image_matches(
    matches: XFeatStarMatches,
    *,
    geometry_evidence: FrameGeometryEvidence,
    config: ImageTargetConfig,
    goal_region_filtered: int = 0,
    precomputed_inlier_mask: np.ndarray | None = None,
    source_match_indices: np.ndarray | None = None,
) -> ImageTargetResult:
    diagnostics: dict[str, Any] = {
        "enabled": True,
        "attempted": True,
        "raw_matches": 0,
        "coarse_filtered_matches": 0,
        "fine_conf_filtered_matches": 0,
        "ransac_inliers": 0,
        "confidence_sum": None,
        "mean_confidence": None,
        "mean_coarse_sim": None,
        "projected_point_count": 0,
        "projected_points": [],
        "cluster_size": 0,
        "cluster_confidence_sum": None,
        "cluster_mean_confidence": None,
        "cluster_mean_coarse_sim": None,
        "target_updated": False,
        "reject_reason": None,
        "goal_region_filtered_matches": int(goal_region_filtered),
    }

    diagnostics.update(
        {
            "raw_matches": int(matches.raw_match_count),
            "coarse_filtered_matches": int(matches.coarse_filtered_count),
            "fine_conf_filtered_matches": int(len(matches.current_xy)),
        }
    )
    match_count = int(len(matches.current_xy))
    if source_match_indices is None:
        source_indices = np.arange(match_count, dtype=np.int64)
    else:
        source_indices = np.asarray(
            source_match_indices,
            dtype=np.int64,
        ).reshape(-1)
    if len(source_indices) != match_count:
        raise ValueError(
            "source_match_indices must align with filtered matches"
        )
    if (
        np.any(source_indices < 0)
        or np.any(source_indices >= geometry_evidence.match_count)
        or len(np.unique(source_indices)) != len(source_indices)
    ):
        raise ValueError(
            "source_match_indices must uniquely identify frame matches"
        )
    visualization = ImageMatchDebugData(
        matches=matches,
        inlier_mask=np.zeros((match_count,), dtype=bool),
        valid_depth_mask=np.zeros((match_count,), dtype=bool),
        cluster_mask=np.zeros((match_count,), dtype=bool),
    )
    if len(matches.current_xy) < int(config.min_matches):
        diagnostics["reject_reason"] = "not_enough_fine_conf_matches"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)

    if precomputed_inlier_mask is None:
        inlier_mask = ransac_inlier_mask(
            matches.goal_xy,
            matches.current_xy,
            reproj_threshold_px=float(config.ransac_reproj_threshold_px),
        )
    else:
        inlier_mask = np.asarray(
            precomputed_inlier_mask,
            dtype=bool,
        ).reshape(-1)
        if len(inlier_mask) != match_count:
            raise ValueError(
                "precomputed_inlier_mask must align with filtered matches"
            )
    visualization.inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)[:match_count].copy()
    inlier_count = int(np.count_nonzero(inlier_mask))
    diagnostics["ransac_inliers"] = inlier_count
    if inlier_count < int(config.min_inliers):
        diagnostics["reject_reason"] = "not_enough_ransac_inliers"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)

    inlier_conf = matches.fine_conf[inlier_mask]
    inlier_coarse = matches.coarse_sim[inlier_mask]
    inlier_confidence_sum = float(np.sum(inlier_conf))
    inlier_mean_confidence = float(np.mean(inlier_conf))
    inlier_mean_coarse_sim = float(np.mean(inlier_coarse))
    diagnostics.update(
        {
            "confidence_sum": _round_float(inlier_confidence_sum, 4),
            "mean_confidence": _round_float(inlier_mean_confidence, 4),
            "mean_coarse_sim": _round_float(inlier_mean_coarse_sim, 4),
        }
    )
    if inlier_mean_confidence < float(config.min_mean_confidence):
        diagnostics["reject_reason"] = "mean_confidence_below_threshold"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)
    if inlier_confidence_sum < float(config.min_confidence_sum):
        diagnostics["reject_reason"] = "confidence_sum_below_threshold"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)
    if inlier_mean_coarse_sim < float(config.min_coarse_sim):
        diagnostics["reject_reason"] = "mean_coarse_similarity_below_threshold"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)

    selected_frame_mask = np.zeros(
        (geometry_evidence.match_count,),
        dtype=bool,
    )
    selected_frame_mask[source_indices[inlier_mask]] = True
    world_points, projected_frame_indices = (
        _projected_geometry_for_match_mask(
            geometry_evidence,
            selected_frame_mask,
        )
    )
    valid_local_mask = (
        np.asarray(inlier_mask, dtype=bool)
        & geometry_evidence.valid_depth_mask[source_indices]
    )
    valid_projected_match_indices = np.flatnonzero(valid_local_mask)
    if not np.array_equal(
        source_indices[valid_projected_match_indices],
        projected_frame_indices,
    ):
        raise ValueError(
            "projected frame geometry is not aligned with filtered matches"
        )
    visualization.valid_depth_mask = np.zeros((match_count,), dtype=bool)
    visualization.valid_depth_mask[valid_projected_match_indices] = True
    projected_conf = matches.fine_conf[valid_local_mask]
    projected_coarse = matches.coarse_sim[valid_local_mask]
    diagnostics.update(
        {
            "projected_point_count": int(len(world_points)),
            "projected_points": _points_to_list(world_points),
        }
    )
    if len(world_points) < int(config.min_inliers):
        diagnostics["reject_reason"] = "not_enough_projected_points"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)

    cluster = select_confidence_cluster(
        world_points,
        projected_conf,
        projected_coarse,
        eps_m=float(config.dbscan_eps_m),
        min_samples=int(config.dbscan_min_samples),
    )
    if cluster is None:
        diagnostics["reject_reason"] = "no_depth_cluster"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)
    visualization.cluster_mask = np.zeros((match_count,), dtype=bool)
    visualization.cluster_mask[valid_projected_match_indices[np.asarray(cluster.mask, dtype=bool).reshape(-1)]] = True
    diagnostics.update(
        {
            "cluster_size": int(cluster.size),
            "cluster_confidence_sum": _round_float(cluster.confidence_sum, 4),
            "cluster_mean_confidence": _round_float(cluster.mean_confidence, 4),
            "cluster_mean_coarse_sim": _round_float(cluster.mean_coarse_sim, 4),
        }
    )
    if int(cluster.size) < int(config.min_inliers):
        diagnostics["reject_reason"] = "cluster_too_small"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)
    if float(cluster.confidence_sum) < float(config.min_confidence_sum):
        diagnostics["reject_reason"] = "cluster_confidence_sum_below_threshold"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)
    if float(cluster.mean_confidence) < float(config.min_mean_confidence):
        diagnostics["reject_reason"] = "cluster_mean_confidence_below_threshold"
        return ImageTargetResult(node=None, diagnostics=diagnostics, visualization=visualization)

    cluster_points = world_points[cluster.mask]
    cluster_conf = projected_conf[cluster.mask]
    weight_sum = float(np.sum(cluster_conf))
    if weight_sum > 1e-6:
        centroid = np.average(cluster_points, axis=0, weights=cluster_conf).astype(np.float32)
    else:
        centroid = np.mean(cluster_points, axis=0).astype(np.float32)
    bbox_min = np.min(cluster_points, axis=0).astype(np.float32)
    bbox_max = np.max(cluster_points, axis=0).astype(np.float32)
    diagnostics["target_updated"] = True

    node = {
        "id": -1,
        "label": "instance_image",
        "match_score": float(cluster.confidence_sum),
        "semantic_confidence": float(cluster.mean_confidence),
        "num_surfels": int(inlier_count),
        "centroid_m": centroid,
        "target_centroid_m": centroid.copy(),
        "bbox_min_m": bbox_min,
        "bbox_max_m": bbox_max,
        "goal_position": centroid.copy(),
        "image_match": {
            "confidence_sum": float(cluster.confidence_sum),
            "mean_confidence": float(cluster.mean_confidence),
            "mean_coarse_sim": float(cluster.mean_coarse_sim),
            "cluster_size": int(cluster.size),
            "ransac_inliers": int(inlier_count),
        },
    }
    return ImageTargetResult(node=node, diagnostics=diagnostics, visualization=visualization)


def _uses_goal_region(config: ImageTargetConfig) -> bool:
    return float(config.goal_min_y_frac) > 0.0


def _goal_image_region_source_indices(
    matches: XFeatStarMatches,
    goal_image_shape: tuple[int, ...],
    *,
    min_x_frac: float,
    max_x_frac: float,
    min_y_frac: float,
    max_y_frac: float,
) -> np.ndarray:
    count = int(len(matches.current_xy))
    height = int(goal_image_shape[0]) if len(goal_image_shape) >= 1 else 0
    width = int(goal_image_shape[1]) if len(goal_image_shape) >= 2 else 0
    if width <= 0 or height <= 0 or count == 0:
        return np.arange(count, dtype=np.int64)

    min_x = float(np.clip(min_x_frac, 0.0, 1.0)) * float(width)
    max_x = float(np.clip(max_x_frac, 0.0, 1.0)) * float(width)
    min_y = float(np.clip(min_y_frac, 0.0, 1.0)) * float(height)
    max_y = float(np.clip(max_y_frac, 0.0, 1.0)) * float(height)
    if (
        min_x <= 0.0
        and max_x >= float(width)
        and min_y <= 0.0
        and max_y >= float(height)
    ):
        return np.arange(count, dtype=np.int64)
    if max_x <= min_x or max_y <= min_y:
        return np.zeros((0,), dtype=np.int64)

    goal_xy = np.asarray(matches.goal_xy, dtype=np.float32).reshape(-1, 2)
    keep = (
        (goal_xy[:, 0] >= min_x)
        & (goal_xy[:, 0] <= max_x)
        & (goal_xy[:, 1] >= min_y)
        & (goal_xy[:, 1] <= max_y)
    )
    return np.flatnonzero(keep).astype(np.int64, copy=False)


def _copy_candidate_diagnostics(
    source: ImageTargetResult,
    *,
    prefix: str,
) -> dict[str, Any]:
    diagnostics = source.diagnostics
    keys = (
        "reject_reason",
        "raw_matches",
        "coarse_filtered_matches",
        "fine_conf_filtered_matches",
        "goal_region_filtered_matches",
        "ransac_inliers",
        "confidence_sum",
        "mean_confidence",
        "projected_point_count",
        "cluster_size",
        "cluster_confidence_sum",
        "cluster_mean_confidence",
        "target_updated",
    )
    return {f"{prefix}_{key}": diagnostics.get(key) for key in keys}


def localize_instance_image_evidence(
    evidence: ImageSpaceMatchEvidence | FrameGeometryEvidence,
    *,
    goal_image_shape: tuple[int, ...],
    config: ImageTargetConfig,
    depth: np.ndarray | None = None,
    intrinsics: CameraIntrinsics | None = None,
    world_from_camera_cv: np.ndarray | None = None,
) -> ImageTargetResult:
    """Localize one match frame, reusing preprojected geometry if given."""

    if isinstance(evidence, FrameGeometryEvidence):
        frame_evidence = evidence
        image_evidence = evidence.image_evidence
    elif isinstance(evidence, ImageSpaceMatchEvidence):
        if depth is None or intrinsics is None or world_from_camera_cv is None:
            raise ValueError(
                "depth, intrinsics, and world_from_camera_cv are required "
                "for image-space evidence"
            )
        image_evidence = evidence
        frame_evidence = build_frame_geometry_evidence(
            image_evidence,
            depth=depth,
            intrinsics=intrinsics,
            world_from_camera_cv=world_from_camera_cv,
            min_depth_m=float(config.min_depth_m),
            max_depth_m=float(config.max_depth_m),
        )
    else:
        raise TypeError(
            "evidence must be ImageSpaceMatchEvidence or "
            "FrameGeometryEvidence"
        )

    matches = image_evidence.matches
    full_result = _localize_instance_image_matches(
        matches,
        geometry_evidence=frame_evidence,
        config=config,
        precomputed_inlier_mask=image_evidence.inlier_mask,
    )
    if not _uses_goal_region(config):
        return full_result

    region_matches, goal_region_filtered = filter_goal_image_region(
        matches,
        goal_image_shape,
        min_x_frac=0.0,
        max_x_frac=1.0,
        min_y_frac=float(config.goal_min_y_frac),
        max_y_frac=1.0,
    )
    region_source_indices = _goal_image_region_source_indices(
        matches,
        goal_image_shape,
        min_x_frac=0.0,
        max_x_frac=1.0,
        min_y_frac=float(config.goal_min_y_frac),
        max_y_frac=1.0,
    )
    if len(region_source_indices) != len(region_matches.current_xy):
        raise ValueError(
            "goal-region filter did not preserve frame match alignment"
        )
    region_result = _localize_instance_image_matches(
        region_matches,
        geometry_evidence=frame_evidence,
        config=config,
        goal_region_filtered=goal_region_filtered,
        source_match_indices=region_source_indices,
    )

    if region_result.node is not None:
        region_result.diagnostics["goal_region_attempted"] = True
        region_result.diagnostics["goal_region_selected"] = True
        region_result.diagnostics.update(_copy_candidate_diagnostics(full_result, prefix="full_candidate"))
        return region_result

    region_inliers = int(region_result.diagnostics.get("ransac_inliers") or 0)
    if region_inliers >= int(config.min_inliers):
        region_result.diagnostics["goal_region_attempted"] = True
        region_result.diagnostics["goal_region_selected"] = False
        region_result.diagnostics["goal_region_blocked_full_candidate"] = True
        region_result.diagnostics.update(_copy_candidate_diagnostics(full_result, prefix="full_candidate"))
        return region_result

    full_result.diagnostics["goal_region_attempted"] = True
    full_result.diagnostics["goal_region_selected"] = False
    full_result.diagnostics.update(_copy_candidate_diagnostics(region_result, prefix="goal_region_candidate"))
    return full_result


def localize_instance_image_target(
    matcher: XFeatStarMatcher,
    *,
    goal_image_rgb: np.ndarray,
    current_image_rgb: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    world_from_camera_cv: np.ndarray,
    config: ImageTargetConfig,
) -> ImageTargetResult:
    """Match the goal image and localize the target in world coordinates."""

    evidence = match_instance_image_evidence(
        matcher,
        goal_image_rgb=goal_image_rgb,
        current_image_rgb=current_image_rgb,
        config=config,
    )
    return localize_instance_image_evidence(
        evidence,
        goal_image_shape=tuple(np.asarray(goal_image_rgb).shape),
        depth=depth,
        intrinsics=intrinsics,
        world_from_camera_cv=world_from_camera_cv,
        config=config,
    )
