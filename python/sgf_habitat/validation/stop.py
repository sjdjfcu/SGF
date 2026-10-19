from __future__ import annotations

import base64
import contextlib
import io
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLOE

from ..decision.hypothesis_projection import (
    ProjectedHypothesis,
    detection_share_inside,
    select_grounded_detection,
)
from .factored_evidence import combine_factored_verdict
from ..runtime.timing import RuntimeTiming, detail_scope
from ..goal.llm_parser import (
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    _extract_json_payload,
    _extract_text_content,
    _load_config_file,
    resolve_endpoint_field,
    warn_once_if_unauthenticated,
)


@dataclass
class VisionLanguageConfig:
    config_path: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 20.0


VERIFICATION_OBJECTS = ("node_footprint", "top_detection", "whole_frame")


@dataclass
class StopValidationResult:
    approved: bool
    validator_mode: str
    reason: str
    detection_found: bool
    detection_confidence: float = 0.0
    detection_label: str = ""
    detection_bbox_xyxy: tuple[int, int, int, int] | None = None
    detection_bbox_area_fraction: float = 0.0
    detection_bbox_center: tuple[float, float] | None = None
    p_yes: float | None = None
    scene_p_yes: float | None = None
    raw_response: str = ""
    evidence_source: str = "unspecified"
    hypothesis_overlap: float = 0.0
    hypothesis_depth_support: float = 0.0
    # Factored appearance/context evidence (validation/factored_evidence.py)
    appearance_p: float | None = None
    context_p: float | None = None
    context_observability: float = 0.0
    factored: bool = False

    def __post_init__(self) -> None:
        if self.p_yes is None:
            self.p_yes = 0.85 if self.approved else 0.15
        self.p_yes = _clamp_probability(self.p_yes)
        if self.scene_p_yes is None:
            self.scene_p_yes = self.p_yes if self.validator_mode.startswith("vlm_") and not self.detection_found else 0.0
        self.scene_p_yes = _clamp_probability(self.scene_p_yes)


def _clamp_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(probability):
        return 0.0
    return float(max(0.0, min(1.0, probability)))


def _payload_probability(payload: dict[str, Any], *, approved: bool) -> float:
    for key in (
        "p_yes",
        "probability",
        "yes_probability",
        "target_probability",
    ):
        if key in payload:
            return _clamp_probability(payload[key])
    for key in (
        "confidence",
        "target_confidence",
        "is_exact_target_confidence",
        "score",
    ):
        if key in payload:
            confidence = _clamp_probability(payload[key])
            if approved:
                return confidence
            # VLMs differ on what confidence means for a negative answer, so cap it
            return min(confidence, 1.0 - confidence, 0.15)
    return 0.85 if approved else 0.15


def resolve_vlm_config(config: VisionLanguageConfig) -> VisionLanguageConfig:
    loaded = _load_config_file(config.config_path)
    base_url = resolve_endpoint_field(
        config.base_url, loaded.get("base_url", ""), LLM_BASE_URL_ENV
    )
    api_key = resolve_endpoint_field(
        config.api_key, loaded.get("api_key", ""), LLM_API_KEY_ENV
    )
    warn_once_if_unauthenticated(base_url, api_key)
    model = config.model or str(loaded.get("vlm_model", "") or loaded.get("llm_model", "")).strip()
    return VisionLanguageConfig(
        config_path=config.config_path,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=config.timeout_seconds,
    )


def _encode_png_base64(image_bgr: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError("failed to encode validation image")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def build_validation_canvas(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    """Draw the box without its label, which would leak the goal query."""

    annotated = image_bgr.copy()
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (24, 82, 255), 3)
    return annotated


class StopValidator:
    def __init__(
        self,
        *,
        yolo_model_path: str,
        detector_queries: list[str],
        yolo_conf_threshold: float,
        yolo_iou_threshold: float,
        vlm_config: VisionLanguageConfig,
        detector_device: str = "auto",
        fallback_queries: list[str] | None = None,
        low_confidence_scene_recheck_threshold: float = 0.08,
        scene_only_min_p_yes: float = 0.0,
        timing: RuntimeTiming | None = None,
    ) -> None:
        self.timing = timing
        self.yolo_model_path = yolo_model_path
        self.detector_queries = [query for query in detector_queries if query]
        self.fallback_queries = [q for q in (fallback_queries or []) if q and q not in self.detector_queries]
        self.yolo_conf_threshold = float(yolo_conf_threshold)
        self.yolo_iou_threshold = float(yolo_iou_threshold)
        self.low_confidence_scene_recheck_threshold = float(low_confidence_scene_recheck_threshold)
        self.scene_only_min_p_yes = float(scene_only_min_p_yes)
        self.vlm_config = resolve_vlm_config(vlm_config)
        self.detector_device = str(detector_device or "auto").strip() or "auto"
        self._detector: YOLOE | None = None
        self._query_embeddings: dict[tuple[str, ...], torch.Tensor] = {}
        self._active_queries: list[str] = list(self.detector_queries)

    def _set_detector_classes(self, model: YOLOE, queries: list[str]) -> None:
        # Cache text embeddings per query set to avoid recomputing on each switch
        key = tuple(queries)
        if key not in self._query_embeddings:
            self._query_embeddings[key] = model.get_text_pe(queries)
        model.set_classes(queries, embeddings=self._query_embeddings[key])

    def _ensure_detector(self) -> YOLOE:
        if self._detector is not None:
            return self._detector
        with contextlib.redirect_stdout(io.StringIO()):
            model = YOLOE(self.yolo_model_path)
        device = self.detector_device
        if device.lower() == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        if self.detector_queries:
            with contextlib.redirect_stdout(io.StringIO()):
                self._set_detector_classes(model, self.detector_queries)
        self._active_queries = list(self.detector_queries)
        self._detector = model
        return model

    def _switch_to_fallback(self) -> bool:
        if not self.fallback_queries or not self._detector:
            return False
        with detail_scope(self.timing, "detector_set_classes"), contextlib.redirect_stdout(io.StringIO()):
            self._set_detector_classes(self._detector, self.fallback_queries)
        self._active_queries = list(self.fallback_queries)
        return True

    def _switch_to_primary(self) -> None:
        if self._detector and self.detector_queries:
            with detail_scope(self.timing, "detector_set_classes"), contextlib.redirect_stdout(io.StringIO()):
                self._set_detector_classes(self._detector, self.detector_queries)
            self._active_queries = list(self.detector_queries)

    def _all_detections(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        if self._detector is None:
            with detail_scope(self.timing, "detector_initialization"):
                self._ensure_detector()
        model = self._detector
        assert model is not None
        with detail_scope(self.timing, "detector_predict"), contextlib.redirect_stdout(io.StringIO()):
            results = model.predict(
                source=image_bgr,
                verbose=False,
                conf=self.yolo_conf_threshold,
                iou=self.yolo_iou_threshold,
                imgsz=640,
                retina_masks=True,
            )

        with detail_scope(self.timing, "detector_postprocess"):
            detections: list[dict[str, Any]] = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                confs = result.boxes.conf.detach().cpu().numpy()
                classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
                names = (
                    result.names
                    if isinstance(result.names, dict)
                    else {idx: name for idx, name in enumerate(result.names)}
                )
                for idx in range(len(classes)):
                    box = boxes[idx].astype(np.int32).tolist()
                    detections.append(
                        {
                            "confidence": float(confs[idx]),
                            "label": str(names.get(int(classes[idx]), classes[idx])),
                            "bbox_xyxy": (
                                int(box[0]),
                                int(box[1]),
                                int(box[2]),
                                int(box[3]),
                            ),
                        }
                    )
        return detections

    @staticmethod
    def _most_confident(
        detections: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for detection in detections:
            if best is None or float(detection["confidence"]) > float(
                best["confidence"]
            ):
                best = detection
        return best

    def _top_detection(self, image_bgr: np.ndarray) -> dict[str, Any] | None:
        return self._most_confident(self._all_detections(image_bgr))

    @staticmethod
    def _detection_geometry(
        image_bgr: np.ndarray,
        detection: dict[str, Any],
    ) -> tuple[tuple[int, int, int, int], float, tuple[float, float]]:
        height, width = image_bgr.shape[:2]
        bbox = detection["bbox_xyxy"]
        x1 = max(0, min(max(0, width - 1), int(bbox[0])))
        y1 = max(0, min(max(0, height - 1), int(bbox[1])))
        x2 = max(x1 + 1, min(width, int(bbox[2])))
        y2 = max(y1 + 1, min(height, int(bbox[3])))
        area_fraction = float((x2 - x1) * (y2 - y1)) / max(1.0, float(width * height))
        center = (
            (float(x1) + float(x2)) * 0.5 / max(1.0, float(width)),
            (float(y1) + float(y2)) * 0.5 / max(1.0, float(height)),
        )
        return (x1, y1, x2, y2), area_fraction, center

    def _call_vlm(self, prompt: str, image_bgr: np.ndarray) -> str:
        if not (self.vlm_config.base_url and self.vlm_config.model):
            raise RuntimeError("VLM config is incomplete")
        with detail_scope(self.timing, "image_encoding"):
            image_base64 = _encode_png_base64(image_bgr)
        url = self.vlm_config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.vlm_config.api_key:
            headers["Authorization"] = f"Bearer {self.vlm_config.api_key}"
        payload = {
            "model": self.vlm_config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You verify whether a highlighted indoor object exactly matches a navigation instruction. Return JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                },
            ],
        }
        with detail_scope(self.timing, "vlm_request"):
            response = requests.post(url, headers=headers, json=payload, timeout=self.vlm_config.timeout_seconds)
            response.raise_for_status()
        with detail_scope(self.timing, "response_parsing"):
            message_content = response.json()["choices"][0]["message"]["content"]
            return _extract_text_content(message_content).strip()

    def _scene_visibility_prompt(self, text_goal: str) -> str:
        return (
            "You are checking whether the exact target object is visible in this image for an indoor navigation episode.\n"
            f"Full navigation instruction:\n{text_goal}\n\n"
            "Is the exact described target visible strongly enough that the agent should STOP now?\n"
            "Note: the camera may be VERY close to the target, so large objects like beds, sofas, tables, and TV monitors "
            "may fill most of the frame or only be partially visible.\n"
            "Return JSON with this schema:\n"
            '{"is_exact_target": true/false, "confidence": 0.0-1.0, "reason": "short explanation"}\n'
            "Rules:\n"
            "- Confidence measures your confidence in your verdict itself (whether is_exact_target is true or false).\n"
            "- Return true only if the target object itself is clearly identifiable somewhere in the current image.\n"
            "- If the instruction contains distinctive surrounding context or multiple same-category cues, require visible evidence that this is that specific target, not just any object of the same category.\n"
            "- If the instruction is sparse and mostly names the category, a clear category match is enough.\n"
            "- Return false if the object is absent, only faintly implied by nearby context, or only a tiny ambiguous fragment is visible.\n"
            "- Partial close-up views are acceptable only when the object is still clearly identifiable as the target.\n"
            "- If the image is ambiguous and the distinctive cues from the instruction are not checkable, prefer false.\n"
            "- Reason must be one short sentence.\n"
        )

    def _scene_level_check(
        self,
        *,
        text_goal: str,
        rgb_bgr: np.ndarray,
        validator_mode: str,
        detection_found: bool,
        detection_confidence: float = 0.0,
        detection_label: str = "",
    ) -> StopValidationResult | None:
        if not self.vlm_config.base_url:
            return None
        with detail_scope(self.timing, "vlm_scene_check"):
            raw_response = self._call_vlm(self._scene_visibility_prompt(text_goal), rgb_bgr)
        payload = _extract_json_payload(raw_response)
        if not payload:
            raise RuntimeError("VLM did not return a JSON object")
        approved = bool(payload.get("is_exact_target", False))
        p_yes = _payload_probability(payload, approved=approved)
        reason = str(payload.get("reason", "")).strip() or ("target clearly visible in scene" if approved else "target not clearly visible")
        return StopValidationResult(
            approved=approved,
            validator_mode=validator_mode,
            reason=reason,
            detection_found=detection_found,
            detection_confidence=float(detection_confidence),
            detection_label=detection_label,
            p_yes=p_yes,
            scene_p_yes=p_yes,
            raw_response=raw_response,
        )

    def validate(
        self,
        *,
        text_goal: str,
        canonical_goal: str,
        rgb_bgr: np.ndarray,
        hypothesis: ProjectedHypothesis | None = None,
        footprint_bbox_xyxy: tuple[int, int, int, int] | None = None,
        verification_object: str = "node_footprint",
    ) -> StopValidationResult:
        """Verify the pursued hypothesis against the navigation instruction."""

        _ = canonical_goal
        if verification_object not in VERIFICATION_OBJECTS:
            raise ValueError(
                f"unknown verification object: {verification_object!r}"
            )
        if verification_object == "whole_frame":
            return self._validate_whole_frame(
                text_goal=text_goal,
                rgb_bgr=rgb_bgr,
                hypothesis=hypothesis,
                footprint_bbox_xyxy=footprint_bbox_xyxy,
            )
        hypothesis_depth_support = (
            0.0 if hypothesis is None else float(hypothesis.depth_support)
        )
        if verification_object == "top_detection":
            hypothesis = None
            footprint_bbox_xyxy = None
        detections = self._all_detections(rgb_bgr)
        if not detections and self.fallback_queries:
            # Try with broader canonical queries before VLM fallback
            if self._switch_to_fallback():
                detections = self._all_detections(rgb_bgr)
                self._switch_to_primary()

        evidence_source = "none"
        hypothesis_overlap = 0.0
        detection: dict[str, Any] | None = None
        if hypothesis is not None and hypothesis.depth_samples > 0:
            grounded, overlap = select_grounded_detection(detections, hypothesis)
            if grounded is not None:
                detection = grounded
                hypothesis_overlap = overlap
                evidence_source = "detector_grounded"
            elif hypothesis.depth_support > 0.5:
                # Detector missed a depth-confirmed node, so review its projected box
                # (a zero detector score routes an approval through the scene recheck)
                detection = {
                    "confidence": 0.0,
                    "label": "",
                    "bbox_xyxy": tuple(hypothesis.bbox_xyxy),
                }
                hypothesis_overlap = 1.0
                evidence_source = "hypothesis_projection"
        if detection is None:
            # No grounded footprint, fall back to the most confident detection
            detection = self._most_confident(detections)
            evidence_source = "detector_top" if detection is not None else "none"
            if (
                detection is not None
                and footprint_bbox_xyxy is not None
                and detection_share_inside(
                    tuple(detection["bbox_xyxy"]), tuple(footprint_bbox_xyxy)
                )
                <= 0.0
            ):
                bbox_xyxy, bbox_area_fraction, bbox_center = self._detection_geometry(
                    rgb_bgr, detection
                )
                return StopValidationResult(
                    approved=False,
                    validator_mode="hypothesis_unobservable",
                    reason=(
                        "The frame's most confident detection lies outside the "
                        "pursued node's projected extent; the verifier was not asked."
                    ),
                    detection_found=True,
                    detection_confidence=float(detection["confidence"]),
                    detection_label=str(detection["label"]),
                    detection_bbox_xyxy=bbox_xyxy,
                    detection_bbox_area_fraction=float(bbox_area_fraction),
                    detection_bbox_center=bbox_center,
                    p_yes=0.5,
                    scene_p_yes=0.0,
                    evidence_source=evidence_source,
                    hypothesis_depth_support=hypothesis_depth_support,
                )

        if detection is None:
            # Fallback: ask VLM directly without a bounding box
            if self.vlm_config.base_url:
                try:
                    scene_result = self._scene_level_check(
                        text_goal=text_goal,
                        rgb_bgr=rgb_bgr,
                        validator_mode="vlm_no_detection_fallback",
                        detection_found=False,
                    )
                    if scene_result is not None:
                        scene_result.evidence_source = evidence_source
                        scene_result.hypothesis_depth_support = (
                            hypothesis_depth_support
                        )
                        if (
                            scene_result.approved
                            and float(scene_result.p_yes or 0.0) < self.scene_only_min_p_yes
                        ):
                            return StopValidationResult(
                                approved=False,
                                validator_mode="vlm_no_detection_low_p_yes",
                                reason=(
                                    "Scene-only verdict below the no-detection "
                                    f"probability floor ({scene_result.p_yes:.2f} < "
                                    f"{self.scene_only_min_p_yes:.2f})."
                                ),
                                detection_found=False,
                                p_yes=scene_result.p_yes,
                                scene_p_yes=scene_result.scene_p_yes,
                                raw_response=scene_result.raw_response,
                                evidence_source="none",
                                hypothesis_depth_support=hypothesis_depth_support,
                            )
                        return scene_result
                except Exception:
                    pass
            return StopValidationResult(
                approved=False,
                validator_mode="no_detection",
                reason="No target-class detection was visible in the current observation.",
                detection_found=False,
                p_yes=0.05,
                scene_p_yes=0.0,
                evidence_source=evidence_source,
                hypothesis_depth_support=hypothesis_depth_support,
            )

        bbox_xyxy, bbox_area_fraction, bbox_center = self._detection_geometry(rgb_bgr, detection)
        reviewed_area_fraction = float(bbox_area_fraction)
        reviewed_containment = (
            1.0 if hypothesis is None else float(hypothesis.containment)
        )
        validation_image = build_validation_canvas(rgb_bgr, bbox_xyxy)
        prompt = self._single_view_prompt(text_goal)
        with detail_scope(self.timing, "vlm_object_check"):
            raw_response = self._call_vlm(prompt, validation_image)
        payload = _extract_json_payload(raw_response)
        if not payload:
            raise RuntimeError("VLM did not return a JSON object")
        pooled_approved = bool(payload.get("is_exact_target", False))
        pooled_p_yes = _payload_probability(payload, approved=pooled_approved)
        verdict = combine_factored_verdict(
            payload,
            pooled_p_yes=pooled_p_yes,
            pooled_approved=pooled_approved,
            area_fraction=reviewed_area_fraction,
            containment=reviewed_containment,
        )
        approved = bool(verdict.approved)
        p_yes = float(verdict.p_yes)
        reason = str(payload.get("reason", "")).strip() or ("validated target match" if approved else "validation rejected target")
        scene_recheck_mode = ""
        scene_confirm_mode = ""
        if approved and float(detection["confidence"]) < self.low_confidence_scene_recheck_threshold:
            scene_recheck_mode = "vlm_low_conf_scene_recheck"
            scene_confirm_mode = "vlm_low_conf_scene_confirmed"
        if approved and scene_recheck_mode:
            try:
                scene_result = self._scene_level_check(
                    text_goal=text_goal,
                    rgb_bgr=rgb_bgr,
                    validator_mode=scene_recheck_mode,
                    detection_found=True,
                    detection_confidence=float(detection["confidence"]),
                    detection_label=str(detection["label"]),
                )
            except Exception:
                scene_result = None
            if scene_result is not None:
                if not scene_result.approved:
                    scene_result.evidence_source = evidence_source
                    scene_result.hypothesis_overlap = hypothesis_overlap
                    scene_result.hypothesis_depth_support = hypothesis_depth_support
                    return scene_result
                return StopValidationResult(
                    approved=True,
                    validator_mode=scene_confirm_mode,
                    reason=scene_result.reason,
                    detection_found=True,
                    detection_confidence=float(detection["confidence"]),
                    detection_label=str(detection["label"]),
                    detection_bbox_xyxy=bbox_xyxy,
                    detection_bbox_area_fraction=float(bbox_area_fraction),
                    detection_bbox_center=bbox_center,
                    p_yes=p_yes,
                    scene_p_yes=scene_result.scene_p_yes,
                    raw_response=scene_result.raw_response,
                    evidence_source=evidence_source,
                    hypothesis_overlap=hypothesis_overlap,
                    hypothesis_depth_support=hypothesis_depth_support,
                    appearance_p=verdict.appearance_p,
                    context_p=verdict.context_p,
                    context_observability=verdict.context_observability,
                    factored=verdict.factored,
                )
        return StopValidationResult(
            approved=approved,
            validator_mode="vlm",
            reason=reason,
            detection_found=True,
            detection_confidence=float(detection["confidence"]),
            detection_label=str(detection["label"]),
            detection_bbox_xyxy=bbox_xyxy,
            detection_bbox_area_fraction=float(bbox_area_fraction),
            detection_bbox_center=bbox_center,
            p_yes=p_yes,
            scene_p_yes=0.0,
            raw_response=raw_response,
            evidence_source=evidence_source,
            hypothesis_overlap=hypothesis_overlap,
            hypothesis_depth_support=hypothesis_depth_support,
            appearance_p=verdict.appearance_p,
            context_p=verdict.context_p,
            context_observability=verdict.context_observability,
            factored=verdict.factored,
        )

    def _validate_whole_frame(
        self,
        *,
        text_goal: str,
        rgb_bgr: np.ndarray,
        hypothesis: ProjectedHypothesis | None,
        footprint_bbox_xyxy: tuple[int, int, int, int] | None,
    ) -> StopValidationResult:
        hypothesis_depth_support = (
            0.0 if hypothesis is None else float(hypothesis.depth_support)
        )
        handle: tuple[int, int, int, int] | None = None
        if footprint_bbox_xyxy is not None:
            handle = tuple(int(value) for value in footprint_bbox_xyxy)
        elif hypothesis is not None and hypothesis.depth_samples > 0:
            handle = tuple(int(value) for value in hypothesis.bbox_xyxy)
        else:
            top = self._most_confident(self._all_detections(rgb_bgr))
            if top is not None:
                handle = tuple(int(value) for value in top["bbox_xyxy"])
        if handle is None:
            return StopValidationResult(
                approved=False,
                validator_mode="whole_frame_no_handle",
                reason="Nothing in view anchors the hypothesis; the verifier was not asked.",
                detection_found=False,
                p_yes=0.05,
                scene_p_yes=0.0,
                evidence_source="none",
                hypothesis_depth_support=hypothesis_depth_support,
            )
        scene_result = self._scene_level_check(
            text_goal=text_goal,
            rgb_bgr=rgb_bgr,
            validator_mode="whole_frame",
            detection_found=True,
        )
        if scene_result is None:
            raise RuntimeError("VLM config is incomplete")
        bbox_xyxy, bbox_area_fraction, bbox_center = self._detection_geometry(
            rgb_bgr,
            {"bbox_xyxy": handle},
        )
        p_yes = float(scene_result.p_yes or 0.0)
        approved = bool(
            scene_result.approved and p_yes >= self.scene_only_min_p_yes
        )
        return StopValidationResult(
            approved=approved,
            validator_mode=(
                "whole_frame"
                if approved or not scene_result.approved
                else "whole_frame_low_p_yes"
            ),
            reason=scene_result.reason,
            detection_found=True,
            detection_confidence=0.0,
            detection_label="",
            detection_bbox_xyxy=bbox_xyxy,
            detection_bbox_area_fraction=float(bbox_area_fraction),
            detection_bbox_center=bbox_center,
            p_yes=p_yes,
            scene_p_yes=scene_result.scene_p_yes,
            raw_response=scene_result.raw_response,
            evidence_source="whole_frame",
            hypothesis_depth_support=hypothesis_depth_support,
        )

    @staticmethod
    def _single_view_prompt(text_goal: str) -> str:
        return (
            "You are checking whether the highlighted object is the exact target for an indoor navigation episode.\n"
            f"Full navigation instruction:\n{text_goal}\n\n"
            "The image shows the current observation with the candidate object boxed in red.\n"
            "Return JSON with this schema:\n"
            '{"is_exact_target": true/false, "confidence": 0.0-1.0, '
            '"appearance_match": 0.0-1.0, "context_match": 0.0-1.0, '
            '"reason": "short explanation"}\n'
            "Rules:\n"
            "- appearance_match is the probability that the boxed object itself matches what the instruction says the target IS (category, material, colour, shape). Judge only the boxed pixels.\n"
            "- context_match is the probability that the visible surroundings match what the instruction says is AROUND the target. Return 0.5 when the instruction states no surroundings, or when the surroundings it describes are not visible in this image.\n"
            "- Report both numbers independently; do not let one decide the other.\n"
            "- Confidence measures your confidence in your verdict itself (whether is_exact_target is true or false).\n"
            "- This is exact-target verification, not category-only matching.\n"
            "- Lighting conditions affect perceived colors; do not treat slight color differences as a mismatch.\n"
            "- Do not invent unseen details.\n"
            "- Reason must be one short sentence.\n"
        )
