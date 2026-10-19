#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import runpy
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from sgf_habitat.runtime.nav_config import (  # noqa: E402
    DEFAULT_NAV_CONFIG_PATH,
    NAV_ARG_CONFIG_DEFAULTS,
    TASK_MODES,
    is_scalar,
    is_sensitive_key,
    normalize_scalar_type,
    parse_nav_override_value as parse_nav_override_scalar,
    resolve_nav_configs,
    validate_nav_override_type,
)


DEFAULT_RUNNER_PATH = REPO_ROOT / "scripts" / "run_habitat_text_image_goal_nav.py"
DEFAULT_BATCH_RUNNER_PATH = REPO_ROOT / "scripts" / "batch_eval_nav.py"
DEFAULT_COMPARE_RUNNER_PATH = REPO_ROOT / "scripts" / "compare_nav_runs.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "sgf_nav_ablations"
DEFAULT_BATCH_MAX_STEPS = 1000
EXECUTION_PLAN_VERSION = 1
EXECUTION_PLAN_PROBE_TIMEOUT_SECONDS = 30
SUBPROCESS_ERROR_TAIL_BYTES = 8192
EXECUTION_PLAN_INTERRUPT_TIMEOUT_SECONDS = 60
ABLATION_SELECTORS = frozenset({"all", "core", "components", "text-stop-controls", "representation-coupling"})
MANAGED_BATCH_OPTIONS = frozenset(
    {
        "--execution-plan",
        "--execution-plan-capabilities",
        "--execution-result",
        "--nav-config-path",
        "--metrics-only",
        "--no-aggregate-summary",
        "--output-root",
        "--overwrite",
        "--python-executable",
        "--resume",
        "--run-tag",
        "--runner-args",
        "--runner-path",
        "--task-mode",
    }
)


Scalar = bool | int | float | str


EVIDENCE_EXPECTATIONS = frozenset(
    {"disabled", "retained", "activated", "capability"}
)
EVIDENCE_ROLES = frozenset({"direct", "dependent", "context", "capability"})


@dataclass(frozen=True)
class EvidenceSpec:
    """One runtime counter and the relationship a profile expects."""

    evidence_key: str
    expectation: str = "disabled"

    def __post_init__(self) -> None:
        if not self.evidence_key:
            raise ValueError("evidence_key must be non-empty")
        if self.expectation not in EVIDENCE_EXPECTATIONS:
            raise ValueError(
                f"unknown evidence expectation: {self.expectation!r}"
            )


@dataclass(frozen=True)
class EvidenceRequirement:
    evidence_key: str
    expectation: str
    role: str

    def __post_init__(self) -> None:
        EvidenceSpec(self.evidence_key, self.expectation)
        if self.role not in EVIDENCE_ROLES:
            raise ValueError(f"unknown evidence role: {self.role!r}")


def disabled(key: str) -> EvidenceSpec:
    return EvidenceSpec(key, "disabled")


def retained(key: str) -> EvidenceSpec:
    return EvidenceSpec(key, "retained")


def activated(key: str) -> EvidenceSpec:
    return EvidenceSpec(key, "activated")


def capability(key: str) -> EvidenceSpec:
    return EvidenceSpec(key, "capability")


def _merged_expectation(key: str, previous: str, current: str) -> str:
    if previous == current:
        return previous
    if {previous, current} == {"retained", "capability"}:
        return "capability"
    raise ValueError(
        f"conflicting evidence expectations for {key}: "
        f"{previous} vs {current}"
    )


def merge_evidence_specs(
    *groups: Sequence[EvidenceSpec],
) -> tuple[EvidenceSpec, ...]:
    merged: dict[str, EvidenceSpec] = {}
    for spec in (item for group in groups for item in group):
        previous = merged.get(spec.evidence_key)
        if previous is None:
            merged[spec.evidence_key] = spec
            continue
        expectation = _merged_expectation(
            spec.evidence_key,
            previous.expectation,
            spec.expectation,
        )
        if expectation != previous.expectation:
            merged[spec.evidence_key] = EvidenceSpec(
                spec.evidence_key,
                expectation,
            )
    return tuple(merged.values())


def merge_evidence_requirements(
    *groups: Sequence[EvidenceRequirement],
) -> tuple[EvidenceRequirement, ...]:
    merged: dict[str, EvidenceRequirement] = {}
    for requirement in (item for group in groups for item in group):
        previous = merged.get(requirement.evidence_key)
        if previous is None:
            merged[requirement.evidence_key] = requirement
            continue
        expectation = _merged_expectation(
            requirement.evidence_key,
            previous.expectation,
            requirement.expectation,
        )
        if expectation != previous.expectation:
            merged[requirement.evidence_key] = EvidenceRequirement(
                requirement.evidence_key,
                expectation,
                "capability",
            )
    return tuple(merged.values())


def evidence_requirement_payload(
    requirement: EvidenceRequirement,
) -> dict[str, str]:
    return {
        "evidence_key": requirement.evidence_key,
        "expectation": requirement.expectation,
        "role": requirement.role,
    }


def parse_evidence_requirement(value: Any) -> EvidenceRequirement:
    if isinstance(value, EvidenceRequirement):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"invalid evidence requirement: {value!r}")
    return EvidenceRequirement(
        evidence_key=str(value.get("evidence_key", "")),
        expectation=str(value.get("expectation", "")),
        role=str(value.get("role", "")),
    )


# Runtime counters each navigation switch should disable or retain
CONFIG_KEY_EVIDENCE_SPECS: dict[str, tuple[EvidenceSpec, ...]] = {
    "text_goal_parser_mode": (
        disabled("llm_goal_parser"),
        disabled("parsed_target_phrases"),
    ),
    "text_context_guided_exploration": (disabled("context_guided_exploration"),),
    "text_query_conditional_labels": (disabled("text_query_conditional_retrieval"),),
    "sgf_stop": (disabled("text_sgf_stop"),),
    "sgf_explore": (disabled("sgf_exploration"),),
    # Enum switch: counter of the branch the full profile selects
    "sgf_explore_utility_form": (disabled("sgf_explore_rate_selection"),),
    "sgf_explore_exchange_rate": (),
    "sgf_explore_exchange_rate_scale": (),
    "sgf_explore_rotation_cost_weight": (
        disabled("sgf_explore_rotation_priced_selection"),
    ),
    "stop_validation": (disabled("stop_validation"),),
    "text_metric_visual_closure": (disabled("metric_visual_closure"),),
    "text_validated_stop_approach": (disabled("validated_stop_approach"),),
    "text_detection_probe": (disabled("detection_probe"),),
    "text_detection_instances": (
        disabled("text_detection_instances"),
        disabled("text_detection_instance_candidates"),
    ),
    "text_stop_verification_object": (
        disabled("text_node_grounded_evidence"),
        retained("stop_validation"),
    ),
    "text_hypothesis_triage": (
        disabled("text_hypothesis_triage"),
        retained("stop_ledger"),
    ),
    "text_stop_evidence_rule": (
        disabled("text_sequential_evidence_stop"),
        retained("stop_ledger"),
    ),
    "text_stop_reobservation": (
        disabled("text_reobservation_maneuver"),
        disabled("text_withheld_press"),
        retained("stop_ledger"),
    ),
    "text_stop_ledger": (disabled("stop_ledger"),),
    "text_stop_sequential_error_rate": (
        disabled("text_sequential_evidence_stop"),
        retained("stop_ledger"),
    ),
    "text_target_fatigue": (disabled("target_fatigue"),),
    "endgame_return": (disabled("endgame_return"),),
    "sgf_visual_memory": (disabled("visual_memory"),),
    "visual_pose_servo": (disabled("visual_pose_servo"),),
    "image_vlm_stop_validation": (
        disabled("image_vlm_stop_validation"),
    ),
    "panoramic_observation": (disabled("panoramic_observation"),),
    "panoramic_observation_trigger_scale": (
        disabled("panoramic_detector_scaled_interval"),
        retained("panoramic_observation"),
    ),
    "panoramic_observation_searched_extent_interval": (
        disabled("panoramic_searched_extent_interval"),
        retained("panoramic_observation"),
    ),
    "image_vlm_panoramic_query": (
        disabled("image_vlm_panoramic_query"),
        retained("panoramic_observation"),
    ),
    "image_vlm_foveated_hypothesis": (
        disabled("image_vlm_foveated_query"),
        disabled("image_vlm_foveated_hypothesis"),
        retained("image_vlm_panoramic_query"),
    ),
    "image_vlm_hypothesis_arrival_sweep": (
        disabled("image_vlm_hypothesis_arrival_sweep"),
        retained("image_vlm_foveated_hypothesis"),
    ),
    "image_reachable_pursuit": (
        disabled("image_reachable_pursuit"),
        retained("visual_pose_servo"),
    ),
    "image_goal_view_terminal": (
        disabled("image_goal_view_terminal"),
        retained("visual_pose_servo"),
    ),
    "image_target_localization": (disabled("visual_pose_servo"),),
    "image_navigation_target": (
        disabled("image_goal_camera_target_steps"),
        retained("image_goal_view_terminal"),
    ),
    "image_pursuit_controller": (disabled("image_straight_servo_actions"),),
    "image_goal_view_min_corroboration": (
        retained("image_goal_view_terminal"),
    ),
    "image_goal_view_commit_band": (
        disabled("image_goal_view_wide_band_commit"),
        retained("image_goal_view_terminal"),
    ),
    "image_goal_view_alignment": (
        disabled("image_goal_view_alignment_turns"),
        retained("image_goal_view_terminal"),
    ),
    "sgf_explore_evidence_verification": (
        disabled("sgf_explore_evidence_verification"),
    ),
    "navigation_graph_occupancy": (
        disabled("navigation_graph_occupancy"),
        disabled("navigation_graph_surfel_votes"),
        disabled("navigation_graph_removed_free_cells"),
        disabled("navigation_graph_added_occupied_cells"),
    ),
    "navigation_floor_support": (
        disabled("navigation_floor_support"),
        disabled("navigation_floor_support_veto_cells"),
    ),
    # Enum switch: counter of the branch the full profile selects
    "text_target_association": (disabled("text_navigation_map_association"),),
    # Counted only when some candidate carries ledger evidence
    "text_evidence_weighted_target_ranking": (
        disabled("text_evidence_weighted_ranking"),
        retained("stop_ledger"),
    ),
}


# Capabilities every non-baseline profile must retain
CAPABILITY_KEYS: dict[str, tuple[str, ...]] = {
    "text": ("sgf_exploration", "target_pursuit_actions", "decision_stop"),
    "instance-image": (
        "sgf_exploration",
        "target_pursuit_actions",
        "decision_stop",
    ),
}


CONFIG_KEY_EVIDENCE: dict[str, tuple[str, ...]] = {
    key: tuple(
        spec.evidence_key
        for spec in specs
        if spec.expectation == "disabled"
    )
    for key, specs in CONFIG_KEY_EVIDENCE_SPECS.items()
}


def evidence_specs_for_config_keys(
    keys: Sequence[str],
    *,
    expectations: frozenset[str] | None = None,
) -> tuple[EvidenceSpec, ...]:
    specs = merge_evidence_specs(
        *(
            tuple(
                spec
                for spec in CONFIG_KEY_EVIDENCE_SPECS.get(key, ())
                if expectations is None
                or spec.expectation in expectations
            )
            for key in keys
        )
    )
    return specs


def evidence_for_config_keys(keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        spec.evidence_key
        for spec in evidence_specs_for_config_keys(
            keys,
            expectations=frozenset({"disabled"}),
        )
    )


@dataclass(frozen=True)
class Ablation:
    description: str
    overrides: tuple[tuple[str, Scalar], ...] = ()
    family: str = ""
    evidence_keys: tuple[str, ...] = ()
    sweep: bool = False
    evidence_specs: tuple[EvidenceSpec, ...] = ()
    # Additional paired baselines besides full
    contrast_with: tuple[str, ...] = ()


def ablation(
    description: str,
    *,
    family: str,
    evidence_keys: Sequence[str] = (),
    evidence_specs: Sequence[EvidenceSpec] = (),
    sweep: bool = False,
    contrast_with: Sequence[str] = (),
    **overrides: Scalar,
) -> Ablation:
    inferred_specs = (
        () if sweep else evidence_specs_for_config_keys(tuple(overrides))
    )
    merged_specs = merge_evidence_specs(
        inferred_specs,
        tuple(disabled(key) for key in evidence_keys),
        tuple(evidence_specs),
    )
    return Ablation(
        description=description,
        overrides=tuple(overrides.items()),
        family=family,
        evidence_keys=tuple(
            spec.evidence_key
            for spec in merged_specs
            if spec.expectation == "disabled"
        ),
        evidence_specs=merged_specs,
        sweep=bool(sweep),
        contrast_with=tuple(contrast_with),
    )


# Overrides shared by the simplified text stopping controls
TEXT_SIMPLE_STOP_OVERRIDES: dict[str, Scalar] = {
    "text_stop_verification_object": "top_detection",
    "text_stop_reobservation": "passive",
    "text_hypothesis_triage": False,
    "text_evidence_weighted_target_ranking": False,
    "text_metric_visual_closure": False,
    "text_validated_stop_approach": False,
}
TEXT_STOP_CONTROL_RULES = {
    "full": "sequential",
    "stop_rule_single_verdict": "single_verdict",
    "stop_rule_count": "count",
    "stop_simple_single_verdict": "single_verdict",
    "stop_simple_count": "count",
    "stop_simple_sequential": "sequential",
}
TEXT_STOP_CONTROLS = tuple(TEXT_STOP_CONTROL_RULES)
TEXT_SIMPLE_STOP_PROFILES = frozenset(
    name for name in TEXT_STOP_CONTROLS if name.startswith("stop_simple_")
)
TEXT_SIMPLE_STOP_EVIDENCE = (
    capability("stop_validation"),
    capability("stop_ledger"),
    activated("text_stop_object_top_detection"),
    disabled("text_face_node_turn"),
    activated("text_passive_release"),
)


TEXT_ABLATIONS: dict[str, Ablation] = {
    "full": ablation(
        "Full text-goal navigation configuration (paired baseline).",
        family="baseline",
    ),
    "no_llm_goal_parser": ablation(
        "Skip LLM goal parsing and use the text goal's first sentence as the "
        "sole detector and graph query, with no parsed context cues.",
        family="semantic_grounding",
        evidence_specs=(activated("first_sentence_query"),),
        text_goal_parser_mode="first_sentence",
    ),
    "no_query_conditional_retrieval": ablation(
        "Replace query-conditioned graph class-posterior retrieval with "
        "single argmax-label matching; keep parsed queries, detection-backed "
        "instances, detection probes, verification and candidate association.",
        family="semantic_grounding",
        evidence_specs=(
            activated("text_argmax_label_retrieval"),
            retained("text_detection_instances"),
            retained("stop_validation"),
            retained("text_navigation_map_association"),
        ),
        text_query_conditional_labels=False,
    ),
    "explore_utility_difference": ablation(
        "Score frontiers by value minus a fixed reference exchange rate "
        "(lambda_ref) times the action cost, instead of by value per action; "
        "lambda_ref is sgf_explore_exchange_rate, the median best rate of the "
        "full objective.",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        sgf_explore_utility_form="difference",
    ),
    "explore_utility_difference_half_lambda": ablation(
        "Difference objective with half the reference exchange rate "
        "(0.5 lambda_ref).",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        sgf_explore_utility_form="difference",
        sgf_explore_exchange_rate_scale=0.5,
    ),
    "explore_utility_difference_double_lambda": ablation(
        "Difference objective with twice the reference exchange rate "
        "(2 lambda_ref).",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        sgf_explore_utility_form="difference",
        sgf_explore_exchange_rate_scale=2.0,
    ),
    "explore_utility_gain_only": ablation(
        "Choose the frontier of largest value regardless of its action cost "
        "(the zero-exchange-rate limit of the difference objective).",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        sgf_explore_utility_form="difference",
        sgf_explore_exchange_rate_scale=0.0,
    ),
    "explore_nearest_frontier": ablation(
        "Choose the frontier of smallest action cost regardless of its value: "
        "the classic nearest-frontier explorer and the infinite-exchange-rate "
        "limit of the difference objective.",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_nearest_selection"),),
        sgf_explore_utility_form="nearest",
    ),
    "explore_cost_translation_only": ablation(
        "Charge translation alone in the frontier action cost, so turning to "
        "face a frontier is free.",
        family="exploration_cost",
        sgf_explore_rotation_cost_weight=0.0,
    ),
    "explore_utility_difference_translation_only": ablation(
        "Difference objective at the reference exchange rate (lambda_ref) "
        "with a translation-only action cost, crossing the objective form "
        "with the rotation cost.",
        family="exploration_cost",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        contrast_with=(
            "explore_utility_difference",
            "explore_cost_translation_only",
        ),
        sgf_explore_utility_form="difference",
        sgf_explore_rotation_cost_weight=0.0,
    ),
    "no_panoramic_observation": ablation(
        "Remove the free-rotation panoramic sweep; goal evidence comes only "
        "from views along the direction of travel.",
        family="sgf_navigation",
        evidence_keys=("panoramic_observation",),
        panoramic_observation=False,
    ),
    "no_stop_validation": ablation(
        "Remove the verifier-dependent bundle: VLM stop validation and its "
        "dependent probes, ledger, triage, evidence ranking, re-observation, "
        "terminal refinement and endgame return.",
        family="verifier_dependent_bundle",
        evidence_keys=("stop_validation",),
        stop_validation=False,
    ),
    "no_metric_visual_closure": ablation(
        "Remove metric-visual one-primitive delayed STOP commitments.",
        family="terminal_refinement",
        evidence_keys=("metric_visual_closure",),
        text_metric_visual_closure=False,
    ),
    "no_detection_probe": ablation(
        "Remove proactive detection-probe pursuit.",
        family="stop_validation",
        evidence_keys=("detection_probe",),
        text_detection_probe=False,
    ),
    "no_detection_instances": ablation(
        "Use only geometric scene-graph nodes as persistent text target "
        "hypotheses; remove detection-backed spatial instances while keeping "
        "the detector, proactive detection probes, verifier, stop ledger and "
        "shared-map candidate association.",
        family="instance_hypotheses",
        evidence_specs=(
            retained("stop_validation"),
            retained("stop_ledger"),
            retained("text_navigation_map_association"),
        ),
        text_detection_instances=False,
    ),
    "stop_object_top_detection": ablation(
        "Show the verifier the frame's most confident target detection, "
        "selected independently of the pursued node and its footprint, "
        "instead of the node's projected footprint; the verdict counts for "
        "the node when the box lies within its identity radius.",
        family="stop_verification_object",
        evidence_specs=(activated("text_stop_object_top_detection"),),
        text_stop_verification_object="top_detection",
    ),
    "stop_object_whole_frame": ablation(
        "Ask the verifier whether the target is anywhere in the unmarked "
        "frame instead of about the pursued node's footprint; the footprint "
        "stays the geometric handle for approach and standpoint projection.",
        family="stop_verification_object",
        evidence_specs=(activated("text_stop_object_whole_frame"),),
        text_stop_verification_object="whole_frame",
    ),
    "no_hypothesis_triage": ablation(
        "Refute a candidate only after travelling to it, instead of at the "
        "range where it first becomes observable.",
        family="stop_validation",
        evidence_keys=("text_hypothesis_triage",),
        text_hypothesis_triage=False,
    ),
    "no_validated_stop_approach": ablation(
        "Remove post-validation close-view approach.",
        family="terminal_refinement",
        text_validated_stop_approach=False,
    ),
    "stop_rule_single_verdict": ablation(
        "Decide each hypothesis from the current verdict alone: an "
        "evidence-bearing approval inside the stop radius stops and an "
        "evidence-bearing refutation excludes, instead of summing "
        "pose-independent verdicts against the sequential bound.",
        family="stop_evidence_accumulation",
        evidence_specs=(
            capability("stop_validation"),
            capability("stop_ledger"),
            activated("text_evidence_rule_override"),
        ),
        text_stop_evidence_rule="single_verdict",
    ),
    "stop_rule_count": ablation(
        "Commit by vote count: at least two pose-independent evidence-bearing "
        "approvals that outnumber evidence-bearing dissents by at least two, "
        "instead of summed log-odds against the sequential bound.",
        family="stop_evidence_accumulation",
        evidence_specs=(
            capability("stop_validation"),
            capability("stop_ledger"),
            activated("text_evidence_rule_override"),
        ),
        contrast_with=("stop_rule_single_verdict",),
        text_stop_evidence_rule="count",
    ),
    "stop_simple_single_verdict": ablation(
        "Same-model simplified stopping scaffold with the single-verdict "
        "rule: verify the top detection from ordinary navigation, without "
        "early triage, evidence-weighted ranking, delayed closure or "
        "post-validation approach. Differs from full in both the scaffold "
        "and the evidence rule.",
        family="same_model_stop_control",
        evidence_specs=(
            *TEXT_SIMPLE_STOP_EVIDENCE,
            activated("text_evidence_rule_override"),
        ),
        contrast_with=("stop_rule_single_verdict",),
        text_stop_evidence_rule="single_verdict",
        **TEXT_SIMPLE_STOP_OVERRIDES,
    ),
    "stop_simple_count": ablation(
        "Use approval votes in the same-model simplified stopping scaffold. "
        "Against stop_simple_single_verdict only the evidence rule changes; "
        "against stop_rule_count only the stopping scaffold changes.",
        family="same_model_stop_control",
        evidence_specs=(
            *TEXT_SIMPLE_STOP_EVIDENCE,
            activated("text_evidence_rule_override"),
        ),
        contrast_with=("stop_simple_single_verdict", "stop_rule_count"),
        text_stop_evidence_rule="count",
        **TEXT_SIMPLE_STOP_OVERRIDES,
    ),
    "stop_simple_sequential": ablation(
        "Use sequential log-odds in the same-model simplified stopping "
        "scaffold. Against the other simplified controls only the evidence "
        "rule changes; against full only the stopping scaffold changes.",
        family="same_model_stop_control",
        evidence_specs=TEXT_SIMPLE_STOP_EVIDENCE,
        contrast_with=("stop_simple_single_verdict", "stop_simple_count"),
        # Inherits the sequential rule from full
        **TEXT_SIMPLE_STOP_OVERRIDES,
    ),
    "stop_reobserve_passive": ablation(
        "Take verdicts only from views ordinary navigation produces: no trip "
        "to a new vantage, no press toward a withheld approval, and no turn "
        "back toward the node after a dissent; a stalled hypothesis is "
        "released and may be readmitted when re-encountered.",
        family="stop_reobservation",
        evidence_specs=(
            disabled("text_face_node_turn"),
            activated("text_passive_release"),
        ),
        text_stop_reobservation="passive",
    ),
    "stop_reobserve_same_pose": ablation(
        "Re-query the verifier from wherever the agent stands at the re-query "
        "interval instead of moving to a new vantage, counting every verdict "
        "as independent evidence.",
        family="stop_reobservation",
        evidence_specs=(
            disabled("text_same_vantage_skip"),
            activated("text_same_pose_requery"),
        ),
        contrast_with=("stop_reobserve_passive",),
        text_stop_reobservation="same_pose",
    ),
    "no_target_fatigue": ablation(
        "Remove stalled-target retirement driven by target fatigue.",
        family="target_recovery",
        evidence_keys=("target_fatigue",),
        text_target_fatigue=False,
    ),
    "no_endgame_return": ablation(
        "Remove best-evidence endgame return.",
        family="endgame",
        evidence_keys=("endgame_return",),
        endgame_return=False,
    ),
    "no_evidence_weighted_ranking": ablation(
        "Pursue the admissible candidate of least travel instead of ranking "
        "by travel divided by the logistic ledger-evidence score; verdicts "
        "still accumulate in the ledger but do not reorder pursuit.",
        family="target_ranking",
        text_evidence_weighted_target_ranking=False,
    ),
    "nav_map_depth_only": ablation(
        "Build the navigation evidence map from depth rays alone; the scene "
        "graph's stable surfels do not vote occupancy, so traversability, "
        "frontiers and planning do not use the graph's geometry.",
        family="shared_representation",
        navigation_graph_occupancy=False,
    ),
    "nav_map_no_floor_support": ablation(
        "Remove the floor-support veto on ray-cleared navigation cells; "
        "retain depth-ray occupancy, stable graph-surface occupancy, obstacle "
        "inflation and shared-map text candidate association.",
        family="navigation_traversability",
        evidence_specs=(
            retained("navigation_graph_occupancy"),
            retained("text_navigation_map_association"),
        ),
        navigation_floor_support=False,
    ),
    "target_association_euclidean": ablation(
        "Associate text candidates as a separate semantic map would: admit "
        "standpoints the shared navigation map cannot reach, and select and "
        "rank candidates by planar distance instead of geodesic travel, "
        "leaving reachability to the planner. Terminal rules keep the "
        "geodesic distance.",
        family="shared_representation",
        evidence_specs=(activated("text_euclidean_association"),),
        text_target_association="euclidean",
    ),
    "target_association_reachable_euclidean": ablation(
        "Retain geodesic reachability filtering and rank reachable candidates "
        "by planar distance. Isolate travel ranking from reachability gating; "
        "terminal distances and evidence weighting remain unchanged.",
        family="shared_representation",
        evidence_specs=(activated("text_reachable_euclidean_association"),),
        contrast_with=("target_association_euclidean",),
        text_target_association="reachable_euclidean",
    ),
    "depth_only_reachable_euclidean": ablation(
        "Rank reachable candidates by planar distance on a depth-only map. "
        "Cross the ranking-only control with surfel occupancy removal.",
        family="shared_representation",
        evidence_specs=(activated("text_reachable_euclidean_association"),),
        contrast_with=(
            "nav_map_depth_only", "target_association_reachable_euclidean",
            "decoupled_representation",
        ),
        navigation_graph_occupancy=False,
        text_target_association="reachable_euclidean",
    ),
    "decoupled_representation": ablation(
        "Decouple the scene graph from the navigation map in both directions: "
        "a depth-only navigation map and Euclidean candidate association "
        "(the two shared-representation factors removed together).",
        family="shared_representation",
        evidence_specs=(activated("text_euclidean_association"),),
        contrast_with=(
            "nav_map_depth_only",
            "target_association_euclidean",
        ),
        navigation_graph_occupancy=False,
        text_target_association="euclidean",
    ),
}


INSTANCE_IMAGE_ABLATIONS: dict[str, Ablation] = {
    "full": ablation(
        "Full instance-image navigation configuration (paired baseline).",
        family="baseline",
    ),
    "explore_classic_frontier": ablation(
        "Replace the SGF exploration stack (value-per-action frontier "
        "scoring and goal-conditioned visual memory) with the classic "
        "nearest-frontier explorer on the same map.",
        family="sgf_exploration",
        evidence_specs=(activated("sgf_explore_nearest_selection"),),
        sgf_explore_utility_form="nearest",
        sgf_visual_memory=False,
    ),
    "explore_utility_difference": ablation(
        "Score frontiers by value minus a fixed reference exchange rate "
        "(lambda_ref) times the action cost, instead of by value per action; "
        "lambda_ref is sgf_explore_exchange_rate, the median best rate of the "
        "full objective.",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        sgf_explore_utility_form="difference",
    ),
    "explore_utility_difference_half_lambda": ablation(
        "Difference objective with half the reference exchange rate "
        "(0.5 lambda_ref).",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        sgf_explore_utility_form="difference",
        sgf_explore_exchange_rate_scale=0.5,
    ),
    "explore_utility_difference_double_lambda": ablation(
        "Difference objective with twice the reference exchange rate "
        "(2 lambda_ref).",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        sgf_explore_utility_form="difference",
        sgf_explore_exchange_rate_scale=2.0,
    ),
    "explore_utility_gain_only": ablation(
        "Choose the frontier of largest value regardless of its action cost "
        "(the zero-exchange-rate limit of the difference objective).",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        sgf_explore_utility_form="difference",
        sgf_explore_exchange_rate_scale=0.0,
    ),
    "explore_nearest_frontier": ablation(
        "Choose the frontier of smallest action cost regardless of its value: "
        "the classic nearest-frontier explorer and the infinite-exchange-rate "
        "limit of the difference objective.",
        family="exploration_objective",
        evidence_specs=(activated("sgf_explore_nearest_selection"),),
        sgf_explore_utility_form="nearest",
    ),
    "explore_cost_translation_only": ablation(
        "Charge translation alone in the frontier action cost, so turning to "
        "face a frontier is free.",
        family="exploration_cost",
        sgf_explore_rotation_cost_weight=0.0,
    ),
    "explore_utility_difference_translation_only": ablation(
        "Difference objective at the reference exchange rate (lambda_ref) "
        "with a translation-only action cost, crossing the objective form "
        "with the rotation cost.",
        family="exploration_cost",
        evidence_specs=(activated("sgf_explore_difference_selection"),),
        contrast_with=(
            "explore_utility_difference",
            "explore_cost_translation_only",
        ),
        sgf_explore_utility_form="difference",
        sgf_explore_rotation_cost_weight=0.0,
    ),
    "no_visual_memory": ablation(
        "Remove goal-conditioned SGF visual-memory frontier scoring.",
        family="sgf_exploration",
        evidence_keys=("visual_memory",),
        sgf_visual_memory=False,
    ),
    "no_panoramic_observation": ablation(
        "Remove the free-rotation panoramic sweep; goal evidence comes only "
        "from views along the direction of travel.",
        family="sgf_exploration",
        evidence_keys=("panoramic_observation",),
        panoramic_observation=False,
    ),
    "no_free_resource_observation": ablation(
        "Use the depth sensor's mapping range as the sweep interval and do "
        "not query the instance-level validator on completed sweeps.",
        family="free_resource_observation",
        panoramic_observation_trigger_scale=1.0,
        image_vlm_panoramic_query=False,
    ),
    "no_detector_scaled_sweeps": ablation(
        "Set the sweep interval to the depth sensor's mapping range instead "
        "of the goal detector's effective range.",
        family="free_resource_observation",
        panoramic_observation_trigger_scale=1.0,
    ),
    "no_panoramic_instance_query": ablation(
        "Do not query the instance-level validator on completed sweeps, so "
        "no bearing prior is produced.",
        family="free_resource_observation",
        image_vlm_panoramic_query=False,
    ),
    "no_foveated_hypothesis": ablation(
        "Skip the full-resolution VLM query on the sweep view selected by "
        "the panel query, so no metric instance hypothesis is created from a "
        "VLM sighting; the dependent arrival sweep becomes inert.",
        family="recognition_limited_search",
        evidence_keys=("image_vlm_foveated_hypothesis",),
        image_vlm_foveated_hypothesis=False,
    ),
    "no_reachable_pursuit": ablation(
        "Keep the straight servo pressing toward the reference viewpoint when "
        "a forward action fails to move, instead of routing to the closest "
        "free standpoint and giving up an unreachable approach.",
        family="terminal_policy",
        evidence_keys=("image_reachable_pursuit",),
        image_reachable_pursuit=False,
    ),
    "no_goal_view_terminal": ablation(
        "Replace the persistent goal-view belief, its R - s commitment band "
        "and object-aligned verification with the latest single-frame "
        "goal-camera estimate, pursued to the depth floor and dropped after "
        "one rotation's worth of steps without a re-match; the verifier is "
        "asked from the arrival heading.",
        family="terminal_policy",
        evidence_keys=("image_goal_view_terminal",),
        image_goal_view_terminal=False,
    ),
    "image_localization_depth_cluster": ablation(
        "Localize the goal by the matched depth cluster alone (no "
        "goal-camera PnP, hence no goal-view belief) and pursue the "
        "measured object point with the same servo, planner and verifier, "
        "facing it and committing within R - s.",
        family="image_goal_localization",
        evidence_specs=(activated("image_depth_cluster_target"),),
        contrast_with=("image_target_object",),
        image_target_localization="depth_cluster",
    ),
    "image_target_object": ablation(
        "Pursue the object anchor on the goal camera's optical axis instead "
        "of the goal camera's viewpoint; belief, band, alignment and verifier "
        "are unchanged.",
        family="image_navigation_target",
        evidence_specs=(activated("image_object_target_steps"),),
        image_navigation_target="object",
    ),
    "image_controller_fmm": ablation(
        "Pursue the same target with the map planner from the first step "
        "instead of the straight bearing servo; the planner must deliver the "
        "agent into the commitment band itself.",
        family="image_controller",
        evidence_specs=(activated("image_pursuit_fmm_actions"),),
        image_pursuit_controller="fmm",
    ),
    "image_terminal_single_corroboration": ablation(
        "Grant the R - s commitment band after one PnP solution instead of "
        "two agreeing ones (evidence accumulation of the goal-view terminal).",
        family="image_terminal",
        evidence_specs=(
            activated("image_goal_view_single_corroboration_commit"),
        ),
        image_goal_view_min_corroboration=1,
    ),
    "image_terminal_depth_floor": ablation(
        "Commit only at the depth floor instead of within R - s of a "
        "corroborated goal-camera estimate.",
        family="image_terminal",
        image_goal_view_commit_band=False,
    ),
    "image_terminal_unaligned": ablation(
        "Ask the verifier from the heading the approach left instead of first "
        "turning to face the object: the verification view and the active "
        "re-observation that produces it are removed together.",
        family="image_terminal",
        image_goal_view_alignment="none",
    ),
    "no_vlm_stop_validation": ablation(
        "Remove instance-image VLM stop validation.",
        family="vlm",
        evidence_keys=("image_vlm_stop_validation",),
        image_vlm_stop_validation=False,
    ),
    "no_vlm": ablation(
        "Use geometric instance-image localization, pursuit and STOP without "
        "VLM calls: disable stop validation and panoramic instance queries; "
        "the dependent foveated hypotheses and arrival sweeps become inert. "
        "Retain XFeat/PnP, goal-view belief and panoramic observation.",
        family="vlm",
        evidence_specs=(
            retained("visual_pose_servo"),
            retained("image_goal_view_terminal"),
        ),
        image_vlm_stop_validation=False,
        image_vlm_panoramic_query=False,
    ),
    "no_endgame_return": ablation(
        "Remove best-evidence endgame return.",
        family="endgame",
        evidence_keys=("endgame_return",),
        endgame_return=False,
    ),
    "nav_map_depth_only": ablation(
        "Build the navigation evidence map from depth rays alone; the scene "
        "graph's stable surfels do not vote occupancy, so traversability, "
        "frontiers, reachable-viewpoint pursuit and planning do not use the "
        "graph's geometry.",
        family="shared_representation",
        navigation_graph_occupancy=False,
    ),
    "nav_map_no_floor_support": ablation(
        "Remove the floor-support veto on ray-cleared navigation cells; "
        "retain depth-ray occupancy, stable graph-surface occupancy, obstacle "
        "inflation and the image-goal localization and pursuit machinery.",
        family="navigation_traversability",
        evidence_specs=(retained("navigation_graph_occupancy"),),
        navigation_floor_support=False,
    ),
}


ABLATIONS_BY_TASK = {
    "text": TEXT_ABLATIONS,
    "instance-image": INSTANCE_IMAGE_ABLATIONS,
}

REPRESENTATION_COUPLING_ABLATIONS = {
    "text": (
        "full", "nav_map_depth_only", "target_association_euclidean",
        "decoupled_representation", "target_association_reachable_euclidean",
        "depth_only_reachable_euclidean",
    ),
    "instance-image": ("full", "nav_map_depth_only"),
}

for _mode, _registry in ABLATIONS_BY_TASK.items():
    for _name, _profile in _registry.items():
        for _reference in _profile.contrast_with:
            if _reference not in _registry or _reference in {"full", _name}:
                raise ValueError(
                    f"{_mode}/{_name} contrasts with unknown or invalid "
                    f"profile {_reference!r}"
                )


CORE_ABLATIONS: dict[str, tuple[str, ...]] = {
    "text": (
        "full",
        "no_llm_goal_parser",
        "no_query_conditional_retrieval",
        "no_detection_instances",
        "explore_nearest_frontier",
        "explore_utility_difference",
        "explore_cost_translation_only",
        "no_panoramic_observation",
        "no_stop_validation",
        "stop_object_top_detection",
        "stop_rule_single_verdict",
        "stop_reobserve_passive",
        "stop_simple_single_verdict",
        "no_metric_visual_closure",
        "no_validated_stop_approach",
        "no_endgame_return",
        "no_evidence_weighted_ranking",
        "nav_map_depth_only",
        "nav_map_no_floor_support",
        "target_association_euclidean",
        "decoupled_representation",
    ),
    "instance-image": (
        "full",
        "explore_classic_frontier",
        "explore_utility_difference",
        "explore_cost_translation_only",
        "no_visual_memory",
        "no_panoramic_observation",
        "no_foveated_hypothesis",
        "no_free_resource_observation",
        "image_localization_depth_cluster",
        "image_target_object",
        "image_controller_fmm",
        "no_goal_view_terminal",
        "no_vlm_stop_validation",
        "no_vlm",
        "no_endgame_return",
        "nav_map_depth_only",
        "nav_map_no_floor_support",
    ),
}

JOINT_ABLATIONS: dict[str, frozenset[str]] = {
    "text": frozenset(
        {
            "no_llm_goal_parser",
            "decoupled_representation",
            "depth_only_reachable_euclidean",
            *TEXT_SIMPLE_STOP_PROFILES,
        }
    ),
    "instance-image": frozenset(
        {
            "no_free_resource_observation",
            "explore_classic_frontier",
            "no_vlm",
        }
    ),
}

COMPONENT_ABLATIONS: dict[str, tuple[str, ...]] = {
    mode: tuple(
        name
        for name in registry
        if name == "full" or name not in JOINT_ABLATIONS[mode]
    )
    for mode, registry in ABLATIONS_BY_TASK.items()
}


def ordered_union(*groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(name for group in groups for name in group))


ALL_ABLATIONS: dict[str, tuple[str, ...]] = {
    mode: ordered_union(
        CORE_ABLATIONS[mode],
        COMPONENT_ABLATIONS[mode],
        REPRESENTATION_COUPLING_ABLATIONS[mode],
        TEXT_STOP_CONTROLS if mode == "text" else (),
    )
    for mode in TASK_MODES
}

@dataclass(frozen=True)
class ActivationRule:
    """Conditions under which one configured navigation key can take effect."""

    all_true: tuple[str, ...] = ()
    all_false: tuple[str, ...] = ()
    any_true: tuple[str, ...] = ()
    any_all_true: tuple[tuple[str, ...], ...] = ()
    all_positive: tuple[str, ...] = ()
    all_nonnegative: tuple[str, ...] = ()
    values_in: tuple[tuple[str, tuple[Scalar, ...]], ...] = ()


def requires(*parents: str) -> ActivationRule:
    return ActivationRule(all_true=tuple(parents))


# Children have no effect once a required parent is disabled
FEATURE_DEPENDENCIES: dict[str, dict[str, tuple[str, ...]]] = {
    "text": {
        "text_detection_probe": ("stop_validation", "sgf_stop"),
        "text_stop_verification_object": ("stop_validation", "sgf_stop"),
        # The runtime triages only into the evidence ledger
        "text_hypothesis_triage": (
            "stop_validation",
            "sgf_stop",
            "text_stop_ledger",
        ),
        "text_stop_evidence_rule": (
            "stop_validation",
            "sgf_stop",
            "text_stop_ledger",
        ),
        "text_stop_reobservation": (
            "stop_validation",
            "sgf_stop",
            "text_stop_ledger",
        ),
        "text_metric_visual_closure": (
            "stop_validation",
            "sgf_stop",
        ),
        "text_validated_stop_approach": (
            "stop_validation",
            "sgf_stop",
        ),
        "text_stop_ledger": (
            "stop_validation",
            "sgf_stop",
        ),
        "text_stop_sequential_error_rate": (
            "stop_validation",
            "sgf_stop",
            "text_stop_ledger",
        ),
        # Reads the ledger log-odds, which only verifier verdicts update
        "text_evidence_weighted_target_ranking": (
            "stop_validation",
            "sgf_stop",
            "text_stop_ledger",
        ),
        "endgame_return": ("stop_validation", "sgf_stop"),
    },
    "instance-image": {
        "sgf_visual_memory": ("sgf_explore",),
        "sgf_explore_evidence_verification": ("sgf_explore",),
        "image_reachable_pursuit": ("visual_pose_servo",),
        "image_goal_view_terminal": ("visual_pose_servo",),
        "image_target_localization": ("visual_pose_servo",),
        "image_navigation_target": (
            "visual_pose_servo",
            "image_goal_view_terminal",
        ),
        "image_pursuit_controller": ("visual_pose_servo", "sgf_explore"),
        # Instance-image builds its SGF navigation map only for the explorer
        "navigation_graph_occupancy": ("sgf_explore",),
        "navigation_floor_support": ("sgf_explore",),
    },
}

# Controls owned by the other task are inert
TASK_INAPPLICABLE_PREFIXES: dict[str, tuple[str, ...]] = {
    "text": (
        "final_image_match",
        "image_",
        "instance_image_",
        "no_image_match_visuals",
        "sgf_visual_memory",
        "visual_",
    ),
    "instance-image": (
        "llm_",
        "sgf_stop",
        "stop_validation",
        "text_",
    ),
}

# Output-only settings that do not affect the action policy
NONBEHAVIORAL_OUTPUT_CONFIG_KEYS = frozenset(
    {
        "dump_sgf_inputs",
        "final_image_match",
        "final_image_match_path",
        "image_every_n_steps",
        "image_format",
        "image_match_every_n_steps",
        "image_match_video",
        "image_match_video_fps",
        "image_match_video_path",
        "image_match_visual_every_n_steps",
        "image_match_visual_max_lines",
        "no_image_match_visuals",
        "overwrite_image_dir",
        "overwrite_sgf_input_dump_dir",
        "sgf_input_dump_dir",
        "sgf_input_dump_every_n_steps",
        "video_fps",
    }
)

VISUAL_MATCH_SWITCHES = ("visual_pose_servo",)

# Settings under these prefixes are inert when the feature is disabled
FEATURE_PREFIX_DEPENDENCIES: dict[
    str,
    tuple[tuple[str, ActivationRule, frozenset[str]], ...],
] = {
    "text": (
        ("stop_validation_", requires("stop_validation"), frozenset()),
        (
            "text_detection_instance_",
            requires("text_detection_instances"),
            frozenset(),
        ),
        (
            "text_detection_probe_",
            requires("text_detection_probe"),
            frozenset(),
        ),
        (
            "text_metric_visual_closure_",
            requires("text_metric_visual_closure"),
            frozenset(),
        ),
        (
            "text_validated_stop_approach_",
            requires("text_validated_stop_approach"),
            frozenset(),
        ),
        (
            "text_stop_ledger_",
            requires("text_stop_ledger"),
            frozenset(
                {
                    # Also used by metric closure
                    "text_stop_ledger_close_up_bbox_area_fraction",
                }
            ),
        ),
        (
            "text_target_fatigue_",
            requires("text_target_fatigue"),
            frozenset(),
        ),
        (
            "endgame_return_",
            requires("endgame_return"),
            frozenset(),
        ),
        (
            "sgf_explore_",
            requires("sgf_explore"),
            frozenset(),
        ),
    ),
    "instance-image": (
        (
            "sgf_visual_memory_",
            requires("sgf_visual_memory"),
            frozenset(),
        ),
        ("visual_pose_", requires("visual_pose_servo"), frozenset()),
        (
            "image_goal_view_",
            requires("image_goal_view_terminal"),
            frozenset(),
        ),
        (
            "image_vlm_stop_validation_",
            requires("image_vlm_stop_validation"),
            frozenset(),
        ),
        ("endgame_return_", requires("endgame_return"), frozenset()),
        (
            "sgf_explore_",
            requires("sgf_explore"),
            frozenset(),
        ),
    ),
}


# Conditions that cannot be represented by a simple positive parent prefix
CONDITIONAL_ACTIVATION_RULES: dict[str, dict[str, tuple[ActivationRule, ...]]] = {
    mode: {} for mode in TASK_MODES
}

_DIFFERENCE_FORM = ActivationRule(
    values_in=(("sgf_explore_utility_form", ("difference",)),),
)
_VALUE_SCORED_FORMS = ActivationRule(
    values_in=(("sgf_explore_utility_form", ("rate", "difference")),),
)
for _mode in TASK_MODES:
    CONDITIONAL_ACTIVATION_RULES[_mode].update(
        {
            "sgf_explore_exchange_rate": (_DIFFERENCE_FORM,),
            "sgf_explore_exchange_rate_scale": (_DIFFERENCE_FORM,),
            "sgf_explore_info_gain_weight": (_VALUE_SCORED_FORMS,),
            "sgf_explore_revisit_weight": (_VALUE_SCORED_FORMS,),
            "sgf_explore_information_range_scale": (_VALUE_SCORED_FORMS,),
        }
    )

CONDITIONAL_ACTIVATION_RULES["text"].update(
    {
        "text_context_guided_exploration": (
            ActivationRule(values_in=(("text_goal_parser_mode", ("llm",)),)),
        ),
        "stop_validation": (
            requires("sgf_stop"),
        ),
        # SPRT error rate applies only to the sequential rule
        "text_stop_sequential_error_rate": (
            ActivationRule(
                values_in=(("text_stop_evidence_rule", ("sequential",)),),
            ),
        ),
        "text_stop_ledger_close_up_bbox_area_fraction": (
            ActivationRule(
                any_true=(
                    "text_stop_ledger",
                    "text_metric_visual_closure",
                ),
            ),
        ),
    }
)
_PNP_LOCALIZATION = ActivationRule(
    values_in=(("image_target_localization", ("pnp_goal_view",)),),
)
CONDITIONAL_ACTIVATION_RULES["instance-image"].update(
    {
        # Depth-cluster localization has no PnP gate or goal-view fusion
        "image_goal_view_terminal": (_PNP_LOCALIZATION,),
        **{
            key: (_PNP_LOCALIZATION,)
            for key in (
                "visual_pose_min_matches",
                "visual_pose_min_inliers",
                "visual_pose_min_inlier_fraction",
                "visual_pose_min_inlier_coverage",
                "visual_pose_reproj_threshold_px",
                "visual_pose_min_confidence",
                "visual_pose_min_agent_distance",
                "visual_pose_max_agent_distance",
                "visual_pose_max_height_delta",
            )
        },
        # The straight-servo fallback planner exists only for the servo
        "image_reachable_pursuit": (
            ActivationRule(
                values_in=(("image_pursuit_controller", ("servo",)),),
            ),
        ),
        "sgf_explore_evidence_verification": (
            ActivationRule(any_true=VISUAL_MATCH_SWITCHES),
        ),
        "endgame_return": (
            ActivationRule(any_true=VISUAL_MATCH_SWITCHES),
        ),
        "image_vlm_stop_validation": (
            ActivationRule(
                # Only pose-servo STOP/arrival proposals reach the VLM gate
                any_true=VISUAL_MATCH_SWITCHES,
            ),
        ),
        "panoramic_observation_trigger_scale": (
            ActivationRule(any_true=("panoramic_observation",)),
        ),
        "panoramic_observation_searched_extent_interval": (
            ActivationRule(any_true=("panoramic_observation",)),
        ),
        "image_vlm_panoramic_query": (
            ActivationRule(any_true=("panoramic_observation",)),
        ),
        # Foveated hypotheses are used only by the visual-memory explorer
        "image_vlm_foveated_hypothesis": (
            ActivationRule(
                any_all_true=(
                    (
                        "panoramic_observation",
                        "image_vlm_panoramic_query",
                        "sgf_visual_memory",
                    ),
                ),
            ),
        ),
        "image_vlm_hypothesis_arrival_sweep": (
            ActivationRule(
                any_all_true=(
                    (
                        "panoramic_observation",
                        "image_vlm_panoramic_query",
                        "sgf_visual_memory",
                        "sgf_explore_evidence_verification",
                        "image_vlm_foveated_hypothesis",
                    ),
                ),
            ),
        ),
        "image_vlm_panoramic_query_panels": (
            ActivationRule(
                any_all_true=(
                    ("panoramic_observation", "image_vlm_panoramic_query"),
                ),
            ),
        ),
    }
)


SUMMARY_COLUMNS = (
    "task_mode", "ablation", "SR", "ΔSR [95% CI]", "SPL", "ΔSPL [95% CI]",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_value(*parts: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *parts],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def navigation_source_sha256(runner_path: Path, batch_runner_path: Path) -> str:
    package_root = REPO_ROOT / "python" / "sgf_habitat"
    files = {runner_path.resolve(), batch_runner_path.resolve()}
    if package_root.is_dir():
        files.update(path.resolve() for path in package_root.rglob("*.py"))
        files.update(path.resolve() for path in package_root.glob("_sgf*.so"))
    digest = hashlib.sha256()
    for path in sorted(files, key=str):
        try:
            label = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def task_modes_for(value: str) -> tuple[str, ...]:
    return TASK_MODES if value == "both" else (value,)


def select_ablations(
    task_modes: Sequence[str], spec: str, *, auto_contrasts: bool = True,
) -> dict[str, list[str]]:
    requested = parse_csv(spec)
    if not requested:
        raise ValueError(
            "--ablations must contain a profile name or a matrix selector"
        )
    if ABLATION_SELECTORS.intersection(requested) and len(requested) != 1:
        raise ValueError(
            "--ablations matrix selectors cannot be combined with named profiles"
        )
    if requested == ["text-stop-controls"]:
        if "text" not in task_modes:
            raise ValueError("text-stop-controls applies only to the text task")
        return {"text": list(TEXT_STOP_CONTROLS)}
    if requested == ["all"]:
        return {mode: list(ALL_ABLATIONS[mode]) for mode in task_modes}
    if requested == ["representation-coupling"]:
        return {mode: list(REPRESENTATION_COUPLING_ABLATIONS[mode]) for mode in task_modes}
    if requested == ["core"]:
        return {mode: list(CORE_ABLATIONS[mode]) for mode in task_modes}
    if requested == ["components"]:
        return {mode: list(COMPONENT_ABLATIONS[mode]) for mode in task_modes}

    known = set().union(*(set(ABLATIONS_BY_TASK[mode]) for mode in task_modes))
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"unknown ablation profile(s): {', '.join(unknown)}")

    selected: dict[str, list[str]] = {}
    for mode in task_modes:
        # Add the reference profiles needed by nested contrasts
        wanted = set(requested)
        pending = list(requested) if auto_contrasts else []
        while pending:
            name = pending.pop()
            profile = ABLATIONS_BY_TASK[mode].get(name)
            if profile is not None:
                for reference in profile.contrast_with:
                    if reference not in wanted:
                        wanted.add(reference)
                        pending.append(reference)
        names = [
            name
            for name in ABLATIONS_BY_TASK[mode]
            if name == "full" or name in wanted
        ]
        if requested == ["full"]:
            names = ["full"]
        elif not any(name in requested for name in ABLATIONS_BY_TASK[mode]):
            names = []
        if names:
            selected[mode] = names
    if not selected:
        raise ValueError("none of the requested profiles applies to the selected task mode(s)")
    return selected


def profile_tier(mode: str, name: str) -> str:
    if name == "full":
        return "baseline"
    if name in JOINT_ABLATIONS[mode]:
        return "core-joint" if name in CORE_ABLATIONS[mode] else "control-joint"
    if name in CORE_ABLATIONS[mode] and name in COMPONENT_ABLATIONS[mode]:
        return "core+component"
    if name in CORE_ABLATIONS[mode]:
        return "core"
    if ABLATIONS_BY_TASK[mode][name].sweep:
        return "sensitivity"
    return "component"


def ablation_catalog_payload(
    profile_analysis: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    modes = tuple(profile_analysis) if profile_analysis is not None else TASK_MODES
    return {
        "selectors": {
            mode: {
                "core": list(CORE_ABLATIONS[mode]),
                "components": list(COMPONENT_ABLATIONS[mode]),
                "all": list(ALL_ABLATIONS[mode]),
                "representation-coupling": list(REPRESENTATION_COUPLING_ABLATIONS[mode]),
                **(
                    {"text-stop-controls": list(TEXT_STOP_CONTROLS)}
                    if mode == "text" else {}
                ),
            }
            for mode in modes
        },
        "tasks": {
            mode: [
                {
                    "name": name,
                    "tier": profile_tier(mode, name),
                    "family": profile.family,
                    "description": profile.description,
                    "contrast_with": list(profile.contrast_with),
                    "overrides": {
                        key: value for key, value in profile.overrides
                    },
                    "direct_evidence": list(
                        (profile_analysis or {})
                        .get(mode, {})
                        .get(name, {})
                        .get("direct_evidence_keys", profile.evidence_keys)
                    ),
                    "dependent_evidence": list(
                        (profile_analysis or {})
                        .get(mode, {})
                        .get(name, {})
                        .get("dependent_evidence_keys", ())
                    ),
                    "retained_evidence": list(
                        (profile_analysis or {})
                        .get(mode, {})
                        .get(name, {})
                        .get(
                            "retained_evidence_keys",
                            tuple(
                                spec.evidence_key
                                for spec in profile.evidence_specs
                                if spec.expectation == "retained"
                            ),
                        )
                    ),
                    "activated_evidence": list(
                        (profile_analysis or {})
                        .get(mode, {})
                        .get(name, {})
                        .get("activated_evidence_keys", ())
                    ),
                    "capability_evidence": list(
                        (profile_analysis or {})
                        .get(mode, {})
                        .get(name, {})
                        .get("capability_evidence_keys", ())
                    ),
                    "evidence_contracts": list(
                        (profile_analysis or {})
                        .get(mode, {})
                        .get(name, {})
                        .get(
                            "evidence_contracts",
                            [
                                evidence_requirement_payload(
                                    EvidenceRequirement(
                                        spec.evidence_key,
                                        spec.expectation,
                                        (
                                            "direct"
                                            if spec.expectation == "disabled"
                                            else "context"
                                        ),
                                    )
                                )
                                for spec in profile.evidence_specs
                            ],
                        )
                    ),
                    "inert_by_dependency": list(
                        (profile_analysis or {})
                        .get(mode, {})
                        .get(name, {})
                        .get("inert_by_dependency", ())
                    ),
                }
                for name, profile in ABLATIONS_BY_TASK[mode].items()
            ]
            for mode in modes
        },
    }


def print_ablation_catalog(
    profile_analysis: dict[str, dict[str, dict[str, Any]]] | None = None,
    *,
    output_format: str = "text",
) -> None:
    catalog = ablation_catalog_payload(profile_analysis)
    if output_format == "json":
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
        return
    if output_format == "markdown":
        for mode, profiles in catalog["tasks"].items():
            selectors = catalog["selectors"][mode]
            print(f"## {mode}\n")
            print(
                f"Selectors: core={len(selectors['core'])}, "
                f"components={len(selectors['components'])}, "
                f"all={len(selectors['all'])}.\n"
            )
            print("| Profile | Tier | Family | Overrides | Direct evidence | Dependency-disabled evidence | Evidence contracts | Inert config keys |")
            print("|---|---|---|---|---|---|---|---|")
            for item in profiles:
                overrides = format_overrides(tuple(item["overrides"].items())) or "—"
                direct = ", ".join(item["direct_evidence"]) or "—"
                dependent = ", ".join(item["dependent_evidence"]) or "—"
                contracts = ", ".join(
                    f"{rule['expectation']}:{rule['evidence_key']}"
                    for rule in item["evidence_contracts"]
                ) or "—"
                inert = ", ".join(item["inert_by_dependency"]) or "—"
                print(
                    f"| `{item['name']}` | {item['tier']} | {item['family']} | "
                    f"`{overrides}` | {direct} | {dependent} | {contracts} | {inert} |"
                )
            print()
        return

    print(
        "Selectors: core = paper-facing high-level matrix; components = individual "
        "switches; text-stop-controls = same-model stopping scaffold x evidence "
        "rule; all = stable ordered union of these matrices.\n"
    )
    modes = tuple(catalog["tasks"])
    for mode in modes:
        selectors = catalog["selectors"][mode]
        print(
            f"{mode}: core={len(selectors['core'])} "
            f"components={len(selectors['components'])} all={len(selectors['all'])}"
        )
        for item in catalog["tasks"][mode]:
            overrides = format_overrides(tuple(item["overrides"].items())) or "<none>"
            direct = item["direct_evidence"]
            dependent = item["dependent_evidence"]
            contracts = item["evidence_contracts"]
            inert = item["inert_by_dependency"]
            status = (
                "baseline"
                if item["name"] == "full"
                else f"{len(item['overrides'])} direct override(s)"
            )
            if inert:
                status += f"; {len(inert)} dependent switch(es) inert"
            print(
                f"  {item['name']:39s} [{item['tier']}; {item['family']}] {status}"
            )
            print(f"    {item['description']}")
            print(f"    {overrides}")
            print(f"    direct_evidence={','.join(direct) or '<none>'}")
            print(f"    dependent_evidence={','.join(dependent) or '<none>'}")
            print(
                "    evidence_contracts="
                + (
                    ",".join(
                        f"{rule['expectation']}:{rule['evidence_key']}"
                        for rule in contracts
                    )
                    or "<none>"
                )
            )
            print(f"    inert_by_dependency={','.join(inert) or '<none>'}")


def load_effective_nav_configs(
    nav_config_path: Path,
    task_modes: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if isinstance(task_modes, (str, Path)):
        raise TypeError("task_modes must be a sequence of navigation modes")
    resolved = resolve_nav_configs(nav_config_path, task_modes)
    return (
        {mode: config.as_dict() for mode, config in resolved.items()},
        dict(NAV_ARG_CONFIG_DEFAULTS),
    )


def scalar_text(value: Scalar) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_overrides(overrides: Sequence[tuple[str, Scalar]]) -> str:
    return " ".join(f"{key}={scalar_text(value)}" for key, value in overrides)


def parse_common_nav_overrides(
    values: Sequence[str],
    *,
    effective_configs: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    seen: set[str] = set()
    task_modes = tuple(effective_configs)
    for item in values:
        if "=" not in item:
            raise ValueError(f"--nav-override must be KEY=VALUE, got {item!r}")
        key, _ = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"--nav-override has an empty key: {item!r}")
        if any(marker in key for marker in (".", "[", "]")):
            raise ValueError(
                f"--nav-override only supports top-level keys, got {key!r}"
            )
        if is_sensitive_key(key):
            raise ValueError(f"--nav-override cannot modify sensitive field: {key}")
        missing = [mode for mode in task_modes if key not in effective_configs[mode]]
        if missing:
            raise ValueError(
                f"unknown --nav-override key {key!r} for: {', '.join(missing)}"
            )
        nonscalar = [
            mode for mode in task_modes if not is_scalar(effective_configs[mode][key])
        ]
        if nonscalar:
            raise ValueError(
                f"--nav-override {key} is not a scalar field for: "
                + ", ".join(nonscalar)
            )
        if key in seen:
            raise ValueError(f"duplicate common --nav-override key: {key}")
        _, raw_value = item.split("=", 1)
        value = parse_nav_override_scalar(raw_value)
        for mode in task_modes:
            validate_nav_override_type(
                key,
                value,
                effective_configs[mode][key],
            )
        seen.add(key)
        parsed.append((key, item))
    return parsed


def common_override_values(
    common_overrides: Sequence[tuple[str, str]],
) -> dict[str, Scalar]:
    values: dict[str, Scalar] = {}
    for key, item in common_overrides:
        _, raw_value = item.split("=", 1)
        values[key] = parse_nav_override_scalar(raw_value)
    return values


def baseline_config_for(
    mode: str,
    effective_configs: dict[str, dict[str, Any]],
    common_overrides: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    config = dict(effective_configs[mode])
    config.update(common_override_values(common_overrides))
    return config


def sweep_profile_name(key: str, value: Scalar) -> str:
    label = f"{key}_{scalar_text(value)}".lower()
    slug = re.sub(r"[^a-z0-9]+", "_", label).strip("_")[:48] or "point"
    digest = hashlib.sha256(
        f"{key}={scalar_text(value)}".encode("utf-8")
    ).hexdigest()[:8]
    return f"sweep_{slug}_{digest}"


def add_sensitivity_sweeps(
    specs: Sequence[str],
    *,
    task_modes: Sequence[str],
    selected: dict[str, list[str]],
    effective_configs: dict[str, dict[str, Any]],
    common_overrides: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_points: set[tuple[str, str, str]] = set()
    for spec in specs:
        if ":" not in spec or "=" not in spec:
            raise ValueError(
                "--sweep must use TASK:KEY=VALUE[,VALUE...] syntax"
            )
        mode, assignment = spec.split(":", 1)
        mode = mode.strip()
        if mode not in TASK_MODES:
            raise ValueError(f"unknown --sweep task mode: {mode!r}")
        if mode not in task_modes:
            raise ValueError(
                f"--sweep task {mode!r} is outside --task-mode"
            )
        key, raw_values = assignment.split("=", 1)
        key = key.strip()
        if not key or not raw_values.strip():
            raise ValueError(
                "--sweep must use TASK:KEY=VALUE[,VALUE...] syntax"
            )
        if any(marker in key for marker in (".", "[", "]")):
            raise ValueError(f"--sweep only supports top-level keys, got {key!r}")
        if is_sensitive_key(key):
            raise ValueError(f"--sweep cannot modify sensitive field: {key}")
        if key == "seed":
            raise ValueError("--sweep cannot vary seed; use --seed for paired runs")
        baseline = baseline_config_for(
            mode,
            effective_configs,
            common_overrides,
        )
        if key not in baseline:
            raise ValueError(f"unknown --sweep key for {mode}: {key}")
        if not is_scalar(baseline[key]):
            raise ValueError(f"--sweep key is not scalar for {mode}: {key}")
        if any(not part.strip() for part in raw_values.split(",")):
            raise ValueError(
                f"--sweep contains an empty value (commas are separators): {spec!r}"
            )
        raw_points = parse_csv(raw_values)
        if not raw_points:
            raise ValueError(f"--sweep has no values: {spec!r}")
        for raw_point in raw_points:
            value = parse_nav_override_scalar(raw_point)
            validate_nav_override_type(
                key,
                value,
                baseline[key],
                option_label="--sweep",
            )
            value = normalize_scalar_type(value, baseline[key])
            point_key = (
                mode,
                key,
                json.dumps(value, sort_keys=True, ensure_ascii=False),
            )
            if point_key in seen_points:
                continue
            seen_points.add(point_key)
            if value == baseline[key]:
                records.append(
                    {
                        "task_mode": mode,
                        "key": key,
                        "value": value,
                        "profile": "full",
                        "reuses_baseline": True,
                    }
                )
                continue
            candidate = {**baseline, key: value}
            if (
                key in dependency_inert_keys(mode, baseline)
                and key in dependency_inert_keys(mode, candidate)
            ):
                raise ValueError(
                    f"--sweep key {mode}:{key} is inert at both the full "
                    "baseline and requested point; enable its owning mode "
                    "or feature first"
                )
            name = sweep_profile_name(key, value)
            existing = ABLATIONS_BY_TASK[mode].get(name)
            profile = ablation(
                f"One-factor sensitivity point: {key}={scalar_text(value)}.",
                family=f"sensitivity:{key}",
                sweep=True,
                **{key: value},
            )
            if existing is not None and existing != profile:
                raise ValueError(f"sweep profile name collision: {mode}/{name}")
            ABLATIONS_BY_TASK[mode][name] = profile
            names = selected.setdefault(mode, ["full"])
            if "full" not in names:
                names.insert(0, "full")
            if name not in names:
                names.append(name)
            records.append(
                {
                    "task_mode": mode,
                    "key": key,
                    "value": value,
                    "profile": name,
                    "reuses_baseline": False,
                }
            )
    return records


def activation_rule_is_satisfied(
    rule: ActivationRule,
    config: dict[str, Any],
    inert: set[str],
) -> bool:
    def active_truthy(key: str) -> bool:
        return key not in inert and bool(config.get(key, False))

    if any(not active_truthy(key) for key in rule.all_true):
        return False
    if any(
        key in inert or bool(config.get(key, False))
        for key in rule.all_false
    ):
        return False
    if rule.any_true and not any(
        active_truthy(key) for key in rule.any_true
    ):
        return False
    if rule.any_all_true and not any(
        all(active_truthy(key) for key in path)
        for path in rule.any_all_true
    ):
        return False
    for key in rule.all_positive:
        if key in inert:
            return False
        try:
            value = float(config.get(key, 0))
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(value) or value <= 0.0:
            return False
    for key in rule.all_nonnegative:
        if key in inert:
            return False
        try:
            value = float(config.get(key, -1))
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(value) or value < 0.0:
            return False
    for key, allowed in rule.values_in:
        if key in inert or config.get(key) not in allowed:
            return False
    return True


def dependency_inert_keys(mode: str, config: dict[str, Any]) -> set[str]:
    rules: dict[str, list[ActivationRule]] = {
        child: [requires(*parents)]
        for child, parents in FEATURE_DEPENDENCIES[mode].items()
    }
    for key in config:
        for prefix, rule, exceptions in FEATURE_PREFIX_DEPENDENCIES[mode]:
            if (
                not key.startswith(prefix)
                or key in exceptions
                or key in rule.all_true
                or key in rule.all_false
                or key in rule.any_true
                or any(key in path for path in rule.any_all_true)
                or key in rule.all_positive
                or key in rule.all_nonnegative
            ):
                continue
            rules.setdefault(key, []).append(rule)
    for key, key_rules in CONDITIONAL_ACTIVATION_RULES[mode].items():
        if key in config:
            rules.setdefault(key, []).extend(key_rules)
    inert: set[str] = {
        key
        for key in config
        if key in NONBEHAVIORAL_OUTPUT_CONFIG_KEYS
        or any(
            key.startswith(prefix) for prefix in TASK_INAPPLICABLE_PREFIXES[mode]
        )
    }
    changed = True
    while changed:
        changed = False
        for child, child_rules in rules.items():
            if child in inert:
                continue
            if any(
                not activation_rule_is_satisfied(rule, config, inert)
                for rule in child_rules
            ):
                inert.add(child)
                changed = True
    return inert


def behavioral_signature(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_inert: set[str],
    candidate_inert: set[str],
) -> tuple[tuple[str, str], ...]:
    keys = sorted(
        key
        for key in set(baseline) | set(candidate)
        if is_scalar(baseline.get(key)) and is_scalar(candidate.get(key))
    )
    changes: list[tuple[str, str]] = []
    for key in keys:
        before: Any = "<inert>" if key in baseline_inert else baseline.get(key)
        after: Any = "<inert>" if key in candidate_inert else candidate.get(key)
        if before != after:
            changes.append(
                (
                    key,
                    json.dumps(after, sort_keys=True, ensure_ascii=False),
                )
            )
    return tuple(changes)


def validate_text_stop_control_baseline(baseline: dict[str, Any]) -> None:
    """Check the baseline settings the six text stop-control arms assume."""

    expected = {
        "sgf_explore": True,
        "sgf_stop": True,
        "stop_validation": True,
        "text_stop_ledger": True,
        "text_stop_verification_object": "node_footprint",
        "text_stop_evidence_rule": "sequential",
        "text_stop_reobservation": "active",
        "text_hypothesis_triage": True,
        "text_evidence_weighted_target_ranking": True,
        "text_metric_visual_closure": True,
        "text_validated_stop_approach": True,
    }
    mismatched = [
        f"{key}={scalar_text(baseline.get(key))} (expected {scalar_text(value)})"
        for key, value in expected.items()
        if baseline.get(key) != value
    ]
    if not 0.0 < float(baseline.get("text_stop_sequential_error_rate", 0.0)) < 0.5:
        mismatched.append("text_stop_sequential_error_rate must be in (0, 0.5)")
    if mismatched:
        raise ValueError(
            "same-model text stop controls require an intact full baseline: "
            + "; ".join(mismatched)
        )


def validate_profiles(
    selected: dict[str, list[str]],
    *,
    effective_configs: dict[str, dict[str, Any]],
    common_overrides: Sequence[tuple[str, str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    common_keys = {key for key, _ in common_overrides}
    analysis: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, names in selected.items():
        baseline = baseline_config_for(mode, effective_configs, common_overrides)
        if mode == "text" and TEXT_SIMPLE_STOP_PROFILES.intersection(names):
            validate_text_stop_control_baseline(baseline)
        baseline_inert = dependency_inert_keys(mode, baseline)
        signatures: dict[tuple[tuple[str, str], ...], str] = {}
        mode_analysis: dict[str, dict[str, Any]] = {}
        for name in names:
            profile = ABLATIONS_BY_TASK[mode][name]
            profile_keys = {key for key, _ in profile.overrides}
            unknown = sorted(profile_keys - set(baseline))
            if unknown:
                raise ValueError(
                    f"{mode}/{name} contains unknown runner override key(s): {', '.join(unknown)}"
                )
            nonscalar = sorted(
                key for key in profile_keys if not is_scalar(baseline[key])
            )
            if nonscalar:
                raise ValueError(
                    f"{mode}/{name} contains non-scalar override key(s): "
                    + ", ".join(nonscalar)
                )
            sensitive = sorted(key for key in profile_keys if is_sensitive_key(key))
            if sensitive:
                raise ValueError(
                    f"{mode}/{name} contains sensitive override key(s): "
                    + ", ".join(sensitive)
                )
            for key, value in profile.overrides:
                validate_nav_override_type(
                    key,
                    value,
                    baseline[key],
                    option_label=f"{mode}/{name}",
                )
            missing_evidence_map = sorted(
                key
                for key in profile_keys
                if key not in CONFIG_KEY_EVIDENCE and not profile.sweep
            )
            if missing_evidence_map:
                raise ValueError(
                    f"{mode}/{name} has no telemetry evidence mapping for: "
                    + ", ".join(missing_evidence_map)
                )
            conflicts = sorted(
                profile_keys & common_keys
                if not profile.sweep
                else set()
            )
            if conflicts:
                raise ValueError(
                    f"common --nav-override conflicts with {mode}/{name}: {', '.join(conflicts)}"
                )
            candidate = dict(baseline)
            candidate.update(dict(profile.overrides))
            candidate_inert = dependency_inert_keys(mode, candidate)
            signature = behavioral_signature(
                baseline,
                candidate,
                baseline_inert=baseline_inert,
                candidate_inert=candidate_inert,
            )
            raw_diff = {
                key: {"before": baseline[key], "after": candidate[key]}
                for key in sorted(profile_keys)
                if baseline[key] != candidate[key]
            }
            if name != "full" and not signature:
                raise ValueError(
                    f"{mode}/{name} is behaviorally identical to full under the "
                    "effective configuration"
                )
            if name != "full" and signature in signatures:
                raise ValueError(
                    f"{mode}/{name} duplicates {mode}/{signatures[signature]} "
                    "after dependency closure"
                )
            if name != "full":
                signatures[signature] = name
            direct_specs = () if profile.sweep else profile.evidence_specs
            direct_requirements = tuple(
                EvidenceRequirement(
                    spec.evidence_key,
                    spec.expectation,
                    (
                        "direct"
                        if spec.expectation == "disabled"
                        else "context"
                    ),
                )
                for spec in direct_specs
            )
            newly_inert_keys = candidate_inert - baseline_inert
            newly_active_keys = baseline_inert - candidate_inert
            dependent_specs = (
                ()
                if profile.sweep
                else evidence_specs_for_config_keys(
                    sorted(newly_inert_keys),
                    expectations=frozenset({"disabled"}),
                )
            )
            dependent_requirements = tuple(
                EvidenceRequirement(
                    spec.evidence_key,
                    "disabled",
                    "dependent",
                )
                for spec in dependent_specs
            )
            activated_specs = (
                ()
                if profile.sweep
                else evidence_specs_for_config_keys(
                    sorted(newly_active_keys),
                    expectations=frozenset({"disabled"}),
                )
            )
            activated_requirements = tuple(
                EvidenceRequirement(
                    spec.evidence_key,
                    "activated",
                    "context",
                )
                for spec in activated_specs
            )
            evidence_requirements = merge_evidence_requirements(
                direct_requirements,
                dependent_requirements,
                activated_requirements,
            )
            if name != "full":
                capability_keys = CAPABILITY_KEYS[mode]
                disabled_capabilities = sorted(
                    requirement.evidence_key
                    for requirement in evidence_requirements
                    if requirement.expectation == "disabled"
                    and requirement.evidence_key in capability_keys
                )
                if disabled_capabilities:
                    raise ValueError(
                        f"{mode}/{name} disables capability evidence "
                        f"{', '.join(disabled_capabilities)}; every ablation "
                        "must keep the agent's ability to search, approach, "
                        "and stop"
                    )
                evidence_requirements = merge_evidence_requirements(
                    evidence_requirements,
                    tuple(
                        EvidenceRequirement(key, "capability", "capability")
                        for key in capability_keys
                    ),
                )
            direct_evidence = tuple(
                requirement.evidence_key
                for requirement in direct_requirements
                if requirement.expectation == "disabled"
            )
            dependent_evidence = tuple(
                requirement.evidence_key
                for requirement in evidence_requirements
                if requirement.expectation == "disabled"
                and requirement.role == "dependent"
            )
            retained_evidence = tuple(
                requirement.evidence_key
                for requirement in evidence_requirements
                if requirement.expectation == "retained"
            )
            activated_evidence = tuple(
                requirement.evidence_key
                for requirement in evidence_requirements
                if requirement.expectation == "activated"
            )
            capability_evidence = tuple(
                requirement.evidence_key
                for requirement in evidence_requirements
                if requirement.expectation == "capability"
            )
            mode_analysis[name] = {
                "tier": profile_tier(mode, name),
                "family": profile.family,
                "evidence_keys": list(direct_evidence),
                "direct_evidence_keys": list(direct_evidence),
                "dependent_evidence_keys": list(dependent_evidence),
                "retained_evidence_keys": list(retained_evidence),
                "activated_evidence_keys": list(activated_evidence),
                "capability_evidence_keys": list(capability_evidence),
                "evidence_contracts": [
                    evidence_requirement_payload(requirement)
                    for requirement in evidence_requirements
                ],
                "config_diff": raw_diff,
                "changed_keys": [key for key, _ in signature],
                "inert_by_dependency": sorted(newly_inert_keys),
                "activated_by_dependency": sorted(newly_active_keys),
                "baseline_inert_by_dependency": sorted(baseline_inert),
                "behavior_signature_sha256": hashlib.sha256(
                    json.dumps(
                        signature,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        analysis[mode] = mode_analysis
    return analysis


def uncalibrated_exchange_rate_profiles(
    selected: dict[str, list[str]],
    *,
    effective_configs: dict[str, dict[str, Any]],
    common_overrides: Sequence[tuple[str, str]],
) -> list[str]:
    pending: list[str] = []
    for mode, names in selected.items():
        baseline = baseline_config_for(mode, effective_configs, common_overrides)
        for name in names:
            candidate = {
                **baseline,
                **dict(ABLATIONS_BY_TASK[mode][name].overrides),
            }
            if (
                candidate.get("sgf_explore_utility_form") == "difference"
                and float(candidate.get("sgf_explore_exchange_rate_scale", 1.0))
                > 0.0
                and float(candidate.get("sgf_explore_exchange_rate", 0.0))
                <= 0.0
            ):
                pending.append(f"{mode}/{name}")
    return pending


def validate_evidence_aggregation(
    analysis: dict[str, dict[str, dict[str, Any]]],
    *,
    batch_runner_path: Path,
) -> None:
    """Reject evidence counters the batch runner does not aggregate."""

    retained = runpy.run_path(
        str(batch_runner_path),
        run_name="batch_runner_schema",
    ).get("RETAINED_FEATURE_ACTIVITY_KEYS")
    if retained is None:
        return
    missing = sorted(
        {
            contract["evidence_key"]
            for mode_analysis in analysis.values()
            for detail in mode_analysis.values()
            for contract in detail["evidence_contracts"]
        }
        - set(retained)
    )
    if missing:
        raise ValueError(
            f"batch runner {batch_runner_path} does not aggregate evidence "
            f"counter(s) the profiles require: {', '.join(missing)}"
        )


def option_name(token: str) -> str:
    return token.split("=", 1)[0] if token.startswith("--") else ""


def normalize_batch_args(values: Sequence[str]) -> list[str]:
    batch_args = list(values)
    if batch_args and batch_args[0] == "--":
        batch_args.pop(0)
    conflicts = sorted(
        {
            name
            for token in batch_args
            if (name := option_name(token))
            and name in MANAGED_BATCH_OPTIONS
        }
    )
    if conflicts:
        raise ValueError(
            "batch arguments cannot override ablation-managed option(s): "
            + ", ".join(conflicts)
        )
    return batch_args


def effective_batch_max_steps(batch_args: Sequence[str]) -> int:
    values: list[str] = []
    index = 0
    while index < len(batch_args):
        token = str(batch_args[index])
        if token == "--max-steps":
            if index + 1 >= len(batch_args):
                raise ValueError("--max-steps in batch arguments requires a value")
            values.append(str(batch_args[index + 1]))
            index += 2
            continue
        if token.startswith("--max-steps="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) > 1:
        raise ValueError("duplicate --max-steps in batch arguments")
    if not values:
        return DEFAULT_BATCH_MAX_STEPS
    try:
        parsed = int(values[0])
    except ValueError as exc:
        raise ValueError("--max-steps in batch arguments must be an integer") from exc
    if parsed <= 0:
        raise ValueError("--max-steps in batch arguments must be positive")
    return parsed


def runner_args_for(
    profile: Ablation,
    common_overrides: Sequence[tuple[str, str]],
) -> str:
    tokens: list[str] = []
    for _, item in common_overrides:
        tokens.extend(["--nav-override", item])
    # Profile overrides take precedence over common overrides
    for key, value in profile.overrides:
        tokens.extend(["--nav-override", f"{key}={scalar_text(value)}"])
    return shlex.join(tokens)


def build_batch_command(
    *,
    python_executable: str,
    batch_runner_path: Path,
    runner_path: Path,
    nav_config_path: Path,
    mode_output_root: Path,
    task_mode: str,
    name: str,
    profile: Ablation,
    common_overrides: Sequence[tuple[str, str]],
    batch_args: Sequence[str],
    resume: bool,
    overwrite: bool,
    metrics_only: bool,
) -> list[str]:
    command = [
        python_executable,
        str(batch_runner_path),
        "--python-executable",
        python_executable,
        "--runner-path",
        str(runner_path),
        "--nav-config-path",
        str(nav_config_path),
        "--task-mode",
        task_mode,
        "--output-root",
        str(mode_output_root),
        "--run-tag",
        name,
        "--no-aggregate-summary",
        "--runner-args",
        runner_args_for(profile, common_overrides),
    ]
    if resume:
        command.append("--resume")
    if overwrite:
        command.append("--overwrite")
    if metrics_only:
        command.append("--metrics-only")
    command.extend(batch_args)
    return command


def execution_plan_payload(
    commands: Sequence[tuple[str, str, Sequence[str]]],
    *,
    keep_going: bool,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for mode, name, command in commands:
        if len(command) < 2:
            raise ValueError(
                f"cannot build execution plan for {mode}/{name}: "
                "batch command is missing its Python executable or runner path"
            )
        runs.append(
            {
                "mode": mode,
                "name": name,
                "argv": list(command[2:]),
            }
        )
    return {
        "version": EXECUTION_PLAN_VERSION,
        "keep_going": bool(keep_going),
        "runs": runs,
    }


def subprocess_detail(value: Any, *, limit: int = 2048) -> str:
    if isinstance(value, bytes):
        detail = value.decode("utf-8", errors="replace")
    else:
        detail = str(value or "")
    detail = detail.strip()
    return detail[-limit:] if len(detail) > limit else detail


def probe_execution_plan_capability(
    *,
    python_executable: str,
    batch_runner_path: Path,
) -> tuple[bool, str | None]:
    command = [
        python_executable,
        str(batch_runner_path),
        "--execution-plan-capabilities",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=EXECUTION_PLAN_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return (
            False,
            "execution-plan capability probe timed out after "
            f"{EXECUTION_PLAN_PROBE_TIMEOUT_SECONDS} seconds",
        )
    except OSError as exc:
        return False, f"execution-plan capability probe could not start: {exc}"
    if completed.returncode != 0:
        detail = subprocess_detail(completed.stderr)
        suffix = f": {detail}" if detail else ""
        return (
            False,
            "execution-plan capability probe exited with return code "
            f"{completed.returncode}{suffix}",
        )
    try:
        capabilities = json.loads(str(completed.stdout or ""))
    except json.JSONDecodeError as exc:
        return False, f"execution-plan capability probe returned invalid JSON: {exc}"
    if not isinstance(capabilities, dict):
        return False, "execution-plan capability probe did not return a JSON object"
    version = capabilities.get("execution_plan_version")
    if isinstance(version, bool) or version != EXECUTION_PLAN_VERSION:
        return (
            False,
            "batch runner does not advertise execution plan protocol version "
            f"{EXECUTION_PLAN_VERSION}",
        )
    return True, None


def run_metrics_only_subprocess(
    command: Sequence[str],
) -> tuple[subprocess.CompletedProcess[Any], str]:
    """Run with stdout discarded and stderr spooled to a temporary file."""

    with tempfile.TemporaryFile(mode="w+b") as stderr_handle:
        completed = subprocess.run(
            list(command),
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=stderr_handle,
            check=False,
        )
        stderr_handle.flush()
        stderr_handle.seek(0, os.SEEK_END)
        size = stderr_handle.tell()
        stderr_handle.seek(
            max(0, size - SUBPROCESS_ERROR_TAIL_BYTES),
            os.SEEK_SET,
        )
        detail = stderr_handle.read().decode("utf-8", errors="replace").strip()
    return completed, detail


def parse_execution_plan_result(
    result_path: Path,
    *,
    expected_runs: Sequence[tuple[str, str, Sequence[str]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not result_path.is_file():
        raise ValueError("execution-plan result file was not created")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read execution-plan result: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("execution-plan result must be a JSON object")
    version = payload.get("version")
    if isinstance(version, bool) or version != EXECUTION_PLAN_VERSION:
        raise ValueError(
            "execution-plan result has an unsupported or missing version"
        )
    scheduler = payload.get("scheduler")
    if not isinstance(scheduler, dict):
        raise ValueError("execution-plan result scheduler must be a JSON object")
    result_runs = payload.get("runs")
    if not isinstance(result_runs, list):
        raise ValueError("execution-plan result runs must be a JSON array")

    expected_identities = {(mode, name) for mode, name, _ in expected_runs}
    identities: set[tuple[str, str]] = set()
    validated_runs: list[dict[str, Any]] = []
    for index, run in enumerate(result_runs):
        if not isinstance(run, dict):
            raise ValueError(
                f"execution-plan result run {index} must be a JSON object"
            )
        mode = run.get("mode")
        name = run.get("name")
        if not isinstance(mode, str) or not isinstance(name, str):
            raise ValueError(
                f"execution-plan result run {index} has an invalid identity"
            )
        identity = (mode, name)
        if identity not in expected_identities:
            raise ValueError(
                "execution-plan result contains an unexpected run: "
                f"{mode}/{name}"
            )
        if identity in identities:
            raise ValueError(
                "execution-plan result contains a duplicate run: "
                f"{mode}/{name}"
            )
        state = run.get("state")
        if state not in {"completed", "failed", "not_started"}:
            raise ValueError(
                f"execution-plan result run {mode}/{name} has invalid state: "
                f"{state!r}"
            )
        returncode = run.get("returncode")
        if returncode is not None and (
            isinstance(returncode, bool) or not isinstance(returncode, int)
        ):
            raise ValueError(
                f"execution-plan result run {mode}/{name} has an invalid returncode"
            )
        error = run.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError(
                f"execution-plan result run {mode}/{name} has an invalid error"
            )
        if state == "completed" and returncode != 0:
            raise ValueError(
                f"execution-plan result run {mode}/{name} must use returncode 0 "
                "when state is completed"
            )
        if state == "failed" and (
            returncode is None or returncode == 0
        ):
            raise ValueError(
                f"execution-plan result run {mode}/{name} must use a non-zero "
                "returncode when state is failed"
            )
        if state == "not_started" and returncode is not None:
            raise ValueError(
                f"execution-plan result run {mode}/{name} must use a null "
                "returncode when state is not_started"
            )
        identities.add(identity)
        validated_runs.append(dict(run))
    missing = expected_identities - identities
    if missing:
        labels = ", ".join(f"{mode}/{name}" for mode, name in sorted(missing))
        raise ValueError(f"execution-plan result is missing run(s): {labels}")
    return dict(scheduler), validated_runs


@dataclass(frozen=True)
class SharedExecutionOutcome:
    wrapper_returncode: int
    wrapper_error: str
    scheduler: dict[str, Any] | None
    runs: tuple[dict[str, Any], ...] | None
    result_error: str | None
    interrupted: bool = False


def run_shared_execution_plan(
    *,
    python_executable: str,
    batch_runner_path: Path,
    experiment_root: Path,
    commands: Sequence[tuple[str, str, Sequence[str]]],
    keep_going: bool,
    metrics_only: bool,
) -> SharedExecutionOutcome:
    payload = execution_plan_payload(commands, keep_going=keep_going)
    with tempfile.TemporaryDirectory(
        prefix=".execution-plan-",
        dir=experiment_root,
    ) as temporary_dir:
        temporary_path = Path(temporary_dir)
        plan_path = temporary_path / "plan.json"
        result_path = temporary_path / "result.json"
        plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        command = [
            python_executable,
            str(batch_runner_path),
            "--execution-plan",
            str(plan_path),
            "--execution-result",
            str(result_path),
        ]
        interrupted = False

        def wait_for_wrapper(
            *,
            stdout_target: Any = None,
            stderr_target: Any = None,
        ) -> int:
            nonlocal interrupted
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=stdout_target,
                stderr=stderr_target,
                start_new_session=True,
            )
            try:
                return int(process.wait())
            except KeyboardInterrupt:
                interrupted = True
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    return int(
                        process.wait(
                            timeout=EXECUTION_PLAN_INTERRUPT_TIMEOUT_SECONDS
                        )
                    )
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        return int(process.wait(timeout=5))
                    except subprocess.TimeoutExpired:
                        process.kill()
                        return int(process.wait())

        try:
            if metrics_only:
                with tempfile.TemporaryFile(mode="w+b") as stderr_handle:
                    wrapper_returncode = wait_for_wrapper(
                        stdout_target=subprocess.DEVNULL,
                        stderr_target=stderr_handle,
                    )
                    stderr_handle.flush()
                    stderr_handle.seek(0, os.SEEK_END)
                    size = stderr_handle.tell()
                    stderr_handle.seek(
                        max(0, size - SUBPROCESS_ERROR_TAIL_BYTES),
                        os.SEEK_SET,
                    )
                    wrapper_error = (
                        stderr_handle.read()
                        .decode("utf-8", errors="replace")
                        .strip()
                    )
            else:
                wrapper_returncode = wait_for_wrapper()
                wrapper_error = ""
        except OSError as exc:
            wrapper_returncode = 1
            wrapper_error = f"execution-plan wrapper could not start: {exc}"
        try:
            scheduler, runs = parse_execution_plan_result(
                result_path,
                expected_runs=commands,
            )
        except ValueError as exc:
            return SharedExecutionOutcome(
                wrapper_returncode=wrapper_returncode,
                wrapper_error=wrapper_error,
                scheduler=None,
                runs=None,
                result_error=str(exc),
                interrupted=interrupted,
            )
    return SharedExecutionOutcome(
        wrapper_returncode=wrapper_returncode,
        wrapper_error=wrapper_error,
        scheduler=scheduler,
        runs=tuple(runs),
        result_error=None,
        interrupted=interrupted,
    )


def run_serial_batches(
    commands: Sequence[tuple[str, str, Sequence[str]]],
    *,
    metrics_only: bool,
    keep_going: bool,
) -> tuple[dict[tuple[str, str], int], bool, list[dict[str, Any]]]:
    returncodes: dict[tuple[str, str], int] = {}
    result_runs: list[dict[str, Any]] = []
    batch_failed = False
    stopped = False
    for mode, name, command in commands:
        if stopped:
            result_runs.append(
                {
                    "mode": mode,
                    "name": name,
                    "state": "not_started",
                    "returncode": None,
                    "error": "not started after an earlier batch failure",
                }
            )
            continue
        if not metrics_only:
            print(f"\n=== running {mode}/{name} ===", flush=True)
        try:
            if metrics_only:
                completed, detail = run_metrics_only_subprocess(command)
            else:
                completed = subprocess.run(
                    list(command),
                    cwd=REPO_ROOT,
                    check=False,
                )
                detail = ""
            returncode = int(completed.returncode)
            error = detail if returncode != 0 and detail else None
        except OSError as exc:
            returncode = 1
            error = f"batch could not start: {exc}"
        returncodes[(mode, name)] = returncode
        state = "completed" if returncode == 0 else "failed"
        if returncode != 0 and error is None:
            error = f"batch exited with return code {returncode}"
        result_runs.append(
            {
                "mode": mode,
                "name": name,
                "state": state,
                "returncode": returncode,
                "error": error,
            }
        )
        if returncode != 0:
            batch_failed = True
            suffix = f": {error}" if error else ""
            print(
                f"batch failed for {mode}/{name} with return code "
                f"{returncode}{suffix}",
                file=sys.stderr,
                flush=True,
            )
            if not keep_going:
                stopped = True
    return returncodes, batch_failed, result_runs


def run_dir_for(experiment_root: Path, task_mode: str, name: str) -> Path:
    return experiment_root / task_mode / "runs" / name


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def summary_is_complete(summary: dict[str, Any]) -> bool:
    if not summary or summary.get("status") != "completed":
        return False
    episodes = optional_number(summary.get("episodes"))
    completed = optional_number(summary.get("completed"))
    pending = optional_number(summary.get("pending"))
    if any(value is None for value in (episodes, completed, pending)):
        return False
    return bool(
        episodes >= 0
        and episodes == int(episodes)
        and completed >= 0
        and completed == int(completed)
        and pending >= 0
        and pending == int(pending)
        and completed == episodes
        and pending == 0
    )


def report_selection(
    experiment_root: Path,
    selected: dict[str, list[str]],
) -> dict[str, list[str]]:
    report: dict[str, list[str]] = {}
    for mode in selected:
        requested = set(selected.get(mode, ()))
        names = [
            name
            for name in ABLATIONS_BY_TASK[mode]
            if name in requested
            or (run_dir_for(experiment_root, mode, name) / "summary.json").exists()
        ]
        if names:
            report[mode] = names
    return report


def expanded_resume_selection(
    experiment_root: Path,
    selected: dict[str, list[str]],
) -> dict[str, list[str]]:
    expanded: dict[str, list[str]] = {}
    for mode, requested_names in selected.items():
        requested = set(requested_names)
        requested.update(
            name
            for name in ABLATIONS_BY_TASK[mode]
            if run_dir_for(experiment_root, mode, name).exists()
        )
        expanded[mode] = [name for name in ABLATIONS_BY_TASK[mode] if name in requested]
    return expanded


def prepare_experiment_root(path: Path, *, resume: bool, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        elif not resume:
            raise FileExistsError(
                f"experiment directory already exists: {path}. Use --resume or --overwrite."
            )
    path.mkdir(parents=True, exist_ok=True)


def normalized_batch_command(command: Sequence[str]) -> tuple[str, ...]:
    return tuple(token for token in command if token not in {"--resume", "--overwrite"})


def validate_resume_compatibility(
    *,
    experiment_root: Path,
    commands: Sequence[tuple[str, str, Sequence[str]]],
    runner_path: Path,
    batch_runner_path: Path,
    compare_runner_path: Path,
    nav_config_path: Path,
) -> None:
    existing = [
        (mode, name, command)
        for mode, name, command in commands
        if run_dir_for(experiment_root, mode, name).exists()
    ]
    if not existing:
        return
    manifest_path = experiment_root / "manifest.json"
    manifest = read_json(manifest_path)
    invocations = manifest.get("invocations", [])
    if not isinstance(invocations, list):
        invocations = []
    current_runner_hash = sha256_file(runner_path)
    current_batch_runner_hash = sha256_file(batch_runner_path)
    current_compare_runner_hash = sha256_file(compare_runner_path)
    current_orchestrator_hash = sha256_file(Path(__file__).resolve())
    current_config_hash = sha256_file(nav_config_path)
    current_source_hash = navigation_source_sha256(runner_path, batch_runner_path)
    current_commit = git_value("rev-parse", "--short", "HEAD")
    for mode, name, command in existing:
        identity = normalized_batch_command(command)
        matching_invocation: dict[str, Any] | None = None
        for invocation in reversed(invocations):
            if not isinstance(invocation, dict):
                continue
            prior_commands = invocation.get("commands", [])
            if not isinstance(prior_commands, list):
                continue
            if any(
                isinstance(prior_command, list)
                and normalized_batch_command(prior_command) == identity
                for prior_command in prior_commands
            ):
                matching_invocation = invocation
                break
        if matching_invocation is None:
            raise ValueError(
                f"cannot resume {mode}/{name}: its command differs from the recorded invocation"
            )
        if matching_invocation.get("runner_sha256") != current_runner_hash:
            raise ValueError(f"cannot resume {mode}/{name}: runner source hash changed")
        if matching_invocation.get("batch_runner_sha256") != current_batch_runner_hash:
            raise ValueError(f"cannot resume {mode}/{name}: batch runner source hash changed")
        if (
            matching_invocation.get("compare_runner_sha256")
            != current_compare_runner_hash
        ):
            raise ValueError(
                f"cannot resume {mode}/{name}: comparison runner source hash changed"
            )
        if (
            matching_invocation.get("orchestrator_sha256")
            != current_orchestrator_hash
        ):
            raise ValueError(
                f"cannot resume {mode}/{name}: ablation orchestrator source hash changed"
            )
        if matching_invocation.get("nav_config_sha256") != current_config_hash:
            raise ValueError(f"cannot resume {mode}/{name}: navigation config hash changed")
        if matching_invocation.get("navigation_source_sha256") != current_source_hash:
            raise ValueError(f"cannot resume {mode}/{name}: navigation source tree changed")
        if matching_invocation.get("commit") != current_commit:
            raise ValueError(f"cannot resume {mode}/{name}: git commit changed")


def sanitized_scalar_config(config: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key in sorted(config):
        value = config[key]
        if not is_scalar(value):
            continue
        sanitized[key] = "<redacted>" if is_sensitive_key(key) else value
    return sanitized


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def update_manifest_execution_scheduler(
    *,
    experiment_root: Path,
    scheduler: dict[str, Any],
    runs: Sequence[dict[str, Any]],
) -> None:
    """Merge execution details into the most recently recorded invocation."""

    manifest_path = experiment_root / "manifest.json"
    manifest = read_json(manifest_path)
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list) or not invocations:
        raise ValueError("cannot update execution scheduler: manifest has no invocation")
    invocation: dict[str, Any] | None = None
    for candidate in reversed(invocations):
        if isinstance(candidate, dict):
            invocation = candidate
            break
    if invocation is None:
        raise ValueError(
            "cannot update execution scheduler: manifest invocation is invalid"
        )
    current = invocation.get("execution_scheduler")
    merged = dict(current) if isinstance(current, dict) else {}
    merged.update(scheduler)
    merged["runs"] = [dict(run) for run in runs]
    invocation["execution_scheduler"] = merged
    manifest["invocations"] = invocations
    manifest["updated_at"] = utc_now_iso()
    write_json_atomic(manifest_path, manifest)


def append_manifest_invocation(
    *,
    experiment_root: Path,
    experiment_name: str,
    runner_path: Path,
    batch_runner_path: Path,
    compare_runner_path: Path,
    nav_config_path: Path,
    selected: dict[str, list[str]],
    common_overrides: Sequence[tuple[str, str]],
    effective_configs: dict[str, dict[str, Any]],
    profile_analysis: dict[str, dict[str, dict[str, Any]]],
    sweep_records: Sequence[dict[str, Any]],
    navigation_seed: int,
    statistics_seed: int,
    bootstrap: int,
    batch_args: Sequence[str],
    commands: Sequence[Sequence[str]],
    metrics_only: bool,
    execution_scheduler: dict[str, Any] | None = None,
) -> None:
    manifest_path = experiment_root / "manifest.json"
    manifest = read_json(manifest_path)
    if not manifest:
        manifest = {
            "experiment_name": experiment_name,
            "created_at": utc_now_iso(),
            "repo_root": str(REPO_ROOT),
            "invocations": [],
        }
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list):
        invocations = []
    invocations.append(
        {
            "timestamp": utc_now_iso(),
            "argv": sys.argv,
            "commit": git_value("rev-parse", "--short", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "runner_path": str(runner_path),
            "batch_runner_path": str(batch_runner_path),
            "compare_runner_path": str(compare_runner_path),
            "nav_config_path": str(nav_config_path),
            "nav_config_sha256": sha256_file(nav_config_path),
            "runner_sha256": sha256_file(runner_path),
            "batch_runner_sha256": sha256_file(batch_runner_path),
            "compare_runner_sha256": sha256_file(compare_runner_path),
            "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
            "navigation_source_sha256": navigation_source_sha256(
                runner_path,
                batch_runner_path,
            ),
            "selected_ablations": selected,
            "matrix_selectors": {
                "core": CORE_ABLATIONS,
                "components": COMPONENT_ABLATIONS,
                "all": ALL_ABLATIONS,
                "representation-coupling": REPRESENTATION_COUPLING_ABLATIONS,
                "text-stop-controls": {"text": TEXT_STOP_CONTROLS},
            },
            "resolved_full_config": {
                mode: sanitized_scalar_config(
                    baseline_config_for(
                        mode,
                        effective_configs,
                        common_overrides,
                    )
                )
                for mode in selected
            },
            "profiles": {
                mode: {
                    name: {
                        "description": ABLATIONS_BY_TASK[mode][name].description,
                        "family": ABLATIONS_BY_TASK[mode][name].family,
                        "tier": profile_analysis[mode][name]["tier"],
                        "sweep": ABLATIONS_BY_TASK[mode][name].sweep,
                        "contrast_with": list(
                            ABLATIONS_BY_TASK[mode][name].contrast_with
                        ),
                        "evidence_keys": list(
                            profile_analysis[mode][name][
                                "direct_evidence_keys"
                            ]
                        ),
                        "dependent_evidence_keys": profile_analysis[mode][name][
                            "dependent_evidence_keys"
                        ],
                        "retained_evidence_keys": profile_analysis[mode][name][
                            "retained_evidence_keys"
                        ],
                        "activated_evidence_keys": profile_analysis[mode][name][
                            "activated_evidence_keys"
                        ],
                        "capability_evidence_keys": profile_analysis[mode][
                            name
                        ]["capability_evidence_keys"],
                        "evidence_contracts": profile_analysis[mode][name][
                            "evidence_contracts"
                        ],
                        "nav_overrides": [
                            f"{key}={scalar_text(value)}"
                            for key, value in ABLATIONS_BY_TASK[mode][name].overrides
                        ],
                        "config_diff": profile_analysis[mode][name]["config_diff"],
                        "changed_keys": profile_analysis[mode][name]["changed_keys"],
                        "inert_by_dependency": profile_analysis[mode][name][
                            "inert_by_dependency"
                        ],
                        "activated_by_dependency": profile_analysis[mode][name][
                            "activated_by_dependency"
                        ],
                        "behavior_signature_sha256": profile_analysis[mode][name][
                            "behavior_signature_sha256"
                        ],
                    }
                    for name in names
                }
                for mode, names in selected.items()
            },
            "common_nav_overrides": [item for _, item in common_overrides],
            "sensitivity_points": list(sweep_records),
            "navigation_seed": int(navigation_seed),
            "statistics": {
                "bootstrap_iterations": int(bootstrap),
                "statistics_seed": int(statistics_seed),
                "failure_policy": "navigation_failure_zero_sr_spl",
            },
            "batch_args": list(batch_args),
            "metrics_only": bool(metrics_only),
            "commands": [list(command) for command in commands],
            "execution_scheduler": dict(execution_scheduler or {}),
        }
    )
    manifest["invocations"] = invocations
    manifest["updated_at"] = utc_now_iso()
    write_json_atomic(manifest_path, manifest)


def recorded_invocation_state(
    experiment_root: Path,
) -> tuple[dict[tuple[str, str], int], bool | None]:
    """Return codes and metrics-only flag of the last recorded invocation."""

    manifest = read_json(experiment_root / "manifest.json")
    invocations = manifest.get("invocations")
    if not isinstance(invocations, list):
        return {}, None
    invocation = next(
        (item for item in reversed(invocations) if isinstance(item, dict)),
        None,
    )
    if invocation is None:
        return {}, None
    returncodes: dict[tuple[str, str], int] = {}
    scheduler = invocation.get("execution_scheduler")
    runs = scheduler.get("runs") if isinstance(scheduler, dict) else None
    for run in runs if isinstance(runs, list) else ():
        if not isinstance(run, dict):
            continue
        returncode = run.get("returncode")
        if isinstance(returncode, int) and not isinstance(returncode, bool):
            returncodes[(str(run.get("mode")), str(run.get("name")))] = returncode
    metrics_only = invocation.get("metrics_only")
    return returncodes, metrics_only if isinstance(metrics_only, bool) else None


def format_delta_ci(metric: dict[str, Any]) -> str:
    delta = f"{metric['delta']:.6f}"
    interval = metric["ci95"]
    if interval is None:
        return delta
    return f"{delta} [{interval[0]:.6f}, {interval[1]:.6f}]"


def write_experiment_reports(
    *,
    compare_runner_path: Path,
    experiment_root: Path,
    selected: dict[str, list[str]],
    returncodes: dict[tuple[str, str], int],
    bootstrap: int,
    statistics_seed: int,
    navigation_seed: int,
) -> tuple[dict[str, Any], bool]:
    """Write the final report, scoring episode errors as navigation failures."""
    comparator = runpy.run_path(str(compare_runner_path))
    for helper in ("read_navigation_rows", "compare_navigation_metrics"):
        if not callable(comparator.get(helper)):
            raise ValueError(f"comparison module {compare_runner_path} lacks {helper}")
    selected_for_report = report_selection(experiment_root, selected)
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for mode, names in selected_for_report.items():
        observations: dict[str, dict] = {}
        mode_rows: dict[str, dict[str, Any]] = {}
        for name in names:
            row = {column: "" for column in SUMMARY_COLUMNS}
            row.update(task_mode=mode, ablation=name)
            mode_rows[name] = row
            rows.append(row)
            run_dir = run_dir_for(experiment_root, mode, name)
            try:
                summary = read_json(run_dir / "summary.json")
                if not summary_is_complete(summary):
                    raise ValueError(f"incomplete ablation run: {mode}/{name}")
                values = comparator["read_navigation_rows"](
                    run_dir, navigation_seed=navigation_seed,
                )
                if len(values) != int(summary["episodes"]):
                    raise ValueError(f"episode count mismatch for {mode}/{name}")
                observations[name] = values
                for column, field in (("SR", "success"), ("SPL", "spl")):
                    average = sum(value[field] for value in values.values()) / len(values)
                    row[column] = f"{average:.6f}"
                if returncodes.get((mode, name), 0) != 0:
                    issues.append(f"batch failed for {mode}/{name}")
            except (OSError, ValueError) as error:
                issues.append(str(error))

        baseline = observations.get("full")
        for name, values in observations.items():
            if baseline is None:
                issues.append(f"missing complete full baseline for {mode}/{name}")
                continue
            try:
                if name == "full":
                    metrics = {
                        field: {"delta": 0.0, "ci95": (0.0, 0.0) if bootstrap else None}
                        for field in ("success", "spl")
                    }
                else:
                    metrics = comparator["compare_navigation_metrics"](
                        f"{mode}/{name}", baseline, values,
                        bootstrap=bootstrap, seed=statistics_seed,
                    )
                mode_rows[name]["ΔSR [95% CI]"] = format_delta_ci(metrics["success"])
                mode_rows[name]["ΔSPL [95% CI]"] = format_delta_ci(metrics["spl"])
            except ValueError as error:
                issues.append(str(error))

    path = experiment_root / "summary.tsv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    for issue in issues:
        print(issue, file=sys.stderr)
    print(f"summary={path}")
    return {"passed": not issues, "runs": rows, "issues": issues}, bool(issues)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired mechanism and bundle ablations for SGF text-goal and "
            "instance-image navigation. Arguments after '--' are forwarded to batch_eval_nav.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
        epilog=(
            "examples:\n"
            "  %(prog)s --list-ablations\n"
            "  %(prog)s --task-mode both --ablations core --dry-run -- "
            "--episodes 0-9 --num-workers 2\n"
            "  %(prog)s --task-mode text --ablations core "
            "--sweep text:text_stop_requery_interval_steps=4,8,16 --dry-run -- "
            "--episodes 0-9\n"
            "  %(prog)s --task-mode text --ablations all --experiment-name text_seed0 -- "
            "--episodes 0-99 --num-workers 4 --timeout-seconds 1500"
        ),
    )
    parser.add_argument(
        "--task-mode",
        choices=(*TASK_MODES, "both"),
        default="both",
        help="Task mode(s) to evaluate.",
    )
    parser.add_argument(
        "--ablations",
        default="core",
        help=(
            "Comma-separated profile names, or one selector: core, components, "
            "text-stop-controls, representation-coupling, all (default: core)."
        ),
    )
    parser.add_argument(
        "--no-auto-contrasts",
        action="store_true",
        help=(
            "Do not add contrast_with reference profiles; with --resume, resume "
            "only the selected profiles."
        ),
    )
    parser.add_argument(
        "--skip-full",
        action="store_true",
        help=(
            "Exclude full from execution; existing full results are still used "
            "as the paired baseline."
        ),
    )
    parser.add_argument("--list-ablations", action="store_true", help="List built-in profiles and exit.")
    parser.add_argument(
        "--list-ablations-format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Catalog output format (default: text).",
    )
    parser.add_argument(
        "--experiment-name",
        default="",
        help="Output directory name. Defaults to a UTC timestamp plus the short commit.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--runner-path", type=Path, default=DEFAULT_RUNNER_PATH)
    parser.add_argument("--batch-runner-path", type=Path, default=DEFAULT_BATCH_RUNNER_PATH)
    parser.add_argument(
        "--compare-runner-path", type=Path, default=DEFAULT_COMPARE_RUNNER_PATH,
        help=(
            "Python module providing the paired comparison helpers "
            "(default: scripts/compare_nav_runs.py)."
        ),
    )
    parser.add_argument("--nav-config-path", type=Path, default=DEFAULT_NAV_CONFIG_PATH)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--nav-override",
        action="append",
        default=[],
        help="Common runner KEY=VALUE override applied to every selected profile. Repeatable.",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="TASK:KEY=V1,V2",
        help=(
            "Add one-factor-at-a-time sensitivity points; repeatable, and axes "
            "are not crossed."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Navigation seed shared by all paired runs (default: 0).",
    )
    parser.add_argument(
        "--statistics-seed",
        type=int,
        default=0,
        help=(
            "Seed for paired bootstrap resampling "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help=(
            "Paired bootstrap iterations used by compare_nav_runs.py "
            "(default: 10000; 0 disables confidence intervals)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without writing or running.",
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help=(
            "Enable metrics-only mode in batch and SGF navigation runners: no episode logs, "
            "images, videos, traces, or SGF dumps."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue with later profiles after a batch failure.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Rebuild summary.tsv from completed runs without launching episodes "
            "or modifying manifest.json."
        ),
    )
    run_state = parser.add_mutually_exclusive_group()
    run_state.add_argument("--resume", action="store_true", help="Resume every selected batch run.")
    run_state.add_argument("--overwrite", action="store_true", help="Replace the experiment directory.")
    parser.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="Allow --resume to mix runs after command, runner, or config changes.",
    )
    parser.add_argument(
        "batch_args",
        nargs=argparse.REMAINDER,
        help="Arguments after '--' forwarded unchanged to batch_eval_nav.py.",
    )
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be >= 0")
    if args.statistics_seed < 0:
        parser.error("--statistics-seed must be >= 0")
    if args.report_only and (args.resume or args.overwrite or args.dry_run):
        parser.error(
            "--report-only cannot be combined with --resume, --overwrite, or --dry-run"
        )
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.allow_resume_mismatch and not args.resume:
        parser.error("--allow-resume-mismatch requires --resume")
    return args


def main() -> int:
    args = parse_args()
    task_modes = task_modes_for(args.task_mode)
    runner_path = resolve_repo_path(args.runner_path)
    batch_runner_path = resolve_repo_path(args.batch_runner_path)
    compare_runner_path = resolve_repo_path(args.compare_runner_path)
    nav_config_path = resolve_repo_path(args.nav_config_path)
    if not nav_config_path.is_file():
        raise FileNotFoundError(f"navigation config does not exist: {nav_config_path}")
    effective_configs, _ = load_effective_nav_configs(
        nav_config_path,
        task_modes,
    )

    if args.list_ablations:
        catalog_selected = {
            mode: list(ABLATIONS_BY_TASK[mode])
            for mode in task_modes
        }
        catalog_analysis = validate_profiles(
            catalog_selected,
            effective_configs=effective_configs,
            common_overrides=(),
        )
        validate_evidence_aggregation(
            catalog_analysis,
            batch_runner_path=batch_runner_path,
        )
        print_ablation_catalog(
            catalog_analysis,
            output_format=args.list_ablations_format,
        )
        return 0

    for label, path in (
        ("runner", runner_path),
        ("batch runner", batch_runner_path),
        ("comparison runner", compare_runner_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    selected = select_ablations(
        task_modes, args.ablations, auto_contrasts=not args.no_auto_contrasts,
    )
    batch_args = normalize_batch_args(args.batch_args)
    batch_max_steps = effective_batch_max_steps(batch_args)
    raw_common_overrides = list(args.nav_override)
    if not any(
        "=" in item and item.split("=", 1)[0].strip() == "max_steps"
        for item in raw_common_overrides
    ):
        # Record the batch runner's implicit --max-steps in the baseline
        raw_common_overrides.append(f"max_steps={batch_max_steps}")
    raw_seed_overrides = [
        item
        for item in raw_common_overrides
        if "=" in item and item.split("=", 1)[0].strip() == "seed"
    ]
    if len(raw_seed_overrides) > 1:
        raise ValueError("duplicate common --nav-override key: seed")
    nav_seed: int | None = None
    if raw_seed_overrides:
        raw_seed = raw_seed_overrides[0].split("=", 1)[1]
        parsed_seed = parse_nav_override_scalar(raw_seed)
        if isinstance(parsed_seed, bool) or not isinstance(parsed_seed, int):
            raise ValueError("--nav-override seed must be a non-negative integer")
        if parsed_seed < 0:
            raise ValueError("--nav-override seed must be a non-negative integer")
        nav_seed = parsed_seed
    if args.seed is not None and nav_seed is not None and args.seed != nav_seed:
        raise ValueError(
            f"--seed={args.seed} conflicts with --nav-override seed={nav_seed}"
        )
    navigation_seed = int(
        args.seed if args.seed is not None else nav_seed if nav_seed is not None else 0
    )
    raw_common_overrides = [
        item
        for item in raw_common_overrides
        if not ("=" in item and item.split("=", 1)[0].strip() == "seed")
    ]
    raw_common_overrides.append(f"seed={navigation_seed}")
    common_overrides = parse_common_nav_overrides(
        raw_common_overrides,
        effective_configs=effective_configs,
    )

    sweep_records = add_sensitivity_sweeps(
        args.sweep,
        task_modes=task_modes,
        selected=selected,
        effective_configs=effective_configs,
        common_overrides=common_overrides,
    )
    experiment_name = str(args.experiment_name or "").strip()
    if not experiment_name:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{timestamp}_{git_value('rev-parse', '--short', 'HEAD')}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_name):
        raise ValueError("--experiment-name must contain only letters, digits, '.', '_', or '-'")
    output_root = resolve_repo_path(args.output_root)
    experiment_root = output_root / experiment_name
    if args.resume and not args.no_auto_contrasts:
        selected = expanded_resume_selection(experiment_root, selected)
    if args.skip_full:
        # Filter after sweeps and resume so neither can restore the baseline
        selected = {
            mode: candidates
            for mode, names in selected.items()
            if (candidates := [name for name in names if name != "full"])
        }
        if not selected:
            raise ValueError(
                "--skip-full leaves no ablation profiles to run; "
                "select a non-full profile or add --sweep"
            )
    profile_analysis = validate_profiles(
        selected,
        effective_configs=effective_configs,
        common_overrides=common_overrides,
    )
    validate_evidence_aggregation(
        profile_analysis,
        batch_runner_path=batch_runner_path,
    )
    if args.report_only:
        if not experiment_root.is_dir():
            raise FileNotFoundError(
                f"experiment directory does not exist: {experiment_root}"
            )
        returncodes, _ = recorded_invocation_state(
            experiment_root
        )
        report_payload, report_failed = write_experiment_reports(
            compare_runner_path=compare_runner_path,
            experiment_root=experiment_root,
            selected=selected,
            returncodes=returncodes,
            bootstrap=int(args.bootstrap),
            statistics_seed=int(args.statistics_seed),
            navigation_seed=int(navigation_seed),
        )
        return 1 if report_failed or not report_payload["passed"] else 0
    pending_exchange_rate = uncalibrated_exchange_rate_profiles(
        selected,
        effective_configs=effective_configs,
        common_overrides=common_overrides,
    )
    if pending_exchange_rate:
        message = (
            "sgf_explore_exchange_rate is uncalibrated (<= 0) for "
            f"{', '.join(pending_exchange_rate)}; run full, then "
            "scripts/calibrate_exploration_exchange_rate.py, and set the "
            "per-task value in the navigation YAML"
        )
        if not args.dry_run:
            raise ValueError(message)
        print(f"warning: {message}", file=sys.stderr, flush=True)

    commands: list[tuple[str, str, list[str]]] = []
    for mode, names in selected.items():
        for name in names:
            profile = ABLATIONS_BY_TASK[mode][name]
            command = build_batch_command(
                python_executable=args.python_executable,
                batch_runner_path=batch_runner_path,
                runner_path=runner_path,
                nav_config_path=nav_config_path,
                mode_output_root=experiment_root / mode,
                task_mode=mode,
                name=name,
                profile=profile,
                common_overrides=common_overrides,
                batch_args=batch_args,
                resume=bool(args.resume),
                overwrite=bool(args.overwrite),
                metrics_only=bool(args.metrics_only),
            )
            commands.append((mode, name, command))

    if args.resume and not args.allow_resume_mismatch:
        validate_resume_compatibility(
            experiment_root=experiment_root,
            commands=commands,
            runner_path=runner_path,
            batch_runner_path=batch_runner_path,
            compare_runner_path=compare_runner_path,
            nav_config_path=nav_config_path,
        )

    if args.dry_run or not args.metrics_only:
        print(f"experiment_root={experiment_root}")
        print(
            f"navigation_seed={navigation_seed} "
            f"statistics_seed={args.statistics_seed} bootstrap={args.bootstrap}"
        )
        for mode, name, command in commands:
            profile = ABLATIONS_BY_TASK[mode][name]
            detail = profile_analysis[mode][name]
            print(
                f"\n[{mode}/{name}] tier={detail['tier']} family={profile.family}"
            )
            print(profile.description)
            if detail["inert_by_dependency"]:
                inert_keys = detail["inert_by_dependency"]
                preview = ",".join(inert_keys[:8])
                if len(inert_keys) > 8:
                    preview += ",..."
                print(
                    f"inert_by_dependency={len(inert_keys)} [{preview}]"
                )
            print(shlex.join(command))
    if args.dry_run:
        return 0

    supports_execution_plan, fallback_reason = (
        probe_execution_plan_capability(
            python_executable=args.python_executable,
            batch_runner_path=batch_runner_path,
        )
    )
    if supports_execution_plan:
        execution_scheduler: dict[str, Any] = {
            "strategy": "ordered_tail_fill",
            "protocol_version": EXECUTION_PLAN_VERSION,
        }
    else:
        assert fallback_reason is not None
        execution_scheduler = {
            "strategy": "serial_fallback",
            "protocol_version": None,
            "fallback_reason": fallback_reason,
        }
        print(
            "warning: batch runner execution-plan scheduling is unavailable; "
            f"using serial fallback ({fallback_reason})",
            file=sys.stderr,
            flush=True,
        )
    for field in (
        "requested_workers",
        "effective_max_workers",
        "resolved_gpu_ids",
        "max_gpu_workers_per_device",
    ):
        # Filled in by an execution-plan runner, otherwise left null
        execution_scheduler.setdefault(field, None)

    prepare_experiment_root(
        experiment_root,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
    )
    append_manifest_invocation(
        experiment_root=experiment_root,
        experiment_name=experiment_name,
        runner_path=runner_path,
        batch_runner_path=batch_runner_path,
        compare_runner_path=compare_runner_path,
        nav_config_path=nav_config_path,
        selected=selected,
        common_overrides=common_overrides,
        effective_configs=effective_configs,
        profile_analysis=profile_analysis,
        sweep_records=sweep_records,
        navigation_seed=navigation_seed,
        statistics_seed=int(args.statistics_seed),
        bootstrap=int(args.bootstrap),
        batch_args=batch_args,
        commands=[command for _, _, command in commands],
        metrics_only=bool(args.metrics_only),
        execution_scheduler=execution_scheduler,
    )
    execution_interrupted = False
    if execution_scheduler["strategy"] == "ordered_tail_fill":
        if not args.metrics_only:
            print("\n=== running shared ablation schedule ===", flush=True)
        outcome = run_shared_execution_plan(
            python_executable=args.python_executable,
            batch_runner_path=batch_runner_path,
            experiment_root=experiment_root,
            commands=commands,
            keep_going=bool(args.keep_going),
            metrics_only=bool(args.metrics_only),
        )
        execution_interrupted = outcome.interrupted
        returncodes: dict[tuple[str, str], int] = {}
        scheduler_update: dict[str, Any]
        if outcome.result_error is not None or outcome.runs is None:
            batch_failed = True
            failure_returncode = (
                outcome.wrapper_returncode
                if outcome.wrapper_returncode != 0
                else 1
            )
            error_parts = [
                part
                for part in (outcome.result_error, outcome.wrapper_error)
                if part
            ]
            wrapper_error = "; ".join(error_parts) or (
                "execution-plan wrapper did not produce a usable result"
            )
            result_runs = [
                {
                    "mode": mode,
                    "name": name,
                    "state": "failed",
                    "returncode": failure_returncode,
                    "error": wrapper_error,
                }
                for mode, name, _ in commands
            ]
            returncodes.update(
                {
                    (mode, name): failure_returncode
                    for mode, name, _ in commands
                }
            )
            scheduler_update = {
                "wrapper_returncode": outcome.wrapper_returncode,
                "result_error": outcome.result_error,
            }
            print(
                "execution-plan wrapper failed: " + wrapper_error,
                file=sys.stderr,
                flush=True,
            )
        else:
            result_runs = list(outcome.runs)
            scheduler_update = dict(outcome.scheduler or {})
            batch_failed = outcome.wrapper_returncode != 0
            if outcome.wrapper_returncode != 0:
                scheduler_update["wrapper_returncode"] = (
                    outcome.wrapper_returncode
                )
                suffix = (
                    f": {outcome.wrapper_error}"
                    if outcome.wrapper_error
                    else ""
                )
                print(
                    "execution-plan wrapper exited with return code "
                    f"{outcome.wrapper_returncode}{suffix}",
                    file=sys.stderr,
                    flush=True,
                )
            for run in result_runs:
                identity = (str(run["mode"]), str(run["name"]))
                returncode = run.get("returncode")
                if isinstance(returncode, int) and not isinstance(
                    returncode, bool
                ):
                    returncodes[identity] = returncode
                failed = run.get("state") != "completed" or returncode != 0
                if not failed:
                    continue
                batch_failed = True
                detail = str(run.get("error") or "").strip()
                suffix = f": {detail}" if detail else ""
                print(
                    f"batch failed for {identity[0]}/{identity[1]} with "
                    f"return code {returncode}{suffix}",
                    file=sys.stderr,
                    flush=True,
                )
    else:
        returncodes, batch_failed, result_runs = run_serial_batches(
            commands,
            metrics_only=bool(args.metrics_only),
            keep_going=bool(args.keep_going),
        )
        scheduler_update = {}

    update_manifest_execution_scheduler(
        experiment_root=experiment_root,
        scheduler=scheduler_update,
        runs=result_runs,
    )
    if execution_interrupted:
        raise KeyboardInterrupt

    report_payload, report_failed = write_experiment_reports(
        compare_runner_path=compare_runner_path,
        experiment_root=experiment_root,
        selected=selected,
        returncodes=returncodes,
        bootstrap=int(args.bootstrap),
        statistics_seed=int(args.statistics_seed),
        navigation_seed=int(navigation_seed),
    )
    return 1 if batch_failed or report_failed or not report_payload["passed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
