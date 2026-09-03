#!/usr/bin/env python3
"""Run the train-only P4 non-adaptive multi-view teacher screen.

The protocol is deliberately split into three durable phases.  ``candidate``
can see only the immutable label-free Pilot64 cache view.  ``outer`` may open
the train-side target mmap only after a complete candidate artifact has been
verified.  ``aggregate`` consumes complete outer records and applies the pure
scientific gate.  No phase uses validation or test payloads, and a P4 pass
never authorizes P5.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, Final
import uuid

import numpy as np
import yaml


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG: Final = PROJECT_ROOT / "configs/nonadaptive_teacher_screen_v1.yaml"
EXPECTED_PROTOCOL_ID: Final = "cr-sitta-nonadaptive-teacher-screen-v1"
DATASETS: Final = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CONDITIONS: Final = (
    ("clean", 0),
    ("gaussian_noise", 1),
    ("gaussian_noise", 3),
    ("gaussian_noise", 5),
    ("gaussian_blur", 1),
    ("gaussian_blur", 3),
    ("gaussian_blur", 5),
    ("low_contrast", 1),
    ("low_contrast", 3),
    ("low_contrast", 5),
    ("stripe_noise", 1),
    ("stripe_noise", 3),
    ("stripe_noise", 5),
)
CANDIDATE_IDS: Final = (
    "flip4_mean",
    "flip4_trimmed",
    "flip4_disagreement_weighted",
    "flip4_source_anchor_beta_0_25",
    "flip4_source_anchor_beta_0_5",
    "tile5_mean",
    "tile5_trimmed",
    "tile5_disagreement_weighted",
    "tile5_source_anchor_beta_0_25",
    "tile5_source_anchor_beta_0_5",
)
METHOD_FIELDS: Final = frozenset(
    {"image", "image_id", "original_size", "dataset", "corruption", "severity", "seed"}
)
BASE_VIEW_NAMES: Final = (
    "identity",
    "hflip",
    "vflip",
    "hvflip",
    "tile_reconstruction",
)
IMAGE_COUNT: Final = 64
IMAGE_SHAPE: Final = (3, 256, 256)
PROBABILITY_SHAPE: Final = (1, 256, 256)
THRESHOLD: Final = 0.5
RNG_STREAMS: Final = (
    "python_random",
    "numpy_random",
    "torch_cpu",
    "torch_current_cuda",
)
MODEL_RUNTIME_FINGERPRINT_FIELDS: Final = (
    "parameter_versions",
    "buffer_versions",
    "module_training",
    "module_runtime",
    "batchnorm_runtime",
    "parameter_requires_grad",
    "parameter_grad_is_none",
)


class P4ProtocolError(RuntimeError):
    """The frozen P4 protocol or one of its artifacts is invalid."""


class ExistingArtifactError(P4ProtocolError):
    """A canonical path exists but is incomplete or conflicts with this run."""


@dataclass(frozen=True, slots=True)
class P4Contract:
    repository: Path
    config_path: Path
    config_sha256: str
    raw: Mapping[str, Any]
    output_root: Path

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(str(value["id"]) for value in self.raw["candidates"])

    @property
    def conditions(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (str(value[0]), int(value[1])) for value in self.raw["ordered_conditions"]
        )


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_key(corruption: str, severity: int) -> str:
    return f"{corruption}_S{severity}"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P4ProtocolError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise P4ProtocolError(f"{label} must be a sequence")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected, key=str)
    if missing or unknown:
        raise P4ProtocolError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )
    return mapping


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise P4ProtocolError(
            f"{label} differs; expected={expected!r}, observed={actual!r}"
        )


def _boolean(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise P4ProtocolError(f"{label} must be {expected}")


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P4ProtocolError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise P4ProtocolError(f"{label} must be finite")
    return result


def _repository_path(repository: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise P4ProtocolError(f"{label} must be a non-empty relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise P4ProtocolError(f"{label} must stay inside the repository")
    return Path(os.path.abspath(repository / relative))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P4ProtocolError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise P4ProtocolError(f"JSON root must be an object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_bytes())
    except (OSError, yaml.YAMLError) as exc:
        raise P4ProtocolError(f"cannot read YAML object: {path}") from exc
    if not isinstance(value, dict):
        raise P4ProtocolError(f"YAML root must be an object: {path}")
    return value


def _validate_scope(config: Mapping[str, Any]) -> None:
    scope = _mapping(config.get("scope"), "scope")
    required = {
        "stage": "P4_nonadaptive_teacher_screen",
        "source_train_derived": True,
        "split_name": "train",
        "split_role": "frozen_pilot64",
        "no_validation_split": True,
        "use_validation_payload": False,
        "use_test_payload": False,
        "dataset_count": 3,
        "condition_count_per_dataset": 13,
        "image_count_per_condition": 64,
        "candidate_count": 10,
        "seed": 42,
        "development_only": True,
        "paper_result": False,
        "paper_test_result": False,
        "parameter_update": False,
        "p5_authorized": False,
        "p5_requires_separate_protocol": True,
    }
    for key, expected in required.items():
        _equal(scope.get(key), expected, f"scope.{key}")
    _equal(
        tuple(scope.get("allowed_payload_roles", ())),
        (
            "train_pilot64_image",
            "train_pilot64_metadata",
            "train_pilot64_outer_target",
        ),
        "scope.allowed_payload_roles",
    )
    _equal(
        tuple(scope.get("forbidden_payload_roles", ())),
        (
            "validation_image",
            "validation_target",
            "test_image",
            "test_target",
            "test_prediction",
        ),
        "scope.forbidden_payload_roles",
    )


def _validate_views_and_candidates(config: Mapping[str, Any]) -> None:
    views = _mapping(config.get("views"), "views")
    _boolean(views.get("probability_space_aggregation"), True, "views probability aggregation")
    _equal(views.get("model_mode"), "frozen_source_eval", "views.model_mode")
    _boolean(views.get("requires_grad"), False, "views.requires_grad")
    flip4 = _mapping(views.get("flip4"), "views.flip4")
    _equal(tuple(flip4.get("ordered_views", ())), BASE_VIEW_NAMES[:4], "flip4 views")
    _boolean(flip4.get("exact_inverse_alignment"), True, "flip4 inverse alignment")
    tile5 = _mapping(views.get("tile5"), "views.tile5")
    _equal(tuple(tile5.get("ordered_views", ())), BASE_VIEW_NAMES, "tile5 views")
    _boolean(
        tile5.get("exact_inverse_alignment_for_flip_views"),
        True,
        "tile5 flip inverse alignment",
    )
    tile = _mapping(views.get("tile_reconstruction"), "views.tile_reconstruction")
    _equal(tile.get("crop_size"), 224, "tile crop size")
    _equal(
        tuple(tuple(int(part) for part in value) for value in tile.get("ordered_origins", ())),
        ((0, 0), (0, 32), (32, 0), (32, 32)),
        "tile origins",
    )
    _equal(tuple(tile.get("canvas_size", ())), (256, 256), "tile canvas")
    _equal(tuple(tile.get("model_input_size", ())), (256, 256), "tile model input")
    _equal(tile.get("overlap_reduction"), "arithmetic_mean_over_valid_predictions", "tile overlap")
    _boolean(tile.get("complete_canvas_coverage_required"), True, "tile coverage")
    for name in ("input_resize", "inverse_probability_resize"):
        resize = _mapping(tile.get(name), f"tile.{name}")
        _equal(resize.get("mode"), "bilinear", f"tile.{name}.mode")
        _boolean(resize.get("align_corners"), False, f"tile.{name}.align_corners")
    for disabled in ("rotation", "scale"):
        entry = _mapping(views.get(disabled), f"views.{disabled}")
        _boolean(entry.get("enabled"), False, f"views.{disabled}.enabled")
        _boolean(entry.get("fail_closed"), True, f"views.{disabled}.fail_closed")

    aggregation = _mapping(config.get("aggregation"), "aggregation")
    _equal(aggregation.get("trim_count_each_tail"), 1, "aggregation trim")
    _equal(_finite_float(aggregation.get("disagreement_tau"), "aggregation tau"), 0.01, "aggregation tau")
    _equal(
        aggregation.get("disagreement_weight_formula"),
        "exp(-(p_m-minus-view_mean)^2/tau)",
        "aggregation disagreement formula",
    )
    _equal(
        aggregation.get("source_anchor_inner_aggregation"),
        "mean",
        "source-anchor inner aggregation",
    )
    _equal(tuple(float(v) for v in aggregation.get("source_anchor_betas", ())), (0.25, 0.5), "anchor betas")
    _equal(tuple(float(v) for v in aggregation.get("probability_bounds", ())), (0.0, 1.0), "probability bounds")
    _boolean(aggregation.get("finite_required"), True, "aggregation finite")

    candidates = tuple(_sequence(config.get("candidates"), "candidates"))
    _equal(tuple(str(value.get("id")) for value in candidates), CANDIDATE_IDS, "candidate IDs")
    expected = (
        ("flip4", "mean", None),
        ("flip4", "trimmed", None),
        ("flip4", "disagreement_weighted", None),
        ("flip4", "source_anchor", 0.25),
        ("flip4", "source_anchor", 0.5),
        ("tile5", "mean", None),
        ("tile5", "trimmed", None),
        ("tile5", "disagreement_weighted", None),
        ("tile5", "source_anchor", 0.25),
        ("tile5", "source_anchor", 0.5),
    )
    for index, (raw, frozen) in enumerate(zip(candidates, expected, strict=True)):
        item = _mapping(raw, f"candidates[{index}]")
        observed = (
            item.get("view_set"),
            item.get("aggregation"),
            None if "beta" not in item else float(item["beta"]),
        )
        _equal(observed, frozen, f"candidates[{index}] recipe")


def _validate_evaluation_and_gate(config: Mapping[str, Any]) -> None:
    evaluation = _mapping(config.get("evaluation"), "evaluation")
    _equal(_finite_float(evaluation.get("probability_threshold"), "threshold"), 0.5, "threshold")
    _equal(evaluation.get("threshold_rule"), "strict_greater_than", "threshold rule")
    _equal(evaluation.get("foreground_connectivity_2d"), 8, "connectivity")
    _equal(evaluation.get("min_component_area"), 1, "minimum component area")
    matching = _mapping(evaluation.get("target_matching"), "target matching")
    _equal(matching.get("assignment"), "hungarian_minimum_centroid_distance", "matching assignment")
    _equal(matching.get("comparison"), "strict_less_than", "matching comparison")
    _equal(float(matching.get("max_centroid_distance_pixels")), 3.0, "matching distance")
    _boolean(matching.get("one_to_one"), True, "matching one-to-one")
    _equal(
        tuple(float(value) for value in evaluation.get("froc_probability_thresholds", ())),
        tuple(index / 20.0 for index in range(21)),
        "FROC thresholds",
    )

    gate = _mapping(config.get("science_gate"), "science gate")
    required_numbers = {
        "comparison_tolerance": 1.0e-12,
        "nonclean_36_macro_delta_iou_strictly_greater_than": 0.001,
        "overall_39_macro_delta_iou_strictly_greater_than": 0.0,
        "minimum_positive_nonclean_datasets": 2.0,
        "worst_nonclean_dataset_delta_iou_minimum": -0.002,
        "clean_macro_delta_iou_minimum": -0.002,
        "each_clean_dataset_delta_iou_minimum": -0.005,
        "nonclean_macro_delta_pd_minimum": -0.01,
        "each_nonclean_dataset_delta_pd_minimum": -0.02,
    }
    for key, expected in required_numbers.items():
        _equal(_finite_float(gate.get(key), f"science_gate.{key}"), expected, f"science_gate.{key}")
    _boolean(gate.get("filter_before_ranking"), True, "gate filter-before-rank")
    _boolean(gate.get("no_eligible_is_normal_scientific_result"), True, "gate no eligible")
    _equal(gate.get("positive_corruption_family_requirement"), None, "family-positive gate")
    _boolean(gate.get("p5_authorized_on_pass"), False, "gate P5 authorization")
    _boolean(gate.get("p5_requires_separate_protocol"), True, "gate P5 separate protocol")
    _equal(
        tuple(gate.get("ranking_tie_break_order", ())),
        (
            "higher_nonclean_macro_delta_iou",
            "higher_overall_macro_delta_iou",
            "higher_worst_dataset_delta_iou",
            "lower_nonclean_fa_delta",
            "lexical_candidate_id",
        ),
        "gate ranking order",
    )
    fa = _mapping(gate.get("fa_delta_maximum_formula"), "gate Fa formula")
    _equal(fa.get("units"), "per_million_pixels", "Fa units")
    _equal(float(fa.get("absolute_allowance")), 10.0, "Fa allowance")
    _equal(float(fa.get("source_multiplier")), 0.25, "Fa multiplier")
    _equal(
        tuple(fa.get("applies_to", ())),
        ("overall", "nonclean", "clean", "each_dataset", "each_corruption_family"),
        "Fa strata",
    )
    foreground = _mapping(gate.get("foreground_fraction"), "gate foreground")
    _equal(float(foreground.get("delta_maximum")), 0.001, "foreground delta")
    _equal(float(foreground.get("teacher_to_source_multiplier")), 1.2, "foreground multiplier")
    _equal(float(foreground.get("epsilon")), 1.0e-6, "foreground epsilon")


def _validate_firewall_and_output(config: Mapping[str, Any]) -> None:
    firewall = _mapping(config.get("phase_firewall"), "phase firewall")
    candidate = _mapping(firewall.get("candidate"), "candidate firewall")
    _equal(candidate.get("allowed_loader"), "SourceCalibrationMethodInputDatasetV2", "candidate loader")
    for key in (
        "target_loader_calls",
        "method_label_accesses",
        "validation_payload_opens",
        "test_payload_opens",
    ):
        _equal(candidate.get(key), 0, f"candidate firewall {key}")
    _boolean(candidate.get("model_state_change_allowed"), False, "candidate state change")
    _boolean(candidate.get("completion_receipt_required"), True, "candidate receipt")
    outer = _mapping(firewall.get("outer"), "outer firewall")
    _boolean(outer.get("candidate_complete_required_before_target_load"), True, "outer completion gate")
    _equal(outer.get("allowed_target_role"), "train_pilot64_outer_target", "outer target role")
    _boolean(outer.get("targets_visible_to_candidate"), False, "outer target visibility")
    _equal(outer.get("validation_payload_opens"), 0, "outer validation access")
    _equal(outer.get("test_payload_opens"), 0, "outer test access")
    aggregate = _mapping(firewall.get("aggregate"), "aggregate firewall")
    _equal(aggregate.get("required_candidate_dataset_artifacts"), 3, "aggregate candidate artifacts")
    _equal(aggregate.get("required_outer_dataset_artifacts"), 3, "aggregate outer artifacts")
    _equal(aggregate.get("required_cell_records_per_candidate"), 39, "aggregate cells")
    _equal(aggregate.get("missing_or_duplicate_cell_action"), "fail_closed", "aggregate missing action")

    output = _mapping(config.get("output"), "output")
    _equal(output.get("root"), "results/cr_sitta/nonadaptive_teacher_screen_v1", "output root")
    _equal(output.get("candidate_phase"), "candidate_phase", "candidate output")
    _equal(output.get("outer_phase"), "outer_phase", "outer output")
    _equal(output.get("aggregate_phase"), "aggregate_phase/R0", "aggregate output")
    _equal(output.get("engineering_phase"), "engineering_dry_runs/candidate_phase", "engineering output")
    _boolean(output.get("atomic_no_replace"), True, "atomic no-replace")
    _boolean(output.get("refuse_overwrite"), True, "refuse overwrite")
    _equal(output.get("existing_verified_complete_action"), "no_op", "existing complete action")
    _equal(output.get("existing_incomplete_or_conflicting_action"), "fail_closed", "existing conflict action")


def _validate_frozen_files(
    repository: Path,
    config: Mapping[str, Any],
    *,
    verify_files: bool,
) -> None:
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    for name, raw in frozen.items():
        record = _mapping(raw, f"frozen_inputs.{name}")
        path = _repository_path(repository, record.get("path"), f"frozen_inputs.{name}.path")
        expected = record.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise P4ProtocolError(f"frozen_inputs.{name}.sha256 is invalid")
        if verify_files:
            if path.is_symlink() or not path.is_file():
                raise P4ProtocolError(f"frozen input is missing: {path}")
            _equal(sha256_file(path), expected, f"frozen input hash {name}")

    cache = _mapping(config.get("cache"), "cache")
    _equal(cache.get("protocol_sha256"), frozen["cache_protocol"]["sha256"], "cache protocol hash")
    _equal(cache.get("target_relative_path"), "outer_evaluator/targets.npy", "cache target path")
    _equal(
        tuple(cache.get("method_facing_fields", ())),
        ("image", "image_id", "original_size", "dataset", "corruption", "severity", "seed"),
        "method-facing fields",
    )
    _equal(cache.get("method_facing_targets"), "forbidden", "method-facing target")

    datasets = _mapping(config.get("datasets"), "datasets")
    _equal(tuple(datasets), DATASETS, "dataset order")
    for dataset in DATASETS:
        record = _mapping(datasets[dataset], f"datasets.{dataset}")
        _equal(record.get("checkpoint_role"), "best_miou", f"{dataset} checkpoint role")
        for field in (
            "cache_manifest_sha256",
            "cache_content_sha256",
            "method_input_manifest_sha256",
            "checkpoint_sha256",
        ):
            value = record.get(field)
            if not isinstance(value, str) or len(value) != 64:
                raise P4ProtocolError(f"datasets.{dataset}.{field} is invalid")
        if not verify_files:
            continue
        cache_root = _repository_path(repository, record.get("cache_root"), f"{dataset} cache root")
        checkpoint = _repository_path(repository, record.get("checkpoint_path"), f"{dataset} checkpoint")
        if cache_root.is_symlink() or not cache_root.is_dir():
            raise P4ProtocolError(f"cache root is missing: {cache_root}")
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise P4ProtocolError(f"checkpoint is missing: {checkpoint}")
        _equal(sha256_file(checkpoint), record["checkpoint_sha256"], f"{dataset} checkpoint hash")
        manifest_path = cache_root / "manifest.json"
        method_path = cache_root / "method_input_manifest.json"
        complete_path = cache_root / "COMPLETE.json"
        _equal(sha256_file(manifest_path), record["cache_manifest_sha256"], f"{dataset} cache manifest")
        _equal(sha256_file(method_path), record["method_input_manifest_sha256"], f"{dataset} method manifest")
        complete = _load_json(complete_path)
        _boolean(complete.get("complete"), True, f"{dataset} cache complete")
        _boolean(complete.get("method_received_labels"), False, f"{dataset} method labels")
        _equal(complete.get("test_images_opened"), 0, f"{dataset} test image opens")
        _equal(complete.get("test_masks_opened"), 0, f"{dataset} test mask opens")
        _equal(complete.get("manifest_sha256"), record["cache_manifest_sha256"], f"{dataset} complete manifest")
        _equal(complete.get("method_input_manifest_sha256"), record["method_input_manifest_sha256"], f"{dataset} complete method manifest")
        _equal(complete.get("cache_content_sha256"), record["cache_content_sha256"], f"{dataset} cache content")

    code_paths = tuple(_sequence(config.get("critical_code_paths"), "critical code paths"))
    if len(code_paths) != len(set(code_paths)) or not code_paths:
        raise P4ProtocolError("critical code paths must be unique and non-empty")
    if verify_files:
        missing = [
            str(path)
            for raw in code_paths
            for path in (_repository_path(repository, raw, "critical code path"),)
            if path.is_symlink() or not path.is_file()
        ]
        if missing:
            raise P4ProtocolError(f"critical code files are missing: {missing}")


def _validate_stage_transition(
    repository: Path,
    config: Mapping[str, Any],
    *,
    verify_files: bool,
) -> None:
    transition = _mapping(config.get("stage_transition"), "stage transition")
    expected = {
        "predecessor_stage": "P3_formal_tent_failure_diagnostics_stage_a",
        "predecessor_formal_protocol_complete": True,
        "predecessor_scientific_status": "scientific_no_eligible",
        "p4_nonadaptive_teacher_screen_authorized": True,
        "authorized_method_class": "nonadaptive_no_parameter_update",
        "parameter_update_authorized": False,
        "stage2_authorized": False,
        "p5_authorized": False,
    }
    _exact_keys(transition, set(expected), "stage transition")
    for key, value in expected.items():
        _equal(transition.get(key), value, f"stage_transition.{key}")
    if not verify_files:
        return
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    receipt_record = _mapping(
        frozen.get("p3_science_decision_receipt"),
        "frozen P3 science receipt",
    )
    complete_record = _mapping(
        frozen.get("p3_aggregate_complete"),
        "frozen P3 completion",
    )
    receipt = _load_json(
        _repository_path(repository, receipt_record.get("path"), "P3 science receipt")
    )
    complete = _load_json(
        _repository_path(repository, complete_record.get("path"), "P3 completion")
    )
    _equal(
        receipt.get("scientific_status"),
        "scientific_no_eligible",
        "P3 scientific status",
    )
    _boolean(
        receipt.get("formal_stage_a_protocol_complete"),
        True,
        "P3 formal completion",
    )
    _equal(receipt.get("eligible_candidates"), [], "P3 eligible candidates")
    _boolean(receipt.get("stage2_allowed"), False, "P3 Stage2 authorization")
    _boolean(complete.get("complete"), True, "P3 aggregate completion")
    _boolean(complete.get("stage2_authorized"), False, "P3 complete Stage2 authorization")
    _equal(
        complete.get("config_sha256"),
        config["frozen_inputs"]["p3_formal_config"]["sha256"],
        "P3 completion config hash",
    )
    science = _mapping(complete.get("science_decision"), "P3 completion science decision")
    _equal(
        science.get("sha256"),
        receipt_record.get("sha256"),
        "P3 completion science receipt hash",
    )
    _equal(
        science.get("scientific_status"),
        "scientific_no_eligible",
        "P3 completion scientific status",
    )


def load_contract(
    path: str | Path = DEFAULT_CONFIG,
    *,
    repository: str | Path = PROJECT_ROOT,
    verify_files: bool = True,
) -> P4Contract:
    repository_path = Path(os.path.abspath(os.fspath(repository)))
    config_path = Path(os.path.abspath(os.fspath(path)))
    config = _load_yaml(config_path)
    _equal(config.get("schema_version"), 1, "schema version")
    _equal(config.get("protocol_id"), EXPECTED_PROTOCOL_ID, "protocol ID")
    _equal(config.get("created_at"), "2026-09-03", "created_at")
    _validate_scope(config)
    _equal(
        tuple((str(value[0]), int(value[1])) for value in config.get("ordered_conditions", ())),
        CONDITIONS,
        "ordered conditions",
    )
    _validate_views_and_candidates(config)
    _validate_evaluation_and_gate(config)
    _validate_firewall_and_output(config)
    _validate_frozen_files(repository_path, config, verify_files=verify_files)
    _validate_stage_transition(repository_path, config, verify_files=verify_files)
    output_root = _repository_path(repository_path, config["output"]["root"], "output root")
    return P4Contract(
        repository=repository_path,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        raw=config,
        output_root=output_root,
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, _canonical_json_bytes(value))


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(_canonical_json_bytes(dict(value)) for value in values)
    _write_bytes(path, payload)


def _safe_relative(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise P4ProtocolError(f"artifact member escaped staging root: {path}") from exc
    if not relative.parts or ".." in relative.parts:
        raise P4ProtocolError(f"unsafe artifact member: {path}")
    return relative.as_posix()


def _file_ledger(root: Path, *, excluded: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    excluded_set = set(excluded)
    ledger: dict[str, dict[str, Any]] = {}
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in tuple(directory_names):
            candidate = current_path / name
            if candidate.is_symlink():
                raise P4ProtocolError(f"artifact contains a symlink directory: {candidate}")
        for name in file_names:
            path = current_path / name
            relative = _safe_relative(path, root)
            if relative in excluded_set:
                continue
            if path.is_symlink() or not path.is_file():
                raise P4ProtocolError(f"artifact member is not a regular file: {path}")
            ledger[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return dict(sorted(ledger.items()))


def _tree_sha256(ledger: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(dict(ledger)))


def _capture_code_hashes(contract: P4Contract) -> dict[str, str]:
    return {
        str(raw): sha256_file(_repository_path(contract.repository, raw, "critical code"))
        for raw in contract.raw["critical_code_paths"]
    }


def _stage_transition_lineage(contract: P4Contract) -> dict[str, Any]:
    frozen = contract.raw["frozen_inputs"]
    transition = contract.raw["stage_transition"]
    return {
        "predecessor_stage": transition["predecessor_stage"],
        "predecessor_formal_config": dict(frozen["p3_formal_config"]),
        "predecessor_science_decision_receipt": dict(
            frozen["p3_science_decision_receipt"]
        ),
        "predecessor_aggregate_complete": dict(frozen["p3_aggregate_complete"]),
        "predecessor_formal_protocol_complete": True,
        "predecessor_scientific_status": "scientific_no_eligible",
        "p4_nonadaptive_teacher_screen_authorized": True,
        "authorized_method_class": "nonadaptive_no_parameter_update",
        "parameter_update_authorized": False,
        "stage2_authorized": False,
        "p5_authorized": False,
    }


def _artifact_destination(contract: P4Contract, phase: str, dataset: str | None = None) -> Path:
    output = contract.raw["output"]
    if phase == "candidate":
        assert dataset is not None
        return contract.output_root / output["candidate_phase"] / dataset
    if phase == "outer":
        assert dataset is not None
        return contract.output_root / output["outer_phase"] / dataset
    if phase == "aggregate":
        if dataset is not None:
            raise ValueError("aggregate artifact has no dataset")
        return contract.output_root / output["aggregate_phase"]
    raise ValueError(f"unknown artifact phase: {phase}")


def _verify_file_ledger(root: Path, expected: Mapping[str, Any]) -> None:
    observed = _file_ledger(root, excluded=("manifest.json", "COMPLETE.json"))
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            key for key in set(expected) & set(observed) if expected[key] != observed[key]
        )
        raise ExistingArtifactError(
            f"artifact payload differs; missing={missing}, extra={extra}, changed={changed}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise P4ProtocolError(f"{label} must be a lowercase SHA256 hex digest")
    return value


def _verify_rng_snapshot(
    value: Any,
    *,
    label: str,
    require_cuda: bool,
) -> dict[str, str | None]:
    snapshot = _exact_keys(value, set(RNG_STREAMS), label)
    result: dict[str, str | None] = {}
    for stream in RNG_STREAMS:
        observed = snapshot[stream]
        if stream == "torch_current_cuda" and not require_cuda:
            _equal(observed, None, f"{label}.{stream}")
            result[stream] = None
            continue
        result[stream] = _require_sha256(observed, f"{label}.{stream}")
    return result


def _verify_formal_candidate_semantics(
    root: Path,
    manifest: Mapping[str, Any],
    contract: P4Contract,
) -> None:
    """Re-derive the formal candidate's state/RNG and per-image evidence contract."""

    _boolean(manifest.get("development_only"), True, "formal candidate development scope")
    _boolean(manifest.get("paper_result"), False, "formal candidate paper scope")
    _boolean(manifest.get("p5_authorized"), False, "formal candidate P5 scope")
    _equal(manifest.get("condition_count"), len(CONDITIONS), "formal condition count")
    _equal(manifest.get("image_count_per_condition"), IMAGE_COUNT, "formal image count")
    _equal(manifest.get("candidate_count"), len(CANDIDATE_IDS), "formal candidate count")
    _equal(tuple(manifest.get("candidate_ids", ())), contract.candidate_ids, "formal candidates")
    _equal(tuple(manifest.get("base_view_names", ())), BASE_VIEW_NAMES, "formal base views")

    execution = _mapping(manifest.get("execution"), "formal candidate execution")
    for required in (
        "seed",
        "requires_grad",
        "optimizer_present",
        "model_fully_eval",
        "state_sha256_before",
        "state_sha256_after",
        "state_bit_exact",
        "per_image_lightweight_state_gate",
        "per_image_rng_state_gate",
        "rng_state_sha256_before",
        "rng_state_sha256_after",
        "rng_state_unchanged",
        "strict_threshold",
        "runtime_environment",
    ):
        if required not in execution:
            raise P4ProtocolError(f"formal candidate execution is missing {required}")
    _equal(execution["seed"], 42, "formal execution seed")
    _boolean(execution["requires_grad"], False, "formal execution requires_grad")
    _boolean(execution["optimizer_present"], False, "formal execution optimizer")
    _boolean(execution["model_fully_eval"], True, "formal execution eval mode")
    state_before = _require_sha256(
        execution["state_sha256_before"], "formal state before"
    )
    state_after = _require_sha256(
        execution["state_sha256_after"], "formal state after"
    )
    _equal(state_after, state_before, "formal source-model state before/after")
    _boolean(execution["state_bit_exact"], True, "formal state bit-exact proof")
    _equal(execution["strict_threshold"], "probability > 0.5", "formal threshold")

    runtime = _mapping(execution["runtime_environment"], "formal runtime environment")
    for required in (
        "python",
        "torch",
        "cuda_runtime",
        "cudnn",
        "device",
        "gpu_name",
        "sfs_extension_module",
        "sfs_extension_file",
        "sfs_extension_sha256",
    ):
        if required not in runtime:
            raise P4ProtocolError(f"formal runtime environment is missing {required}")
    if not isinstance(runtime["device"], str) or not runtime["device"].startswith("cuda"):
        raise P4ProtocolError("formal candidate runtime device must be CUDA")
    if not isinstance(runtime["gpu_name"], str) or not runtime["gpu_name"]:
        raise P4ProtocolError("formal candidate runtime GPU name is missing")
    _equal(
        runtime["sfs_extension_module"],
        "MultiScaleDeformableAttention",
        "formal SFS extension module",
    )
    _require_sha256(runtime["sfs_extension_sha256"], "formal SFS extension hash")

    dataset_rng_before = _verify_rng_snapshot(
        execution["rng_state_sha256_before"],
        label="formal dataset RNG before",
        require_cuda=True,
    )
    dataset_rng_after = _verify_rng_snapshot(
        execution["rng_state_sha256_after"],
        label="formal dataset RNG after",
        require_cuda=True,
    )
    _equal(dataset_rng_after, dataset_rng_before, "formal dataset RNG before/after")
    _boolean(execution["rng_state_unchanged"], True, "formal dataset RNG proof")

    expected_checks = len(CONDITIONS) * IMAGE_COUNT
    model_gate = _mapping(
        execution["per_image_lightweight_state_gate"],
        "formal per-image model-state gate",
    )
    _equal(model_gate.get("checks"), expected_checks, "formal model-gate checks")
    _equal(
        model_gate.get("expected_checks"),
        expected_checks,
        "formal expected model-gate checks",
    )
    _boolean(model_gate.get("passed"), True, "formal model-state gate")
    baseline_fingerprint = _require_sha256(
        model_gate.get("baseline_fingerprint_sha256"),
        "formal model runtime fingerprint",
    )
    _equal(
        tuple(model_gate.get("fingerprint_fields", ())),
        MODEL_RUNTIME_FINGERPRINT_FIELDS,
        "formal model runtime fingerprint fields",
    )
    for proof in (
        "parameter_versions_unchanged",
        "buffer_versions_unchanged",
        "module_training_unchanged",
        "module_type_and_training_unchanged",
        "batchnorm_runtime_attributes_unchanged",
        "parameter_requires_grad_unchanged",
        "parameter_grad_is_none_unchanged",
    ):
        _boolean(model_gate.get(proof), True, f"formal model-gate proof {proof}")
    for count in ("module_count", "batchnorm_module_count", "parameter_count", "buffer_count"):
        observed = model_gate.get(count)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 1:
            raise P4ProtocolError(f"formal model-gate {count} must be a positive integer")

    rng_gate = _mapping(
        execution["per_image_rng_state_gate"], "formal per-image RNG gate"
    )
    _equal(rng_gate.get("checks"), expected_checks, "formal per-image RNG checks")
    _equal(
        rng_gate.get("expected_checks"),
        expected_checks,
        "formal expected per-image RNG checks",
    )
    _equal(tuple(rng_gate.get("streams", ())), RNG_STREAMS, "formal RNG streams")
    _equal(
        rng_gate.get("evidence_location"),
        "conditions/*/per_image.jsonl",
        "formal RNG evidence location",
    )
    _boolean(rng_gate.get("passed"), True, "formal per-image RNG gate")

    boundary = _exact_keys(
        manifest.get("method_boundary"),
        {
            "loader",
            "fields",
            "target_loader_calls",
            "method_label_accesses",
            "validation_payload_opens",
            "test_payload_opens",
        },
        "formal candidate method firewall",
    )
    _equal(boundary["loader"], "SourceCalibrationMethodInputDatasetV2", "formal loader")
    _equal(tuple(boundary["fields"]), tuple(sorted(METHOD_FIELDS)), "formal method fields")
    for counter in (
        "target_loader_calls",
        "method_label_accesses",
        "validation_payload_opens",
        "test_payload_opens",
    ):
        _equal(boundary[counter], 0, f"formal method firewall {counter}")

    image_ids = tuple(str(value) for value in manifest.get("image_ids", ()))
    if len(image_ids) != IMAGE_COUNT or len(set(image_ids)) != IMAGE_COUNT or any(
        not value for value in image_ids
    ):
        raise P4ProtocolError("formal candidate image IDs must be 64 unique non-empty values")
    conditions = tuple(_sequence(manifest.get("conditions"), "formal conditions"))
    _equal(len(conditions), len(CONDITIONS), "formal condition records")
    file_ledger = _mapping(manifest.get("files"), "formal artifact files")
    total_records = 0
    for index, (corruption, severity) in enumerate(CONDITIONS):
        key = _condition_key(corruption, severity)
        condition = _mapping(conditions[index], f"formal condition {key}")
        _equal(condition.get("condition"), key, f"formal condition key {key}")
        _equal(condition.get("corruption"), corruption, f"formal corruption {key}")
        _equal(condition.get("severity"), severity, f"formal severity {key}")
        _equal(condition.get("image_count"), IMAGE_COUNT, f"formal image count {key}")
        _require_sha256(
            condition.get("input_tensor_sequence_sha256"),
            f"formal input sequence {key}",
        )
        relative = f"conditions/{key}/per_image.jsonl"
        _equal(condition.get("per_image_path"), relative, f"formal per-image path {key}")
        if relative not in file_ledger:
            raise P4ProtocolError(f"formal per-image evidence is absent from ledger: {key}")
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise P4ProtocolError(f"formal per-image evidence is unsafe: {path}")
        records = _read_jsonl(path)
        _equal(len(records), IMAGE_COUNT, f"formal per-image record count {key}")
        observed_ids: list[str] = []
        for image_index, record in enumerate(records):
            for field in record:
                lowered = str(field).lower()
                if "target" in lowered or "label" in lowered:
                    raise P4ProtocolError(
                        f"formal candidate per-image evidence exposes forbidden field {field}"
                    )
            _equal(record.get("index"), image_index, f"formal image index {key}")
            image_id = str(record.get("image_id", ""))
            observed_ids.append(image_id)
            _equal(image_id, image_ids[image_index], f"formal image ID order {key}")
            _equal(record.get("dataset"), manifest.get("dataset"), f"formal dataset {key}")
            _equal(record.get("corruption"), corruption, f"formal record corruption {key}")
            _equal(record.get("severity"), severity, f"formal record severity {key}")
            _equal(record.get("seed"), 42, f"formal record seed {key}")
            model_before = _require_sha256(
                record.get("model_runtime_fingerprint_sha256_before"),
                f"formal per-image model fingerprint before {key}/{image_index}",
            )
            model_after = _require_sha256(
                record.get("model_runtime_fingerprint_sha256_after"),
                f"formal per-image model fingerprint after {key}/{image_index}",
            )
            _equal(model_before, baseline_fingerprint, "formal per-image model baseline")
            _equal(model_after, model_before, "formal per-image model before/after")
            _boolean(
                record.get("model_runtime_fingerprint_unchanged"),
                True,
                f"formal per-image model proof {key}/{image_index}",
            )
            rng_before = _verify_rng_snapshot(
                record.get("rng_state_sha256_before"),
                label=f"formal per-image RNG before {key}/{image_index}",
                require_cuda=True,
            )
            rng_after = _verify_rng_snapshot(
                record.get("rng_state_sha256_after"),
                label=f"formal per-image RNG after {key}/{image_index}",
                require_cuda=True,
            )
            _equal(rng_after, rng_before, "formal per-image RNG before/after")
            _boolean(
                record.get("rng_state_unchanged"),
                True,
                f"formal per-image RNG proof {key}/{image_index}",
            )
            _equal(
                _finite_float(
                    record.get("strict_probability_threshold"),
                    f"formal threshold {key}/{image_index}",
                ),
                THRESHOLD,
                f"formal threshold {key}/{image_index}",
            )
            _equal(
                record.get("threshold_rule"),
                "strict_greater_than",
                f"formal threshold rule {key}/{image_index}",
            )
        if tuple(observed_ids) != image_ids or len(set(observed_ids)) != IMAGE_COUNT:
            raise P4ProtocolError(f"formal per-image IDs differ or repeat: {key}")
        total_records += len(records)
    _equal(total_records, expected_checks, "formal total per-image evidence count")


def verify_artifact(
    root: str | Path,
    *,
    contract: P4Contract,
    phase: str,
    dataset: str | None,
) -> dict[str, Any]:
    path = Path(os.path.abspath(os.fspath(root)))
    if path.is_symlink() or not path.is_dir():
        raise ExistingArtifactError(f"artifact root is absent or unsafe: {path}")
    manifest_path = path / "manifest.json"
    complete_path = path / "COMPLETE.json"
    if manifest_path.is_symlink() or complete_path.is_symlink():
        raise ExistingArtifactError("artifact manifest/completion cannot be symlinks")
    if not manifest_path.is_file() or not complete_path.is_file():
        raise ExistingArtifactError(f"artifact is incomplete: {path}")
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    _boolean(complete.get("complete"), True, "artifact complete")
    _equal(manifest.get("protocol_id"), EXPECTED_PROTOCOL_ID, "artifact protocol")
    _equal(manifest.get("config_sha256"), contract.config_sha256, "artifact config")
    _equal(manifest.get("phase"), phase, "artifact phase")
    _equal(manifest.get("dataset"), dataset, "artifact dataset")
    _equal(
        manifest.get("lineage"),
        _stage_transition_lineage(contract),
        "artifact stage-transition lineage",
    )
    recorded_code_hashes = dict(
        _mapping(manifest.get("code_sha256"), "artifact code hashes")
    )
    current_code_hashes = _capture_code_hashes(contract)
    _equal(
        recorded_code_hashes,
        current_code_hashes,
        "artifact code hashes against the current implementation",
    )
    _equal(complete.get("phase"), phase, "completion phase")
    _equal(complete.get("dataset"), dataset, "completion dataset")
    manifest_sha = sha256_file(manifest_path)
    _equal(complete.get("manifest_sha256"), manifest_sha, "completion manifest hash")
    files = _mapping(manifest.get("files"), "artifact files")
    _equal(manifest.get("payload_tree_sha256"), _tree_sha256(files), "payload tree hash")
    _verify_file_ledger(path, files)
    if phase == "candidate":
        formal = manifest.get("formal")
        if type(formal) is not bool:
            raise P4ProtocolError("candidate artifact formal flag must be boolean")
        if formal:
            _verify_formal_candidate_semantics(path, manifest, contract)
    result = dict(manifest)
    result["_manifest_sha256"] = manifest_sha
    result["_complete_sha256"] = sha256_file(complete_path)
    return result


def _prepare_parent(path: Path) -> None:
    from tta.d0_secure_io import ensure_directory_chain_nofollow

    absolute = Path(os.path.abspath(path))
    missing: list[str] = []
    anchor = absolute
    while not anchor.exists() and not anchor.is_symlink():
        missing.append(anchor.name)
        anchor = anchor.parent
    if anchor.is_symlink() or not anchor.is_dir():
        raise P4ProtocolError(f"artifact parent contains an unsafe component: {anchor}")
    ensure_directory_chain_nofollow(anchor, tuple(reversed(missing)))


def _publish_artifact(
    staging: Path,
    destination: Path,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    files = _file_ledger(staging)
    final_manifest = {
        **dict(manifest),
        "files": files,
        "payload_tree_sha256": _tree_sha256(files),
    }
    _write_json(staging / "manifest.json", final_manifest)
    manifest_sha = sha256_file(staging / "manifest.json")
    _write_json(
        staging / "COMPLETE.json",
        {
            "schema_version": 1,
            "artifact_type": "cr_sitta_nonadaptive_teacher_screen_completion",
            "complete": True,
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "phase": final_manifest["phase"],
            "dataset": final_manifest.get("dataset"),
            "manifest_sha256": manifest_sha,
            "atomic_no_replace": True,
            "development_only": True,
            "paper_result": False,
            "p5_authorized": False,
        },
    )
    from tta.d0_secure_io import publish_directory_noreplace

    publish_directory_noreplace(staging, destination)
    return final_manifest


def _existing_complete_or_raise(
    destination: Path,
    *,
    contract: P4Contract,
    phase: str,
    dataset: str | None,
) -> dict[str, Any] | None:
    if not destination.exists() and not destination.is_symlink():
        return None
    try:
        return verify_artifact(
            destination,
            contract=contract,
            phase=phase,
            dataset=dataset,
        )
    except Exception as exc:
        raise ExistingArtifactError(
            f"refusing to overwrite incomplete or conflicting {phase} artifact: {destination}"
        ) from exc


def _new_staging(destination: Path) -> Path:
    _prepare_parent(destination.parent)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir(mode=0o700, exist_ok=False)
    return staging


def _remove_private_staging(path: Path) -> None:
    if path.name.startswith(".") and path.name.endswith(".staging") and path.exists():
        shutil.rmtree(path)


def _raw_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _candidate_recipe(config: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    matching = [value for value in config["candidates"] if value["id"] == candidate_id]
    if len(matching) != 1:
        raise P4ProtocolError(f"candidate recipe is not unique: {candidate_id}")
    return matching[0]


def _aggregate_candidates(base_probabilities: Any, contract: P4Contract) -> tuple[Any, Any, Any]:
    """Return source, two uncertainty maps, and ten candidates from one 5-view stack."""

    import torch
    from tta.proposals.source_multiview_teacher import aggregate_aligned_probabilities

    if not isinstance(base_probabilities, torch.Tensor):
        raise P4ProtocolError("teacher core returned a non-tensor view stack")
    if tuple(base_probabilities.shape) != (5, 1, 1, 256, 256):
        raise P4ProtocolError(
            f"teacher core returned an unexpected view stack: {tuple(base_probabilities.shape)}"
        )
    flip4 = base_probabilities[:4]
    tile5 = base_probabilities
    uncertainties = torch.stack(
        (
            flip4.var(dim=0, unbiased=False),
            tile5.var(dim=0, unbiased=False),
        ),
        dim=0,
    )
    aggregation = contract.raw["aggregation"]
    method_names = {
        "mean": "mean",
        "trimmed": "trimmed_mean",
        "disagreement_weighted": "disagreement_weighted_mean",
        "source_anchor": "source_anchor",
    }
    candidate_values = []
    for candidate_id in contract.candidate_ids:
        recipe = _candidate_recipe(contract.raw, candidate_id)
        stack = flip4 if recipe["view_set"] == "flip4" else tile5
        kwargs: dict[str, Any] = {
            "trim_each_side": int(aggregation["trim_count_each_tail"]),
            "tau": float(aggregation["disagreement_tau"]),
        }
        if recipe["aggregation"] == "source_anchor":
            kwargs["beta"] = float(recipe["beta"])
            kwargs["source_anchor_aggregate"] = "mean"
        candidate_values.append(
            aggregate_aligned_probabilities(
                stack,
                method=method_names[str(recipe["aggregation"])],
                **kwargs,
            )
        )
    candidates = torch.stack(candidate_values, dim=0)
    source = base_probabilities[0]
    for label, value in (
        ("source", source),
        ("uncertainties", uncertainties),
        ("candidates", candidates),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise P4ProtocolError(f"{label} contains NaN/Inf")
        if bool((value < 0).any().item()) or bool((value > 1).any().item()):
            raise P4ProtocolError(f"{label} is outside [0,1]")
        if value.requires_grad or value.grad_fn is not None:
            raise P4ProtocolError(f"{label} is not detached")
    return source.detach(), uncertainties.detach(), candidates.detach()


def _build_teacher_outputs(adapter: Any, image: Any, contract: P4Contract) -> tuple[Any, Any, Any, Any]:
    from tta.proposals.source_multiview_teacher import build_aligned_view_probabilities

    base, names = build_aligned_view_probabilities(
        adapter,
        image,
        include_context_tile=True,
    )
    _equal(tuple(names), ("identity", "hflip", "vflip", "hvflip", "context_tile"), "teacher core view names")
    source, uncertainty, candidates = _aggregate_candidates(base, contract)
    return base.detach(), source.detach(), uncertainty.detach(), candidates.detach()


def _open_memmap(path: Path, *, dtype: str, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _condition_array_paths(condition_dir: Path) -> dict[str, Path]:
    return {
        "base_view_probabilities": condition_dir / "base_view_probabilities.npy",
        "source_probabilities": condition_dir / "source_probabilities.npy",
        "view_uncertainty": condition_dir / "view_uncertainty.npy",
        "candidate_probabilities": condition_dir / "candidate_probabilities.npy",
        "source_masks": condition_dir / "source_masks_gt_0_5.npy",
        "candidate_masks": condition_dir / "candidate_masks_gt_0_5.npy",
        "per_image": condition_dir / "per_image.jsonl",
    }


def _method_input_dataset(contract: P4Contract, dataset: str, condition: str):
    # Candidate phase has exactly one dataset import.  In particular, this
    # function does not import or reference the outer target loader.
    from materialize_binary_tent_ss_calibration_cache_v2 import (
        SourceCalibrationMethodInputDatasetV2,
    )

    cache_root = _repository_path(
        contract.repository,
        contract.raw["datasets"][dataset]["cache_root"],
        f"{dataset} cache root",
    )
    return SourceCalibrationMethodInputDatasetV2(
        cache_root,
        condition_key=condition,
        expected_protocol_sha256=contract.raw["cache"]["protocol_sha256"],
    )


def _build_source_model(contract: P4Contract, dataset: str, device_name: str):
    import torch
    import test_source as source_runner
    from tta.model_adapter import IRSTDModelAdapter

    source_runner.seed_everything(int(contract.raw["scope"]["seed"]))
    device = source_runner.resolve_device(device_name)
    model = source_runner.build_nsfpn_model()
    checkpoint = _repository_path(
        contract.repository,
        contract.raw["datasets"][dataset]["checkpoint_path"],
        f"{dataset} checkpoint",
    )
    if sha256_file(checkpoint) != contract.raw["datasets"][dataset]["checkpoint_sha256"]:
        raise P4ProtocolError(f"checkpoint changed before load: {dataset}")
    wrapper = source_runner.load_trusted_checkpoint(model, checkpoint)
    model.to(device)
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    model.zero_grad(set_to_none=True)
    if any(module.training for module in model.modules()):
        raise P4ProtocolError("source model is not fully in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise P4ProtocolError("source model contains trainable parameters")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise P4ProtocolError("source model contains gradients")
    return source_runner, model, adapter, device, wrapper


def _runtime_environment_receipt(torch: Any, device: Any) -> dict[str, Any]:
    """Seal runtime-only dependencies that cannot use repository-relative paths."""

    extension = importlib.import_module("MultiScaleDeformableAttention")
    raw_extension_path = getattr(extension, "__file__", None)
    if not isinstance(raw_extension_path, str) or not raw_extension_path:
        raise P4ProtocolError(
            "loaded MultiScaleDeformableAttention has no auditable __file__"
        )
    extension_path = Path(os.path.abspath(raw_extension_path))
    if extension_path.is_symlink() or not extension_path.is_file():
        raise P4ProtocolError(
            f"loaded MultiScaleDeformableAttention file is unsafe: {extension_path}"
        )
    device_type = str(getattr(device, "type", device)).split(":", 1)[0]
    if device_type == "cuda":
        gpu_name: str | None = str(torch.cuda.get_device_name(device))
    else:
        gpu_name = None
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn": (
            None
            if torch.backends.cudnn.version() is None
            else int(torch.backends.cudnn.version())
        ),
        "device": str(device),
        "gpu_name": gpu_name,
        "sfs_extension_module": "MultiScaleDeformableAttention",
        "sfs_extension_file": str(extension_path),
        "sfs_extension_sha256": sha256_file(extension_path),
    }


def _capture_lightweight_model_gate(model: Any) -> dict[str, tuple[tuple[str, Any], ...]]:
    """Capture mutation-sensitive metadata without copying tensor contents."""

    from torch.nn.modules.batchnorm import _BatchNorm

    parameters = tuple(model.named_parameters())
    modules = tuple(model.named_modules())
    return {
        "parameter_versions": tuple(
            (name, int(parameter._version)) for name, parameter in parameters
        ),
        "buffer_versions": tuple(
            (name, int(buffer._version)) for name, buffer in model.named_buffers()
        ),
        "module_training": tuple(
            (name, bool(module.training)) for name, module in modules
        ),
        "module_runtime": tuple(
            (
                name,
                f"{type(module).__module__}.{type(module).__qualname__}",
                bool(module.training),
            )
            for name, module in modules
        ),
        "batchnorm_runtime": tuple(
            (
                name,
                f"{type(module).__module__}.{type(module).__qualname__}",
                float(module.eps),
                None if module.momentum is None else float(module.momentum),
                bool(module.affine),
                bool(module.track_running_stats),
                int(module.num_features),
            )
            for name, module in modules
            if isinstance(module, _BatchNorm)
        ),
        "parameter_requires_grad": tuple(
            (name, bool(parameter.requires_grad)) for name, parameter in parameters
        ),
        "parameter_grad_is_none": tuple(
            (name, parameter.grad is None) for name, parameter in parameters
        ),
    }


def _assert_lightweight_model_gate_unchanged(
    model: Any,
    expected: Mapping[str, tuple[tuple[str, Any], ...]],
) -> None:
    observed = _capture_lightweight_model_gate(model)
    for field in MODEL_RUNTIME_FINGERPRINT_FIELDS:
        if observed[field] != expected[field]:
            raise P4ProtocolError(
                f"per-image frozen-model gate detected drift in {field}"
            )


def _numpy_rng_state_sha256() -> str:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    digest = hashlib.sha256()
    digest.update(str(algorithm).encode("ascii"))
    digest.update(_raw_array_sha256(keys).encode("ascii"))
    digest.update(str(int(position)).encode("ascii"))
    digest.update(str(int(has_gauss)).encode("ascii"))
    digest.update(float(cached_gaussian).hex().encode("ascii"))
    return digest.hexdigest()


def _capture_rng_state_sha256(torch: Any, device: Any) -> dict[str, str | None]:
    device_type = str(getattr(device, "type", device)).split(":", 1)[0]
    cuda_state = (
        torch.cuda.get_rng_state(device=device) if device_type == "cuda" else None
    )
    return {
        "python_random": _sha256_bytes(_canonical_json_bytes(random.getstate())),
        "numpy_random": _numpy_rng_state_sha256(),
        "torch_cpu": _raw_array_sha256(torch.get_rng_state().cpu().numpy()),
        "torch_current_cuda": (
            None
            if cuda_state is None
            else _raw_array_sha256(cuda_state.cpu().numpy())
        ),
    }


def _assert_rng_state_unchanged(
    before: Mapping[str, str | None], after: Mapping[str, str | None]
) -> None:
    if dict(after) != dict(before):
        changed = sorted(
            key for key in set(before) | set(after) if before.get(key) != after.get(key)
        )
        raise P4ProtocolError(
            f"non-adaptive teacher changed random-generator state: {changed}"
        )


def _build_teacher_outputs_with_per_image_runtime_gates(
    adapter: Any,
    image: Any,
    model: Any,
    expected_model_gate: Mapping[str, tuple[tuple[str, Any], ...]],
    contract: P4Contract,
    torch: Any,
) -> tuple[Any, Any, Any, Any, dict[str, str | None], dict[str, str | None]]:
    """Run one label-free teacher call behind model and RNG fail-closed gates."""

    rng_before = _capture_rng_state_sha256(torch, image.device)
    outputs = _build_teacher_outputs(adapter, image, contract)
    rng_after = _capture_rng_state_sha256(torch, image.device)
    _assert_lightweight_model_gate_unchanged(model, expected_model_gate)
    _assert_rng_state_unchanged(rng_before, rng_after)
    return (*outputs, rng_before, rng_after)


def _execute_candidate_payload(
    staging: Path,
    *,
    contract: P4Contract,
    dataset: str,
    device_name: str,
    image_limit: int,
    formal: bool,
) -> dict[str, Any]:
    import torch
    from dataio.corruption_cache import TensorSequenceHasher

    source_runner, model, adapter, device, checkpoint_wrapper = _build_source_model(
        contract, dataset, device_name
    )
    lightweight_model_gate = _capture_lightweight_model_gate(model)
    lightweight_model_gate_sha256 = _sha256_bytes(
        _canonical_json_bytes(lightweight_model_gate)
    )
    rng_state_before = _capture_rng_state_sha256(torch, device)
    runtime_environment = _runtime_environment_receipt(torch, device)
    state_before = source_runner.state_dict_sha256(model.state_dict())
    modes_before = tuple((name, bool(module.training)) for name, module in model.named_modules())
    requires_grad_before = tuple(
        (name, bool(parameter.requires_grad)) for name, parameter in model.named_parameters()
    )
    condition_records: list[dict[str, Any]] = []
    ordered_image_ids: tuple[str, ...] | None = None
    per_image_state_gate_checks = 0
    per_image_rng_gate_checks = 0
    started = time.perf_counter()

    for corruption, severity in contract.conditions:
        key = _condition_key(corruption, severity)
        method_dataset = _method_input_dataset(contract, dataset, key)
        if len(method_dataset) != IMAGE_COUNT:
            raise P4ProtocolError(f"method cache {dataset}/{key} does not contain 64 images")
        count = image_limit
        condition_dir = staging / "conditions" / key
        paths = _condition_array_paths(condition_dir)
        base_map = _open_memmap(
            paths["base_view_probabilities"],
            dtype="<f4",
            shape=(count, 5, *PROBABILITY_SHAPE),
        )
        source_map = _open_memmap(
            paths["source_probabilities"], dtype="<f4", shape=(count, *PROBABILITY_SHAPE)
        )
        uncertainty_map = _open_memmap(
            paths["view_uncertainty"], dtype="<f4", shape=(count, 2, *PROBABILITY_SHAPE)
        )
        candidate_map = _open_memmap(
            paths["candidate_probabilities"],
            dtype="<f4",
            shape=(count, len(CANDIDATE_IDS), *PROBABILITY_SHAPE),
        )
        source_masks = _open_memmap(
            paths["source_masks"], dtype="|u1", shape=(count, *PROBABILITY_SHAPE)
        )
        candidate_masks = _open_memmap(
            paths["candidate_masks"],
            dtype="|u1",
            shape=(count, len(CANDIDATE_IDS), *PROBABILITY_SHAPE),
        )
        input_hasher = TensorSequenceHasher()
        per_image: list[dict[str, Any]] = []
        cell_ids: list[str] = []
        for index in range(count):
            sample = dict(method_dataset[index])
            if frozenset(sample) != METHOD_FIELDS:
                raise P4ProtocolError(
                    f"candidate method fields differ: {sorted(sample)}"
                )
            image_id = str(sample["image_id"])
            if not image_id:
                raise P4ProtocolError("candidate image_id cannot be empty")
            _equal(str(sample["dataset"]), dataset, "candidate sample dataset")
            _equal(str(sample["corruption"]), corruption, "candidate sample corruption")
            _equal(int(sample["severity"]), severity, "candidate sample severity")
            _equal(
                int(sample["seed"]),
                int(contract.raw["scope"]["seed"]),
                "candidate sample seed",
            )
            original_size = tuple(int(value) for value in sample["original_size"])
            if len(original_size) != 2 or any(value <= 0 for value in original_size):
                raise P4ProtocolError("candidate original_size must contain two positive integers")
            cell_ids.append(image_id)
            image_cpu = sample.pop("image")
            if tuple(image_cpu.shape) != IMAGE_SHAPE or image_cpu.dtype != torch.float32:
                raise P4ProtocolError("candidate input must be float32 [3,256,256]")
            if not bool(torch.isfinite(image_cpu).all().item()):
                raise P4ProtocolError("candidate input contains NaN/Inf")
            input_hasher.update(image_id, image_cpu)
            input_reference = image_cpu.clone()
            image = image_cpu.unsqueeze(0).to(device, non_blocking=False)
            device_input_reference = image.clone()
            with torch.inference_mode():
                (
                    base,
                    source,
                    uncertainty,
                    candidates,
                    image_rng_before,
                    image_rng_after,
                ) = _build_teacher_outputs_with_per_image_runtime_gates(
                    adapter,
                    image,
                    model,
                    lightweight_model_gate,
                    contract,
                    torch,
                )
            if not torch.equal(image_cpu, input_reference) or not torch.equal(
                image, device_input_reference
            ):
                raise P4ProtocolError("candidate method modified its input")
            per_image_state_gate_checks += 1
            per_image_rng_gate_checks += 1
            base_np = base[:, 0].detach().cpu().numpy().astype("<f4", copy=False)
            source_np = source[0].detach().cpu().numpy().astype("<f4", copy=False)
            uncertainty_np = uncertainty[:, 0].detach().cpu().numpy().astype("<f4", copy=False)
            candidates_np = candidates[:, 0].detach().cpu().numpy().astype("<f4", copy=False)
            if not np.array_equal(source_np, base_np[0]):
                raise P4ProtocolError("source probability differs from identity base view")
            base_map[index] = base_np
            source_map[index] = source_np
            uncertainty_map[index] = uncertainty_np
            candidate_map[index] = candidates_np
            source_mask = np.where(source_np > THRESHOLD, 255, 0).astype(np.uint8)
            masks = np.where(candidates_np > THRESHOLD, 255, 0).astype(np.uint8)
            source_masks[index] = source_mask
            candidate_masks[index] = masks
            per_image.append(
                {
                    "index": index,
                    "image_id": image_id,
                    "original_size": list(original_size),
                    "dataset": str(sample["dataset"]),
                    "corruption": str(sample["corruption"]),
                    "severity": int(sample["severity"]),
                    "seed": int(sample["seed"]),
                    "input_tensor_sha256": _raw_array_sha256(image_cpu.numpy()),
                    "base_view_probabilities_sha256": _raw_array_sha256(base_np),
                    "source_probability_sha256": _raw_array_sha256(source_np),
                    "view_uncertainty_sha256": _raw_array_sha256(uncertainty_np),
                    "candidate_probabilities_sha256": _raw_array_sha256(candidates_np),
                    "source_mask_sha256": _raw_array_sha256(source_mask),
                    "candidate_masks_sha256": _raw_array_sha256(masks),
                    "model_runtime_fingerprint_sha256_before": (
                        lightweight_model_gate_sha256
                    ),
                    "model_runtime_fingerprint_sha256_after": (
                        lightweight_model_gate_sha256
                    ),
                    "model_runtime_fingerprint_unchanged": True,
                    "rng_state_sha256_before": image_rng_before,
                    "rng_state_sha256_after": image_rng_after,
                    "rng_state_unchanged": True,
                    "strict_probability_threshold": THRESHOLD,
                    "threshold_rule": "strict_greater_than",
                }
            )
        current_ids = tuple(cell_ids)
        if len(current_ids) != len(set(current_ids)):
            raise P4ProtocolError(f"Pilot image IDs are duplicated for {dataset}/{key}")
        if ordered_image_ids is None:
            ordered_image_ids = current_ids
        elif current_ids != ordered_image_ids:
            raise P4ProtocolError("Pilot image order differs between conditions")
        for array in (
            base_map,
            source_map,
            uncertainty_map,
            candidate_map,
            source_masks,
            candidate_masks,
        ):
            array.flush()
        del base_map, source_map, uncertainty_map, candidate_map, source_masks, candidate_masks
        _write_jsonl(paths["per_image"], per_image)
        condition_records.append(
            {
                "condition": key,
                "corruption": corruption,
                "severity": severity,
                "image_count": count,
                "input_tensor_sequence_sha256": input_hasher.hexdigest(),
                "arrays": {
                    "base_view_probabilities": {
                        "path": _safe_relative(paths["base_view_probabilities"], staging),
                        "dtype": "little_endian_float32",
                        "shape": [count, 5, 1, 256, 256],
                        "view_names": list(BASE_VIEW_NAMES),
                    },
                    "source_probabilities": {
                        "path": _safe_relative(paths["source_probabilities"], staging),
                        "dtype": "little_endian_float32",
                        "shape": [count, 1, 256, 256],
                    },
                    "view_uncertainty": {
                        "path": _safe_relative(paths["view_uncertainty"], staging),
                        "dtype": "little_endian_float32",
                        "shape": [count, 2, 1, 256, 256],
                        "view_sets": ["flip4", "tile5"],
                    },
                    "candidate_probabilities": {
                        "path": _safe_relative(paths["candidate_probabilities"], staging),
                        "dtype": "little_endian_float32",
                        "shape": [count, 10, 1, 256, 256],
                        "candidate_ids": list(contract.candidate_ids),
                    },
                    "source_masks": {
                        "path": _safe_relative(paths["source_masks"], staging),
                        "dtype": "uint8_0_or_255",
                        "shape": [count, 1, 256, 256],
                    },
                    "candidate_masks": {
                        "path": _safe_relative(paths["candidate_masks"], staging),
                        "dtype": "uint8_0_or_255",
                        "shape": [count, 10, 1, 256, 256],
                        "candidate_ids": list(contract.candidate_ids),
                    },
                },
                "per_image_path": _safe_relative(paths["per_image"], staging),
            }
        )

    assert ordered_image_ids is not None
    _equal(
        per_image_state_gate_checks,
        len(contract.conditions) * image_limit,
        "per-image lightweight state-gate check count",
    )
    _equal(
        per_image_rng_gate_checks,
        len(contract.conditions) * image_limit,
        "per-image RNG-gate check count",
    )
    rng_state_after = _capture_rng_state_sha256(torch, device)
    _assert_rng_state_unchanged(rng_state_before, rng_state_after)
    state_after = source_runner.state_dict_sha256(model.state_dict())
    modes_after = tuple((name, bool(module.training)) for name, module in model.named_modules())
    requires_grad_after = tuple(
        (name, bool(parameter.requires_grad)) for name, parameter in model.named_parameters()
    )
    if state_after != state_before:
        raise P4ProtocolError("model state changed during non-adaptive teacher inference")
    if modes_after != modes_before or any(training for _, training in modes_after):
        raise P4ProtocolError("model train/eval modes changed during teacher inference")
    if requires_grad_after != requires_grad_before or any(value for _, value in requires_grad_after):
        raise P4ProtocolError("requires_grad flags changed during teacher inference")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise P4ProtocolError("teacher inference created gradients")
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_nonadaptive_teacher_candidate_dataset",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "phase": "candidate",
        "dataset": dataset,
        "formal": formal,
        "development_only": True,
        "paper_result": False,
        "p5_authorized": False,
        "config_path": str(contract.config_path),
        "config_sha256": contract.config_sha256,
        "lineage": _stage_transition_lineage(contract),
        "checkpoint_role": "best_miou",
        "checkpoint_path": contract.raw["datasets"][dataset]["checkpoint_path"],
        "checkpoint_sha256": contract.raw["datasets"][dataset]["checkpoint_sha256"],
        "checkpoint_wrapper": checkpoint_wrapper,
        "cache_root": contract.raw["datasets"][dataset]["cache_root"],
        "cache_manifest_sha256": contract.raw["datasets"][dataset]["cache_manifest_sha256"],
        "cache_content_sha256": contract.raw["datasets"][dataset]["cache_content_sha256"],
        "method_input_manifest_sha256": contract.raw["datasets"][dataset]["method_input_manifest_sha256"],
        "condition_count": len(condition_records),
        "image_count_per_condition": image_limit,
        "candidate_count": len(contract.candidate_ids),
        "candidate_ids": list(contract.candidate_ids),
        "base_view_names": list(BASE_VIEW_NAMES),
        "image_ids": list(ordered_image_ids),
        "conditions": condition_records,
        "method_boundary": {
            "loader": "SourceCalibrationMethodInputDatasetV2",
            "fields": sorted(METHOD_FIELDS),
            "target_loader_calls": 0,
            "method_label_accesses": 0,
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
        },
        "execution": {
            "seed": 42,
            "requires_grad": False,
            "optimizer_present": False,
            "model_fully_eval": True,
            "state_sha256_before": state_before,
            "state_sha256_after": state_after,
            "state_bit_exact": True,
            "per_image_lightweight_state_gate": {
                "checks": per_image_state_gate_checks,
                "expected_checks": len(contract.conditions) * image_limit,
                "baseline_fingerprint_sha256": lightweight_model_gate_sha256,
                "fingerprint_fields": list(MODEL_RUNTIME_FINGERPRINT_FIELDS),
                "module_count": len(lightweight_model_gate["module_runtime"]),
                "batchnorm_module_count": len(
                    lightweight_model_gate["batchnorm_runtime"]
                ),
                "parameter_count": len(
                    lightweight_model_gate["parameter_versions"]
                ),
                "buffer_count": len(lightweight_model_gate["buffer_versions"]),
                "parameter_versions_unchanged": True,
                "buffer_versions_unchanged": True,
                "module_training_unchanged": True,
                "module_type_and_training_unchanged": True,
                "batchnorm_runtime_attributes_unchanged": True,
                "parameter_requires_grad_unchanged": True,
                "parameter_grad_is_none_unchanged": True,
                "passed": True,
            },
            "per_image_rng_state_gate": {
                "checks": per_image_rng_gate_checks,
                "expected_checks": len(contract.conditions) * image_limit,
                "streams": list(RNG_STREAMS),
                "evidence_location": "conditions/*/per_image.jsonl",
                "passed": True,
            },
            "rng_state_sha256_before": rng_state_before,
            "rng_state_sha256_after": rng_state_after,
            "rng_state_unchanged": True,
            "strict_threshold": "probability > 0.5",
            "runtime_environment": runtime_environment,
            "wall_time_seconds": time.perf_counter() - started,
        },
        "code_sha256": _capture_code_hashes(contract),
    }


def run_candidate(
    contract: P4Contract,
    *,
    dataset: str,
    device_name: str,
    max_images: int | None = None,
) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise P4ProtocolError(f"unsupported dataset: {dataset}")
    if max_images is not None and (
        isinstance(max_images, bool) or max_images < 1 or max_images > IMAGE_COUNT
    ):
        raise P4ProtocolError("--max-images must lie in [1,64]")
    formal = max_images is None
    if formal:
        destination = _artifact_destination(contract, "candidate", dataset)
        existing = _existing_complete_or_raise(
            destination,
            contract=contract,
            phase="candidate",
            dataset=dataset,
        )
        if existing is not None:
            return {
                "status": "existing_verified_complete_no_op",
                "phase": "candidate",
                "dataset": dataset,
                "path": str(destination),
                "manifest_sha256": existing["_manifest_sha256"],
            }
    else:
        engineering_parent = (
            contract.output_root
            / contract.raw["output"]["engineering_phase"]
            / dataset
            / f"max_images_{max_images}"
        )
        destination = engineering_parent / uuid.uuid4().hex
    staging = _new_staging(destination)
    try:
        manifest = _execute_candidate_payload(
            staging,
            contract=contract,
            dataset=dataset,
            device_name=device_name,
            image_limit=IMAGE_COUNT if max_images is None else max_images,
            formal=formal,
        )
        _publish_artifact(staging, destination, manifest=manifest)
    except BaseException:
        _remove_private_staging(staging)
        raise
    verified = verify_artifact(
        destination,
        contract=contract,
        phase="candidate",
        dataset=dataset,
    )
    return {
        "status": "published",
        "phase": "candidate",
        "dataset": dataset,
        "formal": formal,
        "path": str(destination),
        "manifest_sha256": verified["_manifest_sha256"],
    }


def _evaluation_protocol(config: Mapping[str, Any]):
    from metrics.irstd_metrics import IRSTDEvaluationProtocol

    evaluation = config["evaluation"]
    return IRSTDEvaluationProtocol(
        fixed_probability_threshold=float(evaluation["probability_threshold"]),
        froc_probability_thresholds=tuple(
            float(value) for value in evaluation["froc_probability_thresholds"]
        ),
        connectivity={8: 2}[int(evaluation["foreground_connectivity_2d"])],
        max_centroid_distance=float(
            evaluation["target_matching"]["max_centroid_distance_pixels"]
        ),
        min_component_area=int(evaluation["min_component_area"]),
    )


def _endpoint_counts(result: Any) -> dict[str, int]:
    fixed = result.fixed
    pixel = fixed.pixel
    return {
        "intersection_pixels": int(pixel.true_positive_pixels),
        "false_positive_pixels": int(pixel.false_positive_pixels),
        "false_negative_pixels": int(pixel.false_negative_pixels),
        "true_negative_pixels": int(pixel.true_negative_pixels),
        "union_pixels": int(
            pixel.true_positive_pixels
            + pixel.false_positive_pixels
            + pixel.false_negative_pixels
        ),
        "predicted_positive_pixels": int(pixel.predicted_positive_pixels),
        "target_positive_pixels": int(pixel.target_positive_pixels),
        "total_image_pixels": int(fixed.total_image_pixels),
        "detected_targets": int(fixed.detected_targets),
        "total_targets": int(fixed.total_targets),
        "false_alarm_pixels": int(fixed.false_alarm_pixels),
        "image_count": int(fixed.image_count),
    }


def _summary_metrics(result: Any) -> dict[str, float]:
    fixed = result.fixed
    total_pixels = int(fixed.total_image_pixels)
    return {
        "global_iou": float(fixed.pixel.intersection_over_union),
        "pd": float(fixed.detection_probability),
        "fa_per_million": float(fixed.false_alarm_pixel_rate * 1_000_000.0),
        "foreground_fraction": (
            float(fixed.pixel.predicted_positive_pixels / total_pixels)
            if total_pixels
            else 0.0
        ),
        "normalized_iou": float(result.normalized_iou.normalized_iou),
    }


def _evaluate_probabilities(probabilities: Any, targets: Any, image_ids: Sequence[str]):
    from metrics.irstd_metrics_v2 import UnifiedResearchEvaluatorV2

    evaluator = UnifiedResearchEvaluatorV2(_evaluation_protocol_cached())
    evaluator.update_probabilities(probabilities, targets, image_ids=image_ids)
    result = evaluator.compute()
    result.assert_endpoint_conservation()
    return result


_ACTIVE_EVALUATION_PROTOCOL: Any | None = None


def _evaluation_protocol_cached():
    if _ACTIVE_EVALUATION_PROTOCOL is None:
        raise P4ProtocolError("outer evaluation protocol is not initialized")
    return _ACTIVE_EVALUATION_PROTOCOL


def _sum_endpoints(values: Sequence[Mapping[str, int]]) -> dict[str, int]:
    if not values:
        raise P4ProtocolError("cannot sum zero endpoint records")
    keys = tuple(values[0])
    if any(tuple(value) != keys for value in values):
        raise P4ProtocolError("per-image endpoint keys differ")
    return {key: sum(int(value[key]) for value in values) for key in keys}


def _load_candidate_array(
    candidate_root: Path,
    record: Mapping[str, Any],
    name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
):
    arrays = _mapping(record.get("arrays"), "candidate condition arrays")
    descriptor = _mapping(arrays.get(name), f"candidate array {name}")
    path = candidate_root / str(descriptor["path"])
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(value.shape) != expected_shape or value.dtype.str != expected_dtype:
        raise P4ProtocolError(
            f"candidate array contract drift for {name}: {value.shape}/{value.dtype.str}"
        )
    if bool(value.flags.writeable):
        raise P4ProtocolError(f"candidate array is writable: {name}")
    return value


def _load_outer_targets(contract: P4Contract, dataset: str):
    from materialize_binary_tent_ss_calibration_cache_v2 import (
        load_outer_evaluator_targets_v2,
    )

    cache_root = _repository_path(
        contract.repository,
        contract.raw["datasets"][dataset]["cache_root"],
        f"{dataset} cache root",
    )
    return load_outer_evaluator_targets_v2(
        cache_root,
        expected_protocol_sha256=contract.raw["cache"]["protocol_sha256"],
        episodes_complete=True,
    )


def _execute_outer_payload(
    staging: Path,
    *,
    contract: P4Contract,
    dataset: str,
    candidate_root: Path,
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    global _ACTIVE_EVALUATION_PROTOCOL

    if candidate_manifest.get("formal") is not True:
        raise P4ProtocolError("outer phase refuses an engineering candidate artifact")
    if int(candidate_manifest.get("image_count_per_condition", -1)) != IMAGE_COUNT:
        raise P4ProtocolError("outer phase requires all 64 Pilot images")
    if tuple(candidate_manifest.get("candidate_ids", ())) != contract.candidate_ids:
        raise P4ProtocolError("candidate artifact candidate order differs")
    image_ids = tuple(str(value) for value in candidate_manifest.get("image_ids", ()))
    if len(image_ids) != IMAGE_COUNT or len(set(image_ids)) != IMAGE_COUNT:
        raise P4ProtocolError("candidate artifact Pilot IDs are invalid")
    conditions = tuple(candidate_manifest.get("conditions", ()))
    if tuple(str(value.get("condition")) for value in conditions) != tuple(
        _condition_key(*value) for value in contract.conditions
    ):
        raise P4ProtocolError("candidate artifact condition order differs")

    # This call is intentionally after the complete candidate artifact and all
    # of its method-facing metadata have been verified above.
    targets = _load_outer_targets(contract, dataset)
    if tuple(targets.shape) != (IMAGE_COUNT, 1, 256, 256):
        raise P4ProtocolError("outer target array shape differs")
    _ACTIVE_EVALUATION_PROTOCOL = _evaluation_protocol(contract.raw)
    cell_records: list[dict[str, Any]] = []
    per_image_records: list[dict[str, Any]] = []
    try:
        for condition_record in conditions:
            condition = str(condition_record["condition"])
            source = _load_candidate_array(
                candidate_root,
                condition_record,
                "source_probabilities",
                (IMAGE_COUNT, 1, 256, 256),
                "<f4",
            )
            teachers = _load_candidate_array(
                candidate_root,
                condition_record,
                "candidate_probabilities",
                (IMAGE_COUNT, 10, 1, 256, 256),
                "<f4",
            )
            source_masks = _load_candidate_array(
                candidate_root,
                condition_record,
                "source_masks",
                (IMAGE_COUNT, 1, 256, 256),
                "|u1",
            )
            teacher_masks = _load_candidate_array(
                candidate_root,
                condition_record,
                "candidate_masks",
                (IMAGE_COUNT, 10, 1, 256, 256),
                "|u1",
            )
            if not np.array_equal(source_masks, np.where(source > THRESHOLD, 255, 0).astype(np.uint8)):
                raise P4ProtocolError(f"stored source masks violate strict threshold: {condition}")
            if not np.array_equal(teacher_masks, np.where(teachers > THRESHOLD, 255, 0).astype(np.uint8)):
                raise P4ProtocolError(f"stored teacher masks violate strict threshold: {condition}")
            source_result = _evaluate_probabilities(source, targets, image_ids)
            source_endpoint = _endpoint_counts(source_result)
            source_metrics = _summary_metrics(source_result)
            candidate_results: dict[str, Any] = {}
            for candidate_index, candidate_id in enumerate(contract.candidate_ids):
                teacher_result = _evaluate_probabilities(
                    teachers[:, candidate_index], targets, image_ids
                )
                candidate_results[candidate_id] = teacher_result
                cell_records.append(
                    {
                        "schema_version": 1,
                        "artifact_type": "cr_sitta_nonadaptive_teacher_cell",
                        "candidate_id": candidate_id,
                        "dataset": dataset,
                        "condition": condition,
                        "corruption": str(condition_record["corruption"]),
                        "severity": int(condition_record["severity"]),
                        "image_count": IMAGE_COUNT,
                        "source": source_endpoint,
                        "teacher": _endpoint_counts(teacher_result),
                        "source_metrics": source_metrics,
                        "teacher_metrics": _summary_metrics(teacher_result),
                    }
                )

            per_image_source: list[dict[str, int]] = []
            per_image_teacher: dict[str, list[dict[str, int]]] = {
                candidate_id: [] for candidate_id in contract.candidate_ids
            }
            for image_index, image_id in enumerate(image_ids):
                target = np.asarray(targets[image_index : image_index + 1])
                source_one = _evaluate_probabilities(
                    np.asarray(source[image_index : image_index + 1]), target, (image_id,)
                )
                source_one_endpoint = _endpoint_counts(source_one)
                per_image_source.append(source_one_endpoint)
                teacher_payload: dict[str, Any] = {}
                for candidate_index, candidate_id in enumerate(contract.candidate_ids):
                    teacher_one = _evaluate_probabilities(
                        np.asarray(
                            teachers[
                                image_index : image_index + 1,
                                candidate_index,
                            ]
                        ),
                        target,
                        (image_id,),
                    )
                    endpoint = _endpoint_counts(teacher_one)
                    per_image_teacher[candidate_id].append(endpoint)
                    teacher_payload[candidate_id] = {
                        "counts": endpoint,
                        "metrics": _summary_metrics(teacher_one),
                    }
                per_image_records.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "image_index": image_index,
                        "image_id": image_id,
                        "source": {
                            "counts": source_one_endpoint,
                            "metrics": _summary_metrics(source_one),
                        },
                        "teachers": teacher_payload,
                    }
                )
            if _sum_endpoints(per_image_source) != source_endpoint:
                raise P4ProtocolError(f"source per-image counts do not sum to cell: {condition}")
            for candidate_id in contract.candidate_ids:
                if _sum_endpoints(per_image_teacher[candidate_id]) != _endpoint_counts(
                    candidate_results[candidate_id]
                ):
                    raise P4ProtocolError(
                        f"teacher per-image counts do not sum to cell: {condition}/{candidate_id}"
                    )
    finally:
        _ACTIVE_EVALUATION_PROTOCOL = None

    _write_jsonl(staging / "cell_records.jsonl", cell_records)
    _write_jsonl(staging / "per_image.jsonl", per_image_records)
    _write_json(
        staging / "outer_access_receipt.json",
        {
            "schema_version": 1,
            "artifact_type": "cr_sitta_nonadaptive_teacher_outer_access_receipt",
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "dataset": dataset,
            "candidate_manifest_sha256": candidate_manifest["_manifest_sha256"],
            "candidate_completion_sha256": candidate_manifest["_complete_sha256"],
            "candidate_complete_verified_before_target_load": True,
            "target_loader": "load_outer_evaluator_targets_v2",
            "target_loader_call_count": 1,
            "target_role": "train_pilot64_outer_target",
            "targets_visible_to_candidate": False,
            "candidate_target_loader_calls": 0,
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
            "development_only": True,
            "paper_result": False,
            "p5_authorized": False,
        },
    )
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_nonadaptive_teacher_outer_dataset",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "phase": "outer",
        "dataset": dataset,
        "formal": True,
        "development_only": True,
        "paper_result": False,
        "p5_authorized": False,
        "config_path": str(contract.config_path),
        "config_sha256": contract.config_sha256,
        "lineage": _stage_transition_lineage(contract),
        "candidate_artifact": str(candidate_root),
        "candidate_manifest_sha256": candidate_manifest["_manifest_sha256"],
        "candidate_completion_sha256": candidate_manifest["_complete_sha256"],
        "cell_record_count": len(cell_records),
        "per_image_record_count": len(per_image_records),
        "candidate_count": len(contract.candidate_ids),
        "candidate_ids": list(contract.candidate_ids),
        "condition_count": len(CONDITIONS),
        "image_count_per_condition": IMAGE_COUNT,
        "cell_records_path": "cell_records.jsonl",
        "per_image_path": "per_image.jsonl",
        "outer_access_receipt_path": "outer_access_receipt.json",
        "evaluation": {
            "evaluator": "metrics.irstd_metrics_v2.UnifiedResearchEvaluatorV2",
            "threshold": THRESHOLD,
            "threshold_rule": "strict_greater_than",
            "froc_endpoint_conservation_verified": True,
        },
        "code_sha256": _capture_code_hashes(contract),
    }


def run_outer(contract: P4Contract, *, dataset: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise P4ProtocolError(f"unsupported dataset: {dataset}")
    destination = _artifact_destination(contract, "outer", dataset)
    existing = _existing_complete_or_raise(
        destination,
        contract=contract,
        phase="outer",
        dataset=dataset,
    )
    if existing is not None:
        return {
            "status": "existing_verified_complete_no_op",
            "phase": "outer",
            "dataset": dataset,
            "path": str(destination),
            "manifest_sha256": existing["_manifest_sha256"],
        }
    candidate_root = _artifact_destination(contract, "candidate", dataset)
    # This complete verification must precede both staging creation and the
    # only outer-target loader call.
    candidate_manifest = verify_artifact(
        candidate_root,
        contract=contract,
        phase="candidate",
        dataset=dataset,
    )
    staging = _new_staging(destination)
    try:
        manifest = _execute_outer_payload(
            staging,
            contract=contract,
            dataset=dataset,
            candidate_root=candidate_root,
            candidate_manifest=candidate_manifest,
        )
        _publish_artifact(staging, destination, manifest=manifest)
    except BaseException:
        _remove_private_staging(staging)
        raise
    verified = verify_artifact(
        destination,
        contract=contract,
        phase="outer",
        dataset=dataset,
    )
    return {
        "status": "published",
        "phase": "outer",
        "dataset": dataset,
        "path": str(destination),
        "manifest_sha256": verified["_manifest_sha256"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise P4ProtocolError(f"blank JSONL line at {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise P4ProtocolError(f"JSONL record is not an object: {path}:{line_number}")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise P4ProtocolError(f"cannot read JSONL artifact: {path}") from exc
    return records


def _validate_aggregate_grid(
    records: Sequence[Mapping[str, Any]], contract: P4Contract
) -> None:
    expected = {
        (candidate_id, dataset, _condition_key(*condition))
        for candidate_id in contract.candidate_ids
        for dataset in DATASETS
        for condition in contract.conditions
    }
    observed = [
        (str(record.get("candidate_id")), str(record.get("dataset")), str(record.get("condition")))
        for record in records
    ]
    duplicates = sorted(key for key in set(observed) if observed.count(key) != 1)
    missing = sorted(expected - set(observed))
    extra = sorted(set(observed) - expected)
    if duplicates or missing or extra or len(observed) != len(expected):
        raise P4ProtocolError(
            "aggregate grid is incomplete or duplicated; "
            f"missing={missing[:5]}, extra={extra[:5]}, duplicates={duplicates[:5]}"
        )


def _build_gate_config(config: Mapping[str, Any]):
    from analysis.nonadaptive_teacher_gate import GateConfig

    gate = config["science_gate"]
    fa = gate["fa_delta_maximum_formula"]
    foreground = gate["foreground_fraction"]
    # Keep all coupling to the pure gate's constructor in this one function.
    # GateConfig uses Fraction internally and accepts exact decimal strings.
    values = {
        "comparison_tolerance": str(gate["comparison_tolerance"]),
        "nonclean_macro_delta_iou_epsilon": str(
            gate["nonclean_36_macro_delta_iou_strictly_greater_than"]
        ),
        "overall_macro_delta_iou_threshold": str(
            gate["overall_39_macro_delta_iou_strictly_greater_than"]
        ),
        "minimum_positive_nonclean_datasets": int(gate["minimum_positive_nonclean_datasets"]),
        "worst_nonclean_dataset_delta_iou_minimum": str(
            gate["worst_nonclean_dataset_delta_iou_minimum"]
        ),
        "clean_macro_delta_iou_minimum": str(gate["clean_macro_delta_iou_minimum"]),
        "each_clean_dataset_delta_iou_minimum": str(
            gate["each_clean_dataset_delta_iou_minimum"]
        ),
        "nonclean_macro_delta_pd_minimum": str(
            gate["nonclean_macro_delta_pd_minimum"]
        ),
        "each_nonclean_dataset_delta_pd_minimum": str(
            gate["each_nonclean_dataset_delta_pd_minimum"]
        ),
        "fa_absolute_allowance_per_million": str(fa["absolute_allowance"]),
        "fa_source_multiplier": str(fa["source_multiplier"]),
        "foreground_fraction_delta_maximum": str(foreground["delta_maximum"]),
        "foreground_fraction_source_multiplier": str(
            foreground["teacher_to_source_multiplier"]
        ),
        "foreground_fraction_epsilon": str(foreground["epsilon"]),
        "require_integer_counts": True,
    }
    return GateConfig(**values)


def _evaluate_gate(
    records: Sequence[Mapping[str, Any]], contract: P4Contract
) -> tuple[Any, Mapping[str, Any]]:
    from analysis.nonadaptive_teacher_gate import evaluate_nonadaptive_teacher_gate

    gate_config = _build_gate_config(contract.raw)
    if getattr(gate_config, "require_integer_counts", None) is not True:
        raise P4ProtocolError("formal P4 aggregation requires integer-count evidence")
    decision = evaluate_nonadaptive_teacher_gate(
        records,
        contract.candidate_ids,
        gate_config,
    )
    converter = getattr(decision, "to_receipt", None)
    receipt = converter() if callable(converter) else getattr(decision, "to_dict", lambda: None)()
    if not isinstance(receipt, Mapping):
        raise P4ProtocolError("teacher gate decision did not produce a receipt mapping")
    if getattr(decision, "evidence_kind", None) != "counts" or receipt.get(
        "evidence_kind"
    ) != "counts":
        raise P4ProtocolError("formal P4 aggregation requires count-based evidence")
    if getattr(decision, "integer_count_conservation_verified", None) is not True or receipt.get(
        "integer_count_conservation_verified"
    ) is not True:
        raise P4ProtocolError(
            "formal P4 aggregation requires verified integer-count conservation"
        )
    receipt_gate = _mapping(receipt.get("gate"), "teacher gate receipt gate")
    if receipt_gate.get("require_integer_counts") is not True:
        raise P4ProtocolError(
            "formal P4 aggregation receipt must freeze require_integer_counts=true"
        )
    if receipt.get("p5_authorized") is not False:
        raise P4ProtocolError("P4 gate receipt must never authorize P5")
    if receipt.get("development_only") is not True or receipt.get("paper_result") is not False:
        raise P4ProtocolError("P4 gate receipt eligibility scope differs")
    return decision, receipt


def _execute_aggregate_payload(
    staging: Path,
    *,
    contract: P4Contract,
    outer_artifacts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for dataset in DATASETS:
        artifact = outer_artifacts[dataset]
        root = _artifact_destination(contract, "outer", dataset)
        values = _read_jsonl(root / str(artifact["cell_records_path"]))
        if len(values) != len(CONDITIONS) * len(CANDIDATE_IDS):
            raise P4ProtocolError(f"outer cell count differs for {dataset}")
        records.extend(values)
        bindings.append(
            {
                "dataset": dataset,
                "path": str(root),
                "manifest_sha256": artifact["_manifest_sha256"],
                "completion_sha256": artifact["_complete_sha256"],
                "cell_record_count": len(values),
            }
        )
    _validate_aggregate_grid(records, contract)
    _, gate_receipt = _evaluate_gate(records, contract)
    science_receipt = {
        "schema_version": 1,
        "artifact_type": "cr_sitta_nonadaptive_teacher_science_decision",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "config_sha256": contract.config_sha256,
        "protocol_status": "passed",
        "formal_protocol_complete": True,
        "development_only": True,
        "paper_result": False,
        "paper_test_result": False,
        "p5_authorized": False,
        "p5_requires_separate_protocol": True,
        "candidate_ids": list(contract.candidate_ids),
        "cell_record_count": len(records),
        "outer_artifacts": bindings,
        "gate_decision": dict(gate_receipt),
    }
    _write_jsonl(staging / "cell_records.jsonl", records)
    _write_json(staging / "science_decision_receipt.json", science_receipt)
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_nonadaptive_teacher_aggregate",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "phase": "aggregate",
        "dataset": None,
        "formal": True,
        "formal_protocol_complete": True,
        "development_only": True,
        "paper_result": False,
        "paper_test_result": False,
        "p5_authorized": False,
        "p5_requires_separate_protocol": True,
        "config_path": str(contract.config_path),
        "config_sha256": contract.config_sha256,
        "lineage": _stage_transition_lineage(contract),
        "candidate_count": len(contract.candidate_ids),
        "candidate_ids": list(contract.candidate_ids),
        "cell_record_count": len(records),
        "outer_artifacts": bindings,
        "cell_records_path": "cell_records.jsonl",
        "science_decision_receipt_path": "science_decision_receipt.json",
        "science_decision_receipt_sha256": sha256_file(
            staging / "science_decision_receipt.json"
        ),
        "code_sha256": _capture_code_hashes(contract),
    }


def run_aggregate(contract: P4Contract) -> dict[str, Any]:
    destination = _artifact_destination(contract, "aggregate")
    existing = _existing_complete_or_raise(
        destination,
        contract=contract,
        phase="aggregate",
        dataset=None,
    )
    if existing is not None:
        return {
            "status": "existing_verified_complete_no_op",
            "phase": "aggregate",
            "path": str(destination),
            "manifest_sha256": existing["_manifest_sha256"],
        }
    outer_artifacts: dict[str, Mapping[str, Any]] = {}
    # Completeness is checked before creating an aggregate staging directory.
    for dataset in DATASETS:
        root = _artifact_destination(contract, "outer", dataset)
        outer_artifacts[dataset] = verify_artifact(
            root,
            contract=contract,
            phase="outer",
            dataset=dataset,
        )
    staging = _new_staging(destination)
    try:
        manifest = _execute_aggregate_payload(
            staging,
            contract=contract,
            outer_artifacts=outer_artifacts,
        )
        _publish_artifact(staging, destination, manifest=manifest)
    except BaseException:
        _remove_private_staging(staging)
        raise
    verified = verify_artifact(
        destination,
        contract=contract,
        phase="aggregate",
        dataset=None,
    )
    receipt = _load_json(destination / "science_decision_receipt.json")
    return {
        "status": "published",
        "phase": "aggregate",
        "path": str(destination),
        "manifest_sha256": verified["_manifest_sha256"],
        "science_decision": receipt["gate_decision"],
        "p5_authorized": False,
    }


def verify_all(contract: P4Contract) -> dict[str, Any]:
    candidates = {}
    outers = {}
    for dataset in DATASETS:
        candidate_root = _artifact_destination(contract, "candidate", dataset)
        candidates[dataset] = verify_artifact(
            candidate_root,
            contract=contract,
            phase="candidate",
            dataset=dataset,
        )["_manifest_sha256"]
        outer_root = _artifact_destination(contract, "outer", dataset)
        outers[dataset] = verify_artifact(
            outer_root,
            contract=contract,
            phase="outer",
            dataset=dataset,
        )["_manifest_sha256"]
    aggregate_root = _artifact_destination(contract, "aggregate")
    aggregate = verify_artifact(
        aggregate_root,
        contract=contract,
        phase="aggregate",
        dataset=None,
    )
    receipt = _load_json(aggregate_root / str(aggregate["science_decision_receipt_path"]))
    _equal(receipt.get("p5_authorized"), False, "verified P5 authorization")
    return {
        "valid": True,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "config_sha256": contract.config_sha256,
        "candidate_artifacts": candidates,
        "outer_artifacts": outers,
        "aggregate_manifest_sha256": aggregate["_manifest_sha256"],
        "formal_protocol_complete": True,
        "development_only": True,
        "paper_result": False,
        "p5_authorized": False,
    }


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="validate the frozen P4 contract and inputs")
    candidate = commands.add_parser("candidate", help="run one label-free dataset candidate phase")
    candidate.add_argument("--dataset", choices=DATASETS, required=True)
    candidate.add_argument("--device", required=True)
    candidate.add_argument("--max-images", type=_positive_int, default=None)
    outer = commands.add_parser("outer", help="evaluate one complete candidate dataset")
    outer.add_argument("--dataset", choices=DATASETS, required=True)
    commands.add_parser("aggregate", help="apply the complete 39-cell scientific gate")
    commands.add_parser("verify", help="verify all formal P4 artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = load_contract(args.config)
    if args.command == "validate":
        result = {
            "valid": True,
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "config_path": str(contract.config_path),
            "config_sha256": contract.config_sha256,
            "datasets": list(DATASETS),
            "conditions_per_dataset": len(CONDITIONS),
            "images_per_condition": IMAGE_COUNT,
            "candidate_ids": list(contract.candidate_ids),
            "development_only": True,
            "paper_result": False,
            "p5_authorized": False,
        }
    elif args.command == "candidate":
        result = run_candidate(
            contract,
            dataset=args.dataset,
            device_name=args.device,
            max_images=args.max_images,
        )
    elif args.command == "outer":
        result = run_outer(contract, dataset=args.dataset)
    elif args.command == "aggregate":
        result = run_aggregate(contract)
    elif args.command == "verify":
        result = verify_all(contract)
    else:  # pragma: no cover - argparse owns the command choices.
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_IDS",
    "CONDITIONS",
    "DATASETS",
    "DEFAULT_CONFIG",
    "ExistingArtifactError",
    "P4Contract",
    "P4ProtocolError",
    "build_parser",
    "load_contract",
    "main",
    "run_aggregate",
    "run_candidate",
    "run_outer",
    "verify_all",
    "verify_artifact",
]
