"""Fail-closed raw-record Gate-LF finalization; never decode image/mask payloads.

Scientific eligibility is only an augmentation-visibility screen. It cannot
authorize IPMA meta-training, full source training, a test run, or paper claims.
Shared frozen-contract verification hashes allowlisted source-train files, but
all risk calculations use sealed JSON records, not newly decoded images/masks.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
VIEWS = ("full_256", "train_crop_224")
VARIANTS = ("L1", "L2", "L3", "L4a", "L4b")
PROBES = ("clean",) + VARIANTS
FORBIDDEN_ACCESS = ("test_split_reads", "test_image_opens", "test_mask_opens",
                    "validation_split_reads", "validation_image_opens", "validation_mask_opens")
MIN_CONTRAST = 1.0 / 255.0
EPSILON = 1.0e-8
RETENTION_MIN = 0.30
RISK_MAX = 0.10
RMS_FLOOR = 1.0e-6
RMS_FRACTION_MIN = 0.90


class GateInputError(ValueError):
    """Incomplete, changed, or invalid evidence must never be a scientific pass."""


def assert_finite_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GateInputError(f"non-finite evidence: {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{path}[{index}]")


def required_integer(row: Mapping[str, Any], key: str, minimum: int = 0) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GateInputError(f"missing or invalid integer {key}")
    return value


def required_number(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GateInputError(f"missing or non-finite numeric {key}")
    return float(value)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected) or json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise GateInputError(f"missing or inconsistent {label}: {actual!r} != {expected!r}")


def require_close(actual: Any, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise GateInputError(f"missing or non-finite {label}")
    if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12):
        raise GateInputError(f"raw/recomputed disagreement for {label}")


def validate_access(access: Mapping[str, Any], *, expected_train_opens: int = 128) -> None:
    for key in ("train_image_opens", "train_mask_opens"):
        require_equal(access.get(key), expected_train_opens, key)
    for key in FORBIDDEN_ACCESS:
        require_equal(access.get(key), 0, key)


def recompute_target(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the original E1 and the new compound risk on one denominator."""
    required_integer(row, "target_id", 1)
    required_integer(row, "target_pixels", 1)
    ring_pixels = required_integer(row, "ring_pixels")
    valid = ring_pixels > 0
    require_equal(row.get("ring_valid"), valid, "ring_valid")
    if valid:
        clean = required_number(row, "clean_contrast")
        post = required_number(row, "postclip_contrast")
        pre = required_number(row, "preclip_contrast")
        eligible = abs(clean) >= MIN_CONTRAST
        ratio = (abs(post) + EPSILON) / (abs(clean) + EPSILON)
        pre_ratio = (abs(pre) + EPSILON) / (abs(clean) + EPSILON)
        flip = clean * post < 0.0
        pre_flip = clean * pre < 0.0
        below = ratio < RETENTION_MIN
        require_close(row.get("postclip_contrast_retention_ratio"), ratio, "postclip retention")
        require_close(row.get("preclip_contrast_retention_ratio"), pre_ratio, "preclip retention")
        require_equal(row.get("postclip_contrast_sign_flip"), flip, "postclip sign flip")
        require_equal(row.get("preclip_contrast_sign_flip"), pre_flip, "preclip sign flip")
        require_equal(row.get("postclip_contrast_below_ratio_threshold"), below, "legacy E1 ratio flag")
    else:
        eligible = False
        ratio = pre_ratio = None
        flip = pre_flip = None
        below = None
        for key in ("clean_contrast", "preclip_contrast", "postclip_contrast",
                    "postclip_contrast_retention_ratio", "preclip_contrast_retention_ratio",
                    "postclip_contrast_sign_flip", "preclip_contrast_sign_flip",
                    "postclip_contrast_below_ratio_threshold"):
            if key not in row or row[key] is not None:
                raise GateInputError(f"empty ring must explicitly report undefined {key}")
    require_equal(row.get("contrast_eligible_for_e1"), eligible, "original E1 eligibility")
    e1_failure = bool(eligible and below)
    compound_failure = bool(eligible and (below or flip))
    # Supplemental flags are checked when supplied, never used as the evidence
    # from which risks are derived. They have eligibility-conditioned semantics.
    for key, expected in (("original_E1_failure", e1_failure if eligible else None),
                          ("compound_visibility_failure", compound_failure if eligible else None)):
        if key in row:
            require_equal(row[key], expected, key)
    return {"eligible": eligible, "e1_failure": e1_failure,
            "compound_failure": compound_failure, "ring_valid": valid,
            "low_clean_contrast": valid and not eligible,
            "retention_abs_postclip": ratio, "sign_flip_postclip": flip}


def evaluate_cell(image_rows: Sequence[Mapping[str, Any]], target_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(image_rows) != 64 or len({row["image_id"] for row in image_rows}) != 64:
        raise GateInputError("each dataset/view/probe cell requires exactly 64 unique images")
    results = [recompute_target(row) for row in target_rows]
    eligible = sum(row["eligible"] for row in results)
    e1 = sum(row["e1_failure"] for row in results)
    compound = sum(row["compound_failure"] for row in results)
    rms_values = [required_number(row, "postclip_perturbation_rms_rgb") for row in image_rows]
    if any(value < 0.0 for value in rms_values):
        raise GateInputError("negative image RMS")
    rms_above = sum(value > RMS_FLOOR for value in rms_values)
    median = statistics.median(rms_values)
    informative = median > RMS_FLOOR and rms_above / 64 >= RMS_FRACTION_MIN
    reasons = []
    if not eligible:
        reasons.append("insufficient_evidence_empty_original_e1_denominator")
    if eligible and e1 / eligible > RISK_MAX:
        reasons.append("original_e1_risk_gt_0.10")
    if eligible and compound / eligible > RISK_MAX:
        reasons.append("compound_visibility_risk_gt_0.10")
    if not informative:
        reasons.append("noninformative_probe")
    return {
        "image_count": 64, "target_count": len(target_rows), "eligible_target_count": eligible,
        "invalid_ring_target_count": sum(not row["ring_valid"] for row in results),
        "low_clean_contrast_target_count": sum(row["low_clean_contrast"] for row in results),
        "original_e1_failure_count": e1, "original_e1_denominator": eligible,
        "original_e1_failure_fraction": e1 / eligible if eligible else None,
        "compound_failure_count": compound, "compound_denominator": eligible,
        "compound_failure_fraction": compound / eligible if eligible else None,
        "postclip_rms_median": median, "images_above_rms_floor": rms_above,
        "fraction_above_rms_floor": rms_above / 64, "image_signal_floor_passed": informative,
        "visibility_risk_passed": bool(eligible and e1 / eligible <= RISK_MAX and compound / eligible <= RISK_MAX),
        "candidate_cell_passed": not reasons, "failure_reasons": reasons,
    }


def select_candidate(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {}
    for cell in cells:
        key = (cell["dataset"], cell["view"], cell["probe_id"])
        if key in indexed:
            raise GateInputError(f"duplicate gate cell: {key}")
        indexed[key] = cell
    expected = {(dataset, view, probe) for dataset in DATASETS for view in VIEWS for probe in PROBES}
    if set(indexed) != expected:
        raise GateInputError("missing or unexpected dataset/view/probe gate cells")
    candidates = {}
    for probe in ("L4a", "L4b"):
        candidate_cells = [indexed[(dataset, view, probe)] for dataset in DATASETS for view in VIEWS]
        eligible = all(cell["candidate_cell_passed"] for cell in candidate_cells)
        candidates[probe] = {
            "risk_and_image_floor_eligible": eligible,
            "failed_cells": [{"dataset": cell["dataset"], "view": cell["view"], "reasons": cell["failure_reasons"]}
                             for cell in candidate_cells if not cell["candidate_cell_passed"]],
            "alternate_eligible": probe == "L4b" and eligible,
        }
    selected = "L4a" if candidates["L4a"]["risk_and_image_floor_eligible"] else None
    if selected:
        status = "visibility_screen_passed_signal_smoke_only"
    elif candidates["L4b"]["risk_and_image_floor_eligible"]:
        status = "no_selection_l4b_alternate_requires_separate_unlabeled_signal_evidence"
    else:
        status = "visibility_screen_failed"
    return {
        "protocol_id": "cr-sitta-lf-repair-audit-v2",
        "scientific_scope": "augmentation_visibility_screen_only", "status": status,
        "selected_probe_id": selected, "candidates": candidates,
        "selection_policy": "only_preregistered_L4a_can_be_selected_in_R3_no_automatic_L4b_fallback",
        "signal_smoke_16_allowed": selected is not None,
        "ipma_meta_training_allowed": False, "full_source_training_allowed": False,
        "formal_test_allowed": False, "paper_result": False, "development_only": True,
        "no_validation_split": True,
        "limitations": ["visibility_and_image_floor_not_detection_performance",
                        "image_signal_floor_not_evidence_of_model_proxy_signal",
                        "no_training_or_next_stage_is_launched_by_finalization"],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise GateInputError(f"blank raw-record line: {path}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise GateInputError(f"non-object raw record: {path}")
            assert_finite_tree(row)
            rows.append(row)
    return rows


def identity(row: Mapping[str, Any], *, target: bool = False, probe: bool = True) -> tuple[Any, ...]:
    keys = ("dataset", "image_id", "view") + (("probe_id",) if probe else ()) + (("target_id",) if target else ())
    if any(key not in row for key in keys):
        raise GateInputError("record identity is incomplete")
    return tuple(row[key] for key in keys)


def unique_index(rows: Sequence[Mapping[str, Any]], *, target: bool = False, probe: bool = True) -> dict:
    indexed = {}
    for row in rows:
        key = identity(row, target=target, probe=probe)
        if key in indexed:
            raise GateInputError(f"duplicate raw-record identity: {key}")
        indexed[key] = row
    return indexed


def expected_operator_hash(config: Mapping[str, Any], probe: str) -> str:
    candidate = {"probe_id": "clean"} if probe == "clean" else next(row for row in config["variants"] if row["probe_id"] == probe)
    payload = json.dumps({**config["operator"], **candidate}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def expected_probe_seed(config: Mapping[str, Any], dataset: str, image_id: str, view: str) -> int:
    payload = json.dumps([config["probe_seed_namespace"], config["global_seed"], dataset, image_id, view],
                         ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def evaluate_records(
    image_rows: Sequence[Mapping[str, Any]], target_rows: Sequence[Mapping[str, Any]], *,
    legacy_images: Sequence[Mapping[str, Any]], legacy_targets: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any], access_by_dataset: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Check complete pairing against sealed L0, then recompute all 36 cells."""
    for label, rows in (("images", image_rows), ("targets", target_rows),
                        ("legacy_images", legacy_images), ("legacy_targets", legacy_targets)):
        assert_finite_tree(rows, label)
    require_equal(list(config["datasets"]), list(DATASETS), "preregistered datasets")
    require_equal(list(config["views"]), list(VIEWS), "preregistered views")
    require_equal([row["probe_id"] for row in config["variants"]], list(VARIANTS), "preregistered variants")
    if set(access_by_dataset) != set(DATASETS):
        raise GateInputError("missing dataset access audit")
    for dataset in DATASETS:
        validate_access(access_by_dataset[dataset])
    old_images = unique_index([row for row in legacy_images if row["probe_id"] == "lf_mask"], probe=False)
    old_targets = unique_index([row for row in legacy_targets if row["probe_id"] == "lf_mask"], target=True, probe=False)
    by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    for dataset, image_id, view in old_images:
        if dataset not in DATASETS or view not in VIEWS:
            raise GateInputError("unexpected sealed legacy dataset or view")
        by_cell[(dataset, view)].add(image_id)
    if set(by_cell) != {(dataset, view) for dataset in DATASETS for view in VIEWS} or any(len(ids) != 64 for ids in by_cell.values()):
        raise GateInputError("sealed legacy source must contain all three Pilot64/two-view cells")
    for dataset in DATASETS:
        if by_cell[(dataset, VIEWS[0])] != by_cell[(dataset, VIEWS[1])]:
            raise GateInputError("sealed legacy views use different Pilot64 IDs")
    indexed_images = unique_index(image_rows)
    expected_images = {(dataset, image_id, view, probe) for dataset, image_id, view in old_images for probe in PROBES}
    if set(indexed_images) != expected_images:
        raise GateInputError("missing or unexpected 64-image dataset/view/variant cell coverage")
    indexed_targets = unique_index(target_rows, target=True)
    expected_targets = {(dataset, image_id, view, probe, target_id)
                        for dataset, image_id, view, target_id in old_targets for probe in PROBES}
    if set(indexed_targets) != expected_targets:
        raise GateInputError("new target rows differ from the full sealed target enumeration")
    target_counts = defaultdict(int)
    image_groups: dict[tuple[str, str, str], list] = defaultdict(list)
    target_groups: dict[tuple[str, str, str], list] = defaultdict(list)
    for row in target_rows:
        dataset, image_id, view, probe, target_id = identity(row, target=True)
        target_counts[(dataset, image_id, view, probe)] += 1
        old = old_targets[(dataset, image_id, view, target_id)]
        for field in ("target_pixels", "ring_pixels", "bbox_yxyx_exclusive", "ring_valid",
                      "contrast_eligible_for_e1", "clean_contrast"):
            if field not in row or field not in old:
                raise GateInputError(f"missing target pairing field: {field}")
            require_equal(row[field], old[field], f"sealed target {field}")
        recompute_target(row)
        target_groups[(dataset, view, probe)].append(row)
    for row in image_rows:
        dataset, image_id, view, probe = identity(row)
        old = old_images[(dataset, image_id, view)]
        if not isinstance(row.get("input_metadata"), dict):
            raise GateInputError("input/GT hash or deterministic transform metadata differs from sealed P2")
        require_equal(row["input_metadata"], old.get("input_metadata"), "sealed input/GT/transform metadata")
        metadata = old["input_metadata"]
        for field, expected in (("input_hash", metadata["input_tensor_sha256"]),
                                ("target_hash", metadata["target_tensor_sha256"]),
                                ("operator_config_sha256", expected_operator_hash(config, probe)),
                                ("probe_seed", None if probe == "clean" else expected_probe_seed(config, dataset, image_id, view)),
                                ("sealed_input_pair_exact", True), ("gt_tensor_unchanged", True),
                                ("deterministic_repeat_exact", True), ("preclip_clamp_parity_exact", True)):
            require_equal(row.get(field), expected, field)
        count = required_integer(row, "target_count")
        require_equal(count, required_integer(old, "target_count"), "sealed target count")
        require_equal(count, target_counts[(dataset, image_id, view, probe)], "enumerated target count")
        rms = required_number(row, "postclip_perturbation_rms_rgb")
        if "postclip_rms" in row:
            require_close(row["postclip_rms"], rms, "operator/visibility image RMS")
        if probe == "clean" and rms != 0:
            raise GateInputError("clean control is not identity")
        for key in ("clean_physical_tensor_sha256", "postclip_physical_tensor_sha256", "preclip_physical_tensor_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise GateInputError(f"missing or invalid tensor hash: {key}")
        if probe == "clean" and len({row[key] for key in ("clean_physical_tensor_sha256", "preclip_physical_tensor_sha256", "postclip_physical_tensor_sha256")}) != 1:
            raise GateInputError("clean identity tensor hashes differ")
        image_groups[(dataset, view, probe)].append(row)
    # Per-target descriptors must bind the same operator, seed, image and mask as
    # their parent per-image record. A mask hash is never inferred from filenames.
    for row in target_rows:
        parent = indexed_images[identity(row)]
        for field in ("input_hash", "target_hash", "operator_config_sha256", "probe_seed"):
            require_equal(row.get(field), parent[field], f"target/image {field}")
    cells = []
    for dataset in DATASETS:
        for view in VIEWS:
            for probe in PROBES:
                key = (dataset, view, probe)
                cells.append({"dataset": dataset, "view": view, "probe_id": probe,
                              **evaluate_cell(image_groups[key], target_groups[key])})
    return cells, select_candidate(cells)


def validate_r2_freeze(
    freeze: Mapping[str, Any], run_contract: Mapping[str, Any], *,
    config: Mapping[str, Any], legacy_visibility: Mapping[str, Any], dataset: str,
    preregistration: Mapping[str, Any], preregistration_binding: Mapping[str, str],
    run_contract_binding: Mapping[str, str],
) -> None:
    """A self-consistent seal is insufficient unless it binds this preregistration."""
    require_equal(freeze.get("contract"), dict(run_contract_binding), "R2 own run-contract binding")
    actual = freeze.get("input_bindings")
    if not isinstance(actual, list):
        raise GateInputError("missing R2 frozen input bindings")
    indexed = {}
    for item in actual:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise GateInputError("malformed R2 input binding")
        if item["path"] in indexed:
            raise GateInputError("duplicate R2 input binding")
        indexed[item["path"]] = item["sha256"]
    for item in [*preregistration["input_bindings"], dict(preregistration_binding)]:
        if indexed.get(item["path"]) != item["sha256"]:
            raise GateInputError("R2 freeze does not include this protocol's preregistered inputs")
    for field, expected in (("dataset", dataset), ("protocol_id", config["protocol_id"]),
                            ("variants", config["variants"]), ("operator", config["operator"]),
                            ("gate", config["gate"]), ("visibility", dict(legacy_visibility)),
                            ("stage_scope", config["stage_scope"]),
                            ("scientific_scope", config["scientific_scope"]),
                            ("paper_result", False), ("checkpoint_used", None)):
        require_equal(run_contract.get(field), expected, f"R2 preregistered {field}")


def validate_random_fields(field_rows: Sequence[Mapping[str, Any]], image_rows: Sequence[Mapping[str, Any]],
                           config: Mapping[str, Any]) -> None:
    """Verify the stored float64 fields without loading images or running FFT."""
    import numpy as np
    assert_finite_tree(field_rows)
    indexed = unique_index(field_rows, probe=False)
    expected = {identity(row, probe=False) for row in image_rows}
    if set(indexed) != expected:
        raise GateInputError("random-field sidecar does not cover every unique image/view")
    grouped = defaultdict(list)
    for row in image_rows:
        grouped[identity(row, probe=False)].append(row)
    for key, field in indexed.items():
        dataset, image_id, view = key
        require_equal(field.get("probe_seed"), expected_probe_seed(config, dataset, image_id, view), "random-field seed")
        require_equal(field.get("shared_by"), list(VARIANTS), "random-field shared candidates")
        shape = field.get("random_field_shape")
        if not isinstance(shape, list) or len(shape) != 3 or shape[:2] != [1, 3] or isinstance(shape[2], bool) or not isinstance(shape[2], int) or shape[2] <= 0:
            raise GateInputError("random field must have shape [1,3,positive_orbit_count]")
        try:
            values = np.asarray(field["random_field_values"])
        except (KeyError, ValueError, TypeError) as error:
            raise GateInputError("missing or malformed random-field values") from error
        if values.dtype != np.float64 or list(values.shape) != shape:
            raise GateInputError("random-field dtype or shape drift")
        if not np.isfinite(values).all() or not ((values >= 0) & (values < 1)).all():
            raise GateInputError("random-field values must be finite float64 uniforms in [0,1)")
        descriptor = json.dumps({"schema": "lf_v2_dtype_shape_contiguous_bytes_v1",
                                 "dtype": "torch.float64", "shape": shape},
                                sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(descriptor + b"\n" + values.tobytes(order="C")).hexdigest()
        require_equal(field.get("random_field_sha256"), digest, "random-field sidecar hash")
        variants = [row for row in grouped[key] if row["probe_id"] != "clean"]
        if {row["probe_id"] for row in variants} != set(VARIANTS) or len(variants) != 5:
            raise GateInputError("random-field image group lacks exactly five variants")
        for row in variants:
            require_equal(row.get("random_field_sha256"), digest, "candidate shared random-field hash")
            require_equal(row.get("random_field_shape"), shape, "candidate shared random-field shape")
            require_equal(row.get("tensor_hash_schema"), "lf_v2_dtype_shape_contiguous_bytes_v1", "random-field hash schema")


def finalize(config_path: Path, *, execute: bool = False) -> dict[str, Any]:
    from analysis import d0a_v7_common as artifacts
    from analysis import lf_repair_contract_v2 as contract

    config_path = config_path.resolve()
    config = contract.read_config(config_path)
    preregistration = contract.validate_preregistration(config_path)
    legacy_config = artifacts.read_config(config["legacy_config"])
    result_root = artifacts.absolute(config["result_root"])
    output = result_root / "finalization"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing existing full or partial LF finalization: {output}")
    parent_root = artifacts.absolute(config["parent_result_root"]) / "probe_label_preservation"
    bindings = list(preregistration["input_bindings"])
    bindings.extend(artifacts.binding(path) for path in (
        result_root / "PREREGISTRATION.json", result_root / "P0_P3_EVIDENCE_BINDING.json",
        config_path, Path(__file__).resolve(),
    ))
    image_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    legacy_images: list[dict[str, Any]] = []
    legacy_targets: list[dict[str, Any]] = []
    access_by_dataset = {}
    manifest_hashes = {}
    for dataset in DATASETS:
        dataset_output = result_root / dataset
        verified = contract.verify_complete_dir(dataset_output)
        required = {"per_image.jsonl", "per_target.jsonl", "random_fields.jsonl",
                    "legacy_L0_per_image.jsonl", "legacy_L0_per_target.jsonl"}
        if not required <= set(verified["manifest"]["files"]):
            raise GateInputError(f"required raw audit artifact missing: {dataset}")
        bindings.extend(verified["bindings"])
        r2_freeze = contract.read_json(dataset_output / "PRE_RUN_FREEZE.json")
        r2_contract = contract.read_json(dataset_output / "RUN_CONTRACT.json")
        validate_r2_freeze(r2_freeze, r2_contract, config=config,
            legacy_visibility=legacy_config["visibility"], dataset=dataset,
            preregistration=preregistration,
            preregistration_binding=artifacts.binding(result_root / "PREREGISTRATION.json"),
            run_contract_binding=artifacts.binding(dataset_output / "RUN_CONTRACT.json"))
        contract.assert_bindings(r2_freeze["input_bindings"] + [r2_freeze["contract"]])
        summary = verified["summary"]
        assert_finite_tree(summary)
        require_equal(summary.get("dataset"), dataset, "R2 dataset")
        require_equal(summary.get("protocol_id"), config["protocol_id"], "R2 protocol")
        for key in ("training_started", "paper_result", "ipma_meta_training_allowed",
                    "full_source_training_allowed", "formal_test_allowed"):
            require_equal(summary.get(key), False, f"R2 {key}")
        if not isinstance(summary.get("access"), dict):
            raise GateInputError(f"missing actual data-access counters: {dataset}")
        access_by_dataset[dataset] = summary["access"]
        validate_access(summary["access"])
        manifest_hashes[dataset] = artifacts.sha256_file(dataset_output / "artifact_manifest.json")
        dataset_images = None
        for filename, destination in (("per_image.jsonl", image_rows), ("per_target.jsonl", target_rows)):
            rows = read_jsonl(dataset_output / filename)
            if any(row.get("dataset") != dataset for row in rows):
                raise GateInputError("raw records escaped their dataset artifact")
            destination.extend(rows)
            if filename == "per_image.jsonl":
                dataset_images = rows
        validate_random_fields(read_jsonl(dataset_output / "random_fields.jsonl"), dataset_images, config)
        parent = parent_root / dataset
        parent_verified = contract.verify_complete_dir(
            parent, config["parent_manifests"][f"probe_label_preservation/{dataset}"])
        if not {"per_image.jsonl", "per_target.jsonl"} <= set(parent_verified["manifest"]["files"]):
            raise GateInputError("sealed parent lacks raw target/image evidence")
        bindings.extend(parent_verified["bindings"])
        old_image_rows = read_jsonl(parent / "per_image.jsonl")
        old_target_rows = read_jsonl(parent / "per_target.jsonl")
        old_lf_images = [row for row in old_image_rows if row["probe_id"] == "lf_mask"]
        old_lf_targets = [row for row in old_target_rows if row["probe_id"] == "lf_mask"]
        if unique_index(read_jsonl(dataset_output / "legacy_L0_per_image.jsonl")) != unique_index(old_lf_images):
            raise GateInputError("archived L0 image rows differ from the sealed parent")
        if unique_index(read_jsonl(dataset_output / "legacy_L0_per_target.jsonl"), target=True) != unique_index(old_lf_targets, target=True):
            raise GateInputError("archived L0 target rows differ from the sealed parent")
        legacy_images.extend(old_lf_images)
        legacy_targets.extend(old_lf_targets)
    bindings = contract.deduplicate(bindings)
    cells, gate = evaluate_records(image_rows, target_rows, legacy_images=legacy_images,
                                  legacy_targets=legacy_targets, config=config,
                                  access_by_dataset=access_by_dataset)
    receipt = {
        **gate, "all_required_artifacts_verified": True,
        "gate_recomputed_from_raw_records": True, "raw_summary_pass_flags_used": False,
        "required_variant_cells": 30, "clean_control_cells": 6, "total_cells": len(cells),
        "per_image_rows": len(image_rows), "per_target_rows": len(target_rows),
        "parent_diagnostic_manifest_sha256": dict(config["parent_manifests"]),
        "r2_dataset_manifest_sha256": manifest_hashes,
        "preregistration_sha256": artifacts.sha256_file(result_root / "PREREGISTRATION.json"),
        "r0_evidence_binding_sha256": artifacts.sha256_file(result_root / "P0_P3_EVIDENCE_BINDING.json"),
        "approved_operator_sha256": artifacts.sha256_file(ROOT / "tta/deteriorations/fourier_low_mask_v2.py"),
        "access_by_dataset": access_by_dataset,
        "new_image_decodes_in_finalizer": 0, "new_mask_decodes_in_finalizer": 0,
        "contract_verification_hashes_allowlisted_train_files": True,
        "test_payload_opens": 0, "validation_payload_opens": 0,
    }
    selected = gate["selected_probe_id"]
    approved = None
    if selected is not None:
        variant = next(row for row in config["variants"] if row["probe_id"] == selected)
        approved = {**config["operator"], **variant,
                    "operator_config_sha256": expected_operator_hash(config, selected),
                    "operator_source_sha256": receipt["approved_operator_sha256"],
                    "selection_scope": "one_global_probe_all_datasets_images_views",
                    "scientific_scope": gate["scientific_scope"],
                    "signal_smoke_16_allowed": True, "ipma_meta_training_allowed": False,
                    "full_source_training_allowed": False, "formal_test_allowed": False,
                    "paper_result": False}
    receipt["approved_probe"] = approved
    if not execute:
        return {"ready": True, "writes": 0, "execute": False, "output_dir": str(output),
                "gate_preview": receipt}
    # The criteria and candidate order were already frozen before R2. This
    # independent derived receipt additionally binds every raw input read above.
    contract.assert_bindings(bindings)
    artifacts.reserve_output(output)
    artifacts.freeze_run(output, {
        "protocol_id": config["protocol_id"], "role": "R3_independent_raw_record_gate_finalization",
        "scientific_scope": gate["scientific_scope"], "development_only": True,
        "paper_result": False, "no_validation_split": True,
        "criteria_source": "unchanged_preregistration_before_R2",
        "stage_scope": config["stage_scope"], "gate": config["gate"],
        "selection": config["selection"], "all_next_stage_execution_disabled": True,
        "test_payload_opens": 0, "new_image_or_mask_decodes": 0,
    }, bindings)
    artifacts.write_json_new(output / "recomputed_cells.json", {"cells": cells})
    artifacts.write_json_new(output / "GATE_LF_RECEIPT.json", receipt)
    if approved is not None:
        artifacts.write_json_new(output / "APPROVED_LF_CONFIG.json", approved)
    contract.complete_output(output, receipt)
    return {"complete": True, "output_dir": str(output), **receipt}


def main() -> None:
    from analysis.lf_repair_contract_v2 import DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(finalize(args.config, execute=args.execute), ensure_ascii=False,
                     indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
