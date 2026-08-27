"""Audit three formal corruption Pilots and apply the frozen calibration gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from corruptions.corruption_protocol import NON_CLEAN_CORRUPTIONS
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "results" / "corruption_pilot_fixed_split"
DEFAULT_CRITERIA = PROJECT_ROOT / "configs" / "corruption_pilot_acceptance.yaml"
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
REQUIRED_TRUE_CHECKS = (
    "formal_contract_satisfied",
    "same_ordered_ids_all_conditions",
    "fixed_test_ids_absent",
    "model_state_unchanged",
    "exact_input_reproduction_all_conditions",
    "gt_mask_hash_identical_across_conditions",
    "clean_transform_exact_identity",
    "severity_table_unchanged",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _load_json(path: Path) -> Mapping[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return loaded


def _verify_artifact(dataset_dir: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    pilot_path = dataset_dir / "pilot.json"
    manifest_path = dataset_dir / "artifact_manifest.json"
    complete_path = dataset_dir / "COMPLETE.json"
    for path in (pilot_path, manifest_path, complete_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing Pilot artifact: {path}")
    pilot = _load_json(pilot_path)
    manifest = _load_json(manifest_path)
    complete = _load_json(complete_path)
    mismatches = []
    for relative_path, expected in manifest["files_sha256"].items():
        path = dataset_dir / relative_path
        actual = source_runner.sha256_file(path)
        if actual != expected:
            mismatches.append(
                {"path": relative_path, "expected": expected, "actual": actual}
            )
    pilot_hash = source_runner.sha256_file(pilot_path)
    valid = (
        not mismatches
        and complete.get("complete") is True
        and complete.get("formal_artifact") is True
        and complete.get("pilot_json_sha256") == pilot_hash
        and complete.get("artifact_manifest_sha256")
        == source_runner.sha256_file(manifest_path)
    )
    return pilot, {
        "valid": valid,
        "file_hash_mismatches": mismatches,
        "pilot_json_sha256": pilot_hash,
        "artifact_manifest_sha256": source_runner.sha256_file(manifest_path),
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 1.0
    severity_ranks = np.arange(1, len(values) + 1, dtype=np.float64)
    value_ranks = _rankdata(values)
    if float(np.std(value_ranks)) == 0.0:
        return 0.0
    return float(np.corrcoef(severity_ranks, value_ranks)[0, 1])


def _difficulty(clean_iou: float, clean_pd: float, iou: float, pd: float) -> float:
    iou_drop = np.clip((clean_iou - iou) / max(clean_iou, 1e-12), -1.0, 1.0)
    pd_drop = np.clip((clean_pd - pd) / max(clean_pd, 1e-12), -1.0, 1.0)
    return float(0.65 * iou_drop + 0.35 * pd_drop)


def _condition_map(pilot: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    records = {
        (str(record["corruption"]), int(record["severity"])): record
        for record in pilot["conditions"]
    }
    expected = {("clean", 0)} | {
        (corruption, severity)
        for corruption in NON_CLEAN_CORRUPTIONS
        for severity in range(1, 6)
    }
    if set(records) != expected:
        raise ValueError("Pilot does not contain the exact 21-condition grid")
    return records


def _hard_validity(pilot: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    checks = pilot["checks"]
    failures = [name for name in REQUIRED_TRUE_CHECKS if checks.get(name) is not True]
    if checks.get("test_image_open_count") != 0:
        failures.append("test_image_open_count")
    if checks.get("test_mask_open_count") != 0:
        failures.append("test_mask_open_count")
    if checks.get("io_guard", {}).get("forbidden_open_count") != 0:
        failures.append("io_guard.forbidden_open_count")
    if pilot.get("condition_count") != 21 or pilot.get("sample_condition_count") != 1344:
        failures.append("21_conditions_x_64_images")
    if not artifact["valid"]:
        failures.append("artifact_integrity")
    finite = all(
        np.isfinite(float(record["metrics"]["fixed"][metric]))
        for record in pilot["conditions"]
        for metric in ("iou", "pd", "fa_pixel_rate")
    )
    if not finite:
        failures.append("finite_metrics")
    return {"passed": not failures, "failures": failures}


def _evaluate_corruption(
    records: Mapping[tuple[str, int], Mapping[str, Any]],
    corruption: str,
    criteria: Mapping[str, Any],
) -> dict[str, Any]:
    clean = records[("clean", 0)]["metrics"]["fixed"]
    clean_iou = float(clean["iou"])
    clean_pd = float(clean["pd"])
    severity_records = [records[(corruption, severity)] for severity in range(1, 6)]
    ious = [float(record["metrics"]["fixed"]["iou"]) for record in severity_records]
    pds = [float(record["metrics"]["fixed"]["pd"]) for record in severity_records]
    fas = [
        float(record["metrics"]["fixed"]["fa_per_million_pixels"])
        for record in severity_records
    ]
    difficulty = [
        _difficulty(clean_iou, clean_pd, iou, pd) for iou, pd in zip(ious, pds)
    ]
    gate = criteria["per_dataset_corruption"]
    s1_light = (
        clean_iou - ious[0] <= float(gate["s1_max_absolute_iou_drop"])
        and clean_pd - pds[0] <= float(gate["s1_max_absolute_pd_drop"])
    )
    upper_lower_margin = float(np.mean(difficulty[3:5]) - np.mean(difficulty[0:2]))
    overall_harder = (
        upper_lower_margin
        >= float(gate["upper_tier_min_difficulty_margin_over_lower_tier"])
        and difficulty[4] >= float(gate["s5_min_difficulty_score"])
    )
    s5_significant = (
        ious[4] / max(clean_iou, 1e-12)
        <= 1.0 - float(gate["s5_min_relative_degradation"])
        or pds[4] / max(clean_pd, 1e-12)
        <= 1.0 - float(gate["s5_min_relative_degradation"])
    )
    s5_noncollapse = (
        ious[4]
        >= max(
            float(gate["s5_min_iou_absolute"]),
            float(gate["s5_min_iou_fraction_of_clean"]) * clean_iou,
        )
        and pds[4]
        >= max(
            float(gate["s5_min_pd_absolute"]),
            float(gate["s5_min_pd_fraction_of_clean"]) * clean_pd,
        )
    )
    adjacent_reversals = [
        index + 1
        for index, (left, right) in enumerate(zip(difficulty, difficulty[1:]))
        if right - left < -float(gate["adjacent_difficulty_reversal_warning_above"])
    ]
    spearman = _spearman(difficulty)
    warnings = []
    if spearman < float(gate["spearman_warning_below"]):
        warnings.append("low_severity_difficulty_spearman")
    if adjacent_reversals:
        warnings.append("large_adjacent_difficulty_reversal")
    passed = s1_light and overall_harder and s5_significant and s5_noncollapse
    return {
        "passed": passed,
        "s1_light": s1_light,
        "overall_harder": overall_harder,
        "s5_significant_degradation": s5_significant,
        "s5_noncollapsed": s5_noncollapse,
        "difficulty_score_s1_to_s5": difficulty,
        "upper_minus_lower_tier_difficulty": upper_lower_margin,
        "difficulty_spearman": spearman,
        "adjacent_reversal_after_severity": adjacent_reversals,
        "warnings": warnings,
        "clean": {"iou": clean_iou, "pd": clean_pd},
        "s1_to_s5": {"iou": ious, "pd": pds, "fa_per_million_pixels": fas},
        "s5_fraction_of_clean": {
            "iou": ious[4] / max(clean_iou, 1e-12),
            "pd": pds[4] / max(clean_pd, 1e-12),
        },
    }


def analyze(input_root: Path, criteria_path: Path) -> dict[str, Any]:
    criteria = yaml.safe_load(criteria_path.read_text(encoding="utf-8"))
    pilots: dict[str, Mapping[str, Any]] = {}
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        pilot, artifact = _verify_artifact(input_root / dataset)
        pilots[dataset] = pilot
        records = _condition_map(pilot)
        datasets[dataset] = {
            "artifact": artifact,
            "hard_validity": _hard_validity(pilot, artifact),
            "corruptions": {
                corruption: _evaluate_corruption(records, corruption, criteria)
                for corruption in NON_CLEAN_CORRUPTIONS
            },
        }

    protocol_hashes = sorted({str(pilot["protocol_sha256"]) for pilot in pilots.values()})
    severity_hashes = sorted(
        {str(pilot["checks"]["severity_table_sha256_before"]) for pilot in pilots.values()}
    )
    shared = {}
    for corruption in NON_CLEAN_CORRUPTIONS:
        per_dataset = {
            dataset: datasets[dataset]["corruptions"][corruption]
            for dataset in DATASETS
        }
        pass_count = sum(record["passed"] for record in per_dataset.values())
        collapsed = [
            dataset
            for dataset, record in per_dataset.items()
            if not record["s5_noncollapsed"]
        ]
        passed = (
            pass_count >= int(criteria["shared_table"]["min_dataset_passes_per_corruption"])
            and (
                not criteria["shared_table"]["require_no_s5_collapse_in_any_dataset"]
                or not collapsed
            )
        )
        shared[corruption] = {
            "passed": passed,
            "dataset_pass_count": pass_count,
            "collapsed_datasets": collapsed,
            "failed_datasets": [
                dataset for dataset, record in per_dataset.items() if not record["passed"]
            ],
        }
    hard_validity_passed = (
        len(protocol_hashes) == 1
        and len(severity_hashes) == 1
        and all(record["hard_validity"]["passed"] for record in datasets.values())
    )
    freeze = hard_validity_passed and all(record["passed"] for record in shared.values())
    return {
        "schema_version": 1,
        "criteria_id": criteria["criteria_id"],
        "criteria_path": str(criteria_path.resolve()),
        "criteria_sha256": source_runner.sha256_file(criteria_path),
        "input_root": str(input_root.resolve()),
        "protocol_sha256": protocol_hashes,
        "severity_table_sha256": severity_hashes,
        "hard_validity_passed": hard_validity_passed,
        "datasets": datasets,
        "shared_table": shared,
        "decision": "freeze" if freeze else "adjust_and_rerun_pilot",
        "severity_table_may_be_frozen": freeze,
        "recoverability_claimed": False,
        "interpretation": criteria["interpretation"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = args.input_root.expanduser().resolve()
    criteria_path = args.criteria.expanduser().resolve()
    output = (
        args.output
        or input_root / "calibration_report.json"
    ).expanduser().resolve()
    report = analyze(input_root, criteria_path)
    source_runner.write_json_atomic(output, report)
    print(f"Artifact: {output}")
    print(f"Decision: {report['decision']}")
    for corruption, result in report["shared_table"].items():
        print(
            f"{corruption}: passed={result['passed']}, "
            f"dataset_passes={result['dataset_pass_count']}/3, "
            f"collapsed={result['collapsed_datasets']}"
        )
    return 0 if report["hard_validity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
