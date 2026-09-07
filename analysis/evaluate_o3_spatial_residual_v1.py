#!/usr/bin/env python3
"""Train-only performance evaluation after all three label-free candidates finish.

This module cannot adapt a model.  It evaluates saved probabilities with the
unchanged B4 evaluator, checks paired Source counts against frozen O3/P2, and
reports predeclared advancement goals separately from implementation checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts import run_p3_stage_b4_full_pilot64_v1 as b4
from scripts import run_p3_stage_b_screen_v1 as b3

CANDIDATE_ID = "O3_DecoderSpatialResidual"
DEFAULT_CONFIG = REPOSITORY / "configs/cr_sitta_o3_spatial_residual_v1.yaml"
METRICS = ("iou", "pd", "fa_per_million")
COUNT_FIELDS = b4._COUNT_FIELDS
FAMILIES = ("gaussian_noise", "gaussian_blur", "low_contrast", "stripe_noise")
TOLERANCE = 1e-12


class PerformanceEvaluationError(RuntimeError):
    """Missing, inconsistent, or mutable evidence must stop evaluation."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        handle.write("\n")


def endpoint_from_counts(value: Mapping[str, Any]) -> dict[str, Any]:
    """Derive old aggregate metrics; nIoU cannot be recovered from total counts."""
    if set(value) != set(COUNT_FIELDS):
        raise PerformanceEvaluationError("historical count field set differs")
    counts = dict(value)
    if any(type(counts[key]) is not int or counts[key] < 0 for key in COUNT_FIELDS):
        raise PerformanceEvaluationError("endpoint counts must be nonnegative integers")
    union = (counts["intersection_pixels"] + counts["false_positive_pixels"]
             + counts["false_negative_pixels"])
    total = counts["total_image_pixels"]
    metrics = {
        "iou": counts["intersection_pixels"] / union if union else 1.0,
        "pd": (counts["detected_targets"] / counts["total_targets"]
               if counts["total_targets"] else 0.0),
        "fa_per_million": counts["false_alarm_pixels"] / total * 1e6 if total else 0.0,
        "foreground_fraction": counts["predicted_positive_pixels"] / total if total else 0.0,
    }
    # Reuse the frozen count-conservation validator; the placeholder nIoU is
    # deliberately omitted from the returned historical endpoint.
    b4._validated_endpoint_summary(
        {**counts, **metrics, "normalized_iou": 0.0},
        label="historical O3/P2 endpoint", expected_image_count=64,
    )
    return {**counts, **metrics}


def load_historical_cells(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("candidate_id") != "O3_P2":
            continue
        key = (record.get("dataset"), record.get("condition"))
        if key in result:
            raise PerformanceEvaluationError(f"duplicate historical O3/P2 cell: {key}")
        if record.get("episode_count") != 64:
            raise PerformanceEvaluationError("historical cohort size differs")
        result[key] = {
            **record,
            "source": endpoint_from_counts(record["source_counts"]),
            "previous_o3": endpoint_from_counts(record["adapted_counts"]),
        }
    expected = {(d, b4._condition_key(*c)) for d in b4.DATASETS for c in b4.CONDITIONS}
    if set(result) != expected:
        raise PerformanceEvaluationError("historical 39-cell Cartesian keyset differs")
    for (dataset, condition), record in result.items():
        corruption, severity = next(c for c in b4.CONDITIONS if b4._condition_key(*c) == condition)
        if (record.get("corruption_family") != corruption
                or record.get("severity") != f"S{severity}"):
            raise PerformanceEvaluationError(f"historical condition identity differs: {dataset}/{condition}")
    return result


def assert_paired_source(source: Mapping[str, Any], historical: Mapping[str, Any]) -> None:
    """Every integer endpoint must agree, not merely rounded IoU or Pd."""
    for key in COUNT_FIELDS:
        if type(source.get(key)) is not int or source[key] != historical[key]:
            raise PerformanceEvaluationError(f"Source count differs from frozen B4: {key}")
    for key in METRICS:
        if not math.isclose(float(source[key]), float(historical[key]),
                            rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            raise PerformanceEvaluationError(f"Source metric differs from frozen B4: {key}")


def _comparison(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise PerformanceEvaluationError("cannot average an empty cell group")
    means = {
        endpoint: {metric: math.fsum(float(r[endpoint][metric]) for r in records) / len(records)
                   for metric in METRICS}
        for endpoint in ("source", "previous_o3", "adapted")
    }
    for endpoint in means.values():
        if not all(math.isfinite(v) for v in endpoint.values()):
            raise PerformanceEvaluationError("non-finite endpoint metric")
    result: dict[str, Any] = {"cell_count": len(records), **means}
    for reference in ("source", "previous_o3"):
        result[f"delta_vs_{reference}"] = {
            "iou_pp": 100.0 * (means["adapted"]["iou"] - means[reference]["iou"]),
            "pd_pp": 100.0 * (means["adapted"]["pd"] - means[reference]["pd"]),
            "fa_per_million": means["adapted"]["fa_per_million"] - means[reference]["fa_per_million"],
        }
    return result


def summarize_performance(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(d, b4._condition_key(*c)) for d in b4.DATASETS for c in b4.CONDITIONS}
    keys = [(r.get("dataset"), r.get("condition")) for r in cells]
    if len(keys) != 39 or set(keys) != expected:
        raise PerformanceEvaluationError("new 39-cell Cartesian keyset differs")
    for row in cells:
        corruption, severity = next(c for c in b4.CONDITIONS
                                    if b4._condition_key(*c) == row["condition"])
        if row.get("corruption_family") != corruption or row.get("severity") != severity:
            raise PerformanceEvaluationError("new condition identity differs")
    nonclean = [r for r in cells if r["corruption_family"] != "clean"]
    overall = _comparison(nonclean)
    families = {f: _comparison([r for r in nonclean if r["corruption_family"] == f])
                for f in FAMILIES}
    datasets = {d: _comparison([r for r in nonclean if r["dataset"] == d])
                for d in b4.DATASETS}
    clean = {d: _comparison([r for r in cells if r["dataset"] == d
                            and r["corruption_family"] == "clean"])
             for d in b4.DATASETS}
    goals: dict[str, bool] = {}
    for family in ("gaussian_noise", "gaussian_blur"):
        group = families[family]
        goals[f"{family}_iou_above_source"] = group["adapted"]["iou"] > group["source"]["iou"] + TOLERANCE
    for family in ("low_contrast", "stripe_noise"):
        group = families[family]
        goals[f"{family}_iou_preserves_previous_o3"] = group["adapted"]["iou"] + TOLERANCE >= group["previous_o3"]["iou"]
    goals["nonclean_iou_above_previous_o3"] = overall["adapted"]["iou"] > overall["previous_o3"]["iou"] + TOLERANCE
    goals["nonclean_pd_preserves_previous_o3"] = overall["adapted"]["pd"] + TOLERANCE >= overall["previous_o3"]["pd"]
    goals["nonclean_fa_not_above_source"] = overall["adapted"]["fa_per_million"] <= overall["source"]["fa_per_million"] + TOLERANCE
    for dataset, group in clean.items():
        for metric in ("iou", "pd"):
            goals[f"{dataset}_clean_{metric}_within_0p002"] = group["adapted"][metric] + 0.002 + TOLERANCE >= group["source"][metric]
        goals[f"{dataset}_clean_fa_not_above_source"] = group["adapted"]["fa_per_million"] <= group["source"]["fa_per_million"] + TOLERANCE
    return {
        "candidate_id": CANDIDATE_ID,
        "scope": "source-domain train Pilot64; no validation split; not formal test",
        "paper_result": False, "formal_test": False, "no_validation_split": True,
        "cell_count": 39, "nonclean_cell_count": 36,
        "averaging": "unweighted arithmetic mean of condition-level metrics; clean excluded from nonclean macros",
        "nonclean_macro": overall, "families_nonclean": families,
        "datasets_nonclean": datasets, "clean": clean,
        "advancement_goals": goals, "all_advancement_goals_met": all(goals.values()),
        "failed_advancement_goals": [key for key, passed in goals.items() if not passed],
        "claim_limits": [
            "Advancement goals are performance targets, not statistical-significance evidence.",
            "Comparison changes parameterization, update site and budget; it is not a matched-capacity or matched-budget causal ablation.",
            "Historical nIoU is unavailable from aggregate counts; no historical nIoU is fabricated.",
            "No automatic retry, hyperparameter revision, full training or formal test is authorized by this receipt.",
        ],
    }


def _probabilities(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if value.dtype != np.float32 or value.shape != (64, 1, 256, 256):
        raise PerformanceEvaluationError(f"probability shape/dtype differs: {path}")
    if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
        raise PerformanceEvaluationError(f"probability values invalid: {path}")
    return value


def _artifact_files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PerformanceEvaluationError(f"artifact contains symlink: {path}")
        if path.is_file() and path.name not in ("manifest.json", "COMPLETE.json"):
            result[str(path.relative_to(root))] = _sha(path)
    return result


def _input_bindings(config_path: Path, raw: Mapping[str, Any], candidate_roots: Mapping[str, Path]) -> dict[str, Any]:
    old_root = REPOSITORY / raw["parent_aggregate"]["path"]
    return {
        "config_sha256": _sha(config_path),
        "candidate_artifacts": {dataset: {"path": str(root.relative_to(REPOSITORY)),
                                           "manifest_sha256": _sha(root / "manifest.json"),
                                           "complete_sha256": _sha(root / "COMPLETE.json")}
                                for dataset, root in candidate_roots.items()},
        "historical_aggregate": {"path": str(old_root.relative_to(REPOSITORY)),
                                 "manifest_sha256": _sha(old_root / "manifest.json"),
                                 "complete_sha256": _sha(old_root / "COMPLETE.json"),
                                 "science_gate_cells_sha256": _sha(old_root / "science_gate_cells.jsonl")},
        "evaluator_sha256": _sha(Path(__file__).resolve()),
    }


def evaluate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    # This call is the global all-three-dataset barrier. Do not move any target
    # import, decode or label-based metric ahead of it.
    from analysis.spatial_residual_contract_v1 import verify_all_candidates

    config_path = Path(config_path).resolve()
    verified = verify_all_candidates(config_path=config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    parent = b4.load_contract(REPOSITORY / raw["parent_config"])
    result_root = REPOSITORY / raw["result_root"]
    candidate_roots = {d: result_root / "candidate" / d for d in b4.DATASETS}
    if set(verified) != set(b4.DATASETS):
        raise PerformanceEvaluationError("global candidate barrier returned an incomplete dataset set")
    bindings = _input_bindings(config_path, raw, candidate_roots)
    output = result_root / "outer" / "R0"
    if output.exists():
        if not (output / "COMPLETE.json").is_file():
            raise PerformanceEvaluationError(f"incomplete outer artifact already exists; preserved: {output}")
        manifest, complete = _json(output / "manifest.json"), _json(output / "COMPLETE.json")
        if (complete.get("manifest_sha256") != _sha(output / "manifest.json")
                or manifest.get("inputs") != bindings
                or manifest.get("files") != _artifact_files(output)):
            raise PerformanceEvaluationError("existing outer artifact differs from its bound evidence")
        return {"status": "existing_verified_complete_no_op", "path": str(output),
                "summary": _json(output / "summary.json")}
    historical = load_historical_cells(REPOSITORY / raw["parent_aggregate"]["path"] / "science_gate_cells.jsonl")
    summaries = {d: _json(candidate_roots[d] / "summary.json") for d in b4.DATASETS}
    for dataset, summary in summaries.items():
        image_ids = summary.get("image_ids")
        if (not isinstance(image_ids, list) or len(image_ids) != 64
                or len(set(image_ids)) != 64 or not all(isinstance(v, str) for v in image_ids)):
            raise PerformanceEvaluationError(f"candidate ordered 64 image IDs invalid: {dataset}")
    output.mkdir(parents=True, exist_ok=False)
    cells: list[dict[str, Any]] = []
    targets_loaded = 0
    try:
        for dataset in b4.DATASETS:
            # All three candidate artifacts are already immutable and verified.
            targets = b3._load_outer_targets(parent, dataset)
            targets_loaded += 1
            image_ids = summaries[dataset]["image_ids"]
            for corruption, severity in b4.CONDITIONS:
                condition = b4._condition_key(corruption, severity)
                arrays = candidate_roots[dataset] / "conditions" / condition
                source = b3._evaluation_result(_probabilities(arrays / "source_probabilities.npy"), targets, image_ids)
                adapted = b3._evaluation_result(_probabilities(arrays / "post_probabilities.npy"), targets, image_ids)
                source_endpoint, adapted_endpoint = b3._endpoint_summary(source), b3._endpoint_summary(adapted)
                old = historical[(dataset, condition)]
                assert_paired_source(source_endpoint, old["source"])
                row = {"dataset": dataset, "condition": condition,
                       "corruption_family": corruption, "severity": severity,
                       "candidate_id": CANDIDATE_ID, "episode_count": 64,
                       "source": source_endpoint, "previous_o3": old["previous_o3"],
                       "adapted": adapted_endpoint, "source_counts_exact_historical": True}
                cells.append(row)
                detail_dir = output / "conditions" / dataset / condition
                detail_dir.mkdir(parents=True, exist_ok=False)
                _write_json(detail_dir / "source_metrics.json", source.to_dict())
                _write_json(detail_dir / "adapted_metrics.json", adapted.to_dict())
                print(json.dumps({"event": "condition_evaluated", "dataset": dataset,
                                  "condition": condition, "completed_cells": len(cells)}, sort_keys=True), flush=True)
        summary = summarize_performance(cells)
        summary.update({"outer_target_loader_calls": targets_loaded, "method_label_accesses": 0,
                        "test_payload_opens": 0, "validation_payload_opens": 0,
                        "source_counts_exact_historical": True})
        with (output / "cells.jsonl").open("x", encoding="utf-8") as handle:
            for row in cells:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        _write_json(output / "summary.json", summary)
        verify_all_candidates(config_path=config_path)
        if _input_bindings(config_path, raw, candidate_roots) != bindings:
            raise PerformanceEvaluationError("bound evidence changed during evaluation")
        manifest = {"phase": "outer", "candidate_id": CANDIDATE_ID,
                    "inputs": bindings, "files": _artifact_files(output),
                    "all_candidates_complete_before_gt": True, "paper_result": False,
                    "no_validation_split": True, "formal_test": False}
        _write_json(output / "manifest.json", manifest)
        _write_json(output / "COMPLETE.json", {"complete": True, "phase": "outer",
                    "manifest_sha256": _sha(output / "manifest.json"), "cell_count": 39,
                    "all_advancement_goals_met": summary["all_advancement_goals_met"]})
    except BaseException as exc:
        _write_json(output / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc),
                    "completed_cells": len(cells), "outer_target_loader_calls": targets_loaded,
                    "complete": False, "partial_artifacts_preserved": True})
        raise
    return {"status": "published", "path": str(output), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.config), ensure_ascii=False, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
