"""CPU-only 39-cell aggregate for P3 Stage-B1 mechanism evidence.

The aggregate consumes only completed, publicly verified Stage-B1 cell
artifacts.  It opens no raw image or target, constructs no model/optimizer,
performs no candidate selection, and can never authorize Stage-B3 or P5.

Descriptive tables use float64 episode statistics.  The three preregistered
mechanism flags instead use the separately frozen equal-cell macro: first take
the mean of finite target-present episodes inside each non-clean cell, then
give each of the 36 cells equal weight.  This distinction is intentional and
prevents target-rich cells from silently dominating a mechanism conclusion.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Final

import numpy as np

from analysis.d0_v3_label_free_shard import canonical_json_bytes, parse_canonical_json
from analysis.p3_stage_b1_outer_cell_shard import (
    COMPLETE_FILENAME as CELL_COMPLETE_FILENAME,
    GROUP_IDS,
    MANIFEST_FILENAME as CELL_MANIFEST_FILENAME,
    VerifiedStageB1CellShard,
    canonical_jsonl_bytes,
    parse_canonical_jsonl,
    verify_stage_b1_cell_shard,
)
from tta.d0_secure_io import read_stable_regular_file, snapshot_regular_directory


SCHEMA_VERSION: Final = 1
ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_aggregate"
COMPLETE_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_aggregate_complete"
LINEAGE_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_cell_lineage"
STRATUM_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_descriptive_stratum"
SUMMARY_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_descriptive_summary"
MECHANISM_ARTIFACT_TYPE: Final = "cr_sitta_p3_stage_b1_mechanism_evidence"

EXPECTED_CELL_COUNT: Final = 39
EPISODES_PER_CELL: Final = 64
TOTAL_EPISODE_COUNT: Final = EXPECTED_CELL_COUNT * EPISODES_PER_CELL
TARGET_PRESENCE_VALUES: Final = ("target_present", "empty_target")

CELL_LINEAGE_FILENAME: Final = "cell_lineage.jsonl"
STRATIFIED_STATISTICS_FILENAME: Final = "stratified_statistics.jsonl"
SUMMARY_STATISTICS_FILENAME: Final = "summary_statistics.jsonl"
MECHANISM_EVIDENCE_FILENAME: Final = "mechanism_evidence_flags.json"
MANIFEST_FILENAME: Final = "manifest.json"
COMPLETE_FILENAME: Final = "COMPLETE.json"
MEMBERS: Final = frozenset(
    {
        CELL_LINEAGE_FILENAME,
        STRATIFIED_STATISTICS_FILENAME,
        SUMMARY_STATISTICS_FILENAME,
        MECHANISM_EVIDENCE_FILENAME,
        MANIFEST_FILENAME,
        COMPLETE_FILENAME,
    }
)

AGGREGATE_CRITICAL_CODE_PATHS: Final = (
    "analysis/p3_stage_b1_aggregate.py",
    "analysis/p3_stage_b1_outer_cell_shard.py",
    "analysis/foreground_background_gradient_decomposition_v1.py",
    "analysis/p3_stage_b1_contract.py",
    "tta/d0_secure_io.py",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_LINEAGE_FIELDS = {
    "schema_version",
    "artifact_type",
    "cell_index",
    "dataset",
    "condition",
    "corruption_family",
    "severity",
    "replicate",
    "cell_path",
    "manifest_sha256",
    "complete_sha256",
    "basis_sha256",
    "records_sha256",
    "group_layout_sha256",
    "outer_access_receipt_sha256",
    "ordered_image_ids_sha256",
    "target_identity_sha256",
    "record_count",
    "public_cell_verifier_passed",
    "raw_target_reopened_by_aggregate",
    "candidate_selection_performed",
    "stage_b3_authorized",
    "p5_authorized",
}

# JSON metric ID -> path below ``record['groups'][group_id]``.
METRIC_PATHS: Final = {
    "task_gradient_norm": ("task_gradient_norm",),
    **{
        f"additive_gradient_norm.{name}": ("additive_gradient_norms", name)
        for name in (
            "foreground_subthreshold_add",
            "foreground_suprathreshold_add",
            "background_add",
            "foreground_add",
            "full_add",
        )
    },
    **{
        f"additive_alignment.{component}.{metric}": (
            "additive_entropy_task_alignment",
            component,
            metric,
        )
        for component in (
            "full_entropy_mean",
            "foreground_entropy_add",
            "background_entropy_add",
            "foreground_subthreshold_entropy_add",
            "foreground_suprathreshold_entropy_add",
        )
        for metric in (
            "entropy_gradient_norm",
            "task_gradient_norm",
            "entropy_task_dot",
            "entropy_task_cosine",
            "task_projection",
            "unit_descent_task_change",
        )
    },
    **{
        f"conditional_alignment.{component}.{metric}": (
            "conditional_entropy_task_alignment",
            component,
            metric,
        )
        for component in (
            "full_entropy_mean",
            "foreground_entropy_mean",
            "background_entropy_mean",
            "foreground_subthreshold_entropy_mean",
            "foreground_suprathreshold_entropy_mean",
        )
        for metric in (
            "entropy_gradient_norm",
            "task_gradient_norm",
            "entropy_task_dot",
            "entropy_task_cosine",
            "task_projection",
            "unit_descent_task_change",
        )
    },
    "cross.foreground_background_additive_cosine": (
        "cross_region",
        "foreground_background_additive_alignment",
        "foreground_background_cosine",
    ),
    "cross.foreground_background_additive_dot": (
        "cross_region",
        "foreground_background_additive_alignment",
        "foreground_background_dot",
    ),
    "cross.background_to_foreground_additive_norm_ratio": (
        "cross_region",
        "background_to_foreground_additive_norm_ratio",
        "value",
    ),
    "cross.background_to_foreground_conditional_norm_ratio": (
        "cross_region",
        "background_to_foreground_conditional_norm_ratio",
        "value",
    ),
    "cross.projection_cancellation_ratio": (
        "cross_region",
        "projection_cancellation_ratio",
        "value",
    ),
    "cross.foreground_task_dot": (
        "cross_region",
        "projection_cancellation_ratio",
        "foreground_task_dot",
    ),
    "cross.background_task_dot": (
        "cross_region",
        "projection_cancellation_ratio",
        "background_task_dot",
    ),
    "cross.full_task_dot": (
        "cross_region",
        "projection_cancellation_ratio",
        "full_task_dot",
    ),
}

P0_MECHANISM_METRICS: Final = {
    "background_norm_dominance": (
        "cross.background_to_foreground_additive_norm_ratio"
    ),
    "background_cancellation": "cross.foreground_background_additive_cosine",
    "subthreshold_erasure": (
        "additive_alignment.foreground_subthreshold_entropy_add."
        "entropy_task_cosine"
    ),
}
SMALL_GROUP_METRICS: Final = {
    "full": "additive_alignment.full_entropy_mean.entropy_task_cosine",
    "foreground_total": (
        "additive_alignment.foreground_entropy_add.entropy_task_cosine"
    ),
}

AGGREGATE_AUTHORIZATION: Final = {
    "paper_result": False,
    "paper_test_result": False,
    "performance_claim": False,
    "candidate_selection_performed": False,
    "ranking_performed": False,
    "stage_b3_authorized": False,
    "p5_authorized": False,
}


class P3StageB1AggregateError(ValueError):
    """The B1 39-cell grid or its mechanism aggregate is invalid."""


@dataclass(frozen=True, slots=True)
class StageB1CellPaths:
    index: int
    dataset: str
    condition: str
    path: Path


@dataclass(frozen=True, slots=True)
class StageB1AggregatePreflight:
    repository_root: Path
    output_root_relative: str
    protocol_id: str
    config_sha256: str
    config: Mapping[str, Any]
    lineage: tuple[Mapping[str, Any], ...]
    records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class VerifiedStageB1Aggregate:
    path: Path
    manifest_sha256: str
    complete_sha256: str
    mechanism_evidence_sha256: str
    cell_count: int
    episode_count: int
    stratified_record_count: int
    summary_record_count: int
    mechanism_statuses: Mapping[str, str]
    candidate_selection_performed: bool
    stage_b3_authorized: bool
    p5_authorized: bool


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise P3StageB1AggregateError(f"{label} must be a mapping")
    return value


def _exact(value: Any, fields: set[str], *, label: str) -> Mapping[str, Any]:
    result = _mapping(value, label=label)
    if set(result) != fields:
        missing = sorted(fields - set(result))
        unknown = sorted(set(result) - fields)
        raise P3StageB1AggregateError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )
    return result


def _sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise P3StageB1AggregateError(f"{label} must be lowercase SHA-256")
    return value


def _canonical_output_relative(value: str | os.PathLike[str]) -> str:
    path = Path(os.fspath(value))
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise P3StageB1AggregateError("output root is not canonical relative")
    return path.as_posix()


def _repository_relative(root: Path, path: Path, *, label: str) -> str:
    repository = Path(os.path.abspath(os.fspath(root)))
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = absolute.relative_to(repository)
    except ValueError as exc:
        raise P3StageB1AggregateError(f"{label} is outside repository") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise P3StageB1AggregateError(f"{label} is not canonical")
    return relative.as_posix()


def _config_sequences(config: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    raw_datasets = config.get("datasets")
    raw_conditions = config.get("conditions")
    if isinstance(raw_datasets, Mapping):
        datasets = tuple(str(value) for value in raw_datasets)
    else:
        raise P3StageB1AggregateError("config datasets must be a mapping")
    if isinstance(raw_conditions, Sequence) and not isinstance(raw_conditions, (str, bytes)):
        conditions = tuple(str(value) for value in raw_conditions)
    else:
        raise P3StageB1AggregateError("config conditions must be a sequence")
    if len(datasets) != 3 or len(conditions) != 13:
        raise P3StageB1AggregateError("B1 config must define exactly 3 x 13 cells")
    return datasets, conditions


def _condition_parts(condition: str) -> tuple[str, str]:
    if condition == "clean_S0":
        return "clean", "S0"
    family, severity = condition.rsplit("_", 1)
    return family, severity


def fixed_stage_b1_cells(
    repository_root: str | os.PathLike[str],
    output_root_relative: str | os.PathLike[str],
    *,
    config: Mapping[str, Any],
) -> tuple[StageB1CellPaths, ...]:
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    output = repository / _canonical_output_relative(output_root_relative)
    datasets, conditions = _config_sequences(config)
    cells = tuple(
        StageB1CellPaths(
            index=index,
            dataset=dataset,
            condition=condition,
            path=output / "outer_phase" / "shards" / "R0" / dataset / condition,
        )
        for index, (dataset, condition) in enumerate(
            (dataset, condition)
            for dataset in datasets
            for condition in conditions
        )
    )
    if len(cells) != EXPECTED_CELL_COUNT:
        raise P3StageB1AggregateError("fixed B1 topology is not exactly 39 cells")
    return cells


def _lineage_record(
    *,
    cell: StageB1CellPaths,
    verified: VerifiedStageB1CellShard,
    repository_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": LINEAGE_ARTIFACT_TYPE,
        "cell_index": cell.index,
        "dataset": cell.dataset,
        "condition": cell.condition,
        "corruption_family": verified.corruption_family,
        "severity": verified.severity,
        "replicate": "R0",
        "cell_path": _repository_relative(
            repository_root, cell.path, label="B1 cell path"
        ),
        "manifest_sha256": verified.manifest_sha256,
        "complete_sha256": verified.complete_sha256,
        "basis_sha256": verified.basis_sha256,
        "records_sha256": verified.records_sha256,
        "group_layout_sha256": verified.group_layout_sha256,
        "outer_access_receipt_sha256": verified.outer_access_receipt_sha256,
        "ordered_image_ids_sha256": verified.ordered_image_ids_sha256,
        "target_identity_sha256": verified.target_identity_sha256,
        "record_count": verified.record_count,
        "public_cell_verifier_passed": True,
        "raw_target_reopened_by_aggregate": False,
        "candidate_selection_performed": False,
        "stage_b3_authorized": False,
        "p5_authorized": False,
    }


def _validate_target_invariance(
    records: Sequence[Mapping[str, Any]], datasets: Sequence[str], conditions: Sequence[str]
) -> None:
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    cell_keys: set[tuple[str, str]] = set()
    for record in records:
        dataset = str(record["dataset"])
        condition = str(record["condition"])
        index = int(record["image_index"])
        cell_keys.add((dataset, condition))
        target = record["target"]
        identity = {
            "image_id": record["image_id"],
            "target_present": target["target_present"],
            "total_pixel_count": target["total_pixel_count"],
            "foreground_pixel_count": target["foreground_pixel_count"],
            "background_pixel_count": target["background_pixel_count"],
            "target_value_sum": target["target_value_sum"],
            "target_slice_sha256": target["target_slice_sha256"],
        }
        key = (dataset, index)
        if key in by_key and by_key[key] != identity:
            raise P3StageB1AggregateError(
                f"target identity drifts across conditions: {dataset}/{index}"
            )
        by_key[key] = identity
    expected_cells = {(dataset, condition) for dataset in datasets for condition in conditions}
    if cell_keys != expected_cells:
        raise P3StageB1AggregateError("record grid does not cover exact 39 cells")
    if len(by_key) != len(datasets) * EPISODES_PER_CELL:
        raise P3StageB1AggregateError("target identity grid is not 3 x 64")


def collect_stage_b1_preflight(
    *,
    repository_root: str | os.PathLike[str],
    output_root_relative: str | os.PathLike[str],
    config: Mapping[str, Any],
    config_sha256: str,
    expected_code_seal: Mapping[str, Any] | None = None,
) -> StageB1AggregatePreflight:
    """Read-only public verification of all 39 completed cells."""

    repository = Path(os.path.abspath(os.fspath(repository_root)))
    output_relative = _canonical_output_relative(output_root_relative)
    digest = _sha256(config_sha256, label="config_sha256")
    protocol_id = str(config.get("protocol_id"))
    datasets, conditions = _config_sequences(config)
    lineage: list[Mapping[str, Any]] = []
    records: list[Mapping[str, Any]] = []
    for cell in fixed_stage_b1_cells(repository, output_relative, config=config):
        verified = verify_stage_b1_cell_shard(
            cell.path,
            repository_root=repository,
            config=config,
            expected_config_sha256=digest,
            verify_live_parents=True,
            expected_code_seal=expected_code_seal,
        )
        if (
            verified.dataset != cell.dataset
            or verified.condition != cell.condition
            or verified.record_count != EPISODES_PER_CELL
        ):
            raise P3StageB1AggregateError("verified cell grid binding differs")
        lineage.append(
            _lineage_record(cell=cell, verified=verified, repository_root=repository)
        )
        records.extend(verified.records)
    if len(lineage) != EXPECTED_CELL_COUNT or len(records) != TOTAL_EPISODE_COUNT:
        raise P3StageB1AggregateError("B1 grid is not exact 39 x 64")
    _validate_target_invariance(records, datasets, conditions)
    for dataset in datasets:
        rows = [value for value in lineage if value["dataset"] == dataset]
        for field in ("ordered_image_ids_sha256", "target_identity_sha256"):
            if len({str(value[field]) for value in rows}) != 1:
                raise P3StageB1AggregateError(
                    f"{dataset} does not share one {field} across 13 conditions"
                )
    return StageB1AggregatePreflight(
        repository_root=repository,
        output_root_relative=output_relative,
        protocol_id=protocol_id,
        config_sha256=digest,
        config=config,
        lineage=tuple(lineage),
        records=tuple(records),
    )


def _path_value(record: Mapping[str, Any], group_id: str, path: Sequence[str]) -> Any:
    value: Any = record["groups"][group_id]
    for key in path:
        value = value[key]
    return value


def _metric_value(record: Mapping[str, Any], group_id: str, metric_id: str) -> float | None:
    if metric_id == "target.foreground_fraction":
        target = record["target"]
        return float(target["foreground_pixel_count"]) / float(target["total_pixel_count"])
    if metric_id == "target.foreground_subthreshold_fraction":
        target = record["target"]
        return float(target["foreground_subthreshold_pixel_count"]) / float(
            target["total_pixel_count"]
        )
    raw = _path_value(record, group_id, METRIC_PATHS[metric_id])
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise P3StageB1AggregateError(f"metric {metric_id} is not numeric/null")
    value = float(raw)
    if not math.isfinite(value):
        raise P3StageB1AggregateError(f"metric {metric_id} is non-finite")
    return value


ALL_METRIC_IDS: Final = (
    "target.foreground_fraction",
    "target.foreground_subthreshold_fraction",
    *tuple(METRIC_PATHS),
)


def descriptive_statistics(
    values: Sequence[float | None], *, total_observation_count: int | None = None
) -> dict[str, Any]:
    """Frozen float64 mean/median/linear-quartile sufficient summary."""

    total = len(values) if total_observation_count is None else total_observation_count
    if total < len(values) or total < 0:
        raise P3StageB1AggregateError("invalid descriptive total count")
    materialized = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    if materialized.size and not np.isfinite(materialized).all():
        raise P3StageB1AggregateError("descriptive values contain NaN/Inf")
    count = int(materialized.size)
    null_count = total - count
    if count:
        mean = float(np.mean(materialized, dtype=np.float64))
        median = float(np.median(materialized))
        q1 = float(np.quantile(materialized, 0.25, method="linear"))
        q3 = float(np.quantile(materialized, 0.75, method="linear"))
        minimum = float(np.min(materialized))
        maximum = float(np.max(materialized))
        status = "estimable"
    else:
        mean = median = q1 = q3 = minimum = maximum = None
        status = "not_estimable"
    denominator = float(total) if total else None
    return {
        "status": status,
        "total_observation_count": total,
        "finite_non_null_count": count,
        "null_count": null_count,
        "finite_non_null_fraction": (count / denominator if denominator else None),
        "null_fraction": (null_count / denominator if denominator else None),
        "mean": mean,
        "median": median,
        "q1": q1,
        "q3": q3,
        "minimum": minimum,
        "maximum": maximum,
    }


def _statistics_for_records(
    records: Sequence[Mapping[str, Any]], group_id: str
) -> dict[str, Any]:
    return {
        metric_id: descriptive_statistics(
            [_metric_value(record, group_id, metric_id) for record in records]
        )
        for metric_id in ALL_METRIC_IDS
    }


def build_stratified_statistics(
    records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    datasets, conditions = _config_sequences(config)
    output: list[Mapping[str, Any]] = []
    for dataset in datasets:
        for condition in conditions:
            family, severity = _condition_parts(condition)
            cell = [
                record
                for record in records
                if record["dataset"] == dataset and record["condition"] == condition
            ]
            if len(cell) != EPISODES_PER_CELL:
                raise P3StageB1AggregateError(
                    f"cell episode count differs: {dataset}/{condition}"
                )
            for group_id in GROUP_IDS:
                for presence in TARGET_PRESENCE_VALUES:
                    selected = [
                        record
                        for record in cell
                        if bool(record["target"]["target_present"])
                        is (presence == "target_present")
                    ]
                    output.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "artifact_type": STRATUM_ARTIFACT_TYPE,
                            "stratum_index": len(output),
                            "dataset": dataset,
                            "corruption_family": family,
                            "severity": severity,
                            "condition": condition,
                            "parameter_space": group_id,
                            "target_presence": presence,
                            "episode_count": len(selected),
                            "aggregation_unit": "episode",
                            "accumulation_dtype": "float64",
                            "metrics": _statistics_for_records(selected, group_id),
                            "candidate_selection_performed": False,
                            "stage_b3_authorized": False,
                        }
                    )
    expected = len(datasets) * len(conditions) * len(GROUP_IDS) * len(
        TARGET_PRESENCE_VALUES
    )
    if len(output) != expected:
        raise P3StageB1AggregateError("stratified topology differs")
    return tuple(output)


def _summary_scopes(
    *, datasets: Sequence[str], conditions: Sequence[str]
) -> tuple[tuple[str, str, Callable[[Mapping[str, Any]], bool]], ...]:
    nonclean = lambda value: value["condition"] != "clean_S0"
    scopes: list[tuple[str, str, Callable[[Mapping[str, Any]], bool]]] = [
        ("overall", "overall", lambda _value: True),
        ("clean", "clean", lambda value: value["condition"] == "clean_S0"),
        ("nonclean", "nonclean", nonclean),
    ]
    for dataset in datasets:
        scopes.extend(
            [
                ("dataset", dataset, lambda value, d=dataset: value["dataset"] == d),
                (
                    "dataset_clean",
                    dataset,
                    lambda value, d=dataset: value["dataset"] == d
                    and value["condition"] == "clean_S0",
                ),
                (
                    "dataset_nonclean",
                    dataset,
                    lambda value, d=dataset: value["dataset"] == d
                    and value["condition"] != "clean_S0",
                ),
            ]
        )
    families = tuple(
        dict.fromkeys(_condition_parts(condition)[0] for condition in conditions)
    )
    severities = tuple(
        dict.fromkeys(_condition_parts(condition)[1] for condition in conditions)
    )
    for family in families:
        scopes.append(
            (
                "corruption_family",
                family,
                lambda value, f=family: value["corruption_family"] == f,
            )
        )
    for severity in severities:
        scopes.append(
            (
                "severity",
                severity,
                lambda value, s=severity: value["severity"] == s,
            )
        )
    if "NUAA-SIRST" in datasets:
        scopes.extend(
            [
                (
                    "nuaa",
                    "overall",
                    lambda value: value["dataset"] == "NUAA-SIRST",
                ),
                (
                    "nuaa",
                    "clean",
                    lambda value: value["dataset"] == "NUAA-SIRST"
                    and value["condition"] == "clean_S0",
                ),
                (
                    "nuaa",
                    "nonclean",
                    lambda value: value["dataset"] == "NUAA-SIRST"
                    and value["condition"] != "clean_S0",
                ),
            ]
        )
        for family in families:
            if family == "clean":
                continue
            scopes.append(
                (
                    "nuaa_corruption_family",
                    family,
                    lambda value, f=family: value["dataset"] == "NUAA-SIRST"
                    and value["corruption_family"] == f,
                )
            )
    return tuple(scopes)


def build_summary_statistics(
    records: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    datasets, conditions = _config_sequences(config)
    output: list[Mapping[str, Any]] = []
    for scope_type, scope_id, selector in _summary_scopes(
        datasets=datasets, conditions=conditions
    ):
        scope_records = [record for record in records if selector(record)]
        cell_count = len(
            {(record["dataset"], record["condition"]) for record in scope_records}
        )
        for group_id in GROUP_IDS:
            for presence in TARGET_PRESENCE_VALUES:
                selected = [
                    record
                    for record in scope_records
                    if bool(record["target"]["target_present"])
                    is (presence == "target_present")
                ]
                output.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": SUMMARY_ARTIFACT_TYPE,
                        "summary_index": len(output),
                        "scope_type": scope_type,
                        "scope_id": scope_id,
                        "parameter_space": group_id,
                        "target_presence": presence,
                        "cell_count": cell_count,
                        "episode_count": len(selected),
                        "aggregation_unit": "episode",
                        "accumulation_dtype": "float64",
                        "metrics": _statistics_for_records(selected, group_id),
                        "candidate_selection_performed": False,
                        "stage_b3_authorized": False,
                    }
                )
    return tuple(output)


def _cell_means(
    records: Sequence[Mapping[str, Any]],
    *,
    group_id: str,
    metric_id: str,
    cells: Sequence[tuple[str, str]],
) -> tuple[list[float], list[dict[str, Any]], bool]:
    values: list[float] = []
    evidence: list[dict[str, Any]] = []
    all_estimable = True
    for dataset, condition in cells:
        selected = [
            record
            for record in records
            if record["dataset"] == dataset
            and record["condition"] == condition
            and record["target"]["target_present"] is True
        ]
        finite = [
            value
            for value in (
                _metric_value(record, group_id, metric_id) for record in selected
            )
            if value is not None
        ]
        if not finite:
            mean = None
            all_estimable = False
        else:
            mean = float(np.mean(np.asarray(finite, dtype=np.float64), dtype=np.float64))
            values.append(mean)
        evidence.append(
            {
                "dataset": dataset,
                "condition": condition,
                "target_present_episode_count": len(selected),
                "valid_non_null_episode_count": len(finite),
                "null_or_nonfinite_episode_count": len(selected) - len(finite),
                "cell_mean": mean,
                "estimable": mean is not None,
            }
        )
    return values, evidence, all_estimable and len(values) == len(cells)


def _macro(
    records: Sequence[Mapping[str, Any]],
    *,
    group_id: str,
    metric_id: str,
    cells: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    values, evidence, estimable = _cell_means(
        records, group_id=group_id, metric_id=metric_id, cells=cells
    )
    valid_episode_count = sum(
        int(value["valid_non_null_episode_count"]) for value in evidence
    )
    return {
        "status": "estimable" if estimable else "not_estimable",
        "required_cell_count": len(cells),
        "valid_non_null_cell_count": len(values),
        "valid_non_null_episode_count": valid_episode_count,
        "equal_cell_macro_mean": (
            float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))
            if estimable
            else None
        ),
        "cells": evidence,
    }


def _evaluate_one_mechanism_flag(
    *,
    records: Sequence[Mapping[str, Any]],
    datasets: Sequence[str],
    conditions: Sequence[str],
    group_id: str,
    metric_id: str,
    comparison: str,
    threshold: float,
    tolerance: float,
    minimum_datasets: int,
    minimum_families: int,
) -> dict[str, Any]:
    nonclean_conditions = [value for value in conditions if value != "clean_S0"]
    cells = [(dataset, condition) for dataset in datasets for condition in nonclean_conditions]
    overall = _macro(
        records, group_id=group_id, metric_id=metric_id, cells=cells
    )
    per_dataset = {
        dataset: _macro(
            records,
            group_id=group_id,
            metric_id=metric_id,
            cells=[(dataset, condition) for condition in nonclean_conditions],
        )
        for dataset in datasets
    }
    families = tuple(
        dict.fromkeys(_condition_parts(value)[0] for value in nonclean_conditions)
    )
    per_family = {
        family: _macro(
            records,
            group_id=group_id,
            metric_id=metric_id,
            cells=[
                (dataset, condition)
                for dataset in datasets
                for condition in nonclean_conditions
                if _condition_parts(condition)[0] == family
            ],
        )
        for family in families
    }

    def passes(value: float | None) -> bool:
        if value is None:
            return False
        if comparison == "greater":
            return value > threshold + tolerance
        if comparison == "less":
            return value < threshold - tolerance
        if comparison == "greater_equal":
            return value + tolerance >= threshold
        raise P3StageB1AggregateError("unknown mechanism comparison")

    supporting_datasets = [
        key
        for key, value in per_dataset.items()
        if value["status"] == "estimable"
        and passes(value["equal_cell_macro_mean"])
    ]
    supporting_families = [
        key
        for key, value in per_family.items()
        if value["status"] == "estimable"
        and passes(value["equal_cell_macro_mean"])
    ]
    all_estimable = (
        overall["status"] == "estimable"
        and all(value["status"] == "estimable" for value in per_dataset.values())
        and all(value["status"] == "estimable" for value in per_family.values())
    )
    if not all_estimable:
        status = "not_estimable"
    elif (
        passes(overall["equal_cell_macro_mean"])
        and len(supporting_datasets) >= minimum_datasets
        and len(supporting_families) >= minimum_families
    ):
        status = "supported"
    else:
        status = "not_supported"
    return {
        "status": status,
        "parameter_space": group_id,
        "metric_id": metric_id,
        "comparison": comparison,
        "threshold": threshold,
        "comparison_tolerance": tolerance,
        "overall": overall,
        "per_dataset": per_dataset,
        "per_corruption_family": per_family,
        "supporting_datasets": supporting_datasets,
        "supporting_corruption_families": supporting_families,
        "minimum_supporting_datasets": minimum_datasets,
        "minimum_supporting_corruption_families": minimum_families,
        "candidate_selection_performed": False,
        "stage_b3_authorized": False,
    }


_CANCELLATION_PROJECTION_METRICS: Final = {
    "foreground": "additive_alignment.foreground_entropy_add.task_projection",
    "background": "additive_alignment.background_entropy_add.task_projection",
    "full": "additive_alignment.full_entropy_mean.task_projection",
}


def _joint_projection_macro(
    records: Sequence[Mapping[str, Any]],
    *,
    cells: Sequence[tuple[str, str]],
    tolerance: float,
) -> dict[str, Any]:
    cell_evidence: list[dict[str, Any]] = []
    component_cell_means: dict[str, list[float]] = {
        component: [] for component in _CANCELLATION_PROJECTION_METRICS
    }
    complete = True
    for dataset, condition in cells:
        candidates = [
            record
            for record in records
            if record["dataset"] == dataset
            and record["condition"] == condition
            and record["target"]["target_present"] is True
        ]
        joint: list[dict[str, float]] = []
        for record in candidates:
            values = {
                component: _metric_value(record, "P0", metric_id)
                for component, metric_id in _CANCELLATION_PROJECTION_METRICS.items()
            }
            if all(value is not None for value in values.values()):
                joint.append(
                    {component: float(value) for component, value in values.items()}
                )
        if not joint:
            means = {component: None for component in component_cell_means}
            cell_pass = None
            complete = False
        else:
            means = {
                component: float(
                    np.mean(
                        np.asarray([value[component] for value in joint], dtype=np.float64),
                        dtype=np.float64,
                    )
                )
                for component in component_cell_means
            }
            for component, value in means.items():
                component_cell_means[component].append(value)
            cell_pass = (
                means["foreground"] > tolerance
                and means["background"] < -tolerance
                and abs(means["full"]) + tolerance < abs(means["foreground"])
            )
        cell_evidence.append(
            {
                "dataset": dataset,
                "condition": condition,
                "target_present_episode_count": len(candidates),
                "joint_valid_non_null_episode_count": len(joint),
                "joint_null_or_nonfinite_episode_count": len(candidates) - len(joint),
                "component_cell_means": means,
                "joint_predicate_passed": cell_pass,
                "estimable": bool(joint),
            }
        )
    macro_means = {
        component: (
            float(np.mean(np.asarray(values, dtype=np.float64), dtype=np.float64))
            if complete and len(values) == len(cells)
            else None
        )
        for component, values in component_cell_means.items()
    }
    if complete:
        passed: bool | None = (
            macro_means["foreground"] > tolerance
            and macro_means["background"] < -tolerance
            and abs(macro_means["full"]) + tolerance
            < abs(macro_means["foreground"])
        )
    else:
        passed = None
    return {
        "status": "estimable" if complete else "not_estimable",
        "required_cell_count": len(cells),
        "joint_valid_non_null_cell_count": (
            len(cells) if complete else sum(value["estimable"] for value in cell_evidence)
        ),
        "joint_valid_non_null_episode_count": sum(
            int(value["joint_valid_non_null_episode_count"])
            for value in cell_evidence
        ),
        "joint_support_policy": "all_three_projections_non_null_in_same_episode",
        "equal_cell_macro_means": macro_means,
        "joint_predicate": {
            "foreground_mean_rule": "value>comparison_tolerance",
            "background_mean_rule": "value<0-comparison_tolerance",
            "full_cancellation_rule": (
                "abs(mean_full)+comparison_tolerance<abs(mean_foreground)"
            ),
            "passed": passed,
        },
        "cells": cell_evidence,
    }


def _evaluate_background_cancellation(
    records: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str],
    conditions: Sequence[str],
    tolerance: float,
    minimum_datasets: int,
    minimum_families: int,
) -> dict[str, Any]:
    nonclean = [condition for condition in conditions if condition != "clean_S0"]
    all_cells = [(dataset, condition) for dataset in datasets for condition in nonclean]
    overall = _joint_projection_macro(records, cells=all_cells, tolerance=tolerance)
    per_dataset = {
        dataset: _joint_projection_macro(
            records,
            cells=[(dataset, condition) for condition in nonclean],
            tolerance=tolerance,
        )
        for dataset in datasets
    }
    families = tuple(dict.fromkeys(_condition_parts(value)[0] for value in nonclean))
    per_family = {
        family: _joint_projection_macro(
            records,
            cells=[
                (dataset, condition)
                for dataset in datasets
                for condition in nonclean
                if _condition_parts(condition)[0] == family
            ],
            tolerance=tolerance,
        )
        for family in families
    }
    supporting_datasets = [
        key
        for key, value in per_dataset.items()
        if value["joint_predicate"]["passed"] is True
    ]
    supporting_families = [
        key
        for key, value in per_family.items()
        if value["joint_predicate"]["passed"] is True
    ]
    estimable = (
        overall["status"] == "estimable"
        and all(value["status"] == "estimable" for value in per_dataset.values())
        and all(value["status"] == "estimable" for value in per_family.values())
    )
    if not estimable:
        status = "not_estimable"
    elif (
        overall["joint_predicate"]["passed"] is True
        and len(supporting_datasets) >= minimum_datasets
        and len(supporting_families) >= minimum_families
    ):
        status = "supported"
    else:
        status = "not_supported"
    # This cosine is explicitly descriptive and cannot flip the joint flag.
    cosine_auxiliary = _macro(
        records,
        group_id="P0",
        metric_id="cross.foreground_background_additive_cosine",
        cells=all_cells,
    )
    return {
        "status": status,
        "parameter_space": "P0",
        "metric_ids": dict(_CANCELLATION_PROJECTION_METRICS),
        "comparison_tolerance": tolerance,
        "overall": overall,
        "per_dataset": per_dataset,
        "per_corruption_family": per_family,
        "supporting_datasets": supporting_datasets,
        "supporting_corruption_families": supporting_families,
        "minimum_supporting_datasets": minimum_datasets,
        "minimum_supporting_corruption_families": minimum_families,
        "foreground_background_cosine_auxiliary_only": cosine_auxiliary,
        "candidate_selection_performed": False,
        "stage_b3_authorized": False,
    }


def build_mechanism_evidence(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_sha256: str,
    mechanism_gate: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Evaluate only the preregistered mechanism flags; never rank/select."""

    datasets, conditions = _config_sequences(config)
    gate = _exact(
        mechanism_gate,
        {
            "evidence_scope",
            "aggregation",
            "support_accounting",
            "coverage",
            "comparison_tolerance",
            "allowed_status_values",
            "status_policy",
            "P0_flags",
            "small_group_alignment_hints",
        },
        label="mechanism_gate",
    )
    scope = _exact(
        gate["evidence_scope"],
        {
            "parameter_space", "conditions", "episode_filter",
            "required_nonclean_cell_count", "per_dataset_cell_count",
            "per_corruption_family_cell_count",
        },
        label="mechanism evidence_scope",
    )
    aggregation = _exact(
        gate["aggregation"],
        {
            "only_authorized_statistic", "episode_statistic", "cell_weighting",
            "dataset_statistic", "corruption_family_statistic", "overall_statistic",
            "projection_reduction_order", "absolute_projection_semantics",
            "joint_predicate_semantics", "median_or_alternative_statistic",
        },
        label="mechanism aggregation",
    )
    support = _exact(
        gate["support_accounting"],
        {
            "finite_non_null_definition", "single_quantity_episode_support",
            "joint_quantity_episode_support", "cell_is_estimable",
            "joint_cell_is_estimable", "stratum_is_estimable",
            "joint_stratum_is_estimable", "required_count_fields",
        },
        label="mechanism support_accounting",
    )
    coverage = _exact(
        gate["coverage"],
        {
            "dataset_names", "corruption_family_names",
            "minimum_supporting_datasets", "minimum_supporting_corruption_families",
        },
        label="mechanism coverage",
    )
    flags = _exact(
        gate["P0_flags"],
        {"background_norm_dominance", "background_cancellation", "subthreshold_erasure"},
        label="P0 flags",
    )
    hints = _exact(
        gate["small_group_alignment_hints"],
        {
            "parameter_spaces", "components", "report_field_paths", "quantity",
            "gradient_semantics", "support_mode", "minimum_cosine", "overall_rule",
            "supporting_stratum_rule", "require_dataset_and_family_coverage", "role",
        },
        label="small group hints",
    )
    tolerance = float(gate.get("comparison_tolerance"))
    required_counts = _mapping(
        support["required_count_fields"], label="required count fields"
    )
    expected_required_counts = {
        "per_cell": [
            "target_present_episode_count",
            "valid_non_null_episode_count",
            "null_or_nonfinite_episode_count",
        ],
        "per_joint_cell": [
            "target_present_episode_count",
            "joint_valid_non_null_episode_count",
            "joint_null_or_nonfinite_episode_count",
        ],
        "per_stratum": [
            "required_cell_count",
            "valid_non_null_cell_count",
            "valid_non_null_episode_count",
        ],
        "per_joint_stratum": [
            "required_cell_count",
            "joint_valid_non_null_cell_count",
            "joint_valid_non_null_episode_count",
        ],
    }
    if (
        scope.get("parameter_space") != "P0"
        or scope.get("conditions") != "nonclean_36_cells"
        or scope.get("episode_filter") != "target_present"
        or scope.get("required_nonclean_cell_count") != 36
        or aggregation.get("only_authorized_statistic")
        != "equal_cell_macro_of_equal_valid_target_present_episode_means"
        or aggregation.get("cell_weighting") != "equal"
        or tuple(aggregation.get("projection_reduction_order", ()))
        != (
            "restrict_to_same_joint_valid_episode_support_when_predicate_is_joint",
            "arithmetic_mean_each_projection_within_cell",
            "equal_cell_macro_each_projection_within_requested_stratum",
            "apply_absolute_value_only_to_completed_stratum_mean",
        )
        or aggregation.get("absolute_projection_semantics")
        != "abs(mean_projection)_never_mean(abs(projection))"
        or aggregation.get("median_or_alternative_statistic") != "forbidden"
        or dict(required_counts) != expected_required_counts
        or support.get("finite_non_null_definition")
        != "value_is_not_null_and_math_isfinite"
        or support.get("cell_is_estimable") != "valid_non_null_episode_count>=1"
        or support.get("joint_cell_is_estimable")
        != "joint_valid_non_null_episode_count>=1"
        or support.get("stratum_is_estimable")
        != "valid_non_null_cell_count==required_cell_count"
        or support.get("joint_stratum_is_estimable")
        != "joint_valid_non_null_cell_count==required_cell_count"
        or tuple(coverage.get("dataset_names", ())) != tuple(datasets)
        or tuple(coverage.get("corruption_family_names", ()))
        != ("gaussian_noise", "gaussian_blur", "low_contrast", "stripe_noise")
        or not math.isfinite(tolerance)
        or tolerance < 0.0
        or tuple(gate["allowed_status_values"])
        != ("supported", "not_supported", "not_estimable")
        or set(_mapping(gate["status_policy"], label="status policy"))
        != {"not_estimable", "supported", "not_supported"}
    ):
        raise P3StageB1AggregateError("frozen mechanism gate semantics differ")
    minimum_datasets = int(coverage["minimum_supporting_datasets"])
    minimum_families = int(coverage["minimum_supporting_corruption_families"])
    expected_flag_specs = {
        "background_norm_dominance": {
            "quantity": "background_to_foreground_additive_l2_norm_ratio",
            "report_field_path": (
                "per_group.P0.cross_region."
                "background_to_foreground_additive_norm_ratio.value"
            ),
            "gradient_semantics": "additive_full_image_denominator",
            "support_mode": "single_quantity_finite_non_null",
            "overall_rule": "value>1.0+comparison_tolerance",
            "supporting_stratum_rule": "value>1.0+comparison_tolerance",
        },
        "subthreshold_erasure": {
            "quantity": "additive_foreground_subthreshold_entropy_task_cosine",
            "report_field_path": (
                "per_group.P0.additive_entropy_task_alignment."
                "foreground_subthreshold_entropy_add.entropy_task_cosine"
            ),
            "gradient_semantics": "additive_full_image_denominator",
            "support_mode": "single_quantity_finite_non_null",
            "interpretation": flags["subthreshold_erasure"].get("interpretation"),
            "overall_rule": "value<0.0-comparison_tolerance",
            "supporting_stratum_rule": "value<0.0-comparison_tolerance",
        },
    }
    p0_results: dict[str, Any] = {}
    for flag_id in ("background_norm_dominance", "subthreshold_erasure"):
        metric_id = P0_MECHANISM_METRICS[flag_id]
        spec = _mapping(flags.get(flag_id), label=f"P0 flag {flag_id}")
        if dict(spec) != expected_flag_specs[flag_id] or (
            flag_id == "subthreshold_erasure"
            and not isinstance(spec.get("interpretation"), str)
        ):
            raise P3StageB1AggregateError(f"P0 flag spec differs: {flag_id}")
        comparison = "greater" if flag_id == "background_norm_dominance" else "less"
        threshold = 1.0 if comparison == "greater" else 0.0
        p0_results[flag_id] = _evaluate_one_mechanism_flag(
            records=records,
            datasets=datasets,
            conditions=conditions,
            group_id="P0",
            metric_id=metric_id,
            comparison=comparison,
            threshold=threshold,
            tolerance=tolerance,
            minimum_datasets=minimum_datasets,
            minimum_families=minimum_families,
        )
    cancellation_spec = _exact(
        flags.get("background_cancellation"), label="P0 flag background_cancellation"
        , fields={
            "quantity", "report_field_paths", "gradient_semantics", "support_mode",
            "projection_aggregation", "absolute_projection_semantics",
            "joint_predicate", "overall_rule", "supporting_stratum_rule",
            "auxiliary_foreground_background_cosine",
        }
    )
    expected_config_paths = {
        "foreground_task_projection": (
            "per_group.P0.additive_entropy_task_alignment."
            "foreground_entropy_add.task_projection"
        ),
        "background_task_projection": (
            "per_group.P0.additive_entropy_task_alignment."
            "background_entropy_add.task_projection"
        ),
        "full_task_projection": (
            "per_group.P0.additive_entropy_task_alignment."
            "full_entropy_mean.task_projection"
        ),
    }
    joint = _mapping(cancellation_spec["joint_predicate"], label="joint predicate")
    conditions_spec = _mapping(joint.get("conditions"), label="joint conditions")
    auxiliary = _mapping(
        cancellation_spec["auxiliary_foreground_background_cosine"],
        label="cancellation auxiliary",
    )
    expected_joint_conditions = {
        "foreground_projection_positive": {
            "left": "mean_foreground_task_projection",
            "operator": ">",
            "right": "+comparison_tolerance",
        },
        "background_projection_negative": {
            "left": "mean_background_task_projection",
            "operator": "<",
            "right": "-comparison_tolerance",
        },
        "full_projection_magnitude_reduced": {
            "left": "abs(mean_full_task_projection)+comparison_tolerance",
            "operator": "<",
            "right": "abs(mean_foreground_task_projection)",
        },
    }
    if (
        cancellation_spec["quantity"] != "joint_additive_task_projection_cancellation"
        or dict(cancellation_spec["report_field_paths"]) != expected_config_paths
        or cancellation_spec["gradient_semantics"]
        != "additive_full_image_denominator"
        or cancellation_spec["support_mode"]
        != "same_episode_joint_finite_non_null"
        or cancellation_spec["projection_aggregation"]
        != "mean_projection_then_absolute_value"
        or cancellation_spec["absolute_projection_semantics"]
        != "abs(mean_projection)_never_mean(abs(projection))"
        or joint.get("connective") != "all"
        or joint.get("evaluate_on_same_stratum") is not True
        or joint.get("use_same_joint_valid_episode_support") is not True
        or dict(conditions_spec) != expected_joint_conditions
        or cancellation_spec["overall_rule"]
        != "evaluate_joint_predicate_on_overall_stratum"
        or cancellation_spec["supporting_stratum_rule"]
        != "evaluate_same_joint_predicate_on_each_dataset_or_corruption_family_stratum"
        or auxiliary.get("report_field_path")
        != (
            "per_group.P0.cross_region.foreground_background_additive_alignment."
            "foreground_background_cosine"
        )
        or auxiliary.get("gradient_semantics")
        != "additive_full_image_denominator"
        or auxiliary.get("role") != "descriptive_only_not_used_by_primary_flag_status"
        or auxiliary.get("support_mode") != "single_quantity_finite_non_null"
    ):
        raise P3StageB1AggregateError("cancellation joint spec differs")
    p0_results["background_cancellation"] = _evaluate_background_cancellation(
        records,
        datasets=datasets,
        conditions=conditions,
        tolerance=tolerance,
        minimum_datasets=minimum_datasets,
        minimum_families=minimum_families,
    )
    if (
        tuple(hints.get("parameter_spaces", ())) != ("P1", "P2", "P3", "P4")
        or tuple(hints.get("components", ())) != ("full", "foreground_total")
        or dict(hints.get("report_field_paths", {}))
        != {
            "full": (
                "per_group.{parameter_space}.additive_entropy_task_alignment."
                "full_entropy_mean.entropy_task_cosine"
            ),
            "foreground_total": (
                "per_group.{parameter_space}.additive_entropy_task_alignment."
                "foreground_entropy_add.entropy_task_cosine"
            ),
        }
        or hints.get("quantity") != "additive_entropy_task_cosine"
        or hints.get("gradient_semantics") != "additive_full_image_denominator"
        or hints.get("support_mode") != "single_quantity_finite_non_null"
        or hints.get("require_dataset_and_family_coverage") is not True
        or hints.get("role") != "mechanism_hint_only_not_candidate_selection"
    ):
        raise P3StageB1AggregateError("small-group hint contract differs")
    minimum_cosine = float(hints["minimum_cosine"])
    small_results: dict[str, Any] = {}
    for group_id in ("P1", "P2", "P3", "P4"):
        small_results[group_id] = {
            component: _evaluate_one_mechanism_flag(
                records=records,
                datasets=datasets,
                conditions=conditions,
                group_id=group_id,
                metric_id=metric_id,
                comparison="greater_equal",
                threshold=minimum_cosine,
                tolerance=tolerance,
                minimum_datasets=minimum_datasets,
                minimum_families=minimum_families,
            )
            for component, metric_id in SMALL_GROUP_METRICS.items()
        }
    nuaa_hints = {
        group_id: {
            component: result["per_dataset"].get("NUAA-SIRST")
            for component, result in components.items()
        }
        for group_id, components in small_results.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MECHANISM_ARTIFACT_TYPE,
        "protocol_id": str(config["protocol_id"]),
        "config_sha256": _sha256(config_sha256, label="config_sha256"),
        "scope": {
            "source_train_pilot64_only": True,
            "nonclean_cell_count": 36,
            "episode_filter": "target_present",
            "aggregation": "equal_cell_macro",
            "outer_oracle_mechanism_analysis": True,
        },
        "P0_flags": p0_results,
        "small_group_alignment_hints": small_results,
        "nuaa_small_group_alignment": nuaa_hints,
        "mechanism_answers": {
            "background_norm_dominance": p0_results["background_norm_dominance"]["status"],
            "background_cancellation": p0_results["background_cancellation"]["status"],
            "foreground_subthreshold_erasure": p0_results["subthreshold_erasure"]["status"],
            "direction_vs_magnitude_attribution": (
                "descriptive_only_no_frozen_binary_rule"
            ),
        },
        "selection": {
            "selection_or_ranking_performed": False,
            "selected_candidates": [],
            "mechanism_flags_are_not_candidate_gate": True,
        },
        "authorization": AGGREGATE_AUTHORIZATION,
    }


def _validate_lineage(
    lineage: Sequence[Mapping[str, Any]], *, preflight: StageB1AggregatePreflight
) -> tuple[Mapping[str, Any], ...]:
    cells = fixed_stage_b1_cells(
        preflight.repository_root,
        preflight.output_root_relative,
        config=preflight.config,
    )
    if isinstance(lineage, (str, bytes, Mapping)) or len(lineage) != len(cells):
        raise P3StageB1AggregateError("cell lineage must contain exactly 39 rows")
    result: list[Mapping[str, Any]] = []
    paths: set[str] = set()
    for expected, raw in zip(cells, lineage, strict=True):
        record = _exact(raw, _LINEAGE_FIELDS, label="cell lineage")
        family, severity = _condition_parts(expected.condition)
        expected_path = _repository_relative(
            preflight.repository_root, expected.path, label="cell path"
        )
        if (
            record["schema_version"] != SCHEMA_VERSION
            or record["artifact_type"] != LINEAGE_ARTIFACT_TYPE
            or record["cell_index"] != expected.index
            or record["dataset"] != expected.dataset
            or record["condition"] != expected.condition
            or record["corruption_family"] != family
            or record["severity"] != severity
            or record["replicate"] != "R0"
            or record["cell_path"] != expected_path
            or record["record_count"] != 64
            or record["public_cell_verifier_passed"] is not True
            or record["raw_target_reopened_by_aggregate"] is not False
            or record["candidate_selection_performed"] is not False
            or record["stage_b3_authorized"] is not False
            or record["p5_authorized"] is not False
        ):
            raise P3StageB1AggregateError("cell lineage identity/order differs")
        for field in (
            "manifest_sha256", "complete_sha256", "basis_sha256", "records_sha256",
            "group_layout_sha256", "outer_access_receipt_sha256",
            "ordered_image_ids_sha256", "target_identity_sha256",
        ):
            _sha256(record[field], label=f"lineage {field}")
        if expected_path in paths:
            raise P3StageB1AggregateError("duplicate cell lineage path")
        paths.add(expected_path)
        result.append(record)
    for dataset in _config_sequences(preflight.config)[0]:
        rows = [value for value in result if value["dataset"] == dataset]
        for field in ("ordered_image_ids_sha256", "target_identity_sha256"):
            if len({str(value[field]) for value in rows}) != 1:
                raise P3StageB1AggregateError(
                    f"{dataset} lineage target/order SHA drifts"
                )
    return tuple(result)


def build_aggregate_code_seal(repository_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(repository_root)))
    files = [
        {"path": path, "sha256": read_stable_regular_file(root / path).sha256}
        for path in AGGREGATE_CRITICAL_CODE_PATHS
    ]
    return {
        "files": files,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(files)).hexdigest(),
    }


def build_stage_b1_aggregate_payloads(
    preflight: StageB1AggregatePreflight,
    *,
    mechanism_gate: Mapping[str, Any],
) -> dict[str, bytes]:
    if not isinstance(preflight, StageB1AggregatePreflight):
        raise P3StageB1AggregateError("preflight has wrong type")
    lineage = _validate_lineage(preflight.lineage, preflight=preflight)
    if len(preflight.records) != TOTAL_EPISODE_COUNT:
        raise P3StageB1AggregateError("aggregate requires exactly 2496 episodes")
    datasets, conditions = _config_sequences(preflight.config)
    _validate_target_invariance(preflight.records, datasets, conditions)
    stratified = build_stratified_statistics(preflight.records, config=preflight.config)
    summaries = build_summary_statistics(preflight.records, config=preflight.config)
    mechanism = build_mechanism_evidence(
        preflight.records,
        config=preflight.config,
        config_sha256=preflight.config_sha256,
        mechanism_gate=mechanism_gate,
    )
    lineage_bytes = canonical_jsonl_bytes(lineage)
    stratified_bytes = canonical_jsonl_bytes(stratified)
    summary_bytes = canonical_jsonl_bytes(summaries)
    mechanism_bytes = canonical_json_bytes(mechanism, newline=True)
    code_seal = build_aggregate_code_seal(preflight.repository_root)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": preflight.protocol_id,
        "config_sha256": preflight.config_sha256,
        "scope": {
            "output_root": preflight.output_root_relative,
            "cell_count": EXPECTED_CELL_COUNT,
            "episodes_per_cell": EPISODES_PER_CELL,
            "episode_count": TOTAL_EPISODE_COUNT,
            "cell_order": "dataset_major_condition_minor",
            "source_train_pilot64_only": True,
        },
        "cell_lineage": {
            "path": CELL_LINEAGE_FILENAME,
            "sha256": hashlib.sha256(lineage_bytes).hexdigest(),
            "count": EXPECTED_CELL_COUNT,
            "target_identity_invariance_verified": True,
        },
        "stratified_statistics": {
            "path": STRATIFIED_STATISTICS_FILENAME,
            "sha256": hashlib.sha256(stratified_bytes).hexdigest(),
            "count": len(stratified),
            "accumulation_dtype": "float64",
        },
        "summary_statistics": {
            "path": SUMMARY_STATISTICS_FILENAME,
            "sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "count": len(summaries),
            "includes": [
                "overall", "nonclean", "clean", "dataset", "corruption_family",
                "severity", "NUAA-SIRST",
            ],
            "accumulation_dtype": "float64",
        },
        "mechanism_evidence": {
            "path": MECHANISM_EVIDENCE_FILENAME,
            "sha256": hashlib.sha256(mechanism_bytes).hexdigest(),
            "P0_flag_count": 3,
            "small_group_hint_count": 8,
            "selection_or_ranking_performed": False,
            "stage_b3_authorized": False,
        },
        "aggregate_code_seal": code_seal,
        "execution": {
            "cpu_only": True,
            "public_cell_verifier_count": EXPECTED_CELL_COUNT,
            "episode_parse_count": TOTAL_EPISODE_COUNT,
            "raw_image_open_count": 0,
            "raw_target_open_count": 0,
            "model_build_count": 0,
            "optimizer_build_count": 0,
            "gpu_compute_count": 0,
            "candidate_selection_count": 0,
        },
        "data_boundary": {
            "source_train_completed_cell_artifacts_only": True,
            "raw_train_images_opened": 0,
            "raw_train_targets_opened": 0,
            "validation_payloads_opened": 0,
            "test_split_files_opened": 0,
            "test_images_opened": 0,
            "test_targets_opened": 0,
        },
        "authorization": AGGREGATE_AUTHORIZATION,
    }
    manifest_bytes = canonical_json_bytes(manifest, newline=True)
    payload = {
        CELL_LINEAGE_FILENAME: lineage_bytes,
        STRATIFIED_STATISTICS_FILENAME: stratified_bytes,
        SUMMARY_STATISTICS_FILENAME: summary_bytes,
        MECHANISM_EVIDENCE_FILENAME: mechanism_bytes,
        MANIFEST_FILENAME: manifest_bytes,
    }
    complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": preflight.protocol_id,
        "complete": True,
        "config_sha256": preflight.config_sha256,
        "cell_count": EXPECTED_CELL_COUNT,
        "episode_count": TOTAL_EPISODE_COUNT,
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "payload_files": [
            {"path": name, "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in sorted(payload.items())
        ],
        "mechanism_evidence": {
            "path": MECHANISM_EVIDENCE_FILENAME,
            "sha256": hashlib.sha256(mechanism_bytes).hexdigest(),
        },
        "atomic_no_replace": True,
        "immutable": True,
        **AGGREGATE_AUTHORIZATION,
    }
    payload[COMPLETE_FILENAME] = canonical_json_bytes(complete, newline=True)
    return payload


def _validate_local_aggregate_members(
    *,
    by_name: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root_relative: str,
    config_sha256: str,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]], Mapping[str, Any]]:
    protocol_id = str(config["protocol_id"])
    manifest = parse_canonical_json(
        by_name[MANIFEST_FILENAME].data, label=MANIFEST_FILENAME, newline=True
    )
    expected_manifest_fields = {
        "schema_version", "artifact_type", "protocol_id", "config_sha256", "scope",
        "cell_lineage", "stratified_statistics", "summary_statistics",
        "mechanism_evidence", "aggregate_code_seal", "execution", "data_boundary",
        "authorization",
    }
    if set(manifest) != expected_manifest_fields or (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["artifact_type"] != ARTIFACT_TYPE
        or manifest["protocol_id"] != protocol_id
        or manifest["config_sha256"] != config_sha256
        or manifest["authorization"] != AGGREGATE_AUTHORIZATION
    ):
        raise P3StageB1AggregateError("aggregate manifest identity differs")
    scope = manifest["scope"]
    if scope != {
        "output_root": output_root_relative,
        "cell_count": EXPECTED_CELL_COUNT,
        "episodes_per_cell": EPISODES_PER_CELL,
        "episode_count": TOTAL_EPISODE_COUNT,
        "cell_order": "dataset_major_condition_minor",
        "source_train_pilot64_only": True,
    }:
        raise P3StageB1AggregateError("aggregate scope differs")
    lineage = parse_canonical_jsonl(
        by_name[CELL_LINEAGE_FILENAME].data,
        count=EXPECTED_CELL_COUNT,
        label=CELL_LINEAGE_FILENAME,
    )
    expected_strata = 3 * 13 * len(GROUP_IDS) * 2
    stratified = parse_canonical_jsonl(
        by_name[STRATIFIED_STATISTICS_FILENAME].data,
        count=expected_strata,
        label=STRATIFIED_STATISTICS_FILENAME,
    )
    summary_count = int(manifest["summary_statistics"]["count"])
    summaries = parse_canonical_jsonl(
        by_name[SUMMARY_STATISTICS_FILENAME].data,
        count=summary_count,
        label=SUMMARY_STATISTICS_FILENAME,
    )
    mechanism = parse_canonical_json(
        by_name[MECHANISM_EVIDENCE_FILENAME].data,
        label=MECHANISM_EVIDENCE_FILENAME,
        newline=True,
    )
    refs = (
        ("cell_lineage", CELL_LINEAGE_FILENAME, len(lineage)),
        ("stratified_statistics", STRATIFIED_STATISTICS_FILENAME, len(stratified)),
        ("summary_statistics", SUMMARY_STATISTICS_FILENAME, len(summaries)),
        ("mechanism_evidence", MECHANISM_EVIDENCE_FILENAME, None),
    )
    for field, filename, count in refs:
        ref = _mapping(manifest[field], label=f"manifest.{field}")
        if ref.get("path") != filename or ref.get("sha256") != by_name[filename].sha256:
            raise P3StageB1AggregateError(f"aggregate {field} reference differs")
        if count is not None and ref.get("count") != count:
            raise P3StageB1AggregateError(f"aggregate {field} count differs")
    if mechanism.get("authorization") != AGGREGATE_AUTHORIZATION or mechanism.get("selection") != {
        "selection_or_ranking_performed": False,
        "selected_candidates": [],
        "mechanism_flags_are_not_candidate_gate": True,
    }:
        raise P3StageB1AggregateError("mechanism evidence authorization differs")
    return manifest, lineage, stratified, summaries, mechanism


def verify_stage_b1_aggregate_shard(
    path: str | os.PathLike[str],
    *,
    repository_root: str | os.PathLike[str],
    output_root_relative: str | os.PathLike[str],
    config: Mapping[str, Any],
    expected_config_sha256: str,
    mechanism_gate: Mapping[str, Any],
    verify_live_cells: bool = True,
    expected_cell_code_seal: Mapping[str, Any] | None = None,
) -> VerifiedStageB1Aggregate:
    root = Path(os.path.abspath(os.fspath(path)))
    repository = Path(os.path.abspath(os.fspath(repository_root)))
    output_relative = _canonical_output_relative(output_root_relative)
    config_sha = _sha256(expected_config_sha256, label="expected_config_sha256")
    try:
        snapshot = snapshot_regular_directory(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise P3StageB1AggregateError(f"cannot snapshot B1 aggregate: {root}") from exc
    by_name = {member.path.name: member for member in snapshot.members}
    if set(by_name) != MEMBERS:
        raise P3StageB1AggregateError("aggregate member set differs")
    manifest, lineage, stratified, summaries, mechanism = _validate_local_aggregate_members(
        by_name=by_name,
        config=config,
        output_root_relative=output_relative,
        config_sha256=config_sha,
    )
    if manifest["aggregate_code_seal"] != build_aggregate_code_seal(repository):
        raise P3StageB1AggregateError("aggregate live code seal differs")
    complete = parse_canonical_json(
        by_name[COMPLETE_FILENAME].data, label=COMPLETE_FILENAME, newline=True
    )
    expected_complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "protocol_id": str(config["protocol_id"]),
        "complete": True,
        "config_sha256": config_sha,
        "cell_count": EXPECTED_CELL_COUNT,
        "episode_count": TOTAL_EPISODE_COUNT,
        "manifest": {"path": MANIFEST_FILENAME, "sha256": by_name[MANIFEST_FILENAME].sha256},
        "payload_files": [
            {"path": name, "sha256": by_name[name].sha256}
            for name in sorted(set(by_name) - {COMPLETE_FILENAME})
        ],
        "mechanism_evidence": {
            "path": MECHANISM_EVIDENCE_FILENAME,
            "sha256": by_name[MECHANISM_EVIDENCE_FILENAME].sha256,
        },
        "atomic_no_replace": True,
        "immutable": True,
        **AGGREGATE_AUTHORIZATION,
    }
    if complete != expected_complete:
        raise P3StageB1AggregateError("aggregate COMPLETE differs")
    if verify_live_cells:
        live = collect_stage_b1_preflight(
            repository_root=repository,
            output_root_relative=output_relative,
            config=config,
            config_sha256=config_sha,
            expected_code_seal=expected_cell_code_seal,
        )
        expected = build_stage_b1_aggregate_payloads(live, mechanism_gate=mechanism_gate)
        for name in MEMBERS:
            if by_name[name].data != expected[name]:
                raise P3StageB1AggregateError(
                    f"aggregate differs from live 39-cell rebuild: {name}"
                )
    # Locally validate every descriptive numeric/null schema even when a live
    # rebuild is intentionally disabled for an offline audit.
    for collection in (stratified, summaries):
        for row in collection:
            metrics = _mapping(row.get("metrics"), label="descriptive metrics")
            if set(metrics) != set(ALL_METRIC_IDS):
                raise P3StageB1AggregateError("descriptive metric order differs")
            for metric_id, statistic in metrics.items():
                stats = _mapping(statistic, label=f"statistics {metric_id}")
                if stats.get("status") not in {"estimable", "not_estimable"}:
                    raise P3StageB1AggregateError("descriptive status differs")
                for field in ("mean", "median", "q1", "q3", "minimum", "maximum"):
                    value = stats.get(field)
                    if value is not None and (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise P3StageB1AggregateError("descriptive value is non-finite")
    statuses = {
        key: str(value["status"]) for key, value in mechanism["P0_flags"].items()
    }
    return VerifiedStageB1Aggregate(
        path=root,
        manifest_sha256=by_name[MANIFEST_FILENAME].sha256,
        complete_sha256=by_name[COMPLETE_FILENAME].sha256,
        mechanism_evidence_sha256=by_name[MECHANISM_EVIDENCE_FILENAME].sha256,
        cell_count=EXPECTED_CELL_COUNT,
        episode_count=TOTAL_EPISODE_COUNT,
        stratified_record_count=len(stratified),
        summary_record_count=len(summaries),
        mechanism_statuses=statuses,
        candidate_selection_performed=False,
        stage_b3_authorized=False,
        p5_authorized=False,
    )


__all__ = [
    "AGGREGATE_AUTHORIZATION", "AGGREGATE_CRITICAL_CODE_PATHS", "ALL_METRIC_IDS",
    "ARTIFACT_TYPE", "CELL_LINEAGE_FILENAME", "COMPLETE_ARTIFACT_TYPE",
    "COMPLETE_FILENAME", "EPISODES_PER_CELL", "EXPECTED_CELL_COUNT",
    "LINEAGE_ARTIFACT_TYPE", "MANIFEST_FILENAME", "MECHANISM_ARTIFACT_TYPE",
    "MECHANISM_EVIDENCE_FILENAME", "MEMBERS", "METRIC_PATHS",
    "P0_MECHANISM_METRICS", "P3StageB1AggregateError", "SMALL_GROUP_METRICS",
    "STRATIFIED_STATISTICS_FILENAME", "STRATUM_ARTIFACT_TYPE",
    "SUMMARY_ARTIFACT_TYPE", "SUMMARY_STATISTICS_FILENAME", "StageB1AggregatePreflight",
    "StageB1CellPaths", "TARGET_PRESENCE_VALUES", "TOTAL_EPISODE_COUNT",
    "VerifiedStageB1Aggregate", "build_aggregate_code_seal",
    "build_mechanism_evidence", "build_stage_b1_aggregate_payloads",
    "build_stratified_statistics", "build_summary_statistics",
    "collect_stage_b1_preflight", "descriptive_statistics", "fixed_stage_b1_cells",
    "verify_stage_b1_aggregate_shard",
]
