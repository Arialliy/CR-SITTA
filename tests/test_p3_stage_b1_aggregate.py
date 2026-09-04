from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from analysis import p3_stage_b1_aggregate as aggregate
from analysis.d0_v3_label_free_shard import canonical_json_bytes


PROTOCOL = "cr-sitta-p3-stage-b1-gradient-decomposition-v1"
CONFIG_SHA = "a" * 64
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CONDITIONS = (
    "clean_S0",
    "gaussian_noise_S1",
    "gaussian_noise_S3",
    "gaussian_noise_S5",
    "gaussian_blur_S1",
    "gaussian_blur_S3",
    "gaussian_blur_S5",
    "low_contrast_S1",
    "low_contrast_S3",
    "low_contrast_S5",
    "stripe_noise_S1",
    "stripe_noise_S3",
    "stripe_noise_S5",
)
OUTPUT = "results/test_stage_b1"
REPOSITORY = Path(__file__).resolve().parents[1]


def _config() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL,
        "datasets": {dataset: {} for dataset in DATASETS},
        "conditions": list(CONDITIONS),
    }


def _mechanism_gate() -> dict[str, Any]:
    parsed = yaml.safe_load(
        (REPOSITORY / "configs/p3_stage_b_gradient_decomposition_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    return copy.deepcopy(parsed["mechanism_evidence_flags"])


def _alignment(*, cosine: float | None, projection: float | None) -> dict[str, Any]:
    return {
        "entropy_gradient_norm": None if cosine is None else 1.0,
        "task_gradient_norm": 1.0,
        "entropy_task_dot": cosine,
        "entropy_task_cosine": cosine,
        "task_projection": projection,
        "unit_descent_task_change": None if cosine is None else -cosine,
    }


def _group(*, present: bool, small_positive: bool = True) -> dict[str, Any]:
    small_cosine = 0.10 if small_positive else -0.10
    full = _alignment(cosine=small_cosine, projection=0.05)
    foreground = _alignment(
        cosine=small_cosine if present else None,
        projection=0.40 if present else None,
    )
    background = _alignment(cosine=-0.10, projection=-0.30)
    subthreshold = _alignment(
        cosine=-0.20 if present else None,
        projection=0.20 if present else None,
    )
    suprathreshold = _alignment(
        cosine=0.20 if present else None,
        projection=0.20 if present else None,
    )
    return {
        "task_gradient_norm": 1.0,
        "additive_gradient_norms": {
            "foreground_subthreshold_add": 1.0 if present else None,
            "foreground_suprathreshold_add": 1.0 if present else None,
            "background_add": 2.0,
            "foreground_add": 1.0 if present else None,
            "full_add": 1.5,
        },
        "additive_entropy_task_alignment": {
            "full_entropy_mean": full,
            "foreground_entropy_add": foreground,
            "background_entropy_add": background,
            "foreground_subthreshold_entropy_add": subthreshold,
            "foreground_suprathreshold_entropy_add": suprathreshold,
        },
        "conditional_entropy_task_alignment": {
            "full_entropy_mean": full,
            "foreground_entropy_mean": foreground,
            "background_entropy_mean": background,
            "foreground_subthreshold_entropy_mean": subthreshold,
            "foreground_suprathreshold_entropy_mean": suprathreshold,
        },
        "cross_region": {
            "foreground_background_additive_alignment": {
                # Positive on purpose: the frozen cancellation flag must use
                # the joint projection predicate, not this auxiliary cosine.
                "foreground_background_cosine": 0.75 if present else None,
                "foreground_background_dot": 0.75 if present else None,
            },
            "background_to_foreground_additive_norm_ratio": {
                "value": 2.0 if present else None,
            },
            "background_to_foreground_conditional_norm_ratio": {
                "value": 2.0 if present else None,
            },
            "projection_cancellation_ratio": {
                "value": 0.9 if present else None,
                "foreground_task_dot": 0.4 if present else None,
                "background_task_dot": -0.3 if present else None,
                "full_task_dot": 0.05,
            },
        },
    }


def _records() -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            family, severity = aggregate._condition_parts(condition)
            for index in range(64):
                present = index != 0
                target_hash = hashlib.sha256(
                    f"{dataset}:target:{index}".encode()
                ).hexdigest()
                values.append(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "corruption_family": family,
                        "severity": severity,
                        "image_index": index,
                        "image_id": f"{dataset}_pilot_{index:02d}",
                        "target": {
                            "target_present": present,
                            "total_pixel_count": 65536,
                            "foreground_pixel_count": 8 if present else 0,
                            "background_pixel_count": 65528 if present else 65536,
                            "foreground_subthreshold_pixel_count": 4 if present else 0,
                            "foreground_suprathreshold_pixel_count": 4 if present else 0,
                            "target_value_sum": 8.0 if present else 0.0,
                            "target_slice_sha256": target_hash,
                        },
                        "groups": {
                            group_id: _group(present=present)
                            for group_id in aggregate.GROUP_IDS
                        },
                    }
                )
    assert len(values) == aggregate.TOTAL_EPISODE_COUNT
    return tuple(values)


def _lineage() -> tuple[dict[str, Any], ...]:
    config = _config()
    result: list[dict[str, Any]] = []
    dataset_target_sha = {
        dataset: hashlib.sha256(f"{dataset}:targets".encode()).hexdigest()
        for dataset in DATASETS
    }
    dataset_ids_sha = {
        dataset: hashlib.sha256(f"{dataset}:ids".encode()).hexdigest()
        for dataset in DATASETS
    }
    for cell in aggregate.fixed_stage_b1_cells(REPOSITORY, OUTPUT, config=config):
        family, severity = aggregate._condition_parts(cell.condition)
        prefix = f"{cell.dataset}:{cell.condition}"
        result.append(
            {
                "schema_version": 1,
                "artifact_type": aggregate.LINEAGE_ARTIFACT_TYPE,
                "cell_index": cell.index,
                "dataset": cell.dataset,
                "condition": cell.condition,
                "corruption_family": family,
                "severity": severity,
                "replicate": "R0",
                "cell_path": cell.path.relative_to(REPOSITORY).as_posix(),
                "manifest_sha256": hashlib.sha256((prefix + ":m").encode()).hexdigest(),
                "complete_sha256": hashlib.sha256((prefix + ":c").encode()).hexdigest(),
                "basis_sha256": hashlib.sha256((prefix + ":b").encode()).hexdigest(),
                "records_sha256": hashlib.sha256((prefix + ":r").encode()).hexdigest(),
                "group_layout_sha256": hashlib.sha256((prefix + ":g").encode()).hexdigest(),
                "outer_access_receipt_sha256": hashlib.sha256((prefix + ":o").encode()).hexdigest(),
                "ordered_image_ids_sha256": dataset_ids_sha[cell.dataset],
                "target_identity_sha256": dataset_target_sha[cell.dataset],
                "record_count": 64,
                "public_cell_verifier_passed": True,
                "raw_target_reopened_by_aggregate": False,
                "candidate_selection_performed": False,
                "stage_b3_authorized": False,
                "p5_authorized": False,
            }
        )
    return tuple(result)


def _preflight(records: tuple[dict[str, Any], ...] | None = None) -> aggregate.StageB1AggregatePreflight:
    return aggregate.StageB1AggregatePreflight(
        repository_root=REPOSITORY,
        output_root_relative=OUTPUT,
        protocol_id=PROTOCOL,
        config_sha256=CONFIG_SHA,
        config=_config(),
        lineage=_lineage(),
        records=_records() if records is None else records,
    )


def test_descriptive_empty_stratum_is_null_not_estimable() -> None:
    result = aggregate.descriptive_statistics([])
    assert result["status"] == "not_estimable"
    assert result["mean"] is None
    assert result["median"] is None
    assert result["q1"] is None
    assert result["q3"] is None
    assert result["finite_non_null_fraction"] is None


def test_joint_projection_cancellation_drives_flag_not_auxiliary_cosine() -> None:
    result = aggregate.build_mechanism_evidence(
        _records(),
        config=_config(),
        config_sha256=CONFIG_SHA,
        mechanism_gate=_mechanism_gate(),
    )
    flags = result["P0_flags"]
    assert flags["background_norm_dominance"]["status"] == "supported"
    assert flags["subthreshold_erasure"]["status"] == "supported"
    cancellation = flags["background_cancellation"]
    assert cancellation["status"] == "supported"
    assert cancellation["overall"]["joint_predicate"]["passed"] is True
    assert (
        cancellation["foreground_background_cosine_auxiliary_only"]
        ["equal_cell_macro_mean"]
        > 0.0
    )
    assert result["selection"]["selected_candidates"] == []
    assert result["authorization"]["stage_b3_authorized"] is False


def test_joint_cancellation_requires_same_episode_support_in_every_cell() -> None:
    records = [copy.deepcopy(value) for value in _records()]
    for record in records:
        if (
            record["dataset"] == DATASETS[0]
            and record["condition"] == "gaussian_noise_S1"
            and record["target"]["target_present"]
        ):
            record["groups"]["P0"]["additive_entropy_task_alignment"][
                "background_entropy_add"
            ]["task_projection"] = None
    result = aggregate.build_mechanism_evidence(
        records,
        config=_config(),
        config_sha256=CONFIG_SHA,
        mechanism_gate=_mechanism_gate(),
    )
    cancellation = result["P0_flags"]["background_cancellation"]
    assert cancellation["status"] == "not_estimable"
    assert cancellation["overall"]["joint_valid_non_null_cell_count"] == 35


def test_aggregate_build_and_local_verify_are_non_authorizing(tmp_path: Path) -> None:
    payloads = aggregate.build_stage_b1_aggregate_payloads(
        _preflight(), mechanism_gate=_mechanism_gate()
    )
    assert set(payloads) == aggregate.MEMBERS
    artifact = tmp_path / "R0"
    artifact.mkdir()
    for name, value in payloads.items():
        (artifact / name).write_bytes(value)
    verified = aggregate.verify_stage_b1_aggregate_shard(
        artifact,
        repository_root=REPOSITORY,
        output_root_relative=OUTPUT,
        config=_config(),
        expected_config_sha256=CONFIG_SHA,
        mechanism_gate=_mechanism_gate(),
        verify_live_cells=False,
    )
    assert verified.cell_count == 39
    assert verified.episode_count == 2496
    assert verified.stratified_record_count == 390
    assert verified.mechanism_statuses == {
        "background_cancellation": "supported",
        "background_norm_dominance": "supported",
        "subthreshold_erasure": "supported",
    }
    assert verified.candidate_selection_performed is False
    assert verified.stage_b3_authorized is False
    assert verified.p5_authorized is False


def test_target_identity_drift_across_conditions_fails_closed() -> None:
    records = [copy.deepcopy(value) for value in _records()]
    records[64]["target"]["target_slice_sha256"] = "f" * 64
    with pytest.raises(aggregate.P3StageB1AggregateError, match="target identity drifts"):
        aggregate.build_stage_b1_aggregate_payloads(
            _preflight(tuple(records)), mechanism_gate=_mechanism_gate()
        )
