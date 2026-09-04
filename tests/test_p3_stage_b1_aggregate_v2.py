from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from analysis import p3_stage_b1_aggregate_v2 as aggregate


PROTOCOL = "cr-sitta-p3-stage-b1-gradient-decomposition-v2"
CONFIG_SHA = "a" * 64
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CONDITIONS = (
    "clean_S0",
    "gaussian_noise_S1", "gaussian_noise_S3", "gaussian_noise_S5",
    "gaussian_blur_S1", "gaussian_blur_S3", "gaussian_blur_S5",
    "low_contrast_S1", "low_contrast_S3", "low_contrast_S5",
    "stripe_noise_S1", "stripe_noise_S3", "stripe_noise_S5",
)
OUTPUT = "results/test_stage_b1_v2"
REPOSITORY = Path(__file__).resolve().parents[1]


def _config() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL,
        "datasets": {dataset: {} for dataset in DATASETS},
        "conditions": list(CONDITIONS),
    }


def _mechanism_gate() -> dict[str, Any]:
    value = yaml.safe_load(
        (REPOSITORY / "configs/p3_stage_b_gradient_decomposition_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    return copy.deepcopy(value["mechanism_evidence_flags"])


def _alignment(cosine: float | None, projection: float | None) -> dict[str, Any]:
    return {
        "entropy_gradient_norm": None if cosine is None else 1.0,
        "task_gradient_norm": 1.0,
        "entropy_task_dot": cosine,
        "entropy_task_cosine": cosine,
        "task_projection": projection,
        "unit_descent_task_change": None if cosine is None else -cosine,
    }


def _group(present: bool) -> dict[str, Any]:
    full = _alignment(0.10, 0.05)
    foreground = _alignment(0.10 if present else None, 0.40 if present else None)
    background = _alignment(-0.10, -0.30)
    sub = _alignment(-0.20 if present else None, 0.20 if present else None)
    supra = _alignment(0.20 if present else None, 0.20 if present else None)
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
            "foreground_subthreshold_entropy_add": sub,
            "foreground_suprathreshold_entropy_add": supra,
        },
        "conditional_entropy_task_alignment": {
            "full_entropy_mean": full,
            "foreground_entropy_mean": foreground,
            "background_entropy_mean": background,
            "foreground_subthreshold_entropy_mean": sub,
            "foreground_suprathreshold_entropy_mean": supra,
        },
        "cross_region": {
            "foreground_background_additive_alignment": {
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


def _raw_consistency(*, within: bool = True, error: float = 0.0) -> dict[str, Any]:
    component = {
        "max_abs_error": error,
        "l2_error": error,
        "relative_l2_error": error,
        "reference_l2": 1.0,
        "observed_l2": 1.0,
        "max_abs_tolerance": 1e-7,
        "relative_l2_tolerance": 1e-4,
        "within_dual_tolerance": within,
    }
    return {
        "role": "numeric_diagnostic_only",
        "failure_action": "record_only_never_protocol_or_science_gate",
        "enters_science_metrics": False,
        "enters_mechanism_flags": False,
        "enters_selection_or_ranking": False,
        "components": {
            "foreground_suprathreshold": dict(component),
            "background": dict(component),
        },
    }


def _records(*, raw_within: bool = True, raw_error: float = 0.0):
    values = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            family, severity = aggregate._condition_parts(condition)
            for index in range(64):
                present = index != 0
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
                            "target_slice_sha256": hashlib.sha256(
                                f"{dataset}:target:{index}".encode()
                            ).hexdigest(),
                        },
                        "gradient_integrity": {
                            "raw_consistency": _raw_consistency(
                                within=raw_within, error=raw_error
                            )
                        },
                        "groups": {
                            group_id: _group(present) for group_id in aggregate.GROUP_IDS
                        },
                    }
                )
    assert len(values) == aggregate.TOTAL_EPISODE_COUNT
    return tuple(values)


def _lineage() -> tuple[dict[str, Any], ...]:
    ids = {
        dataset: hashlib.sha256(f"{dataset}:ids".encode()).hexdigest()
        for dataset in DATASETS
    }
    targets = {
        dataset: hashlib.sha256(f"{dataset}:targets".encode()).hexdigest()
        for dataset in DATASETS
    }
    rows = []
    for cell in aggregate.fixed_stage_b1_cells(REPOSITORY, OUTPUT, config=_config()):
        family, severity = aggregate._condition_parts(cell.condition)
        prefix = f"{cell.dataset}:{cell.condition}"
        digest = lambda suffix: hashlib.sha256((prefix + suffix).encode()).hexdigest()
        rows.append(
            {
                "schema_version": 2,
                "artifact_type": aggregate.LINEAGE_ARTIFACT_TYPE,
                "cell_index": cell.index,
                "dataset": cell.dataset,
                "condition": cell.condition,
                "corruption_family": family,
                "severity": severity,
                "replicate": "R0",
                "cell_path": cell.path.relative_to(REPOSITORY).as_posix(),
                "manifest_sha256": digest(":m"),
                "complete_sha256": digest(":c"),
                "cumulative_vjps_filename": "cumulative_and_raw_entropy_vjps.npy",
                "cumulative_vjps_sha256": digest(":v"),
                "records_sha256": digest(":r"),
                "group_layout_sha256": digest(":g"),
                "outer_access_receipt_sha256": digest(":o"),
                "ordered_image_ids_sha256": ids[cell.dataset],
                "target_identity_sha256": targets[cell.dataset],
                "record_count": 64,
                "public_v2_cell_verifier_passed": True,
                "raw_target_reopened_by_aggregate": False,
                "raw_vjps_enter_science_metrics": False,
                "candidate_selection_performed": False,
                "stage_b3_authorized": False,
                "p5_authorized": False,
            }
        )
    return tuple(rows)


def _preflight(records=None) -> aggregate.StageB1AggregatePreflight:
    return aggregate.StageB1AggregatePreflight(
        repository_root=REPOSITORY,
        output_root_relative=OUTPUT,
        protocol_id=PROTOCOL,
        config_sha256=CONFIG_SHA,
        config=_config(),
        lineage=_lineage(),
        records=_records() if records is None else records,
    )


def test_v2_mechanism_flags_equal_v1_rules_and_never_authorize() -> None:
    result = aggregate.build_mechanism_evidence(
        _records(),
        config=_config(),
        config_sha256=CONFIG_SHA,
        mechanism_gate=_mechanism_gate(),
    )
    assert {key: value["status"] for key, value in result["P0_flags"].items()} == {
        "background_norm_dominance": "supported",
        "background_cancellation": "supported",
        "subthreshold_erasure": "supported",
    }
    assert result["numeric_estimator"]["raw_direct_audit_used"] is False
    assert result["selection"]["selected_candidates"] == []
    assert result["authorization"]["stage_b3_authorized"] is False


def test_raw_audit_changes_do_not_change_science_or_coverage() -> None:
    good = _records(raw_within=True, raw_error=0.0)
    bad = _records(raw_within=False, raw_error=9.0)
    good_science = aggregate.build_mechanism_evidence(
        good, config=_config(), config_sha256=CONFIG_SHA, mechanism_gate=_mechanism_gate()
    )
    bad_science = aggregate.build_mechanism_evidence(
        bad, config=_config(), config_sha256=CONFIG_SHA, mechanism_gate=_mechanism_gate()
    )
    assert good_science == bad_science
    audit = aggregate.build_raw_numeric_audit(
        bad, protocol_id=PROTOCOL, config_sha256=CONFIG_SHA
    )
    assert audit["components"]["background"]["outside_dual_tolerance_count"] == 2496
    assert all(value is False for value in audit["science_firewall"].values())


def test_aggregate_build_and_offline_verify_rebuild_hashes(tmp_path: Path) -> None:
    payloads = aggregate.build_stage_b1_aggregate_payloads(
        _preflight(_records(raw_within=False, raw_error=1.0)),
        mechanism_gate=_mechanism_gate(),
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
    assert verified.stage_b3_authorized is False
    assert verified.candidate_selection_performed is False


def test_target_drift_and_raw_firewall_tamper_fail_closed() -> None:
    drift = [copy.deepcopy(value) for value in _records()]
    drift[64]["target"]["target_slice_sha256"] = "f" * 64
    with pytest.raises(aggregate.P3StageB1AggregateV2Error, match="target identity drifts"):
        aggregate.build_stage_b1_aggregate_payloads(
            _preflight(tuple(drift)), mechanism_gate=_mechanism_gate()
        )
    firewall = [copy.deepcopy(value) for value in _records()]
    firewall[0]["gradient_integrity"]["raw_consistency"]["enters_mechanism_flags"] = True
    with pytest.raises(aggregate.P3StageB1AggregateV2Error, match="firewall"):
        aggregate.build_stage_b1_aggregate_payloads(
            _preflight(tuple(firewall)), mechanism_gate=_mechanism_gate()
        )


def test_lineage_schema_count_and_boundary_tamper_fail_closed() -> None:
    for field, value in (
        ("record_count", 63),
        ("cell_path", "results/elsewhere"),
        ("raw_target_reopened_by_aggregate", True),
    ):
        preflight = _preflight()
        lineage = [copy.deepcopy(row) for row in preflight.lineage]
        lineage[0][field] = value
        forged = aggregate.StageB1AggregatePreflight(
            repository_root=preflight.repository_root,
            output_root_relative=preflight.output_root_relative,
            protocol_id=preflight.protocol_id,
            config_sha256=preflight.config_sha256,
            config=preflight.config,
            lineage=tuple(lineage),
            records=preflight.records,
        )
        with pytest.raises(
            aggregate.P3StageB1AggregateV2Error,
            match="lineage identity/firewall differs",
        ):
            aggregate.build_stage_b1_aggregate_payloads(
                forged, mechanism_gate=_mechanism_gate()
            )

    preflight = _preflight()
    lineage = [copy.deepcopy(row) for row in preflight.lineage]
    lineage[0]["unknown_field"] = "forbidden"
    forged = aggregate.StageB1AggregatePreflight(
        repository_root=preflight.repository_root,
        output_root_relative=preflight.output_root_relative,
        protocol_id=preflight.protocol_id,
        config_sha256=preflight.config_sha256,
        config=preflight.config,
        lineage=tuple(lineage),
        records=preflight.records,
    )
    with pytest.raises(
        aggregate.P3StageB1AggregateV2Error, match="lineage record fields differ"
    ):
        aggregate.build_stage_b1_aggregate_payloads(
            forged, mechanism_gate=_mechanism_gate()
        )


def test_aggregate_byte_tamper_is_rejected(tmp_path: Path) -> None:
    payloads = aggregate.build_stage_b1_aggregate_payloads(
        _preflight(), mechanism_gate=_mechanism_gate()
    )
    artifact = tmp_path / "R0"
    artifact.mkdir()
    for name, value in payloads.items():
        (artifact / name).write_bytes(value)
    target = artifact / aggregate.RAW_NUMERIC_AUDIT_FILENAME
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(aggregate.P3StageB1AggregateV2Error):
        aggregate.verify_stage_b1_aggregate_shard(
            artifact,
            repository_root=REPOSITORY,
            output_root_relative=OUTPUT,
            config=_config(),
            expected_config_sha256=CONFIG_SHA,
            mechanism_gate=_mechanism_gate(),
            verify_live_cells=False,
        )


def test_live_collection_and_verification_require_explicit_cell_seal(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        aggregate.P3StageB1AggregateV2Error,
        match="requires the expected cell code seal",
    ):
        aggregate.collect_stage_b1_preflight(
            repository_root=tmp_path,
            output_root_relative=OUTPUT,
            config=_config(),
            config_sha256=CONFIG_SHA,
            expected_code_seal=None,
        )
    with pytest.raises(
        aggregate.P3StageB1AggregateV2Error,
        match="requires the expected cell code seal",
    ):
        aggregate.verify_stage_b1_aggregate_shard(
            tmp_path / "missing",
            repository_root=tmp_path,
            output_root_relative=OUTPUT,
            config=_config(),
            expected_config_sha256=CONFIG_SHA,
            mechanism_gate=_mechanism_gate(),
            verify_live_cells=True,
            expected_cell_code_seal=None,
        )
