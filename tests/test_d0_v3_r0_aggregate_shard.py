from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from analysis.d0_v3_formal_contract import CONDITIONS, DATASETS, FROZEN_CANDIDATES
from analysis.d0_v3_r0_aggregate_shard import (
    COMPLETE_FILENAME,
    D0V3R0AggregateError,
    EXPECTED_CELL_COUNT,
    MANIFEST_FILENAME,
    MEMBERS,
    R0AggregatePreflight,
    build_r0_aggregate_payloads,
    collect_r0_preflight,
    fixed_r0_cells,
    verify_r0_aggregate_shard,
)
from analysis.d0_v3_science_gate import (
    ALIGNMENT_OBSERVATIONS_PER_CANDIDATE_REPLICATE,
    EPISODES_PER_CANDIDATE_REPLICATE,
    SAFETY_STRATA,
    evaluate_stage_a_r0,
)
from tta.d0_v3_atomic_shard import publish_flat_directory_noreplace


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_RELATIVE = (
    "results/cr_sitta/tent_failure_diagnostics_v3_formal_stage_a"
)
CONFIG_SHA = "f0ed056520f840d1432b90b6426f55803afa1867b98108712dc242371dd1c05a"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_evidence(index: int, *, passing: bool) -> dict[str, Any]:
    candidate = FROZEN_CANDIDATES[index]
    nonclean = 0.003 if passing else 0.0
    overall = 0.002 if passing else 0.0
    return {
        "schema_version": 3,
        "artifact_type": "cr_sitta_d0_v3_stage_a_replicate_evidence",
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "optimizer": candidate.optimizer,
            "learning_rate": candidate.learning_rate,
        },
        "replicate_id": "R0",
        "episode_count": EPISODES_PER_CANDIDATE_REPLICATE,
        "metrics": {
            "nonclean_macro_delta_iou": nonclean,
            "overall_macro_delta_iou": overall,
            "dataset_nonclean_delta_iou": {
                "IRSTD-1K": 0.003 if passing else 0.0,
                "NUAA-SIRST": 0.002 if passing else 0.0,
                "NUDT-SIRST": -0.001,
            },
            "family_delta_iou": {
                "gaussian_noise": 0.003 if passing else 0.0,
                "gaussian_blur": 0.002 if passing else 0.0,
                "low_contrast": 0.001 if passing else 0.0,
                "stripe_noise": -0.004,
            },
            "clean_macro_delta_iou": 0.0,
            "clean_dataset_delta_iou": {dataset: 0.0 for dataset in DATASETS},
            "nonclean_macro_delta_pd": 0.0,
            "dataset_nonclean_delta_pd": {dataset: 0.0 for dataset in DATASETS},
            "clean_macro_delta_pd": 0.0,
        },
        "safety": {
            "source_fa_per_million": {key: 20.0 for key in SAFETY_STRATA},
            "fa_delta_per_million": {key: 0.0 for key in SAFETY_STRATA},
            "source_foreground_fraction": {key: 0.01 for key in SAFETY_STRATA},
            "adapted_foreground_fraction": {key: 0.0105 for key in SAFETY_STRATA},
        },
        "activity": {
            "finite_gradient_episodes": EPISODES_PER_CANDIDATE_REPLICATE,
            "parameter_changed_episodes": 2372,
            "functional_logit_threshold": 1.0e-6,
            "functional_logit_changed_episodes": 250,
            "threshold_crossing_episodes": 50,
            "metric_sufficient_count_changed_episodes": 50,
            "entropy_decrease_episodes": 1997,
            "both_gradients_nonzero_episodes": 1997,
            "fine_group_alignment_observation_count": (
                ALIGNMENT_OBSERVATIONS_PER_CANDIDATE_REPLICATE
            ),
        },
        "alignment": {
            "macro_cosine": 0.06,
            "dataset_median_cosine": {
                "IRSTD-1K": 0.1,
                "NUAA-SIRST": 0.1,
                "NUDT-SIRST": -0.1,
            },
        },
    }


def _lineage() -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    for cell in fixed_r0_cells(REPOSITORY, OUTPUT_RELATIVE):
        prefix = f"{cell.index}:{cell.dataset}:{cell.condition}"
        values.append(
            {
                "schema_version": 3,
                "artifact_type": (
                    "cr_sitta_d0_v3_formal_stage_a_r0_cell_lineage"
                ),
                "cell_index": cell.index,
                "dataset": cell.dataset,
                "condition": cell.condition,
                "replicate": "R0",
                "label_free_shard_path": cell.label_free_path.relative_to(
                    REPOSITORY
                ).as_posix(),
                "outer_shard_path": cell.outer_path.relative_to(
                    REPOSITORY
                ).as_posix(),
                "label_free_manifest_sha256": _digest(prefix + ":lm"),
                "label_free_complete_sha256": _digest(prefix + ":lc"),
                "label_free_phase_receipt_sha256": _digest(prefix + ":lp"),
                "outer_manifest_sha256": _digest(prefix + ":om"),
                "outer_complete_sha256": _digest(prefix + ":oc"),
                "outer_access_receipt_sha256": _digest(prefix + ":oa"),
                "outer_records_sha256": _digest(prefix + ":or"),
                "ordered_image_ids_sha256": _digest(cell.dataset + ":ids"),
                "train_split_sha256": _digest(cell.dataset + ":split"),
                "checkpoint_sha256": _digest(cell.dataset + ":checkpoint"),
                "outer_record_count": 640,
                "live_input_bindings_verified": True,
                "raw_target_reopened_by_aggregate": False,
                "stage2_authorized": False,
            }
        )
    return tuple(values)


def _preflight(*, passing: set[int]) -> R0AggregatePreflight:
    evidence = tuple(
        _candidate_evidence(index, passing=index in passing)
        for index in range(len(FROZEN_CANDIDATES))
    )
    decision = evaluate_stage_a_r0(evidence, protocol_status="passed").to_receipt()
    return R0AggregatePreflight(
        repository_root=REPOSITORY,
        output_root_relative=OUTPUT_RELATIVE,
        config_sha256=CONFIG_SHA,
        lineage=_lineage(),
        evidence=evidence,
        decision_receipt=decision,
    )


def _artifact(tmp_path: Path, *, passing: set[int]) -> Path:
    destination = tmp_path / "aggregate"
    destination.mkdir(parents=True)
    payloads = build_r0_aggregate_payloads(_preflight(passing=passing))
    assert set(payloads) == MEMBERS
    for name, payload in payloads.items():
        (destination / name).write_bytes(payload)
    return destination


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_no_eligible_is_normal_formal_early_stop_and_forbids_r1_r2_stage2(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, passing=set())
    verified = verify_r0_aggregate_shard(
        artifact,
        repository_root=REPOSITORY,
        output_root_relative=OUTPUT_RELATIVE,
        expected_config_sha256=CONFIG_SHA,
        verify_live_cells=False,
    )
    assert verified.cell_count == 39
    assert verified.outer_record_count == 39 * 640
    assert verified.evidence_count == 10
    assert verified.eligible_candidate_ids == ()
    assert verified.required_followup_replicates == ()
    assert verified.formal_stage_a_protocol_complete is True
    assert verified.scientific_status == "scientific_no_eligible"

    complete = _json(artifact / COMPLETE_FILENAME)
    assert complete["complete"] is True
    assert complete["followup_scope"]["r1_r2_forbidden"] is True
    assert complete["followup_scope"]["required_replicates"] == []
    assert complete["paper_result"] is False
    assert complete["paper_test_result"] is False
    assert complete["stage2_authorized"] is False
    manifest = _json(artifact / MANIFEST_FILENAME)
    assert manifest["data_boundary"]["raw_gt_opened_by_aggregate"] == 0
    assert manifest["data_boundary"]["test_labels_opened"] == 0
    assert manifest["execution"]["gpu_compute_count"] == 0
    assert manifest["authorization"]["stage2_authorized"] is False


def test_eligible_receipt_lists_exact_candidates_and_only_required_r1_r2(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, passing={0, 3})
    verified = verify_r0_aggregate_shard(
        artifact,
        repository_root=REPOSITORY,
        output_root_relative=OUTPUT_RELATIVE,
        expected_config_sha256=CONFIG_SHA,
        verify_live_cells=False,
    )
    assert verified.eligible_candidate_ids == (
        FROZEN_CANDIDATES[0].candidate_id,
        FROZEN_CANDIDATES[3].candidate_id,
    )
    assert verified.required_followup_replicates == ("R1", "R2")
    assert verified.formal_stage_a_protocol_complete is False
    complete = _json(artifact / COMPLETE_FILENAME)
    assert complete["followup_scope"] == {
        "policy": "required_eligible_candidates_only",
        "eligible_candidates": [
            {
                "candidate_id": FROZEN_CANDIDATES[0].candidate_id,
                "optimizer": FROZEN_CANDIDATES[0].optimizer,
                "learning_rate": FROZEN_CANDIDATES[0].learning_rate,
            },
            {
                "candidate_id": FROZEN_CANDIDATES[3].candidate_id,
                "optimizer": FROZEN_CANDIDATES[3].optimizer,
                "learning_rate": FROZEN_CANDIDATES[3].learning_rate,
            },
        ],
        "required_replicates": ["R1", "R2"],
        "all_noneligible_candidates_forbidden": True,
        "r1_r2_forbidden": False,
        "stage2_authorized": False,
    }
    assert complete["stage2_authorized"] is False


def test_public_verifier_rejects_byte_tampering(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, passing=set())
    manifest = artifact / MANIFEST_FILENAME
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(D0V3R0AggregateError, match="member differs"):
        verify_r0_aggregate_shard(
            artifact,
            repository_root=REPOSITORY,
            output_root_relative=OUTPUT_RELATIVE,
            expected_config_sha256=CONFIG_SHA,
            verify_live_cells=False,
        )


def test_missing_or_duplicate_cell_lineage_fails_before_payload_creation() -> None:
    base = _preflight(passing=set())
    missing = R0AggregatePreflight(
        repository_root=base.repository_root,
        output_root_relative=base.output_root_relative,
        config_sha256=base.config_sha256,
        lineage=base.lineage[:-1],
        evidence=base.evidence,
        decision_receipt=base.decision_receipt,
    )
    with pytest.raises(D0V3R0AggregateError, match="exactly 39"):
        build_r0_aggregate_payloads(missing)

    duplicated_values = [copy.deepcopy(value) for value in base.lineage]
    duplicated_values[1] = copy.deepcopy(duplicated_values[0])
    duplicated = R0AggregatePreflight(
        repository_root=base.repository_root,
        output_root_relative=base.output_root_relative,
        config_sha256=base.config_sha256,
        lineage=tuple(duplicated_values),
        evidence=base.evidence,
        decision_receipt=base.decision_receipt,
    )
    with pytest.raises(D0V3R0AggregateError, match="identity/order|duplicate"):
        build_r0_aggregate_payloads(duplicated)

    split_drift_values = [copy.deepcopy(value) for value in base.lineage]
    split_drift_values[1]["train_split_sha256"] = "e" * 64
    split_drift = R0AggregatePreflight(
        repository_root=base.repository_root,
        output_root_relative=base.output_root_relative,
        config_sha256=base.config_sha256,
        lineage=tuple(split_drift_values),
        evidence=base.evidence,
        decision_receipt=base.decision_receipt,
    )
    with pytest.raises(D0V3R0AggregateError, match="one fixed train_split"):
        build_r0_aggregate_payloads(split_drift)


def test_authorization_tamper_cannot_be_made_self_consistent_locally(
    tmp_path: Path,
) -> None:
    artifact = _artifact(tmp_path, passing=set())
    manifest_path = artifact / MANIFEST_FILENAME
    manifest = _json(manifest_path)
    manifest["authorization"]["stage2_authorized"] = True
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(D0V3R0AggregateError, match="member differs"):
        verify_r0_aggregate_shard(
            artifact,
            repository_root=REPOSITORY,
            output_root_relative=OUTPUT_RELATIVE,
            expected_config_sha256=CONFIG_SHA,
            verify_live_cells=False,
        )


def test_collect_uses_exact_39_fixed_cells_and_creates_no_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    dummy = {"not": "parsed here"}

    def fake_cell(cell: Any, **_: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        calls.append((cell.dataset, cell.condition))
        lineage = copy.deepcopy(_lineage()[cell.index])
        return lineage, [dummy] * 640

    expected_evidence = tuple(
        _candidate_evidence(index, passing=False)
        for index in range(len(FROZEN_CANDIDATES))
    )
    monkeypatch.setattr(
        "analysis.d0_v3_r0_aggregate_shard._read_verified_cell", fake_cell
    )
    monkeypatch.setattr(
        "analysis.d0_v3_r0_aggregate_shard.build_replicate_evidence_set",
        lambda records, *, replicate_id: list(expected_evidence),
    )
    value = collect_r0_preflight(
        repository_root=REPOSITORY,
        output_root_relative=OUTPUT_RELATIVE,
        config_sha256=CONFIG_SHA,
    )
    assert calls == [
        (dataset, condition) for dataset in DATASETS for condition in CONDITIONS
    ]
    assert len(calls) == EXPECTED_CELL_COUNT
    assert len(value.lineage) == EXPECTED_CELL_COUNT
    assert value.decision_receipt["stage2_allowed"] is False


def test_live_public_verifier_requires_live_rebuild_to_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(tmp_path, passing=set())
    valid = _preflight(passing=set())
    calls = 0

    def same(**_: Any) -> R0AggregatePreflight:
        nonlocal calls
        calls += 1
        return valid

    monkeypatch.setattr(
        "analysis.d0_v3_r0_aggregate_shard.collect_r0_preflight", same
    )
    verify_r0_aggregate_shard(
        artifact,
        repository_root=REPOSITORY,
        output_root_relative=OUTPUT_RELATIVE,
        expected_config_sha256=CONFIG_SHA,
        verify_live_cells=True,
    )
    assert calls == 1

    changed = copy.deepcopy(list(valid.lineage))
    changed[0]["outer_complete_sha256"] = "f" * 64
    mismatch = R0AggregatePreflight(
        repository_root=valid.repository_root,
        output_root_relative=valid.output_root_relative,
        config_sha256=valid.config_sha256,
        lineage=tuple(changed),
        evidence=valid.evidence,
        decision_receipt=valid.decision_receipt,
    )
    monkeypatch.setattr(
        "analysis.d0_v3_r0_aggregate_shard.collect_r0_preflight",
        lambda **_: mismatch,
    )
    with pytest.raises(D0V3R0AggregateError, match="live 39-cell rebuild differs"):
        verify_r0_aggregate_shard(
            artifact,
            repository_root=REPOSITORY,
            output_root_relative=OUTPUT_RELATIVE,
            expected_config_sha256=CONFIG_SHA,
            verify_live_cells=True,
        )


def test_aggregate_publication_is_immutable_atomic_and_never_replaces(
    tmp_path: Path,
) -> None:
    staging = _artifact(tmp_path / "first", passing=set())
    # Production likewise renames a private staging directory to ``R0`` in
    # the same aggregate_phase parent.
    destination = staging.parent / "R0"

    def verifier(path: Path) -> object:
        return verify_r0_aggregate_shard(
            path,
            repository_root=REPOSITORY,
            output_root_relative=OUTPUT_RELATIVE,
            expected_config_sha256=CONFIG_SHA,
            verify_live_cells=False,
        )

    published = publish_flat_directory_noreplace(
        staging,
        destination,
        expected_members=tuple(MEMBERS),
        semantic_verifier=verifier,
    )
    original_complete = (published / COMPLETE_FILENAME).read_bytes()
    assert published == destination
    assert (published.stat().st_mode & 0o222) == 0

    second = _artifact(tmp_path / "second", passing=set())
    with pytest.raises(FileExistsError, match="already exists"):
        publish_flat_directory_noreplace(
            second,
            destination,
            expected_members=tuple(MEMBERS),
            semantic_verifier=verifier,
        )
    assert (destination / COMPLETE_FILENAME).read_bytes() == original_complete
