from __future__ import annotations

import copy

import pytest

from analysis.d0_v3_formal_contract import (
    CONDITIONS,
    DATASETS,
    FINE_ALIGNMENT_GROUP_IDS,
    FROZEN_CANDIDATES,
)
from analysis.d0_v3_replicate_aggregate import (
    D0V3ReplicateAggregateError,
    EPISODE_ARTIFACT_TYPE,
    build_candidate_replicate_evidence,
    build_replicate_evidence_set,
)
from analysis.d0_v3_science_gate import (
    evaluate_replicate_hard_gates,
    parse_replicate_evidence,
)


def _counts(*, adapted: bool) -> dict[str, int]:
    if adapted:
        tp, fp, fn, tn, false_alarm = 6, 4, 4, 86, 4
    else:
        tp, fp, fn, tn, false_alarm = 5, 5, 5, 85, 5
    return {
        "detected_targets": 1,
        "false_alarm_pixels": false_alarm,
        "false_negative_pixels": fn,
        "false_positive_components": 1,
        "false_positive_pixels": fp,
        "intersection_pixels": tp,
        "predicted_positive_pixels": tp + fp,
        "target_positive_pixels": tp + fn,
        "total_image_pixels": tp + fp + fn + tn,
        "total_targets": 1,
        "true_negative_pixels": tn,
        "union_pixels": tp + fp + fn,
    }


def _alignment_metrics(
    *, scalar_count: int, parameter_tensor_count: int | None = None
) -> dict:
    result = {
        "scalar_count": scalar_count,
        "entropy_gradient_norm": 1.0,
        "supervised_gradient_norm": 1.0,
        "adaptation_step_norm": 0.1,
        "entropy_supervised_dot": 0.5,
        "entropy_supervised_cosine": 0.5,
        "supervised_dot_adaptation_step": -0.1,
        "first_order_task_loss_change": -0.1,
        "first_order_task_effect": "predicted_task_loss_decrease",
    }
    if parameter_tensor_count is not None:
        result["parameter_tensor_count"] = parameter_tensor_count
    return result


def _records(*, candidate_index: int = 0, replicate_id: str = "R0") -> list[dict]:
    candidate = FROZEN_CANDIDATES[candidate_index]
    before = _counts(adapted=False)
    after = _counts(adapted=True)
    per_group = {
        group: _alignment_metrics(
            scalar_count=1,
            parameter_tensor_count=(87 if index == 0 else 1),
        )
        for index, group in enumerate(FINE_ALIGNMENT_GROUP_IDS)
    }
    analysis = {
        "schema_version": 3,
        "artifact_type": "cr_sitta_p3_stage_a_outer_episode",
        "noop": {
            "logit_delta_abs_max": 0.1,
            "binary_pixel_xor_count": 1,
            "metric_counts_identical": False,
            "entropy_pre": 0.6,
            "entropy_post": 0.5,
            "metric_counts": {
                "identical": False,
                "pre": before,
                "post": after,
            },
        },
        "entropy_task_alignment": {
            "global": _alignment_metrics(scalar_count=20),
            "per_group": per_group,
        },
        "scientific_selection_performed": False,
        "stage2_authorized": False,
    }
    return [
        {
            "schema_version": 3,
            "artifact_type": EPISODE_ARTIFACT_TYPE,
            "dataset": dataset,
            "condition": condition,
            "replicate_id": replicate_id,
            "image_index": index,
            "image_id": f"{dataset}-pilot-{index}",
            "candidate": {
                "candidate_id": candidate.candidate_id,
                "optimizer": candidate.optimizer,
                "learning_rate": candidate.learning_rate,
            },
            "finite_gradient": True,
            "changed_parameter_tensor_count": 106,
            "analysis": analysis,
        }
        for dataset in DATASETS
        for condition in CONDITIONS
        for index in range(64)
    ]


def _candidate_value(index: int) -> dict:
    candidate = FROZEN_CANDIDATES[index]
    return {
        "candidate_id": candidate.candidate_id,
        "optimizer": candidate.optimizer,
        "learning_rate": candidate.learning_rate,
    }


def _r0_receipt(*, eligible: tuple[int, ...]) -> dict:
    eligible_set = set(eligible)
    return {
        "schema_version": 3,
        "receipt_type": "cr_sitta_d0_v3_stage_a_science_decision",
        "evaluation_phase": "R0",
        "protocol_status": "passed",
        "formal_stage_a_protocol_complete": not bool(eligible),
        "scientific_status": (
            "scientific_pending_R1_R2" if eligible else "scientific_no_eligible"
        ),
        "stage2_allowed": False,
        "eligible_candidates": [_candidate_value(index) for index in eligible],
        "rejected_candidates": [
            {
                "candidate": _candidate_value(index),
                "failed_gates": ["synthetic_rejection"],
            }
            for index in range(len(FROZEN_CANDIDATES))
            if index not in eligible_set
        ],
        "ranking": [],
        "required_followup_replicates": ["R1", "R2"] if eligible else [],
    }


def test_replicate_aggregate_builds_strict_gate_evidence() -> None:
    evidence = build_candidate_replicate_evidence(
        _records(), candidate=FROZEN_CANDIDATES[0], replicate_id="R0"
    )
    parsed = parse_replicate_evidence(evidence)
    assert parsed.episode_count == 3 * 13 * 64
    assert parsed.activity.fine_group_alignment_observation_count == 20 * 3 * 13 * 64
    assert parsed.activity.both_gradients_nonzero_episodes == 3 * 13 * 64
    assert not evaluate_replicate_hard_gates(parsed)


def test_replicate_aggregate_rejects_duplicate_or_incomplete_grid() -> None:
    records = _records()
    tampered = copy.deepcopy(records)
    tampered[-1]["image_index"] = 62
    with pytest.raises(D0V3ReplicateAggregateError, match="duplicate episode key"):
        build_candidate_replicate_evidence(
            tampered, candidate=FROZEN_CANDIDATES[0], replicate_id="R0"
        )


def test_metric_identical_is_recomputed_from_pre_and_post_counts() -> None:
    records = _records()
    for record in records:
        metric_counts = record["analysis"]["noop"]["metric_counts"]
        metric_counts["post"] = copy.deepcopy(metric_counts["pre"])
    with pytest.raises(
        D0V3ReplicateAggregateError, match="differs from recomputed"
    ):
        build_candidate_replicate_evidence(
            records, candidate=FROZEN_CANDIDATES[0], replicate_id="R0"
        )


def test_fine_group_observations_must_be_complete_not_empty_placeholders() -> None:
    records = _records()
    records[0]["analysis"]["entropy_task_alignment"]["per_group"][
        FINE_ALIGNMENT_GROUP_IDS[0]
    ] = {}
    with pytest.raises(D0V3ReplicateAggregateError, match="fields must be exact"):
        build_candidate_replicate_evidence(
            records, candidate=FROZEN_CANDIDATES[0], replicate_id="R0"
        )


def test_r1_subset_is_defined_only_by_canonical_r0_eligibility_receipt() -> None:
    receipt = _r0_receipt(eligible=(0,))
    result = build_replicate_evidence_set(
        _records(replicate_id="R1"),
        replicate_id="R1",
        r0_eligibility_receipt=receipt,
    )
    assert len(result) == 1
    assert result[0]["candidate"] == _candidate_value(0)

    with pytest.raises(D0V3ReplicateAggregateError, match="requires"):
        build_replicate_evidence_set(
            _records(replicate_id="R1"), replicate_id="R1"
        )


def test_r1_subset_rejects_receipt_required_missing_and_extra_candidates() -> None:
    with pytest.raises(D0V3ReplicateAggregateError, match="missing"):
        build_replicate_evidence_set(
            _records(candidate_index=0, replicate_id="R1"),
            replicate_id="R1",
            r0_eligibility_receipt=_r0_receipt(eligible=(0, 1)),
        )

    records = _records(candidate_index=0, replicate_id="R2")
    records.append(_records(candidate_index=1, replicate_id="R2")[0])
    with pytest.raises(D0V3ReplicateAggregateError, match="outside"):
        build_replicate_evidence_set(
            records,
            replicate_id="R2",
            r0_eligibility_receipt=_r0_receipt(eligible=(0,)),
        )


def test_empty_r0_eligibility_receipt_forbids_r1_r2_execution() -> None:
    with pytest.raises(D0V3ReplicateAggregateError, match="forbidden"):
        build_replicate_evidence_set(
            [],
            replicate_id="R1",
            r0_eligibility_receipt=_r0_receipt(eligible=()),
        )
