from __future__ import annotations

import copy
from decimal import Decimal
from itertools import permutations

import pytest

from analysis.d0_v3_formal_contract import FROZEN_CANDIDATES, FormalCandidate
from analysis.d0_v3_science_gate import (
    ALIGNMENT_OBSERVATIONS_PER_CANDIDATE_REPLICATE,
    CORRUPTION_FAMILIES,
    CandidateRankingSummary,
    DATASETS,
    D0V3ProtocolGateError,
    D0V3ScienceGateError,
    EPISODES_PER_CANDIDATE_REPLICATE,
    SAFETY_STRATA,
    evaluate_replicate_hard_gates,
    evaluate_stage_a_final,
    evaluate_stage_a_r0,
    parse_r0_eligibility_receipt,
    parse_replicate_evidence,
    rank_eligible_candidates,
)


def _evidence(
    candidate_index: int,
    replicate_id: str,
    *,
    nonclean_iou: float = 0.003,
    overall_iou: float = 0.002,
) -> dict:
    candidate = FROZEN_CANDIDATES[candidate_index]
    return {
        "schema_version": 3,
        "artifact_type": "cr_sitta_d0_v3_stage_a_replicate_evidence",
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "optimizer": candidate.optimizer,
            "learning_rate": candidate.learning_rate,
        },
        "replicate_id": replicate_id,
        "episode_count": EPISODES_PER_CANDIDATE_REPLICATE,
        "metrics": {
            "nonclean_macro_delta_iou": nonclean_iou,
            "overall_macro_delta_iou": overall_iou,
            "dataset_nonclean_delta_iou": {
                "IRSTD-1K": 0.003,
                "NUAA-SIRST": 0.002,
                "NUDT-SIRST": -0.001,
            },
            "family_delta_iou": {
                "gaussian_noise": 0.003,
                "gaussian_blur": 0.002,
                "low_contrast": 0.001,
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


def _r0_records(*, passing: set[int]) -> list[dict]:
    return [
        _evidence(index, "R0", nonclean_iou=0.003 if index in passing else 0.0)
        for index in range(10)
    ]


def _final_records(*, passing: set[int]) -> list[dict]:
    records = _r0_records(passing=passing)
    for index in sorted(passing):
        records.extend((_evidence(index, "R1"), _evidence(index, "R2")))
    return records


def test_strict_mapping_parser_accepts_complete_record_and_rejects_unknown() -> None:
    value = _evidence(0, "R0")
    evidence = parse_replicate_evidence(value)
    assert evidence.candidate == FROZEN_CANDIDATES[0]
    assert evidence.replicate_id == "R0"
    assert evidence.episode_count == 2496
    assert evidence.activity.both_gradients_nonzero_episodes == 1997
    assert evidence.activity.fine_group_alignment_observation_count == 49920
    assert evaluate_replicate_hard_gates(evidence) == ()

    unknown = copy.deepcopy(value)
    unknown["activity"]["unsafe"] = 1
    with pytest.raises(D0V3ScienceGateError, match="unknown"):
        parse_replicate_evidence(unknown)


def test_protocol_failure_is_exception_not_scientific_negative() -> None:
    with pytest.raises(D0V3ProtocolGateError, match="not a scientific result"):
        evaluate_stage_a_r0(_r0_records(passing=set()), protocol_status="failed")


def test_r0_no_eligible_is_complete_normal_negative_and_stops() -> None:
    decision = evaluate_stage_a_r0(
        _r0_records(passing=set()), protocol_status="passed"
    )
    assert decision.protocol_status == "passed"
    assert decision.formal_stage_a_protocol_complete is True
    assert decision.scientific_status == "scientific_no_eligible"
    assert decision.stage2_allowed is False
    assert decision.eligible_candidates == ()
    assert decision.required_followup_replicates == ()
    assert len(decision.rejected_candidates) == 10
    assert all(
        "nonclean_macro_iou_nonpositive" in item.failed_gates
        for item in decision.rejected_candidates
    )

    final_decision = evaluate_stage_a_final(
        _r0_records(passing=set()), protocol_status="passed"
    )
    assert final_decision == decision
    assert final_decision.evaluation_phase == "R0"


def test_r0_eligibility_receipt_parser_enforces_exact_candidate_partition() -> None:
    decision = evaluate_stage_a_r0(
        _r0_records(passing={0, 3}), protocol_status="passed"
    )
    receipt = decision.to_receipt()
    assert parse_r0_eligibility_receipt(receipt) == (
        FROZEN_CANDIDATES[0],
        FROZEN_CANDIDATES[3],
    )

    reordered = copy.deepcopy(receipt)
    reordered["eligible_candidates"].reverse()
    with pytest.raises(D0V3ProtocolGateError, match="partition/order"):
        parse_r0_eligibility_receipt(reordered)

    contradictory = copy.deepcopy(receipt)
    contradictory["formal_stage_a_protocol_complete"] = True
    with pytest.raises(D0V3ProtocolGateError, match="must require R1/R2"):
        parse_r0_eligibility_receipt(contradictory)


def test_r0_runs_r1_r2_for_all_and_only_eligible_candidates() -> None:
    decision = evaluate_stage_a_r0(
        _r0_records(passing={0, 3}), protocol_status="passed"
    )
    assert decision.formal_stage_a_protocol_complete is False
    assert decision.scientific_status == "scientific_pending_R1_R2"
    assert decision.eligible_candidates == (
        FROZEN_CANDIDATES[0],
        FROZEN_CANDIDATES[3],
    )
    assert decision.required_followup_replicates == ("R1", "R2")
    assert decision.stage2_allowed is False


def test_final_requires_exact_eligible_only_replicate_topology() -> None:
    missing = _final_records(passing={0})
    missing.pop()
    with pytest.raises(D0V3ProtocolGateError, match="eligible-only"):
        evaluate_stage_a_final(missing, protocol_status="passed")

    extra = _final_records(passing={0})
    extra.extend((_evidence(1, "R1"), _evidence(1, "R2")))
    with pytest.raises(D0V3ProtocolGateError, match="eligible-only"):
        evaluate_stage_a_final(extra, protocol_status="passed")


def test_three_replicate_pass_is_formal_complete_but_never_stage2_authority() -> None:
    records = _final_records(passing={0, 1})
    records[-2]["metrics"]["nonclean_macro_delta_iou"] = 0.004
    records[-1]["metrics"]["nonclean_macro_delta_iou"] = 0.004
    decision = evaluate_stage_a_final(records, protocol_status="passed")

    assert decision.formal_stage_a_protocol_complete is True
    assert decision.scientific_status == "scientific_passed"
    assert decision.stage2_allowed is False
    assert set(decision.eligible_candidates) == {
        FROZEN_CANDIDATES[0],
        FROZEN_CANDIDATES[1],
    }
    assert decision.ranking[0].candidate == FROZEN_CANDIDATES[1]
    receipt = decision.to_receipt()
    assert receipt["protocol_status"] == "passed"
    assert receipt["scientific_status"] == "scientific_passed"
    assert receipt["stage2_allowed"] is False


def test_each_replicate_hard_gate_and_three_replicate_mean_are_both_required() -> None:
    records = _final_records(passing={0})
    records[-1]["metrics"]["clean_macro_delta_iou"] = -0.003
    decision = evaluate_stage_a_final(records, protocol_status="passed")
    assert decision.scientific_status == "scientific_no_eligible"
    assert "R2:clean_macro_iou_safety" in decision.rejected_candidates[0].failed_gates

    mean_failure = _final_records(passing={0})
    for record in mean_failure:
        if record["candidate"]["candidate_id"] == FROZEN_CANDIDATES[0].candidate_id:
            record["metrics"]["nonclean_macro_delta_iou"] = 0.0015
    decision = evaluate_stage_a_final(mean_failure, protocol_status="passed")
    assert decision.scientific_status == "scientific_no_eligible"
    rejected = next(
        item for item in decision.rejected_candidates if item.candidate == FROZEN_CANDIDATES[0]
    )
    assert "R0_R1_R2:nonclean_macro_iou_mean" in rejected.failed_gates


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (
            lambda value: value["metrics"]["dataset_nonclean_delta_iou"].update(
                {dataset: -0.003 for dataset in DATASETS}
            ),
            "positive_dataset_coverage",
        ),
        (
            lambda value: value["metrics"]["family_delta_iou"].update(
                {family: -0.006 for family in CORRUPTION_FAMILIES}
            ),
            "positive_family_coverage",
        ),
        (
            lambda value: value["metrics"].update(
                {"nonclean_macro_delta_pd": -0.011}
            ),
            "nonclean_pd_safety",
        ),
        (
            lambda value: value["safety"]["fa_delta_per_million"].update(
                {"nonclean": 15.0000001}
            ),
            "fa_inflation",
        ),
        (
            lambda value: value["safety"]["adapted_foreground_fraction"].update(
                {"nonclean": 0.012002}
            ),
            "foreground_inflation",
        ),
        (
            lambda value: value["activity"].update(
                {"parameter_changed_episodes": 2371}
            ),
            "parameter_changed_fraction",
        ),
        (
            lambda value: value["activity"].update(
                {"functional_logit_changed_episodes": 249}
            ),
            "functional_logit_changed_fraction",
        ),
        (
            lambda value: value["activity"].update(
                {"both_gradients_nonzero_episodes": 1996}
            ),
            "alignment_nonzero_gradient_fraction",
        ),
        (
            lambda value: value["alignment"].update({"macro_cosine": 0.049}),
            "alignment_macro_cosine",
        ),
    ],
)
def test_representative_preregistered_hard_gate_boundaries(mutator, reason: str) -> None:
    value = _evidence(0, "R0")
    mutator(value)
    failures = evaluate_replicate_hard_gates(parse_replicate_evidence(value))
    assert reason in failures


def _summary(
    candidate: FormalCandidate,
    *,
    nonclean: str = "0.003",
    worst: str = "0.003",
    dataset: str = "0.002",
    clean: str = "0",
    fa: str = "0",
    foreground: str = "0.0005",
    pd: str = "0",
) -> CandidateRankingSummary:
    return CandidateRankingSummary(
        candidate=candidate,
        mean_nonclean_delta_iou=Decimal(nonclean),
        worst_replicate_nonclean_delta_iou=Decimal(worst),
        minimum_dataset_mean_nonclean_delta_iou=Decimal(dataset),
        mean_clean_delta_iou=Decimal(clean),
        mean_nonclean_fa_delta=Decimal(fa),
        mean_nonclean_foreground_inflation=Decimal(foreground),
        mean_nonclean_delta_pd=Decimal(pd),
    )


def test_ranking_is_exact_then_uses_lower_lr_and_adam_before_sgd() -> None:
    lower_lr = _summary(FROZEN_CANDIDATES[0], nonclean="0.0030000000000")
    higher_lr = _summary(FROZEN_CANDIDATES[1], nonclean="0.0030000000005")
    assert rank_eligible_candidates((higher_lr, lower_lr))[0] == higher_lr

    exact_primary_tie_lower_lr = _summary(
        FROZEN_CANDIDATES[0], nonclean="0.0030000000005"
    )
    assert (
        rank_eligible_candidates((higher_lr, exact_primary_tie_lower_lr))[0]
        == exact_primary_tie_lower_lr
    )

    adam = _summary(FROZEN_CANDIDATES[0])
    sgd = _summary(FROZEN_CANDIDATES[5])
    assert rank_eligible_candidates((sgd, adam))[0] == adam


def test_ranking_is_transitive_and_input_permutation_independent() -> None:
    values = (
        _summary(FROZEN_CANDIDATES[0], nonclean="0"),
        _summary(FROZEN_CANDIDATES[1], nonclean="0.00000000000075"),
        _summary(FROZEN_CANDIDATES[2], nonclean="0.0000000000015"),
    )
    rankings = {
        tuple(item.candidate for item in rank_eligible_candidates(order))
        for order in permutations(values)
    }
    assert rankings == {
        (
            FROZEN_CANDIDATES[2],
            FROZEN_CANDIDATES[1],
            FROZEN_CANDIDATES[0],
        )
    }


def test_comparison_tolerance_applies_to_gate_boundaries_not_ranking() -> None:
    within_strict_zero = _evidence(
        0, "R0", nonclean_iou=5.0e-13, overall_iou=5.0e-13
    )
    failures = evaluate_replicate_hard_gates(
        parse_replicate_evidence(within_strict_zero)
    )
    assert "nonclean_macro_iou_nonpositive" in failures
    assert "overall_macro_iou_nonpositive" in failures

    beyond_strict_zero = _evidence(
        0, "R0", nonclean_iou=2.0e-12, overall_iou=2.0e-12
    )
    failures = evaluate_replicate_hard_gates(
        parse_replicate_evidence(beyond_strict_zero)
    )
    assert "nonclean_macro_iou_nonpositive" not in failures
    assert "overall_macro_iou_nonpositive" not in failures

    within_inclusive_minimum = _evidence(0, "R0")
    within_inclusive_minimum["metrics"]["clean_macro_delta_iou"] = -0.0020000000005
    failures = evaluate_replicate_hard_gates(
        parse_replicate_evidence(within_inclusive_minimum)
    )
    assert "clean_macro_iou_safety" not in failures

    beyond_inclusive_minimum = _evidence(0, "R0")
    beyond_inclusive_minimum["metrics"]["clean_macro_delta_iou"] = -0.002000000002
    failures = evaluate_replicate_hard_gates(
        parse_replicate_evidence(beyond_inclusive_minimum)
    )
    assert "clean_macro_iou_safety" in failures


def test_ranking_primary_metric_dominates_later_tie_breaks() -> None:
    primary_better = _summary(
        FROZEN_CANDIDATES[4], nonclean="0.004", fa="10", pd="-0.005"
    )
    safe_but_lower_primary = _summary(
        FROZEN_CANDIDATES[0], nonclean="0.003", fa="-10", pd="0.01"
    )
    ranking = rank_eligible_candidates((safe_but_lower_primary, primary_better))
    assert ranking[0] == primary_better
