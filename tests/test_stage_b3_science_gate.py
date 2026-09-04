from __future__ import annotations

import copy
from fractions import Fraction
import json
from pathlib import Path

import pytest
import yaml

from analysis.proxy_objective_space_screen_v1 import (
    CELLS_PER_CANDIDATE,
    CONDITIONS,
    DATASETS,
    EPISODES_PER_CANDIDATE,
    GATE_CONFIG_FIELDS,
    StageB3ScienceGateError,
    evaluate_candidates,
    evaluate_stage_b3_science_gate,
)


CANDIDATES = tuple(f"screen_candidate_{index}" for index in range(6))


def _gate(**overrides) -> dict:
    value = {
        "comparison_tolerance": "0.000000000001",
        "maximum_selected_candidates": 4,
        "normalized_task_directional_derivative_maximum": 0,
        "minimum_negative_derivative_datasets": 2,
        "minimum_negative_derivative_families": 3,
        "minimum_negative_derivative_severities": 2,
        "nonclean_macro_delta_iou_strictly_greater_than": 0,
        "clean_macro_delta_iou_minimum": -0.002,
        "foreground_fraction_delta_maximum": 0.001,
        "foreground_fraction_source_multiplier": 1.2,
        "foreground_fraction_epsilon": 0.000001,
        "minimum_functional_changed_episodes": 1,
        "minimum_threshold_crossing_count": 1,
        "minimum_positive_delta_iou_severities": 2,
        "filter_before_ranking": True,
        "no_eligible_is_normal_scientific_result": True,
        "negative_controls_selectable": False,
    }
    value.update(overrides)
    return value


def _parts(condition: str) -> tuple[str, str]:
    if condition == "clean_S0":
        return "clean", "S0"
    return condition.rsplit("_S", 1)[0], f"S{condition.rsplit('_S', 1)[1]}"


def _grid(
    candidate_ids=CANDIDATES,
    *,
    gains: dict[str, float] | None = None,
    derivative: float = -0.1,
    clean_gain: float = 0.0,
    foreground_gain: float = 0.0005,
    functional_count: int = 1,
    crossing_count: int = 1,
) -> list[dict]:
    gains = gains or {candidate: 0.003 for candidate in candidate_ids}
    records = []
    for candidate_id in candidate_ids:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                family, severity = _parts(condition)
                gain = clean_gain if condition == "clean_S0" else gains[candidate_id]
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "dataset": dataset,
                        "condition": condition,
                        "corruption_family": family,
                        "severity": severity,
                        "normalized_task_directional_derivative": derivative,
                        "source_iou": 0.5,
                        "adapted_iou": 0.5 + gain,
                        "delta_pd": (
                            0.001
                            if condition != "clean_S0" and functional_count > 0
                            else 0.0
                        ),
                        "delta_fa": (
                            -1.0
                            if condition != "clean_S0" and functional_count > 0
                            else 0.0
                        ),
                        "source_foreground_fraction": 0.01,
                        "adapted_foreground_fraction": 0.01 + foreground_gain,
                        "valid_alignment_episode_count": 16,
                        "episode_count": 16,
                        "functional_changed_episode_count": functional_count,
                        "threshold_crossing_count": crossing_count,
                    }
                )
    return records


def _evaluation(decision, candidate_id: str):
    return next(
        item
        for item in decision.candidate_evaluations
        if item.candidate_id == candidate_id
    )


def test_complete_grid_passes_and_receipt_is_json_safe_development_evidence() -> None:
    records = _grid((CANDIDATES[0],))
    before = copy.deepcopy(records)
    decision = evaluate_stage_b3_science_gate(
        records, (CANDIDATES[0],), _gate()
    )

    assert records == before
    assert decision.scientific_status == "scientific_passed"
    assert decision.stage_b4_allowed is True
    assert decision.selected_candidate_ids == (CANDIDATES[0],)
    aggregate = decision.candidate_evaluations[0].aggregate
    assert aggregate.cell_count == CELLS_PER_CANDIDATE == 39
    assert aggregate.nonclean_cell_count == 36
    assert aggregate.episode_count == EPISODES_PER_CANDIDATE == 624
    assert aggregate.valid_alignment_episode_count == 624
    assert aggregate.functional_changed_episode_count == 39
    assert aggregate.threshold_crossing_count == 39
    assert aggregate.nonclean_functional_changed_episode_count == 36
    assert aggregate.nonclean_threshold_crossing_count == 36
    assert aggregate.nonclean_macro_directional_derivative == Fraction(-1, 10)
    assert aggregate.nonclean_macro_delta_iou == Fraction(3, 1000)
    assert aggregate.nonclean_macro_delta_pd == Fraction(1, 1000)
    assert aggregate.nonclean_macro_delta_fa == Fraction(-1, 1)
    assert dict(aggregate.severity_nonclean_delta_iou) == {
        "S1": Fraction(3, 1000),
        "S3": Fraction(3, 1000),
        "S5": Fraction(3, 1000),
    }

    receipt = decision.to_receipt()
    assert receipt["protocol_status"] == "passed"
    assert receipt["paper_result"] is False
    assert receipt["test"] is False
    assert receipt["stage_b4_allowed"] is True
    assert receipt["aggregation_policy"]["missing_value_policy"] == (
        "forbidden_no_imputation"
    )
    assert "not_eligible" in receipt["aggregation_policy"][
        "zero_valid_alignment_policy"
    ]
    assert receipt["aggregation_policy"][
        "nonclean_alignment_completeness_required"
    ] is True
    assert receipt["aggregation_policy"][
        "incomplete_nonclean_alignment_action"
    ] == "candidate_science_ineligible_other_candidates_continue"
    assert receipt["selection_policy"]["filter_before_ranking"] is True
    assert receipt["gate"]["comparison_tolerance"] == {
        "numerator": 1,
        "denominator": 1_000_000_000_000,
    }
    assert set(receipt["gate"]) == GATE_CONFIG_FIELDS
    json.dumps(receipt, allow_nan=False)


def test_filter_before_ranking_excludes_unsafe_high_gain_and_caps_at_four() -> None:
    gains = {
        candidate: gain
        for candidate, gain in zip(CANDIDATES, (0.004, 0.003, 0.006, 0.002, 0.005, 0.1))
    }
    records = _grid(CANDIDATES, gains=gains)
    for record in records:
        if record["candidate_id"] == CANDIDATES[-1]:
            record["adapted_foreground_fraction"] = 0.5

    decision = evaluate_candidates(records, CANDIDATES, _gate())
    unsafe = _evaluation(decision, CANDIDATES[-1])
    assert unsafe.eligible is False
    assert "foreground_absolute_inflation:overall" in unsafe.reason_codes
    assert CANDIDATES[-1] not in decision.eligible_candidate_ids
    assert decision.selected_candidate_ids == (
        CANDIDATES[2],
        CANDIDATES[4],
        CANDIDATES[0],
        CANDIDATES[1],
    )
    assert len(decision.ranking) == 4


def test_zero_eligible_is_normal_scientific_negative_without_stage_b4() -> None:
    records = _grid(
        (CANDIDATES[0],),
        gains={CANDIDATES[0]: 0.0},
        derivative=0.0,
        foreground_gain=0.0,
        functional_count=0,
        crossing_count=0,
    )
    decision = evaluate_candidates(records, (CANDIDATES[0],), _gate())
    assert decision.scientific_status == "scientific_no_eligible"
    assert decision.stage_b4_allowed is False
    assert decision.eligible_candidate_ids == ()
    assert decision.ranking == ()
    assert decision.to_receipt()["selected_for_stage_b4"] == []
    reasons = decision.candidate_evaluations[0].reason_codes
    assert "macro_directional_derivative_not_negative" in reasons
    assert "nonclean_macro_delta_iou_not_positive" in reasons
    assert "functional_change_nonzero" in reasons
    assert "threshold_change_nonzero" in reasons


def test_dataset_and_family_negative_derivative_coverage_are_hard_gates() -> None:
    records = _grid((CANDIDATES[0],))
    for record in records:
        if record["condition"] != "clean_S0" and record["dataset"] != DATASETS[0]:
            record["normalized_task_directional_derivative"] = 0.02
    decision = evaluate_candidates(records, (CANDIDATES[0],), _gate())
    assert "negative_directional_derivative_dataset_coverage" in (
        decision.candidate_evaluations[0].reason_codes
    )

    records = _grid((CANDIDATES[0],))
    for record in records:
        if record["corruption_family"] in ("low_contrast", "stripe_noise"):
            record["normalized_task_directional_derivative"] = 0.02
    decision = evaluate_candidates(records, (CANDIDATES[0],), _gate())
    assert "negative_directional_derivative_family_coverage" in (
        decision.candidate_evaluations[0].reason_codes
    )


def test_gain_and_direction_evidence_must_each_span_two_severities() -> None:
    records = _grid((CANDIDATES[0],))
    for record in records:
        if record["severity"] in ("S3", "S5"):
            record["normalized_task_directional_derivative"] = 0.02
    decision = evaluate_candidates(records, (CANDIDATES[0],), _gate())
    evaluation = decision.candidate_evaluations[0]
    assert evaluation.aggregate.nonclean_macro_directional_derivative < 0
    assert "negative_directional_derivative_severity_coverage" in (
        evaluation.reason_codes
    )

    records = _grid((CANDIDATES[0],))
    for record in records:
        if record["severity"] == "S1":
            record["adapted_iou"] = 0.509
        elif record["severity"] in ("S3", "S5"):
            record["adapted_iou"] = 0.499
    decision = evaluate_candidates(records, (CANDIDATES[0],), _gate())
    evaluation = decision.candidate_evaluations[0]
    assert evaluation.aggregate.nonclean_macro_delta_iou > 0
    assert "positive_delta_iou_severity_coverage" in evaluation.reason_codes


def test_exact_two_dataset_three_family_and_two_severity_boundaries_pass() -> None:
    records = _grid((CANDIDATES[0],))
    for record in records:
        if record["dataset"] == DATASETS[-1]:
            record["normalized_task_directional_derivative"] = 0.01
        if record["corruption_family"] == "stripe_noise":
            record["normalized_task_directional_derivative"] = 0.01
        if record["severity"] == "S5":
            record["normalized_task_directional_derivative"] = 0.01
            record["adapted_iou"] = 0.499
    decision = evaluate_candidates(records, (CANDIDATES[0],), _gate())
    assert decision.scientific_status == "scientific_passed"
    assert decision.candidate_evaluations[0].reason_codes == ()


def test_clean_safety_is_an_independent_hard_gate() -> None:
    records = _grid((CANDIDATES[0],), clean_gain=-0.006)
    decision = evaluate_candidates(records, (CANDIDATES[0],), _gate())
    reasons = decision.candidate_evaluations[0].reason_codes
    assert "clean_macro_delta_iou_safety" in reasons
    assert "functional_change_nonzero" not in reasons


def test_clean_only_activity_does_not_satisfy_nonclean_activity_gates() -> None:
    records = _grid(
        (CANDIDATES[0],),
        gains={CANDIDATES[0]: 0.0},
        foreground_gain=0.0,
        functional_count=0,
        crossing_count=0,
    )
    for record in records:
        if record["condition"] == "clean_S0":
            record["functional_changed_episode_count"] = 1
            record["threshold_crossing_count"] = 1
    decision = evaluate_candidates(records, (CANDIDATES[0],), _gate())
    aggregate = decision.candidate_evaluations[0].aggregate
    assert aggregate.functional_changed_episode_count == 3
    assert aggregate.threshold_crossing_count == 3
    assert aggregate.nonclean_functional_changed_episode_count == 0
    assert aggregate.nonclean_threshold_crossing_count == 0
    assert "functional_change_nonzero" in decision.candidate_evaluations[0].reason_codes
    assert "threshold_change_nonzero" in decision.candidate_evaluations[0].reason_codes


def test_foreground_absolute_and_ratio_bounds_are_inclusive_with_tolerance() -> None:
    tolerance = 1e-12
    at_absolute_boundary = _grid(
        (CANDIDATES[0],), foreground_gain=0.001 + tolerance
    )
    decision = evaluate_candidates(
        at_absolute_boundary, (CANDIDATES[0],), _gate()
    )
    assert not any(
        reason.startswith("foreground_absolute_inflation")
        for reason in decision.candidate_evaluations[0].reason_codes
    )

    above_absolute_boundary = _grid(
        (CANDIDATES[0],), foreground_gain=0.001 + tolerance + 1e-13
    )
    decision = evaluate_candidates(
        above_absolute_boundary, (CANDIDATES[0],), _gate()
    )
    assert "foreground_absolute_inflation:overall" in (
        decision.candidate_evaluations[0].reason_codes
    )

    ratio_failure = _grid((CANDIDATES[0],), foreground_gain=0.002002)
    decision = evaluate_candidates(
        ratio_failure,
        (CANDIDATES[0],),
        _gate(foreground_fraction_delta_maximum=0.1),
    )
    assert "foreground_ratio_inflation:nonclean" in (
        decision.candidate_evaluations[0].reason_codes
    )

    ratio_at_boundary = _grid(
        (CANDIDATES[0],), foreground_gain=0.0
    )
    for record in ratio_at_boundary:
        record["adapted_foreground_fraction"] = "0.012001000001"
    decision = evaluate_candidates(
        ratio_at_boundary,
        (CANDIDATES[0],),
        _gate(foreground_fraction_delta_maximum=0.1),
    )
    assert not any(
        reason.startswith("foreground_ratio_inflation")
        for reason in decision.candidate_evaluations[0].reason_codes
    )

    ratio_above_boundary = copy.deepcopy(ratio_at_boundary)
    for record in ratio_above_boundary:
        record["adapted_foreground_fraction"] = "0.0120010000011"
    decision = evaluate_candidates(
        ratio_above_boundary,
        (CANDIDATES[0],),
        _gate(foreground_fraction_delta_maximum=0.1),
    )
    assert "foreground_ratio_inflation:overall" in (
        decision.candidate_evaluations[0].reason_codes
    )


def test_strict_direction_and_iou_boundaries_respect_comparison_tolerance() -> None:
    tolerance = 1e-12
    at_boundary = _grid(
        (CANDIDATES[0],),
        gains={CANDIDATES[0]: tolerance},
        derivative=-tolerance,
    )
    decision = evaluate_candidates(at_boundary, (CANDIDATES[0],), _gate())
    reasons = decision.candidate_evaluations[0].reason_codes
    assert "macro_directional_derivative_not_negative" in reasons
    assert "nonclean_macro_delta_iou_not_positive" in reasons

    beyond_boundary = _grid(
        (CANDIDATES[0],),
        gains={CANDIDATES[0]: tolerance + 1e-13},
        derivative=-(tolerance + 1e-13),
    )
    decision = evaluate_candidates(
        beyond_boundary, (CANDIDATES[0],), _gate()
    )
    assert decision.scientific_status == "scientific_passed"


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda rows: rows[0].update({"unknown": 1}), "fields must be exact"),
        (
            lambda rows: rows[0].update(
                {"normalized_task_directional_derivative": float("nan")}
            ),
            "must be finite",
        ),
        (
            lambda rows: rows[0].update({"corruption_family": "stripe_noise"}),
            "disagree with condition",
        ),
        (lambda rows: rows[0].update({"episode_count": 15}), "must equal 16"),
        (
            lambda rows: rows[0].update({"functional_changed_episode_count": 17}),
            "exceeds episode_count",
        ),
    ],
)
def test_cell_schema_finite_metadata_and_count_validation_fail_closed(
    mutator, match: str
) -> None:
    records = _grid((CANDIDATES[0],))
    mutator(records)
    with pytest.raises(StageB3ScienceGateError, match=match):
        evaluate_candidates(records, (CANDIDATES[0],), _gate())


def test_alignment_and_activity_claims_cannot_contradict_cell_evidence() -> None:
    records = _grid((CANDIDATES[0],))
    records[1]["valid_alignment_episode_count"] = 0
    with pytest.raises(StageB3ScienceGateError, match="zero valid alignment"):
        evaluate_candidates(records, (CANDIDATES[0],), _gate())

    records = _grid((CANDIDATES[0],))
    records[1]["functional_changed_episode_count"] = 0
    records[1]["threshold_crossing_count"] = 0
    with pytest.raises(StageB3ScienceGateError, match="endpoint/activity change"):
        evaluate_candidates(records, (CANDIDATES[0],), _gate())

    records = _grid((CANDIDATES[0],))
    records[1]["threshold_crossing_count"] = 0
    with pytest.raises(StageB3ScienceGateError, match="zero threshold crossings"):
        evaluate_candidates(records, (CANDIDATES[0],), _gate())


def test_incomplete_nonclean_alignment_rejects_only_affected_candidate() -> None:
    candidate_ids = (CANDIDATES[0], CANDIDATES[1])
    records = _grid(candidate_ids)
    incomplete = next(
        record
        for record in records
        if record["candidate_id"] == CANDIDATES[0]
        and record["condition"] != "clean_S0"
    )
    incomplete["valid_alignment_episode_count"] = 0
    incomplete["normalized_task_directional_derivative"] = 0.0

    decision = evaluate_candidates(records, candidate_ids, _gate())
    rejected = _evaluation(decision, CANDIDATES[0])
    accepted = _evaluation(decision, CANDIDATES[1])
    assert rejected.eligible is False
    assert rejected.reason_codes == ("nonclean_alignment_evidence_incomplete",)
    assert (
        rejected.aggregate.nonclean_valid_alignment_episode_count
        == rejected.aggregate.nonclean_episode_count - 16
    )
    assert accepted.eligible is True
    assert accepted.reason_codes == ()
    assert decision.selected_candidate_ids == (CANDIDATES[1],)
    assert decision.stage_b4_allowed is True


def test_clean_alignment_count_does_not_enter_nonclean_completeness_gate() -> None:
    records = _grid((CANDIDATES[0],))
    clean = next(record for record in records if record["condition"] == "clean_S0")
    clean["valid_alignment_episode_count"] = 0
    clean["normalized_task_directional_derivative"] = 0.0
    decision = evaluate_candidates(records, (CANDIDATES[0],), _gate())
    assert decision.scientific_status == "scientific_passed"
    assert "nonclean_alignment_evidence_incomplete" not in (
        decision.candidate_evaluations[0].reason_codes
    )


def test_exact_topology_duplicate_roster_and_source_endpoint_fail_closed() -> None:
    with pytest.raises(StageB3ScienceGateError, match=r"candidate_count\*39"):
        evaluate_candidates(
            _grid((CANDIDATES[0],))[:-1], (CANDIDATES[0],), _gate()
        )

    duplicate = _grid((CANDIDATES[0],))
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(StageB3ScienceGateError, match="duplicate"):
        evaluate_candidates(duplicate, (CANDIDATES[0],), _gate())

    with pytest.raises(StageB3ScienceGateError, match="unique"):
        evaluate_candidates(
            _grid((CANDIDATES[0], CANDIDATES[1])),
            (CANDIDATES[0], CANDIDATES[0]),
            _gate(),
        )

    mismatched_source = _grid((CANDIDATES[0], CANDIDATES[1]))
    mismatched_source[CELLS_PER_CANDIDATE]["source_iou"] = 0.49
    with pytest.raises(StageB3ScienceGateError, match="Source endpoint differs"):
        evaluate_candidates(
            mismatched_source,
            (CANDIDATES[0], CANDIDATES[1]),
            _gate(),
        )


def test_config_mapping_is_exact_and_cannot_relax_v5_minimums() -> None:
    assert set(_gate()) == GATE_CONFIG_FIELDS
    incomplete = _gate()
    incomplete.pop("comparison_tolerance")
    with pytest.raises(StageB3ScienceGateError, match="fields must be exact"):
        evaluate_candidates(
            _grid((CANDIDATES[0],)), (CANDIDATES[0],), incomplete
        )

    with pytest.raises(StageB3ScienceGateError, match="must be <= 0"):
        evaluate_candidates(
            _grid((CANDIDATES[0],)),
            (CANDIDATES[0],),
            _gate(normalized_task_directional_derivative_maximum=0.1),
        )
    with pytest.raises(StageB3ScienceGateError, match=r"must be in \[2, 3\]"):
        evaluate_candidates(
            _grid((CANDIDATES[0],)),
            (CANDIDATES[0],),
            _gate(minimum_negative_derivative_datasets=1),
        )
    with pytest.raises(StageB3ScienceGateError, match=r"must be in \[1, 4\]"):
        evaluate_candidates(
            _grid((CANDIDATES[0],)),
            (CANDIDATES[0],),
            _gate(maximum_selected_candidates=5),
        )


def test_gate_mapping_directly_matches_frozen_stage_b3_yaml() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repository / "configs/p3_stage_b_objective_space_screen_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    frozen_gate = config["stage_b3_gate"]
    assert set(frozen_gate) == GATE_CONFIG_FIELDS
    decision = evaluate_candidates(
        _grid(("O1_P2",)), ("O1_P2",), frozen_gate
    )
    assert decision.scientific_status == "scientific_passed"


def test_o0_negative_control_is_evaluated_but_never_selected() -> None:
    decision = evaluate_candidates(
        _grid(("O0_P2",)), ("O0_P2",), _gate()
    )
    evaluation = decision.candidate_evaluations[0]
    assert evaluation.eligible is False
    assert evaluation.reason_codes == ("negative_control_not_selectable",)
    assert decision.scientific_status == "scientific_no_eligible"
    assert decision.stage_b4_allowed is False


def test_record_order_does_not_affect_deterministic_decision() -> None:
    candidate_ids = (CANDIDATES[0], CANDIDATES[1])
    records = _grid(
        candidate_ids,
        gains={CANDIDATES[0]: 0.003, CANDIDATES[1]: 0.004},
    )
    forward = evaluate_candidates(records, candidate_ids, _gate()).to_receipt()
    reversed_records = evaluate_candidates(
        list(reversed(records)), candidate_ids, _gate()
    ).to_receipt()
    assert forward == reversed_records
    assert forward["selected_for_stage_b4"] == [CANDIDATES[1], CANDIDATES[0]]


def test_exact_ranking_ties_use_candidate_id_lexical_order() -> None:
    candidate_ids = ("z_candidate", "a_candidate")
    records = _grid(candidate_ids)
    decision = evaluate_candidates(records, candidate_ids, _gate())
    assert decision.selected_candidate_ids == ("a_candidate", "z_candidate")


def test_ranking_activity_ties_use_only_nonclean_counts_and_receipt_names() -> None:
    candidate_ids = ("z_candidate", "a_candidate")
    clean_only_difference = _grid(candidate_ids)
    for record in clean_only_difference:
        if (
            record["candidate_id"] == "z_candidate"
            and record["condition"] == "clean_S0"
        ):
            record["functional_changed_episode_count"] = 16
            record["threshold_crossing_count"] = 16
    decision = evaluate_candidates(clean_only_difference, candidate_ids, _gate())
    assert decision.selected_candidate_ids == ("a_candidate", "z_candidate")

    nonclean_difference = copy.deepcopy(clean_only_difference)
    for record in nonclean_difference:
        if (
            record["candidate_id"] == "z_candidate"
            and record["condition"] != "clean_S0"
        ):
            record["functional_changed_episode_count"] = 2
            record["threshold_crossing_count"] = 2
    decision = evaluate_candidates(nonclean_difference, candidate_ids, _gate())
    assert decision.selected_candidate_ids == ("z_candidate", "a_candidate")
    first = decision.to_receipt()["ranking"][0]
    assert first["nonclean_functional_changed_episode_count"] == 72
    assert first["nonclean_threshold_crossing_count"] == 72
    assert "functional_changed_episode_count" not in first
    assert "threshold_crossing_count" not in first
