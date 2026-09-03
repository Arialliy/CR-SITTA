from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from analysis.nonadaptive_teacher_gate import (
    CELLS_PER_CANDIDATE,
    CONDITIONS,
    DATASETS,
    EXPECTED_CANDIDATE_COUNT,
    GateConfig,
    IMAGES_PER_CELL,
    NonadaptiveTeacherGateError,
    PIXELS_PER_CELL,
    evaluate_candidates,
    evaluate_nonadaptive_teacher_gate,
)


CANDIDATES = tuple(f"teacher_{index:02d}" for index in range(EXPECTED_CANDIDATE_COUNT))


def _gate(**overrides) -> GateConfig:
    values = {
        "comparison_tolerance": Fraction(1, 10**12),
        "nonclean_macro_delta_iou_epsilon": Fraction(1, 1000),
        "overall_macro_delta_iou_threshold": Fraction(0, 1),
        "minimum_positive_nonclean_datasets": 2,
        "worst_nonclean_dataset_delta_iou_minimum": Fraction(-2, 1000),
        "clean_macro_delta_iou_minimum": Fraction(-2, 1000),
        "each_clean_dataset_delta_iou_minimum": Fraction(-5, 1000),
        "nonclean_macro_delta_pd_minimum": Fraction(-1, 100),
        "each_nonclean_dataset_delta_pd_minimum": Fraction(-2, 100),
        "fa_absolute_allowance_per_million": Fraction(10, 1),
        "fa_source_multiplier": Fraction(1, 4),
        "foreground_fraction_delta_maximum": Fraction(1, 1000),
        "foreground_fraction_source_multiplier": Fraction(6, 5),
        "foreground_fraction_epsilon": Fraction(1, 1_000_000),
        # Metrics-only records remain useful for unit-level parser/gate tests,
        # but the formal default is True and is tested separately below.
        "require_integer_counts": False,
    }
    values.update(overrides)
    return GateConfig(**values)


def _metrics(
    *,
    iou: Fraction = Fraction(1, 2),
    pd: Fraction = Fraction(4, 5),
    fa: Fraction = Fraction(20, 1),
    fg: Fraction = Fraction(1, 100),
) -> dict:
    return {
        "global_iou": iou,
        "pd": pd,
        "fa_per_million": fa,
        "foreground_fraction": fg,
    }


def _metric_grid(
    candidate_ids=CANDIDATES,
    *,
    nonclean_iou_gains=None,
    clean_iou_gain=Fraction(0, 1),
    pd_gain=Fraction(0, 1),
    fa_gain=Fraction(0, 1),
    fg_gain=Fraction(0, 1),
) -> list[dict]:
    if nonclean_iou_gains is None:
        nonclean_iou_gains = {
            dataset: Fraction(3, 1000) for dataset in DATASETS
        }
    records = []
    for candidate_id in candidate_ids:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                source = _metrics()
                iou_gain = (
                    clean_iou_gain
                    if condition == "clean_S0"
                    else nonclean_iou_gains[dataset]
                )
                teacher = _metrics(
                    iou=source["global_iou"] + iou_gain,
                    pd=source["pd"] + pd_gain,
                    fa=source["fa_per_million"] + fa_gain,
                    fg=source["foreground_fraction"] + fg_gain,
                )
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "dataset": dataset,
                        "condition": condition,
                        "source": source,
                        "teacher": teacher,
                    }
                )
    return records


def _counts(*, intersection=50, fp=30, fn=50, tn=None, detected=8) -> dict:
    occupied = intersection + fp + fn
    if tn is None:
        tn = PIXELS_PER_CELL - occupied
    return {
        "image_count": IMAGES_PER_CELL,
        "intersection_pixels": intersection,
        "false_positive_pixels": fp,
        "false_negative_pixels": fn,
        "true_negative_pixels": tn,
        "union_pixels": intersection + fp + fn,
        "predicted_positive_pixels": intersection + fp,
        "target_positive_pixels": intersection + fn,
        "total_image_pixels": intersection + fp + fn + tn,
        "detected_targets": detected,
        "total_targets": 10,
        "false_alarm_pixels": fp,
    }


def _count_grid(candidate_ids=CANDIDATES) -> list[dict]:
    records = []
    for candidate_id in candidate_ids:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "dataset": dataset,
                        "condition": condition,
                        "source": _counts(),
                        "adapted": _counts(),
                    }
                )
    return records


def _evaluation(decision, candidate_id):
    return next(
        item
        for item in decision.candidate_evaluations
        if item.candidate_id == candidate_id
    )


def test_metrics_grid_passes_with_exact_39_cell_aggregation_and_safe_labels() -> None:
    decision = evaluate_candidates(
        _metric_grid(), CANDIDATES, _gate()
    )
    assert decision.evidence_kind == "metrics"
    assert decision.integer_count_conservation_verified is False
    assert decision.scientific_status == "scientific_passed"
    assert decision.result_tier == "development"
    assert decision.development_only is True
    assert decision.paper_result is False
    assert decision.p5_authorized is False
    assert decision.eligible_candidate_ids == CANDIDATES
    aggregate = decision.candidate_evaluations[0].aggregate
    assert aggregate.cell_count == CELLS_PER_CANDIDATE == 39
    assert aggregate.nonclean_cell_count == 36
    assert aggregate.nonclean_macro_delta_iou == Fraction(3, 1000)
    assert aggregate.overall_macro_delta_iou == Fraction(36, 39) * Fraction(3, 1000)
    receipt = decision.to_receipt()
    assert receipt["paper_result"] is False
    assert receipt["p5_authorized"] is False
    assert receipt["gate"]["nonclean_macro_delta_iou_epsilon"] == {
        "numerator": 1,
        "denominator": 1000,
    }
    assert receipt["topology"]["cells_per_candidate"] == 39
    assert receipt["ranking"][0]["rank"] == 1


def test_descriptive_alias_has_same_explicit_candidate_roster_api() -> None:
    records = _metric_grid()
    assert evaluate_nonadaptive_teacher_gate(
        records, CANDIDATES, _gate()
    ) == evaluate_candidates(records, CANDIDATES, _gate())


def test_filter_happens_before_deterministic_ranking() -> None:
    fillers = tuple(f"failing_filler_{index}" for index in range(7))
    candidates = ("weaker_safe", "stronger_safe", "huge_but_unsafe", *fillers)
    records = []
    records.extend(
        _metric_grid(
            ("weaker_safe",),
            nonclean_iou_gains={dataset: Fraction(2, 1000) for dataset in DATASETS},
        )
    )
    records.extend(
        _metric_grid(
            fillers,
            nonclean_iou_gains={dataset: Fraction(0, 1) for dataset in DATASETS},
        )
    )
    records.extend(
        _metric_grid(
            ("stronger_safe",),
            nonclean_iou_gains={dataset: Fraction(4, 1000) for dataset in DATASETS},
        )
    )
    records.extend(
        _metric_grid(
            ("huge_but_unsafe",),
            nonclean_iou_gains={dataset: Fraction(1, 100) for dataset in DATASETS},
            fa_gain=Fraction(16, 1),
        )
    )
    decision = evaluate_candidates(records, candidates, _gate())
    assert decision.eligible_candidate_ids == ("stronger_safe", "weaker_safe")
    unsafe = _evaluation(decision, "huge_but_unsafe")
    assert unsafe.eligible is False
    assert "fa_inflation:overall" in unsafe.reason_codes
    assert all(
        entry.candidate_id != "huge_but_unsafe" for entry in decision.ranking
    )


def test_main_iou_gates_are_strict_and_empty_pool_is_normal_negative() -> None:
    exact_epsilon = _metric_grid(
        nonclean_iou_gains={dataset: Fraction(1, 1000) for dataset in DATASETS}
    )
    decision = evaluate_candidates(exact_epsilon, CANDIDATES, _gate())
    assert decision.scientific_status == "scientific_no_eligible"
    assert decision.ranking == ()
    assert "nonclean_macro_iou_not_above_epsilon" in (
        decision.candidate_evaluations[0].reason_codes
    )

    zero_overall = _metric_grid(
        nonclean_iou_gains={dataset: Fraction(3, 1000) for dataset in DATASETS},
        clean_iou_gain=Fraction(-36, 1000),
    )
    decision = evaluate_candidates(zero_overall, CANDIDATES, _gate())
    assert "overall_macro_iou_not_above_threshold" in (
        decision.candidate_evaluations[0].reason_codes
    )


def test_comparison_tolerance_matches_frozen_strict_minimum_and_maximum_semantics() -> None:
    tolerance = Fraction(1, 10**12)
    at_strict_boundary = _metric_grid(
        nonclean_iou_gains={
            dataset: Fraction(1, 1000) + tolerance for dataset in DATASETS
        }
    )
    decision = evaluate_candidates(at_strict_boundary, CANDIDATES, _gate())
    assert "nonclean_macro_iou_not_above_epsilon" in (
        decision.candidate_evaluations[0].reason_codes
    )

    above_strict_boundary = _metric_grid(
        nonclean_iou_gains={
            dataset: Fraction(1, 1000) + tolerance + Fraction(1, 10**15)
            for dataset in DATASETS
        }
    )
    decision = evaluate_candidates(above_strict_boundary, CANDIDATES, _gate())
    assert "nonclean_macro_iou_not_above_epsilon" not in (
        decision.candidate_evaluations[0].reason_codes
    )

    # A dataset delta equal to +tolerance is not strictly positive.
    coverage = _metric_grid(
        nonclean_iou_gains={
            DATASETS[0]: Fraction(4, 1000),
            DATASETS[1]: tolerance,
            DATASETS[2]: Fraction(0, 1),
        }
    )
    decision = evaluate_candidates(coverage, CANDIDATES, _gate())
    assert "positive_nonclean_dataset_coverage" in (
        decision.candidate_evaluations[0].reason_codes
    )

    # Inclusive maximum permits maximum+tolerance and rejects anything larger.
    fa_at_tolerance = _metric_grid(fa_gain=Fraction(15, 1) + tolerance)
    decision = evaluate_candidates(fa_at_tolerance, CANDIDATES, _gate())
    assert not any(
        reason.startswith("fa_inflation")
        for reason in decision.candidate_evaluations[0].reason_codes
    )
    fa_above_tolerance = _metric_grid(
        fa_gain=Fraction(15, 1) + tolerance + Fraction(1, 10**15)
    )
    decision = evaluate_candidates(fa_above_tolerance, CANDIDATES, _gate())
    assert "fa_inflation:overall" in decision.candidate_evaluations[0].reason_codes


def test_ranking_uses_only_yaml_order_and_lexical_final_tie_break() -> None:
    fillers = tuple(f"failing_{index}" for index in range(8))
    candidates = ("z_better_overall", "a_better_worst", *fillers)
    records = []
    records.extend(
        _metric_grid(
            ("z_better_overall",),
            nonclean_iou_gains={
                DATASETS[0]: Fraction(-19, 10_000),
                DATASETS[1]: Fraction(109, 20_000),
                DATASETS[2]: Fraction(109, 20_000),
            },
            clean_iou_gain=Fraction(1, 1000),
        )
    )
    records.extend(
        _metric_grid(
            ("a_better_worst",),
            nonclean_iou_gains={dataset: Fraction(3, 1000) for dataset in DATASETS},
        )
    )
    records.extend(
        _metric_grid(
            fillers,
            nonclean_iou_gains={dataset: Fraction(0, 1) for dataset in DATASETS},
        )
    )
    decision = evaluate_candidates(records, candidates, _gate())
    # Both have nonclean macro .003. YAML puts overall before worst dataset,
    # so z wins despite a's better worst-dataset value and lexical name.
    assert decision.eligible_candidate_ids[:2] == (
        "z_better_overall",
        "a_better_worst",
    )

    tied = _metric_grid(CANDIDATES)
    decision = evaluate_candidates(tied, tuple(reversed(CANDIDATES)), _gate())
    assert decision.eligible_candidate_ids == tuple(sorted(CANDIDATES))


def test_dataset_iou_clean_and_pd_gates_return_specific_reason_codes() -> None:
    records = _metric_grid(
        nonclean_iou_gains={
            DATASETS[0]: Fraction(4, 1000),
            DATASETS[1]: Fraction(-3, 1000),
            DATASETS[2]: Fraction(-1, 1000),
        },
        clean_iou_gain=Fraction(-6, 1000),
        pd_gain=Fraction(-21, 1000),
    )
    decision = evaluate_candidates(records, CANDIDATES, _gate())
    reasons = decision.candidate_evaluations[0].reason_codes
    assert "positive_nonclean_dataset_coverage" in reasons
    assert "worst_nonclean_dataset_iou" in reasons
    assert "clean_macro_iou_safety" in reasons
    assert "clean_dataset_iou_safety:IRSTD-1K" in reasons
    assert "nonclean_pd_safety" in reasons
    assert "dataset_nonclean_pd_safety:NUAA-SIRST" in reasons


def test_fa_and_foreground_boundaries_are_inclusive_then_fail_exactly_above() -> None:
    # Source Fa is 20, so the inclusive maximum delta is 10 + .25*20 = 15.
    at_boundary = _metric_grid(fa_gain=Fraction(15, 1))
    decision = evaluate_candidates(at_boundary, CANDIDATES, _gate())
    assert not any(
        reason.startswith("fa_inflation")
        for reason in decision.candidate_evaluations[0].reason_codes
    )
    above = _metric_grid(fa_gain=Fraction(15001, 1000))
    decision = evaluate_candidates(above, CANDIDATES, _gate())
    assert "fa_inflation:nonclean" in decision.candidate_evaluations[0].reason_codes

    # Delta guard permits .001, but ratio guard for source .01 permits only
    # .002001.  Exercise each guard independently with overridden gates.
    delta_above = _metric_grid(fg_gain=Fraction(1001, 1_000_000))
    decision = evaluate_candidates(delta_above, CANDIDATES, _gate())
    assert "foreground_delta_inflation:overall" in (
        decision.candidate_evaluations[0].reason_codes
    )
    ratio_above = _metric_grid(fg_gain=Fraction(2002, 1_000_000))
    decision = evaluate_candidates(
        ratio_above,
        CANDIDATES,
        _gate(foreground_fraction_delta_maximum=Fraction(1, 100)),
    )
    assert "foreground_ratio_inflation:clean" in (
        decision.candidate_evaluations[0].reason_codes
    )


def test_each_dataset_safety_stratum_covers_clean_and_nonclean_cells() -> None:
    records = _metric_grid()
    # A +208 clean-only Fa delta is diluted below the overall gate (208/39),
    # but exceeds the per-dataset gate when each dataset correctly includes
    # all thirteen cells (208/13 = 16 > 15).  A nonclean-only implementation
    # would miss this dataset-specific failure entirely.
    for record in records:
        if record["dataset"] == DATASETS[0] and record["condition"] == "clean_S0":
            record["teacher"]["fa_per_million"] += Fraction(208, 1)
    decision = evaluate_candidates(records, CANDIDATES, _gate())
    assert "fa_inflation:dataset:IRSTD-1K" in (
        decision.candidate_evaluations[0].reason_codes
    )


def test_count_input_verifies_conservation_and_zero_union_conventions() -> None:
    decision = evaluate_candidates(
        _count_grid(), CANDIDATES, _gate(require_integer_counts=True)
    )
    assert decision.evidence_kind == "counts"
    assert decision.integer_count_conservation_verified is True
    assert decision.scientific_status == "scientific_no_eligible"

    records = _count_grid()
    zero = _counts(intersection=0, fp=0, fn=0, detected=0)
    zero["total_targets"] = 0
    for record in records:
        record["source"] = copy.deepcopy(zero)
        record["adapted"] = copy.deepcopy(zero)
    # IoU=1 and Pd=0 for an empty cell; exact equality remains a valid no-gain
    # scientific result rather than a protocol failure.
    decision = evaluate_candidates(
        records, CANDIDATES, _gate(require_integer_counts=True)
    )
    assert decision.integer_count_conservation_verified is True


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda records: records[0]["adapted"].update(
                {"predicted_positive_pixels": 999}
            ),
            "predicted-positive conservation",
        ),
        (
            lambda records: records[0]["adapted"].update(
                {
                    "intersection_pixels": 49,
                    "false_negative_pixels": 50,
                    "union_pixels": 129,
                    "predicted_positive_pixels": 79,
                    "target_positive_pixels": 99,
                    "true_negative_pixels": PIXELS_PER_CELL - 129,
                }
            ),
            "ground-truth denominators differ",
        ),
    ],
)
def test_count_conservation_or_source_teacher_gt_mismatch_is_protocol_error(
    mutator, match
) -> None:
    records = _count_grid()
    mutator(records)
    with pytest.raises(NonadaptiveTeacherGateError, match=match):
        evaluate_candidates(
            records, CANDIDATES, _gate(require_integer_counts=True)
        )


def test_source_endpoint_must_be_identical_across_candidates() -> None:
    records = _metric_grid(CANDIDATES)
    records[CELLS_PER_CANDIDATE]["source"]["fa_per_million"] = Fraction(21, 1)
    with pytest.raises(NonadaptiveTeacherGateError, match="Source endpoint differs"):
        evaluate_candidates(records, CANDIDATES, _gate())


def test_exact_topology_candidate_roster_duplicates_and_mixed_kinds_fail_closed() -> None:
    missing = _metric_grid()[:-1]
    with pytest.raises(NonadaptiveTeacherGateError, match=r"candidate_count\*39"):
        evaluate_candidates(missing, CANDIDATES, _gate())

    duplicate = _metric_grid()
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(NonadaptiveTeacherGateError, match="duplicate"):
        evaluate_candidates(duplicate, CANDIDATES, _gate())

    with pytest.raises(NonadaptiveTeacherGateError, match="unique"):
        evaluate_candidates(
            _metric_grid(),
            CANDIDATES[:-1] + (CANDIDATES[0],),
            _gate(),
        )

    with pytest.raises(NonadaptiveTeacherGateError, match="exactly 10"):
        evaluate_candidates(
            _metric_grid(CANDIDATES[:-1]), CANDIDATES[:-1], _gate()
        )

    with pytest.raises(NonadaptiveTeacherGateError, match="outside frozen roster"):
        evaluate_candidates(
            _metric_grid(("other", *CANDIDATES[1:])), CANDIDATES, _gate()
        )

    mixed = _metric_grid()
    mixed[0]["source"] = _counts()
    mixed[0]["teacher"] = _counts()
    with pytest.raises(NonadaptiveTeacherGateError, match="cannot mix"):
        evaluate_candidates(mixed, CANDIDATES, _gate())


def test_formal_gate_rejects_metrics_only_scientific_evidence() -> None:
    with pytest.raises(
        NonadaptiveTeacherGateError,
        match="requires integer sufficient counts",
    ):
        evaluate_candidates(
            _metric_grid(),
            CANDIDATES,
            _gate(require_integer_counts=True),
        )


def test_count_cell_shape_and_cross_condition_gt_are_frozen() -> None:
    bad_images = _count_grid()
    bad_images[0]["source"]["image_count"] = IMAGES_PER_CELL - 1
    with pytest.raises(NonadaptiveTeacherGateError, match="image_count must equal 64"):
        evaluate_candidates(
            bad_images, CANDIDATES, _gate(require_integer_counts=True)
        )

    bad_pixels = _count_grid()
    bad_pixels[0]["source"]["total_image_pixels"] -= 1
    bad_pixels[0]["source"]["true_negative_pixels"] -= 1
    with pytest.raises(
        NonadaptiveTeacherGateError, match=r"64\*256\*256"
    ):
        evaluate_candidates(
            bad_pixels, CANDIDATES, _gate(require_integer_counts=True)
        )

    varying_gt = _count_grid()
    replacement = _counts(intersection=50, fp=30, fn=51, detected=8)
    replacement["total_targets"] = 11
    for candidate_index in range(EXPECTED_CANDIDATE_COUNT):
        index = candidate_index * CELLS_PER_CANDIDATE + 1
        varying_gt[index]["source"] = copy.deepcopy(replacement)
        varying_gt[index]["adapted"] = copy.deepcopy(replacement)
    with pytest.raises(
        NonadaptiveTeacherGateError,
        match="ground-truth denominators vary across conditions",
    ):
        evaluate_candidates(
            varying_gt, CANDIDATES, _gate(require_integer_counts=True)
        )


def test_precomputed_metric_validation_rejects_nonfinite_and_out_of_range() -> None:
    records = _metric_grid()
    records[0]["teacher"]["global_iou"] = float("nan")
    with pytest.raises(NonadaptiveTeacherGateError, match="finite"):
        evaluate_candidates(records, CANDIDATES, _gate())

    records = _metric_grid()
    records[0]["teacher"]["pd"] = Fraction(11, 10)
    with pytest.raises(NonadaptiveTeacherGateError, match=r"pd must be in \[0, 1\]"):
        evaluate_candidates(records, CANDIDATES, _gate())


def test_caller_gate_config_is_validated_and_serialized_exactly() -> None:
    gate = _gate()
    assert gate.to_receipt()["fa_source_multiplier"] == {
        "numerator": 1,
        "denominator": 4,
    }
    with pytest.raises(NonadaptiveTeacherGateError, match=r"must be in \[1, 3\]"):
        _gate(minimum_positive_nonclean_datasets=4)
    with pytest.raises(NonadaptiveTeacherGateError, match="must be non-negative"):
        _gate(fa_absolute_allowance_per_million=-1)
