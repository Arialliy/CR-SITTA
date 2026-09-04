from __future__ import annotations

import copy
from fractions import Fraction
import json
from pathlib import Path

import pytest
import yaml

from analysis.p3_stage_b4_science_gate_v1 import (
    CELLS_PER_CANDIDATE,
    CONDITIONS,
    COUNT_FIELDS,
    DATASETS,
    EPISODES_PER_CANDIDATE,
    EPISODES_PER_CELL,
    FROZEN_CANDIDATES,
    GATE_CONFIG_FIELDS,
    NONCLEAN_EPISODES_PER_CANDIDATE,
    RANKING_TIE_BREAK_ORDER,
    StageB4R0ProtocolError,
    TOTAL_IMAGE_PIXELS_PER_CELL,
    default_gate_config,
    evaluate_candidates,
    evaluate_stage_b4_r0_science_gate,
)


def _parts(condition: str) -> tuple[str, str]:
    if condition == "clean_S0":
        return "clean", "S0"
    family, severity_number = condition.rsplit("_S", 1)
    return family, f"S{severity_number}"


def _source_counts() -> dict[str, int]:
    return {
        "intersection_pixels": 5_000,
        "false_positive_pixels": 5_000,
        "false_negative_pixels": 5_000,
        "true_negative_pixels": TOTAL_IMAGE_PIXELS_PER_CELL - 15_000,
        "predicted_positive_pixels": 10_000,
        "target_positive_pixels": 10_000,
        "detected_targets": 500,
        "total_targets": 1_000,
        "false_alarm_pixels": 1_000,
        "total_image_pixels": TOTAL_IMAGE_PIXELS_PER_CELL,
        "image_count": 64,
    }


def _adapted_counts(gain_pixels: int) -> dict[str, int]:
    result = _source_counts()
    result.update(
        {
            "intersection_pixels": 5_000 + gain_pixels,
            "false_positive_pixels": 5_000 - gain_pixels,
            "false_negative_pixels": 5_000 - gain_pixels,
            "true_negative_pixels": (
                TOTAL_IMAGE_PIXELS_PER_CELL - 15_000 + gain_pixels
            ),
            "predicted_positive_pixels": 10_000,
            "target_positive_pixels": 10_000,
            "detected_targets": 500 + gain_pixels // 10,
            "false_alarm_pixels": 1_000 - gain_pixels,
        }
    )
    return result


def _grid(
    gains: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    gains = gains or {
        "O3_P2": 100,
        "O4_P2": 130,
        "O3_DecoderFiLM": 120,
        "O4_DecoderFiLM": 110,
    }
    cells: list[dict] = []
    episodes: list[dict] = []
    for candidate in FROZEN_CANDIDATES:
        for dataset in DATASETS:
            for condition in CONDITIONS:
                family, severity = _parts(condition)
                gain = 0 if condition == "clean_S0" else gains[candidate]
                cells.append(
                    {
                        "candidate_id": candidate,
                        "dataset": dataset,
                        "condition": condition,
                        "corruption_family": family,
                        "severity": severity,
                        "episode_count": 64,
                        "source_counts": _source_counts(),
                        "adapted_counts": _adapted_counts(gain),
                    }
                )
                for episode_index in range(EPISODES_PER_CELL):
                    accepted = episode_index < 32
                    episodes.append(
                        {
                            "candidate_id": candidate,
                            "dataset": dataset,
                            "condition": condition,
                            "episode_index": episode_index,
                            "proxy_gradient_nonzero": True,
                            "task_gradient_nonzero": True,
                            "gradient_cosine": "0.10",
                            "accepted_update": accepted,
                            "finite": True,
                            "maximum_absolute_logit_delta": (
                                "0.000002" if accepted else "0"
                            ),
                            "proposal_loss_before": "1.0",
                            "proposal_loss_after": "0.9" if accepted else "1.0",
                            "threshold_crossing_count": 1 if accepted else 0,
                        }
                    )
    return cells, episodes


def _evaluation(decision, candidate: str):
    return next(
        item
        for item in decision.candidate_evaluations
        if item.candidate_id == candidate
    )


def _candidate_cells(cells: list[dict], candidate: str) -> list[dict]:
    return [cell for cell in cells if cell["candidate_id"] == candidate]


def _candidate_episodes(episodes: list[dict], candidate: str) -> list[dict]:
    return [episode for episode in episodes if episode["candidate_id"] == candidate]


def _set_no_metric_change(cells: list[dict], candidate: str) -> None:
    for cell in _candidate_cells(cells, candidate):
        cell["adapted_counts"] = copy.deepcopy(cell["source_counts"])


def _set_nonclean_acceptance(
    episodes: list[dict], candidate: str, accepted_count: int
) -> None:
    selected = [
        episode
        for episode in _candidate_episodes(episodes, candidate)
        if episode["condition"] != "clean_S0"
    ]
    assert len(selected) == NONCLEAN_EPISODES_PER_CANDIDATE
    for index, episode in enumerate(selected):
        accepted = index < accepted_count
        episode["accepted_update"] = accepted
        episode["finite"] = True
        episode["maximum_absolute_logit_delta"] = (
            "0.000002" if accepted else "0"
        )
        episode["proposal_loss_after"] = "0.9" if accepted else "1.0"
        episode["threshold_crossing_count"] = 1 if accepted else 0


def test_complete_exact_grid_passes_and_receipt_is_json_safe() -> None:
    cells, episodes = _grid()
    before_cells = copy.deepcopy(cells)
    before_episodes = copy.deepcopy(episodes)
    decision = evaluate_stage_b4_r0_science_gate(
        cells, episodes, default_gate_config()
    )

    assert cells == before_cells
    assert episodes == before_episodes
    assert decision.scientific_status == "scientific_pending_R1_R2"
    assert decision.r1_r2_allowed is True
    assert decision.next_stage_allowed is True
    assert decision.selected_candidate_ids == (
        "O4_P2",
        "O3_DecoderFiLM",
        "O4_DecoderFiLM",
        "O3_P2",
    )
    aggregate = _evaluation(decision, "O3_P2").aggregate
    assert aggregate.cell_count == CELLS_PER_CANDIDATE == 39
    assert aggregate.episode_count == EPISODES_PER_CANDIDATE == 2496
    assert aggregate.nonclean_episode_count == 2304
    assert aggregate.nonclean_accepted_update_episode_count == 1152
    assert aggregate.nonclean_no_update_episode_count == 1152
    assert aggregate.nonclean_accepted_update_fraction == Fraction(1, 2)
    assert aggregate.nonclean_accepted_functional_fraction == 1
    assert aggregate.nonclean_accepted_proposal_loss_decrease_fraction == 1
    assert aggregate.nonclean_accepted_finite_fraction == 1
    assert aggregate.both_gradients_nonzero_fraction == 1
    assert aggregate.alignment_macro_cosine == Fraction(1, 10)

    receipt = decision.to_receipt()
    assert set(receipt) == {
        "schema_version",
        "receipt_type",
        "evaluation_phase",
        "protocol_status",
        "scientific_status",
        "result_tier",
        "development_only",
        "paper_result",
        "test",
        "r1_r2_allowed",
        "next_stage_allowed",
        "final_candidate_selection_allowed",
        "gate",
        "topology",
        "aggregation_policy",
        "selection_policy",
        "across_replicate_policy",
        "gain_attribution_policy",
        "candidate_evaluations",
        "eligible_candidate_ids",
        "selected_for_r1_r2",
        "ranking",
        "required_followup_replicates",
        "exit_semantics",
    }
    assert receipt["protocol_status"] == "passed"
    assert receipt["paper_result"] is False
    assert receipt["test"] is False
    assert receipt["topology"]["candidate_ids"] == list(FROZEN_CANDIDATES)
    assert receipt["topology"]["episodes_per_candidate"] == 2496
    assert receipt["topology"]["nonclean_episodes_per_candidate"] == 2304
    assert receipt["required_followup_replicates"] == ["R1", "R2"]
    assert receipt["across_replicate_policy"]["application_at_r0"] == (
        "recorded_only_not_applied"
    )
    assert receipt["across_replicate_policy"]["maximum_final_top_candidates"] == 3
    assert receipt["aggregation_policy"]["missing_value_policy"] == (
        "forbidden_no_imputation"
    )
    assert set(receipt["gate"]) == GATE_CONFIG_FIELDS
    json.dumps(receipt, allow_nan=False)


def test_repository_yaml_gate_is_parsed_completely() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (repository / "configs/p3_stage_b4_full_pilot64_proposal_gate_v1.yaml")
        .read_text(encoding="utf-8")
    )
    cells, episodes = _grid()
    decision = evaluate_candidates(
        cells, episodes, config["stage_b4_r0_science_gate"]
    )
    assert decision.r1_r2_allowed is True
    assert decision.gate.ranking_tie_break_order == RANKING_TIE_BREAK_ORDER
    assert decision.gate.comparison_tolerance == Fraction(1, 10**12)


def test_zero_eligible_is_normal_scientific_early_stop() -> None:
    cells, episodes = _grid()
    for candidate in FROZEN_CANDIDATES:
        _set_no_metric_change(cells, candidate)
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    assert decision.scientific_status == "scientific_no_eligible"
    assert decision.r1_r2_allowed is False
    assert decision.eligible_candidate_ids == ()
    assert decision.ranking == ()
    receipt = decision.to_receipt()
    assert receipt["protocol_status"] == "passed"
    assert receipt["selected_for_r1_r2"] == []
    assert receipt["required_followup_replicates"] == []
    assert receipt["exit_semantics"] == "normal_scientific_early_stop_exit_0"


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda cells, episodes: cells[0].update({"unknown": 1}), "fields must be exact"),
        (
            lambda cells, episodes: cells[0]["source_counts"].update(
                {"true_negative_pixels": 984_999}
            ),
            "do not conserve",
        ),
        (
            lambda cells, episodes: episodes[0].update(
                {"gradient_cosine": float("nan")}
            ),
            "must be finite",
        ),
        (
            lambda cells, episodes: episodes[0].update(
                {"proxy_gradient_nonzero": False}
            ),
            "must be exact zero",
        ),
        (
            lambda cells, episodes: episodes[63].update(
                {"maximum_absolute_logit_delta": "0.1"}
            ),
            "no-update outcome",
        ),
    ],
)
def test_schema_counts_finiteness_and_no_update_contradictions_fail_protocol(
    mutator, match: str
) -> None:
    cells, episodes = _grid()
    mutator(cells, episodes)
    with pytest.raises(StageB4R0ProtocolError, match=match):
        evaluate_candidates(cells, episodes, default_gate_config())


def test_exact_cell_episode_topology_and_frozen_roster_fail_closed() -> None:
    cells, episodes = _grid()
    with pytest.raises(StageB4R0ProtocolError, match=r"4\*39"):
        evaluate_candidates(cells[:-1], episodes, default_gate_config())
    with pytest.raises(StageB4R0ProtocolError, match=r"4\*2496"):
        evaluate_candidates(cells, episodes[:-1], default_gate_config())

    duplicate_cells = copy.deepcopy(cells)
    duplicate_cells[-1] = copy.deepcopy(duplicate_cells[0])
    with pytest.raises(StageB4R0ProtocolError, match="duplicate"):
        evaluate_candidates(duplicate_cells, episodes, default_gate_config())

    bad_roster = copy.deepcopy(cells)
    bad_roster[0]["candidate_id"] = "not_frozen"
    with pytest.raises(StageB4R0ProtocolError, match="frozen four-candidate"):
        evaluate_candidates(bad_roster, episodes, default_gate_config())


def test_source_integer_sufficient_counts_must_match_across_candidates() -> None:
    cells, episodes = _grid()
    second_candidate_first_cell = cells[CELLS_PER_CANDIDATE]
    second_candidate_first_cell["source_counts"]["false_alarm_pixels"] += 1
    with pytest.raises(StageB4R0ProtocolError, match="Source sufficient counts differ"):
        evaluate_candidates(cells, episodes, default_gate_config())


def test_gt_denominators_must_be_invariant_across_corruption_conditions() -> None:
    cells, episodes = _grid()
    changed = cells[1]
    for endpoint in ("source_counts", "adapted_counts"):
        counts = changed[endpoint]
        counts["false_negative_pixels"] += 1
        counts["target_positive_pixels"] += 1
        counts["true_negative_pixels"] -= 1
    # Keep the same Source count for this condition across all candidates so
    # the failure specifically exercises cross-condition Pilot64/GT integrity.
    for candidate_index in range(1, len(FROZEN_CANDIDATES)):
        other = cells[candidate_index * CELLS_PER_CANDIDATE + 1]
        other["source_counts"] = copy.deepcopy(changed["source_counts"])
        other["adapted_counts"]["false_negative_pixels"] += 1
        other["adapted_counts"]["target_positive_pixels"] += 1
        other["adapted_counts"]["true_negative_pixels"] -= 1
    with pytest.raises(StageB4R0ProtocolError, match="denominators differ"):
        evaluate_candidates(cells, episodes, default_gate_config())


def test_missing_and_duplicate_episode_are_protocol_errors() -> None:
    cells, episodes = _grid()
    duplicate = copy.deepcopy(episodes)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(StageB4R0ProtocolError, match="duplicate"):
        evaluate_candidates(cells, duplicate, default_gate_config())

    wrong_index = copy.deepcopy(episodes)
    wrong_index[0]["episode_index"] = 64
    with pytest.raises(StageB4R0ProtocolError, match=r"\[0, 63\]"):
        evaluate_candidates(cells, wrong_index, default_gate_config())


def test_metric_change_requires_accepted_and_threshold_crossing_evidence() -> None:
    cells, episodes = _grid()
    first_key = (
        cells[1]["candidate_id"],
        cells[1]["dataset"],
        cells[1]["condition"],
    )
    for episode in episodes:
        if (
            episode["candidate_id"],
            episode["dataset"],
            episode["condition"],
        ) == first_key:
            episode["accepted_update"] = False
            episode["maximum_absolute_logit_delta"] = "0"
            episode["proposal_loss_after"] = "1.0"
            episode["threshold_crossing_count"] = 0
    with pytest.raises(StageB4R0ProtocolError, match="zero accepted updates"):
        evaluate_candidates(cells, episodes, default_gate_config())


def test_dataset_and_family_performance_coverage_are_hard_gates() -> None:
    cells, episodes = _grid()
    candidate = FROZEN_CANDIDATES[0]
    for cell in _candidate_cells(cells, candidate):
        if cell["dataset"] != DATASETS[0]:
            cell["adapted_counts"] = copy.deepcopy(cell["source_counts"])
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    assert "positive_dataset_coverage" in _evaluation(
        decision, candidate
    ).reason_codes

    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        if cell["corruption_family"] in ("low_contrast", "stripe_noise"):
            cell["adapted_counts"] = copy.deepcopy(cell["source_counts"])
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    assert "positive_family_coverage" in _evaluation(
        decision, candidate
    ).reason_codes


def test_worst_dataset_and_family_iou_are_independent_hard_gates() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        if cell["dataset"] == DATASETS[-1] and cell["condition"] != "clean_S0":
            cell["adapted_counts"] = _adapted_counts(-100)
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    assert "worst_dataset_iou" in _evaluation(decision, candidate).reason_codes

    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        if cell["corruption_family"] == "stripe_noise":
            cell["adapted_counts"] = _adapted_counts(-100)
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    assert "worst_family_iou" in _evaluation(decision, candidate).reason_codes


def test_clean_iou_and_pd_safety_are_independent_hard_gates() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        if cell["condition"] == "clean_S0":
            cell["adapted_counts"] = _adapted_counts(-500)
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    reasons = _evaluation(decision, candidate).reason_codes
    assert "clean_macro_iou_safety" in reasons
    assert "clean_dataset_iou_safety" in reasons

    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        if cell["condition"] != "clean_S0":
            cell["adapted_counts"]["detected_targets"] = 450
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    reasons = _evaluation(decision, candidate).reason_codes
    assert "nonclean_pd_safety" in reasons
    assert "dataset_pd_safety" in reasons

    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        if cell["condition"] == "clean_S0":
            cell["adapted_counts"]["detected_targets"] = 480
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    assert "clean_pd_safety" in _evaluation(decision, candidate).reason_codes


def test_fa_and_foreground_guards_cover_all_configured_strata() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        cell["adapted_counts"]["false_alarm_pixels"] = 2_000
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    reasons = _evaluation(decision, candidate).reason_codes
    assert "fa_inflation:overall" in reasons
    assert "fa_inflation:nonclean" in reasons
    assert "fa_inflation:clean" in reasons
    assert f"fa_inflation:dataset:{DATASETS[0]}" in reasons
    assert "fa_inflation:corruption_family:gaussian_noise" in reasons

    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        counts = cell["adapted_counts"]
        counts["false_positive_pixels"] += 5_000
        counts["predicted_positive_pixels"] += 5_000
        counts["true_negative_pixels"] -= 5_000
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    reasons = _evaluation(decision, candidate).reason_codes
    assert "foreground_absolute_inflation:overall" in reasons
    assert "foreground_ratio_inflation:overall" in reasons
    assert f"foreground_absolute_inflation:dataset:{DATASETS[1]}" in reasons
    assert "foreground_ratio_inflation:corruption_family:stripe_noise" in reasons


def test_alignment_macro_dataset_medians_and_nonzero_fraction_are_separate() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    for episode in _candidate_episodes(episodes, candidate):
        episode["gradient_cosine"] = (
            "0.20" if episode["dataset"] == DATASETS[0] else "-0.01"
        )
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    evaluation = _evaluation(decision, candidate)
    assert evaluation.aggregate.alignment_macro_cosine == Fraction(3, 50)
    assert "alignment_macro_cosine" not in evaluation.reason_codes
    assert "alignment_dataset_coverage" in evaluation.reason_codes

    cells, episodes = _grid()
    for episode in _candidate_episodes(episodes, candidate)[:600]:
        episode["proxy_gradient_nonzero"] = False
        episode["gradient_cosine"] = "0"
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    evaluation = _evaluation(decision, candidate)
    assert "alignment_nonzero_gradient_fraction" in evaluation.reason_codes
    assert "alignment_macro_cosine" not in evaluation.reason_codes

    cells, episodes = _grid()
    for episode in _candidate_episodes(episodes, candidate):
        episode["gradient_cosine"] = "0.04"
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    evaluation = _evaluation(decision, candidate)
    assert "alignment_macro_cosine" in evaluation.reason_codes
    assert "alignment_dataset_coverage" not in evaluation.reason_codes


def test_accept_and_no_update_exact_integer_boundaries() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    _set_no_metric_change(cells, candidate)
    _set_nonclean_acceptance(episodes, candidate, 460)
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    reasons = _evaluation(decision, candidate).reason_codes
    assert "accepted_update_fraction_nonclean" in reasons
    assert "no_update_fraction_nonclean" in reasons

    cells, episodes = _grid()
    _set_no_metric_change(cells, candidate)
    _set_nonclean_acceptance(episodes, candidate, 461)
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    reasons = _evaluation(decision, candidate).reason_codes
    assert "accepted_update_fraction_nonclean" not in reasons
    assert "no_update_fraction_nonclean" not in reasons


def test_accepted_quality_and_finite_gates_use_only_accepted_nonclean() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    accepted = [
        episode
        for episode in _candidate_episodes(episodes, candidate)
        if episode["condition"] != "clean_S0" and episode["accepted_update"]
    ]
    assert len(accepted) == 1152
    for episode in accepted[:116]:
        episode["maximum_absolute_logit_delta"] = "0.000001"
    for episode in accepted[116:232]:
        episode["proposal_loss_after"] = "1.0"
    accepted[232]["finite"] = False
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    reasons = _evaluation(decision, candidate).reason_codes
    assert "accepted_functional_logit_changed_fraction" in reasons
    assert "accepted_proposal_loss_decrease_fraction" in reasons
    assert "accepted_finite_fraction" in reasons


def test_threshold_crossing_episode_fraction_uses_episode_not_crossing_count() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    nonclean = [
        episode
        for episode in _candidate_episodes(episodes, candidate)
        if episode["condition"] != "clean_S0"
    ]
    for episode in nonclean:
        episode["threshold_crossing_count"] = 0
    # One crossing episode in each changed cell preserves protocol consistency,
    # but 36/2304 remains below the frozen 0.02 science threshold.
    for episode in nonclean:
        if episode["episode_index"] == 0:
            episode["threshold_crossing_count"] = 10_000
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    evaluation = _evaluation(decision, candidate)
    assert evaluation.aggregate.nonclean_threshold_crossing_episode_count == 36
    assert "nonclean_threshold_crossing_episode_fraction" in evaluation.reason_codes


def test_zero_accepted_quality_denominators_are_exact_zero_not_missing() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    _set_no_metric_change(cells, candidate)
    _set_nonclean_acceptance(episodes, candidate, 0)
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    aggregate = _evaluation(decision, candidate).aggregate
    assert aggregate.nonclean_accepted_functional_fraction == 0
    assert aggregate.nonclean_accepted_proposal_loss_decrease_fraction == 0
    assert aggregate.nonclean_accepted_finite_fraction == 0


def test_filter_before_ranking_excludes_unsafe_highest_gain() -> None:
    gains = {
        "O3_P2": 100,
        "O4_P2": 400,
        "O3_DecoderFiLM": 120,
        "O4_DecoderFiLM": 110,
    }
    cells, episodes = _grid(gains)
    for cell in _candidate_cells(cells, "O4_P2"):
        counts = cell["adapted_counts"]
        counts["false_positive_pixels"] += 2_100
        counts["predicted_positive_pixels"] += 2_100
        counts["true_negative_pixels"] -= 2_100
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    unsafe = _evaluation(decision, "O4_P2")
    assert unsafe.eligible is False
    assert "O4_P2" not in decision.eligible_candidate_ids
    assert "O4_P2" not in decision.selected_candidate_ids
    assert decision.selected_candidate_ids[0] == "O3_DecoderFiLM"


def test_across_replicate_thresholds_are_recorded_but_not_applied_at_r0() -> None:
    gains = {
        "O3_P2": 10,
        "O4_P2": 130,
        "O3_DecoderFiLM": 120,
        "O4_DecoderFiLM": 110,
    }
    cells, episodes = _grid(gains)
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    evaluation = _evaluation(decision, "O3_P2")
    assert evaluation.aggregate.nonclean_macro_delta_iou > 0
    assert (
        evaluation.aggregate.nonclean_macro_delta_iou
        < decision.gate.across_replicate_nonclean_mean_delta_iou_minimum
    )
    assert (
        evaluation.aggregate.overall_macro_delta_iou
        < decision.gate.across_replicate_overall_mean_delta_iou_minimum
    )
    assert evaluation.eligible is True
    assert "O3_P2" in decision.selected_candidate_ids


def test_nested_config_unknown_relaxed_ranking_and_actions_fail_protocol() -> None:
    cells, episodes = _grid()
    missing = default_gate_config()
    missing["activity"].pop("scope")
    with pytest.raises(StageB4R0ProtocolError, match="fields must be exact"):
        evaluate_candidates(cells, episodes, missing)

    relaxed = default_gate_config()
    relaxed["per_replicate_performance_and_safety"][
        "minimum_positive_nonclean_datasets"
    ] = 1
    with pytest.raises(StageB4R0ProtocolError, match="frozen value 2"):
        evaluate_candidates(cells, episodes, relaxed)

    reordered = default_gate_config()
    reordered["ranking_tie_break_order"] = list(
        reversed(reordered["ranking_tie_break_order"])
    )
    with pytest.raises(StageB4R0ProtocolError, match="ranking_tie_break_order"):
        evaluate_candidates(cells, episodes, reordered)

    wrong_action = default_gate_config()
    wrong_action["r0_eligible_action"] = "run_everything"
    with pytest.raises(StageB4R0ProtocolError, match="r0_eligible_action"):
        evaluate_candidates(cells, episodes, wrong_action)


def test_record_order_does_not_change_exact_decision() -> None:
    cells, episodes = _grid()
    forward = evaluate_candidates(cells, episodes, default_gate_config()).to_receipt()
    reverse = evaluate_candidates(
        list(reversed(cells)), list(reversed(episodes)), default_gate_config()
    ).to_receipt()
    assert forward == reverse


def test_count_field_contract_matches_endpoint_sufficient_statistics() -> None:
    assert COUNT_FIELDS == {
        "intersection_pixels",
        "false_positive_pixels",
        "false_negative_pixels",
        "true_negative_pixels",
        "predicted_positive_pixels",
        "target_positive_pixels",
        "detected_targets",
        "total_targets",
        "false_alarm_pixels",
        "total_image_pixels",
        "image_count",
    }


def _empty_counts(predicted_positive_pixels: int = 0) -> dict[str, int]:
    return {
        "intersection_pixels": 0,
        "false_positive_pixels": predicted_positive_pixels,
        "false_negative_pixels": 0,
        "true_negative_pixels": (
            TOTAL_IMAGE_PIXELS_PER_CELL - predicted_positive_pixels
        ),
        "predicted_positive_pixels": predicted_positive_pixels,
        "target_positive_pixels": 0,
        "detected_targets": 0,
        "total_targets": 0,
        "false_alarm_pixels": predicted_positive_pixels,
        "total_image_pixels": TOTAL_IMAGE_PIXELS_PER_CELL,
        "image_count": 64,
    }


def test_empty_union_iou_matches_official_one_and_empty_to_nonempty_is_minus_one() -> None:
    cells, episodes = _grid()
    candidate = FROZEN_CANDIDATES[0]
    for cell in cells:
        cell["source_counts"] = _empty_counts()
        cell["adapted_counts"] = (
            _empty_counts(100)
            if cell["candidate_id"] == candidate
            and cell["condition"] != "clean_S0"
            else _empty_counts()
        )
    decision = evaluate_candidates(cells, episodes, default_gate_config())
    empty_to_nonempty = _evaluation(decision, candidate).aggregate
    assert empty_to_nonempty.nonclean_macro_delta_iou == -1
    assert empty_to_nonempty.clean_macro_delta_iou == 0
    assert empty_to_nonempty.overall_macro_delta_iou == Fraction(-12, 13)
    double_empty = _evaluation(decision, FROZEN_CANDIDATES[1]).aggregate
    assert double_empty.nonclean_macro_delta_iou == 0
    assert double_empty.overall_macro_delta_iou == 0


def test_standalone_gate_rejects_non_64x256x256_pixel_scale() -> None:
    cells, episodes = _grid()
    counts = cells[0]["source_counts"]
    counts["total_image_pixels"] -= 1
    counts["true_negative_pixels"] -= 1
    with pytest.raises(StageB4R0ProtocolError, match=r"64\*256\*256"):
        evaluate_candidates(cells, episodes, default_gate_config())


def test_alignment_and_accepted_quality_use_frozen_tolerance_exactly() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    for episode in _candidate_episodes(episodes, candidate):
        episode["gradient_cosine"] = "0.049999999999"
    at_inclusive_boundary = evaluate_candidates(
        cells, episodes, default_gate_config()
    )
    assert "alignment_macro_cosine" not in _evaluation(
        at_inclusive_boundary, candidate
    ).reason_codes

    for episode in _candidate_episodes(episodes, candidate):
        episode["gradient_cosine"] = "0.0499999999989"
    below_inclusive_boundary = evaluate_candidates(
        cells, episodes, default_gate_config()
    )
    assert "alignment_macro_cosine" in _evaluation(
        below_inclusive_boundary, candidate
    ).reason_codes

    cells, episodes = _grid()
    accepted = [
        episode
        for episode in _candidate_episodes(episodes, candidate)
        if episode["condition"] != "clean_S0" and episode["accepted_update"]
    ]
    for episode in accepted:
        episode["maximum_absolute_logit_delta"] = "0.000001000001"
        episode["proposal_loss_after"] = "0.999999999999"
    at_strict_boundary = evaluate_candidates(cells, episodes, default_gate_config())
    reasons = _evaluation(at_strict_boundary, candidate).reason_codes
    assert "accepted_functional_logit_changed_fraction" in reasons
    assert "accepted_proposal_loss_decrease_fraction" in reasons

    for episode in accepted:
        episode["maximum_absolute_logit_delta"] = "0.0000010000011"
        episode["proposal_loss_after"] = "0.9999999999989"
    beyond_strict_boundary = evaluate_candidates(
        cells, episodes, default_gate_config()
    )
    reasons = _evaluation(beyond_strict_boundary, candidate).reason_codes
    assert "accepted_functional_logit_changed_fraction" not in reasons
    assert "accepted_proposal_loss_decrease_fraction" not in reasons


def test_inclusive_fa_and_foreground_exact_boundaries_pass_then_fail() -> None:
    candidate = FROZEN_CANDIDATES[0]
    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        cell["adapted_counts"]["false_alarm_pixels"] = 1_291
    boundary = evaluate_candidates(cells, episodes, default_gate_config())
    assert not any(
        reason.startswith("fa_inflation")
        for reason in _evaluation(boundary, candidate).reason_codes
    )
    for cell in _candidate_cells(cells, candidate):
        cell["adapted_counts"]["false_alarm_pixels"] = 1_292
    above = evaluate_candidates(cells, episodes, default_gate_config())
    assert "fa_inflation:overall" in _evaluation(above, candidate).reason_codes

    cells, episodes = _grid()
    for cell in _candidate_cells(cells, candidate):
        counts = cell["adapted_counts"]
        counts["false_positive_pixels"] += 2_004
        counts["predicted_positive_pixels"] += 2_004
        counts["true_negative_pixels"] -= 2_004
    fg_boundary = evaluate_candidates(cells, episodes, default_gate_config())
    reasons = _evaluation(fg_boundary, candidate).reason_codes
    assert not any(reason.startswith("foreground_ratio_inflation") for reason in reasons)
    for cell in _candidate_cells(cells, candidate):
        counts = cell["adapted_counts"]
        counts["false_positive_pixels"] += 1
        counts["predicted_positive_pixels"] += 1
        counts["true_negative_pixels"] -= 1
    fg_above = evaluate_candidates(cells, episodes, default_gate_config())
    assert "foreground_ratio_inflation:overall" in _evaluation(
        fg_above, candidate
    ).reason_codes
