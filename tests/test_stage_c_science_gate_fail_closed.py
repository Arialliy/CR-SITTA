from __future__ import annotations

import ast
import builtins
import copy
import json
from pathlib import Path

import pytest
import yaml

from analysis.stage_c_science_gate_v1 import (
    StageCProtocolError,
    authorize_stage_c_followup,
    evaluate_stage_c0_science_gate,
    evaluate_stage_c_replicate,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "low_contrast",
    "stripe_noise",
)
SPACES = ("R-E1", "R-D0", "P2")


def _config() -> dict:
    config = yaml.safe_load(
        (REPOSITORY / "configs/p3_stage_c_science_gate_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    return config["stage_c0_science_gate"]


def _step(space: str) -> dict:
    relative = space == "P2"
    return {
        "radius_mode": (
            "relative_to_source_parameter_l2" if relative else "absolute_l2"
        ),
        "radius_value": "0.0005" if relative else "0.25",
        "nonzero_epsilon": "1e-12",
        "direction": "negative_proxy_gradient",
        "clip_rule": "min_1_radius_over_norm_plus_epsilon",
    }


def _space(space: str) -> dict:
    return {
        "parameter_space": space,
        # Same 4,608 unique base observations in each grouped-gradient view;
        # these are not added to form a 13,824-episode denominator.
        "nonclean_probe_episode_count": 4608,
        "nonclean_nonfinite_measurement_episode_count": 0,
        "nonclean_finite_nonzero_proxy_gradient_episode_count": 4000,
        "nonclean_threshold_crossing_episode_count": 14,
        "proxy_gradient_norm_mean": "0.20",
        "task_gradient_norm_mean": "0.30",
        "candidate_absolute_response_derivative_mean": "-0.01",
        "candidate_local_contrast_derivative_mean": "-0.02",
        "dataset_aggregates": [
            {
                "dataset_id": dataset,
                "nonclean_probe_episode_count": 1536,
                "both_gradients_nonzero_episode_count": 1400,
                "macro_outer_gradient_cosine": "0.10",
            }
            for dataset in DATASETS
        ],
        "family_aggregates": [
            {
                "family_id": family,
                "nonclean_probe_episode_count": 1152,
                "normalized_virtual_task_loss_directional_derivative": derivative,
            }
            for family, derivative in zip(
                FAMILIES, ("-0.03", "-0.02", "-0.01", "0.01"), strict=True
            )
        ],
        "macro_outer_gradient_cosine": "0.10",
        "virtual_step_contract": _step(space),
        "identity": {
            "comparison_count": 192,
            "exact_match_count": 192,
            "mismatch_count": 0,
            "maximum_absolute_output_difference": "0",
        },
    }


def _evidence() -> dict:
    return {
        "schema_version": "stage_c0_aggregate_evidence_v1",
        "replicate_id": "R0",
        "scope": {
            "data_role": "train",
            "pilot_role": "fixed_Pilot64",
            "pilot_image_count_per_dataset": 64,
            "development_only": True,
            "paper_result": False,
            "thresholds_frozen_before_run": True,
            "validation_access_count": 0,
            "test_access_count": 0,
        },
        "mechanism_aggregate": {
            "mechanism_id": "ASB-SFR_C0",
            "nonclean_probe_episode_count": 4608,
            "nonclean_nonfinite_signal_episode_count": 0,
            "nonclean_active_support_episode_count": 2000,
            "nonclean_active_pixel_count": 100_000,
            "nonclean_candidate_proximal_active_episode_count": 200,
            "nonclean_candidate_proximal_active_pixel_count": 5_000,
            "nonclean_total_pixel_count": 4608 * 256 * 256,
            "signal_summary": {
                "teacher_student_probability_l1_mean": "0.03",
                "teacher_student_logit_gap_mean": "0.10",
                "teacher_student_logit_gap_max": "1.0",
                "active_pixel_fraction_lf_mean": "0.10",
                "active_pixel_fraction_hf_mean": "0.08",
                "active_target_weight_lf_mean": "0.40",
                "active_background_weight_lf_mean": "0.60",
                "active_target_weight_hf_mean": "0.35",
                "active_background_weight_hf_mean": "0.65",
            },
            "space_aggregates": [_space(space) for space in SPACES],
            "o4_activity": {
                "configured": False,
                "active_episode_count": 0,
                "contribution_episode_count": 0,
            },
        },
    }


def _space_evaluation(receipt, name: str):
    return next(
        item
        for item in receipt.mechanism_evaluation.space_evaluations
        if item.parameter_space == name
    )


def test_complete_train_only_unique_mechanism_aggregate_passes() -> None:
    raw = _evidence()
    before = copy.deepcopy(raw)
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    assert raw == before
    assert receipt.protocol_status == "protocol_complete"
    assert receipt.scientific_status == "scientific_eligible"
    assert receipt.eligible_space_ids == SPACES
    assert evaluate_stage_c_replicate(raw, _config()) == SPACES
    assert receipt.mechanism_evaluation.unique_nonclean_probe_episode_count == 4608
    output = receipt.to_receipt()
    assert output["data_role"] == "train_fixed_Pilot64"
    assert output["validation_access_count"] == output["test_access_count"] == 0
    assert output["stage_c1_allowed"] is True
    assert output["stage_c_r1_r2_allowed"] is False
    assert output["formal_test_allowed"] is False
    json.dumps(output, allow_nan=False)


def test_repository_thresholds_are_preregistered_and_crossing_is_report_only() -> None:
    config = _config()
    assert config["minimum_nonclean_finite_nonzero_proxy_gradient_fraction"] == 0.80
    assert config["minimum_nonclean_active_support_episode_fraction"] == 0.30
    assert config["minimum_positive_dataset_cosines"] == 2
    assert config["macro_cosine_strictly_greater_than"] == 0.08
    assert config["minimum_improving_families"] == 3
    assert "minimum_nonclean_threshold_crossing_episode_fraction" not in config
    assert all("candidate" not in key for key in config if key.startswith("expected_"))

    # 14/4608 ~= 0.3%.  C0 records it but §7.2 does not gate on it; the 2%
    # threshold belongs to the later C3 proposal-activity gate (§7.5).
    receipt = evaluate_stage_c0_science_gate(_evidence(), config)
    assert receipt.scientific_status == "scientific_eligible"
    reported = receipt.to_receipt()["mechanism_evaluation"]["space_evaluations"][0]
    assert reported["threshold_crossing_episode_fraction_report_only"] == {
        "numerator": 7,
        "denominator": 2304,
    }


def test_shared_activity_is_counted_once_and_cannot_be_repeated_as_candidates() -> None:
    raw = _evidence()
    mechanism = raw.pop("mechanism_aggregate")
    raw["candidate_aggregates"] = [mechanism, copy.deepcopy(mechanism)]
    with pytest.raises(StageCProtocolError, match="fields must be exact"):
        evaluate_stage_c0_science_gate(raw, _config())

    raw = _evidence()
    raw["mechanism_aggregate"]["space_aggregates"][0][
        "nonclean_probe_episode_count"
    ] = 13_824
    with pytest.raises(StageCProtocolError, match="same unique 4608"):
        evaluate_stage_c0_science_gate(raw, _config())


def test_shared_active_support_and_proximal_far_background_gates() -> None:
    raw = _evidence()
    raw["mechanism_aggregate"]["nonclean_active_support_episode_count"] = 1382
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    assert receipt.scientific_status == "scientific_no_eligible"
    assert "nonclean_active_support_episode_fraction" in (
        receipt.mechanism_evaluation.shared_reason_codes
    )

    raw = _evidence()
    raw["mechanism_aggregate"].update(
        {
            "nonclean_candidate_proximal_active_episode_count": 0,
            "nonclean_candidate_proximal_active_pixel_count": 0,
        }
    )
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    assert "active_support_only_far_background" in (
        receipt.mechanism_evaluation.shared_reason_codes
    )
    assert receipt.eligible_space_ids == ()


def test_space_proxy_gradient_fraction_is_at_least_eighty_percent() -> None:
    raw = _evidence()
    raw["mechanism_aggregate"]["space_aggregates"][0][
        "nonclean_finite_nonzero_proxy_gradient_episode_count"
    ] = 3686
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    evaluation = _space_evaluation(receipt, "R-E1")
    assert "nonclean_finite_nonzero_proxy_gradient_fraction" in evaluation.reason_codes
    assert evaluation.eligible is False
    assert receipt.eligible_space_ids == ("R-D0", "P2")


def test_one_of_three_positive_datasets_fails_even_when_macro_exceeds_point_zero_eight() -> None:
    raw = _evidence()
    space = raw["mechanism_aggregate"]["space_aggregates"][0]
    for item, value in zip(
        space["dataset_aggregates"], ("0.29", "-0.01", "-0.01"), strict=True
    ):
        item["macro_outer_gradient_cosine"] = value
    space["macro_outer_gradient_cosine"] = "0.09"
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    reasons = _space_evaluation(receipt, "R-E1").reason_codes
    assert "positive_dataset_cosine_coverage" in reasons
    assert "macro_outer_gradient_cosine" not in reasons


def test_dataset_coverage_uses_configured_strict_cosine_threshold() -> None:
    raw = _evidence()
    space = raw["mechanism_aggregate"]["space_aggregates"][0]
    for item, value in zip(
        space["dataset_aggregates"], ("0.04", "0.04", "0.19"), strict=True
    ):
        item["macro_outer_gradient_cosine"] = value
    space["macro_outer_gradient_cosine"] = "0.09"
    config = _config()
    config["dataset_cosine_strictly_greater_than"] = "0.05"

    receipt = evaluate_stage_c0_science_gate(raw, config)

    reasons = _space_evaluation(receipt, "R-E1").reason_codes
    assert "positive_dataset_cosine_coverage" in reasons
    assert "macro_outer_gradient_cosine" not in reasons


def test_macro_cosine_must_be_strictly_above_point_zero_eight() -> None:
    raw = _evidence()
    space = raw["mechanism_aggregate"]["space_aggregates"][0]
    for item in space["dataset_aggregates"]:
        item["macro_outer_gradient_cosine"] = "0.08"
    space["macro_outer_gradient_cosine"] = "0.08"
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    assert "macro_outer_gradient_cosine" in _space_evaluation(
        receipt, "R-E1"
    ).reason_codes


def test_exact_fraction_json_supports_nonterminating_dataset_macro_mean() -> None:
    raw = _evidence()
    space = raw["mechanism_aggregate"]["space_aggregates"][0]
    values = (
        {"numerator": 1, "denominator": 10},
        {"numerator": 1, "denominator": 10},
        {"numerator": 1, "denominator": 3},
    )
    for item, value in zip(space["dataset_aggregates"], values, strict=True):
        item["macro_outer_gradient_cosine"] = value
    space["macro_outer_gradient_cosine"] = {"numerator": 8, "denominator": 45}
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    assert _space_evaluation(receipt, "R-E1").eligible is True

    raw = _evidence()
    raw["mechanism_aggregate"]["space_aggregates"][0][
        "macro_outer_gradient_cosine"
    ] = {"numerator": 1, "denominator": 0}
    with pytest.raises(StageCProtocolError, match="positive integer"):
        evaluate_stage_c0_science_gate(raw, _config())


def test_only_two_of_four_improving_families_fails() -> None:
    raw = _evidence()
    family_rows = raw["mechanism_aggregate"]["space_aggregates"][0][
        "family_aggregates"
    ]
    for item, derivative in zip(
        family_rows, ("-0.1", "-0.1", "0", "0.1"), strict=True
    ):
        item["normalized_virtual_task_loss_directional_derivative"] = derivative
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    assert "improving_family_coverage" in _space_evaluation(
        receipt, "R-E1"
    ).reason_codes


def test_identity_is_bit_exact_and_diagnostic_step_contract_is_frozen() -> None:
    raw = _evidence()
    identity = raw["mechanism_aggregate"]["space_aggregates"][0]["identity"]
    identity.update(
        {
            "exact_match_count": 191,
            "mismatch_count": 1,
            "maximum_absolute_output_difference": "1e-8",
        }
    )
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    assert "identity_not_bit_exact" in _space_evaluation(
        receipt, "R-E1"
    ).reason_codes

    raw = _evidence()
    raw["mechanism_aggregate"]["space_aggregates"][0][
        "virtual_step_contract"
    ]["radius_value"] = "0.5"
    with pytest.raises(StageCProtocolError, match="differs from preregistration"):
        evaluate_stage_c0_science_gate(raw, _config())


def test_zero_gradient_spaces_are_never_eligible_or_ranked() -> None:
    raw = _evidence()
    for space in raw["mechanism_aggregate"]["space_aggregates"]:
        space["nonclean_finite_nonzero_proxy_gradient_episode_count"] = 0
        space["proxy_gradient_norm_mean"] = "0"
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    assert receipt.scientific_status == "scientific_no_eligible"
    assert receipt.eligible_space_ids == ()
    assert all(
        "zero_signal_space" in item.reason_codes
        for item in receipt.mechanism_evaluation.space_evaluations
    )
    assert "ranking" not in receipt.to_receipt()


def test_inactive_o4_cannot_claim_contribution() -> None:
    raw = _evidence()
    raw["mechanism_aggregate"]["o4_activity"].update(
        {"configured": True, "contribution_episode_count": 1}
    )
    with pytest.raises(StageCProtocolError, match="inactive or absent O4"):
        evaluate_stage_c0_science_gate(raw, _config())


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda raw: raw["mechanism_aggregate"].pop("signal_summary"),
            "fields must be exact",
        ),
        (
            lambda raw: raw["mechanism_aggregate"]["signal_summary"].update(
                {"teacher_student_gap_l1_mean": float("nan")}
            ),
            "fields must be exact",
        ),
        (
            lambda raw: raw["mechanism_aggregate"]["signal_summary"].update(
                {"teacher_student_probability_l1_mean": float("nan")}
            ),
            "must be finite",
        ),
        (
            lambda raw: raw["mechanism_aggregate"].update(
                {"nonclean_active_pixel_count": 10, "nonclean_candidate_proximal_active_pixel_count": 11}
            ),
            "do not conserve",
        ),
        (
            lambda raw: raw["mechanism_aggregate"].update(
                {"nonclean_nonfinite_signal_episode_count": 1}
            ),
            "non-finite signal",
        ),
        (
            lambda raw: raw["scope"].update({"validation_access_count": 1}),
            "no validation access",
        ),
        (
            lambda raw: raw["scope"].update({"test_access_count": 1}),
            "no test access",
        ),
        (
            lambda raw: raw["scope"].update({"thresholds_frozen_before_run": False}),
            "thresholds must be frozen",
        ),
    ],
)
def test_missing_nonfinite_count_and_access_errors_fail_closed(mutator, match: str) -> None:
    raw = _evidence()
    mutator(raw)
    with pytest.raises(StageCProtocolError, match=match):
        evaluate_stage_c0_science_gate(raw, _config())


def test_scientific_no_eligible_is_normal_and_authorization_has_no_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _evidence()
    for space in raw["mechanism_aggregate"]["space_aggregates"]:
        space["nonclean_finite_nonzero_proxy_gradient_episode_count"] = 0
        space["proxy_gradient_norm_mean"] = "0"
    receipt = evaluate_stage_c0_science_gate(raw, _config())
    assert receipt.to_receipt()["exit_semantics"] == "normal_scientific_early_stop_exit_0"

    def forbidden(*args, **kwargs):  # pragma: no cover - called on regression
        raise AssertionError("authorization attempted a side effect")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    authorization = authorize_stage_c_followup(receipt)
    assert authorization.stage_c1_allowed is False
    assert authorization.parameter_space_ids == ()
    assert authorization.stage_c_r1_r2_allowed is False
    assert authorization.formal_test_allowed is False
    assert authorization.reason == "scientific_no_eligible"


def test_positive_authorization_is_only_for_predefined_train_side_c1() -> None:
    receipt = evaluate_stage_c0_science_gate(_evidence(), _config())
    authorization = authorize_stage_c_followup(receipt)
    assert authorization.stage_c1_allowed is True
    assert authorization.parameter_space_ids == SPACES
    assert authorization.stage_c_r1_r2_allowed is False
    assert authorization.formal_test_allowed is False


def test_gate_module_has_no_torch_cuda_project_or_filesystem_imports() -> None:
    source = (REPOSITORY / "analysis/stage_c_science_gate_v1.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    assert imports.isdisjoint(
        {"torch", "numpy", "yaml", "pathlib", "os", "dataset", "model", "tta"}
    )
