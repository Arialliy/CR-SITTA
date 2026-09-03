from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
import os
from pathlib import Path

import pytest
import yaml

from analysis.d0_v3_formal_contract import (
    BASE_ENGINEERING_CONFIG_SHA256,
    CONDITIONS,
    CONFIG_CANONICAL_MAPPING_SHA256,
    CONFIG_FILE_SHA256,
    DATASETS,
    D0V3FormalContractError,
    ENGINEERING_SMOKE_AGGREGATE_SHA256,
    FINE_ALIGNMENT_GROUP_IDS,
    load_d0_v3_formal_contract,
    parse_d0_v3_formal_contract,
    verify_frozen_parent_bindings,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tent_failure_diagnostics_v3_formal_stage_a.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _mapping() -> dict:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_canonical_contract_is_exact_train_pilot64_and_non_authorizing() -> None:
    contract = load_d0_v3_formal_contract(CONFIG)

    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_FILE_SHA256
    assert contract.config_file_sha256 == CONFIG_FILE_SHA256
    assert contract.canonical_mapping_sha256() == CONFIG_CANONICAL_MAPPING_SHA256
    assert contract.datasets == DATASETS
    assert contract.conditions == CONDITIONS
    assert len(contract.candidates) == 10
    assert len(contract.fine_alignment_group_ids) == 20
    assert contract.fine_alignment_group_ids == FINE_ALIGNMENT_GROUP_IDS
    assert contract.formal_protocol_complete_initial is False
    assert contract.scientific_status_initial == "not_evaluated"
    assert contract.stage2_authorized is False

    scope = contract.raw["scope"]
    assert scope["split_name"] == "train"
    assert scope["split_role"] == "frozen_pilot64"
    assert scope["no_validation_split"] is True
    assert scope["use_validation_payload"] is False
    assert scope["use_test_payload"] is False
    assert scope["dataset_count"] == 3
    assert scope["condition_count_per_dataset"] == 13
    assert scope["image_count_per_condition"] == 64
    assert scope["candidate_count"] == 10


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_parent_config_smoke_and_checkpoint_paths_are_frozen() -> None:
    contract = load_d0_v3_formal_contract(CONFIG)
    bindings = contract.raw["frozen_parent_bindings"]
    assert bindings["engineering_config"]["sha256"] == (
        BASE_ENGINEERING_CONFIG_SHA256
    )
    assert bindings["engineering_smoke"]["aggregate_sha256"] == (
        ENGINEERING_SMOKE_AGGREGATE_SHA256
    )
    verified = verify_frozen_parent_bindings(contract, repository_root=ROOT)
    assert verified[0][1] == BASE_ENGINEERING_CONFIG_SHA256
    assert verified[1][1] == ENGINEERING_SMOKE_AGGREGATE_SHA256
    for dataset in DATASETS:
        entry = contract.raw["datasets"][dataset]
        assert entry["checkpoint_role"] == "best_miou"
        assert entry["checkpoint_path"].endswith("/best_miou.pth.tar")
        assert len(entry["checkpoint_sha256"]) == 64


def test_candidate_then_outer_target_firewall_is_unambiguous() -> None:
    contract = load_d0_v3_formal_contract(CONFIG)
    firewall = contract.raw["payload_firewall"]
    candidate = firewall["candidate_phase"]
    assert candidate["method_label_accesses"] == 0
    assert candidate["outer_evaluator_label_accesses"] == 0
    assert candidate["train_target_bytes_opened"] == 0
    assert candidate["train_target_deserialized"] is False
    assert candidate["validation_payload_opens"] == 0
    assert candidate["test_payload_opens"] == 0
    assert candidate["completion_receipt_required"] is True
    outer = firewall["outer_evaluator"]
    assert outer["allowed_target_role"] == "train_pilot64_outer_target"
    assert outer["target_load_timing"] == (
        "after_candidate_phase_completion_receipt_for_dataset_condition_replicate"
    )
    assert outer["target_visible_to_adaptation"] is False


def test_numeric_protocol_and_outer_loss_are_fully_frozen() -> None:
    contract = load_d0_v3_formal_contract(CONFIG)
    numeric = contract.raw["formal_numeric_protocol"]
    assert dict(numeric["no_op_thresholds"]) == {
        "parameter_null_floor": 0.0,
        "logit_null_floor": 0.0,
        "probability_null_floor": 0.0,
        "probability_logit_consistency_atol": 1.0e-6,
        "prediction_threshold": 0.5,
        "near_threshold_lower": 0.45,
        "near_threshold_upper": 0.55,
        "entropy_eps": 1.0e-6,
        "connectivity": 2,
        "min_component_area": 1,
        "max_centroid_distance": 3.0,
    }
    geometry = numeric["optimizer_geometry"]
    assert geometry["inherited_from_parent_v2"] is True
    assert geometry["small_gradient_threshold"] == 1.0e-8
    assert geometry["adam_near_sign_step_threshold"] == 0.9
    assert geometry["verification_rtol"] == 1.0e-5
    assert geometry["verification_atol"] == 1.0e-7
    task = numeric["outer_oracle_task_loss"]
    assert task["used_by_adaptation"] is False
    assert task["lambda_bce"] == task["lambda_soft_iou"] == 1.0
    assert task["eps"] == 1.0e-6
    assert task["bce_reduction"] == "mean_over_all_pixels"
    assert task["soft_iou_reduction"] == "mean_over_sample"
    alignment = numeric["alignment"]
    assert alignment["first_order_zero_tolerance"] == 0.0
    assert alignment["selection_cosine_unit"] == (
        "per_episode_global_all_BN_affine_vector"
    )
    assert alignment["zero_gradient_norm_policy"] == (
        "cosine_null_with_reason_code"
    )
    assert alignment["both_gradients_nonzero_episode_denominator"] == 2496
    assert alignment["fine_group_observations_per_candidate_replicate"] == 49920
    assert alignment["fine_group_role"] == "diagnostic_only_not_selection"

    aggregation = numeric["aggregation_semantics"]
    assert aggregation["cell_sufficient_statistics"] == (
        "pooled_over_64_images_before_metric_computation"
    )
    assert aggregation["global_iou"]["zero_union_value"] == 1.0
    assert aggregation["pd"]["zero_target_value"] == 0.0
    assert aggregation["fa_per_million_pixels"]["multiplier"] == 1000000.0
    assert aggregation["cross_cell_aggregation"] == "equal_cell_macro"
    assert dict(aggregation["macro_cell_counts"]) == {
        "overall": 39,
        "nonclean": 36,
        "clean": 3,
        "per_dataset_nonclean": 12,
        "per_corruption_family": 9,
    }


def test_replicate_gate_ranking_and_stage_b_preregistration_are_exact() -> None:
    contract = load_d0_v3_formal_contract(CONFIG)
    execution = contract.raw["replicate_execution"]
    assert execution["replicate_ids"] == ("R0", "R1", "R2")
    assert execution["r0_total_candidate_episodes"] == 24960
    assert execution["followup_candidate_policy"] == (
        "r0_scientifically_eligible_only"
    )
    assert execution["same_seed_across_replicates"] is True
    assert execution["independent_fresh_process_per_replicate"] is True

    gate = contract.raw["stage_a_science_gate"]
    assert gate["across_R0_R1_R2"][
        "nonclean_36_macro_delta_iou_mean_minimum"
    ] == 0.002
    assert gate["across_R0_R1_R2"][
        "overall_39_macro_delta_iou_mean_minimum"
    ] == 0.001
    assert gate["comparison_tolerance"] == 1.0e-12
    assert gate["top_k"] is None
    assert gate["stage2_authorized_by_gate"] is False
    assert len(gate["ranking_tie_break_order"]) == 10

    stage_b = contract.raw["stage_b_preregistration"]
    assert stage_b["authorized_by_this_config"] is False
    assert tuple(stage_b["runnable_coarse_groups"]) == ("P1", "P2", "P3", "P4")
    assert stage_b["p0_role"] == (
        "stage_a_all_bn_reference_not_a_stage_b_candidate"
    )
    assert stage_b["unavailable_group"]["P5"]["blocked_reason"] == (
        "current_final_head_has_no_BatchNorm2d"
    )
    assert stage_b["threshold_overrides"][
        "nonclean_36_macro_delta_iou_mean_minimum"
    ] == 0.003
    assert stage_b["threshold_overrides"][
        "overall_39_macro_delta_iou_mean_minimum"
    ] == 0.002
    assert stage_b["fine_20_group_role"] == "alignment_diagnostic_only"


@pytest.mark.parametrize(
    ("mutation",),
    [
        ("unknown",),
        ("missing",),
        ("test_payload",),
        ("validation_payload",),
        ("stage2",),
        ("threshold",),
        ("candidate_order",),
    ],
)
def test_any_unknown_missing_or_value_drift_fails_closed(mutation: str) -> None:
    value = copy.deepcopy(_mapping())
    if mutation == "unknown":
        value["payload_firewall"]["candidate_phase"]["unsafe"] = True
    elif mutation == "missing":
        del value["scope"]["no_validation_split"]
    elif mutation == "test_payload":
        value["scope"]["use_test_payload"] = True
    elif mutation == "validation_payload":
        value["scope"]["use_validation_payload"] = True
    elif mutation == "stage2":
        value["status_contract"]["stage2_authorized"] = True
    elif mutation == "threshold":
        value["stage_a_science_gate"]["across_R0_R1_R2"][
            "nonclean_36_macro_delta_iou_mean_minimum"
        ] = 0.001
    else:
        value["candidates"].reverse()
    with pytest.raises(D0V3FormalContractError, match="drifted"):
        parse_d0_v3_formal_contract(value)


def test_contract_is_deeply_immutable() -> None:
    contract = load_d0_v3_formal_contract(CONFIG)
    with pytest.raises(FrozenInstanceError):
        contract.output_root = "unsafe"  # type: ignore[misc]
    with pytest.raises(TypeError):
        contract.raw["scope"]["stage2_authorized_default"] = True  # type: ignore[index]


def test_cpu_only_load_does_not_create_registered_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "results").exists()
    contract = load_d0_v3_formal_contract(CONFIG)
    assert contract.output_root.startswith("results/")
    assert not (tmp_path / "results").exists()


def test_symlink_config_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "formal.yaml"
    link.symlink_to(CONFIG)
    with pytest.raises(D0V3FormalContractError, match="securely read"):
        load_d0_v3_formal_contract(link)
