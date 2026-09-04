from __future__ import annotations

import ast
import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

import analysis.p3_stage_b1_contract_v2 as contract_module
from analysis.p3_stage_b1_contract_v2 import (
    CONDITIONS,
    CONFIG_CANONICAL_MAPPING_SHA256,
    CONFIG_FILE_SHA256,
    DATASETS,
    DIRECT_VJP_ORDER,
    PARAMETER_GROUP_SCALAR_COUNTS,
    PILOT64_ORDERED_ID_SHA256,
    PRIMARY_SCIENTIFIC_BASIS_ORDER,
    P3StageB1ContractV2Error,
    REPLICATE_IDS,
    V1_PREFLIGHT_FAILURE_RELATIVE_PATH,
    V1_PREFLIGHT_FAILURE_SHA256,
    load_p3_stage_b1_contract_v2,
    parse_p3_stage_b1_contract_v2,
    verify_all_frozen_file_bindings,
    verify_live_parent_aggregate,
    verify_v1_preflight_failure_binding,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3_stage_b_gradient_decomposition_v2.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _mapping() -> dict:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_exact_file_and_mapping_hashes_are_independently_pinned() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)

    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_FILE_SHA256
    assert contract.config_file_sha256 == CONFIG_FILE_SHA256
    assert contract.canonical_mapping_sha256() == CONFIG_CANONICAL_MAPPING_SHA256
    assert contract.protocol_id == "cr-sitta-p3-stage-b1-gradient-decomposition-v2"
    assert contract.schema_version == 2


def test_v2_is_a_numerical_estimator_only_version_with_frozen_v1_incident() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    versioning = contract.raw["versioning"]
    assert versioning["predecessor_protocol_id"] == (
        "cr-sitta-p3-stage-b1-gradient-decomposition-v1"
    )
    assert versioning["change_class"] == "versioned_numerical_estimator_only"
    assert versioning["scientific_predicates_changed"] is False
    assert versioning["scientific_thresholds_changed"] is False
    assert versioning["dataset_or_family_coverage_changed"] is False
    assert versioning["parent_numeric_tolerances_changed"] is False
    assert versioning["v1_retry_forbidden"] is True

    incident = contract.raw["frozen_v1_preflight_failure_binding"]
    assert incident["path"] == V1_PREFLIGHT_FAILURE_RELATIVE_PATH
    assert incident["sha256"] == V1_PREFLIGHT_FAILURE_SHA256
    assert incident["required_status"] == "aborted_numeric_invariant"
    assert incident["required_scientific_conclusion"] == "not_evaluated"
    assert incident["read_only"] is True


def test_v1_scientific_predicates_coverage_and_parent_tolerances_are_unchanged() -> None:
    v1_path = ROOT / "configs/p3_stage_b_gradient_decomposition_v1.yaml"
    v1 = yaml.safe_load(v1_path.read_text(encoding="utf-8"))
    v2 = _mapping()
    assert v2["mechanism_evidence_flags"] == v1["mechanism_evidence_flags"]
    assert v2["datasets"] == v1["datasets"]
    assert v2["conditions"] == v1["conditions"]
    assert v2["parameter_spaces"] == v1["parameter_spaces"]
    for key in (
        "compare_all_parent_candidate_slices",
        "compared_slice_count",
        "reference_slice_index",
        "compare_each_slice_to_recomputed_full",
        "max_abs_tolerance",
        "relative_l2_tolerance",
        "relative_l2_denominator",
        "any_nonfinite_action",
        "tolerance_failure_action",
    ):
        assert v2["entropy_gradient_consistency"][key] == (
            v1["entropy_gradient_consistency"][key]
        )


def test_scope_is_exact_train_pilot64_R0_and_non_authorizing() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    scope = contract.raw["scope"]

    assert contract.datasets == DATASETS
    assert contract.conditions == CONDITIONS
    assert contract.replicate_ids == REPLICATE_IDS == ("R0",)
    assert scope["split_name"] == "train"
    assert scope["split_role"] == "frozen_pilot64"
    assert scope["dataset_count"] == 3
    assert scope["condition_count_per_dataset"] == 13
    assert scope["total_dataset_condition_cells"] == 39
    assert scope["image_count_per_condition"] == 64
    assert scope["paper_result"] is False
    assert scope["paper_test_result"] is False
    assert scope["performance_claim"] is False
    assert scope["candidate_selection_performed"] is False
    assert contract.paper_result is False
    assert scope["stage_b3_authorized"] is False
    assert contract.stage_b3_authorized is False

    assert scope["use_validation_payload"] is False
    assert scope["use_test_payload"] is False
    assert scope["optimizer_construction"] == "forbidden"
    assert scope["optimizer_step"] == "forbidden"
    assert scope["model_weight_update"] == "forbidden"
    assert scope["checkpoint_write"] == "forbidden"
    assert tuple(scope["forbidden_payload_roles"]) == (
        "validation_image",
        "validation_target",
        "test_image",
        "test_target",
        "test_prediction",
    )
    assert scope["target_visibility"] == "outer_analyzer_only_never_adaptation"


def test_parent_decision_and_required_file_bindings_are_frozen() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    decision = contract.raw["required_parent_decision"]
    assert dict(decision) == {
        "protocol_status": "passed",
        "formal_stage_a_protocol_complete": True,
        "scientific_status": "scientific_no_eligible",
        "eligible_candidates": (),
        "required_followup_replicates": (),
        "stage2_allowed": False,
    }

    bindings = contract.raw["frozen_parent_bindings"]
    assert tuple(bindings) == (
        "d0_formal_stage_a_config",
        "d0_formal_stage_a_aggregate_complete",
        "d0_formal_stage_a_aggregate_manifest",
        "d0_formal_stage_a_science_decision",
        "parameter_group_config",
        "cache_protocol",
    )
    assert bindings["d0_formal_stage_a_config"]["sha256"] == (
        "f0ed056520f840d1432b90b6426f55803afa1867b98108712dc242371dd1c05a"
    )
    assert bindings["parameter_group_config"]["sha256"] == (
        "3cfcdb768ca7adfbb85446fd14a9d02a89ba6fecd43ea25843783e571cb7b637"
    )
    assert bindings["cache_protocol"]["sha256"] == (
        "e225311cce252125eaf3a2b47eeeea4cde60d0363c8c71f48494d70e41c2aa3e"
    )


def test_ordered_critical_code_bundle_must_be_hashed_at_runtime() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    implementation = contract.raw["implementation"]
    assert tuple(implementation["critical_code_paths"]) == (
        "analysis/foreground_background_gradient_decomposition_v2.py",
        "analysis/foreground_background_gradient_decomposition_v1.py",
        "analysis/p3_stage_b1_contract_v2.py",
        "analysis/p3_stage_b1_outer_cell_shard_v2.py",
        "analysis/p3_stage_b1_outer_cell_shard.py",
        "analysis/p3_stage_b1_aggregate_v2.py",
        "analysis/p3_stage_b1_aggregate.py",
        "scripts/run_p3_stage_b_gradient_decomposition_v2.py",
        "tta/parameter_groups.py",
        "tta/d0_v2_parameter_groups.py",
        "analysis/analyze_entropy_task_alignment.py",
        "analysis/d0_v2_task_loss.py",
        "tta/d0_v3_outer_source_gradient.py",
        "analysis/d0_v3_outer_analyzer.py",
        "analysis/d0_v3_label_free_shard.py",
        "analysis/d0_v3_outer_cell_shard.py",
        "analysis/d0_v3_phase_receipt.py",
        "analysis/source_train_provenance.py",
        "materialize_binary_tent_ss_calibration_cache_v2.py",
        "tta/model_adapter.py",
        "tta/state_manager.py",
        "tta/binary_tent.py",
        "model/MSHNet_NSFPN.py",
        "model/NS_FPN.py",
        "test_source.py",
        "tta/d0_secure_io.py",
        "tta/d0_v3_atomic_shard.py",
    )
    assert implementation["path_order_is_frozen"] is True
    runtime = implementation["runtime_manifest"]
    assert runtime["hash_algorithm"] == "sha256"
    assert runtime["hash_every_critical_code_path"] is True
    assert runtime["require_regular_non_symlink_files"] is True
    assert runtime["reject_missing_path"] is True
    assert runtime["reject_extra_path"] is True
    assert runtime["reject_reordered_path"] is True
    assert runtime["verify_live_bytes_before_any_outer_target_load"] is True
    assert runtime["store_per_file_sha256"] is True
    assert runtime["store_canonical_bundle_sha256"] is True


def test_dataset_checkpoint_split_and_complete_ordered_id_hashes_are_exact() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    expected_id_hashes = dict(PILOT64_ORDERED_ID_SHA256)
    expected_checkpoint_hashes = {
        "IRSTD-1K": "ee8d5c3af67d7c93ff6be25b169bd8fd59afcd776afd173b4e230ad8dc8aa8b2",
        "NUAA-SIRST": "23feea50a847cdfc9874fe67d338a3884d9afa75d7a27b9b71a62ff2d00379cf",
        "NUDT-SIRST": "a67d3a7e2597d201e1fecf32b7f218aefa8274ad0443d2726197d84f44d1343e",
    }
    expected_split_hashes = {
        "IRSTD-1K": "689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744",
        "NUAA-SIRST": "324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4",
        "NUDT-SIRST": "e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3",
    }

    assert contract.pilot64_ordered_id_sha256 == PILOT64_ORDERED_ID_SHA256
    for dataset in DATASETS:
        entry = contract.raw["datasets"][dataset]
        assert entry["checkpoint_role"] == "best_miou"
        assert entry["checkpoint_sha256"] == expected_checkpoint_hashes[dataset]
        assert entry["train_split_sha256"] == expected_split_hashes[dataset]
        assert entry["ordered_pilot64_image_ids_sha256"] == expected_id_hashes[dataset]
        assert entry["ordered_id_hash_source"] == (
            "all_13_parent_candidate_manifests_R0"
        )


def test_parent_array_layout_and_P0_through_P4_counts_are_exact() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    artifacts = contract.raw["parent_cell_artifacts"]
    layout = artifacts["parameter_layout"]
    assert layout["parameter_tensor_count"] == 106
    assert layout["scalar_parameter_count"] == 8736
    assert layout["layout_sha256"] == (
        "fbe39c8cee7b1e6fd46bfc7bc2eebd1aab3b9fcb4194d52a21ccf44d05984a4b"
    )
    assert tuple(artifacts["parent_entropy_gradients"]["shape"]) == (64, 10, 8736)
    assert artifacts["parent_entropy_gradients"]["candidate_axis_count"] == 10
    assert tuple(artifacts["parent_task_gradients"]["shape"]) == (64, 8736)
    assert artifacts["parent_task_gradients"]["source"] == (
        "frozen_parent_outer_shard"
    )
    assert artifacts["parent_task_gradients"]["recompute"] is False
    assert artifacts["parent_task_gradients"]["used_by_adaptation"] is False

    spaces = contract.raw["parameter_spaces"]
    assert contract.parameter_group_scalar_counts == PARAMETER_GROUP_SCALAR_COUNTS
    assert tuple(spaces["nested_order"]) == ("P0", "P1", "P2", "P3", "P4")
    assert tuple(
        (group, spaces["groups"][group]["scalar_parameter_count"])
        for group in spaces["nested_order"]
    ) == PARAMETER_GROUP_SCALAR_COUNTS
    assert spaces["one_full_P0_backward_then_layout_slice"] is True
    assert spaces["repeated_backward_per_parameter_group"] == "forbidden"


def test_five_direct_vjps_and_three_term_scientific_partition_are_frozen() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    partition = contract.raw["region_partition"]
    assert partition["foreground_predicate"] == "target>0"
    assert partition["background_predicate"] == "target==0"
    assert partition["probability_definition"] == "sigmoid(source_logits)"
    assert partition["probability_partition_is_detached"] is True
    assert partition["subthreshold_predicate"] == "sigmoid(source_logits)<=0.5"
    assert partition["suprathreshold_predicate"] == "sigmoid(source_logits)>0.5"
    assert partition["threshold"] == 0.5
    assert dict(partition["masks"]) == {
        "background": "target==0",
        "foreground_subthreshold": "(target>0)&(sigmoid(source_logits)<=0.5)",
        "foreground_suprathreshold": "(target>0)&(sigmoid(source_logits)>0.5)",
    }
    assert partition["disjoint_masks_required"] is True
    assert partition["exhaustive_masks_required"] is True

    basis = contract.raw["gradient_basis"]
    assert contract.direct_vjp_order == DIRECT_VJP_ORDER
    assert tuple(basis["direct_vjp_order"]) == DIRECT_VJP_ORDER == (
        "full",
        "foreground_total",
        "foreground_subthreshold",
        "raw_foreground_suprathreshold",
        "raw_background",
    )
    assert basis["direct_vjp_count"] == 5
    assert contract.primary_scientific_basis_order == (
        PRIMARY_SCIENTIFIC_BASIS_ORDER
    )
    assert tuple(basis["primary_scientific_basis_order"]) == (
        PRIMARY_SCIENTIFIC_BASIS_ORDER
    )
    assert basis["primary_scientific_basis_term_count"] == 3
    assert basis["denominator_for_every_loss"] == "total_image_pixel_count"
    assert basis["masked_loss_definition"] == (
        "sum(mask*entropy)/total_image_pixel_count"
    )
    stored = basis["stored_direct_vjps"]
    assert stored["filename"] == "cumulative_and_raw_entropy_vjps.npy"
    assert tuple(stored["shape"]) == (64, 5, 8736)
    assert stored["storage_dtype"] == "float32"
    assert stored["axis_order_matches_direct_vjp_order"] is True
    assert stored["store_before_primary_telescoping"] is True

    derivation = basis["primary_scientific_derivation"]
    assert derivation["device"] == "cpu"
    assert derivation["dtype"] == "float64"
    assert derivation["foreground_subthreshold"] == (
        "direct_vjp.foreground_subthreshold"
    )
    assert derivation["foreground_suprathreshold"] == (
        "direct_vjp.foreground_total-direct_vjp.foreground_subthreshold"
    )
    assert derivation["background"] == (
        "direct_vjp.full-direct_vjp.foreground_total"
    )
    assert derivation["data_dependent_residual_allocation"] == "forbidden"
    assert derivation["raw_audit_vectors_used_by_science"] is False

    audit = basis["raw_component_audit"]
    assert audit["role"] == "numeric_diagnostic_only"
    assert tuple(audit["pairs"]) == (
        "foreground_suprathreshold",
        "background",
    )
    assert audit["max_abs_tolerance"] == 1.0e-7
    assert audit["relative_l2_tolerance"] == 1.0e-4
    assert audit["failure_action"] == (
        "record_only_never_protocol_or_science_gate"
    )
    assert audit["used_by_mechanism_flags"] is False
    assert audit["used_by_selection_or_ranking"] is False
    assert basis["task_gradient_source"] == "parent_outer_supervised_gradients.npy"
    assert basis["task_gradient_recomputed"] is False


def test_consistency_float64_null_and_vjp_device_contract_are_exact() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    consistency = contract.raw["entropy_gradient_consistency"]
    assert consistency["compare_all_parent_candidate_slices"] is True
    assert consistency["compared_slice_count"] == 10
    assert consistency["reference_slice_index"] == 0
    assert consistency["recomputed_full_source"] == "direct_vjp.full"
    assert consistency["compare_each_slice_to_recomputed_full"] is True
    assert consistency["compare_primary_telescoping_sum_to_recomputed_full"] is True
    assert consistency["max_abs_tolerance"] == 1.0e-7
    assert consistency["relative_l2_tolerance"] == 1.0e-4
    assert consistency["parent_pass_rule"] == (
        "max_abs_and_relative_l2_must_both_pass_for_every_slice"
    )
    assert consistency["primary_telescoping_failure_action"] == "fail_closed"
    assert consistency["any_nonfinite_action"] == "fail_closed"
    assert consistency["tolerance_failure_action"] == "fail_closed"

    metrics = contract.raw["metric_protocol"]
    assert metrics["metric_accumulation_device"] == "cpu"
    assert metrics["accumulation_dtype"] == "float64"
    assert metrics["input_storage_cast_before_metric"] == "float32_to_float64"
    assert metrics["primary_metric_vector_source"] == (
        "cpu_float64_telescoping_from_stored_cumulative_vjps"
    )
    assert metrics["raw_direct_audit_vector_use"] == "numeric_diagnostic_only"
    assert metrics["empty_region_vector_metrics"] is None
    assert metrics["zero_norm_cosine"] is None
    assert metrics["zero_denominator_ratio"] is None
    vjp = metrics["region_vjp_execution"]
    assert vjp["formal_device"] == "one_explicitly_selected_visible_cuda_device"
    assert vjp["cpu_role"] == "smoke_test_only"
    assert vjp["one_image_at_a_time"] is True
    assert vjp["optimizer_construction"] == "forbidden"
    assert vjp["optimizer_step"] == "forbidden"
    assert vjp["model_weight_update"] == "forbidden"


def test_mechanism_flags_pin_additive_fields_joint_gate_support_and_three_states() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    flags = contract.raw["mechanism_evidence_flags"]
    assert flags["evidence_scope"]["parameter_space"] == "P0"
    assert flags["evidence_scope"]["conditions"] == "nonclean_36_cells"
    assert flags["evidence_scope"]["episode_filter"] == "target_present"
    assert flags["aggregation"]["only_authorized_statistic"] == (
        "equal_cell_macro_of_equal_valid_target_present_episode_means"
    )
    assert tuple(flags["aggregation"]["projection_reduction_order"]) == (
        "restrict_to_same_joint_valid_episode_support_when_predicate_is_joint",
        "arithmetic_mean_each_projection_within_cell",
        "equal_cell_macro_each_projection_within_requested_stratum",
        "apply_absolute_value_only_to_completed_stratum_mean",
    )
    assert flags["aggregation"]["absolute_projection_semantics"] == (
        "abs(mean_projection)_never_mean(abs(projection))"
    )
    assert "same aggregated stratum" in flags["aggregation"][
        "joint_predicate_semantics"
    ]
    assert flags["aggregation"]["median_or_alternative_statistic"] == "forbidden"

    support = flags["support_accounting"]
    assert support["finite_non_null_definition"] == (
        "value_is_not_null_and_math_isfinite"
    )
    assert support["single_quantity_episode_support"] == (
        "requested_report_field_is_finite_non_null"
    )
    assert "same episode" in support["joint_quantity_episode_support"]
    assert support["cell_is_estimable"] == "valid_non_null_episode_count>=1"
    assert support["joint_cell_is_estimable"] == (
        "joint_valid_non_null_episode_count>=1"
    )
    assert support["stratum_is_estimable"] == (
        "valid_non_null_cell_count==required_cell_count"
    )
    assert support["joint_stratum_is_estimable"] == (
        "joint_valid_non_null_cell_count==required_cell_count"
    )
    assert tuple(support["required_count_fields"]["per_cell"]) == (
        "target_present_episode_count",
        "valid_non_null_episode_count",
        "null_or_nonfinite_episode_count",
    )
    assert tuple(support["required_count_fields"]["per_joint_cell"]) == (
        "target_present_episode_count",
        "joint_valid_non_null_episode_count",
        "joint_null_or_nonfinite_episode_count",
    )
    assert tuple(support["required_count_fields"]["per_stratum"]) == (
        "required_cell_count",
        "valid_non_null_cell_count",
        "valid_non_null_episode_count",
    )
    assert tuple(support["required_count_fields"]["per_joint_stratum"]) == (
        "required_cell_count",
        "joint_valid_non_null_cell_count",
        "joint_valid_non_null_episode_count",
    )
    assert flags["coverage"]["minimum_supporting_datasets"] == 2
    assert flags["coverage"]["minimum_supporting_corruption_families"] == 3
    assert flags["comparison_tolerance"] == 1.0e-12
    assert tuple(flags["allowed_status_values"]) == (
        "supported",
        "not_supported",
        "not_estimable",
    )
    assert tuple(flags["P0_flags"]) == (
        "background_norm_dominance",
        "background_cancellation",
        "subthreshold_erasure",
    )
    p0 = flags["P0_flags"]
    norm = p0["background_norm_dominance"]
    assert norm["quantity"] == "background_to_foreground_additive_l2_norm_ratio"
    assert norm["report_field_path"] == (
        "per_group.P0.cross_region."
        "background_to_foreground_additive_norm_ratio.value"
    )
    assert norm["gradient_semantics"] == "additive_full_image_denominator"

    cancellation = p0["background_cancellation"]
    assert cancellation["quantity"] == "joint_additive_task_projection_cancellation"
    assert dict(cancellation["report_field_paths"]) == {
        "foreground_task_projection": (
            "per_group.P0.additive_entropy_task_alignment."
            "foreground_entropy_add.task_projection"
        ),
        "background_task_projection": (
            "per_group.P0.additive_entropy_task_alignment."
            "background_entropy_add.task_projection"
        ),
        "full_task_projection": (
            "per_group.P0.additive_entropy_task_alignment."
            "full_entropy_mean.task_projection"
        ),
    }
    assert cancellation["gradient_semantics"] == "additive_full_image_denominator"
    assert cancellation["support_mode"] == "same_episode_joint_finite_non_null"
    assert cancellation["projection_aggregation"] == (
        "mean_projection_then_absolute_value"
    )
    assert cancellation["absolute_projection_semantics"] == (
        "abs(mean_projection)_never_mean(abs(projection))"
    )
    predicate = cancellation["joint_predicate"]
    assert predicate["connective"] == "all"
    assert predicate["evaluate_on_same_stratum"] is True
    assert predicate["use_same_joint_valid_episode_support"] is True
    assert dict(predicate["conditions"]) == {
        "foreground_projection_positive": {
            "left": "mean_foreground_task_projection",
            "operator": ">",
            "right": "+comparison_tolerance",
        },
        "background_projection_negative": {
            "left": "mean_background_task_projection",
            "operator": "<",
            "right": "-comparison_tolerance",
        },
        "full_projection_magnitude_reduced": {
            "left": "abs(mean_full_task_projection)+comparison_tolerance",
            "operator": "<",
            "right": "abs(mean_foreground_task_projection)",
        },
    }
    assert cancellation["overall_rule"] == (
        "evaluate_joint_predicate_on_overall_stratum"
    )
    assert cancellation["supporting_stratum_rule"] == (
        "evaluate_same_joint_predicate_on_each_dataset_or_corruption_family_stratum"
    )
    auxiliary = cancellation["auxiliary_foreground_background_cosine"]
    assert auxiliary["report_field_path"] == (
        "per_group.P0.cross_region.foreground_background_additive_alignment."
        "foreground_background_cosine"
    )
    assert auxiliary["role"] == "descriptive_only_not_used_by_primary_flag_status"

    erasure = p0["subthreshold_erasure"]
    assert erasure["quantity"] == (
        "additive_foreground_subthreshold_entropy_task_cosine"
    )
    assert erasure["report_field_path"] == (
        "per_group.P0.additive_entropy_task_alignment."
        "foreground_subthreshold_entropy_add.entropy_task_cosine"
    )
    assert erasure["gradient_semantics"] == "additive_full_image_denominator"

    hints = flags["small_group_alignment_hints"]
    assert tuple(hints["parameter_spaces"]) == ("P1", "P2", "P3", "P4")
    assert tuple(hints["components"]) == ("full", "foreground_total")
    assert dict(hints["report_field_paths"]) == {
        "full": (
            "per_group.{parameter_space}.additive_entropy_task_alignment."
            "full_entropy_mean.entropy_task_cosine"
        ),
        "foreground_total": (
            "per_group.{parameter_space}.additive_entropy_task_alignment."
            "foreground_entropy_add.entropy_task_cosine"
        ),
    }
    assert hints["quantity"] == "additive_entropy_task_cosine"
    assert hints["gradient_semantics"] == "additive_full_image_denominator"
    assert hints["minimum_cosine"] == 0.05
    assert hints["require_dataset_and_family_coverage"] is True
    assert hints["role"] == "mechanism_hint_only_not_candidate_selection"


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown",
        "missing",
        "condition_order",
        "test_payload",
        "optimizer",
        "direct_vjp_order",
        "raw_audit_role",
        "incident_hash",
        "mask",
        "tolerance",
        "paper",
        "stage_b3",
        "aggregation",
    ),
)
def test_unknown_missing_reordered_or_value_drift_fails_closed(mutation: str) -> None:
    value = copy.deepcopy(_mapping())
    if mutation == "unknown":
        value["metric_protocol"]["post_hoc_metric"] = "forbidden"
    elif mutation == "missing":
        del value["scope"]["use_test_payload"]
    elif mutation == "condition_order":
        value["conditions"].reverse()
    elif mutation == "test_payload":
        value["scope"]["use_test_payload"] = True
    elif mutation == "optimizer":
        value["scope"]["optimizer_construction"] = "allowed"
    elif mutation == "direct_vjp_order":
        value["gradient_basis"]["direct_vjp_order"].reverse()
    elif mutation == "raw_audit_role":
        value["gradient_basis"]["raw_component_audit"][
            "used_by_mechanism_flags"
        ] = True
    elif mutation == "incident_hash":
        value["frozen_v1_preflight_failure_binding"]["sha256"] = "0" * 64
    elif mutation == "mask":
        value["region_partition"]["foreground_predicate"] = "target>0.5"
    elif mutation == "tolerance":
        value["entropy_gradient_consistency"]["max_abs_tolerance"] = 1.0e-6
    elif mutation == "paper":
        value["scope"]["paper_result"] = True
    elif mutation == "stage_b3":
        value["scope"]["stage_b3_authorized"] = True
    else:
        value["mechanism_evidence_flags"]["aggregation"][
            "only_authorized_statistic"
        ] = "median"
    with pytest.raises(P3StageB1ContractV2Error, match="drifted"):
        parse_p3_stage_b1_contract_v2(value)


def test_duplicate_yaml_key_is_rejected_before_byte_hash_check(tmp_path: Path) -> None:
    source = CONFIG.read_text(encoding="utf-8")
    duplicate = source.replace(
        "schema_version: 2\n",
        "schema_version: 2\nschema_version: 2\n",
        1,
    )
    path = tmp_path / "duplicate.yaml"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(P3StageB1ContractV2Error, match="duplicate YAML key"):
        load_p3_stage_b1_contract_v2(path)


def test_same_mapping_with_different_yaml_bytes_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "whitespace-drift.yaml"
    path.write_bytes(CONFIG.read_bytes() + b"\n")
    with pytest.raises(P3StageB1ContractV2Error, match="YAML bytes drifted"):
        load_p3_stage_b1_contract_v2(path)


def test_symlink_config_is_rejected(tmp_path: Path) -> None:
    link = tmp_path / "config.yaml"
    link.symlink_to(CONFIG)
    with pytest.raises(P3StageB1ContractV2Error, match="securely read"):
        load_p3_stage_b1_contract_v2(link)


def test_contract_is_deeply_immutable() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    with pytest.raises(FrozenInstanceError):
        contract.output_root = "unsafe"  # type: ignore[misc]
    with pytest.raises(TypeError):
        contract.raw["scope"]["paper_result"] = True  # type: ignore[index]


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires the ignored local Stage-B1 v1 preflight incident; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_live_v1_preflight_failure_is_hash_and_semantics_bound() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    binding = verify_v1_preflight_failure_binding(
        contract, repository_root=ROOT
    )
    assert binding.key == "v1_preflight_failure"
    assert binding.relative_path == V1_PREFLIGHT_FAILURE_RELATIVE_PATH
    assert binding.sha256 == V1_PREFLIGHT_FAILURE_SHA256

    incident_path = ROOT / binding.relative_path
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    assert incident["status"] == "aborted_numeric_invariant"
    assert incident["scientific_conclusion"] == "not_evaluated"
    assert incident["failure_scope"]["image_id"] == "XDU407"
    assert incident["failure_scope"]["max_abs_error"] == 2.64e-9
    assert incident["failure_scope"]["relative_l2_error"] == 1.0826e-4
    assert incident["formal_artifact_audit_before_incident_write"] == {
        "output_root_was_absent": True,
        "formal_shard_count": 0,
        "manifest_count": 0,
        "complete_receipt_count": 0,
        "formal_artifact_published": False,
        "partial_result_promoted": False,
    }
    boundary = incident["data_boundary"]
    assert boundary["validation_payload_accesses"] == 0
    assert boundary["test_payload_accesses"] == 0
    assert boundary["optimizer_constructed"] is False
    assert boundary["optimizer_steps"] == 0
    assert boundary["model_parameters_updated"] is False
    assert boundary["checkpoint_written"] is False

    v1_root = incident_path.parent
    assert sorted(
        str(path.relative_to(v1_root))
        for path in v1_root.rglob("*")
        if path.is_file()
    ) == ["PREFLIGHT_FAILURE.json"]
    assert not tuple(v1_root.rglob("manifest.json"))
    assert not tuple(v1_root.rglob("COMPLETE.json"))
    assert not (v1_root / "outer_phase" / "shards").exists()


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires the ignored local Stage-B1 v1 preflight incident; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_v1_preflight_root_rejects_any_later_artifact(
    tmp_path: Path,
) -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    incident_path = tmp_path / V1_PREFLIGHT_FAILURE_RELATIVE_PATH
    incident_path.parent.mkdir(parents=True)
    incident_path.write_bytes((ROOT / V1_PREFLIGHT_FAILURE_RELATIVE_PATH).read_bytes())
    (incident_path.parent / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        P3StageB1ContractV2Error,
        match="must contain only the frozen incident",
    ):
        verify_v1_preflight_failure_binding(contract, repository_root=tmp_path)


def test_load_is_cpu_metadata_only_and_creates_no_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(contract_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        node.names[0].name.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node.names
    }
    imported_roots.update(
        node.module.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert "torch" not in imported_roots
    assert "cuda" not in imported_roots

    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "results").exists()
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    assert contract.output_root == (
        "results/cr_sitta/p3_stage_b_gradient_decomposition_v2"
    )
    assert not (tmp_path / "results").exists()


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local Stage-A artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_live_parent_aggregate_and_all_frozen_file_hashes_verify() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    aggregate = verify_live_parent_aggregate(contract, repository_root=ROOT)
    assert tuple(binding.key for binding in aggregate) == (
        "d0_formal_stage_a_aggregate_complete",
        "d0_formal_stage_a_aggregate_manifest",
        "d0_formal_stage_a_science_decision",
    )
    all_bindings = verify_all_frozen_file_bindings(contract, repository_root=ROOT)
    assert len(all_bindings) == 8
    assert all_bindings[0].key == "stage_b_v5_document"
    assert all_bindings[1].key == "v1_preflight_failure"


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local Stage-A artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_ordered_pilot64_hash_is_recomputed_across_all_39_parent_cells() -> None:
    contract = load_p3_stage_b1_contract_v2(CONFIG)
    candidate_root = ROOT / (
        "results/cr_sitta/tent_failure_diagnostics_v3_formal_stage_a/"
        "candidate_phase/shards/R0"
    )
    outer_root = ROOT / (
        "results/cr_sitta/tent_failure_diagnostics_v3_formal_stage_a/"
        "outer_phase/shards/R0"
    )
    for dataset in DATASETS:
        expected = dict(PILOT64_ORDERED_ID_SHA256)[dataset]
        first_ids: tuple[str, ...] | None = None
        for condition in CONDITIONS:
            candidate = json.loads(
                (candidate_root / dataset / condition / "manifest.json").read_bytes()
            )
            outer = json.loads(
                (outer_root / dataset / condition / "manifest.json").read_bytes()
            )
            image_ids = tuple(candidate["ordered_image_ids"])
            assert len(image_ids) == 64
            assert len(set(image_ids)) == 64
            observed = hashlib.sha256(
                json.dumps(
                    list(image_ids), ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest()
            assert observed == expected
            assert candidate["ordered_image_ids_sha256"] == expected
            assert outer["ordered_image_ids_sha256"] == expected
            if first_ids is None:
                first_ids = image_ids
            assert image_ids == first_ids
        assert contract.raw["datasets"][dataset][
            "ordered_pilot64_image_ids_sha256"
        ] == expected
