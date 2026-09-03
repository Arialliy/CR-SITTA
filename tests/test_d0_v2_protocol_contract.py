from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path

import pytest
import torch
import yaml

from analysis.d0_v2_protocol_contract import (
    CONDITIONS,
    D0V2ProtocolContractError,
    load_d0_v2_protocol_contract,
    parse_d0_v2_protocol_contract,
    verify_sealed_v1_inputs,
)
from analysis.d0_v2_task_loss import D0V2TaskLossConfig
from tta.d0_v2_parameter_groups import ELIGIBLE_GROUP_IDS, FINAL_HEAD_REASON


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tent_failure_diagnostics_v2_independent_candidates.yaml"
CONFIG_SHA256 = "a839af3599111854196548cfa0b05ab5b3b4ba3f60ab0829ee5a827ce871728d"


def _mapping() -> dict:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _set_nested(value: dict, path: tuple[object, ...], replacement: object) -> None:
    cursor: object = value
    for key in path[:-1]:
        if isinstance(key, int):
            assert isinstance(cursor, list)
            cursor = cursor[key]
        else:
            assert isinstance(cursor, dict)
            cursor = cursor[key]
    final = path[-1]
    if isinstance(final, int):
        assert isinstance(cursor, list)
        cursor[final] = replacement
    else:
        assert isinstance(cursor, dict)
        cursor[final] = replacement


def test_canonical_config_is_strict_train_pilot_only_and_non_authorizing() -> None:
    contract = load_d0_v2_protocol_contract(CONFIG)

    assert contract.config_file_sha256 == CONFIG_SHA256
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256
    assert contract.conditions == CONDITIONS
    assert len(contract.conditions) == 13
    assert len(contract.candidates) == 10
    assert contract.candidates[0].slug == "Adam_lr_1em5"
    assert contract.candidates[-1].slug == "SGD_lr_1em3"
    assert contract.eligible_group_ids == ELIGIBLE_GROUP_IDS
    assert len(contract.eligible_group_ids) == 20
    assert contract.scientific_gate_status == "unresolved"
    assert contract.stage2_authorized is False
    assert contract.formal_p3_complete is False

    raw = contract.as_frozen_mapping()
    assert raw["scope"]["no_validation_split"] is True
    assert raw["scope"]["use_validation"] is False
    assert raw["scope"]["use_test_images"] is False
    assert raw["scope"]["use_test_labels"] is False
    assert raw["scope"]["method_label_accesses"] == 0
    assert raw["cache"]["split_name"] == "train"
    assert raw["cache"]["split_role"] == "frozen_pilot64"
    assert raw["cache"]["subset_size_per_dataset"] == 64
    assert raw["cache"]["target_load_timing"] == (
        "after_all_10_label_free_candidates_in_cell"
    )
    assert raw["independent_candidate_execution"]["gradient_reuse"] == (
        "forbidden"
    )
    assert raw["independent_candidate_execution"][
        "shared_gradient_buffers_across_candidates"
    ] is False
    assert raw["implementation_status"]["formal_gpu_runner"] == "not_implemented"
    assert raw["fine_parameter_groups"]["structurally_ineligible"] == {
        "final_head": FINAL_HEAD_REASON
    }
    assert raw["fine_parameter_groups"][
        "single_group_update_runner_implemented"
    ] is False
    smoke = raw["engineering_smoke"]
    assert smoke["sample"] == {
        "dataset": "NUAA-SIRST",
        "condition": "clean_S0",
        "image_index": 0,
        "image_id": "Misc_421",
        "original_size": (225, 334),
        "split_name": "train",
        "split_role": "frozen_pilot64",
    }
    assert smoke["execution"]["fresh_process_count"] == 3
    assert smoke["execution"]["candidate_count_per_process"] == 10
    assert smoke["execution"]["gradient_reuse"] == "forbidden"
    assert smoke["execution"]["method_label_accesses"] == 0
    assert smoke["execution"]["target_payload_deserialized"] is False
    assert smoke["execution"]["test_images_opened"] == 0
    assert smoke["execution"]["test_labels_opened"] == 0
    assert smoke["runtime_binding"]["method_facing_input_seal"] is True
    assert smoke["runtime_binding"][
        "frozen_v2_full_payload_seal_reused"
    ] is False
    assert smoke["runtime_binding"]["target_payload_bytes_opened"] == 0
    assert smoke["runtime_binding"]["target_payload_deserialized"] is False
    assert smoke["runtime_binding"]["test_split_files_opened"] == 0
    assert "analysis/analyze_tent_optimizer_geometry.py" in smoke[
        "runtime_binding"
    ]["critical_code_paths"]
    assert smoke["paper_result"] is False
    assert smoke["formal_p3_complete"] is False
    assert smoke["stage2_authorized"] is False
    assert smoke["cross_process_gate"]["frozen_source_state_components"] == (
        "model_runtime_topology_gradients_extras"
    )
    assert smoke["cross_process_gate"][
        "candidate_optimizer_excluded_from_shared_hash"
    ] is True
    assert smoke["cross_process_gate"][
        "candidate_optimizer_exact_reset_verified_per_candidate"
    ] is True
    assert raw["implementation_status"]["gpu_engineering_smoke_runner"] == (
        "implemented_not_executed"
    )
    assert raw["implementation_status"]["fresh_process_smoke_aggregate"] == (
        "implemented_not_executed"
    )
    assert raw["output"]["registration_status"] == "path_only_not_materialized"
    assert raw["output"]["engineering_smoke"] == "engineering_smoke"
    assert raw["output"]["create_on_contract_load"] is False
    assert raw["output"]["formal_result_publication_authorized"] is False


def test_task_loss_and_diagnostic_extensions_are_exactly_bound() -> None:
    contract = load_d0_v2_protocol_contract(CONFIG)
    assert contract.task_loss == D0V2TaskLossConfig(
        lambda_bce=1.0,
        lambda_soft_iou=1.0,
        eps=1.0e-6,
    )
    diagnostics = contract.raw["diagnostics"]
    assert diagnostics["prediction_threshold"] == 0.5
    assert diagnostics["threshold_rule"] == (
        "strict_probability_greater_than_0_5"
    )
    assert diagnostics["four_level_noop_order"] == (
        "parameter_numeric_noop",
        "functional_logit_noop",
        "threshold_noop",
        "metric_noop",
    )
    assert diagnostics["required_extensions"] == (
        "p95_abs_delta_logit",
        "p95_abs_delta_probability",
        "near_threshold_pixel_count_by_margin_bin",
        "foreground_probability_mass_sum_pre_post_delta",
        "largest_component_area_pre_post_delta",
    )
    assert len(diagnostics["margin_bins"]) == 4


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("scope", "scientific_gate_status"), "passed"),
        (("scope", "stage2_authorized"), True),
        (("scope", "use_validation"), True),
        (("scope", "use_test_images"), True),
        (("scope", "use_test_labels"), True),
        (("cache", "split_name"), "test"),
        (("cache", "subset_size_per_dataset"), 63),
        (("conditions", 0), "test_clean"),
        (("independent_candidate_execution", "gradient_reuse"), "allowed"),
        (
            (
                "independent_candidate_execution",
                "shared_autograd_graph_across_candidates",
            ),
            True,
        ),
        (
            ("independent_candidate_execution", "candidates", 0, "learning_rate"),
            0,
        ),
        (("diagnostics", "prediction_threshold"), 0.49),
        (("outer_oracle_task_loss", "used_by_adaptation"), True),
        (("fine_parameter_groups", "eligible", 0), "final_head"),
        (
            ("fine_parameter_groups", "single_group_update_runner_implemented"),
            True,
        ),
        (("engineering_smoke", "sample", "condition"), "gaussian_noise_S1"),
        (("engineering_smoke", "sample", "image_id"), "wrong"),
        (("engineering_smoke", "execution", "fresh_process_count"), 2),
        (("engineering_smoke", "execution", "gradient_reuse"), "allowed"),
        (("engineering_smoke", "execution", "method_label_accesses"), 1),
        (("engineering_smoke", "execution", "test_images_opened"), 1),
        (
            (
                "engineering_smoke",
                "runtime_binding",
                "target_payload_bytes_opened",
            ),
            1,
        ),
        (
            (
                "engineering_smoke",
                "runtime_binding",
                "frozen_v2_full_payload_seal_reused",
            ),
            True,
        ),
        (
            (
                "engineering_smoke",
                "runtime_binding",
                "critical_code_paths",
                6,
            ),
            "unsafe.py",
        ),
        (("engineering_smoke", "paper_result"), True),
        (("engineering_smoke", "formal_p3_complete"), True),
        (("engineering_smoke", "stage2_authorized"), True),
        (("implementation_status", "formal_p3_complete"), True),
        (("output", "create_on_contract_load"), True),
        (("sealed_v1_inputs", "read_only_reference"), False),
    ],
)
def test_any_scope_execution_or_authorization_drift_fails_closed(
    path: tuple[object, ...], replacement: object
) -> None:
    value = copy.deepcopy(_mapping())
    _set_nested(value, path, replacement)
    with pytest.raises(D0V2ProtocolContractError, match="drifted"):
        parse_d0_v2_protocol_contract(value)


def test_missing_unknown_and_reordered_values_fail_closed() -> None:
    missing = _mapping()
    del missing["scope"]["no_validation_split"]
    with pytest.raises(D0V2ProtocolContractError, match="missing"):
        parse_d0_v2_protocol_contract(missing)

    unknown = _mapping()
    unknown["scope"]["validation_split"] = "test"
    with pytest.raises(D0V2ProtocolContractError, match="unknown"):
        parse_d0_v2_protocol_contract(unknown)

    reordered = _mapping()
    reordered["conditions"][0], reordered["conditions"][1] = (
        reordered["conditions"][1],
        reordered["conditions"][0],
    )
    with pytest.raises(D0V2ProtocolContractError, match="drifted"):
        parse_d0_v2_protocol_contract(reordered)


def test_contract_is_deeply_immutable() -> None:
    contract = load_d0_v2_protocol_contract(CONFIG)
    with pytest.raises(FrozenInstanceError):
        contract.output_root = "unsafe"  # type: ignore[misc]
    with pytest.raises(TypeError):
        contract.raw["scope"]["stage2_authorized"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        contract.raw["datasets"]["unsafe"] = {}  # type: ignore[index]


def test_loading_does_not_materialize_registered_result_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda_initialized_before = torch.cuda.is_initialized()
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "results").exists()
    contract = load_d0_v2_protocol_contract(CONFIG)
    assert contract.output_root.startswith("results/")
    assert not (tmp_path / "results").exists()
    assert torch.cuda.is_initialized() is cuda_initialized_before


def test_symlink_config_is_rejected(tmp_path: Path) -> None:
    symlink = tmp_path / "unsafe.yaml"
    symlink.symlink_to(CONFIG)
    with pytest.raises(D0V2ProtocolContractError, match="securely read"):
        load_d0_v2_protocol_contract(symlink)


def test_all_referenced_d0_v1_files_remain_exactly_sealed() -> None:
    contract = load_d0_v2_protocol_contract(CONFIG)
    verified = verify_sealed_v1_inputs(contract, repository_root=ROOT)
    assert len(verified) == 5
    assert verified[0] == (
        "configs/tent_failure_diagnostics_v1.yaml",
        "6f6cabbc9d4b43b54c3d2dc97539eb368daabe934aee0c7f0ef9a4f5735ef815",
    )
    assert verified[-1] == (
        "analysis/d0_equivalence_repro_contract.py",
        "18560bec3467e0a286a2ae33dfddf209ef484d1eb9dbc0a059c280fc62e42615",
    )


def test_seal_verifier_fails_on_wrong_repository_root(tmp_path: Path) -> None:
    contract = load_d0_v2_protocol_contract(CONFIG)
    with pytest.raises(D0V2ProtocolContractError, match="sealed D0-v1"):
        verify_sealed_v1_inputs(contract, repository_root=tmp_path)
