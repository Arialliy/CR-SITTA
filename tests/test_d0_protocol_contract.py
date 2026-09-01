from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from analysis.d0_protocol_contract import (
    D0ProtocolContractError,
    load_d0_protocol_contract,
    parse_d0_protocol_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "tent_failure_diagnostics_v1.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _mapping() -> dict:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _set_nested(value: dict, path: tuple[str, ...], replacement: object) -> None:
    cursor = value
    for key in path[:-1]:
        child = cursor[key]
        assert isinstance(child, dict)
        cursor = child
    cursor[path[-1]] = replacement


def test_canonical_contract_freezes_actual_d0_execution_semantics() -> None:
    contract = load_d0_protocol_contract(CONFIG)

    assert contract.cache.subset_size_per_dataset == 64
    assert contract.cache.condition_count == 13
    assert contract.method.bn_protocol == "source_running_statistics"
    assert contract.method.adaptable_parameters == "all_batchnorm2d_affine"
    assert len(contract.method.candidates) == 10
    assert contract.method.candidates[0].slug == "Adam_lr_1em5"
    assert contract.method.candidates[-1].slug == "SGD_lr_1em3"
    determinism = contract.method.determinism
    assert determinism.policy == "strict_forwards_temporary_backward_disable"
    assert determinism.strict_forward.deterministic_algorithms_enabled is True
    assert determinism.strict_forward.warn_only is False
    assert determinism.strict_forward.cudnn_deterministic is True
    assert determinism.strict_forward.cudnn_benchmark is False
    assert determinism.temporary_backward_disable_scopes == (
        "entropy_backward",
        "supervised_task_backward",
    )
    assert determinism.restore_strict_policy_before_optimizer_step is True
    assert determinism.restore_strict_policy_before_post_forward is True
    assert determinism.restore_strict_policy_after_backward_exception is True
    assert contract.output.canonical_shards == "shards"
    assert contract.output.smoke_shards == "smoke"
    assert contract.output.equivalence == "equivalence"
    assert contract.output.aggregate == "aggregate"
    assert contract.comparison.archive_file_count == 113
    assert contract.comparison.archive_source_inventory_sha256 == (
        "bf8954dd2692f23e6b65761e4337f1daa77a64439386b05c6bb8df716800dfbf"
    )
    assert contract.comparison.stage1_records_sha256 == (
        "ee6a21b29ceee32fa6f9f70c6509436971133f81e815596fd148716807bbe3d3"
    )

    task = contract.source_task_gradient
    assert task.contract == "post_warm_SLSIoU_on_source_eval_graph"
    assert task.model_runtime == "source_eval"
    assert task.bn_protocol_during_diagnostic == "source_running_statistics"
    assert task.human_epoch == 1000
    assert task.epoch_index == 999
    assert task.warm_epochs == 5
    assert task.warm_flag is False
    assert task.with_shape is True
    assert task.full_source_training_runtime_exact is False

    floors = contract.evaluation.no_op_null_floors
    assert floors.policy == "exact_bitwise_zero"
    assert floors.scope == "D0_existing_result_diagnosis_only"
    assert floors.is_future_v3_activity_margin is False
    assert (floors.parameter, floors.logit, floors.probability) == (0.0, 0.0, 0.0)
    geometry = contract.evaluation.optimizer_geometry
    assert geometry.runtime_hard_gate_reference == (
        "pytorch_2_1_2_single_tensor_same_device_native_dtype_bit_exact_after_v1"
    )
    assert geometry.cross_backend_cpu_storage_replay_reference == (
        "pytorch_2_1_2_cpu_snapshot_native_dtype_cross_backend_"
        "storage_replay_diagnostic_v1"
    )
    assert geometry.continuous_ideal_reference == (
        "continuous_float64_empty_state_first_step_v1"
    )


def test_equivalence_samples_are_frozen_per_dataset() -> None:
    equivalence = load_d0_protocol_contract(CONFIG).equivalence
    assert equivalence.fresh_process_repetitions == 3
    assert equivalence.fresh_process_identity == (
        "linux_pid_and_proc_start_time_ticks"
    )
    assert equivalence.cross_process_pair_count == 3
    expected = {
        "IRSTD-1K": "XDU102",
        "NUAA-SIRST": "Misc_421",
        "NUDT-SIRST": "000891",
    }
    for dataset, image_id in expected.items():
        sample = equivalence.sample_for(dataset)
        assert sample.condition == "clean_S0"
        assert sample.image_index == 0
        assert sample.image_id == image_id
    with pytest.raises(D0ProtocolContractError, match="not uniquely frozen"):
        equivalence.sample_for("unknown")


def test_frozen_equivalence_samples_match_tracked_calibration_index_zero() -> None:
    contract = load_d0_protocol_contract(CONFIG)
    protocol = ROOT / contract.cache.protocol_path
    protocol_bytes = protocol.read_bytes()
    assert hashlib.sha256(protocol_bytes).hexdigest() == contract.cache.protocol_sha256
    cache_protocol = yaml.safe_load(protocol_bytes)
    corruption, severity = cache_protocol["input_protocol"]["ordered_conditions"][0]
    assert f"{corruption}_S{severity}" == "clean_S0"
    for sample in contract.equivalence.samples:
        dataset = cache_protocol["datasets"][sample.dataset]
        calibration_ids = ROOT / dataset["calibration_ids"]
        ids_bytes = calibration_ids.read_bytes()
        assert hashlib.sha256(ids_bytes).hexdigest() == dataset[
            "calibration_ids_file_sha256"
        ]
        image_ids = ids_bytes.decode("utf-8").splitlines()
        assert image_ids[sample.image_index] == sample.image_id


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_materialized_cache_manifests_match_frozen_equivalence_samples() -> None:
    contract = load_d0_protocol_contract(CONFIG)
    cache_root = ROOT / contract.cache.root
    manifest_paths = tuple(
        cache_root / sample.dataset / "manifest.json"
        for sample in contract.equivalence.samples
    )
    assert all(path.is_file() and not path.is_symlink() for path in manifest_paths)
    for sample in contract.equivalence.samples:
        manifest = json.loads(
            (cache_root / sample.dataset / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["image_ids"][sample.image_index] == sample.image_id
        assert manifest["conditions"][0]["key"] == sample.condition


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("cache", "condition_count"), 12),
        (("method", "entropy_eps"), 1.0e-5),
        (
            ("method", "determinism", "policy"),
            "strict_everywhere",
        ),
        (
            ("method", "determinism", "temporary_backward_disable_scopes"),
            ["entropy_backward"],
        ),
        (
            (
                "method",
                "determinism",
                "restore_strict_policy_before_post_forward",
            ),
            False,
        ),
        (("evaluation", "prediction_threshold"), 0.49),
        (
            ("evaluation", "optimizer_geometry", "runtime_hard_gate_reference"),
            "unfrozen_reference",
        ),
        (
            ("source_task_gradient", "contract"),
            "source_training_post_warm_exact",
        ),
        (
            ("equivalence", "samples", "IRSTD-1K", "image_id"),
            "XDU407",
        ),
        (("equivalence", "fresh_process_repetitions"), 1),
        (("output", "atomic_publish"), False),
        (("scope", "allowed_split_roles"), ["train"]),
        (
            ("datasets", "IRSTD-1K", "train_split_sha256"),
            "0" * 64,
        ),
        (("comparison", "stage1_records_sha256"), "0" * 64),
    ],
)
def test_execution_section_value_drift_is_rejected(
    path: tuple[str, ...], replacement: object
) -> None:
    value = copy.deepcopy(_mapping())
    _set_nested(value, path, replacement)
    with pytest.raises(D0ProtocolContractError, match="drifted"):
        parse_d0_protocol_contract(value)


@pytest.mark.parametrize(
    "section",
    [
        "cache",
        "method",
        "evaluation",
        "source_task_gradient",
        "equivalence",
        "output",
        "scope",
        "datasets",
        "comparison",
    ],
)
def test_decorative_execution_field_is_rejected(section: str) -> None:
    value = copy.deepcopy(_mapping())
    value[section]["informational_note"] = "ignored by execution"
    with pytest.raises(D0ProtocolContractError, match="fields are not exact"):
        parse_d0_protocol_contract(value)


def test_missing_execution_field_and_wrong_scalar_type_are_rejected() -> None:
    missing = copy.deepcopy(_mapping())
    del missing["evaluation"]["threshold_rule"]
    with pytest.raises(D0ProtocolContractError, match="missing=.*threshold_rule"):
        parse_d0_protocol_contract(missing)

    wrong_type = copy.deepcopy(_mapping())
    wrong_type["method"]["optimizer_steps_per_image"] = True
    with pytest.raises(D0ProtocolContractError, match="drifted"):
        parse_d0_protocol_contract(wrong_type)


def test_determinism_nested_fields_are_exact_and_not_decorative() -> None:
    value = copy.deepcopy(_mapping())
    value["method"]["determinism"]["informational_note"] = "not executable"
    with pytest.raises(D0ProtocolContractError, match="fields are not exact"):
        parse_d0_protocol_contract(value)

    wrong_order = copy.deepcopy(_mapping())
    wrong_order["method"]["determinism"][
        "temporary_backward_disable_scopes"
    ].reverse()
    with pytest.raises(D0ProtocolContractError, match="drifted"):
        parse_d0_protocol_contract(wrong_order)


def test_condition_value_and_order_drift_are_rejected() -> None:
    changed = copy.deepcopy(_mapping())
    changed["conditions"][0] = "clean_S1"
    with pytest.raises(D0ProtocolContractError, match="drifted"):
        parse_d0_protocol_contract(changed)

    reordered = copy.deepcopy(_mapping())
    reordered["conditions"][0], reordered["conditions"][1] = (
        reordered["conditions"][1],
        reordered["conditions"][0],
    )
    with pytest.raises(D0ProtocolContractError, match="drifted"):
        parse_d0_protocol_contract(reordered)


def test_dataclasses_and_exported_mapping_are_deeply_immutable() -> None:
    contract = load_d0_protocol_contract(CONFIG)
    with pytest.raises(FrozenInstanceError):
        contract.method.bn_protocol = "batch_statistics"  # type: ignore[misc]

    frozen = contract.as_frozen_mapping()
    with pytest.raises(TypeError):
        frozen["protocol_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen["evaluation"]["prediction_threshold"] = 0.2  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen["method"]["candidates"][0]["optimizer"] = "SGD"  # type: ignore[index]
    with pytest.raises(TypeError):
        frozen["method"]["determinism"]["strict_forward"][
            "warn_only"
        ] = True  # type: ignore[index]


def test_loader_rejects_symlink(tmp_path: Path) -> None:
    link = tmp_path / "d0.yaml"
    link.symlink_to(CONFIG)
    with pytest.raises(D0ProtocolContractError, match="symlink"):
        load_d0_protocol_contract(link)
