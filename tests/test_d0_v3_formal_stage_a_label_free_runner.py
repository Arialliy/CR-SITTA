from __future__ import annotations

import os
from pathlib import Path

import pytest

from analysis.d0_v2_independent_candidate_contract import FROZEN_CANDIDATES
from analysis.d0_v3_formal_contract import (
    CONFIG_FILE_SHA256,
    FROZEN_CANDIDATES as FORMAL_CANDIDATES,
)
from analysis.d0_v3_label_free_shard import (
    D0V3LabelFreeShardError,
    REQUIRED_CRITICAL_CODE_PATHS,
    _validate_input_seal,
    canonical_sha256,
    validate_method_facing_seal_path,
)
from scripts.run_d0_v3_formal_stage_a_label_free import (
    DEFAULT_CONFIG,
    D0V3StageAWorkerError,
    _validate_candidate_grid,
    run_parent,
    validate_contract_only,
)


LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_validate_is_cpu_only_and_does_not_create_output(tmp_path: Path) -> None:
    del tmp_path
    before = Path("results/cr_sitta/tent_failure_diagnostics_v3_formal_stage_a")
    existed = before.exists()
    value = validate_contract_only(DEFAULT_CONFIG)
    assert value["valid"] is True
    assert value["config_sha256"] == CONFIG_FILE_SHA256
    assert value["filesystem_created"] is False
    assert value["gpu_initialized"] is False
    assert value["target_payload_opened"] is False
    assert value["test_payload_opened"] is False
    assert value["formal_protocol_complete"] is False
    assert value["stage2_authorized"] is False
    assert before.exists() is existed


def test_candidate_count_requires_both_sides_exactly_ten() -> None:
    _validate_candidate_grid(FORMAL_CANDIDATES, FROZEN_CANDIDATES)
    with pytest.raises(D0V3StageAWorkerError, match="count drifted"):
        _validate_candidate_grid(FORMAL_CANDIDATES[:-1], FROZEN_CANDIDATES)
    with pytest.raises(D0V3StageAWorkerError, match="count drifted"):
        _validate_candidate_grid(FORMAL_CANDIDATES, FROZEN_CANDIDATES[:-1])


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_r1_r2_are_blocked_before_cuda_or_output_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.run_d0_v3_formal_stage_a_label_free._make_destination_parent",
        lambda _destination: pytest.fail("must reject before output creation"),
    )
    with pytest.raises(D0V3StageAWorkerError, match="R1/R2 are blocked"):
        run_parent(
            config_path=DEFAULT_CONFIG,
            dataset="NUAA-SIRST",
            condition="clean_S0",
            replicate="R1",
            max_images=64,
            cuda_visible_device="0",
        )


def test_test_source_code_is_allowed_but_test_payloads_are_rejected() -> None:
    assert (
        validate_method_facing_seal_path(
            "critical_code:test_source.py", "test_source.py"
        )
        == "test_source.py"
    )
    for role, path in (
        ("test_split", "datasets/NUAA-SIRST/test.txt"),
        ("test_image", "data/test_images/a.png"),
        ("cache_target", "cache/outer_evaluator/targets.npy"),
    ):
        with pytest.raises(D0V3LabelFreeShardError, match="forbidden"):
            validate_method_facing_seal_path(role, path)


def _input_seal() -> tuple[dict, dict]:
    digest = "a" * 64
    base = {
        "formal_config": "configs/formal.yaml",
        "parent_engineering_config": "configs/parent.yaml",
        "engineering_smoke_aggregate": "results/smoke/aggregate.json",
        "cache_execution_protocol": "configs/cache_execution.yaml",
        "cache_protocol": "configs/cache.yaml",
        "source_train_split": "datasets/NUAA-SIRST/train.txt",
        "frozen_pilot_ids": "datasets/NUAA-SIRST/pilot64.txt",
        "source_checkpoint": "results/baseline/NUAA-SIRST/best_miou.pth.tar",
        "cache_manifest": "results/cache/manifest.json",
        "cache_method_manifest": "results/cache/method_input_manifest.json",
        "cache_complete": "results/cache/COMPLETE.json",
        "method_condition": "results/cache/conditions/clean_S0.npy",
    }
    records = [
        {"role": role, "path": path, "sha256": digest}
        for role, path in base.items()
    ] + [
        {
            "role": f"critical_code:{path}",
            "path": path,
            "sha256": digest,
        }
        for path in REQUIRED_CRITICAL_CODE_PATHS
    ]
    records.sort(key=lambda value: value["role"])
    # The formal config has its own binding.
    records[next(index for index, value in enumerate(records) if value["role"] == "formal_config")][
        "sha256"
    ] = "b" * 64
    seal = {
        "files": records,
        "seal_sha256": canonical_sha256(records),
        "target_payload_bytes_opened": 0,
        "target_payload_deserialized": False,
        "validation_payload_opens": 0,
        "test_split_files_opened": 0,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "test_labels_opened": 0,
    }
    dataset_binding = {
        "train_split_sha256": digest,
        "checkpoint_role": "best_miou",
        "checkpoint_path": base["source_checkpoint"],
        "checkpoint_sha256": digest,
    }
    return seal, dataset_binding


def test_input_seal_binds_dataset_checkpoint_split_and_zero_boundary() -> None:
    seal, binding = _input_seal()
    records, code = _validate_input_seal(
        seal, config_sha256="b" * 64, dataset_binding=binding
    )
    assert len(records) == 12 + len(REQUIRED_CRITICAL_CODE_PATHS)
    assert set(code) == set(REQUIRED_CRITICAL_CODE_PATHS)

    drifted_binding = dict(binding, checkpoint_sha256="c" * 64)
    with pytest.raises(D0V3LabelFreeShardError, match="differs from input seal"):
        _validate_input_seal(
            seal,
            config_sha256="b" * 64,
            dataset_binding=drifted_binding,
        )
    nonzero = dict(seal, test_images_opened=1)
    with pytest.raises(D0V3LabelFreeShardError, match="boundary differs"):
        _validate_input_seal(
            nonzero,
            config_sha256="b" * 64,
            dataset_binding=binding,
        )
