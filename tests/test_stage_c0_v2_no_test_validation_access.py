from __future__ import annotations

import inspect
import os
from pathlib import Path
import sys
import types

import pytest
import yaml

import run_p3_stage_c0_signal_audit_v2 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3_stage_c0_signal_audit_v2.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


@pytest.fixture()
def contract(monkeypatch: pytest.MonkeyPatch):
    if not LOCAL_ARTIFACT_TESTS_ENABLED:
        pytest.skip(
            "requires the frozen local Stage-C environment; set "
            f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
        )
    monkeypatch.setattr(
        runner, "FROZEN_CONFIG_SHA256", runner.current_config_sha256()
    )
    return runner.load_contract()


def test_stage_c0_config_has_no_validation_or_test_data_binding() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scope = raw["scope"]
    assert scope["split_name"] == "train"
    assert scope["no_validation_split"] is True
    assert scope["use_validation_payload"] is False
    assert scope["use_test_payload"] is False
    assert scope["method_label_accesses"] == 0
    assert scope["candidate_outer_target_loader_module_imports"] == 0
    assert scope["candidate_target_payload_deserializations"] == 0
    for dataset in raw["datasets"].values():
        assert set(dataset).isdisjoint(
            {"test_split", "validation_split", "val_split"}
        )
        assert "trainval" not in dataset["train_split"].lower()
        assert "/img_idx/train_" in dataset["train_split"]
    assert raw["stage_c0_signal_gate"]["formal_test_authorized"] is False


def test_candidate_and_probe_public_interfaces_cannot_receive_labels() -> None:
    forbidden = {"target", "label", "mask", "ground_truth", "validation", "test"}
    for function in (
        runner.build_probe_image,
        runner.derive_probe_seed,
        runner._label_free_objective,
        runner._run_candidate,
    ):
        parameters = set(inspect.signature(function).parameters)
        assert parameters.isdisjoint(forbidden)
    candidate_source = inspect.getsource(runner._run_candidate)
    assert "load_outer_evaluator_targets_v2" not in candidate_source
    assert "outer_evaluator/targets.npy" not in candidate_source


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local Stage-C method-input artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_candidate_method_reader_has_a_real_module_capability_firewall(
    contract, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = "materialize_binary_tent_ss_calibration_cache_v2"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    method = runner._method_input_dataset(
        contract, "IRSTD-1K", "clean_S0"
    )
    assert len(method) == 64
    assert module_name not in sys.modules
    from tta import stage_c0_method_input

    assert module_name not in inspect.getsource(stage_c0_method_input)


def test_outer_target_import_is_unreachable_before_candidate_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = False

    def forbidden_loader(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("target loader must be unreachable")

    fake_module = types.ModuleType("materialize_binary_tent_ss_calibration_cache_v2")
    fake_module.load_outer_evaluator_targets_v2 = forbidden_loader
    monkeypatch.setitem(
        sys.modules, "materialize_binary_tent_ss_calibration_cache_v2", fake_module
    )

    def fail_preflight(*args, **kwargs):
        raise runner.StageC0ProtocolError("candidate invalid")

    monkeypatch.setattr(runner, "verify_candidate_artifact", fail_preflight)
    token = runner.VerifiedCandidateArtifact(
        path=ROOT / "does-not-exist",
        dataset="IRSTD-1K",
        config_sha256="0" * 64,
        manifest_sha256="1" * 64,
        payload_tree_sha256="2" * 64,
        formal=True,
        condition_count=13,
        image_count_per_condition=64,
        probe_count=2,
    )
    with pytest.raises(runner.StageC0ProtocolError, match="candidate invalid"):
        runner.open_outer_train_targets(
            object(), dataset="IRSTD-1K", verified_candidate=token
        )
    assert opened is False


def test_formal_and_engineering_namespaces_are_disjoint(contract) -> None:
    formal = runner._artifact_destination(
        contract,
        phase="candidate",
        dataset="IRSTD-1K",
        formal=True,
        smoke_id=None,
    )
    smoke = runner._artifact_destination(
        contract,
        phase="candidate",
        dataset="IRSTD-1K",
        formal=False,
        smoke_id="cpu-toy",
    )
    assert formal != smoke
    assert "engineering_smoke" not in formal.parts
    assert "engineering_smoke" in smoke.parts
    assert formal == (
        ROOT
        / "results/cr_sitta/p3_stage_c0_signal_audit_v2/candidate_phase/R0/IRSTD-1K"
    )


def test_access_counter_validator_fails_closed() -> None:
    clean = {
        "method_label_accesses": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
    }
    runner._validate_access_zero(clean, label="toy")
    for field in clean:
        changed = dict(clean)
        changed[field] = 1
        with pytest.raises(runner.StageC0ProtocolError, match=field):
            runner._validate_access_zero(changed, label="toy")
