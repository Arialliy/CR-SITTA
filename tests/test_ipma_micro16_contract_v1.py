"""Synthetic contract tests: no real payload, checkpoint load or GPU access."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import types

import pytest

from analysis import d0a_v7_common as legacy
from analysis import ipma_micro16_contract_v1 as contract
from analysis import lf_repair_contract_v2 as lf


def config_fixture():
    return contract.read_config()


def test_registered_configuration_is_exact_and_engineering_only():
    config = config_fixture()
    assert config["scope"]["outer_optimizer_steps"] == 0
    assert all(config["scope"][key] is False for key in contract.NO_AUTHORIZATION)
    assert config["sampling"]["images_total"] == len(config["sampling"]["ids"]) == 16
    assert config["sampling"]["check_gt_decode_allowed"] is False


def test_unknown_config_path_rejected_before_hash_or_read(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unapproved path must not be read")
    monkeypatch.setattr(legacy, "sha256_file", forbidden)
    with pytest.raises(ValueError, match="canonical"):
        contract.read_config(contract.ROOT / "datasets/NUDT-SIRST/img_idx/test_NUDT-SIRST.txt")


def test_config_hash_drift_rejected_before_yaml_read(monkeypatch):
    monkeypatch.setattr(legacy, "sha256_file", lambda _: "0" * 64)
    with pytest.raises(ValueError, match="byte hash"):
        contract.read_config()


def test_config_symlink_rejected_without_following_payload(tmp_path, monkeypatch):
    destination = tmp_path / "payload.txt"
    destination.write_text("not a config", encoding="utf-8")
    path = tmp_path / "canonical.yaml"
    path.symlink_to(destination)
    monkeypatch.setattr(contract, "DEFAULT_CONFIG", path)
    monkeypatch.setattr(legacy, "sha256_file", lambda _: pytest.fail("must not hash symlink"))
    with pytest.raises(ValueError, match="canonical"):
        contract.read_config(path)


def host_fixture(tmp_path):
    source, safe, runtime, extension, run, receipt_path = [tmp_path / name for name in (
        "source.pth.tar", "safe.pth.tar", "model.py", "MultiScaleDeformableAttention.synthetic.so",
        "run_contract.json", "SAFE_EXPORT.json")]
    for path in (source, safe, runtime, extension):
        path.write_bytes(b"synthetic byte artifact, never loaded")
    run.write_text(json.dumps({"run_config": {"dataset": "NUDT-SIRST", "epochs": 1000,
        "host_architecture": "MSHNet_NSFPN", "checkpoint_selection": "fixed_final_epoch_train_only"},
        "runtime_sha256": {str(path): legacy.sha256_file(path) for path in (runtime, extension)}}), encoding="utf-8")
    receipt = {"receipt_type": "cr_sitta_d0a_safe_export_receipt_v1", "status": "published_weights_only_verified",
        "dataset": "NUDT-SIRST", "checkpoint_contract": {"architecture": "MSHNet_NSFPN", "epoch": 1000,
            "state_dict_keys": 505, "torch_load_weights_only": True, "all_state_dict_tensors_cpu": True,
            "test_selected": False, "selection_rule": "fixed_final_epoch_train_only",
            "repository_model_loads_verified": True},
        "artifacts": {key: legacy.binding(path) for key, path in
                      (("source_checkpoint", source), ("safe_checkpoint", safe), ("run_contract", run))}}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    host = {}
    for key, path in (("source_checkpoint", source), ("safe_checkpoint", safe), ("safe_export", receipt_path), ("run_contract", run)):
        host[key] = str(path); host[f"{key}_sha256"] = legacy.sha256_file(path)
    return {"host": host}, receipt, extension


def test_host_chain_and_extension_bytes_verified_without_loading(tmp_path):
    config, _, _ = host_fixture(tmp_path)
    result = contract.validate_host_metadata(config)
    assert result["weights_loaded"] is False
    assert result["original_runtime_files_verified"] == 2
    assert any(Path(item["path"]).suffix == ".so" for item in result["bindings"])


def test_host_runtime_extension_drift_fails(tmp_path):
    config, _, extension = host_fixture(tmp_path)
    extension.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="runtime"):
        contract.validate_host_metadata(config)


@pytest.mark.parametrize("key,value", [("torch_load_weights_only", False), ("test_selected", True),
                                       ("state_dict_keys", 504), ("epoch", 999),
                                       ("selection_rule", "best_test_miou")])
def test_invalid_safe_metadata_is_rejected_even_if_receipt_hash_matches(tmp_path, key, value):
    config, receipt, _ = host_fixture(tmp_path)
    receipt["checkpoint_contract"][key] = value
    path = Path(config["host"]["safe_export"])
    path.write_text(json.dumps(receipt), encoding="utf-8")
    config["host"]["safe_export_sha256"] = legacy.sha256_file(path)
    with pytest.raises(ValueError, match="safe-export"):
        contract.validate_host_metadata(config)


def replay_fixture():
    config = config_fixture()
    records = [{"dataset": "NUDT-SIRST", "image_id": name} for name in config["sampling"]["ids"]]
    old, current = [], []
    approved = {"operator_config_sha256": "a" * 64}
    for record in records:
        common = {**record, "view": "train_crop_224", "target_count": 1,
            "input_metadata": {"input_tensor_sha256": "b" * 64, "target_tensor_sha256": "c" * 64,
                               "view": "train_crop_224", "augmentation_seed": 42}}
        old.append({**common, "probe_id": "lf_mask"})
        current.append({**copy.deepcopy(common), "probe_id": "L4a", "input_hash": "b" * 64,
            "target_hash": "c" * 64, "operator_config_sha256": approved["operator_config_sha256"],
            "sealed_input_pair_exact": True, "gt_tensor_unchanged": True,
            **{key: "d" * 64 for key in ("clean_physical_tensor_sha256", "preclip_physical_tensor_sha256",
                                      "postclip_physical_tensor_sha256", "random_field_sha256")}})
    return config, records, current, old, approved


def test_replay_binding_preserves_original_prefix_order_not_row_sort():
    config, records, current, old, approved = replay_fixture()
    result = contract.validate_replay_rows(config, records, list(reversed(current)), old, approved)
    assert [row["image_id"] for row in result] == config["sampling"]["ids"]


@pytest.mark.parametrize("fault", ["input", "target", "operator", "physical", "missing", "duplicate", "reordered_ids"])
def test_input_gt_lf_or_order_mismatch_fails(fault):
    config, records, current, old, approved = replay_fixture()
    if fault == "input": current[0]["input_hash"] = "x" * 64
    if fault == "target": current[0]["input_metadata"]["target_tensor_sha256"] = "x" * 64
    if fault == "operator": current[0]["operator_config_sha256"] = "x" * 64
    if fault == "physical": current[0]["postclip_physical_tensor_sha256"] = "invalid"
    if fault == "missing": current.pop()
    if fault == "duplicate": current.append(current[0])
    if fault == "reordered_ids": records.reverse()
    with pytest.raises(ValueError):
        contract.validate_replay_rows(config, records, current, old, approved)


def approval_fixture(tmp_path, monkeypatch):
    config = config_fixture()
    source = tmp_path / "operator.py"
    source.write_text("synthetic source", encoding="utf-8")
    approved = {"probe_id": "L4a", "mask_ratio": .2, "pair_keep_probability": .5, "attenuation": .25,
        "protect_dc": True, "shared_channels": True, "signal_smoke_16_allowed": True,
        "operator_source_sha256": legacy.sha256_file(source), **{key: False for key in contract.NO_AUTHORIZATION}}
    gate = {"selected_probe_id": "L4a", "signal_smoke_16_allowed": True,
        "scientific_scope": "augmentation_visibility_screen_only", "approved_probe": approved,
        "approved_operator_sha256": approved["operator_source_sha256"],
        **{key: False for key in contract.NO_AUTHORIZATION}}
    output = tmp_path / "lf/finalization"
    output.mkdir(parents=True)
    gate_path, approved_path = output / "GATE_LF_RECEIPT.json", output / "APPROVED_LF_CONFIG.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    approved_path.write_text(json.dumps(approved), encoding="utf-8")
    config["lf_gate_sha256"] = legacy.sha256_file(gate_path)
    config["lf_approved_config_sha256"] = legacy.sha256_file(approved_path)
    monkeypatch.setattr(lf, "verify_complete_dir", lambda *args: {"summary": gate, "bindings": []})
    monkeypatch.setattr(contract, "_verify_freeze", lambda *args: [])
    original_bound = contract._bound
    def bound(path, expected, label):
        return original_bound(source if label == "approved LF implementation" else path, expected, label)
    monkeypatch.setattr(contract, "_bound", bound)
    return config, {"result_root": str(tmp_path / "lf")}, gate, approved, gate_path


def test_lf_approval_scope_is_visibility_only(tmp_path, monkeypatch):
    config, lf_config, _, _, _ = approval_fixture(tmp_path, monkeypatch)
    result = contract.validate_lf_approval(config, lf_config)
    assert result["approved_lf"]["probe_id"] == "L4a"
    assert result["approved_lf"]["ipma_meta_training_allowed"] is False


@pytest.mark.parametrize("key,value", [("selected_probe_id", "L4b"), ("signal_smoke_16_allowed", False),
                                      ("formal_test_allowed", True), ("ipma_meta_training_allowed", True)])
def test_lf_wrong_selection_or_scope_is_rejected(tmp_path, monkeypatch, key, value):
    config, lf_config, gate, _, path = approval_fixture(tmp_path, monkeypatch)
    gate[key] = value
    path.write_text(json.dumps(gate), encoding="utf-8")
    config["lf_gate_sha256"] = legacy.sha256_file(path)
    with pytest.raises(ValueError):
        contract.validate_lf_approval(config, lf_config)


def completion_fixture(tmp_path, monkeypatch):
    config = config_fixture()
    config_path = tmp_path / "config.yaml"
    config_path.write_text("synthetic frozen config", encoding="utf-8")
    source = tmp_path / "input.txt"
    source.write_text("synthetic input bytes", encoding="utf-8")
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    monkeypatch.setattr(contract, "DEFAULT_CONFIG", config_path)
    monkeypatch.setattr(contract, "read_config", lambda *args: config)
    monkeypatch.setattr(lf, "validate_preregistration", lambda *args: {})
    prepared = {"config_path": config_path, "new_config": config,
        "records": [{"image_id": name} for name in config["sampling"]["ids"]],
        "contract": {"configuration": copy.deepcopy(config), "config_binding": legacy.binding(config_path),
            "scope": dict(config["scope"]), "runtime": {"injected": "synthetic CPU test"}},
        "output": tmp_path / contract.OUTPUT_RELATIVE,
        "bindings": [legacy.binding(source), legacy.binding(config_path)]}
    return prepared, source


def test_append_only_complete_can_be_negative_and_never_authorizes_training(tmp_path, monkeypatch):
    prepared, _ = completion_fixture(tmp_path, monkeypatch)
    contract.freeze_run(prepared)
    contract.complete_run(prepared, {"engineering_passed": False, "status": "signal_insufficient"})
    result = contract.verify_run(prepared["output"])
    assert result["engineering_passed"] is False
    assert result["outer_optimizer_steps"] == 0
    assert result["ipma_meta_training_allowed"] is False
    assert result["r5_requires_separate_frozen_contract"] is True
    with pytest.raises(FileExistsError): contract.freeze_run(prepared)
    with pytest.raises(FileExistsError): contract.complete_run(prepared, {})


def test_current_frozen_input_drift_prevents_completion(tmp_path, monkeypatch):
    prepared, source = completion_fixture(tmp_path, monkeypatch)
    contract.freeze_run(prepared)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        contract.complete_run(prepared, {})
    assert not (prepared["output"] / "COMPLETE.json").exists()


def test_even_empty_partial_output_is_not_reused(tmp_path, monkeypatch):
    prepared, _ = completion_fixture(tmp_path, monkeypatch)
    prepared["output"].mkdir(parents=True)
    with pytest.raises(FileExistsError):
        contract.freeze_run(prepared)
    assert not list(prepared["output"].iterdir())


def test_current_input_drift_is_detected_when_verifying_completed_run(tmp_path, monkeypatch):
    prepared, source = completion_fixture(tmp_path, monkeypatch)
    contract.freeze_run(prepared)
    contract.complete_run(prepared, {"engineering_passed": True})
    source.write_text("changed after completion", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        contract.verify_run(prepared["output"])


@pytest.mark.parametrize("key,value", [("outer_optimizer_steps", 1), ("ipma_meta_training_allowed", True),
                                      ("formal_test_allowed", True), ("paper_result", True), ("new_validation_split", True)])
def test_technical_pass_cannot_promote_scope(tmp_path, monkeypatch, key, value):
    prepared, _ = completion_fixture(tmp_path, monkeypatch)
    contract.freeze_run(prepared)
    with pytest.raises(ValueError):
        contract.complete_run(prepared, {"engineering_passed": True, key: value})
    assert not (prepared["output"] / "COMPLETE.json").exists()


def test_wrong_verification_path_is_rejected_before_read(monkeypatch):
    monkeypatch.setattr(contract, "read_config", lambda *args: pytest.fail("must reject path first"))
    with pytest.raises(ValueError, match="canonical"):
        contract.verify_run(contract.ROOT / "datasets/NUDT-SIRST/masks/not_in_scope.png")


def test_prepare_is_read_only_metadata_and_preserves_exact_prefix(monkeypatch, tmp_path):
    config, records, current, old_rows, approved_probe = replay_fixture()
    for record in records:
        record.update(image_path=str(tmp_path / "images" / (record["image_id"] + ".png")),
                      mask_path=str(tmp_path / "masks" / (record["image_id"] + ".png")))
    lf_config = {"result_root": str(tmp_path / "lf"), "legacy_config": "synthetic_legacy.yaml",
        "parent_result_root": str(tmp_path / "parent"),
        "parent_manifests": {"probe_label_preservation/NUDT-SIRST": "p2manifest"}}
    monkeypatch.setattr(contract, "read_config", lambda *args: config)
    monkeypatch.setattr(lf, "read_config", lambda *args: lf_config)
    monkeypatch.setattr(lf, "validate_preregistration", lambda *args: {"input_bindings": []})
    monkeypatch.setattr(contract, "validate_lf_approval", lambda *args: {"approved_lf": approved_probe,
        "gate": {"r2_dataset_manifest_sha256": {"NUDT-SIRST": "r2manifest"}}, "bindings": []})
    monkeypatch.setattr(contract, "validate_host_metadata", lambda *args: {"bindings": [], "original_runtime_files_verified": 13})
    monkeypatch.setattr(legacy, "read_config", lambda *args: {"protocol_id": "synthetic_legacy"})
    monkeypatch.setattr(legacy, "load_pilot_records", lambda *args: records + [{"image_id": "not-selected"}])
    byte_checks = []
    monkeypatch.setattr(lf, "assert_pilot_raw_bindings", lambda cfg, entries: byte_checks.extend(entries))
    monkeypatch.setattr(lf, "verify_complete_dir", lambda *args: {"bindings": []})
    monkeypatch.setattr(contract, "_verify_freeze", lambda *args: [])
    monkeypatch.setattr(lf, "read_jsonl", lambda path: old_rows if "parent" in str(path) else current)
    monkeypatch.setattr(legacy, "binding", lambda path: {"path": str(path), "sha256": "0" * 64})
    monkeypatch.setattr(contract.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(stdout="syntheticcommit\n"))
    monkeypatch.setattr(legacy, "load_sample", lambda *args: pytest.fail("must not decode"))
    monkeypatch.setattr(legacy, "freeze_run", lambda *args: pytest.fail("prepare must not write"))
    prepared = contract.prepare_run()
    assert [row["image_id"] for row in prepared["records"]] == config["sampling"]["ids"]
    assert prepared["records"] == byte_checks
    assert prepared["legacy_lf_image_rows"] == current
    assert prepared["preflight"]["writes"] == prepared["preflight"]["image_decodes"] == 0
    assert prepared["preflight"]["weights_loaded"] is False
    assert prepared["preflight"]["cuda_initialized"] is False
    assert prepared["contract"]["configuration"] == config
    assert len(prepared["bindings"]) >= len(contract.NEW_SOURCES)
