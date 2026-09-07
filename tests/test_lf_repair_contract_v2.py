"""Contract checks use synthetic text artifacts, never real image payloads."""
import copy
import json

import pytest
import yaml

from analysis import d0a_v7_common as legacy
from analysis import lf_repair_contract_v2 as contract


def config_fixture():
    return yaml.safe_load(contract.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_registered_config_keeps_old_validator_and_scope():
    new = contract.read_config()
    old = legacy.read_config(new["legacy_config"])
    from analysis.audit_d0a_probe_label_preservation import validate_diagnostic_parameters
    validate_diagnostic_parameters(old)
    changed = copy.deepcopy(old)
    changed["probes"]["lf_mask"]["keep_probability"] = .6
    with pytest.raises(ValueError):
        validate_diagnostic_parameters(changed)
    assert new["datasets"] == list(contract.DATASETS)
    assert all(v is False for v in new["stage_scope"].values())


@pytest.mark.parametrize("path,value", [
    (("stage_scope", "formal_test_allowed"), True),
    (("stage_scope", "ipma_meta_training_allowed"), True),
    (("stage_scope", "new_validation_split"), True),
    (("gate", "risk_fraction_max"), .2),
    (("gate", "image_rms_floor"), 0.0),
    (("gate", "image_rms_fraction_min"), .5),
    (("gate", "compound_denominator"), "all_defined"),
    (("bootstrap", "replicates"), 100),
    (("bootstrap", "seed"), 43),
    (("random_field", "include_view_in_seed"), False),
    (("operator", "pair_keep_probability"), True),
    (("operator", "mask_ratio"), .3),
    (("selection", "r3_selected_probe_must_be"), "L4b"),
    (("device",), "cuda:0"),
    (("global_seed",), 43),
    (("pilot_images_per_dataset",), 32),
    (("result_root",), "results/cr_sitta/d0a_v7_diagnostics_v1"),
    (("design_document",), "datasets/NUDT-SIRST/images/not_allowlisted.png"),
    (("implementation_addendum",), "datasets/NUDT-SIRST/img_idx/test_NUDT-SIRST.txt"),
    (("parent_code_anchor",), "HEAD"),
    (("parent_execution_receipt_sha256",), "0" * 64),
    (("parent_manifests", "branch_gradients/NUDT-SIRST"), "0" * 64),
])
def test_unregistered_protocol_change_fails(tmp_path, path, value):
    config = config_fixture()
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    candidate = tmp_path / "config.yaml"
    candidate.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError):
        contract.read_config(candidate)


def sealed_fixture(tmp_path):
    source = tmp_path / "输入.txt"
    source.write_text("冻结文本", encoding="utf-8")
    output = tmp_path / "结果"
    legacy.freeze_run(output, {"scope": "synthetic_only"}, [legacy.binding(source)])
    contract.complete_output(output, {"status": "risk_failed", "scientific_pass": False})
    return source, output


def test_complete_is_utf8_and_scientific_failure_not_training_permission(tmp_path):
    _, output = sealed_fixture(tmp_path)
    verified = contract.verify_complete_dir(output)
    assert verified["summary"]["scientific_pass"] is False
    assert verified["complete"]["complete"] is True
    assert verified["complete"]["new_training_authorized"] is False
    assert len(verified["bindings"]) == 5
    with pytest.raises(FileExistsError):
        contract.complete_output(output, {})


def test_frozen_input_changed_cannot_complete(tmp_path):
    source = tmp_path / "input"
    source.write_text("original", encoding="utf-8")
    output = tmp_path / "new"
    legacy.freeze_run(output, {}, [legacy.binding(source)])
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError):
        contract.complete_output(output, {})
    assert not (output / "COMPLETE.json").exists()
    assert not (output / "summary.json").exists()


def test_sealed_record_tamper_fails(tmp_path):
    _, output = sealed_fixture(tmp_path)
    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        contract.verify_complete_dir(output)


def test_manifest_hash_and_missing_complete_fail(tmp_path):
    _, output = sealed_fixture(tmp_path)
    with pytest.raises(ValueError):
        contract.verify_complete_dir(output, "0" * 64)
    with pytest.raises(FileNotFoundError):
        contract.verify_complete_dir(tmp_path / "missing")


def test_manifest_path_escape_fails(tmp_path):
    _, output = sealed_fixture(tmp_path)
    manifest = contract.read_json(output / "artifact_manifest.json")
    manifest["files"]["../输入.txt"] = legacy.sha256_file(tmp_path / "输入.txt")
    (output / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    complete = contract.read_json(output / "COMPLETE.json")
    complete["artifact_manifest"] = legacy.binding(output / "artifact_manifest.json")
    (output / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        contract.verify_complete_dir(output)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_rejected(tmp_path, token):
    path = tmp_path / "bad.json"
    path.write_text('{"number":' + token + '}', encoding="utf-8")
    with pytest.raises(ValueError):
        contract.read_json(path)
    with pytest.raises(ValueError):
        contract.read_jsonl(path)


def test_conflicting_bindings_rejected():
    with pytest.raises(ValueError):
        contract.deduplicate([{"path": "a", "sha256": "1"}, {"path": "a", "sha256": "2"}])
    assert contract.deduplicate([{"path": "a", "sha256": "1"}] * 2) == [{"path": "a", "sha256": "1"}]


def test_missing_prereg_cannot_freeze_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "validate_preregistration", lambda _: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(FileNotFoundError):
        contract.freeze_dataset({"config_path": tmp_path / "config", "output": tmp_path / "run"})
    assert not (tmp_path / "run").exists()


def test_raw_train_binding_drift_fails_without_decoder(tmp_path, monkeypatch):
    image, mask = tmp_path / "a.png", tmp_path / "mask.png"
    image.write_bytes(b"synthetic-not-an-image")
    mask.write_bytes(b"synthetic-not-a-mask")
    bindings = [legacy.binding(image), legacy.binding(mask)]
    monkeypatch.setattr(contract, "read_json", lambda _: {"input_bindings": bindings})
    records = [{"dataset": "NUDT-SIRST", "image_path": str(image), "mask_path": str(mask)}]
    contract.assert_pilot_raw_bindings({"parent_result_root": str(tmp_path)}, records)
    mask.write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw payload"):
        contract.assert_pilot_raw_bindings({"parent_result_root": str(tmp_path)}, records)
