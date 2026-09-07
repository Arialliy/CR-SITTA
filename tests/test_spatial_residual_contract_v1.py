"""Synthetic metadata/byte contract checks; no real data, weights or GPU."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import spatial_residual_contract_v1 as contract


def test_canonical_config_is_frozen_train_only():
    raw = contract.read_config()
    assert raw["scope"]["split_name"] == "train"
    assert raw["scope"]["no_validation_split"] is True
    assert raw["scope"]["all_candidates_complete_before_outer_targets"] is True
    assert raw["scope"]["formal_test_allowed"] is False
    assert raw["candidate"]["absolute_radius"] == .25
    assert raw["candidate"]["trainable_scalars"] == 144


def test_noncanonical_config_rejected_before_any_read(monkeypatch):
    monkeypatch.setattr(contract, "sha256_file", lambda *_: pytest.fail("unexpected byte read"))
    with pytest.raises(contract.SpatialResidualProtocolError, match="canonical"):
        contract.read_config(contract.ROOT / "datasets/unknown/test.txt")


def test_config_hash_drift_fails_before_yaml(monkeypatch):
    monkeypatch.setattr(contract, "sha256_file", lambda *_: "0" * 64)
    with pytest.raises(contract.SpatialResidualProtocolError, match="byte hash"):
        contract.read_config()


@pytest.mark.parametrize("field,value", [("formal_test_allowed", True), ("no_validation_split", False),
                                        ("new_validation_split", True), ("full_source_training_allowed", True),
                                        ("paper_result", True), ("split_name", "test")])
def test_scope_drift_rejected(field, value):
    raw = contract.read_config()
    raw["scope"][field] = value
    with pytest.raises(contract.SpatialResidualProtocolError, match="scope"):
        contract._validate_config(raw)


def test_incomplete_code_roster_rejected():
    raw = contract.read_config()
    raw["implementation_files"].pop()
    with pytest.raises(contract.SpatialResidualProtocolError, match="code roster"):
        contract._validate_config(raw)


@pytest.mark.parametrize("relative", ["../data.npy", "/tmp/data.npy", ""])
def test_unsafe_path_rejected(relative):
    with pytest.raises(contract.SpatialResidualProtocolError):
        contract._path(relative)


def test_symlink_ancestor_rejected_without_payload_read(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(contract.SpatialResidualProtocolError, match="symlink"):
        contract._path("link/absent.npy")


def test_historical_code_pinned_to_saved_result_not_current_snapshot(monkeypatch):
    parent = SimpleNamespace(config_path=contract.ROOT / contract.PARENT_CONFIG)
    manifest = {"protocol_id": contract.b4.PROTOCOL_ID, "phase": "aggregate",
                "code_sha256": {"module.py": "f" * 64}}
    complete = {"complete": True, "manifest_sha256": contract.PARENT_MANIFEST_SHA256}
    monkeypatch.setattr(contract, "_binding", lambda path, expected=None: {"path": str(path)})
    monkeypatch.setattr(contract, "_json", lambda path: complete if path.name == "COMPLETE.json" else manifest)
    monkeypatch.setattr(contract.b4, "_capture_code_hashes", lambda _: {"module.py": "0" * 64})
    with pytest.raises(contract.SpatialResidualProtocolError, match="historical critical code"):
        contract._historical_bindings(parent, contract.read_config())


def export_transition_fixture():
    raw = contract.read_config()
    historical = {key: pair["historical_sha256"] for key, pair in contract.HISTORICAL_EXPORT_ONLY_DRIFT.items()}
    current = {key: pair["current_sha256"] for key, pair in contract.HISTORICAL_EXPORT_ONLY_DRIFT.items()}
    historical["tta/objectives/region_balanced_consistency.py"] = "a" * 64
    current["tta/objectives/region_balanced_consistency.py"] = "a" * 64
    return raw, historical, current


def test_exact_two_export_only_transitions_are_accepted():
    raw, historical, current = export_transition_fixture()
    contract._validate_historical_code(historical, current, raw)
    assert historical != current


@pytest.mark.parametrize("fault", ["unexpected_drift", "changed_current", "changed_historical",
                                   "changed_exception", "missing_exception", "extra_exception", "missing_code"])
def test_any_other_historical_drift_or_exception_hash_is_rejected(fault):
    raw, historical, current = export_transition_fixture()
    if fault == "unexpected_drift":
        current["tta/objectives/region_balanced_consistency.py"] = "b" * 64
    elif fault == "changed_current":
        current["tta/adapters/__init__.py"] = "b" * 64
    elif fault == "changed_historical":
        historical["tta/adapters/__init__.py"] = "b" * 64
    elif fault == "changed_exception":
        raw["historical_export_only_drift"]["tta/adapters/__init__.py"]["current_sha256"] = "b" * 64
    elif fault == "missing_exception":
        raw["historical_export_only_drift"].pop("tta/adapters/__init__.py")
    elif fault == "extra_exception":
        raw["historical_export_only_drift"]["tta/extra.py"] = {
            "historical_sha256": "a" * 64, "current_sha256": "b" * 64}
    else:
        current.pop("tta/adapters/__init__.py")
    with pytest.raises(contract.SpatialResidualProtocolError, match="historical critical code"):
        contract._validate_historical_code(historical, current, raw)


def test_input_hashing_never_requests_outer_target(monkeypatch):
    calls = []
    ids = [f"image_{index:03d}" for index in range(64)]
    record = {"cache_root": "synthetic/cache", "teacher_artifact_root": "synthetic/teacher",
              "checkpoint_path": "synthetic/checkpoint.pth.tar", "checkpoint_sha256": "a" * 64,
              "teacher_manifest_sha256": "b" * 64, "teacher_complete_sha256": "c" * 64}
    parent = SimpleNamespace(raw={"datasets": {"IRSTD-1K": record}})
    monkeypatch.setattr(contract.b4, "_verify_consumed_payloads",
                        lambda *args, **kwargs: calls.append(kwargs["include_outer_target"]))
    monkeypatch.setattr(contract.b4, "_teacher_manifest", lambda *_: {"image_ids": ids})
    monkeypatch.setattr(contract, "_binding", lambda path, expected=None: {"path": str(path)})
    actual, _ = contract._input_bindings(parent, "IRSTD-1K")
    assert actual == ids and calls == [False]


def synthetic_run(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    monkeypatch.setattr(contract, "PILOT_COUNT", 2)
    monkeypatch.setattr(contract, "CONDITIONS", (("clean", 0),))
    monkeypatch.setattr(contract, "EPISODE_COUNT", 2)
    monkeypatch.setattr(contract, "_assert_prepared_unchanged", lambda _: None)
    output = tmp_path / contract.RESULT_ROOT / "candidate" / "IRSTD-1K"
    manifest = {"protocol_id": contract.PROTOCOL_ID, "phase": "candidate", "dataset": "IRSTD-1K",
                "image_ids": ["b", "a"]}
    prepared = {"output": output, "contract": manifest,
                "preflight": {"outer_target_loader_calls": 0}, "new_config": {}, "parent_contract": None}
    contract.freeze_run(prepared)
    contract._write_json(output / "runtime.json", {"synthetic": True})
    condition = output / "conditions/clean_S0"
    masks = condition / "masks"
    masks.mkdir(parents=True)
    for name, shape, dtype in (
        ("source_probabilities", (2, 1, 256, 256), "float32"),
        ("post_probabilities", (2, 1, 256, 256), "float32"),
        ("proxy_gradients", (2, 144), "float64"),
        ("proposal_directions", (2, 144), "float64"),
        ("endpoint_kernels", (2, 16, 1, 3, 3), "float32"),
    ):
        np.save(condition / f"{name}.npy", np.zeros(shape, dtype=dtype), allow_pickle=False)
    rows = [{"dataset": "IRSTD-1K", "condition": "clean_S0", "image_id": image_id, "image_index": index,
             "method_label_accesses": 0, "episode_reset_exact": True}
            for index, image_id in enumerate(manifest["image_ids"])]
    (condition / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    for image_id in manifest["image_ids"]:
        (masks / f"{image_id}.png").write_bytes(b"\x89PNG\r\n\x1a\nsynthetic header only")
    summary = {"phase": "candidate", "dataset": "IRSTD-1K", "episode_count": 2,
               "condition_count": 1, "image_count_per_condition": 2, "image_ids": ["b", "a"],
               "method_label_accesses": 0, "outer_target_loader_calls": 0,
               "test_payload_opens": 0, "validation_payload_opens": 0,
               "source_state_restored": True, "predictions_complete": True,
               "no_validation_split": True, "paper_result": False}
    monkeypatch.setattr(contract, "prepare_run", lambda *args, **kwargs: prepared)
    return prepared, summary


def test_freeze_refuses_overwrite(tmp_path, monkeypatch):
    prepared, _ = synthetic_run(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        contract.freeze_run(prepared)


def test_complete_verify_hashes_only_no_numpy_load(tmp_path, monkeypatch):
    prepared, summary = synthetic_run(tmp_path, monkeypatch)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("must not deserialize array"))
    completion = contract.complete_run(prepared, summary)
    result = contract.verify_candidate(dataset="IRSTD-1K")
    assert result["summary"] == summary
    assert completion["complete"] is True
    assert result["bindings"]["conditions/clean_S0/post_probabilities.npy"]["bytes"] > 0
    with pytest.raises(contract.SpatialResidualProtocolError, match="overwrite"):
        contract.complete_run(prepared, summary)


@pytest.mark.parametrize("key,value", [("method_label_accesses", 1), ("outer_target_loader_calls", 1),
                                      ("test_payload_opens", 1), ("validation_payload_opens", 1),
                                      ("source_state_restored", False), ("predictions_complete", False),
                                      ("episode_count", 1), ("image_ids", ["a", "b"]),
                                      ("no_validation_split", False), ("paper_result", True)])
def test_invalid_completion_summary_refused(tmp_path, monkeypatch, key, value):
    prepared, summary = synthetic_run(tmp_path, monkeypatch)
    summary[key] = value
    with pytest.raises(contract.SpatialResidualProtocolError, match="summary field"):
        contract.complete_run(prepared, summary)
    assert not (prepared["output"] / "COMPLETE.json").exists()
    assert not (prepared["output"] / "summary.json").exists()


@pytest.mark.parametrize("fault", ["missing_mask", "extra_mask", "bad_array_shape", "reordered_episode", "not_reset"])
def test_incomplete_or_wrong_payload_refused(tmp_path, monkeypatch, fault):
    prepared, summary = synthetic_run(tmp_path, monkeypatch)
    condition = prepared["output"] / "conditions/clean_S0"
    if fault == "missing_mask":
        (condition / "masks/a.png").unlink()
    elif fault == "extra_mask":
        (condition / "masks/extra.png").write_bytes(b"extra")
    elif fault == "bad_array_shape":
        np.save(condition / "post_probabilities.npy", np.zeros((1, 1, 256, 256), np.float32))
    else:
        path = condition / "episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "reordered_episode":
            rows.reverse()
        else:
            rows[0]["episode_reset_exact"] = False
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(contract.SpatialResidualProtocolError):
        contract.complete_run(prepared, summary)
    assert not (prepared["output"] / "COMPLETE.json").exists()


@pytest.mark.parametrize("fault", ["mutated", "new_file", "new_symlink"])
def test_post_completion_file_or_roster_drift_refused(tmp_path, monkeypatch, fault):
    prepared, summary = synthetic_run(tmp_path, monkeypatch)
    contract.complete_run(prepared, summary)
    if fault == "mutated":
        (prepared["output"] / "runtime.json").write_text("{}")
    elif fault == "new_file":
        (prepared["output"] / "extra.txt").write_text("extra")
    else:
        (prepared["output"] / "extra.txt").symlink_to(prepared["output"] / "runtime.json")
    with pytest.raises(contract.SpatialResidualProtocolError):
        contract.verify_candidate(dataset="IRSTD-1K")


def test_global_completion_barrier_checked_before_dataset_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    monkeypatch.setattr(contract, "read_config", lambda *_: {"datasets": list(contract.DATASETS)})
    monkeypatch.setattr(contract, "verify_candidate", lambda *_: pytest.fail("barrier not satisfied"))
    with pytest.raises(contract.SpatialResidualProtocolError, match="all-candidate barrier"):
        contract.verify_all_candidates()
