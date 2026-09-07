"""Synthetic full-ledger checks; no real arrays/checkpoints/images are read."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from analysis import o3_multiscale_guard_contract_v2 as contract


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def descriptor(path):
    value = path.read_bytes()
    return {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}


def seal(root, repository):
    ledger = {name: descriptor(root / name) for name in sorted(contract._expected_files())}
    write(root / "artifact_ledger.json", ledger)
    manifest_sha = descriptor(root / "manifest.json")["sha256"]
    ledger_sha = descriptor(root / "artifact_ledger.json")["sha256"]
    write(root / "COMPLETE.json", {"automatic_full_training_allowed": False, "complete": True,
        "ledger_sha256": ledger_sha, "manifest_sha256": manifest_sha,
        "paper_result": False, "samples": 104, "steps": 128})
    return {"root": root, "fixture_repository": repository,
            "expected_manifest_sha256": manifest_sha, "expected_ledger_sha256": ledger_sha}


@pytest.fixture
def fixture_run(tmp_path):
    root = tmp_path / contract.PARENT_RELATIVE
    for name in contract._expected_files():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        # Intentionally invalid NPY/checkpoint/image bytes: hash-only verification
        # must not attempt to deserialize any of these files.
        path.write_bytes(b"synthetic opaque payload")
    train_ids = list(contract.IMAGE_IDS) + [f"synthetic_{i:04d}" for i in range(655)]
    train_path = tmp_path / contract.TRAIN_SPLIT
    train_path.parent.mkdir(parents=True)
    train_path.write_text("\n".join(train_ids) + "\n")
    train_sha = descriptor(train_path)["sha256"]
    write(tmp_path / contract.CACHE_MANIFEST, {"dataset": contract.DATASET, "image_ids": train_ids[:64],
          "train_split": contract.TRAIN_SPLIT, "train_split_sha256": train_sha})
    write(tmp_path / contract.TEACHER_MANIFEST, {"image_ids": train_ids[:64]})
    paths = [contract.CACHE_MANIFEST, contract.TEACHER_MANIFEST]
    for i in range(85):
        relative = f"synthetic_inputs/module_{i:03d}.py"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic bound bytes {i}".encode())
        paths.append(relative)
    bindings = [{"path": name, **descriptor(tmp_path / name)} for name in paths]
    config = {"protocol_id": contract.PARENT_PROTOCOL, "dataset": contract.DATASET, "image_count": 8,
              "result_root": contract.PARENT_RELATIVE, "seed": 42, "scope": copy.deepcopy(contract.SCOPE),
              "condition_source": "unchanged_parent_13_conditions",
              "training": {"steps": 128, "batch_size": 4, "checkpoint_rule": "fixed_last_step_no_metric_selection"}}
    manifest = {"bindings": bindings, "configuration": config, "protocol_id": contract.PARENT_PROTOCOL,
                "phase": contract.SCOPE["role"], "image_ids": list(contract.IMAGE_IDS),
                "fit_gt_access_authorized": True, "formal_test": False, "no_validation_split": True,
                "paper_result": False, "original_o3_method_label_accesses": 0}
    write(root / "manifest.json", manifest)
    summary = {"dataset": contract.DATASET, "image_ids": list(contract.IMAGE_IDS), "scope": contract.SCOPE,
               "samples": 104, "optimizer_steps": 128, "test_payload_opens": 0,
               "original_o3_method_label_accesses": 0, "host_restored": True,
               "generalization_claim": False, "automatic_full_training_allowed": False}
    write(root / "summary.json", summary)
    receipt = {"dataset": contract.DATASET, "image_ids": list(contract.IMAGE_IDS), "split_name": "train",
               "role": "source_supervised_residual_training_train8",
               "sampling": "first8_in_original_frozen_B4_Pilot64_order",
               "target_shape": [8, 1, 256, 256], "target_dtype": "float32", "train_split_sha256": train_sha,
               "formal_test": False, "paper_result": False, "no_validation_split": True,
               "supervised_labels_enter_original_o3_update": False,
               "train_mask_png_decodes": 8, "unique_train_mask_files": 8,
               **{key: 0 for key in ("outer_target_loader_calls", "sealed_outer_target_payload_opens",
                    "test_image_decodes", "test_mask_decodes", "test_split_reads", "validation_image_decodes",
                    "validation_mask_decodes", "validation_split_reads", "other_pilot_mask_decodes")}}
    write(root / "train_target_receipt.json", receipt)
    cells = [{"corruption": family, "severity": severity, "condition": contract._condition_key(family, severity),
              "dataset": contract.DATASET, **{key: {"image_count": 8} for key in ("source", "o3", "trained")}}
             for family, severity in contract.CONDITIONS]
    (root / "cells.jsonl").write_text("\n".join(json.dumps(row) for row in cells) + "\n")
    episodes = [{"condition": contract._condition_key(*condition), "image_id": image_id}
                for condition in contract.CONDITIONS for image_id in contract.IMAGE_IDS]
    (root / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes) + "\n")
    indices = [[(step * 4 + j) % 104 for j in range(4)] for step in range(128)]
    write(root / "training_order.json", {"indices": indices, "sample_order": "condition_major_then_image_id"})
    training = [{"step": i + 1, "sample_indices": batch, "loss": 1.} for i, batch in enumerate(indices)]
    (root / "training.jsonl").write_text("\n".join(json.dumps(row) for row in training) + "\n")
    return seal(root, tmp_path)


def test_complete_fixture_hashes_all_files_without_deserializing(fixture_run):
    result = contract.verify_parent_run(**fixture_run)
    assert len(result["parent_ledger"]) == 366 and len(result["bindings"]) == 87
    assert len(result["parent_cells"]) == 13
    assert result["parent_summary"]["image_ids"] == list(contract.IMAGE_IDS)
    assert result["receipt"]["arrays_deserialized"] == result["receipt"]["checkpoints_deserialized"] == 0


def test_production_hash_override_rejected_before_read(monkeypatch):
    monkeypatch.setattr(contract, "_json", lambda *args: pytest.fail("must not read"))
    with pytest.raises(contract.ParentRunVerificationError, match="overrides"):
        contract.verify_parent_run(expected_manifest_sha256="0" * 64)


def test_noncanonical_parent_requires_explicit_fixture_bindings(tmp_path):
    with pytest.raises(contract.ParentRunVerificationError, match="synthetic fixture"):
        contract.verify_parent_run(tmp_path / contract.PARENT_RELATIVE)


@pytest.mark.parametrize("name", ["manifest.json", "artifact_ledger.json", "o3_features.npy",
                                  "initial.pth.tar", "train_targets.npy", "training_order.json"])
def test_modified_bound_bytes_fail(fixture_run, name):
    path = fixture_run["root"] / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(contract.ParentRunVerificationError):
        contract.verify_parent_run(**fixture_run)


@pytest.mark.parametrize("fault", ["extra", "missing", "symlink", "external_binding"])
def test_file_roster_and_external_binding_failures(fixture_run, fault):
    root, repository = fixture_run["root"], fixture_run["fixture_repository"]
    if fault == "extra":
        (root / "extra.txt").write_text("not in sealed roster")
    elif fault == "missing":
        (root / "initial.pth.tar").unlink()
    elif fault == "symlink":
        (root / "extra.txt").symlink_to(root / "initial.pth.tar")
    else:
        (repository / "synthetic_inputs/module_000.py").write_text("changed external bytes")
    with pytest.raises(contract.ParentRunVerificationError):
        contract.verify_parent_run(**fixture_run)


@pytest.mark.parametrize("fault", ["test_scope", "reordered_ids", "labels_in_o3", "short_budget",
                                  "cell_order", "episode_order", "order_index", "executed_order", "binding_count"])
def test_metadata_scope_failures_even_with_explicit_fixture_hashes(fixture_run, fault):
    root, repository = fixture_run["root"], fixture_run["fixture_repository"]
    if fault in ("test_scope", "reordered_ids", "short_budget", "binding_count"):
        path = root / "manifest.json"
        data = json.loads(path.read_text())
        if fault == "test_scope": data["formal_test"] = True
        if fault == "reordered_ids": data["image_ids"].reverse()
        if fault == "short_budget": data["configuration"]["training"]["steps"] = 1
        if fault == "binding_count": data["bindings"].pop()
        write(path, data)
    elif fault == "labels_in_o3":
        path = root / "train_target_receipt.json"
        data = json.loads(path.read_text()); data["supervised_labels_enter_original_o3_update"] = True
        write(path, data)
    elif fault == "order_index":
        path = root / "training_order.json"
        data = json.loads(path.read_text()); data["indices"][0][0] = 104
        write(path, data)
    else:
        name = {"cell_order": "cells.jsonl", "episode_order": "episodes.jsonl", "executed_order": "training.jsonl"}[fault]
        path = root / name; rows = path.read_text().splitlines(); rows.reverse()
        path.write_text("\n".join(rows) + "\n")
    with pytest.raises(contract.ParentRunVerificationError):
        contract.verify_parent_run(**seal(root, repository))


def test_current_train_split_drift_is_rejected(fixture_run):
    path = fixture_run["fixture_repository"] / contract.TRAIN_SPLIT
    path.write_text(path.read_text() + "not-authorized\n")
    with pytest.raises(contract.ParentRunVerificationError, match="actual train split"):
        contract.verify_parent_run(**fixture_run)


def test_rewritten_complete_marker_cannot_claim_new_scope(fixture_run):
    path = fixture_run["root"] / "COMPLETE.json"
    value = json.loads(path.read_text()); value["automatic_full_training_allowed"] = True
    write(path, value)
    with pytest.raises(contract.ParentRunVerificationError, match="COMPLETE"):
        contract.verify_parent_run(**fixture_run)
