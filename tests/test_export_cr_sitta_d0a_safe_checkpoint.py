from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from torch import Tensor, nn

import export_cr_sitta_d0a_safe_checkpoint as exporter


def _sha(path: Path) -> str:
    return exporter.sha256_file(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


class _ToyMSHNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for index in range(exporter.EXPECTED_STATE_DICT_KEYS):
            self.register_buffer(f"tensor_{index:03d}", torch.zeros(1))


class _ToyAdaptable:
    def __init__(self) -> None:
        self.source = _ToyMSHNet()

    def load_source_state_dict(self, state_dict: dict[str, Tensor]):
        return self.source.load_state_dict(state_dict, strict=True)


def _strict_original_validator(state_dict: dict[str, Tensor]) -> None:
    result = _ToyMSHNet().load_state_dict(state_dict, strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def _adaptable_source_validator(state_dict: dict[str, Tensor]) -> None:
    result = _ToyAdaptable().load_source_state_dict(state_dict)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def _state_dict() -> dict[str, Tensor]:
    return {
        f"tensor_{index:03d}": torch.tensor([float(index)], dtype=torch.float32)
        for index in range(exporter.EXPECTED_STATE_DICT_KEYS)
    }


def _fixture(tmp_path: Path) -> dict[str, Any]:
    repository = tmp_path / "repository"
    output_dir = repository / "results" / "cr_sitta" / "d0a" / "TOY-SIRST"
    output_dir.mkdir(parents=True)
    (output_dir / "splits").mkdir()

    runtime_files = {
        "train_cr_sitta_d0a.py": b"# frozen training runner\n",
        "model/MSHNet_NSFPN.py": b"# frozen original architecture\n",
    }
    for relative, payload in runtime_files.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    adaptable_path = repository / "model" / "MSHNet_NSFPN_adaptable.py"
    adaptable_path.write_bytes(b"# separately bound adaptable loader\n")

    train_split = repository / "datasets" / "TOY-SIRST" / "img_idx" / "train.txt"
    train_split.parent.mkdir(parents=True)
    train_split.write_text("a\nb\nc\nd\n", encoding="utf-8")
    (output_dir / "splits" / "train.txt").write_bytes(train_split.read_bytes())
    train_split_sha = _sha(train_split)
    corpus_sha = "c" * 64

    protocol_path = repository / "configs" / "cr_sitta_d0a_train_v2.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol = {
        "schema_version": 1,
        "protocol_id": exporter.PROTOCOL_ID,
        "scope": {
            "stage": "D0-A",
            "no_validation_split": True,
            "use_validation_payload": False,
            "use_test_payload": False,
            "validation_payload_opens_required": 0,
            "test_payload_opens_required": 0,
        },
        "training": {
            "architecture": "MSHNet_NSFPN",
            "epochs": 1000,
            "official_train_only": True,
            "test_evaluation_during_training": False,
            "validation_evaluation_during_training": False,
            "checkpoint_selection": {
                "rule": "fixed_final_epoch",
                "epoch": 1000,
                "output_name": exporter.SOURCE_NAME,
            },
        },
        "datasets": {
            "TOY-SIRST": {
                "root": "datasets/TOY-SIRST",
                "train_split": "datasets/TOY-SIRST/img_idx/train.txt",
                "train_split_sha256": train_split_sha,
                "train_images": 32,
                "train_corpus_manifest_sha256": corpus_sha,
                # These poison metadata-only paths intentionally do not exist.
                "test_split_metadata_only": "must-not-open/test.txt",
                "test_images_metadata_only": 99,
            }
        },
    }
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    protocol_sha = _sha(protocol_path)

    run_config = {
        "schema_version": 1,
        "protocol_id": exporter.PROTOCOL_ID,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "host_architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "dataset": "TOY-SIRST",
        "root": str(repository / "datasets" / "TOY-SIRST"),
        "train_split": str(train_split),
        "expected_train_split_sha256": train_split_sha,
        "expected_train_images": 32,
        "expected_train_corpus_manifest_sha256": corpus_sha,
        "expected_state_dict_keys": 505,
        "epochs": 1000,
        "batch_size": 16,
        "data_mode": "full_train_only",
        "train_only_smoke": False,
        "max_train_batches": None,
        "checkpoint_selection": "fixed_final_epoch_train_only",
        "test_payload_access_allowed": False,
        "validation_payload_access_allowed": False,
        "output_dir": str(output_dir),
    }
    split_manifest = {
        "role": "official_train_only",
        "train_count": 32,
        "train_split_sha256": train_split_sha,
        "train_corpus_manifest_sha256": corpus_sha,
        "test_split_reads": 0,
        "test_image_opens": 0,
        "test_mask_opens": 0,
        "validation_split_reads": 0,
        "validation_image_opens": 0,
        "validation_mask_opens": 0,
    }
    access_firewall = {
        "implementation_has_test_loader": False,
        "implementation_has_validation_loader": False,
        "test_split_reads": 0,
        "test_image_opens": 0,
        "test_mask_opens": 0,
        "validation_split_reads": 0,
        "validation_image_opens": 0,
        "validation_mask_opens": 0,
    }
    runtime_sha256 = {
        "train_cr_sitta_d0a.py": _sha(repository / "train_cr_sitta_d0a.py"),
        "model/MSHNet_NSFPN.py": _sha(repository / "model/MSHNet_NSFPN.py"),
        "configs/cr_sitta_d0a_train_v2.yaml": protocol_sha,
    }
    contract = {
        "run_config": run_config,
        "split_manifest": split_manifest,
        "access_firewall": access_firewall,
        "runtime_sha256": runtime_sha256,
    }
    contract_path = output_dir / "run_contract.json"
    _write_json(contract_path, contract)

    smoke_path = repository / "results" / "smoke" / "SMOKE_GATE.json"
    _write_json(smoke_path, {"status": "passed_engineering_gate"})
    freeze = {
        "schema_version": 1,
        "protocol_id": exporter.PROTOCOL_ID,
        "status": "runtime_bytes_frozen_full_training_in_progress",
        "science_result": False,
        "formal_test_authorized": False,
        "tta_authorized": False,
        "smoke_gate": {
            "path": str(smoke_path.relative_to(repository)),
            "sha256": _sha(smoke_path),
            "status": "passed_engineering_gate",
        },
        "runtime_sha256": runtime_sha256,
        "started_run_contract": {
            "dataset": "TOY-SIRST",
            "path": str(contract_path.relative_to(repository)),
            "sha256": _sha(contract_path),
        },
        "known_recovery_limitations": {
            "safe_weights_only_inference_export_required": True,
        },
    }
    freeze_path = repository / "results" / "cr_sitta" / "d0a" / "FULL_TRAIN_FREEZE.json"
    _write_json(freeze_path, freeze)

    expected_steps = (32 // 16) * 1000
    source_payload: dict[str, Any] = {
        "schema_version": 1,
        "architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "test_selected": False,
        "selection_rule": "fixed_final_epoch_train_only",
        "epoch": 1000,
        "global_optimizer_step": expected_steps,
        "state_dict": _state_dict(),
        "optimizer": {"state": {0: {"sum": torch.ones(1)}}, "param_groups": []},
        "latest_train_metrics": {
            "epoch": 1000,
            "ending_optimizer_step": expected_steps,
        },
        "split_manifest": split_manifest,
        "run_config": run_config,
        "rng_state": {
            "python": (3, (1, 2, 3), None),
            "numpy": np.random.RandomState(7).get_state(),
            "torch_cpu": torch.get_rng_state(),
        },
    }
    source_path = output_dir / exporter.SOURCE_NAME
    torch.save(source_payload, source_path)
    summary = {
        "dataset": "TOY-SIRST",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "data_mode": "full_train_only",
        "completed_epochs": 1000,
        "global_optimizer_steps": expected_steps,
        "fixed_final_checkpoint": str(source_path),
        "test_selected": False,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
    }
    _write_json(output_dir / "summary.json", summary)
    return {
        "repository": repository,
        "output_dir": output_dir,
        "source": source_path,
        "source_payload": source_payload,
        "source_sha256": _sha(source_path),
        "contract": contract_path,
        "freeze": freeze_path,
        "protocol": protocol_path,
        "train_split": train_split,
    }


def _export(fixture: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    arguments = {
        "expected_source_sha256": _sha(fixture["source"]),
        "expected_run_contract_sha256": _sha(fixture["contract"]),
        "expected_full_train_freeze_sha256": _sha(fixture["freeze"]),
        "trusted_local_source": True,
        "repository": fixture["repository"],
        "protocol_config": fixture["protocol"],
        "train_split": fixture["train_split"],
        "state_dict_validators": (
            _strict_original_validator,
            _adaptable_source_validator,
        ),
    }
    arguments.update(overrides)
    return exporter.export_safe_checkpoint(
        fixture["source"], fixture["contract"], fixture["freeze"], **arguments
    )


def test_valid_export_is_weights_only_cpu_exact_and_hash_bound(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = _export(fixture)
    safe_path = fixture["output_dir"] / exporter.SAFE_NAME
    receipt_path = fixture["output_dir"] / exporter.RECEIPT_NAME

    assert safe_path.is_file()
    assert receipt_path.is_file()
    safe = torch.load(safe_path, map_location="cpu", weights_only=True)
    assert set(safe) == {"provenance", "state_dict"}
    assert len(safe["state_dict"]) == 505
    assert all(value.device.type == "cpu" for value in safe["state_dict"].values())
    json.dumps(safe["provenance"], allow_nan=False)

    def mapping_keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(
                *(mapping_keys(child) for child in value.values())
            )
        if isinstance(value, list):
            return set().union(*(mapping_keys(child) for child in value))
        return set()

    assert not {
        "optimizer",
        "rng_state",
        "latest_train_metrics",
    }.intersection(mapping_keys(safe["provenance"]))
    assert safe["provenance"]["test_selected"] is False
    assert receipt["artifacts"]["source_checkpoint"]["sha256"] == _sha(
        fixture["source"]
    )
    assert receipt["artifacts"]["safe_checkpoint"]["sha256"] == _sha(safe_path)
    assert receipt["artifacts"]["run_contract"]["sha256"] == _sha(
        fixture["contract"]
    )
    assert receipt["artifacts"]["full_train_freeze"]["sha256"] == _sha(
        fixture["freeze"]
    )
    assert receipt["artifacts"]["exporter"]["sha256"] == _sha(
        Path(exporter.__file__)
    )
    assert receipt["checkpoint_contract"]["repository_model_loads_verified"] is False
    assert receipt["checkpoint_contract"]["repository_model_loads"] == []
    assert not (safe_path.stat().st_mode & 0o222)
    assert not (receipt_path.stat().st_mode & 0o222)
    assert exporter.verify_safe_export(
        receipt_path,
        state_dict_validators=(
            _strict_original_validator,
            _adaptable_source_validator,
        ),
    ) == receipt


@pytest.mark.parametrize(
    ("override_name", "message"),
    [
        ("expected_source_sha256", "source checkpoint hash"),
        ("expected_run_contract_sha256", "externally anchored run_contract"),
        (
            "expected_full_train_freeze_sha256",
            "externally anchored FULL_TRAIN_FREEZE",
        ),
    ],
)
def test_external_hash_anchors_are_checked_before_general_pickle_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_name: str,
    message: str,
) -> None:
    fixture = _fixture(tmp_path)
    called = False

    def forbidden_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("untrusted checkpoint must not be unpickled")

    monkeypatch.setattr(exporter.torch, "load", forbidden_load)
    with pytest.raises(exporter.SafeCheckpointExportError, match=message):
        _export(fixture, **{override_name: "0" * 64})
    assert called is False
    assert not (fixture["output_dir"] / exporter.SAFE_NAME).exists()
    assert not (fixture["output_dir"] / exporter.RECEIPT_NAME).exists()


def test_explicit_trust_acknowledgement_is_required(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(exporter.SafeCheckpointExportError, match="general pickle"):
        _export(fixture, trusted_local_source=False)
    assert not (fixture["output_dir"] / exporter.SAFE_NAME).exists()


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("method_stage", "D0-B", "source.method_stage"),
        ("epoch", 999, "source.epoch"),
        ("test_selected", True, "source.test_selected"),
        ("selection_rule", "best_miou", "source.selection_rule"),
    ],
)
def test_rejects_wrong_or_incomplete_source_contract(
    tmp_path: Path, field: str, bad_value: Any, message: str
) -> None:
    fixture = _fixture(tmp_path)
    fixture["source_payload"][field] = bad_value
    torch.save(fixture["source_payload"], fixture["source"])
    with pytest.raises(exporter.SafeCheckpointExportError, match=message):
        _export(fixture)
    assert not (fixture["output_dir"] / exporter.SAFE_NAME).exists()
    assert not (fixture["output_dir"] / exporter.RECEIPT_NAME).exists()


def test_rejects_protocol_hash_drift_without_unpickling_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    fixture["protocol"].write_text("drift: true\n", encoding="utf-8")
    monkeypatch.setattr(
        exporter.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("source was loaded before protocol checks"),
    )
    with pytest.raises(exporter.SafeCheckpointExportError, match="protocol_sha256"):
        _export(fixture)
    assert not (fixture["output_dir"] / exporter.SAFE_NAME).exists()


def test_rejects_train_split_and_archive_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["train_split"].write_text("different\n", encoding="utf-8")
    with pytest.raises(exporter.SafeCheckpointExportError, match="train_split_sha256"):
        _export(fixture)
    assert not (fixture["output_dir"] / exporter.SAFE_NAME).exists()


def test_rejects_run_contract_hash_disagreement_with_freeze(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    contract = json.loads(fixture["contract"].read_text(encoding="utf-8"))
    contract["untrusted_post_start_field"] = True
    _write_json(fixture["contract"], contract)
    fixture["source_payload"]["run_config"] = contract["run_config"]
    torch.save(fixture["source_payload"], fixture["source"])
    with pytest.raises(exporter.SafeCheckpointExportError, match="run_contract hash"):
        _export(fixture)
    assert not (fixture["output_dir"] / exporter.SAFE_NAME).exists()


def test_validator_failure_publishes_neither_artifact(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    def rejecting_validator(state_dict: dict[str, Tensor]) -> None:
        assert len(state_dict) == 505
        raise RuntimeError("toy strict-load failure")

    with pytest.raises(RuntimeError, match="toy strict-load failure"):
        _export(fixture, state_dict_validators=(rejecting_validator,))
    assert not (fixture["output_dir"] / exporter.SAFE_NAME).exists()
    assert not (fixture["output_dir"] / exporter.RECEIPT_NAME).exists()
    assert not list(fixture["output_dir"].glob(".*.staging"))


def test_refuses_to_overwrite_existing_safe_artifacts(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _export(fixture)
    safe = fixture["output_dir"] / exporter.SAFE_NAME
    receipt = fixture["output_dir"] / exporter.RECEIPT_NAME
    safe_sha = _sha(safe)
    receipt_sha = _sha(receipt)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _export(fixture)
    assert _sha(safe) == safe_sha
    assert _sha(receipt) == receipt_sha


def test_missing_completion_summary_rejects_epoch_checkpoint(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["output_dir"] / "summary.json").unlink()
    with pytest.raises(exporter.SafeCheckpointExportError, match="completion summary"):
        _export(fixture)
    assert not (fixture["output_dir"] / exporter.SAFE_NAME).exists()


def test_safe_verifier_detects_post_export_tensor_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _export(fixture)
    safe = fixture["output_dir"] / exporter.SAFE_NAME
    receipt = fixture["output_dir"] / exporter.RECEIPT_NAME
    safe.chmod(0o644)
    with safe.open("ab") as handle:
        handle.write(b"drift")
    with pytest.raises(exporter.SafeCheckpointExportError, match="artifact hash drift"):
        exporter.verify_safe_export(receipt)
