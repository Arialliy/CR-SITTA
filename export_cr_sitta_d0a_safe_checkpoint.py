#!/usr/bin/env python3
"""Fail-closed, weights-only export for a completed CR-SITTA D0-A run.

The 1000-epoch training checkpoint is a trusted-local resume artifact: it
contains optimizer and Python/NumPy/PyTorch RNG objects and therefore is not a
production inference artifact.  This exporter verifies the frozen train-only
contract, loads that explicitly trusted input once, and publishes a new
checkpoint containing exactly a JSON-safe provenance dictionary and a CPU
tensor state_dict.

Publication never replaces an existing path.  ``SAFE_EXPORT.json`` is written
last and is the completion sentinel for the two-file export.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import Tensor


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_NAME = "epoch_1000_train_only.pth.tar"
SAFE_NAME = "epoch_1000_train_only_safe.pth.tar"
RECEIPT_NAME = "SAFE_EXPORT.json"
EXPECTED_STATE_DICT_KEYS = 505
EXPECTED_EPOCHS = 1000
PROTOCOL_ID = "cr-sitta-d0a-supervised-lfhf-train-1000e-v2"
SAFE_ARTIFACT_TYPE = "cr_sitta_d0a_safe_checkpoint_v1"
RECEIPT_TYPE = "cr_sitta_d0a_safe_export_receipt_v1"
JSON_SEPARATORS = (",", ":")

StateDictValidator = Callable[[Mapping[str, Tensor]], None]


class SafeCheckpointExportError(RuntimeError):
    """Raised when an export contract fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_expected_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SafeCheckpointExportError(
            f"expected {label} SHA-256 must be 64 lowercase hexadecimal characters"
        )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=JSON_SEPARATORS,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SafeCheckpointExportError(f"{label} must be a mapping")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise SafeCheckpointExportError(
            f"{label} mismatch: expected {expected!r}, got {value!r}"
        )


def _regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink():
        raise SafeCheckpointExportError(f"{label} must not be a symlink: {candidate}")
    if not candidate.is_file():
        raise SafeCheckpointExportError(f"{label} is not a regular file: {candidate}")
    return candidate.resolve(strict=True)


def _new_output_path(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    parent = candidate.parent.resolve(strict=True)
    candidate = parent / candidate.name
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"refusing to overwrite {label}: {candidate}")
    return candidate


def _declared_path(value: Any, repository: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SafeCheckpointExportError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository / path
    return _regular_file(path, label)


def _declared_directory(value: Any, repository: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SafeCheckpointExportError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository / path
    path = path.absolute()
    if path.is_symlink() or not path.is_dir():
        raise SafeCheckpointExportError(f"{label} is not a regular directory: {path}")
    return path.resolve(strict=True)


def _same_path(left: Path, right: Path) -> bool:
    return left.resolve(strict=False) == right.resolve(strict=False)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _regular_file(path, label)
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SafeCheckpointExportError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise SafeCheckpointExportError(f"{label} JSON root must be an object")
    return value, _sha256_bytes(payload)


def _read_yaml(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _regular_file(path, label)
    payload = path.read_bytes()
    try:
        value = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise SafeCheckpointExportError(f"{label} is not valid YAML: {path}") from error
    if not isinstance(value, dict):
        raise SafeCheckpointExportError(f"{label} YAML root must be a mapping")
    return value, _sha256_bytes(payload)


def _json_clone(value: Any, label: str) -> Any:
    """Return a JSON-only clone and reject tuples, tensors, NaN, and objects."""

    def validate(item: Any, location: str) -> None:
        if item is None or isinstance(item, (str, bool, int)):
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise SafeCheckpointExportError(
                    f"{label}{location} contains a non-finite float"
                )
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate(child, f"{location}[{index}]")
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise SafeCheckpointExportError(
                        f"{label}{location} contains a non-string dictionary key"
                    )
                validate(child, f"{location}.{key}")
            return
        raise SafeCheckpointExportError(
            f"{label}{location} is not JSON-safe: {type(item).__name__}"
        )

    validate(value, "")
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=JSON_SEPARATORS,
            allow_nan=False,
        )
    )


def _resolve_runtime_path(repository: Path, declared: str) -> Path:
    path = Path(declared).expanduser()
    if not path.is_absolute():
        path = repository / path
    return _regular_file(path, f"frozen runtime {declared!r}")


def _validate_hash_map(
    hashes: Any,
    *,
    repository: Path,
    label: str,
) -> dict[Path, str]:
    mapping = _require_mapping(hashes, label)
    if not mapping:
        raise SafeCheckpointExportError(f"{label} must not be empty")
    resolved: dict[Path, str] = {}
    for declared, expected_sha in mapping.items():
        if not isinstance(declared, str) or not isinstance(expected_sha, str):
            raise SafeCheckpointExportError(f"{label} entries must be string:string")
        path = _resolve_runtime_path(repository, declared)
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise SafeCheckpointExportError(
                f"{label} hash drift for {declared}: expected {expected_sha}, "
                f"got {actual_sha}"
            )
        if path in resolved and resolved[path] != expected_sha:
            raise SafeCheckpointExportError(
                f"{label} contains conflicting bindings for {path}"
            )
        resolved[path] = expected_sha
    return resolved


def _binding_for_path(
    bindings: Mapping[Path, str], path: Path, label: str
) -> str:
    target = path.resolve(strict=True)
    if target not in bindings:
        raise SafeCheckpointExportError(f"{label} is not covered by frozen hashes: {path}")
    return bindings[target]


def _check_access_firewall(contract: Mapping[str, Any]) -> dict[str, Any]:
    firewall = _require_mapping(contract.get("access_firewall"), "access_firewall")
    expected = {
        "implementation_has_test_loader": False,
        "implementation_has_validation_loader": False,
        "test_split_reads": 0,
        "test_image_opens": 0,
        "test_mask_opens": 0,
        "validation_split_reads": 0,
        "validation_image_opens": 0,
        "validation_mask_opens": 0,
    }
    for key, value in expected.items():
        _require_exact(firewall.get(key), value, f"access_firewall.{key}")
    return dict(_json_clone(dict(firewall), "access_firewall"))


def _validate_protocol(
    protocol: Mapping[str, Any],
    *,
    dataset: str,
    run_config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> None:
    _require_exact(protocol.get("schema_version"), 1, "protocol.schema_version")
    _require_exact(protocol.get("protocol_id"), PROTOCOL_ID, "protocol.protocol_id")
    scope = _require_mapping(protocol.get("scope"), "protocol.scope")
    _require_exact(scope.get("stage"), "D0-A", "protocol.scope.stage")
    _require_exact(scope.get("no_validation_split"), True, "no_validation_split")
    _require_exact(scope.get("use_validation_payload"), False, "use_validation_payload")
    _require_exact(scope.get("use_test_payload"), False, "use_test_payload")
    _require_exact(scope.get("validation_payload_opens_required"), 0, "validation opens")
    _require_exact(scope.get("test_payload_opens_required"), 0, "test opens")

    training = _require_mapping(protocol.get("training"), "protocol.training")
    _require_exact(training.get("architecture"), "MSHNet_NSFPN", "architecture")
    _require_exact(training.get("epochs"), EXPECTED_EPOCHS, "training.epochs")
    _require_exact(training.get("official_train_only"), True, "official_train_only")
    _require_exact(
        training.get("test_evaluation_during_training"),
        False,
        "test_evaluation_during_training",
    )
    _require_exact(
        training.get("validation_evaluation_during_training"),
        False,
        "validation_evaluation_during_training",
    )
    checkpoint_selection = _require_mapping(
        training.get("checkpoint_selection"), "training.checkpoint_selection"
    )
    _require_exact(checkpoint_selection.get("rule"), "fixed_final_epoch", "selection rule")
    _require_exact(checkpoint_selection.get("epoch"), EXPECTED_EPOCHS, "selection epoch")
    _require_exact(checkpoint_selection.get("output_name"), SOURCE_NAME, "output_name")

    datasets = _require_mapping(protocol.get("datasets"), "protocol.datasets")
    dataset_config = _require_mapping(datasets.get(dataset), f"protocol.datasets.{dataset}")
    checks = (
        ("train_split_sha256", "expected_train_split_sha256"),
        ("train_images", "expected_train_images"),
        ("train_corpus_manifest_sha256", "expected_train_corpus_manifest_sha256"),
    )
    for protocol_key, run_key in checks:
        _require_exact(
            dataset_config.get(protocol_key),
            run_config.get(run_key),
            f"protocol dataset {protocol_key}",
        )
    _require_exact(
        split_manifest.get("train_split_sha256"),
        dataset_config.get("train_split_sha256"),
        "split_manifest train hash",
    )
    _require_exact(
        split_manifest.get("train_count"),
        dataset_config.get("train_images"),
        "split_manifest train count",
    )


def _load_trusted_checkpoint(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    _validate_expected_sha256(expected_sha256, "source checkpoint")
    before = sha256_file(path)
    if before != expected_sha256:
        raise SafeCheckpointExportError(
            f"source checkpoint hash mismatch: expected {expected_sha256}, got {before}"
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise SafeCheckpointExportError(
            f"trusted local source checkpoint could not be loaded: {path}"
        ) from error
    after = sha256_file(path)
    if after != before:
        raise SafeCheckpointExportError("source checkpoint changed while it was loaded")
    return _require_mapping(payload, "source checkpoint")


def _cpu_tensor_state_dict(value: Any) -> dict[str, Tensor]:
    state_dict = _require_mapping(value, "source state_dict")
    if len(state_dict) != EXPECTED_STATE_DICT_KEYS:
        raise SafeCheckpointExportError(
            "source state_dict key-count mismatch: expected "
            f"{EXPECTED_STATE_DICT_KEYS}, got {len(state_dict)}"
        )
    result: dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        if not isinstance(key, str) or not key:
            raise SafeCheckpointExportError("state_dict keys must be non-empty strings")
        if not isinstance(tensor, Tensor):
            raise SafeCheckpointExportError(
                f"state_dict[{key!r}] is not a tensor: {type(tensor).__name__}"
            )
        cpu_tensor = tensor.detach().cpu().clone()
        if (cpu_tensor.is_floating_point() or cpu_tensor.is_complex()) and not bool(
            torch.isfinite(cpu_tensor).all().item()
        ):
            raise SafeCheckpointExportError(
                f"state_dict[{key!r}] contains NaN or Inf"
            )
        result[key] = cpu_tensor
    if len(result) != EXPECTED_STATE_DICT_KEYS:
        raise SafeCheckpointExportError("state_dict contains duplicate/non-string keys")
    return result


def _validate_source_payload(
    source: Mapping[str, Any],
    *,
    run_config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> tuple[dict[str, Tensor], int]:
    required_keys = {
        "schema_version",
        "architecture",
        "method_name",
        "method_stage",
        "development_only",
        "test_selected",
        "selection_rule",
        "epoch",
        "global_optimizer_step",
        "state_dict",
        "optimizer",
        "latest_train_metrics",
        "split_manifest",
        "run_config",
        "rng_state",
    }
    missing = sorted(required_keys - set(source))
    if missing:
        raise SafeCheckpointExportError(f"source checkpoint is incomplete; missing={missing}")
    unexpected = sorted(set(source) - required_keys)
    if unexpected:
        raise SafeCheckpointExportError(
            f"source checkpoint schema has unexpected keys: {unexpected}"
        )
    _require_exact(source.get("schema_version"), 1, "source.schema_version")
    _require_exact(source.get("architecture"), "MSHNet_NSFPN", "source.architecture")
    _require_exact(source.get("method_name"), "CR-SITTA", "source.method_name")
    _require_exact(source.get("method_stage"), "D0-A", "source.method_stage")
    _require_exact(source.get("development_only"), True, "source.development_only")
    _require_exact(source.get("test_selected"), False, "source.test_selected")
    _require_exact(
        source.get("selection_rule"),
        "fixed_final_epoch_train_only",
        "source.selection_rule",
    )
    _require_exact(source.get("epoch"), EXPECTED_EPOCHS, "source.epoch")
    if source.get("run_config") != run_config:
        raise SafeCheckpointExportError("source run_config differs from run_contract")
    if source.get("split_manifest") != split_manifest:
        raise SafeCheckpointExportError("source split_manifest differs from run_contract")

    latest = _require_mapping(source.get("latest_train_metrics"), "latest_train_metrics")
    _require_exact(latest.get("epoch"), EXPECTED_EPOCHS, "latest_train_metrics.epoch")
    global_step = source.get("global_optimizer_step")
    if type(global_step) is not int or global_step <= 0:
        raise SafeCheckpointExportError("global_optimizer_step must be a positive integer")
    _require_exact(
        latest.get("ending_optimizer_step"),
        global_step,
        "latest_train_metrics.ending_optimizer_step",
    )
    train_images = int(run_config["expected_train_images"])
    batch_size = int(run_config["batch_size"])
    expected_steps = (train_images // batch_size) * EXPECTED_EPOCHS
    _require_exact(global_step, expected_steps, "source.global_optimizer_step")
    return _cpu_tensor_state_dict(source.get("state_dict")), global_step


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    source_path: Path,
    dataset: str,
    global_step: int,
) -> None:
    _require_exact(summary.get("dataset"), dataset, "summary.dataset")
    _require_exact(summary.get("method_name"), "CR-SITTA", "summary.method_name")
    _require_exact(summary.get("method_stage"), "D0-A", "summary.method_stage")
    _require_exact(summary.get("data_mode"), "full_train_only", "summary.data_mode")
    _require_exact(summary.get("completed_epochs"), EXPECTED_EPOCHS, "completed_epochs")
    _require_exact(summary.get("global_optimizer_steps"), global_step, "summary steps")
    _require_exact(summary.get("test_selected"), False, "summary.test_selected")
    _require_exact(summary.get("validation_payload_opens"), 0, "summary validation opens")
    _require_exact(summary.get("test_payload_opens"), 0, "summary test opens")
    declared_final = summary.get("fixed_final_checkpoint")
    if not isinstance(declared_final, str) or not _same_path(
        Path(declared_final), source_path
    ):
        raise SafeCheckpointExportError(
            "summary fixed_final_checkpoint does not identify the source checkpoint"
        )


def _validate_safe_payload(value: Any) -> tuple[dict[str, Any], Mapping[str, Tensor]]:
    payload = _require_mapping(value, "safe checkpoint")
    if set(payload) != {"provenance", "state_dict"}:
        raise SafeCheckpointExportError(
            "safe checkpoint must contain exactly provenance and state_dict"
        )
    provenance = _json_clone(payload.get("provenance"), "safe provenance")
    provenance = dict(_require_mapping(provenance, "safe provenance"))
    _require_exact(
        provenance.get("artifact_type"), SAFE_ARTIFACT_TYPE, "safe artifact_type"
    )
    _require_exact(provenance.get("method_name"), "CR-SITTA", "safe method_name")
    _require_exact(provenance.get("method_stage"), "D0-A", "safe method_stage")
    _require_exact(provenance.get("epoch"), EXPECTED_EPOCHS, "safe epoch")
    _require_exact(provenance.get("test_selected"), False, "safe test_selected")
    state = _require_mapping(payload.get("state_dict"), "safe state_dict")
    if len(state) != EXPECTED_STATE_DICT_KEYS:
        raise SafeCheckpointExportError(
            f"safe state_dict must contain {EXPECTED_STATE_DICT_KEYS} keys"
        )
    for key, tensor in state.items():
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            raise SafeCheckpointExportError("safe state_dict must be string-to-tensor")
        if tensor.device.type != "cpu":
            raise SafeCheckpointExportError(f"safe state_dict tensor is not on CPU: {key}")
    return provenance, state


def _weights_only_load(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise SafeCheckpointExportError(
            f"safe checkpoint failed torch.load(weights_only=True): {path}"
        ) from error
    return _require_mapping(value, "weights-only checkpoint")


def _run_validators(
    state_dict: Mapping[str, Tensor], validators: Sequence[StateDictValidator]
) -> list[str]:
    names: list[str] = []
    for index, validator in enumerate(validators):
        if not callable(validator):
            raise TypeError(f"state_dict validator {index} is not callable")
        validator(state_dict)
        names.append(getattr(validator, "__name__", type(validator).__name__))
    return names


def validate_repository_models(state_dict: Mapping[str, Tensor]) -> None:
    """Perform both required repository model loads without running CUDA."""

    from model.MSHNet_NSFPN import MSHNet_NSFPN
    from model.MSHNet_NSFPN_adaptable import MSHNetNSFPNAdaptable

    original = MSHNet_NSFPN(3)
    result = original.load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise SafeCheckpointExportError("original MSHNet_NSFPN strict load was not exact")
    del original

    adaptable = MSHNetNSFPNAdaptable(3)
    result = adaptable.load_source_state_dict(state_dict)
    if result.missing_keys or result.unexpected_keys:
        raise SafeCheckpointExportError(
            "MSHNetNSFPNAdaptable source-state load was not exact"
        )
    del adaptable


def _write_torch_staging(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _write_bytes_staging(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _publish_noreplace(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite export artifact: {destination}")
    os.link(staging, destination)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _artifact(path: Path, sha256: str) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256}


def _validate_live_bindings(bindings: Mapping[Path, str]) -> None:
    for path, expected in bindings.items():
        if path.is_symlink() or not path.is_file():
            raise SafeCheckpointExportError(f"bound input disappeared or became symlink: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise SafeCheckpointExportError(
                f"bound input changed before publication: {path}; "
                f"expected {expected}, got {actual}"
            )


def _prepare_contract(
    *,
    repository: Path,
    source_checkpoint: Path,
    run_contract_path: Path,
    full_train_freeze_path: Path,
    expected_run_contract_sha256: str,
    expected_full_train_freeze_sha256: str,
    protocol_config_path: Path | None,
    train_split_path: Path | None,
) -> dict[str, Any]:
    repository = repository.expanduser().resolve(strict=True)
    source_checkpoint = _regular_file(source_checkpoint, "source checkpoint")
    run_contract_path = _regular_file(run_contract_path, "run contract")
    full_train_freeze_path = _regular_file(
        full_train_freeze_path, "FULL_TRAIN_FREEZE"
    )
    if source_checkpoint.name != SOURCE_NAME:
        raise SafeCheckpointExportError(
            f"source checkpoint must be named exactly {SOURCE_NAME}"
        )

    contract, contract_sha = _read_json(run_contract_path, "run contract")
    freeze, freeze_sha = _read_json(full_train_freeze_path, "FULL_TRAIN_FREEZE")
    _validate_expected_sha256(expected_run_contract_sha256, "run contract")
    _validate_expected_sha256(
        expected_full_train_freeze_sha256, "FULL_TRAIN_FREEZE"
    )
    _require_exact(
        contract_sha,
        expected_run_contract_sha256,
        "externally anchored run_contract SHA-256",
    )
    _require_exact(
        freeze_sha,
        expected_full_train_freeze_sha256,
        "externally anchored FULL_TRAIN_FREEZE SHA-256",
    )
    run_config = _require_mapping(contract.get("run_config"), "run_contract.run_config")
    split_manifest = _require_mapping(
        contract.get("split_manifest"), "run_contract.split_manifest"
    )
    dataset = run_config.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise SafeCheckpointExportError("run_config.dataset must be a non-empty string")

    required_run_values = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "host_architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "epochs": EXPECTED_EPOCHS,
        "expected_state_dict_keys": EXPECTED_STATE_DICT_KEYS,
        "data_mode": "full_train_only",
        "train_only_smoke": False,
        "max_train_batches": None,
        "checkpoint_selection": "fixed_final_epoch_train_only",
        "test_payload_access_allowed": False,
        "validation_payload_access_allowed": False,
    }
    for key, expected in required_run_values.items():
        _require_exact(run_config.get(key), expected, f"run_config.{key}")
    _require_exact(split_manifest.get("role"), "official_train_only", "split role")
    for key in (
        "test_split_reads",
        "test_image_opens",
        "test_mask_opens",
        "validation_split_reads",
        "validation_image_opens",
        "validation_mask_opens",
    ):
        _require_exact(split_manifest.get(key), 0, f"split_manifest.{key}")
    firewall = _check_access_firewall(contract)

    output_dir_value = run_config.get("output_dir")
    if not isinstance(output_dir_value, str):
        raise SafeCheckpointExportError("run_config.output_dir must be a string")
    output_dir = Path(output_dir_value).expanduser().resolve(strict=True)
    if source_checkpoint.parent != output_dir:
        raise SafeCheckpointExportError(
            "source checkpoint is outside the immutable run_config.output_dir"
        )
    if run_contract_path != output_dir / "run_contract.json":
        raise SafeCheckpointExportError("run contract path does not match output directory")

    declared_protocol = _declared_path(
        run_config.get("protocol_path"), repository, "protocol config"
    )
    if protocol_config_path is not None:
        supplied_protocol = _regular_file(protocol_config_path, "protocol config")
        if supplied_protocol != declared_protocol:
            raise SafeCheckpointExportError("supplied protocol path differs from run_config")
    protocol_path = declared_protocol
    protocol, protocol_sha = _read_yaml(protocol_path, "protocol config")
    _require_exact(
        run_config.get("protocol_sha256"), protocol_sha, "run_config.protocol_sha256"
    )

    declared_split = _declared_path(
        run_config.get("train_split"), repository, "official train split"
    )
    if train_split_path is not None:
        supplied_split = _regular_file(train_split_path, "official train split")
        if supplied_split != declared_split:
            raise SafeCheckpointExportError("supplied train split differs from run_config")
    train_split = declared_split
    train_split_sha = sha256_file(train_split)
    _require_exact(
        run_config.get("expected_train_split_sha256"),
        train_split_sha,
        "run_config.expected_train_split_sha256",
    )
    _require_exact(
        split_manifest.get("train_split_sha256"),
        train_split_sha,
        "split_manifest.train_split_sha256",
    )
    _require_exact(
        split_manifest.get("train_count"),
        run_config.get("expected_train_images"),
        "split_manifest.train_count",
    )
    _require_exact(
        split_manifest.get("train_corpus_manifest_sha256"),
        run_config.get("expected_train_corpus_manifest_sha256"),
        "split_manifest.train_corpus_manifest_sha256",
    )

    protocol_datasets = _require_mapping(
        protocol.get("datasets"), "protocol.datasets"
    )
    protocol_dataset = _require_mapping(
        protocol_datasets.get(dataset), f"protocol.datasets.{dataset}"
    )
    run_dataset_root = _declared_directory(
        run_config.get("root"), repository, "run_config dataset root"
    )
    protocol_dataset_root = _declared_directory(
        protocol_dataset.get("root"), repository, "protocol dataset root"
    )
    if run_dataset_root != protocol_dataset_root:
        raise SafeCheckpointExportError("protocol/run_config dataset roots differ")
    protocol_train_split = _declared_path(
        protocol_dataset.get("train_split"), repository, "protocol train split"
    )
    if protocol_train_split != train_split:
        raise SafeCheckpointExportError("protocol/run_config train split paths differ")
    datasets_root = (repository / "datasets").resolve(strict=True)
    try:
        run_dataset_root.relative_to(datasets_root)
        train_split.relative_to(run_dataset_root)
    except ValueError as error:
        raise SafeCheckpointExportError(
            "D0-A export is not bound to repository datasets/<dataset> train split"
        ) from error

    split_archive = _regular_file(output_dir / "splits" / "train.txt", "archived train split")
    archive_sha = sha256_file(split_archive)
    _require_exact(archive_sha, train_split_sha, "archived train split hash")

    _validate_protocol(
        protocol,
        dataset=dataset,
        run_config=run_config,
        split_manifest=split_manifest,
    )

    _require_exact(freeze.get("schema_version"), 1, "freeze.schema_version")
    _require_exact(freeze.get("protocol_id"), PROTOCOL_ID, "freeze.protocol_id")
    _require_exact(
        freeze.get("status"),
        "runtime_bytes_frozen_full_training_in_progress",
        "freeze.status",
    )
    _require_exact(freeze.get("science_result"), False, "freeze.science_result")
    _require_exact(
        freeze.get("formal_test_authorized"), False, "freeze.formal_test_authorized"
    )
    _require_exact(freeze.get("tta_authorized"), False, "freeze.tta_authorized")
    limitations = _require_mapping(
        freeze.get("known_recovery_limitations"), "freeze.known_recovery_limitations"
    )
    _require_exact(
        limitations.get("safe_weights_only_inference_export_required"),
        True,
        "freeze safe export requirement",
    )

    freeze_runtime = _validate_hash_map(
        freeze.get("runtime_sha256"), repository=repository, label="freeze.runtime_sha256"
    )
    contract_runtime = _validate_hash_map(
        contract.get("runtime_sha256"),
        repository=repository,
        label="run_contract.runtime_sha256",
    )
    for path in set(freeze_runtime).intersection(contract_runtime):
        if freeze_runtime[path] != contract_runtime[path]:
            raise SafeCheckpointExportError(
                f"freeze/run_contract runtime hash disagreement for {path}"
            )
    _require_exact(
        _binding_for_path(freeze_runtime, protocol_path, "protocol config"),
        protocol_sha,
        "freeze protocol hash",
    )
    _require_exact(
        _binding_for_path(contract_runtime, protocol_path, "protocol config"),
        protocol_sha,
        "run_contract protocol hash",
    )

    smoke = _require_mapping(freeze.get("smoke_gate"), "freeze.smoke_gate")
    _require_exact(smoke.get("status"), "passed_engineering_gate", "smoke gate status")
    smoke_path = _declared_path(smoke.get("path"), repository, "SMOKE_GATE")
    smoke_sha = sha256_file(smoke_path)
    _require_exact(smoke.get("sha256"), smoke_sha, "freeze smoke gate hash")

    started_contract = freeze.get("started_run_contract")
    if isinstance(started_contract, Mapping) and started_contract.get("dataset") == dataset:
        frozen_contract_path = _declared_path(
            started_contract.get("path"), repository, "frozen started run contract"
        )
        if frozen_contract_path != run_contract_path:
            raise SafeCheckpointExportError("freeze identifies a different run contract")
        _require_exact(
            started_contract.get("sha256"), contract_sha, "freeze run_contract hash"
        )

    runner_path = repository / "train_cr_sitta_d0a.py"
    runner_sha = _binding_for_path(freeze_runtime, runner_path, "D0-A runner")
    _require_exact(
        _binding_for_path(contract_runtime, runner_path, "D0-A runner"),
        runner_sha,
        "runner hash across freeze/run_contract",
    )

    original_model_path = repository / "model" / "MSHNet_NSFPN.py"
    original_model_sha = _binding_for_path(
        freeze_runtime, original_model_path, "original model implementation"
    )
    adaptable_model_path = _regular_file(
        repository / "model" / "MSHNet_NSFPN_adaptable.py",
        "adaptable model implementation",
    )
    adaptable_model_sha = sha256_file(adaptable_model_path)

    summary_path = _regular_file(output_dir / "summary.json", "completion summary")
    summary, summary_sha = _read_json(summary_path, "completion summary")

    bindings: dict[Path, str] = {}
    for path, digest in (
        (run_contract_path, contract_sha),
        (full_train_freeze_path, freeze_sha),
        (protocol_path, protocol_sha),
        (train_split, train_split_sha),
        (split_archive, archive_sha),
        (smoke_path, smoke_sha),
        (summary_path, summary_sha),
        (adaptable_model_path, adaptable_model_sha),
    ):
        bindings[path] = digest
    bindings.update(freeze_runtime)
    bindings.update(contract_runtime)

    return {
        "repository": repository,
        "source_checkpoint": source_checkpoint,
        "run_contract_path": run_contract_path,
        "run_contract": contract,
        "run_contract_sha256": contract_sha,
        "full_train_freeze_path": full_train_freeze_path,
        "full_train_freeze_sha256": freeze_sha,
        "protocol_path": protocol_path,
        "protocol_sha256": protocol_sha,
        "train_split": train_split,
        "train_split_sha256": train_split_sha,
        "split_archive": split_archive,
        "split_archive_sha256": archive_sha,
        "smoke_path": smoke_path,
        "smoke_sha256": smoke_sha,
        "summary_path": summary_path,
        "summary": summary,
        "summary_sha256": summary_sha,
        "dataset": dataset,
        "run_config": run_config,
        "split_manifest": split_manifest,
        "access_firewall": firewall,
        "runner_path": runner_path.resolve(strict=True),
        "runner_sha256": runner_sha,
        "original_model_path": original_model_path.resolve(strict=True),
        "original_model_sha256": original_model_sha,
        "adaptable_model_path": adaptable_model_path,
        "adaptable_model_sha256": adaptable_model_sha,
        "bindings": bindings,
    }


def export_safe_checkpoint(
    source_checkpoint: Path,
    run_contract: Path,
    full_train_freeze: Path,
    *,
    expected_source_sha256: str,
    expected_run_contract_sha256: str,
    expected_full_train_freeze_sha256: str,
    trusted_local_source: bool,
    repository: Path = PROJECT_ROOT,
    protocol_config: Path | None = None,
    train_split: Path | None = None,
    output_checkpoint: Path | None = None,
    receipt_path: Path | None = None,
    state_dict_validators: Sequence[StateDictValidator] = (),
) -> dict[str, Any]:
    """Validate and atomically publish one safe D0-A checkpoint and receipt."""

    if not trusted_local_source:
        raise SafeCheckpointExportError(
            "source uses general pickle loading; pass trusted_local_source=True only "
            "for the locally produced, independently hashed training checkpoint"
        )
    context = _prepare_contract(
        repository=repository,
        source_checkpoint=source_checkpoint,
        run_contract_path=run_contract,
        full_train_freeze_path=full_train_freeze,
        expected_run_contract_sha256=expected_run_contract_sha256,
        expected_full_train_freeze_sha256=expected_full_train_freeze_sha256,
        protocol_config_path=protocol_config,
        train_split_path=train_split,
    )
    source_path: Path = context["source_checkpoint"]
    output = _new_output_path(
        output_checkpoint or source_path.with_name(SAFE_NAME), "safe checkpoint"
    )
    receipt = _new_output_path(
        receipt_path or source_path.with_name(RECEIPT_NAME), "safe export receipt"
    )
    if output != source_path.with_name(SAFE_NAME):
        raise SafeCheckpointExportError(
            f"safe checkpoint path must be exactly {source_path.with_name(SAFE_NAME)}"
        )
    if receipt != source_path.with_name(RECEIPT_NAME):
        raise SafeCheckpointExportError(
            f"receipt path must be exactly {source_path.with_name(RECEIPT_NAME)}"
        )

    source_sha = sha256_file(source_path)
    if source_sha != expected_source_sha256:
        raise SafeCheckpointExportError(
            f"source checkpoint hash mismatch: expected {expected_source_sha256}, "
            f"got {source_sha}"
        )
    source = _load_trusted_checkpoint(source_path, expected_source_sha256)
    state_dict, global_step = _validate_source_payload(
        source,
        run_config=context["run_config"],
        split_manifest=context["split_manifest"],
    )
    _validate_summary(
        context["summary"],
        source_path=source_path,
        dataset=context["dataset"],
        global_step=global_step,
    )

    exporter_path = _regular_file(Path(__file__), "safe checkpoint exporter")
    exporter_sha = sha256_file(exporter_path)
    context["bindings"][source_path] = source_sha
    context["bindings"][exporter_path] = exporter_sha

    provenance = {
        "schema_version": 1,
        "artifact_type": SAFE_ARTIFACT_TYPE,
        "architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "test_selected": False,
        "selection_rule": "fixed_final_epoch_train_only",
        "epoch": EXPECTED_EPOCHS,
        "global_optimizer_step": global_step,
        "dataset": context["dataset"],
        "state_dict_keys": EXPECTED_STATE_DICT_KEYS,
        "source_checkpoint_sha256": source_sha,
        "protocol_sha256": context["protocol_sha256"],
        "run_contract_sha256": context["run_contract_sha256"],
        "full_train_freeze_sha256": context["full_train_freeze_sha256"],
        "train_split_sha256": context["train_split_sha256"],
        "run_config": _json_clone(dict(context["run_config"]), "run_config"),
        "split_manifest": _json_clone(
            dict(context["split_manifest"]), "split_manifest"
        ),
        "access_firewall": _json_clone(
            dict(context["access_firewall"]), "access_firewall"
        ),
    }
    safe_payload = {
        "provenance": _json_clone(provenance, "provenance"),
        "state_dict": state_dict,
    }

    output_fd, output_staging_name = tempfile.mkstemp(
        prefix=f".{SAFE_NAME}.", suffix=".staging", dir=output.parent
    )
    os.close(output_fd)
    output_staging = Path(output_staging_name)
    output_staging.unlink()
    receipt_fd, receipt_staging_name = tempfile.mkstemp(
        prefix=f".{RECEIPT_NAME}.", suffix=".staging", dir=receipt.parent
    )
    os.close(receipt_fd)
    receipt_staging = Path(receipt_staging_name)
    receipt_staging.unlink()
    output_published = False
    receipt_published = False
    observed: dict[str, Any] | None = None
    try:
        _write_torch_staging(output_staging, safe_payload)
        staged = _weights_only_load(output_staging)
        staged_provenance, staged_state = _validate_safe_payload(staged)
        if staged_provenance != safe_payload["provenance"]:
            raise SafeCheckpointExportError("staged safe provenance changed on round trip")
        validator_names = _run_validators(staged_state, state_dict_validators)
        repository_models_verified = any(
            validator is validate_repository_models
            for validator in state_dict_validators
        )
        safe_sha = sha256_file(output_staging)

        receipt_payload = {
            "schema_version": 1,
            "receipt_type": RECEIPT_TYPE,
            "status": "published_weights_only_verified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": context["dataset"],
            "checkpoint_contract": {
                "architecture": "MSHNet_NSFPN",
                "method_name": "CR-SITTA",
                "method_stage": "D0-A",
                "epoch": EXPECTED_EPOCHS,
                "test_selected": False,
                "selection_rule": "fixed_final_epoch_train_only",
                "state_dict_keys": EXPECTED_STATE_DICT_KEYS,
                "all_state_dict_tensors_cpu": True,
                "torch_load_weights_only": True,
                "general_pickle_objects_removed": [
                    "optimizer",
                    "rng_state",
                    "latest_train_metrics",
                ],
                "repository_model_loads_verified": repository_models_verified,
                "repository_model_loads": (
                    [
                        "MSHNet_NSFPN.load_state_dict(strict=True)",
                        "MSHNetNSFPNAdaptable.load_source_state_dict",
                    ]
                    if repository_models_verified
                    else []
                ),
                "validators": validator_names,
            },
            "access_firewall": {
                "validation_split_reads": 0,
                "validation_image_opens": 0,
                "validation_mask_opens": 0,
                "test_split_reads": 0,
                "test_image_opens": 0,
                "test_mask_opens": 0,
            },
            "publication_contract": {
                "completion_sentinel": str(receipt),
                "completion_rule": (
                    "export is complete only when this canonical receipt and the "
                    "hash-bound safe checkpoint both exist and verify"
                ),
                "checkpoint_published_before_receipt": True,
                "existing_paths_never_replaced": True,
            },
            "artifacts": {
                "source_checkpoint": _artifact(source_path, source_sha),
                "safe_checkpoint": _artifact(output, safe_sha),
                "exporter": _artifact(exporter_path, exporter_sha),
                "run_contract": _artifact(
                    context["run_contract_path"], context["run_contract_sha256"]
                ),
                "full_train_freeze": _artifact(
                    context["full_train_freeze_path"],
                    context["full_train_freeze_sha256"],
                ),
                "protocol_config": _artifact(
                    context["protocol_path"], context["protocol_sha256"]
                ),
                "train_split": _artifact(
                    context["train_split"], context["train_split_sha256"]
                ),
                "archived_train_split": _artifact(
                    context["split_archive"], context["split_archive_sha256"]
                ),
                "completion_summary": _artifact(
                    context["summary_path"], context["summary_sha256"]
                ),
                "training_runner": _artifact(
                    context["runner_path"], context["runner_sha256"]
                ),
                "original_model_implementation": _artifact(
                    context["original_model_path"], context["original_model_sha256"]
                ),
                "adaptable_model_implementation": _artifact(
                    context["adaptable_model_path"],
                    context["adaptable_model_sha256"],
                ),
                "smoke_gate": _artifact(
                    context["smoke_path"], context["smoke_sha256"]
                ),
            },
        }
        receipt_payload = dict(_json_clone(receipt_payload, "receipt"))
        receipt_bytes = _canonical_json_bytes(receipt_payload)
        _write_bytes_staging(receipt_staging, receipt_bytes)

        _validate_live_bindings(context["bindings"])
        if sha256_file(output_staging) != safe_sha:
            raise SafeCheckpointExportError("staged safe checkpoint changed before publish")
        if receipt_staging.read_bytes() != receipt_bytes:
            raise SafeCheckpointExportError("staged receipt changed before publish")
        if output.exists() or output.is_symlink() or receipt.exists() or receipt.is_symlink():
            raise FileExistsError("safe checkpoint or receipt appeared before publication")

        _publish_noreplace(output_staging, output)
        output_published = True
        _publish_noreplace(receipt_staging, receipt)
        receipt_published = True
        _fsync_directory(output.parent)
        observed = verify_safe_export(receipt)
        if observed != receipt_payload:
            raise SafeCheckpointExportError(
                "published receipt differs from verified receipt"
            )
    except BaseException:
        # Roll back only exact inodes linked by this invocation.  A process crash
        # between the two links may still leave a checkpoint without its receipt;
        # the no-overwrite rule deliberately requires manual audit in that case.
        if receipt_published:
            try:
                if receipt.stat().st_ino == receipt_staging.stat().st_ino:
                    receipt.unlink()
            except OSError:
                pass
        if output_published:
            try:
                if output.stat().st_ino == output_staging.stat().st_ino:
                    output.unlink()
            except OSError:
                pass
        if output_published or receipt_published:
            _fsync_directory(output.parent)
        raise
    finally:
        output_staging.unlink(missing_ok=True)
        receipt_staging.unlink(missing_ok=True)

    if observed is None:
        raise SafeCheckpointExportError("safe export publication did not complete")
    return receipt_payload


def verify_safe_export(
    receipt_path: Path,
    *,
    state_dict_validators: Sequence[StateDictValidator] = (),
) -> dict[str, Any]:
    receipt_path = _regular_file(receipt_path, "safe export receipt")
    payload, _receipt_sha = _read_json(receipt_path, "safe export receipt")
    if receipt_path.read_bytes() != _canonical_json_bytes(payload):
        raise SafeCheckpointExportError("safe export receipt is not canonical JSON")
    _require_exact(payload.get("schema_version"), 1, "receipt.schema_version")
    _require_exact(payload.get("receipt_type"), RECEIPT_TYPE, "receipt.receipt_type")
    _require_exact(
        payload.get("status"), "published_weights_only_verified", "receipt.status"
    )
    artifacts = _require_mapping(payload.get("artifacts"), "receipt.artifacts")
    required = {
        "source_checkpoint",
        "safe_checkpoint",
        "exporter",
        "run_contract",
        "full_train_freeze",
    }
    if not required.issubset(artifacts):
        raise SafeCheckpointExportError(
            f"receipt lacks required hash bindings: {sorted(required - set(artifacts))}"
        )
    resolved_artifacts: dict[str, Path] = {}
    for name, raw_binding in artifacts.items():
        binding = _require_mapping(raw_binding, f"receipt.artifacts.{name}")
        path_value = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(path_value, str) or not isinstance(digest, str):
            raise SafeCheckpointExportError(f"invalid receipt binding for {name}")
        path = _regular_file(Path(path_value), f"receipt artifact {name}")
        if sha256_file(path) != digest:
            raise SafeCheckpointExportError(f"receipt artifact hash drift: {name}")
        resolved_artifacts[name] = path

    safe = _weights_only_load(resolved_artifacts["safe_checkpoint"])
    provenance, state = _validate_safe_payload(safe)
    _run_validators(state, state_dict_validators)
    for provenance_key, artifact_name in (
        ("source_checkpoint_sha256", "source_checkpoint"),
        ("protocol_sha256", "protocol_config"),
        ("run_contract_sha256", "run_contract"),
        ("full_train_freeze_sha256", "full_train_freeze"),
        ("train_split_sha256", "train_split"),
    ):
        artifact = _require_mapping(artifacts.get(artifact_name), artifact_name)
        _require_exact(
            provenance.get(provenance_key),
            artifact.get("sha256"),
            f"safe provenance {provenance_key}",
        )
    if receipt_path.stat().st_mode & 0o222:
        raise SafeCheckpointExportError("safe export receipt must be read-only")
    if resolved_artifacts["safe_checkpoint"].stat().st_mode & 0o222:
        raise SafeCheckpointExportError("safe checkpoint must be read-only")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--expected-run-contract-sha256", required=True)
    parser.add_argument("--full-train-freeze", type=Path, required=True)
    parser.add_argument("--expected-full-train-freeze-sha256", required=True)
    parser.add_argument("--repository", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--protocol-config", type=Path)
    parser.add_argument("--train-split", type=Path)
    parser.add_argument("--output-checkpoint", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--trust-local-source",
        action="store_true",
        help=(
            "acknowledge that the source is the trusted local D0-A training "
            "artifact whose independently recorded SHA-256 was supplied"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.trust_local_source:
        parser.error("--trust-local-source is required for general pickle input")
    receipt = export_safe_checkpoint(
        args.source_checkpoint,
        args.run_contract,
        args.full_train_freeze,
        expected_source_sha256=args.expected_source_sha256,
        expected_run_contract_sha256=args.expected_run_contract_sha256,
        expected_full_train_freeze_sha256=(
            args.expected_full_train_freeze_sha256
        ),
        trusted_local_source=True,
        repository=args.repository,
        protocol_config=args.protocol_config,
        train_split=args.train_split,
        output_checkpoint=args.output_checkpoint,
        receipt_path=args.receipt,
        state_dict_validators=(validate_repository_models,),
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
