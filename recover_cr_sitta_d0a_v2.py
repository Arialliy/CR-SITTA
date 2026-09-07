#!/usr/bin/env python3
"""Fail-closed recovery preparation for an interrupted CR-SITTA D0-A v2 run.

This utility is deliberately *not* a training runner and does not inspect any
validation or test payload.  It accepts only the trusted-local ``last.pth.tar``
inside an immutable full-train-only run directory, verifies the frozen run
contract and the complete train JSONL prefix, then publishes a copy with the
process RNG payload removed.  The original v2 runner restores all model and
optimizer state from that copy, but its RNG restore helper returns immediately
because ``rng_state`` is absent.  This is safe at an epoch boundary because the
v2 DataLoader and deterioration probes derive their seeds explicitly from the
epoch/protocol identifiers and the frozen model has no Dropout.

The source checkpoint contains Python/NumPy objects and is loaded with pickle.
It must therefore be a trusted local artifact produced by the frozen runner.
Publication is atomic and no-replace; ``RECOVERY.json`` is written last as the
completion sentinel.  A changed CUDA process, including custom CUDA kernels,
is explicitly not claimed to be bit-exact.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import torch
from torch import Tensor
from torch.optim import Adagrad

from tta.d0_secure_io import (
    SecureIOError,
    publish_directory_noreplace,
    read_stable_regular_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TOOL_PATH = Path(__file__).resolve()
PROTOCOL_ID = "cr-sitta-d0a-supervised-lfhf-train-1000e-v2"
EXPECTED_EPOCHS = 1000
SOURCE_NAME = "last.pth.tar"
FINAL_NAME = "epoch_1000_train_only.pth.tar"
RECOVERY_DIRECTORY_NAME = "recovery"
RECOVERY_RECEIPT_NAME = "RECOVERY.json"
FINAL_RECOVERY_RECEIPT_NAME = "FINAL_RECOVERY_REQUIRED.json"
FINAL_EXPORT_REQUIRED_EXIT_STATUS = 42
SECURE_IO_PATH = PROJECT_ROOT / "tta" / "d0_secure_io.py"

RECOVERY_ARTIFACT_TYPE = "cr_sitta_d0a_v2_rng_sanitized_resume_checkpoint"
RECOVERY_RECEIPT_TYPE = "cr_sitta_d0a_v2_interrupted_run_recovery_receipt"
FINAL_RECOVERY_RECEIPT_TYPE = "cr_sitta_d0a_v2_final_recovery_refusal"

REQUIRED_FROZEN_RUNTIME = (
    "train_cr_sitta_d0a.py",
    "configs/cr_sitta_d0a_train_v2.yaml",
    "train_fixed_split.py",
    "model/loss.py",
    "model/MSHNet_NSFPN.py",
    "model/NS_FPN.py",
    "model/diff_cross_attns.py",
    "tta/deteriorations/__init__.py",
    "tta/deteriorations/fourier_low_mask.py",
    "tta/deteriorations/high_frequency_noise.py",
    "tta/deteriorations/image_space.py",
    "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
    "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
)

ZERO_ACCESS_FIELDS = (
    "test_split_reads",
    "test_image_opens",
    "test_mask_opens",
    "validation_split_reads",
    "validation_image_opens",
    "validation_mask_opens",
)


class RecoveryError(RuntimeError):
    """A named fail-closed recovery refusal."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


ResumeValidator = Callable[[Mapping[str, Any], Mapping[str, Any]], None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RecoveryError(
            "D0A_NON_JSON_CONTRACT", "run configuration/manifest is not JSON-safe"
        ) from error
    return _sha256_bytes(payload)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryError("D0A_INVALID_STRUCTURE", f"{label} must be a mapping")
    return value


def _require_exact(value: Any, expected: Any, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise RecoveryError(
            "D0A_CONTRACT_MISMATCH",
            f"{label} mismatch: expected {expected!r}, got {value!r}",
        )


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RecoveryError(
            "D0A_INVALID_HASH", f"{label} must be lowercase hexadecimal SHA-256"
        )
    return value


def _require_expected_hash(actual: str, expected_value: Any, label: str) -> None:
    expected = _require_sha256(expected_value, f"expected {label}")
    if actual != expected:
        raise RecoveryError(
            "D0A_EXTERNAL_HASH_MISMATCH",
            f"{label} differs from external SHA-256 anchor: expected {expected}, got {actual}",
        )


def _regular_file(path: Path, label: str) -> Path:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink():
        raise RecoveryError("D0A_UNSAFE_PATH", f"{label} must not be a symlink: {candidate}")
    if not candidate.is_file():
        raise RecoveryError("D0A_MISSING_INPUT", f"{label} is not a file: {candidate}")
    return candidate.resolve(strict=True)


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _declared_file(value: Any, repository: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RecoveryError("D0A_INVALID_PATH", f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository / path
    return _regular_file(path, label)


def _display_path(path: Path, repository: Path) -> str:
    try:
        return str(path.relative_to(repository))
    except ValueError:
        return str(path)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = _regular_file(path, label)
    try:
        snapshot = read_stable_regular_file(path)
    except (OSError, ValueError, SecureIOError) as error:
        raise RecoveryError("D0A_UNSTABLE_INPUT", f"cannot stably read {label}") from error
    raw = snapshot.data
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryError("D0A_INVALID_JSON", f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RecoveryError("D0A_INVALID_STRUCTURE", f"{label} root must be an object")
    return value, snapshot.sha256


def _parse_python(path: Path, label: str) -> ast.Module:
    try:
        source = path.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise RecoveryError(
            "D0A_RUNTIME_SEMANTICS_UNVERIFIED", f"cannot parse {label}: {path}"
        ) from error


def _function(tree: ast.Module, name: str, label: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            if isinstance(node, ast.FunctionDef):
                return node
    raise RecoveryError(
        "D0A_RUNTIME_SEMANTICS_UNVERIFIED", f"{label} does not define {name}"
    )


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _validate_rng_restore_guard(runner_path: Path) -> None:
    tree = _parse_python(runner_path, "frozen D0-A runner")
    function = _function(tree, "_restore_rng_state", "frozen D0-A runner")
    if len(function.body) < 2 or not isinstance(function.body[0], ast.Assign):
        raise RecoveryError(
            "D0A_RNG_GUARD_UNVERIFIED",
            "_restore_rng_state does not begin with a guarded rng_state lookup",
        )
    assignment = function.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        raise RecoveryError("D0A_RNG_GUARD_UNVERIFIED", "unexpected RNG guard assignment")
    state_name = assignment.targets[0].id
    call = assignment.value
    if not (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "get"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "rng_state"
    ):
        raise RecoveryError(
            "D0A_RNG_GUARD_UNVERIFIED", "RNG restore does not use payload.get('rng_state')"
        )
    guard = function.body[1]
    if not (
        isinstance(guard, ast.If)
        and isinstance(guard.test, ast.UnaryOp)
        and isinstance(guard.test.op, ast.Not)
        and isinstance(guard.test.operand, ast.Name)
        and guard.test.operand.id == state_name
        and len(guard.body) == 1
        and isinstance(guard.body[0], ast.Return)
    ):
        raise RecoveryError(
            "D0A_RNG_GUARD_UNVERIFIED",
            "missing rng_state is not proven to return before RNG restoration",
        )


def _validate_explicit_seed_semantics(runner_path: Path, loader_path: Path) -> None:
    runner_tree = _parse_python(runner_path, "frozen D0-A runner")
    derive = _function(runner_tree, "derive_probe_seed", "frozen D0-A runner")
    required_arguments = {
        "protocol_id",
        "global_seed",
        "dataset_id",
        "human_epoch",
        "image_id",
        "probe_id",
    }
    argument_names = {argument.arg for argument in derive.args.args}
    used_names = {node.id for node in ast.walk(derive) if isinstance(node, ast.Name)}
    if not required_arguments.issubset(argument_names) or not required_arguments.issubset(
        used_names
    ):
        raise RecoveryError(
            "D0A_SEED_SCHEDULE_UNVERIFIED",
            "probe seed is not visibly derived from all frozen identifiers",
        )

    loader_tree = _parse_python(loader_path, "frozen train loader")
    loader = _function(loader_tree, "make_train_loader", "frozen train loader")
    seeded_per_epoch = False
    for node in ast.walk(loader):
        if not isinstance(node, ast.Call) or _dotted_name(node.func) is None:
            continue
        if not _dotted_name(node.func).endswith("manual_seed") or len(node.args) != 1:
            continue
        expression_nodes = tuple(ast.walk(node.args[0]))
        has_epoch = any(
            isinstance(child, ast.Name) and child.id == "human_epoch"
            for child in expression_nodes
        )
        has_config_seed = any(
            isinstance(child, ast.Constant) and child.value == "seed"
            for child in expression_nodes
        )
        if has_epoch and has_config_seed:
            seeded_per_epoch = True
            break
    if not seeded_per_epoch:
        raise RecoveryError(
            "D0A_SEED_SCHEDULE_UNVERIFIED",
            "train DataLoader seed is not visibly derived from config seed and human_epoch",
        )


def _validate_no_dropout(model_paths: list[Path]) -> None:
    if not model_paths:
        raise RecoveryError("D0A_DROPOUT_UNVERIFIED", "no frozen model sources were found")
    dropout_calls = {
        "dropout",
        "dropout1d",
        "dropout2d",
        "dropout3d",
        "alphadropout",
        "featurealphadropout",
    }
    for path in model_paths:
        tree = _parse_python(path, "frozen model source")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted_name(node.func)
            if dotted and dotted.rsplit(".", 1)[-1].lower() in dropout_calls:
                raise RecoveryError(
                    "D0A_MODEL_HAS_DROPOUT",
                    f"frozen model instantiates/calls Dropout in {path}",
                )
            if dotted and dotted.rsplit(".", 1)[-1] == "getattr" and len(node.args) >= 2:
                member = node.args[1]
                if isinstance(member, ast.Constant) and str(member.value).lower() in dropout_calls:
                    raise RecoveryError(
                        "D0A_MODEL_HAS_DROPOUT",
                        f"frozen model dynamically accesses Dropout in {path}",
                    )


def _validate_hash_map(
    value: Any, *, repository: Path, label: str
) -> tuple[dict[str, str], dict[Path, str]]:
    hashes = _require_mapping(value, label)
    if not hashes:
        raise RecoveryError("D0A_EMPTY_RUNTIME_HASHES", f"{label} must not be empty")
    declared_result: dict[str, str] = {}
    resolved_result: dict[Path, str] = {}
    for declared, expected_value in hashes.items():
        if not isinstance(declared, str) or not declared:
            raise RecoveryError("D0A_INVALID_RUNTIME_HASH", f"{label} has invalid path key")
        expected = _require_sha256(expected_value, f"{label}[{declared!r}]")
        path = Path(declared).expanduser()
        if not path.is_absolute():
            path = repository / path
        path = _regular_file(path, f"{label} runtime {declared!r}")
        actual = sha256_file(path)
        if actual != expected:
            raise RecoveryError(
                "D0A_RUNTIME_HASH_DRIFT",
                f"{label} hash drift for {declared}: expected {expected}, got {actual}",
            )
        if path in resolved_result and resolved_result[path] != expected:
            raise RecoveryError(
                "D0A_RUNTIME_HASH_CONFLICT", f"conflicting runtime bindings for {path}"
            )
        declared_result[declared] = expected
        resolved_result[path] = expected
    return declared_result, resolved_result


def _validate_access_firewall(contract: Mapping[str, Any]) -> dict[str, Any]:
    firewall = _require_mapping(contract.get("access_firewall"), "access_firewall")
    _require_exact(
        firewall.get("implementation_has_test_loader"),
        False,
        "access_firewall.implementation_has_test_loader",
    )
    _require_exact(
        firewall.get("implementation_has_validation_loader"),
        False,
        "access_firewall.implementation_has_validation_loader",
    )
    for field in ZERO_ACCESS_FIELDS:
        _require_exact(firewall.get(field), 0, f"access_firewall.{field}")
    return dict(firewall)


def _validate_run_config(run_config: Mapping[str, Any], run_dir: Path) -> None:
    required = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "host_architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "epochs": EXPECTED_EPOCHS,
        "data_mode": "full_train_only",
        "train_only_smoke": False,
        "max_train_batches": None,
        "probe_schedule": "deterministic_alternating_per_optimizer_step",
        "checkpoint_selection": "fixed_final_epoch_train_only",
        "validation_payload_access_allowed": False,
        "test_payload_access_allowed": False,
    }
    for field, expected in required.items():
        _require_exact(run_config.get(field), expected, f"run_config.{field}")
    dataset = run_config.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise RecoveryError("D0A_INVALID_DATASET", "run_config.dataset is invalid")
    for field in ("expected_train_images", "batch_size", "expected_state_dict_keys"):
        value = run_config.get(field)
        if type(value) is not int or value <= 0:
            raise RecoveryError("D0A_INVALID_RUN_CONFIG", f"run_config.{field} is invalid")
    seed = run_config.get("seed")
    if type(seed) is not int:
        raise RecoveryError("D0A_INVALID_RUN_CONFIG", "run_config.seed must be an integer")
    output_value = run_config.get("output_dir")
    if not isinstance(output_value, str):
        raise RecoveryError("D0A_INVALID_RUN_CONFIG", "run_config.output_dir is invalid")
    if Path(output_value).expanduser().resolve(strict=True) != run_dir:
        raise RecoveryError(
            "D0A_OUTPUT_DIR_MISMATCH", "run_config.output_dir differs from run directory"
        )


def _validate_split_manifest(
    manifest: Mapping[str, Any], run_config: Mapping[str, Any]
) -> None:
    _require_exact(manifest.get("role"), "official_train_only", "split_manifest.role")
    _require_exact(
        manifest.get("train_count"),
        run_config.get("expected_train_images"),
        "split_manifest.train_count",
    )
    _require_exact(
        manifest.get("train_split_sha256"),
        run_config.get("expected_train_split_sha256"),
        "split_manifest.train_split_sha256",
    )
    _require_exact(
        manifest.get("train_corpus_manifest_sha256"),
        run_config.get("expected_train_corpus_manifest_sha256"),
        "split_manifest.train_corpus_manifest_sha256",
    )
    if "known_train_size_mismatches" in run_config:
        _require_exact(
            manifest.get("known_train_size_mismatches"),
            run_config.get("known_train_size_mismatches"),
            "split_manifest.known_train_size_mismatches",
        )
    for field in ZERO_ACCESS_FIELDS:
        _require_exact(manifest.get(field), 0, f"split_manifest.{field}")


def _read_split_identifiers(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise RecoveryError("D0A_INVALID_SPLIT", f"split is not UTF-8: {path}") from error
    identifiers = [line.strip() for line in lines]
    if not identifiers or any(not identifier for identifier in identifiers):
        raise RecoveryError("D0A_INVALID_SPLIT", "train split is empty or has blank IDs")
    if len(set(identifiers)) != len(identifiers):
        raise RecoveryError("D0A_INVALID_SPLIT", "train split has duplicate IDs")
    return identifiers


def _load_trusted_checkpoint(
    path: Path, *, expected_sha256: str
) -> tuple[Mapping[str, Any], str]:
    try:
        snapshot = read_stable_regular_file(path)
    except (OSError, ValueError, SecureIOError) as error:
        raise RecoveryError(
            "D0A_UNSTABLE_INPUT", "cannot stably read trusted local last checkpoint"
        ) from error
    _require_expected_hash(snapshot.sha256, expected_sha256, "last checkpoint")
    try:
        payload = torch.load(
            io.BytesIO(snapshot.data), map_location="cpu", weights_only=False
        )
    except TypeError as error:
        if "weights_only" not in str(error):
            raise RecoveryError(
                "D0A_CHECKPOINT_LOAD_FAILED", "trusted local last checkpoint failed to load"
            ) from error
        try:
            payload = torch.load(io.BytesIO(snapshot.data), map_location="cpu")
        except Exception as fallback_error:
            raise RecoveryError(
                "D0A_CHECKPOINT_LOAD_FAILED", "trusted local last checkpoint failed to load"
            ) from fallback_error
    except Exception as error:
        raise RecoveryError(
            "D0A_CHECKPOINT_LOAD_FAILED", "trusted local last checkpoint failed to load"
        ) from error
    return _require_mapping(payload, "last checkpoint"), snapshot.sha256


def validate_v2_resume_state(
    checkpoint: Mapping[str, Any], run_config: Mapping[str, Any]
) -> None:
    """Prove that the frozen v2 model and optimizer accept the resume payload."""

    from model.MSHNet_NSFPN import MSHNet_NSFPN

    model = MSHNet_NSFPN(3)
    model.load_state_dict(
        _require_mapping(checkpoint.get("state_dict"), "resume state_dict"),
        strict=True,
    )
    optimizer = Adagrad(model.parameters(), lr=float(run_config["learning_rate"]))
    optimizer.load_state_dict(
        _require_mapping(checkpoint.get("optimizer"), "resume optimizer")
    )
    parameters = list(model.parameters())
    if len(optimizer.state) != len(parameters):
        raise RecoveryError(
            "D0A_OPTIMIZER_STATE_INVALID",
            "Adagrad state does not cover every frozen model parameter",
        )
    for index, parameter in enumerate(parameters):
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or not {"step", "sum"}.issubset(state):
            raise RecoveryError(
                "D0A_OPTIMIZER_STATE_INVALID",
                f"Adagrad parameter {index} lacks step/sum state",
            )
        accumulator = state["sum"]
        if not isinstance(accumulator, Tensor) or accumulator.shape != parameter.shape:
            raise RecoveryError(
                "D0A_OPTIMIZER_STATE_INVALID",
                f"Adagrad accumulator shape mismatch at parameter {index}",
            )
        if not bool(torch.isfinite(accumulator).all().item()):
            raise RecoveryError(
                "D0A_OPTIMIZER_STATE_INVALID",
                f"Adagrad accumulator is non-finite at parameter {index}",
            )


def _validate_checkpoint(
    source: Mapping[str, Any],
    *,
    run_config: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> tuple[int, int]:
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
        raise RecoveryError(
            "D0A_INCOMPLETE_CHECKPOINT", f"last checkpoint is missing keys: {missing}"
        )
    unexpected = sorted(set(source) - required_keys)
    if unexpected:
        raise RecoveryError(
            "D0A_UNEXPECTED_CHECKPOINT_KEYS",
            f"last checkpoint has unexpected payload keys: {unexpected}",
        )
    fixed_values = {
        "schema_version": 1,
        "architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "test_selected": False,
        "selection_rule": "fixed_final_epoch_train_only",
    }
    for field, expected in fixed_values.items():
        _require_exact(source.get(field), expected, f"checkpoint.{field}")
    if source.get("run_config") != run_config:
        raise RecoveryError(
            "D0A_CHECKPOINT_CONFIG_MISMATCH",
            "checkpoint run_config differs from immutable run_contract",
        )
    if source.get("split_manifest") != split_manifest:
        raise RecoveryError(
            "D0A_CHECKPOINT_SPLIT_MISMATCH",
            "checkpoint split_manifest differs from immutable run_contract",
        )
    epoch = source.get("epoch")
    if type(epoch) is not int or epoch < 1 or epoch > EXPECTED_EPOCHS:
        raise RecoveryError("D0A_INVALID_EPOCH", "checkpoint epoch must be in [1, 1000]")
    global_step = source.get("global_optimizer_step")
    if type(global_step) is not int or global_step <= 0:
        raise RecoveryError("D0A_INVALID_GLOBAL_STEP", "global step must be positive")
    steps_per_epoch = int(run_config["expected_train_images"]) // int(
        run_config["batch_size"]
    )
    if steps_per_epoch <= 0:
        raise RecoveryError("D0A_INVALID_RUN_CONFIG", "training has zero full batches")
    _require_exact(
        global_step,
        steps_per_epoch * epoch,
        "checkpoint.global_optimizer_step",
    )
    state_dict = _require_mapping(source.get("state_dict"), "checkpoint.state_dict")
    _require_exact(
        len(state_dict),
        int(run_config["expected_state_dict_keys"]),
        "checkpoint.state_dict key count",
    )
    for key, value in state_dict.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise RecoveryError(
                "D0A_INVALID_STATE_DICT", "checkpoint state_dict must be string-to-tensor"
            )
    _require_mapping(source.get("optimizer"), "checkpoint.optimizer")
    rng_state = _require_mapping(source.get("rng_state"), "checkpoint.rng_state")
    if not rng_state:
        raise RecoveryError(
            "D0A_SOURCE_ALREADY_SANITIZED", "last checkpoint has no process RNG state"
        )
    return epoch, global_step


def _validate_train_metrics(
    path: Path,
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_epoch: int,
    global_step: int,
    run_config: Mapping[str, Any],
) -> tuple[str, int]:
    try:
        snapshot = read_stable_regular_file(path)
    except (OSError, ValueError, SecureIOError) as error:
        raise RecoveryError(
            "D0A_UNSTABLE_INPUT", "cannot stably read train_metrics.jsonl"
        ) from error
    raw = snapshot.data
    metrics_sha = snapshot.sha256
    if not raw or not raw.endswith(b"\n"):
        raise RecoveryError(
            "D0A_INCOMPLETE_METRICS_LINE",
            "train_metrics.jsonl is empty or its last append is incomplete",
        )
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RecoveryError("D0A_INVALID_METRICS", "train metrics is not UTF-8") from error
    lines = decoded.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise RecoveryError("D0A_INVALID_METRICS", "train metrics contains blank records")
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecoveryError(
                "D0A_INVALID_METRICS", f"invalid JSON at train metrics line {line_number}"
            ) from error
        rows.append(_require_mapping(row, f"train metrics line {line_number}"))
    expected_epochs = list(range(1, checkpoint_epoch + 1))
    observed_epochs = [row.get("epoch") for row in rows]
    if observed_epochs != expected_epochs:
        raise RecoveryError(
            "D0A_METRICS_EPOCH_MISMATCH",
            "train metrics must be the contiguous complete prefix 1..checkpoint epoch; "
            f"observed tail={observed_epochs[-5:] if observed_epochs else []}",
        )
    steps_per_epoch = int(run_config["expected_train_images"]) // int(
        run_config["batch_size"]
    )
    for human_epoch, row in enumerate(rows, start=1):
        required = ("batches", "starting_optimizer_step", "ending_optimizer_step")
        if any(field not in row for field in required):
            raise RecoveryError(
                "D0A_INCOMPLETE_METRICS_RECORD",
                f"train metrics epoch {human_epoch} lacks complete step accounting",
            )
        _require_exact(row.get("batches"), steps_per_epoch, f"metrics[{human_epoch}].batches")
        _require_exact(
            row.get("starting_optimizer_step"),
            steps_per_epoch * (human_epoch - 1),
            f"metrics[{human_epoch}].starting_optimizer_step",
        )
        _require_exact(
            row.get("ending_optimizer_step"),
            steps_per_epoch * human_epoch,
            f"metrics[{human_epoch}].ending_optimizer_step",
        )
        for loss_name in (
            "mean_clean_loss",
            "mean_degraded_loss",
            "mean_combined_loss",
            "last_gradient_l2",
        ):
            if loss_name in row and row[loss_name] is not None:
                value = row[loss_name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise RecoveryError(
                        "D0A_INVALID_METRICS", f"metrics[{human_epoch}].{loss_name} is invalid"
                    )
                if not math.isfinite(float(value)):
                    raise RecoveryError(
                        "D0A_INVALID_METRICS", f"metrics[{human_epoch}].{loss_name} is non-finite"
                    )
    last = rows[-1]
    latest = _require_mapping(
        checkpoint.get("latest_train_metrics"), "checkpoint.latest_train_metrics"
    )
    if dict(last) != dict(latest):
        raise RecoveryError(
            "D0A_METRICS_CHECKPOINT_MISMATCH",
            "last complete train_metrics record is not identical to checkpoint metrics",
        )
    _require_exact(last.get("epoch"), checkpoint_epoch, "last metrics epoch")
    _require_exact(last.get("ending_optimizer_step"), global_step, "last metrics step")
    return metrics_sha, len(rows)


def _validate_started_contract_binding(
    freeze: Mapping[str, Any],
    *,
    dataset: str,
    contract_path: Path,
    contract_sha: str,
    repository: Path,
) -> None:
    candidates: list[Mapping[str, Any]] = []
    single = freeze.get("started_run_contract")
    if isinstance(single, Mapping) and single.get("dataset") == dataset:
        candidates.append(single)
    plural = freeze.get("started_run_contracts")
    if isinstance(plural, Mapping) and isinstance(plural.get(dataset), Mapping):
        candidates.append(plural[dataset])
    for index, binding in enumerate(candidates):
        declared = _declared_file(
            binding.get("path"), repository, f"frozen started run contract {index}"
        )
        if declared != contract_path:
            raise RecoveryError(
                "D0A_RUN_CONTRACT_PATH_MISMATCH",
                "FULL_TRAIN_FREEZE binds a different run_contract path",
            )
        expected = _require_sha256(
            binding.get("sha256"), "FULL_TRAIN_FREEZE started run contract hash"
        )
        if contract_sha != expected:
            raise RecoveryError(
                "D0A_RUN_CONTRACT_HASH_MISMATCH",
                f"run_contract hash differs from FULL_TRAIN_FREEZE: {contract_sha}",
            )


def _bind(bindings: dict[Path, str], path: Path, sha256: str) -> None:
    previous = bindings.get(path)
    if previous is not None and previous != sha256:
        raise RecoveryError("D0A_INPUT_BINDING_CONFLICT", f"conflicting hashes for {path}")
    bindings[path] = sha256


def _prepare_context(
    run_dir: Path,
    full_train_freeze: Path | None,
    repository: Path,
    *,
    expected_source_sha256: str,
    expected_run_contract_sha256: str,
    expected_full_train_freeze_sha256: str,
) -> dict[str, Any]:
    repository = repository.expanduser().resolve(strict=True)
    run_candidate = run_dir.expanduser().absolute()
    if run_candidate.is_symlink() or not run_candidate.is_dir():
        raise RecoveryError("D0A_UNSAFE_PATH", f"run directory is invalid: {run_candidate}")
    run_dir = run_candidate.resolve(strict=True)
    if not _inside(run_dir, repository):
        raise RecoveryError("D0A_UNSAFE_PATH", "run directory must be inside repository")
    source_path = _regular_file(run_dir / SOURCE_NAME, "trusted local last checkpoint")
    contract_path = _regular_file(run_dir / "run_contract.json", "run contract")
    metrics_path = _regular_file(run_dir / "train_metrics.jsonl", "train metrics")
    freeze_path = _regular_file(
        full_train_freeze or run_dir.parent / "FULL_TRAIN_FREEZE.json",
        "FULL_TRAIN_FREEZE",
    )
    tool_path = _regular_file(TOOL_PATH, "recovery tool")

    contract, contract_sha = _read_json(contract_path, "run contract")
    freeze, freeze_sha = _read_json(freeze_path, "FULL_TRAIN_FREEZE")
    _require_expected_hash(
        contract_sha, expected_run_contract_sha256, "run_contract"
    )
    _require_expected_hash(
        freeze_sha, expected_full_train_freeze_sha256, "FULL_TRAIN_FREEZE"
    )
    run_config = _require_mapping(contract.get("run_config"), "run_contract.run_config")
    split_manifest = _require_mapping(
        contract.get("split_manifest"), "run_contract.split_manifest"
    )
    _validate_run_config(run_config, run_dir)
    _validate_split_manifest(split_manifest, run_config)
    firewall = _validate_access_firewall(contract)

    _require_exact(freeze.get("schema_version"), 1, "FULL_TRAIN_FREEZE.schema_version")
    _require_exact(freeze.get("protocol_id"), PROTOCOL_ID, "FULL_TRAIN_FREEZE.protocol_id")
    _require_exact(
        freeze.get("status"),
        "runtime_bytes_frozen_full_training_in_progress",
        "FULL_TRAIN_FREEZE.status",
    )
    _require_exact(freeze.get("science_result"), False, "FULL_TRAIN_FREEZE.science_result")
    _require_exact(
        freeze.get("formal_test_authorized"), False, "FULL_TRAIN_FREEZE.formal_test_authorized"
    )
    _require_exact(freeze.get("tta_authorized"), False, "FULL_TRAIN_FREEZE.tta_authorized")

    freeze_declared, freeze_runtime = _validate_hash_map(
        freeze.get("runtime_sha256"), repository=repository, label="FULL_TRAIN_FREEZE.runtime_sha256"
    )
    missing_runtime = sorted(set(REQUIRED_FROZEN_RUNTIME) - set(freeze_declared))
    if missing_runtime:
        raise RecoveryError(
            "D0A_INCOMPLETE_FREEZE",
            f"FULL_TRAIN_FREEZE lacks required runtime bindings: {missing_runtime}",
        )
    contract_declared, contract_runtime = _validate_hash_map(
        contract.get("runtime_sha256"), repository=repository, label="run_contract.runtime_sha256"
    )
    for declared, expected in contract_declared.items():
        if declared in freeze_declared and freeze_declared[declared] != expected:
            raise RecoveryError(
                "D0A_RUNTIME_MAP_MISMATCH",
                f"run_contract and FULL_TRAIN_FREEZE disagree for {declared}",
            )
    for required in REQUIRED_FROZEN_RUNTIME:
        if required == "tta/deteriorations/__init__.py":
            continue
        if required not in contract_declared:
            raise RecoveryError(
                "D0A_INCOMPLETE_RUN_CONTRACT",
                f"run_contract lacks required runtime binding: {required}",
            )

    dataset = str(run_config["dataset"])
    _validate_started_contract_binding(
        freeze,
        dataset=dataset,
        contract_path=contract_path,
        contract_sha=contract_sha,
        repository=repository,
    )

    protocol_path = _declared_file(
        run_config.get("protocol_path"), repository, "frozen protocol config"
    )
    protocol_sha = sha256_file(protocol_path)
    _require_exact(
        run_config.get("protocol_sha256"), protocol_sha, "run_config.protocol_sha256"
    )
    if freeze_declared["configs/cr_sitta_d0a_train_v2.yaml"] != protocol_sha:
        raise RecoveryError(
            "D0A_PROTOCOL_HASH_MISMATCH", "run_config protocol differs from FULL_TRAIN_FREEZE"
        )

    official_split = _declared_file(
        run_config.get("train_split"), repository, "official train split"
    )
    archived_split = _regular_file(run_dir / "splits" / "train.txt", "archived train split")
    official_split_sha = sha256_file(official_split)
    archived_split_sha = sha256_file(archived_split)
    expected_split_sha = _require_sha256(
        run_config.get("expected_train_split_sha256"), "expected train split hash"
    )
    if official_split_sha != expected_split_sha or archived_split_sha != expected_split_sha:
        raise RecoveryError(
            "D0A_TRAIN_SPLIT_HASH_MISMATCH",
            "official/archive train split does not match the frozen split hash",
        )
    identifiers = _read_split_identifiers(archived_split)
    _require_exact(
        len(identifiers), int(run_config["expected_train_images"]), "archived train split count"
    )

    _validate_rng_restore_guard(repository / "train_cr_sitta_d0a.py")
    _validate_explicit_seed_semantics(
        repository / "train_cr_sitta_d0a.py", repository / "train_fixed_split.py"
    )
    model_paths = [repository / path for path in REQUIRED_FROZEN_RUNTIME if path.startswith("model/")]
    _validate_no_dropout(model_paths)

    source, source_sha = _load_trusted_checkpoint(
        source_path, expected_sha256=expected_source_sha256
    )
    epoch, global_step = _validate_checkpoint(
        source, run_config=run_config, split_manifest=split_manifest
    )
    metrics_sha, metrics_records = _validate_train_metrics(
        metrics_path,
        checkpoint=source,
        checkpoint_epoch=epoch,
        global_step=global_step,
        run_config=run_config,
    )
    secure_io_path = _regular_file(SECURE_IO_PATH, "secure publication dependency")
    secure_io_sha = sha256_file(secure_io_path)

    bindings: dict[Path, str] = {}
    for path, expected in freeze_runtime.items():
        _bind(bindings, path, expected)
    for path, expected in contract_runtime.items():
        _bind(bindings, path, expected)
    for path, expected in (
        (source_path, source_sha),
        (contract_path, contract_sha),
        (freeze_path, freeze_sha),
        (metrics_path, metrics_sha),
        (official_split, official_split_sha),
        (archived_split, archived_split_sha),
        (tool_path, sha256_file(tool_path)),
        (secure_io_path, secure_io_sha),
    ):
        _bind(bindings, path, expected)

    return {
        "repository": repository,
        "run_dir": run_dir,
        "source_path": source_path,
        "source": source,
        "source_sha": source_sha,
        "contract_path": contract_path,
        "contract_sha": contract_sha,
        "freeze_path": freeze_path,
        "freeze_sha": freeze_sha,
        "metrics_path": metrics_path,
        "metrics_sha": metrics_sha,
        "metrics_records": metrics_records,
        "official_split": official_split,
        "archived_split": archived_split,
        "split_sha": archived_split_sha,
        "tool_path": tool_path,
        "tool_sha": sha256_file(tool_path),
        "secure_io_path": secure_io_path,
        "secure_io_sha": secure_io_sha,
        "run_config": dict(run_config),
        "split_manifest": dict(split_manifest),
        "firewall": firewall,
        "epoch": epoch,
        "global_step": global_step,
        "dataset": dataset,
        "bindings": bindings,
    }


def _new_destination(path: Path, label: str) -> Path:
    if os.path.lexists(path):
        raise RecoveryError("D0A_NO_REPLACE", f"refusing to replace {label}: {path}")
    return path


def _new_recovery_staging(run_dir: Path) -> tuple[Path, Path]:
    destination = run_dir / RECOVERY_DIRECTORY_NAME
    if os.path.lexists(destination):
        raise RecoveryError(
            "D0A_NO_REPLACE", f"refusing to replace recovery directory: {destination}"
        )
    staging = Path(tempfile.mkdtemp(prefix=".recovery-staging-", dir=run_dir))
    staging.chmod(0o700)
    return staging.resolve(strict=True), destination


def _publish_recovery_directory(
    staging: Path,
    destination: Path,
    *,
    bindings: Mapping[Path, str],
) -> None:
    def guard() -> None:
        _validate_live_bindings(bindings)
        if os.path.lexists(destination):
            raise RecoveryError(
                "D0A_NO_REPLACE",
                f"refusing to replace recovery directory: {destination}",
            )

    try:
        publish_directory_noreplace(staging, destination, pre_rename_guard=guard)
    except FileExistsError as error:
        raise RecoveryError(
            "D0A_NO_REPLACE", f"refusing to replace recovery directory: {destination}"
        ) from error
    except SecureIOError as error:
        raise RecoveryError(
            "D0A_ATOMIC_PUBLICATION_FAILED", "recovery directory publication failed"
        ) from error


def _stage_bytes(directory: Path, payload: bytes, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, dir=directory)
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _stage_torch(directory: Path, payload: Mapping[str, Any]) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=".resume-checkpoint-", dir=directory)
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _publish_noreplace(staging: Path, destination: Path) -> None:
    if os.path.lexists(destination):
        raise RecoveryError("D0A_NO_REPLACE", f"refusing to replace {destination}")
    try:
        os.link(staging, destination)
    except FileExistsError as error:
        raise RecoveryError("D0A_NO_REPLACE", f"refusing to replace {destination}") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_live_bindings(bindings: Mapping[Path, str]) -> None:
    for path, expected in bindings.items():
        if path.is_symlink() or not path.is_file():
            raise RecoveryError("D0A_BOUND_INPUT_CHANGED", f"bound input disappeared: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RecoveryError(
                "D0A_BOUND_INPUT_CHANGED",
                f"bound input changed before publication: {path}; expected {expected}, got {actual}",
            )


def _load_sanitized_for_verification(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        if "weights_only" not in str(error):
            raise
        value = torch.load(path, map_location="cpu")
    except Exception as error:
        raise RecoveryError(
            "D0A_SANITIZATION_FAILED",
            "sanitized resume checkpoint is not weights_only loadable",
        ) from error
    return _require_mapping(value, "staged sanitized checkpoint")


def _base_receipt(context: Mapping[str, Any]) -> dict[str, Any]:
    repository = context["repository"]
    run_config = context["run_config"]
    split_manifest = context["split_manifest"]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "data_mode": "full_train_only",
        "dataset": context["dataset"],
        "science_result": False,
        "formal_test_authorized": False,
        "tta_authorized": False,
        "original_process_stopped": {
            "operator_confirmed": True,
            "independently_proven_by_tool": False,
        },
        "external_hash_anchors_verified": [
            "source_checkpoint",
            "run_contract",
            "FULL_TRAIN_FREEZE",
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": {
            "path": _display_path(context["source_path"], repository),
            "sha256": context["source_sha"],
            "trusted_local_pickle": True,
            "loaded_with_map_location": "cpu",
            "epoch": context["epoch"],
            "global_optimizer_step": context["global_step"],
        },
        "frozen_bindings": {
            "run_contract": {
                "path": _display_path(context["contract_path"], repository),
                "sha256": context["contract_sha"],
            },
            "FULL_TRAIN_FREEZE": {
                "path": _display_path(context["freeze_path"], repository),
                "sha256": context["freeze_sha"],
            },
            "run_config_sha256": _canonical_json_sha256(run_config),
            "split_manifest_sha256": _canonical_json_sha256(split_manifest),
            "official_train_split": {
                "path": _display_path(context["official_split"], repository),
                "sha256": context["split_sha"],
            },
            "archived_train_split": {
                "path": _display_path(context["archived_split"], repository),
                "sha256": context["split_sha"],
            },
            "train_metrics": {
                "path": _display_path(context["metrics_path"], repository),
                "sha256": context["metrics_sha"],
                "complete_records": context["metrics_records"],
                "last_complete_epoch": context["epoch"],
            },
            "recovery_tool": {
                "path": _display_path(context["tool_path"], repository),
                "sha256": context["tool_sha"],
            },
            "secure_publication_dependency": {
                "path": _display_path(context["secure_io_path"], repository),
                "sha256": context["secure_io_sha"],
            },
        },
        "access_firewall": context["firewall"],
    }


def _final_recovery_refusal(context: Mapping[str, Any], recovery_dir: Path) -> dict[str, Any]:
    receipt_path = _new_destination(
        recovery_dir / FINAL_RECOVERY_RECEIPT_NAME, "final-recovery refusal receipt"
    )
    if os.path.lexists(recovery_dir / RECOVERY_RECEIPT_NAME):
        raise RecoveryError("D0A_NO_REPLACE", "RECOVERY.json already exists")
    receipt = _base_receipt(context)
    receipt.update(
        {
            "artifact_type": FINAL_RECOVERY_RECEIPT_TYPE,
            "status": "resume_refused_final_checkpoint_recovery_required",
            "resume_permitted": False,
            "refusal_code": "D0A_EPOCH_1000_FINAL_EXPORT_REQUIRED",
            "process_exit_status": FINAL_EXPORT_REQUIRED_EXIT_STATUS,
            "reason": (
                "last.pth.tar records the complete epoch-1000 boundary, but the canonical "
                "epoch_1000_train_only.pth.tar is missing; starting epoch 1001 is forbidden"
            ),
            "recommended_action": {
                "first": (
                    "use a separately reviewed no-resume final-checkpoint recovery/export "
                    "procedure bound to this receipt"
                ),
                "then": (
                    "after the canonical final checkpoint exists, run the separately reviewed "
                    "safe weights-only exporter"
                ),
                "forbidden": "do not invoke train_cr_sitta_d0a.py --resume",
            },
            "output_checkpoint": None,
        }
    )
    staging: Path | None = None
    try:
        staging = _stage_bytes(
            recovery_dir,
            (json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
            ".final-recovery-receipt-",
        )
        _validate_live_bindings(context["bindings"])
        _publish_noreplace(staging, receipt_path)
        _fsync_directory(recovery_dir)
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
    return receipt


def recover_interrupted_run(
    run_dir: str | Path,
    *,
    expected_source_sha256: str,
    expected_run_contract_sha256: str,
    expected_full_train_freeze_sha256: str,
    original_process_stopped: bool,
    trusted_local_source: bool,
    full_train_freeze: str | Path | None = None,
    repository: str | Path = PROJECT_ROOT,
    resume_validator: ResumeValidator = validate_v2_resume_state,
) -> dict[str, Any]:
    """Validate and publish one no-replace RNG-sanitized epoch-boundary resume.

    No validation/test split or image is resolved.  If the trusted ``last`` is
    already at epoch 1000 and the canonical final checkpoint is absent, this
    publishes only ``FINAL_RECOVERY_REQUIRED.json`` and returns a refusal
    outcome; no resume checkpoint is created.
    """

    if original_process_stopped is not True:
        raise RecoveryError(
            "D0A_ACTIVE_RUN_NOT_EXCLUDED",
            "recovery requires explicit confirmation that the original training process stopped",
        )
    if trusted_local_source is not True:
        raise RecoveryError(
            "D0A_UNTRUSTED_PICKLE",
            "general pickle loading requires explicit trusted-local-source acknowledgement",
        )
    lexical_run_dir = Path(run_dir).expanduser().absolute()
    if os.path.lexists(lexical_run_dir / RECOVERY_DIRECTORY_NAME):
        raise RecoveryError(
            "D0A_NO_REPLACE",
            "refusing to replace existing recovery directory before checkpoint load",
        )
    context = _prepare_context(
        Path(run_dir),
        Path(full_train_freeze) if full_train_freeze is not None else None,
        Path(repository),
        expected_source_sha256=expected_source_sha256,
        expected_run_contract_sha256=expected_run_contract_sha256,
        expected_full_train_freeze_sha256=expected_full_train_freeze_sha256,
    )
    epoch = int(context["epoch"])
    final_path = context["run_dir"] / FINAL_NAME
    if epoch == EXPECTED_EPOCHS:
        if os.path.lexists(final_path):
            raise RecoveryError(
                "D0A_ALREADY_COMPLETE",
                "epoch-1000 last and canonical final checkpoint already exist; resume is forbidden",
            )
        staging_dir, recovery_dir = _new_recovery_staging(context["run_dir"])
        try:
            receipt = _final_recovery_refusal(context, staging_dir)
            _publish_recovery_directory(
                staging_dir, recovery_dir, bindings=context["bindings"]
            )
            return receipt
        finally:
            if staging_dir.exists() and not staging_dir.is_symlink():
                shutil.rmtree(staging_dir)

    staging_dir, recovery_dir = _new_recovery_staging(context["run_dir"])
    resume_path = _new_destination(
        staging_dir / f"resume_epoch_{epoch}_rng_sanitized.pth.tar",
        "sanitized resume checkpoint",
    )
    receipt_path = _new_destination(
        staging_dir / RECOVERY_RECEIPT_NAME, "recovery receipt"
    )
    sanitized = {key: value for key, value in context["source"].items() if key != "rng_state"}
    if set(sanitized) != set(context["source"]) - {"rng_state"}:
        raise RecoveryError("D0A_SANITIZATION_FAILED", "sanitization changed unexpected keys")

    checkpoint_staging: Path | None = None
    receipt_staging: Path | None = None
    try:
        checkpoint_staging = _stage_torch(staging_dir, sanitized)
        verified = _load_sanitized_for_verification(checkpoint_staging)
        if "rng_state" in verified:
            raise RecoveryError("D0A_SANITIZATION_FAILED", "rng_state survived serialization")
        if set(verified) != set(context["source"]) - {"rng_state"}:
            raise RecoveryError(
                "D0A_SANITIZATION_FAILED", "serialized checkpoint key set changed"
            )
        for field in (
            "epoch",
            "global_optimizer_step",
            "run_config",
            "split_manifest",
            "method_stage",
            "state_dict",
            "optimizer",
        ):
            if field not in verified:
                raise RecoveryError(
                    "D0A_SANITIZATION_FAILED", f"sanitized checkpoint lost {field}"
                )
        _require_exact(verified.get("epoch"), epoch, "sanitized checkpoint epoch")
        _require_exact(
            verified.get("global_optimizer_step"),
            context["global_step"],
            "sanitized checkpoint global step",
        )
        if verified.get("run_config") != context["run_config"]:
            raise RecoveryError("D0A_SANITIZATION_FAILED", "run_config was not preserved")
        if verified.get("split_manifest") != context["split_manifest"]:
            raise RecoveryError("D0A_SANITIZATION_FAILED", "split_manifest was not preserved")
        resume_validator(verified, context["run_config"])
        repository_resume_validated = resume_validator is validate_v2_resume_state
        output_sha = sha256_file(checkpoint_staging)

        receipt = _base_receipt(context)
        receipt.update(
            {
                "artifact_type": RECOVERY_RECEIPT_TYPE,
                "checkpoint_artifact_type": RECOVERY_ARTIFACT_TYPE,
                "status": "resume_checkpoint_rng_sanitized",
                "resume_permitted": True,
                "refusal_code": None,
                "output_checkpoint": {
                    "path": _display_path(
                        recovery_dir / resume_path.name, context["repository"]
                    ),
                    "sha256": output_sha,
                    "epoch": epoch,
                    "global_optimizer_step": context["global_step"],
                    "next_epoch": epoch + 1,
                },
                "sanitization": {
                    "removed_keys": ["rng_state"],
                    "torch_load_weights_only_verified": True,
                    "preserved_payload_keys": sorted(sanitized),
                    "model_state_preserved": True,
                    "optimizer_state_preserved": True,
                    "run_config_preserved": True,
                    "split_manifest_preserved": True,
                    "global_optimizer_step_preserved": True,
                    "original_v2_rng_restore_called": False,
                    "reason": (
                        "the frozen _restore_rng_state helper returns immediately when "
                        "rng_state is absent"
                    ),
                },
                "resume_semantics": {
                    "epoch_boundary_only": True,
                    "next_epoch": epoch + 1,
                    "per_epoch_dataloader_seed_explicit": True,
                    "per_image_probe_seed_explicit": True,
                    "model_dropout_absent": True,
                    "frozen_v2_model_optimizer_load_validated": (
                        repository_resume_validated
                    ),
                    "resume_validator": getattr(
                        resume_validator, "__name__", type(resume_validator).__name__
                    ),
                    "cuda_continuation_bit_exact_guaranteed": False,
                    "cuda_disclosure": (
                        "A resumed process using custom CUDA operators is not claimed to be "
                        "bit-exact to an uninterrupted CUDA process; compare scientific gates "
                        "under the frozen protocol rather than claiming byte-identical training."
                    ),
                },
                "publication": {
                    "atomic_no_replace": True,
                    "artifact_pair_published_as_one_directory_rename": True,
                    "source_checkpoint_unchanged": True,
                    "completion_sentinel": RECOVERY_RECEIPT_NAME,
                },
            }
        )
        receipt_bytes = (
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        receipt_staging = _stage_bytes(
            staging_dir, receipt_bytes, ".recovery-receipt-"
        )
        _validate_live_bindings(context["bindings"])
        _publish_noreplace(checkpoint_staging, resume_path)
        checkpoint_staging.unlink()
        checkpoint_staging = None
        _fsync_directory(staging_dir)
        _publish_noreplace(receipt_staging, receipt_path)
        receipt_staging.unlink()
        receipt_staging = None
        _fsync_directory(staging_dir)
        _publish_recovery_directory(
            staging_dir, recovery_dir, bindings=context["bindings"]
        )
        return receipt
    finally:
        if checkpoint_staging is not None:
            checkpoint_staging.unlink(missing_ok=True)
        if receipt_staging is not None:
            receipt_staging.unlink(missing_ok=True)
        if staging_dir.exists() and not staging_dir.is_symlink():
            shutil.rmtree(staging_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a trusted-local, epoch-boundary CR-SITTA D0-A v2 resume "
            "without opening validation/test data."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-run-contract-sha256", required=True)
    parser.add_argument("--expected-full-train-freeze-sha256", required=True)
    parser.add_argument(
        "--confirm-original-process-stopped",
        action="store_true",
        help="confirm that the original D0-A training process is no longer running",
    )
    parser.add_argument(
        "--trust-local-source",
        action="store_true",
        help="acknowledge that last.pth.tar is a trusted local frozen-run artifact",
    )
    parser.add_argument("--full-train-freeze", type=Path, default=None)
    parser.add_argument("--repository", type=Path, default=PROJECT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outcome = recover_interrupted_run(
            args.run_dir,
            expected_source_sha256=args.expected_source_sha256,
            expected_run_contract_sha256=args.expected_run_contract_sha256,
            expected_full_train_freeze_sha256=(
                args.expected_full_train_freeze_sha256
            ),
            original_process_stopped=args.confirm_original_process_stopped,
            trusted_local_source=args.trust_local_source,
            full_train_freeze=args.full_train_freeze,
            repository=args.repository,
        )
    except RecoveryError as error:
        print(
            json.dumps(
                {"status": "recovery_refused", "refusal_code": error.code, "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(json.dumps(outcome, ensure_ascii=False, sort_keys=True), flush=True)
    if outcome.get("refusal_code") == "D0A_EPOCH_1000_FINAL_EXPORT_REQUIRED":
        return FINAL_EXPORT_REQUIRED_EXIT_STATUS
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
