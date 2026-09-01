#!/usr/bin/env python3
"""Run Phase-D0 Binary-TENT failure diagnostics on frozen source-train data.

This runner is deliberately separate from the frozen v2 calibration runner.
It explains the v2 negative result; it cannot select a candidate or authorize
Stage 2.  A canonical shard is accepted only after exactly three fresh GPU
processes have produced an aggregate equivalence receipt showing that reusing
one label-free entropy gradient gives exactly the same TENT-pre/TENT-post
logits as the historical independent BinaryTentMethod path.

Targets are not deserialized until every selected label-free episode in the
condition has completed.  They are then used only by the outer diagnostic
phase for no-op classification and supervised-gradient alignment.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
import uuid

import numpy as np
import torch
from torch import Tensor, nn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_binary_tent_ss_calibration_v2 as frozen_v2
from analysis.analyze_entropy_task_alignment import (
    analyze_entropy_task_alignment,
)
from analysis.analyze_tent_optimizer_geometry import (
    ANALYSIS_SCHEMA_VERSION as OPTIMIZER_GEOMETRY_SCHEMA_VERSION,
    CROSS_BACKEND_CPU_STORAGE_REPLAY_REFERENCE,
    CONTINUOUS_IDEAL_REFERENCE,
    OptimizerFirstStepSpec,
    PYTORCH_REFERENCE_VERSION,
    RUNTIME_HARD_GATE_REFERENCE,
    analyze_optimizer_first_step,
    pytorch_first_step_reference,
)
from analysis.d0_determinism_runtime import (
    D0DeterminismRuntimeError,
    assert_strict_forward_policy,
    backward_with_d0_determinism,
    frozen_determinism_sha256,
    validate_audit as validate_determinism_audit,
)
from analysis.d0_equivalence_repro_contract import (
    D0EquivalenceReproContractError,
    EquivalenceReceiptBinding,
    LOGITS_TENSOR_SHA256_CONTRACT,
    build_d0_equivalence_repro_receipt,
    validate_d0_equivalence_repro_receipt,
)
from analysis.d0_protocol_contract import (
    D0ProtocolContractError,
    parse_d0_protocol_contract,
)
from analysis.source_train_provenance import SourceTrainAnalysisProvenance
from materialize_binary_tent_ss_calibration_cache_v2 import (
    SourceCalibrationMethodInputDatasetV2,
    load_outer_evaluator_targets_v2,
)
from scripts.archive_binary_tent_ss_stage1_negative_v2 import (
    NegativeArchiveError,
    verify_archive as verify_negative_archive,
)
from model.loss import SLSIoULoss
from tta.binary_tent import binary_entropy_map, build_binary_tent_optimizer
from tta.d0_secure_io import (
    ensure_directory_chain_nofollow,
    fsync_directory,
    publish_directory_noreplace,
    publish_file_noreplace,
    read_stable_regular_file,
    snapshot_regular_directory,
)
from tta.diagnostics import NoOpThresholds, analyze_noop_episode
from tta.model_adapter import IRSTDModelAdapter
from tta.parameter_groups import (
    AdaptableGroupSpec,
    build_parameter_group_inventory,
    collect_adaptable_params,
    verify_frozen_nsfpn_inventory,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tent_failure_diagnostics_v1.yaml"
ARTIFACT_TYPE = "cr_sitta_tent_failure_diagnostics_dataset_shard_v1"
EQUIVALENCE_TYPE = "cr_sitta_tent_shared_gradient_equivalence_v1"
AGGREGATE_TYPE = "cr_sitta_tent_failure_diagnostics_three_dataset_aggregate_v1"
JSON_SEPARATORS = (",", ":")

# These are the frozen Phase-D0 dimensions inherited from Stage-1 v2.  Keep
# the cardinalities derived from the tuples instead of duplicating magic
# record counts: one formal dataset shard is 13 * 10 * 64 = 8,320 diagnostic
# episode records, and the three-dataset aggregate is 24,960 records.
FORMAL_DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
FORMAL_IMAGES_PER_CELL = 64


class DiagnosticRunnerError(RuntimeError):
    """The D0 protocol or an execution invariant failed."""


@dataclass(frozen=True, order=True)
class Candidate:
    optimizer: str
    learning_rate: float

    def __post_init__(self) -> None:
        if self.optimizer not in ("Adam", "SGD"):
            raise DiagnosticRunnerError("candidate optimizer must be Adam or SGD")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise DiagnosticRunnerError("candidate learning rate must be positive")

    @property
    def slug(self) -> str:
        text = format(self.learning_rate, ".0e").replace("e-0", "e-")
        return f"{self.optimizer}_lr_{text.replace('-', 'm').replace('+', 'p')}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
        }


@dataclass
class CandidateState:
    logits_post: Tensor
    parameters_after: dict[str, Tensor]
    step: dict[str, Tensor]
    geometry: dict[str, Any]
    strict_policy_before_optimizer_step: bool
    strict_policy_before_post_forward: bool


@dataclass
class LabelFreeEpisode:
    metadata: dict[str, Any]
    logits_pre: Tensor
    entropy_pre: float
    parameters_before: dict[str, Tensor]
    entropy_gradients: dict[str, Tensor | None]
    entropy_backward_determinism: dict[str, Any]
    candidates: dict[str, CandidateState]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_tensor_bundle_sha256(
    names: Sequence[str], values: Mapping[str, Tensor | None]
) -> str:
    """Hash an exact ordered named-tensor bundle, including topology.

    This is intentionally stronger than a norm/count comparison.  It is used
    by the shared-gradient equivalence gate to prove that the historical
    independent Binary-TENT path produced the same entropy gradient and the
    same optimizer delta for every adaptable tensor.
    """

    if tuple(values) != tuple(names) or len(set(names)) != len(names):
        raise DiagnosticRunnerError("named tensor hash topology differs")
    digest = hashlib.sha256()
    digest.update(b"cr-sitta-named-tensor-bundle-v1\0")
    for name in names:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        value = values[name]
        if value is None:
            digest.update(b"N")
            continue
        if not isinstance(value, Tensor) or value.layout != torch.strided:
            raise DiagnosticRunnerError(f"invalid tensor in hash bundle: {name}")
        tensor = value.detach().cpu().contiguous()
        if not torch.is_floating_point(tensor) or not bool(
            torch.isfinite(tensor).all().item()
        ):
            raise DiagnosticRunnerError(f"non-finite tensor in hash bundle: {name}")
        dtype = str(tensor.dtype).encode("ascii")
        shape = ",".join(str(item) for item in tensor.shape).encode("ascii")
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(b"T")
        for component in (dtype, shape, payload):
            digest.update(len(component).to_bytes(8, "big"))
            digest.update(component)
    return digest.hexdigest()


def _optimizer_state_bundle_sha256(
    names: Sequence[str], states: Mapping[str, Mapping[str, Tensor]]
) -> tuple[str, int]:
    """Hash ordered per-parameter optimizer state, including field topology."""

    if tuple(states) != tuple(names):
        raise DiagnosticRunnerError("optimizer state hash topology differs")
    flattened_names: list[str] = []
    flattened_values: dict[str, Tensor] = {}
    for parameter_name in names:
        state = states[parameter_name]
        if not isinstance(state, Mapping):
            raise DiagnosticRunnerError("optimizer state hash entry is invalid")
        for field_name in sorted(state):
            value = state[field_name]
            if not isinstance(field_name, str) or not field_name or not isinstance(
                value, Tensor
            ):
                raise DiagnosticRunnerError("optimizer state hash field is invalid")
            qualified = f"{parameter_name}\0{field_name}"
            flattened_names.append(qualified)
            flattened_values[qualified] = value
    if not flattened_names:
        raise DiagnosticRunnerError("optimizer state hash cannot be empty")
    return (
        _named_tensor_bundle_sha256(flattened_names, flattened_values),
        len(flattened_names),
    )


def _strided_tensor_sha256(value: Tensor) -> str:
    """Hash dtype, shape and contiguous bytes under the frozen logits contract."""

    if not isinstance(value, Tensor) or value.layout != torch.strided:
        raise DiagnosticRunnerError("logits hash requires a strided tensor")
    tensor = value.detach().cpu().contiguous()
    if not torch.is_floating_point(tensor) or not bool(
        torch.isfinite(tensor).all().item()
    ):
        raise DiagnosticRunnerError("logits hash requires finite floating values")
    dtype = str(tensor.dtype).encode("ascii")
    shape = ",".join(str(item) for item in tensor.shape).encode("ascii")
    payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(LOGITS_TENSOR_SHA256_CONTRACT.encode("ascii") + b"\0")
    for component in (dtype, shape, payload):
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return digest.hexdigest()


def _install_first_step_capture(
    optimizer: torch.optim.Optimizer, names: Sequence[str]
) -> tuple[Any, dict[str, Any]]:
    """Wrap exactly one optimizer step and capture its raw gradients/deltas."""

    parameters = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    if len(parameters) != len(names):
        raise DiagnosticRunnerError("optimizer capture topology differs")
    original_step = optimizer.step
    capture: dict[str, Any] = {}

    def captured_step(*args: Any, **kwargs: Any) -> Any:
        if capture:
            raise DiagnosticRunnerError("equivalence optimizer stepped more than once")
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in zip(names, parameters, strict=True)
        }
        gradients = {
            name: (
                None
                if parameter.grad is None
                else parameter.grad.detach().cpu().clone()
            )
            for name, parameter in zip(names, parameters, strict=True)
        }
        result = original_step(*args, **kwargs)
        after = {
            name: parameter.detach().cpu().clone()
            for name, parameter in zip(names, parameters, strict=True)
        }
        deltas = {name: after[name] - before[name] for name in names}
        capture.update(
            {
                "gradient_bundle_sha256": _named_tensor_bundle_sha256(
                    names, gradients
                ),
                "delta_bundle_sha256": _named_tensor_bundle_sha256(names, deltas),
            }
        )
        return result

    optimizer.step = captured_step  # type: ignore[method-assign]
    return original_step, capture


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=JSON_SEPARATORS,
        allow_nan=False,
    ).encode("utf-8")


def _command_sha256(command: Sequence[str]) -> str:
    """Bind one exact exec argv vector without exposing it in the receipt."""

    if isinstance(command, (str, bytes, bytearray)) or not command or not all(
        isinstance(value, str) and value for value in command
    ):
        raise DiagnosticRunnerError("equivalence worker command is invalid")
    return hashlib.sha256(_canonical_json(list(command))).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for value in values:
            handle.write(_canonical_json(value) + b"\n")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticRunnerError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticRunnerError(f"JSON root must be an object: {path}")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    path = _lexical_absolute(path)
    try:
        snapshot = read_stable_regular_file(path)
        value = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DiagnosticRunnerError(f"invalid YAML: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticRunnerError("config root must be a mapping")
    try:
        parse_d0_protocol_contract(value)
    except D0ProtocolContractError as exc:
        raise DiagnosticRunnerError(f"D0 execution contract drifted: {exc}") from exc
    scope = value.get("scope")
    required_scope = {
        "source_train_derived": True,
        "paper_result": False,
        "paper_test_result": False,
        "oracle_analysis": True,
        "no_validation_split": True,
        "use_test_images": False,
        "use_test_labels": False,
        "method_label_accesses": 0,
    }
    if not isinstance(scope, Mapping) or any(
        scope.get(key) != expected for key, expected in required_scope.items()
    ):
        raise DiagnosticRunnerError("D0 scope is not source-train-only/non-paper")
    return value


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving any symlink component."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _output_root(config: Mapping[str, Any]) -> Path:
    return _lexical_absolute(PROJECT_ROOT / str(config["output"]["root"]))


def _validated_output_destination(
    *,
    config: Mapping[str, Any],
    destination: Path,
    role: str,
    dataset: str | None = None,
    process_id: str | None = None,
) -> Path:
    output = config["output"]
    root = _output_root(config)
    observed = _lexical_absolute(destination)
    if role == "formal_shard":
        if dataset is None:
            raise DiagnosticRunnerError("formal shard output requires dataset")
        expected = root / str(output["canonical_shards"]) / dataset
        if observed != expected:
            raise DiagnosticRunnerError(
                f"formal D0 output must be exactly {expected}"
            )
        return observed
    if role == "smoke_shard":
        parent = root / str(output["smoke_shards"])
        if observed.parent != parent or observed.name in {"", ".", ".."}:
            raise DiagnosticRunnerError(
                f"D0 smoke output must be one direct child of {parent}"
            )
        return observed
    if role == "equivalence":
        if dataset is None:
            raise DiagnosticRunnerError("equivalence output requires dataset")
        expected = root / str(output["equivalence"]) / f"{dataset}.json"
        if observed != expected:
            raise DiagnosticRunnerError(
                f"D0 equivalence output must be exactly {expected}"
            )
        return observed
    if role == "equivalence_single":
        if dataset is None or process_id is None:
            raise DiagnosticRunnerError(
                "single-process equivalence output requires dataset/process_id"
            )
        if (
            not process_id
            or len(process_id) > 128
            or process_id[0] not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
                for character in process_id
            )
        ):
            raise DiagnosticRunnerError("unsafe equivalence process_id")
        expected = (
            root
            / str(output["equivalence"])
            / "runs"
            / dataset
            / f"{process_id}.json"
        )
        if observed != expected:
            raise DiagnosticRunnerError(
                f"single-process equivalence output must be exactly {expected}"
            )
        return observed
    if role == "aggregate":
        expected = root / str(output["aggregate"])
        if observed != expected:
            raise DiagnosticRunnerError(
                f"D0 aggregate output must be exactly {expected}"
            )
        return observed
    raise DiagnosticRunnerError(f"unknown D0 output role: {role}")


def _ensure_output_parent(
    *, config: Mapping[str, Any], role: str, dataset: str | None = None
) -> Path:
    root_parts = tuple(Path(str(config["output"]["root"])).parts)
    if role == "formal_shard":
        suffix = (str(config["output"]["canonical_shards"]),)
    elif role == "smoke_shard":
        suffix = (str(config["output"]["smoke_shards"]),)
    elif role == "equivalence":
        suffix = (str(config["output"]["equivalence"]),)
    elif role == "equivalence_single":
        if dataset is None:
            raise DiagnosticRunnerError(
                "single-process equivalence parent requires dataset"
            )
        suffix = (str(config["output"]["equivalence"]), "runs", dataset)
    elif role == "aggregate":
        suffix = ()
    else:
        raise DiagnosticRunnerError(f"unknown D0 output role: {role}")
    return ensure_directory_chain_nofollow(PROJECT_ROOT, (*root_parts, *suffix))


def _config_candidates(config: Mapping[str, Any]) -> tuple[Candidate, ...]:
    raw = config.get("method", {}).get("candidates")
    if not isinstance(raw, list) or len(raw) != 10:
        raise DiagnosticRunnerError("D0 requires exactly ten frozen candidates")
    values = tuple(
        Candidate(str(value.get("optimizer")), float(value.get("learning_rate")))
        for value in raw
        if isinstance(value, Mapping)
    )
    if len(values) != 10 or len(set(values)) != 10:
        raise DiagnosticRunnerError("D0 candidates are incomplete or duplicated")
    expected = tuple(
        Candidate(candidate.optimizer, float(candidate.learning_rate))
        for candidate in frozen_v2.ALL_CANDIDATES
    )
    if values != expected:
        raise DiagnosticRunnerError("D0 candidates differ from frozen v2 order")
    return values


def _config_conditions(config: Mapping[str, Any]) -> tuple[str, ...]:
    raw = config.get("conditions")
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise DiagnosticRunnerError("conditions must be a string list")
    expected = tuple(f"{name}_S{severity}" for name, severity in frozen_v2.CONDITIONS)
    values = tuple(raw)
    if values != expected:
        raise DiagnosticRunnerError("D0 conditions differ from frozen v2 order")
    return values


def _parameter_group_assignment(model: nn.Module) -> dict[str, str]:
    inventory = build_parameter_group_inventory(model)
    verify_frozen_nsfpn_inventory(inventory)
    assignment: dict[str, str] = {}
    for group in inventory.groups:
        for name in group.parameter_names:
            if name in assignment:
                raise DiagnosticRunnerError(f"parameter assigned twice: {name}")
            assignment[name] = group.group_id
    if len(assignment) != 106:
        raise DiagnosticRunnerError("frozen all-BN assignment must contain 106 tensors")
    return assignment


def _clone_named(
    names: Sequence[str], values: Sequence[Tensor | None]
) -> dict[str, Tensor | None]:
    if len(names) != len(values):
        raise DiagnosticRunnerError("named tensor lengths differ")
    return {
        name: None if value is None else value.detach().cpu().clone()
        for name, value in zip(names, values, strict=True)
    }


def _nonnull(values: Mapping[str, Tensor | None]) -> dict[str, Tensor]:
    return {
        name: value
        for name, value in values.items()
        if value is not None
    }


def _restore_parameters(
    parameters: Sequence[nn.Parameter], source: Sequence[Tensor]
) -> None:
    if len(parameters) != len(source):
        raise DiagnosticRunnerError("parameter restore topology changed")
    with torch.no_grad():
        for parameter, value in zip(parameters, source, strict=True):
            parameter.copy_(value)
            parameter.grad = None


def _assert_parameters_exact(
    parameters: Sequence[nn.Parameter], source: Sequence[Tensor], label: str
) -> None:
    if len(parameters) != len(source) or any(
        not torch.equal(parameter.detach(), value)
        for parameter, value in zip(parameters, source, strict=True)
    ):
        raise DiagnosticRunnerError(f"adaptable parameters not exact at {label}")


def _model_state_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _assert_model_state_exact(model: nn.Module, source: Mapping[str, Tensor]) -> None:
    current = model.state_dict()
    if tuple(current) != tuple(source):
        raise DiagnosticRunnerError("model state topology changed")
    for name, expected in source.items():
        if not torch.equal(current[name].detach().cpu(), expected):
            raise DiagnosticRunnerError(f"model state was not restored: {name}")
    if any(module.training for module in model.modules()):
        raise DiagnosticRunnerError("model runtime was not restored to Source eval")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise DiagnosticRunnerError("model gradients were not refrozen at Source")
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and (
            not module.track_running_stats
            or module.running_mean is None
            or module.running_var is None
        ):
            raise DiagnosticRunnerError(f"BatchNorm Source runtime not restored: {name}")


def _backward(
    loss: Tensor,
    *,
    scope: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one named backward under the frozen, auditable D0 policy."""

    try:
        return backward_with_d0_determinism(
            loss,
            scope=scope,
            determinism_config=config["method"]["determinism"],
        )
    except (KeyError, D0DeterminismRuntimeError) as exc:
        raise DiagnosticRunnerError(
            f"D0 determinism contract failed in {scope}"
        ) from exc


def _assert_cuda_process_environment() -> None:
    expected = {
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    mismatches = {
        key: {"expected": value, "observed": os.environ.get(key)}
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise DiagnosticRunnerError(
            f"D0 CUDA process environment is not frozen: {mismatches}"
        )


def _linux_process_start_time_ticks(process_id: int | None = None) -> int:
    """Read Linux /proc start-time field 22 without misparsing spaces in comm."""

    pid = os.getpid() if process_id is None else process_id
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise DiagnosticRunnerError("invalid OS process ID")
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        close = value.rfind(")")
        if close < 0:
            raise ValueError("missing comm terminator")
        fields_from_three = value[close + 2 :].split()
        ticks = int(fields_from_three[19])
    except (OSError, UnicodeDecodeError, ValueError, IndexError) as exc:
        raise DiagnosticRunnerError(
            f"cannot establish Linux process start time for PID {pid}"
        ) from exc
    if ticks <= 0:
        raise DiagnosticRunnerError("Linux process start time must be positive")
    return ticks


def _configure_d0_cuda(device_name: str) -> torch.device:
    _assert_cuda_process_environment()
    if device_name != "cuda:0":
        raise DiagnosticRunnerError(
            "D0 worker requires isolated logical device cuda:0"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise DiagnosticRunnerError(
            "D0 worker must see exactly one available CUDA device"
        )
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise DiagnosticRunnerError("D0 strict deterministic policy was not enabled")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    return device


def _provenance(
    *, config: Mapping[str, Any], dataset: str, oracle: bool, accesses: int
) -> SourceTrainAnalysisProvenance:
    entry = config["datasets"][dataset]
    return SourceTrainAnalysisProvenance(
        dataset=dataset,
        split_name="train",
        split_sha256=str(entry["train_split_sha256"]),
        checkpoint_sha256=str(entry["checkpoint"]["sha256"]),
        seed=42,
        oracle_analysis=oracle,
        outer_evaluator_label_accesses=accesses,
        supervised_gradient_role=(
            "outer_oracle_train_labels_only" if oracle else "none"
        ),
    )


def _optimizer_mapping(candidate: Candidate) -> dict[str, Any]:
    common = {
        "name": candidate.optimizer,
        "learning_rate": candidate.learning_rate,
        "weight_decay": 0.0,
        "maximize": False,
        "initial_optimizer_state_empty": True,
    }
    if candidate.optimizer == "SGD":
        return {
            **common,
            "momentum": 0.9,
            "dampening": 0.0,
            "nesterov": True,
        }
    return {
        **common,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "amsgrad": False,
        "decoupled_weight_decay": False,
    }


def _json_optimizer_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _verify_actual_optimizer_pre_step(
    optimizer: torch.optim.Optimizer,
    *,
    candidate: Candidate,
    parameters: Sequence[nn.Parameter],
) -> tuple[OptimizerFirstStepSpec, dict[str, Any]]:
    """Bind the runtime hard gate to the actual PyTorch optimizer object."""

    observed_version = str(torch.__version__).split("+", 1)[0]
    if observed_version != PYTORCH_REFERENCE_VERSION:
        raise DiagnosticRunnerError(
            "D0 optimizer reference requires torch=="
            f"{PYTORCH_REFERENCE_VERSION}, observed {torch.__version__}"
        )
    default_scalar = torch.empty(())
    if default_scalar.dtype != torch.float32 or default_scalar.device.type != "cpu":
        raise DiagnosticRunnerError(
            "D0 optimizer reference requires CPU float32 PyTorch default tensors"
        )
    if optimizer.state:
        raise DiagnosticRunnerError("runtime optimizer state was not empty before step")
    if len(optimizer.param_groups) != 1:
        raise DiagnosticRunnerError("runtime optimizer must contain exactly one group")
    group = optimizer.param_groups[0]
    if tuple(map(id, group.get("params", ()))) != tuple(map(id, parameters)):
        raise DiagnosticRunnerError("runtime optimizer parameter order differs")

    if candidate.optimizer == "Adam":
        if type(optimizer) is not torch.optim.Adam:
            raise DiagnosticRunnerError("runtime optimizer class differs from Adam")
        expected = {
            "lr": candidate.learning_rate,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        }
        mapping = {
            "name": "Adam",
            "learning_rate": group.get("lr"),
            "betas": group.get("betas"),
            "eps": group.get("eps"),
            "weight_decay": group.get("weight_decay"),
            "amsgrad": group.get("amsgrad"),
            "maximize": group.get("maximize"),
            "decoupled_weight_decay": False,
            "initial_optimizer_state_empty": True,
        }
        implementation_flags = {
            "foreach": group.get("foreach"),
            "fused": group.get("fused"),
            "capturable": group.get("capturable"),
            "differentiable": group.get("differentiable"),
        }
    else:
        if type(optimizer) is not torch.optim.SGD:
            raise DiagnosticRunnerError("runtime optimizer class differs from SGD")
        expected = {
            "lr": candidate.learning_rate,
            "momentum": 0.9,
            "dampening": 0.0,
            "weight_decay": 0.0,
            "nesterov": True,
            "maximize": False,
            "foreach": False,
            "differentiable": False,
        }
        mapping = {
            "name": "SGD",
            "learning_rate": group.get("lr"),
            "momentum": group.get("momentum"),
            "dampening": group.get("dampening"),
            "weight_decay": group.get("weight_decay"),
            "nesterov": group.get("nesterov"),
            "maximize": group.get("maximize"),
            "initial_optimizer_state_empty": True,
        }
        implementation_flags = {
            "foreach": group.get("foreach"),
            "fused": None,
            "capturable": None,
            "differentiable": group.get("differentiable"),
        }

    if set(optimizer.defaults) != set(expected) or set(group) != {
        "params",
        *expected,
    }:
        raise DiagnosticRunnerError("runtime optimizer fields differ from frozen contract")
    for source_name, source in (("defaults", optimizer.defaults), ("param_group", group)):
        mismatches = {
            key: {"expected": expected_value, "observed": source.get(key)}
            for key, expected_value in expected.items()
            if source.get(key) != expected_value
        }
        if mismatches:
            raise DiagnosticRunnerError(
                f"runtime optimizer {source_name} differs: {mismatches}"
            )
    try:
        spec = OptimizerFirstStepSpec.from_mapping(mapping)
    except ValueError as exc:
        raise DiagnosticRunnerError("runtime optimizer spec is invalid") from exc
    return spec, {
        "schema_version": 1,
        "reference_name": RUNTIME_HARD_GATE_REFERENCE,
        "pytorch_version_required": PYTORCH_REFERENCE_VERSION,
        "pytorch_version_observed": observed_version,
        "pytorch_default_dtype": str(default_scalar.dtype),
        "pytorch_default_device_type": default_scalar.device.type,
        "optimizer_class": type(optimizer).__name__,
        "optimizer_config_from_actual_param_group": spec.to_dict(),
        "implementation_flags_from_actual_param_group": implementation_flags,
        "defaults_from_actual_optimizer": {
            key: _json_optimizer_value(optimizer.defaults[key])
            for key in sorted(expected)
        },
        "single_param_group_verified": True,
        "ordered_parameter_identity_verified": True,
        "parameter_tensor_count": len(parameters),
        "initial_optimizer_state_empty": True,
    }


def _verify_actual_optimizer_post_step(
    optimizer: torch.optim.Optimizer,
    *,
    spec: OptimizerFirstStepSpec,
    names: Sequence[str],
    parameters: Sequence[nn.Parameter],
    reference_state: Mapping[str, Mapping[str, Tensor]],
) -> dict[str, Any]:
    """Bit-exactly verify actual optimizer state after the one allowed step."""

    if (
        len(names) != len(parameters)
        or tuple(reference_state) != tuple(names)
        or len(optimizer.state) != len(parameters)
    ):
        raise DiagnosticRunnerError("runtime optimizer did not initialize every parameter")
    native_step_counter_count = 0
    actual_state_by_name: dict[str, Mapping[str, Tensor]] = {}
    bit_exact_state_tensor_count = 0
    for name, parameter in zip(names, parameters, strict=True):
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping):
            raise DiagnosticRunnerError("runtime optimizer parameter state is missing")
        expected_state = reference_state[name]
        if not isinstance(expected_state, Mapping) or set(state) != set(expected_state):
            raise DiagnosticRunnerError(
                f"runtime optimizer state fields differ: {name}"
            )
        if spec.name == "Adam":
            expected_keys = {"step", "exp_avg", "exp_avg_sq"}
            if spec.amsgrad:
                expected_keys.add("max_exp_avg_sq")
            if set(state) != expected_keys or set(expected_state) != expected_keys:
                raise DiagnosticRunnerError("runtime Adam state fields differ")
            step = state["step"]
            expected_step = expected_state["step"]
            if (
                not isinstance(step, Tensor)
                or not isinstance(expected_step, Tensor)
                or step.numel() != 1
                or step.shape != expected_step.shape
                or step.dtype != torch.float32
                or expected_step.dtype != torch.float32
                or step.device.type != "cpu"
                or expected_step.device.type != "cpu"
                or float(step.detach().cpu().item()) != 1.0
                or not torch.equal(step, expected_step)
            ):
                raise DiagnosticRunnerError(
                    "runtime Adam parameter step value/dtype/device differs"
                )
            native_step_counter_count += 1
            state_tensor_names = expected_keys - {"step"}
        else:
            if set(state) != {"momentum_buffer"} or set(expected_state) != {
                "momentum_buffer"
            }:
                raise DiagnosticRunnerError("runtime SGD state fields differ")
            state_tensor_names = {"momentum_buffer"}
        for field_name in state_tensor_names:
            value = state[field_name]
            expected_value = expected_state[field_name]
            if (
                not isinstance(value, Tensor)
                or not isinstance(expected_value, Tensor)
                or value.shape != parameter.shape
                or value.shape != expected_value.shape
                or value.dtype != parameter.dtype
                or value.dtype != expected_value.dtype
                or value.device != parameter.device
                or value.device != expected_value.device
                or not bool(torch.isfinite(value).all().item())
                or not torch.equal(value, expected_value)
            ):
                raise DiagnosticRunnerError(
                    f"runtime optimizer state tensor differs: {name}.{field_name}"
                )
            bit_exact_state_tensor_count += 1
        if spec.name == "Adam":
            bit_exact_state_tensor_count += 1  # step
        actual_state_by_name[name] = {
            field_name: state[field_name] for field_name in sorted(state)
        }
    reference_state_sha256, reference_state_tensor_count = (
        _optimizer_state_bundle_sha256(names, reference_state)
    )
    actual_state_sha256, actual_state_tensor_count = _optimizer_state_bundle_sha256(
        names, actual_state_by_name
    )
    if (
        reference_state_tensor_count != actual_state_tensor_count
        or bit_exact_state_tensor_count != actual_state_tensor_count
        or reference_state_sha256 != actual_state_sha256
    ):
        raise DiagnosticRunnerError("runtime optimizer state bundle differs")
    return {
        "optimizer_step_call_count": 1,
        "logical_step_count_per_parameter": 1,
        "all_parameters_reached_logical_step_one": True,
        "native_step_counter_exposed": spec.name == "Adam",
        "native_step_counter_one_parameter_count": native_step_counter_count,
        "state_parameter_count": len(optimizer.state),
        "state_tensor_count": actual_state_tensor_count,
        "bit_exact_state_tensor_count": bit_exact_state_tensor_count,
        "all_optimizer_state_tensors_bit_exact": True,
        "reference_optimizer_state_bundle_sha256": reference_state_sha256,
        "actual_optimizer_state_bundle_sha256": actual_state_sha256,
        "adam_step_dtype": "torch.float32" if spec.name == "Adam" else None,
        "adam_step_device_type": "cpu" if spec.name == "Adam" else None,
    }


def _build_model(config: Mapping[str, Any], dataset: str, device: torch.device):
    import test_source as source_runner

    if dataset not in config.get("datasets", {}):
        raise DiagnosticRunnerError(f"unknown dataset: {dataset}")
    checkpoint = _lexical_absolute(
        PROJECT_ROOT / config["datasets"][dataset]["checkpoint"]["path"]
    )
    expected = str(config["datasets"][dataset]["checkpoint"]["sha256"])
    try:
        checkpoint_before = read_stable_regular_file(checkpoint)
    except (OSError, ValueError) as exc:
        raise DiagnosticRunnerError(f"checkpoint binding failed: {dataset}") from exc
    if checkpoint_before.sha256 != expected:
        raise DiagnosticRunnerError(f"checkpoint binding failed: {dataset}")
    model = source_runner.build_nsfpn_model()
    # Load the exact bytes captured by the stable no-follow snapshot.  Opening
    # the pathname again inside ``torch.load`` would leave a swap-and-restore
    # window between the two surrounding identity checks.
    checkpoint_buffer = io.BytesIO(checkpoint_before.data)
    try:
        try:
            checkpoint_value = torch.load(
                checkpoint_buffer,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:  # Compatibility with the frozen supported Torch.
            checkpoint_buffer.seek(0)
            checkpoint_value = torch.load(checkpoint_buffer, map_location="cpu")
        state_dict, wrapper = source_runner.extract_state_dict(checkpoint_value)
        expected_state = model.state_dict()
        missing = sorted(set(expected_state) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected_state))
        if missing or unexpected:
            raise DiagnosticRunnerError(
                "checkpoint keys do not exactly match the D0 model; "
                f"missing={missing}, unexpected={unexpected}"
            )
        shape_mismatches = {
            key: (tuple(state_dict[key].shape), tuple(expected_state[key].shape))
            for key in expected_state
            if tuple(state_dict[key].shape) != tuple(expected_state[key].shape)
        }
        if shape_mismatches:
            raise DiagnosticRunnerError(
                f"checkpoint tensor shapes differ for D0: {shape_mismatches}"
            )
        model.load_state_dict(state_dict, strict=True)
    finally:
        checkpoint_buffer.close()
    checkpoint_after = read_stable_regular_file(checkpoint)
    if (
        checkpoint_after.sha256 != checkpoint_before.sha256
        or checkpoint_after.device != checkpoint_before.device
        or checkpoint_after.inode != checkpoint_before.inode
        or checkpoint_after.size_bytes != checkpoint_before.size_bytes
        or checkpoint_after.mtime_ns != checkpoint_before.mtime_ns
        or checkpoint_after.ctime_ns != checkpoint_before.ctime_ns
    ):
        raise DiagnosticRunnerError(f"checkpoint changed while loading: {dataset}")
    model.to(device)
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    parameters, names = collect_adaptable_params(
        model, group_spec=AdaptableGroupSpec(("all_bn",))
    )
    adapter_parameters, adapter_names = adapter.collect_adaptable_params()
    if names != adapter_names or tuple(map(id, parameters)) != tuple(
        map(id, adapter_parameters)
    ):
        raise DiagnosticRunnerError("parameter-group and adapter inventories differ")
    assignment = _parameter_group_assignment(model)
    source_parameters = tuple(parameter.detach().clone() for parameter in parameters)
    source_state = _model_state_snapshot(model)
    return (
        model,
        adapter,
        tuple(parameters),
        tuple(names),
        assignment,
        source_parameters,
        source_state,
        wrapper,
    )


def _label_free_episode(
    *,
    adapter: IRSTDModelAdapter,
    parameters: Sequence[nn.Parameter],
    names: Sequence[str],
    source_parameters: Sequence[Tensor],
    assignment: Mapping[str, str],
    image: Tensor,
    metadata: Mapping[str, Any],
    candidates: Sequence[Candidate],
    config: Mapping[str, Any],
    dataset: str,
) -> LabelFreeEpisode:
    determinism_config = config["method"]["determinism"]
    _restore_parameters(parameters, source_parameters)
    adapter.set_tent_mode(use_batch_stats=False)
    adapter.model.zero_grad(set_to_none=True)
    assert_strict_forward_policy(determinism_config)
    logits_pre = adapter.forward_logits(image)
    entropy = binary_entropy_map(
        logits_pre, eps=float(config["method"]["entropy_eps"])
    ).mean()
    entropy_backward_determinism = _backward(
        entropy,
        scope="entropy_backward",
        config=config,
    )
    gradients_gpu = tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )
    if any(value is None for value in gradients_gpu):
        missing = [
            name
            for name, value in zip(names, gradients_gpu, strict=True)
            if value is None
        ]
        raise DiagnosticRunnerError(
            f"entropy gradient missing adaptable parameters: {missing[:5]}"
        )
    if not any(
        value is not None and bool(torch.count_nonzero(value).item())
        for value in gradients_gpu
    ):
        raise DiagnosticRunnerError("entropy gradient is globally zero")
    before_cpu = _nonnull(_clone_named(names, source_parameters))
    gradient_cpu = _clone_named(names, gradients_gpu)
    geometry_scope = _provenance(
        config=config, dataset=dataset, oracle=False, accesses=0
    )
    candidate_states: dict[str, CandidateState] = {}
    geometry_config = config["evaluation"]["optimizer_geometry"]
    for candidate in candidates:
        _restore_parameters(parameters, source_parameters)
        for parameter, gradient in zip(parameters, gradients_gpu, strict=True):
            parameter.grad = None if gradient is None else gradient.clone()
        optimizer = build_binary_tent_optimizer(
            parameters,
            name=candidate.optimizer,
            learning_rate=candidate.learning_rate,
        )
        runtime_spec, runtime_hard_gate = _verify_actual_optimizer_pre_step(
            optimizer,
            candidate=candidate,
            parameters=parameters,
        )
        native_reference = pytorch_first_step_reference(
            parameters_before=tuple(zip(names, source_parameters, strict=True)),
            gradients=tuple(zip(names, gradients_gpu, strict=True)),
            optimizer=runtime_spec,
        )
        reference_after_gpu = native_reference.parameters_after
        strict_before_optimizer_step = bool(
            assert_strict_forward_policy(determinism_config)
        )
        optimizer.step()
        runtime_hard_gate.update(
            _verify_actual_optimizer_post_step(
                optimizer,
                spec=runtime_spec,
                names=names,
                parameters=parameters,
                reference_state=native_reference.optimizer_state,
            )
        )
        per_tensor_bit_exact = {
            name: bool(torch.equal(parameter.detach(), reference_after_gpu[name]))
            for name, parameter in zip(names, parameters, strict=True)
        }
        if not all(per_tensor_bit_exact.values()):
            mismatched = [
                name for name, exact in per_tensor_bit_exact.items() if not exact
            ]
            raise DiagnosticRunnerError(
                "optimizer same-device storage reference mismatch: "
                f"{candidate.slug}; tensors={mismatched[:5]}"
            )
        after_cpu = _nonnull(_clone_named(names, tuple(parameters)))
        reference_after_cpu = {
            name: reference_after_gpu[name].detach().cpu().clone() for name in names
        }
        runtime_hard_gate.update(
            {
                "same_device_reference": True,
                "native_storage_dtype_reference": True,
                "comparison": "torch.equal_per_parameter_tensor",
                "bit_exact_parameter_tensor_count": sum(
                    per_tensor_bit_exact.values()
                ),
                "all_parameters_bit_exact": True,
                "reference_after_bundle_sha256": _named_tensor_bundle_sha256(
                    names, reference_after_cpu
                ),
                "actual_after_bundle_sha256": _named_tensor_bundle_sha256(
                    names, after_cpu
                ),
            }
        )
        if (
            runtime_hard_gate["reference_after_bundle_sha256"]
            != runtime_hard_gate["actual_after_bundle_sha256"]
        ):
            raise DiagnosticRunnerError(
                f"optimizer hard-gate bundle mismatch: {candidate.slug}"
            )
        step_cpu = {
            name: after_cpu[name] - before_cpu[name]
            for name in names
        }
        geometry = analyze_optimizer_first_step(
            parameters_before=before_cpu,
            gradients=gradient_cpu,
            parameters_after=after_cpu,
            optimizer_config=runtime_spec.to_dict(),
            provenance=geometry_scope,
            parameter_groups=assignment,
            small_gradient_threshold=float(
                geometry_config["small_gradient_threshold"]
            ),
            near_sign_step_threshold=float(
                geometry_config["adam_near_sign_step_threshold"]
            ),
            verification_rtol=float(geometry_config["verification_rtol"]),
            verification_atol=float(geometry_config["verification_atol"]),
        )
        geometry["runtime_same_device_hard_gate"] = runtime_hard_gate
        strict_before_post_forward = bool(
            assert_strict_forward_policy(determinism_config)
        )
        with torch.no_grad():
            logits_post = adapter.forward_logits(image).detach().cpu().clone()
        candidate_states[candidate.slug] = CandidateState(
            logits_post=logits_post,
            parameters_after=after_cpu,
            step=step_cpu,
            geometry=geometry,
            strict_policy_before_optimizer_step=strict_before_optimizer_step,
            strict_policy_before_post_forward=strict_before_post_forward,
        )
        del optimizer
        _restore_parameters(parameters, source_parameters)
        _assert_parameters_exact(parameters, source_parameters, candidate.slug)
    adapter.model.zero_grad(set_to_none=True)
    _restore_parameters(parameters, source_parameters)
    adapter.set_source_eval_mode()
    return LabelFreeEpisode(
        metadata=dict(metadata),
        logits_pre=logits_pre.detach().cpu().clone(),
        entropy_pre=float(entropy.detach().cpu().item()),
        parameters_before=before_cpu,
        entropy_gradients=gradient_cpu,
        entropy_backward_determinism=entropy_backward_determinism,
        candidates=candidate_states,
    )


def _supervised_gradients(
    *,
    model: nn.Module,
    adapter: IRSTDModelAdapter,
    parameters: Sequence[nn.Parameter],
    names: Sequence[str],
    source_parameters: Sequence[Tensor],
    image: Tensor,
    target: Tensor,
    task_contract: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Tensor], float, dict[str, Any]]:
    _restore_parameters(parameters, source_parameters)
    adapter.set_tent_mode(use_batch_stats=False)
    model.zero_grad(set_to_none=True)
    warm_flag = bool(task_contract["warm_flag"])
    warm_epochs = int(task_contract["warm_epochs"])
    epoch_index = int(task_contract["epoch_index"])
    with_shape = bool(task_contract["with_shape"])
    expected_auxiliary_count = int(task_contract["expected_auxiliary_count"])
    assert_strict_forward_policy(config["method"]["determinism"])
    auxiliary, prediction = model(image, warm_flag)
    if (
        not isinstance(auxiliary, Sequence)
        or len(auxiliary) != expected_auxiliary_count
    ):
        raise DiagnosticRunnerError(
            "source-eval post-warm task graph auxiliary count drifted"
        )
    loss = SLSIoULoss()(
        prediction,
        target,
        warm_epochs,
        epoch_index,
        with_shape=with_shape,
    )
    if not torch.isfinite(loss):
        raise DiagnosticRunnerError("supervised diagnostic loss is non-finite")
    backward_audit = _backward(
        loss,
        scope="supervised_task_backward",
        config=config,
    )
    gradients = tuple(parameter.grad for parameter in parameters)
    if any(value is None for value in gradients):
        missing = [
            name for name, value in zip(names, gradients, strict=True) if value is None
        ]
        raise DiagnosticRunnerError(
            f"supervised task gradient missing parameters: {missing[:5]}"
        )
    result = _nonnull(_clone_named(names, gradients))
    model.zero_grad(set_to_none=True)
    _restore_parameters(parameters, source_parameters)
    adapter.set_source_eval_mode()
    return result, float(loss.detach().cpu().item()), backward_audit


def _thresholds(config: Mapping[str, Any]) -> NoOpThresholds:
    evaluation = config["evaluation"]
    floors = evaluation["no_op_null_floors"]
    interval = evaluation["near_threshold_interval"]
    return NoOpThresholds(
        parameter_null_floor=float(floors["parameter"]),
        logit_null_floor=float(floors["logit"]),
        probability_null_floor=float(floors["probability"]),
        probability_logit_consistency_atol=1e-6,
        prediction_threshold=float(evaluation["prediction_threshold"]),
        near_threshold_lower=float(interval[0]),
        near_threshold_upper=float(interval[1]),
        entropy_eps=float(config["method"]["entropy_eps"]),
        connectivity=int(evaluation["connectivity"]),
        min_component_area=int(evaluation["min_component_area"]),
        max_centroid_distance=float(evaluation["max_centroid_distance"]),
    )


def _prune_noop(value: dict[str, Any], *, formal: bool) -> dict[str, Any]:
    if formal:
        parameter = value.get("parameter_change")
        if isinstance(parameter, dict):
            parameter.pop("per_tensor", None)
    return value


def _iou(counts: Mapping[str, Any]) -> float:
    # ``analyze_noop_episode`` exposes the frozen evaluator's sufficient
    # statistics as ``intersection_pixels`` and ``union_pixels``.  Consume
    # that canonical schema directly so the D0 summary cannot silently drift
    # from the evaluator or rely on a test-only alias.
    intersection = int(counts["intersection_pixels"])
    union = int(counts["union_pixels"])
    return 1.0 if union == 0 else intersection / union


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise DiagnosticRunnerError("cannot summarize zero D0 records")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["candidate_slug"])].append(record)
    candidates: dict[str, Any] = {}
    for slug, values in sorted(grouped.items()):
        classifications = Counter(
            str(value["noop"]["classification"]) for value in values
        )
        iou_deltas = [
            _iou(value["noop"]["metric_counts"]["post"])
            - _iou(value["noop"]["metric_counts"]["pre"])
            for value in values
        ]
        group_ids = tuple(values[0]["entropy_task_alignment"]["per_group"])
        if any(
            tuple(value["entropy_task_alignment"]["per_group"]) != group_ids
            or tuple(value["optimizer_geometry"]["per_group"]) != group_ids
            for value in values
        ):
            raise DiagnosticRunnerError("per-group diagnostic topology changed")
        per_group: dict[str, Any] = {}
        for group_id in group_ids:
            alignment_values = [
                value["entropy_task_alignment"]["per_group"][group_id]
                for value in values
            ]
            geometry_values = [
                value["optimizer_geometry"]["per_group"][group_id]
                for value in values
            ]
            cosines = [
                float(value["entropy_supervised_cosine"])
                for value in alignment_values
                if value["entropy_supervised_cosine"] is not None
            ]
            sign_fractions = [
                float(value["near_sign_step_fraction"])
                for value in geometry_values
                if value["near_sign_step_fraction"] is not None
            ]
            per_group[group_id] = {
                "episode_count": len(values),
                "mean_entropy_supervised_cosine": (
                    None if not cosines else math.fsum(cosines) / len(cosines)
                ),
                "mean_supervised_dot_adaptation_step": math.fsum(
                    float(value["supervised_dot_adaptation_step"])
                    for value in alignment_values
                )
                / len(values),
                "predicted_task_loss_decrease_count": sum(
                    value["first_order_task_effect"]
                    == "predicted_task_loss_decrease"
                    for value in alignment_values
                ),
                "predicted_task_loss_increase_count": sum(
                    value["first_order_task_effect"]
                    == "predicted_task_loss_increase"
                    for value in alignment_values
                ),
                "first_order_neutral_count": sum(
                    value["first_order_task_effect"] == "first_order_neutral"
                    for value in alignment_values
                ),
                "mean_actual_step_norm": math.fsum(
                    float(value["actual_step_norm"]) for value in geometry_values
                )
                / len(values),
                "mean_relative_actual_step_norm": math.fsum(
                    float(value["relative_actual_step_norm"] or 0.0)
                    for value in geometry_values
                )
                / len(values),
                "mean_small_gradient_fraction": math.fsum(
                    float(value["small_gradient_fraction"] or 0.0)
                    for value in geometry_values
                )
                / len(values),
                "mean_adam_near_sign_step_fraction": (
                    None
                    if not sign_fractions
                    else math.fsum(sign_fractions) / len(sign_fractions)
                ),
            }
        margin_strata = tuple(values[0]["noop"]["threshold_margin"]["strata"])
        threshold_margin: dict[str, Any] = {}
        for stratum in margin_strata:
            strata = [
                value["noop"]["threshold_margin"]["strata"][stratum]
                for value in values
            ]
            pixel_count = sum(int(value["pixel_count"]) for value in strata)
            gt_margin = sum(
                int(value["abs_delta_gt_margin_count"]) for value in strata
            )
            gt_tenth = sum(
                int(value["abs_delta_gt_0_1_margin_count"]) for value in strata
            )
            threshold_margin[stratum] = {
                "pixel_count": pixel_count,
                "abs_delta_gt_margin_count": gt_margin,
                "fraction_abs_delta_gt_margin": (
                    None if pixel_count == 0 else gt_margin / pixel_count
                ),
                "abs_delta_gt_0_1_margin_count": gt_tenth,
                "fraction_abs_delta_gt_0_1_margin": (
                    None if pixel_count == 0 else gt_tenth / pixel_count
                ),
                "binary_xor_count": sum(
                    int(value["binary_xor_count"]) for value in strata
                ),
            }
        candidates[slug] = {
            "candidate": values[0]["candidate"],
            "episode_count": len(values),
            "classification_counts": dict(sorted(classifications.items())),
            "mean_iou_delta": math.fsum(iou_deltas) / len(iou_deltas),
            "positive_iou_episode_count": sum(value > 0.0 for value in iou_deltas),
            "negative_iou_episode_count": sum(value < 0.0 for value in iou_deltas),
            "threshold_xor_episode_count": sum(
                int(value["noop"]["binary_transitions"]["binary_pixel_xor_count"])
                > 0
                for value in values
            ),
            "cpu_storage_replay_within_frozen_tolerance_episode_count": sum(
                value["optimizer_geometry"]["global"][
                    "cpu_storage_replay_within_frozen_tolerance"
                ]
                is True
                for value in values
            ),
            "first_order_task_decrease_episode_count": sum(
                value["entropy_task_alignment"]["global"][
                    "first_order_task_effect"
                ]
                == "predicted_task_loss_decrease"
                for value in values
            ),
            "first_order_task_increase_episode_count": sum(
                value["entropy_task_alignment"]["global"][
                    "first_order_task_effect"
                ]
                == "predicted_task_loss_increase"
                for value in values
            ),
            "per_group": per_group,
            "threshold_margin": threshold_margin,
        }
    return {
        "schema_version": 1,
        "artifact_type": "tent_failure_diagnostics_summary_v1",
        "oracle_analysis": True,
        "paper_test_result": False,
        "record_count": len(records),
        "candidates": candidates,
    }


_D0_RECORD_FIELDS = {
    "schema_version",
    "dataset",
    "split_role",
    "image_id",
    "corruption",
    "severity",
    "candidate_slug",
    "candidate",
    "optimization_entropy_pre",
    "source_eval_post_warm_task_loss",
    "noop",
    "optimizer_geometry",
    "entropy_task_alignment",
    "determinism",
    "scope",
}

_D0_RECORD_SCOPE = {
    "oracle_analysis": True,
    "paper_test_result": False,
    "source_train_derived": True,
    "method_label_accesses": 0,
    "outer_evaluator_label_accesses": 1,
    "use_test_images": False,
    "use_test_labels": False,
}

_D0_NOOP_CLASSIFICATIONS = {
    "numeric_noop",
    "functional_noop",
    "threshold_noop",
    "metric_noop",
    "task_effective_change",
}


def _require_exact_keys(
    value: Any, expected: set[str], label: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DiagnosticRunnerError(f"{label} must be a mapping")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise DiagnosticRunnerError(
            f"{label} fields differ; missing={missing}, unknown={unknown}"
        )
    return value


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiagnosticRunnerError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise DiagnosticRunnerError(f"{label} must be a finite number")
    return result


def _validate_analysis_scope(
    value: Any,
    *,
    require_outer_oracle: bool,
    dataset: str,
    config: Mapping[str, Any],
    label: str,
) -> SourceTrainAnalysisProvenance:
    try:
        parsed = SourceTrainAnalysisProvenance.from_mapping(
            value, require_outer_oracle=require_outer_oracle
        )
    except ValueError as exc:
        raise DiagnosticRunnerError(f"{label} is invalid: {exc}") from exc
    expected = config["datasets"][dataset]
    if (
        parsed.dataset != dataset
        or parsed.split_sha256 != str(expected["train_split_sha256"])
        or parsed.checkpoint_sha256 != str(expected["checkpoint"]["sha256"])
        or parsed.seed != 42
    ):
        raise DiagnosticRunnerError(f"{label} source binding differs")
    return parsed


def validate_d0_records(
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    dataset: str,
    conditions: Sequence[str],
    candidates: Sequence[Candidate],
    images_per_condition: int,
) -> None:
    """Validate the complete diagnostic payload, not only metric counts.

    The frozen-v2 equality gate deliberately consumes only integer evaluator
    sufficient statistics.  This independent validator proves that the same
    records also contain the D0-specific no-op, optimizer-geometry, and
    supervised-alignment evidence before a shard can be accepted.
    """

    if dataset not in config.get("datasets", {}):
        raise DiagnosticRunnerError("D0 record dataset is outside the config")
    condition_set = set(conditions)
    candidate_by_slug = {candidate.slug: candidate for candidate in candidates}
    expected_count = len(conditions) * len(candidates) * images_per_condition
    if len(records) != expected_count:
        raise DiagnosticRunnerError(
            f"D0 diagnostic payload count differs: {len(records)} != {expected_count}"
        )
    topology: tuple[Any, ...] | None = None
    observed_cells: Counter[tuple[str, str]] = Counter()
    for index, raw_record in enumerate(records):
        label = f"D0 record[{index}]"
        record = _require_exact_keys(raw_record, _D0_RECORD_FIELDS, label)
        if (
            record.get("schema_version") != 1
            or record.get("dataset") != dataset
            or record.get("split_role") != "train"
            or not isinstance(record.get("image_id"), str)
            or not record.get("image_id")
        ):
            raise DiagnosticRunnerError(f"{label} identity/scope is invalid")
        condition = _condition_from_record(record, label)
        if condition not in condition_set:
            raise DiagnosticRunnerError(f"{label} condition is outside the run scope")
        candidate = _candidate_from_mapping(record.get("candidate"), f"{label}.candidate")
        if (
            candidate_by_slug.get(candidate.slug) != candidate
            or record.get("candidate_slug") != candidate.slug
        ):
            raise DiagnosticRunnerError(f"{label} candidate binding differs")
        observed_cells[(condition, candidate.slug)] += 1
        _finite_number(
            record.get("optimization_entropy_pre"),
            f"{label}.optimization_entropy_pre",
            nonnegative=True,
        )
        _finite_number(
            record.get("source_eval_post_warm_task_loss"),
            f"{label}.source_eval_post_warm_task_loss",
            nonnegative=True,
        )
        scope = _require_exact_keys(
            record.get("scope"), set(_D0_RECORD_SCOPE), f"{label}.scope"
        )
        if dict(scope) != _D0_RECORD_SCOPE:
            raise DiagnosticRunnerError(f"{label}.scope differs from outer-oracle D0")

        noop = record.get("noop")
        if not isinstance(noop, Mapping) or noop.get("schema_version") != 1:
            raise DiagnosticRunnerError(f"{label}.noop schema is invalid")
        if noop.get("classification") not in _D0_NOOP_CLASSIFICATIONS:
            raise DiagnosticRunnerError(f"{label}.noop classification is invalid")
        metric_counts = noop.get("metric_counts")
        if (
            not isinstance(metric_counts, Mapping)
            or set(metric_counts) != {"identical", "post", "pre"}
            or not isinstance(metric_counts.get("identical"), bool)
        ):
            raise DiagnosticRunnerError(f"{label}.noop metric counts are invalid")
        for endpoint in ("pre", "post"):
            values = metric_counts.get(endpoint)
            if not isinstance(values, Mapping):
                raise DiagnosticRunnerError(
                    f"{label}.noop.metric_counts.{endpoint} is invalid"
                )
            _endpoint_fractions(values, f"{label}.noop.metric_counts.{endpoint}")

        geometry = record.get("optimizer_geometry")
        alignment = record.get("entropy_task_alignment")
        if not isinstance(geometry, Mapping) or not isinstance(alignment, Mapping):
            raise DiagnosticRunnerError(f"{label} diagnostic analyses are missing")
        _validate_analysis_scope(
            geometry.get("scope"),
            require_outer_oracle=False,
            dataset=dataset,
            config=config,
            label=f"{label}.optimizer_geometry.scope",
        )
        _validate_analysis_scope(
            alignment.get("scope"),
            require_outer_oracle=True,
            dataset=dataset,
            config=config,
            label=f"{label}.entropy_task_alignment.scope",
        )
        if (
            geometry.get("schema_version") != OPTIMIZER_GEOMETRY_SCHEMA_VERSION
            or geometry.get("analysis_type") != "tent_optimizer_first_step_geometry"
            or geometry.get("empty_optimizer_state_verified") is not True
            or geometry.get("current_binary_tent_optimizer_configuration") is not True
            or alignment.get("analysis_type") != "entropy_task_gradient_alignment"
        ):
            raise DiagnosticRunnerError(f"{label} analysis contract differs")
        geometry_config = config["evaluation"]["optimizer_geometry"]
        references = _require_exact_keys(
            geometry.get("references"),
            {"cross_backend_cpu_storage_replay", "continuous_ideal"},
            f"{label}.optimizer_geometry.references",
        )
        verification_reference = _require_exact_keys(
            references["cross_backend_cpu_storage_replay"],
            {
                "name",
                "role",
                "used_for_hard_gate",
                "pytorch_version",
                "single_tensor_operation_order",
                "native_storage_dtype",
                "final_parameter_storage_rounding_included",
            },
            f"{label}.optimizer_geometry.references.cross_backend_cpu_storage_replay",
        )
        ideal_reference = _require_exact_keys(
            references["continuous_ideal"],
            {"name", "role", "dtype", "used_for_hard_gate"},
            f"{label}.optimizer_geometry.references.continuous_ideal",
        )
        if (
            verification_reference.get("name")
            != geometry_config["cross_backend_cpu_storage_replay_reference"]
            or verification_reference.get("role")
            != "cross_backend_diagnostic_only"
            or verification_reference.get("used_for_hard_gate") is not False
            or verification_reference.get("pytorch_version")
            != PYTORCH_REFERENCE_VERSION
            or verification_reference.get("single_tensor_operation_order") is not True
            or verification_reference.get("native_storage_dtype") is not True
            or verification_reference.get("final_parameter_storage_rounding_included")
            is not True
            or ideal_reference.get("name")
            != geometry_config["continuous_ideal_reference"]
            or ideal_reference.get("role")
            != "scientific_geometry_explanation_only"
            or ideal_reference.get("dtype") != "torch.float64"
            or ideal_reference.get("used_for_hard_gate") is not False
            or geometry.get("first_step_formula_role")
            != "continuous_float64_explanation_only"
        ):
            raise DiagnosticRunnerError(f"{label} optimizer reference contract differs")
        gate_policy = _require_exact_keys(
            geometry.get("gate_policy"),
            {
                "optimizer_correctness_acceptance_gate",
                "runtime_same_device_hard_gate_is_sole_acceptance_gate",
                "cpu_storage_replay_used_for_hard_gate",
                "continuous_ideal_used_for_hard_gate",
            },
            f"{label}.optimizer_geometry.gate_policy",
        )
        if (
            gate_policy["optimizer_correctness_acceptance_gate"]
            != "runtime_same_device_hard_gate"
            or gate_policy["runtime_same_device_hard_gate_is_sole_acceptance_gate"]
            is not True
            or gate_policy["cpu_storage_replay_used_for_hard_gate"] is not False
            or gate_policy["continuous_ideal_used_for_hard_gate"] is not False
        ):
            raise DiagnosticRunnerError(f"{label} optimizer gate policy differs")
        thresholds = _require_exact_keys(
            geometry.get("thresholds"),
            {
                "small_gradient_absolute_threshold",
                "adam_near_sign_step_ratio_threshold",
                "verification_rtol",
                "verification_atol",
            },
            f"{label}.optimizer_geometry.thresholds",
        )
        threshold_bindings = {
            "small_gradient_absolute_threshold": "small_gradient_threshold",
            "adam_near_sign_step_ratio_threshold": "adam_near_sign_step_threshold",
            "verification_rtol": "verification_rtol",
            "verification_atol": "verification_atol",
        }
        for receipt_name, config_name in threshold_bindings.items():
            observed_threshold = thresholds[receipt_name]
            frozen_threshold = geometry_config[config_name]
            if (
                type(observed_threshold) is not float
                or type(frozen_threshold) is not float
                or observed_threshold != frozen_threshold
            ):
                raise DiagnosticRunnerError(
                    f"{label} optimizer threshold binding differs: {receipt_name}"
                )
        hard_gate_value = geometry.get("runtime_same_device_hard_gate")
        hard_gate_fields = {
            "schema_version",
            "reference_name",
            "pytorch_version_required",
            "pytorch_version_observed",
            "pytorch_default_dtype",
            "pytorch_default_device_type",
            "optimizer_class",
            "optimizer_config_from_actual_param_group",
            "implementation_flags_from_actual_param_group",
            "defaults_from_actual_optimizer",
            "single_param_group_verified",
            "ordered_parameter_identity_verified",
            "parameter_tensor_count",
            "initial_optimizer_state_empty",
            "optimizer_step_call_count",
            "logical_step_count_per_parameter",
            "all_parameters_reached_logical_step_one",
            "native_step_counter_exposed",
            "native_step_counter_one_parameter_count",
            "state_parameter_count",
            "state_tensor_count",
            "bit_exact_state_tensor_count",
            "all_optimizer_state_tensors_bit_exact",
            "reference_optimizer_state_bundle_sha256",
            "actual_optimizer_state_bundle_sha256",
            "adam_step_dtype",
            "adam_step_device_type",
            "same_device_reference",
            "native_storage_dtype_reference",
            "comparison",
            "bit_exact_parameter_tensor_count",
            "all_parameters_bit_exact",
            "reference_after_bundle_sha256",
            "actual_after_bundle_sha256",
        }
        hard_gate = _require_exact_keys(
            hard_gate_value,
            hard_gate_fields,
            f"{label}.optimizer_geometry.runtime_same_device_hard_gate",
        )
        expected_flags = (
            {
                "foreach": False,
                "fused": False,
                "capturable": False,
                "differentiable": False,
            }
            if candidate.optimizer == "Adam"
            else {
                "foreach": False,
                "fused": None,
                "capturable": None,
                "differentiable": False,
            }
        )
        expected_defaults = (
            {
                "amsgrad": False,
                "betas": [0.9, 0.999],
                "capturable": False,
                "differentiable": False,
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
                "lr": candidate.learning_rate,
                "maximize": False,
                "weight_decay": 0.0,
            }
            if candidate.optimizer == "Adam"
            else {
                "dampening": 0.0,
                "differentiable": False,
                "foreach": False,
                "lr": candidate.learning_rate,
                "maximize": False,
                "momentum": 0.9,
                "nesterov": True,
                "weight_decay": 0.0,
            }
        )
        expected_native_step_count = 106 if candidate.optimizer == "Adam" else 0
        expected_state_tensor_count = (
            106 * 3 if candidate.optimizer == "Adam" else 106
        )
        if (
            hard_gate.get("schema_version") != 1
            or hard_gate.get("reference_name")
            != geometry_config["runtime_hard_gate_reference"]
            or hard_gate.get("pytorch_version_required")
            != PYTORCH_REFERENCE_VERSION
            or hard_gate.get("pytorch_version_observed")
            != PYTORCH_REFERENCE_VERSION
            or hard_gate.get("pytorch_default_dtype") != "torch.float32"
            or hard_gate.get("pytorch_default_device_type") != "cpu"
            or hard_gate.get("optimizer_class") != candidate.optimizer
            or hard_gate.get("optimizer_config_from_actual_param_group")
            != _optimizer_mapping(candidate)
            or hard_gate.get("implementation_flags_from_actual_param_group")
            != expected_flags
            or hard_gate.get("defaults_from_actual_optimizer") != expected_defaults
            or hard_gate.get("single_param_group_verified") is not True
            or hard_gate.get("ordered_parameter_identity_verified") is not True
            or hard_gate.get("parameter_tensor_count") != 106
            or hard_gate.get("initial_optimizer_state_empty") is not True
            or hard_gate.get("optimizer_step_call_count") != 1
            or hard_gate.get("logical_step_count_per_parameter") != 1
            or hard_gate.get("all_parameters_reached_logical_step_one") is not True
            or hard_gate.get("native_step_counter_exposed")
            is not (candidate.optimizer == "Adam")
            or hard_gate.get("native_step_counter_one_parameter_count")
            != expected_native_step_count
            or hard_gate.get("state_parameter_count") != 106
            or hard_gate.get("state_tensor_count") != expected_state_tensor_count
            or hard_gate.get("bit_exact_state_tensor_count")
            != expected_state_tensor_count
            or hard_gate.get("all_optimizer_state_tensors_bit_exact") is not True
            or not _is_lower_sha256(
                hard_gate.get("reference_optimizer_state_bundle_sha256")
            )
            or hard_gate.get("reference_optimizer_state_bundle_sha256")
            != hard_gate.get("actual_optimizer_state_bundle_sha256")
            or hard_gate.get("adam_step_dtype")
            != ("torch.float32" if candidate.optimizer == "Adam" else None)
            or hard_gate.get("adam_step_device_type")
            != ("cpu" if candidate.optimizer == "Adam" else None)
            or hard_gate.get("same_device_reference") is not True
            or hard_gate.get("native_storage_dtype_reference") is not True
            or hard_gate.get("comparison") != "torch.equal_per_parameter_tensor"
            or hard_gate.get("bit_exact_parameter_tensor_count") != 106
            or hard_gate.get("all_parameters_bit_exact") is not True
            or not _is_lower_sha256(
                hard_gate.get("reference_after_bundle_sha256")
            )
            or hard_gate.get("reference_after_bundle_sha256")
            != hard_gate.get("actual_after_bundle_sha256")
        ):
            raise DiagnosticRunnerError(f"{label} optimizer hard gate differs")
        geometry_global = geometry.get("global")
        geometry_groups = geometry.get("per_group")
        alignment_groups = alignment.get("per_group")
        common_geometry_metric_keys = {
            "scalar_count",
            "active_gradient_scalar_count",
            "actual_step_norm",
            "cross_backend_cpu_storage_replay_expected_step_norm",
            "cross_backend_cpu_storage_replay_cos_actual_expected",
            "cross_backend_cpu_storage_replay_actual_to_expected_norm_ratio",
            "cross_backend_cpu_storage_replay_residual_norm",
            "cross_backend_cpu_storage_replay_max_abs_residual",
            "continuous_ideal_expected_step_norm",
            "continuous_ideal_cos_actual_expected",
            "continuous_ideal_actual_to_expected_norm_ratio",
            "continuous_ideal_residual_norm",
            "continuous_ideal_max_abs_residual",
            "small_gradient_scalar_count",
            "small_gradient_fraction",
            "near_sign_step_threshold",
            "near_sign_step_scalar_count",
            "near_sign_step_fraction",
            "gradient_norm",
            "effective_gradient_norm",
            "parameter_norm_before",
            "relative_actual_step_norm",
            "cpu_storage_replay_within_frozen_tolerance",
        }
        if (
            not isinstance(geometry_global, Mapping)
            or not isinstance(geometry_groups, Mapping)
            or not geometry_groups
            or not isinstance(alignment_groups, Mapping)
            or tuple(geometry_groups) != tuple(alignment_groups)
            or any(
                not isinstance(value, Mapping)
                or type(value.get(
                    "cpu_storage_replay_within_frozen_tolerance"
                )) is not bool
                for value in geometry_groups.values()
            )
        ):
            raise DiagnosticRunnerError(f"{label} group geometry/alignment is invalid")
        for metrics_label, metrics in (
            ("global", geometry_global),
            *((f"per_group.{name}", value) for name, value in geometry_groups.items()),
        ):
            expected_metric_keys = common_geometry_metric_keys | (
                {"parameter_tensor_count"}
                if metrics_label != "global"
                else set()
            )
            _require_exact_keys(
                metrics,
                expected_metric_keys,
                f"{label}.optimizer_geometry.{metrics_label}",
            )
            count_names = {
                "scalar_count",
                "active_gradient_scalar_count",
                "small_gradient_scalar_count",
                "parameter_tensor_count",
            }
            for metric_name in count_names & expected_metric_keys:
                if type(metrics[metric_name]) is not int or metrics[metric_name] < 0:
                    raise DiagnosticRunnerError(
                        f"{label}.optimizer_geometry.{metrics_label}.{metric_name} "
                        "must be a non-negative integer"
                    )
            if (
                metrics["active_gradient_scalar_count"] != metrics["scalar_count"]
                or metrics["small_gradient_scalar_count"]
                > metrics["active_gradient_scalar_count"]
                or metrics["scalar_count"] <= 0
                or (
                    metrics_label != "global"
                    and metrics["parameter_tensor_count"] <= 0
                )
            ):
                raise DiagnosticRunnerError(
                    f"{label}.optimizer_geometry.{metrics_label} count range differs"
                )
            replay_match = metrics["cpu_storage_replay_within_frozen_tolerance"]
            if type(replay_match) is not bool:
                raise DiagnosticRunnerError(
                    f"{label}.optimizer_geometry.{metrics_label} replay metric must be bool"
                )
            optional_float_names = {
                "cross_backend_cpu_storage_replay_cos_actual_expected",
                "cross_backend_cpu_storage_replay_actual_to_expected_norm_ratio",
                "continuous_ideal_cos_actual_expected",
                "continuous_ideal_actual_to_expected_norm_ratio",
                "small_gradient_fraction",
                "near_sign_step_threshold",
                "near_sign_step_fraction",
            }
            optional_count = metrics["near_sign_step_scalar_count"]
            if candidate.optimizer == "Adam":
                if (
                    type(optional_count) is not int
                    or optional_count < 0
                    or optional_count > metrics["active_gradient_scalar_count"]
                ):
                    raise DiagnosticRunnerError(
                        f"{label}.optimizer_geometry.{metrics_label}.near_sign_step_scalar_count differs"
                    )
            elif optional_count is not None:
                raise DiagnosticRunnerError(
                    f"{label}.optimizer_geometry.{metrics_label}.near_sign_step_scalar_count differs"
                )
            for metric_name in expected_metric_keys - count_names - {
                "cpu_storage_replay_within_frozen_tolerance",
                "near_sign_step_scalar_count",
            }:
                metric_value = metrics[metric_name]
                if metric_value is None and metric_name in optional_float_names:
                    continue
                if type(metric_value) is not float or not math.isfinite(metric_value):
                    raise DiagnosticRunnerError(
                        f"{label}.optimizer_geometry.{metrics_label}.{metric_name} must be finite float"
                    )
                if "cos_" in metric_name:
                    if not -1.0 <= metric_value <= 1.0:
                        raise DiagnosticRunnerError(
                            f"{label}.optimizer_geometry.{metrics_label}.{metric_name} out of range"
                        )
                elif metric_value < 0.0:
                    raise DiagnosticRunnerError(
                        f"{label}.optimizer_geometry.{metrics_label}.{metric_name} must be non-negative"
                    )
                if metric_name.endswith("fraction") and metric_value > 1.0:
                    raise DiagnosticRunnerError(
                        f"{label}.optimizer_geometry.{metrics_label}.{metric_name} out of range"
                    )
            active_count = metrics["active_gradient_scalar_count"]
            expected_small_fraction = metrics["small_gradient_scalar_count"] / active_count
            if metrics["small_gradient_fraction"] != expected_small_fraction:
                raise DiagnosticRunnerError(
                    f"{label}.optimizer_geometry.{metrics_label}.small_gradient_fraction differs"
                )
            near_threshold = metrics["near_sign_step_threshold"]
            near_fraction = metrics["near_sign_step_fraction"]
            if candidate.optimizer == "SGD":
                if near_threshold is not None or optional_count is not None or near_fraction is not None:
                    raise DiagnosticRunnerError(
                        f"{label}.optimizer_geometry.{metrics_label} SGD near-sign metrics differ"
                    )
            elif (
                type(near_threshold) is not float
                or near_threshold != geometry_config["adam_near_sign_step_threshold"]
                or type(optional_count) is not int
                or type(near_fraction) is not float
                or near_fraction != optional_count / active_count
            ):
                raise DiagnosticRunnerError(
                    f"{label}.optimizer_geometry.{metrics_label} Adam near-sign metrics differ"
                )
        geometry_layout = geometry.get("parameter_layout")
        alignment_layout = alignment.get("parameter_layout")
        if (
            not isinstance(geometry_layout, Mapping)
            or geometry_layout != alignment_layout
            or geometry_layout.get("parameter_tensor_count") != 106
            or geometry_layout.get("parameter_scalar_count") != 8736
            or geometry_global["scalar_count"]
            != geometry_layout.get("parameter_scalar_count")
            or sum(
                value["parameter_tensor_count"] for value in geometry_groups.values()
            )
            != geometry_layout.get("parameter_tensor_count")
            or sum(value["scalar_count"] for value in geometry_groups.values())
            != geometry_layout.get("parameter_scalar_count")
        ):
            raise DiagnosticRunnerError(f"{label} parameter topology differs")
        current_topology = (
            tuple(geometry_groups),
            geometry_layout.get("parameter_names_sha256"),
            geometry_layout.get("topology_sha256"),
        )
        if topology is None:
            topology = current_topology
        elif current_topology != topology:
            raise DiagnosticRunnerError("D0 diagnostic topology changed across records")
        label_isolation = alignment.get("label_isolation")
        if not isinstance(label_isolation, Mapping) or any(
            label_isolation.get(key) != expected
            for key, expected in {
                "entropy_gradient_label_free": True,
                "adaptation_step_label_free": True,
                "supervised_gradient_used_by_method": False,
                "supervised_gradient_role": "outer_oracle_train_labels_only",
                "method_label_accesses": 0,
                "outer_evaluator_label_accesses": 1,
            }.items()
        ):
            raise DiagnosticRunnerError(f"{label} label-isolation receipt differs")

        determinism = _require_exact_keys(
            record.get("determinism"),
            {
                "entropy_backward",
                "supervised_task_backward",
                "strict_policy_before_optimizer_step",
                "strict_policy_before_post_forward",
            },
            f"{label}.determinism",
        )
        if (
            determinism.get("strict_policy_before_optimizer_step") is not True
            or determinism.get("strict_policy_before_post_forward") is not True
        ):
            raise DiagnosticRunnerError(
                f"{label} strict deterministic forward restoration differs"
            )
        try:
            validate_determinism_audit(
                determinism["entropy_backward"],
                determinism_config=config["method"]["determinism"],
                expected_scope="entropy_backward",
                expected_loss_device_type="cuda",
                expected_backward_completed=True,
            )
            validate_determinism_audit(
                determinism["supervised_task_backward"],
                determinism_config=config["method"]["determinism"],
                expected_scope="supervised_task_backward",
                expected_loss_device_type="cuda",
                expected_backward_completed=True,
            )
        except (KeyError, D0DeterminismRuntimeError) as exc:
            raise DiagnosticRunnerError(
                f"{label} determinism runtime receipt differs"
            ) from exc

    expected_cells = {
        (condition, candidate.slug): images_per_condition
        for condition in conditions
        for candidate in candidates
    }
    if dict(observed_cells) != expected_cells:
        raise DiagnosticRunnerError("D0 diagnostic cell cardinalities differ")


# ---------------------------------------------------------------------------
# Exact Stage-1 cell reconstruction and frozen-v2 comparison
# ---------------------------------------------------------------------------

ENDPOINT_COUNT_FIELDS = (
    "intersection_pixels",
    "union_pixels",
    "false_alarm_pixels",
    "total_image_pixels",
    "detected_targets",
    "total_targets",
)


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DiagnosticRunnerError(f"{label} must be a nonnegative integer")
    return value


def _fraction_receipt(value: Fraction) -> dict[str, Any]:
    """Return a lossless JSON representation of an exact rational metric."""

    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": f"{value.numerator}/{value.denominator}",
    }


def _endpoint_fractions(counts: Mapping[str, Any], label: str) -> dict[str, Fraction]:
    values = {
        field: _nonnegative_integer(counts.get(field), f"{label}.{field}")
        for field in ENDPOINT_COUNT_FIELDS
    }
    for denominator in ("union_pixels", "total_image_pixels", "total_targets"):
        if values[denominator] <= 0:
            raise DiagnosticRunnerError(f"{label}.{denominator} must be positive")
    return {
        "global_iou": Fraction(
            values["intersection_pixels"], values["union_pixels"]
        ),
        "pd": Fraction(values["detected_targets"], values["total_targets"]),
        "fa": Fraction(
            values["false_alarm_pixels"], values["total_image_pixels"]
        ),
        "fa_per_million_pixels": Fraction(
            values["false_alarm_pixels"] * 1_000_000,
            values["total_image_pixels"],
        ),
    }


def _exact_metric_bundle(
    pre: Mapping[str, Any], post: Mapping[str, Any], label: str
) -> dict[str, Any]:
    pre_values = _endpoint_fractions(pre, f"{label}.pre")
    post_values = _endpoint_fractions(post, f"{label}.post")
    return {
        "pre": {
            metric: _fraction_receipt(value)
            for metric, value in pre_values.items()
        },
        "post": {
            metric: _fraction_receipt(value)
            for metric, value in post_values.items()
        },
        "delta": {
            metric: _fraction_receipt(post_values[metric] - pre_values[metric])
            for metric in pre_values
        },
    }


def _candidate_from_mapping(value: Any, label: str) -> Candidate:
    if not isinstance(value, Mapping):
        raise DiagnosticRunnerError(f"{label} must be a mapping")
    try:
        return Candidate(str(value["optimizer"]), float(value["learning_rate"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise DiagnosticRunnerError(f"{label} is not a valid candidate") from exc


def _condition_from_record(record: Mapping[str, Any], label: str) -> str:
    corruption = record.get("corruption")
    severity = record.get("severity")
    if not isinstance(corruption, str) or not corruption:
        raise DiagnosticRunnerError(f"{label}.corruption must be a nonempty string")
    if isinstance(severity, bool) or not isinstance(severity, int) or severity < 0:
        raise DiagnosticRunnerError(f"{label}.severity must be a nonnegative integer")
    return f"{corruption}_S{severity}"


def _read_jsonl_objects(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        snapshot = read_stable_regular_file(_lexical_absolute(path))
    except (OSError, ValueError, RuntimeError) as exc:
        raise DiagnosticRunnerError(
            f"{label} missing, unstable, or symlinked: {path}"
        ) from exc
    return _decode_jsonl_bytes(snapshot.data, label)


def _sum_endpoint_counts(
    values: Sequence[Mapping[str, Any]], label: str
) -> dict[str, int]:
    result = {field: 0 for field in ENDPOINT_COUNT_FIELDS}
    for index, value in enumerate(values):
        for field in ENDPOINT_COUNT_FIELDS:
            result[field] += _nonnegative_integer(
                value.get(field), f"{label}[{index}].{field}"
            )
    return result


def build_exact_cell_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_datasets: Sequence[str] | None = None,
    expected_conditions: Sequence[str] | None = None,
    expected_candidates: Sequence[Candidate] | None = None,
    expected_images_per_cell: int | None = None,
) -> list[dict[str, Any]]:
    """Collapse episode diagnostics into exact dataset/condition/candidate cells.

    All task metrics are reconstructed from integer sufficient statistics with
    :class:`fractions.Fraction`.  When expected dimensions are provided this
    function is also the formal episode/cell identity gate.
    """

    if not records:
        raise DiagnosticRunnerError("cannot build exact cells from zero records")
    expected_dataset_tuple = (
        None if expected_datasets is None else tuple(expected_datasets)
    )
    expected_condition_tuple = (
        None if expected_conditions is None else tuple(expected_conditions)
    )
    expected_candidate_tuple = (
        None if expected_candidates is None else tuple(expected_candidates)
    )
    if expected_images_per_cell is not None and (
        isinstance(expected_images_per_cell, bool)
        or expected_images_per_cell <= 0
    ):
        raise DiagnosticRunnerError("expected_images_per_cell must be positive")

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    candidate_by_slug: dict[str, Candidate] = {}
    for index, record in enumerate(records):
        label = f"D0 record[{index}]"
        dataset = record.get("dataset")
        image_id = record.get("image_id")
        if not isinstance(dataset, str) or not dataset:
            raise DiagnosticRunnerError(f"{label}.dataset must be nonempty")
        if record.get("split_role") != "train":
            raise DiagnosticRunnerError(f"{label} is not source-train derived")
        if not isinstance(image_id, str) or not image_id:
            raise DiagnosticRunnerError(f"{label}.image_id must be nonempty")
        condition = _condition_from_record(record, label)
        candidate = _candidate_from_mapping(record.get("candidate"), f"{label}.candidate")
        slug = record.get("candidate_slug")
        if slug != candidate.slug:
            raise DiagnosticRunnerError(f"{label} candidate slug mismatch")
        previous_candidate = candidate_by_slug.setdefault(candidate.slug, candidate)
        if previous_candidate != candidate:
            raise DiagnosticRunnerError(f"candidate slug collision: {candidate.slug}")
        noop = record.get("noop")
        if not isinstance(noop, Mapping):
            raise DiagnosticRunnerError(f"{label}.noop must be a mapping")
        metric_counts = noop.get("metric_counts")
        if (
            not isinstance(metric_counts, Mapping)
            or not isinstance(metric_counts.get("pre"), Mapping)
            or not isinstance(metric_counts.get("post"), Mapping)
        ):
            raise DiagnosticRunnerError(f"{label} lacks metric sufficient statistics")
        grouped[(dataset, condition, candidate.slug)].append(record)

    if expected_dataset_tuple is not None and set(candidate_by_slug) and (
        set(dataset for dataset, _condition, _slug in grouped)
        != set(expected_dataset_tuple)
    ):
        raise DiagnosticRunnerError("D0 dataset identity set differs from formal scope")
    if expected_candidate_tuple is not None:
        expected_by_slug = {
            candidate.slug: candidate for candidate in expected_candidate_tuple
        }
        if candidate_by_slug != expected_by_slug:
            raise DiagnosticRunnerError("D0 candidate identity set differs from formal scope")
    else:
        expected_by_slug = dict(candidate_by_slug)

    if expected_condition_tuple is not None:
        observed_conditions = {condition for _dataset, condition, _slug in grouped}
        if observed_conditions != set(expected_condition_tuple):
            raise DiagnosticRunnerError("D0 condition identity set differs from formal scope")
    datasets = (
        expected_dataset_tuple
        if expected_dataset_tuple is not None
        else tuple(sorted({key[0] for key in grouped}))
    )
    conditions = (
        expected_condition_tuple
        if expected_condition_tuple is not None
        else tuple(sorted({key[1] for key in grouped}))
    )
    candidates = (
        expected_candidate_tuple
        if expected_candidate_tuple is not None
        else tuple(sorted(candidate_by_slug.values()))
    )
    expected_keys = {
        (dataset, condition, candidate.slug)
        for dataset in datasets
        for condition in conditions
        for candidate in candidates
    }
    if set(grouped) != expected_keys:
        missing = sorted(expected_keys - set(grouped))
        extra = sorted(set(grouped) - expected_keys)
        raise DiagnosticRunnerError(
            "D0 cell identity set differs; "
            f"first_missing={missing[:3]}, first_extra={extra[:3]}"
        )

    dataset_image_ids: dict[str, tuple[str, ...]] = {}
    result: list[dict[str, Any]] = []
    for dataset in datasets:
        for condition in conditions:
            corruption, severity_text = condition.rsplit("_S", 1)
            severity = int(severity_text)
            for candidate in candidates:
                key = (dataset, condition, candidate.slug)
                values = grouped[key]
                image_ids = tuple(str(value["image_id"]) for value in values)
                if len(set(image_ids)) != len(image_ids):
                    raise DiagnosticRunnerError(f"duplicate D0 episode identity in {key}")
                canonical_image_ids = tuple(sorted(image_ids))
                if expected_images_per_cell is not None and (
                    len(canonical_image_ids) != expected_images_per_cell
                ):
                    raise DiagnosticRunnerError(
                        f"D0 episode count differs in {key}: {len(canonical_image_ids)}"
                    )
                previous_ids = dataset_image_ids.setdefault(dataset, canonical_image_ids)
                if previous_ids != canonical_image_ids:
                    raise DiagnosticRunnerError(
                        f"D0 source episode identity set changed across cells: {key}"
                    )
                pre_counts = _sum_endpoint_counts(
                    [value["noop"]["metric_counts"]["pre"] for value in values],
                    f"{key}.pre",
                )
                post_counts = _sum_endpoint_counts(
                    [value["noop"]["metric_counts"]["post"] for value in values],
                    f"{key}.post",
                )
                if pre_counts["total_image_pixels"] != post_counts["total_image_pixels"]:
                    raise DiagnosticRunnerError(f"image-pixel denominator changed in {key}")
                if pre_counts["total_targets"] != post_counts["total_targets"]:
                    raise DiagnosticRunnerError(f"target denominator changed in {key}")
                result.append(
                    {
                        "schema_version": 1,
                        "dataset": dataset,
                        "condition": condition,
                        "corruption": corruption,
                        "severity": severity,
                        "candidate_slug": candidate.slug,
                        "candidate": candidate.to_dict(),
                        "episode_count": len(values),
                        "episode_identity": {
                            "unique": True,
                            "image_ids": list(canonical_image_ids),
                            "image_ids_sha256": hashlib.sha256(
                                _canonical_json(list(canonical_image_ids))
                            ).hexdigest(),
                        },
                        "endpoint_counts": {
                            "pre": pre_counts,
                            "post": post_counts,
                        },
                        "metrics_exact": _exact_metric_bundle(
                            pre_counts, post_counts, str(key)
                        ),
                    }
                )
    return result


def _archive_candidate_key(record: Mapping[str, Any], label: str) -> Candidate:
    return _candidate_from_mapping(record.get("candidate"), f"{label}.candidate")


def verify_formal_dataset_against_frozen_v2(
    *,
    records: Sequence[Mapping[str, Any]],
    dataset: str,
    config: Mapping[str, Any],
    archive_records_path: Path,
) -> dict[str, Any]:
    """Require exact 130-cell equality with the frozen negative Stage-1 v2."""

    archive_verification = _verify_frozen_negative_archive(config)
    conditions = _config_conditions(config)
    candidates = _config_candidates(config)
    if dataset not in config.get("datasets", {}) or dataset not in FORMAL_DATASETS:
        raise DiagnosticRunnerError(f"dataset is outside formal D0 scope: {dataset}")
    cells = build_exact_cell_records(
        records,
        expected_datasets=(dataset,),
        expected_conditions=conditions,
        expected_candidates=candidates,
        expected_images_per_cell=FORMAL_IMAGES_PER_CELL,
    )
    try:
        archive_snapshot = read_stable_regular_file(archive_records_path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise DiagnosticRunnerError(
            "frozen v2 stage1_records became unavailable or unstable"
        ) from exc
    if archive_snapshot.sha256 != config["comparison"]["stage1_records_sha256"]:
        raise DiagnosticRunnerError("frozen v2 stage1_records SHA-256 drifted")
    archive_records = _decode_jsonl_bytes(
        archive_snapshot.data, "frozen v2 stage1_records"
    )
    expected_archive_count = (
        len(FORMAL_DATASETS) * len(conditions) * len(candidates)
    )
    if len(archive_records) != expected_archive_count:
        raise DiagnosticRunnerError(
            "frozen v2 stage1_records count differs from 390-cell contract"
        )
    archive_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for index, record in enumerate(archive_records):
        label = f"frozen v2 stage1_records[{index}]"
        record_dataset = record.get("dataset")
        condition = _condition_from_record(record, label)
        candidate = _archive_candidate_key(record, label)
        key = (str(record_dataset), condition, candidate.slug)
        if key in archive_index:
            raise DiagnosticRunnerError(f"duplicate frozen v2 cell: {key}")
        if (
            record_dataset not in FORMAL_DATASETS
            or condition not in conditions
            or candidate not in candidates
            or record.get("stage") != 1
            or record.get("bn_protocol") != "source_running_statistics"
            or record.get("image_count") != FORMAL_IMAGES_PER_CELL
            or record.get("optimizer_steps_total") != FORMAL_IMAGES_PER_CELL
            or record.get("method_label_accesses") != 0
            or record.get("test_image_opens") != 0
            or record.get("test_label_opens") != 0
        ):
            raise DiagnosticRunnerError(f"frozen v2 cell protocol drifted: {key}")
        endpoints = record.get("endpoints")
        if (
            not isinstance(endpoints, Mapping)
            or set(endpoints) != {"tent_pre", "tent_post"}
        ):
            raise DiagnosticRunnerError(f"frozen v2 endpoints invalid: {key}")
        for endpoint in ("tent_pre", "tent_post"):
            values = endpoints[endpoint]
            if not isinstance(values, Mapping) or set(values) != set(
                ENDPOINT_COUNT_FIELDS
            ):
                raise DiagnosticRunnerError(
                    f"frozen v2 endpoint field set drifted: {key}.{endpoint}"
                )
            _endpoint_fractions(values, f"{key}.{endpoint}")
        archive_index[key] = record
    if len(archive_index) != expected_archive_count:
        raise DiagnosticRunnerError("frozen v2 stage1 cell identity set is incomplete")

    d0_index = {
        (cell["dataset"], cell["condition"], cell["candidate_slug"]): cell
        for cell in cells
    }
    expected_dataset_keys = {
        (dataset, condition, candidate.slug)
        for condition in conditions
        for candidate in candidates
    }
    if set(d0_index) != expected_dataset_keys or not expected_dataset_keys.issubset(
        archive_index
    ):
        raise DiagnosticRunnerError("D0/archive formal cell identities differ")
    for key in sorted(expected_dataset_keys):
        cell = d0_index[key]
        archive = archive_index[key]
        archive_endpoints = archive["endpoints"]
        for d0_endpoint, archive_endpoint in (
            ("pre", "tent_pre"),
            ("post", "tent_post"),
        ):
            d0_counts = cell["endpoint_counts"][d0_endpoint]
            frozen_counts = dict(archive_endpoints[archive_endpoint])
            if d0_counts != frozen_counts:
                differing = [
                    field
                    for field in ENDPOINT_COUNT_FIELDS
                    if d0_counts.get(field) != frozen_counts.get(field)
                ]
                raise DiagnosticRunnerError(
                    "formal D0 differs from frozen v2 Stage-1 at "
                    f"{key}.{d0_endpoint}; fields={differing}"
                )
        frozen_metrics = _exact_metric_bundle(
            archive_endpoints["tent_pre"],
            archive_endpoints["tent_post"],
            f"frozen.{key}",
        )
        if cell["metrics_exact"] != frozen_metrics:
            raise DiagnosticRunnerError(
                f"formal D0 exact Fraction metrics differ from frozen v2: {key}"
            )
        if cell["episode_count"] != int(archive["image_count"]):
            raise DiagnosticRunnerError(
                f"formal D0/archive episode identity count differs: {key}"
            )

    image_identity_hashes = {
        cell["episode_identity"]["image_ids_sha256"] for cell in cells
    }
    if len(image_identity_hashes) != 1:
        raise DiagnosticRunnerError("formal D0 image identities changed across cells")
    repeated_archive_verification = _verify_frozen_negative_archive(config)
    if repeated_archive_verification != archive_verification:
        raise DiagnosticRunnerError(
            "frozen v2 negative archive changed during formal comparison"
        )
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0_frozen_v2_cell_equivalence_v1",
        "passed": True,
        "dataset": dataset,
        "cell_count": len(cells),
        "diagnostic_record_count": len(records),
        "episodes_per_cell": FORMAL_IMAGES_PER_CELL,
        "cell_identity_exact": True,
        "episode_count_identity_exact": True,
        "unique_image_ids_per_cell": True,
        "same_source_image_ids_across_all_cells": True,
        "source_image_ids_sha256": next(iter(image_identity_hashes)),
        "endpoint_integer_counts_exact": True,
        "global_iou_fraction_exact": True,
        "pd_fraction_exact": True,
        "fa_fraction_exact": True,
        "matched_endpoint_fields": list(ENDPOINT_COUNT_FIELDS),
        "frozen_v2_stage1_records": {
            "path": str(_lexical_absolute(archive_records_path)),
            "sha256": archive_snapshot.sha256,
            "record_count": len(archive_records),
        },
        "frozen_v2_negative_archive": archive_verification,
        "d0_cell_records_sha256": hashlib.sha256(
            _canonical_json(cells)
        ).hexdigest(),
    }


def _source_hashes(config_path: Path) -> dict[str, str]:
    paths = (
        config_path,
        Path(__file__).resolve(),
        PROJECT_ROOT / "run_binary_tent_ss_calibration_v2.py",
        PROJECT_ROOT / "materialize_binary_tent_ss_calibration_cache_v2.py",
        PROJECT_ROOT / "test_source.py",
        PROJECT_ROOT / "tta" / "diagnostics.py",
        PROJECT_ROOT / "tta" / "d0_secure_io.py",
        PROJECT_ROOT / "tta" / "model_adapter.py",
        PROJECT_ROOT / "tta" / "parameter_groups.py",
        PROJECT_ROOT / "tta" / "parameter_vector.py",
        PROJECT_ROOT / "tta" / "binary_tent_fast_runner.py",
        PROJECT_ROOT / "tta" / "binary_tent_fast_runner_v2.py",
        PROJECT_ROOT / "tta" / "episodic_runner.py",
        PROJECT_ROOT / "analysis" / "source_train_provenance.py",
        PROJECT_ROOT / "analysis" / "d0_protocol_contract.py",
        PROJECT_ROOT / "analysis" / "d0_determinism_runtime.py",
        PROJECT_ROOT / "analysis" / "d0_equivalence_repro_contract.py",
        PROJECT_ROOT / "analysis" / "analyze_tent_optimizer_geometry.py",
        PROJECT_ROOT / "analysis" / "analyze_entropy_task_alignment.py",
        PROJECT_ROOT / "scripts" / "archive_binary_tent_ss_stage1_negative_v2.py",
        PROJECT_ROOT / "tta" / "binary_tent.py",
        PROJECT_ROOT / "metrics" / "irstd_metrics.py",
        PROJECT_ROOT / "metrics" / "connected_components.py",
        PROJECT_ROOT / "metrics" / "target_matching.py",
        PROJECT_ROOT / "model" / "loss.py",
        PROJECT_ROOT / "model" / "MSHNet_NSFPN.py",
        PROJECT_ROOT / "model" / "NS_FPN.py",
        PROJECT_ROOT / "model" / "diff_cross_attns.py",
        PROJECT_ROOT / "train_fixed_split.py",
    )
    result: dict[str, str] = {}
    for path in paths:
        lexical = _lexical_absolute(path)
        try:
            snapshot = read_stable_regular_file(lexical)
        except (OSError, ValueError, RuntimeError) as exc:
            raise DiagnosticRunnerError(
                f"D0 diagnostic source is missing, unstable, or symlinked: {lexical}"
            ) from exc
        result[lexical.relative_to(PROJECT_ROOT).as_posix()] = snapshot.sha256
    return result


def _assert_source_hashes_unchanged(
    *, config_path: Path, expected: Mapping[str, str], stage: str
) -> dict[str, Any]:
    current = _source_hashes(config_path)
    if current != dict(expected):
        changed = sorted(
            key
            for key in set(current) | set(expected)
            if current.get(key) != expected.get(key)
        )
        raise DiagnosticRunnerError(
            f"D0 source byte seal drifted at {stage}: {changed[:5]}"
        )
    return {
        "stage": stage,
        "verified": True,
        "bound_file_count": len(current),
        "all_files_rehashed": True,
    }


def _publish_shard(
    *,
    destination: Path,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    formal_scope: bool,
    source_hashes: Mapping[str, str],
    prepublish_guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if summary != summarize_records(records):
        raise DiagnosticRunnerError("refusing to publish a non-recomputable D0 summary")
    if formal_scope and prepublish_guard is None:
        raise DiagnosticRunnerError(
            "formal D0 publication requires a live prepublish guard"
        )
    destination = _lexical_absolute(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite D0 output: {destination}")
    fsync_directory(destination.parent)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        _write_jsonl(staging / "episode_records.jsonl", records)
        _write_json(staging / "summary.json", summary)
        _write_json(staging / "provenance.json", provenance)
        payloads = ("episode_records.jsonl", "summary.json", "provenance.json")
        files = {
            name: {
                "sha256": sha256_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in payloads
        }
        manifest = {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "artifact_complete": True,
            "formal_d0_complete": formal_scope,
            "paper_test_result": False,
            "oracle_analysis": True,
            "method_label_accesses": 0,
            "selection_authorized": False,
            "stage2_authorized": False,
            "stage3_authorized": False,
            "record_count": len(records),
            "files": files,
            "source_code_sha256": dict(sorted(source_hashes.items())),
        }
        _write_json(staging / "artifact_manifest.json", manifest)
        complete = {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "complete": True,
            "formal_d0_complete": formal_scope,
            "paper_test_result": False,
            "selection_authorized": False,
            "stage2_authorized": False,
            "stage3_authorized": False,
            "manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
            "record_count": len(records),
        }
        _write_json(staging / "COMPLETE.json", complete)
        managed = (*payloads, "artifact_manifest.json", "COMPLETE.json")
        checksum_lines = [
            f"{sha256_file(staging / name)}  {name}\n" for name in sorted(managed)
        ]
        (staging / "SHA256SUMS").write_text("".join(checksum_lines), encoding="utf-8")
        # Perform the complete semantic/live verification while the artifact
        # still has a private staging name.  A failure is therefore removable
        # and the canonical no-overwrite path remains immediately retryable.
        staged_verification = verify_shard(staging)
        publish_directory_noreplace(
            staging,
            destination,
            pre_rename_guard=prepublish_guard,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**staged_verification, "path": str(destination)}


def _decode_json_bytes(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticRunnerError(f"invalid JSON: {label}") from exc
    if not isinstance(value, dict):
        raise DiagnosticRunnerError(f"JSON root must be an object: {label}")
    return value


def _decode_jsonl_bytes(data: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DiagnosticRunnerError(f"invalid UTF-8 JSONL: {label}") from exc
    result: list[dict[str, Any]] = []
    for index, line in enumerate(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DiagnosticRunnerError(f"invalid JSONL: {label}[{index}]") from exc
        if not isinstance(value, dict):
            raise DiagnosticRunnerError(f"JSONL row is not an object: {label}[{index}]")
        result.append(value)
    return result


def _is_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_record_source_bindings(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    config: Mapping[str, Any],
) -> None:
    for index, record in enumerate(records):
        geometry = record["optimizer_geometry"]
        alignment = record["entropy_task_alignment"]
        for label, value, oracle in (
            (f"record[{index}].optimizer_geometry.scope", geometry["scope"], False),
            (f"record[{index}].entropy_task_alignment.scope", alignment["scope"], True),
        ):
            _validate_analysis_scope(
                value,
                require_outer_oracle=oracle,
                dataset=dataset,
                config=config,
                label=label,
            )


def _validate_shard_provenance(
    *,
    provenance: Mapping[str, Any],
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    config_path: Path,
    formal_scope: bool,
) -> tuple[str, tuple[str, ...], tuple[Candidate, ...], int]:
    expected_fields = {
        "schema_version",
        "artifact_type",
        "dataset",
        "formal_d0_complete",
        "paper_test_result",
        "oracle_analysis",
        "source_train_derived",
        "method_label_accesses",
        "outer_evaluator_label_accesses",
        "test_images_opened",
        "test_labels_opened",
        "conditions",
        "candidates",
        "images_per_condition",
        "checkpoint_wrapper",
        "checkpoint_sha256",
        "train_split_sha256",
        "config_path",
        "config_sha256",
        "equivalence_receipt_sha256",
        "equivalence_receipt",
        "global_runtime_seal_sha256",
        "v2_runtime_seal",
        "runtime_audits",
        "diagnostic_source_code_sha256",
        "diagnostic_source_audits",
        "frozen_v2_stage1_exact_comparison",
        "cache_lineage",
        "target_deserialized_only_after_cell_label_free_completion",
        "source_eval_task_gradient_contract",
        "determinism_contract",
        "determinism_contract_sha256",
        "selection_authorized",
        "stage2_authorized",
        "stage3_authorized",
        "elapsed_seconds",
    }
    provenance = _require_exact_keys(provenance, expected_fields, "D0 provenance")
    dataset = provenance.get("dataset")
    if not isinstance(dataset, str) or dataset not in config.get("datasets", {}):
        raise DiagnosticRunnerError("D0 provenance dataset is invalid")
    conditions_value = provenance.get("conditions")
    candidates_value = provenance.get("candidates")
    images = provenance.get("images_per_condition")
    if (
        not isinstance(conditions_value, list)
        or not conditions_value
        or not all(isinstance(value, str) for value in conditions_value)
        or not isinstance(candidates_value, list)
        or not candidates_value
        or isinstance(images, bool)
        or not isinstance(images, int)
        or not 1 <= images <= FORMAL_IMAGES_PER_CELL
    ):
        raise DiagnosticRunnerError("D0 provenance execution dimensions are invalid")
    conditions = tuple(conditions_value)
    candidates = tuple(
        _candidate_from_mapping(value, f"D0 provenance candidate[{index}]")
        for index, value in enumerate(candidates_value)
    )
    all_conditions = _config_conditions(config)
    all_candidates = _config_candidates(config)
    if (
        any(value not in all_conditions for value in conditions)
        or any(value not in all_candidates for value in candidates)
        or len(set(conditions)) != len(conditions)
        or len(set(candidates)) != len(candidates)
    ):
        raise DiagnosticRunnerError("D0 provenance dimensions differ from config")
    inferred_formal = (
        conditions == all_conditions
        and candidates == all_candidates
        and images == FORMAL_IMAGES_PER_CELL
    )
    if inferred_formal is not formal_scope:
        raise DiagnosticRunnerError("D0 formal scope does not match execution dimensions")
    fixed_flags = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "formal_d0_complete": formal_scope,
        "paper_test_result": False,
        "oracle_analysis": True,
        "source_train_derived": True,
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": len(conditions) * images,
        "test_images_opened": 0,
        "test_labels_opened": 0,
        "target_deserialized_only_after_cell_label_free_completion": True,
        "selection_authorized": False,
        "stage2_authorized": False,
        "stage3_authorized": False,
    }
    if any(provenance.get(key) != expected for key, expected in fixed_flags.items()):
        raise DiagnosticRunnerError("D0 provenance safety flags differ")
    if (
        provenance.get("checkpoint_sha256")
        != config["datasets"][dataset]["checkpoint"]["sha256"]
        or provenance.get("train_split_sha256")
        != config["datasets"][dataset]["train_split_sha256"]
        or not isinstance(provenance.get("checkpoint_wrapper"), str)
        or not provenance.get("checkpoint_wrapper")
        or provenance.get("source_eval_task_gradient_contract")
        != config["source_task_gradient"]
        or provenance.get("determinism_contract")
        != config["method"]["determinism"]
        or provenance.get("determinism_contract_sha256")
        != frozen_determinism_sha256(config["method"]["determinism"])
        or _finite_number(
            provenance.get("elapsed_seconds"),
            "D0 provenance elapsed_seconds",
            nonnegative=True,
        )
        < 0.0
    ):
        raise DiagnosticRunnerError("D0 provenance source/task binding differs")
    config_snapshot = read_stable_regular_file(config_path)
    if (
        provenance.get("config_path") != str(config_path)
        or provenance.get("config_sha256") != config_snapshot.sha256
    ):
        raise DiagnosticRunnerError("D0 provenance config binding differs")
    source_hashes = provenance.get("diagnostic_source_code_sha256")
    if (
        not isinstance(source_hashes, Mapping)
        or source_hashes != manifest.get("source_code_sha256")
        or any(not _is_lower_sha256(value) for value in source_hashes.values())
    ):
        raise DiagnosticRunnerError("D0 provenance source-code binding differs")
    validate_d0_records(
        records,
        config=config,
        dataset=dataset,
        conditions=conditions,
        candidates=candidates,
        images_per_condition=images,
    )
    return dataset, conditions, candidates, images


def _verify_formal_shard_live_bindings(
    *,
    provenance: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    config_path: Path,
    dataset: str,
    candidates: Sequence[Candidate],
) -> None:
    contract = frozen_v2.load_contract()
    seal, caches = frozen_v2.capture_runtime_seal(contract)
    monitor = frozen_v2.RuntimeSealMonitor(seal)
    monitor.assert_unchanged(stage="d0_verify_shard", full_byte_rehash=True)
    if (
        provenance.get("global_runtime_seal_sha256")
        != seal.global_runtime_seal_sha256
        or provenance.get("v2_runtime_seal") != seal.to_dict()
    ):
        raise DiagnosticRunnerError("formal D0 live runtime seal differs")
    source_hashes = _source_hashes(config_path)
    if provenance.get("diagnostic_source_code_sha256") != dict(
        sorted(source_hashes.items())
    ):
        raise DiagnosticRunnerError("formal D0 live source-code seal differs")
    runtime_audits = provenance.get("runtime_audits")
    source_audits = provenance.get("diagnostic_source_audits")
    if (
        not isinstance(runtime_audits, list)
        or len(runtime_audits) < 2
        or any(
            not isinstance(value, Mapping)
            or value.get("verified") is not True
            or value.get("global_runtime_seal_sha256")
            != seal.global_runtime_seal_sha256
            for value in runtime_audits
        )
        or runtime_audits[-1].get("full_byte_rehash") is not True
        or not isinstance(source_audits, list)
        or len(source_audits) != len(runtime_audits)
        or any(
            not isinstance(value, Mapping)
            or value.get("verified") is not True
            or value.get("all_files_rehashed") is not True
            or value.get("bound_file_count") != len(source_hashes)
            for value in source_audits
        )
    ):
        raise DiagnosticRunnerError("formal D0 runtime/source audit ledger differs")
    if dataset not in caches:
        raise DiagnosticRunnerError("formal D0 cache dataset is missing")
    cache = caches[dataset]
    targets = cache.manifest["targets"]
    expected_cache = {
        "root": str(cache.root.resolve()),
        "manifest_sha256": cache.manifest_sha256,
        "complete_sha256": cache.complete_sha256,
        "cache_content_sha256": cache.manifest["cache_content_sha256"],
        "method_input_manifest_sha256": cache.complete[
            "method_input_manifest_sha256"
        ],
        "ordered_ids_sha256": cache.manifest["ordered_ids_sha256"],
        "calibration_ids_file_sha256": cache.manifest[
            "calibration_ids_file_sha256"
        ],
        "target_file_sha256": targets["file_sha256"],
        "target_tensor_sequence_sha256": targets["tensor_sequence_sha256"],
        "test_images_opened": cache.complete["test_images_opened"],
        "test_masks_opened": cache.complete["test_masks_opened"],
    }
    if provenance.get("cache_lineage") != expected_cache:
        raise DiagnosticRunnerError("formal D0 live cache lineage differs")
    comparison = verify_formal_dataset_against_frozen_v2(
        records=records,
        dataset=dataset,
        config=config,
        archive_records_path=_archive_records_path_from_config(config),
    )
    if provenance.get("frozen_v2_stage1_exact_comparison") != comparison:
        raise DiagnosticRunnerError("formal D0 frozen-v2 exact comparison differs")
    receipt_binding = provenance.get("equivalence_receipt")
    receipt_sha = provenance.get("equivalence_receipt_sha256")
    if (
        not isinstance(receipt_binding, Mapping)
        or receipt_binding.get("sha256") != receipt_sha
        or not _is_lower_sha256(receipt_sha)
    ):
        raise DiagnosticRunnerError("formal D0 equivalence receipt binding is invalid")
    receipt_path = _validated_output_destination(
        config=config,
        destination=Path(str(receipt_binding.get("path", ""))),
        role="equivalence",
        dataset=dataset,
    )
    observed_receipt_sha = _verify_equivalence_receipt(
        receipt_path,
        dataset=dataset,
        candidates=candidates,
        config_sha256=str(provenance["config_sha256"]),
        global_runtime_seal_sha256=seal.global_runtime_seal_sha256,
        diagnostic_source_code_sha256=source_hashes,
        expected_condition=str(config["equivalence"]["samples"][dataset]["condition"]),
        expected_image_index=int(
            config["equivalence"]["samples"][dataset]["image_index"]
        ),
        expected_image_id=str(config["equivalence"]["samples"][dataset]["image_id"]),
    )
    if observed_receipt_sha != receipt_sha:
        raise DiagnosticRunnerError("formal D0 equivalence receipt bytes differ")


def verify_shard(path: Path) -> dict[str, Any]:
    root = _lexical_absolute(path)
    expected_names = {
        "episode_records.jsonl",
        "summary.json",
        "provenance.json",
        "artifact_manifest.json",
        "COMPLETE.json",
        "SHA256SUMS",
    }
    try:
        directory = snapshot_regular_directory(root)
    except (OSError, ValueError) as exc:
        raise DiagnosticRunnerError(f"D0 shard is not a stable regular directory: {root}") from exc
    if set(directory.member_names) != expected_names:
        raise DiagnosticRunnerError("D0 shard member set is not exact")
    members = {value.path.name: value for value in directory.members}
    checksums: dict[str, str] = {}
    try:
        checksum_text = members["SHA256SUMS"].data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DiagnosticRunnerError("invalid SHA256SUMS encoding") from exc
    for line in checksum_text.splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in checksums or not _is_lower_sha256(digest):
            raise DiagnosticRunnerError("invalid SHA256SUMS")
        checksums[name] = digest
    if set(checksums) != expected_names - {"SHA256SUMS"}:
        raise DiagnosticRunnerError("SHA256SUMS member set differs")
    for name, digest in checksums.items():
        if members[name].sha256 != digest:
            raise DiagnosticRunnerError(f"D0 shard hash mismatch: {name}")
    manifest = _decode_json_bytes(
        members["artifact_manifest.json"].data, "artifact_manifest.json"
    )
    complete = _decode_json_bytes(members["COMPLETE.json"].data, "COMPLETE.json")
    formal_scope = manifest.get("formal_d0_complete") is True
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "artifact_complete",
            "formal_d0_complete",
            "paper_test_result",
            "oracle_analysis",
            "method_label_accesses",
            "selection_authorized",
            "stage2_authorized",
            "stage3_authorized",
            "record_count",
            "files",
            "source_code_sha256",
        },
        "D0 manifest",
    )
    _require_exact_keys(
        complete,
        {
            "schema_version",
            "artifact_type",
            "complete",
            "formal_d0_complete",
            "paper_test_result",
            "selection_authorized",
            "stage2_authorized",
            "stage3_authorized",
            "manifest_sha256",
            "record_count",
        },
        "D0 completion",
    )
    if (
        manifest.get("schema_version") != 1
        or complete.get("schema_version") != 1
        or manifest.get("artifact_type") != ARTIFACT_TYPE
        or complete.get("artifact_type") != ARTIFACT_TYPE
        or manifest.get("artifact_complete") is not True
        or manifest.get("formal_d0_complete") is not formal_scope
        or complete.get("formal_d0_complete") is not formal_scope
        or manifest.get("paper_test_result") is not False
        or complete.get("paper_test_result") is not False
        or manifest.get("oracle_analysis") is not True
        or manifest.get("method_label_accesses") != 0
        or any(
            value.get(key) is not False
            for value in (manifest, complete)
            for key in (
                "selection_authorized",
                "stage2_authorized",
                "stage3_authorized",
            )
        )
        or complete.get("complete") is not True
        or complete.get("manifest_sha256")
        != members["artifact_manifest.json"].sha256
        or manifest.get("record_count") != complete.get("record_count")
    ):
        raise DiagnosticRunnerError("D0 manifest/completion contract failed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "episode_records.jsonl",
        "summary.json",
        "provenance.json",
    }:
        raise DiagnosticRunnerError("D0 manifest payload set differs")
    for name, binding in files.items():
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"sha256", "size_bytes"}
            or binding.get("sha256") != members[name].sha256
            or binding.get("size_bytes") != members[name].size_bytes
        ):
            raise DiagnosticRunnerError(f"D0 manifest binding failed: {name}")
    records = _decode_jsonl_bytes(
        members["episode_records.jsonl"].data, "D0 episode records"
    )
    record_count = len(records)
    if (
        isinstance(manifest.get("record_count"), bool)
        or not isinstance(manifest.get("record_count"), int)
        or record_count != manifest["record_count"]
    ):
        raise DiagnosticRunnerError("D0 JSONL record count differs")
    summary = _decode_json_bytes(members["summary.json"].data, "summary.json")
    if summary != summarize_records(records):
        raise DiagnosticRunnerError("D0 summary does not recompute from episode records")
    provenance = _decode_json_bytes(members["provenance.json"].data, "provenance.json")
    config_path_value = provenance.get("config_path")
    if not isinstance(config_path_value, str) or not config_path_value:
        raise DiagnosticRunnerError("D0 config path binding missing")
    config_path = _lexical_absolute(Path(config_path_value))
    config = _load_config(config_path)
    dataset, _conditions, candidates, _images = _validate_shard_provenance(
        provenance=provenance,
        manifest=manifest,
        records=records,
        config=config,
        config_path=config_path,
        formal_scope=formal_scope,
    )
    if formal_scope:
        _verify_formal_shard_live_bindings(
            provenance=provenance,
            records=records,
            config=config,
            config_path=config_path,
            dataset=dataset,
            candidates=candidates,
        )
    elif (
        provenance.get("equivalence_receipt") is not None
        or provenance.get("equivalence_receipt_sha256") is not None
        or provenance.get("frozen_v2_stage1_exact_comparison") is not None
    ):
        raise DiagnosticRunnerError("D0 smoke shard contains formal-only authority")
    try:
        final_directory = snapshot_regular_directory(root)
    except (OSError, ValueError) as exc:
        raise DiagnosticRunnerError(
            "D0 shard changed or became unsafe during verification"
        ) from exc
    if final_directory != directory:
        raise DiagnosticRunnerError("D0 shard changed during live verification")
    return {
        "status": "verified",
        "path": str(root),
        "record_count": record_count,
        "formal_d0_complete": formal_scope,
        "paper_test_result": False,
        "artifact_manifest_sha256": members["artifact_manifest.json"].sha256,
        "complete_sha256": members["COMPLETE.json"].sha256,
        "episode_records_sha256": members["episode_records.jsonl"].sha256,
    }


def _verify_equivalence_receipt(
    path: Path,
    *,
    dataset: str,
    candidates: Sequence[Candidate],
    config_sha256: str,
    global_runtime_seal_sha256: str,
    diagnostic_source_code_sha256: Mapping[str, str],
    expected_condition: str,
    expected_image_index: int,
    expected_image_id: str,
) -> str:
    path = _lexical_absolute(path)
    try:
        snapshot = read_stable_regular_file(path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise DiagnosticRunnerError(
            "equivalence receipt missing, unstable, or symlinked"
        ) from exc
    receipt = _decode_json_bytes(snapshot.data, str(path))
    expected_slugs = [candidate.slug for candidate in candidates]
    expected_source_hashes = dict(sorted(diagnostic_source_code_sha256.items()))
    if (
        receipt.get("dataset") != dataset
        or receipt.get("config_sha256") != config_sha256
        or receipt.get("global_runtime_seal_sha256")
        != global_runtime_seal_sha256
        or receipt.get("diagnostic_source_code_sha256")
        != expected_source_hashes
        or receipt.get("determinism_contract") is None
        or receipt.get("candidate_slugs") != expected_slugs
        or receipt.get("candidate_count") != len(candidates)
        or receipt.get("sample")
        != {
            "condition": expected_condition,
            "image_index": expected_image_index,
            "image_id": expected_image_id,
        }
    ):
        raise DiagnosticRunnerError(
            "three-process equivalence receipt live bindings differ"
        )
    try:
        canonical_sha = validate_d0_equivalence_repro_receipt(
            receipt, revalidate_inputs=True
        )
    except D0EquivalenceReproContractError as exc:
        raise DiagnosticRunnerError(
            "three-process equivalence receipt did not pass exact gates"
        ) from exc
    if canonical_sha != snapshot.sha256:
        raise DiagnosticRunnerError(
            "three-process equivalence receipt canonical bytes differ"
        )
    final_snapshot = read_stable_regular_file(path)
    if final_snapshot != snapshot:
        raise DiagnosticRunnerError(
            "three-process equivalence receipt changed during verification"
        )
    return snapshot.sha256


def _write_once_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    prepublish_validator: Callable[[Path], None] | None = None,
) -> str:
    destination = _lexical_absolute(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {destination}")
    fsync_directory(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        staged_snapshot = read_stable_regular_file(temporary)

        def pre_rename_guard() -> None:
            if prepublish_validator is None:
                return
            prepublish_validator(temporary)
            if read_stable_regular_file(temporary) != staged_snapshot:
                raise DiagnosticRunnerError(
                    "equivalence receipt staging bytes changed during prepublication"
                )

        publish_file_noreplace(
            temporary,
            destination,
            pre_rename_guard=pre_rename_guard,
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    # ``publish_file_noreplace`` proves the published inode is the fsynced
    # staging inode.  Returning the prepublication digest avoids a failure mode
    # where a redundant post-publication read leaves an invalid canonical name
    # that cannot be retried under the no-overwrite contract.
    return staged_snapshot.sha256


def run_equivalence_smoke(
    *,
    config_path: Path,
    dataset: str,
    device_name: str,
    condition_key: str,
    image_index: int,
    process_id: str,
    parent_run_nonce: str,
    child_launch_nonce: str,
    output: Path,
) -> dict[str, Any]:
    """Compare shared-gradient D0 with the frozen independent v2 method path.

    This gate is label-free.  It intentionally builds a fresh historical
    model/method/optimizer for every candidate, just as Stage 1 did.  A failed
    receipt is still written for audit, but :func:`_verify_equivalence_receipt`
    will never authorize a canonical shard from it.
    """

    config_path = _lexical_absolute(config_path)
    config = _load_config(config_path)
    if os.environ.get("CR_SITTA_D0_PROCESS_ID") != process_id:
        raise DiagnosticRunnerError(
            "equivalence worker lacks its parent-bound fresh process ID"
        )
    if (
        not _is_lower_sha256(parent_run_nonce)
        or os.environ.get("CR_SITTA_D0_PARENT_RUN_NONCE") != parent_run_nonce
        or not _is_lower_sha256(child_launch_nonce)
        or os.environ.get("CR_SITTA_D0_CHILD_LAUNCH_NONCE")
        != child_launch_nonce
    ):
        raise DiagnosticRunnerError(
            "equivalence worker lacks its parent-issued launch nonce binding"
        )
    observed_command_sha256 = _command_sha256((sys.executable, *sys.argv))
    if (
        os.environ.get("CR_SITTA_D0_COMMAND_SHA256")
        != observed_command_sha256
    ):
        raise DiagnosticRunnerError(
            "equivalence worker command differs from the parent launch transcript"
        )
    os_process_id = os.getpid()
    process_start_time_ticks = _linux_process_start_time_ticks()
    output = _validated_output_destination(
        config=config,
        destination=output,
        role="equivalence_single",
        dataset=dataset,
        process_id=process_id,
    )
    candidates = _config_candidates(config)
    equivalence_samples = config.get("equivalence", {}).get("samples", {})
    expected_sample = equivalence_samples.get(dataset)
    if not isinstance(expected_sample, Mapping):
        raise DiagnosticRunnerError("equivalence sample is not frozen for dataset")
    if (
        condition_key != expected_sample.get("condition")
        or image_index != expected_sample.get("image_index")
    ):
        raise DiagnosticRunnerError(
            "equivalence condition/image_index differs from frozen sample"
        )
    if condition_key not in _config_conditions(config):
        raise DiagnosticRunnerError("unknown equivalence condition")
    if isinstance(image_index, bool) or not 0 <= image_index < 64:
        raise DiagnosticRunnerError("equivalence image_index must lie in [0, 63]")
    device = _configure_d0_cuda(device_name)
    contract = frozen_v2.load_contract()
    seal, caches = frozen_v2.capture_runtime_seal(contract)
    runtime_monitor = frozen_v2.RuntimeSealMonitor(seal)
    runtime_monitor.assert_unchanged(stage="d0_equivalence_entry")
    source_hashes = _source_hashes(config_path)
    source_audits = [
        _assert_source_hashes_unchanged(
            config_path=config_path,
            expected=source_hashes,
            stage="d0_equivalence_entry",
        )
    ]
    if dataset not in caches:
        raise DiagnosticRunnerError(f"unknown dataset: {dataset}")
    cache = caches[dataset]
    method_inputs = SourceCalibrationMethodInputDatasetV2(
        cache.root,
        condition_key=condition_key,
        expected_protocol_sha256=frozen_v2.CACHE_PROTOCOL_SHA256,
    )
    shared_sample = method_inputs[image_index]
    if shared_sample.get("image_id") != expected_sample.get("image_id"):
        raise DiagnosticRunnerError(
            "equivalence image_id differs from frozen sample binding"
        )
    shared_image = shared_sample.pop("image").unsqueeze(0).to(device)
    (
        shared_model,
        shared_adapter,
        shared_parameters,
        shared_names,
        assignment,
        shared_source_parameters,
        shared_source_state,
        _wrapper,
    ) = _build_model(config, dataset, device)
    shared = _label_free_episode(
        adapter=shared_adapter,
        parameters=shared_parameters,
        names=shared_names,
        source_parameters=shared_source_parameters,
        assignment=assignment,
        image=shared_image,
        metadata=shared_sample,
        candidates=candidates,
        config=config,
        dataset=dataset,
    )
    validate_determinism_audit(
        shared.entropy_backward_determinism,
        determinism_config=config["method"]["determinism"],
        expected_scope="entropy_backward",
        expected_loss_device_type="cuda",
        expected_backward_completed=True,
    )
    _assert_model_state_exact(shared_model, shared_source_state)
    comparisons: list[dict[str, Any]] = []
    for candidate in candidates:
        historical_candidate = next(
            value
            for value in frozen_v2.ALL_CANDIDATES
            if value.optimizer == candidate.optimizer
            and float(value.learning_rate) == candidate.learning_rate
        )
        runner, _build = frozen_v2._build_fast_runner_v2(
            contract=contract,
            dataset=dataset,
            candidate=historical_candidate,
            bn_protocol=frozen_v2.SS_BN_PROTOCOL,
            device=device,
        )
        historical_names = tuple(runner.method.parameter_names)
        original_optimizer_step, step_capture = _install_first_step_capture(
            runner.method.optimizer, historical_names
        )
        original_sample = method_inputs[image_index]
        original_image = original_sample.pop("image").unsqueeze(0).to(device)
        try:
            result = frozen_v2.core.run_label_free_episode(
                runner,
                image=original_image,
                metadata=original_sample,
            )
        finally:
            runner.method.optimizer.step = original_optimizer_step
        if set(step_capture) != {
            "gradient_bundle_sha256",
            "delta_bundle_sha256",
        }:
            raise DiagnosticRunnerError(
                "historical equivalence path did not expose exactly one optimizer step"
            )
        runner.state.assert_source_state()
        historical_reset = all(
            result.checks.get(key) is True
            for key in (
                "source_runtime_restored_per_image",
                "bn_affine_restored_per_image",
                "optimizer_restored_per_image",
                "rng_restored",
            )
        )
        historical_determinism = all(
            (
                result.diagnostics.get(
                    "forward_deterministic_algorithms_enabled"
                )
                is True,
                result.diagnostics.get(
                    "forward_deterministic_algorithms_warn_only"
                )
                is False,
                result.diagnostics.get(
                    "backward_deterministic_algorithms_enabled"
                )
                is False,
                result.diagnostics.get(
                    "backward_deterministic_algorithms_warn_only"
                )
                is False,
                result.diagnostics.get("episode_device_type") == "cuda",
                result.diagnostics.get("cuda_backward_determinism_policy")
                == "temporarily_disable",
                result.diagnostics.get(
                    "deterministic_policy_restored_after_backward"
                )
                is True,
                result.diagnostics.get("deterministic_algorithms_enabled")
                is True,
            )
        )
        if not historical_determinism:
            raise DiagnosticRunnerError(
                f"historical determinism receipt failed: {candidate.slug}"
            )
        shared_state = shared.candidates[candidate.slug]
        if (
            shared_state.strict_policy_before_optimizer_step is not True
            or shared_state.strict_policy_before_post_forward is not True
        ):
            raise DiagnosticRunnerError(
                f"shared determinism restoration failed: {candidate.slug}"
            )
        pre_exact = torch.equal(shared.logits_pre, result.logits_tent_pre)
        post_exact = torch.equal(shared_state.logits_post, result.logits_tent_post)
        shared_step_norm = float(
            shared_state.geometry["global"]["actual_step_norm"]
        )
        original_step_norm = float(result.diagnostics["step_norm"])
        shared_changed = sum(
            not torch.equal(
                shared.parameters_before[name],
                shared_state.parameters_after[name],
            )
            for name in shared_names
        )
        original_changed = int(
            result.diagnostics["number_updated_bn_affine_tensors"]
        )
        shared_gradient_hash = _named_tensor_bundle_sha256(
            shared_names, shared.entropy_gradients
        )
        shared_delta_hash = _named_tensor_bundle_sha256(
            shared_names, shared_state.step
        )
        historical_gradient_hash = str(step_capture["gradient_bundle_sha256"])
        historical_delta_hash = str(step_capture["delta_bundle_sha256"])
        comparisons.append(
            {
                "candidate": candidate.to_dict(),
                "candidate_slug": candidate.slug,
                "pre_logits_bit_exact": pre_exact,
                "post_logits_bit_exact": post_exact,
                "post_logits_max_abs_difference": float(
                    torch.max(
                        torch.abs(
                            shared_state.logits_post.to(torch.float64)
                            - result.logits_tent_post.to(torch.float64)
                        )
                    ).item()
                ),
                "shared_post_logits_tensor_sha256": _strided_tensor_sha256(
                    shared_state.logits_post
                ),
                "shared_step_norm": shared_step_norm,
                "historical_step_norm": original_step_norm,
                "step_norm_abs_difference": abs(
                    shared_step_norm - original_step_norm
                ),
                "shared_changed_parameter_tensors": shared_changed,
                "historical_changed_parameter_tensors": original_changed,
                "changed_parameter_tensor_count_equal": (
                    shared_changed == original_changed
                ),
                "shared_entropy_gradient_bundle_sha256": shared_gradient_hash,
                "historical_entropy_gradient_bundle_sha256": (
                    historical_gradient_hash
                ),
                "entropy_gradient_bundle_sha256_equal": (
                    shared_gradient_hash == historical_gradient_hash
                ),
                "shared_parameter_delta_bundle_sha256": shared_delta_hash,
                "historical_parameter_delta_bundle_sha256": historical_delta_hash,
                "parameter_delta_bundle_sha256_equal": (
                    shared_delta_hash == historical_delta_hash
                ),
                "historical_reset_exact_source": historical_reset,
            }
        )
        del runner
    pre_all = all(value["pre_logits_bit_exact"] for value in comparisons)
    post_all = all(value["post_logits_bit_exact"] for value in comparisons)
    changed_all = all(
        value["changed_parameter_tensor_count_equal"] for value in comparisons
    )
    gradient_hashes_all = all(
        value["entropy_gradient_bundle_sha256_equal"] for value in comparisons
    )
    delta_hashes_all = all(
        value["parameter_delta_bundle_sha256_equal"] for value in comparisons
    )
    resets_all = all(value["historical_reset_exact_source"] for value in comparisons)
    step_all = all(
        value["step_norm_abs_difference"] <= 1e-12 for value in comparisons
    )
    complete = cache.complete
    zero_test = (
        int(complete["test_images_opened"]) == 0
        and int(complete["test_masks_opened"]) == 0
    )
    runtime_monitor.assert_unchanged(
        stage="d0_equivalence_pre_receipt", full_byte_rehash=True
    )
    source_audits.append(
        _assert_source_hashes_unchanged(
            config_path=config_path,
            expected=source_hashes,
            stage="d0_equivalence_pre_receipt",
        )
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": EQUIVALENCE_TYPE,
        "paper_test_result": False,
        "source_train_derived": True,
        "oracle_analysis": False,
        "fresh_process": True,
        "parent_run_nonce": parent_run_nonce,
        "child_launch_nonce": child_launch_nonce,
        "command_sha256": observed_command_sha256,
        "process_id": process_id,
        "os_process_id": os_process_id,
        "process_start_time_ticks": process_start_time_ticks,
        "dataset": dataset,
        "condition": condition_key,
        "image_index": image_index,
        "image_id": shared.metadata["image_id"],
        "config_sha256": read_stable_regular_file(config_path).sha256,
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "diagnostic_source_code_sha256": dict(sorted(source_hashes.items())),
        "determinism_contract": config["method"]["determinism"],
        "logits_tensor_sha256_contract": LOGITS_TENSOR_SHA256_CONTRACT,
        "shared_pre_logits_tensor_sha256": _strided_tensor_sha256(
            shared.logits_pre
        ),
        "runtime_audits": runtime_monitor.audits,
        "diagnostic_source_audits": source_audits,
        "candidate_slugs": [candidate.slug for candidate in candidates],
        "candidate_count": len(candidates),
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": 0,
        "test_images_opened": 0,
        "test_labels_opened": 0,
        "target_payload_deserialized": False,
        "pre_logits_all_bit_exact": pre_all,
        "post_logits_all_bit_exact": post_all,
        "step_norm_all_within_1e_minus_12": step_all,
        "changed_parameter_tensor_counts_all_equal": changed_all,
        "entropy_gradient_hashes_all_equal": gradient_hashes_all,
        "parameter_delta_hashes_all_equal": delta_hashes_all,
        "historical_resets_all_exact_source": resets_all,
        "cache_zero_test_opens_verified": zero_test,
        "comparisons": comparisons,
        "passed": bool(
            pre_all
            and post_all
            and step_all
            and changed_all
            and gradient_hashes_all
            and delta_hashes_all
            and resets_all
            and zero_test
        ),
    }
    _ensure_output_parent(
        config=config, role="equivalence_single", dataset=dataset
    )
    digest = _write_once_json(output, receipt)
    return {
        "status": "passed" if receipt["passed"] else "failed",
        "output": str(output.resolve()),
        "receipt_sha256": digest,
        "candidate_count": len(comparisons),
        "process_id": process_id,
        "parent_run_nonce": parent_run_nonce,
        "child_launch_nonce": child_launch_nonce,
        "command_sha256": observed_command_sha256,
        "os_process_id": os_process_id,
        "process_start_time_ticks": process_start_time_ticks,
    }


def run_equivalence_repro(
    *,
    config_path: Path,
    dataset: str,
    device_name: str,
    output: Path,
) -> dict[str, Any]:
    """Launch and aggregate exactly three independent exec-based GPU workers."""

    config_path = _lexical_absolute(config_path)
    config = _load_config(config_path)
    output = _validated_output_destination(
        config=config,
        destination=output,
        role="equivalence",
        dataset=dataset,
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {output}")
    if dataset not in config["datasets"]:
        raise DiagnosticRunnerError(f"unknown equivalence dataset: {dataset}")
    if device_name != "cuda:0":
        raise DiagnosticRunnerError(
            "fresh-process equivalence requires isolated logical cuda:0"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        not isinstance(visible, str)
        or not visible.strip()
        or len([value for value in visible.split(",") if value.strip()]) != 1
    ):
        raise DiagnosticRunnerError(
            "equivalence parent must expose exactly one CUDA_VISIBLE_DEVICES entry"
        )
    if torch.cuda.is_initialized():
        raise DiagnosticRunnerError(
            "equivalence parent initialized CUDA before spawning fresh workers"
        )
    sample = config["equivalence"]["samples"][dataset]
    _ensure_output_parent(config=config, role="equivalence")
    run_parent = _ensure_output_parent(
        config=config, role="equivalence_single", dataset=dataset
    )
    bindings: list[EquivalenceReceiptBinding] = []
    launches: list[dict[str, Any]] = []
    worker_script = _lexical_absolute(Path(__file__))
    replicate_count = int(config["equivalence"]["fresh_process_repetitions"])
    parent_run_nonce = secrets.token_hex(32)
    if not _is_lower_sha256(parent_run_nonce):
        raise DiagnosticRunnerError("parent run nonce generation failed closed")
    parent_os_process_id = os.getpid()
    parent_process_start_time_ticks = _linux_process_start_time_ticks()
    for replicate in range(1, replicate_count + 1):
        process_id = f"repro-{replicate}-{uuid.uuid4().hex}"
        child_launch_nonce = secrets.token_hex(32)
        worker_output = _lexical_absolute(run_parent / f"{process_id}.json")
        _validated_output_destination(
            config=config,
            destination=worker_output,
            role="equivalence_single",
            dataset=dataset,
            process_id=process_id,
        )
        command = [
            sys.executable,
            str(worker_script),
            "equivalence-smoke",
            "--config",
            str(config_path),
            "--dataset",
            dataset,
            "--device",
            device_name,
            "--condition",
            str(sample["condition"]),
            "--image-index",
            str(sample["image_index"]),
            "--process-id",
            process_id,
            "--parent-run-nonce",
            parent_run_nonce,
            "--child-launch-nonce",
            child_launch_nonce,
            "--output",
            str(worker_output),
        ]
        command_sha256 = _command_sha256(command)
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CR_SITTA_D0_PROCESS_ID": process_id,
                "CR_SITTA_D0_PARENT_RUN_NONCE": parent_run_nonce,
                "CR_SITTA_D0_CHILD_LAUNCH_NONCE": child_launch_nonce,
                "CR_SITTA_D0_COMMAND_SHA256": command_sha256,
            }
        )
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        observed_start_ticks = _linux_process_start_time_ticks(process.pid)
        stdout, stderr = process.communicate()
        stdout_sha256 = hashlib.sha256(stdout).hexdigest()
        stderr_sha256 = hashlib.sha256(stderr).hexdigest()
        if process.returncode != 0:
            raise DiagnosticRunnerError(
                "fresh equivalence worker failed; immutable failed-run evidence "
                f"is retained when present: process_id={process_id}, "
                f"exit={process.returncode}, stderr_tail={stderr[-4000:]!r}"
            )
        try:
            worker_snapshot = read_stable_regular_file(worker_output)
            worker_receipt = _decode_json_bytes(
                worker_snapshot.data, str(worker_output)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise DiagnosticRunnerError(
                f"fresh equivalence worker did not publish a stable receipt: {process_id}"
            ) from exc
        if (
            worker_receipt.get("process_id") != process_id
            or worker_receipt.get("parent_run_nonce") != parent_run_nonce
            or worker_receipt.get("child_launch_nonce") != child_launch_nonce
            or worker_receipt.get("command_sha256") != command_sha256
            or worker_receipt.get("os_process_id") != process.pid
            or worker_receipt.get("process_start_time_ticks")
            != observed_start_ticks
        ):
            raise DiagnosticRunnerError(
                f"fresh equivalence process identity differs: {process_id}"
            )
        bindings.append(
            EquivalenceReceiptBinding(
                path=worker_output,
                sha256=worker_snapshot.sha256,
                process_id=process_id,
                os_process_id=process.pid,
                process_start_time_ticks=observed_start_ticks,
            )
        )
        launches.append(
            {
                "process_id": process_id,
                "parent_run_nonce": parent_run_nonce,
                "child_launch_nonce": child_launch_nonce,
                "command_sha256": command_sha256,
                "os_process_id": process.pid,
                "process_start_time_ticks": observed_start_ticks,
                "returncode": int(process.returncode),
                "stdout_sha256": stdout_sha256,
                "stderr_sha256": stderr_sha256,
                "receipt_path": str(worker_output),
                "receipt_sha256": worker_snapshot.sha256,
            }
        )
        if stdout.strip():
            print(
                stdout.decode("utf-8", errors="replace").strip(),
                file=sys.stderr,
                flush=True,
            )
    if torch.cuda.is_initialized():
        raise DiagnosticRunnerError(
            "equivalence parent unexpectedly initialized CUDA while orchestrating"
        )
    try:
        aggregate_receipt = build_d0_equivalence_repro_receipt(
            bindings,
            parent_run={
                "parent_run_nonce": parent_run_nonce,
                "parent_os_process_id": parent_os_process_id,
                "parent_process_start_time_ticks": (
                    parent_process_start_time_ticks
                ),
                "launches": launches,
            },
        )
    except D0EquivalenceReproContractError as exc:
        raise DiagnosticRunnerError(
            "fresh-process equivalence did not reproduce exactly"
        ) from exc
    contract = frozen_v2.load_contract()
    seal, _caches = frozen_v2.capture_runtime_seal(contract)
    expected_config_sha256 = read_stable_regular_file(config_path).sha256
    expected_source_hashes = _source_hashes(config_path)

    def validate_staged_receipt(staged_path: Path) -> None:
        _verify_equivalence_receipt(
            staged_path,
            dataset=dataset,
            candidates=_config_candidates(config),
            config_sha256=expected_config_sha256,
            global_runtime_seal_sha256=seal.global_runtime_seal_sha256,
            diagnostic_source_code_sha256=expected_source_hashes,
            expected_condition=str(sample["condition"]),
            expected_image_index=int(sample["image_index"]),
            expected_image_id=str(sample["image_id"]),
        )

    digest = _write_once_json(
        output,
        aggregate_receipt,
        prepublish_validator=validate_staged_receipt,
    )
    return {
        "status": "passed",
        "output": str(output),
        "receipt_sha256": digest,
        "fresh_process_count": replicate_count,
        "parent_run_nonce": parent_run_nonce,
        "process_ids": [binding.process_id for binding in bindings],
    }


def run_dataset(
    *,
    config_path: Path,
    dataset: str,
    device_name: str,
    destination: Path,
    condition_filters: Sequence[str],
    candidate_filters: Sequence[str],
    max_images: int,
    equivalence_receipt: Path | None,
) -> dict[str, Any]:
    config_path = _lexical_absolute(config_path)
    config = _load_config(config_path)
    all_conditions = _config_conditions(config)
    all_candidates = _config_candidates(config)
    conditions = (
        tuple(condition_filters) if condition_filters else all_conditions
    )
    candidates = (
        tuple(
            candidate
            for candidate in all_candidates
            if candidate.slug in set(candidate_filters)
        )
        if candidate_filters
        else all_candidates
    )
    if not conditions or any(value not in all_conditions for value in conditions):
        raise DiagnosticRunnerError("condition filter is empty or unknown")
    if not candidates or (
        candidate_filters and set(candidate_filters) != {value.slug for value in candidates}
    ):
        raise DiagnosticRunnerError("candidate filter is empty or unknown")
    if isinstance(max_images, bool) or not 1 <= max_images <= 64:
        raise DiagnosticRunnerError("max_images must lie in [1, 64]")
    formal_scope = (
        conditions == all_conditions
        and candidates == all_candidates
        and max_images == 64
    )
    config_sha = read_stable_regular_file(config_path).sha256
    equivalence_sha = None
    if formal_scope:
        if equivalence_receipt is None:
            raise DiagnosticRunnerError(
                "canonical D0 is blocked until exact shared-gradient equivalence passes"
            )
        equivalence_receipt = _validated_output_destination(
            config=config,
            destination=equivalence_receipt,
            role="equivalence",
            dataset=dataset,
        )
    output_role = "formal_shard" if formal_scope else "smoke_shard"
    destination = _validated_output_destination(
        config=config,
        destination=destination,
        role=output_role,
        dataset=dataset,
    )
    device = _configure_d0_cuda(device_name)

    contract = frozen_v2.load_contract()
    seal, caches = frozen_v2.capture_runtime_seal(contract)
    runtime_monitor = frozen_v2.RuntimeSealMonitor(seal)
    runtime_monitor.assert_unchanged(stage="d0_dataset_entry")
    source_hashes = _source_hashes(config_path)
    source_audits = [
        _assert_source_hashes_unchanged(
            config_path=config_path,
            expected=source_hashes,
            stage="d0_dataset_entry",
        )
    ]
    if formal_scope:
        assert equivalence_receipt is not None
        equivalence_sha = _verify_equivalence_receipt(
            equivalence_receipt,
            dataset=dataset,
            candidates=candidates,
            config_sha256=config_sha,
            global_runtime_seal_sha256=seal.global_runtime_seal_sha256,
            diagnostic_source_code_sha256=source_hashes,
            expected_condition=str(
                config["equivalence"]["samples"][dataset]["condition"]
            ),
            expected_image_index=int(
                config["equivalence"]["samples"][dataset]["image_index"]
            ),
            expected_image_id=str(
                config["equivalence"]["samples"][dataset]["image_id"]
            ),
        )
    if dataset not in caches:
        raise DiagnosticRunnerError(f"dataset absent from frozen cache: {dataset}")
    (
        model,
        adapter,
        parameters,
        names,
        assignment,
        source_parameters,
        source_state,
        checkpoint_wrapper,
    ) = _build_model(config, dataset, device)
    cache = caches[dataset]
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for condition_key in conditions:
        method_inputs = SourceCalibrationMethodInputDatasetV2(
            cache.root,
            condition_key=condition_key,
            expected_protocol_sha256=frozen_v2.CACHE_PROTOCOL_SHA256,
        )
        episodes: list[LabelFreeEpisode] = []
        # Label-free phase.  No target loader is reachable inside this loop.
        for index in range(max_images):
            sample = method_inputs[index]
            image = sample.pop("image").unsqueeze(0).to(device)
            episodes.append(
                _label_free_episode(
                    adapter=adapter,
                    parameters=parameters,
                    names=names,
                    source_parameters=source_parameters,
                    assignment=assignment,
                    image=image,
                    metadata=sample,
                    candidates=candidates,
                    config=config,
                    dataset=dataset,
                )
            )
        _assert_model_state_exact(model, source_state)
        # The target file is first deserialized here, after all selected
        # label-free episodes and candidates in this condition are complete.
        targets = load_outer_evaluator_targets_v2(
            cache.root,
            expected_protocol_sha256=frozen_v2.CACHE_PROTOCOL_SHA256,
            episodes_complete=True,
        )
        for index, episode in enumerate(episodes):
            sample = method_inputs[index]
            image = sample.pop("image").unsqueeze(0).to(device)
            target_np = np.array(targets[index], dtype=np.float32, copy=True)
            target = torch.from_numpy(target_np).unsqueeze(0).to(device)
            supervised, task_loss, supervised_backward_determinism = (
                _supervised_gradients(
                    model=model,
                    adapter=adapter,
                    parameters=parameters,
                    names=names,
                    source_parameters=source_parameters,
                    image=image,
                    target=target,
                    task_contract=config["source_task_gradient"],
                    config=config,
                )
            )
            oracle_scope = _provenance(
                config=config, dataset=dataset, oracle=True, accesses=1
            )
            for candidate in candidates:
                candidate_state = episode.candidates[candidate.slug]
                noop = analyze_noop_episode(
                    logits_pre=episode.logits_pre,
                    logits_post=candidate_state.logits_post,
                    target=target_np,
                    parameter_pre=episode.parameters_before,
                    parameter_post=candidate_state.parameters_after,
                    thresholds=_thresholds(config),
                )
                alignment = analyze_entropy_task_alignment(
                    entropy_gradients=_nonnull(episode.entropy_gradients),
                    supervised_gradients=supervised,
                    adaptation_step=candidate_state.step,
                    provenance=oracle_scope,
                    parameter_groups=assignment,
                    first_order_zero_tolerance=float(
                        config["evaluation"]["alignment"][
                            "first_order_zero_tolerance"
                        ]
                    ),
                )
                records.append(
                    {
                        "schema_version": 1,
                        "dataset": dataset,
                        "split_role": "train",
                        "image_id": episode.metadata["image_id"],
                        "corruption": episode.metadata["corruption"],
                        "severity": episode.metadata["severity"],
                        "candidate_slug": candidate.slug,
                        "candidate": candidate.to_dict(),
                        "optimization_entropy_pre": episode.entropy_pre,
                        "source_eval_post_warm_task_loss": task_loss,
                        "noop": _prune_noop(noop, formal=formal_scope),
                        "optimizer_geometry": candidate_state.geometry,
                        "entropy_task_alignment": alignment,
                        "determinism": {
                            "entropy_backward": (
                                episode.entropy_backward_determinism
                            ),
                            "supervised_task_backward": (
                                supervised_backward_determinism
                            ),
                            "strict_policy_before_optimizer_step": (
                                candidate_state
                                .strict_policy_before_optimizer_step
                            ),
                            "strict_policy_before_post_forward": (
                                candidate_state.strict_policy_before_post_forward
                            ),
                        },
                        "scope": {
                            "oracle_analysis": True,
                            "paper_test_result": False,
                            "source_train_derived": True,
                            "method_label_accesses": 0,
                            "outer_evaluator_label_accesses": 1,
                            "use_test_images": False,
                            "use_test_labels": False,
                        },
                    }
                )
        del targets, episodes, method_inputs
        _assert_model_state_exact(model, source_state)
        runtime_monitor.assert_unchanged(
            stage=f"d0_condition_complete:{dataset}:{condition_key}"
        )
        source_audits.append(
            _assert_source_hashes_unchanged(
                config_path=config_path,
                expected=source_hashes,
                stage=f"d0_condition_complete:{dataset}:{condition_key}",
            )
        )
        print(
            json.dumps(
                {
                    "event": "d0_condition_complete",
                    "dataset": dataset,
                    "condition": condition_key,
                    "records": len(records),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
    expected_records = len(conditions) * max_images * len(candidates)
    if len(records) != expected_records:
        raise DiagnosticRunnerError("D0 record count differs from execution scope")
    _assert_model_state_exact(model, source_state)
    frozen_v2_comparison: dict[str, Any] | None = None
    frozen_v2_records_path: Path | None = None
    if formal_scope:
        # This is a scientific identity gate, not a contextual comparison:
        # all 64*13*10 episode sufficient statistics must collapse to the
        # exact 130 frozen-v2 cells before a formal shard can be published.
        frozen_v2_records_path = _archive_records_path_from_config(config)
        frozen_v2_comparison = verify_formal_dataset_against_frozen_v2(
            records=records,
            dataset=dataset,
            config=config,
            archive_records_path=frozen_v2_records_path,
        )
    summary = summarize_records(records)
    validate_d0_records(
        records,
        config=config,
        dataset=dataset,
        conditions=conditions,
        candidates=candidates,
        images_per_condition=max_images,
    )
    if formal_scope:
        assert frozen_v2_records_path is not None
        assert frozen_v2_comparison is not None
        # Repeat the complete comparison after the runtime/source prepublish
        # audits.  Equality includes the archive SHA binding and closes the
        # archive-read/publish TOCTOU window fail-closed.
        repeated_comparison = verify_formal_dataset_against_frozen_v2(
            records=records,
            dataset=dataset,
            config=config,
            archive_records_path=frozen_v2_records_path,
        )
        if repeated_comparison != frozen_v2_comparison:
            raise DiagnosticRunnerError(
                "frozen v2 Stage-1 comparison changed before D0 publish"
            )
        assert equivalence_receipt is not None
        repeated_equivalence_sha = _verify_equivalence_receipt(
            equivalence_receipt,
            dataset=dataset,
            candidates=candidates,
            config_sha256=config_sha,
            global_runtime_seal_sha256=seal.global_runtime_seal_sha256,
            diagnostic_source_code_sha256=source_hashes,
            expected_condition=str(
                config["equivalence"]["samples"][dataset]["condition"]
            ),
            expected_image_index=int(
                config["equivalence"]["samples"][dataset]["image_index"]
            ),
            expected_image_id=str(
                config["equivalence"]["samples"][dataset]["image_id"]
            ),
        )
        if repeated_equivalence_sha != equivalence_sha:
            raise DiagnosticRunnerError(
                "equivalence receipt changed before D0 publish"
            )
    # External archive/receipt reads above happen first.  Runtime and source
    # bytes are the final recorded audit before provenance serialization.
    runtime_monitor.assert_unchanged(
        stage="d0_dataset_pre_publish", full_byte_rehash=True
    )
    source_audits.append(
        _assert_source_hashes_unchanged(
            config_path=config_path,
            expected=source_hashes,
            stage="d0_dataset_pre_publish",
        )
    )
    cache_targets = cache.manifest["targets"]
    provenance = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "dataset": dataset,
        "formal_d0_complete": formal_scope,
        "paper_test_result": False,
        "oracle_analysis": True,
        "source_train_derived": True,
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": len(conditions) * max_images,
        "test_images_opened": 0,
        "test_labels_opened": 0,
        "conditions": list(conditions),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "images_per_condition": max_images,
        "checkpoint_wrapper": checkpoint_wrapper,
        "checkpoint_sha256": config["datasets"][dataset]["checkpoint"]["sha256"],
        "train_split_sha256": config["datasets"][dataset]["train_split_sha256"],
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "equivalence_receipt_sha256": equivalence_sha,
        "equivalence_receipt": (
            None
            if equivalence_receipt is None
            else {
                "path": str(_lexical_absolute(equivalence_receipt)),
                "sha256": equivalence_sha,
            }
        ),
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "v2_runtime_seal": seal.to_dict(),
        "runtime_audits": runtime_monitor.audits,
        "diagnostic_source_code_sha256": dict(sorted(source_hashes.items())),
        "diagnostic_source_audits": source_audits,
        "frozen_v2_stage1_exact_comparison": frozen_v2_comparison,
        "cache_lineage": {
            "root": str(cache.root.resolve()),
            "manifest_sha256": cache.manifest_sha256,
            "complete_sha256": cache.complete_sha256,
            "cache_content_sha256": cache.manifest["cache_content_sha256"],
            "method_input_manifest_sha256": cache.complete[
                "method_input_manifest_sha256"
            ],
            "ordered_ids_sha256": cache.manifest["ordered_ids_sha256"],
            "calibration_ids_file_sha256": cache.manifest[
                "calibration_ids_file_sha256"
            ],
            "target_file_sha256": cache_targets["file_sha256"],
            "target_tensor_sequence_sha256": cache_targets[
                "tensor_sequence_sha256"
            ],
            "test_images_opened": cache.complete["test_images_opened"],
            "test_masks_opened": cache.complete["test_masks_opened"],
        },
        "target_deserialized_only_after_cell_label_free_completion": True,
        "source_eval_task_gradient_contract": dict(
            config["source_task_gradient"]
        ),
        "determinism_contract": dict(config["method"]["determinism"]),
        "determinism_contract_sha256": frozen_determinism_sha256(
            config["method"]["determinism"]
        ),
        "selection_authorized": False,
        "stage2_authorized": False,
        "stage3_authorized": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _ensure_output_parent(config=config, role=output_role)

    def prepublish_guard() -> None:
        # Run after every staging byte has been created and immediately before
        # the atomic no-replace rename.  This prevents an immutable but already
        # stale shard from being published when an input changes late.
        if formal_scope:
            assert frozen_v2_records_path is not None
            assert frozen_v2_comparison is not None
            assert equivalence_receipt is not None
            if verify_formal_dataset_against_frozen_v2(
                records=records,
                dataset=dataset,
                config=config,
                archive_records_path=frozen_v2_records_path,
            ) != frozen_v2_comparison:
                raise DiagnosticRunnerError(
                    "frozen v2 Stage-1 comparison changed at publication boundary"
                )
            if _verify_equivalence_receipt(
                equivalence_receipt,
                dataset=dataset,
                candidates=candidates,
                config_sha256=config_sha,
                global_runtime_seal_sha256=seal.global_runtime_seal_sha256,
                diagnostic_source_code_sha256=source_hashes,
                expected_condition=str(
                    config["equivalence"]["samples"][dataset]["condition"]
                ),
                expected_image_index=int(
                    config["equivalence"]["samples"][dataset]["image_index"]
                ),
                expected_image_id=str(
                    config["equivalence"]["samples"][dataset]["image_id"]
                ),
            ) != equivalence_sha:
                raise DiagnosticRunnerError(
                    "equivalence receipt changed at publication boundary"
                )
        runtime_monitor.assert_unchanged(
            stage="d0_atomic_publish_boundary", full_byte_rehash=True
        )
        _assert_source_hashes_unchanged(
            config_path=config_path,
            expected=source_hashes,
            stage="d0_atomic_publish_boundary",
        )

    return _publish_shard(
        destination=destination,
        records=records,
        summary=summary,
        provenance=provenance,
        formal_scope=formal_scope,
        source_hashes=source_hashes,
        prepublish_guard=prepublish_guard,
    )


# ---------------------------------------------------------------------------
# Immutable three-dataset D0 aggregate
# ---------------------------------------------------------------------------


def _fraction_from_receipt(value: Any, label: str) -> Fraction:
    if not isinstance(value, Mapping):
        raise DiagnosticRunnerError(f"{label} must be an exact Fraction receipt")
    numerator = _nonnegative_or_signed_integer(value.get("numerator"), f"{label}.numerator")
    denominator = _nonnegative_or_signed_integer(
        value.get("denominator"), f"{label}.denominator"
    )
    if denominator <= 0:
        raise DiagnosticRunnerError(f"{label}.denominator must be positive")
    result = Fraction(numerator, denominator)
    if (
        value.get("exact") != f"{result.numerator}/{result.denominator}"
        or numerator != result.numerator
        or denominator != result.denominator
    ):
        raise DiagnosticRunnerError(f"{label} is not a canonical Fraction receipt")
    return result


def _nonnegative_or_signed_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiagnosticRunnerError(f"{label} must be an integer")
    return value


def build_candidate_macro_summary(
    *,
    cell_records: Sequence[Mapping[str, Any]],
    episode_records: Sequence[Mapping[str, Any]],
    expected_candidates: Sequence[Candidate],
) -> dict[str, Any]:
    """Build candidate-level exact macro metrics plus diagnostic counts."""

    grouped_cells: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in cell_records:
        grouped_cells[str(cell.get("candidate_slug"))].append(cell)
    grouped_episodes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in episode_records:
        grouped_episodes[str(record.get("candidate_slug"))].append(record)
    expected_slugs = {candidate.slug for candidate in expected_candidates}
    if set(grouped_cells) != expected_slugs or set(grouped_episodes) != expected_slugs:
        raise DiagnosticRunnerError("aggregate candidate identity set differs")
    diagnostic_summary = summarize_records(episode_records)
    diagnostic_candidates = diagnostic_summary.get("candidates")
    if not isinstance(diagnostic_candidates, Mapping) or set(
        diagnostic_candidates
    ) != expected_slugs:
        raise DiagnosticRunnerError("aggregate diagnostic summary candidate set differs")

    candidates: dict[str, Any] = {}
    for candidate in expected_candidates:
        cells = grouped_cells[candidate.slug]
        episodes = grouped_episodes[candidate.slug]
        diagnostic = diagnostic_candidates[candidate.slug]
        if not isinstance(diagnostic, Mapping):
            raise DiagnosticRunnerError(
                f"aggregate diagnostic summary invalid: {candidate.slug}"
            )
        metric_deltas: dict[str, list[Fraction]] = defaultdict(list)
        per_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for index, cell in enumerate(cells):
            per_dataset[str(cell["dataset"])].append(cell)
            delta = cell.get("metrics_exact", {}).get("delta")
            if not isinstance(delta, Mapping):
                raise DiagnosticRunnerError(
                    f"candidate cell[{index}] lacks exact delta metrics"
                )
            for metric in (
                "global_iou",
                "pd",
                "fa",
                "fa_per_million_pixels",
            ):
                metric_deltas[metric].append(
                    _fraction_from_receipt(
                        delta.get(metric), f"{candidate.slug}.cell[{index}].{metric}"
                    )
                )
        macro = {
            metric: _fraction_receipt(
                sum(values, Fraction(0, 1)) / len(values)
            )
            for metric, values in metric_deltas.items()
        }
        dataset_macro: dict[str, Any] = {}
        for dataset, dataset_cells in sorted(per_dataset.items()):
            dataset_macro[dataset] = {
                metric: _fraction_receipt(
                    sum(
                        (
                            _fraction_from_receipt(
                                cell["metrics_exact"]["delta"][metric],
                                f"{candidate.slug}.{dataset}.{metric}",
                            )
                            for cell in dataset_cells
                        ),
                        Fraction(0, 1),
                    )
                    / len(dataset_cells)
                )
                for metric in (
                    "global_iou",
                    "pd",
                    "fa",
                    "fa_per_million_pixels",
                )
            }
        classifications = Counter()
        first_order_effects = Counter()
        threshold_xor_episode_count = 0
        for index, record in enumerate(episodes):
            noop = record.get("noop")
            alignment = record.get("entropy_task_alignment")
            if not isinstance(noop, Mapping) or not isinstance(alignment, Mapping):
                raise DiagnosticRunnerError(
                    f"aggregate episode[{index}] lacks D0 diagnostics"
                )
            classification = noop.get("classification")
            if not isinstance(classification, str) or not classification:
                raise DiagnosticRunnerError(
                    f"aggregate episode[{index}] classification invalid"
                )
            classifications[classification] += 1
            transitions = noop.get("binary_transitions")
            if not isinstance(transitions, Mapping):
                raise DiagnosticRunnerError(
                    f"aggregate episode[{index}] transitions invalid"
                )
            threshold_xor_episode_count += int(
                _nonnegative_integer(
                    transitions.get("binary_pixel_xor_count"),
                    f"aggregate episode[{index}].binary_pixel_xor_count",
                )
                > 0
            )
            global_alignment = alignment.get("global")
            if not isinstance(global_alignment, Mapping):
                raise DiagnosticRunnerError(
                    f"aggregate episode[{index}] global alignment invalid"
                )
            effect = global_alignment.get("first_order_task_effect")
            if effect not in {
                "predicted_task_loss_decrease",
                "predicted_task_loss_increase",
                "first_order_neutral",
            }:
                raise DiagnosticRunnerError(
                    f"aggregate episode[{index}] first-order effect invalid"
                )
            first_order_effects[str(effect)] += 1
        candidates[candidate.slug] = {
            "candidate": candidate.to_dict(),
            "cell_count": len(cells),
            "diagnostic_record_count": len(episodes),
            "macro_exact_delta": macro,
            "per_dataset_macro_exact_delta": dataset_macro,
            "classification_counts": dict(sorted(classifications.items())),
            "threshold_xor_episode_count": threshold_xor_episode_count,
            "first_order_task_effect_counts": dict(
                sorted(first_order_effects.items())
            ),
            "cpu_storage_replay_within_frozen_tolerance_episode_count": diagnostic[
                "cpu_storage_replay_within_frozen_tolerance_episode_count"
            ],
            "per_group": diagnostic["per_group"],
            "threshold_margin": diagnostic["threshold_margin"],
        }
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0_candidate_macro_summary_v1",
        "paper_test_result": False,
        "oracle_analysis": True,
        "candidate_count": len(candidates),
        "cell_count": len(cell_records),
        "diagnostic_record_count": len(episode_records),
        "candidates": candidates,
    }


def _archive_records_path_from_config(config: Mapping[str, Any]) -> Path:
    comparison = config.get("comparison")
    expected = {
        "frozen_v2_negative_archive": (
            "results/binary_tent/ss_calibration_v2_negative_archive"
        ),
        "archive_source_inventory_sha256": (
            "bf8954dd2692f23e6b65761e4337f1daa77a64439386b05c6bb8df716800dfbf"
        ),
        "archive_file_count": 113,
        "stage1_records": "stage1/aggregate/stage1_records.jsonl",
        "stage1_records_sha256": (
            "ee6a21b29ceee32fa6f9f70c6509436971133f81e815596fd148716807bbe3d3"
        ),
        "direct_equality_required_only_for_full_64_image_cells": True,
        "smoke_comparison_role": (
            "contextual_full_cell_reference_not_direct_equality"
        ),
    }
    if not isinstance(comparison, Mapping) or dict(comparison) != expected:
        raise DiagnosticRunnerError("D0 frozen-v2 comparison contract drifted")
    archive = expected["frozen_v2_negative_archive"]
    records = expected["stage1_records"]
    root = Path(archive)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    path = _lexical_absolute(root / records)
    try:
        snapshot = read_stable_regular_file(path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise DiagnosticRunnerError(
            "frozen v2 stage1_records missing, unstable, or symlinked"
        ) from exc
    if snapshot.sha256 != expected["stage1_records_sha256"]:
        raise DiagnosticRunnerError("frozen v2 stage1_records SHA-256 drifted")
    return path


def _verify_frozen_negative_archive(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the complete immutable 113-file negative archive and bindings."""

    comparison = config["comparison"]
    root_value = Path(str(comparison["frozen_v2_negative_archive"]))
    root = _lexical_absolute(
        root_value if root_value.is_absolute() else PROJECT_ROOT / root_value
    )
    checksum_path = root / "SHA256SUMS"
    negative_path = root / "NEGATIVE_RESULT.json"
    records_path = root / str(comparison["stage1_records"])
    try:
        checksum_before = read_stable_regular_file(checksum_path)
        negative_before = read_stable_regular_file(negative_path)
        records_before = read_stable_regular_file(records_path)
        verification = verify_negative_archive(
            root,
            expected_source_inventory_sha256=str(
                comparison["archive_source_inventory_sha256"]
            ),
        )
        checksum_after = read_stable_regular_file(checksum_path)
        negative_after = read_stable_regular_file(negative_path)
        records_after = read_stable_regular_file(records_path)
    except (OSError, ValueError, RuntimeError, NegativeArchiveError) as exc:
        raise DiagnosticRunnerError(
            "frozen v2 negative archive did not pass its complete ledger gate"
        ) from exc
    if (
        checksum_before != checksum_after
        or negative_before != negative_after
        or records_before != records_after
        or verification.get("status") != "verified"
        or verification.get("paper_result") is not False
        or verification.get("file_count") != comparison["archive_file_count"]
        or verification.get("source_inventory_sha256")
        != comparison["archive_source_inventory_sha256"]
        or records_before.sha256 != comparison["stage1_records_sha256"]
    ):
        raise DiagnosticRunnerError(
            "frozen v2 negative archive identity or frozen binding drifted"
        )
    return {
        "path": str(root),
        "file_count": verification["file_count"],
        "source_inventory_sha256": verification["source_inventory_sha256"],
        "sha256sums_sha256": checksum_before.sha256,
        "negative_result_sha256": negative_before.sha256,
        "stage1_records_sha256": records_before.sha256,
        "paper_result": False,
        "verified": True,
    }


def _load_shard_records(root: Path) -> list[dict[str, Any]]:
    return _read_jsonl_objects(root / "episode_records.jsonl", "D0 episode records")


def _formal_shard_payload(
    *,
    path: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    archive_records_path: Path,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    path = _lexical_absolute(path)
    if path.is_symlink() or not path.is_dir():
        raise DiagnosticRunnerError(f"aggregate input shard missing or symlink: {path}")
    verified = verify_shard(path)
    if verified.get("formal_d0_complete") is not True:
        raise DiagnosticRunnerError(f"aggregate input is not a formal D0 shard: {path}")
    root = path
    directory = snapshot_regular_directory(root)
    snapshot_members = {value.path.name: value for value in directory.members}
    initial_payload_hashes = {
        name: snapshot_members[name].sha256
        for name in (
            "artifact_manifest.json",
            "COMPLETE.json",
            "episode_records.jsonl",
            "provenance.json",
        )
    }
    provenance = _decode_json_bytes(
        snapshot_members["provenance.json"].data,
        f"{root}/provenance.json",
    )
    dataset = provenance.get("dataset")
    if dataset not in FORMAL_DATASETS:
        raise DiagnosticRunnerError(f"aggregate shard dataset invalid: {dataset}")
    if (
        provenance.get("formal_d0_complete") is not True
        or provenance.get("paper_test_result") is not False
        or provenance.get("source_train_derived") is not True
        or provenance.get("config_sha256") != config_sha256
    ):
        raise DiagnosticRunnerError(f"aggregate shard provenance invalid: {root}")
    records = _decode_jsonl_bytes(
        snapshot_members["episode_records.jsonl"].data,
        f"{root}/episode_records.jsonl",
    )
    receipt = verify_formal_dataset_against_frozen_v2(
        records=records,
        dataset=str(dataset),
        config=config,
        archive_records_path=archive_records_path,
    )
    if provenance.get("frozen_v2_stage1_exact_comparison") != receipt:
        raise DiagnosticRunnerError(
            f"formal shard lacks exact frozen-v2 comparison receipt: {root}"
        )
    repeated_verified = verify_shard(root)
    repeated_directory = snapshot_regular_directory(root)
    repeated_members = {value.path.name: value for value in repeated_directory.members}
    repeated_payload_hashes = {
        name: repeated_members[name].sha256 for name in initial_payload_hashes
    }
    if repeated_verified != verified or repeated_payload_hashes != initial_payload_hashes:
        raise DiagnosticRunnerError(
            f"formal shard changed while being aggregated: {root}"
        )
    binding = {
        "path": str(root),
        "artifact_manifest_sha256": initial_payload_hashes[
            "artifact_manifest.json"
        ],
        "complete_sha256": initial_payload_hashes["COMPLETE.json"],
        "episode_records_sha256": initial_payload_hashes[
            "episode_records.jsonl"
        ],
        "record_count": len(records),
        "dataset": dataset,
    }
    return str(dataset), records, receipt, binding


def _canonical_record_order(
    records: Sequence[Mapping[str, Any]],
    *,
    datasets: Sequence[str],
    conditions: Sequence[str],
    candidates: Sequence[Candidate],
) -> list[Mapping[str, Any]]:
    dataset_index = {value: index for index, value in enumerate(datasets)}
    condition_index = {value: index for index, value in enumerate(conditions)}
    candidate_index = {
        value.slug: index for index, value in enumerate(candidates)
    }

    def key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            dataset_index[str(record["dataset"])],
            condition_index[_condition_from_record(record, "aggregate record")],
            candidate_index[str(record["candidate_slug"])],
            str(record["image_id"]),
        )

    return sorted(records, key=key)


def _publish_aggregate(
    *,
    destination: Path,
    episode_records: Sequence[Mapping[str, Any]],
    cell_records: Sequence[Mapping[str, Any]],
    candidate_summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    prepublish_guard: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if prepublish_guard is None:
        raise DiagnosticRunnerError(
            "formal D0 aggregate publication requires a live prepublish guard"
        )
    destination = _lexical_absolute(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite D0 aggregate: {destination}")
    fsync_directory(destination.parent)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        _write_jsonl(staging / "episode_records.jsonl", episode_records)
        _write_jsonl(staging / "cell_records.jsonl", cell_records)
        _write_json(staging / "candidate_summary.json", candidate_summary)
        _write_json(staging / "provenance.json", provenance)
        payloads = (
            "episode_records.jsonl",
            "cell_records.jsonl",
            "candidate_summary.json",
            "provenance.json",
        )
        files = {
            name: {
                "sha256": sha256_file(staging / name),
                "size_bytes": (staging / name).stat().st_size,
            }
            for name in payloads
        }
        manifest = {
            "schema_version": 1,
            "artifact_type": AGGREGATE_TYPE,
            "artifact_complete": True,
            "formal_d0_complete": True,
            "paper_test_result": False,
            "oracle_analysis": True,
            "method_label_accesses": 0,
            "selection_authorized": False,
            "stage2_authorized": False,
            "stage3_authorized": False,
            "dataset_count": len(FORMAL_DATASETS),
            "candidate_count": len(candidate_summary["candidates"]),
            "cell_count": len(cell_records),
            "diagnostic_record_count": len(episode_records),
            "files": files,
        }
        _write_json(staging / "artifact_manifest.json", manifest)
        complete = {
            "schema_version": 1,
            "artifact_type": AGGREGATE_TYPE,
            "complete": True,
            "formal_d0_complete": True,
            "paper_test_result": False,
            "selection_authorized": False,
            "stage2_authorized": False,
            "stage3_authorized": False,
            "manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
            "dataset_count": len(FORMAL_DATASETS),
            "candidate_count": len(candidate_summary["candidates"]),
            "cell_count": len(cell_records),
            "diagnostic_record_count": len(episode_records),
        }
        _write_json(staging / "COMPLETE.json", complete)
        managed = (*payloads, "artifact_manifest.json", "COMPLETE.json")
        (staging / "SHA256SUMS").write_text(
            "".join(
                f"{sha256_file(staging / name)}  {name}\n"
                for name in sorted(managed)
            ),
            encoding="utf-8",
        )
        staged_verification = verify_aggregate(
            staging, expected_canonical_destination=destination
        )
        publish_directory_noreplace(
            staging,
            destination,
            pre_rename_guard=prepublish_guard,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {**staged_verification, "path": str(destination)}


def aggregate_formal_shards(
    *,
    config_path: Path,
    shard_paths: Sequence[Path],
    destination: Path,
) -> dict[str, Any]:
    """Verify and immutably aggregate exactly three formal dataset shards."""

    config_path = _lexical_absolute(config_path)
    if len(shard_paths) != len(FORMAL_DATASETS):
        raise DiagnosticRunnerError("aggregate requires exactly three shard paths")
    resolved_shards = tuple(_lexical_absolute(path) for path in shard_paths)
    if len(set(resolved_shards)) != len(resolved_shards):
        raise DiagnosticRunnerError("aggregate shard paths contain duplicates")
    config = _load_config(config_path)
    destination = _validated_output_destination(
        config=config,
        destination=destination,
        role="aggregate",
    )
    if tuple(config.get("datasets", {})) != FORMAL_DATASETS:
        raise DiagnosticRunnerError("formal D0 dataset order drifted")
    conditions = _config_conditions(config)
    candidates = _config_candidates(config)
    config_sha = read_stable_regular_file(config_path).sha256
    archive_records_path = _archive_records_path_from_config(config)
    archive_verification = _verify_frozen_negative_archive(config)
    archive_records_snapshot = read_stable_regular_file(archive_records_path)
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for path in resolved_shards:
        dataset, records, receipt, binding = _formal_shard_payload(
            path=path,
            config=config,
            config_sha256=config_sha,
            archive_records_path=archive_records_path,
        )
        if dataset in by_dataset:
            raise DiagnosticRunnerError(f"duplicate formal dataset shard: {dataset}")
        expected_path = _validated_output_destination(
            config=config,
            destination=path,
            role="formal_shard",
            dataset=dataset,
        )
        if path != expected_path:
            raise DiagnosticRunnerError(
                f"aggregate shard is not at its canonical path: {path}"
            )
        by_dataset[dataset] = records
        receipts[dataset] = receipt
        bindings[dataset] = binding
    if set(by_dataset) != set(FORMAL_DATASETS):
        raise DiagnosticRunnerError("aggregate does not contain all three datasets")
    combined = [
        record
        for dataset in FORMAL_DATASETS
        for record in by_dataset[dataset]
    ]
    ordered_records = _canonical_record_order(
        combined,
        datasets=FORMAL_DATASETS,
        conditions=conditions,
        candidates=candidates,
    )
    expected_record_count = (
        len(FORMAL_DATASETS)
        * len(conditions)
        * len(candidates)
        * FORMAL_IMAGES_PER_CELL
    )
    if len(ordered_records) != expected_record_count:
        raise DiagnosticRunnerError(
            "formal aggregate diagnostic record count differs from 24,960"
        )
    cells = build_exact_cell_records(
        ordered_records,
        expected_datasets=FORMAL_DATASETS,
        expected_conditions=conditions,
        expected_candidates=candidates,
        expected_images_per_cell=FORMAL_IMAGES_PER_CELL,
    )
    expected_cell_count = len(FORMAL_DATASETS) * len(conditions) * len(candidates)
    if len(cells) != expected_cell_count:
        raise DiagnosticRunnerError("formal aggregate cell count differs from 390")
    candidate_summary = build_candidate_macro_summary(
        cell_records=cells,
        episode_records=ordered_records,
        expected_candidates=candidates,
    )
    for dataset in FORMAL_DATASETS:
        repeated_receipt = verify_formal_dataset_against_frozen_v2(
            records=by_dataset[dataset],
            dataset=dataset,
            config=config,
            archive_records_path=archive_records_path,
        )
        if repeated_receipt != receipts[dataset]:
            raise DiagnosticRunnerError(
                f"frozen v2 archive changed during aggregate: {dataset}"
            )
    provenance = {
        "schema_version": 1,
        "artifact_type": AGGREGATE_TYPE,
        "formal_d0_complete": True,
        "paper_test_result": False,
        "oracle_analysis": True,
        "source_train_derived": True,
        "method_label_accesses": 0,
        "test_images_opened": 0,
        "test_labels_opened": 0,
        "datasets": list(FORMAL_DATASETS),
        "conditions": list(conditions),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "images_per_cell": FORMAL_IMAGES_PER_CELL,
        "dataset_count": len(FORMAL_DATASETS),
        "candidate_count": len(candidates),
        "cell_count": len(cells),
        "diagnostic_record_count": len(ordered_records),
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "frozen_v2_stage1_records": {
            "path": str(archive_records_path),
            "sha256": archive_records_snapshot.sha256,
        },
        "frozen_v2_negative_archive": archive_verification,
        "per_dataset_frozen_v2_exact_comparison": {
            dataset: receipts[dataset] for dataset in FORMAL_DATASETS
        },
        "source_shards": {
            dataset: bindings[dataset] for dataset in FORMAL_DATASETS
        },
        "selection_authorized": False,
        "stage2_authorized": False,
        "stage3_authorized": False,
    }
    _ensure_output_parent(config=config, role="aggregate")

    def prepublish_guard() -> None:
        if read_stable_regular_file(config_path).sha256 != config_sha:
            raise DiagnosticRunnerError(
                "D0 aggregate config changed at publication boundary"
            )
        if read_stable_regular_file(archive_records_path).sha256 != provenance[
            "frozen_v2_stage1_records"
        ]["sha256"]:
            raise DiagnosticRunnerError(
                "frozen v2 archive changed at aggregate publication boundary"
            )
        if _verify_frozen_negative_archive(config) != archive_verification:
            raise DiagnosticRunnerError(
                "frozen v2 negative archive ledger changed at publication boundary"
            )
        repeated_by_dataset: dict[str, list[dict[str, Any]]] = {}
        repeated_receipts: dict[str, dict[str, Any]] = {}
        repeated_bindings: dict[str, dict[str, Any]] = {}
        for path in resolved_shards:
            repeated_dataset, repeated_records, repeated_receipt, repeated_binding = (
                _formal_shard_payload(
                    path=path,
                    config=config,
                    config_sha256=config_sha,
                    archive_records_path=archive_records_path,
                )
            )
            repeated_by_dataset[repeated_dataset] = repeated_records
            repeated_receipts[repeated_dataset] = repeated_receipt
            repeated_bindings[repeated_dataset] = repeated_binding
        if (
            repeated_by_dataset != by_dataset
            or repeated_receipts != receipts
            or repeated_bindings != bindings
        ):
            raise DiagnosticRunnerError(
                "formal D0 shards changed at aggregate publication boundary"
            )

    return _publish_aggregate(
        destination=destination,
        episode_records=ordered_records,
        cell_records=cells,
        candidate_summary=candidate_summary,
        provenance=provenance,
        prepublish_guard=prepublish_guard,
    )


def verify_aggregate(
    path: Path,
    *,
    expected_canonical_destination: Path | None = None,
) -> dict[str, Any]:
    root = _lexical_absolute(path)
    expected_names = {
        "episode_records.jsonl",
        "cell_records.jsonl",
        "candidate_summary.json",
        "provenance.json",
        "artifact_manifest.json",
        "COMPLETE.json",
        "SHA256SUMS",
    }
    try:
        directory = snapshot_regular_directory(root)
    except (OSError, ValueError) as exc:
        raise DiagnosticRunnerError(
            f"D0 aggregate is not a stable regular directory: {root}"
        ) from exc
    if set(directory.member_names) != expected_names:
        raise DiagnosticRunnerError("D0 aggregate member set is not exact")
    members = {value.path.name: value for value in directory.members}
    checksums: dict[str, str] = {}
    try:
        checksum_text = members["SHA256SUMS"].data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DiagnosticRunnerError("invalid aggregate SHA256SUMS encoding") from exc
    for line in checksum_text.splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in checksums or not _is_lower_sha256(digest):
            raise DiagnosticRunnerError("invalid aggregate SHA256SUMS")
        checksums[name] = digest
    if set(checksums) != expected_names - {"SHA256SUMS"}:
        raise DiagnosticRunnerError("aggregate SHA256SUMS member set differs")
    for name, digest in checksums.items():
        if members[name].sha256 != digest:
            raise DiagnosticRunnerError(f"D0 aggregate hash mismatch: {name}")
    manifest = _decode_json_bytes(
        members["artifact_manifest.json"].data, "aggregate artifact_manifest.json"
    )
    complete = _decode_json_bytes(
        members["COMPLETE.json"].data, "aggregate COMPLETE.json"
    )
    provenance = _decode_json_bytes(
        members["provenance.json"].data, "aggregate provenance.json"
    )
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "artifact_type",
            "artifact_complete",
            "formal_d0_complete",
            "paper_test_result",
            "oracle_analysis",
            "method_label_accesses",
            "selection_authorized",
            "stage2_authorized",
            "stage3_authorized",
            "dataset_count",
            "candidate_count",
            "cell_count",
            "diagnostic_record_count",
            "files",
        },
        "D0 aggregate manifest",
    )
    _require_exact_keys(
        complete,
        {
            "schema_version",
            "artifact_type",
            "complete",
            "formal_d0_complete",
            "paper_test_result",
            "selection_authorized",
            "stage2_authorized",
            "stage3_authorized",
            "manifest_sha256",
            "dataset_count",
            "candidate_count",
            "cell_count",
            "diagnostic_record_count",
        },
        "D0 aggregate completion",
    )
    _require_exact_keys(
        provenance,
        {
            "schema_version",
            "artifact_type",
            "formal_d0_complete",
            "paper_test_result",
            "oracle_analysis",
            "source_train_derived",
            "method_label_accesses",
            "test_images_opened",
            "test_labels_opened",
            "datasets",
            "conditions",
            "candidates",
            "images_per_cell",
            "dataset_count",
            "candidate_count",
            "cell_count",
            "diagnostic_record_count",
            "config_path",
            "config_sha256",
            "frozen_v2_stage1_records",
            "frozen_v2_negative_archive",
            "per_dataset_frozen_v2_exact_comparison",
            "source_shards",
            "selection_authorized",
            "stage2_authorized",
            "stage3_authorized",
        },
        "D0 aggregate provenance",
    )
    if (
        manifest.get("schema_version") != 1
        or complete.get("schema_version") != 1
        or provenance.get("schema_version") != 1
        or manifest.get("artifact_type") != AGGREGATE_TYPE
        or complete.get("artifact_type") != AGGREGATE_TYPE
        or provenance.get("artifact_type") != AGGREGATE_TYPE
        or manifest.get("formal_d0_complete") is not True
        or complete.get("formal_d0_complete") is not True
        or provenance.get("formal_d0_complete") is not True
        or manifest.get("paper_test_result") is not False
        or complete.get("paper_test_result") is not False
        or provenance.get("paper_test_result") is not False
        or manifest.get("oracle_analysis") is not True
        or provenance.get("oracle_analysis") is not True
        or provenance.get("source_train_derived") is not True
        or manifest.get("method_label_accesses") != 0
        or provenance.get("method_label_accesses") != 0
        or provenance.get("test_images_opened") != 0
        or provenance.get("test_labels_opened") != 0
        or provenance.get("selection_authorized") is not False
        or provenance.get("stage2_authorized") is not False
        or provenance.get("stage3_authorized") is not False
        or manifest.get("selection_authorized") is not False
        or manifest.get("stage2_authorized") is not False
        or manifest.get("stage3_authorized") is not False
        or complete.get("selection_authorized") is not False
        or complete.get("stage2_authorized") is not False
        or complete.get("stage3_authorized") is not False
        or manifest.get("artifact_complete") is not True
        or complete.get("complete") is not True
        or complete.get("manifest_sha256")
        != members["artifact_manifest.json"].sha256
    ):
        raise DiagnosticRunnerError("D0 aggregate manifest contract failed")
    files = manifest.get("files")
    payload_names = {
        "episode_records.jsonl",
        "cell_records.jsonl",
        "candidate_summary.json",
        "provenance.json",
    }
    if not isinstance(files, Mapping) or set(files) != payload_names:
        raise DiagnosticRunnerError("D0 aggregate payload set differs")
    for name, binding in files.items():
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"sha256", "size_bytes"}
            or binding.get("sha256") != members[name].sha256
            or binding.get("size_bytes") != members[name].size_bytes
        ):
            raise DiagnosticRunnerError(f"D0 aggregate binding failed: {name}")

    config_path_value = provenance.get("config_path")
    if not isinstance(config_path_value, str) or not config_path_value:
        raise DiagnosticRunnerError("D0 aggregate config path binding missing")
    config_path = _lexical_absolute(Path(config_path_value))
    try:
        config_snapshot = read_stable_regular_file(config_path)
    except (OSError, ValueError) as exc:
        raise DiagnosticRunnerError("D0 aggregate config binding failed") from exc
    if provenance.get("config_sha256") != config_snapshot.sha256:
        raise DiagnosticRunnerError("D0 aggregate config binding failed")
    config = _load_config(config_path)
    canonical_path = (
        root
        if expected_canonical_destination is None
        else _lexical_absolute(expected_canonical_destination)
    )
    if _validated_output_destination(
        config=config,
        destination=canonical_path,
        role="aggregate",
    ) != canonical_path:
        raise DiagnosticRunnerError("D0 aggregate path is not canonical")

    datasets = provenance.get("datasets")
    conditions = provenance.get("conditions")
    raw_candidates = provenance.get("candidates")
    if datasets != list(FORMAL_DATASETS) or not isinstance(conditions, list) or not isinstance(
        raw_candidates, list
    ):
        raise DiagnosticRunnerError("D0 aggregate formal dimensions invalid")
    candidates = tuple(
        _candidate_from_mapping(value, f"aggregate candidate[{index}]")
        for index, value in enumerate(raw_candidates)
    )
    if tuple(conditions) != tuple(
        f"{name}_S{severity}" for name, severity in frozen_v2.CONDITIONS
    ) or candidates != tuple(
        Candidate(value.optimizer, float(value.learning_rate))
        for value in frozen_v2.ALL_CANDIDATES
    ):
        raise DiagnosticRunnerError("D0 aggregate frozen dimensions drifted")
    episode_records = _decode_jsonl_bytes(
        members["episode_records.jsonl"].data, "aggregate episode records"
    )
    cell_records = _decode_jsonl_bytes(
        members["cell_records.jsonl"].data, "aggregate cell records"
    )
    expected_record_count = (
        len(FORMAL_DATASETS)
        * len(conditions)
        * len(candidates)
        * FORMAL_IMAGES_PER_CELL
    )
    expected_cell_count = len(FORMAL_DATASETS) * len(conditions) * len(candidates)
    if (
        len(episode_records) != expected_record_count
        or len(cell_records) != expected_cell_count
        or manifest.get("diagnostic_record_count") != expected_record_count
        or complete.get("diagnostic_record_count") != expected_record_count
        or provenance.get("diagnostic_record_count") != expected_record_count
        or manifest.get("cell_count") != expected_cell_count
        or complete.get("cell_count") != expected_cell_count
        or provenance.get("cell_count") != expected_cell_count
        or manifest.get("dataset_count") != len(FORMAL_DATASETS)
        or complete.get("dataset_count") != len(FORMAL_DATASETS)
        or provenance.get("dataset_count") != len(FORMAL_DATASETS)
        or manifest.get("candidate_count") != len(candidates)
        or complete.get("candidate_count") != len(candidates)
        or provenance.get("candidate_count") != len(candidates)
        or provenance.get("images_per_cell") != FORMAL_IMAGES_PER_CELL
    ):
        raise DiagnosticRunnerError("D0 aggregate 390/24,960 count contract failed")
    for dataset in FORMAL_DATASETS:
        validate_d0_records(
            [record for record in episode_records if record.get("dataset") == dataset],
            config=config,
            dataset=dataset,
            conditions=tuple(conditions),
            candidates=candidates,
            images_per_condition=FORMAL_IMAGES_PER_CELL,
        )
    canonical_episode_records = _canonical_record_order(
        episode_records,
        datasets=FORMAL_DATASETS,
        conditions=tuple(conditions),
        candidates=candidates,
    )
    if canonical_episode_records != episode_records:
        raise DiagnosticRunnerError("D0 aggregate episode order is not canonical")
    reconstructed_cells = build_exact_cell_records(
        episode_records,
        expected_datasets=FORMAL_DATASETS,
        expected_conditions=tuple(conditions),
        expected_candidates=candidates,
        expected_images_per_cell=FORMAL_IMAGES_PER_CELL,
    )
    if reconstructed_cells != cell_records:
        raise DiagnosticRunnerError("D0 aggregate cell reconstruction differs")
    expected_summary = build_candidate_macro_summary(
        cell_records=cell_records,
        episode_records=episode_records,
        expected_candidates=candidates,
    )
    if _decode_json_bytes(
        members["candidate_summary.json"].data, "aggregate candidate summary"
    ) != expected_summary:
        raise DiagnosticRunnerError("D0 aggregate candidate macro summary differs")
    archive_binding = provenance.get("frozen_v2_stage1_records")
    if not isinstance(archive_binding, Mapping):
        raise DiagnosticRunnerError("D0 aggregate frozen-v2 binding missing")
    archive_path = _lexical_absolute(Path(str(archive_binding.get("path", ""))))
    try:
        archive_snapshot = read_stable_regular_file(archive_path)
    except (OSError, ValueError) as exc:
        raise DiagnosticRunnerError("D0 aggregate frozen-v2 binding failed") from exc
    if archive_binding.get("sha256") != archive_snapshot.sha256:
        raise DiagnosticRunnerError("D0 aggregate frozen-v2 binding failed")
    if provenance.get("frozen_v2_negative_archive") != (
        _verify_frozen_negative_archive(config)
    ):
        raise DiagnosticRunnerError(
            "D0 aggregate frozen negative-archive ledger binding failed"
        )
    per_dataset_receipts = provenance.get("per_dataset_frozen_v2_exact_comparison")
    if not isinstance(per_dataset_receipts, Mapping) or set(per_dataset_receipts) != set(
        FORMAL_DATASETS
    ):
        raise DiagnosticRunnerError("D0 aggregate per-dataset receipts invalid")
    source_shards = provenance.get("source_shards")
    if not isinstance(source_shards, Mapping) or set(source_shards) != set(
        FORMAL_DATASETS
    ):
        raise DiagnosticRunnerError("D0 aggregate source-shard bindings invalid")
    per_shard_record_count = len(conditions) * len(candidates) * FORMAL_IMAGES_PER_CELL
    for dataset in FORMAL_DATASETS:
        binding = source_shards[dataset]
        if (
            not isinstance(binding, Mapping)
            or set(binding)
            != {
                "path",
                "artifact_manifest_sha256",
                "complete_sha256",
                "episode_records_sha256",
                "record_count",
                "dataset",
            }
            or binding.get("dataset") != dataset
            or binding.get("record_count") != per_shard_record_count
            or any(
                not _is_lower_sha256(binding.get(field))
                for field in (
                    "artifact_manifest_sha256",
                    "complete_sha256",
                    "episode_records_sha256",
                )
            )
        ):
            raise DiagnosticRunnerError(
                f"D0 aggregate source-shard binding invalid: {dataset}"
            )
    for dataset in FORMAL_DATASETS:
        dataset_records = [
            record for record in episode_records if record.get("dataset") == dataset
        ]
        receipt = verify_formal_dataset_against_frozen_v2(
            records=dataset_records,
            dataset=dataset,
            config=config,
            archive_records_path=archive_path,
        )
        if per_dataset_receipts.get(dataset) != receipt:
            raise DiagnosticRunnerError(
                f"D0 aggregate frozen-v2 receipt differs: {dataset}"
            )
        live_path = _lexical_absolute(Path(str(source_shards[dataset].get("path", ""))))
        expected_live_path = _validated_output_destination(
            config=config,
            destination=live_path,
            role="formal_shard",
            dataset=dataset,
        )
        if live_path != expected_live_path:
            raise DiagnosticRunnerError(
                f"D0 aggregate source-shard path is not canonical: {dataset}"
            )
        (
            live_dataset,
            live_records,
            live_receipt,
            live_binding,
        ) = _formal_shard_payload(
            path=live_path,
            config=config,
            config_sha256=config_snapshot.sha256,
            archive_records_path=archive_path,
        )
        if (
            live_dataset != dataset
            or live_records != dataset_records
            or live_receipt != receipt
            or live_binding != source_shards[dataset]
        ):
            raise DiagnosticRunnerError(
                f"D0 aggregate live source-shard lineage differs: {dataset}"
            )
    try:
        final_directory = snapshot_regular_directory(root)
    except (OSError, ValueError) as exc:
        raise DiagnosticRunnerError(
            "D0 aggregate changed or became unsafe during verification"
        ) from exc
    if final_directory != directory:
        raise DiagnosticRunnerError("D0 aggregate changed during live verification")
    return {
        "status": "verified",
        "path": str(root),
        "formal_d0_complete": True,
        "paper_test_result": False,
        "dataset_count": len(FORMAL_DATASETS),
        "cell_count": len(cell_records),
        "diagnostic_record_count": len(episode_records),
        "candidate_count": len(candidates),
    }


def _parse_candidate_filters(values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(values)
    if len(set(result)) != len(result):
        raise DiagnosticRunnerError("candidate filters contain duplicates")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="run one dataset shard; omit filters and keep 64 images for formal D0",
        description=(
            "Run one D0 dataset shard. A formal shard must omit --condition and "
            "--candidate and retain --max-images 64; filtered runs are smoke only."
        ),
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--dataset", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--condition", action="append", default=[])
    run.add_argument("--candidate", action="append", default=[])
    run.add_argument("--max-images", type=int, default=64)
    run.add_argument("--equivalence-receipt", type=Path)
    verify = subparsers.add_parser("verify", help="verify an immutable shard")
    verify.add_argument("--path", type=Path, required=True)
    receipt = subparsers.add_parser(
        "verify-equivalence",
        help="verify a three-fresh-process aggregate equivalence receipt",
        description=(
            "Verify the aggregate receipt produced by exactly three supported-runner "
            "fresh processes; a single-process receipt is not accepted."
        ),
    )
    receipt.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    receipt.add_argument("--dataset", required=True)
    receipt.add_argument("--path", type=Path, required=True)
    equivalence = subparsers.add_parser(
        "equivalence-smoke",
        help=(
            "produce one internal-process equivalence receipt; this alone cannot "
            "authorize a formal shard"
        ),
        description=(
            "Internal child command used by equivalence-repro. Its single-process "
            "receipt cannot authorize a formal D0 shard on its own."
        ),
    )
    equivalence.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    equivalence.add_argument("--dataset", required=True)
    equivalence.add_argument("--device", default="cuda:0")
    equivalence.add_argument("--condition", default="clean_S0")
    equivalence.add_argument("--image-index", type=int, default=0)
    equivalence.add_argument("--process-id", required=True)
    equivalence.add_argument("--parent-run-nonce", required=True)
    equivalence.add_argument("--child-launch-nonce", required=True)
    equivalence.add_argument("--output", type=Path, required=True)
    equivalence_repro = subparsers.add_parser(
        "equivalence-repro",
        help="launch exactly three fresh equivalence workers and aggregate them",
        description=(
            "Launch exactly three supported-runner fresh GPU processes, bind their "
            "parent transcripts, and publish one aggregate equivalence receipt."
        ),
    )
    equivalence_repro.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    equivalence_repro.add_argument("--dataset", required=True)
    equivalence_repro.add_argument("--device", default="cuda:0")
    equivalence_repro.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser(
        "aggregate",
        help="verify and combine exactly three formal dataset shards",
    )
    aggregate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    aggregate.add_argument("--shard", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    verify_aggregate_parser = subparsers.add_parser(
        "verify-aggregate", help="verify an immutable three-dataset aggregate"
    )
    verify_aggregate_parser.add_argument("--path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "verify":
            result = verify_shard(args.path)
        elif args.command == "verify-aggregate":
            result = verify_aggregate(args.path)
        elif args.command == "aggregate":
            result = aggregate_formal_shards(
                config_path=args.config,
                shard_paths=tuple(args.shard),
                destination=args.output,
            )
        elif args.command == "equivalence-smoke":
            result = run_equivalence_smoke(
                config_path=args.config,
                dataset=args.dataset,
                device_name=args.device,
                condition_key=args.condition,
                image_index=args.image_index,
                process_id=args.process_id,
                parent_run_nonce=args.parent_run_nonce,
                child_launch_nonce=args.child_launch_nonce,
                output=args.output,
            )
        elif args.command == "equivalence-repro":
            result = run_equivalence_repro(
                config_path=args.config,
                dataset=args.dataset,
                device_name=args.device,
                output=args.output,
            )
        elif args.command == "verify-equivalence":
            config_path = _lexical_absolute(args.config)
            config = _load_config(config_path)
            candidates = _config_candidates(config)
            contract = frozen_v2.load_contract()
            seal, _caches = frozen_v2.capture_runtime_seal(contract)
            digest = _verify_equivalence_receipt(
                args.path,
                dataset=args.dataset,
                candidates=candidates,
                config_sha256=read_stable_regular_file(config_path).sha256,
                global_runtime_seal_sha256=seal.global_runtime_seal_sha256,
                diagnostic_source_code_sha256=_source_hashes(
                    config_path
                ),
                expected_condition=str(
                    config["equivalence"]["samples"][args.dataset]["condition"]
                ),
                expected_image_index=int(
                    config["equivalence"]["samples"][args.dataset]["image_index"]
                ),
                expected_image_id=str(
                    config["equivalence"]["samples"][args.dataset]["image_id"]
                ),
            )
            result = {"status": "verified", "receipt_sha256": digest}
        else:
            result = run_dataset(
                config_path=args.config,
                dataset=args.dataset,
                device_name=args.device,
                destination=args.output,
                condition_filters=tuple(args.condition),
                candidate_filters=_parse_candidate_filters(args.candidate),
                max_images=args.max_images,
                equivalence_receipt=args.equivalence_receipt,
            )
    except (DiagnosticRunnerError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=JSON_SEPARATORS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_TYPE",
    "ARTIFACT_TYPE",
    "Candidate",
    "DiagnosticRunnerError",
    "aggregate_formal_shards",
    "build_candidate_macro_summary",
    "build_exact_cell_records",
    "build_parser",
    "main",
    "run_dataset",
    "run_equivalence_repro",
    "run_equivalence_smoke",
    "sha256_file",
    "summarize_records",
    "verify_aggregate",
    "verify_formal_dataset_against_frozen_v2",
    "verify_shard",
]
