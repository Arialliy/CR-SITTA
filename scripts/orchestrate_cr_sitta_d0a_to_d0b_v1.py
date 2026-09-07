#!/usr/bin/env python3
"""Fail-closed D0-A completion -> D0-B orchestration for CR-SITTA.

The default command is read-only ``status``.  ``dry-run`` performs every
available D0-A completion check but writes nothing and starts no subprocess.
Only the explicit ``execute`` command may wait, publish immutable completion
anchors, export weights-only checkpoints, or launch D0-B.

The orchestrator is intentionally limited to the three fixed D0-A result
directories below.  It never opens a validation or test split/payload.  D0-B
GPU phases run serially on physical GPU 2 under an exclusive filesystem lock;
all other CUDA devices are hidden.  The terminal action is D0-B verification
and publication of ``PIPELINE_COMPLETE.json``.  D1 and formal test are never
launched here, including when D0-B is scientifically eligible.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import stat
from typing import Any, Final
import uuid

import yaml

# A script launched as ``python scripts/<name>.py`` receives ``scripts/`` as
# sys.path[0].  Add the fixed repository root before importing project modules.
_BOOTSTRAP_REPOSITORY = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPOSITORY))

from tta.d0_secure_io import (
    ensure_directory_chain_nofollow,
    publish_file_noreplace,
    read_stable_regular_file,
)


REPOSITORY: Final = Path(__file__).resolve().parents[1]
PROTOCOL_ID: Final = "cr-sitta-d0a-to-d0b-orchestrator-v1"
D0A_PROTOCOL_ID: Final = "cr-sitta-d0a-supervised-lfhf-train-1000e-v2"
D0B_PROTOCOL_ID: Final = "cr-sitta-d0b-checkpoint-rebound-gradient-gate-v1"
DATASETS: Final = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
D0A_RESULT_RELATIVE: Final = Path(
    "results/cr_sitta/d0a_supervised_lfhf_train_v2"
)
ORCHESTRATOR_RESULT_RELATIVE: Final = Path(
    "results/cr_sitta/d0a_to_d0b_orchestrator_v1"
)
D0B_RESULT_RELATIVE: Final = Path(
    "results/cr_sitta/d0b_checkpoint_rebound_gradient_gate_v1"
)
D0A_CONFIG_RELATIVE: Final = Path("configs/cr_sitta_d0a_train_v2.yaml")
D0A_FREEZE_RELATIVE: Final = D0A_RESULT_RELATIVE / "FULL_TRAIN_FREEZE.json"
EXPORTER_RELATIVE: Final = Path("export_cr_sitta_d0a_safe_checkpoint.py")
D0B_RUNNER_RELATIVE: Final = Path("run_cr_sitta_d0b_gradient_gate_v1.py")
D0B_CONFIG_RELATIVE: Final = Path("configs/cr_sitta_d0b_gradient_gate_v1.yaml")

EXPECTED_D0A_CONFIG_SHA256: Final = (
    "6118f3c54d1918b9716e8d6124003054da9d49b0dcdf2a79d8aa920cd2dd1bbc"
)
EXPECTED_D0A_FREEZE_SHA256: Final = (
    "fa164bcf33185656c1eeb5bbcc0cbcd752c59dbeab37b0a82fe64cfaa6446bca"
)
EXPECTED_D0A_RUNNER_SHA256: Final = (
    "d88de89452db70b35dc59711100e759d17eedcc25470cfa4259e8e702428b650"
)
EXPECTED_EXPORTER_SHA256: Final = (
    "abc8be1f252cf353143089122ae8056aaff32f2b29ff2c22de64407578b23a00"
)
EXPECTED_D0B_RUNNER_SHA256: Final = (
    "7c5844fe4fd76eed548a24e76bd35c0e0a1d8919e32d98f5d00454fbc9cd98f5"
)
EXPECTED_D0B_CONFIG_SHA256: Final = (
    "1fe5cb076da8451487282566e1721eecb759b7a5484253b0c60881978f0e8605"
)
EXPECTED_D0B_GATE_SHA256: Final = (
    "5deca272bf1c675c4bfec0ea0e7265e4ecb5076a32926c100e4ec3fbcf9de2b9"
)
EXPECTED_EPOCHS: Final = 1000
EXPECTED_STATE_DICT_KEYS: Final = 505
PHYSICAL_GPU_INDEX: Final = 2
GPU_POLL_SECONDS: Final = 30.0
TRAIN_POLL_SECONDS: Final = 30.0
ZERO_ACCESS_FIELDS: Final = (
    "test_split_reads",
    "test_image_opens",
    "test_mask_opens",
    "validation_split_reads",
    "validation_image_opens",
    "validation_mask_opens",
)
SHA256_HEX: Final = frozenset("0123456789abcdef")


class OrchestrationError(RuntimeError):
    """A fail-closed D0-A/D0-B handoff violation."""


@dataclass(frozen=True, slots=True)
class DatasetPaths:
    dataset: str
    output_dir: Path
    run_contract: Path
    metrics: Path
    summary: Path
    final_checkpoint: Path
    last_checkpoint: Path
    archived_train_split: Path
    safe_checkpoint: Path
    safe_export_receipt: Path
    anchor: Path


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


CommandRunner = Callable[[Sequence[str], Mapping[str, str]], CommandResult]
Sleep = Callable[[float], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OrchestrationError("receipt is not canonical-JSON safe") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return read_stable_regular_file(path).sha256
    except (OSError, ValueError, RuntimeError) as exc:
        raise OrchestrationError(f"cannot hash stable regular file: {path}") from exc


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in SHA256_HEX for character in value)
    ):
        raise OrchestrationError(f"{label} must be a lowercase SHA-256")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OrchestrationError(f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OrchestrationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise OrchestrationError(f"{label} must be >= {minimum}")
    return value


def _finite_number(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OrchestrationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise OrchestrationError(f"{label} must be finite")
    return result


def _require_zero_access(mapping: Mapping[str, Any], label: str) -> None:
    for field in ZERO_ACCESS_FIELDS:
        value = mapping.get(field)
        if type(value) is not int or value != 0:
            raise OrchestrationError(f"{label}.{field} must be integer zero")


def _repository_path(repository: Path, relative: Path | str, label: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise OrchestrationError(f"{label} must be repository-relative")
    candidate = repository / raw
    if not candidate.absolute().is_relative_to(repository.absolute()):
        raise OrchestrationError(f"{label} escapes repository")
    return candidate


def _display_path(path: Path, repository: Path) -> str:
    absolute = path.absolute()
    try:
        return str(absolute.relative_to(repository.absolute()))
    except ValueError:
        return str(absolute)


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        snapshot = read_stable_regular_file(path)
        value = json.loads(snapshot.data)
    except (OSError, ValueError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrchestrationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise OrchestrationError(f"{label} root must be an object")
    return value, snapshot.sha256


def _load_yaml(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        snapshot = read_stable_regular_file(path)
        value = yaml.safe_load(snapshot.data)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        raise OrchestrationError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise OrchestrationError(f"{label} root must be a mapping")
    return value, snapshot.sha256


def _dataset_paths(repository: Path, dataset: str) -> DatasetPaths:
    if dataset not in DATASETS:
        raise OrchestrationError(f"unexpected dataset: {dataset}")
    output = _repository_path(repository, D0A_RESULT_RELATIVE / dataset, "D0-A output")
    anchor = _repository_path(
        repository,
        ORCHESTRATOR_RESULT_RELATIVE / "anchors" / dataset / "D0A_COMPLETION_ANCHOR.json",
        "D0-A anchor",
    )
    return DatasetPaths(
        dataset=dataset,
        output_dir=output,
        run_contract=output / "run_contract.json",
        metrics=output / "train_metrics.jsonl",
        summary=output / "summary.json",
        final_checkpoint=output / "epoch_1000_train_only.pth.tar",
        last_checkpoint=output / "last.pth.tar",
        archived_train_split=output / "splits" / "train.txt",
        safe_checkpoint=output / "epoch_1000_train_only_safe.pth.tar",
        safe_export_receipt=output / "SAFE_EXPORT.json",
        anchor=anchor,
    )


def _cmdline_output_dirs(argv: Sequence[str], cwd: Path) -> tuple[Path, ...]:
    result: list[Path] = []
    for index, argument in enumerate(argv):
        raw: str | None = None
        if argument == "--output-dir" and index + 1 < len(argv):
            raw = argv[index + 1]
        elif argument.startswith("--output-dir="):
            raw = argument.split("=", 1)[1]
        if raw:
            path = Path(raw)
            result.append((path if path.is_absolute() else cwd / path).absolute())
    return tuple(result)


def trainer_processes_for_output(
    output_dir: Path, *, proc_root: Path = Path("/proc")
) -> tuple[int, ...]:
    """Return live D0-A trainer PIDs bound to exactly ``output_dir``."""

    expected = output_dir.absolute()
    found: list[int] = []
    try:
        processes = tuple(proc_root.iterdir())
    except OSError:
        return ()
    for process in processes:
        if not process.name.isdecimal():
            continue
        try:
            raw = (process / "cmdline").read_bytes()
            argv = tuple(
                item.decode("utf-8") for item in raw.split(b"\0") if item
            )
            cwd = Path(os.readlink(process / "cwd"))
        except (OSError, UnicodeError):
            continue
        if not any(Path(item).name == "train_cr_sitta_d0a.py" for item in argv):
            continue
        if expected in _cmdline_output_dirs(argv, cwd):
            found.append(int(process.name))
    return tuple(sorted(found))


def _required_d0a_files(paths: DatasetPaths) -> tuple[Path, ...]:
    return (
        paths.run_contract,
        paths.metrics,
        paths.summary,
        paths.final_checkpoint,
        paths.last_checkpoint,
        paths.archived_train_split,
    )


def _last_metrics_progress(path: Path) -> tuple[int | None, int | None]:
    try:
        snapshot = read_stable_regular_file(path)
        lines = [line for line in snapshot.data.splitlines() if line.strip()]
        value = json.loads(lines[-1]) if lines else None
    except (OSError, ValueError, RuntimeError, UnicodeError, json.JSONDecodeError):
        return None, None
    if not isinstance(value, Mapping):
        return None, None
    epoch = value.get("epoch")
    step = value.get("ending_optimizer_step")
    return (
        epoch if isinstance(epoch, int) and not isinstance(epoch, bool) else None,
        step if isinstance(step, int) and not isinstance(step, bool) else None,
    )


def inspect_dataset_state(
    repository: Path,
    dataset: str,
    *,
    proc_root: Path = Path("/proc"),
    now: float | None = None,
    stale_grace_seconds: float = 300.0,
) -> dict[str, Any]:
    paths = _dataset_paths(repository, dataset)
    pids = trainer_processes_for_output(paths.output_dir, proc_root=proc_root)
    required = _required_d0a_files(paths)
    present = {path.name: path.is_file() and not path.is_symlink() for path in required}
    all_present = all(present.values())
    any_present = any(present.values()) or paths.output_dir.exists()
    epoch, global_step = _last_metrics_progress(paths.metrics)
    if pids:
        state = "training"
    elif all_present:
        state = "ready_for_deep_validation"
    elif not any_present:
        state = "waiting_not_started"
    else:
        timestamps: list[float] = []
        try:
            output_stat = paths.output_dir.stat(follow_symlinks=False)
            timestamps.append(output_stat.st_mtime)
        except OSError:
            pass
        for path in required:
            try:
                timestamps.append(path.stat(follow_symlinks=False).st_mtime)
            except OSError:
                pass
        age = (time.time() if now is None else now) - max(timestamps or [0.0])
        state = "waiting_transient" if age < stale_grace_seconds else "failed_or_stale"
    return {
        "dataset": dataset,
        "state": state,
        "trainer_pids": list(pids),
        "epoch": epoch,
        "global_optimizer_step": global_step,
        "required_files": present,
        "anchor_exists": paths.anchor.is_file() and not paths.anchor.is_symlink(),
        "safe_checkpoint_exists": (
            paths.safe_checkpoint.is_file() and not paths.safe_checkpoint.is_symlink()
        ),
        "safe_export_receipt_exists": (
            paths.safe_export_receipt.is_file()
            and not paths.safe_export_receipt.is_symlink()
        ),
    }


def build_status(
    repository: Path = REPOSITORY,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    datasets = {
        dataset: inspect_dataset_state(repository, dataset, proc_root=proc_root)
        for dataset in DATASETS
    }
    root = _repository_path(repository, ORCHESTRATOR_RESULT_RELATIVE, "result root")
    d0b = _repository_path(repository, D0B_RESULT_RELATIVE, "D0-B result root")
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0a_to_d0b_read_only_status_v1",
        "protocol_id": PROTOCOL_ID,
        "mode": "status_read_only",
        "writes_performed": 0,
        "datasets": datasets,
        "anchor_set_exists": (root / "D0A_COMPLETION_ANCHOR_SET.json").is_file(),
        "handoff_freeze_exists": (root / "HANDOFF_EXECUTION_FREEZE.json").is_file(),
        "pipeline_complete_exists": (root / "PIPELINE_COMPLETE.json").is_file(),
        "d0b_outputs": {
            "pre_run_freeze_exists": (d0b / "PRE_RUN_FREEZE.json").is_file(),
            "aggregate_exists": (d0b / "aggregate_phase" / "R0").is_dir(),
        },
        "d1_launched": False,
        "formal_test_allowed": False,
    }


def _validate_protocol_constants(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = _repository_path(repository, D0A_CONFIG_RELATIVE, "D0-A config")
    config, config_sha = _load_yaml(config_path, "D0-A config")
    if config_sha != EXPECTED_D0A_CONFIG_SHA256:
        raise OrchestrationError("D0-A config hash drift")
    freeze_path = _repository_path(repository, D0A_FREEZE_RELATIVE, "D0-A freeze")
    freeze, freeze_sha = _load_json(freeze_path, "D0-A freeze")
    if freeze_sha != EXPECTED_D0A_FREEZE_SHA256:
        raise OrchestrationError("D0-A full-train freeze hash drift")
    if config.get("protocol_id") != D0A_PROTOCOL_ID or freeze.get("protocol_id") != D0A_PROTOCOL_ID:
        raise OrchestrationError("D0-A protocol identity drift")
    scope = _mapping(config.get("scope"), "D0-A scope")
    training = _mapping(config.get("training"), "D0-A training")
    if (
        scope.get("no_validation_split") is not True
        or scope.get("use_validation_payload") is not False
        or scope.get("use_test_payload") is not False
        or training.get("official_train_only") is not True
        or training.get("epochs") != EXPECTED_EPOCHS
        or training.get("test_evaluation_during_training") is not False
        or training.get("validation_evaluation_during_training") is not False
    ):
        raise OrchestrationError("D0-A config is not fixed train-only 1000-epoch scope")
    runtime = _mapping(freeze.get("runtime_sha256"), "D0-A frozen runtime")
    if runtime.get(str(D0A_CONFIG_RELATIVE)) != EXPECTED_D0A_CONFIG_SHA256:
        raise OrchestrationError("D0-A freeze/config binding drift")
    if runtime.get("train_cr_sitta_d0a.py") != EXPECTED_D0A_RUNNER_SHA256:
        raise OrchestrationError("D0-A freeze/runner binding drift")
    if freeze.get("formal_test_authorized") is not False or freeze.get("tta_authorized") is not False:
        raise OrchestrationError("D0-A freeze authorization widened")
    return config, freeze


def _validate_d0b_static_contract(repository: Path) -> dict[str, Any]:
    """Pin the complete D0-B YAML bytes and prove Pilot64 is train-internal."""

    runner = repository / D0B_RUNNER_RELATIVE
    config_path = repository / D0B_CONFIG_RELATIVE
    exporter = repository / EXPORTER_RELATIVE
    gate = repository / "analysis/cr_sitta_d0b_gate.py"
    if sha256_file(runner) != EXPECTED_D0B_RUNNER_SHA256:
        raise OrchestrationError("D0-B runner hash differs from the reviewed implementation")
    if sha256_file(config_path) != EXPECTED_D0B_CONFIG_SHA256:
        raise OrchestrationError("D0-B config hash differs from the reviewed full YAML")
    if sha256_file(exporter) != EXPECTED_EXPORTER_SHA256:
        raise OrchestrationError("safe exporter hash differs from the reviewed implementation")
    if sha256_file(gate) != EXPECTED_D0B_GATE_SHA256:
        raise OrchestrationError("D0-B authorization gate differs from the reviewed implementation")
    raw, _digest = _load_yaml(config_path, "D0-B config")
    if raw.get("protocol_id") != D0B_PROTOCOL_ID:
        raise OrchestrationError("D0-B protocol identity drift")
    scope = _mapping(raw.get("scope"), "D0-B scope")
    if (
        scope.get("data_role") != "train"
        or scope.get("no_validation_split") is not True
        or scope.get("use_validation_payload") is not False
        or scope.get("use_test_payload") is not False
        or type(scope.get("validation_access_count")) is not int
        or scope.get("validation_access_count") != 0
        or type(scope.get("test_access_count")) is not int
        or scope.get("test_access_count") != 0
    ):
        raise OrchestrationError("D0-B scope is not train-only")
    datasets = _mapping(raw.get("datasets"), "D0-B datasets")
    if tuple(datasets) != DATASETS:
        raise OrchestrationError("D0-B dataset roster/order drift")
    subset_receipt: dict[str, Any] = {}
    for dataset in DATASETS:
        record = _mapping(datasets.get(dataset), f"D0-B {dataset}")
        train_path = _repository_path(repository, record.get("train_split"), "train split")
        pilot_path = _repository_path(repository, record.get("pilot_ids"), "Pilot64 IDs")
        train_snapshot = read_stable_regular_file(train_path)
        pilot_snapshot = read_stable_regular_file(pilot_path)
        try:
            train_lines = train_snapshot.data.decode("utf-8").splitlines()
            pilot_lines = pilot_snapshot.data.decode("utf-8").splitlines()
        except UnicodeError as exc:
            raise OrchestrationError(f"{dataset} train/Pilot IDs are not UTF-8") from exc
        if any(not value.strip() for value in train_lines + pilot_lines):
            raise OrchestrationError(f"{dataset} train/Pilot ID file contains a blank row")
        train_ids = tuple(value.strip() for value in train_lines)
        pilot_ids = tuple(value.strip() for value in pilot_lines)
        if len(train_ids) != len(set(train_ids)):
            raise OrchestrationError(f"{dataset} official train IDs contain duplicates")
        if len(pilot_ids) != 64 or len(pilot_ids) != len(set(pilot_ids)):
            raise OrchestrationError(f"{dataset} Pilot64 IDs are not 64 unique values")
        if not set(pilot_ids).issubset(train_ids):
            raise OrchestrationError(f"{dataset} Pilot64 is not a subset of official train")
        if train_snapshot.sha256 != record.get("train_split_sha256") or pilot_snapshot.sha256 != record.get("pilot_ids_file_sha256"):
            raise OrchestrationError(f"{dataset} train/Pilot ID hash drift")
        subset_receipt[dataset] = {
            "pilot_count": 64,
            "train_count": len(train_ids),
            "pilot_unique": True,
            "pilot_subset_of_official_train": True,
            "train_split_sha256": train_snapshot.sha256,
            "pilot_ids_sha256": pilot_snapshot.sha256,
        }
    return subset_receipt


def _validate_metrics(
    path: Path,
    *,
    expected_batches: int,
    warm_epochs: int,
    learning_rate: float,
) -> tuple[list[dict[str, Any]], str]:
    try:
        snapshot = read_stable_regular_file(path)
        text = snapshot.data.decode("utf-8")
    except (OSError, ValueError, RuntimeError, UnicodeError) as exc:
        raise OrchestrationError(f"cannot stably read D0-A metrics: {path}") from exc
    if not text.endswith("\n"):
        raise OrchestrationError("D0-A metrics lacks a complete trailing newline")
    raw_lines = text.splitlines()
    if len(raw_lines) != EXPECTED_EPOCHS or any(not line.strip() for line in raw_lines):
        raise OrchestrationError("D0-A metrics must contain exactly 1000 nonempty rows")
    rows: list[dict[str, Any]] = []
    expected_start = 0
    for human_epoch, line in enumerate(raw_lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OrchestrationError(f"invalid metrics JSON at epoch {human_epoch}") from exc
        if not isinstance(row, dict):
            raise OrchestrationError(f"metrics row {human_epoch} is not an object")
        if row.get("epoch") != human_epoch or row.get("epoch_index") != human_epoch - 1:
            raise OrchestrationError("D0-A metric epoch ordering is not exact 1..1000")
        if row.get("batches") != expected_batches:
            raise OrchestrationError(f"D0-A batch count drift at epoch {human_epoch}")
        if row.get("starting_optimizer_step") != expected_start:
            raise OrchestrationError(f"D0-A optimizer-step discontinuity at epoch {human_epoch}")
        expected_end = expected_start + expected_batches
        if row.get("ending_optimizer_step") != expected_end:
            raise OrchestrationError(f"D0-A ending optimizer step drift at epoch {human_epoch}")
        expected_probe = {
            "lf_mask": sum(1 for value in range(expected_start, expected_end) if value % 2 == 0),
            "hf_noise": sum(1 for value in range(expected_start, expected_end) if value % 2 == 1),
        }
        if row.get("probe_counts") != expected_probe:
            raise OrchestrationError(f"D0-A LF/HF schedule drift at epoch {human_epoch}")
        if row.get("probe_tensor_sha256s") != []:
            raise OrchestrationError("full D0-A metrics unexpectedly contain smoke payload hashes")
        if row.get("warm_flag") is not ((human_epoch - 1) < warm_epochs):
            raise OrchestrationError(f"D0-A warm schedule drift at epoch {human_epoch}")
        if _finite_number(row.get("learning_rate"), f"metrics[{human_epoch}].learning_rate") != learning_rate:
            raise OrchestrationError(f"D0-A learning rate drift at epoch {human_epoch}")
        for field in (
            "mean_clean_loss",
            "mean_degraded_loss",
            "mean_combined_loss",
            "last_gradient_l2",
            "duration_seconds",
        ):
            _finite_number(row.get(field), f"metrics[{human_epoch}].{field}", nonnegative=True)
        rows.append(row)
        expected_start = expected_end
    return rows, snapshot.sha256


def _deep_equal(left: Any, right: Any) -> bool:
    """Exact recursive equality for trusted-local checkpoint payloads."""

    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - repository runtime always has both
        raise OrchestrationError("checkpoint comparison requires NumPy and PyTorch") from exc
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and tuple(left.keys()) == tuple(right.keys())
            and all(_deep_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_deep_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _artifact(path: Path, repository: Path) -> dict[str, Any]:
    snapshot = read_stable_regular_file(path)
    return {
        "path": _display_path(path, repository),
        "sha256": snapshot.sha256,
        "bytes": snapshot.size_bytes,
    }


def _runtime_binding_path(repository: Path, declared: str) -> Path:
    path = Path(declared).expanduser()
    return path if path.is_absolute() else repository / path


def _verify_runtime_bindings(repository: Path, bindings: Mapping[str, Any], label: str) -> None:
    if not bindings:
        raise OrchestrationError(f"{label} is empty")
    for declared, expected in bindings.items():
        if not isinstance(declared, str):
            raise OrchestrationError(f"{label} contains a non-string path")
        expected_sha = _require_sha256(expected, f"{label}.{declared}")
        if sha256_file(_runtime_binding_path(repository, declared)) != expected_sha:
            raise OrchestrationError(f"{label} hash drift: {declared}")


def _validate_runtime_rosters(
    repository: Path,
    run_runtime: Mapping[str, Any],
    freeze_runtime: Mapping[str, Any],
) -> None:
    expected_freeze = {
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
    }
    if set(freeze_runtime) != expected_freeze:
        raise OrchestrationError("D0-A freeze runtime path roster differs")
    expected_run_relative = expected_freeze - {"tta/deteriorations/__init__.py"}
    run_keys = set(run_runtime)
    if not expected_run_relative.issubset(run_keys) or len(run_keys) != len(expected_run_relative) + 1:
        raise OrchestrationError("D0-A run_contract runtime path roster differs")
    extension_keys = run_keys - expected_run_relative
    extension_expected = (
        repository
        / ".conda/lib/python3.10/site-packages/"
        "MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so"
    ).resolve()
    extension_key = next(iter(extension_keys))
    if not Path(extension_key).is_absolute() or Path(extension_key).resolve() != extension_expected:
        raise OrchestrationError("D0-A run_contract extension path differs")
    for declared in run_keys | set(freeze_runtime):
        lowered = declared.lower()
        if any(token in lowered for token in ("trainval", "test_", "validation_")):
            raise OrchestrationError("D0-A runtime roster contains a data/evaluation path")


def validate_dataset_completion(
    repository: Path,
    dataset: str,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Deep-validate one stopped, completed D0-A run and build its anchor."""

    config, freeze = _validate_protocol_constants(repository)
    paths = _dataset_paths(repository, dataset)
    if paths.output_dir.is_symlink() or not paths.output_dir.is_dir():
        raise OrchestrationError(f"{dataset} output directory is missing or a symlink")
    if trainer_processes_for_output(paths.output_dir, proc_root=proc_root):
        raise OrchestrationError(f"{dataset} trainer is still running")
    for path in _required_d0a_files(paths):
        if path.is_symlink() or not path.is_file():
            raise OrchestrationError(f"{dataset} completion input missing/unsafe: {path}")
    for root, directory_names, file_names in os.walk(paths.output_dir, followlinks=False):
        for name in tuple(directory_names) + tuple(file_names):
            candidate = Path(root) / name
            if candidate.is_symlink():
                raise OrchestrationError(f"{dataset} run directory contains a symlink: {candidate}")
            lowered = name.lower()
            if (
                "trainval" in lowered
                or "best_miou" in lowered
                or "best_pd" in lowered
                or lowered in {"test", "validation", "val"}
                or lowered.startswith(("test_", "validation_", "val_"))
            ):
                raise OrchestrationError(f"{dataset} run directory contains a forbidden artifact: {candidate}")

    dataset_config = _mapping(_mapping(config.get("datasets"), "datasets").get(dataset), dataset)
    run_contract, run_contract_sha = _load_json(paths.run_contract, "D0-A run contract")
    run_config = _mapping(run_contract.get("run_config"), "run_contract.run_config")
    split_manifest = _mapping(run_contract.get("split_manifest"), "run_contract.split_manifest")
    firewall = _mapping(run_contract.get("access_firewall"), "run_contract.access_firewall")
    contract_runtime = _mapping(run_contract.get("runtime_sha256"), "run runtime hashes")
    freeze_runtime = _mapping(freeze.get("runtime_sha256"), "freeze runtime hashes")
    # Validate the key roster before opening a single path declared by the run
    # contract.  This prevents an injected data/evaluation path from becoming a
    # read primitive in either this orchestrator or the exporter.
    _validate_runtime_rosters(repository, contract_runtime, freeze_runtime)
    _verify_runtime_bindings(repository, contract_runtime, "run runtime hashes")
    _verify_runtime_bindings(repository, freeze_runtime, "freeze runtime hashes")
    expected_output = str(paths.output_dir.resolve(strict=True))
    required_run = {
        "schema_version": 1,
        "protocol_id": D0A_PROTOCOL_ID,
        "host_architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "dataset": dataset,
        "epochs": EXPECTED_EPOCHS,
        "expected_state_dict_keys": EXPECTED_STATE_DICT_KEYS,
        "data_mode": "full_train_only",
        "train_only_smoke": False,
        "max_train_batches": None,
        "checkpoint_selection": "fixed_final_epoch_train_only",
        "test_payload_access_allowed": False,
        "validation_payload_access_allowed": False,
        "output_dir": expected_output,
        "seed": 42,
        "num_workers": 8,
        "batch_size": 16,
        "learning_rate": 0.05,
        "warm_epochs": 5,
        "base_size": 256,
        "crop_size": 224,
        "lambda_degraded": 1.0,
        "degraded_branch_batchnorm": "train_batch_stats_no_running_update",
        "probe_schedule": "deterministic_alternating_per_optimizer_step",
        "lf_mask_ratio": 0.2,
        "lf_keep_probability": 0.5,
        "hf_target_rms": 0.02,
        "hf_low_cut_ratio": 0.2,
        "device": "cuda:0",
        "protocol_path": str((repository / D0A_CONFIG_RELATIVE).resolve(strict=True)),
        "protocol_sha256": EXPECTED_D0A_CONFIG_SHA256,
        "root": str((repository / Path(str(dataset_config["root"]))).resolve(strict=True)),
        "train_split": str((repository / Path(str(dataset_config["train_split"]))).resolve(strict=True)),
        "known_train_size_mismatches": dataset_config.get("known_train_size_mismatches"),
    }
    for field, expected in required_run.items():
        if type(run_config.get(field)) is not type(expected) or run_config.get(field) != expected:
            raise OrchestrationError(f"{dataset} run_config.{field} drift")
    _require_zero_access(split_manifest, f"{dataset}.split_manifest")
    _require_zero_access(firewall, f"{dataset}.access_firewall")
    if (
        firewall.get("implementation_has_test_loader") is not False
        or firewall.get("implementation_has_validation_loader") is not False
        or split_manifest.get("role") != "official_train_only"
    ):
        raise OrchestrationError(f"{dataset} train-only firewall drift")

    expected_images = _integer(dataset_config.get("train_images"), "train_images", minimum=1)
    batch_size = _integer(run_config.get("batch_size"), "batch_size", minimum=1)
    expected_batches = expected_images // batch_size
    expected_steps = expected_batches * EXPECTED_EPOCHS
    if (
        run_config.get("expected_train_images") != expected_images
        or split_manifest.get("train_count") != expected_images
        or run_config.get("expected_train_split_sha256") != dataset_config.get("train_split_sha256")
        or split_manifest.get("train_split_sha256") != dataset_config.get("train_split_sha256")
        or run_config.get("expected_train_corpus_manifest_sha256") != dataset_config.get("train_corpus_manifest_sha256")
        or split_manifest.get("train_corpus_manifest_sha256") != dataset_config.get("train_corpus_manifest_sha256")
    ):
        raise OrchestrationError(f"{dataset} train split/corpus binding drift")

    rows, metrics_sha = _validate_metrics(
        paths.metrics,
        expected_batches=expected_batches,
        warm_epochs=_integer(run_config.get("warm_epochs"), "warm_epochs", minimum=0),
        learning_rate=float(run_config.get("learning_rate")),
    )
    final_row = rows[-1]
    summary, summary_sha = _load_json(paths.summary, "D0-A completion summary")
    if (
        type(summary.get("validation_payload_opens")) is not int
        or summary.get("validation_payload_opens") != 0
        or type(summary.get("test_payload_opens")) is not int
        or summary.get("test_payload_opens") != 0
    ):
        raise OrchestrationError(f"{dataset} summary access counters must be integer zero")
    if (
        summary.get("dataset") != dataset
        or summary.get("method_name") != "CR-SITTA"
        or summary.get("method_stage") != "D0-A"
        or summary.get("data_mode") != "full_train_only"
        or summary.get("completed_epochs") != EXPECTED_EPOCHS
        or summary.get("global_optimizer_steps") != expected_steps
        or summary.get("latest_train_metrics") != final_row
        or summary.get("test_selected") is not False
        or summary.get("validation_payload_opens") != 0
        or summary.get("test_payload_opens") != 0
        or Path(str(summary.get("fixed_final_checkpoint"))).resolve() != paths.final_checkpoint.resolve()
    ):
        raise OrchestrationError(f"{dataset} completion summary drift")

    final_sha = sha256_file(paths.final_checkpoint)
    last_sha = sha256_file(paths.last_checkpoint)
    try:
        import export_cr_sitta_d0a_safe_checkpoint as exporter

        context = exporter._prepare_contract(
            repository=repository,
            source_checkpoint=paths.final_checkpoint,
            run_contract_path=paths.run_contract,
            full_train_freeze_path=_repository_path(repository, D0A_FREEZE_RELATIVE, "freeze"),
            expected_run_contract_sha256=run_contract_sha,
            expected_full_train_freeze_sha256=EXPECTED_D0A_FREEZE_SHA256,
            protocol_config_path=_repository_path(repository, D0A_CONFIG_RELATIVE, "config"),
            train_split_path=None,
        )
        final_payload = exporter._load_trusted_checkpoint(paths.final_checkpoint, final_sha)
        _final_state, final_step = exporter._validate_source_payload(
            final_payload, run_config=run_config, split_manifest=split_manifest
        )
        exporter._validate_summary(
            summary,
            source_path=paths.final_checkpoint,
            dataset=dataset,
            global_step=final_step,
        )
        last_payload = exporter._load_trusted_checkpoint(paths.last_checkpoint, last_sha)
        _last_state, last_step = exporter._validate_source_payload(
            last_payload, run_config=run_config, split_manifest=split_manifest
        )
    except Exception as exc:
        raise OrchestrationError(f"{dataset} checkpoint contract failed: {exc}") from exc
    if final_step != expected_steps or last_step != expected_steps:
        raise OrchestrationError(f"{dataset} final checkpoint optimizer-step drift")
    if final_payload.get("latest_train_metrics") != final_row or last_payload.get("latest_train_metrics") != final_row:
        raise OrchestrationError(f"{dataset} checkpoint/metrics tail mismatch")
    if not _deep_equal(final_payload, last_payload):
        raise OrchestrationError(f"{dataset} final and last checkpoint payloads differ")

    for key in set(contract_runtime).intersection(freeze_runtime):
        if contract_runtime[key] != freeze_runtime[key]:
            raise OrchestrationError(f"{dataset} run/freeze hash disagreement: {key}")
    if dataset == "NUDT-SIRST":
        started = _mapping(freeze.get("started_run_contract"), "started run contract")
        if started.get("dataset") != dataset or started.get("sha256") != run_contract_sha:
            raise OrchestrationError("NUDT-SIRST run contract differs from pre-run anchor")
    if trainer_processes_for_output(paths.output_dir, proc_root=proc_root):
        raise OrchestrationError(f"{dataset} trainer appeared during validation")
    if sha256_file(paths.metrics) != metrics_sha or sha256_file(paths.summary) != summary_sha:
        raise OrchestrationError(f"{dataset} completion metadata changed during validation")
    if sha256_file(paths.final_checkpoint) != final_sha or sha256_file(paths.last_checkpoint) != last_sha:
        raise OrchestrationError(f"{dataset} checkpoint changed during validation")

    official_train_split = Path(str(context["train_split"]))
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0a_completion_hash_anchor_v1",
        "protocol_id": PROTOCOL_ID,
        "upstream_protocol_id": D0A_PROTOCOL_ID,
        "dataset": dataset,
        "status": "d0a_1000_epoch_train_only_complete",
        "created_at_utc": _utc_now(),
        "completion": {
            "epochs": EXPECTED_EPOCHS,
            "batches_per_epoch": expected_batches,
            "global_optimizer_steps": expected_steps,
            "state_dict_keys": EXPECTED_STATE_DICT_KEYS,
            "final_last_semantically_identical": True,
            "metrics_rows": EXPECTED_EPOCHS,
            "checkpoint_selection": "fixed_final_epoch_train_only",
            "test_selected": False,
        },
        "process_firewall": {
            "trainer_processes_at_validation": [],
            "exact_output_dir": _display_path(paths.output_dir, repository),
        },
        "access_firewall": {
            **{field: 0 for field in ZERO_ACCESS_FIELDS},
            "implementation_has_test_loader": False,
            "implementation_has_validation_loader": False,
            "no_validation_split": True,
            "data_role": "official_train_only",
        },
        "artifacts": {
            "full_train_freeze": _artifact(_repository_path(repository, D0A_FREEZE_RELATIVE, "freeze"), repository),
            "protocol_config": _artifact(_repository_path(repository, D0A_CONFIG_RELATIVE, "config"), repository),
            "training_runner": _artifact(repository / "train_cr_sitta_d0a.py", repository),
            "run_contract": _artifact(paths.run_contract, repository),
            "train_metrics": _artifact(paths.metrics, repository),
            "completion_summary": _artifact(paths.summary, repository),
            "final_checkpoint": _artifact(paths.final_checkpoint, repository),
            "last_checkpoint": _artifact(paths.last_checkpoint, repository),
            "official_train_split": _artifact(official_train_split, repository),
            "archived_train_split": _artifact(paths.archived_train_split, repository),
            "safe_exporter": _artifact(repository / EXPORTER_RELATIVE, repository),
        },
        "runtime_sha256": dict(contract_runtime),
        "frozen_runtime_sha256": dict(freeze_runtime),
        "d0b_authorized_next": True,
        "d1_launched": False,
        "formal_test_allowed": False,
    }


def _artifact_path(repository: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise OrchestrationError(f"{label} path is invalid")
    raw = Path(value)
    if ".." in raw.parts:
        raise OrchestrationError(f"{label} path contains '..'")
    path = raw if raw.is_absolute() else repository / raw
    absolute = path.absolute()
    if not absolute.is_relative_to(repository.absolute()):
        raise OrchestrationError(f"{label} path is outside the repository")
    return absolute


def verify_completion_anchor(
    repository: Path,
    dataset: str,
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[dict[str, Any], str]:
    paths = _dataset_paths(repository, dataset)
    anchor, anchor_sha = _load_json(paths.anchor, "D0-A completion anchor")
    config, freeze = _validate_protocol_constants(repository)
    dataset_config = _mapping(
        _mapping(config.get("datasets"), "D0-A datasets").get(dataset), dataset
    )
    expected_images = _integer(dataset_config.get("train_images"), "train_images", minimum=1)
    expected_batches = expected_images // 16
    expected_completion = {
        "epochs": EXPECTED_EPOCHS,
        "batches_per_epoch": expected_batches,
        "global_optimizer_steps": expected_batches * EXPECTED_EPOCHS,
        "state_dict_keys": EXPECTED_STATE_DICT_KEYS,
        "final_last_semantically_identical": True,
        "metrics_rows": EXPECTED_EPOCHS,
        "checkpoint_selection": "fixed_final_epoch_train_only",
        "test_selected": False,
    }
    expected_process = {
        "trainer_processes_at_validation": [],
        "exact_output_dir": _display_path(paths.output_dir, repository),
    }
    expected_access = {
        **{field: 0 for field in ZERO_ACCESS_FIELDS},
        "implementation_has_test_loader": False,
        "implementation_has_validation_loader": False,
        "no_validation_split": True,
        "data_role": "official_train_only",
    }
    completion = _mapping(anchor.get("completion"), "anchor completion")
    process_firewall = _mapping(anchor.get("process_firewall"), "anchor process firewall")
    if (
        type(anchor.get("schema_version")) is not int
        or anchor.get("schema_version") != 1
        or anchor.get("artifact_type") != "cr_sitta_d0a_completion_hash_anchor_v1"
        or anchor.get("protocol_id") != PROTOCOL_ID
        or anchor.get("upstream_protocol_id") != D0A_PROTOCOL_ID
        or anchor.get("dataset") != dataset
        or anchor.get("status") != "d0a_1000_epoch_train_only_complete"
        or not isinstance(anchor.get("created_at_utc"), str)
        or not anchor.get("created_at_utc")
        or anchor.get("completion") != expected_completion
        or anchor.get("process_firewall") != expected_process
        or anchor.get("access_firewall") != expected_access
        or anchor.get("d0b_authorized_next") is not True
        or anchor.get("d1_launched") is not False
        or anchor.get("formal_test_allowed") is not False
    ):
        raise OrchestrationError(f"{dataset} completion anchor semantics differ")
    for field in (
        "epochs",
        "batches_per_epoch",
        "global_optimizer_steps",
        "state_dict_keys",
        "metrics_rows",
    ):
        if type(completion.get(field)) is not int:
            raise OrchestrationError(f"{dataset} anchor completion.{field} must be an integer")
    if (
        completion.get("final_last_semantically_identical") is not True
        or completion.get("test_selected") is not False
        or process_firewall.get("trainer_processes_at_validation") != []
    ):
        raise OrchestrationError(f"{dataset} anchor completion/process types differ")
    access = _mapping(anchor.get("access_firewall"), "anchor access firewall")
    _require_zero_access(access, f"{dataset}.anchor.access_firewall")
    if (
        access.get("implementation_has_test_loader") is not False
        or access.get("implementation_has_validation_loader") is not False
        or access.get("no_validation_split") is not True
        or access.get("data_role") != "official_train_only"
    ):
        raise OrchestrationError(f"{dataset} anchor access semantics differ")
    if trainer_processes_for_output(paths.output_dir, proc_root=proc_root):
        raise OrchestrationError(f"{dataset} trainer is running after anchor publication")
    artifacts = _mapping(anchor.get("artifacts"), "anchor artifacts")
    required = {
        "full_train_freeze",
        "protocol_config",
        "training_runner",
        "run_contract",
        "train_metrics",
        "completion_summary",
        "final_checkpoint",
        "last_checkpoint",
        "official_train_split",
        "archived_train_split",
        "safe_exporter",
    }
    if set(artifacts) != required:
        raise OrchestrationError(f"{dataset} completion anchor artifact roster differs")
    official_train_split = repository / Path(str(dataset_config["train_split"]))
    expected_paths = {
        "full_train_freeze": repository / D0A_FREEZE_RELATIVE,
        "protocol_config": repository / D0A_CONFIG_RELATIVE,
        "training_runner": repository / "train_cr_sitta_d0a.py",
        "run_contract": paths.run_contract,
        "train_metrics": paths.metrics,
        "completion_summary": paths.summary,
        "final_checkpoint": paths.final_checkpoint,
        "last_checkpoint": paths.last_checkpoint,
        "official_train_split": official_train_split,
        "archived_train_split": paths.archived_train_split,
        "safe_exporter": repository / EXPORTER_RELATIVE,
    }
    for name, raw in artifacts.items():
        binding = _mapping(raw, f"anchor artifact {name}")
        path = _artifact_path(repository, binding.get("path"), f"anchor artifact {name}")
        if path.resolve() != expected_paths[name].resolve():
            raise OrchestrationError(f"{dataset} anchored artifact path differs: {name}")
        if sha256_file(path) != _require_sha256(binding.get("sha256"), name):
            raise OrchestrationError(f"{dataset} anchored artifact drift: {name}")
        size = binding.get("bytes")
        if type(size) is not int or size <= 0 or path.stat(follow_symlinks=False).st_size != size:
            raise OrchestrationError(f"{dataset} anchored artifact size drift: {name}")
    run_contract, _ = _load_json(paths.run_contract, "anchored run contract")
    run_runtime = _mapping(run_contract.get("runtime_sha256"), "run contract runtime")
    freeze_runtime = _mapping(freeze.get("runtime_sha256"), "freeze runtime")
    if anchor.get("runtime_sha256") != run_runtime:
        raise OrchestrationError(f"{dataset} anchor/run-contract runtime bindings differ")
    if anchor.get("frozen_runtime_sha256") != freeze_runtime:
        raise OrchestrationError(f"{dataset} anchor/freeze runtime bindings differ")
    _validate_runtime_rosters(repository, run_runtime, freeze_runtime)
    _verify_runtime_bindings(repository, run_runtime, "runtime")
    _verify_runtime_bindings(repository, freeze_runtime, "frozen runtime")
    if paths.anchor.stat().st_mode & 0o222:
        raise OrchestrationError(f"{dataset} completion anchor is not read-only")
    return anchor, anchor_sha


def _publish_json_noreplace(
    repository: Path,
    destination: Path,
    payload: Mapping[str, Any],
    *,
    guard: Callable[[], None] | None = None,
) -> None:
    relative_parent = destination.parent.absolute().relative_to(repository.absolute())
    ensure_directory_chain_nofollow(repository, tuple(relative_parent.parts))
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        staging.chmod(0o444)
        publish_file_noreplace(staging, destination, pre_rename_guard=guard)
    finally:
        if staging.exists() and staging.is_file() and not staging.is_symlink():
            staging.chmod(0o600)
            staging.unlink()


def publish_completion_anchor(
    repository: Path,
    dataset: str,
    payload: Mapping[str, Any],
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[dict[str, Any], str]:
    paths = _dataset_paths(repository, dataset)
    if paths.anchor.exists() or paths.anchor.is_symlink():
        return verify_completion_anchor(repository, dataset, proc_root=proc_root)

    expected_artifacts = _mapping(payload.get("artifacts"), "anchor artifacts")

    def guard() -> None:
        if trainer_processes_for_output(paths.output_dir, proc_root=proc_root):
            raise OrchestrationError(f"{dataset} trainer appeared before anchor publish")
        for name, raw in expected_artifacts.items():
            binding = _mapping(raw, f"anchor artifact {name}")
            path = _artifact_path(repository, binding.get("path"), name)
            if sha256_file(path) != binding.get("sha256"):
                raise OrchestrationError(f"{dataset} input changed before anchor publish: {name}")

    _publish_json_noreplace(repository, paths.anchor, payload, guard=guard)
    return verify_completion_anchor(repository, dataset, proc_root=proc_root)


def _anchor_set_path(repository: Path) -> Path:
    return _repository_path(
        repository,
        ORCHESTRATOR_RESULT_RELATIVE / "D0A_COMPLETION_ANCHOR_SET.json",
        "anchor set",
    )


def publish_anchor_set(
    repository: Path,
    anchors: Mapping[str, tuple[Mapping[str, Any], str]],
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[dict[str, Any], str]:
    destination = _anchor_set_path(repository)
    runtime_maps = {
        dataset: dict(_mapping(anchors[dataset][0].get("runtime_sha256"), f"{dataset} runtime"))
        for dataset in DATASETS
    }
    reference_runtime = runtime_maps[DATASETS[0]]
    if any(runtime_maps[dataset] != reference_runtime for dataset in DATASETS[1:]):
        raise OrchestrationError("three D0-A run_contract runtime bindings are not identical")
    expected_bindings = {
        dataset: {
            "path": _display_path(_dataset_paths(repository, dataset).anchor, repository),
            "sha256": anchors[dataset][1],
        }
        for dataset in DATASETS
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0a_completion_anchor_set_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "all_three_d0a_runs_anchored",
        "created_at_utc": _utc_now(),
        "ordered_datasets": list(DATASETS),
        "anchors": expected_bindings,
        "shared_run_contract_runtime_sha256": reference_runtime,
        "safe_exports_started_after_anchor_set": True,
        "d1_launched": False,
        "formal_test_allowed": False,
    }

    def verify() -> tuple[dict[str, Any], str]:
        observed, digest = _load_json(destination, "D0-A anchor set")
        if (
            observed.get("artifact_type") != payload["artifact_type"]
            or observed.get("protocol_id") != PROTOCOL_ID
            or observed.get("status") != payload["status"]
            or observed.get("ordered_datasets") != list(DATASETS)
            or observed.get("anchors") != expected_bindings
            or observed.get("shared_run_contract_runtime_sha256") != reference_runtime
            or observed.get("d1_launched") is not False
            or observed.get("formal_test_allowed") is not False
        ):
            raise OrchestrationError("D0-A anchor set differs")
        if destination.stat().st_mode & 0o222:
            raise OrchestrationError("D0-A anchor set is not read-only")
        for dataset in DATASETS:
            verify_completion_anchor(repository, dataset, proc_root=proc_root)
        return observed, digest

    if destination.exists() or destination.is_symlink():
        return verify()

    def guard() -> None:
        for dataset in DATASETS:
            _anchor, digest = verify_completion_anchor(repository, dataset, proc_root=proc_root)
            if digest != expected_bindings[dataset]["sha256"]:
                raise OrchestrationError("D0-A anchor changed before set publication")

    _publish_json_noreplace(repository, destination, payload, guard=guard)
    return verify()


def _handoff_tool_bindings(repository: Path) -> dict[str, dict[str, Any]]:
    values = {
        "orchestrator": Path(__file__).resolve(),
        "safe_exporter": repository / EXPORTER_RELATIVE,
        "d0b_runner": repository / D0B_RUNNER_RELATIVE,
        "d0b_config": repository / D0B_CONFIG_RELATIVE,
        "d0b_authorization_gate": repository / "analysis/cr_sitta_d0b_gate.py",
        "original_model": repository / "model/MSHNet_NSFPN.py",
        "adaptable_model": repository / "model/MSHNet_NSFPN_adaptable.py",
        "secure_io": repository / "tta/d0_secure_io.py",
    }
    return {name: _artifact(path, repository) for name, path in values.items()}


def _handoff_freeze_path(repository: Path) -> Path:
    return _repository_path(
        repository,
        ORCHESTRATOR_RESULT_RELATIVE / "HANDOFF_EXECUTION_FREEZE.json",
        "handoff freeze",
    )


def publish_handoff_freeze(
    repository: Path,
    anchor_set_sha256: str,
) -> tuple[dict[str, Any], str]:
    pilot_subset_receipt = _validate_d0b_static_contract(repository)
    destination = _handoff_freeze_path(repository)
    bindings = _handoff_tool_bindings(repository)
    payload = {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0a_to_d0b_handoff_freeze_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_before_safe_export_and_d0b",
        "created_at_utc": _utc_now(),
        "anchor_set": {
            "path": _display_path(_anchor_set_path(repository), repository),
            "sha256": anchor_set_sha256,
        },
        "tools": bindings,
        "pilot64_train_subset_proof": pilot_subset_receipt,
        "ordered_next_phases": [
            "safe_export_all_three",
            "d0b_preflight",
            "d0b_freeze",
            "d0b_all_teacher_then_all_candidate_then_all_outer",
            "d0b_aggregate",
            "d0b_verify",
            "pipeline_complete",
        ],
        "gpu_contract": {
            "physical_gpu_index": PHYSICAL_GPU_INDEX,
            "cuda_visible_devices": "2",
            "child_device": "cuda:0",
            "exclusive_lock_required": True,
            "wait_while_foreign_compute_process_exists": True,
        },
        "d1_launched": False,
        "formal_test_allowed": False,
    }

    if destination.exists() or destination.is_symlink():
        return verify_handoff_freeze(repository)

    def guard() -> None:
        if sha256_file(_anchor_set_path(repository)) != anchor_set_sha256:
            raise OrchestrationError("anchor set changed before handoff freeze")
        if _handoff_tool_bindings(repository) != bindings:
            raise OrchestrationError("handoff tool changed before freeze")

    _publish_json_noreplace(repository, destination, payload, guard=guard)
    return verify_handoff_freeze(repository)


def verify_handoff_freeze(repository: Path) -> tuple[dict[str, Any], str]:
    """Pure verification: unlike publication, this function never creates files."""

    destination = _handoff_freeze_path(repository)
    observed, digest = _load_json(destination, "handoff freeze")
    anchor_set_sha = sha256_file(_anchor_set_path(repository))
    expected_anchor = {
        "path": _display_path(_anchor_set_path(repository), repository),
        "sha256": anchor_set_sha,
    }
    expected_tools = _handoff_tool_bindings(repository)
    expected_subset = _validate_d0b_static_contract(repository)
    expected_order = [
        "safe_export_all_three",
        "d0b_preflight",
        "d0b_freeze",
        "d0b_all_teacher_then_all_candidate_then_all_outer",
        "d0b_aggregate",
        "d0b_verify",
        "pipeline_complete",
    ]
    expected_gpu = {
        "physical_gpu_index": PHYSICAL_GPU_INDEX,
        "cuda_visible_devices": "2",
        "child_device": "cuda:0",
        "exclusive_lock_required": True,
        "wait_while_foreign_compute_process_exists": True,
    }
    if (
        type(observed.get("schema_version")) is not int
        or observed.get("schema_version") != 1
        or observed.get("artifact_type") != "cr_sitta_d0a_to_d0b_handoff_freeze_v1"
        or observed.get("protocol_id") != PROTOCOL_ID
        or observed.get("status") != "frozen_before_safe_export_and_d0b"
        or not isinstance(observed.get("created_at_utc"), str)
        or not observed.get("created_at_utc")
        or observed.get("anchor_set") != expected_anchor
        or observed.get("tools") != expected_tools
        or observed.get("pilot64_train_subset_proof") != expected_subset
        or observed.get("ordered_next_phases") != expected_order
        or observed.get("gpu_contract") != expected_gpu
        or observed.get("d1_launched") is not False
        or observed.get("formal_test_allowed") is not False
    ):
        raise OrchestrationError("handoff execution freeze differs")
    if destination.stat().st_mode & 0o222:
        raise OrchestrationError("handoff execution freeze is not read-only")
    return observed, digest


def _anchor_artifact(anchor: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _mapping(_mapping(anchor.get("artifacts"), "anchor artifacts").get(name), name)


def build_export_command(
    repository: Path,
    dataset: str,
    anchor: Mapping[str, Any],
    *,
    python_executable: Path,
) -> tuple[str, ...]:
    final = _anchor_artifact(anchor, "final_checkpoint")
    contract = _anchor_artifact(anchor, "run_contract")
    freeze = _anchor_artifact(anchor, "full_train_freeze")
    return (
        str(python_executable),
        str(repository / EXPORTER_RELATIVE),
        "--source-checkpoint",
        str(_artifact_path(repository, final.get("path"), "final checkpoint")),
        "--expected-source-sha256",
        _require_sha256(final.get("sha256"), "final checkpoint"),
        "--run-contract",
        str(_artifact_path(repository, contract.get("path"), "run contract")),
        "--expected-run-contract-sha256",
        _require_sha256(contract.get("sha256"), "run contract"),
        "--full-train-freeze",
        str(_artifact_path(repository, freeze.get("path"), "full-train freeze")),
        "--expected-full-train-freeze-sha256",
        _require_sha256(freeze.get("sha256"), "full-train freeze"),
        "--repository",
        str(repository),
        "--trust-local-source",
    )


def _default_command_runner(command: Sequence[str], environment: Mapping[str, str]) -> CommandResult:
    completed = subprocess.run(
        tuple(command),
        cwd=REPOSITORY,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=False,
    )
    return CommandResult(tuple(command), completed.returncode, completed.stdout, completed.stderr)


def _command_json(result: CommandResult, name: str, *, gpu: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_orchestrator_command_receipt_v1",
        "protocol_id": PROTOCOL_ID,
        "name": name,
        "created_at_utc": _utc_now(),
        "command": list(result.command),
        "returncode": result.returncode,
        "stdout_sha256": _sha256_bytes(result.stdout.encode("utf-8")),
        "stderr_sha256": _sha256_bytes(result.stderr.encode("utf-8")),
        "gpu_phase": gpu,
        "physical_gpu_index": PHYSICAL_GPU_INDEX if gpu else None,
        "cuda_visible_devices": "2" if gpu else "",
        "child_device": "cuda:0" if gpu else None,
        "formal_test_allowed": False,
    }


def _run_logged_command(
    repository: Path,
    session_dir: Path,
    name: str,
    command: Sequence[str],
    *,
    gpu: bool,
    command_runner: CommandRunner,
) -> CommandResult:
    verify_handoff_freeze(repository)
    environment = dict(os.environ)
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = "2" if gpu else ""
    result = command_runner(tuple(command), environment)
    log_path = session_dir / f"{name}.log"
    receipt_path = session_dir / f"{name}.json"
    with log_path.open("xb") as stream:
        stream.write(result.stdout.encode("utf-8"))
        if result.stderr:
            stream.write(b"\n[stderr]\n")
            stream.write(result.stderr.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    log_path.chmod(0o444)
    _publish_json_noreplace(repository, receipt_path, _command_json(result, name, gpu=gpu))
    if result.returncode != 0:
        raise OrchestrationError(f"command failed ({result.returncode}): {name}")
    return result


def verify_safe_export_against_anchor(
    repository: Path,
    dataset: str,
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    paths = _dataset_paths(repository, dataset)
    if paths.safe_checkpoint.is_symlink() or paths.safe_export_receipt.is_symlink():
        raise OrchestrationError(f"{dataset} safe export uses a symlink")
    # Parse and constrain the receipt before calling the generic exporter
    # verifier: that verifier opens every artifact named by the receipt.
    # Exact roster/path admission here prevents a forged data/evaluation path
    # from becoming a read primitive.
    pre_receipt, _pre_receipt_sha = _load_json(
        paths.safe_export_receipt, "SAFE_EXPORT preflight receipt"
    )
    pre_artifacts = _mapping(pre_receipt.get("artifacts"), "SAFE_EXPORT artifacts")
    expected_roster = {
        "source_checkpoint",
        "safe_checkpoint",
        "exporter",
        "run_contract",
        "full_train_freeze",
        "protocol_config",
        "train_split",
        "archived_train_split",
        "completion_summary",
        "training_runner",
        "original_model_implementation",
        "adaptable_model_implementation",
        "smoke_gate",
    }
    if set(pre_artifacts) != expected_roster:
        raise OrchestrationError(f"{dataset} SAFE_EXPORT artifact roster differs")
    anchor_names = {
        "source_checkpoint": "final_checkpoint",
        "exporter": "safe_exporter",
        "run_contract": "run_contract",
        "full_train_freeze": "full_train_freeze",
        "protocol_config": "protocol_config",
        "train_split": "official_train_split",
        "archived_train_split": "archived_train_split",
        "completion_summary": "completion_summary",
        "training_runner": "training_runner",
    }
    expected_paths = {
        name: _artifact_path(
            repository, _anchor_artifact(anchor, anchor_name).get("path"), anchor_name
        )
        for name, anchor_name in anchor_names.items()
    }
    expected_paths.update(
        {
            "safe_checkpoint": paths.safe_checkpoint,
            "original_model_implementation": repository / "model/MSHNet_NSFPN.py",
            "adaptable_model_implementation": repository / "model/MSHNet_NSFPN_adaptable.py",
        }
    )
    freeze, _freeze_sha = _load_json(
        repository / D0A_FREEZE_RELATIVE, "D0-A full-train freeze"
    )
    smoke_expected = _mapping(freeze.get("smoke_gate"), "D0-A smoke gate")
    expected_paths["smoke_gate"] = _artifact_path(
        repository, smoke_expected.get("path"), "smoke gate"
    )
    for name, expected_path in expected_paths.items():
        binding = _mapping(pre_artifacts.get(name), f"SAFE_EXPORT artifact {name}")
        observed_path = _artifact_path(repository, binding.get("path"), name)
        if observed_path.resolve() != expected_path.resolve():
            raise OrchestrationError(f"{dataset} SAFE_EXPORT preflight path mismatch: {name}")
    for receipt_name, anchor_name in anchor_names.items():
        if _mapping(pre_artifacts[receipt_name], receipt_name).get("sha256") != _anchor_artifact(anchor, anchor_name).get("sha256"):
            raise OrchestrationError(f"{dataset} SAFE_EXPORT preflight hash mismatch: {receipt_name}")
    if _mapping(pre_artifacts["smoke_gate"], "smoke gate").get("sha256") != smoke_expected.get("sha256"):
        raise OrchestrationError(f"{dataset} SAFE_EXPORT preflight smoke hash mismatch")
    try:
        from export_cr_sitta_d0a_safe_checkpoint import (
            _cpu_tensor_state_dict,
            _load_trusted_checkpoint,
            _validate_safe_payload,
            _weights_only_load,
            validate_repository_models,
            verify_safe_export,
        )

        receipt = verify_safe_export(
            paths.safe_export_receipt,
            state_dict_validators=(validate_repository_models,),
        )
    except Exception as exc:
        raise OrchestrationError(f"{dataset} SAFE_EXPORT verification failed: {exc}") from exc
    if receipt != pre_receipt:
        raise OrchestrationError(f"{dataset} SAFE_EXPORT changed during verification")
    source_anchor = _anchor_artifact(anchor, "final_checkpoint")
    source_path = _artifact_path(
        repository, source_anchor.get("path"), "anchored final checkpoint"
    )
    source_sha = _require_sha256(
        source_anchor.get("sha256"), "anchored final checkpoint"
    )
    try:
        source_payload = _load_trusted_checkpoint(source_path, source_sha)
        source_state = _cpu_tensor_state_dict(source_payload.get("state_dict"))
        _safe_provenance, safe_state = _validate_safe_payload(
            _weights_only_load(paths.safe_checkpoint)
        )
    except Exception as exc:
        raise OrchestrationError(
            f"{dataset} safe/source state comparison could not be established: {exc}"
        ) from exc
    if tuple(source_state) != tuple(safe_state) or any(
        not _deep_equal(source_state[key], safe_state[key]) for key in source_state
    ):
        raise OrchestrationError(
            f"{dataset} SAFE_EXPORT state_dict differs from anchored final checkpoint"
        )
    if receipt.get("dataset") != dataset:
        raise OrchestrationError(f"{dataset} SAFE_EXPORT dataset mismatch")
    artifacts = _mapping(receipt.get("artifacts"), "SAFE_EXPORT artifacts")
    if set(artifacts) != expected_roster:
        raise OrchestrationError(f"{dataset} SAFE_EXPORT artifact roster differs")
    comparisons = {
        "source_checkpoint": "final_checkpoint",
        "run_contract": "run_contract",
        "full_train_freeze": "full_train_freeze",
        "completion_summary": "completion_summary",
        "train_split": "official_train_split",
        "archived_train_split": "archived_train_split",
        "protocol_config": "protocol_config",
        "training_runner": "training_runner",
        "exporter": "safe_exporter",
    }
    for receipt_name, anchor_name in comparisons.items():
        receipt_binding = _mapping(artifacts.get(receipt_name), receipt_name)
        anchor_binding = _anchor_artifact(anchor, anchor_name)
        if receipt_binding.get("sha256") != anchor_binding.get("sha256"):
            raise OrchestrationError(f"{dataset} SAFE_EXPORT/anchor mismatch: {receipt_name}")
        if _artifact_path(repository, receipt_binding.get("path"), receipt_name).resolve() != _artifact_path(repository, anchor_binding.get("path"), anchor_name).resolve():
            raise OrchestrationError(f"{dataset} SAFE_EXPORT/anchor path mismatch: {receipt_name}")
    for name in ("safe_checkpoint", "original_model_implementation", "adaptable_model_implementation"):
        expected_path = expected_paths[name]
        binding = _mapping(artifacts.get(name), name)
        if _artifact_path(repository, binding.get("path"), name).resolve() != expected_path.resolve():
            raise OrchestrationError(f"{dataset} SAFE_EXPORT path mismatch: {name}")
    frozen_runtime = _mapping(anchor.get("frozen_runtime_sha256"), "anchor frozen runtime")
    original_binding = _mapping(
        artifacts.get("original_model_implementation"), "original model"
    )
    if original_binding.get("sha256") != frozen_runtime.get("model/MSHNet_NSFPN.py"):
        raise OrchestrationError(f"{dataset} SAFE_EXPORT original-model hash mismatch")
    handoff, _handoff_sha = verify_handoff_freeze(repository)
    handoff_tools = _mapping(handoff.get("tools"), "handoff tools")
    adaptable_expected = _mapping(handoff_tools.get("adaptable_model"), "adaptable model")
    adaptable_observed = _mapping(
        artifacts.get("adaptable_model_implementation"), "adaptable model artifact"
    )
    if (
        adaptable_observed.get("sha256") != adaptable_expected.get("sha256")
        or _artifact_path(repository, adaptable_expected.get("path"), "adaptable model").resolve()
        != (repository / "model/MSHNet_NSFPN_adaptable.py").resolve()
    ):
        raise OrchestrationError(f"{dataset} SAFE_EXPORT adaptable-model binding mismatch")
    smoke_observed = _mapping(artifacts.get("smoke_gate"), "SAFE_EXPORT smoke gate")
    if (
        smoke_observed.get("sha256") != smoke_expected.get("sha256")
        or _artifact_path(repository, smoke_observed.get("path"), "smoke gate").resolve()
        != _artifact_path(repository, smoke_expected.get("path"), "smoke gate").resolve()
    ):
        raise OrchestrationError(f"{dataset} SAFE_EXPORT smoke-gate binding mismatch")
    contract = _mapping(receipt.get("checkpoint_contract"), "SAFE_EXPORT checkpoint contract")
    firewall = _mapping(receipt.get("access_firewall"), "SAFE_EXPORT access firewall")
    if (
        contract.get("epoch") != EXPECTED_EPOCHS
        or contract.get("state_dict_keys") != EXPECTED_STATE_DICT_KEYS
        or contract.get("test_selected") is not False
        or contract.get("selection_rule") != "fixed_final_epoch_train_only"
        or contract.get("torch_load_weights_only") is not True
        or contract.get("repository_model_loads_verified") is not True
        or any(type(firewall.get(field)) is not int or firewall.get(field) != 0 for field in ZERO_ACCESS_FIELDS)
    ):
        raise OrchestrationError(f"{dataset} SAFE_EXPORT contract widened")
    return receipt


def _parse_csv_rows(text: str, columns: int) -> tuple[tuple[str, ...], ...]:
    result: list[tuple[str, ...]] = []
    for line in text.splitlines():
        if not line.strip() or "No running processes" in line:
            continue
        fields = tuple(field.strip() for field in line.split(","))
        if len(fields) != columns:
            raise OrchestrationError(f"unexpected nvidia-smi row: {line!r}")
        result.append(fields)
    return tuple(result)


def physical_gpu_uuid(
    *, command_runner: CommandRunner = _default_command_runner
) -> str:
    environment = dict(os.environ)
    result = command_runner(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ),
        environment,
    )
    if result.returncode != 0:
        raise OrchestrationError("cannot query physical GPU inventory")
    matches = [uuid_value for index, uuid_value in _parse_csv_rows(result.stdout, 2) if index == str(PHYSICAL_GPU_INDEX)]
    if len(matches) != 1:
        raise OrchestrationError("physical GPU 2 is missing or ambiguous")
    return matches[0]


def physical_gpu_compute_pids(
    gpu_uuid: str,
    *,
    command_runner: CommandRunner = _default_command_runner,
) -> tuple[int, ...]:
    result = command_runner(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ),
        dict(os.environ),
    )
    if result.returncode != 0:
        raise OrchestrationError("cannot query GPU compute processes")
    pids: list[int] = []
    for observed_uuid, raw_pid in _parse_csv_rows(result.stdout, 2):
        if observed_uuid == gpu_uuid:
            try:
                pids.append(int(raw_pid))
            except ValueError as exc:
                raise OrchestrationError("nvidia-smi returned a noninteger PID") from exc
    return tuple(sorted(pids))


@contextmanager
def gpu2_exclusive_lease(
    repository: Path,
    *,
    command_runner: CommandRunner = _default_command_runner,
    sleep: Sleep = time.sleep,
    poll_seconds: float = GPU_POLL_SECONDS,
):
    """Hold the pipeline lock and wait until physical GPU 2 has no compute PID."""

    root = _repository_path(repository, ORCHESTRATOR_RESULT_RELATIVE, "orchestrator root")
    ensure_directory_chain_nofollow(repository, tuple(root.relative_to(repository).parts))
    lock_path = root / ".physical_gpu2.execution.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise OrchestrationError("GPU2 execution lock is not a private regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        gpu_uuid = physical_gpu_uuid(command_runner=command_runner)
        while physical_gpu_compute_pids(gpu_uuid, command_runner=command_runner):
            sleep(poll_seconds)
        yield gpu_uuid
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _d0b_command(
    repository: Path,
    python_executable: Path,
    phase: str,
    dataset: str | None = None,
) -> tuple[str, ...]:
    allowed = {"preflight", "freeze", "teacher", "candidate", "outer", "aggregate", "verify"}
    if phase not in allowed:
        raise OrchestrationError(f"forbidden D0-B phase: {phase}")
    command = [
        str(python_executable),
        str(repository / D0B_RUNNER_RELATIVE),
        "--config",
        str(repository / D0B_CONFIG_RELATIVE),
        phase,
    ]
    if phase in {"teacher", "candidate", "outer"}:
        if dataset not in DATASETS:
            raise OrchestrationError(f"{phase} requires one fixed dataset")
        command.extend(("--dataset", str(dataset), "--device", "cuda:0"))
    elif dataset is not None:
        raise OrchestrationError(f"{phase} does not accept a dataset")
    return tuple(command)


def _ensure_d0b_phase_parent(
    repository: Path, phase: str, dataset: str | None = None
) -> Path:
    """Create/verify only the reviewed D0-B phase parent with O_NOFOLLOW."""

    mapping = {
        "teacher": Path("teacher_phase/R0"),
        "candidate": Path("candidate_phase/R0"),
        "outer": Path("outer_phase/R0"),
        "aggregate": Path("aggregate_phase"),
    }
    if phase not in mapping:
        raise OrchestrationError(f"unsupported D0-B phase parent: {phase}")
    if phase != "aggregate" and dataset not in DATASETS:
        raise OrchestrationError(f"{phase} requires a fixed dataset")
    relative = D0B_RESULT_RELATIVE / mapping[phase]
    parent = ensure_directory_chain_nofollow(repository, tuple(relative.parts))
    if phase == "aggregate":
        return parent
    destination = parent / str(dataset)
    if destination.is_symlink():
        raise OrchestrationError(f"D0-B destination is a symlink: {destination}")
    return parent


def _new_session_dir(repository: Path) -> Path:
    root = _repository_path(repository, ORCHESTRATOR_RESULT_RELATIVE, "orchestrator root")
    executions = ensure_directory_chain_nofollow(
        repository, tuple((ORCHESTRATOR_RESULT_RELATIVE / "executions").parts)
    )
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    session = executions / name
    session.mkdir(mode=0o700)
    return session


def _wait_for_d0a(
    repository: Path,
    *,
    proc_root: Path,
    sleep: Sleep,
    poll_seconds: float,
    stale_grace_seconds: float,
) -> None:
    while True:
        states = [
            inspect_dataset_state(
                repository,
                dataset,
                proc_root=proc_root,
                stale_grace_seconds=stale_grace_seconds,
            )
            for dataset in DATASETS
        ]
        failed = [item["dataset"] for item in states if item["state"] == "failed_or_stale"]
        if failed:
            raise OrchestrationError(f"stopped incomplete D0-A run(s): {failed}")
        if all(item["state"] == "ready_for_deep_validation" for item in states):
            return
        rendered = ", ".join(f"{item['dataset']}={item['state']}" for item in states)
        print(f"waiting for D0-A: {rendered}", flush=True)
        sleep(poll_seconds)


def _verify_pipeline_complete(repository: Path) -> tuple[dict[str, Any], str]:
    path = _repository_path(repository, ORCHESTRATOR_RESULT_RELATIVE / "PIPELINE_COMPLETE.json", "pipeline completion")
    payload, digest = _load_json(path, "pipeline completion")
    eligible = payload.get("eligible_parameter_space_ids")
    expected_artifacts = {
        "anchor_set",
        "handoff_freeze",
        "d0b_pre_run_freeze",
        "d0b_aggregate_manifest",
        "d0b_science_decision",
        "d1_authorization_not_execution",
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("artifact_type") != "cr_sitta_d0a_to_d0b_pipeline_complete_v1"
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "protocol_complete_at_d0b"
        or payload.get("d0b_protocol_status") != "protocol_complete"
        or payload.get("scientific_status") not in {"scientific_eligible", "scientific_no_eligible"}
        or not isinstance(eligible, list)
        or (payload.get("scientific_status") == "scientific_eligible") is not bool(eligible)
        or payload.get("d1_train_internal_oof_allowed") is not bool(eligible)
        or payload.get("scientific_negative_is_normal_completion")
        is not (payload.get("scientific_status") == "scientific_no_eligible")
        or payload.get("terminal_action") != "stop_after_d0b_verify"
        or payload.get("d1_launched") is not False
        or payload.get("formal_test_allowed") is not False
    ):
        raise OrchestrationError("PIPELINE_COMPLETE semantics differ")
    artifacts = _mapping(payload.get("artifacts"), "pipeline completion artifacts")
    if set(artifacts) != expected_artifacts:
        raise OrchestrationError("PIPELINE_COMPLETE artifact roster differs")
    d0b_root = _repository_path(repository, D0B_RESULT_RELATIVE, "D0-B root")
    aggregate = d0b_root / "aggregate_phase" / "R0"
    expected_paths = {
        "anchor_set": _anchor_set_path(repository),
        "handoff_freeze": _handoff_freeze_path(repository),
        "d0b_pre_run_freeze": d0b_root / "PRE_RUN_FREEZE.json",
        "d0b_aggregate_manifest": aggregate / "manifest.json",
        "d0b_science_decision": aggregate / "D0B_SCIENCE_DECISION.json",
        "d1_authorization_not_execution": aggregate / "D1_AUTHORIZATION.json",
    }
    for name, raw in artifacts.items():
        binding = _mapping(raw, name)
        bound_path = _artifact_path(repository, binding.get("path"), name)
        if bound_path.resolve() != expected_paths[name].resolve():
            raise OrchestrationError(f"PIPELINE_COMPLETE artifact path differs: {name}")
        if sha256_file(bound_path) != binding.get("sha256"):
            raise OrchestrationError(f"PIPELINE_COMPLETE artifact drift: {name}")
    science, _ = _load_json(expected_paths["d0b_science_decision"], "D0-B science decision")
    authorization, _ = _load_json(
        expected_paths["d1_authorization_not_execution"], "D1 authorization"
    )
    if (
        science.get("protocol_status") != payload.get("d0b_protocol_status")
        or science.get("scientific_status") != payload.get("scientific_status")
        or science.get("eligible_parameter_space_ids") != eligible
        or science.get("d1_train_internal_oof_allowed")
        is not payload.get("d1_train_internal_oof_allowed")
        or authorization.get("protocol_status") != payload.get("d0b_protocol_status")
        or authorization.get("scientific_status") != payload.get("scientific_status")
        or authorization.get("eligible_parameter_space_ids") != eligible
        or authorization.get("d1_train_internal_oof_allowed")
        is not payload.get("d1_train_internal_oof_allowed")
        or science.get("formal_test_allowed") is not False
        or authorization.get("formal_test_allowed") is not False
    ):
        raise OrchestrationError("PIPELINE_COMPLETE disagrees with D0-B decisions")
    if path.stat().st_mode & 0o222:
        raise OrchestrationError("PIPELINE_COMPLETE is not read-only")
    return payload, digest


def _publish_pipeline_complete(repository: Path) -> tuple[dict[str, Any], str]:
    destination = _repository_path(repository, ORCHESTRATOR_RESULT_RELATIVE / "PIPELINE_COMPLETE.json", "pipeline completion")
    if destination.exists() or destination.is_symlink():
        return _verify_pipeline_complete(repository)
    d0b_root = _repository_path(repository, D0B_RESULT_RELATIVE, "D0-B root")
    aggregate = d0b_root / "aggregate_phase" / "R0"
    science, science_sha = _load_json(aggregate / "D0B_SCIENCE_DECISION.json", "D0-B science decision")
    authorization, authorization_sha = _load_json(aggregate / "D1_AUTHORIZATION.json", "D1 authorization")
    manifest, manifest_sha = _load_json(aggregate / "manifest.json", "D0-B aggregate manifest")
    eligible = science.get("eligible_parameter_space_ids")
    d1_allowed = authorization.get("d1_train_internal_oof_allowed")
    if (
        science.get("protocol_status") != "protocol_complete"
        or science.get("scientific_status") not in {"scientific_eligible", "scientific_no_eligible"}
        or not isinstance(eligible, list)
        or (science.get("scientific_status") == "scientific_eligible") is not bool(eligible)
        or authorization.get("protocol_status") != "protocol_complete"
        or authorization.get("scientific_status") != science.get("scientific_status")
        or authorization.get("eligible_parameter_space_ids") != eligible
        or d1_allowed is not bool(eligible)
        or science.get("d1_train_internal_oof_allowed") is not d1_allowed
        or science.get("formal_test_allowed") is not False
        or authorization.get("formal_test_allowed") is not False
        or manifest.get("formal_test_allowed") is not False
    ):
        raise OrchestrationError("D0-B terminal decision receipts disagree")
    anchor_set = _anchor_set_path(repository)
    handoff = _handoff_freeze_path(repository)
    d0b_freeze = d0b_root / "PRE_RUN_FREEZE.json"
    payload = {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0a_to_d0b_pipeline_complete_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "protocol_complete_at_d0b",
        "created_at_utc": _utc_now(),
        "d0b_protocol_status": "protocol_complete",
        "scientific_status": science["scientific_status"],
        "eligible_parameter_space_ids": eligible,
        "d1_train_internal_oof_allowed": d1_allowed,
        "scientific_negative_is_normal_completion": science["scientific_status"] == "scientific_no_eligible",
        "terminal_action": "stop_after_d0b_verify",
        "d1_launched": False,
        "formal_test_allowed": False,
        "artifacts": {
            "anchor_set": {"path": _display_path(anchor_set, repository), "sha256": sha256_file(anchor_set)},
            "handoff_freeze": {"path": _display_path(handoff, repository), "sha256": sha256_file(handoff)},
            "d0b_pre_run_freeze": {"path": _display_path(d0b_freeze, repository), "sha256": sha256_file(d0b_freeze)},
            "d0b_aggregate_manifest": {"path": _display_path(aggregate / "manifest.json", repository), "sha256": manifest_sha},
            "d0b_science_decision": {"path": _display_path(aggregate / "D0B_SCIENCE_DECISION.json", repository), "sha256": science_sha},
            "d1_authorization_not_execution": {"path": _display_path(aggregate / "D1_AUTHORIZATION.json", repository), "sha256": authorization_sha},
        },
    }

    def guard() -> None:
        verify_handoff_freeze(repository)
        if sha256_file(aggregate / "D0B_SCIENCE_DECISION.json") != science_sha or sha256_file(aggregate / "D1_AUTHORIZATION.json") != authorization_sha or sha256_file(aggregate / "manifest.json") != manifest_sha:
            raise OrchestrationError("D0-B terminal artifact changed before completion publish")

    _publish_json_noreplace(repository, destination, payload, guard=guard)
    return _verify_pipeline_complete(repository)


def _publish_failure_receipt(
    repository: Path,
    session_dir: Path,
    error: BaseException,
    completed_steps: Sequence[str],
) -> None:
    destination = session_dir / "ENGINEERING_FAILURE.json"
    if destination.exists() or destination.is_symlink():
        return
    payload = {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0a_to_d0b_engineering_failure_v1",
        "protocol_id": PROTOCOL_ID,
        "status": "engineering_or_protocol_failure",
        "created_at_utc": _utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "completed_steps": list(completed_steps),
        "scientific_result": False,
        "does_not_overwrite_science_artifacts": True,
        "d1_launched": False,
        "formal_test_allowed": False,
    }
    try:
        _publish_json_noreplace(repository, destination, payload)
    except Exception:
        pass


def dry_run(
    repository: Path = REPOSITORY,
    *,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    """Read-only completion audit; never waits, publishes, or starts a child."""

    status = build_status(repository, proc_root=proc_root)
    static_error: str | None = None
    try:
        _validate_d0b_static_contract(repository)
    except Exception as exc:
        static_error = str(exc)
    deep: dict[str, Any] = {}
    ready = static_error is None
    for dataset in DATASETS:
        state = status["datasets"][dataset]["state"]
        if state != "ready_for_deep_validation":
            ready = False
            deep[dataset] = {"ready": False, "reason": state}
            continue
        try:
            anchor = validate_dataset_completion(repository, dataset, proc_root=proc_root)
            deep[dataset] = {
                "ready": True,
                "would_publish_anchor": _display_path(_dataset_paths(repository, dataset).anchor, repository),
                "final_checkpoint_sha256": anchor["artifacts"]["final_checkpoint"]["sha256"],
                "run_contract_sha256": anchor["artifacts"]["run_contract"]["sha256"],
            }
        except Exception as exc:
            ready = False
            deep[dataset] = {"ready": False, "reason": str(exc)}
    return {
        "schema_version": 1,
        "artifact_type": "cr_sitta_d0a_to_d0b_dry_run_v1",
        "protocol_id": PROTOCOL_ID,
        "mode": "dry_run_read_only",
        "ready_to_execute": ready,
        "datasets": deep,
        "d0b_static_contract_ready": static_error is None,
        "d0b_static_contract_error": static_error,
        "planned_order": [
            "publish_three_independent_completion_anchors",
            "publish_anchor_set_and_handoff_freeze",
            "safe_export_and_verify_all_three",
            "d0b_preflight",
            "d0b_freeze",
            "per_dataset_teacher_candidate_outer_on_physical_gpu2",
            "d0b_aggregate",
            "d0b_verify",
            "publish_PIPELINE_COMPLETE_and_stop",
        ],
        "writes_performed": 0,
        "subprocesses_started": 0,
        "d1_launched": False,
        "formal_test_allowed": False,
    }


def execute_pipeline(
    repository: Path = REPOSITORY,
    *,
    proc_root: Path = Path("/proc"),
    python_executable: Path | None = None,
    command_runner: CommandRunner = _default_command_runner,
    gpu_command_runner: CommandRunner | None = None,
    sleep: Sleep = time.sleep,
    train_poll_seconds: float = TRAIN_POLL_SECONDS,
    gpu_poll_seconds: float = GPU_POLL_SECONDS,
    stale_grace_seconds: float = 300.0,
) -> dict[str, Any]:
    """Execute the finite D0-A -> D0-B pipeline; never continue into D1."""

    existing_complete = _repository_path(repository, ORCHESTRATOR_RESULT_RELATIVE / "PIPELINE_COMPLETE.json", "pipeline completion")
    if existing_complete.exists() or existing_complete.is_symlink():
        return _verify_pipeline_complete(repository)[0]
    # Full reviewed D0-B bytes and train-only Pilot64 membership are a zero-write
    # admission gate, before even creating an execution session.
    _validate_d0b_static_contract(repository)
    interpreter = Path(sys.executable if python_executable is None else python_executable).resolve(strict=True)
    # The unique session directory exists only after explicit ``execute`` and
    # gives every subsequent failure a no-overwrite receipt location.
    session_dir: Path | None = _new_session_dir(repository)
    completed_steps: list[str] = []
    try:
        _wait_for_d0a(
            repository,
            proc_root=proc_root,
            sleep=sleep,
            poll_seconds=train_poll_seconds,
            stale_grace_seconds=stale_grace_seconds,
        )
        anchors: dict[str, tuple[Mapping[str, Any], str]] = {}
        for dataset in DATASETS:
            candidate = validate_dataset_completion(repository, dataset, proc_root=proc_root)
            anchors[dataset] = publish_completion_anchor(
                repository, dataset, candidate, proc_root=proc_root
            )
            completed_steps.append(f"anchor:{dataset}")
        _anchor_set, anchor_set_sha = publish_anchor_set(repository, anchors, proc_root=proc_root)
        completed_steps.append("anchor_set")
        publish_handoff_freeze(repository, anchor_set_sha)
        completed_steps.append("handoff_freeze")
        for dataset in DATASETS:
            anchor, _digest = verify_completion_anchor(repository, dataset, proc_root=proc_root)
            paths = _dataset_paths(repository, dataset)
            safe_pair = (paths.safe_checkpoint.exists(), paths.safe_export_receipt.exists())
            if safe_pair == (True, False) or safe_pair == (False, True):
                raise OrchestrationError(f"{dataset} has a partial SAFE_EXPORT publication")
            if safe_pair == (False, False):
                command = build_export_command(
                    repository, dataset, anchor, python_executable=interpreter
                )
                _run_logged_command(
                    repository,
                    session_dir,
                    f"safe_export_{dataset}",
                    command,
                    gpu=False,
                    command_runner=command_runner,
                )
            verify_safe_export_against_anchor(repository, dataset, anchor)
            completed_steps.append(f"safe_export_verified:{dataset}")

        runner = gpu_command_runner or command_runner
        for phase in ("preflight", "freeze"):
            _run_logged_command(
                repository,
                session_dir,
                f"d0b_{phase}",
                _d0b_command(repository, interpreter, phase),
                gpu=False,
                command_runner=command_runner,
            )
            completed_steps.append(f"d0b:{phase}")

        with gpu2_exclusive_lease(
            repository,
            command_runner=runner,
            sleep=sleep,
            poll_seconds=gpu_poll_seconds,
        ) as gpu_uuid:
            # Global barriers are deliberate: every teacher is verified before
            # any candidate, and every candidate before any outer phase.
            for phase in ("teacher", "candidate", "outer"):
                for dataset in DATASETS:
                    while physical_gpu_compute_pids(
                        gpu_uuid,
                        command_runner=runner,
                    ):
                        sleep(gpu_poll_seconds)
                    _ensure_d0b_phase_parent(repository, phase, dataset)
                    _run_logged_command(
                        repository,
                        session_dir,
                        f"d0b_{dataset}_{phase}",
                        _d0b_command(repository, interpreter, phase, dataset),
                        gpu=True,
                        command_runner=command_runner,
                    )
                    completed_steps.append(f"d0b:{dataset}:{phase}")

        for phase in ("aggregate", "verify"):
            if phase == "aggregate":
                _ensure_d0b_phase_parent(repository, phase)
            _run_logged_command(
                repository,
                session_dir,
                f"d0b_{phase}",
                _d0b_command(repository, interpreter, phase),
                gpu=False,
                command_runner=command_runner,
            )
            completed_steps.append(f"d0b:{phase}")
        completion, _digest = _publish_pipeline_complete(repository)
        completed_steps.append("pipeline_complete")
        return completion
    except BaseException as exc:
        if session_dir is not None:
            _publish_failure_receipt(repository, session_dir, exc, completed_steps)
        raise


def _render_status(payload: Mapping[str, Any]) -> str:
    lines = ["CR-SITTA D0-A -> D0-B orchestrator (read-only status)"]
    for dataset in DATASETS:
        item = payload["datasets"][dataset]
        lines.append(
            f"{dataset}: {item['state']} epoch={item['epoch']} "
            f"step={item['global_optimizer_step']} pids={item['trainer_pids']} "
            f"anchor={item['anchor_exists']} safe_export={item['safe_export_receipt_exists']}"
        )
    lines.append(
        f"anchor_set={payload['anchor_set_exists']} handoff_freeze={payload['handoff_freeze_exists']} "
        f"pipeline_complete={payload['pipeline_complete_exists']}"
    )
    lines.append("D1 not launched; formal test not authorized.")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    status = commands.add_parser("status", help="read-only lightweight status (default)")
    status.add_argument("--json", action="store_true", dest="as_json")
    commands.add_parser("dry-run", help="read-only deep completion validation")
    execute = commands.add_parser("execute", help="explicitly wait and execute through D0-B verify")
    execute.add_argument("--train-poll-seconds", type=float, default=TRAIN_POLL_SECONDS)
    execute.add_argument("--gpu-poll-seconds", type=float, default=GPU_POLL_SECONDS)
    execute.add_argument("--stale-grace-seconds", type=float, default=300.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "status"
    try:
        if command == "status":
            payload = build_status()
            if getattr(args, "as_json", False):
                print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            else:
                print(_render_status(payload))
            return 0
        if command == "dry-run":
            payload = dry_run()
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 0 if payload["ready_to_execute"] else 2
        if command == "execute":
            if args.train_poll_seconds <= 0 or args.gpu_poll_seconds <= 0 or args.stale_grace_seconds < 0:
                raise OrchestrationError("poll intervals must be positive and stale grace nonnegative")
            payload = execute_pipeline(
                train_poll_seconds=args.train_poll_seconds,
                gpu_poll_seconds=args.gpu_poll_seconds,
                stale_grace_seconds=args.stale_grace_seconds,
            )
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 0
        raise OrchestrationError(f"unsupported command: {command}")
    except (OrchestrationError, OSError, ValueError, RuntimeError) as exc:
        print(f"orchestrator error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
