#!/usr/bin/env python3
"""Serial, fail-closed orchestration for the 39 formal P3 Stage-A R0 cells.

This launcher owns no scientific computation.  It invokes the two frozen
single-cell programs in this exact order for every dataset/condition cell:

1. formal label-free R0/64 candidate phase;
2. CPU verification of the candidate shard;
3. formal outer phase;
4. CPU verification of the outer shard.

Resume is artifact-driven.  An existing shard is skipped only after its
public CPU verifier accepts it.  An invalid, partial, symlinked, or otherwise
unverifiable destination is never removed or replaced and stops the grid.
All launcher records are append-only files in a unique orchestration run
directory; each file is atomically linked into place with no replacement.

Completing this launcher means only that the R0 evidence grid exists and was
verified.  It is not a paper result, does not complete the formal protocol,
does not perform scientific selection, and never authorizes Stage 2.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid
from typing import Any, Final


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.d0_v3_formal_contract import (
    CONDITIONS,
    CONFIG_FILE_SHA256,
    CONFIG_RELATIVE_PATH,
    DATASETS,
    D0V3FormalContract,
    load_d0_v3_formal_contract,
    verify_frozen_parent_bindings,
)
from tta.d0_secure_io import ensure_directory_chain_nofollow, read_stable_regular_file


SCHEMA_VERSION: Final = 1
ARTIFACT_TYPE: Final = "cr_sitta_d0_v3_formal_stage_a_r0_grid_orchestration"
DEFAULT_CONFIG: Final = PROJECT_ROOT / CONFIG_RELATIVE_PATH
CANDIDATE_SCRIPT_RELATIVE: Final = (
    "scripts/run_d0_v3_formal_stage_a_label_free.py"
)
OUTER_SCRIPT_RELATIVE: Final = "scripts/run_d0_v3_formal_stage_a_outer.py"
LAUNCHER_SCRIPT_RELATIVE: Final = "scripts/run_d0_v3_formal_stage_a_r0_grid.py"
ORCHESTRATION_RELATIVE: Final = ("orchestration", "r0_grid", "runs")
EXPECTED_CELL_COUNT: Final = 39
FORMAL_IMAGE_COUNT: Final = 64
CANDIDATE_COUNT: Final = 10
EPISODE_COUNT: Final = FORMAL_IMAGE_COUNT * CANDIDATE_COUNT
_SAFE_DEVICE_REJECT: Final = {"", ".", ".."}
_SAFE_LEAF = re.compile(r"^[A-Za-z0-9_.-]+$")

AUTHORIZATION: Final[dict[str, Any]] = {
    "source_train_derived": True,
    "paper_result": False,
    "paper_test_result": False,
    "development_test_selected_result": False,
    "scientific_selection_performed": False,
    "scientific_gate_status": "not_evaluated",
    "formal_protocol_complete": False,
    "stage2_authorized": False,
}


class D0V3R0GridLauncherError(RuntimeError):
    """The fixed R0 grid could not be safely completed or resumed."""


@dataclass(frozen=True, slots=True)
class GridCell:
    index: int
    dataset: str
    condition: str
    candidate_path: Path
    outer_path: Path

    @property
    def slug(self) -> str:
        return f"{self.index:02d}_{self.dataset}_{self.condition}"


@dataclass(frozen=True, slots=True)
class GridDefinition:
    config_path: Path
    config_sha256: str
    output_root: Path
    cells: tuple[GridCell, ...]


@dataclass(frozen=True, slots=True)
class LoggedCommand:
    sequence: int
    role: str
    log_path: Path
    log_sha256: str
    stdout: str
    stderr: str


CommandExecutor = Callable[
    [tuple[str, ...], Path, Mapping[str, str]], subprocess.CompletedProcess[str]
]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise D0V3R0GridLauncherError(
            "orchestration record is not canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_leaf(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or _SAFE_LEAF.fullmatch(value) is None
    ):
        raise D0V3R0GridLauncherError(f"{label} is not a safe file name")
    return value


def _validate_visible_device(value: str) -> str:
    if (
        not isinstance(value, str)
        or value in _SAFE_DEVICE_REJECT
        or "," in value
        or any(character.isspace() for character in value)
        or any(character in value for character in ("/", "\x00", "\r", "\n"))
    ):
        raise D0V3R0GridLauncherError(
            "--cuda-visible-device must identify exactly one safe device"
        )
    return value


def _repository_relative(project_root: Path, path: Path, *, label: str) -> str:
    root = Path(os.path.abspath(os.fspath(project_root)))
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise D0V3R0GridLauncherError(f"{label} must remain inside repository") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise D0V3R0GridLauncherError(f"{label} is not canonical")
    return relative.as_posix()


def _atomic_write_bytes_noreplace(path: Path, payload: bytes) -> str:
    """Atomically publish one immutable regular file without replacement."""

    destination = Path(os.path.abspath(os.fspath(path)))
    _safe_leaf(destination.name, label="orchestration record")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        parent_fd = os.open(destination.parent, flags)
    except OSError as exc:
        raise D0V3R0GridLauncherError(
            f"cannot securely open orchestration directory: {destination.parent}"
        ) from exc
    temporary = f".{destination.name}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    linked = False
    try:
        write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            write_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            write_flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, write_flags, 0o600, dir_fd=parent_fd)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:  # pragma: no cover - defensive kernel boundary
                raise D0V3R0GridLauncherError("short orchestration record write")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        try:
            os.link(
                temporary,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"orchestration record already exists: {destination}"
            ) from exc
        linked = True
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)
    digest = _sha256_bytes(payload)
    if not linked:
        raise D0V3R0GridLauncherError("orchestration record was not published")
    snapshot = read_stable_regular_file(destination)
    if snapshot.sha256 != digest or snapshot.data != payload:
        raise D0V3R0GridLauncherError(
            f"published orchestration record differs: {destination}"
        )
    return digest


def _atomic_write_json_noreplace(path: Path, value: Mapping[str, Any]) -> str:
    return _atomic_write_bytes_noreplace(path, _canonical_json_bytes(dict(value)))


def _new_session_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}_{uuid.uuid4().hex}"


def _mkdir_unique_session(base: Path, session_id: str) -> Path:
    _safe_leaf(session_id, label="orchestration session id")
    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    base_fd = os.open(base, flags)
    try:
        try:
            os.mkdir(session_id, mode=0o700, dir_fd=base_fd)
        except FileExistsError as exc:
            raise FileExistsError(
                f"orchestration session already exists: {base / session_id}"
            ) from exc
        os.fsync(base_fd)
    finally:
        os.close(base_fd)
    session = base / session_id
    os.mkdir(session / "commands", mode=0o700)
    os.mkdir(session / "cells", mode=0o700)
    return session


def _freeze_session(session: Path) -> None:
    for child in (session / "commands", session / "cells"):
        if child.exists() and not child.is_symlink():
            os.chmod(child, 0o555, follow_symlinks=False)
    if session.exists() and not session.is_symlink():
        os.chmod(session, 0o555, follow_symlinks=False)


def _default_executor(
    command: tuple[str, ...], cwd: Path, environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        close_fds=True,
        check=False,
    )


def _load_definition(
    config_path: Path = DEFAULT_CONFIG,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[D0V3FormalContract, GridDefinition]:
    root = Path(os.path.abspath(os.fspath(project_root)))
    config = Path(os.path.abspath(os.fspath(config_path)))
    _repository_relative(root, config, label="formal config")
    contract = load_d0_v3_formal_contract(config)
    verify_frozen_parent_bindings(contract, repository_root=root)
    if contract.config_file_sha256 != CONFIG_FILE_SHA256:
        raise D0V3R0GridLauncherError("formal config SHA differs")
    if contract.stage2_authorized:
        raise D0V3R0GridLauncherError("formal config unexpectedly authorizes Stage 2")
    if tuple(contract.datasets) != DATASETS or tuple(contract.conditions) != CONDITIONS:
        raise D0V3R0GridLauncherError("formal 3 x 13 grid differs")
    output_relative = Path(contract.output_root)
    if (
        output_relative.is_absolute()
        or not output_relative.parts
        or any(part in {"", ".", ".."} for part in output_relative.parts)
    ):
        raise D0V3R0GridLauncherError("formal output root is not canonical")
    output_root = root / output_relative
    cells: list[GridCell] = []
    for dataset in contract.datasets:
        for condition in contract.conditions:
            index = len(cells)
            cells.append(
                GridCell(
                    index=index,
                    dataset=dataset,
                    condition=condition,
                    candidate_path=(
                        output_root
                        / "candidate_phase"
                        / "shards"
                        / "R0"
                        / dataset
                        / condition
                    ),
                    outer_path=(
                        output_root
                        / "outer_phase"
                        / "shards"
                        / "R0"
                        / dataset
                        / condition
                    ),
                )
            )
    if len(cells) != EXPECTED_CELL_COUNT or len(set(cells)) != EXPECTED_CELL_COUNT:
        raise D0V3R0GridLauncherError("formal R0 grid must contain exactly 39 cells")
    return contract, GridDefinition(
        config_path=config,
        config_sha256=str(contract.config_file_sha256),
        output_root=output_root,
        cells=tuple(cells),
    )


def validate_grid_only(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contract, definition = _load_definition(config_path)
    del contract
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "valid": True,
        "mode": "validate_only_cpu_no_output",
        "config_sha256": definition.config_sha256,
        "dataset_count": len(DATASETS),
        "condition_count_per_dataset": len(CONDITIONS),
        "cell_count": len(definition.cells),
        "replicate": "R0",
        "image_count_per_cell": FORMAL_IMAGE_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "gpu_initialized": False,
        "filesystem_created": False,
        **AUTHORIZATION,
    }


class R0GridLauncher:
    """One strictly serial invocation over a fixed :class:`GridDefinition`."""

    def __init__(
        self,
        *,
        definition: GridDefinition,
        project_root: Path,
        python_executable: Path,
        cuda_visible_device: str,
        executor: CommandExecutor = _default_executor,
        session_id: str | None = None,
    ) -> None:
        self.definition = definition
        self.project_root = Path(os.path.abspath(os.fspath(project_root)))
        self.python_executable = Path(
            os.path.abspath(os.fspath(python_executable))
        )
        self.cuda_visible_device = _validate_visible_device(cuda_visible_device)
        self.executor = executor
        self.session_id = session_id or _new_session_id()
        _safe_leaf(self.session_id, label="orchestration session id")
        if not self.python_executable.is_file() or not os.access(
            self.python_executable, os.X_OK
        ):
            raise D0V3R0GridLauncherError(
                f"Python executable is unavailable: {self.python_executable}"
            )
        if len(definition.cells) != EXPECTED_CELL_COUNT:
            raise D0V3R0GridLauncherError("launcher accepts only the full 39-cell grid")
        for expected_index, cell in enumerate(definition.cells):
            if cell.index != expected_index:
                raise D0V3R0GridLauncherError("grid cell order/index differs")
        self.candidate_script = self.project_root / CANDIDATE_SCRIPT_RELATIVE
        self.outer_script = self.project_root / OUTER_SCRIPT_RELATIVE
        self.launcher_script = self.project_root / LAUNCHER_SCRIPT_RELATIVE
        self.code_seals = self._capture_code_seals()
        self.sequence = 0
        self.session: Path | None = None

    def _capture_code_seals(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for relative, path in (
            (CANDIDATE_SCRIPT_RELATIVE, self.candidate_script),
            (OUTER_SCRIPT_RELATIVE, self.outer_script),
            (LAUNCHER_SCRIPT_RELATIVE, self.launcher_script),
        ):
            snapshot = read_stable_regular_file(path)
            values[relative] = snapshot.sha256
        return values

    def _assert_code_seals(self) -> None:
        for relative, expected in self.code_seals.items():
            observed = read_stable_regular_file(self.project_root / relative).sha256
            if observed != expected:
                raise D0V3R0GridLauncherError(
                    f"orchestration code changed during grid: {relative}"
                )

    @staticmethod
    def _environment(*, cuda_visible_devices: str) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            }
        )
        return environment

    def _run_and_log(
        self,
        *,
        role: str,
        cell: GridCell | None,
        command: Sequence[str],
        cuda_visible_devices: str,
    ) -> LoggedCommand:
        if self.session is None:
            raise D0V3R0GridLauncherError("orchestration session is not initialized")
        self._assert_code_seals()
        argv = tuple(str(value) for value in command)
        if len(argv) < 2 or Path(argv[1]) not in {
            self.candidate_script,
            self.outer_script,
        }:
            raise D0V3R0GridLauncherError("launcher may invoke only the two fixed workers")
        sequence = self.sequence
        self.sequence += 1
        started = datetime.now(timezone.utc).isoformat()
        launch_error: BaseException | None = None
        try:
            completed = self.executor(
                argv,
                self.project_root,
                self._environment(cuda_visible_devices=cuda_visible_devices),
            )
            returncode = int(completed.returncode)
            stdout = str(completed.stdout or "")
            stderr = str(completed.stderr or "")
        except BaseException as exc:  # preserve a durable launch-error record
            launch_error = exc
            returncode = -1
            stdout = ""
            stderr = f"{type(exc).__name__}: {exc}"
        finished = datetime.now(timezone.utc).isoformat()
        cell_slug = "grid" if cell is None else cell.slug
        filename = _safe_leaf(
            f"{sequence:04d}_{cell_slug}_{role}.json", label="command log"
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": f"{ARTIFACT_TYPE}_command",
            "session_id": self.session_id,
            "sequence": sequence,
            "role": role,
            "cell": (
                None
                if cell is None
                else {
                    "index": cell.index,
                    "dataset": cell.dataset,
                    "condition": cell.condition,
                    "replicate": "R0",
                }
            ),
            "command": list(argv),
            "command_sha256": _sha256_bytes(_canonical_json_bytes(list(argv))),
            "environment_overrides": {
                "PYTHONHASHSEED": "42",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            },
            "started_at": started,
            "finished_at": finished,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "atomic_no_replace": True,
            **AUTHORIZATION,
        }
        log_path = self.session / "commands" / filename
        log_sha = _atomic_write_json_noreplace(log_path, payload)
        if launch_error is not None:
            raise D0V3R0GridLauncherError(
                f"worker launch failed for {role}; log={log_path}"
            ) from launch_error
        if returncode != 0:
            raise D0V3R0GridLauncherError(
                f"worker command failed for {role} with exit {returncode}; log={log_path}"
            )
        return LoggedCommand(
            sequence=sequence,
            role=role,
            log_path=log_path,
            log_sha256=log_sha,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _parse_json_stdout(command: LoggedCommand, *, label: str) -> Mapping[str, Any]:
        try:
            value = json.loads(command.stdout)
        except json.JSONDecodeError as exc:
            raise D0V3R0GridLauncherError(
                f"{label} did not emit one JSON object; log={command.log_path}"
            ) from exc
        if not isinstance(value, Mapping):
            raise D0V3R0GridLauncherError(f"{label} JSON root is not an object")
        return value

    def _candidate_verify(self, cell: GridCell) -> LoggedCommand:
        command = self._run_and_log(
            role="candidate_cpu_verify",
            cell=cell,
            command=(
                str(self.python_executable),
                str(self.candidate_script),
                "verify",
                "--config",
                str(self.definition.config_path),
                "--path",
                str(cell.candidate_path),
            ),
            cuda_visible_devices="",
        )
        value = self._parse_json_stdout(command, label="candidate CPU verifier")
        expected = {
            "valid": True,
            "formal": True,
            "dry_run": False,
            "episode_count": EPISODE_COUNT,
            "formal_protocol_complete": False,
            "stage2_authorized": False,
        }
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise D0V3R0GridLauncherError(
                f"candidate CPU verifier summary differs; log={command.log_path}"
            )
        return command

    def _outer_verify(self, cell: GridCell) -> LoggedCommand:
        command = self._run_and_log(
            role="outer_cpu_verify",
            cell=cell,
            command=(
                str(self.python_executable),
                str(self.outer_script),
                "--config",
                str(self.definition.config_path),
                "verify",
                "--dataset",
                cell.dataset,
                "--condition",
                cell.condition,
            ),
            cuda_visible_devices="",
        )
        value = self._parse_json_stdout(command, label="outer CPU verifier")
        expected = {
            "dataset": cell.dataset,
            "condition": cell.condition,
            "replicate": "R0",
            "image_count": FORMAL_IMAGE_COUNT,
            "candidate_count": CANDIDATE_COUNT,
            "record_count": EPISODE_COUNT,
        }
        if any(value.get(key) != expected_value for key, expected_value in expected.items()):
            raise D0V3R0GridLauncherError(
                f"outer CPU verifier summary differs; log={command.log_path}"
            )
        return command

    def _process_cell(self, cell: GridCell) -> dict[str, Any]:
        candidate_existed = cell.candidate_path.exists() or cell.candidate_path.is_symlink()
        if not candidate_existed:
            self._run_and_log(
                role="candidate_gpu_run",
                cell=cell,
                command=(
                    str(self.python_executable),
                    str(self.candidate_script),
                    "run",
                    "--config",
                    str(self.definition.config_path),
                    "--dataset",
                    cell.dataset,
                    "--condition",
                    cell.condition,
                    "--replicate",
                    "R0",
                    "--max-images",
                    "64",
                    "--cuda-visible-device",
                    self.cuda_visible_device,
                ),
                cuda_visible_devices=self.cuda_visible_device,
            )
            if not (cell.candidate_path.exists() or cell.candidate_path.is_symlink()):
                raise D0V3R0GridLauncherError(
                    "candidate worker returned success without publishing its fixed destination"
                )
        candidate_verified = self._candidate_verify(cell)

        outer_existed = cell.outer_path.exists() or cell.outer_path.is_symlink()
        if not outer_existed:
            self._run_and_log(
                role="outer_gpu_run",
                cell=cell,
                command=(
                    str(self.python_executable),
                    str(self.outer_script),
                    "--config",
                    str(self.definition.config_path),
                    "run",
                    "--dataset",
                    cell.dataset,
                    "--condition",
                    cell.condition,
                    "--device",
                    "cuda:0",
                ),
                cuda_visible_devices=self.cuda_visible_device,
            )
            if not (cell.outer_path.exists() or cell.outer_path.is_symlink()):
                raise D0V3R0GridLauncherError(
                    "outer worker returned success without publishing its fixed destination"
                )
        outer_verified = self._outer_verify(cell)
        return {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": f"{ARTIFACT_TYPE}_cell_status",
            "session_id": self.session_id,
            "cell": {
                "index": cell.index,
                "dataset": cell.dataset,
                "condition": cell.condition,
                "replicate": "R0",
            },
            "candidate_phase": {
                "path": _repository_relative(
                    self.project_root, cell.candidate_path, label="candidate shard"
                ),
                "disposition": "existing_verified" if candidate_existed else "ran_verified",
                "cpu_verified": True,
                "verification_log": _repository_relative(
                    self.project_root,
                    candidate_verified.log_path,
                    label="candidate verification log",
                ),
                "verification_log_sha256": candidate_verified.log_sha256,
            },
            "outer_phase": {
                "path": _repository_relative(
                    self.project_root, cell.outer_path, label="outer shard"
                ),
                "disposition": "existing_verified" if outer_existed else "ran_verified",
                "cpu_verified": True,
                "verification_log": _repository_relative(
                    self.project_root,
                    outer_verified.log_path,
                    label="outer verification log",
                ),
                "verification_log_sha256": outer_verified.log_sha256,
            },
            "serial_cell_complete": True,
            "atomic_no_replace": True,
            **AUTHORIZATION,
        }

    def run(self) -> dict[str, Any]:
        orchestration_parts = (
            *self.definition.output_root.relative_to(self.project_root).parts,
            *ORCHESTRATION_RELATIVE,
        )
        base = ensure_directory_chain_nofollow(self.project_root, orchestration_parts)
        self.session = _mkdir_unique_session(base, self.session_id)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "session_id": self.session_id,
            "mode": "formal_stage_a_r0_full_39_cell_serial_resumable",
            "config_path": _repository_relative(
                self.project_root, self.definition.config_path, label="formal config"
            ),
            "config_sha256": self.definition.config_sha256,
            "python_executable": str(self.python_executable),
            "cuda_visible_device": self.cuda_visible_device,
            "replicate": "R0",
            "dataset_count": len(DATASETS),
            "condition_count_per_dataset": len(CONDITIONS),
            "cell_count": len(self.definition.cells),
            "execution_order": "dataset_major_then_condition_serial",
            "resume_policy": "verify_existing_or_run_missing_fail_closed",
            "overwrite_policy": "forbidden",
            "candidate_script": CANDIDATE_SCRIPT_RELATIVE,
            "outer_script": OUTER_SCRIPT_RELATIVE,
            "code_seals": dict(sorted(self.code_seals.items())),
            "cells": [
                {
                    "index": cell.index,
                    "dataset": cell.dataset,
                    "condition": cell.condition,
                    "candidate_path": _repository_relative(
                        self.project_root, cell.candidate_path, label="candidate shard"
                    ),
                    "outer_path": _repository_relative(
                        self.project_root, cell.outer_path, label="outer shard"
                    ),
                }
                for cell in self.definition.cells
            ],
            "atomic_no_replace": True,
            **AUTHORIZATION,
        }
        _atomic_write_json_noreplace(self.session / "MANIFEST.json", manifest)
        counts = {
            "candidate_ran_verified": 0,
            "candidate_existing_verified": 0,
            "outer_ran_verified": 0,
            "outer_existing_verified": 0,
        }
        try:
            preflight = self._run_and_log(
                role="candidate_contract_cpu_validate",
                cell=None,
                command=(
                    str(self.python_executable),
                    str(self.candidate_script),
                    "validate",
                    "--config",
                    str(self.definition.config_path),
                ),
                cuda_visible_devices="",
            )
            preflight_value = self._parse_json_stdout(
                preflight, label="candidate contract CPU validation"
            )
            required_preflight = {
                "valid": True,
                "config_sha256": self.definition.config_sha256,
                "gpu_initialized": False,
                "target_payload_opened": False,
                "validation_payload_opened": False,
                "test_payload_opened": False,
                "formal_protocol_complete": False,
                "stage2_authorized": False,
            }
            if any(
                preflight_value.get(key) != expected
                for key, expected in required_preflight.items()
            ):
                raise D0V3R0GridLauncherError(
                    f"candidate contract preflight differs; log={preflight.log_path}"
                )
            for cell in self.definition.cells:
                status = self._process_cell(cell)
                candidate_disposition = status["candidate_phase"]["disposition"]
                outer_disposition = status["outer_phase"]["disposition"]
                counts[f"candidate_{candidate_disposition}"] += 1
                counts[f"outer_{outer_disposition}"] += 1
                _atomic_write_json_noreplace(
                    self.session / "cells" / f"{cell.slug}.json", status
                )
            complete = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": f"{ARTIFACT_TYPE}_complete",
                "complete": True,
                "session_id": self.session_id,
                "r0_grid_complete": True,
                "verified_cell_count": len(self.definition.cells),
                "counts": counts,
                "command_count": self.sequence,
                "resume_policy": "verify_existing_or_run_missing_fail_closed",
                "atomic_no_replace": True,
                **AUTHORIZATION,
            }
            complete_sha = _atomic_write_json_noreplace(
                self.session / "COMPLETE.json", complete
            )
            result = {
                **complete,
                "path": str(self.session),
                "complete_sha256": complete_sha,
            }
            _freeze_session(self.session)
            return result
        except BaseException as exc:
            failure = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": f"{ARTIFACT_TYPE}_failure",
                "complete": False,
                "session_id": self.session_id,
                "r0_grid_complete": False,
                "completed_cell_status_count": len(list((self.session / "cells").glob("*.json"))),
                "command_count": self.sequence,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "no_artifact_was_removed_or_replaced": True,
                "atomic_no_replace": True,
                **AUTHORIZATION,
            }
            try:
                _atomic_write_json_noreplace(self.session / "FAILED.json", failure)
                _freeze_session(self.session)
            except BaseException as record_error:
                raise D0V3R0GridLauncherError(
                    "grid failed and immutable failure record could not be "
                    f"published: {self.session}"
                ) from record_error
            raise


def run_grid(
    *,
    config_path: Path,
    cuda_visible_device: str,
    python_executable: Path,
) -> dict[str, Any]:
    _contract, definition = _load_definition(config_path)
    launcher = R0GridLauncher(
        definition=definition,
        project_root=PROJECT_ROOT,
        python_executable=python_executable,
        cuda_visible_device=cuda_visible_device,
    )
    return launcher.run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate", help="CPU-only exact-contract and 39-cell plan validation"
    )
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run = subparsers.add_parser(
        "run", help="serially complete/resume and verify the fixed R0 grid"
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--cuda-visible-device", required=True)
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        value = validate_grid_only(Path(os.path.abspath(os.fspath(args.config))))
    else:
        value = run_grid(
            config_path=Path(os.path.abspath(os.fspath(args.config))),
            cuda_visible_device=args.cuda_visible_device,
            python_executable=Path(os.path.abspath(os.fspath(args.python))),
        )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
