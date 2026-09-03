#!/usr/bin/env python3
"""Run the non-scientific D0-v2 three-process GPU engineering smoke.

The public ``validate`` and ``verify`` commands are CPU-only.  ``run`` keeps
CUDA uninitialized in the parent, launches three fresh subprocesses
sequentially, and each child independently builds/runs all ten optimizer/LR
candidates on the frozen NUAA-SIRST clean Pilot64 image at index zero.

Successful publication is an atomic no-replace directory named
``engineering_smoke``.  It is not a paper result, does not perform scientific
selection, does not complete formal P3, and cannot authorize Stage 2.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import subprocess
import sys
import tempfile
from typing import Any
import uuid

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.d0_v2_independent_candidate_contract import (
    FROZEN_CANDIDATES,
    IndependentCandidateExecutionLedger,
    IndependentCandidateReceipt,
)
from analysis.d0_v2_protocol_contract import (
    D0V2ProtocolContract,
    load_d0_v2_protocol_contract,
    verify_sealed_v1_inputs,
)
from analysis.d0_v2_smoke_repro_contract import (
    FreshSubprocessIdentity,
    aggregate_three,
    build_d0_v2_smoke_process_receipt,
    canonical_d0_v2_smoke_process_receipt_bytes,
    canonical_d0_v2_smoke_repro_aggregate_bytes,
    d0_v2_smoke_repro_aggregate_sha256,
    validate_d0_v2_smoke_process_receipt,
    validate_d0_v2_smoke_repro_aggregate,
)
from tta.d0_secure_io import (
    ensure_directory_chain_nofollow,
    publish_directory_noreplace,
    publish_file_noreplace,
    read_stable_regular_file,
    snapshot_regular_directory,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs/tent_failure_diagnostics_v2_independent_candidates.yaml"
)
ENGINEERING_SMOKE_DIRECTORY = "engineering_smoke"
PROCESS_IDS = ("replicate_0", "replicate_1", "replicate_2")
PROCESS_FILENAMES = tuple(f"{value}.json" for value in PROCESS_IDS)
AGGREGATE_FILENAME = "aggregate.json"
COMPLETE_FILENAME = "COMPLETE.json"
EXPECTED_DIRECTORY_MEMBERS = frozenset(
    (*PROCESS_FILENAMES, AGGREGATE_FILENAME, COMPLETE_FILENAME)
)
COMPLETE_ARTIFACT_TYPE = "cr_sitta_d0_v2_engineering_smoke_complete"
TENSOR_HASH_CONTRACT = "cr-sitta-d0-v2-strided-tensor-v1"
SOURCE_STATE_HASH_CONTRACT = (
    "cr-sitta-d0-v2-source-state-excluding-candidate-optimizer-v1"
)

ENV_PROCESS_ID = "CR_SITTA_D0_V2_PROCESS_ID"
ENV_PARENT_NONCE = "CR_SITTA_D0_V2_PARENT_RUN_NONCE"
ENV_CHILD_NONCE = "CR_SITTA_D0_V2_CHILD_LAUNCH_NONCE"
ENV_COMMAND_SHA256 = "CR_SITTA_D0_V2_COMMAND_SHA256"
ENV_CONFIG_SHA256 = "CR_SITTA_D0_V2_CONFIG_SHA256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_VISIBLE_DEVICE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")


class D0V2EngineeringSmokeError(RuntimeError):
    """The engineering smoke failed before canonical publication."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as exc:
        raise D0V2EngineeringSmokeError(
            "engineering smoke artifact is not canonical-JSON serializable"
        ) from exc


def _strict_json_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise D0V2EngineeringSmokeError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=pairs_hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D0V2EngineeringSmokeError(f"{label} is invalid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise D0V2EngineeringSmokeError(f"{label} root must be an object")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise D0V2EngineeringSmokeError(
            f"{label} must be lowercase 64-hex SHA-256"
        )
    return value


def _command_sha256(command: Sequence[str]) -> str:
    if (
        isinstance(command, (str, bytes))
        or not command
        or not all(isinstance(value, str) and value for value in command)
    ):
        raise D0V2EngineeringSmokeError("worker command vector is invalid")
    return _sha256(_canonical_json_bytes(list(command)))


def _tensor_sha256(value: Any) -> str:
    import torch

    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise D0V2EngineeringSmokeError("tensor hash requires a strided tensor")
    tensor = value.detach().cpu().contiguous()
    if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all()):
        raise D0V2EngineeringSmokeError(
            "tensor hash requires finite floating-point values"
        )
    components = (
        str(tensor.dtype).encode("ascii"),
        ",".join(str(value) for value in tensor.shape).encode("ascii"),
        tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
    )
    digest = hashlib.sha256()
    digest.update(TENSOR_HASH_CONTRACT.encode("ascii") + b"\0")
    for component in components:
        digest.update(len(component).to_bytes(8, "big"))
        digest.update(component)
    return digest.hexdigest()


def _candidate_invariant_source_state_sha256(fingerprint: Any) -> str:
    """Hash complete Source state except candidate-specific optimizer config."""

    fields = {
        "hash_contract": SOURCE_STATE_HASH_CONTRACT,
        "model_sha256": _require_sha256(
            getattr(fingerprint, "model_sha256", None),
            label="source.model_sha256",
        ),
        "runtime_sha256": _require_sha256(
            getattr(fingerprint, "runtime_sha256", None),
            label="source.runtime_sha256",
        ),
        "topology_sha256": _require_sha256(
            getattr(fingerprint, "topology_sha256", None),
            label="source.topology_sha256",
        ),
        "gradients_sha256": _require_sha256(
            getattr(fingerprint, "gradients_sha256", None),
            label="source.gradients_sha256",
        ),
        "extras_sha256": _require_sha256(
            getattr(fingerprint, "extras_sha256", None),
            label="source.extras_sha256",
        ),
    }
    return _sha256(_canonical_json_bytes(fields))


def _load_contract(path: Path) -> D0V2ProtocolContract:
    contract = load_d0_v2_protocol_contract(path)
    verify_sealed_v1_inputs(contract, repository_root=PROJECT_ROOT)
    smoke = contract.raw["engineering_smoke"]
    if (
        smoke["role"] != "engineering_smoke"
        or smoke["paper_result"] is not False
        or smoke["paper_test_result"] is not False
        or smoke["scientific_selection_performed"] is not False
        or smoke["formal_p3_complete"] is not False
        or smoke["stage2_authorized"] is not False
    ):
        raise D0V2EngineeringSmokeError(
            "engineering smoke authorization boundary drifted"
        )
    if contract.config_file_sha256 is None:
        raise D0V2EngineeringSmokeError(
            "engineering smoke requires a file-backed config SHA-256"
        )
    return contract


def _output_paths(contract: D0V2ProtocolContract) -> tuple[Path, Path]:
    raw_output = contract.raw["output"]
    root = PROJECT_ROOT / str(raw_output["root"])
    expected_root = (
        PROJECT_ROOT
        / "results"
        / "cr_sitta"
        / "tent_failure_diagnostics_v2_independent_candidates"
    )
    if root.absolute() != expected_root:
        raise D0V2EngineeringSmokeError("D0-v2 output root drifted")
    relative = str(raw_output["engineering_smoke"])
    publication = contract.raw["engineering_smoke"]["publication"]
    if (
        relative != ENGINEERING_SMOKE_DIRECTORY
        or publication["relative_directory"] != relative
    ):
        raise D0V2EngineeringSmokeError(
            "engineering smoke output directory drifted"
        )
    return root, root / relative


def _capture_critical_source_seal(
    contract: D0V2ProtocolContract,
) -> tuple[dict[str, tuple[Any, ...]], str]:
    paths = contract.raw["engineering_smoke"]["runtime_binding"][
        "critical_code_paths"
    ]
    seal: dict[str, tuple[Any, ...]] = {}
    digest_input: dict[str, str] = {}
    for relative in paths:
        snapshot = read_stable_regular_file(PROJECT_ROOT / str(relative))
        seal[str(relative)] = (
            snapshot.sha256,
            snapshot.device,
            snapshot.inode,
            snapshot.mode,
            snapshot.link_count,
            snapshot.size_bytes,
            snapshot.mtime_ns,
            snapshot.ctime_ns,
        )
        digest_input[str(relative)] = snapshot.sha256
    return seal, _sha256(_canonical_json_bytes(digest_input))


def _assert_critical_source_seal(
    contract: D0V2ProtocolContract,
    expected: Mapping[str, tuple[Any, ...]],
) -> None:
    current, _digest = _capture_critical_source_seal(contract)
    if dict(current) != dict(expected):
        raise D0V2EngineeringSmokeError(
            "D0-v2 engineering smoke critical source changed during execution"
        )


@dataclass(frozen=True)
class _MethodFacingInputSeal:
    identities: tuple[tuple[str, tuple[Any, ...]], ...]
    seal_sha256: str
    cache_root: Path
    complete: Mapping[str, Any]


def _snapshot_identity(snapshot: Any) -> tuple[Any, ...]:
    return (
        snapshot.sha256,
        snapshot.device,
        snapshot.inode,
        snapshot.mode,
        snapshot.link_count,
        snapshot.size_bytes,
        snapshot.mtime_ns,
        snapshot.ctime_ns,
    )


def _strict_yaml_mapping(data: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise D0V2EngineeringSmokeError(
            f"{label} is not valid UTF-8 YAML"
        ) from exc
    if not isinstance(value, Mapping):
        raise D0V2EngineeringSmokeError(f"{label} root must be a mapping")
    return value


def _project_relative_path(value: Any, *, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise D0V2EngineeringSmokeError(f"{label} must be a non-empty path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise D0V2EngineeringSmokeError(
            f"{label} must be a canonical project-relative path"
        )
    path = PROJECT_ROOT / relative
    if not path.absolute().is_relative_to(PROJECT_ROOT):
        raise D0V2EngineeringSmokeError(f"{label} escapes the project root")
    return relative.as_posix(), path


def _capture_method_facing_input_seal(
    contract: D0V2ProtocolContract,
    *,
    config_path: Path,
) -> _MethodFacingInputSeal:
    """Seal only the fixed method-facing NUAA clean input lineage.

    Deliberately excluded: every test split file and the train-target payload.
    Manifest metadata is read to prove the target firewall, but
    ``outer_evaluator/targets.npy`` is never opened, hashed, deserialized, or
    indexed by this path.
    """

    smoke = contract.raw["engineering_smoke"]
    runtime_binding = smoke["runtime_binding"]
    if (
        runtime_binding["method_facing_input_seal"] is not True
        or runtime_binding["frozen_v2_full_payload_seal_reused"] is not False
        or runtime_binding["target_payload_bytes_opened"] != 0
        or runtime_binding["target_payload_deserialized"] is not False
        or runtime_binding["test_split_files_opened"] != 0
        or runtime_binding["test_images_opened"] != 0
        or runtime_binding["test_labels_opened"] != 0
    ):
        raise D0V2EngineeringSmokeError(
            "method-facing input seal boundary drifted"
        )

    snapshots: dict[str, Any] = {}

    def capture(
        role: str,
        path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> Any:
        if role in snapshots:
            raise D0V2EngineeringSmokeError(
                f"duplicate method-facing input role: {role}"
            )
        snapshot = read_stable_regular_file(path)
        if expected_sha256 is not None and snapshot.sha256 != _require_sha256(
            expected_sha256, label=f"{role}.expected_sha256"
        ):
            raise D0V2EngineeringSmokeError(
                f"method-facing input SHA-256 differs: {role}"
            )
        snapshots[role] = snapshot
        return snapshot

    config_snapshot = capture(
        "d0_v2_config",
        config_path,
        expected_sha256=str(contract.config_file_sha256),
    )
    del config_snapshot

    cache_config = contract.raw["cache"]
    execution_relative, execution_path = _project_relative_path(
        cache_config["execution_protocol_path"],
        label="cache.execution_protocol_path",
    )
    execution_snapshot = capture(
        f"protocol:{execution_relative}",
        execution_path,
        expected_sha256=str(cache_config["execution_protocol_sha256"]),
    )
    execution = _strict_yaml_mapping(
        execution_snapshot.data, label="frozen v2 execution protocol"
    )
    execution_cache = execution.get("cache_protocol")
    if not isinstance(execution_cache, Mapping):
        raise D0V2EngineeringSmokeError(
            "frozen v2 execution cache section is invalid"
        )
    expected_cache_link = {
        "path": cache_config["protocol_path"],
        "sha256": cache_config["protocol_sha256"],
        "root": cache_config["root"],
    }
    if any(
        execution_cache.get(key) != expected
        for key, expected in expected_cache_link.items()
    ):
        raise D0V2EngineeringSmokeError(
            "frozen v2 execution/cache protocol binding differs"
        )
    artifacts = execution_cache.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise D0V2EngineeringSmokeError(
            "frozen v2 execution artifact anchors are invalid"
        )
    artifact = artifacts.get("NUAA-SIRST")
    if not isinstance(artifact, Mapping):
        raise D0V2EngineeringSmokeError(
            "NUAA-SIRST cache artifact anchors are unavailable"
        )

    protocol_relative, protocol_path = _project_relative_path(
        cache_config["protocol_path"], label="cache.protocol_path"
    )
    protocol_snapshot = capture(
        f"protocol:{protocol_relative}",
        protocol_path,
        expected_sha256=str(cache_config["protocol_sha256"]),
    )
    cache_protocol = _strict_yaml_mapping(
        protocol_snapshot.data, label="cache protocol"
    )
    protocol_datasets = cache_protocol.get("datasets")
    dataset_protocol = (
        protocol_datasets.get("NUAA-SIRST")
        if isinstance(protocol_datasets, Mapping)
        else None
    )
    if not isinstance(dataset_protocol, Mapping):
        raise D0V2EngineeringSmokeError(
            "NUAA-SIRST cache protocol is unavailable"
        )
    dataset_config = contract.raw["datasets"]["NUAA-SIRST"]
    if dataset_protocol.get("train_split_sha256") != dataset_config[
        "train_split_sha256"
    ]:
        raise D0V2EngineeringSmokeError("NUAA train split binding differs")

    for role, path_field, hash_field in (
        ("source_train_split", "train_split", "train_split_sha256"),
        ("frozen_pilot_ids", "calibration_ids", "calibration_ids_file_sha256"),
    ):
        relative, path = _project_relative_path(
            dataset_protocol.get(path_field), label=f"cache.{path_field}"
        )
        capture(
            f"{role}:{relative}",
            path,
            expected_sha256=str(dataset_protocol.get(hash_field)),
        )

    checkpoint_relative, checkpoint_path = _project_relative_path(
        dataset_config["checkpoint_path"], label="checkpoint_path"
    )
    capture(
        f"source_checkpoint:{checkpoint_relative}",
        checkpoint_path,
        expected_sha256=str(dataset_config["checkpoint_sha256"]),
    )

    cache_root = PROJECT_ROOT / str(cache_config["root"]) / "NUAA-SIRST"
    cache_files = {
        "cache_manifest": ("manifest.json", "manifest_sha256"),
        "cache_method_manifest": (
            "method_input_manifest.json",
            "method_input_manifest_sha256",
        ),
        "cache_complete": ("COMPLETE.json", "complete_sha256"),
    }
    cache_snapshots: dict[str, Any] = {}
    for role, (filename, anchor) in cache_files.items():
        expected = artifact.get(anchor)
        cache_snapshots[role] = capture(
            f"{role}:{filename}",
            cache_root / filename,
            expected_sha256=str(expected),
        )

    complete = _strict_json_bytes(
        cache_snapshots["cache_complete"].data, label="NUAA cache COMPLETE"
    )
    required_complete = {
        "complete": True,
        "dataset": "NUAA-SIRST",
        "protocol_sha256": cache_config["protocol_sha256"],
        "manifest_sha256": artifact.get("manifest_sha256"),
        "method_input_manifest_sha256": artifact.get(
            "method_input_manifest_sha256"
        ),
        "cache_content_sha256": artifact.get("cache_content_sha256"),
        "method_received_labels": False,
        "test_images_opened": 0,
        "test_masks_opened": 0,
    }
    complete_mismatches = {
        key: {"expected": expected, "observed": complete.get(key)}
        for key, expected in required_complete.items()
        if complete.get(key) != expected
    }
    if complete_mismatches:
        raise D0V2EngineeringSmokeError(
            f"NUAA cache COMPLETE boundary differs: {complete_mismatches}"
        )

    manifest = _strict_json_bytes(
        cache_snapshots["cache_manifest"].data, label="NUAA cache manifest"
    )
    method_manifest = _strict_json_bytes(
        cache_snapshots["cache_method_manifest"].data,
        label="NUAA method manifest",
    )
    frozen_sample = smoke["sample"]
    image_ids = manifest.get("image_ids")
    original_sizes = manifest.get("original_sizes")
    if (
        manifest.get("dataset") != "NUAA-SIRST"
        or manifest.get("protocol_sha256") != cache_config["protocol_sha256"]
        or manifest.get("train_split_sha256")
        != dataset_config["train_split_sha256"]
        or manifest.get("cache_content_sha256")
        != artifact.get("cache_content_sha256")
        or not isinstance(image_ids, list)
        or len(image_ids) != 64
        or image_ids[0] != frozen_sample["image_id"]
        or not isinstance(original_sizes, list)
        or len(original_sizes) != 64
        or original_sizes[0] != list(frozen_sample["original_size"])
    ):
        raise D0V2EngineeringSmokeError(
            "NUAA cache/sample manifest binding differs"
        )
    if (
        method_manifest.get("dataset") != "NUAA-SIRST"
        or method_manifest.get("outer_manifest_sha256")
        != artifact.get("manifest_sha256")
        or method_manifest.get("targets_exposed") is not False
        or method_manifest.get("image_ids") != image_ids
    ):
        raise D0V2EngineeringSmokeError(
            "NUAA method-facing target firewall or lineage differs"
        )

    clean_relative = "conditions/clean_S0.npy"
    manifest_files = manifest.get("files")
    method_files = method_manifest.get("files")
    manifest_clean = (
        manifest_files.get(clean_relative)
        if isinstance(manifest_files, Mapping)
        else None
    )
    method_clean = (
        method_files.get(clean_relative)
        if isinstance(method_files, Mapping)
        else None
    )
    if (
        not isinstance(manifest_clean, Mapping)
        or not isinstance(method_clean, Mapping)
        or dict(manifest_clean) != dict(method_clean)
    ):
        raise D0V2EngineeringSmokeError(
            "NUAA clean method-facing shard binding differs"
        )
    clean_snapshot = capture(
        f"method_input:{clean_relative}",
        cache_root / clean_relative,
        expected_sha256=str(manifest_clean.get("sha256")),
    )
    expected_bytes = manifest_clean.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or clean_snapshot.size_bytes != expected_bytes
    ):
        raise D0V2EngineeringSmokeError(
            "NUAA clean method-facing shard byte size differs"
        )

    identities = tuple(
        (role, _snapshot_identity(snapshot))
        for role, snapshot in sorted(snapshots.items())
    )
    digest_input = {role: identity[0] for role, identity in identities}
    return _MethodFacingInputSeal(
        identities=identities,
        seal_sha256=_sha256(_canonical_json_bytes(digest_input)),
        cache_root=cache_root,
        complete=complete,
    )


def _runtime_sha256(
    *,
    method_facing_input_seal_sha256: str,
    critical_source_sha256: str,
    determinism_sha256: str,
    device: Any,
) -> str:
    import torch

    value = {
        "schema_version": 2,
        "method_facing_input_seal_sha256": _require_sha256(
            method_facing_input_seal_sha256,
            label="method_facing_input_seal_sha256",
        ),
        "critical_source_sha256": _require_sha256(
            critical_source_sha256, label="critical_source_sha256"
        ),
        "determinism_sha256": _require_sha256(
            determinism_sha256, label="determinism_sha256"
        ),
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version()),
        "device_type": str(device.type),
        "device_name": str(torch.cuda.get_device_name(device)),
        "device_capability": list(torch.cuda.get_device_capability(device)),
    }
    return _sha256(_canonical_json_bytes(value))


def validate_contract_only(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """CPU-only validation; performs no CUDA query and creates no directory."""

    contract = _load_contract(config_path)
    root, destination = _output_paths(contract)
    smoke = contract.raw["engineering_smoke"]
    return {
        "schema_version": 2,
        "role": "engineering_smoke_validate_only",
        "valid": True,
        "config_sha256": contract.config_file_sha256,
        "sample": dict(smoke["sample"]),
        "candidate_count_per_process": smoke["execution"][
            "candidate_count_per_process"
        ],
        "fresh_process_count": smoke["execution"]["fresh_process_count"],
        "output_root_registered": root.relative_to(PROJECT_ROOT).as_posix(),
        "destination_registered": destination.relative_to(PROJECT_ROOT).as_posix(),
        "filesystem_created": False,
        "gpu_initialized": False,
        "paper_result": False,
        "formal_p3_complete": False,
        "stage2_authorized": False,
    }


def _configure_cuda_worker() -> Any:
    import numpy as np
    import torch

    expected_environment = {
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    mismatches = {
        key: {"expected": expected, "observed": os.environ.get(key)}
        for key, expected in expected_environment.items()
        if os.environ.get(key) != expected
    }
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        not isinstance(visible, str)
        or _VISIBLE_DEVICE_RE.fullmatch(visible) is None
        or "," in visible
    ):
        mismatches["CUDA_VISIBLE_DEVICES"] = {
            "expected": "exactly one safe device identifier",
            "observed": visible,
        }
    if mismatches:
        raise D0V2EngineeringSmokeError(
            f"worker CUDA environment is not frozen: {mismatches}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise D0V2EngineeringSmokeError(
            "worker must see exactly one available CUDA device"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise D0V2EngineeringSmokeError(
            "worker failed to establish strict deterministic forwards"
        )
    return device


def _seed_candidate() -> None:
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def _validate_worker_environment(
    *,
    process_id: str,
    parent_run_nonce: str,
    child_launch_nonce: str,
    config_sha256: str,
) -> str:
    bindings = {
        ENV_PROCESS_ID: process_id,
        ENV_PARENT_NONCE: _require_sha256(
            parent_run_nonce, label="parent_run_nonce"
        ),
        ENV_CHILD_NONCE: _require_sha256(
            child_launch_nonce, label="child_launch_nonce"
        ),
        ENV_CONFIG_SHA256: _require_sha256(
            config_sha256, label="config_sha256"
        ),
    }
    mismatches = {
        key: {"expected": expected, "observed": os.environ.get(key)}
        for key, expected in bindings.items()
        if os.environ.get(key) != expected
    }
    observed_command = _command_sha256((sys.executable, *sys.argv))
    expected_command = os.environ.get(ENV_COMMAND_SHA256)
    if expected_command != observed_command:
        mismatches[ENV_COMMAND_SHA256] = {
            "expected": observed_command,
            "observed": expected_command,
        }
    if mismatches:
        raise D0V2EngineeringSmokeError(
            f"worker lacks parent-issued fresh-process binding: {mismatches}"
        )
    return observed_command


def _load_fixed_method_dataset(
    contract: D0V2ProtocolContract,
    *,
    cache_root: Path,
) -> Any:
    from materialize_binary_tent_ss_calibration_cache_v2 import (
        SourceCalibrationMethodInputDatasetV2,
    )

    sample = contract.raw["engineering_smoke"]["sample"]
    return SourceCalibrationMethodInputDatasetV2(
        cache_root,
        condition_key=str(sample["condition"]),
        expected_protocol_sha256=str(contract.raw["cache"]["protocol_sha256"]),
    )


def _fixed_sample(
    dataset: Any,
    contract: D0V2ProtocolContract,
) -> tuple[Any, dict[str, Any], str]:
    import torch

    frozen = contract.raw["engineering_smoke"]["sample"]
    index = int(frozen["image_index"])
    value = dataset[index]
    expected_fields = {
        "image",
        "image_id",
        "original_size",
        "dataset",
        "corruption",
        "severity",
        "seed",
    }
    if set(value) != expected_fields:
        raise D0V2EngineeringSmokeError(
            "method-facing sample fields differ from label-free contract"
        )
    image = value.pop("image")
    if (
        not isinstance(image, torch.Tensor)
        or tuple(image.shape) != (3, 256, 256)
        or image.dtype != torch.float32
        or not bool(torch.isfinite(image).all())
    ):
        raise D0V2EngineeringSmokeError("fixed method image tensor is invalid")
    expected_metadata = {
        "image_id": frozen["image_id"],
        "original_size": tuple(frozen["original_size"]),
        "dataset": frozen["dataset"],
        "corruption": "clean",
        "severity": 0,
        "seed": 42,
    }
    if dict(value) != expected_metadata:
        raise D0V2EngineeringSmokeError(
            f"fixed method sample binding drifted: {value}"
        )
    batched = image.unsqueeze(0)
    return batched, dict(value), _tensor_sha256(batched)


@dataclass(frozen=True)
class _CandidateExecutionIdentity:
    model_instance_id: str
    method_instance_id: str
    optimizer_instance_id: str
    autograd_graph_id: str
    backward_execution_id: str
    gradient_buffer_owner_id: str


def _execution_identity(process_id: str, candidate_index: int) -> _CandidateExecutionIdentity:
    def value(role: str) -> str:
        return f"{process_id}:{candidate_index}:{role}:{uuid.uuid4().hex}"

    return _CandidateExecutionIdentity(
        model_instance_id=value("model"),
        method_instance_id=value("method"),
        optimizer_instance_id=value("optimizer"),
        autograd_graph_id=value("graph"),
        backward_execution_id=value("backward"),
        gradient_buffer_owner_id=value("gradient-owner"),
    )


def _require_true_mapping(value: Mapping[str, Any], *, label: str) -> None:
    failures = sorted(key for key, passed in value.items() if passed is not True)
    if failures:
        raise D0V2EngineeringSmokeError(f"{label} failed: {failures}")


def _validate_runner_result_checks(
    value: Mapping[str, Any],
    *,
    changed_parameter_tensor_count: int,
    candidate_slug: str,
) -> None:
    checks = dict(value)
    changed_gate = checks.pop("bn_affine_changed_by_one_step", None)
    _require_true_mapping(
        checks, label=f"{candidate_slug} invariant runner checks"
    )
    if changed_gate is not (changed_parameter_tensor_count > 0):
        raise D0V2EngineeringSmokeError(
            f"runner update-activity flag is inconsistent: {candidate_slug}"
        )


def _candidate_receipt(
    *,
    contract: D0V2ProtocolContract,
    process_id: str,
    candidate_index: int,
    dataset_config: Mapping[str, Any],
    method_dataset: Any,
    device: Any,
    runtime_sha256: str,
    determinism_sha256: str,
    ledger: IndependentCandidateExecutionLedger,
    retained_builds: list[Any],
) -> IndependentCandidateReceipt:
    import torch

    from tta.d0_v2_candidate_worker import build_fresh_d0_v2_candidate
    from tta.d0_v2_native_step import (
        NativeFirstStepObserver,
        frozen_first_step_spec,
    )

    candidate = FROZEN_CANDIDATES[candidate_index]
    _seed_candidate()
    smoke = contract.raw["engineering_smoke"]
    method_config = smoke["method"]
    build = build_fresh_d0_v2_candidate(
        project_root=PROJECT_ROOT,
        dataset_config=dataset_config,
        candidate=candidate,
        device=device,
        entropy_eps=float(method_config["entropy_eps"]),
        diagnostic_detail=str(method_config["diagnostic_detail"]),
    )
    retained_builds.append(build)
    if build.checkpoint_sha256 != dataset_config["checkpoint_sha256"]:
        raise D0V2EngineeringSmokeError("fresh build checkpoint binding drifted")
    ledger.claim_candidate_objects(
        candidate,
        model=build.model,
        method=build.method,
        optimizer=build.method.optimizer,
        optimizer_state_entry_count=len(build.method.optimizer.state),
    )
    identity = _execution_identity(process_id, candidate_index)
    autograd_graph_token = object()

    def register_actual_gradients(
        named_gradients: tuple[tuple[str, Any], ...],
    ) -> None:
        if tuple(name for name, _value in named_gradients) != build.parameter_names:
            raise D0V2EngineeringSmokeError(
                "native observer gradient name topology drifted"
            )
        ledger.claim_candidate_backward(
            candidate,
            autograd_graph=autograd_graph_token,
            gradient_buffers=tuple(value for _name, value in named_gradients),
        )

    named_parameters = tuple(
        zip(
            build.parameter_names,
            (
                parameter
                for group in build.method.optimizer.param_groups
                for parameter in group["params"]
            ),
            strict=True,
        )
    )
    sample_image, metadata, input_sha256 = _fixed_sample(method_dataset, contract)
    observer = NativeFirstStepObserver(
        build.method.optimizer,
        named_parameters=named_parameters,
        expected_spec=frozen_first_step_spec(
            candidate.optimizer, candidate.learning_rate
        ),
        gradient_callback=register_actual_gradients,
    )
    with observer:
        result = build.runner.run_one_image(
            image=sample_image.to(device, non_blocking=False),
            metadata=metadata,
            force_full_audit=True,
        )
    observation = observer.observation
    native = observation.to_dict()
    native_hashes = native.get("hashes")
    native_counts = native.get("counts")
    native_gates = native.get("gates")
    native_authorization = native.get("authorization")
    if not all(
        isinstance(value, Mapping)
        for value in (
            native_hashes,
            native_counts,
            native_gates,
            native_authorization,
        )
    ):
        raise D0V2EngineeringSmokeError(
            f"native first-step observation schema drifted: {candidate.slug}"
        )
    assert isinstance(native_hashes, Mapping)
    assert isinstance(native_counts, Mapping)
    assert isinstance(native_gates, Mapping)
    assert isinstance(native_authorization, Mapping)

    build.state_manager.assert_source_state()
    if (
        result.full_audit_performed is not True
        or result.full_audit_reason != "forced"
        or result.reset_full_fingerprint != build.state_manager.source_fingerprint
        or result.source_state_sha256
        != build.state_manager.source_fingerprint.full_sha256
    ):
        raise D0V2EngineeringSmokeError(
            f"fresh candidate exact reset audit failed: {candidate.slug}"
        )
    diagnostics = result.diagnostics
    required_determinism = {
        "forward_deterministic_algorithms_enabled": True,
        "forward_deterministic_algorithms_warn_only": False,
        "backward_deterministic_algorithms_enabled": False,
        "backward_deterministic_algorithms_warn_only": False,
        "deterministic_policy_restored_after_backward": True,
        "deterministic_algorithms_enabled": True,
        "episode_device_type": "cuda",
        "cuda_backward_determinism_policy": "temporarily_disable",
        "amp_enabled": False,
        "finite": True,
        "optimizer_params_exact_all_bn_affine": True,
        "only_bn_affine_requires_grad": True,
        "source_tent_pre_bit_exact": True,
        "fast_source_reset_complete": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": diagnostics.get(key)}
        for key, expected in required_determinism.items()
        if diagnostics.get(key) != expected
    }
    if mismatches:
        raise D0V2EngineeringSmokeError(
            f"fresh candidate runtime gates failed: {candidate.slug}: {mismatches}"
        )
    if diagnostics.get("source_pre_raw_sha256") != diagnostics.get(
        "tent_pre_raw_sha256"
    ):
        raise D0V2EngineeringSmokeError(
            f"Source/TENT-pre raw hash differs: {candidate.slug}"
        )
    _require_true_mapping(native_gates, label=f"{candidate.slug} native gates")
    if (
        native.get("device") != str(device)
        or native_counts.get("optimizer_step_call_count") != 1
        or native_counts.get("parameter_tensor_count") != len(build.parameter_names)
        or native_counts.get("gradient_tensor_count") != len(build.parameter_names)
        or native_counts.get("optimizer_state_parameter_count")
        != len(build.parameter_names)
        or native_counts.get("bit_exact_parameter_tensor_count")
        != len(build.parameter_names)
        or native_counts.get("bit_exact_optimizer_state_tensor_count")
        != native_counts.get("optimizer_state_tensor_count")
        or native_hashes.get("reference_parameter_after_bundle_sha256")
        != native_hashes.get("actual_parameter_after_bundle_sha256")
        or native_hashes.get("reference_optimizer_state_bundle_sha256")
        != native_hashes.get("actual_optimizer_state_bundle_sha256")
        or native_authorization
        != {
            "engineering_observation_only": True,
            "scientific_gate_status": "unresolved",
            "scientific_selection_performed": False,
            "stage2_authorized": False,
        }
    ):
        raise D0V2EngineeringSmokeError(
            f"native first-step gate failed: {candidate.slug}"
        )
    if int(native_counts["changed_parameter_tensor_count"]) != int(
        diagnostics["number_updated_bn_affine_tensors"]
    ):
        raise D0V2EngineeringSmokeError(
            f"native/runner changed tensor counts differ: {candidate.slug}"
        )
    _validate_runner_result_checks(
        result.checks,
        changed_parameter_tensor_count=int(
            native_counts["changed_parameter_tensor_count"]
        ),
        candidate_slug=candidate.slug,
    )
    if not math.isclose(
        float(native["step_norm_l2"]),
        float(diagnostics["step_norm"]),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise D0V2EngineeringSmokeError(
            f"native/runner step norms differ: {candidate.slug}"
        )
    if result.logits_source_pre.dtype != torch.float32:
        raise D0V2EngineeringSmokeError("stored source logits are not float32")

    selected_names_sha256 = hashlib.sha256(
        "\0".join(build.parameter_names).encode("utf-8")
    ).hexdigest()
    if diagnostics["parameter_names_sha256"] != selected_names_sha256:
        raise D0V2EngineeringSmokeError(
            "selected parameter-name hash differs from method receipt"
        )
    frozen_sample = smoke["sample"]
    return IndependentCandidateReceipt(
        candidate_index=candidate_index,
        candidate=candidate,
        config_sha256=str(contract.config_file_sha256),
        dataset=str(frozen_sample["dataset"]),
        condition=str(frozen_sample["condition"]),
        sample_index=int(frozen_sample["image_index"]),
        sample_id=str(frozen_sample["image_id"]),
        split_sha256=str(dataset_config["train_split_sha256"]),
        checkpoint_sha256=str(dataset_config["checkpoint_sha256"]),
        source_state_sha256=_candidate_invariant_source_state_sha256(
            build.state_manager.source_fingerprint
        ),
        runtime_sha256=runtime_sha256,
        determinism_sha256=determinism_sha256,
        input_sha256=input_sha256,
        selected_parameter_names_sha256=selected_names_sha256,
        pre_logits_sha256=str(diagnostics["source_pre_raw_sha256"]),
        post_logits_sha256=str(diagnostics["tent_post_raw_sha256"]),
        entropy_gradient_bundle_sha256=str(
            native_hashes["gradient_bundle_sha256"]
        ),
        parameter_delta_bundle_sha256=str(
            native_hashes["parameter_delta_bundle_sha256"]
        ),
        model_instance_id=identity.model_instance_id,
        method_instance_id=identity.method_instance_id,
        optimizer_instance_id=identity.optimizer_instance_id,
        autograd_graph_id=identity.autograd_graph_id,
        backward_execution_id=identity.backward_execution_id,
        gradient_buffer_owner_id=identity.gradient_buffer_owner_id,
        gradient_tensor_count=int(native_counts["gradient_tensor_count"]),
        changed_parameter_tensor_count=int(
            native_counts["changed_parameter_tensor_count"]
        ),
        optimizer_state_entry_count_after_step=int(
            native_counts["optimizer_state_parameter_count"]
        ),
        native_reference_parameter_tensor_count=int(
            native_counts["native_reference_parameter_tensor_count"]
        ),
        native_reference_optimizer_state_tensor_count=int(
            native_counts["native_reference_optimizer_state_tensor_count"]
        ),
        step_norm_l2=float(native["step_norm_l2"]),
    )


def _publish_bytes_noreplace(destination: Path, data: bytes) -> None:
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        publish_file_noreplace(temporary, destination)
    finally:
        if temporary.exists() and temporary.is_file() and not temporary.is_symlink():
            temporary.unlink()


def run_worker(
    *,
    config_path: Path,
    process_id: str,
    parent_run_nonce: str,
    child_launch_nonce: str,
    output: Path,
) -> dict[str, Any]:
    """Run all ten independent candidates in one parent-bound GPU child."""

    if process_id not in PROCESS_IDS:
        raise D0V2EngineeringSmokeError("worker process_id is not frozen")
    contract = _load_contract(config_path)
    config_sha256 = str(contract.config_file_sha256)
    observed_command_sha256 = _validate_worker_environment(
        process_id=process_id,
        parent_run_nonce=parent_run_nonce,
        child_launch_nonce=child_launch_nonce,
        config_sha256=config_sha256,
    )
    if output.name != f"{process_id}.json" or not output.parent.is_dir():
        raise D0V2EngineeringSmokeError("worker output path is not parent staging")
    if output.exists():
        raise FileExistsError(f"worker receipt already exists: {output}")

    critical_seal, critical_sha256 = _capture_critical_source_seal(contract)
    method_input_seal = _capture_method_facing_input_seal(
        contract, config_path=config_path
    )
    device = _configure_cuda_worker()
    from analysis.d0_determinism_runtime import frozen_determinism_sha256
    smoke = contract.raw["engineering_smoke"]
    determinism_sha256 = frozen_determinism_sha256(
        smoke["method"]["determinism"]
    )
    runtime_sha256 = _runtime_sha256(
        method_facing_input_seal_sha256=method_input_seal.seal_sha256,
        critical_source_sha256=critical_sha256,
        determinism_sha256=determinism_sha256,
        device=device,
    )
    dataset_name = str(smoke["sample"]["dataset"])
    if dataset_name != "NUAA-SIRST":
        raise D0V2EngineeringSmokeError("fixed smoke dataset is unavailable")
    configured_cache_root = PROJECT_ROOT / str(contract.raw["cache"]["root"])
    if method_input_seal.cache_root != configured_cache_root / dataset_name:
        raise D0V2EngineeringSmokeError("frozen cache root binding differs")
    complete = method_input_seal.complete
    if (
        complete.get("test_images_opened") != 0
        or complete.get("test_masks_opened") != 0
    ):
        raise D0V2EngineeringSmokeError("frozen train cache test-open audit failed")
    method_dataset = _load_fixed_method_dataset(
        contract, cache_root=method_input_seal.cache_root
    )
    first_image, _metadata, _input_sha256 = _fixed_sample(method_dataset, contract)
    del first_image

    ledger = IndependentCandidateExecutionLedger()
    retained_builds: list[Any] = []
    dataset_config = contract.raw["datasets"][dataset_name]
    receipts = [
        _candidate_receipt(
            contract=contract,
            process_id=process_id,
            candidate_index=index,
            dataset_config=dataset_config,
            method_dataset=method_dataset,
            device=device,
            runtime_sha256=runtime_sha256,
            determinism_sha256=determinism_sha256,
            ledger=ledger,
            retained_builds=retained_builds,
        )
        for index in range(len(FROZEN_CANDIDATES))
    ]
    ledger_summary = ledger.assert_complete()
    if (
        ledger_summary["gradient_reuse"] is not False
        or ledger_summary["engineering_gate_passed"] is not True
        or ledger_summary["stage2_authorized"] is not False
    ):
        raise D0V2EngineeringSmokeError(
            "independent candidate live-object ledger did not close"
        )
    if len(retained_builds) != 10:
        raise D0V2EngineeringSmokeError("fresh build retention count differs")
    exit_method_input_seal = _capture_method_facing_input_seal(
        contract, config_path=config_path
    )
    if exit_method_input_seal.identities != method_input_seal.identities:
        raise D0V2EngineeringSmokeError(
            "method-facing input lineage changed during execution"
        )
    if exit_method_input_seal.seal_sha256 != method_input_seal.seal_sha256:
        raise D0V2EngineeringSmokeError(
            "method-facing input seal digest changed during execution"
        )
    _assert_critical_source_seal(contract, critical_seal)
    process_receipt = build_d0_v2_smoke_process_receipt(
        receipts,
        parent_run_nonce=parent_run_nonce,
        child_launch_nonce=child_launch_nonce,
        command_sha256=observed_command_sha256,
        process_identity=FreshSubprocessIdentity.capture_current(process_id),
    )
    canonical = canonical_d0_v2_smoke_process_receipt_bytes(process_receipt)
    _publish_bytes_noreplace(output, canonical)
    stable = read_stable_regular_file(output)
    if stable.data != canonical:
        raise D0V2EngineeringSmokeError("published process receipt bytes differ")
    return process_receipt


def _complete_manifest(
    *,
    config_sha256: str,
    process_file_sha256s: Mapping[str, str],
    aggregate_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "complete": True,
        "role": "engineering_smoke_only",
        "config_sha256": _require_sha256(
            config_sha256, label="complete.config_sha256"
        ),
        "process_files": [
            {"path": name, "sha256": process_file_sha256s[name]}
            for name in PROCESS_FILENAMES
        ],
        "aggregate_file": {
            "path": AGGREGATE_FILENAME,
            "sha256": _require_sha256(
                aggregate_sha256, label="complete.aggregate_sha256"
            ),
        },
        "payload_file_count": 4,
        "atomic_no_replace": True,
        "paper_result": False,
        "paper_test_result": False,
        "scientific_gate_status": "unresolved",
        "scientific_selection_performed": False,
        "formal_p3_complete": False,
        "stage2_authorized": False,
    }


def verify_engineering_smoke_directory(
    path: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """CPU-only, fail-closed verification of one flat smoke publication."""

    try:
        snapshot = snapshot_regular_directory(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise D0V2EngineeringSmokeError(
            f"cannot securely snapshot engineering smoke: {path}"
        ) from exc
    by_name = {member.path.name: member for member in snapshot.members}
    if set(by_name) != EXPECTED_DIRECTORY_MEMBERS:
        raise D0V2EngineeringSmokeError(
            "engineering smoke directory member set is not exact"
        )
    process_receipts = []
    for process_id, filename in zip(PROCESS_IDS, PROCESS_FILENAMES, strict=True):
        value = _strict_json_bytes(by_name[filename].data, label=filename)
        normalized = validate_d0_v2_smoke_process_receipt(value)
        if normalized["process_identity"]["process_id"] != process_id:
            raise D0V2EngineeringSmokeError(
                f"process receipt filename/identity differs: {filename}"
            )
        if canonical_d0_v2_smoke_process_receipt_bytes(normalized) != by_name[
            filename
        ].data:
            raise D0V2EngineeringSmokeError(
                f"process receipt is not canonical bytes: {filename}"
            )
        process_receipts.append(normalized)
    rebuilt = aggregate_three(process_receipts)
    observed_aggregate = _strict_json_bytes(
        by_name[AGGREGATE_FILENAME].data, label=AGGREGATE_FILENAME
    )
    normalized_aggregate = validate_d0_v2_smoke_repro_aggregate(
        observed_aggregate
    )
    canonical_aggregate = canonical_d0_v2_smoke_repro_aggregate_bytes(
        normalized_aggregate
    )
    if canonical_aggregate != by_name[AGGREGATE_FILENAME].data:
        raise D0V2EngineeringSmokeError("aggregate file is not canonical bytes")
    if _canonical_json_bytes(rebuilt) != canonical_aggregate:
        raise D0V2EngineeringSmokeError(
            "aggregate does not rebuild from separate process files"
        )
    config_sha256 = normalized_aggregate["shared_cell_binding"]["config_sha256"]
    if config_path is not None:
        contract = _load_contract(config_path)
        if contract.config_file_sha256 != config_sha256:
            raise D0V2EngineeringSmokeError(
                "published smoke config SHA differs from requested config"
            )
    process_hashes = {
        name: by_name[name].sha256 for name in PROCESS_FILENAMES
    }
    expected_complete = _complete_manifest(
        config_sha256=config_sha256,
        process_file_sha256s=process_hashes,
        aggregate_sha256=by_name[AGGREGATE_FILENAME].sha256,
    )
    observed_complete = _strict_json_bytes(
        by_name[COMPLETE_FILENAME].data, label=COMPLETE_FILENAME
    )
    if _canonical_json_bytes(expected_complete) != _canonical_json_bytes(
        observed_complete
    ):
        raise D0V2EngineeringSmokeError("COMPLETE manifest binding differs")
    return normalized_aggregate


def _validate_visible_device(value: str) -> str:
    if _VISIBLE_DEVICE_RE.fullmatch(value) is None or "," in value:
        raise D0V2EngineeringSmokeError(
            "--cuda-visible-device must name exactly one safe physical device"
        )
    return value


def _worker_command(
    *,
    config_path: Path,
    process_id: str,
    parent_run_nonce: str,
    child_launch_nonce: str,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--config",
        str(config_path.absolute()),
        "--process-id",
        process_id,
        "--parent-run-nonce",
        parent_run_nonce,
        "--child-launch-nonce",
        child_launch_nonce,
        "--output",
        str(output.absolute()),
    ]


def run_three_process_engineering_smoke(
    *,
    config_path: Path = DEFAULT_CONFIG,
    cuda_visible_device: str,
) -> Path:
    """Launch exactly three fresh GPU workers and atomically publish results."""

    import torch

    if torch.cuda.is_initialized():
        raise D0V2EngineeringSmokeError(
            "parent initialized CUDA before launching fresh subprocesses"
        )
    visible = _validate_visible_device(cuda_visible_device)
    contract = _load_contract(config_path)
    output_root, destination = _output_paths(contract)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite engineering smoke: {destination}"
        )
    relative_parts = output_root.relative_to(PROJECT_ROOT).parts
    ensured_root = ensure_directory_chain_nofollow(PROJECT_ROOT, relative_parts)
    parent_nonce = secrets.token_hex(32)

    with tempfile.TemporaryDirectory(
        prefix=".engineering_smoke.", dir=ensured_root
    ) as temporary:
        staging = Path(temporary)
        process_receipts: list[dict[str, Any]] = []
        for process_id, filename in zip(PROCESS_IDS, PROCESS_FILENAMES, strict=True):
            child_nonce = secrets.token_hex(32)
            output = staging / filename
            command = _worker_command(
                config_path=config_path,
                process_id=process_id,
                parent_run_nonce=parent_nonce,
                child_launch_nonce=child_nonce,
                output=output,
            )
            command_sha256 = _command_sha256(command)
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONHASHSEED": "42",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "CUDA_VISIBLE_DEVICES": visible,
                    ENV_PROCESS_ID: process_id,
                    ENV_PARENT_NONCE: parent_nonce,
                    ENV_CHILD_NONCE: child_nonce,
                    ENV_COMMAND_SHA256: command_sha256,
                    ENV_CONFIG_SHA256: str(contract.config_file_sha256),
                }
            )
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise D0V2EngineeringSmokeError(
                    f"fresh worker failed: {process_id}; "
                    f"stdout={completed.stdout[-2000:]!r}; "
                    f"stderr={completed.stderr[-4000:]!r}"
                )
            stable = read_stable_regular_file(output)
            receipt = validate_d0_v2_smoke_process_receipt(
                _strict_json_bytes(stable.data, label=filename)
            )
            if receipt["command_sha256"] != command_sha256:
                raise D0V2EngineeringSmokeError(
                    f"worker command binding differs: {process_id}"
                )
            if receipt["process_identity"]["os_process_id"] == os.getpid():
                raise D0V2EngineeringSmokeError(
                    "worker receipt reused the parent OS process"
                )
            process_receipts.append(receipt)

        aggregate = aggregate_three(process_receipts)
        aggregate_bytes = canonical_d0_v2_smoke_repro_aggregate_bytes(aggregate)
        _publish_bytes_noreplace(staging / AGGREGATE_FILENAME, aggregate_bytes)
        process_hashes = {
            filename: read_stable_regular_file(staging / filename).sha256
            for filename in PROCESS_FILENAMES
        }
        complete = _complete_manifest(
            config_sha256=str(contract.config_file_sha256),
            process_file_sha256s=process_hashes,
            aggregate_sha256=_sha256(aggregate_bytes),
        )
        _publish_bytes_noreplace(
            staging / COMPLETE_FILENAME, _canonical_json_bytes(complete)
        )
        verify_engineering_smoke_directory(staging, config_path=config_path)
        publish_directory_noreplace(
            staging,
            destination,
            pre_rename_guard=lambda: verify_engineering_smoke_directory(
                staging, config_path=config_path
            ),
        )
    verify_engineering_smoke_directory(destination, config_path=config_path)
    return destination


def _parser(*, include_internal_worker: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "D0-v2 independent-candidate engineering smoke. This is never a "
            "paper result, never scientific selection, and never Stage-2 authority."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate",
        help="CPU-only strict config/seal validation; creates no files",
    )
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    run = subparsers.add_parser(
        "run",
        help=(
            "launch 3 fresh GPU subprocesses and atomically publish only "
            "engineering_smoke"
        ),
    )
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument(
        "--cuda-visible-device",
        required=True,
        help="one physical CUDA index/UUID exposed to every isolated worker",
    )

    verify = subparsers.add_parser(
        "verify",
        help="CPU-only verification of a published engineering_smoke directory",
    )
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify.add_argument("--path", type=Path, default=None)

    if include_internal_worker:
        worker = subparsers.add_parser("_worker")
        worker.add_argument("--config", type=Path, required=True)
        worker.add_argument("--process-id", required=True)
        worker.add_argument("--parent-run-nonce", required=True)
        worker.add_argument("--child-launch-nonce", required=True)
        worker.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _parser(
        include_internal_worker=bool(arguments and arguments[0] == "_worker")
    ).parse_args(arguments)
    try:
        if args.command == "validate":
            value = validate_contract_only(args.config)
        elif args.command == "run":
            destination = run_three_process_engineering_smoke(
                config_path=args.config,
                cuda_visible_device=args.cuda_visible_device,
            )
            aggregate = verify_engineering_smoke_directory(
                destination, config_path=args.config
            )
            value = {
                "published": destination.relative_to(PROJECT_ROOT).as_posix(),
                "aggregate_sha256": d0_v2_smoke_repro_aggregate_sha256(
                    aggregate
                ),
                "engineering_smoke": True,
                "paper_result": False,
                "formal_p3_complete": False,
                "stage2_authorized": False,
            }
        elif args.command == "verify":
            contract = _load_contract(args.config)
            _root, canonical = _output_paths(contract)
            aggregate = verify_engineering_smoke_directory(
                canonical if args.path is None else args.path,
                config_path=args.config,
            )
            value = {
                "verified": True,
                "aggregate_sha256": d0_v2_smoke_repro_aggregate_sha256(
                    aggregate
                ),
                "engineering_smoke": True,
                "paper_result": False,
                "formal_p3_complete": False,
                "stage2_authorized": False,
            }
        elif args.command == "_worker":
            receipt = run_worker(
                config_path=args.config,
                process_id=args.process_id,
                parent_run_nonce=args.parent_run_nonce,
                child_launch_nonce=args.child_launch_nonce,
                output=args.output,
            )
            value = {
                "worker_complete": True,
                "process_id": receipt["process_identity"]["process_id"],
                "engineering_smoke": True,
                "paper_result": False,
                "formal_p3_complete": False,
                "stage2_authorized": False,
            }
        else:
            raise D0V2EngineeringSmokeError("unsupported CLI command")
    except (
        D0V2EngineeringSmokeError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"D0-v2 engineering smoke failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
