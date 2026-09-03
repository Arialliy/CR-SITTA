#!/usr/bin/env python3
"""Run one formal P3 Stage-A label-free R0 cell in a fresh GPU process.

The public ``run`` command creates a private hidden staging directory, launches
exactly one CUDA-visible child, verifies the completed flat shard on CPU and
publishes it with a snapshot-bound no-replace rename.  The candidate child
uses only ``SourceCalibrationMethodInputDatasetV2``; it never imports or calls
the outer-target loader and never opens validation/test payloads.

Formal execution is deliberately narrow: ``R0`` means 64 images x all ten
frozen candidates.  ``--max-images 1`` is an engineering dry-run and is
persisted outside the formal shard tree with ``formal=false`` and
``stage2_authorized=false``.  R1/R2 require a future, signed R0 eligibility
receipt and are therefore rejected by this worker.
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
import subprocess
import sys
import uuid
from typing import Any, Final

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.d0_v2_independent_candidate_contract import (
    FROZEN_CANDIDATES,
    IndependentCandidateExecutionLedger,
    IndependentCandidateReceipt,
    validate_independent_candidate_cell,
)
from analysis.d0_v2_protocol_contract import (
    D0V2ProtocolContract,
    load_d0_v2_protocol_contract,
    verify_sealed_v1_inputs,
)
from analysis.d0_v2_smoke_repro_contract import FreshSubprocessIdentity
from analysis.d0_v3_formal_contract import (
    CONFIG_RELATIVE_PATH,
    D0V3FormalContract,
    FROZEN_CANDIDATES as FORMAL_CANDIDATES,
    load_d0_v3_formal_contract,
    verify_frozen_parent_bindings,
)
from analysis.d0_v3_label_free_shard import (
    ARRAY_FILENAMES,
    ARTIFACT_TYPE,
    CANDIDATE_COUNT,
    COMPLETE_ARTIFACT_TYPE,
    COMPLETE_FILENAME,
    DRY_RUN_MEMBERS,
    ENTROPY_GRADIENTS_FILENAME,
    EPISODES_FILENAME,
    EPISODE_ARTIFACT_TYPE,
    FLOAT_DTYPE,
    FORMAL_IMAGE_COUNT,
    FORMAL_MEMBERS,
    LAYOUT_FILENAME,
    MANIFEST_FILENAME,
    PARAMETERS_AFTER_FILENAME,
    PARAMETER_SCALAR_COUNT,
    PARAMETER_TENSOR_COUNT,
    PHASE_RECEIPT_FILENAME,
    POST_LOGITS_FILENAME,
    REQUIRED_CRITICAL_CODE_PATHS,
    SOURCE_LOGITS_FILENAME,
    SOURCE_PARAMETERS_FILENAME,
    array_slice_sha256,
    canonical_json_bytes,
    canonical_sha256,
    expected_array_specs,
    input_tensor_sha256,
    named_bundle_sha256,
    raw_float32_sha256,
    validate_parameter_layout,
    verify_label_free_shard,
)
from analysis.d0_v3_outer_analyzer import FlatParameterLayout
from analysis.d0_v3_phase_receipt import (
    build_label_free_cell_receipt,
    canonical_label_free_cell_receipt_bytes,
    zero_candidate_phase_data_boundary,
)
from analysis.source_train_provenance import SourceTrainAnalysisProvenance
from tta.d0_secure_io import (
    ensure_directory_chain_nofollow,
    read_stable_regular_file,
)
from tta.d0_v3_atomic_shard import publish_flat_directory_noreplace
from tta.d0_v3_formal_capture import FormalFirstStepCapture


DEFAULT_CONFIG: Final = PROJECT_ROOT / CONFIG_RELATIVE_PATH
PARENT_CONFIG: Final = (
    PROJECT_ROOT / "configs/tent_failure_diagnostics_v2_independent_candidates.yaml"
)
SEED: Final = 42
REPLICATES: Final = ("R0", "R1", "R2")
VISIBLE_DEVICE_ENV_REJECT = {"", ".", ".."}

ENV_PROCESS_ID: Final = "CR_SITTA_D0_V3_PROCESS_ID"
ENV_PARENT_NONCE: Final = "CR_SITTA_D0_V3_PARENT_NONCE"
ENV_CHILD_NONCE: Final = "CR_SITTA_D0_V3_CHILD_NONCE"
ENV_CONFIG_SHA256: Final = "CR_SITTA_D0_V3_CONFIG_SHA256"
ENV_COMMAND_SHA256: Final = "CR_SITTA_D0_V3_COMMAND_SHA256"

_EXECUTION_GATES: Final = {
    "fresh_model_method_optimizer": True,
    "own_forward_backward": True,
    "gradient_reuse": False,
    "only_selected_trainable": True,
    "optimizer_state_empty_before_step": True,
    "native_same_device_endpoint_exact": True,
    "native_optimizer_state_exact": True,
    "source_tent_pre_bit_exact": True,
    "full_audit_forced": True,
    "exact_source_reset": True,
    "formal_native_hash_parity": True,
    "live_gradient_storage_owned": True,
    "finite": True,
}
_DATA_BOUNDARY: Final = {
    "source_train_derived": True,
    "split_name": "train",
    "split_role": "frozen_pilot64",
    "no_validation_split": True,
    "method_label_accesses": 0,
    "outer_evaluator_label_accesses": 0,
    "train_target_payload_bytes_opened": 0,
    "train_target_payload_deserialization_count": 0,
    "validation_payload_opens": 0,
    "test_split_files_opened": 0,
    "test_images_opened": 0,
    "test_masks_opened": 0,
    "test_labels_opened": 0,
}
_AUTHORIZATION: Final = {
    "paper_result": False,
    "paper_test_result": False,
    "scientific_gate_status": "not_evaluated",
    "scientific_selection_performed": False,
    "formal_protocol_complete": False,
    "stage2_authorized": False,
}


class D0V3StageAWorkerError(RuntimeError):
    """The formal label-free worker violated a frozen execution boundary."""


@dataclass(frozen=True, slots=True)
class LoadedContracts:
    formal: D0V3FormalContract
    parent: D0V2ProtocolContract


@dataclass(frozen=True, slots=True)
class InputSeal:
    files: tuple[dict[str, str], ...]
    seal_sha256: str
    cache_root: Path
    image_ids: tuple[str, ...]
    original_sizes: tuple[tuple[int, int], ...]
    method_manifest_sha256: str
    code_seals: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [dict(value) for value in self.files],
            "seal_sha256": self.seal_sha256,
            "target_payload_bytes_opened": 0,
            "target_payload_deserialized": False,
            "validation_payload_opens": 0,
            "test_split_files_opened": 0,
            "test_images_opened": 0,
            "test_masks_opened": 0,
            "test_labels_opened": 0,
        }


@dataclass(frozen=True, slots=True)
class EpisodeEvidence:
    image_index: int
    image_id: str
    candidate_index: int
    independent: Mapping[str, Any]
    native: Mapping[str, Any]
    optimizer_geometry: Mapping[str, Any]


def _strict_yaml(data: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise D0V3StageAWorkerError(f"{label} is not valid UTF-8 YAML") from exc
    if not isinstance(value, Mapping):
        raise D0V3StageAWorkerError(f"{label} root must be a mapping")
    return value


def _strict_json(data: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D0V3StageAWorkerError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise D0V3StageAWorkerError(f"{label} root must be a mapping")
    return value


def _project_path(value: Any, *, label: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise D0V3StageAWorkerError(f"{label} must be a non-empty path")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise D0V3StageAWorkerError(
            f"{label} must be a canonical repository-relative path"
        )
    absolute = PROJECT_ROOT / relative
    if not absolute.absolute().is_relative_to(PROJECT_ROOT):
        raise D0V3StageAWorkerError(f"{label} escapes repository root")
    return relative.as_posix(), absolute


def _validate_candidate_grid(
    formal_candidates: Sequence[Any], parent_candidates: Sequence[Any]
) -> None:
    """Require both versioned candidate grids to contain exactly ten items."""

    if len(formal_candidates) != 10 or len(parent_candidates) != 10:
        raise D0V3StageAWorkerError("candidate grid count drifted")
    for index, (left, right, parent, v2) in enumerate(
        zip(
            formal_candidates,
            FORMAL_CANDIDATES,
            parent_candidates,
            FROZEN_CANDIDATES,
            strict=True,
        )
    ):
        if (
            left != right
            or parent != v2
            or left.optimizer != v2.optimizer
            or left.learning_rate != v2.learning_rate
            or left.candidate_id != v2.slug
        ):
            raise D0V3StageAWorkerError(
                f"formal/v2 candidate grid differs at index {index}"
            )


def _load_contracts(config_path: Path) -> LoadedContracts:
    formal = load_d0_v3_formal_contract(config_path)
    verify_frozen_parent_bindings(formal, repository_root=PROJECT_ROOT)
    parent_binding = formal.raw["frozen_parent_bindings"]["engineering_config"]
    relative, parent_path = _project_path(
        parent_binding["path"], label="frozen engineering config path"
    )
    if relative != PARENT_CONFIG.relative_to(PROJECT_ROOT).as_posix():
        raise D0V3StageAWorkerError("formal parent config path differs")
    parent = load_d0_v2_protocol_contract(parent_path)
    if parent.config_file_sha256 != parent_binding["sha256"]:
        raise D0V3StageAWorkerError("formal parent config SHA differs")
    verify_sealed_v1_inputs(parent, repository_root=PROJECT_ROOT)
    if formal.stage2_authorized or formal.formal_protocol_complete_initial:
        raise D0V3StageAWorkerError("formal contract authorization initial state drifted")
    if formal.scientific_status_initial != "not_evaluated":
        raise D0V3StageAWorkerError("formal science status initial state drifted")
    _validate_candidate_grid(formal.candidates, parent.candidates)
    return LoadedContracts(formal=formal, parent=parent)


def _snapshot_record(role: str, relative: str, sha256: str) -> dict[str, str]:
    return {"role": role, "path": relative, "sha256": sha256}


def _capture_input_seal(
    contracts: LoadedContracts,
    *,
    config_path: Path,
    dataset: str,
    condition: str,
) -> InputSeal:
    """Seal only method-facing train-Pilot inputs and executable code."""

    formal = contracts.formal
    parent = contracts.parent
    records: dict[str, dict[str, str]] = {}

    def capture(
        role: str,
        relative: str,
        *,
        expected_sha256: str | None = None,
    ) -> Any:
        if role in records:
            raise D0V3StageAWorkerError(f"duplicate input-seal role: {role}")
        canonical, path = _project_path(relative, label=f"input seal {role}")
        if canonical.endswith("outer_evaluator/targets.npy"):
            raise D0V3StageAWorkerError("target payload is forbidden in candidate seal")
        snapshot = read_stable_regular_file(path)
        if expected_sha256 is not None and snapshot.sha256 != expected_sha256:
            raise D0V3StageAWorkerError(f"input SHA differs: {role}")
        records[role] = _snapshot_record(role, canonical, snapshot.sha256)
        return snapshot

    config_relative = config_path.relative_to(PROJECT_ROOT).as_posix()
    capture(
        "formal_config",
        config_relative,
        expected_sha256=str(formal.config_file_sha256),
    )
    parent_relative = PARENT_CONFIG.relative_to(PROJECT_ROOT).as_posix()
    capture(
        "parent_engineering_config",
        parent_relative,
        expected_sha256=str(parent.config_file_sha256),
    )
    smoke_binding = formal.raw["frozen_parent_bindings"]["engineering_smoke"]
    smoke_relative = f"{smoke_binding['root']}/aggregate.json"
    capture(
        "engineering_smoke_aggregate",
        smoke_relative,
        expected_sha256=str(smoke_binding["aggregate_sha256"]),
    )

    cache = formal.raw["cache"]
    execution_relative = str(cache["execution_protocol_path"])
    execution_snapshot = capture(
        "cache_execution_protocol",
        execution_relative,
        expected_sha256=str(cache["execution_protocol_sha256"]),
    )
    execution = _strict_yaml(execution_snapshot.data, label="cache execution protocol")
    execution_cache = execution.get("cache_protocol")
    if not isinstance(execution_cache, Mapping):
        raise D0V3StageAWorkerError("cache execution binding is unavailable")
    expected_cache_binding = {
        "path": cache["protocol_path"],
        "sha256": cache["protocol_sha256"],
        "root": cache["root"],
    }
    if any(
        execution_cache.get(key) != expected
        for key, expected in expected_cache_binding.items()
    ):
        raise D0V3StageAWorkerError("cache execution/protocol binding differs")
    artifacts = execution_cache.get("artifacts")
    artifact = artifacts.get(dataset) if isinstance(artifacts, Mapping) else None
    if not isinstance(artifact, Mapping):
        raise D0V3StageAWorkerError(f"cache artifact anchors missing: {dataset}")

    protocol_relative = str(cache["protocol_path"])
    protocol_snapshot = capture(
        "cache_protocol",
        protocol_relative,
        expected_sha256=str(cache["protocol_sha256"]),
    )
    protocol = _strict_yaml(protocol_snapshot.data, label="cache protocol")
    datasets = protocol.get("datasets")
    dataset_protocol = datasets.get(dataset) if isinstance(datasets, Mapping) else None
    if not isinstance(dataset_protocol, Mapping):
        raise D0V3StageAWorkerError(f"cache dataset protocol missing: {dataset}")
    dataset_binding = formal.raw["datasets"][dataset]
    parent_binding = parent.raw["datasets"][dataset]
    if dict(dataset_binding) != dict(parent_binding):
        raise D0V3StageAWorkerError("formal/parent dataset binding differs")
    if dataset_protocol.get("train_split_sha256") != dataset_binding[
        "train_split_sha256"
    ]:
        raise D0V3StageAWorkerError("train split SHA binding differs")
    for role, path_field, sha_field in (
        ("source_train_split", "train_split", "train_split_sha256"),
        ("frozen_pilot_ids", "calibration_ids", "calibration_ids_file_sha256"),
    ):
        capture(
            role,
            str(dataset_protocol[path_field]),
            expected_sha256=str(dataset_protocol[sha_field]),
        )
    capture(
        "source_checkpoint",
        str(dataset_binding["checkpoint_path"]),
        expected_sha256=str(dataset_binding["checkpoint_sha256"]),
    )

    cache_root = PROJECT_ROOT / str(cache["root"]) / dataset
    cache_relative = cache_root.relative_to(PROJECT_ROOT).as_posix()
    cache_snapshots: dict[str, Any] = {}
    for role, filename, anchor in (
        ("cache_manifest", "manifest.json", "manifest_sha256"),
        (
            "cache_method_manifest",
            "method_input_manifest.json",
            "method_input_manifest_sha256",
        ),
        ("cache_complete", "COMPLETE.json", "complete_sha256"),
    ):
        cache_snapshots[role] = capture(
            role,
            f"{cache_relative}/{filename}",
            expected_sha256=str(artifact[anchor]),
        )
    complete = _strict_json(cache_snapshots["cache_complete"].data, label="cache COMPLETE")
    if (
        complete.get("complete") is not True
        or complete.get("dataset") != dataset
        or complete.get("protocol_sha256") != cache["protocol_sha256"]
        or complete.get("test_images_opened") != 0
        or complete.get("test_masks_opened") != 0
        or complete.get("method_received_labels") is not False
    ):
        raise D0V3StageAWorkerError("cache completion/firewall receipt differs")
    manifest = _strict_json(cache_snapshots["cache_manifest"].data, label="cache manifest")
    method = _strict_json(
        cache_snapshots["cache_method_manifest"].data,
        label="cache method manifest",
    )
    image_ids = manifest.get("image_ids")
    original_sizes = manifest.get("original_sizes")
    if (
        manifest.get("dataset") != dataset
        or manifest.get("protocol_sha256") != cache["protocol_sha256"]
        or manifest.get("train_split_sha256") != dataset_binding["train_split_sha256"]
        or not isinstance(image_ids, list)
        or len(image_ids) != FORMAL_IMAGE_COUNT
        or len(set(image_ids)) != FORMAL_IMAGE_COUNT
        or not isinstance(original_sizes, list)
        or len(original_sizes) != FORMAL_IMAGE_COUNT
        or method.get("dataset") != dataset
        or method.get("targets_exposed") is not False
        or method.get("image_ids") != image_ids
    ):
        raise D0V3StageAWorkerError("cache method-facing lineage differs")
    manifest_files = manifest.get("files")
    method_files = method.get("files")
    relative_condition = f"conditions/{condition}.npy"
    outer_record = (
        manifest_files.get(relative_condition)
        if isinstance(manifest_files, Mapping)
        else None
    )
    method_record = (
        method_files.get(relative_condition)
        if isinstance(method_files, Mapping)
        else None
    )
    if (
        not isinstance(outer_record, Mapping)
        or not isinstance(method_record, Mapping)
        or dict(outer_record) != dict(method_record)
    ):
        raise D0V3StageAWorkerError("active method condition binding differs")
    condition_snapshot = capture(
        "method_condition",
        f"{cache_relative}/{relative_condition}",
        expected_sha256=str(method_record["sha256"]),
    )
    if condition_snapshot.size_bytes != method_record.get("bytes"):
        raise D0V3StageAWorkerError("active method condition byte count differs")

    code_seals: dict[str, str] = {}
    for relative in REQUIRED_CRITICAL_CODE_PATHS:
        role = f"critical_code:{relative}"
        snapshot = capture(role, relative)
        code_seals[relative] = snapshot.sha256
    ordered = tuple(records[key] for key in sorted(records))
    seal_sha = canonical_sha256(list(ordered))
    return InputSeal(
        files=ordered,
        seal_sha256=seal_sha,
        cache_root=cache_root,
        image_ids=tuple(str(value) for value in image_ids),
        original_sizes=tuple(tuple(int(item) for item in value) for value in original_sizes),
        method_manifest_sha256=cache_snapshots["cache_method_manifest"].sha256,
        code_seals=code_seals,
    )


def _canonical_command_sha256(command: Sequence[str]) -> str:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise D0V3StageAWorkerError("worker command vector is invalid")
    return hashlib.sha256(canonical_json_bytes(list(command))).hexdigest()


def _validate_visible_device(value: str) -> str:
    if (
        not isinstance(value, str)
        or value in VISIBLE_DEVICE_ENV_REJECT
        or "," in value
        or any(character.isspace() for character in value)
        or any(character in value for character in ("/", "\x00", "\r", "\n"))
    ):
        raise D0V3StageAWorkerError(
            "--cuda-visible-device must identify exactly one safe device"
        )
    return value


def _configure_cuda_worker() -> Any:
    import numpy as np
    import torch

    required = {
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    mismatches = {
        key: {"expected": expected, "observed": os.environ.get(key)}
        for key, expected in required.items()
        if os.environ.get(key) != expected
    }
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        _validate_visible_device(str(visible))
    except D0V3StageAWorkerError:
        mismatches["CUDA_VISIBLE_DEVICES"] = {"expected": "one device", "observed": visible}
    if mismatches:
        raise D0V3StageAWorkerError(f"worker CUDA environment differs: {mismatches}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise D0V3StageAWorkerError("worker must see exactly one CUDA device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise D0V3StageAWorkerError("strict deterministic forwards were not enabled")
    return device


def _seed_candidate() -> None:
    import numpy as np
    import torch

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _validate_worker_binding(
    *,
    process_id: str,
    parent_nonce: str,
    child_nonce: str,
    config_sha256: str,
) -> str:
    expected = {
        ENV_PROCESS_ID: process_id,
        ENV_PARENT_NONCE: parent_nonce,
        ENV_CHILD_NONCE: child_nonce,
        ENV_CONFIG_SHA256: config_sha256,
    }
    mismatches = {
        key: {"expected": value, "observed": os.environ.get(key)}
        for key, value in expected.items()
        if os.environ.get(key) != value
    }
    observed_command = _canonical_command_sha256((sys.executable, *sys.argv))
    if os.environ.get(ENV_COMMAND_SHA256) != observed_command:
        mismatches[ENV_COMMAND_SHA256] = {
            "expected": observed_command,
            "observed": os.environ.get(ENV_COMMAND_SHA256),
        }
    if mismatches:
        raise D0V3StageAWorkerError(
            f"worker lacks fresh-process launch binding: {mismatches}"
        )
    return observed_command


def _runtime_sha256(
    *,
    input_seal_sha256: str,
    code_seals: Mapping[str, str],
    determinism_sha256: str,
    device: Any,
) -> str:
    import torch

    value = {
        "schema_version": 3,
        "input_seal_sha256": input_seal_sha256,
        "critical_code_bundle_sha256": canonical_sha256(dict(code_seals)),
        "determinism_sha256": determinism_sha256,
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn": int(torch.backends.cudnn.version()),
        "device_type": str(device.type),
        "device_name": str(torch.cuda.get_device_name(device)),
        "device_capability": list(torch.cuda.get_device_capability(device)),
    }
    return canonical_sha256(value)


def _source_state_sha256(fingerprint: Any) -> str:
    value = {
        "hash_contract": "cr-sitta-d0-v2-source-state-excluding-candidate-optimizer-v1",
        "model_sha256": fingerprint.model_sha256,
        "runtime_sha256": fingerprint.runtime_sha256,
        "topology_sha256": fingerprint.topology_sha256,
        "gradients_sha256": fingerprint.gradients_sha256,
        "extras_sha256": fingerprint.extras_sha256,
    }
    return canonical_sha256(value)


def _storage_identity(value: Any) -> tuple[str, int, int, int, str]:
    return (
        str(value.device),
        int(value.untyped_storage().data_ptr()),
        int(value.storage_offset()),
        int(value.numel()),
        str(value.dtype),
    )


def _execution_identity(process_id: str, image_index: int, candidate_index: int) -> dict[str, str]:
    def token(role: str) -> str:
        return f"{process_id}:{image_index}:{candidate_index}:{role}:{uuid.uuid4().hex}"

    return {
        "model_instance_id": token("model"),
        "method_instance_id": token("method"),
        "optimizer_instance_id": token("optimizer"),
        "autograd_graph_id": token("graph"),
        "backward_execution_id": token("backward"),
        "gradient_buffer_owner_id": token("gradient-owner"),
    }


def _fine_group_assignment(build: Any) -> tuple[Mapping[str, str], str]:
    from tta.d0_v2_parameter_groups import (
        build_d0_v2_fine_group_inventory,
        verify_frozen_d0_v2_fine_inventory,
    )

    inventory = build_d0_v2_fine_group_inventory(build.model)
    verify_frozen_d0_v2_fine_inventory(inventory)
    by_name: dict[str, str] = {}
    for group in inventory.groups:
        if group.eligibility != "eligible":
            continue
        for name in group.parameter_names:
            if name in by_name:
                raise D0V3StageAWorkerError("fine group parameter overlap detected")
            by_name[name] = group.group_id
    if set(by_name) != set(build.parameter_names):
        raise D0V3StageAWorkerError("fine group assignment is incomplete")
    return (
        {name: by_name[name] for name in build.parameter_names},
        inventory.fine_inventory_sha256,
    )


def _sample(method_dataset: Any, index: int, expected_id: str) -> tuple[Any, dict[str, Any], str]:
    import torch

    value = method_dataset[index]
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
        raise D0V3StageAWorkerError("method sample exposed an unexpected field")
    image = value.pop("image")
    if (
        not isinstance(image, torch.Tensor)
        or image.shape != (3, 256, 256)
        or image.dtype != torch.float32
        or not bool(torch.isfinite(image).all().item())
        or value["image_id"] != expected_id
        or value["seed"] != SEED
    ):
        raise D0V3StageAWorkerError("method sample binding differs")
    batched = image.unsqueeze(0).contiguous()
    input_sha = input_tensor_sha256(batched.numpy())
    return batched, dict(value), input_sha


def _write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _flush_memmaps(values: Mapping[str, Any], staging: Path) -> None:
    for value in values.values():
        value.flush()
    for filename in ARRAY_FILENAMES:
        descriptor = os.open(staging / filename, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _payload_record(
    *,
    filename: str,
    file_sha256: str,
    index: list[int],
    value: Any,
    layout: FlatParameterLayout | None,
) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(value, dtype=np.dtype(FLOAT_DTYPE))
    return {
        "file": filename,
        "file_sha256": file_sha256,
        "array_index": index,
        "slice_shape": list(array.shape),
        "dtype": FLOAT_DTYPE,
        "slice_sha256": array_slice_sha256(array),
        "named_bundle_sha256": (
            None if layout is None else named_bundle_sha256(array, layout)
        ),
    }


def _build_episode_receipt(
    evidence: EpisodeEvidence,
    *,
    config_sha256: str,
    dataset: str,
    condition: str,
    replicate: str,
    process_identity: Mapping[str, Any],
    arrays: Mapping[str, Any],
    array_sha256s: Mapping[str, str],
    layout: FlatParameterLayout,
) -> dict[str, Any]:
    image_index = evidence.image_index
    candidate_index = evidence.candidate_index
    candidate = FROZEN_CANDIDATES[candidate_index]
    return {
        "schema_version": 3,
        "artifact_type": EPISODE_ARTIFACT_TYPE,
        "protocol_id": "cr-sitta-d0-v3-formal-stage-a",
        "cell_binding": {
            "config_sha256": config_sha256,
            "dataset": dataset,
            "condition": condition,
            "replicate": replicate,
            "image_index": image_index,
            "image_id": evidence.image_id,
            "candidate_index": candidate_index,
            "candidate_slug": candidate.slug,
            "split_name": "train",
            "split_role": "frozen_pilot64",
        },
        "process_identity": dict(process_identity),
        "tensor_slices": {
            "source_logits_pre": _payload_record(
                filename=SOURCE_LOGITS_FILENAME,
                file_sha256=array_sha256s[SOURCE_LOGITS_FILENAME],
                index=[image_index],
                value=arrays[SOURCE_LOGITS_FILENAME][image_index],
                layout=None,
            ),
            "tent_logits_post": _payload_record(
                filename=POST_LOGITS_FILENAME,
                file_sha256=array_sha256s[POST_LOGITS_FILENAME],
                index=[image_index, candidate_index],
                value=arrays[POST_LOGITS_FILENAME][image_index, candidate_index],
                layout=None,
            ),
            "source_parameters": _payload_record(
                filename=SOURCE_PARAMETERS_FILENAME,
                file_sha256=array_sha256s[SOURCE_PARAMETERS_FILENAME],
                index=[image_index, candidate_index],
                value=arrays[SOURCE_PARAMETERS_FILENAME][image_index, candidate_index],
                layout=layout,
            ),
            "parameters_after": _payload_record(
                filename=PARAMETERS_AFTER_FILENAME,
                file_sha256=array_sha256s[PARAMETERS_AFTER_FILENAME],
                index=[image_index, candidate_index],
                value=arrays[PARAMETERS_AFTER_FILENAME][image_index, candidate_index],
                layout=layout,
            ),
            "entropy_gradients": _payload_record(
                filename=ENTROPY_GRADIENTS_FILENAME,
                file_sha256=array_sha256s[ENTROPY_GRADIENTS_FILENAME],
                index=[image_index, candidate_index],
                value=arrays[ENTROPY_GRADIENTS_FILENAME][image_index, candidate_index],
                layout=layout,
            ),
        },
        "independent_candidate_receipt": dict(evidence.independent),
        "native_first_step": dict(evidence.native),
        "optimizer_geometry": dict(evidence.optimizer_geometry),
        "execution_gates": dict(_EXECUTION_GATES),
        "data_boundary": dict(_DATA_BOUNDARY),
        "authorization": dict(_AUTHORIZATION),
    }


def _run_candidate(
    *,
    contracts: LoadedContracts,
    dataset: str,
    condition: str,
    image_index: int,
    image_id: str,
    candidate_index: int,
    sample_image: Any,
    metadata: Mapping[str, Any],
    input_sha256: str,
    device: Any,
    process_id: str,
    runtime_sha256: str,
    determinism_sha256: str,
    ledger: IndependentCandidateExecutionLedger,
    retained_builds: list[Any],
    layout: FlatParameterLayout | None,
) -> tuple[EpisodeEvidence, Any, Any, Any, Any, FlatParameterLayout, str]:
    import numpy as np
    import torch

    from analysis.analyze_tent_optimizer_geometry import analyze_optimizer_first_step
    from tta.d0_v2_candidate_worker import build_fresh_d0_v2_candidate
    from tta.d0_v2_native_step import NativeFirstStepObserver, frozen_first_step_spec

    candidate = FROZEN_CANDIDATES[candidate_index]
    formal_candidate = contracts.formal.candidates[candidate_index]
    if (
        formal_candidate.optimizer != candidate.optimizer
        or formal_candidate.learning_rate != candidate.learning_rate
        or formal_candidate.candidate_id != candidate.slug
    ):
        raise D0V3StageAWorkerError("candidate version mapping differs")
    _seed_candidate()
    parent_dataset = dict(contracts.parent.raw["datasets"][dataset])
    if set(parent_dataset) != {
        "train_split_sha256",
        "checkpoint_role",
        "checkpoint_path",
        "checkpoint_sha256",
    }:
        raise D0V3StageAWorkerError("parent dataset builder binding fields differ")
    method_config = contracts.parent.raw["engineering_smoke"]["method"]
    build = build_fresh_d0_v2_candidate(
        project_root=PROJECT_ROOT,
        dataset_config=parent_dataset,
        candidate=candidate,
        device=device,
        entropy_eps=float(method_config["entropy_eps"]),
        diagnostic_detail=str(method_config["diagnostic_detail"]),
    )
    retained_builds.append(build)
    ledger.claim_candidate_objects(
        candidate,
        model=build.model,
        method=build.method,
        optimizer=build.method.optimizer,
        optimizer_state_entry_count=len(build.method.optimizer.state),
    )
    identity = _execution_identity(process_id, image_index, candidate_index)
    autograd_graph_token = object()
    callback_storage: tuple[tuple[str, tuple[str, int, int, int, str]], ...] | None = None

    def register_gradients(named_gradients: tuple[tuple[str, Any], ...]) -> None:
        nonlocal callback_storage
        if tuple(name for name, _ in named_gradients) != build.parameter_names:
            raise D0V3StageAWorkerError("live gradient name order differs")
        callback_storage = tuple(
            (name, _storage_identity(value)) for name, value in named_gradients
        )
        ledger.claim_candidate_backward(
            candidate,
            autograd_graph=autograd_graph_token,
            gradient_buffers=tuple(value for _, value in named_gradients),
        )

    parameters = tuple(
        parameter
        for group in build.method.optimizer.param_groups
        for parameter in group["params"]
    )
    named_parameters = tuple(zip(build.parameter_names, parameters, strict=True))
    native_observer = NativeFirstStepObserver(
        build.method.optimizer,
        named_parameters=named_parameters,
        expected_spec=frozen_first_step_spec(
            candidate.optimizer, candidate.learning_rate
        ),
        gradient_callback=register_gradients,
    )
    with native_observer:
        with FormalFirstStepCapture(
            build.method.optimizer, named_parameters=named_parameters
        ) as formal_capture:
            result = build.runner.run_one_image(
                image=sample_image.to(device, non_blocking=False),
                metadata=metadata,
                force_full_audit=True,
            )
    build.state_manager.assert_source_state()
    formal = formal_capture.tensors
    native_observation = native_observer.observation
    native = native_observation.to_dict()
    if callback_storage is None or callback_storage != formal.gradient_live_storage:
        raise D0V3StageAWorkerError("formal/callback live gradient storage differs")
    if (
        formal.parameter_names != build.parameter_names
        or native_observation.parameter_names != build.parameter_names
        or len(build.parameter_names) != PARAMETER_TENSOR_COUNT
    ):
        raise D0V3StageAWorkerError("formal/native parameter topology differs")
    candidate_layout = FlatParameterLayout.from_named_tensors(
        formal.parameters_before
    )
    validate_parameter_layout(candidate_layout.to_dict())
    if layout is not None and candidate_layout.to_dict() != layout.to_dict():
        raise D0V3StageAWorkerError("parameter layout drifted across candidates")
    source_flat = candidate_layout.pack(formal.before_dict()).numpy()
    gradient_flat = candidate_layout.pack(formal.gradient_dict()).numpy()
    after_flat = candidate_layout.pack(formal.after_dict()).numpy()
    step_flat = candidate_layout.pack(formal.step_dict()).numpy()
    derived_step = np.ascontiguousarray(after_flat - source_flat, dtype=np.dtype(FLOAT_DTYPE))
    if not np.array_equal(step_flat, derived_step):
        raise D0V3StageAWorkerError("formal adaptation step is not after-before")
    hashes = native["hashes"]
    parity = {
        "parameter_before_bundle_sha256": named_bundle_sha256(source_flat, candidate_layout),
        "gradient_bundle_sha256": named_bundle_sha256(gradient_flat, candidate_layout),
        "actual_parameter_after_bundle_sha256": named_bundle_sha256(after_flat, candidate_layout),
        "parameter_delta_bundle_sha256": named_bundle_sha256(step_flat, candidate_layout),
    }
    if any(hashes[key] != expected for key, expected in parity.items()):
        raise D0V3StageAWorkerError("formal/native lossless hash parity failed")
    if not math.isclose(
        formal.step_norm_l2,
        native_observation.step_norm_l2,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise D0V3StageAWorkerError("formal/native step norm differs")
    if (
        not torch.equal(result.logits_source_pre, result.logits_tent_pre)
        or result.source_tent_pre_bit_exact is not True
        or result.full_audit_performed is not True
        or result.full_audit_reason != "forced"
        or result.reset_full_fingerprint != build.state_manager.source_fingerprint
        or result.source_state_sha256
        != build.state_manager.source_fingerprint.full_sha256
    ):
        raise D0V3StageAWorkerError("source/pre/full-audit/reset gate failed")
    required_diagnostics = {
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
        key: {"expected": expected, "observed": result.diagnostics.get(key)}
        for key, expected in required_diagnostics.items()
        if result.diagnostics.get(key) != expected
    }
    if mismatches:
        raise D0V3StageAWorkerError(f"runner diagnostic gates failed: {mismatches}")
    counts = native["counts"]
    if (
        counts["parameter_tensor_count"] != PARAMETER_TENSOR_COUNT
        or counts["gradient_tensor_count"] != PARAMETER_TENSOR_COUNT
        or counts["scalar_parameter_count"] != PARAMETER_SCALAR_COUNT
        or counts["optimizer_state_parameter_count"] != PARAMETER_TENSOR_COUNT
        or hashes["actual_parameter_after_bundle_sha256"]
        != hashes["reference_parameter_after_bundle_sha256"]
        or hashes["actual_optimizer_state_bundle_sha256"]
        != hashes["reference_optimizer_state_bundle_sha256"]
        or any(value is not True for value in native["gates"].values())
    ):
        raise D0V3StageAWorkerError("native first-step gate failed")
    if int(counts["changed_parameter_tensor_count"]) != int(
        result.diagnostics["number_updated_bn_affine_tensors"]
    ):
        raise D0V3StageAWorkerError("native/runner changed tensor count differs")
    selected_names_sha = hashlib.sha256(
        "\0".join(build.parameter_names).encode("utf-8")
    ).hexdigest()
    if result.diagnostics["parameter_names_sha256"] != selected_names_sha:
        raise D0V3StageAWorkerError("selected parameter name SHA differs")
    source_logits = result.logits_source_pre.detach().cpu().contiguous().numpy()
    post_logits = result.logits_tent_post.detach().cpu().contiguous().numpy()
    if source_logits.shape != (1, 1, 256, 256) or post_logits.shape != (1, 1, 256, 256):
        raise D0V3StageAWorkerError("runner logit shape differs")

    split_sha = str(parent_dataset["train_split_sha256"])
    checkpoint_sha = str(parent_dataset["checkpoint_sha256"])
    provenance = SourceTrainAnalysisProvenance(
        dataset=dataset,
        split_name="train",
        split_sha256=split_sha,
        checkpoint_sha256=checkpoint_sha,
        seed=SEED,
        oracle_analysis=False,
        outer_evaluator_label_accesses=0,
        supervised_gradient_role="none",
    )
    group_assignment, inventory_sha = _fine_group_assignment(build)
    numeric = contracts.formal.raw["formal_numeric_protocol"]["optimizer_geometry"]
    geometry = analyze_optimizer_first_step(
        parameters_before=formal.parameters_before,
        gradients=formal.gradients,
        parameters_after=formal.parameters_after,
        optimizer_config=frozen_first_step_spec(
            candidate.optimizer, candidate.learning_rate
        ).to_dict(),
        provenance=provenance,
        parameter_groups=group_assignment,
        small_gradient_threshold=float(numeric["small_gradient_threshold"]),
        near_sign_step_threshold=float(numeric["adam_near_sign_step_threshold"]),
        verification_rtol=float(numeric["verification_rtol"]),
        verification_atol=float(numeric["verification_atol"]),
    )
    independent = IndependentCandidateReceipt(
        candidate_index=candidate_index,
        candidate=candidate,
        config_sha256=str(contracts.formal.config_file_sha256),
        dataset=dataset,
        condition=condition,
        sample_index=image_index,
        sample_id=image_id,
        split_sha256=split_sha,
        checkpoint_sha256=checkpoint_sha,
        source_state_sha256=_source_state_sha256(
            build.state_manager.source_fingerprint
        ),
        runtime_sha256=runtime_sha256,
        determinism_sha256=determinism_sha256,
        input_sha256=input_sha256,
        selected_parameter_names_sha256=selected_names_sha,
        pre_logits_sha256=str(result.diagnostics["source_pre_raw_sha256"]),
        post_logits_sha256=str(result.diagnostics["tent_post_raw_sha256"]),
        entropy_gradient_bundle_sha256=str(hashes["gradient_bundle_sha256"]),
        parameter_delta_bundle_sha256=str(hashes["parameter_delta_bundle_sha256"]),
        model_instance_id=identity["model_instance_id"],
        method_instance_id=identity["method_instance_id"],
        optimizer_instance_id=identity["optimizer_instance_id"],
        autograd_graph_id=identity["autograd_graph_id"],
        backward_execution_id=identity["backward_execution_id"],
        gradient_buffer_owner_id=identity["gradient_buffer_owner_id"],
        gradient_tensor_count=int(counts["gradient_tensor_count"]),
        changed_parameter_tensor_count=int(counts["changed_parameter_tensor_count"]),
        optimizer_state_entry_count_after_step=int(counts["optimizer_state_parameter_count"]),
        native_reference_parameter_tensor_count=int(
            counts["native_reference_parameter_tensor_count"]
        ),
        native_reference_optimizer_state_tensor_count=int(
            counts["native_reference_optimizer_state_tensor_count"]
        ),
        step_norm_l2=float(native["step_norm_l2"]),
    ).to_dict()
    evidence = EpisodeEvidence(
        image_index=image_index,
        image_id=image_id,
        candidate_index=candidate_index,
        independent=independent,
        native=native,
        optimizer_geometry=geometry,
    )
    return (
        evidence,
        source_logits[0],
        post_logits[0],
        source_flat,
        (after_flat, gradient_flat),
        candidate_layout,
        inventory_sha,
    )


def _destination(
    contract: D0V3FormalContract,
    *,
    dataset: str,
    condition: str,
    replicate: str,
    max_images: int,
) -> Path:
    root = PROJECT_ROOT / contract.output_root
    if max_images == FORMAL_IMAGE_COUNT:
        return root / "candidate_phase" / "shards" / replicate / dataset / condition
    return (
        root
        / "engineering_dry_runs"
        / "candidate_phase"
        / replicate
        / dataset
        / condition
        / "max_images_1"
    )


def _make_destination_parent(destination: Path) -> None:
    relative = destination.parent.relative_to(PROJECT_ROOT)
    ensure_directory_chain_nofollow(PROJECT_ROOT, relative.parts)


def validate_contract_only(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    contracts = _load_contracts(config_path)
    return {
        "schema_version": 3,
        "role": "d0_v3_formal_stage_a_label_free_validate_only",
        "valid": True,
        "config_sha256": contracts.formal.config_file_sha256,
        "formal_execution": "R0_64_images_x_10_candidates_only",
        "followup_execution": "R1_R2_require_verified_signed_R0_eligibility_receipt",
        "dry_run": "max_images_1_is_nonformal",
        "filesystem_created": False,
        "gpu_initialized": False,
        "target_payload_opened": False,
        "validation_payload_opened": False,
        "test_payload_opened": False,
        "formal_protocol_complete": False,
        "stage2_authorized": False,
    }


def run_worker(
    *,
    config_path: Path,
    dataset: str,
    condition: str,
    replicate: str,
    max_images: int,
    staging: Path,
    process_id: str,
    parent_nonce: str,
    child_nonce: str,
) -> dict[str, Any]:
    """Execute one complete R0 cell or one-image non-formal dry-run."""

    import numpy as np
    import torch

    contracts = _load_contracts(config_path)
    if dataset not in contracts.formal.datasets or condition not in contracts.formal.conditions:
        raise D0V3StageAWorkerError("dataset/condition is outside frozen formal grid")
    if replicate != "R0":
        raise D0V3StageAWorkerError(
            "R1/R2 require a verified signed R0 eligibility receipt and a subset-aware worker"
        )
    if max_images not in {1, FORMAL_IMAGE_COUNT}:
        raise D0V3StageAWorkerError("max_images must be exactly 1 or 64")
    if not staging.is_absolute() or staging.parent == staging or not staging.name.startswith("."):
        raise D0V3StageAWorkerError("worker staging must be an absolute hidden directory")
    if not staging.is_dir() or any(staging.iterdir()):
        raise D0V3StageAWorkerError("worker staging must exist and be empty")
    config_sha = str(contracts.formal.config_file_sha256)
    command_sha = _validate_worker_binding(
        process_id=process_id,
        parent_nonce=parent_nonce,
        child_nonce=child_nonce,
        config_sha256=config_sha,
    )
    input_seal = _capture_input_seal(
        contracts,
        config_path=config_path,
        dataset=dataset,
        condition=condition,
    )
    device = _configure_cuda_worker()
    from analysis.d0_determinism_runtime import frozen_determinism_sha256
    determinism_sha = frozen_determinism_sha256(
        contracts.parent.raw["engineering_smoke"]["method"]["determinism"]
    )
    runtime_sha = _runtime_sha256(
        input_seal_sha256=input_seal.seal_sha256,
        code_seals=input_seal.code_seals,
        determinism_sha256=determinism_sha,
        device=device,
    )
    from materialize_binary_tent_ss_calibration_cache_v2 import (
        SourceCalibrationMethodInputDatasetV2,
    )
    method_dataset = SourceCalibrationMethodInputDatasetV2(
        input_seal.cache_root,
        condition_key=condition,
        expected_protocol_sha256=str(contracts.formal.raw["cache"]["protocol_sha256"]),
    )
    if len(method_dataset) != FORMAL_IMAGE_COUNT or tuple(method_dataset.image_ids) != input_seal.image_ids:
        raise D0V3StageAWorkerError("method dataset Pilot64 order differs")

    specs = expected_array_specs(max_images)
    arrays = {
        filename: np.lib.format.open_memmap(
            staging / filename,
            mode="w+",
            dtype=np.dtype(FLOAT_DTYPE),
            shape=shape,
            fortran_order=False,
        )
        for filename, shape in specs.items()
    }
    evidence: list[EpisodeEvidence] = []
    layout: FlatParameterLayout | None = None
    inventory_sha: str | None = None
    for image_index in range(max_images):
        image_id = input_seal.image_ids[image_index]
        sample_image, metadata, input_sha = _sample(method_dataset, image_index, image_id)
        ledger = IndependentCandidateExecutionLedger()
        retained_builds: list[Any] = []
        image_receipts: list[Mapping[str, Any]] = []
        shared_source_logits: Any | None = None
        for candidate_index in range(CANDIDATE_COUNT):
            (
                episode,
                source_logits,
                post_logits,
                source_parameters,
                after_and_gradient,
                observed_layout,
                observed_inventory_sha,
            ) = _run_candidate(
                contracts=contracts,
                dataset=dataset,
                condition=condition,
                image_index=image_index,
                image_id=image_id,
                candidate_index=candidate_index,
                sample_image=sample_image,
                metadata=metadata,
                input_sha256=input_sha,
                device=device,
                process_id=process_id,
                runtime_sha256=runtime_sha,
                determinism_sha256=determinism_sha,
                ledger=ledger,
                retained_builds=retained_builds,
                layout=layout,
            )
            if layout is None:
                layout = observed_layout
                inventory_sha = observed_inventory_sha
            elif observed_inventory_sha != inventory_sha:
                raise D0V3StageAWorkerError("fine inventory drifted across candidates")
            if shared_source_logits is None:
                shared_source_logits = source_logits.copy()
                arrays[SOURCE_LOGITS_FILENAME][image_index] = source_logits
            elif not np.array_equal(shared_source_logits, source_logits):
                raise D0V3StageAWorkerError(
                    "source logits differ across candidates for one image"
                )
            arrays[POST_LOGITS_FILENAME][image_index, candidate_index] = post_logits
            arrays[SOURCE_PARAMETERS_FILENAME][image_index, candidate_index] = source_parameters
            arrays[PARAMETERS_AFTER_FILENAME][image_index, candidate_index] = after_and_gradient[0]
            arrays[ENTROPY_GRADIENTS_FILENAME][image_index, candidate_index] = after_and_gradient[1]
            evidence.append(episode)
            image_receipts.append(episode.independent)
        ledger_summary = ledger.assert_complete()
        if (
            ledger_summary["gradient_reuse"] is not False
            or ledger_summary["engineering_gate_passed"] is not True
            or len(retained_builds) != CANDIDATE_COUNT
        ):
            raise D0V3StageAWorkerError("live independent-candidate ledger failed")
        validate_independent_candidate_cell(image_receipts)
        retained_builds.clear()
        del retained_builds
        torch.cuda.empty_cache()
    if layout is None or inventory_sha is None:
        raise D0V3StageAWorkerError("no candidate layout/inventory was captured")
    _flush_memmaps(arrays, staging)
    _write_exclusive(
        staging / LAYOUT_FILENAME,
        canonical_json_bytes(layout.to_dict(), newline=True),
    )
    array_sha256s = {
        filename: read_stable_regular_file(staging / filename).sha256
        for filename in ARRAY_FILENAMES
    }
    process_identity = FreshSubprocessIdentity.capture_current(process_id)
    persisted_process = {
        "logical_process_id": process_identity.process_id,
        "os_process_id": process_identity.os_process_id,
        "process_start_time_ticks": process_identity.process_start_time_ticks,
        "parent_run_nonce": parent_nonce,
        "child_launch_nonce": child_nonce,
        "command_sha256": command_sha,
    }
    episode_receipts = [
        _build_episode_receipt(
            item,
            config_sha256=config_sha,
            dataset=dataset,
            condition=condition,
            replicate=replicate,
            process_identity=persisted_process,
            arrays=arrays,
            array_sha256s=array_sha256s,
            layout=layout,
        )
        for item in evidence
    ]
    episode_lines = b"".join(
        canonical_json_bytes(value, newline=True) for value in episode_receipts
    )
    _write_exclusive(staging / EPISODES_FILENAME, episode_lines)
    episode_hashes = [
        hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        for value in episode_receipts
    ]
    formal = max_images == FORMAL_IMAGE_COUNT
    phase_receipt: Mapping[str, Any] | None = None
    phase_sha: str | None = None
    if formal:
        phase_records = [
            {
                "image_index": item.image_index,
                "image_id": item.image_id,
                "candidate_index": item.candidate_index,
                "candidate_slug": FROZEN_CANDIDATES[item.candidate_index].slug,
                "complete": True,
                "episode_receipt_sha256": digest,
            }
            for item, digest in zip(evidence, episode_hashes, strict=True)
        ]
        phase_receipt = build_label_free_cell_receipt(
            dataset=dataset,
            condition=condition,
            replicate=0,
            ordered_image_ids=input_seal.image_ids,
            episode_records=phase_records,
            cache_protocol_sha256=str(contracts.formal.raw["cache"]["protocol_sha256"]),
            cache_method_manifest_sha256=input_seal.method_manifest_sha256,
            checkpoint_sha256=str(contracts.formal.raw["datasets"][dataset]["checkpoint_sha256"]),
            config_sha256=config_sha,
            code_seals=input_seal.code_seals,
            candidate_phase_data_boundary=zero_candidate_phase_data_boundary(),
        )
        phase_bytes = canonical_label_free_cell_receipt_bytes(phase_receipt)
        _write_exclusive(staging / PHASE_RECEIPT_FILENAME, phase_bytes)
        phase_sha = hashlib.sha256(phase_bytes).hexdigest()
    array_records = {
        filename: {
            "path": filename,
            "sha256": array_sha256s[filename],
            "shape": list(specs[filename]),
            "dtype": FLOAT_DTYPE,
            "c_order": True,
            "finite": True,
            "lossless": True,
        }
        for filename in ARRAY_FILENAMES
    }
    dataset_binding = dict(contracts.formal.raw["datasets"][dataset])
    manifest = {
        "schema_version": 3,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": "cr-sitta-d0-v3-formal-stage-a",
        "config_sha256": config_sha,
        "cell": {"dataset": dataset, "condition": condition, "replicate": replicate},
        "mode": {
            "formal": formal,
            "dry_run": not formal,
            "image_count": max_images,
            "candidate_count": CANDIDATE_COUNT,
            "episode_count": len(episode_receipts),
            "episode_order": "image_major_candidate_minor",
        },
        "process_identity": persisted_process,
        "dataset_binding": dataset_binding,
        "ordered_image_ids": list(input_seal.image_ids[:max_images]),
        "ordered_image_ids_sha256": hashlib.sha256(
            json.dumps(
                list(input_seal.image_ids[:max_images]),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "candidate_slugs": [value.slug for value in FROZEN_CANDIDATES],
        "parameter_layout": {
            "path": LAYOUT_FILENAME,
            "sha256": read_stable_regular_file(staging / LAYOUT_FILENAME).sha256,
            "layout_sha256": layout.layout_sha256,
            "parameter_tensor_count": PARAMETER_TENSOR_COUNT,
            "scalar_parameter_count": PARAMETER_SCALAR_COUNT,
        },
        "fine_inventory_sha256": inventory_sha,
        "input_seal": input_seal.to_dict(),
        "arrays": array_records,
        "episode_receipts": {
            "path": EPISODES_FILENAME,
            "sha256": read_stable_regular_file(staging / EPISODES_FILENAME).sha256,
            "count": len(episode_receipts),
            "order": "image_major_candidate_minor",
            "episode_receipt_hashes_sha256": canonical_sha256(episode_hashes),
        },
        "phase_receipt": (
            {
                "path": PHASE_RECEIPT_FILENAME,
                "sha256": phase_sha,
                "label_free_phase_complete": True,
                "outer_target_access_authorized_for_this_cell": True,
                "formal_protocol_complete": False,
                "stage2_authorized": False,
            }
            if formal
            else None
        ),
        "data_boundary": dict(_DATA_BOUNDARY),
        "authorization": dict(_AUTHORIZATION),
    }
    _write_exclusive(
        staging / MANIFEST_FILENAME,
        canonical_json_bytes(manifest, newline=True),
    )
    members_without_complete = FORMAL_MEMBERS if formal else DRY_RUN_MEMBERS
    payload_names = sorted(set(members_without_complete) - {COMPLETE_FILENAME})
    complete = {
        "schema_version": 3,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "complete": True,
        "candidate_phase_complete": True,
        "formal": formal,
        "dry_run": not formal,
        "config_sha256": config_sha,
        "dataset": dataset,
        "condition": condition,
        "replicate": replicate,
        "image_count": max_images,
        "candidate_count": CANDIDATE_COUNT,
        "episode_count": len(episode_receipts),
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": read_stable_regular_file(staging / MANIFEST_FILENAME).sha256,
        },
        "payload_files": [
            {"path": name, "sha256": read_stable_regular_file(staging / name).sha256}
            for name in payload_names
        ],
        "atomic_no_replace": True,
        "paper_result": False,
        "formal_protocol_complete": False,
        "scientific_gate_status": "not_evaluated",
        "stage2_authorized": False,
    }
    _write_exclusive(
        staging / COMPLETE_FILENAME,
        canonical_json_bytes(complete, newline=True),
    )
    exit_seal = _capture_input_seal(
        contracts,
        config_path=config_path,
        dataset=dataset,
        condition=condition,
    )
    if exit_seal.files != input_seal.files or exit_seal.seal_sha256 != input_seal.seal_sha256:
        raise D0V3StageAWorkerError("method/code input lineage changed during worker")
    verified = verify_label_free_shard(
        staging,
        expected_config_sha256=config_sha,
        verify_live_inputs=True,
        repository_root=PROJECT_ROOT,
    )
    return {
        "valid": True,
        "path": str(staging),
        "dataset": verified.dataset,
        "condition": verified.condition,
        "replicate": verified.replicate,
        "formal": verified.formal,
        "dry_run": verified.dry_run,
        "image_count": verified.image_count,
        "episode_count": verified.episode_count,
        "manifest_sha256": verified.manifest_sha256,
        "stage2_authorized": False,
    }


def _worker_command(
    *,
    config_path: Path,
    dataset: str,
    condition: str,
    replicate: str,
    max_images: int,
    staging: Path,
    process_id: str,
    parent_nonce: str,
    child_nonce: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_worker",
        "--config",
        str(config_path),
        "--dataset",
        dataset,
        "--condition",
        condition,
        "--replicate",
        replicate,
        "--max-images",
        str(max_images),
        "--staging",
        str(staging),
        "--process-id",
        process_id,
        "--parent-run-nonce",
        parent_nonce,
        "--child-launch-nonce",
        child_nonce,
    ]


def run_parent(
    *,
    config_path: Path,
    dataset: str,
    condition: str,
    replicate: str,
    max_images: int,
    cuda_visible_device: str,
) -> dict[str, Any]:
    contracts = _load_contracts(config_path)
    if dataset not in contracts.formal.datasets or condition not in contracts.formal.conditions:
        raise D0V3StageAWorkerError("dataset/condition is outside frozen formal grid")
    if replicate != "R0":
        raise D0V3StageAWorkerError(
            "R1/R2 are blocked until a verified signed R0 eligibility receipt exists"
        )
    if max_images not in {1, FORMAL_IMAGE_COUNT}:
        raise D0V3StageAWorkerError("--max-images must be exactly 1 or 64")
    visible = _validate_visible_device(cuda_visible_device)
    destination = _destination(
        contracts.formal,
        dataset=dataset,
        condition=condition,
        replicate=replicate,
        max_images=max_images,
    )
    _make_destination_parent(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"formal shard destination already exists: {destination}")
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.staging"
    os.mkdir(staging, mode=0o700)
    parent_nonce = hashlib.sha256(os.urandom(32)).hexdigest()
    child_nonce = hashlib.sha256(os.urandom(32)).hexdigest()
    process_id = f"{replicate}-{dataset}-{condition}-{uuid.uuid4().hex}"
    command = _worker_command(
        config_path=config_path,
        dataset=dataset,
        condition=condition,
        replicate=replicate,
        max_images=max_images,
        staging=staging,
        process_id=process_id,
        parent_nonce=parent_nonce,
        child_nonce=child_nonce,
    )
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
            ENV_CONFIG_SHA256: str(contracts.formal.config_file_sha256),
            ENV_COMMAND_SHA256: _canonical_command_sha256(command),
        }
    )
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        close_fds=True,
        check=False,
    )
    if completed.returncode != 0:
        raise D0V3StageAWorkerError(
            "fresh label-free worker failed; private staging is retained at "
            f"{staging}; stdout={completed.stdout[-4000:]!r}; "
            f"stderr={completed.stderr[-4000:]!r}"
        )
    expected_members = FORMAL_MEMBERS if max_images == FORMAL_IMAGE_COUNT else DRY_RUN_MEMBERS
    published = publish_flat_directory_noreplace(
        staging,
        destination,
        expected_members=tuple(expected_members),
        semantic_verifier=lambda path: verify_label_free_shard(
            path,
            expected_config_sha256=str(contracts.formal.config_file_sha256),
            verify_live_inputs=True,
            repository_root=PROJECT_ROOT,
        ),
    )
    verified = verify_label_free_shard(
        published,
        expected_config_sha256=str(contracts.formal.config_file_sha256),
        verify_live_inputs=True,
        repository_root=PROJECT_ROOT,
    )
    return {
        "published": True,
        "path": str(published),
        "dataset": verified.dataset,
        "condition": verified.condition,
        "replicate": verified.replicate,
        "formal": verified.formal,
        "dry_run": verified.dry_run,
        "image_count": verified.image_count,
        "episode_count": verified.episode_count,
        "manifest_sha256": verified.manifest_sha256,
        "paper_result": False,
        "formal_protocol_complete": False,
        "stage2_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "D0-v3 formal Stage-A label-free R0 worker. Formal means 64x10; "
            "--max-images 1 is a non-formal engineering dry-run. This command "
            "never authorizes Stage 2."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="CPU-only frozen contract validation")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify = sub.add_parser("verify", help="CPU-only full-byte shard verification")
    verify.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    verify.add_argument("--path", type=Path, required=True)
    run = sub.add_parser("run", help="launch one fresh GPU R0 child and publish")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--dataset", required=True)
    run.add_argument("--condition", required=True)
    run.add_argument("--replicate", choices=REPLICATES, default="R0")
    run.add_argument("--max-images", type=int, choices=(1, 64), default=64)
    run.add_argument("--cuda-visible-device", required=True)
    worker = sub.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--config", type=Path, required=True)
    worker.add_argument("--dataset", required=True)
    worker.add_argument("--condition", required=True)
    worker.add_argument("--replicate", required=True)
    worker.add_argument("--max-images", type=int, required=True)
    worker.add_argument("--staging", type=Path, required=True)
    worker.add_argument("--process-id", required=True)
    worker.add_argument("--parent-run-nonce", required=True)
    worker.add_argument("--child-launch-nonce", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = args.config.absolute()
    if not config.is_relative_to(PROJECT_ROOT):
        raise D0V3StageAWorkerError("config must be inside the repository")
    if args.command == "validate":
        result = validate_contract_only(config)
    elif args.command == "verify":
        contract = _load_contracts(config).formal
        verified = verify_label_free_shard(
            args.path,
            expected_config_sha256=str(contract.config_file_sha256),
            verify_live_inputs=True,
            repository_root=PROJECT_ROOT,
        )
        result = {
            "valid": True,
            "path": str(verified.path),
            "formal": verified.formal,
            "dry_run": verified.dry_run,
            "episode_count": verified.episode_count,
            "manifest_sha256": verified.manifest_sha256,
            "formal_protocol_complete": False,
            "stage2_authorized": False,
        }
    elif args.command == "run":
        result = run_parent(
            config_path=config,
            dataset=args.dataset,
            condition=args.condition,
            replicate=args.replicate,
            max_images=args.max_images,
            cuda_visible_device=args.cuda_visible_device,
        )
    else:
        result = run_worker(
            config_path=config,
            dataset=args.dataset,
            condition=args.condition,
            replicate=args.replicate,
            max_images=args.max_images,
            staging=args.staging.absolute(),
            process_id=args.process_id,
            parent_nonce=args.parent_run_nonce,
            child_nonce=args.child_launch_nonce,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
