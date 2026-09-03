#!/usr/bin/env python3
"""Run or CPU-verify one formal P3 Stage-A outer-analysis cell.

Formal execution is restricted to R0 and all 64 frozen train Pilot images.
The one-image dry-run command performs only candidate-shard verification and
cannot reach the real target loader.  This script never ranks candidates and
every published artifact remains paper=false and stage2=false.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.d0_v2_task_loss import D0V2TaskLossConfig  # noqa: E402
from analysis.d0_v3_formal_contract import (  # noqa: E402
    CONFIG_FILE_SHA256,
    CONFIG_RELATIVE_PATH,
    D0V3FormalContract,
    FINE_ALIGNMENT_GROUP_IDS,
    FROZEN_CANDIDATES,
    load_d0_v3_formal_contract,
    verify_frozen_parent_bindings,
)
from analysis.d0_v3_label_free_shard import (  # noqa: E402
    CANDIDATE_COUNT,
    ENTROPY_GRADIENTS_FILENAME,
    EPISODES_FILENAME,
    FLOAT_DTYPE,
    FORMAL_IMAGE_COUNT,
    LAYOUT_FILENAME,
    MANIFEST_FILENAME as LABEL_FREE_MANIFEST_FILENAME,
    PARAMETERS_AFTER_FILENAME,
    PHASE_RECEIPT_FILENAME,
    POST_LOGITS_FILENAME,
    SOURCE_LOGITS_FILENAME,
    SOURCE_PARAMETERS_FILENAME,
    array_slice_sha256,
    canonical_json_bytes,
    parse_canonical_json,
    validate_parameter_layout,
    verify_label_free_shard,
)
from analysis.d0_v3_outer_analyzer import analyze_formal_stage_a_episode  # noqa: E402
from analysis.d0_v3_outer_cell_shard import (  # noqa: E402
    ARTIFACT_TYPE,
    COMPLETE_ARTIFACT_TYPE,
    COMPLETE_FILENAME,
    EPISODE_ORDER,
    MANIFEST_FILENAME,
    MEMBERS,
    OUTER_ACCESS_RECEIPT_FILENAME,
    OUTER_RECORDS_FILENAME,
    OUTER_SOURCE_LOGITS_FILENAME,
    RECORD_ARTIFACT_TYPE,
    SCHEMA_VERSION,
    SUPERVISED_GRADIENTS_FILENAME,
    TASK_AUDIT_ARTIFACT_TYPE,
    TASK_LOSS_AUDITS_FILENAME,
    build_outer_code_seal,
    canonical_jsonl_bytes,
    verify_outer_cell_shard,
)
from analysis.d0_v3_phase_receipt import (  # noqa: E402
    canonical_label_free_cell_receipt_bytes,
    guarded_load_outer_targets,
    validate_label_free_cell_receipt,
)
from analysis.source_train_provenance import (  # noqa: E402
    OUTER_ORACLE_ROLE,
    SourceTrainAnalysisProvenance,
)
from materialize_binary_tent_ss_calibration_cache_v2 import (  # noqa: E402
    SourceCalibrationMethodInputDatasetV2,
)
from tta.d0_secure_io import read_stable_regular_file  # noqa: E402
from tta.d0_v3_atomic_shard import publish_flat_directory_noreplace  # noqa: E402
from tta.d0_v3_outer_source_gradient import (  # noqa: E402
    build_d0_v3_outer_source_model,
    compute_d0_v3_outer_source_gradient,
)
from tta.diagnostics import NoOpThresholds  # noqa: E402


class D0V3FormalOuterRunnerError(RuntimeError):
    """Formal outer execution violated its frozen phase or output contract."""


def _freeze_fine_group_order(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Seal per-group diagnostics in the protocol's frozen semantic order."""

    result = dict(analysis)
    alignment_raw = result.get("entropy_task_alignment")
    if not isinstance(alignment_raw, Mapping):
        raise D0V3FormalOuterRunnerError("outer alignment must be a mapping")
    alignment = dict(alignment_raw)
    per_group_raw = alignment.get("per_group")
    if not isinstance(per_group_raw, Mapping):
        raise D0V3FormalOuterRunnerError("outer per_group must be a mapping")
    observed = set(per_group_raw)
    expected = set(FINE_ALIGNMENT_GROUP_IDS)
    if observed != expected or len(per_group_raw) != len(FINE_ALIGNMENT_GROUP_IDS):
        raise D0V3FormalOuterRunnerError(
            "outer per_group keys differ from frozen fine-group IDs"
        )
    alignment["per_group"] = {
        group_id: per_group_raw[group_id]
        for group_id in FINE_ALIGNMENT_GROUP_IDS
    }
    result["entropy_task_alignment"] = alignment
    return result


def _configure_formal_forward_runtime(*, device: torch.device, seed: int) -> dict[str, Any]:
    """Reproduce the sealed candidate worker's forward-time determinism policy."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise D0V3FormalOuterRunnerError("formal forward seed must be an integer")
    if device.type == "cuda":
        required = {
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        mismatches = {
            key: {"expected": expected, "observed": os.environ.get(key)}
            for key, expected in required.items()
            if os.environ.get(key) != expected
        }
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if (
            not isinstance(visible, str)
            or not visible
            or "," in visible
            or any(character.isspace() for character in visible)
            or any(character in visible for character in ("/", "\x00", "\r", "\n"))
        ):
            mismatches["CUDA_VISIBLE_DEVICES"] = {
                "expected": "exactly one safe device",
                "observed": visible,
            }
        if mismatches:
            raise D0V3FormalOuterRunnerError(
                f"outer CUDA environment differs from candidate runtime: {mismatches}"
            )
        if device != torch.device("cuda:0"):
            raise D0V3FormalOuterRunnerError(
                "formal outer worker must address its sole visible GPU as cuda:0"
            )
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise D0V3FormalOuterRunnerError(
                "formal outer worker must see exactly one CUDA device"
            )
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise D0V3FormalOuterRunnerError("formal forward device must be CPU or CUDA")

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    audit = {
        "seed": seed,
        "device": str(device),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "visible_cuda_device_count": (
            int(torch.cuda.device_count()) if device.type == "cuda" else 0
        ),
    }
    if not audit["deterministic_algorithms"] or audit["deterministic_warn_only"]:
        raise D0V3FormalOuterRunnerError(
            "strict deterministic outer forwards were not enabled"
        )
    return audit


def _sha256_file(path: Path) -> str:
    return read_stable_regular_file(path).sha256


def _repository_relative(path: Path) -> str:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError as exc:
        raise D0V3FormalOuterRunnerError(
            f"formal path is outside project root: {absolute}"
        ) from exc


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npy(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype=np.dtype(FLOAT_DTYPE))
    if not np.isfinite(array).all():
        raise D0V3FormalOuterRunnerError(f"refusing non-finite array: {path.name}")
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _load_contract(path: Path) -> D0V3FormalContract:
    contract = load_d0_v3_formal_contract(path)
    if contract.config_file_sha256 != CONFIG_FILE_SHA256:
        raise D0V3FormalOuterRunnerError("formal config file SHA differs")
    verify_frozen_parent_bindings(contract, repository_root=PROJECT_ROOT)
    if contract.stage2_authorized:
        raise D0V3FormalOuterRunnerError("formal config unexpectedly authorizes Stage 2")
    return contract


def _fixed_label_free_path(
    contract: D0V3FormalContract, dataset: str, condition: str
) -> Path:
    root = PROJECT_ROOT / contract.output_root
    return root / "candidate_phase" / "shards" / "R0" / dataset / condition


def _fixed_outer_path(
    contract: D0V3FormalContract, dataset: str, condition: str
) -> Path:
    root = PROJECT_ROOT / contract.output_root
    return root / "outer_phase" / "shards" / "R0" / dataset / condition


def _jsonl(data: bytes, *, expected_count: int, label: str) -> list[Mapping[str, Any]]:
    if not data.endswith(b"\n"):
        raise D0V3FormalOuterRunnerError(f"{label} is not newline-terminated JSONL")
    lines = data[:-1].split(b"\n")
    if len(lines) != expected_count or any(not line for line in lines):
        raise D0V3FormalOuterRunnerError(f"{label} record count differs")
    values: list[Mapping[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise D0V3FormalOuterRunnerError(
                f"{label}[{index}] is invalid JSON"
            ) from exc
        if not isinstance(value, Mapping):
            raise D0V3FormalOuterRunnerError(f"{label}[{index}] is not a mapping")
        values.append(value)
    return values


def _input_seal_by_role(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    seal = manifest.get("input_seal")
    if not isinstance(seal, Mapping) or not isinstance(seal.get("files"), Sequence):
        raise D0V3FormalOuterRunnerError("label-free input seal is missing")
    by_role: dict[str, Mapping[str, Any]] = {}
    for raw in seal["files"]:
        if not isinstance(raw, Mapping) or set(raw) != {"role", "path", "sha256"}:
            raise D0V3FormalOuterRunnerError("label-free input seal record differs")
        role = raw["role"]
        if not isinstance(role, str) or not role or role in by_role:
            raise D0V3FormalOuterRunnerError("label-free input seal role differs")
        by_role[role] = raw
    for role in ("cache_protocol", "cache_method_manifest"):
        if role not in by_role:
            raise D0V3FormalOuterRunnerError(f"label-free input seal lacks {role}")
    return by_role


def _formal_label_free_preflight(
    *,
    label_free_path: Path,
    contract: D0V3FormalContract,
    dataset: str,
    condition: str,
) -> tuple[Any, Mapping[str, Any], Any, Any]:
    """Complete the public CPU and canonical phase checks before any target call."""

    verified = verify_label_free_shard(
        label_free_path,
        expected_config_sha256=str(contract.config_file_sha256),
        verify_live_inputs=True,
        repository_root=PROJECT_ROOT,
    )
    if (
        verified.dataset != dataset
        or verified.condition != condition
        or verified.replicate != "R0"
        or not verified.formal
        or verified.dry_run
        or verified.image_count != FORMAL_IMAGE_COUNT
        or verified.candidate_count != CANDIDATE_COUNT
        or verified.episode_count != FORMAL_IMAGE_COUNT * CANDIDATE_COUNT
        or verified.phase_receipt_sha256 is None
    ):
        raise D0V3FormalOuterRunnerError(
            "outer execution requires a complete matching formal R0/64 label-free shard"
        )
    manifest = parse_canonical_json(
        read_stable_regular_file(label_free_path / LABEL_FREE_MANIFEST_FILENAME).data,
        label="label-free manifest",
        newline=True,
    )
    phase_snapshot = read_stable_regular_file(
        label_free_path / PHASE_RECEIPT_FILENAME
    )
    phase = validate_label_free_cell_receipt(
        phase_snapshot.data,
        expected_dataset=dataset,
        expected_condition=condition,
        expected_replicate=0,
        expected_cache_protocol_sha256=contract.raw["cache"]["protocol_sha256"],
        expected_checkpoint_sha256=contract.raw["datasets"][dataset][
            "checkpoint_sha256"
        ],
        expected_config_sha256=str(contract.config_file_sha256),
        expected_ordered_image_ids=manifest["ordered_image_ids"],
        expected_receipt_sha256=verified.phase_receipt_sha256,
    )
    if canonical_label_free_cell_receipt_bytes(phase_snapshot.data) != phase_snapshot.data:
        raise D0V3FormalOuterRunnerError("phase receipt is not canonical")
    dataset_binding = manifest.get("dataset_binding")
    expected_dataset_binding = {
        "train_split_sha256": contract.raw["datasets"][dataset][
            "train_split_sha256"
        ],
        "checkpoint_role": "best_miou",
        "checkpoint_path": contract.raw["datasets"][dataset]["checkpoint_path"],
        "checkpoint_sha256": contract.raw["datasets"][dataset][
            "checkpoint_sha256"
        ],
    }
    if dataset_binding != expected_dataset_binding:
        raise D0V3FormalOuterRunnerError(
            "label-free dataset/checkpoint binding differs from formal config"
        )
    input_by_role = _input_seal_by_role(manifest)
    if input_by_role["cache_protocol"]["sha256"] != contract.raw["cache"][
        "protocol_sha256"
    ]:
        raise D0V3FormalOuterRunnerError(
            "label-free cache protocol binding differs from formal config"
        )
    if input_by_role["cache_method_manifest"]["sha256"] != (
        phase.cache_method_manifest_sha256
    ):
        raise D0V3FormalOuterRunnerError(
            "phase receipt/cache method manifest binding differs"
        )
    layout = validate_parameter_layout(
        parse_canonical_json(
            read_stable_regular_file(label_free_path / LAYOUT_FILENAME).data,
            label="label-free layout",
            newline=True,
        )
    )
    return verified, manifest, phase, layout


def dry_run_preflight(
    *,
    label_free_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Verify a one-image candidate dry-run without any target-loader path."""

    contract = _load_contract(config_path)
    verified = verify_label_free_shard(
        label_free_path,
        expected_config_sha256=str(contract.config_file_sha256),
        verify_live_inputs=False,
    )
    if (
        not verified.dry_run
        or verified.formal
        or verified.image_count != 1
        or verified.phase_receipt_sha256 is not None
    ):
        raise D0V3FormalOuterRunnerError(
            "dry-run preflight accepts only a one-image non-formal candidate shard"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "cr_sitta_d0_v3_outer_dry_run_preflight",
        "dry_run": True,
        "formal": False,
        "label_free_candidate_shard_verified": True,
        "real_target_opened": False,
        "outer_target_loader_calls": 0,
        "outer_source_model_builds": 0,
        "paper_result": False,
        "stage2_authorized": False,
    }


def _load_label_free_arrays(label_free_path: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for filename in (
        SOURCE_LOGITS_FILENAME,
        POST_LOGITS_FILENAME,
        SOURCE_PARAMETERS_FILENAME,
        PARAMETERS_AFTER_FILENAME,
        ENTROPY_GRADIENTS_FILENAME,
    ):
        try:
            array = np.load(label_free_path / filename, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise D0V3FormalOuterRunnerError(
                f"cannot reopen verified label-free array {filename}"
            ) from exc
        if not isinstance(array, np.ndarray) or array.dtype.str != FLOAT_DTYPE:
            raise D0V3FormalOuterRunnerError(
                f"verified label-free array schema drifted: {filename}"
            )
        values[filename] = array
    return values


def _task_config(contract: D0V3FormalContract) -> D0V2TaskLossConfig:
    raw = contract.raw["formal_numeric_protocol"]["outer_oracle_task_loss"]
    return D0V2TaskLossConfig.from_mapping(
        {
            key: raw[key]
            for key in (
                "lambda_bce",
                "lambda_soft_iou",
                "eps",
                "bce_reduction",
                "soft_iou_reduction",
                "empty_target_convention",
            )
        }
    )


def _candidate_dict(index: int) -> dict[str, Any]:
    candidate = FROZEN_CANDIDATES[index]
    return {
        "candidate_id": candidate.candidate_id,
        "optimizer": candidate.optimizer,
        "learning_rate": candidate.learning_rate,
    }


def _source_parameter_match(
    worker_source: torch.Tensor, source_parameters: np.ndarray
) -> None:
    source = np.ascontiguousarray(worker_source.detach().cpu().numpy(), dtype="<f4")
    if source.shape != (source_parameters.shape[-1],):
        raise D0V3FormalOuterRunnerError("outer/label-free Source layout differs")
    if not np.array_equal(source_parameters, source[None, None, :]):
        # NumPy broadcasting is not applied by array_equal; check every value
        # without allocating a repeated 64 x 10 copy.
        if not bool(np.all(source_parameters == source[None, None, :])):
            raise D0V3FormalOuterRunnerError(
                "outer Source parameters are not bit-exact to label-free Source"
            )


def _output_manifest(
    *,
    staging: Path,
    contract: D0V3FormalContract,
    label_free_path: Path,
    label_verified: Any,
    label_manifest: Mapping[str, Any],
    layout: Any,
    dataset: str,
    condition: str,
    image_ids: Sequence[str],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    def file_record(filename: str, value: np.ndarray) -> dict[str, Any]:
        return {
            "path": filename,
            "sha256": _sha256_file(staging / filename),
            "shape": list(value.shape),
            "dtype": FLOAT_DTYPE,
            "c_order": True,
            "finite": True,
            "lossless": True,
        }

    label_binding = label_manifest["dataset_binding"]
    input_by_role = _input_seal_by_role(label_manifest)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "protocol_id": contract.protocol_id,
        "config_sha256": str(contract.config_file_sha256),
        "cell": {"dataset": dataset, "condition": condition, "replicate": "R0"},
        "mode": {
            "formal": True,
            "dry_run": False,
            "image_count": FORMAL_IMAGE_COUNT,
            "candidate_count": CANDIDATE_COUNT,
            "record_count": FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
            "record_order": EPISODE_ORDER,
        },
        "label_free_shard": {
            "path": _repository_relative(label_free_path),
            "manifest_sha256": label_verified.manifest_sha256,
            "complete_sha256": label_verified.complete_sha256,
            "phase_receipt_sha256": label_verified.phase_receipt_sha256,
            "cpu_verified_before_target_load": True,
        },
        "dataset_binding": {
            "train_split_sha256": label_binding["train_split_sha256"],
            "checkpoint_path": label_binding["checkpoint_path"],
            "checkpoint_sha256": label_binding["checkpoint_sha256"],
            "cache_protocol_sha256": input_by_role["cache_protocol"]["sha256"],
            "cache_method_manifest_sha256": input_by_role[
                "cache_method_manifest"
            ]["sha256"],
        },
        "ordered_image_ids": list(image_ids),
        "ordered_image_ids_sha256": label_manifest["ordered_image_ids_sha256"],
        "parameter_layout": {
            "source_path": (
                _repository_relative(label_free_path) + f"/{LAYOUT_FILENAME}"
            ),
            "source_file_sha256": _sha256_file(label_free_path / LAYOUT_FILENAME),
            "layout_sha256": layout.layout_sha256,
            "parameter_tensor_count": len(layout.names),
            "scalar_parameter_count": layout.scalar_count,
        },
        "outer_code_seal": build_outer_code_seal(PROJECT_ROOT),
        "arrays": {
            filename: file_record(filename, value)
            for filename, value in arrays.items()
        },
        "outer_records": {
            "path": OUTER_RECORDS_FILENAME,
            "sha256": _sha256_file(staging / OUTER_RECORDS_FILENAME),
            "count": FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
            "order": EPISODE_ORDER,
        },
        "task_loss_audits": {
            "path": TASK_LOSS_AUDITS_FILENAME,
            "sha256": _sha256_file(staging / TASK_LOSS_AUDITS_FILENAME),
            "count": FORMAL_IMAGE_COUNT,
            "one_supervised_gradient_per_image": True,
            "used_by_adaptation": False,
        },
        "outer_access_receipt": {
            "path": OUTER_ACCESS_RECEIPT_FILENAME,
            "sha256": _sha256_file(staging / OUTER_ACCESS_RECEIPT_FILENAME),
            "used_by_adaptation": False,
            "outer_target_loader_call_count": 1,
        },
        "execution": {
            "outer_source_model_build_count": 1,
            "outer_optimizer_build_count": 0,
            "source_forward_count": FORMAL_IMAGE_COUNT,
            "supervised_backward_count": FORMAL_IMAGE_COUNT,
            "supervised_gradient_count": FORMAL_IMAGE_COUNT,
            "source_logits_bit_exact_count": FORMAL_IMAGE_COUNT,
            "outer_episode_analysis_count": FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
            "candidate_optimizer_step_count_in_outer": 0,
        },
        "data_boundary": {
            "candidate_phase_cpu_verified_before_outer_target_load": True,
            "canonical_phase_receipt_verified_before_outer_target_load": True,
            "candidate_phase_target_access_count": 0,
            "outer_target_loader_call_count": 1,
            "outer_target_indexing_count": FORMAL_IMAGE_COUNT,
            "method_label_accesses": 0,
            "supervised_gradient_used_by_adaptation": False,
            "adaptation_optimizer_constructed_by_outer": False,
            "validation_payload_opens": 0,
            "test_split_files_opened": 0,
            "test_images_opened": 0,
            "test_masks_opened": 0,
            "test_labels_opened": 0,
        },
        "authorization": {
            "source_train_derived": True,
            "paper_result": False,
            "paper_test_result": False,
            "development_test_selected_result": False,
            "scientific_selection_performed": False,
            "formal_protocol_complete": False,
            "stage2_authorized": False,
        },
    }


def run_formal_outer_cell(
    *,
    dataset: str,
    condition: str,
    config_path: Path,
    device: torch.device,
    label_free_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Run one formal R0/64 outer cell and publish it atomically."""

    contract = _load_contract(config_path)
    if dataset not in contract.datasets or condition not in contract.conditions:
        raise D0V3FormalOuterRunnerError("dataset/condition is outside frozen grid")
    label_path = label_free_path or _fixed_label_free_path(
        contract, dataset, condition
    )
    destination = output_path or _fixed_outer_path(contract, dataset, condition)
    _repository_relative(label_path)
    _repository_relative(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"outer shard destination exists: {destination}")

    # Mandatory order: complete CPU label-free verification and canonical
    # phase receipt validation occur before this function can reach the guard.
    label_verified, label_manifest, phase, layout = _formal_label_free_preflight(
        label_free_path=label_path,
        contract=contract,
        dataset=dataset,
        condition=condition,
    )
    label_arrays = _load_label_free_arrays(label_path)
    episodes = _jsonl(
        read_stable_regular_file(label_path / EPISODES_FILENAME).data,
        expected_count=FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
        label="label-free episodes",
    )
    forward_runtime = _configure_formal_forward_runtime(
        device=device,
        seed=int(contract.raw["replicate_execution"]["seed"]),
    )
    worker = build_d0_v3_outer_source_model(
        project_root=PROJECT_ROOT,
        checkpoint_path=contract.raw["datasets"][dataset]["checkpoint_path"],
        checkpoint_sha256=contract.raw["datasets"][dataset]["checkpoint_sha256"],
        device=device,
    )
    if worker.layout.to_dict() != layout.to_dict():
        raise D0V3FormalOuterRunnerError(
            "outer Source parameter layout differs from label-free layout"
        )
    _source_parameter_match(
        worker.source_parameters_flat,
        label_arrays[SOURCE_PARAMETERS_FILENAME],
    )

    cache_root = PROJECT_ROOT / contract.raw["cache"]["root"] / dataset
    guarded = guarded_load_outer_targets(
        cache_root,
        contract.raw["cache"]["protocol_sha256"],
        read_stable_regular_file(label_path / PHASE_RECEIPT_FILENAME).data,
        dataset=dataset,
        condition=condition,
        replicate=0,
        expected_checkpoint_sha256=worker.checkpoint_sha256,
        expected_config_sha256=str(contract.config_file_sha256),
        expected_code_seals=dict(phase.code_files),
        expected_receipt_sha256=label_verified.phase_receipt_sha256,
    )
    method_inputs = SourceCalibrationMethodInputDatasetV2(
        cache_root,
        condition_key=condition,
        expected_protocol_sha256=contract.raw["cache"]["protocol_sha256"],
    )
    image_ids = tuple(label_manifest["ordered_image_ids"])
    if tuple(method_inputs.image_ids) != image_ids:
        raise D0V3FormalOuterRunnerError("outer method image order differs")

    task_config = _task_config(contract)
    thresholds = NoOpThresholds.from_mapping(
        contract.raw["formal_numeric_protocol"]["no_op_thresholds"]
    )
    first_order_tolerance = float(
        contract.raw["formal_numeric_protocol"]["alignment"][
            "first_order_zero_tolerance"
        ]
    )
    provenance = SourceTrainAnalysisProvenance(
        dataset=dataset,
        split_name="train",
        split_sha256=contract.raw["datasets"][dataset]["train_split_sha256"],
        checkpoint_sha256=worker.checkpoint_sha256,
        seed=int(contract.raw["replicate_execution"]["seed"]),
        oracle_analysis=True,
        outer_evaluator_label_accesses=1,
        supervised_gradient_role=OUTER_ORACLE_ROLE,
    )
    outer_logits = np.empty((FORMAL_IMAGE_COUNT, 1, 256, 256), dtype="<f4")
    supervised_gradients = np.empty(
        (FORMAL_IMAGE_COUNT, layout.scalar_count), dtype="<f4"
    )
    records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for image_index in range(FORMAL_IMAGE_COUNT):
        sample = method_inputs[image_index]
        image = sample["image"].unsqueeze(0).to(device)
        target_np = np.array(guarded.targets[image_index], dtype="<f4", copy=True)
        target = torch.from_numpy(target_np).unsqueeze(0).to(device)
        gradient = compute_d0_v3_outer_source_gradient(
            worker,
            image=image,
            target=target,
            task_loss_config=task_config,
            provenance=provenance,
        )
        logits_np = np.ascontiguousarray(
            gradient.source_logits.numpy()[0], dtype="<f4"
        )
        expected_logits = label_arrays[SOURCE_LOGITS_FILENAME][image_index]
        if not np.array_equal(logits_np, expected_logits):
            raise D0V3FormalOuterRunnerError(
                f"outer Source logits are not bit-exact at image {image_index}"
            )
        gradient_np = np.ascontiguousarray(
            gradient.supervised_gradient_flat.numpy(), dtype="<f4"
        )
        outer_logits[image_index] = logits_np
        supervised_gradients[image_index] = gradient_np
        audits.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": TASK_AUDIT_ARTIFACT_TYPE,
                "dataset": dataset,
                "condition": condition,
                "replicate_id": "R0",
                "image_index": image_index,
                "image_id": image_ids[image_index],
                "source_logits_bit_exact": True,
                "label_free_source_logits_slice_sha256": array_slice_sha256(
                    np.ascontiguousarray(expected_logits, dtype="<f4")
                ),
                "outer_source_logits_slice_sha256": array_slice_sha256(logits_np),
                "supervised_gradient_slice_sha256": array_slice_sha256(gradient_np),
                "supervised_gradient_finite": bool(np.isfinite(gradient_np).all()),
                "source_state_sha256": gradient.source_state_sha256,
                "reset_source_state_sha256": gradient.reset_source_state_sha256,
                "task_loss": dict(gradient.task_loss_audit),
                "forward_runtime": dict(forward_runtime),
                "used_by_adaptation": False,
                "stage2_authorized": False,
            }
        )
        for candidate_index in range(CANDIDATE_COUNT):
            position = image_index * CANDIDATE_COUNT + candidate_index
            nested = episodes[position]["independent_candidate_receipt"]
            changed = nested["numeric_evidence"]["changed_parameter_tensor_count"]
            analysis = analyze_formal_stage_a_episode(
                layout=layout,
                source_parameters_flat=label_arrays[SOURCE_PARAMETERS_FILENAME][
                    image_index, candidate_index
                ],
                parameters_after_flat=label_arrays[PARAMETERS_AFTER_FILENAME][
                    image_index, candidate_index
                ],
                entropy_gradient_flat=label_arrays[ENTROPY_GRADIENTS_FILENAME][
                    image_index, candidate_index
                ],
                supervised_gradient_flat=gradient_np,
                logits_pre=expected_logits,
                logits_post=label_arrays[POST_LOGITS_FILENAME][
                    image_index, candidate_index
                ],
                target=target_np,
                thresholds=thresholds,
                provenance=provenance,
                fine_group_assignment=worker.fine_group_assignment,
                first_order_zero_tolerance=first_order_tolerance,
            )
            analysis = _freeze_fine_group_order(analysis)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": RECORD_ARTIFACT_TYPE,
                    "dataset": dataset,
                    "condition": condition,
                    "replicate_id": "R0",
                    "image_index": image_index,
                    "image_id": image_ids[image_index],
                    "candidate": _candidate_dict(candidate_index),
                    "finite_gradient": bool(
                        np.isfinite(gradient_np).all()
                        and np.isfinite(
                            label_arrays[ENTROPY_GRADIENTS_FILENAME][
                                image_index, candidate_index
                            ]
                        ).all()
                    ),
                    "changed_parameter_tensor_count": int(changed),
                    "analysis": analysis,
                }
            )
    if len(records) != FORMAL_IMAGE_COUNT * CANDIDATE_COUNT or len(audits) != 64:
        raise D0V3FormalOuterRunnerError("outer output grid is incomplete")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{condition}.outer-build-", dir=destination.parent
        )
    )
    output_arrays = {
        OUTER_SOURCE_LOGITS_FILENAME: outer_logits,
        SUPERVISED_GRADIENTS_FILENAME: supervised_gradients,
    }
    for filename, value in output_arrays.items():
        _write_npy(staging / filename, value)
    _write_bytes(
        staging / OUTER_RECORDS_FILENAME,
        canonical_jsonl_bytes(records, ordered=True),
    )
    _write_bytes(
        staging / TASK_LOSS_AUDITS_FILENAME,
        canonical_jsonl_bytes(audits, ordered=False),
    )
    _write_bytes(staging / OUTER_ACCESS_RECEIPT_FILENAME, guarded.access_receipt_bytes)
    manifest = _output_manifest(
        staging=staging,
        contract=contract,
        label_free_path=label_path,
        label_verified=label_verified,
        label_manifest=label_manifest,
        layout=layout,
        dataset=dataset,
        condition=condition,
        image_ids=image_ids,
        arrays=output_arrays,
    )
    _write_bytes(staging / MANIFEST_FILENAME, canonical_json_bytes(manifest, newline=True))
    payload_files = [
        {"path": filename, "sha256": _sha256_file(staging / filename)}
        for filename in sorted(MEMBERS - {COMPLETE_FILENAME})
    ]
    complete = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": COMPLETE_ARTIFACT_TYPE,
        "complete": True,
        "candidate_phase_complete": True,
        "outer_phase_complete": True,
        "formal": True,
        "dry_run": False,
        "config_sha256": str(contract.config_file_sha256),
        "dataset": dataset,
        "condition": condition,
        "replicate": "R0",
        "image_count": FORMAL_IMAGE_COUNT,
        "candidate_count": CANDIDATE_COUNT,
        "record_count": FORMAL_IMAGE_COUNT * CANDIDATE_COUNT,
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": _sha256_file(staging / MANIFEST_FILENAME),
        },
        "payload_files": payload_files,
        "atomic_no_replace": True,
        "paper_result": False,
        "paper_test_result": False,
        "formal_protocol_complete": False,
        "scientific_gate_status": "not_evaluated",
        "stage2_authorized": False,
    }
    _write_bytes(staging / COMPLETE_FILENAME, canonical_json_bytes(complete, newline=True))

    def semantic_verifier(candidate_path: Path) -> Any:
        return verify_outer_cell_shard(
            candidate_path,
            repository_root=PROJECT_ROOT,
            label_free_shard_path=label_path,
            expected_config_sha256=str(contract.config_file_sha256),
        )

    published = publish_flat_directory_noreplace(
        staging,
        destination,
        expected_members=sorted(MEMBERS),
        semantic_verifier=semantic_verifier,
    )
    semantic_verifier(published)
    return published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / CONFIG_RELATIVE_PATH,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--dataset", required=True)
        child.add_argument("--condition", required=True)
        child.add_argument("--label-free", type=Path)
        child.add_argument("--output", type=Path)
        if command == "run":
            child.add_argument("--device", default="cuda:0")
    dry = subparsers.add_parser("dry-run")
    dry.add_argument("--label-free", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(os.path.abspath(os.fspath(args.config)))
    if args.command == "dry-run":
        value = dry_run_preflight(
            label_free_path=Path(os.path.abspath(os.fspath(args.label_free))),
            config_path=config_path,
        )
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    contract = _load_contract(config_path)
    label_path = (
        Path(os.path.abspath(os.fspath(args.label_free)))
        if args.label_free
        else _fixed_label_free_path(contract, args.dataset, args.condition)
    )
    output_path = (
        Path(os.path.abspath(os.fspath(args.output)))
        if args.output
        else _fixed_outer_path(contract, args.dataset, args.condition)
    )
    if args.command == "verify":
        verified = verify_outer_cell_shard(
            output_path,
            repository_root=PROJECT_ROOT,
            label_free_shard_path=label_path,
            expected_config_sha256=str(contract.config_file_sha256),
        )
        print(json.dumps(asdict(verified), default=str, sort_keys=True))
        return 0
    published = run_formal_outer_cell(
        dataset=args.dataset,
        condition=args.condition,
        config_path=config_path,
        device=torch.device(args.device),
        label_free_path=label_path,
        output_path=output_path,
    )
    print(str(published))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
