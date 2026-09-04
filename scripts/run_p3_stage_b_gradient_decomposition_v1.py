#!/usr/bin/env python3
"""Run the train-only P3 Stage-B1 region-gradient decomposition.

Stage-B1 is an outer-oracle mechanism diagnostic.  It never adapts a model,
constructs an optimizer, writes a checkpoint, reads validation/test payloads,
or authorizes a downstream stage.  For each frozen Source-train Pilot64 cell
it first verifies the completed D0-v3 label-free and outer shards, then opens
the matching train targets through the existing receipt-gated loader.
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
import sys
import tempfile
from typing import Any, Final

import numpy as np
import torch


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.d0_v3_formal_contract import (  # noqa: E402
    CONFIG_FILE_SHA256 as PARENT_CONFIG_SHA256,
)
from analysis.d0_v3_label_free_shard import (  # noqa: E402
    COMPLETE_FILENAME as PARENT_CANDIDATE_COMPLETE_FILENAME,
    ENTROPY_GRADIENTS_FILENAME,
    LAYOUT_FILENAME as PARENT_LAYOUT_FILENAME,
    MANIFEST_FILENAME as PARENT_CANDIDATE_MANIFEST_FILENAME,
    PHASE_RECEIPT_FILENAME,
    SOURCE_LOGITS_FILENAME,
    array_slice_sha256,
    canonical_json_bytes,
    parse_canonical_json,
    validate_parameter_layout,
    verify_label_free_shard,
)
from analysis.d0_v3_outer_cell_shard import (  # noqa: E402
    COMPLETE_FILENAME as PARENT_OUTER_COMPLETE_FILENAME,
    MANIFEST_FILENAME as PARENT_OUTER_MANIFEST_FILENAME,
    OUTER_ACCESS_RECEIPT_FILENAME as PARENT_OUTER_ACCESS_FILENAME,
    OUTER_SOURCE_LOGITS_FILENAME,
    SUPERVISED_GRADIENTS_FILENAME,
    verify_outer_cell_shard,
)
from analysis.d0_v3_phase_receipt import (  # noqa: E402
    guarded_load_outer_targets,
    validate_label_free_cell_receipt,
)
from analysis.foreground_background_gradient_decomposition_v1 import (  # noqa: E402
    BACKWARD_BASIS,
    ForegroundBackgroundGradientError,
    GROUP_IDS,
    GradientDecompositionConfig,
    analyze_foreground_background_gradient_decomposition,
)
from analysis.p3_stage_b1_contract import (  # noqa: E402
    CONFIG_FILE_SHA256,
    CONFIG_RELATIVE_PATH,
    P3StageB1Contract,
    load_p3_stage_b1_contract,
    verify_all_frozen_file_bindings,
)
from analysis.p3_stage_b1_aggregate import (  # noqa: E402
    MEMBERS as AGGREGATE_MEMBERS,
    StageB1AggregatePreflight,
    VerifiedStageB1Aggregate,
    build_stage_b1_aggregate_payloads,
    collect_stage_b1_preflight,
    verify_stage_b1_aggregate_shard,
)
from analysis.p3_stage_b1_outer_cell_shard import (  # noqa: E402
    ARTIFACT_TYPE,
    CELL_AUTHORIZATION,
    CELL_DATA_BOUNDARY,
    COARSE_GROUP_LAYOUT_FILENAME,
    COMPLETE_ARTIFACT_TYPE,
    COMPLETE_FILENAME,
    EPISODE_RECORDS_FILENAME,
    GROUP_SCALAR_COUNTS,
    MANIFEST_DATA_BOUNDARY,
    MANIFEST_FILENAME,
    MEMBERS as CELL_MEMBERS,
    OUTER_ACCESS_RECEIPT_FILENAME,
    RECORD_ARTIFACT_TYPE,
    REGION_GRADIENT_BASIS_FILENAME,
    build_coarse_group_layout,
    build_stage_b1_cell_payloads,
    verify_stage_b1_cell_shard,
)
from materialize_binary_tent_ss_calibration_cache_v2 import (  # noqa: E402
    SourceCalibrationMethodInputDatasetV2,
)
from tta.d0_secure_io import read_stable_regular_file  # noqa: E402
from tta.d0_v3_atomic_shard import publish_flat_directory_noreplace  # noqa: E402
from tta.d0_v3_outer_source_gradient import (  # noqa: E402
    D0V3OuterSourceModel,
    build_d0_v3_outer_source_model,
)
from tta.parameter_groups import (  # noqa: E402
    PILOT_GROUP_SPECS,
    collect_adaptable_params,
)


SCHEMA_VERSION: Final = 1
FORMAL_IMAGE_COUNT: Final = 64
PARENT_CANDIDATE_COUNT: Final = 10
SCALAR_COUNT: Final = 8736
STORAGE_BASIS: Final = (
    "foreground_subthreshold",
    "foreground_suprathreshold",
    "background",
)


class P3StageB1RunnerError(RuntimeError):
    """The B1 outer-oracle execution violated its frozen contract."""


@dataclass(frozen=True)
class _RNGSnapshot:
    python_state: object
    numpy_state: tuple[Any, ...]
    torch_cpu: torch.Tensor
    torch_cuda: torch.Tensor | None
    digest: str


@dataclass(frozen=True)
class _ParentCell:
    candidate_path: Path
    outer_path: Path
    candidate_verified: Any
    outer_verified: Any
    candidate_manifest: Mapping[str, Any]
    outer_manifest: Mapping[str, Any]
    phase_receipt: Any
    ordered_image_ids: tuple[str, ...]
    layout: Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return read_stable_regular_file(path).sha256


def _repository_relative(path: Path) -> str:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = absolute.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise P3StageB1RunnerError(f"path escapes project root: {absolute}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise P3StageB1RunnerError(f"path is not canonical: {absolute}")
    return relative.as_posix()


def _load_contract(path: Path) -> P3StageB1Contract:
    absolute = Path(os.path.abspath(os.fspath(path)))
    _repository_relative(absolute)
    contract = load_p3_stage_b1_contract(absolute)
    if contract.config_file_sha256 != CONFIG_FILE_SHA256:
        raise P3StageB1RunnerError("B1 config SHA-256 differs")
    verify_all_frozen_file_bindings(contract, repository_root=PROJECT_ROOT)
    if contract.paper_result or contract.stage_b3_authorized:
        raise P3StageB1RunnerError("B1 diagnostic cannot authorize paper/B3 use")
    required = contract.raw["required_parent_decision"]
    if (
        required["protocol_status"] != "passed"
        or required["formal_stage_a_protocol_complete"] is not True
        or required["scientific_status"] != "scientific_no_eligible"
        or tuple(required["eligible_candidates"]) != ()
        or tuple(required["required_followup_replicates"]) != ()
        or required["stage2_allowed"] is not False
    ):
        raise P3StageB1RunnerError("B1 parent negative-decision binding differs")
    return contract


def _fixed_parent_paths(
    contract: P3StageB1Contract, dataset: str, condition: str
) -> tuple[Path, Path]:
    raw = contract.raw["parent_cell_artifacts"]
    candidate = str(raw["candidate_shard_template"]).format(
        dataset=dataset, condition=condition
    )
    outer = str(raw["outer_shard_template"]).format(
        dataset=dataset, condition=condition
    )
    return PROJECT_ROOT / candidate, PROJECT_ROOT / outer


def _fixed_output_path(
    contract: P3StageB1Contract, dataset: str, condition: str
) -> Path:
    return (
        PROJECT_ROOT
        / str(contract.output_root)
        / "outer_phase"
        / "shards"
        / "R0"
        / dataset
        / condition
    )


def _fixed_aggregate_path(contract: P3StageB1Contract) -> Path:
    return PROJECT_ROOT / str(contract.output_root) / "aggregate_phase" / "R0"


def _read_canonical(path: Path, *, label: str) -> Mapping[str, Any]:
    return parse_canonical_json(
        read_stable_regular_file(path).data,
        label=label,
        newline=True,
    )


def _preflight_parent_cell(
    *, contract: P3StageB1Contract, dataset: str, condition: str
) -> _ParentCell:
    if dataset not in contract.datasets or condition not in contract.conditions:
        raise P3StageB1RunnerError("dataset/condition is outside frozen B1 grid")
    candidate_path, outer_path = _fixed_parent_paths(contract, dataset, condition)
    candidate = verify_label_free_shard(
        candidate_path,
        expected_config_sha256=PARENT_CONFIG_SHA256,
        verify_live_inputs=True,
        repository_root=PROJECT_ROOT,
    )
    outer = verify_outer_cell_shard(
        outer_path,
        repository_root=PROJECT_ROOT,
        label_free_shard_path=candidate_path,
        expected_config_sha256=PARENT_CONFIG_SHA256,
    )
    if (
        candidate.dataset != dataset
        or candidate.condition != condition
        or candidate.replicate != "R0"
        or not candidate.formal
        or candidate.dry_run
        or candidate.image_count != FORMAL_IMAGE_COUNT
        or candidate.candidate_count != PARENT_CANDIDATE_COUNT
        or outer.dataset != dataset
        or outer.condition != condition
        or outer.replicate != "R0"
        or outer.image_count != FORMAL_IMAGE_COUNT
    ):
        raise P3StageB1RunnerError("parent cell topology differs from B1 contract")
    candidate_manifest = _read_canonical(
        candidate_path / PARENT_CANDIDATE_MANIFEST_FILENAME,
        label="parent candidate manifest",
    )
    outer_manifest = _read_canonical(
        outer_path / PARENT_OUTER_MANIFEST_FILENAME,
        label="parent outer manifest",
    )
    ordered = tuple(candidate_manifest["ordered_image_ids"])
    if tuple(outer_manifest["ordered_image_ids"]) != ordered or len(ordered) != 64:
        raise P3StageB1RunnerError("parent candidate/outer Pilot order differs")
    expected_ids_sha = dict(contract.pilot64_ordered_id_sha256)[dataset]
    if (
        candidate_manifest["ordered_image_ids_sha256"] != expected_ids_sha
        or outer_manifest["ordered_image_ids_sha256"] != expected_ids_sha
    ):
        raise P3StageB1RunnerError("parent Pilot64 ordered-ID SHA differs")
    layout = validate_parameter_layout(
        _read_canonical(
            candidate_path / PARENT_LAYOUT_FILENAME,
            label="parent parameter layout",
        )
    )
    expected_layout = contract.raw["parent_cell_artifacts"]["parameter_layout"]
    if (
        layout.layout_sha256 != expected_layout["layout_sha256"]
        or len(layout.names) != expected_layout["parameter_tensor_count"]
        or layout.scalar_count != expected_layout["scalar_parameter_count"]
    ):
        raise P3StageB1RunnerError("parent parameter layout differs")
    phase_bytes = read_stable_regular_file(candidate_path / PHASE_RECEIPT_FILENAME).data
    phase = validate_label_free_cell_receipt(
        phase_bytes,
        expected_dataset=dataset,
        expected_condition=condition,
        expected_replicate=0,
        expected_cache_protocol_sha256=contract.raw["frozen_parent_bindings"][
            "cache_protocol"
        ]["sha256"],
        expected_checkpoint_sha256=contract.raw["datasets"][dataset][
            "checkpoint_sha256"
        ],
        expected_config_sha256=PARENT_CONFIG_SHA256,
        expected_ordered_image_ids=ordered,
        expected_receipt_sha256=candidate.phase_receipt_sha256,
    )
    return _ParentCell(
        candidate_path=candidate_path,
        outer_path=outer_path,
        candidate_verified=candidate,
        outer_verified=outer,
        candidate_manifest=candidate_manifest,
        outer_manifest=outer_manifest,
        phase_receipt=phase,
        ordered_image_ids=ordered,
        layout=layout,
    )


def _configure_runtime(*, device: torch.device, seed: int) -> Mapping[str, Any]:
    if device.type == "cuda":
        required = {
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        mismatch = {
            key: {"expected": expected, "observed": os.environ.get(key)}
            for key, expected in required.items()
            if os.environ.get(key) != expected
        }
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not isinstance(visible, str) or not visible or "," in visible:
            mismatch["CUDA_VISIBLE_DEVICES"] = {
                "expected": "exactly one physical GPU",
                "observed": visible,
            }
        if mismatch:
            raise P3StageB1RunnerError(f"CUDA environment differs: {mismatch}")
        if device != torch.device("cuda:0"):
            raise P3StageB1RunnerError("sole visible CUDA device must be addressed as cuda:0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise P3StageB1RunnerError("B1 CUDA worker must see exactly one GPU")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise P3StageB1RunnerError("B1 device must be CPU or CUDA")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "device": str(device),
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "visible_cuda_device_count": (
            int(torch.cuda.device_count()) if device.type == "cuda" else 0
        ),
        "gradient_backward_policy": "temporarily_disable_strict_determinism_then_restore",
    }


def _capture_rng(device: torch.device) -> _RNGSnapshot:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state().clone()
    cuda_state = (
        torch.cuda.get_rng_state(device=device).clone()
        if device.type == "cuda"
        else None
    )
    digest = hashlib.sha256()
    digest.update(repr(python_state).encode("utf-8"))
    digest.update(str(numpy_state[0]).encode("ascii"))
    digest.update(np.asarray(numpy_state[1], dtype="<u4").tobytes())
    digest.update(repr(tuple(numpy_state[2:])).encode("ascii"))
    digest.update(cpu_state.cpu().numpy().tobytes())
    if cuda_state is not None:
        digest.update(cuda_state.cpu().numpy().tobytes())
    return _RNGSnapshot(
        python_state=python_state,
        numpy_state=numpy_state,
        torch_cpu=cpu_state,
        torch_cuda=cuda_state,
        digest=digest.hexdigest(),
    )


def _rng_equal(left: _RNGSnapshot, right: _RNGSnapshot) -> bool:
    numpy_equal = (
        left.numpy_state[0] == right.numpy_state[0]
        and np.array_equal(left.numpy_state[1], right.numpy_state[1])
        and left.numpy_state[2:] == right.numpy_state[2:]
    )
    cuda_equal = (
        left.torch_cuda is None
        and right.torch_cuda is None
        or left.torch_cuda is not None
        and right.torch_cuda is not None
        and torch.equal(left.torch_cuda, right.torch_cuda)
    )
    return (
        left.python_state == right.python_state
        and numpy_equal
        and torch.equal(left.torch_cpu, right.torch_cpu)
        and cuda_equal
        and left.digest == right.digest
    )


def _restore_rng(snapshot: _RNGSnapshot, device: torch.device) -> None:
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_cpu)
    if snapshot.torch_cuda is not None:
        torch.cuda.set_rng_state(snapshot.torch_cuda, device=device)


def _load_array(path: Path, *, shape: tuple[int, ...]) -> np.ndarray:
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise P3StageB1RunnerError(f"cannot load parent array: {path}") from exc
    if (
        not isinstance(value, np.ndarray)
        or value.dtype.str != "<f4"
        or tuple(value.shape) != shape
    ):
        raise P3StageB1RunnerError(f"parent array schema differs: {path.name}")
    return value


def _group_parameter_names(worker: D0V3OuterSourceModel) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {"P0": tuple(worker.parameter_names)}
    for group_id in GROUP_IDS[1:]:
        _parameters, names = collect_adaptable_params(
            worker.model, group_spec=PILOT_GROUP_SPECS[group_id]
        )
        result[group_id] = tuple(names)
    expected = {"P0": 8736, "P1": 96, "P2": 416, "P3": 2080, "P4": 2592}
    offsets = worker.layout.offsets
    ends = (*offsets[1:], worker.layout.scalar_count)
    sizes = {
        name: end - start
        for name, start, end in zip(worker.layout.names, offsets, ends, strict=True)
    }
    observed = {
        group_id: sum(sizes[name] for name in names)
        for group_id, names in result.items()
    }
    if observed != expected or tuple(result) != GROUP_IDS:
        raise P3StageB1RunnerError(f"P0-P4 scalar topology differs: {observed}")
    return result


def _coarse_group_layout(
    *, contract: P3StageB1Contract, worker: D0V3OuterSourceModel,
    group_names: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    offsets = worker.layout.offsets
    ends = (*offsets[1:], worker.layout.scalar_count)
    span_by_name = dict(
        zip(worker.layout.names, zip(offsets, ends, strict=True), strict=True)
    )
    groups: dict[str, Any] = {}
    for group_id in GROUP_IDS:
        names = tuple(group_names[group_id])
        spans = [
            {"parameter_name": name, "start": span_by_name[name][0], "end": span_by_name[name][1]}
            for name in names
        ]
        groups[group_id] = {
            "parameter_tensor_count": len(names),
            "parameter_scalar_count": sum(item["end"] - item["start"] for item in spans),
            "spans": spans,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "cr_sitta_p3_stage_b1_coarse_group_layout",
        "protocol_id": contract.protocol_id,
        "source_layout_sha256": worker.layout.layout_sha256,
        "source_parameter_tensor_count": len(worker.layout.names),
        "source_scalar_parameter_count": worker.layout.scalar_count,
        "group_order": list(GROUP_IDS),
        "groups": groups,
    }


def _target_sha256(value: np.ndarray) -> str:
    return array_slice_sha256(np.ascontiguousarray(value, dtype="<f4"))


def _condition_parts(condition: str) -> tuple[str, str]:
    if condition == "clean_S0":
        return "clean", "S0"
    family, severity = condition.rsplit("_", 1)
    if family not in {"gaussian_noise", "gaussian_blur", "low_contrast", "stripe_noise"}:
        raise P3StageB1RunnerError(f"unknown frozen condition: {condition}")
    if severity not in {"S1", "S3", "S5"}:
        raise P3StageB1RunnerError(f"unknown frozen severity: {condition}")
    return family, severity


def _max_parent_errors(full: torch.Tensor, parent: np.ndarray) -> tuple[float, float]:
    reference = np.asarray(parent, dtype=np.float64)
    candidate = full.detach().cpu().numpy().astype(np.float64, copy=False)
    residual = reference - candidate[None, :]
    max_abs = float(np.max(np.abs(residual)))
    residual_l2 = np.linalg.norm(residual, axis=1)
    reference_l2 = np.linalg.norm(reference, axis=1)
    relative = residual_l2 / np.maximum(reference_l2, 1.0e-12)
    max_relative = float(np.max(relative))
    if not math.isfinite(max_abs) or not math.isfinite(max_relative):
        raise P3StageB1RunnerError("parent entropy comparison is non-finite")
    return max_abs, max_relative


def _code_seal(contract: P3StageB1Contract) -> dict[str, Any]:
    paths = tuple(contract.raw["implementation"]["critical_code_paths"])
    records = [
        {"path": path, "sha256": _sha256_file(PROJECT_ROOT / path)}
        for path in paths
    ]
    return {
        "files": records,
        "bundle_sha256": _sha256_bytes(canonical_json_bytes(records)),
    }


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_npy(path: Path, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype="<f4")
    if not np.isfinite(array).all():
        raise P3StageB1RunnerError("refusing non-finite B1 gradient array")
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _compute_cell(
    *, contract: P3StageB1Contract, dataset: str, condition: str,
    device: torch.device, image_count: int,
) -> tuple[
    np.ndarray,
    list[dict[str, Any]],
    dict[str, Any],
    bytes,
    _ParentCell,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    # Fail on a missing or drifting implementation before any parent payload,
    # CUDA context, model, or outer target can be reached.
    code_seal = _code_seal(contract)
    parent = _preflight_parent_cell(
        contract=contract, dataset=dataset, condition=condition
    )
    if tuple(contract.raw["gradient_basis"]["basis_order"]) != STORAGE_BASIS:
        raise P3StageB1RunnerError("B1 storage basis order differs from config")
    if BACKWARD_BASIS != tuple(f"{name}_add" for name in STORAGE_BASIS):
        raise P3StageB1RunnerError("analyzer VJP basis order differs from storage")
    runtime = _configure_runtime(device=device, seed=42)
    dataset_config = contract.raw["datasets"][dataset]
    worker = build_d0_v3_outer_source_model(
        project_root=PROJECT_ROOT,
        checkpoint_path=str(dataset_config["checkpoint_path"]),
        checkpoint_sha256=str(dataset_config["checkpoint_sha256"]),
        device=device,
    )
    if worker.layout.to_dict() != parent.layout.to_dict():
        raise P3StageB1RunnerError("live Source parameter layout differs from parent")
    group_names = _group_parameter_names(worker)
    group_layout = build_coarse_group_layout(
        protocol_id=contract.protocol_id,
        source_parameter_layout=worker.layout.to_dict(),
        group_parameter_names=group_names,
    )
    candidate_logits = _load_array(
        parent.candidate_path / SOURCE_LOGITS_FILENAME,
        shape=(64, 1, 256, 256),
    )
    outer_logits = _load_array(
        parent.outer_path / OUTER_SOURCE_LOGITS_FILENAME,
        shape=(64, 1, 256, 256),
    )
    entropy_gradients = _load_array(
        parent.candidate_path / ENTROPY_GRADIENTS_FILENAME,
        shape=(64, 10, SCALAR_COUNT),
    )
    task_gradients = _load_array(
        parent.outer_path / SUPERVISED_GRADIENTS_FILENAME,
        shape=(64, SCALAR_COUNT),
    )
    if not np.array_equal(candidate_logits, outer_logits):
        raise P3StageB1RunnerError("parent candidate/outer Source logits differ")
    phase_path = parent.candidate_path / PHASE_RECEIPT_FILENAME
    cache_root = PROJECT_ROOT / "results/binary_tent/ss_calibration_cache_v2" / dataset
    guarded = guarded_load_outer_targets(
        cache_root,
        str(contract.raw["frozen_parent_bindings"]["cache_protocol"]["sha256"]),
        read_stable_regular_file(phase_path).data,
        dataset=dataset,
        condition=condition,
        replicate=0,
        expected_checkpoint_sha256=str(dataset_config["checkpoint_sha256"]),
        expected_config_sha256=PARENT_CONFIG_SHA256,
        expected_code_seals=dict(parent.phase_receipt.code_files),
        expected_receipt_sha256=parent.candidate_verified.phase_receipt_sha256,
    )
    parent_access = read_stable_regular_file(
        parent.outer_path / PARENT_OUTER_ACCESS_FILENAME
    )
    if guarded.access_receipt_bytes != parent_access.data:
        raise P3StageB1RunnerError("regenerated outer access receipt differs from parent")
    method_inputs = SourceCalibrationMethodInputDatasetV2(
        cache_root,
        condition_key=condition,
        expected_protocol_sha256=str(
            contract.raw["frozen_parent_bindings"]["cache_protocol"]["sha256"]
        ),
    )
    if tuple(method_inputs.image_ids) != parent.ordered_image_ids:
        raise P3StageB1RunnerError("method-input Pilot order differs from parent")
    if image_count < 1 or image_count > FORMAL_IMAGE_COUNT:
        raise P3StageB1RunnerError("image_count must lie in [1,64]")
    basis = np.empty((image_count, len(STORAGE_BASIS), SCALAR_COUNT), dtype="<f4")
    records: list[dict[str, Any]] = []
    family, severity = _condition_parts(condition)
    numeric = contract.raw["entropy_gradient_consistency"]
    analyzer_config = GradientDecompositionConfig(
        entropy_eps=float(contract.raw["gradient_basis"]["entropy_eps"]),
        parent_entropy_max_abs_tolerance=float(numeric["max_abs_tolerance"]),
        parent_entropy_relative_l2_tolerance=float(numeric["relative_l2_tolerance"]),
        cosine_zero_norm_tolerance=0.0,
    )
    named_parameters = dict(
        zip(worker.parameter_names, worker.parameters, strict=True)
    )
    for image_index in range(image_count):
        before_rng = _capture_rng(device)
        source_state = worker.state_manager.assert_source_state()
        report: Mapping[str, Any] | None = None
        logits_cpu: torch.Tensor | None = None
        target_np: np.ndarray | None = None
        vectors = None
        reset = None
        after_rng = None
        try:
            worker.adapter.set_tent_mode(use_batch_stats=False)
            worker.model.zero_grad(set_to_none=True)
            sample = method_inputs[image_index]
            image = sample["image"].unsqueeze(0).to(device)
            target_np = np.array(guarded.targets[image_index], dtype="<f4", copy=True)
            target = torch.from_numpy(target_np).unsqueeze(0).to(device)
            source_logits = worker.adapter.forward_logits(image)
            logits_cpu = source_logits.detach().cpu().contiguous()
            expected_logits = np.ascontiguousarray(
                candidate_logits[image_index], dtype="<f4"
            )
            if not np.array_equal(logits_cpu.numpy()[0], expected_logits):
                raise P3StageB1RunnerError(
                    f"Source logits differ from parent at image {image_index}"
                )
            deterministic_before = bool(torch.are_deterministic_algorithms_enabled())
            warn_before = bool(torch.is_deterministic_algorithms_warn_only_enabled())
            try:
                if device.type == "cuda":
                    torch.use_deterministic_algorithms(False)
                try:
                    result = analyze_foreground_background_gradient_decomposition(
                        source_logits=source_logits,
                        target=target,
                        named_parameters=named_parameters,
                        parameter_layout=worker.layout,
                        group_parameter_names=group_names,
                        parent_entropy_gradient_flat=torch.from_numpy(
                            np.array(entropy_gradients[image_index, 0], copy=True)
                        ),
                        task_gradient_flat=torch.from_numpy(
                            np.array(task_gradients[image_index], copy=True)
                        ),
                        config=analyzer_config,
                    )
                except ForegroundBackgroundGradientError as exc:
                    raise P3StageB1RunnerError(
                        "B1 gradient decomposition failed at "
                        f"{dataset}/{condition}/image_index={image_index}/"
                        f"image_id={parent.ordered_image_ids[image_index]}: {exc}"
                    ) from exc
            finally:
                torch.use_deterministic_algorithms(
                    deterministic_before, warn_only=warn_before
                )
            report = result.report
            vectors = result.vectors
            max_abs, max_relative = _max_parent_errors(
                vectors.full_add,
                np.asarray(entropy_gradients[image_index]),
            )
            if (
                max_abs > float(numeric["max_abs_tolerance"])
                or max_relative > float(numeric["relative_l2_tolerance"])
            ):
                raise P3StageB1RunnerError(
                    f"all-candidate parent entropy tolerance failed at {image_index}"
                )
            basis_slice = np.stack(
                [
                    vectors.foreground_subthreshold_add.numpy(),
                    vectors.foreground_suprathreshold_add.numpy(),
                    vectors.background_add.numpy(),
                ],
                axis=0,
            ).astype("<f4", copy=False)
            basis[image_index] = basis_slice
        finally:
            try:
                reset = worker.state_manager.reset_to_source()
            finally:
                after_rng = _capture_rng(device)
                if not _rng_equal(before_rng, after_rng):
                    _restore_rng(before_rng, device)
        assert after_rng is not None
        rng_restored = _rng_equal(before_rng, after_rng)
        if not rng_restored:
            _restore_rng(before_rng, device)
            raise P3StageB1RunnerError(
                f"RNG state changed during B1 image {image_index}"
            )
        if (
            report is None
            or logits_cpu is None
            or target_np is None
            or vectors is None
            or reset is None
        ):
            raise P3StageB1RunnerError("B1 analysis produced no result")
        if reset != source_state or worker.state_manager.assert_source_state() != source_state:
            raise P3StageB1RunnerError(
                f"Source state did not reset exactly at image {image_index}"
            )
        stats = report["target_statistics"]
        target_present = bool(stats["foreground_pixel_count"] > 0)
        parent_logit_slice = np.ascontiguousarray(
            candidate_logits[image_index], dtype="<f4"
        )
        recomputed_logit_slice = np.ascontiguousarray(
            logits_cpu.numpy()[0], dtype="<f4"
        )
        max_abs, max_relative = _max_parent_errors(
            vectors.full_add, np.asarray(entropy_gradients[image_index])
        )
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": RECORD_ARTIFACT_TYPE,
                "protocol_id": contract.protocol_id,
                "config_sha256": str(contract.config_file_sha256),
                "dataset": dataset,
                "condition": condition,
                "corruption_family": family,
                "severity": severity,
                "replicate": "R0",
                "image_index": image_index,
                "image_id": parent.ordered_image_ids[image_index],
                "target": {
                    "target_present": target_present,
                    "total_pixel_count": int(stats["total_pixel_count"]),
                    "foreground_pixel_count": int(stats["foreground_pixel_count"]),
                    "background_pixel_count": int(stats["background_pixel_count"]),
                    "foreground_subthreshold_pixel_count": int(
                        stats["foreground_subthreshold_pixel_count"]
                    ),
                    "foreground_suprathreshold_pixel_count": int(
                        stats["foreground_suprathreshold_pixel_count"]
                    ),
                    "target_value_sum": float(np.sum(target_np, dtype=np.float64)),
                    "target_slice_sha256": _target_sha256(target_np),
                    "partition_disjoint": True,
                    "partition_exhaustive": True,
                },
                "source_integrity": {
                    "source_logits_bit_exact": True,
                    "parent_source_logits_slice_sha256": array_slice_sha256(parent_logit_slice),
                    "recomputed_source_logits_slice_sha256": array_slice_sha256(recomputed_logit_slice),
                    "source_state_before_sha256": source_state.full_sha256,
                    "source_state_after_sha256": reset.full_sha256,
                    "state_restored": reset == source_state,
                    "rng_before_sha256": before_rng.digest,
                    "rng_after_sha256": after_rng.digest,
                    "rng_restored": rng_restored,
                },
                "gradient_integrity": {
                    "basis_order": list(STORAGE_BASIS),
                    "basis_slice_sha256": array_slice_sha256(
                        np.ascontiguousarray(basis[image_index], dtype="<f4")
                    ),
                    "finite": bool(np.isfinite(basis[image_index]).all()),
                    "parent_candidate_slice_count": PARENT_CANDIDATE_COUNT,
                    "parent_max_abs_error": max_abs,
                    "parent_max_relative_l2_error": max_relative,
                    "max_abs_tolerance": float(numeric["max_abs_tolerance"]),
                    "relative_l2_tolerance": float(numeric["relative_l2_tolerance"]),
                    "parent_consistency_passed": True,
                },
                "groups": report["per_group"],
                "data_boundary": dict(CELL_DATA_BOUNDARY),
                "authorization": dict(CELL_AUTHORIZATION),
            }
        )
    return (
        basis,
        records,
        group_layout,
        guarded.access_receipt_bytes,
        parent,
        runtime,
        code_seal,
    )


def _parent_lineage(parent: _ParentCell) -> dict[str, Any]:
    return {
        "candidate_shard_path": _repository_relative(parent.candidate_path),
        "outer_shard_path": _repository_relative(parent.outer_path),
        "candidate_manifest_sha256": parent.candidate_verified.manifest_sha256,
        "candidate_complete_sha256": parent.candidate_verified.complete_sha256,
        "candidate_phase_receipt_sha256": parent.candidate_verified.phase_receipt_sha256,
        "outer_manifest_sha256": parent.outer_verified.manifest_sha256,
        "outer_complete_sha256": parent.outer_verified.complete_sha256,
        "outer_access_receipt_sha256": parent.outer_verified.outer_access_receipt_sha256,
        "parent_entropy_gradients_sha256": parent.candidate_manifest["arrays"][ENTROPY_GRADIENTS_FILENAME]["sha256"],
        "parent_task_gradients_sha256": parent.outer_manifest["arrays"][SUPERVISED_GRADIENTS_FILENAME]["sha256"],
    }


def run_formal_cell(
    *, dataset: str, condition: str, config_path: Path, device: torch.device,
    output_path: Path | None = None,
) -> Path:
    if device != torch.device("cuda:0"):
        raise P3StageB1RunnerError(
            "formal Stage-B1 cells require the sole visible device cuda:0; "
            "CPU is reserved for smoke tests"
        )
    contract = _load_contract(config_path)
    destination = output_path or _fixed_output_path(contract, dataset, condition)
    _repository_relative(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"B1 destination exists: {destination}")
    (
        basis,
        records,
        group_layout,
        access_bytes,
        parent,
        runtime,
        code_seal,
    ) = _compute_cell(
        contract=contract,
        dataset=dataset,
        condition=condition,
        device=device,
        image_count=FORMAL_IMAGE_COUNT,
    )
    if _code_seal(contract) != code_seal:
        raise P3StageB1RunnerError("B1 implementation changed during cell execution")
    dataset_binding = contract.raw["datasets"][dataset]
    group_layout_bytes = canonical_json_bytes(group_layout, newline=True)
    parent_layout_path = parent.candidate_path / PARENT_LAYOUT_FILENAME
    parameter_layout = {
        "parent_source_path": _repository_relative(parent_layout_path),
        "parent_source_file_sha256": _sha256_file(parent_layout_path),
        "source_layout_sha256": parent.layout.layout_sha256,
        "source_parameter_tensor_count": len(parent.layout.names),
        "source_scalar_parameter_count": parent.layout.scalar_count,
        "coarse_group_layout_path": COARSE_GROUP_LAYOUT_FILENAME,
        "coarse_group_layout_sha256": _sha256_bytes(group_layout_bytes),
        "group_order": list(GROUP_IDS),
        "group_scalar_counts": dict(GROUP_SCALAR_COUNTS),
    }
    family, severity = _condition_parts(condition)
    payloads = build_stage_b1_cell_payloads(
        protocol_id=contract.protocol_id,
        config_sha256=str(contract.config_file_sha256),
        cell={
            "dataset": dataset,
            "condition": condition,
            "corruption_family": family,
            "severity": severity,
            "replicate": "R0",
        },
        ordered_image_ids=parent.ordered_image_ids,
        dataset_binding={
            "split_name": "train",
            "train_split_sha256": dataset_binding["train_split_sha256"],
            "checkpoint_role": dataset_binding["checkpoint_role"],
            "checkpoint_path": dataset_binding["checkpoint_path"],
            "checkpoint_sha256": dataset_binding["checkpoint_sha256"],
        },
        parent_lineage=_parent_lineage(parent),
        parameter_layout=parameter_layout,
        region_gradient_basis=basis,
        episode_records=records,
        coarse_group_layout=group_layout,
        outer_access_receipt_bytes=access_bytes,
        code_seal=code_seal,
        execution={
            "runtime": dict(runtime),
            "source_model_build_count": 1,
            "source_forward_count": FORMAL_IMAGE_COUNT,
            "region_backward_count": FORMAL_IMAGE_COUNT * len(STORAGE_BASIS),
            "optimizer_build_count": 0,
            "optimizer_step_count": 0,
            "model_weight_update_count": 0,
            "checkpoint_write_count": 0,
            "state_reset_passed_count": FORMAL_IMAGE_COUNT,
            "rng_reset_passed_count": FORMAL_IMAGE_COUNT,
            "validation_payload_access_count": 0,
            "test_payload_access_count": 0,
        },
        data_boundary=MANIFEST_DATA_BOUNDARY,
        authorization=CELL_AUTHORIZATION,
    )
    if set(payloads) != CELL_MEMBERS:
        raise P3StageB1RunnerError("B1 cell builder returned a different member set")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{condition}.b1-build-", dir=destination.parent)
    )
    for name in sorted(payloads):
        _write_bytes(staging / name, payloads[name])

    def semantic_verifier(path: Path) -> Any:
        if _code_seal(contract) != code_seal:
            raise P3StageB1RunnerError(
                "B1 implementation changed across cell publication"
            )
        return verify_stage_b1_cell_shard(
            path,
            repository_root=PROJECT_ROOT,
            config=contract.raw,
            expected_config_sha256=str(contract.config_file_sha256),
            verify_live_parents=(path == destination),
            expected_code_seal=code_seal,
        )

    published = publish_flat_directory_noreplace(
        staging,
        destination,
        expected_members=sorted(CELL_MEMBERS),
        semantic_verifier=semantic_verifier,
    )
    verified = semantic_verifier(published)
    if _code_seal(contract) != code_seal:
        raise P3StageB1RunnerError(
            "B1 implementation changed after cell publication"
        )
    if verified.dataset != dataset or verified.condition != condition:
        raise P3StageB1RunnerError("published B1 cell identity differs")
    return published


def smoke_cell(
    *, dataset: str, condition: str, config_path: Path, device: torch.device
) -> dict[str, Any]:
    contract = _load_contract(config_path)
    basis, records, _layout, _access, parent, runtime, _code_seal_value = _compute_cell(
        contract=contract,
        dataset=dataset,
        condition=condition,
        device=device,
        image_count=1,
    )
    first = records[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "p3_stage_b1_one_image_smoke",
        "passed": True,
        "dataset": dataset,
        "condition": condition,
        "image_id": parent.ordered_image_ids[0],
        "basis_shape": list(basis.shape),
        "basis_finite": bool(np.isfinite(basis).all()),
        "source_logits_bit_exact": first["source_integrity"]["source_logits_bit_exact"],
        "parent_entropy_consistency_passed": first["gradient_integrity"]["parent_consistency_passed"],
        "state_restored": first["source_integrity"]["state_restored"],
        "rng_restored": first["source_integrity"]["rng_restored"],
        "runtime": dict(runtime),
        "filesystem_created": False,
        "optimizer_build_count": 0,
        "parameter_update_count": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
        "paper_result": False,
        "stage_b3_authorized": False,
    }


def validate_only(config_path: Path) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise P3StageB1RunnerError(
            "validate-only must run before any CUDA context is initialized"
        )
    contract = _load_contract(config_path)
    if torch.cuda.is_initialized():
        raise P3StageB1RunnerError(
            "B1 contract validation unexpectedly initialized CUDA"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "p3_stage_b1_contract_validate_only",
        "valid": True,
        "config_sha256": contract.config_file_sha256,
        "dataset_count": len(contract.datasets),
        "condition_count_per_dataset": len(contract.conditions),
        "required_cell_count": len(contract.datasets) * len(contract.conditions),
        "required_episode_count": len(contract.datasets) * len(contract.conditions) * 64,
        "filesystem_created": False,
        "dataset_payload_opened": False,
        "cuda_initialized": False,
        "paper_result": False,
        "stage_b3_authorized": False,
    }


_MECHANISM_STATUS_IDS: Final = frozenset(
    {
        "background_norm_dominance",
        "background_cancellation",
        "subthreshold_erasure",
    }
)
_MECHANISM_STATUS_VALUES: Final = frozenset(
    {"supported", "not_supported", "not_estimable"}
)


def _assert_aggregate_cpu_only(*, boundary: str) -> None:
    if torch.cuda.is_initialized():
        raise P3StageB1RunnerError(
            f"Stage-B1 aggregate must remain CPU-only ({boundary})"
        )


def _aggregate_result(verified: VerifiedStageB1Aggregate) -> dict[str, Any]:
    statuses = dict(verified.mechanism_statuses)
    if set(statuses) != _MECHANISM_STATUS_IDS or any(
        value not in _MECHANISM_STATUS_VALUES for value in statuses.values()
    ):
        raise P3StageB1RunnerError(
            "aggregate did not report the exact three frozen mechanism statuses"
        )
    if (
        verified.candidate_selection_performed
        or verified.stage_b3_authorized
        or verified.p5_authorized
    ):
        raise P3StageB1RunnerError(
            "Stage-B1 aggregate cannot select candidates or authorize B3/P5"
        )
    return {
        "path": str(verified.path),
        "cell_count": verified.cell_count,
        "episode_count": verified.episode_count,
        "stratified_record_count": verified.stratified_record_count,
        "summary_record_count": verified.summary_record_count,
        "manifest_sha256": verified.manifest_sha256,
        "complete_sha256": verified.complete_sha256,
        "mechanism_evidence_sha256": verified.mechanism_evidence_sha256,
        "mechanism_statuses": statuses,
        "candidate_selection_performed": False,
        "stage_b3_authorized": False,
        "p5_authorized": False,
    }


def _collect_aggregate_preflight(
    *, contract: P3StageB1Contract, cell_code_seal: Mapping[str, Any]
) -> StageB1AggregatePreflight:
    return collect_stage_b1_preflight(
        repository_root=PROJECT_ROOT,
        output_root_relative=str(contract.output_root),
        config=contract.raw,
        config_sha256=str(contract.config_file_sha256),
        expected_code_seal=cell_code_seal,
    )


def run_formal_aggregate(*, config_path: Path) -> dict[str, Any]:
    """Build and atomically publish the CPU-only fixed 39-cell aggregate."""

    _assert_aggregate_cpu_only(boundary="entry")
    contract = _load_contract(config_path)
    destination = _fixed_aggregate_path(contract)
    _repository_relative(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"B1 aggregate destination exists: {destination}")

    # This is deliberately a live public verification of every one of the
    # 39 completed cells.  The frozen cell implementation seal must agree in
    # every cell and must remain unchanged throughout aggregation/publication.
    cell_code_seal = _code_seal(contract)
    preflight = _collect_aggregate_preflight(
        contract=contract, cell_code_seal=cell_code_seal
    )
    if len(preflight.lineage) != 39 or len(preflight.records) != 39 * 64:
        raise P3StageB1RunnerError("aggregate preflight is not exact 39 x 64")
    if _code_seal(contract) != cell_code_seal:
        raise P3StageB1RunnerError(
            "B1 implementation changed during aggregate preflight"
        )
    _assert_aggregate_cpu_only(boundary="after live 39-cell preflight")

    payloads = build_stage_b1_aggregate_payloads(
        preflight,
        mechanism_gate=contract.raw["mechanism_evidence_flags"],
    )
    if set(payloads) != AGGREGATE_MEMBERS:
        raise P3StageB1RunnerError(
            "B1 aggregate builder returned a different member set"
        )
    # Revalidate immediately before creating the publication parent or any
    # staging artifact.  This is the frozen pre-publication seal boundary.
    if _code_seal(contract) != cell_code_seal:
        raise P3StageB1RunnerError(
            "B1 implementation changed before aggregate publication"
        )
    _assert_aggregate_cpu_only(boundary="before publication")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".R0.b1-aggregate-build-", dir=destination.parent)
    )
    for name in sorted(payloads):
        _write_bytes(staging / name, payloads[name])

    verified_by_path: dict[Path, VerifiedStageB1Aggregate] = {}

    def semantic_verifier(path: Path) -> VerifiedStageB1Aggregate:
        _assert_aggregate_cpu_only(boundary="semantic verification")
        if _code_seal(contract) != cell_code_seal:
            raise P3StageB1RunnerError(
                "B1 implementation changed across aggregate publication"
            )
        absolute = Path(os.path.abspath(os.fspath(path)))
        verified = verify_stage_b1_aggregate_shard(
            absolute,
            repository_root=PROJECT_ROOT,
            output_root_relative=str(contract.output_root),
            config=contract.raw,
            expected_config_sha256=str(contract.config_file_sha256),
            mechanism_gate=contract.raw["mechanism_evidence_flags"],
            # The staging payload is rebuilt from the already live-verified
            # preflight.  At the canonical name, repeat all 39 public live
            # verifications so a concurrent cell drift triggers rollback.
            verify_live_cells=(absolute == destination),
            expected_cell_code_seal=cell_code_seal,
        )
        _aggregate_result(verified)
        verified_by_path[absolute] = verified
        return verified

    published = publish_flat_directory_noreplace(
        staging,
        destination,
        expected_members=sorted(AGGREGATE_MEMBERS),
        semantic_verifier=semantic_verifier,
    )
    published = Path(os.path.abspath(os.fspath(published)))
    verified = verified_by_path.get(published)
    if verified is None:
        verified = semantic_verifier(published)
    if _code_seal(contract) != cell_code_seal:
        raise P3StageB1RunnerError(
            "B1 implementation changed after aggregate publication"
        )
    _assert_aggregate_cpu_only(boundary="return")
    return _aggregate_result(verified)


def verify_formal_aggregate(*, config_path: Path) -> dict[str, Any]:
    """Publicly reverify the canonical aggregate and all 39 live cells."""

    _assert_aggregate_cpu_only(boundary="entry")
    contract = _load_contract(config_path)
    destination = _fixed_aggregate_path(contract)
    _repository_relative(destination)
    cell_code_seal = _code_seal(contract)
    verified = verify_stage_b1_aggregate_shard(
        destination,
        repository_root=PROJECT_ROOT,
        output_root_relative=str(contract.output_root),
        config=contract.raw,
        expected_config_sha256=str(contract.config_file_sha256),
        mechanism_gate=contract.raw["mechanism_evidence_flags"],
        verify_live_cells=True,
        expected_cell_code_seal=cell_code_seal,
    )
    if _code_seal(contract) != cell_code_seal:
        raise P3StageB1RunnerError(
            "B1 implementation changed during aggregate verification"
        )
    _assert_aggregate_cpu_only(boundary="return")
    return _aggregate_result(verified)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / CONFIG_RELATIVE_PATH
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("aggregate")
    sub.add_parser("verify-aggregate")
    for command in ("smoke", "run-cell", "verify-cell"):
        child = sub.add_parser(command)
        child.add_argument("--dataset", required=True)
        child.add_argument("--condition", required=True)
        if command in {"smoke", "run-cell"}:
            child.add_argument("--device", default="cuda:0")
        if command == "verify-cell":
            child.add_argument("--path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(os.path.abspath(os.fspath(args.config)))
    if args.command == "validate":
        result: Any = validate_only(config_path)
    elif args.command == "aggregate":
        result = run_formal_aggregate(config_path=config_path)
    elif args.command == "verify-aggregate":
        result = verify_formal_aggregate(config_path=config_path)
    elif args.command == "smoke":
        result = smoke_cell(
            dataset=args.dataset,
            condition=args.condition,
            config_path=config_path,
            device=torch.device(args.device),
        )
    elif args.command == "run-cell":
        result = {
            "published_path": str(
                run_formal_cell(
                    dataset=args.dataset,
                    condition=args.condition,
                    config_path=config_path,
                    device=torch.device(args.device),
                )
            )
        }
    else:
        contract = _load_contract(config_path)
        path = args.path or _fixed_output_path(
            contract, args.dataset, args.condition
        )
        verified = verify_stage_b1_cell_shard(
            path,
            repository_root=PROJECT_ROOT,
            config=contract.raw,
            expected_config_sha256=str(contract.config_file_sha256),
            verify_live_parents=True,
            expected_code_seal=_code_seal(contract),
        )
        if verified.dataset != args.dataset or verified.condition != args.condition:
            raise P3StageB1RunnerError("verified B1 cell identity differs from CLI")
        result = {
            "path": str(verified.path),
            "dataset": verified.dataset,
            "condition": verified.condition,
            "image_count": verified.image_count,
            "record_count": verified.record_count,
            "manifest_sha256": verified.manifest_sha256,
            "complete_sha256": verified.complete_sha256,
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
