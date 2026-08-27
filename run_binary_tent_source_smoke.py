"""Run the train-derived Binary TENT implementation/repeatability smoke.

The run is intentionally separate from the formal fixed-test benchmark.  It
reuses the exact 32 IRSTD-1K train IDs and Gaussian-noise S3 materialization
contract of the completed AdaBN source pilot.  Ground-truth masks are retained
only by the outer diagnostic evaluator and never cross the episodic method
boundary.  Adam(lr=1e-3) is provisional implementation-smoke plumbing, not a
selected or frozen scientific hyperparameter.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any, Iterator
import warnings

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import yaml

from dataio.corruption_cache import TensorSequenceHasher, ordered_ids_sha256
from metrics.irstd_metrics import IRSTDEvaluationProtocol, UnifiedResearchEvaluator
from metrics.official_metric_adapter import OfficialMetricAdapter
from run_adabn_source_pilot import (
    PilotPaths,
    SourcePilotSample,
    materialize_label_free_inputs,
    validate_protocol_contract,
)
import test_source as source_runner
from tta.binary_tent import (
    BN_PROTOCOL_BATCH_STATS,
    BN_PROTOCOL_SOURCE_STATS,
    CUDA_BACKWARD_TEMPORARILY_DISABLE,
    BinaryTentMethod,
)
from tta.binary_tent_runner import (
    BinaryTentEpisodeResult,
    BinaryTentEpisodicRunner,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager, StateFingerprint


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "binary_tent_source_smoke_v1.yaml"
EXPECTED_PROTOCOL_ID = "cr-sitta-binary-tent-source-implementation-smoke-v1"
PREDICTION_NAMES = ("source_pre", "tent_pre", "tent_post")
PROTOCOL_DIRECTORY = {
    BN_PROTOCOL_BATCH_STATS: "batch_stats",
    BN_PROTOCOL_SOURCE_STATS: "source_stats",
}
PROVENANCE_PATHS = (
    "THIRD_PARTY.md",
    "THIRD_PARTY_COMMITS.txt",
    "SFS_MSDeformAttn/ops/__init__.py",
    "SFS_MSDeformAttn/ops/functions/__init__.py",
    "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
    "SFS_MSDeformAttn/ops/modules/__init__.py",
    "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
    "SFS_MSDeformAttn/ops/src/cpu/ms_deform_attn_cpu.cpp",
    "SFS_MSDeformAttn/ops/src/cuda/ms_deform_attn_cuda.cu",
    "SFS_MSDeformAttn/ops/src/vision.cpp",
    "configs/binary_tent_source_smoke_v1.yaml",
    "configs/adabn_source_pilot_smoke_v1.yaml",
    "configs/adabn_source_pilot_ids_irstd1k.txt",
    "corruptions/corruption_protocol.py",
    "corruptions/infrared_corruptions.py",
    "corruptions/severity_tables.yaml",
    "dataio/corruption_cache.py",
    "dataio/research_dataset.py",
    "environment.cr-sitta.yml",
    "environment.linux-64.explicit.txt",
    "metrics/connected_components.py",
    "metrics/irstd_metrics.py",
    "metrics/official_metric_adapter.py",
    "metrics/target_matching.py",
    "model/MSHNet_NSFPN.py",
    "model/NS_FPN.py",
    "model/diff_cross_attns.py",
    "requirements.lock.txt",
    "run_adabn_source_pilot.py",
    "run_binary_tent_source_smoke.py",
    "test_source.py",
    "third_party/binary_tent_reference_v1.json",
    "third_party/tent/LICENSE",
    "third_party/tent/conf.py",
    "third_party/tent/tent.py",
    "tta/binary_tent.py",
    "tta/binary_tent_runner.py",
    "tta/episodic_runner.py",
    "tta/model_adapter.py",
    "tta/state_manager.py",
    "utils/metric.py",
)


@dataclass(frozen=True)
class SmokePaths:
    config: Path
    adabn_config: Path
    selected_ids_file: Path
    frozen_adabn_root: Path
    output_root: Path


@dataclass(frozen=True)
class RuntimePolicySnapshot:
    deterministic_algorithms_enabled: bool
    deterministic_algorithms_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_present: bool
    cublas_workspace_value: str | None
    pythonhashseed_present: bool
    pythonhashseed_value: str | None


@dataclass(frozen=True)
class PredictionArrays:
    logits: np.ndarray
    probability: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class EpisodeArrays:
    predictions: Mapping[str, PredictionArrays]
    step_norm: float
    diagnostics: Mapping[str, Any]
    state: Mapping[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional atomically published output-root override.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen contracts without opening pixels or running a model.",
    )
    return parser


def _project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a JSON mapping: {path}")
    return dict(value)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def protocol_scope_gate_checks(config: Mapping[str, Any]) -> dict[str, bool]:
    """Express non-result declarations as positive, composable gate predicates."""

    scope = _require_mapping(config.get("scope"), "scope")
    return {
        "paper_result_is_false": scope.get("paper_result") is False,
        "scientific_result_is_not_frozen": scope.get(
            "scientific_result_frozen"
        )
        is False,
        "tuning_is_disallowed": scope.get("tuning_allowed") is False,
        "optimizer_selection_is_disallowed": scope.get(
            "optimizer_selection_allowed"
        )
        is False,
        "test_images_are_disallowed": scope.get("use_test_images") is False,
        "test_labels_are_disallowed": scope.get("use_test_labels") is False,
        "train_labels_are_outer_evaluator_only": scope.get(
            "train_labels_outer_evaluator_only"
        )
        is True,
    }


def load_smoke_config(path: str | Path = DEFAULT_CONFIG) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Binary TENT smoke config does not exist: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("Binary TENT smoke config must be a YAML mapping")
    config = dict(loaded)
    _require_equal(config.get("schema_version"), 1, "schema_version")
    _require_equal(config.get("protocol_id"), EXPECTED_PROTOCOL_ID, "protocol_id")

    scope = _require_mapping(config.get("scope"), "scope")
    for key, expected in (
        ("paper_result", False),
        ("scientific_result_frozen", False),
        ("tuning_allowed", False),
        ("optimizer_selection_allowed", False),
        ("use_test_images", False),
        ("use_test_labels", False),
        ("train_labels_outer_evaluator_only", True),
    ):
        _require_equal(scope.get(key), expected, f"scope.{key}")

    frozen = _require_mapping(config.get("frozen_source_input"), "frozen_source_input")
    _require_equal(frozen.get("selected_count"), 32, "selected_count")
    _require_equal(frozen.get("dataset"), "IRSTD-1K", "dataset")
    _require_equal(
        dict(_require_mapping(frozen.get("condition"), "condition")),
        {"corruption": "gaussian_noise", "severity": 3, "seed": 42},
        "condition",
    )

    method = _require_mapping(config.get("method"), "method")
    _require_equal(
        tuple(method.get("protocols", ())),
        (BN_PROTOCOL_BATCH_STATS, BN_PROTOCOL_SOURCE_STATS),
        "method.protocols",
    )
    _require_equal(method.get("entropy_eps"), 1e-6, "method.entropy_eps")
    _require_equal(method.get("optimizer_steps_per_image"), 1, "optimizer steps")
    optimizer = _require_mapping(method.get("optimizer"), "method.optimizer")
    for key, expected in (
        ("name", "Adam"),
        ("learning_rate", 1e-3),
        ("status", "provisional_implementation_smoke_only"),
        ("selected_by_this_smoke", False),
    ):
        _require_equal(optimizer.get(key), expected, f"method.optimizer.{key}")

    execution = _require_mapping(config.get("execution"), "execution")
    _require_equal(execution.get("device_type"), "cuda", "execution.device_type")
    repeats = execution.get("canonical_repeat_runs")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 5:
        raise ValueError("execution.canonical_repeat_runs must be an integer in [1, 5]")
    _require_equal(execution.get("reverse_order_runs"), 1, "reverse_order_runs")
    _require_equal(execution.get("batch_size"), 1, "batch_size")
    _require_equal(execution.get("num_workers"), 0, "num_workers")
    deterministic = _require_mapping(
        execution.get("deterministic_algorithms"), "deterministic_algorithms"
    )
    for key, expected in (
        ("policy", "strict_forwards_temporary_backward_disable"),
        ("forward_enabled", True),
        ("forward_warn_only", False),
        ("backward_enabled", False),
        ("backward_disable_scope", "entropy_backward_only"),
        ("post_backward_enabled", True),
        ("post_forward_enabled", True),
    ):
        _require_equal(
            deterministic.get(key), expected, f"deterministic_algorithms.{key}"
        )
    _require_equal(execution.get("cublas_workspace_config"), ":4096:8", "CUBLAS")
    _require_equal(execution.get("cudnn_deterministic"), True, "cuDNN deterministic")
    _require_equal(execution.get("cudnn_benchmark"), False, "cuDNN benchmark")

    repeatability = _require_mapping(config.get("repeatability"), "repeatability")
    _require_equal(
        repeatability.get("tent_post_bit_exact_required"),
        False,
        "tent_post_bit_exact_required",
    )
    outputs = _require_mapping(config.get("outputs"), "outputs")
    _require_equal(outputs.get("atomic_sibling_staging"), True, "atomic staging")
    _require_equal(outputs.get("refuse_overwrite"), True, "refuse overwrite")
    return config_path, config


def resolve_smoke_paths(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    output_override: Path | None = None,
) -> SmokePaths:
    frozen = config["frozen_source_input"]
    artifact = frozen["fixed_adabn_artifact"]
    output = (
        _project_path(config["outputs"]["root"])
        if output_override is None
        else output_override.expanduser().resolve()
    )
    paths = SmokePaths(
        config=config_path,
        adabn_config=_project_path(frozen["adabn_config"]),
        selected_ids_file=_project_path(frozen["selected_ids_file"]),
        frozen_adabn_root=_project_path(artifact["root"]),
        output_root=output,
    )
    for label, path, directory in (
        ("AdaBN config", paths.adabn_config, False),
        ("selected IDs", paths.selected_ids_file, False),
        ("frozen AdaBN artifact", paths.frozen_adabn_root, True),
    ):
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            raise FileNotFoundError(f"{label} does not exist: {path}")
    return paths


def _capture_runtime_policy() -> RuntimePolicySnapshot:
    present = "CUBLAS_WORKSPACE_CONFIG" in os.environ
    pythonhash_present = "PYTHONHASHSEED" in os.environ
    return RuntimePolicySnapshot(
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        deterministic_algorithms_warn_only=(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cublas_workspace_present=present,
        cublas_workspace_value=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        pythonhashseed_present=pythonhash_present,
        pythonhashseed_value=os.environ.get("PYTHONHASHSEED"),
    )


def _restore_runtime_policy(snapshot: RuntimePolicySnapshot) -> None:
    torch.use_deterministic_algorithms(
        snapshot.deterministic_algorithms_enabled,
        warn_only=snapshot.deterministic_algorithms_warn_only,
    )
    torch.backends.cudnn.deterministic = snapshot.cudnn_deterministic
    torch.backends.cudnn.benchmark = snapshot.cudnn_benchmark
    if snapshot.cublas_workspace_present:
        assert snapshot.cublas_workspace_value is not None
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = snapshot.cublas_workspace_value
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if snapshot.pythonhashseed_present:
        assert snapshot.pythonhashseed_value is not None
        os.environ["PYTHONHASHSEED"] = snapshot.pythonhashseed_value
    else:
        os.environ.pop("PYTHONHASHSEED", None)


@contextmanager
def binary_tent_cuda_policy(
    *, seed: int, workspace_config: str
) -> Iterator[dict[str, Any]]:
    """Apply the audited real-NS-FPN CUDA policy and restore it exactly.

    ``CUBLAS_WORKSPACE_CONFIG`` is installed before ``seed_everything`` because
    that helper initializes CUDA RNG state.  The helper enables strict Torch
    deterministic algorithms.  Source, TENT-pre and TENT-post forwards remain
    strict so they retain byte-level parity with the frozen Source/AdaBN
    references.  :class:`BinaryTentMethod` alone temporarily disables the
    global Torch policy around entropy backward, then restores it before the
    optimizer step and post-update forward.  Repeatability of that backward
    path is measured empirically instead of being falsely asserted.
    """

    before = _capture_runtime_policy()
    cuda_initialized_before = bool(torch.cuda.is_initialized())
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace_config
    try:
        source_runner.seed_everything(seed)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        active = _capture_runtime_policy()
        if (
            not active.deterministic_algorithms_enabled
            or active.deterministic_algorithms_warn_only
        ):
            raise RuntimeError(
                "Binary TENT CUDA smoke requires strict deterministic forwards"
            )
        yield {
            "before": asdict(before),
            "active": asdict(active),
            "cuda_initialized_before_workspace_config": cuda_initialized_before,
            "workspace_installed_before_seed_everything": True,
            "episode_backward_policy": CUDA_BACKWARD_TEMPORARILY_DISABLE,
            "backward_disable_scope": "entropy_backward_only",
        }
    finally:
        _restore_runtime_policy(before)


def runtime_policy_is_restored(before: Mapping[str, Any]) -> bool:
    return asdict(_capture_runtime_policy()) == dict(before)


def _verify_manifest_file(
    root: Path,
    manifest: Mapping[str, Any],
    relative: str,
) -> Path:
    files = _require_mapping(manifest.get("files"), "AdaBN artifact manifest files")
    record = _require_mapping(files.get(relative), f"manifest entry {relative}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"frozen AdaBN file is missing: {path}")
    _require_equal(path.stat().st_size, int(record["bytes"]), f"{relative} bytes")
    _require_equal(
        source_runner.sha256_file(path), record["sha256"], f"{relative} SHA256"
    )
    return path


def validate_frozen_source_reference(
    config: Mapping[str, Any],
    paths: SmokePaths,
) -> tuple[
    dict[str, Any],
    PilotPaths,
    tuple[str, ...],
    dict[tuple[str, str], Path],
    dict[str, Any],
    dict[str, Any],
]:
    """Verify the immutable train-side AdaBN contract without opening test pixels."""

    frozen = config["frozen_source_input"]
    artifact_contract = frozen["fixed_adabn_artifact"]
    for path, expected, label in (
        (paths.adabn_config, frozen["adabn_config_sha256"], "AdaBN config"),
        (
            paths.selected_ids_file,
            frozen["selected_ids_file_sha256"],
            "selected IDs file",
        ),
        (
            paths.frozen_adabn_root / "COMPLETE.json",
            artifact_contract["complete_sha256"],
            "AdaBN COMPLETE",
        ),
        (
            paths.frozen_adabn_root / "artifact_manifest.json",
            artifact_contract["artifact_manifest_sha256"],
            "AdaBN artifact manifest",
        ),
        (
            paths.frozen_adabn_root / "provenance.json",
            artifact_contract["provenance_sha256"],
            "AdaBN provenance",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        _require_equal(source_runner.sha256_file(path), expected, f"{label} SHA256")

    source_config, pilot_paths, selected_ids, selected_files, source_contract = (
        validate_protocol_contract(paths.adabn_config)
    )
    _require_equal(len(selected_ids), int(frozen["selected_count"]), "selected count")
    _require_equal(
        ordered_ids_sha256(selected_ids), frozen["selected_ids_sha256"], "selected IDs"
    )
    _require_equal(
        pilot_paths.selected_ids_file, paths.selected_ids_file, "selected IDs path"
    )
    _require_equal(source_config["dataset"]["name"], frozen["dataset"], "dataset")
    _require_equal(
        {
            "corruption": source_config["condition"]["corruption"],
            "severity": int(source_config["condition"]["severity"]),
            "seed": int(source_config["condition"]["seed"]),
        },
        dict(frozen["condition"]),
        "source condition",
    )

    complete = _load_json(paths.frozen_adabn_root / "COMPLETE.json")
    manifest = _load_json(paths.frozen_adabn_root / "artifact_manifest.json")
    provenance = _load_json(paths.frozen_adabn_root / "provenance.json")
    _require_equal(complete.get("complete"), True, "AdaBN complete flag")
    _require_equal(
        complete.get("artifact_manifest_sha256"),
        artifact_contract["artifact_manifest_sha256"],
        "AdaBN completion manifest lineage",
    )
    _require_equal(
        complete.get("protocol_sha256"),
        frozen["adabn_config_sha256"],
        "AdaBN protocol lineage",
    )
    _require_equal(
        provenance.get("selected_ids_sha256"),
        frozen["selected_ids_sha256"],
        "AdaBN provenance selected IDs",
    )
    materialized = _require_mapping(
        provenance.get("materialized_inputs"), "AdaBN materialized inputs"
    )
    _require_equal(
        materialized.get("input_tensor_sequence_sha256"),
        frozen["input_tensor_sequence_sha256"],
        "AdaBN materialized input hash",
    )
    _require_equal(
        materialized.get("source_train_mask_tensor_sequence_sha256"),
        frozen["mask_tensor_sequence_sha256"],
        "AdaBN materialized mask hash",
    )
    _require_equal(materialized.get("method_received_mask"), False, "method label firewall")
    boundary = _require_mapping(
        provenance.get("fixed_test_boundary"), "AdaBN fixed-test boundary"
    )
    for key, expected in (
        ("test_dataset_constructed", False),
        ("test_images_opened", 0),
        ("test_masks_opened", 0),
    ):
        _require_equal(boundary.get(key), expected, f"fixed-test boundary {key}")

    metrics_relative = "IRSTD-1K/gaussian_noise_S3/metrics.json"
    metrics_path = _verify_manifest_file(
        paths.frozen_adabn_root, manifest, metrics_relative
    )
    _require_equal(
        source_runner.sha256_file(metrics_path),
        artifact_contract["condition_metrics_sha256"],
        "AdaBN condition metrics hash",
    )
    for image_id in selected_ids:
        stem = source_runner.safe_artifact_stem(image_id)
        for directory in ("probability_maps_256", "source_probability_maps_256"):
            _verify_manifest_file(
                paths.frozen_adabn_root,
                manifest,
                f"IRSTD-1K/gaussian_noise_S3/{directory}/{stem}.npy",
            )

    reference = {
        "complete": complete,
        "manifest": manifest,
        "provenance": provenance,
        "all_64_referenced_probability_files_hash_verified": True,
    }
    return (
        source_config,
        pilot_paths,
        selected_ids,
        selected_files,
        source_contract,
        reference,
    )


def _raw_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    dtype = str(contiguous.dtype).encode("ascii")
    digest.update(struct.pack(">Q", len(dtype)))
    digest.update(dtype)
    digest.update(struct.pack(">Q", contiguous.ndim))
    for dimension in contiguous.shape:
        digest.update(struct.pack(">q", int(dimension)))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(temporary, path)


def _atomic_save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(
        temporary, format="PNG"
    )
    os.replace(temporary, path)


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_outer_train_targets(
    *,
    selected_ids: Sequence[str],
    selected_files: Mapping[tuple[str, str], Path],
    image_size: int,
    expected_sequence_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load only selected training masks for the outer stability evaluator."""

    targets: dict[str, np.ndarray] = {}
    hasher = TensorSequenceHasher()
    file_digest = hashlib.sha256()
    for image_id in selected_ids:
        path = selected_files[("mask", image_id)]
        with Image.open(path) as handle:
            resized = handle.convert("L").resize(
                (image_size, image_size), resample=Image.Resampling.NEAREST
            )
            target = np.asarray(resized, dtype=np.float32) / 255.0
        target = np.ascontiguousarray(target, dtype=np.float32)
        if target.shape != (image_size, image_size) or not np.isfinite(target).all():
            raise RuntimeError(f"invalid outer-evaluator target for {image_id}")
        targets[image_id] = target
        hasher.update(image_id, torch.from_numpy(target[None, ...]))
        file_digest.update(
            f"{image_id}\t{source_runner.sha256_file(path)}\n".encode("utf-8")
        )
    sequence_hash = hasher.hexdigest()
    _require_equal(sequence_hash, expected_sequence_sha256, "outer target sequence")
    return targets, {
        "role": "outer_train_side_repeatability_evaluator_only",
        "selected_train_masks_opened": len(targets),
        "test_masks_opened": 0,
        "target_tensor_sequence_sha256": sequence_hash,
        "selected_mask_files_manifest_sha256": file_digest.hexdigest(),
        "method_received_mask": False,
        "retained_until_outer_evaluation_complete": True,
    }


def _probability_float32(logits: Tensor | np.ndarray) -> np.ndarray:
    tensor = logits if isinstance(logits, Tensor) else torch.from_numpy(logits)
    probability = torch.sigmoid(tensor.detach().cpu().float()).numpy()
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim == 4 and probability.shape[:2] == (1, 1):
        probability = probability[0, 0]
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise RuntimeError("probability must be a finite two-dimensional float32 array")
    return np.ascontiguousarray(probability, dtype=np.float32)


def _prediction_arrays(logits: Tensor, *, threshold: float) -> PredictionArrays:
    value = logits.detach().cpu().numpy().astype(np.float32, copy=False)
    if value.ndim == 4 and value.shape[:2] == (1, 1):
        value = value[0, 0]
    if value.ndim != 2 or not np.isfinite(value).all():
        raise RuntimeError("logits must be a finite [H,W] float32 array")
    value = np.ascontiguousarray(value, dtype=np.float32)
    probability = _probability_float32(value)
    mask = np.where(probability > threshold, 255, 0).astype(np.uint8)
    return PredictionArrays(logits=value, probability=probability, mask=mask)


def _fingerprint_dict(value: StateFingerprint) -> dict[str, Any]:
    return asdict(value)


def audit_binary_tent_episode(
    result: BinaryTentEpisodeResult,
    *,
    bn_protocol: str,
    expected_trainable_tensors: int,
    expected_trainable_scalars: int,
) -> dict[str, bool]:
    episode = result.episode
    diagnostics = dict(result.diagnostics)
    source = episode.source_fingerprint
    prepared = episode.state_after_prepare_fingerprint
    adapted = episode.state_after_adapt_fingerprint
    post = episode.state_after_post_fingerprint
    checks = {
        "input_unchanged": episode.input_unchanged,
        "decision_adapted": episode.outcome.decision == "adapted",
        "exactly_one_optimizer_step": episode.outcome.optimizer_steps == 1
        and diagnostics.get("optimizer_steps") == 1,
        "configured_bn_protocol": diagnostics.get("bn_protocol") == bn_protocol,
        "expected_trainable_tensors": int(
            diagnostics.get("number_trainable_tensors", -1)
        )
        == expected_trainable_tensors,
        "expected_trainable_scalars": int(
            diagnostics.get("number_trainable_scalars", -1)
        )
        == expected_trainable_scalars,
        "finite_diagnostics": diagnostics.get("finite") is True
        and math.isfinite(float(diagnostics.get("gradient_norm", math.nan)))
        and math.isfinite(float(diagnostics.get("step_norm", math.nan))),
        "nonzero_gradient": diagnostics.get("gradient_nonzero") is True,
        "nonzero_parameter_step": diagnostics.get(
            "actual_parameter_delta_nonzero"
        )
        is True,
        "update_unconditionally_accepted": diagnostics.get("update_accepted") is True,
        "amp_disabled": diagnostics.get("amp_enabled") is False,
        "float32_numeric_precision": diagnostics.get("numeric_precision")
        == "float32",
        "optimizer_params_exact_all_bn_affine": diagnostics.get(
            "optimizer_params_exact_all_bn_affine"
        )
        is True,
        "only_bn_affine_requires_grad": diagnostics.get(
            "only_bn_affine_requires_grad"
        )
        is True,
        "non_adaptable_parameters_resident_bit_exact": diagnostics.get(
            "non_adaptable_parameters_resident_bit_exact"
        )
        is True,
        "registered_buffers_resident_bit_exact": diagnostics.get(
            "registered_buffers_resident_bit_exact"
        )
        is True,
        "forward_deterministic_algorithms_strict": diagnostics.get(
            "forward_deterministic_algorithms_enabled"
        )
        is True
        and diagnostics.get("forward_deterministic_algorithms_warn_only")
        is False,
        "cuda_backward_temporarily_nondeterministic": diagnostics.get(
            "episode_device_type"
        )
        == "cuda"
        and diagnostics.get("backward_deterministic_algorithms_enabled")
        is False
        and diagnostics.get("backward_deterministic_algorithms_warn_only")
        is False,
        "cuda_backward_policy_configured": diagnostics.get(
            "cuda_backward_determinism_policy"
        )
        == CUDA_BACKWARD_TEMPORARILY_DISABLE,
        "deterministic_policy_restored_after_backward": diagnostics.get(
            "deterministic_policy_restored_after_backward"
        )
        is True,
        "post_backward_deterministic_algorithms_strict": diagnostics.get(
            "deterministic_algorithms_enabled"
        )
        is True,
        "prepare_changes_runtime_only": episode.state_changes_after_prepare
        == ("runtime",),
        "adapt_changes_model_optimizer_runtime_only": set(
            episode.state_changes_after_adapt
        )
        == {"model", "optimizer", "runtime"},
        "post_changes_model_optimizer_runtime_only": set(
            episode.state_changes_after_post
        )
        == {"model", "optimizer", "runtime"},
        "prepare_model_unchanged": prepared.model_sha256 == source.model_sha256,
        "prepare_optimizer_unchanged": prepared.optimizer_sha256
        == source.optimizer_sha256,
        "adapt_model_changed": adapted.model_sha256 != source.model_sha256,
        "adapt_optimizer_changed": adapted.optimizer_sha256
        != source.optimizer_sha256,
        "post_forward_state_frozen": adapted == post,
        "topology_unchanged": prepared.topology_sha256
        == adapted.topology_sha256
        == post.topology_sha256
        == source.topology_sha256,
        "gradients_cleared": prepared.gradients_sha256
        == adapted.gradients_sha256
        == post.gradients_sha256
        == source.gradients_sha256,
        "reset_exact_source": episode.reset_fingerprint == source,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"Binary TENT episode invariant failures ({bn_protocol}): {failed}"
        )
    return checks


def _episode_state_record(result: BinaryTentEpisodeResult) -> dict[str, Any]:
    episode = result.episode
    return {
        "state_changes_after_prepare": list(episode.state_changes_after_prepare),
        "state_changes_after_adapt": list(episode.state_changes_after_adapt),
        "state_changes_after_post": list(episode.state_changes_after_post),
        "source_fingerprint": _fingerprint_dict(episode.source_fingerprint),
        "state_after_prepare_fingerprint": _fingerprint_dict(
            episode.state_after_prepare_fingerprint
        ),
        "state_after_adapt_fingerprint": _fingerprint_dict(
            episode.state_after_adapt_fingerprint
        ),
        "state_after_post_fingerprint": _fingerprint_dict(
            episode.state_after_post_fingerprint
        ),
        "reset_fingerprint": _fingerprint_dict(episode.reset_fingerprint),
    }


def execute_episode_pass(
    *,
    samples: Sequence[SourcePilotSample],
    order: Sequence[str],
    runner: BinaryTentEpisodicRunner,
    method: BinaryTentMethod,
    device: torch.device,
    threshold: float,
    expected_trainable_tensors: int,
    expected_trainable_scalars: int,
) -> dict[str, EpisodeArrays]:
    sample_by_id = {sample.image_id: sample for sample in samples}
    if set(order) != set(sample_by_id) or len(order) != len(samples):
        raise ValueError("episode pass order must be a permutation of all samples")
    outputs: dict[str, EpisodeArrays] = {}
    for image_id in order:
        sample = sample_by_id[image_id]
        result = runner.run_one_image(
            image=sample.image.to(device, non_blocking=False),
            metadata=sample.metadata,
            method=method,
        )
        checks = audit_binary_tent_episode(
            result,
            bn_protocol=method.bn_protocol,
            expected_trainable_tensors=expected_trainable_tensors,
            expected_trainable_scalars=expected_trainable_scalars,
        )
        predictions = {
            "source_pre": _prediction_arrays(
                result.logits_source_pre, threshold=threshold
            ),
            "tent_pre": _prediction_arrays(result.logits_tent_pre, threshold=threshold),
            "tent_post": _prediction_arrays(
                result.logits_tent_post, threshold=threshold
            ),
        }
        diagnostics = dict(result.diagnostics)
        diagnostics["episode_invariants"] = checks
        outputs[image_id] = EpisodeArrays(
            predictions=predictions,
            step_norm=float(diagnostics["step_norm"]),
            diagnostics=diagnostics,
            state=_episode_state_record(result),
        )
    return outputs


def _metric_summary(
    predictions: Mapping[str, EpisodeArrays],
    *,
    prediction_name: str,
    selected_ids: Sequence[str],
    targets: Mapping[str, np.ndarray],
    protocol: IRSTDEvaluationProtocol,
) -> dict[str, Any]:
    if prediction_name not in PREDICTION_NAMES:
        raise ValueError(f"unknown prediction name: {prediction_name}")
    official = OfficialMetricAdapter(image_size=256)
    unified = UnifiedResearchEvaluator(protocol)
    for image_id in selected_ids:
        value = predictions[image_id].predictions[prediction_name]
        logits = torch.from_numpy(value.logits)[None, None, ...]
        target = torch.from_numpy(targets[image_id])[None, None, ...]
        source_runner._update_official_evaluator(official, logits, target)
        unified.update_probabilities(value.probability, targets[image_id])
    official_result = official.compute()
    unified_result = unified.compute().fixed
    return {
        "official_nsfpn_operating_point": {
            "iou": float(official_result.mean_iou),
            "pd": float(official_result.detection_probability[0]),
            "fa_per_million_pixels": float(
                official_result.false_alarm_pixel_rate[0] * 1_000_000.0
            ),
        },
        "unified_fixed_probability_0_5": {
            "iou": float(unified_result.pixel.intersection_over_union),
            "pd": float(unified_result.target.detection_probability),
            "fa_per_million_pixels": float(
                unified_result.target.false_alarm_pixel_rate * 1_000_000.0
            ),
        },
    }


def evaluate_pass_metrics(
    predictions: Mapping[str, EpisodeArrays],
    *,
    selected_ids: Sequence[str],
    targets: Mapping[str, np.ndarray],
    protocol: IRSTDEvaluationProtocol,
) -> dict[str, Any]:
    return {
        name: _metric_summary(
            predictions,
            prediction_name=name,
            selected_ids=selected_ids,
            targets=targets,
            protocol=protocol,
        )
        for name in PREDICTION_NAMES
    }


def metric_differences(
    candidate: Mapping[str, Any], canonical: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for prediction_name in PREDICTION_NAMES:
        result[prediction_name] = {}
        for evaluator_name in (
            "official_nsfpn_operating_point",
            "unified_fixed_probability_0_5",
        ):
            result[prediction_name][evaluator_name] = {
                metric: float(
                    candidate[prediction_name][evaluator_name][metric]
                    - canonical[prediction_name][evaluator_name][metric]
                )
                for metric in ("iou", "pd", "fa_per_million_pixels")
            }
    return result


def compare_episode_passes(
    canonical: Mapping[str, EpisodeArrays],
    candidate: Mapping[str, EpisodeArrays],
    *,
    selected_ids: Sequence[str],
) -> dict[str, Any]:
    per_image: dict[str, Any] = {}
    aggregate: dict[str, Any] = {}
    for prediction_name in PREDICTION_NAMES:
        logits_max = 0.0
        probability_max = 0.0
        probability_abs_sum = 0.0
        probability_values = 0
        mask_disagreement = 0
        mask_values = 0
        exact_logits = 0
        exact_probabilities = 0
        exact_masks = 0
        for image_id in selected_ids:
            reference = canonical[image_id].predictions[prediction_name]
            observed = candidate[image_id].predictions[prediction_name]
            logits_difference = np.abs(observed.logits - reference.logits)
            probability_difference = np.abs(
                observed.probability - reference.probability
            )
            disagreements = int(np.count_nonzero(observed.mask != reference.mask))
            record = per_image.setdefault(image_id, {})
            record[prediction_name] = {
                "logits_bit_exact": bool(
                    np.array_equal(observed.logits, reference.logits)
                ),
                "logits_max_abs_difference": float(logits_difference.max()),
                "probability_bit_exact": bool(
                    np.array_equal(observed.probability, reference.probability)
                ),
                "probability_mean_abs_difference": float(
                    probability_difference.mean(dtype=np.float64)
                ),
                "probability_max_abs_difference": float(
                    probability_difference.max()
                ),
                "mask_bit_exact": disagreements == 0,
                "mask_disagreement_pixels": disagreements,
                "mask_disagreement_rate": float(disagreements / observed.mask.size),
            }
            logits_max = max(logits_max, float(logits_difference.max()))
            probability_max = max(
                probability_max, float(probability_difference.max())
            )
            probability_abs_sum += float(
                probability_difference.sum(dtype=np.float64)
            )
            probability_values += int(probability_difference.size)
            mask_disagreement += disagreements
            mask_values += int(observed.mask.size)
            exact_logits += int(record[prediction_name]["logits_bit_exact"])
            exact_probabilities += int(
                record[prediction_name]["probability_bit_exact"]
            )
            exact_masks += int(record[prediction_name]["mask_bit_exact"])
        aggregate[prediction_name] = {
            "logits_bit_exact_images": exact_logits,
            "logits_global_max_abs_difference": logits_max,
            "probability_bit_exact_images": exact_probabilities,
            "probability_global_mean_abs_difference": (
                probability_abs_sum / probability_values
            ),
            "probability_global_max_abs_difference": probability_max,
            "mask_bit_exact_images": exact_masks,
            "mask_disagreement_pixels": mask_disagreement,
            "mask_disagreement_rate": float(mask_disagreement / mask_values),
        }

    step_differences = {
        image_id: abs(candidate[image_id].step_norm - canonical[image_id].step_norm)
        for image_id in selected_ids
    }
    for image_id, difference in step_differences.items():
        per_image[image_id]["optimizer_step_norm_absolute_difference"] = float(
            difference
        )
    aggregate["optimizer_step_norm"] = {
        "global_max_absolute_difference": float(max(step_differences.values())),
        "global_mean_absolute_difference": float(
            sum(step_differences.values()) / len(step_differences)
        ),
    }
    return {"aggregate": aggregate, "per_image": per_image}


def derive_train_tolerance_candidate(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a labelled, non-gating envelope derived only from train smoke runs."""

    if not comparisons:
        raise ValueError("at least one comparison is required")
    post = [comparison["aggregate"]["tent_post"] for comparison in comparisons]
    steps = [
        comparison["aggregate"]["optimizer_step_norm"]
        for comparison in comparisons
    ]
    observed = {
        "logits_max_abs": max(
            float(value["logits_global_max_abs_difference"]) for value in post
        ),
        "probability_mean_abs": max(
            float(value["probability_global_mean_abs_difference"]) for value in post
        ),
        "probability_max_abs": max(
            float(value["probability_global_max_abs_difference"]) for value in post
        ),
        "mask_disagreement_rate": max(
            float(value["mask_disagreement_rate"]) for value in post
        ),
        "step_norm_abs": max(
            float(value["global_max_absolute_difference"]) for value in steps
        ),
    }
    floors = {
        "logits_max_abs": 1e-7,
        "probability_mean_abs": 1e-10,
        "probability_max_abs": 1e-8,
        "mask_disagreement_rate": 0.0,
        "step_norm_abs": 1e-10,
    }
    candidate = {
        key: max(floors[key], value * 1.5) for key, value in observed.items()
    }
    return {
        "status": "train_derived_candidate_only",
        "formal_acceptance_gate": False,
        "scientific_result_frozen": False,
        "selection_or_tuning_performed": False,
        "derivation": "1.5x_max_observed_across_reverse_and_configured_repeats",
        "observed_envelope": observed,
        "candidate_envelope": candidate,
    }


def _save_canonical_predictions(
    *,
    protocol_root: Path,
    canonical: Mapping[str, EpisodeArrays],
    samples: Sequence[SourcePilotSample],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for index, sample in enumerate(samples):
        episode = canonical[sample.image_id]
        prediction_records: dict[str, Any] = {}
        for prediction_name in PREDICTION_NAMES:
            arrays = episode.predictions[prediction_name]
            stem = source_runner.safe_artifact_stem(sample.image_id)
            probability_relative = (
                Path("predictions")
                / prediction_name
                / "probability_maps_256"
                / f"{stem}.npy"
            )
            mask_relative = (
                Path("predictions")
                / prediction_name
                / "prediction_masks_256"
                / f"{stem}.png"
            )
            _atomic_save_npy(protocol_root / probability_relative, arrays.probability)
            _atomic_save_png(protocol_root / mask_relative, arrays.mask)
            prediction_records[prediction_name] = {
                "probability_map": str(probability_relative),
                "probability_map_sha256": source_runner.sha256_file(
                    protocol_root / probability_relative
                ),
                "probability_raw_sha256": _raw_array_sha256(arrays.probability),
                "probability_dtype": str(arrays.probability.dtype),
                "probability_shape": list(arrays.probability.shape),
                "probability_min": float(arrays.probability.min()),
                "probability_max": float(arrays.probability.max()),
                "prediction_mask": str(mask_relative),
                "prediction_mask_sha256": source_runner.sha256_file(
                    protocol_root / mask_relative
                ),
                "foreground_pixels": int(np.count_nonzero(arrays.mask)),
            }
            counts[f"{prediction_name}_probability_maps"] = (
                counts.get(f"{prediction_name}_probability_maps", 0) + 1
            )
            counts[f"{prediction_name}_prediction_masks"] = (
                counts.get(f"{prediction_name}_prediction_masks", 0) + 1
            )
        records.append(
            {
                "index": index,
                "image_id": sample.image_id,
                "metadata": dict(sample.metadata),
                "input_raw_sha256": sample.input_raw_sha256,
                "predictions": prediction_records,
                "canonical_optimizer_step_norm": episode.step_norm,
                "batch_tent_pre_frozen_adabn_probability_bit_exact": None,
                "source_pre_frozen_adabn_source_probability_bit_exact": None,
                "source_stats_tent_pre_source_pre_bit_exact": None,
                "run_comparisons": {},
            }
        )
        diagnostics.append(
            {
                "index": index,
                "image_id": sample.image_id,
                "method_diagnostics": dict(episode.diagnostics),
                **dict(episode.state),
            }
        )
    return records, diagnostics, counts


def _load_frozen_probability(
    *, root: Path, image_id: str, source: bool
) -> np.ndarray:
    stem = source_runner.safe_artifact_stem(image_id)
    directory = "source_probability_maps_256" if source else "probability_maps_256"
    path = root / "IRSTD-1K" / "gaussian_noise_S3" / directory / f"{stem}.npy"
    value = np.load(path, allow_pickle=False)
    if value.shape != (256, 256) or value.dtype != np.float32:
        raise RuntimeError(
            f"frozen AdaBN probability contract drift for {image_id}: "
            f"{value.shape}/{value.dtype}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError(f"frozen AdaBN probability is non-finite: {image_id}")
    return np.asarray(value)


def _build_protocol_runner(
    *,
    source_config: Mapping[str, Any],
    device: torch.device,
    bn_protocol: str,
    optimizer_name: str,
    learning_rate: float,
    entropy_eps: float,
) -> tuple[
    IRSTDModelAdapter,
    BinaryTentMethod,
    EpisodicStateManager,
    BinaryTentEpisodicRunner,
    dict[str, Any],
]:
    model = source_runner.build_nsfpn_model()
    checkpoint = _project_path(source_config["source"]["checkpoint"])
    checkpoint_wrapper = source_runner.load_trusted_checkpoint(model, checkpoint)
    model.to(device)
    adapter = IRSTDModelAdapter(
        model, warm_flag=bool(source_config["source"]["warm_flag"])
    )
    adapter.set_source_eval_mode()
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        bn_protocol=bn_protocol,
        entropy_eps=entropy_eps,
        cuda_backward_determinism_policy=CUDA_BACKWARD_TEMPORARILY_DISABLE,
    )
    state = EpisodicStateManager(model, optimizer=method.optimizer)
    runner = BinaryTentEpisodicRunner(adapter, state)
    return adapter, method, state, runner, {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source_runner.sha256_file(checkpoint),
        "checkpoint_wrapper": checkpoint_wrapper,
    }


def execute_protocol_smoke(
    *,
    config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    paths: SmokePaths,
    samples: Sequence[SourcePilotSample],
    targets: Mapping[str, np.ndarray],
    device: torch.device,
    bn_protocol: str,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"protocol destination already exists: {destination}")
    destination.mkdir(parents=True)
    method_config = config["method"]
    optimizer_config = method_config["optimizer"]
    execution = config["execution"]
    evaluation = config["evaluation"]
    threshold = float(evaluation["fixed_probability_threshold"])
    selected_ids = tuple(sample.image_id for sample in samples)

    adapter, method, state, runner, checkpoint = _build_protocol_runner(
        source_config=source_config,
        device=device,
        bn_protocol=bn_protocol,
        optimizer_name=str(optimizer_config["name"]),
        learning_rate=float(optimizer_config["learning_rate"]),
        entropy_eps=float(method_config["entropy_eps"]),
    )
    bn_count = sum(isinstance(module, nn.BatchNorm2d) for module in adapter.model.modules())
    _require_equal(
        bn_count, int(execution["expected_batchnorm2d_modules"]), "BatchNorm2d count"
    )
    adaptable, adaptable_names = adapter.collect_adaptable_params()
    _require_equal(
        len(adaptable), int(execution["expected_trainable_tensors"]), "BN affine tensors"
    )
    _require_equal(
        sum(value.numel() for value in adaptable),
        int(execution["expected_trainable_scalars"]),
        "BN affine scalars",
    )
    optimizer_ids = {
        id(parameter)
        for group in method.optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in adaptable}:
        raise RuntimeError("optimizer is not bound exactly to all BN affine parameters")

    canonical = execute_episode_pass(
        samples=samples,
        order=selected_ids,
        runner=runner,
        method=method,
        device=device,
        threshold=threshold,
        expected_trainable_tensors=len(adaptable),
        expected_trainable_scalars=sum(value.numel() for value in adaptable),
    )
    state.assert_source_state()
    records, diagnostics, output_counts = _save_canonical_predictions(
        protocol_root=destination, canonical=canonical, samples=samples
    )
    record_by_id = {record["image_id"]: record for record in records}

    source_reference_exact = 0
    batch_adabn_exact = 0
    source_tent_exact = 0
    for image_id in selected_ids:
        episode = canonical[image_id]
        frozen_source = _load_frozen_probability(
            root=paths.frozen_adabn_root, image_id=image_id, source=True
        )
        source_exact = bool(
            np.array_equal(
                episode.predictions["source_pre"].probability, frozen_source
            )
        )
        if not source_exact:
            difference = float(
                np.max(
                    np.abs(
                        episode.predictions["source_pre"].probability - frozen_source
                    )
                )
            )
            raise RuntimeError(
                f"Source-pre diverged from frozen train AdaBN Source at {image_id}; "
                f"max_abs={difference}"
            )
        source_reference_exact += 1
        record_by_id[image_id][
            "source_pre_frozen_adabn_source_probability_bit_exact"
        ] = source_exact

        if bn_protocol == BN_PROTOCOL_BATCH_STATS:
            frozen_adabn = _load_frozen_probability(
                root=paths.frozen_adabn_root, image_id=image_id, source=False
            )
            tent_exact = bool(
                np.array_equal(
                    episode.predictions["tent_pre"].probability, frozen_adabn
                )
            )
            if not tent_exact:
                difference = float(
                    np.max(
                        np.abs(
                            episode.predictions["tent_pre"].probability
                            - frozen_adabn
                        )
                    )
                )
                raise RuntimeError(
                    f"batch-stat TENT-pre diverged from frozen AdaBN at {image_id}; "
                    f"max_abs={difference}"
                )
            batch_adabn_exact += 1
            record_by_id[image_id][
                "batch_tent_pre_frozen_adabn_probability_bit_exact"
            ] = tent_exact
        else:
            tent_exact = bool(
                np.array_equal(
                    episode.predictions["tent_pre"].logits,
                    episode.predictions["source_pre"].logits,
                )
            )
            if not tent_exact:
                raise RuntimeError(
                    f"source-stat TENT-pre is not Source bit-exact at {image_id}"
                )
            source_tent_exact += 1
            record_by_id[image_id][
                "source_stats_tent_pre_source_pre_bit_exact"
            ] = tent_exact

    evaluation_protocol = IRSTDEvaluationProtocol(
        fixed_probability_threshold=threshold,
        connectivity=int(evaluation["connectivity"]),
        max_centroid_distance=float(evaluation["max_centroid_distance"]),
        min_component_area=int(evaluation["min_component_area"]),
    )
    canonical_metrics = evaluate_pass_metrics(
        canonical,
        selected_ids=selected_ids,
        targets=targets,
        protocol=evaluation_protocol,
    )

    pass_specs: list[tuple[str, tuple[str, ...]]] = [
        ("reverse_1", tuple(reversed(selected_ids)))
    ]
    pass_specs.extend(
        (f"canonical_repeat_{index}", selected_ids)
        for index in range(1, int(execution["canonical_repeat_runs"]) + 1)
    )
    comparisons: list[dict[str, Any]] = []
    comparison_by_name: dict[str, Any] = {}
    source_tent_pre_exact_all = True
    for pass_name, order in pass_specs:
        candidate = execute_episode_pass(
            samples=samples,
            order=order,
            runner=runner,
            method=method,
            device=device,
            threshold=threshold,
            expected_trainable_tensors=len(adaptable),
            expected_trainable_scalars=sum(value.numel() for value in adaptable),
        )
        state.assert_source_state()
        comparison = compare_episode_passes(
            canonical, candidate, selected_ids=selected_ids
        )
        candidate_metrics = evaluate_pass_metrics(
            candidate,
            selected_ids=selected_ids,
            targets=targets,
            protocol=evaluation_protocol,
        )
        comparison["pass_name"] = pass_name
        comparison["order"] = (
            "exact_reverse" if pass_name == "reverse_1" else "canonical"
        )
        comparison["metrics"] = candidate_metrics
        comparison["metric_differences_from_canonical"] = metric_differences(
            candidate_metrics, canonical_metrics
        )
        for prediction_name in ("source_pre", "tent_pre"):
            summary = comparison["aggregate"][prediction_name]
            exact = all(
                (
                    int(summary["logits_bit_exact_images"]) == len(selected_ids),
                    int(summary["probability_bit_exact_images"])
                    == len(selected_ids),
                    int(summary["mask_bit_exact_images"]) == len(selected_ids),
                )
            )
            source_tent_pre_exact_all = source_tent_pre_exact_all and exact
            if not exact:
                raise RuntimeError(
                    f"{prediction_name} drifted in {pass_name} ({bn_protocol})"
                )
        for image_id in selected_ids:
            record_by_id[image_id]["run_comparisons"][pass_name] = comparison[
                "per_image"
            ][image_id]
        comparisons.append(comparison)
        comparison_by_name[pass_name] = {
            key: value for key, value in comparison.items() if key != "per_image"
        }
        del candidate

    tolerance_candidate = derive_train_tolerance_candidate(comparisons)
    source_runner.write_jsonl_atomic(destination / "per_image.jsonl", records)
    source_runner.write_jsonl_atomic(
        destination / "adaptation_diagnostics.jsonl", diagnostics
    )
    expected_outputs = len(selected_ids)
    all_outputs_saved = len(output_counts) == 6 and all(
        count == expected_outputs for count in output_counts.values()
    )
    checks = {
        **protocol_scope_gate_checks(config),
        "method_received_no_label": True,
        "expected_batchnorm2d_modules": bn_count
        == int(execution["expected_batchnorm2d_modules"]),
        "optimizer_bound_exactly_to_bn_affine": True,
        "canonical_source_matches_frozen_source": source_reference_exact
        == len(selected_ids),
        "batch_tent_pre_matches_frozen_adabn": (
            batch_adabn_exact == len(selected_ids)
            if bn_protocol == BN_PROTOCOL_BATCH_STATS
            else True
        ),
        "source_stats_tent_pre_matches_source": (
            source_tent_exact == len(selected_ids)
            if bn_protocol == BN_PROTOCOL_SOURCE_STATS
            else True
        ),
        "source_and_tent_pre_repeat_order_bit_exact": source_tent_pre_exact_all,
        "tent_post_bit_exact_not_required": True,
        "every_episode_reset_exact_source": True,
        "all_canonical_predictions_saved": all_outputs_saved,
    }
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise RuntimeError(f"Binary TENT protocol smoke gates failed: {failed}")
    metrics = {
        "schema_version": 1,
        "scope": "source_domain_implementation_smoke",
        "paper_result": False,
        "scientific_result_frozen": False,
        "tuning_allowed": False,
        "method": "Binary Episodic TENT",
        "bn_protocol": bn_protocol,
        "optimizer": {
            **dict(optimizer_config),
            "selection_statement": "provisional_only; no selection performed",
        },
        "selected_train_images": len(selected_ids),
        "execution_passes": ["canonical", *(name for name, _ in pass_specs)],
        "canonical_metrics_outer_train_diagnostic_only": canonical_metrics,
        "comparisons": comparison_by_name,
        "train_derived_tolerance_candidate": tolerance_candidate,
        "saved_output_counts": output_counts,
        "checks": checks,
        "forbidden_claims": {
            "formal_test_performance": "not_computed",
            "optimizer_or_learning_rate_selected": False,
            "post_update_bit_exact": "not_asserted",
        },
    }
    source_runner.write_json_atomic(destination / "metrics.json", metrics)
    del runner, state, method, adapter
    return {
        "metrics": metrics,
        "checkpoint": checkpoint,
        "adaptable_parameter_names": adaptable_names,
    }


def _artifact_files(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        if relative in {"artifact_manifest.json", "COMPLETE.json"}:
            continue
        if path.is_symlink():
            raise ValueError(f"artifact cannot contain a symlink: {path}")
        files[relative] = {
            "sha256": source_runner.sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return files


def finalize_and_publish(
    *,
    staging: Path,
    final_root: Path,
    protocol_id: str,
    protocol_sha256: str,
    checks: Mapping[str, Any],
) -> dict[str, Any]:
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    if failed:
        raise RuntimeError(f"refusing COMPLETE because hard gates failed: {failed}")
    required = {
        "run_config.yaml",
        "provenance.json",
        "aggregate_metrics.json",
        "batch_stats/metrics.json",
        "batch_stats/per_image.jsonl",
        "batch_stats/adaptation_diagnostics.jsonl",
        "source_stats/metrics.json",
        "source_stats/per_image.jsonl",
        "source_stats/adaptation_diagnostics.jsonl",
    }
    present = {
        str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()
    }
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"required Binary TENT smoke artifacts missing: {missing}")
    files = _artifact_files(staging)
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "scope": "source_domain_implementation_smoke",
        "paper_result": False,
        "scientific_result_frozen": False,
        "optimizer_selected": False,
        "files": files,
    }
    manifest_path = staging / "artifact_manifest.json"
    source_runner.write_json_atomic(manifest_path, manifest)
    completion = {
        "complete": True,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "scope": "source_domain_implementation_smoke",
        "paper_result": False,
        "scientific_result_frozen": False,
        "optimizer_selected": False,
        "all_required_hard_gates_passed": True,
        "artifact_manifest_sha256": source_runner.sha256_file(manifest_path),
        "aggregate_metrics_sha256": files["aggregate_metrics.json"]["sha256"],
    }
    source_runner.write_json_atomic(staging / "COMPLETE.json", completion)
    if final_root.exists():
        raise FileExistsError(f"output already exists: {final_root}")
    os.replace(staging, final_root)
    return completion


def _runtime_device_summary(device: torch.device) -> dict[str, Any]:
    driver_command = [
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            driver_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        driver_versions = sorted(
            {line.strip() for line in completed.stdout.splitlines() if line.strip()}
        )
        driver_audit: dict[str, Any] = {
            "query": driver_command,
            "status": "available",
            "versions": driver_versions,
        }
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        driver_audit = {
            "query": driver_command,
            "status": "unavailable",
            "error_type": type(error).__name__,
            "error": str(error),
            "versions": [],
        }

    extension = importlib.import_module("MultiScaleDeformableAttention")
    extension_raw_path = getattr(extension, "__file__", None)
    if not isinstance(extension_raw_path, str) or not extension_raw_path:
        raise RuntimeError(
            "loaded MultiScaleDeformableAttention has no auditable __file__"
        )
    extension_path = Path(extension_raw_path).resolve()
    if not extension_path.is_file():
        raise FileNotFoundError(
            f"loaded MultiScaleDeformableAttention file is missing: {extension_path}"
        )
    summary: dict[str, Any] = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver": driver_audit,
        "tf32": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "loaded_multiscale_deformable_attention": {
            "module": "MultiScaleDeformableAttention",
            "file": str(extension_path),
            "sha256": source_runner.sha256_file(extension_path),
            "bytes": extension_path.stat().st_size,
        },
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        summary.update(
            {
                "cuda_device_index": index,
                "cuda_device_name": properties.name,
                "cuda_compute_capability": [properties.major, properties.minor],
                "cuda_total_memory_bytes": int(properties.total_memory),
            }
        )
    return summary


def _warning_summary(captured: Sequence[warnings.WarningMessage]) -> dict[str, Any]:
    counts: dict[tuple[str, str], int] = {}
    for warning in captured:
        key = (warning.category.__name__, str(warning.message))
        counts[key] = counts.get(key, 0) + 1
    return {
        "captured_count": len(captured),
        "unique": [
            {"category": category, "message": message, "count": count}
            for (category, message), count in sorted(counts.items())
        ],
    }


def run_source_smoke(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config_path, config = load_smoke_config(args.config)
    paths = resolve_smoke_paths(
        config_path, config, output_override=getattr(args, "output_dir", None)
    )
    (
        source_config,
        pilot_paths,
        selected_ids,
        selected_files,
        source_contract,
        frozen_reference,
    ) = validate_frozen_source_reference(config, paths)

    validation = {
        "protocol_id": config["protocol_id"],
        "selected_train_images": len(selected_ids),
        "selected_ids_sha256": ordered_ids_sha256(selected_ids),
        "frozen_input_hash": config["frozen_source_input"][
            "input_tensor_sequence_sha256"
        ],
        "test_dataset_constructed": False,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "frozen_adabn_reference_verified": True,
    }
    if getattr(args, "validate_only", False):
        return {"validate_only": True, "validation": validation}

    if paths.output_root.exists():
        raise FileExistsError(f"Binary TENT smoke output already exists: {paths.output_root}")
    paths.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = paths.output_root.with_name(
        f".{paths.output_root.name}.build-{os.getpid()}"
    )
    if staging.exists():
        raise FileExistsError(f"Binary TENT smoke staging already exists: {staging}")

    samples, materialization = materialize_label_free_inputs(
        source_config, pilot_paths, selected_ids, selected_files
    )
    frozen_inputs = config["frozen_source_input"]
    _require_equal(
        materialization["input_tensor_sequence_sha256"],
        frozen_inputs["input_tensor_sequence_sha256"],
        "re-materialized input sequence",
    )
    _require_equal(
        materialization["source_train_mask_tensor_sequence_sha256"],
        frozen_inputs["mask_tensor_sequence_sha256"],
        "re-materialized target sequence",
    )
    _require_equal(materialization["method_received_mask"], False, "label firewall")
    targets, target_evaluator = load_outer_train_targets(
        selected_ids=selected_ids,
        selected_files=selected_files,
        image_size=256,
        expected_sequence_sha256=frozen_inputs["mask_tensor_sequence_sha256"],
    )

    staging.mkdir(parents=False)
    execution = config["execution"]
    protocol_results: dict[str, Any] = {}
    policy_record: dict[str, Any]
    captured_warnings: list[warnings.WarningMessage]
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        with binary_tent_cuda_policy(
            seed=int(source_config["condition"]["seed"]),
            workspace_config=str(execution["cublas_workspace_config"]),
        ) as policy_record:
            device = source_runner.resolve_device(args.device)
            if device.type != "cuda":
                raise RuntimeError(
                    "Binary TENT GPU repeatability smoke requires a CUDA device"
                )
            if device.type == "cuda" and policy_record[
                "cuda_initialized_before_workspace_config"
            ]:
                raise RuntimeError(
                    "CUDA was initialized before CUBLAS_WORKSPACE_CONFIG; run this "
                    "smoke in a fresh Python process"
                )
            if device.type == "cuda" and (
                not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
            ):
                raise RuntimeError(
                    "real NS-FPN Binary TENT forwards require strict "
                    "deterministic algorithms"
                )
            for bn_protocol in config["method"]["protocols"]:
                directory = PROTOCOL_DIRECTORY[bn_protocol]
                protocol_results[directory] = execute_protocol_smoke(
                    config=config,
                    source_config=source_config,
                    paths=paths,
                    samples=samples,
                    targets=targets,
                    device=device,
                    bn_protocol=bn_protocol,
                    destination=staging / directory,
                )
            runtime_device = _runtime_device_summary(device)
        captured_warnings = list(warning_records)

    policy_record["after"] = asdict(_capture_runtime_policy())
    policy_restored = runtime_policy_is_restored(policy_record["before"])
    if not policy_restored:
        raise RuntimeError("process-global deterministic/CuDNN/CUBLAS policy was not restored")

    all_comparisons = [
        comparison
        for result in protocol_results.values()
        for comparison in result["metrics"]["comparisons"].values()
    ]
    global_tolerance_candidate = derive_train_tolerance_candidate(all_comparisons)
    protocol_checks_pass = all(
        all(value is True for value in result["metrics"]["checks"].values())
        for result in protocol_results.values()
    )
    hard_checks = {
        "frozen_32_train_ids_and_input_hashes_verified": len(selected_ids) == 32
        and materialization["input_tensor_sequence_sha256"]
        == frozen_inputs["input_tensor_sequence_sha256"],
        "zero_test_pixel_opens": source_contract["test_images_opened"] == 0
        and source_contract["test_masks_opened"] == 0
        and target_evaluator["test_masks_opened"] == 0,
        "method_received_no_label": materialization["method_received_mask"] is False
        and target_evaluator["method_received_mask"] is False,
        "both_bn_protocols_executed": set(protocol_results)
        == {"batch_stats", "source_stats"},
        "all_protocol_episode_and_prediction_gates_passed": protocol_checks_pass,
        "provisional_optimizer_not_selected": config["method"]["optimizer"][
            "selected_by_this_smoke"
        ]
        is False,
        "scientific_result_not_frozen": config["scope"][
            "scientific_result_frozen"
        ]
        is False,
        "process_global_policy_restored": policy_restored,
    }
    failed = sorted(name for name, passed in hard_checks.items() if passed is not True)
    if failed:
        raise RuntimeError(f"Binary TENT aggregate hard gates failed: {failed}")

    effective_config = deepcopy(config)
    effective_config["runtime"] = {
        "smoke_config": str(paths.config),
        "smoke_config_sha256": source_runner.sha256_file(paths.config),
        "source_config": str(paths.adabn_config),
        "source_config_sha256": source_runner.sha256_file(paths.adabn_config),
        "device": runtime_device,
        "output_root": str(paths.output_root),
        "process_policy": policy_record,
    }
    _write_yaml_atomic(staging / "run_config.yaml", effective_config)
    provenance = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "scope": "source_domain_implementation_smoke",
        "paper_result": False,
        "scientific_result_frozen": False,
        "optimizer_selected": False,
        "config": str(paths.config),
        "config_sha256": source_runner.sha256_file(paths.config),
        "selected_ids_file": str(paths.selected_ids_file),
        "selected_ids_file_sha256": source_runner.sha256_file(
            paths.selected_ids_file
        ),
        "selected_ids": list(selected_ids),
        "selected_ids_sha256": ordered_ids_sha256(selected_ids),
        "materialized_inputs": materialization,
        "outer_train_target_evaluator": target_evaluator,
        "fixed_test_boundary": {
            "split_metadata_hash_verified": source_contract[
                "fixed_test_split_metadata_hash_verified"
            ],
            "selected_ids_absent_from_fixed_test": source_contract[
                "selected_ids_absent_from_fixed_test"
            ],
            "test_dataset_constructed": False,
            "test_images_opened": 0,
            "test_masks_opened": 0,
        },
        "frozen_adabn_reference": {
            "root": str(paths.frozen_adabn_root),
            "complete_sha256": source_runner.sha256_file(
                paths.frozen_adabn_root / "COMPLETE.json"
            ),
            "artifact_manifest_sha256": source_runner.sha256_file(
                paths.frozen_adabn_root / "artifact_manifest.json"
            ),
            "provenance_sha256": source_runner.sha256_file(
                paths.frozen_adabn_root / "provenance.json"
            ),
            "referenced_probability_files_hash_verified": frozen_reference[
                "all_64_referenced_probability_files_hash_verified"
            ],
        },
        "checkpoint_by_protocol": {
            name: result["checkpoint"] for name, result in protocol_results.items()
        },
        "adaptable_parameter_names_by_protocol": {
            name: result["adaptable_parameter_names"]
            for name, result in protocol_results.items()
        },
        "runtime": {
            **runtime_device,
            "process_policy": policy_record,
            "warnings": _warning_summary(captured_warnings),
        },
        "repository": source_runner.repository_provenance(PROVENANCE_PATHS),
    }
    source_runner.write_json_atomic(staging / "provenance.json", provenance)
    aggregate = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "scope": "source_domain_implementation_smoke",
        "paper_result": False,
        "scientific_result_frozen": False,
        "optimizer_or_learning_rate_selected": False,
        "selected_train_images": len(selected_ids),
        "bn_protocols": list(config["method"]["protocols"]),
        "canonical_repeat_runs": int(execution["canonical_repeat_runs"]),
        "episodes_per_protocol": len(selected_ids)
        * (2 + int(execution["canonical_repeat_runs"])),
        "total_episodes": len(selected_ids)
        * (2 + int(execution["canonical_repeat_runs"]))
        * len(protocol_results),
        "protocol_metrics": {
            name: f"{name}/metrics.json" for name in protocol_results
        },
        "global_train_derived_tolerance_candidate": global_tolerance_candidate,
        "hard_checks": hard_checks,
        "runtime_seconds_before_publication": time.perf_counter() - started,
    }
    source_runner.write_json_atomic(staging / "aggregate_metrics.json", aggregate)
    completion = finalize_and_publish(
        staging=staging,
        final_root=paths.output_root,
        protocol_id=config["protocol_id"],
        protocol_sha256=source_runner.sha256_file(paths.config),
        checks=hard_checks,
    )
    return {
        "published_output_dir": str(paths.output_root),
        "aggregate": aggregate,
        "completion": completion,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_source_smoke(args)
    if result.get("validate_only"):
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print(f"Artifacts: {result['published_output_dir']}")
    print(
        "Binary TENT source implementation smoke passed: "
        f"{result['aggregate']['selected_train_images']} train images, "
        f"{result['aggregate']['total_episodes']} episodes; no tuning performed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "EXPECTED_PROTOCOL_ID",
    "RuntimePolicySnapshot",
    "SmokePaths",
    "audit_binary_tent_episode",
    "binary_tent_cuda_policy",
    "compare_episode_passes",
    "derive_train_tolerance_candidate",
    "evaluate_pass_metrics",
    "finalize_and_publish",
    "load_outer_train_targets",
    "load_smoke_config",
    "resolve_smoke_paths",
    "run_source_smoke",
    "runtime_policy_is_restored",
    "protocol_scope_gate_checks",
    "validate_frozen_source_reference",
]
