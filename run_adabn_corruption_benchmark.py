"""Run episodic AdaBN on the frozen 3 x 13 corruption benchmark.

Default execution runs all 39 dataset/condition shards sequentially and then
aggregates them. ``--dataset`` alone runs a complete formal 13-condition
dataset shard. ``--condition`` selects one or three full-split formal condition
shards; combining it with ``--dataset`` is the intended 39-process GPU launch.
Only ``--max-images``, ``--smoke``, or ``--output-dir`` makes a run non-formal.
``--aggregate-only`` verifies the 39 independently published condition shards,
rebuilds the three dataset summaries, and publishes the global ``COMPLETE``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Literal

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
import yaml

from dataio.corruption_cache import (
    CachedCorruptionDataset,
    TensorSequenceHasher,
    condition_key,
    ordered_ids_sha256,
    sha256_file,
    verify_cache_artifact,
)
from metrics.irstd_metrics import (
    IRSTDEvaluationProtocol,
    UnifiedResearchEvaluator,
    probabilities_from_logits,
)
from metrics.official_metric_adapter import OfficialMetricAdapter
from run_adabn_source_pilot import _episode_invariants
import run_source_corruption_benchmark as source_benchmark
import test_fixed_split_source as fixed_source
import test_source as source_runner
from tta.adabn import AdaBNMethod
from tta.adabn_fast_runner import AdaBNFastEpisodeResult, AdaBNFastRunner
from tta.episodic_runner import EpisodicRunner
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager, StateFingerprint


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "adabn_batch_stats_fixed_splits_v1.yaml"
EXPECTED_PROTOCOL_ID = "cr-sitta-adabn-batch-stats-fixed-splits-v1"
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
CONDITIONS: tuple[tuple[str, int], ...] = (
    ("clean", 0),
    ("gaussian_noise", 1),
    ("gaussian_noise", 3),
    ("gaussian_noise", 5),
    ("gaussian_blur", 1),
    ("gaussian_blur", 3),
    ("gaussian_blur", 5),
    ("low_contrast", 1),
    ("low_contrast", 3),
    ("low_contrast", 5),
    ("stripe_noise", 1),
    ("stripe_noise", 3),
    ("stripe_noise", 5),
)
CONDITION_BY_KEY = {condition_key(*value): value for value in CONDITIONS}
PROVENANCE_PATHS = (
    ".conda/lib/python3.10/site-packages/MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so",
    "SFS_MSDeformAttn/ops/__init__.py",
    "SFS_MSDeformAttn/ops/functions/__init__.py",
    "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
    "SFS_MSDeformAttn/ops/modules/__init__.py",
    "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
    "configs/adabn_batch_stats_fixed_splits_v1.yaml",
    "configs/adabn_source_pilot_smoke_v1.yaml",
    "configs/source_corruption_benchmark_fixed_splits.yaml",
    "configs/source_corruption_cache_generation_v1.yaml",
    "corruptions/__init__.py",
    "corruptions/corruption_protocol.py",
    "corruptions/infrared_corruptions.py",
    "dataio/__init__.py",
    "dataio/corruption_cache.py",
    "dataio/research_dataset.py",
    "environment.cr-sitta.yml",
    "environment.linux-64.explicit.txt",
    "metrics/__init__.py",
    "metrics/connected_components.py",
    "metrics/irstd_metrics.py",
    "metrics/official_metric_adapter.py",
    "metrics/target_matching.py",
    "model/MSHNet_NSFPN.py",
    "model/NS_FPN.py",
    "model/diff_cross_attns.py",
    "requirements.lock.txt",
    "run_adabn_corruption_benchmark.py",
    "run_adabn_source_pilot.py",
    "run_source_corruption_benchmark.py",
    "scripts/run_adabn_batch_stats_v1_parallel.sh",
    "results/relocations.json",
    "test_fixed_split_source.py",
    "test_source.py",
    "THIRD_PARTY.md",
    "THIRD_PARTY_COMMITS.txt",
    "third_party/tent/LICENSE",
    "third_party/tent/norm.py",
    "third_party/tent_reference.json",
    "tta/__init__.py",
    "tta/adabn.py",
    "tta/adabn_fast_runner.py",
    "tta/episodic_runner.py",
    "tta/model_adapter.py",
    "tta/state_manager.py",
    "utils/metric.py",
)


@dataclass(frozen=True)
class RunSelection:
    mode: Literal[
        "formal_all",
        "formal_dataset_shard",
        "formal_condition_shard",
        "smoke",
        "aggregate_only",
    ]
    datasets: tuple[str, ...]
    conditions: tuple[tuple[str, int], ...]
    max_images: int | None
    formal_artifact: bool
    output_root: Path


@dataclass(frozen=True)
class DatasetContext:
    dataset_name: str
    config_contract: Mapping[str, Any]
    source_protocol: Mapping[str, Any]
    source_dataset_contract: Mapping[str, Any]
    cache_dir: Path
    cache_manifest: Mapping[str, Any]
    cache_audit: Mapping[str, Any]
    image_ids: tuple[str, ...]
    checkpoint: Path
    checkpoint_payload: Mapping[str, Any]
    checkpoint_summary: Mapping[str, Any]
    source_root: Path
    source_benchmark: Mapping[str, Any]
    source_artifact_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class SourceConditionReference:
    condition_root: Path
    probabilities: np.ndarray
    records: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    probability_shard_sha256: str


def positive_integer(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dataset", choices=DATASETS, default=None)
    parser.add_argument("--condition", choices=tuple(CONDITION_BY_KEY), default=None)
    parser.add_argument("--max-images", type=positive_integer, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def _project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return dict(loaded)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            loaded = json.loads(line)
            if not isinstance(loaded, Mapping):
                raise TypeError(f"expected JSON object at {path}:{line_number}")
            records.append(dict(loaded))
    return records


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_protocol(path: str | Path = DEFAULT_PROTOCOL) -> tuple[Path, dict[str, Any]]:
    protocol_path = Path(path).expanduser().resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"AdaBN benchmark protocol does not exist: {protocol_path}")
    loaded = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("AdaBN benchmark protocol must be a YAML mapping")
    protocol = dict(loaded)
    _require_equal(protocol.get("schema_version"), 1, "schema_version")
    _require_equal(protocol.get("protocol_id"), EXPECTED_PROTOCOL_ID, "protocol_id")
    _require_equal(tuple(protocol.get("datasets_order", ())), DATASETS, "datasets_order")
    ordered = tuple(
        (str(corruption), int(severity))
        for corruption, severity in protocol.get("ordered_conditions", ())
    )
    _require_equal(ordered, CONDITIONS, "ordered_conditions")
    _require_equal(
        protocol["scope"].get("formal_conditions_per_dataset"),
        13,
        "formal_conditions_per_dataset",
    )
    _require_equal(
        protocol["scope"].get("formal_global_condition_count"),
        39,
        "formal_global_condition_count",
    )
    for key, expected in (
        ("name", "AdaBN"),
        ("learnable_update", False),
        ("backward", False),
        ("optimizer", None),
        ("optimizer_steps_per_image", 0),
    ):
        _require_equal(protocol["method"].get(key), expected, f"method.{key}")
    execution = protocol["execution"]
    for key, expected in (
        ("seed", 42),
        ("batch_size", 1),
        ("num_workers", 0),
        ("image_size", 256),
        ("expected_batchnorm2d_modules", 53),
        ("formal_worker_visible_cuda_device_count", 1),
        ("episodic_reset", "every_image"),
    ):
        _require_equal(execution.get(key), expected, f"execution.{key}")
    evaluation = protocol["evaluation"]
    for key, expected in (
        ("probability_transform", "sigmoid_once_stable_numpy"),
        ("fixed_probability_threshold", 0.5),
        ("threshold_rule", "strict_greater_than"),
        ("foreground_connectivity_2d", 8),
    ):
        _require_equal(evaluation.get(key), expected, f"evaluation.{key}")
    _require_equal(
        protocol["outputs"].get("formal_root"),
        "results/adabn/adabn_batch_stats_v1",
        "outputs.formal_root",
    )
    outputs = protocol["outputs"]
    for key, expected in (
        ("refuse_overwrite", True),
        ("atomic_condition_publish", True),
        ("atomic_dataset_publish", True),
        ("atomic_global_publish", True),
        ("save_all_adabn_probability_maps", True),
        ("save_all_adabn_binary_masks", True),
        ("formal_condition_sentinel", "CONDITION_COMPLETE.json"),
        ("formal_dataset_sentinel", "DATASET_COMPLETE.json"),
        ("formal_global_sentinel", "COMPLETE.json"),
        ("smoke_condition_sentinel", "SMOKE_CONDITION_COMPLETE.json"),
        ("smoke_dataset_sentinel", "SMOKE_DATASET_COMPLETE.json"),
        ("smoke_global_sentinel", "SMOKE_COMPLETE.json"),
    ):
        _require_equal(outputs.get(key), expected, f"outputs.{key}")
    storage = outputs.get("probability_storage")
    if not isinstance(storage, Mapping):
        raise TypeError("outputs.probability_storage must be a mapping")
    for key, expected in (
        ("format", "one_numpy_npy_shard_per_condition"),
        ("filename", "probabilities_256.npy"),
        ("dtype", "little_endian_float32"),
        ("shape", ["N", 256, 256]),
        ("per_image_lookup", ["probability_shard", "probability_shard_index"]),
    ):
        _require_equal(storage.get(key), expected, f"probability_storage.{key}")
    state_audit = execution.get("state_audit")
    if not isinstance(state_audit, Mapping):
        raise TypeError("execution.state_audit must be a mapping")
    for key, expected in (
        ("per_image_exact_gpu_parameter_and_buffer_equality", True),
        ("failure_action", "full_reset_then_abort"),
        ("diagnostics_record_audit_kind", True),
    ):
        _require_equal(state_audit.get(key), expected, f"state_audit.{key}")
    cadence = state_audit.get("full_state_sha256_cadence")
    _require_equal(
        cadence,
        {"first_image": True, "last_image": True, "every_n_images": 64},
        "state_audit.full_state_sha256_cadence",
    )
    pilot = protocol.get("implementation_pilot_lineage")
    if not isinstance(pilot, Mapping):
        raise TypeError("implementation_pilot_lineage must be a mapping")
    pilot_config = _project_path(pilot["config"])
    pilot_root = _project_path(pilot["result"])
    for actual, expected, label in (
        (sha256_file(pilot_config), pilot["config_sha256"], "Pilot config"),
        (
            sha256_file(pilot_root / "COMPLETE.json"),
            pilot["complete_sha256"],
            "Pilot COMPLETE",
        ),
        (
            sha256_file(pilot_root / "artifact_manifest.json"),
            pilot["artifact_manifest_sha256"],
            "Pilot artifact manifest",
        ),
    ):
        _require_equal(actual, expected, f"{label} SHA256")
    pilot_complete = _load_json(pilot_root / "COMPLETE.json")
    _require_equal(
        pilot_complete.get("all_required_gates_passed"),
        True,
        "Pilot all_required_gates_passed",
    )
    relocation = protocol.get("historical_relocations")
    if not isinstance(relocation, Mapping):
        raise TypeError("historical_relocations must be a mapping")
    _require_equal(
        sha256_file(_project_path(relocation["path"])),
        relocation["sha256"],
        "historical relocations SHA256",
    )
    datasets = protocol.get("datasets")
    if not isinstance(datasets, Mapping) or tuple(datasets) != DATASETS:
        raise ValueError("protocol datasets must be the exact ordered three-dataset mapping")

    source_contract = protocol["source_benchmark"]
    source_protocol_path = _project_path(source_contract["protocol"])
    _require_equal(
        sha256_file(source_protocol_path),
        source_contract["protocol_sha256"],
        "Source benchmark protocol SHA256",
    )
    source_loaded = yaml.safe_load(source_protocol_path.read_text(encoding="utf-8"))
    if not isinstance(source_loaded, Mapping):
        raise TypeError("Source benchmark protocol must be a mapping")
    _require_equal(
        source_loaded.get("protocol_id"),
        source_contract["protocol_id"],
        "Source benchmark protocol_id",
    )
    _require_equal(source_benchmark._conditions(source_loaded), CONDITIONS, "Source conditions")
    source_evaluation = source_loaded["evaluation"]
    for key in (
        "fixed_probability_threshold",
        "threshold_rule",
        "froc_probability_thresholds",
        "foreground_connectivity_2d",
        "target_matching",
    ):
        _require_equal(evaluation[key], source_evaluation[key], f"Source evaluation {key}")
    for dataset_name in DATASETS:
        entry = datasets[dataset_name]
        source_entry = source_loaded["datasets"][dataset_name]
        for left, right, label in (
            (entry["test_images"], source_entry["test_images"], "test_images"),
            (
                entry["ordered_test_ids_sha256"],
                source_entry["ordered_test_ids_sha256"],
                "ordered_test_ids_sha256",
            ),
            (entry["checkpoint_sha256"], source_entry["checkpoint_sha256"], "checkpoint"),
            (entry["checkpoint_epoch"], source_entry["checkpoint_epoch"], "checkpoint_epoch"),
            (
                entry["cache_manifest_sha256"],
                source_entry["materialized_cache"]["manifest_sha256"],
                "cache_manifest",
            ),
            (
                entry["cache_content_sha256"],
                source_entry["materialized_cache"]["content_sha256"],
                "cache_content",
            ),
            (entry["cache_bytes"], source_entry["materialized_cache"]["bytes"], "cache_bytes"),
        ):
            _require_equal(left, right, f"{dataset_name} {label}")
    return protocol_path, protocol


def resolve_selection(args: argparse.Namespace, protocol: Mapping[str, Any]) -> RunSelection:
    if args.aggregate_only:
        conflicts = (
            args.dataset is not None,
            args.condition is not None,
            args.max_images is not None,
            bool(args.smoke),
            args.output_dir is not None,
        )
        if any(conflicts):
            raise ValueError("--aggregate-only cannot be combined with run filters")
        return RunSelection(
            mode="aggregate_only",
            datasets=DATASETS,
            conditions=CONDITIONS,
            max_images=None,
            formal_artifact=True,
            output_root=_project_path(protocol["outputs"]["formal_root"]),
        )

    selected_datasets = (args.dataset,) if args.dataset else DATASETS
    selected_conditions = (
        (CONDITION_BY_KEY[args.condition],) if args.condition else CONDITIONS
    )
    smoke = bool(args.smoke or args.max_images is not None or args.output_dir is not None)
    if smoke:
        dataset_tag = args.dataset or "all_datasets"
        condition_tag = args.condition or "all_conditions"
        count_tag = f"n{args.max_images}" if args.max_images else "full"
        default = (
            _project_path(protocol["outputs"]["smoke_root"])
            / f"{dataset_tag}__{condition_tag}__{count_tag}"
        )
        output_root = (
            args.output_dir.expanduser().resolve() if args.output_dir else default
        )
        return RunSelection(
            mode="smoke",
            datasets=selected_datasets,
            conditions=selected_conditions,
            max_images=args.max_images,
            formal_artifact=False,
            output_root=output_root,
        )
    if args.dataset is not None:
        if args.condition is not None:
            return RunSelection(
                mode="formal_condition_shard",
                datasets=selected_datasets,
                conditions=selected_conditions,
                max_images=None,
                formal_artifact=True,
                output_root=_project_path(protocol["outputs"]["formal_root"]),
            )
        return RunSelection(
            mode="formal_dataset_shard",
            datasets=selected_datasets,
            conditions=CONDITIONS,
            max_images=None,
            formal_artifact=True,
            output_root=_project_path(protocol["outputs"]["formal_root"]),
        )
    if args.condition is not None:
        return RunSelection(
            mode="formal_condition_shard",
            datasets=DATASETS,
            conditions=selected_conditions,
            max_images=None,
            formal_artifact=True,
            output_root=_project_path(protocol["outputs"]["formal_root"]),
        )
    return RunSelection(
        mode="formal_all",
        datasets=DATASETS,
        conditions=CONDITIONS,
        max_images=None,
        formal_artifact=True,
        output_root=_project_path(protocol["outputs"]["formal_root"]),
    )


def probability_float32_from_logits(logits: Tensor) -> np.ndarray:
    """Match the frozen Source benchmark's NumPy float64 sigmoid exactly."""

    probability = probabilities_from_logits(logits)[0, 0].astype(np.float32, copy=False)
    if probability.ndim != 2 or not np.isfinite(probability).all():
        raise ValueError("single-image logits produced an invalid probability map")
    return probability


def _raw_array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _fingerprint_dict(value: StateFingerprint) -> dict[str, Any]:
    return asdict(value)


def _png_sha256(array: np.ndarray) -> str:
    stream = BytesIO()
    Image.fromarray(array, mode="L").save(stream, format="PNG")
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _repository_contract() -> dict[str, Any]:
    provenance = source_runner.repository_provenance(PROVENANCE_PATHS)
    provenance["code_bundle_sha256"] = _canonical_json_sha256(
        provenance["file_sha256"]
    )
    return provenance


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    """Capture the actual interpreter and accelerator used by one shard."""

    try:
        driver_query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        driver_versions = sorted(
            {line.strip() for line in driver_query.stdout.splitlines() if line.strip()}
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver_versions = []

    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_runtime_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "numpy_version": np.__version__,
        "pillow_version": getattr(Image, "__version__", None),
        "nvidia_driver_versions_visible_to_nvidia_smi": driver_versions,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resolved_device": str(device),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        result["cuda"] = {
            "visible_device_count": torch.cuda.device_count(),
            "logical_device_index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    return result


def _source_protocol(protocol: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _project_path(protocol["source_benchmark"]["protocol"])
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("Source benchmark protocol must be a mapping")
    _require_equal(
        sha256_file(path),
        protocol["source_benchmark"]["protocol_sha256"],
        "Source protocol SHA256",
    )
    return path, dict(loaded)


def _verify_source_benchmark_root(
    dataset_name: str,
    contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = _project_path(contract["source_result"])
    benchmark_path = root / "benchmark.json"
    manifest_path = root / "artifact_manifest.json"
    complete_path = root / "COMPLETE.json"
    for path in (benchmark_path, manifest_path, complete_path):
        if not path.is_file():
            raise FileNotFoundError(f"Source benchmark artifact is incomplete: {path}")
    _require_equal(
        sha256_file(benchmark_path),
        contract["source_benchmark_sha256"],
        f"{dataset_name} Source benchmark SHA256",
    )
    _require_equal(
        sha256_file(manifest_path),
        contract["source_artifact_manifest_sha256"],
        f"{dataset_name} Source artifact manifest SHA256",
    )
    complete = _load_json(complete_path)
    for actual, expected, label in (
        (complete.get("complete"), True, "complete"),
        (complete.get("dataset"), dataset_name, "dataset"),
        (
            complete.get("protocol_sha256"),
            protocol["source_benchmark"]["protocol_sha256"],
            "protocol SHA256",
        ),
        (
            complete.get("benchmark_sha256"),
            contract["source_benchmark_sha256"],
            "benchmark SHA256",
        ),
        (
            complete.get("artifact_manifest_sha256"),
            contract["source_artifact_manifest_sha256"],
            "manifest SHA256",
        ),
        (
            complete.get("cache_content_sha256"),
            contract["cache_content_sha256"],
            "cache content SHA256",
        ),
    ):
        _require_equal(actual, expected, f"{dataset_name} Source completion {label}")
    benchmark = _load_json(benchmark_path)
    _require_equal(benchmark.get("method"), "Source", "Source benchmark method")
    _require_equal(benchmark.get("dataset"), dataset_name, "Source benchmark dataset")
    _require_equal(benchmark.get("condition_count"), 13, "Source condition count")
    _require_equal(
        benchmark.get("evaluated_images_per_condition"),
        int(contract["test_images"]),
        "Source evaluated image count",
    )
    actual_conditions = tuple(
        (record["corruption"], int(record["severity"]))
        for record in benchmark["conditions"]
    )
    _require_equal(actual_conditions, CONDITIONS, "Source benchmark condition order")
    manifest = _load_json(manifest_path)
    _require_equal(manifest.get("dataset"), dataset_name, "Source manifest dataset")
    _require_equal(
        manifest.get("cache_content_sha256"),
        contract["cache_content_sha256"],
        "Source manifest cache content",
    )
    if not isinstance(manifest.get("files"), Mapping):
        raise ValueError("Source artifact manifest contains no files mapping")
    return root, benchmark, manifest


def prepare_dataset_context(
    protocol: Mapping[str, Any],
    dataset_name: str,
    *,
    verify_all_cache_file_hashes: bool,
) -> DatasetContext:
    if dataset_name not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset_name}")
    contract = dict(protocol["datasets"][dataset_name])
    source_protocol_path, source_protocol = _source_protocol(protocol)
    source_dataset = dict(source_protocol["datasets"][dataset_name])
    _require_equal(
        sha256_file(_project_path(protocol["materialized_cache"]["generation_protocol"])),
        protocol["materialized_cache"]["generation_protocol_sha256"],
        "cache generation protocol SHA256",
    )

    cache_dir = _project_path(contract["cache_dir"])
    cache_manifest, cache_audit = verify_cache_artifact(
        cache_dir,
        expected_protocol_sha256=protocol["materialized_cache"][
            "generation_protocol_sha256"
        ],
        verify_file_hashes=verify_all_cache_file_hashes,
    )
    for actual, expected, label in (
        (cache_audit["manifest_sha256"], contract["cache_manifest_sha256"], "manifest"),
        (cache_manifest["cache_content_sha256"], contract["cache_content_sha256"], "content"),
        (cache_manifest["dataset"], dataset_name, "dataset"),
        (cache_manifest["ordered_ids_sha256"], contract["ordered_test_ids_sha256"], "IDs"),
        (len(cache_manifest["image_ids"]), int(contract["test_images"]), "image count"),
    ):
        _require_equal(actual, expected, f"{dataset_name} cache {label}")
    if verify_all_cache_file_hashes:
        _require_equal(
            int(cache_audit["verified_total_bytes"]),
            int(contract["cache_bytes"]),
            f"{dataset_name} cache bytes",
        )
    cached_conditions = tuple(
        (record["corruption"], int(record["severity"]))
        for record in cache_manifest["conditions"]
    )
    _require_equal(cached_conditions, CONDITIONS, f"{dataset_name} cached conditions")
    image_ids = tuple(str(value) for value in cache_manifest["image_ids"])
    _require_equal(
        ordered_ids_sha256(image_ids),
        contract["ordered_test_ids_sha256"],
        f"{dataset_name} ordered IDs SHA256",
    )

    checkpoint = _project_path(contract["checkpoint"])
    _require_equal(
        sha256_file(checkpoint),
        contract["checkpoint_sha256"],
        f"{dataset_name} checkpoint SHA256",
    )
    checkpoint_payload, checkpoint_summary = source_benchmark._validate_checkpoint(
        checkpoint,
        dataset_name=dataset_name,
        dataset_contract=source_dataset,
        protocol=source_protocol,
    )
    source_root, source_result, source_manifest = _verify_source_benchmark_root(
        dataset_name, contract, protocol
    )
    _require_equal(
        source_result["checkpoint_sha256"],
        contract["checkpoint_sha256"],
        f"{dataset_name} Source checkpoint",
    )
    _require_equal(
        source_result["cache_content_sha256"],
        contract["cache_content_sha256"],
        f"{dataset_name} Source cache",
    )
    return DatasetContext(
        dataset_name=dataset_name,
        config_contract=contract,
        source_protocol=source_protocol,
        source_dataset_contract=source_dataset,
        cache_dir=cache_dir,
        cache_manifest=cache_manifest,
        cache_audit=cache_audit,
        image_ids=image_ids,
        checkpoint=checkpoint,
        checkpoint_payload=checkpoint_payload,
        checkpoint_summary=checkpoint_summary,
        source_root=source_root,
        source_benchmark=source_result,
        source_artifact_manifest=source_manifest,
    )


def load_source_condition_reference(
    context: DatasetContext,
    corruption: str,
    severity: int,
) -> SourceConditionReference:
    key = condition_key(corruption, severity)
    condition_root = context.source_root / "conditions" / key
    files = context.source_artifact_manifest["files"]
    relative_paths = {
        "probability": f"conditions/{key}/probabilities_256.npy",
        "records": f"conditions/{key}/per_image.jsonl",
        "metrics": f"conditions/{key}/metrics.json",
    }
    for label, relative in relative_paths.items():
        record = files.get(relative)
        if not isinstance(record, Mapping):
            raise ValueError(f"Source manifest omits {relative}")
        path = context.source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Source condition file is missing: {path}")
        _require_equal(path.stat().st_size, int(record["bytes"]), f"Source {label} bytes")
        _require_equal(sha256_file(path), record["sha256"], f"Source {label} SHA256")

    probability_path = context.source_root / relative_paths["probability"]
    probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(context.image_ids), 256, 256)
    if probabilities.shape != expected_shape or probabilities.dtype.str != "<f4":
        raise ValueError(
            f"Source probability shard contract drift: {probabilities.shape}/{probabilities.dtype}"
        )
    records = tuple(_load_jsonl(context.source_root / relative_paths["records"]))
    _require_equal(len(records), len(context.image_ids), "Source per-image count")
    probability_hash = sha256_file(probability_path)
    for index, (image_id, record) in enumerate(zip(context.image_ids, records)):
        if int(record["index"]) != index or record["image_id"] != image_id:
            raise ValueError(f"Source per-image ID order drift at {index}")
        _require_equal(
            record["probability_shard_sha256"],
            probability_hash,
            f"Source probability shard lineage at {image_id}",
        )
    metrics = _load_json(context.source_root / relative_paths["metrics"])
    _require_equal(metrics.get("condition_key"), key, "Source metrics condition key")
    _require_equal(metrics.get("evaluated_images"), len(context.image_ids), "Source metrics count")
    return SourceConditionReference(
        condition_root=condition_root,
        probabilities=probabilities,
        records=records,
        metrics=metrics,
        probability_shard_sha256=probability_hash,
    )


def _run_episode(
    runner: EpisodicRunner | AdaBNFastRunner,
    *,
    image: Tensor,
    metadata: Mapping[str, Any],
    force_full_audit: bool = False,
) -> Any:
    """Single call boundary kept intentionally small for the audited fast runner."""

    if isinstance(runner, AdaBNFastRunner):
        return runner.run_one_image(
            image=image,
            metadata=metadata,
            force_full_audit=force_full_audit,
        )
    return runner.run_one_image(
        image=image,
        metadata=metadata,
        method=AdaBNMethod(),
    )


def _audit_episode_result(
    result: Any,
    expected_batchnorm_count: int,
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    """Normalize evidence without inventing fingerprints skipped by fast mode."""

    if isinstance(result, AdaBNFastEpisodeResult):
        failed = sorted(name for name, passed in result.checks.items() if passed is not True)
        if failed:
            raise RuntimeError(f"AdaBN fast episode check failures: {failed}")
        for check in result.exact_checks.values():
            if check.batchnorm_running_mean_buffers != expected_batchnorm_count:
                raise RuntimeError("AdaBN fast BatchNorm count drifted")
        if result.full_audit_performed:
            if (
                result.post_full_fingerprint is None
                or result.reset_full_fingerprint is None
                or result.post_full_state_differences != ("runtime",)
                or result.reset_full_fingerprint.full_sha256
                != result.source_state_sha256
            ):
                raise RuntimeError("AdaBN fast full SHA audit failed")
        invariants = {
            "input_unchanged": result.input_unchanged,
            "optimizer_steps_zero": True,
            "learnable_update_false": True,
            "expected_batchnorm_count": True,
            "all_parameters_and_buffers_exact": True,
            "all_bn_running_buffers_exact": True,
            "optimizer_absent": True,
            "gradients_unchanged": True,
            "topology_unchanged": True,
            "runtime_reset_exact": True,
            "reset_matches_source": True,
        }
        state_record = {
            "source_state_sha256": result.source_state_sha256,
            "state_after_prepare_sha256": None,
            "state_after_post_sha256": (
                result.post_full_fingerprint.full_sha256
                if result.post_full_fingerprint is not None
                else None
            ),
            "reset_state_sha256": (
                result.reset_full_fingerprint.full_sha256
                if result.reset_full_fingerprint is not None
                else None
            ),
        }
        diagnostic = {
            "method": result.method,
            "decision": "batch_statistics_only",
            "optimizer_steps": 0,
            "method_diagnostics": {"learnable_update": False},
            "state_changes_after_prepare": None,
            "state_changes_after_adapt": None,
            "state_changes_after_post": (
                list(result.post_full_state_differences)
                if result.post_full_state_differences is not None
                else None
            ),
            "source_fingerprint": {
                "full_sha256": result.source_state_sha256,
                "audit_kind": "construction_full_sha256",
            },
            "state_after_prepare_fingerprint": None,
            "state_after_adapt_fingerprint": None,
            "state_after_post_fingerprint": (
                _fingerprint_dict(result.post_full_fingerprint)
                if result.post_full_fingerprint is not None
                else None
            ),
            "reset_fingerprint": (
                _fingerprint_dict(result.reset_full_fingerprint)
                if result.reset_full_fingerprint is not None
                else None
            ),
            "state_audit": {
                "kind": result.check_kind,
                "full_sha_performed": result.full_audit_performed,
                "full_sha_reason": result.full_audit_reason,
                "exact_parameter_and_buffer_equality_performed": True,
                "exact_checks": {
                    stage: asdict(check)
                    for stage, check in result.exact_checks.items()
                },
            },
        }
        return invariants, state_record, diagnostic

    invariants = _episode_invariants(result, expected_batchnorm_count)
    state_record = {
        "source_state_sha256": result.source_fingerprint.full_sha256,
        "state_after_prepare_sha256": (
            result.state_after_prepare_fingerprint.full_sha256
        ),
        "state_after_post_sha256": result.state_after_post_fingerprint.full_sha256,
        "reset_state_sha256": result.reset_fingerprint.full_sha256,
    }
    diagnostic = {
        "method": result.method,
        "decision": result.outcome.decision,
        "optimizer_steps": result.outcome.optimizer_steps,
        "method_diagnostics": dict(result.outcome.diagnostics),
        "state_changes_after_prepare": list(result.state_changes_after_prepare),
        "state_changes_after_adapt": list(result.state_changes_after_adapt),
        "state_changes_after_post": list(result.state_changes_after_post),
        "source_fingerprint": _fingerprint_dict(result.source_fingerprint),
        "state_after_prepare_fingerprint": _fingerprint_dict(
            result.state_after_prepare_fingerprint
        ),
        "state_after_adapt_fingerprint": _fingerprint_dict(
            result.state_after_adapt_fingerprint
        ),
        "state_after_post_fingerprint": _fingerprint_dict(
            result.state_after_post_fingerprint
        ),
        "reset_fingerprint": _fingerprint_dict(result.reset_fingerprint),
        "state_audit": {
            "kind": "generic_full_state_sha256",
            "full_sha_performed": True,
            "full_sha_reason": "every_episode",
            "exact_parameter_and_buffer_equality_performed": True,
        },
    }
    return invariants, state_record, diagnostic


def _order_isolation_sentinel(
    *,
    dataset: Dataset,
    runner: EpisodicRunner,
    adapter: IRSTDModelAdapter,
    expected_batchnorm_count: int,
    available_images: int,
) -> dict[str, Any]:
    """Prove A->B and B->A have identical per-image pre/post logits."""

    if available_images < 2:
        return {
            "performed": False,
            "reason": "fewer_than_two_images_in_non_formal_smoke",
            "contributes_to_metrics": False,
            "passed": None,
        }
    loader = DataLoader(
        Subset(dataset, (0, 1)),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    batches = tuple(loader)
    device = next(adapter.model.parameters()).device
    observations: dict[str, dict[str, Tensor]] = {}
    execution: list[dict[str, Any]] = []
    for order_name, order in (("A_then_B", (0, 1)), ("B_then_A", (1, 0))):
        for slot in order:
            batch = batches[slot]
            metadata = source_runner.metadata_from_batch(batch)
            result = _run_episode(
                runner,
                image=batch["image"].detach().cpu().to(device, non_blocking=False),
                metadata=metadata,
            )
            invariants = _episode_invariants(result, expected_batchnorm_count)
            image_id = str(metadata["image_id"])
            observation = {
                "pre": result.logits_pre.detach().cpu().clone(),
                "post": result.logits_post.detach().cpu().clone(),
            }
            prior = observations.get(image_id)
            if prior is None:
                observations[image_id] = observation
            elif not (
                torch.equal(prior["pre"], observation["pre"])
                and torch.equal(prior["post"], observation["post"])
            ):
                raise RuntimeError(
                    f"episodic A/B order isolation failed for {image_id}"
                )
            execution.append(
                {
                    "order": order_name,
                    "image_id": image_id,
                    "source_pre_logits_raw_sha256": _raw_array_sha256(
                        observation["pre"].numpy()
                    ),
                    "adabn_post_logits_raw_sha256": _raw_array_sha256(
                        observation["post"].numpy()
                    ),
                    "reset_matches_source": invariants["reset_matches_source"],
                }
            )
    return {
        "performed": True,
        "implementation": "generic_EpisodicRunner",
        "orders": ["A_then_B", "B_then_A"],
        "images": [str(source_runner.metadata_from_batch(batch)["image_id"]) for batch in batches],
        "pre_logits_bit_exact_across_orders": True,
        "post_logits_bit_exact_across_orders": True,
        "reset_bit_exact_after_every_episode": True,
        "contributes_to_metrics": False,
        "execution": execution,
        "passed": True,
    }


def execute_condition(
    *,
    context: DatasetContext,
    corruption: str,
    severity: int,
    dataset: Dataset,
    source_reference: SourceConditionReference,
    adapter: IRSTDModelAdapter,
    runner: EpisodicRunner | AdaBNFastRunner,
    sentinel_runner: EpisodicRunner | None = None,
    precomputed_order_sentinel: Mapping[str, Any] | None = None,
    evaluation_protocol: IRSTDEvaluationProtocol,
    destination: Path,
    expected_batchnorm_count: int,
    max_images: int | None,
    formal_artifact: bool,
) -> dict[str, Any]:
    """Run one condition; masks stay outside the label-free method boundary."""

    if destination.exists():
        raise FileExistsError(f"condition destination already exists: {destination}")
    destination.mkdir(parents=True)
    key = condition_key(corruption, severity)
    full_count = len(dataset)
    count = full_count if max_images is None else min(max_images, full_count)
    if formal_artifact and count != full_count:
        raise ValueError("formal condition must evaluate the complete fixed test split")
    indices = range(count)
    loader = DataLoader(
        dataset if count == full_count else Subset(dataset, indices),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=next(adapter.model.parameters()).device.type == "cuda",
    )
    source_official_evaluator = OfficialMetricAdapter(image_size=256)
    source_unified_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
    adabn_official_evaluator = OfficialMetricAdapter(image_size=256)
    adabn_unified_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
    input_hasher = TensorSequenceHasher()
    target_hasher = TensorSequenceHasher()
    probability_hasher = TensorSequenceHasher()
    device = next(adapter.model.parameters()).device
    cache_conditions = {
        str(record["key"]): record for record in context.cache_manifest["conditions"]
    }
    cache_condition = cache_conditions[key]
    source_probability_exact_count = 0
    source_mask_array_exact_count = 0
    source_mask_file_hash_exact_count = 0
    changed_probability_images = 0
    global_max_abs_change = 0.0
    absolute_change_sum = 0.0
    absolute_change_values = 0
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    full_sha_image_indexes: list[int] = []
    probability_path = destination / "probabilities_256.npy"
    probability_partial, probability_map = source_benchmark._atomic_probability_memmap(
        probability_path, count
    )
    if precomputed_order_sentinel is not None:
        order_sentinel = dict(precomputed_order_sentinel)
    else:
        if isinstance(runner, AdaBNFastRunner) and sentinel_runner is None:
            raise ValueError(
                "fast execution requires a precomputed generic A/B sentinel"
            )
        order_sentinel = _order_isolation_sentinel(
            dataset=dataset,
            runner=sentinel_runner or runner,
            adapter=adapter,
            expected_batchnorm_count=expected_batchnorm_count,
            available_images=count,
        )
    if formal_artifact and order_sentinel["passed"] is not True:
        raise RuntimeError("formal condition requires a passing A/B order sentinel")

    for index, batch in enumerate(loader):
        metadata = source_runner.metadata_from_batch(batch)
        image_id = str(metadata["image_id"])
        _require_equal(image_id, context.image_ids[index], f"{key} cached ID {index}")
        image_cpu = batch["image"].detach().cpu()
        target_cpu = batch["mask"].detach().cpu()
        input_hasher.update(image_id, image_cpu[0])
        target_hasher.update(image_id, target_cpu[0])

        result = _run_episode(
            runner,
            image=image_cpu.to(device, non_blocking=False),
            metadata=metadata,
            force_full_audit=index == 0 or index == count - 1,
        )
        invariants, state_record, diagnostic_state = _audit_episode_result(
            result, expected_batchnorm_count
        )
        if diagnostic_state["state_audit"]["full_sha_performed"]:
            full_sha_image_indexes.append(index)
        source_probability = probability_float32_from_logits(result.logits_pre)
        adabn_probability = probability_float32_from_logits(result.logits_post)

        source_record = source_reference.records[index]
        frozen_probability = np.asarray(source_reference.probabilities[index])
        probability_exact = bool(np.array_equal(source_probability, frozen_probability))
        raw_probability_exact = (
            _raw_array_sha256(source_probability)
            == source_record["probability_tensor_raw_sha256"]
        )
        if not probability_exact or not raw_probability_exact:
            maximum = float(np.max(np.abs(source_probability - frozen_probability)))
            raise RuntimeError(
                f"Source-pre probability diverged from frozen Source at {image_id}; "
                f"max_abs={maximum}"
            )
        source_probability_exact_count += 1
        source_mask = np.where(source_probability > 0.5, 255, 0).astype(np.uint8)
        frozen_mask_path = source_reference.condition_root / source_record["prediction_mask"]
        _require_equal(
            sha256_file(frozen_mask_path),
            source_record["prediction_mask_sha256"],
            f"frozen Source mask SHA256 at {image_id}",
        )
        with Image.open(frozen_mask_path) as handle:
            frozen_mask = np.asarray(handle.convert("L"))
        mask_array_exact = bool(np.array_equal(source_mask, frozen_mask))
        mask_file_hash_exact = (
            _png_sha256(source_mask) == source_record["prediction_mask_sha256"]
        )
        if not mask_array_exact or not mask_file_hash_exact:
            raise RuntimeError(f"Source-pre binary mask diverged at {image_id}")
        source_mask_array_exact_count += 1
        source_mask_file_hash_exact_count += 1

        source_runner._update_official_evaluator(
            source_official_evaluator, result.logits_pre, target_cpu
        )
        source_unified_evaluator.update_logits(result.logits_pre, target_cpu)
        source_runner._update_official_evaluator(
            adabn_official_evaluator, result.logits_post, target_cpu
        )
        adabn_unified_evaluator.update_logits(result.logits_post, target_cpu)
        probability_hasher.update(image_id, adabn_probability)
        probability_map[index] = adabn_probability
        adabn_mask = np.where(adabn_probability > 0.5, 255, 0).astype(np.uint8)
        relative_mask = Path("prediction_masks_256") / fixed_source._relative_prediction_path(
            image_id, ".png"
        )
        fixed_source._write_png_atomic(destination / relative_mask, adabn_mask)

        difference = np.abs(adabn_probability - source_probability)
        probability_changed = not np.array_equal(adabn_probability, source_probability)
        changed_probability_images += int(probability_changed)
        max_abs_change = float(difference.max())
        mean_abs_change = float(difference.mean(dtype=np.float64))
        global_max_abs_change = max(global_max_abs_change, max_abs_change)
        absolute_change_sum += float(difference.sum(dtype=np.float64))
        absolute_change_values += int(difference.size)
        target = target_cpu.numpy()[0, 0]
        record = {
            "index": index,
            **dict(result.metadata),
            "cache_content_sha256": context.cache_manifest["cache_content_sha256"],
            "cache_condition_tensor_sha256": cache_condition[
                "tensor_sequence_sha256"
            ],
            "source_reference_probability_shard_sha256": (
                source_reference.probability_shard_sha256
            ),
            "source_pre_logits_raw_sha256": _raw_array_sha256(
                result.logits_pre.numpy()
            ),
            "source_pre_probability_raw_sha256": _raw_array_sha256(
                source_probability
            ),
            "source_pre_probability_reference_bit_exact": probability_exact,
            "source_pre_binary_mask_reference_bit_exact": mask_array_exact,
            "source_pre_binary_mask_file_hash_exact": mask_file_hash_exact,
            "probability_shard": "probabilities_256.npy",
            "probability_shard_index": index,
            "probability_tensor_raw_sha256": _raw_array_sha256(adabn_probability),
            "probability_min": float(adabn_probability.min()),
            "probability_max": float(adabn_probability.max()),
            "prediction_mask": str(relative_mask),
            "prediction_mask_sha256": sha256_file(destination / relative_mask),
            "source_adabn_probability_bit_exact": not probability_changed,
            "source_adabn_max_abs_probability_change": max_abs_change,
            "source_adabn_mean_abs_probability_change": mean_abs_change,
            "source_adabn_binary_mask_changed_pixels": int(
                np.count_nonzero(source_mask != adabn_mask)
            ),
            "pixel_metrics_at_probability_gt_0_5": fixed_source._pixel_record(
                adabn_probability, target
            ),
            **state_record,
        }
        diagnostic = {
            "index": index,
            "image_id": image_id,
            **diagnostic_state,
            "invariants": invariants,
            "source_pre_parity": {
                "stored_source_logits_available": False,
                "probability_array_bit_exact": probability_exact,
                "binary_mask_array_bit_exact": mask_array_exact,
                "binary_mask_file_hash_exact": mask_file_hash_exact,
                "numeric_tolerance_used": False,
            },
        }
        records.append(record)
        diagnostics.append(diagnostic)

    probability_map.flush()
    del probability_map
    os.replace(probability_partial, probability_path)
    evaluated = len(records)
    _require_equal(evaluated, count, f"{key} evaluated count")
    full_split = evaluated == full_count
    input_hash_matches = (
        input_hasher.hexdigest() == cache_condition["tensor_sequence_sha256"]
        if full_split
        else None
    )
    target_hash_matches = (
        target_hasher.hexdigest()
        == context.cache_manifest["targets"]["tensor_sequence_sha256"]
        if full_split
        else None
    )
    if formal_artifact and (input_hash_matches is not True or target_hash_matches is not True):
        raise RuntimeError(f"formal cache tensor hash mismatch for {key}")
    source_official = source_official_evaluator.compute()
    source_unified = source_unified_evaluator.compute()
    source_summary = source_benchmark._condition_summary(
        source_official, source_unified
    )
    source_official_exact = (
        source_benchmark._canonical_json_equal(
            source_official.to_dict(), source_reference.metrics["official"]
        )
        if full_split
        else None
    )
    source_unified_exact = (
        source_benchmark._canonical_json_equal(
            source_unified.to_dict(), source_reference.metrics["unified"]
        )
        if full_split
        else None
    )
    source_summary_exact = (
        source_benchmark._canonical_json_equal(
            source_summary, source_reference.metrics["summary"]
        )
        if full_split
        else None
    )
    source_parity = {
        "stored_source_logits_available": False,
        "logits_proxy_gate": "exact_float32_probability_plus_binary_mask",
        "comparison_rule": "exact_after_json_container_canonicalization",
        "aggregate_gate_required": full_split,
        "probability_arrays_compared": evaluated,
        "all_probability_arrays_bit_exact": source_probability_exact_count == evaluated,
        "all_binary_mask_arrays_bit_exact": source_mask_array_exact_count == evaluated,
        "all_binary_mask_file_hashes_exact": (
            source_mask_file_hash_exact_count == evaluated
        ),
        "official_metrics_exact": source_official_exact,
        "unified_metrics_exact": source_unified_exact,
        "summary_exact": source_summary_exact,
        "numeric_tolerance_used": False,
    }
    parity_checks = [
        source_parity["all_probability_arrays_bit_exact"],
        source_parity["all_binary_mask_arrays_bit_exact"],
        source_parity["all_binary_mask_file_hashes_exact"],
    ]
    if full_split:
        parity_checks.extend(
            (
                source_parity["official_metrics_exact"],
                source_parity["unified_metrics_exact"],
                source_parity["summary_exact"],
            )
        )
    if not all(parity_checks):
        raise RuntimeError(f"Source-pre parity gate failed for {key}")

    probability_file_hash = sha256_file(probability_path)
    saved_probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    probability_storage_exact = (
        saved_probabilities.shape == (evaluated, 256, 256)
        and saved_probabilities.dtype.str == "<f4"
    )
    del saved_probabilities
    for record in records:
        record["probability_shard_sha256"] = probability_file_hash
    mask_files = tuple((destination / "prediction_masks_256").rglob("*.png"))
    all_outputs_saved = probability_storage_exact and len(mask_files) == evaluated
    if not all_outputs_saved:
        raise RuntimeError(
            f"incomplete AdaBN outputs for {key}: "
            f"probability_shard={probability_storage_exact}, masks={len(mask_files)}/{evaluated}"
        )
    official = adabn_official_evaluator.compute()
    unified = adabn_unified_evaluator.compute()
    summary = source_benchmark._condition_summary(official, unified)
    fast_execution = isinstance(runner, AdaBNFastRunner)
    expected_full_sha_indexes = sorted(
        {
            0,
            evaluated - 1,
            *(
                index
                for index in range(evaluated)
                if (index + 1) % 64 == 0
            ),
        }
    )
    full_sha_schedule_exact = (
        full_sha_image_indexes == expected_full_sha_indexes
        if fast_execution
        else None
    )
    if formal_artifact and fast_execution and full_sha_schedule_exact is not True:
        raise RuntimeError(
            f"full SHA cadence drifted for {key}: "
            f"{full_sha_image_indexes} != {expected_full_sha_indexes}"
        )
    source_runner.write_jsonl_atomic(destination / "per_image.jsonl", records)
    source_runner.write_jsonl_atomic(
        destination / "adaptation_diagnostics.jsonl", diagnostics
    )
    source_runner.write_json_atomic(
        destination / "order_isolation_sentinel.json", order_sentinel
    )
    metrics = {
        "schema_version": 1,
        "method": "AdaBN",
        "formal_artifact": formal_artifact,
        "paper_result": formal_artifact,
        "dataset": context.dataset_name,
        "condition_key": key,
        "corruption": corruption,
        "severity": severity,
        "evaluated_images": evaluated,
        "full_fixed_test_split": full_split,
        "summary": summary,
        "official": official.to_dict(),
        "unified": unified.to_dict(),
        "source_summary": source_summary,
        "deltas_from_source": {
            field: float(summary[field]) - float(source_summary[field])
            for field in summary
        },
        "delta_role": "report_only_never_used_for_selection_or_tuning",
        "behavioral_diagnostics": {
            "source_adabn_probability_changed_images": changed_probability_images,
            "source_adabn_probability_unchanged_images": evaluated
            - changed_probability_images,
            "source_adabn_global_max_abs_probability_change": global_max_abs_change,
            "source_adabn_global_mean_abs_probability_change": (
                absolute_change_sum / absolute_change_values
            ),
        },
        "source_pre_parity": source_parity,
        "order_isolation_sentinel": order_sentinel,
        "state_audit_schedule": {
            "execution_kind": (
                "AdaBNFastRunner" if fast_execution else "generic_EpisodicRunner"
            ),
            "policy": "first_last_and_every_64_images",
            "image_indexes_zero_based": full_sha_image_indexes,
            "count": len(full_sha_image_indexes),
            "expected_image_indexes_zero_based": expected_full_sha_indexes,
            "exact": full_sha_schedule_exact,
            "per_image_exact_resident_parameter_and_buffer_checks": fast_execution,
        },
        "probability_shard": "probabilities_256.npy",
        "probability_shard_sha256": probability_file_hash,
        "cache_lineage": {
            "cache_dir": str(context.cache_dir),
            "cache_manifest_sha256": context.cache_audit["manifest_sha256"],
            "cache_content_sha256": context.cache_manifest["cache_content_sha256"],
            "condition_tensor_sequence_sha256": cache_condition[
                "tensor_sequence_sha256"
            ],
            "target_tensor_sequence_sha256": context.cache_manifest["targets"][
                "tensor_sequence_sha256"
            ],
            "ordered_ids_sha256": context.cache_manifest["ordered_ids_sha256"],
        },
        "adabn_probability_tensor_sequence_sha256": probability_hasher.hexdigest(),
        "checks": {
            "method_label_firewall_passed": True,
            "input_hash_matches_cache": input_hash_matches,
            "target_hash_matches_cache": target_hash_matches,
            "ordered_ids_match_cache": True,
            "source_pre_parity_passed": True,
            "all_parameters_and_buffers_unchanged_before_reset": True,
            "all_bn_running_buffers_unchanged": True,
            "only_runtime_state_changed": True,
            "no_gradients_or_optimizer": True,
            "per_image_reset_matches_source": True,
            "order_isolation_sentinel_passed": order_sentinel["passed"],
            "full_sha_schedule_exact": full_sha_schedule_exact,
            "all_probability_maps_saved": probability_storage_exact,
            "all_prediction_masks_saved": len(mask_files) == evaluated,
        },
    }
    source_runner.write_json_atomic(destination / "metrics.json", metrics)
    return metrics


def _artifact_files(
    root: Path,
    *,
    excluded: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    excluded_set = set(excluded)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        if relative in excluded_set:
            continue
        if path.is_symlink():
            raise ValueError(f"artifact cannot contain a symlink: {path}")
        files[relative] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return files


def _verify_files_mapping(
    root: Path,
    files: Mapping[str, Any],
    *,
    allowed_unlisted: Sequence[str],
) -> None:
    for relative, expected in files.items():
        if not isinstance(expected, Mapping):
            raise TypeError(f"invalid artifact record for {relative}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"artifact file missing or symlinked: {path}")
        _require_equal(path.stat().st_size, int(expected["bytes"]), f"{relative} bytes")
        _require_equal(sha256_file(path), expected["sha256"], f"{relative} SHA256")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_paths = set(files) | set(allowed_unlisted)
    _require_equal(actual, expected_paths, f"{root} exact artifact file set")


def _condition_destination(
    output_root: Path,
    dataset_name: str,
    corruption: str,
    severity: int,
) -> Path:
    return output_root / dataset_name / "conditions" / condition_key(corruption, severity)


def publish_condition(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    context: DatasetContext,
    corruption: str,
    severity: int,
    device_spec: str,
    output_root: Path,
    formal_artifact: bool,
    max_images: int | None = None,
    repository_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run, seal, and atomically publish one dataset/condition shard."""

    key = condition_key(corruption, severity)
    if (corruption, severity) not in CONDITIONS:
        raise ValueError(f"condition is outside the frozen protocol: {key}")
    if formal_artifact and max_images is not None:
        raise ValueError("a formal condition shard cannot use --max-images")
    final = _condition_destination(output_root, context.dataset_name, corruption, severity)
    if final.exists():
        raise FileExistsError(f"condition output already exists: {final}")
    if formal_artifact:
        if (output_root / "COMPLETE.json").exists():
            raise FileExistsError("global benchmark is already complete")
        if (output_root / context.dataset_name / "DATASET_COMPLETE.json").exists():
            raise FileExistsError(f"dataset {context.dataset_name} is already complete")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.with_name(f".{final.name}.build-{os.getpid()}-{time.time_ns()}")
    if staging.exists():
        raise FileExistsError(f"condition staging path exists: {staging}")

    started = time.perf_counter()
    if formal_artifact and str(device_spec).startswith("cuda"):
        expected_visible = int(
            protocol["execution"]["formal_worker_visible_cuda_device_count"]
        )
        _require_equal(
            torch.cuda.device_count(),
            expected_visible,
            "formal worker visible CUDA device count",
        )
    source_runner.seed_everything(int(protocol["execution"]["seed"]))
    device = source_runner.resolve_device(device_spec)
    model = source_runner.build_nsfpn_model()
    checkpoint_wrapper = source_runner.load_trusted_checkpoint(model, context.checkpoint)
    model.to(device)
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    state = EpisodicStateManager(model, optimizer=None)
    generic_sentinel_runner = EpisodicRunner(adapter, state)
    batchnorm_count = sum(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    _require_equal(
        batchnorm_count,
        int(protocol["execution"]["expected_batchnorm2d_modules"]),
        "real NS-FPN BatchNorm2d count",
    )
    if any(module.training for module in model.modules()):
        raise RuntimeError("Source model must be entirely in eval mode before AdaBN")
    invalid_bn = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        and (
            not module.track_running_stats
            or module.running_mean is None
            or module.running_var is None
            or module.num_batches_tracked is None
        )
    ]
    if invalid_bn:
        raise RuntimeError(f"invalid Source BatchNorm buffers: {invalid_bn[:10]}")
    dataset = CachedCorruptionDataset(
        context.cache_dir,
        corruption=corruption,
        severity=severity,
        manifest=context.cache_manifest,
    )
    sentinel_available = len(dataset) if max_images is None else min(max_images, len(dataset))
    order_sentinel = _order_isolation_sentinel(
        dataset=dataset,
        runner=generic_sentinel_runner,
        adapter=adapter,
        expected_batchnorm_count=batchnorm_count,
        available_images=sentinel_available,
    )
    state.assert_source_state()
    # Construct only after the generic sentinel: StateManager's full reset may
    # restore registered buffers, so the fast runner must snapshot identities last.
    fast_runner = AdaBNFastRunner(
        adapter,
        state,
        full_audit_cadence=int(
            protocol["execution"]["state_audit"]["full_state_sha256_cadence"][
                "every_n_images"
            ]
        ),
    )
    source_reference = load_source_condition_reference(context, corruption, severity)
    evaluation_protocol = source_benchmark._evaluation_protocol(context.source_protocol)
    metrics = execute_condition(
        context=context,
        corruption=corruption,
        severity=severity,
        dataset=dataset,
        source_reference=source_reference,
        adapter=adapter,
        runner=fast_runner,
        precomputed_order_sentinel=order_sentinel,
        evaluation_protocol=evaluation_protocol,
        destination=staging,
        expected_batchnorm_count=batchnorm_count,
        max_images=max_images,
        formal_artifact=formal_artifact,
    )
    state.assert_source_state()
    if fast_runner.aborted:
        raise RuntimeError("fast AdaBN runner aborted")
    _require_equal(fast_runner.completed_episodes, metrics["evaluated_images"], "fast episodes")

    protocol_sha256 = sha256_file(protocol_path)
    repository = dict(repository_contract or _repository_contract())
    effective_config = deepcopy(dict(protocol))
    effective_config["runtime"] = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "dataset": context.dataset_name,
        "condition_key": key,
        "device": str(device),
        "formal_artifact": formal_artifact,
        "max_images": max_images,
        "condition_artifact_relative_path": str(final.relative_to(output_root)),
    }
    _write_yaml_atomic(staging / "condition_run_config.yaml", effective_config)
    provenance = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "formal_artifact": formal_artifact,
        "paper_result": formal_artifact,
        "dataset": context.dataset_name,
        "condition_key": key,
        "checkpoint": str(context.checkpoint),
        "checkpoint_sha256": context.config_contract["checkpoint_sha256"],
        "checkpoint_wrapper": checkpoint_wrapper,
        "checkpoint_metadata": context.checkpoint_summary,
        "checkpoint_test_selected_disclosure": context.source_protocol["source"][
            "checkpoint_disclosure"
        ],
        "cache": {
            "directory": str(context.cache_dir),
            "manifest_sha256": context.cache_audit["manifest_sha256"],
            "content_sha256": context.cache_manifest["cache_content_sha256"],
            "all_file_hashes_verified": bool(
                context.cache_audit.get("file_hashes_verified", False)
                or context.cache_audit.get("verify_file_hashes", False)
            ),
        },
        "source_benchmark": {
            "root": str(context.source_root),
            "benchmark_sha256": context.config_contract["source_benchmark_sha256"],
            "artifact_manifest_sha256": context.config_contract[
                "source_artifact_manifest_sha256"
            ],
            "condition_probability_shard_sha256": (
                source_reference.probability_shard_sha256
            ),
        },
        "implementation_pilot_lineage": dict(
            protocol["implementation_pilot_lineage"]
        ),
        "historical_relocations": dict(protocol["historical_relocations"]),
        "repository": repository,
        "runtime_environment": _runtime_environment(device),
        "method_label_boundary": {
            "method_received_image_and_metadata_only": True,
            "method_received_mask": False,
            "labels_used_only_by_external_evaluators_after_logits": True,
        },
        "runtime_seconds_before_publication": time.perf_counter() - started,
    }
    source_runner.write_json_atomic(staging / "condition_provenance.json", provenance)

    sentinel_name = (
        protocol["outputs"]["formal_condition_sentinel"]
        if formal_artifact
        else protocol["outputs"]["smoke_condition_sentinel"]
    )
    manifest_files = _artifact_files(
        staging,
        excluded=("artifact_manifest.json", sentinel_name),
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "repository_code_bundle_sha256": repository["code_bundle_sha256"],
        "formal_artifact": formal_artifact,
        "dataset": context.dataset_name,
        "condition_key": key,
        "cache_manifest_sha256": context.cache_audit["manifest_sha256"],
        "cache_content_sha256": context.cache_manifest["cache_content_sha256"],
        "checkpoint_sha256": context.config_contract["checkpoint_sha256"],
        "source_benchmark_sha256": context.config_contract["source_benchmark_sha256"],
        "files": manifest_files,
    }
    manifest_path = staging / "artifact_manifest.json"
    source_runner.write_json_atomic(manifest_path, manifest)
    completion = {
        "complete": True,
        "scope": "condition",
        "formal_artifact": formal_artifact,
        "paper_result": formal_artifact,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "repository_code_bundle_sha256": repository["code_bundle_sha256"],
        "dataset": context.dataset_name,
        "condition_key": key,
        "corruption": corruption,
        "severity": severity,
        "evaluated_images": metrics["evaluated_images"],
        "full_fixed_test_split": metrics["full_fixed_test_split"],
        "cache_manifest_sha256": context.cache_audit["manifest_sha256"],
        "cache_content_sha256": context.cache_manifest["cache_content_sha256"],
        "checkpoint_sha256": context.config_contract["checkpoint_sha256"],
        "source_benchmark_sha256": context.config_contract["source_benchmark_sha256"],
        "source_artifact_manifest_sha256": context.config_contract[
            "source_artifact_manifest_sha256"
        ],
        "metrics_sha256": manifest_files["metrics.json"]["sha256"],
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "all_required_gates_passed": (
            all(value is True for value in metrics["checks"].values())
            if formal_artifact
            else all(value is not False for value in metrics["checks"].values())
        ),
    }
    if formal_artifact:
        required = (
            completion["full_fixed_test_split"],
            completion["all_required_gates_passed"],
            metrics["source_pre_parity"]["official_metrics_exact"],
            metrics["source_pre_parity"]["unified_metrics_exact"],
            metrics["source_pre_parity"]["summary_exact"],
        )
        if not all(value is True for value in required):
            raise RuntimeError(f"refusing formal CONDITION_COMPLETE for {key}")
    source_runner.write_json_atomic(staging / sentinel_name, completion)
    os.replace(staging, final)
    return {
        "published_output_dir": str(final),
        "metrics": metrics,
        "completion": completion,
    }


def _dataset_top_files(dataset_root: Path) -> tuple[str, ...]:
    return (
        "dataset_run_config.yaml",
        "dataset_provenance.json",
        "dataset_summary.json",
    )


def finalize_dataset_from_conditions(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    output_root: Path,
    dataset_name: str,
    conditions: Sequence[tuple[str, int]],
    formal_artifact: bool,
    repository_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify condition shards and publish a dataset index/sentinel last."""

    conditions = tuple(conditions)
    if formal_artifact and conditions != CONDITIONS:
        raise ValueError("formal DATASET_COMPLETE requires the exact 13 conditions")
    dataset_root = output_root / dataset_name
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset artifact root is missing: {dataset_root}")
    sentinel_name = (
        protocol["outputs"]["formal_dataset_sentinel"]
        if formal_artifact
        else protocol["outputs"]["smoke_dataset_sentinel"]
    )
    index_root = dataset_root / "dataset_index"
    if (dataset_root / sentinel_name).exists():
        raise FileExistsError(
            f"refusing to overwrite completed dataset index at {dataset_root}"
        )
    if index_root.exists():
        preserved = dataset_root / f".dataset_index.incomplete-{time.time_ns()}"
        os.replace(index_root, preserved)
    protocol_sha256 = sha256_file(protocol_path)
    repository = dict(repository_contract or _repository_contract())
    verified: list[dict[str, Any]] = []
    verified_results: list[dict[str, Any]] = []
    expected_keys = tuple(condition_key(*condition) for condition in conditions)
    condition_root = dataset_root / "conditions"
    actual_keys = tuple(
        sorted(
            path.name
            for path in condition_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
    )
    _require_equal(
        set(actual_keys), set(expected_keys), f"{dataset_name} exact condition directories"
    )
    for index, (corruption, severity) in enumerate(conditions):
        key = condition_key(corruption, severity)
        result = verify_condition_artifact(
            condition_root=condition_root / key,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            dataset_name=dataset_name,
            corruption=corruption,
            severity=severity,
            repository_code_bundle_sha256=repository["code_bundle_sha256"],
            formal_artifact=formal_artifact,
        )
        verified_results.append(result)
        verified.append(
            {
                "condition_index": CONDITIONS.index((corruption, severity)),
                "condition_key": key,
                "corruption": corruption,
                "severity": severity,
                **dict(result["metrics"]["summary"]),
                "source_summary": dict(result["metrics"]["source_summary"]),
                "deltas_from_source": dict(result["metrics"]["deltas_from_source"]),
                "metrics": f"conditions/{key}/metrics.json",
                "metrics_sha256": result["manifest"]["files"]["metrics.json"]["sha256"],
                "condition_complete_sha256": result["completion_sha256"],
                "condition_artifact_manifest_sha256": result["manifest_sha256"],
            }
        )

    summary = {
        "schema_version": 1,
        "method": "AdaBN",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "formal_artifact": formal_artifact,
        "paper_result": formal_artifact,
        "dataset": dataset_name,
        "condition_count": len(verified),
        "evaluated_images_per_condition": (
            int(protocol["datasets"][dataset_name]["test_images"])
            if formal_artifact
            else None
        ),
        "conditions": verified,
        "delta_role": "report_only_never_used_for_selection_or_tuning",
        "checks": {
            "exact_13_condition_contract": formal_artifact and conditions == CONDITIONS,
            "all_condition_artifacts_verified": True,
            "all_source_pre_parity_gates_passed": all(
                result["metrics"]["checks"]["source_pre_parity_passed"] is True
                for result in verified_results
            ),
        },
    }
    index_staging = dataset_root / f".dataset_index.build-{os.getpid()}-{time.time_ns()}"
    index_staging.mkdir(parents=False, exist_ok=False)
    source_runner.write_json_atomic(index_staging / "dataset_summary.json", summary)
    effective = deepcopy(dict(protocol))
    effective["runtime"] = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "dataset": dataset_name,
        "conditions": list(expected_keys),
        "formal_artifact": formal_artifact,
    }
    _write_yaml_atomic(index_staging / "dataset_run_config.yaml", effective)
    provenance = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "formal_artifact": formal_artifact,
        "dataset": dataset_name,
        "repository": repository,
        "cache_manifest_sha256": protocol["datasets"][dataset_name][
            "cache_manifest_sha256"
        ],
        "cache_content_sha256": protocol["datasets"][dataset_name][
            "cache_content_sha256"
        ],
        "checkpoint_sha256": protocol["datasets"][dataset_name]["checkpoint_sha256"],
        "source_benchmark_sha256": protocol["datasets"][dataset_name][
            "source_benchmark_sha256"
        ],
        "implementation_pilot_lineage": dict(
            protocol["implementation_pilot_lineage"]
        ),
        "historical_relocations": dict(protocol["historical_relocations"]),
        "condition_complete_sha256": {
            record["condition_key"]: record["condition_complete_sha256"]
            for record in verified
        },
    }
    source_runner.write_json_atomic(index_staging / "dataset_provenance.json", provenance)
    files = {
        relative: {
            "sha256": sha256_file(index_staging / relative),
            "bytes": (index_staging / relative).stat().st_size,
        }
        for relative in _dataset_top_files(dataset_root)
    }
    condition_links = {
        record["condition_key"]: {
            "artifact_manifest": f"../conditions/{record['condition_key']}/artifact_manifest.json",
            "artifact_manifest_sha256": record[
                "condition_artifact_manifest_sha256"
            ],
            "completion": (
                f"../conditions/{record['condition_key']}/"
                f"{protocol['outputs']['formal_condition_sentinel'] if formal_artifact else protocol['outputs']['smoke_condition_sentinel']}"
            ),
            "completion_sha256": record["condition_complete_sha256"],
        }
        for record in verified
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "repository_code_bundle_sha256": repository["code_bundle_sha256"],
        "formal_artifact": formal_artifact,
        "dataset": dataset_name,
        "files": files,
        "conditions": condition_links,
    }
    manifest_path = index_staging / "artifact_manifest.json"
    source_runner.write_json_atomic(manifest_path, manifest)
    completion = {
        "complete": True,
        "scope": "dataset",
        "formal_artifact": formal_artifact,
        "paper_result": formal_artifact,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "repository_code_bundle_sha256": repository["code_bundle_sha256"],
        "dataset": dataset_name,
        "condition_count": len(conditions),
        "exact_13_condition_contract": formal_artifact and conditions == CONDITIONS,
        "cache_manifest_sha256": protocol["datasets"][dataset_name][
            "cache_manifest_sha256"
        ],
        "cache_content_sha256": protocol["datasets"][dataset_name][
            "cache_content_sha256"
        ],
        "checkpoint_sha256": protocol["datasets"][dataset_name]["checkpoint_sha256"],
        "source_benchmark_sha256": protocol["datasets"][dataset_name][
            "source_benchmark_sha256"
        ],
        "dataset_summary_sha256": files["dataset_summary.json"]["sha256"],
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "all_required_gates_passed": True,
    }
    if formal_artifact and completion["exact_13_condition_contract"] is not True:
        raise RuntimeError("refusing DATASET_COMPLETE without exact 13 conditions")
    os.replace(index_staging, index_root)
    source_runner.write_json_atomic(dataset_root / sentinel_name, completion)
    return {"summary": summary, "manifest": manifest, "completion": completion}


def verify_dataset_artifact(
    *,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    output_root: Path,
    dataset_name: str,
    repository_code_bundle_sha256: str,
) -> dict[str, Any]:
    dataset_root = output_root / dataset_name
    sentinel_path = dataset_root / protocol["outputs"]["formal_dataset_sentinel"]
    index_root = dataset_root / "dataset_index"
    manifest_path = index_root / "artifact_manifest.json"
    for path in (sentinel_path, manifest_path, index_root / "dataset_summary.json"):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete formal dataset artifact: {path}")
    completion = _load_json(sentinel_path)
    contract = protocol["datasets"][dataset_name]
    for actual, expected, label in (
        (completion.get("complete"), True, "complete"),
        (completion.get("formal_artifact"), True, "formal"),
        (completion.get("dataset"), dataset_name, "dataset"),
        (completion.get("condition_count"), 13, "condition count"),
        (completion.get("exact_13_condition_contract"), True, "condition contract"),
        (completion.get("protocol_sha256"), protocol_sha256, "protocol"),
        (completion.get("repository_code_bundle_sha256"), repository_code_bundle_sha256, "code"),
        (completion.get("cache_content_sha256"), contract["cache_content_sha256"], "cache"),
        (completion.get("checkpoint_sha256"), contract["checkpoint_sha256"], "checkpoint"),
        (completion.get("source_benchmark_sha256"), contract["source_benchmark_sha256"], "Source"),
        (completion.get("all_required_gates_passed"), True, "gates"),
        (sha256_file(manifest_path), completion.get("artifact_manifest_sha256"), "manifest"),
    ):
        _require_equal(actual, expected, f"{dataset_name} DATASET_COMPLETE {label}")
    manifest = _load_json(manifest_path)
    _require_equal(manifest.get("formal_artifact"), True, "dataset manifest formal")
    _require_equal(manifest.get("protocol_sha256"), protocol_sha256, "dataset manifest protocol")
    _require_equal(
        manifest.get("repository_code_bundle_sha256"),
        repository_code_bundle_sha256,
        "dataset manifest code",
    )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("dataset manifest files must be a mapping")
    for relative, expected in files.items():
        path = index_root / relative
        _require_equal(path.stat().st_size, int(expected["bytes"]), f"{relative} bytes")
        _require_equal(sha256_file(path), expected["sha256"], f"{relative} SHA256")
    links = manifest.get("conditions")
    _require_exact_condition_link_keys(links)
    for condition in CONDITIONS:
        key = condition_key(*condition)
        verified = verify_condition_artifact(
            condition_root=dataset_root / "conditions" / key,
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            dataset_name=dataset_name,
            corruption=condition[0],
            severity=condition[1],
            repository_code_bundle_sha256=repository_code_bundle_sha256,
            formal_artifact=True,
        )
        _require_equal(
            links[key]["artifact_manifest_sha256"],
            verified["manifest_sha256"],
            f"{key} linked manifest",
        )
        _require_equal(
            links[key]["completion_sha256"],
            verified["completion_sha256"],
            f"{key} linked completion",
        )
    summary = _load_json(index_root / "dataset_summary.json")
    _require_equal(
        sha256_file(index_root / "dataset_summary.json"),
        completion["dataset_summary_sha256"],
        "dataset summary",
    )
    return {
        "completion": completion,
        "completion_sha256": sha256_file(sentinel_path),
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "summary": summary,
    }


def _require_exact_condition_link_keys(links: Any) -> None:
    """Require all formal keys without relying on JSON object order.

    The atomic JSON writer uses ``sort_keys=True``.  Protocol order remains an
    ordered list in ``dataset_summary.json``; this manifest mapping is an exact
    key-set contract.
    """

    expected = {condition_key(*condition) for condition in CONDITIONS}
    if not isinstance(links, Mapping) or set(links) != expected:
        raise ValueError("dataset manifest condition key set drifted")


def aggregate_formal_results(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Rebuild dataset indices from 39 shards, then publish global COMPLETE."""

    output_root.mkdir(parents=True, exist_ok=True)
    complete_path = output_root / protocol["outputs"]["formal_global_sentinel"]
    if complete_path.exists():
        raise FileExistsError(f"formal global result already exists: {complete_path}")
    protocol_sha256 = sha256_file(protocol_path)
    repository = _repository_contract()
    datasets: list[dict[str, Any]] = []
    for dataset_name in DATASETS:
        dataset_complete = (
            output_root
            / dataset_name
            / protocol["outputs"]["formal_dataset_sentinel"]
        )
        if not dataset_complete.exists():
            finalize_dataset_from_conditions(
                protocol_path=protocol_path,
                protocol=protocol,
                output_root=output_root,
                dataset_name=dataset_name,
                conditions=CONDITIONS,
                formal_artifact=True,
                repository_contract=repository,
            )
        verified = verify_dataset_artifact(
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            output_root=output_root,
            dataset_name=dataset_name,
            repository_code_bundle_sha256=repository["code_bundle_sha256"],
        )
        datasets.append(
            {
                "dataset": dataset_name,
                "condition_count": 13,
                "dataset_summary": f"../{dataset_name}/dataset_index/dataset_summary.json",
                "dataset_summary_sha256": verified["completion"][
                    "dataset_summary_sha256"
                ],
                "dataset_artifact_manifest": (
                    f"../{dataset_name}/dataset_index/artifact_manifest.json"
                ),
                "dataset_artifact_manifest_sha256": verified["manifest_sha256"],
                "dataset_complete": f"../{dataset_name}/DATASET_COMPLETE.json",
                "dataset_complete_sha256": verified["completion_sha256"],
                "conditions": verified["summary"]["conditions"],
            }
        )

    index_root = output_root / "global_index"
    if index_root.exists():
        os.replace(
            index_root,
            output_root / f".global_index.incomplete-{time.time_ns()}",
        )
    staging = output_root / f".global_index.build-{os.getpid()}-{time.time_ns()}"
    staging.mkdir(parents=False, exist_ok=False)
    aggregate = {
        "schema_version": 1,
        "method": "AdaBN",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "formal_artifact": True,
        "paper_result": True,
        "dataset_count": 3,
        "condition_count_per_dataset": 13,
        "global_dataset_condition_count": 39,
        "datasets": datasets,
        "delta_role": "report_only_never_used_for_selection_or_tuning",
        "checks": {
            "exact_three_datasets": True,
            "exact_13_conditions_per_dataset": True,
            "exact_39_dataset_conditions": True,
            "all_condition_manifests_recursively_verified": True,
            "common_protocol_config_code_cache_checkpoint_lineage": True,
        },
    }
    source_runner.write_json_atomic(staging / "aggregate_metrics.json", aggregate)
    effective = deepcopy(dict(protocol))
    effective["runtime"] = {
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "mode": "aggregate_only",
        "formal_artifact": True,
    }
    _write_yaml_atomic(staging / "run_config.yaml", effective)
    provenance = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "repository": repository,
        "implementation_pilot_lineage": dict(
            protocol["implementation_pilot_lineage"]
        ),
        "historical_relocations": dict(protocol["historical_relocations"]),
        "datasets": {
            record["dataset"]: {
                "dataset_complete_sha256": record["dataset_complete_sha256"],
                "dataset_artifact_manifest_sha256": record[
                    "dataset_artifact_manifest_sha256"
                ],
                "cache_manifest_sha256": protocol["datasets"][record["dataset"]][
                    "cache_manifest_sha256"
                ],
                "checkpoint_sha256": protocol["datasets"][record["dataset"]][
                    "checkpoint_sha256"
                ],
                "source_benchmark_sha256": protocol["datasets"][record["dataset"]][
                    "source_benchmark_sha256"
                ],
            }
            for record in datasets
        },
    }
    source_runner.write_json_atomic(staging / "provenance.json", provenance)
    files = _artifact_files(staging, excluded=("artifact_manifest.json",))
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "repository_code_bundle_sha256": repository["code_bundle_sha256"],
        "formal_artifact": True,
        "files": files,
        "datasets": {
            record["dataset"]: {
                "artifact_manifest": record["dataset_artifact_manifest"],
                "artifact_manifest_sha256": record[
                    "dataset_artifact_manifest_sha256"
                ],
                "completion": record["dataset_complete"],
                "completion_sha256": record["dataset_complete_sha256"],
            }
            for record in datasets
        },
    }
    manifest_path = staging / "artifact_manifest.json"
    source_runner.write_json_atomic(manifest_path, manifest)
    completion = {
        "complete": True,
        "scope": "global",
        "formal_artifact": True,
        "paper_result": True,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "repository_code_bundle_sha256": repository["code_bundle_sha256"],
        "dataset_count": 3,
        "condition_count_per_dataset": 13,
        "global_dataset_condition_count": 39,
        "aggregate_metrics_sha256": files["aggregate_metrics.json"]["sha256"],
        "artifact_manifest_sha256": sha256_file(manifest_path),
        "all_required_gates_passed": True,
    }
    os.replace(staging, index_root)
    source_runner.write_json_atomic(complete_path, completion)
    return {
        "published_output_dir": str(output_root),
        "aggregate": aggregate,
        "completion": completion,
    }


def _run_formal_dataset_atomic(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    dataset_name: str,
    device_spec: str,
    output_root: Path,
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    final_dataset = output_root / dataset_name
    if final_dataset.exists():
        raise FileExistsError(f"formal dataset output already exists: {final_dataset}")
    output_root.mkdir(parents=True, exist_ok=True)
    staging_root = output_root / f".{dataset_name}.build-{os.getpid()}-{time.time_ns()}"
    staging_root.mkdir(parents=False, exist_ok=False)
    context = prepare_dataset_context(
        protocol, dataset_name, verify_all_cache_file_hashes=True
    )
    results = []
    for corruption, severity in CONDITIONS:
        results.append(
            publish_condition(
                protocol_path=protocol_path,
                protocol=protocol,
                context=context,
                corruption=corruption,
                severity=severity,
                device_spec=device_spec,
                output_root=staging_root,
                formal_artifact=True,
                repository_contract=repository,
            )
        )
    finalized = finalize_dataset_from_conditions(
        protocol_path=protocol_path,
        protocol=protocol,
        output_root=staging_root,
        dataset_name=dataset_name,
        conditions=CONDITIONS,
        formal_artifact=True,
        repository_contract=repository,
    )
    os.replace(staging_root / dataset_name, final_dataset)
    staging_root.rmdir()
    return {"conditions": results, "dataset": finalized}


def _finalize_smoke_root(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    staging: Path,
    datasets: Sequence[str],
    conditions: Sequence[tuple[str, int]],
    repository: Mapping[str, Any],
    max_images: int | None,
) -> dict[str, Any]:
    dataset_records = []
    for dataset_name in datasets:
        result = finalize_dataset_from_conditions(
            protocol_path=protocol_path,
            protocol=protocol,
            output_root=staging,
            dataset_name=dataset_name,
            conditions=conditions,
            formal_artifact=False,
            repository_contract=repository,
        )
        dataset_records.append(
            {
                "dataset": dataset_name,
                "condition_count": len(conditions),
                "summary": result["summary"],
            }
        )
    aggregate = {
        "schema_version": 1,
        "method": "AdaBN",
        "formal_artifact": False,
        "paper_result": False,
        "dataset_count": len(datasets),
        "condition_count_per_dataset": len(conditions),
        "max_images": max_images,
        "datasets": dataset_records,
    }
    source_runner.write_json_atomic(staging / "aggregate_metrics.json", aggregate)
    source_runner.write_json_atomic(
        staging / "provenance.json",
        {
            "schema_version": 1,
            "protocol_sha256": sha256_file(protocol_path),
            "repository": dict(repository),
            "formal_artifact": False,
        },
    )
    _write_yaml_atomic(staging / "run_config.yaml", dict(protocol))
    sentinel_name = protocol["outputs"]["smoke_global_sentinel"]
    files = _artifact_files(
        staging, excluded=("artifact_manifest.json", sentinel_name)
    )
    manifest = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(protocol_path),
        "formal_artifact": False,
        "files": files,
    }
    source_runner.write_json_atomic(staging / "artifact_manifest.json", manifest)
    completion = {
        "complete": True,
        "scope": "smoke",
        "formal_artifact": False,
        "paper_result": False,
        "protocol_sha256": sha256_file(protocol_path),
        "aggregate_metrics_sha256": files["aggregate_metrics.json"]["sha256"],
        "artifact_manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
    }
    source_runner.write_json_atomic(staging / sentinel_name, completion)
    return {"aggregate": aggregate, "completion": completion}


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path, protocol = load_protocol(args.protocol)
    selection = resolve_selection(args, protocol)
    if selection.mode == "aggregate_only":
        return aggregate_formal_results(protocol_path, protocol, selection.output_root)
    repository = _repository_contract()
    if selection.mode == "smoke":
        final = selection.output_root
        if final.exists():
            raise FileExistsError(f"smoke output already exists: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        staging = final.with_name(f".{final.name}.build-{os.getpid()}-{time.time_ns()}")
        staging.mkdir(parents=False, exist_ok=False)
        for dataset_name in selection.datasets:
            context = prepare_dataset_context(
                protocol, dataset_name, verify_all_cache_file_hashes=False
            )
            for corruption, severity in selection.conditions:
                publish_condition(
                    protocol_path=protocol_path,
                    protocol=protocol,
                    context=context,
                    corruption=corruption,
                    severity=severity,
                    device_spec=args.device,
                    output_root=staging,
                    formal_artifact=False,
                    max_images=selection.max_images,
                    repository_contract=repository,
                )
        result = _finalize_smoke_root(
            protocol_path=protocol_path,
            protocol=protocol,
            staging=staging,
            datasets=selection.datasets,
            conditions=selection.conditions,
            repository=repository,
            max_images=selection.max_images,
        )
        os.replace(staging, final)
        result["published_output_dir"] = str(final)
        return result

    if selection.mode == "formal_condition_shard":
        selection.output_root.mkdir(parents=True, exist_ok=True)
        results = []
        for dataset_name in selection.datasets:
            context = prepare_dataset_context(
                protocol, dataset_name, verify_all_cache_file_hashes=True
            )
            for corruption, severity in selection.conditions:
                results.append(
                    publish_condition(
                        protocol_path=protocol_path,
                        protocol=protocol,
                        context=context,
                        corruption=corruption,
                        severity=severity,
                        device_spec=args.device,
                        output_root=selection.output_root,
                        formal_artifact=True,
                        repository_contract=repository,
                    )
                )
        return {
            "published_output_dir": str(selection.output_root),
            "formal_condition_shards": results,
            "global_complete_created": False,
        }

    datasets_result = []
    for dataset_name in selection.datasets:
        datasets_result.append(
            _run_formal_dataset_atomic(
                protocol_path=protocol_path,
                protocol=protocol,
                dataset_name=dataset_name,
                device_spec=args.device,
                output_root=selection.output_root,
                repository=repository,
            )
        )
    if selection.mode == "formal_all":
        return aggregate_formal_results(protocol_path, protocol, selection.output_root)
    return {
        "published_output_dir": str(selection.output_root / selection.datasets[0]),
        "formal_dataset_shards": datasets_result,
        "global_complete_created": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_benchmark(args)
    print(f"Artifacts: {result['published_output_dir']}")
    return 0


def _verify_condition_internal_semantics(
    *,
    condition_root: Path,
    metrics: Mapping[str, Any],
    probabilities: np.ndarray,
    per_image: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
    manifest_files: Mapping[str, Any],
) -> None:
    """Cross-check a sealed condition's indexed outputs, not only file hashes."""

    count = int(metrics["evaluated_images"])
    _require_equal(len(per_image), count, "per-image semantic count")
    _require_equal(len(diagnostics), count, "diagnostics semantic count")
    probability_hasher = TensorSequenceHasher()
    observed_full_sha_indexes: list[int] = []
    image_ids: set[str] = set()

    for index, (record, diagnostic) in enumerate(
        zip(per_image, diagnostics, strict=True)
    ):
        image_id = str(record.get("image_id", ""))
        if not image_id or image_id in image_ids:
            raise ValueError(f"invalid or duplicate image_id at index {index}")
        image_ids.add(image_id)
        for actual, expected, label in (
            (record.get("index"), index, "record index"),
            (record.get("probability_shard_index"), index, "shard index"),
            (record.get("probability_shard"), "probabilities_256.npy", "shard name"),
            (
                record.get("probability_shard_sha256"),
                metrics["probability_shard_sha256"],
                "shard lineage",
            ),
            (diagnostic.get("index"), index, "diagnostic index"),
            (diagnostic.get("image_id"), image_id, "diagnostic image_id"),
            (diagnostic.get("method"), "adabn", "diagnostic method"),
            (diagnostic.get("optimizer_steps"), 0, "optimizer steps"),
        ):
            _require_equal(actual, expected, f"{label} at image {index}")

        probability = np.asarray(probabilities[index])
        _require_equal(
            _raw_array_sha256(probability),
            record.get("probability_tensor_raw_sha256"),
            f"probability raw SHA256 at image {index}",
        )
        _require_equal(
            float(probability.min()),
            float(record["probability_min"]),
            f"probability min at image {index}",
        )
        _require_equal(
            float(probability.max()),
            float(record["probability_max"]),
            f"probability max at image {index}",
        )
        probability_hasher.update(image_id, probability)

        relative_mask = Path(str(record.get("prediction_mask", "")))
        if (
            not relative_mask.parts
            or relative_mask.is_absolute()
            or ".." in relative_mask.parts
        ):
            raise ValueError(f"unsafe prediction mask path at image {index}")
        relative_text = str(relative_mask)
        mask_manifest = manifest_files.get(relative_text)
        if not isinstance(mask_manifest, Mapping):
            raise ValueError(f"prediction mask is absent from manifest: {relative_text}")
        _require_equal(
            record.get("prediction_mask_sha256"),
            mask_manifest.get("sha256"),
            f"prediction mask lineage at image {index}",
        )
        with Image.open(condition_root / relative_mask) as handle:
            saved_mask = np.asarray(handle.convert("L"))
        expected_mask = np.where(probability > 0.5, 255, 0).astype(np.uint8)
        if not np.array_equal(saved_mask, expected_mask):
            raise ValueError(f"prediction mask/probability mismatch at image {index}")

        for field in (
            "source_pre_probability_reference_bit_exact",
            "source_pre_binary_mask_reference_bit_exact",
            "source_pre_binary_mask_file_hash_exact",
        ):
            _require_equal(record.get(field), True, f"{field} at image {index}")
        source_parity = diagnostic.get("source_pre_parity")
        if not isinstance(source_parity, Mapping):
            raise TypeError(f"missing source parity diagnostic at image {index}")
        for field in (
            "probability_array_bit_exact",
            "binary_mask_array_bit_exact",
            "binary_mask_file_hash_exact",
        ):
            _require_equal(
                source_parity.get(field), True, f"diagnostic {field} at image {index}"
            )

        state_audit = diagnostic.get("state_audit")
        if not isinstance(state_audit, Mapping):
            raise TypeError(f"missing state audit at image {index}")
        _require_equal(
            state_audit.get("exact_parameter_and_buffer_equality_performed"),
            True,
            f"exact state audit at image {index}",
        )
        full_sha = state_audit.get("full_sha_performed")
        if not isinstance(full_sha, bool):
            raise TypeError(f"invalid full SHA flag at image {index}")
        if full_sha:
            observed_full_sha_indexes.append(index)
            if (
                record.get("state_after_post_sha256") is None
                or record.get("reset_state_sha256")
                != record.get("source_state_sha256")
                or diagnostic.get("state_after_post_fingerprint") is None
                or diagnostic.get("reset_fingerprint") is None
            ):
                raise ValueError(f"incomplete full SHA evidence at image {index}")
        elif any(
            value is not None
            for value in (
                record.get("state_after_post_sha256"),
                record.get("reset_state_sha256"),
                diagnostic.get("state_after_post_fingerprint"),
                diagnostic.get("reset_fingerprint"),
            )
        ):
            raise ValueError(f"non-audited image fabricates SHA evidence at {index}")

    _require_equal(
        probability_hasher.hexdigest(),
        metrics.get("adabn_probability_tensor_sequence_sha256"),
        "probability tensor sequence SHA256",
    )
    schedule = metrics.get("state_audit_schedule")
    if not isinstance(schedule, Mapping):
        raise TypeError("condition metrics omit state_audit_schedule")
    _require_equal(
        observed_full_sha_indexes,
        schedule.get("image_indexes_zero_based"),
        "diagnostic full SHA schedule",
    )
    _require_equal(
        schedule.get("image_indexes_zero_based"),
        schedule.get("expected_image_indexes_zero_based"),
        "expected full SHA schedule",
    )
    _require_equal(schedule.get("count"), len(observed_full_sha_indexes), "full SHA count")
    _require_equal(schedule.get("exact"), True, "full SHA schedule exact")

    summary = metrics.get("summary")
    source_summary = metrics.get("source_summary")
    deltas = metrics.get("deltas_from_source")
    if not all(isinstance(value, Mapping) for value in (summary, source_summary, deltas)):
        raise TypeError("condition metrics omit summary/delta mappings")
    expected_deltas = {
        field: float(summary[field]) - float(source_summary[field])
        for field in summary
    }
    if not source_benchmark._canonical_json_equal(expected_deltas, deltas):
        raise ValueError("AdaBN deltas_from_source are internally inconsistent")


def verify_condition_artifact(
    *,
    condition_root: Path,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    dataset_name: str,
    corruption: str,
    severity: int,
    repository_code_bundle_sha256: str,
    formal_artifact: bool,
) -> dict[str, Any]:
    key = condition_key(corruption, severity)
    sentinel_name = (
        protocol["outputs"]["formal_condition_sentinel"]
        if formal_artifact
        else protocol["outputs"]["smoke_condition_sentinel"]
    )
    sentinel_path = condition_root / sentinel_name
    manifest_path = condition_root / "artifact_manifest.json"
    for path in (sentinel_path, manifest_path, condition_root / "metrics.json"):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete condition artifact: {path}")
    completion = _load_json(sentinel_path)
    contract = protocol["datasets"][dataset_name]
    for actual, expected, label in (
        (completion.get("complete"), True, "complete"),
        (completion.get("scope"), "condition", "scope"),
        (completion.get("formal_artifact"), formal_artifact, "formal_artifact"),
        (completion.get("protocol_sha256"), protocol_sha256, "protocol"),
        (
            completion.get("repository_code_bundle_sha256"),
            repository_code_bundle_sha256,
            "repository code bundle",
        ),
        (completion.get("dataset"), dataset_name, "dataset"),
        (completion.get("condition_key"), key, "condition"),
        (completion.get("corruption"), corruption, "corruption"),
        (completion.get("severity"), severity, "severity"),
        (completion.get("cache_manifest_sha256"), contract["cache_manifest_sha256"], "cache"),
        (completion.get("cache_content_sha256"), contract["cache_content_sha256"], "cache content"),
        (completion.get("checkpoint_sha256"), contract["checkpoint_sha256"], "checkpoint"),
        (
            completion.get("source_benchmark_sha256"),
            contract["source_benchmark_sha256"],
            "Source benchmark",
        ),
        (
            completion.get("source_artifact_manifest_sha256"),
            contract["source_artifact_manifest_sha256"],
            "Source artifact manifest",
        ),
        (completion.get("all_required_gates_passed"), True, "required gates"),
        (sha256_file(manifest_path), completion.get("artifact_manifest_sha256"), "manifest"),
    ):
        _require_equal(actual, expected, f"{dataset_name}/{key} completion {label}")
    manifest = _load_json(manifest_path)
    for actual, expected, label in (
        (manifest.get("protocol_sha256"), protocol_sha256, "protocol"),
        (manifest.get("repository_code_bundle_sha256"), repository_code_bundle_sha256, "code"),
        (manifest.get("formal_artifact"), formal_artifact, "formal"),
        (manifest.get("dataset"), dataset_name, "dataset"),
        (manifest.get("condition_key"), key, "condition"),
        (manifest.get("cache_manifest_sha256"), contract["cache_manifest_sha256"], "cache"),
        (manifest.get("cache_content_sha256"), contract["cache_content_sha256"], "cache content"),
        (manifest.get("checkpoint_sha256"), contract["checkpoint_sha256"], "checkpoint"),
        (manifest.get("source_benchmark_sha256"), contract["source_benchmark_sha256"], "Source"),
    ):
        _require_equal(actual, expected, f"condition manifest {label}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise TypeError("condition artifact manifest files must be a mapping")
    _verify_files_mapping(
        condition_root,
        files,
        allowed_unlisted=("artifact_manifest.json", sentinel_name),
    )
    _require_equal(files["metrics.json"]["sha256"], completion["metrics_sha256"], "metrics")
    metrics = _load_json(condition_root / "metrics.json")
    for actual, expected, label in (
        (metrics.get("dataset"), dataset_name, "dataset"),
        (metrics.get("condition_key"), key, "condition"),
        (metrics.get("corruption"), corruption, "corruption"),
        (metrics.get("severity"), severity, "severity"),
    ):
        _require_equal(actual, expected, f"condition metrics {label}")
    probability_path = condition_root / "probabilities_256.npy"
    _require_equal(
        files["probabilities_256.npy"]["sha256"],
        metrics["probability_shard_sha256"],
        "AdaBN probability shard SHA256",
    )
    probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    expected_count = int(metrics["evaluated_images"])
    _require_equal(probabilities.shape, (expected_count, 256, 256), "probability shape")
    _require_equal(probabilities.dtype.str, "<f4", "probability dtype")
    per_image = _load_jsonl(condition_root / "per_image.jsonl")
    diagnostics = _load_jsonl(condition_root / "adaptation_diagnostics.jsonl")
    _require_equal(len(per_image), expected_count, "per-image count")
    _require_equal(len(diagnostics), expected_count, "diagnostics count")
    mask_count = len(tuple((condition_root / "prediction_masks_256").rglob("*.png")))
    _require_equal(mask_count, expected_count, "prediction mask count")
    _verify_condition_internal_semantics(
        condition_root=condition_root,
        metrics=metrics,
        probabilities=probabilities,
        per_image=per_image,
        diagnostics=diagnostics,
        manifest_files=files,
    )
    del probabilities
    if formal_artifact:
        for actual, expected, label in (
            (metrics.get("formal_artifact"), True, "formal"),
            (metrics.get("full_fixed_test_split"), True, "full split"),
            (metrics.get("evaluated_images"), int(contract["test_images"]), "image count"),
            (completion.get("evaluated_images"), int(contract["test_images"]), "sentinel image count"),
            (completion.get("full_fixed_test_split"), True, "sentinel full split"),
            (metrics["source_pre_parity"].get("official_metrics_exact"), True, "official parity"),
            (metrics["source_pre_parity"].get("unified_metrics_exact"), True, "unified parity"),
            (metrics["source_pre_parity"].get("summary_exact"), True, "summary parity"),
            (metrics["state_audit_schedule"].get("exact"), True, "full SHA schedule"),
        ):
            _require_equal(actual, expected, f"formal condition {label}")
        failed_checks = sorted(
            name for name, passed in metrics["checks"].items() if passed is not True
        )
        if failed_checks:
            raise RuntimeError(f"formal condition metrics gates failed: {failed_checks}")
    return {
        "completion": completion,
        "manifest": manifest,
        "metrics": metrics,
        "completion_sha256": sha256_file(sentinel_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


if __name__ == "__main__":
    raise SystemExit(main())
