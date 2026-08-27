"""Run the fixed 32-image source-domain AdaBN implementation smoke.

This is not a training run and not a paper-result benchmark.  It opens pixels
only for the configured prefix of the frozen source-domain corruption Pilot,
never constructs the fixed-test dataset, never gives a mask to the episodic
method, and never computes IoU/Pd/Fa or any other label-derived metric.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
import yaml

from corruptions.corruption_protocol import load_severity_table
from corruptions.infrared_corruptions import apply_corruption
from dataio.corruption_cache import TensorSequenceHasher, ordered_ids_sha256
from dataio.research_dataset import (
    DEFAULT_EXTENSIONS,
    IRSTDResearchDataset,
    read_split_ids,
    resolve_dataset_layout,
)
import test_source as source_runner
from tta.adabn import AdaBNMethod
from tta.episodic_runner import EpisodeResult, EpisodicRunner
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager, StateFingerprint


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "adabn_source_pilot_smoke_v1.yaml"
EXPECTED_PROTOCOL_ID = "cr-sitta-adabn-source-pilot-smoke-v1"
PROVENANCE_PATHS = (
    "configs/adabn_source_pilot_ids_irstd1k.txt",
    "configs/adabn_source_pilot_smoke_v1.yaml",
    "corruptions/corruption_protocol.py",
    "corruptions/infrared_corruptions.py",
    "corruptions/severity_tables.yaml",
    "dataio/corruption_cache.py",
    "dataio/research_dataset.py",
    "model/MSHNet_NSFPN.py",
    "run_adabn_source_pilot.py",
    "test_source.py",
    "THIRD_PARTY.md",
    "THIRD_PARTY_COMMITS.txt",
    "third_party/tent/LICENSE",
    "third_party/tent/norm.py",
    "third_party/tent_reference.json",
    "tta/adabn.py",
    "tta/episodic_runner.py",
    "tta/model_adapter.py",
    "tta/state_manager.py",
)


@dataclass(frozen=True)
class PilotPaths:
    config: Path
    dataset_root: Path
    train_split: Path
    test_split: Path
    selected_ids_file: Path
    parent_pilot: Path
    parent_manifest: Path
    parent_complete: Path
    severity_table: Path
    checkpoint: Path
    output_root: Path


@dataclass(frozen=True)
class SourcePilotSample:
    """One materialized label-free input passed to the episodic runner."""

    image_id: str
    image: Tensor
    metadata: Mapping[str, Any]
    input_raw_sha256: str


class SourcePilotIOGuard:
    """Allow pixel I/O only for the exact selected fixed-train files."""

    def __init__(self, allowed: Mapping[tuple[str, str], Path]) -> None:
        self.allowed = {key: path.resolve() for key, path in allowed.items()}
        self.image_open_count = 0
        self.mask_open_count = 0
        self.forbidden_open_count = 0
        self.opened_ids: set[str] = set()
        self._digest = hashlib.sha256()

    def __call__(self, role: str, image_id: str, path: Path) -> None:
        resolved = path.resolve()
        expected = self.allowed.get((role, image_id))
        if role not in {"image", "mask"} or expected is None or resolved != expected:
            self.forbidden_open_count += 1
            raise PermissionError(
                f"source Pilot I/O guard refused {role} {image_id!r}: {resolved}"
            )
        if role == "image":
            self.image_open_count += 1
        else:
            self.mask_open_count += 1
        self.opened_ids.add(image_id)
        self._digest.update(f"{role}\t{image_id}\t{resolved}\n".encode("utf-8"))

    def summary(self) -> dict[str, Any]:
        return {
            "allowed_unique_ids": len({image_id for _, image_id in self.allowed}),
            "opened_unique_ids": len(self.opened_ids),
            "image_open_count": self.image_open_count,
            "mask_open_count": self.mask_open_count,
            "forbidden_open_count": self.forbidden_open_count,
            "ordered_io_events_sha256": self._digest.hexdigest(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional sibling-atomically-published output root override.",
    )
    return parser


def _project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"expected a JSON object: {path}")
    return dict(loaded)


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"AdaBN source Pilot config does not exist: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("AdaBN source Pilot config must be a YAML mapping")
    config = dict(loaded)
    _require_equal(config.get("schema_version"), 1, "config schema_version")
    _require_equal(config.get("protocol_id"), EXPECTED_PROTOCOL_ID, "protocol_id")
    scope = config.get("scope")
    if not isinstance(scope, Mapping):
        raise TypeError("config scope must be a mapping")
    _require_equal(scope.get("paper_result"), False, "paper_result")
    _require_equal(
        scope.get("performance_metrics_computed"),
        False,
        "performance_metrics_computed",
    )
    _require_equal(scope.get("use_test_images"), False, "use_test_images")
    _require_equal(scope.get("use_test_labels"), False, "use_test_labels")
    method = config.get("method")
    if not isinstance(method, Mapping):
        raise TypeError("config method must be a mapping")
    for key, expected in (
        ("name", "AdaBN"),
        ("learnable_update", False),
        ("backward", False),
        ("optimizer", None),
        ("optimizer_steps_per_image", 0),
    ):
        _require_equal(method.get(key), expected, f"method.{key}")

    dataset = config.get("dataset")
    condition = config.get("condition")
    execution = config.get("execution")
    outputs = config.get("outputs")
    for name, value in (
        ("dataset", dataset),
        ("condition", condition),
        ("execution", execution),
        ("outputs", outputs),
    ):
        if not isinstance(value, Mapping):
            raise TypeError(f"config {name} must be a mapping")
    literals = (
        (dataset.get("name"), "IRSTD-1K", "dataset.name"),
        (dataset.get("selected_count"), 32, "dataset.selected_count"),
        (
            dataset.get("selection_rule"),
            "first_32_ranks_of_frozen_round_02_pilot_64",
            "dataset.selection_rule",
        ),
        (condition.get("corruption"), "gaussian_noise", "condition.corruption"),
        (condition.get("severity"), 3, "condition.severity"),
        (condition.get("seed"), 42, "condition.seed"),
        (
            condition.get("condition_key"),
            "gaussian_noise_S3",
            "condition.condition_key",
        ),
        (execution.get("batch_size"), 1, "execution.batch_size"),
        (execution.get("num_workers"), 0, "execution.num_workers"),
        (
            execution.get("canonical_order"),
            "selected_ids_file_order",
            "execution.canonical_order",
        ),
        (
            execution.get("order_check"),
            "canonical_then_exact_reverse",
            "execution.order_check",
        ),
        (
            execution.get("reuse_identical_materialized_inputs_across_orders"),
            True,
            "execution.reuse_identical_materialized_inputs_across_orders",
        ),
        (
            execution.get("fixed_probability_threshold"),
            0.5,
            "execution.fixed_probability_threshold",
        ),
        (
            execution.get("threshold_rule"),
            "strict_greater_than",
            "execution.threshold_rule",
        ),
        (
            outputs.get("dataset_directory"),
            "IRSTD-1K",
            "outputs.dataset_directory",
        ),
        (
            outputs.get("condition_directory"),
            "gaussian_noise_S3",
            "outputs.condition_directory",
        ),
    )
    for actual, expected, label in literals:
        _require_equal(actual, expected, label)
    return config_path, config


def resolve_paths(
    config_path: Path,
    config: Mapping[str, Any],
    output_override: Path | None = None,
) -> PilotPaths:
    dataset = config["dataset"]
    parent = config["parent_pilot"]
    condition = config["condition"]
    source = config["source"]
    output = output_override or _project_path(config["outputs"]["root"])
    if output_override is not None:
        output = output_override.expanduser().resolve()
    paths = PilotPaths(
        config=config_path,
        dataset_root=_project_path(dataset["root"]),
        train_split=_project_path(dataset["train_split"]),
        test_split=_project_path(dataset["test_split"]),
        selected_ids_file=_project_path(dataset["selected_ids_file"]),
        parent_pilot=_project_path(parent["artifact"]),
        parent_manifest=_project_path(parent["artifact_manifest"]),
        parent_complete=_project_path(parent["complete"]),
        severity_table=_project_path(condition["severity_table"]),
        checkpoint=_project_path(source["checkpoint"]),
        output_root=Path(output),
    )
    for label, path in (
        ("dataset root", paths.dataset_root),
        ("train split", paths.train_split),
        ("test split", paths.test_split),
        ("selected IDs", paths.selected_ids_file),
        ("parent Pilot", paths.parent_pilot),
        ("parent manifest", paths.parent_manifest),
        ("parent completion sentinel", paths.parent_complete),
        ("severity table", paths.severity_table),
        ("checkpoint", paths.checkpoint),
    ):
        exists = path.is_dir() if label == "dataset root" else path.is_file()
        if not exists:
            raise FileNotFoundError(f"{label} does not exist: {path}")
    return paths


def _canonical_id(identifier: str) -> str:
    return Path(identifier).with_suffix("").as_posix()


def _resolve_selected_files(
    dataset_root: Path,
    selected_ids: Sequence[str],
) -> dict[tuple[str, str], Path]:
    layout = resolve_dataset_layout(dataset_root)
    resolved: dict[tuple[str, str], Path] = {}
    for image_id in selected_ids:
        identifier = Path(image_id)
        for role, directory in (("image", layout.images_dir), ("mask", layout.masks_dir)):
            candidates = (
                [directory / identifier]
                if identifier.suffix
                else [directory / f"{image_id}{suffix}" for suffix in DEFAULT_EXTENSIONS]
            )
            matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"expected exactly one {role} for {image_id!r}, got {matches}"
                )
            resolved[(role, image_id)] = matches[0]
    return resolved


def _source_file_manifests(
    selected_ids: Sequence[str],
    paths: Mapping[tuple[str, str], Path],
) -> dict[str, str]:
    combined = hashlib.sha256()
    images = hashlib.sha256()
    masks = hashlib.sha256()
    for image_id in sorted(selected_ids):
        image_hash = source_runner.sha256_file(paths[("image", image_id)])
        mask_hash = source_runner.sha256_file(paths[("mask", image_id)])
        combined.update(f"{image_id}\t{image_hash}\t{mask_hash}\n".encode("utf-8"))
        images.update(f"{image_id}\t{image_hash}\n".encode("utf-8"))
        masks.update(f"{image_id}\t{mask_hash}\n".encode("utf-8"))
    return {
        "algorithm": "sorted-id-image_sha256-mask_sha256-lf-v1",
        "combined_sha256": combined.hexdigest(),
        "images_sha256": images.hexdigest(),
        "masks_sha256": masks.hexdigest(),
    }


def _validate_parent_pilot(
    paths: PilotPaths,
    config: Mapping[str, Any],
    selected_ids: tuple[str, ...],
) -> dict[str, Any]:
    parent_contract = config["parent_pilot"]
    _require_equal(
        source_runner.sha256_file(paths.parent_pilot),
        parent_contract["artifact_sha256"],
        "parent Pilot SHA256",
    )
    _require_equal(
        source_runner.sha256_file(paths.parent_manifest),
        parent_contract["artifact_manifest_sha256"],
        "parent Pilot manifest SHA256",
    )
    complete = _load_json(paths.parent_complete)
    _require_equal(complete.get("complete"), True, "parent Pilot complete flag")
    _require_equal(
        complete.get("pilot_json_sha256"),
        parent_contract["artifact_sha256"],
        "parent completion Pilot SHA256",
    )
    _require_equal(
        complete.get("artifact_manifest_sha256"),
        parent_contract["artifact_manifest_sha256"],
        "parent completion manifest SHA256",
    )
    parent = _load_json(paths.parent_pilot)
    _require_equal(parent.get("formal_artifact"), True, "parent formal_artifact")
    parent_ids = tuple(str(value) for value in parent["selection"]["selected_ids"])
    _require_equal(
        len(parent_ids),
        int(parent_contract["parent_selected_count"]),
        "parent selected count",
    )
    _require_equal(
        ordered_ids_sha256(parent_ids),
        parent_contract["parent_selected_ids_sha256"],
        "parent ordered IDs SHA256",
    )
    _require_equal(
        selected_ids,
        parent_ids[: len(selected_ids)],
        "selected IDs versus frozen parent prefix",
    )
    return {
        "artifact_sha256": parent_contract["artifact_sha256"],
        "artifact_manifest_sha256": parent_contract["artifact_manifest_sha256"],
        "parent_selected_count": len(parent_ids),
        "selected_prefix_count": len(selected_ids),
        "selected_prefix_exact": True,
    }


def validate_protocol_contract(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_override: Path | None = None,
) -> tuple[dict[str, Any], PilotPaths, tuple[str, ...], dict[tuple[str, str], Path], dict[str, Any]]:
    """Validate every frozen source-side input without opening test pixels."""

    resolved_config, config = load_config(config_path)
    paths = resolve_paths(resolved_config, config, output_override)
    dataset = config["dataset"]
    condition = config["condition"]
    preprocessing = config["preprocessing"]

    for path, expected, label in (
        (paths.train_split, dataset["train_split_sha256"], "train split SHA256"),
        (paths.test_split, dataset["test_split_sha256"], "test split SHA256"),
        (
            paths.selected_ids_file,
            dataset["selected_ids_file_sha256"],
            "selected IDs file SHA256",
        ),
        (paths.severity_table, condition["severity_table_sha256"], "severity table SHA256"),
        (paths.checkpoint, config["source"]["checkpoint_sha256"], "checkpoint SHA256"),
    ):
        _require_equal(source_runner.sha256_file(path), expected, label)

    selected_ids = tuple(_canonical_id(value) for value in read_split_ids(paths.selected_ids_file))
    _require_equal(len(selected_ids), int(dataset["selected_count"]), "selected count")
    _require_equal(
        ordered_ids_sha256(selected_ids),
        dataset["selected_ids_sha256"],
        "selected ordered IDs SHA256",
    )
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected source Pilot IDs are not unique")

    train_ids = tuple(_canonical_id(value) for value in read_split_ids(paths.train_split))
    test_ids = tuple(_canonical_id(value) for value in read_split_ids(paths.test_split))
    _require_equal(len(train_ids), int(dataset["train_images"]), "train image count")
    _require_equal(len(test_ids), int(dataset["test_images"]), "test image count")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("fixed train/test splits must each contain unique IDs")
    split_overlap = tuple(sorted(set(train_ids) & set(test_ids)))
    if split_overlap:
        raise ValueError(f"fixed train/test overlap: {split_overlap[:10]}")
    missing_train = tuple(value for value in selected_ids if value not in set(train_ids))
    if missing_train:
        raise ValueError(f"selected IDs missing from fixed train: {missing_train}")
    selected_test_overlap = tuple(sorted(set(selected_ids) & set(test_ids)))
    if selected_test_overlap:
        raise ValueError(f"selected IDs overlap fixed test: {selected_test_overlap}")

    parent_summary = _validate_parent_pilot(paths, config, selected_ids)
    selected_files = _resolve_selected_files(paths.dataset_root, selected_ids)
    source_manifests = _source_file_manifests(selected_ids, selected_files)
    _require_equal(
        source_manifests,
        dict(dataset["selected_source_manifests"]),
        "selected source file manifests",
    )

    table = load_severity_table(paths.severity_table)
    _require_equal(table.frozen, True, "severity table frozen flag")
    _require_equal(
        table.parameters(condition["corruption"], int(condition["severity"])),
        condition["parameters"],
        "corruption parameters",
    )
    _require_equal(
        list(preprocessing["image_resize"]["size"]), [256, 256], "image resize"
    )
    _require_equal(
        preprocessing["corruption_before_normalization"],
        True,
        "corruption-before-normalization flag",
    )
    validation = {
        "fixed_train_split_hash_verified": True,
        "fixed_test_split_metadata_hash_verified": True,
        "fixed_train_test_disjoint": True,
        "selected_ids_in_fixed_train": True,
        "selected_ids_absent_from_fixed_test": True,
        "selected_source_manifests_verified": True,
        "severity_table_frozen_and_verified": True,
        "checkpoint_hash_verified": True,
        "test_dataset_constructed": False,
        "test_images_opened": 0,
        "test_masks_opened": 0,
        "parent_pilot": parent_summary,
        "source_manifests": source_manifests,
    }
    return config, paths, selected_ids, selected_files, validation


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


def materialize_label_free_inputs(
    config: Mapping[str, Any],
    paths: PilotPaths,
    selected_ids: tuple[str, ...],
    selected_files: Mapping[tuple[str, str], Path],
) -> tuple[tuple[SourcePilotSample, ...], dict[str, Any]]:
    """Read selected train samples once; retain no mask in method-facing samples."""

    condition = config["condition"]
    image_size = int(config["preprocessing"]["image_resize"]["size"][0])
    train_ids = tuple(_canonical_id(value) for value in read_split_ids(paths.train_split))
    index_by_id = {image_id: index for index, image_id in enumerate(train_ids)}
    selected_indices = [index_by_id[image_id] for image_id in selected_ids]
    guard = SourcePilotIOGuard(selected_files)
    dataset = IRSTDResearchDataset(
        paths.dataset_root,
        split_file=paths.train_split,
        image_size=image_size,
        dataset_name=config["dataset"]["name"],
        corruption=condition["corruption"],
        severity=int(condition["severity"]),
        seed=int(condition["seed"]),
        corruption_transform=apply_corruption,
        io_access_guard=guard,
    )
    loader = DataLoader(
        Subset(dataset, selected_indices),
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    image_hasher = TensorSequenceHasher()
    mask_hasher = TensorSequenceHasher()
    samples: list[SourcePilotSample] = []
    for index, batch in enumerate(loader):
        metadata = source_runner.metadata_from_batch(batch)
        image_id = str(metadata["image_id"])
        _require_equal(image_id, selected_ids[index], f"materialized ID at {index}")
        image = batch["image"].detach().cpu().clone()
        source_train_mask = batch["mask"].detach().cpu()
        image_hasher.update(image_id, image[0])
        mask_hasher.update(image_id, source_train_mask[0])
        samples.append(
            SourcePilotSample(
                image_id=image_id,
                image=image,
                metadata=dict(metadata),
                input_raw_sha256=_raw_array_sha256(image[0].numpy()),
            )
        )
        # The source-train mask is deliberately not stored. It cannot cross the
        # EpisodicRunner label firewall and cannot affect any smoke gate.
        del source_train_mask

    input_hash = image_hasher.hexdigest()
    mask_hash = mask_hasher.hexdigest()
    _require_equal(
        input_hash,
        config["preprocessing"]["expected_input_tensor_sequence_sha256"],
        "materialized corrupted input tensor sequence SHA256",
    )
    _require_equal(
        mask_hash,
        config["preprocessing"]["expected_mask_tensor_sequence_sha256"],
        "source-train mask tensor sequence SHA256",
    )
    io_summary = guard.summary()
    expected_count = len(selected_ids)
    for key, expected in (
        ("allowed_unique_ids", expected_count),
        ("opened_unique_ids", expected_count),
        ("image_open_count", expected_count),
        ("mask_open_count", expected_count),
        ("forbidden_open_count", 0),
    ):
        _require_equal(io_summary[key], expected, f"I/O guard {key}")
    return tuple(samples), {
        "input_tensor_sequence_sha256": input_hash,
        "source_train_mask_tensor_sequence_sha256": mask_hash,
        "source_train_mask_retained_after_hashing": False,
        "method_received_mask": False,
        "io_guard": io_summary,
    }


def _fingerprint_dict(value: StateFingerprint) -> dict[str, Any]:
    return asdict(value)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    os.replace(temporary, path)


def _atomic_save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(array, mode="L").save(temporary, format="PNG")
    os.replace(temporary, path)


def _episode_invariants(result: EpisodeResult, expected_bn_count: int) -> dict[str, bool]:
    source = result.source_fingerprint
    prepared = result.state_after_prepare_fingerprint
    adapted = result.state_after_adapt_fingerprint
    post = result.state_after_post_fingerprint
    expected_changes = ("runtime",)
    checks = {
        "input_unchanged": result.input_unchanged,
        "optimizer_steps_zero": result.outcome.optimizer_steps == 0,
        "learnable_update_false": result.outcome.diagnostics.get("learnable_update") is False,
        "expected_batchnorm_count": int(
            result.outcome.diagnostics.get("batchnorm2d_modules", -1)
        )
        == expected_bn_count,
        "only_runtime_changed_after_prepare": result.state_changes_after_prepare
        == expected_changes,
        "only_runtime_changed_after_adapt": result.state_changes_after_adapt
        == expected_changes,
        "only_runtime_changed_after_post": result.state_changes_after_post
        == expected_changes,
        "parameters_and_buffers_unchanged_after_prepare": prepared.model_sha256
        == source.model_sha256,
        "parameters_and_buffers_unchanged_after_adapt": adapted.model_sha256
        == source.model_sha256,
        "parameters_and_buffers_unchanged_after_post": post.model_sha256
        == source.model_sha256,
        "optimizer_absent": source.optimizer_sha256 is None
        and prepared.optimizer_sha256 is None
        and adapted.optimizer_sha256 is None
        and post.optimizer_sha256 is None,
        "gradients_unchanged": prepared.gradients_sha256
        == adapted.gradients_sha256
        == post.gradients_sha256
        == source.gradients_sha256,
        "topology_unchanged": prepared.topology_sha256
        == adapted.topology_sha256
        == post.topology_sha256
        == source.topology_sha256,
        "state_frozen_after_prepare": prepared == adapted == post,
        "runtime_differs_from_source": prepared.runtime_sha256 != source.runtime_sha256,
        "reset_matches_source": result.reset_fingerprint == source,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(f"AdaBN episode invariant failures: {failed}")
    return checks


def execute_episode_order_gate(
    *,
    samples: Sequence[SourcePilotSample],
    adapter: IRSTDModelAdapter,
    runner: EpisodicRunner,
    condition_dir: Path,
    expected_bn_count: int,
    expected_spatial_size: int,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Save canonical outputs and prove identical logits in exact reverse order."""

    if not samples:
        raise ValueError("source Pilot samples cannot be empty")
    if threshold != 0.5:
        raise ValueError("AdaBN source Pilot requires the frozen threshold 0.5")
    method = AdaBNMethod()
    device = next(adapter.model.parameters()).device
    canonical: dict[str, EpisodeResult] = {}
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    record_by_id: dict[str, dict[str, Any]] = {}
    source_probability_hasher = TensorSequenceHasher()
    adabn_probability_hasher = TensorSequenceHasher()
    changed_images = 0
    global_max_abs_change = 0.0
    absolute_change_sum = 0.0
    absolute_change_values = 0

    condition_dir.mkdir(parents=True, exist_ok=False)
    for index, sample in enumerate(samples):
        result = runner.run_one_image(
            image=sample.image.to(device, non_blocking=False),
            metadata=sample.metadata,
            method=method,
        )
        checks = _episode_invariants(result, expected_bn_count)
        source_probability = adapter.logits_to_prob(result.logits_pre)[0, 0].numpy().astype(
            np.float32, copy=False
        )
        adabn_probability = adapter.logits_to_prob(result.logits_post)[0, 0].numpy().astype(
            np.float32, copy=False
        )
        expected_shape = (expected_spatial_size, expected_spatial_size)
        if source_probability.shape != expected_shape or adabn_probability.shape != expected_shape:
            raise RuntimeError(
                f"probability shape drift for {sample.image_id}: "
                f"{source_probability.shape}/{adabn_probability.shape}"
            )
        if not np.isfinite(source_probability).all() or not np.isfinite(adabn_probability).all():
            raise RuntimeError(f"non-finite probability for {sample.image_id}")
        difference = np.abs(adabn_probability - source_probability)
        probability_equal = bool(np.array_equal(source_probability, adabn_probability))
        if not probability_equal:
            changed_images += 1
        max_abs_change = float(difference.max())
        mean_abs_change = float(difference.mean(dtype=np.float64))
        global_max_abs_change = max(global_max_abs_change, max_abs_change)
        absolute_change_sum += float(difference.sum(dtype=np.float64))
        absolute_change_values += int(difference.size)

        source_mask = np.where(source_probability > threshold, 255, 0).astype(np.uint8)
        adabn_mask = np.where(adabn_probability > threshold, 255, 0).astype(np.uint8)
        stem = source_runner.safe_artifact_stem(sample.image_id)
        paths = {
            "source_probability_map": Path("source_probability_maps_256") / f"{stem}.npy",
            "adabn_probability_map": Path("probability_maps_256") / f"{stem}.npy",
            "source_prediction_mask": Path("source_prediction_masks_256") / f"{stem}.png",
            "adabn_prediction_mask": Path("prediction_masks_256") / f"{stem}.png",
        }
        _atomic_save_npy(condition_dir / paths["source_probability_map"], source_probability)
        _atomic_save_npy(condition_dir / paths["adabn_probability_map"], adabn_probability)
        _atomic_save_png(condition_dir / paths["source_prediction_mask"], source_mask)
        _atomic_save_png(condition_dir / paths["adabn_prediction_mask"], adabn_mask)
        source_probability_hasher.update(sample.image_id, source_probability)
        adabn_probability_hasher.update(sample.image_id, adabn_probability)

        record = {
            "index": index,
            "image_id": sample.image_id,
            "metadata": dict(result.metadata),
            "input_raw_sha256": sample.input_raw_sha256,
            "source_probability_map": str(paths["source_probability_map"]),
            "source_probability_map_sha256": source_runner.sha256_file(
                condition_dir / paths["source_probability_map"]
            ),
            "adabn_probability_map": str(paths["adabn_probability_map"]),
            "adabn_probability_map_sha256": source_runner.sha256_file(
                condition_dir / paths["adabn_probability_map"]
            ),
            "source_prediction_mask": str(paths["source_prediction_mask"]),
            "source_prediction_mask_sha256": source_runner.sha256_file(
                condition_dir / paths["source_prediction_mask"]
            ),
            "adabn_prediction_mask": str(paths["adabn_prediction_mask"]),
            "adabn_prediction_mask_sha256": source_runner.sha256_file(
                condition_dir / paths["adabn_prediction_mask"]
            ),
            "source_probability_raw_sha256": _raw_array_sha256(source_probability),
            "adabn_probability_raw_sha256": _raw_array_sha256(adabn_probability),
            "source_probability_min": float(source_probability.min()),
            "source_probability_max": float(source_probability.max()),
            "adabn_probability_min": float(adabn_probability.min()),
            "adabn_probability_max": float(adabn_probability.max()),
            "source_adabn_probability_bit_exact": probability_equal,
            "source_adabn_max_abs_probability_change": max_abs_change,
            "source_adabn_mean_abs_probability_change": mean_abs_change,
            "binary_mask_changed_pixels": int(np.count_nonzero(source_mask != adabn_mask)),
            "canonical_reverse_source_logits_bit_exact": None,
            "canonical_reverse_adabn_logits_bit_exact": None,
            "canonical_reverse_state_bit_exact": None,
            "invariants": checks,
        }
        diagnostic = {
            "index": index,
            "image_id": sample.image_id,
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
        }
        canonical[sample.image_id] = result
        records.append(record)
        diagnostics.append(diagnostic)
        record_by_id[sample.image_id] = record

    if changed_images < 1 or global_max_abs_change <= 0.0:
        raise RuntimeError(
            "AdaBN matched Source on every probability map; batch statistics may not be active"
        )

    order_exact_count = 0
    for sample in reversed(samples):
        reverse = runner.run_one_image(
            image=sample.image.to(device, non_blocking=False),
            metadata=sample.metadata,
            method=method,
        )
        _episode_invariants(reverse, expected_bn_count)
        forward = canonical[sample.image_id]
        source_exact = bool(torch.equal(forward.logits_pre, reverse.logits_pre))
        adabn_exact = bool(torch.equal(forward.logits_post, reverse.logits_post))
        state_exact = all(
            (
                forward.state_after_prepare_fingerprint
                == reverse.state_after_prepare_fingerprint,
                forward.state_after_adapt_fingerprint
                == reverse.state_after_adapt_fingerprint,
                forward.state_after_post_fingerprint
                == reverse.state_after_post_fingerprint,
                forward.reset_fingerprint == reverse.reset_fingerprint,
            )
        )
        if not source_exact or not adabn_exact or not state_exact:
            raise RuntimeError(f"canonical/reverse order drift for {sample.image_id}")
        order_exact_count += 1
        record = record_by_id[sample.image_id]
        record["canonical_reverse_source_logits_bit_exact"] = source_exact
        record["canonical_reverse_adabn_logits_bit_exact"] = adabn_exact
        record["canonical_reverse_state_bit_exact"] = state_exact

    source_runner.write_jsonl_atomic(condition_dir / "per_image.jsonl", records)
    source_runner.write_jsonl_atomic(
        condition_dir / "adaptation_diagnostics.jsonl", diagnostics
    )
    output_counts = {
        "source_probability_maps": len(
            tuple((condition_dir / "source_probability_maps_256").glob("*.npy"))
        ),
        "adabn_probability_maps": len(
            tuple((condition_dir / "probability_maps_256").glob("*.npy"))
        ),
        "source_prediction_masks": len(
            tuple((condition_dir / "source_prediction_masks_256").glob("*.png"))
        ),
        "adabn_prediction_masks": len(
            tuple((condition_dir / "prediction_masks_256").glob("*.png"))
        ),
    }
    all_outputs_saved = all(count == len(samples) for count in output_counts.values())
    if not all_outputs_saved:
        raise RuntimeError(f"incomplete canonical output set: {output_counts}")
    checks = {
        "paper_result": False,
        "performance_metrics_computed": False,
        "method_received_mask": False,
        "finite_source_and_adabn_outputs": True,
        "all_parameters_and_buffers_unchanged_before_reset": True,
        "all_bn_running_mean_unchanged": True,
        "all_bn_running_var_unchanged": True,
        "all_bn_num_batches_tracked_unchanged": True,
        "only_runtime_state_changes_during_adabn": True,
        "no_gradients_or_optimizer": True,
        "non_batchnorm_modules_remain_eval": True,
        "per_image_reset_matches_source": True,
        "canonical_reverse_logits_bit_exact": order_exact_count == len(samples),
        "at_least_one_probability_map_differs_from_source": changed_images >= 1,
        "all_source_and_adabn_outputs_saved": all_outputs_saved,
    }
    metrics = {
        "schema_version": 1,
        "method": "AdaBN",
        "scope": "source_domain_implementation_smoke",
        "paper_result": False,
        "performance_metrics_computed": False,
        "evaluated_images": len(samples),
        "execution_orders": ["canonical", "exact_reverse"],
        "probability_transform": "sigmoid_once",
        "fixed_probability_threshold": threshold,
        "threshold_rule": "strict_greater_than",
        "behavioral_diagnostics": {
            "source_adabn_probability_changed_images": changed_images,
            "source_adabn_probability_unchanged_images": len(samples) - changed_images,
            "source_adabn_global_max_abs_probability_change": global_max_abs_change,
            "source_adabn_global_mean_abs_probability_change": (
                absolute_change_sum / absolute_change_values
            ),
            "canonical_reverse_exact_images": order_exact_count,
            "source_probability_tensor_sequence_sha256": source_probability_hasher.hexdigest(),
            "adabn_probability_tensor_sequence_sha256": adabn_probability_hasher.hexdigest(),
            "saved_output_counts": output_counts,
        },
        "checks": checks,
        "forbidden_label_metrics": {
            "iou": "not_computed",
            "niou": "not_computed",
            "pd": "not_computed",
            "fa": "not_computed",
            "froc": "not_computed",
        },
    }
    source_runner.write_json_atomic(condition_dir / "metrics.json", metrics)
    return {"metrics": metrics, "records": records, "diagnostics": diagnostics}


def _checkpoint_summary(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("source checkpoint must be a metadata mapping")
    expected = config["source"]
    for key, value in (
        ("architecture", expected["architecture"]),
        ("dataset", config["dataset"]["name"]),
        ("epoch", int(expected["checkpoint_epoch"])),
        ("selection_metric", expected["checkpoint_selection_metric"]),
        ("test_selected", bool(expected["checkpoint_test_selected"])),
    ):
        _require_equal(payload.get(key), value, f"checkpoint metadata {key}")
    if "state_dict" not in payload:
        raise KeyError("source checkpoint is missing state_dict")
    return {
        "architecture": payload["architecture"],
        "dataset": payload["dataset"],
        "epoch": int(payload["epoch"]),
        "selection_metric": payload["selection_metric"],
        "selection_rule": payload.get("selection_rule"),
        "test_selected": bool(payload["test_selected"]),
    }


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


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
    aggregate_checks: Mapping[str, Any],
) -> dict[str, Any]:
    """Write manifest/COMPLETE and atomically publish only after every gate."""

    failed = sorted(name for name, passed in aggregate_checks.items() if passed is not True)
    if failed:
        raise RuntimeError(f"refusing COMPLETE because gates failed: {failed}")
    required = {
        "run_config.yaml",
        "provenance.json",
        "aggregate_metrics.json",
        "IRSTD-1K/gaussian_noise_S3/metrics.json",
        "IRSTD-1K/gaussian_noise_S3/per_image.jsonl",
        "IRSTD-1K/gaussian_noise_S3/adaptation_diagnostics.jsonl",
    }
    present = {str(path.relative_to(staging)) for path in staging.rglob("*") if path.is_file()}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"refusing COMPLETE because required artifacts are missing: {missing}")
    files = _artifact_files(staging)
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "paper_result": False,
        "performance_metrics_computed": False,
        "files": files,
    }
    manifest_path = staging / "artifact_manifest.json"
    source_runner.write_json_atomic(manifest_path, manifest)
    completion = {
        "complete": True,
        "protocol_id": protocol_id,
        "protocol_sha256": protocol_sha256,
        "paper_result": False,
        "performance_metrics_computed": False,
        "all_required_gates_passed": True,
        "artifact_manifest_sha256": source_runner.sha256_file(manifest_path),
        "aggregate_metrics_sha256": files["aggregate_metrics.json"]["sha256"],
    }
    source_runner.write_json_atomic(staging / "COMPLETE.json", completion)
    if final_root.exists():
        raise FileExistsError(f"output already exists: {final_root}")
    os.replace(staging, final_root)
    return completion


def run_source_pilot(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config, paths, selected_ids, selected_files, contract = validate_protocol_contract(
        args.config,
        output_override=args.output_dir,
    )
    if paths.output_root.exists():
        raise FileExistsError(f"AdaBN source Pilot output already exists: {paths.output_root}")
    paths.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = paths.output_root.with_name(
        f".{paths.output_root.name}.build-{os.getpid()}"
    )
    if staging.exists():
        raise FileExistsError(f"AdaBN source Pilot staging directory exists: {staging}")

    source_runner.seed_everything(int(config["condition"]["seed"]))
    samples, materialization = materialize_label_free_inputs(
        config, paths, selected_ids, selected_files
    )
    checkpoint_summary = _checkpoint_summary(paths.checkpoint, config)
    device = source_runner.resolve_device(args.device)
    model = source_runner.build_nsfpn_model()
    checkpoint_wrapper = source_runner.load_trusted_checkpoint(model, paths.checkpoint)
    model.to(device)
    adapter = IRSTDModelAdapter(model, warm_flag=bool(config["source"]["warm_flag"]))
    adapter.set_source_eval_mode()
    state = EpisodicStateManager(model, optimizer=None)
    runner = EpisodicRunner(adapter, state)
    batchnorm_count = sum(isinstance(module, nn.BatchNorm2d) for module in model.modules())
    _require_equal(
        batchnorm_count,
        int(config["execution"]["expected_batchnorm2d_modules"]),
        "real NS-FPN BatchNorm2d count",
    )

    staging.mkdir(parents=False)
    condition_dir = (
        staging
        / config["outputs"]["dataset_directory"]
        / config["outputs"]["condition_directory"]
    )
    episode_result = execute_episode_order_gate(
        samples=samples,
        adapter=adapter,
        runner=runner,
        condition_dir=condition_dir,
        expected_bn_count=batchnorm_count,
        expected_spatial_size=int(config["preprocessing"]["image_resize"]["size"][0]),
        threshold=float(config["execution"]["fixed_probability_threshold"]),
    )
    state.assert_source_state()

    effective_config = deepcopy(config)
    effective_config["runtime"] = {
        "source_config": str(paths.config),
        "source_config_sha256": source_runner.sha256_file(paths.config),
        "device": str(device),
        "output_root": str(paths.output_root),
    }
    _write_yaml_atomic(staging / "run_config.yaml", effective_config)
    provenance = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "paper_result": False,
        "performance_metrics_computed": False,
        "config": str(paths.config),
        "config_sha256": source_runner.sha256_file(paths.config),
        "selected_ids_file": str(paths.selected_ids_file),
        "selected_ids_file_sha256": source_runner.sha256_file(paths.selected_ids_file),
        "selected_ids_sha256": ordered_ids_sha256(selected_ids),
        "selected_ids": list(selected_ids),
        "fixed_train_split": str(paths.train_split),
        "fixed_train_split_sha256": source_runner.sha256_file(paths.train_split),
        "fixed_test_boundary": {
            "split_file": str(paths.test_split),
            "split_sha256": source_runner.sha256_file(paths.test_split),
            "metadata_read_for_leakage_guard_only": True,
            "test_dataset_constructed": False,
            "test_images_opened": 0,
            "test_masks_opened": 0,
        },
        "checkpoint": str(paths.checkpoint),
        "checkpoint_sha256": source_runner.sha256_file(paths.checkpoint),
        "checkpoint_wrapper": checkpoint_wrapper,
        "checkpoint_metadata": checkpoint_summary,
        "checkpoint_test_selected_disclosure": config["source"]["checkpoint_disclosure"],
        "parent_pilot": contract["parent_pilot"],
        "selected_source_manifests": contract["source_manifests"],
        "materialized_inputs": materialization,
        "severity_table": str(paths.severity_table),
        "severity_table_sha256": source_runner.sha256_file(paths.severity_table),
        "condition": dict(config["condition"]),
        "repository": source_runner.repository_provenance(PROVENANCE_PATHS),
    }
    source_runner.write_json_atomic(staging / "provenance.json", provenance)

    condition_checks = episode_result["metrics"]["checks"]
    aggregate_checks = {
        "exact_32_parent_pilot_prefix": bool(
            contract["parent_pilot"]["selected_prefix_exact"]
        ),
        "zero_fixed_test_id_overlap": bool(contract["selected_ids_absent_from_fixed_test"]),
        "zero_test_image_or_mask_opens": contract["test_images_opened"] == 0
        and contract["test_masks_opened"] == 0,
        "exact_corrupted_input_hash": materialization["input_tensor_sequence_sha256"]
        == config["preprocessing"]["expected_input_tensor_sequence_sha256"],
        "method_received_no_mask": materialization["method_received_mask"] is False,
        "performance_metrics_not_computed": condition_checks[
            "performance_metrics_computed"
        ]
        is False,
        "finite_source_and_adabn_outputs": condition_checks[
            "finite_source_and_adabn_outputs"
        ],
        "all_parameters_and_buffers_unchanged_before_reset": condition_checks[
            "all_parameters_and_buffers_unchanged_before_reset"
        ],
        "all_bn_running_buffers_unchanged": all(
            (
                condition_checks["all_bn_running_mean_unchanged"],
                condition_checks["all_bn_running_var_unchanged"],
                condition_checks["all_bn_num_batches_tracked_unchanged"],
            )
        ),
        "only_runtime_state_changes_during_adabn": condition_checks[
            "only_runtime_state_changes_during_adabn"
        ],
        "no_gradients_or_optimizer": condition_checks["no_gradients_or_optimizer"],
        "non_batchnorm_modules_remain_eval": condition_checks[
            "non_batchnorm_modules_remain_eval"
        ],
        "per_image_reset_matches_source": condition_checks[
            "per_image_reset_matches_source"
        ],
        "canonical_reverse_logits_bit_exact": condition_checks[
            "canonical_reverse_logits_bit_exact"
        ],
        "at_least_one_probability_map_differs_from_source": condition_checks[
            "at_least_one_probability_map_differs_from_source"
        ],
        "all_source_and_adabn_outputs_saved": condition_checks[
            "all_source_and_adabn_outputs_saved"
        ],
    }
    aggregate = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "method": "AdaBN",
        "scope": "source_domain_implementation_smoke",
        "paper_result": False,
        "performance_metrics_computed": False,
        "dataset": config["dataset"]["name"],
        "condition_key": config["condition"]["condition_key"],
        "selected_images": len(samples),
        "episode_executions": len(samples) * 2,
        "condition_metrics": str(
            (condition_dir / "metrics.json").relative_to(staging)
        ),
        "behavioral_diagnostics": episode_result["metrics"]["behavioral_diagnostics"],
        "checks": aggregate_checks,
        "runtime_seconds_before_publication": time.perf_counter() - started,
    }
    source_runner.write_json_atomic(staging / "aggregate_metrics.json", aggregate)
    completion = finalize_and_publish(
        staging=staging,
        final_root=paths.output_root,
        protocol_id=config["protocol_id"],
        protocol_sha256=source_runner.sha256_file(paths.config),
        aggregate_checks=aggregate_checks,
    )
    return {
        "published_output_dir": str(paths.output_root),
        "aggregate": aggregate,
        "completion": completion,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_source_pilot(args)
    print(f"Artifacts: {result['published_output_dir']}")
    diagnostics = result["aggregate"]["behavioral_diagnostics"]
    print(
        "AdaBN source-domain Pilot passed: "
        f"{result['aggregate']['selected_images']} images, canonical+reverse, "
        f"changed probability maps="
        f"{diagnostics['source_adabn_probability_changed_images']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CONFIG",
    "PilotPaths",
    "SourcePilotIOGuard",
    "SourcePilotSample",
    "execute_episode_order_gate",
    "finalize_and_publish",
    "load_config",
    "materialize_label_free_inputs",
    "run_source_pilot",
    "validate_protocol_contract",
]
