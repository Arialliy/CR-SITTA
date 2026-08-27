"""Run the fail-closed fixed-train corruption severity Pilot.

The CLI evaluates one SHA256-ranked subset from the fixed *train* split
under clean input and the four configured corruptions at severities S1--S5.
It is diagnostic only: this runner never edits ``severity_tables.yaml``.  The
formal mode reads the fixed test split *text* solely to prove ID disjointness;
it never constructs a test dataset or opens a test image/mask.  The locked
``best_miou`` checkpoint is nevertheless disclosed as test-selected by the
completed source-training protocol.

The real NS-FPN model is constructed lazily through :mod:`test_source`, so
importing this module and exercising its pure/fake-model tests does not require
the custom SFS CUDA extension.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset
import yaml

from corruptions.corruption_protocol import (
    NON_CLEAN_CORRUPTIONS,
    NON_CLEAN_SEVERITIES,
    get_default_severity_table,
    validate_corruption_request,
)
from corruptions.infrared_corruptions import apply_corruption
from dataio.research_dataset import (
    DEFAULT_EXTENSIONS,
    IRSTDResearchDataset,
    read_split_ids,
    resolve_dataset_layout,
)
from metrics.irstd_metrics import (
    IRSTDEvaluationProtocol,
    IRSTDEvaluationResult,
    UnifiedResearchEvaluator,
)
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "corruption_pilot_fixed_splits.yaml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "corruption_pilot_fixed_split"
DEFAULT_SUBSET_SIZE = 64
DEFAULT_SEED = source_runner.SEED
PILOT_PROVENANCE_PATHS = (
    "analyze_corruption_pilots.py",
    "configs/corruption_pilot_acceptance.yaml",
    "configs/corruption_pilot_fixed_splits.yaml",
    "configs/retrain_fixed_splits.yaml",
    "corruptions/corruption_protocol.py",
    "corruptions/infrared_corruptions.py",
    "corruptions/severity_tables.yaml",
    "dataio/research_dataset.py",
    "metrics/connected_components.py",
    "metrics/irstd_metrics.py",
    "metrics/target_matching.py",
    "model/MSHNet_NSFPN.py",
    "run_corruption_pilot.py",
    "test_fixed_split_source.py",
    "test_source.py",
    "tta/model_adapter.py",
)

PILOT_DATASET_DEFAULTS = {
    dataset_name: {
        "root": PROJECT_ROOT / "datasets" / dataset_name,
        "split": (
            PROJECT_ROOT
            / "datasets"
            / dataset_name
            / "img_idx"
            / f"train_{dataset_name}.txt"
        ),
        "test_split": (
            PROJECT_ROOT
            / "datasets"
            / dataset_name
            / "img_idx"
            / f"test_{dataset_name}.txt"
        ),
        "checkpoint": (
            PROJECT_ROOT
            / "results"
            / "retraining_fixed_split"
            / dataset_name
            / "best_miou.pth.tar"
        ),
    }
    for dataset_name in ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
}

DEFAULT_CONDITIONS: tuple[tuple[str, int], ...] = (
    ("clean", 0),
    *tuple(
        (corruption, severity)
        for corruption in NON_CLEAN_CORRUPTIONS
        for severity in NON_CLEAN_SEVERITIES
    ),
)


@dataclass(frozen=True)
class RankedImage:
    image_id: str
    image_id_sha256: str


@dataclass(frozen=True)
class PilotPaths:
    protocol: Path
    dataset_root: Path
    split_file: Path
    test_split_file: Path
    checkpoint: Path
    output_dir: Path


def uint64_integer(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if not 0 <= value < 2**64:
        raise argparse.ArgumentTypeError("value must be in [0, 2**64)")
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-train corruption severity pilot; never evaluates the "
            "fixed test split."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(PILOT_DATASET_DEFAULTS),
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Formal default is the dataset root frozen by the Pilot protocol.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help=(
            "Source-domain split (default: fixed train_<dataset>.txt); test files "
            "and any split containing fixed test IDs are refused."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Locked source checkpoint (default: retrained best_miou checkpoint).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--subset-size",
        type=source_runner.positive_integer,
        default=DEFAULT_SUBSET_SIZE,
        help="Formal mode requires exactly 64 SHA256-ranked train images.",
    )
    parser.add_argument(
        "--image-size",
        type=source_runner.positive_integer,
        default=source_runner.IMAGE_SIZE,
    )
    parser.add_argument("--seed", type=uint64_integer, default=DEFAULT_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: results/corruption_pilot_fixed_split/<dataset>.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Allow reduced/custom inputs for testing. Smoke artifacts are marked "
            "non-formal and cannot satisfy the Pilot contract."
        ),
    )
    return parser


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_pilot_protocol(
    protocol_path: str | Path,
    dataset_name: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(protocol_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pilot protocol does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("Pilot protocol must be a YAML mapping")
    protocol = dict(loaded)
    if protocol.get("protocol_id") != "nsfpn-fixed-split-corruption-pilot-v1":
        raise ValueError("unexpected Pilot protocol_id")
    datasets = protocol.get("datasets")
    if not isinstance(datasets, Mapping) or dataset_name not in datasets:
        available = ", ".join(sorted(datasets or {}))
        raise ValueError(
            f"unknown dataset {dataset_name!r}; expected one of {available}"
        )
    dataset = datasets[dataset_name]
    if not isinstance(dataset, Mapping):
        raise TypeError(f"Pilot dataset entry {dataset_name!r} must be a mapping")
    return path, protocol, dict(dataset)


def resolve_pilot_paths(
    args: argparse.Namespace,
    dataset_contract: Mapping[str, Any] | None = None,
) -> PilotPaths:
    protocol_path = Path(getattr(args, "protocol", DEFAULT_PROTOCOL)).expanduser().resolve()
    if dataset_contract is None:
        _, _, dataset_contract = load_pilot_protocol(protocol_path, args.dataset)
    dataset_root = _project_path(args.root or dataset_contract["root"])
    split_file = _project_path(args.split or dataset_contract["train_split"])
    test_split_file = _project_path(dataset_contract["test_split"])
    checkpoint = _project_path(args.checkpoint or dataset_contract["checkpoint"])
    output_dir = (
        args.output_dir
        or DEFAULT_OUTPUT_ROOT / args.dataset
    ).expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    if not test_split_file.is_file():
        raise FileNotFoundError(f"Fixed test split does not exist: {test_split_file}")
    if split_file == test_split_file or split_file.name.casefold() in {
        "test.txt",
        f"test_{args.dataset}.txt".casefold(),
    }:
        raise ValueError(
            "Corruption Pilot refuses a fixed test split; use the fixed train split."
        )
    return PilotPaths(
        protocol_path,
        dataset_root,
        split_file,
        test_split_file,
        checkpoint,
        output_dir,
    )


def _canonical_dataset_image_id(identifier: str) -> str:
    return Path(identifier).with_suffix("").as_posix()


def fixed_test_id_overlap(
    dataset_name: str,
    candidate_image_ids: Sequence[str],
    *,
    test_split: Path | None = None,
) -> tuple[str, ...]:
    """Return sorted candidate IDs that occur in the frozen fixed test split."""

    test_split = test_split or Path(PILOT_DATASET_DEFAULTS[dataset_name]["test_split"])
    test_ids = {
        _canonical_dataset_image_id(identifier)
        for identifier in read_split_ids(test_split)
    }
    return tuple(sorted(set(candidate_image_ids) & test_ids))

def sha256_ranked_subset(
    image_ids: Sequence[str],
    limit: int = DEFAULT_SUBSET_SIZE,
) -> tuple[RankedImage, ...]:
    """Select IDs by ascending SHA256(UTF-8 image_id), independent of file order."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if not image_ids:
        raise ValueError("image_ids cannot be empty")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("image_ids must be unique before SHA256 ranking")

    ranked = [
        RankedImage(
            image_id=image_id,
            image_id_sha256=hashlib.sha256(image_id.encode("utf-8")).hexdigest(),
        )
        for image_id in image_ids
    ]
    ranked.sort(key=lambda item: (item.image_id_sha256, item.image_id))
    return tuple(ranked[: min(limit, len(ranked))])


def sequence_sha256(values: Sequence[str]) -> str:
    """Hash an ordered string sequence using canonical compact JSON UTF-8."""

    payload = json.dumps(
        list(values), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalise_conditions(
    conditions: Sequence[tuple[str, int]] | None,
    *,
    require_full: bool = False,
) -> tuple[tuple[str, int], ...]:
    selected = DEFAULT_CONDITIONS if conditions is None else tuple(conditions)
    if not selected:
        raise ValueError("at least one pilot condition is required")
    canonical = tuple(validate_corruption_request(*condition) for condition in selected)
    if len(set(canonical)) != len(canonical):
        raise ValueError("pilot conditions must be unique")
    if canonical[0] != ("clean", 0) or canonical.count(("clean", 0)) != 1:
        raise ValueError("pilot conditions must contain clean/0 exactly once and first")
    if require_full and canonical != DEFAULT_CONDITIONS:
        raise ValueError(
            "formal Pilot requires the exact ordered clean + 4 corruptions x S1-S5 "
            "contract (21 conditions)"
        )
    return canonical


def _load_checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("formal Pilot checkpoint must be a metadata mapping")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected {expected!r}, got {actual!r}")


def validate_checkpoint_contract(
    path: Path,
    *,
    dataset_name: str,
    dataset_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Fail closed unless ``path`` is the exact disclosed best_miou checkpoint."""

    checkpoint_hash = source_runner.sha256_file(path)
    _require_equal(
        checkpoint_hash,
        str(dataset_contract["checkpoint_sha256"]),
        "checkpoint SHA256",
    )
    payload = _load_checkpoint_payload(path)
    required = {
        "schema_version",
        "architecture",
        "dataset",
        "epoch",
        "selection_metric",
        "selection_rule",
        "selection_value",
        "test_metrics",
        "test_selected",
        "split_manifest",
        "run_config",
        "state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"checkpoint is missing formal metadata: {missing}")

    _require_equal(int(payload["schema_version"]), 1, "checkpoint schema_version")
    _require_equal(payload["architecture"], protocol["source"]["architecture"], "architecture")
    _require_equal(payload["dataset"], dataset_name, "checkpoint dataset")
    _require_equal(int(payload["epoch"]), int(dataset_contract["checkpoint_epoch"]), "checkpoint epoch")
    if not 500 <= int(payload["epoch"]) <= 1000:
        raise ValueError("best_miou checkpoint epoch must be in the frozen 500..1000 selection window")
    _require_equal(payload["selection_metric"], "miou", "checkpoint selection_metric")
    _require_equal(
        payload["selection_rule"],
        "maximize_miou_then_pd_then_minimize_fa",
        "checkpoint selection_rule",
    )
    if payload["test_selected"] is not True:
        raise ValueError("checkpoint must explicitly disclose test_selected=true")

    metrics = payload["test_metrics"]
    split_manifest = payload["split_manifest"]
    run_config = payload["run_config"]
    if not all(isinstance(value, Mapping) for value in (metrics, split_manifest, run_config)):
        raise TypeError("checkpoint test_metrics, split_manifest and run_config must be mappings")
    for name in ("miou", "pd", "fa_per_pixel_x1e6"):
        if name not in metrics or not np.isfinite(float(metrics[name])):
            raise ValueError(f"checkpoint metric {name!r} is missing or non-finite")
    _require_equal(float(payload["selection_value"]), float(metrics["miou"]), "selection value")
    _require_equal(int(metrics["epoch"]), int(payload["epoch"]), "checkpoint metric epoch")
    _require_equal(int(metrics["images"]), int(dataset_contract["test_images"]), "checkpoint metric image count")

    expected_split = {
        "train_count": int(dataset_contract["train_images"]),
        "test_count": int(dataset_contract["test_images"]),
        "overlap_count": 0,
        "train_split_sha256": str(dataset_contract["train_split_sha256"]),
        "test_split_sha256": str(dataset_contract["test_split_sha256"]),
        "corpus_manifest_algorithm": "sorted-id-image_sha256-mask_sha256-lf-v1",
        "corpus_manifest_sha256": str(dataset_contract["corpus_manifest_sha256"]),
    }
    for key, expected in expected_split.items():
        _require_equal(split_manifest.get(key), expected, f"checkpoint split_manifest.{key}")

    _require_equal(
        run_config.get("protocol_id"),
        "nsfpn-fixed-train-test-1000e-v1",
        "parent training protocol_id",
    )
    _require_equal(
        run_config.get("protocol_sha256"),
        protocol["source"]["parent_training_protocol_sha256"],
        "parent training protocol SHA256",
    )
    _require_equal(run_config.get("base_size"), 256, "checkpoint base_size")
    _require_equal(run_config.get("seed"), 42, "checkpoint training seed")
    _require_equal(run_config.get("max_train_batches"), None, "checkpoint max_train_batches")
    _require_equal(run_config.get("max_test_images"), None, "checkpoint max_test_images")
    _require_equal(
        run_config.get("test_selected_checkpoints"),
        True,
        "checkpoint test_selected_checkpoints",
    )
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint state_dict must be a non-empty mapping")
    if not all(torch.is_tensor(value) for value in state_dict.values()):
        raise TypeError("checkpoint state_dict must contain only tensors")

    summary = {
        "schema_version": int(payload["schema_version"]),
        "architecture": str(payload["architecture"]),
        "dataset": str(payload["dataset"]),
        "epoch": int(payload["epoch"]),
        "selection_metric": str(payload["selection_metric"]),
        "selection_rule": str(payload["selection_rule"]),
        "selection_value": float(payload["selection_value"]),
        "test_selected": True,
        "test_metrics": dict(metrics),
        "split_manifest": dict(split_manifest),
        "state_dict_tensor_count": len(state_dict),
    }
    return payload, summary


def build_evaluation_protocol(protocol: Mapping[str, Any]) -> IRSTDEvaluationProtocol:
    config = protocol["evaluation"]
    connectivity_2d = int(config["foreground_connectivity_2d"])
    return IRSTDEvaluationProtocol(
        fixed_probability_threshold=float(config["fixed_probability_threshold"]),
        froc_probability_thresholds=tuple(
            float(value) for value in config["froc_probability_thresholds"]
        ),
        connectivity={4: 1, 8: 2}[connectivity_2d],
        max_centroid_distance=float(
            config["target_matching"]["max_centroid_distance_pixels"]
        ),
        min_component_area=1,
    )


def _resolve_selected_source_paths(
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
                else [directory / f"{image_id}{extension}" for extension in DEFAULT_EXTENSIONS]
            )
            existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
            if len(existing) != 1:
                raise FileNotFoundError(
                    f"expected one {role} for selected ID {image_id!r}, got {existing}"
                )
            resolved[(role, image_id)] = existing[0]
    return resolved


def source_file_manifests(
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
        "combined_sha256": combined.hexdigest(),
        "images_sha256": images.hexdigest(),
        "masks_sha256": masks.hexdigest(),
        "algorithm": "sorted-id-image_sha256-mask_sha256-lf-v1",
    }


class PilotIOGuard:
    """Allow pixel I/O only for the exact pre-registered Pilot source files."""

    def __init__(self, allowed: Mapping[tuple[str, str], Path]) -> None:
        self.allowed = {key: path.resolve() for key, path in allowed.items()}
        self._digest = hashlib.sha256()
        self.image_open_count = 0
        self.mask_open_count = 0
        self.forbidden_open_count = 0
        self.opened_ids: set[str] = set()

    def __call__(self, role: str, image_id: str, path: Path) -> None:
        resolved = path.resolve()
        expected = self.allowed.get((role, image_id))
        if role not in {"image", "mask"} or expected is None or resolved != expected:
            self.forbidden_open_count += 1
            raise PermissionError(
                f"Pilot I/O guard refused {role} {image_id!r} at {resolved}"
            )
        if role == "image":
            self.image_open_count += 1
        else:
            self.mask_open_count += 1
        self.opened_ids.add(image_id)
        event = f"{role}\t{image_id}\t{resolved}\n".encode("utf-8")
        self._digest.update(event)

    def summary(self) -> dict[str, Any]:
        return {
            "allowed_unique_ids": len({image_id for _, image_id in self.allowed}),
            "opened_unique_ids": len(self.opened_ids),
            "image_open_count": self.image_open_count,
            "mask_open_count": self.mask_open_count,
            "forbidden_open_count": self.forbidden_open_count,
            "ordered_io_events_sha256": self._digest.hexdigest(),
        }


def _update_tensor_digest(
    digest: Any,
    image_id: str,
    tensor: torch.Tensor,
) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    header = json.dumps(
        {
            "image_id": image_id,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(struct.pack(">Q", len(header)))
    digest.update(header)
    raw = array.tobytes(order="C")
    digest.update(struct.pack(">Q", len(raw)))
    digest.update(raw)


def dataset_tensor_hashes(
    dataset: IRSTDResearchDataset,
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    image_digest = hashlib.sha256()
    mask_digest = hashlib.sha256()
    observed_ids: list[str] = []
    for index in selected_indices:
        sample = dataset[index]
        image_id = str(sample["image_id"])
        observed_ids.append(image_id)
        _update_tensor_digest(image_digest, image_id, sample["image"])
        _update_tensor_digest(mask_digest, image_id, sample["mask"])
    return {
        "ordered_ids": tuple(observed_ids),
        "image_sha256": image_digest.hexdigest(),
        "mask_sha256": mask_digest.hexdigest(),
    }


def validate_formal_pilot_contract(
    args: argparse.Namespace,
    *,
    paths: PilotPaths,
    protocol: Mapping[str, Any],
    dataset_contract: Mapping[str, Any],
    selected_conditions: Sequence[tuple[str, int]],
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    selected_ids: Sequence[str],
    source_manifests: Mapping[str, str],
    severity_table_hash: str,
) -> dict[str, Any]:
    """Validate every pre-registered invariant of a formal Pilot run."""

    if bool(getattr(args, "smoke", False)):
        return {"formal_contract_satisfied": False, "execution_mode": "smoke"}

    pilot = protocol["pilot"]
    preprocessing = protocol["preprocessing"]
    severity = protocol["severity_table"]
    canonical_root = _project_path(dataset_contract["root"])
    canonical_train = _project_path(dataset_contract["train_split"])
    canonical_test = _project_path(dataset_contract["test_split"])
    canonical_checkpoint = _project_path(dataset_contract["checkpoint"])
    for actual, expected, label in (
        (paths.dataset_root, canonical_root, "dataset root"),
        (paths.split_file, canonical_train, "train split path"),
        (paths.test_split_file, canonical_test, "test split path"),
        (paths.checkpoint, canonical_checkpoint, "checkpoint path"),
    ):
        _require_equal(actual, expected, label)

    _require_equal(int(args.subset_size), int(pilot["subset_size_per_dataset"]), "subset size")
    _require_equal(int(args.seed), int(pilot["base_seed"]), "Pilot seed")
    expected_size = int(preprocessing["image_resize"]["size"][0])
    _require_equal(int(args.image_size), expected_size, "Pilot image size")
    if list(preprocessing["image_resize"]["size"]) != [256, 256]:
        raise ValueError("formal Pilot preprocessing must remain 256x256")
    if tuple(selected_conditions) != DEFAULT_CONDITIONS:
        raise ValueError("formal Pilot condition sequence drifted")
    _require_equal(len(selected_conditions), int(pilot["condition_count_per_dataset"]), "condition count")

    _require_equal(
        source_runner.sha256_file(paths.split_file),
        str(dataset_contract["train_split_sha256"]),
        "train split SHA256",
    )
    _require_equal(
        source_runner.sha256_file(paths.test_split_file),
        str(dataset_contract["test_split_sha256"]),
        "test split SHA256",
    )
    _require_equal(len(train_ids), int(dataset_contract["train_images"]), "train split count")
    _require_equal(len(test_ids), int(dataset_contract["test_images"]), "test split count")
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("fixed train/test IDs must each be unique")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"fixed train/test ID overlap: {overlap[:10]}")
    _require_equal(len(selected_ids), DEFAULT_SUBSET_SIZE, "selected Pilot ID count")
    _require_equal(
        sequence_sha256(selected_ids),
        str(dataset_contract["pilot_ordered_ids_sha256"]),
        "ordered Pilot ID SHA256",
    )
    _require_equal(
        source_manifests["combined_sha256"],
        str(dataset_contract["pilot_source_manifest_sha256"]),
        "selected source corpus SHA256",
    )
    _require_equal(
        source_manifests["images_sha256"],
        str(dataset_contract["pilot_image_manifest_sha256"]),
        "selected image corpus SHA256",
    )
    _require_equal(
        source_manifests["masks_sha256"],
        str(dataset_contract["pilot_mask_manifest_sha256"]),
        "selected mask corpus SHA256",
    )
    _require_equal(severity_table_hash, str(severity["sha256"]), "severity table SHA256")
    _require_equal(bool(severity["frozen"]), False, "pre-Pilot severity frozen flag")
    _require_equal(
        source_runner.sha256_file(_project_path(severity["acceptance_criteria"])),
        str(severity["acceptance_criteria_sha256"]),
        "Pilot acceptance criteria SHA256",
    )
    _require_equal(
        source_runner.sha256_file(_project_path(protocol["source"]["parent_training_protocol"])),
        str(protocol["source"]["parent_training_protocol_sha256"]),
        "parent training protocol SHA256",
    )
    if paths.output_dir.exists() and any(paths.output_dir.iterdir()):
        raise FileExistsError(
            f"formal Pilot output directory is not empty; refusing overwrite: {paths.output_dir}"
        )
    return {
        "formal_contract_satisfied": True,
        "execution_mode": "formal",
        "fixed_train_test_disjoint": True,
        "fixed_train_split_hash_verified": True,
        "fixed_test_split_metadata_hash_verified": True,
        "selected_source_manifest_verified": True,
    }


def _compact_evaluation(result: IRSTDEvaluationResult) -> dict[str, Any]:
    fixed = result.fixed
    return {
        "fixed": {
            "probability_threshold": fixed.probability_threshold,
            "iou": fixed.pixel.intersection_over_union,
            "pd": fixed.target.detection_probability,
            "fa_pixel_rate": fixed.target.false_alarm_pixel_rate,
            "fa_per_million_pixels": fixed.target.false_alarm_pixel_rate * 1_000_000.0,
            "false_positives_per_image": fixed.target.false_positives_per_image,
            "detected_targets": fixed.target.detected_targets,
            "total_targets": fixed.target.total_targets,
            "false_positive_components": fixed.target.false_positive_components,
            "false_alarm_pixels": fixed.target.false_alarm_pixels,
            "image_count": fixed.target.image_count,
        },
        "froc": [
            {
                "probability_threshold": point.probability_threshold,
                "pd": point.target.detection_probability,
                "false_positives_per_image": point.target.false_positives_per_image,
                "fa_pixel_rate": point.target.false_alarm_pixel_rate,
                "fa_per_million_pixels": point.target.false_alarm_pixel_rate
                * 1_000_000.0,
            }
            for point in result.froc
        ],
    }


def monotonic_summary(values: Sequence[float], *, tolerance: float = 1e-12) -> dict[str, Any]:
    """Describe observed step directions without enforcing a pilot outcome."""

    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("values cannot be empty")
    if not np.isfinite(numeric).all():
        raise ValueError("trend values must be finite")
    deltas = [right - left for left, right in zip(numeric, numeric[1:])]
    increases = sum(delta > tolerance for delta in deltas)
    decreases = sum(delta < -tolerance for delta in deltas)
    flats = len(deltas) - increases - decreases
    if increases == 0 and decreases == 0:
        classification = "constant"
    elif increases == 0:
        classification = "non_increasing"
    elif decreases == 0:
        classification = "non_decreasing"
    else:
        classification = "mixed"
    return {
        "values": numeric,
        "adjacent_deltas": deltas,
        "classification": classification,
        "non_increasing": increases == 0,
        "non_decreasing": decreases == 0,
        "increasing_steps": increases,
        "decreasing_steps": decreases,
        "flat_steps": flats,
    }


def build_trend_report(
    condition_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    clean_records = [
        record
        for record in condition_results
        if record["corruption"] == "clean" and record["severity"] == 0
    ]
    if len(clean_records) != 1:
        raise ValueError("exactly one clean condition result is required")
    clean_fixed = clean_records[0]["metrics"]["fixed"]
    report: dict[str, Any] = {}
    for corruption in NON_CLEAN_CORRUPTIONS:
        records = sorted(
            (
                record
                for record in condition_results
                if record["corruption"] == corruption
            ),
            key=lambda record: int(record["severity"]),
        )
        if not records:
            continue
        severities = [int(record["severity"]) for record in records]
        metrics = {}
        for output_name in ("iou", "pd", "fa_pixel_rate"):
            values = [float(record["metrics"]["fixed"][output_name]) for record in records]
            summary = monotonic_summary(values)
            summary["clean_value"] = float(clean_fixed[output_name])
            summary["delta_s1_from_clean"] = values[0] - float(clean_fixed[output_name])
            summary["delta_last_from_clean"] = values[-1] - float(clean_fixed[output_name])
            if output_name in {"iou", "pd"}:
                summary["diagnostic_expected_direction"] = "non_increasing"
                summary["expected_direction_satisfied"] = summary["non_increasing"]
            else:
                summary["diagnostic_expected_direction"] = (
                    "not_asserted; false alarms may rise or fall by corruption"
                )
            metrics[output_name] = summary
        report[corruption] = {
            "severities": severities,
            "metrics": metrics,
            "pilot_only_no_automatic_freeze": True,
        }
    return report


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _labelled_thumbnail(
    physical_rgb: np.ndarray,
    label: str,
    *,
    maximum_edge: int = 160,
) -> Image.Image:
    if physical_rgb.ndim != 3 or physical_rgb.shape[2] != 3:
        raise ValueError("severity grid images must be HWC RGB arrays")
    if physical_rgb.dtype != np.uint8:
        raise ValueError("severity grid images must be physical-domain uint8")
    panel = Image.fromarray(physical_rgb, mode="RGB")
    panel.thumbnail((maximum_edge, maximum_edge), Image.Resampling.BILINEAR)
    header_height = 22
    labelled = Image.new("RGB", (panel.width, panel.height + header_height), "white")
    labelled.paste(panel, (0, header_height))
    ImageDraw.Draw(labelled).text((5, 4), label, fill="black")
    return labelled


def save_physical_severity_grid(
    corruption: str,
    image_id: str,
    clean_image: np.ndarray,
    severity_images: Mapping[int, np.ndarray],
    destination: Path,
) -> list[str]:
    """Save clean plus available S-level physical-domain RGB thumbnails."""

    panels = [_labelled_thumbnail(clean_image, "clean")]
    labels = ["clean"]
    for severity in sorted(severity_images):
        panels.append(_labelled_thumbnail(severity_images[severity], f"S{severity}"))
        labels.append(f"S{severity}")
    canvas = Image.new(
        "RGB",
        (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
        "white",
    )
    left = 0
    for panel in panels:
        canvas.paste(panel, (left, 0))
        left += panel.width
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")
    return labels


def _condition_dataset(
    paths: PilotPaths,
    args: argparse.Namespace,
    corruption: str,
    severity: int,
    *,
    io_access_guard: PilotIOGuard | None = None,
    apply_transform: bool = True,
) -> IRSTDResearchDataset:
    return IRSTDResearchDataset(
        paths.dataset_root,
        split_file=paths.split_file,
        image_size=args.image_size,
        dataset_name=args.dataset,
        corruption=corruption,
        severity=severity,
        seed=args.seed,
        # The injected API guarantees corruption occurs in [0,1] before the
        # dataset performs ImageNet normalisation, including identity clean/0.
        corruption_transform=apply_corruption if apply_transform else None,
        io_access_guard=io_access_guard,
    )


def run_corruption_pilot(
    args: argparse.Namespace,
    *,
    conditions: Sequence[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Execute one bounded Pilot and write a completion-audited artifact."""

    total_start = time.perf_counter()
    protocol_path, protocol, dataset_contract = load_pilot_protocol(
        getattr(args, "protocol", DEFAULT_PROTOCOL), args.dataset
    )
    paths = resolve_pilot_paths(args, dataset_contract)
    formal = not bool(getattr(args, "smoke", False))
    selected_conditions = normalise_conditions(conditions, require_full=formal)
    source_runner.seed_everything(args.seed)

    raw_train_ids = read_split_ids(paths.split_file)
    raw_test_ids = read_split_ids(paths.test_split_file)
    canonical_ids = tuple(_canonical_dataset_image_id(value) for value in raw_train_ids)
    canonical_test_ids = tuple(
        _canonical_dataset_image_id(value) for value in raw_test_ids
    )
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("train IDs are not unique after image_id canonicalisation")
    if len(set(canonical_test_ids)) != len(canonical_test_ids):
        raise ValueError("test IDs are not unique after image_id canonicalisation")
    test_overlap = tuple(sorted(set(canonical_ids) & set(canonical_test_ids)))
    if test_overlap:
        raise ValueError(
            "Corruption Pilot train split contains fixed test IDs; refusing leakage: "
            f"{list(test_overlap[:10])}"
        )
    selected = sha256_ranked_subset(canonical_ids, args.subset_size)
    selected_ids = tuple(item.image_id for item in selected)
    index_by_id = {image_id: index for index, image_id in enumerate(canonical_ids)}
    selected_indices = [index_by_id[image_id] for image_id in selected_ids]
    allowed_source_paths = _resolve_selected_source_paths(paths.dataset_root, selected_ids)
    source_manifests = source_file_manifests(selected_ids, allowed_source_paths)

    severity_table = get_default_severity_table()
    severity_hash_before = source_runner.sha256_file(severity_table.source_path)
    if formal:
        _require_equal(
            severity_table.status,
            protocol["severity_table"]["status"],
            "severity table status",
        )
        _require_equal(severity_table.frozen, False, "severity table frozen flag")
    contract_checks = validate_formal_pilot_contract(
        args,
        paths=paths,
        protocol=protocol,
        dataset_contract=dataset_contract,
        selected_conditions=selected_conditions,
        train_ids=canonical_ids,
        test_ids=canonical_test_ids,
        selected_ids=selected_ids,
        source_manifests=source_manifests,
        severity_table_hash=severity_hash_before,
    )
    checkpoint_summary: dict[str, Any] | None = None
    if formal:
        _, checkpoint_summary = validate_checkpoint_contract(
            paths.checkpoint,
            dataset_name=args.dataset,
            dataset_contract=dataset_contract,
            protocol=protocol,
        )

    evaluation_protocol = build_evaluation_protocol(protocol)
    device = source_runner.resolve_device(args.device)
    io_guard = PilotIOGuard(allowed_source_paths)

    model_load_start = time.perf_counter()
    model = source_runner.build_nsfpn_model()
    checkpoint_wrapper = source_runner.load_trusted_checkpoint(model, paths.checkpoint)
    model.to(device)
    adapter = source_runner.IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    _synchronise(device)
    model_load_seconds = time.perf_counter() - model_load_start
    state_hash_before = source_runner.state_dict_sha256(model.state_dict())

    clean_reference_dataset = _condition_dataset(
        paths,
        args,
        "clean",
        0,
        io_access_guard=io_guard,
        apply_transform=False,
    )
    clean_reference_hashes = dataset_tensor_hashes(
        clean_reference_dataset, selected_indices
    )
    if clean_reference_hashes["ordered_ids"] != selected_ids:
        raise RuntimeError("clean identity reference changed the selected ID order")

    condition_results: list[dict[str, Any]] = []
    captured_physical: dict[tuple[str, int], np.ndarray] = {}
    all_conditions_identical_ids = True

    for corruption, severity in selected_conditions:
        condition_start = time.perf_counter()
        dataset = _condition_dataset(
            paths,
            args,
            corruption,
            severity,
            io_access_guard=io_guard,
        )
        if len(dataset) != len(canonical_ids):
            raise RuntimeError("condition dataset changed train split cardinality")
        loader = DataLoader(
            Subset(dataset, selected_indices),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        evaluator = UnifiedResearchEvaluator(evaluation_protocol)
        observed_ids: list[str] = []
        image_digest = hashlib.sha256()
        mask_digest = hashlib.sha256()

        _synchronise(device)
        inference_start = time.perf_counter()
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError("research dataset must yield mapping batches")
            image_cpu = batch["image"]
            mask_cpu = batch["mask"]
            metadata = source_runner.metadata_from_batch(batch)
            image_id = str(metadata["image_id"])
            observed_ids.append(image_id)
            _update_tensor_digest(image_digest, image_id, image_cpu[0])
            _update_tensor_digest(mask_digest, image_id, mask_cpu[0])

            image = image_cpu.to(device, non_blocking=False)
            mask = mask_cpu.to(device, non_blocking=False)
            logits, _ = source_runner.checked_source_forward(adapter, image, mask)
            evaluator.update_logits(logits.detach().cpu(), mask.detach().cpu())
            if image_id == selected_ids[0]:
                captured_physical[(corruption, severity)] = (
                    source_runner.normalised_image_to_uint8(image.detach().cpu())
                )
        _synchronise(device)
        inference_seconds = time.perf_counter() - inference_start

        ids_identical = tuple(observed_ids) == selected_ids
        all_conditions_identical_ids = all_conditions_identical_ids and ids_identical
        if not ids_identical:
            raise RuntimeError(
                f"Condition {corruption}/S{severity} did not evaluate the exact "
                "ranked image ID sequence."
            )
        first_image_hash = image_digest.hexdigest()
        first_mask_hash = mask_digest.hexdigest()
        replay_hashes = dataset_tensor_hashes(dataset, selected_indices)
        exact_input_reproduction = (
            replay_hashes["ordered_ids"] == selected_ids
            and replay_hashes["image_sha256"] == first_image_hash
        )
        exact_mask_reproduction = (
            replay_hashes["ordered_ids"] == selected_ids
            and replay_hashes["mask_sha256"] == first_mask_hash
        )
        if not exact_input_reproduction or not exact_mask_reproduction:
            raise RuntimeError(
                f"Condition {corruption}/S{severity} was not bit-exact on replay"
            )
        clean_transform_exact_identity = None
        if (corruption, severity) == ("clean", 0):
            clean_transform_exact_identity = (
                first_image_hash == clean_reference_hashes["image_sha256"]
                and first_mask_hash == clean_reference_hashes["mask_sha256"]
            )
            if not clean_transform_exact_identity:
                raise RuntimeError("clean/0 transform is not an exact identity")

        state_hash_after_condition = source_runner.state_dict_sha256(model.state_dict())
        if state_hash_after_condition != state_hash_before:
            raise RuntimeError(
                f"Source model state changed during {corruption}/S{severity}: "
                f"before={state_hash_before}, after={state_hash_after_condition}."
            )
        condition_seconds = time.perf_counter() - condition_start
        condition_results.append(
            {
                "corruption": corruption,
                "severity": severity,
                "evaluated_images": len(observed_ids),
                "evaluated_ids_sha256": sequence_sha256(observed_ids),
                "ids_identical_to_selected": ids_identical,
                "model_input_tensor_sha256": first_image_hash,
                "gt_mask_tensor_sha256": first_mask_hash,
                "replay_model_input_tensor_sha256": replay_hashes["image_sha256"],
                "replay_gt_mask_tensor_sha256": replay_hashes["mask_sha256"],
                "exact_input_reproduction": exact_input_reproduction,
                "exact_mask_reproduction": exact_mask_reproduction,
                "clean_transform_exact_identity": clean_transform_exact_identity,
                "metrics": _compact_evaluation(evaluator.compute()),
                "runtime": {
                    "condition_seconds": condition_seconds,
                    "inference_seconds": inference_seconds,
                    "images_per_inference_second": (
                        len(observed_ids) / inference_seconds
                        if inference_seconds > 0.0
                        else 0.0
                    ),
                },
                "model_state_sha256_after": state_hash_after_condition,
            }
        )

    state_hash_after = source_runner.state_dict_sha256(model.state_dict())
    state_unchanged = state_hash_after == state_hash_before
    if not state_unchanged:
        raise RuntimeError("Source model state changed during corruption Pilot")

    input_reproducible = all(
        record["exact_input_reproduction"] for record in condition_results
    )
    mask_hashes = {record["gt_mask_tensor_sha256"] for record in condition_results}
    gt_masks_identical = len(mask_hashes) == 1
    if not gt_masks_identical:
        raise RuntimeError("GT mask tensors changed across corruption conditions")
    clean_identity = condition_results[0]["clean_transform_exact_identity"] is True

    clean_image = captured_physical.get(("clean", 0))
    if clean_image is None:
        raise RuntimeError("clean visualization source was not captured")
    visualization_records = []
    included_corruptions = tuple(
        corruption
        for corruption in NON_CLEAN_CORRUPTIONS
        if any(condition[0] == corruption for condition in selected_conditions)
    )
    for corruption in included_corruptions:
        severity_images = {
            severity: captured_physical[(corruption, severity)]
            for condition_corruption, severity in selected_conditions
            if condition_corruption == corruption
        }
        visualization_path = (
            paths.output_dir
            / "severity_grids"
            / f"{corruption}_{source_runner.safe_artifact_stem(selected_ids[0])}.png"
        )
        panel_labels = save_physical_severity_grid(
            corruption,
            selected_ids[0],
            clean_image,
            severity_images,
            visualization_path,
        )
        visualization_records.append(
            {
                "corruption": corruption,
                "image_id": selected_ids[0],
                "physical_domain": "RGB_[0,1]_rendered_as_uint8",
                "panels": panel_labels,
                "path": str(visualization_path.relative_to(paths.output_dir)),
                "sha256": source_runner.sha256_file(visualization_path),
            }
        )

    severity_hash_after = source_runner.sha256_file(severity_table.source_path)
    severity_table_unchanged = severity_hash_before == severity_hash_after
    if not severity_table_unchanged:
        raise RuntimeError("severity_tables.yaml changed during the read-only Pilot")

    full_protocol = selected_conditions == DEFAULT_CONDITIONS
    if full_protocol:
        for visualization in visualization_records:
            if visualization["panels"] != ["clean", "S1", "S2", "S3", "S4", "S5"]:
                raise RuntimeError("full Pilot severity grid is incomplete")

    io_summary = io_guard.summary()
    if io_summary["forbidden_open_count"] != 0:
        raise RuntimeError("Pilot I/O guard recorded a forbidden file open")
    if formal:
        expected_opens_per_role = len(selected_ids) * (1 + 2 * len(selected_conditions))
        _require_equal(io_summary["opened_unique_ids"], 64, "I/O opened unique IDs")
        _require_equal(io_summary["image_open_count"], expected_opens_per_role, "image open count")
        _require_equal(io_summary["mask_open_count"], expected_opens_per_role, "mask open count")

    total_seconds = time.perf_counter() - total_start
    split_role = "fixed_train" if formal else "smoke_train_only"
    payload = {
        "schema_version": 2,
        "method": "source_corruption_pilot",
        "execution_mode": "formal" if formal else "smoke",
        "formal_artifact": formal,
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": source_runner.sha256_file(protocol_path),
        "calibration_round": int(protocol["pilot"].get("calibration_round", 1)),
        "repository_provenance": source_runner.repository_provenance(PILOT_PROVENANCE_PATHS),
        "scope": "fixed_train_corruption_calibration_only",
        "split_role": split_role,
        "dataset": args.dataset,
        "dataset_root": str(paths.dataset_root),
        "split_file": str(paths.split_file),
        "split_sha256": source_runner.sha256_file(paths.split_file),
        "fixed_train_split": str(_project_path(dataset_contract["train_split"])),
        "fixed_train_split_sha256": str(dataset_contract["train_split_sha256"]),
        "fixed_test_boundary": {
            "split_file": str(paths.test_split_file),
            "split_sha256": source_runner.sha256_file(paths.test_split_file),
            "split_manifest_metadata_read_for_leakage_guard": True,
            "test_dataset_constructed": False,
            "test_images_opened": 0,
            "test_masks_opened": 0,
            "checkpoint_was_test_selected": bool(
                checkpoint_summary["test_selected"] if checkpoint_summary else True
            ),
            "disclosure": protocol["source"]["checkpoint_disclosure"],
        },
        "checkpoint": str(paths.checkpoint),
        "checkpoint_sha256": source_runner.sha256_file(paths.checkpoint),
        "checkpoint_wrapper": checkpoint_wrapper,
        "checkpoint_summary": checkpoint_summary,
        "device": str(device),
        "seed": args.seed,
        "image_size": args.image_size,
        "selection": {
            "strategy": "ascending_sha256_of_utf8_canonical_image_id",
            "available_images": len(canonical_ids),
            "requested_subset_size": args.subset_size,
            "selected_count": len(selected),
            "selected_ids": [item.image_id for item in selected],
            "selected": [
                {
                    "rank": rank,
                    "image_id": item.image_id,
                    "image_id_sha256": item.image_id_sha256,
                }
                for rank, item in enumerate(selected)
            ],
            "ordered_selected_ids_sha256": sequence_sha256(selected_ids),
            "selected_source_manifests": source_manifests,
        },
        "conditions": condition_results,
        "condition_count": len(condition_results),
        "sample_condition_count": len(selected_ids) * len(condition_results),
        "full_clean_plus_4x5_protocol": full_protocol,
        "evaluation_protocol": {
            "probability_transform": "sigmoid_once",
            "fixed_probability_threshold": evaluation_protocol.fixed_probability_threshold,
            "froc_probability_thresholds": list(evaluation_protocol.froc_probability_thresholds),
            "connectivity": evaluation_protocol.connectivity,
            "max_centroid_distance": evaluation_protocol.max_centroid_distance,
            "min_component_area": evaluation_protocol.min_component_area,
        },
        "checks": {
            **contract_checks,
            "batch_size": 1,
            "same_ordered_ids_all_conditions": all_conditions_identical_ids,
            "fixed_test_id_overlap_count": len(test_overlap),
            "fixed_test_ids_absent": not test_overlap,
            "model_loaded_once": True,
            "model_state_sha256_before": state_hash_before,
            "model_state_sha256_after": state_hash_after,
            "model_state_unchanged": state_unchanged,
            "exact_input_reproduction_all_conditions": input_reproducible,
            "gt_mask_hash_identical_across_conditions": gt_masks_identical,
            "gt_mask_tensor_sha256": next(iter(mask_hashes)),
            "clean_transform_exact_identity": clean_identity,
            "corruption_injected_before_normalization": True,
            "test_manifest_metadata_only": True,
            "test_image_open_count": 0,
            "test_mask_open_count": 0,
            "io_guard": io_summary,
            "severity_table_sha256_before": severity_hash_before,
            "severity_table_sha256_after": severity_hash_after,
            "severity_table_unchanged": severity_table_unchanged,
            "severity_table_automatically_frozen": False,
        },
        "severity_table": {
            "path": str(severity_table.source_path),
            "status": severity_table.status,
            "frozen": severity_table.frozen,
            "calibration_required": severity_table.calibration_required,
            "calibration_completed": severity_table.calibration_completed,
        },
        "trends": build_trend_report(condition_results),
        "visualizations": visualization_records,
        "artifact_contract": {
            "primary": "pilot.json",
            "manifest": "artifact_manifest.json",
            "completion_sentinel": "COMPLETE.json",
            "overwrite_refused_in_formal_mode": True,
        },
        "runtime": {
            "total_seconds": total_seconds,
            "model_load_seconds": model_load_seconds,
            "condition_seconds_total": sum(
                record["runtime"]["condition_seconds"] for record in condition_results
            ),
        },
    }
    pilot_path = paths.output_dir / "pilot.json"
    source_runner.write_json_atomic(pilot_path, payload)
    artifact_files = {
        "pilot.json": source_runner.sha256_file(pilot_path),
        **{
            record["path"]: record["sha256"] for record in visualization_records
        },
    }
    artifact_manifest = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": payload["protocol_sha256"],
        "dataset": args.dataset,
        "execution_mode": payload["execution_mode"],
        "files_sha256": artifact_files,
    }
    manifest_path = paths.output_dir / "artifact_manifest.json"
    source_runner.write_json_atomic(manifest_path, artifact_manifest)
    completion = {
        "complete": True,
        "formal_artifact": formal,
        "dataset": args.dataset,
        "pilot_json_sha256": artifact_files["pilot.json"],
        "artifact_manifest_sha256": source_runner.sha256_file(manifest_path),
    }
    source_runner.write_json_atomic(paths.output_dir / "COMPLETE.json", completion)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_corruption_pilot(args)
    output_dir = (
        args.output_dir
        or DEFAULT_OUTPUT_ROOT / args.dataset
    ).expanduser().resolve()
    print(f"Artifact: {output_dir / 'pilot.json'}")
    print(
        f"Selected {result['selection']['selected_count']} fixed-train images; "
        f"evaluated {result['condition_count']} conditions in "
        f"{result['runtime']['total_seconds']:.2f}s."
    )
    for corruption, trend in result["trends"].items():
        iou = trend["metrics"]["iou"]
        pd = trend["metrics"]["pd"]
        fa = trend["metrics"]["fa_pixel_rate"]
        print(
            f"{corruption}: IoU={iou['classification']}, "
            f"Pd={pd['classification']}, Fa={fa['classification']}"
        )
    print("Severity table was not modified or frozen automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
