"""Run the bounded source-domain corruption severity pilot.

The CLI evaluates one SHA256-ranked subset from the official *trainval* split
under clean input and the four configured corruptions at severities S1--S5.
It is diagnostic only: this runner never edits ``severity_tables.yaml`` and
never selects samples from an official target-test split.

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
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader, Subset

from corruptions.corruption_protocol import (
    NON_CLEAN_CORRUPTIONS,
    NON_CLEAN_SEVERITIES,
    get_default_severity_table,
    validate_corruption_request,
)
from corruptions.infrared_corruptions import apply_corruption
from dataset.research_dataset import IRSTDResearchDataset, read_split_ids
from metrics.irstd_metrics import IRSTDEvaluationResult, UnifiedResearchEvaluator
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SUBSET_SIZE = 64
DEFAULT_SEED = source_runner.SEED
PILOT_PROVENANCE_PATHS = source_runner.SOURCE_PROVENANCE_PATHS + (
    "corruptions/corruption_protocol.py",
    "corruptions/infrared_corruptions.py",
    "corruptions/severity_tables.yaml",
    "run_corruption_pilot.py",
)

PILOT_DATASET_DEFAULTS = {
    dataset_name: {
        "split": PROJECT_ROOT / "dataset" / dataset_name / "trainval.txt",
        "test_split": source_runner.DATASET_DEFAULTS[dataset_name]["split"],
        "checkpoint": source_runner.DATASET_DEFAULTS[dataset_name]["checkpoint"],
    }
    for dataset_name in source_runner.DATASET_DEFAULTS
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
    dataset_root: Path
    split_file: Path
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
            "Source-trainval corruption severity pilot; never evaluates the "
            "official target-test split."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=tuple(PILOT_DATASET_DEFAULTS),
    )
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="Dataset root containing img/label or images/masks.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help=(
            "Source-domain split (default: official trainval.txt); test.txt and "
            "any split containing official test IDs are refused."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trusted source checkpoint (default: local official checkpoint).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--subset-size",
        type=source_runner.positive_integer,
        default=DEFAULT_SUBSET_SIZE,
        help="Maximum SHA256-ranked source images (default: 64).",
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
        help="Default: results/corruption_pilot/<dataset>.",
    )
    return parser


def resolve_pilot_paths(args: argparse.Namespace) -> PilotPaths:
    defaults = PILOT_DATASET_DEFAULTS[args.dataset]
    dataset_root = args.root.expanduser().resolve()
    split_file = (args.split or defaults["split"]).expanduser().resolve()
    checkpoint = (args.checkpoint or defaults["checkpoint"]).expanduser().resolve()
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "results" / "corruption_pilot" / args.dataset
    ).expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    official_test_split = Path(defaults["test_split"]).expanduser().resolve()
    if split_file == official_test_split or split_file.name.casefold() == "test.txt":
        raise ValueError(
            "Corruption pilot refuses test.txt; use the official trainval split "
            "or another explicitly source-domain holdout split."
        )
    return PilotPaths(dataset_root, split_file, checkpoint, output_dir)


def _canonical_dataset_image_id(identifier: str) -> str:
    return Path(identifier).with_suffix("").as_posix()


def official_test_id_overlap(
    dataset_name: str,
    candidate_image_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return sorted candidate IDs that occur in the frozen official test split."""

    test_split = Path(PILOT_DATASET_DEFAULTS[dataset_name]["test_split"])
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
) -> tuple[tuple[str, int], ...]:
    selected = DEFAULT_CONDITIONS if conditions is None else tuple(conditions)
    if not selected:
        raise ValueError("at least one pilot condition is required")
    canonical = tuple(validate_corruption_request(*condition) for condition in selected)
    if len(set(canonical)) != len(canonical):
        raise ValueError("pilot conditions must be unique")
    if canonical[0] != ("clean", 0) or canonical.count(("clean", 0)) != 1:
        raise ValueError("pilot conditions must contain clean/0 exactly once and first")
    return canonical


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
        corruption_transform=apply_corruption,
    )


def run_corruption_pilot(
    args: argparse.Namespace,
    *,
    conditions: Sequence[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Execute one bounded pilot and atomically write ``pilot.json``."""

    total_start = time.perf_counter()
    paths = resolve_pilot_paths(args)
    selected_conditions = normalise_conditions(conditions)
    source_runner.seed_everything(args.seed)

    raw_split_ids = read_split_ids(paths.split_file)
    canonical_ids = tuple(_canonical_dataset_image_id(value) for value in raw_split_ids)
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("split IDs are not unique after dataset image_id canonicalisation")
    test_overlap = official_test_id_overlap(args.dataset, canonical_ids)
    if test_overlap:
        preview = list(test_overlap[:10])
        raise ValueError(
            "Corruption pilot split contains IDs from the frozen official test "
            f"split; refusing target-test leakage: {preview}"
        )
    selected = sha256_ranked_subset(canonical_ids, args.subset_size)
    selected_ids = tuple(item.image_id for item in selected)
    index_by_id = {image_id: index for index, image_id in enumerate(canonical_ids)}
    selected_indices = [index_by_id[image_id] for image_id in selected_ids]

    severity_table = get_default_severity_table()
    severity_hash_before = source_runner.sha256_file(severity_table.source_path)
    evaluation_protocol = source_runner.build_unified_evaluation_protocol()
    device = source_runner.resolve_device(args.device)

    model_load_start = time.perf_counter()
    model = source_runner.build_nsfpn_model()
    checkpoint_wrapper = source_runner.load_trusted_checkpoint(model, paths.checkpoint)
    model.to(device)
    adapter = source_runner.IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    _synchronise(device)
    model_load_seconds = time.perf_counter() - model_load_start
    state_hash_before = source_runner.state_dict_sha256(model.state_dict())

    condition_results: list[dict[str, Any]] = []
    captured_physical: dict[tuple[str, int], np.ndarray] = {}
    all_conditions_identical_ids = True

    for corruption, severity in selected_conditions:
        condition_start = time.perf_counter()
        dataset = _condition_dataset(paths, args, corruption, severity)
        if len(dataset) != len(canonical_ids):
            raise RuntimeError("condition dataset changed split cardinality")
        loader = DataLoader(
            Subset(dataset, selected_indices),
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        evaluator = UnifiedResearchEvaluator(evaluation_protocol)
        observed_ids: list[str] = []

        _synchronise(device)
        inference_start = time.perf_counter()
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise TypeError("research dataset must yield mapping batches")
            image = batch["image"].to(device, non_blocking=False)
            mask = batch["mask"].to(device, non_blocking=False)
            metadata = source_runner.metadata_from_batch(batch)
            image_id = str(metadata["image_id"])
            observed_ids.append(image_id)

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
        raise RuntimeError("Source model state changed during corruption pilot")

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
        raise RuntimeError("severity_tables.yaml changed during the read-only pilot")

    full_protocol = selected_conditions == DEFAULT_CONDITIONS
    if full_protocol:
        for visualization in visualization_records:
            if visualization["panels"] != [
                "clean",
                "S1",
                "S2",
                "S3",
                "S4",
                "S5",
            ]:
                raise RuntimeError("full pilot severity grid is incomplete")

    total_seconds = time.perf_counter() - total_start
    official_trainval_split = Path(
        PILOT_DATASET_DEFAULTS[args.dataset]["split"]
    ).resolve()
    split_role = (
        "official_trainval"
        if paths.split_file == official_trainval_split
        else "explicit_source_holdout_without_official_test_ids"
    )
    payload = {
        "schema_version": 1,
        "method": "source_corruption_pilot",
        "repository_provenance": source_runner.repository_provenance(
            PILOT_PROVENANCE_PATHS
        ),
        "scope": "source_domain_only",
        "split_role": split_role,
        "dataset": args.dataset,
        "dataset_root": str(paths.dataset_root),
        "split_file": str(paths.split_file),
        "split_sha256": source_runner.sha256_file(paths.split_file),
        "official_trainval_split": str(official_trainval_split),
        "official_trainval_split_sha256": source_runner.sha256_file(
            official_trainval_split
        ),
        "official_test_split_refused": True,
        "official_test_split_sha256": source_runner.sha256_file(
            Path(PILOT_DATASET_DEFAULTS[args.dataset]["test_split"])
        ),
        "checkpoint": str(paths.checkpoint),
        "checkpoint_sha256": source_runner.sha256_file(paths.checkpoint),
        "checkpoint_wrapper": checkpoint_wrapper,
        "device": str(device),
        "seed": args.seed,
        "image_size": args.image_size,
        "selection": {
            "strategy": "ascending_sha256_of_utf8_image_id",
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
        },
        "conditions": condition_results,
        "condition_count": len(condition_results),
        "full_clean_plus_4x5_protocol": full_protocol,
        "checks": {
            "batch_size": 1,
            "same_ordered_ids_all_conditions": all_conditions_identical_ids,
            "official_test_id_overlap_count": len(test_overlap),
            "official_test_ids_absent": not test_overlap,
            "model_loaded_once": True,
            "model_state_sha256_before": state_hash_before,
            "model_state_sha256_after": state_hash_after,
            "model_state_unchanged": state_unchanged,
            "corruption_injected_before_normalization": True,
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
        "runtime": {
            "total_seconds": total_seconds,
            "model_load_seconds": model_load_seconds,
            "condition_seconds_total": sum(
                record["runtime"]["condition_seconds"]
                for record in condition_results
            ),
        },
    }
    source_runner.write_json_atomic(paths.output_dir / "pilot.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    result = run_corruption_pilot(args)
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "results" / "corruption_pilot" / args.dataset
    ).expanduser().resolve()
    print(f"Artifact: {output_dir / 'pilot.json'}")
    print(
        f"Selected {result['selection']['selected_count']} source-trainval images; "
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
