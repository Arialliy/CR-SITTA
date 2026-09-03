#!/usr/bin/env python3
"""Export clean Source predictions for one frozen checkpoint-axis role.

The default role is the test-selected ``best_pd`` development axis.  It is
hard-blocked until the global clean+Source+AdaBN ``best_miou`` parity receipt
exists and passes the shared contract.  ``best_miou`` itself is available only
as a parity run with an explicit non-formal output directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

from benchmark.checkpoint_axis import (
    DEFAULT_AXIS_CONFIG,
    CheckpointAxis,
    finalize_and_publish,
    load_and_verify_checkpoint,
    load_axis_config,
    load_checkpoint_into_model,
    prepare_staging,
    remove_private_staging,
    resolve_axis,
    verify_checkpoint_file,
)
from dataio.research_dataset import IRSTDResearchDataset, read_split_ids
from metrics.irstd_metrics import UnifiedResearchEvaluator, probabilities_from_logits
from metrics.official_metric_adapter import OfficialMetricAdapter
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parent
METHOD = "locked_checkpoint_axis_clean_source_export_v2"
PROVENANCE_PATHS = (
    "benchmark/__init__.py",
    "benchmark/checkpoint_axis.py",
    "configs/checkpoint_axis_best_pd_v1.yaml",
    "configs/retrain_fixed_splits.yaml",
    "dataio/research_dataset.py",
    "export_fixed_split_source_axis_v2.py",
    "metrics/connected_components.py",
    "metrics/irstd_metrics.py",
    "metrics/official_metric_adapter.py",
    "metrics/target_matching.py",
    "model/MSHNet_NSFPN.py",
    "test_source.py",
    "tta/d0_secure_io.py",
    "tta/model_adapter.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--checkpoint-role",
        choices=("best_miou", "best_pd"),
        default="best_pd",
        help="Default: best_pd. best_miou is parity-only and needs --output-dir.",
    )
    parser.add_argument("--axis-config", type=Path, default=DEFAULT_AXIS_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Explicit result directory. Mandatory for best_miou parity and "
            "for any prefix smoke run; omitted best_pd uses the formal role-first root."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-images",
        type=source_runner.positive_integer,
        default=None,
        help="Deterministic-prefix smoke only; requires --output-dir.",
    )
    parser.add_argument(
        "--visualization-count",
        type=int,
        default=20,
        help="Number of optional visualization panels; all masks/maps are always saved.",
    )
    return parser


def _load_run_contract(args: argparse.Namespace) -> tuple[dict[str, Any], CheckpointAxis]:
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.visualization_count < 0:
        raise ValueError("visualization-count must be non-negative")
    if args.max_images is not None and args.output_dir is None:
        raise ValueError("prefix smoke runs require an explicit --output-dir")
    if (
        args.checkpoint_role == "best_pd"
        and args.output_dir is not None
        and args.max_images is None
    ):
        raise ValueError(
            "a full best_pd export must use the configured formal role-first root; "
            "--output-dir is reserved for prefix smoke runs"
        )
    config = load_axis_config(args.axis_config, project_root=PROJECT_ROOT)
    axis = resolve_axis(
        config,
        dataset=args.dataset,
        role=args.checkpoint_role,
        artifact_kind="clean",
        output_override=args.output_dir,
    )
    return config, axis


def _portable_prediction_path(image_id: str, suffix: str) -> Path:
    if not isinstance(image_id, str) or not image_id or "\\" in image_id or "\x00" in image_id:
        raise ValueError(f"unsafe image ID for output: {image_id!r}")
    relative = PurePosixPath(image_id)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"unsafe image ID for output: {image_id!r}")
    return Path(*relative.with_suffix(suffix).parts)


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    Image.fromarray(array, mode="L").save(temporary, format="PNG")
    os.replace(temporary, path)


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")
    os.replace(temporary, path)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=True, allow_unicode=True)
    os.replace(temporary, path)


def _sha256_ids(identifiers: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{identifier}\n" for identifier in identifiers).encode("utf-8")
    ).hexdigest()


def _pixel_record(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = probability > 0.5
    foreground = target > 0
    true_positive = int(np.logical_and(prediction, foreground).sum())
    false_positive = int(np.logical_and(prediction, ~foreground).sum())
    false_negative = int(np.logical_and(~prediction, foreground).sum())
    true_negative = int(
        prediction.size - true_positive - false_positive - false_negative
    )
    union = true_positive + false_positive + false_negative
    return {
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "true_negative_pixels": true_negative,
        "predicted_positive_pixels": int(prediction.sum()),
        "target_positive_pixels": int(foreground.sum()),
        "intersection_over_union": float(true_positive / union) if union else 1.0,
    }


def _checkpoint_summary(axis: CheckpointAxis, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "architecture": payload.get("architecture"),
        "dataset": axis.dataset,
        "checkpoint_role": axis.role,
        "epoch": axis.expected_epoch,
        "selection_metric": axis.expected_selection_metric,
        "selection_rule": axis.expected_selection_rule,
        "selection_value": axis.expected_selection_value,
        "test_metrics": dict(axis.recorded_test_metrics),
        "test_selected": True,
        "selection_split": axis.checkpoint_selection_split,
        "development_only": axis.development_only,
        "split_manifest": dict(payload["split_manifest"]),
        "training_protocol_id": axis.training_protocol_id,
        "training_protocol_sha256": axis.training_protocol_sha256,
    }


def _checkpoint_metric_comparison(
    measured: Mapping[str, float],
    axis: CheckpointAxis,
    *,
    complete_split: bool,
) -> dict[str, Any]:
    if not complete_split:
        return {
            "evaluated": False,
            "passed": None,
            "reason": "checkpoint comparison requires the complete fixed test split",
        }
    comparisons: dict[str, Any] = {}
    for name in ("miou", "pd", "fa_per_pixel_x1e6"):
        actual = float(measured[name])
        expected = float(axis.recorded_test_metrics[name])
        comparisons[name] = {
            "measured": actual,
            "checkpoint_recorded": expected,
            "comparison": "exact_float_equality",
            "passed": actual == expected,
        }
    return {
        "evaluated": True,
        "passed": all(item["passed"] for item in comparisons.values()),
        "metrics": comparisons,
    }


def _runtime_provenance(axis: CheckpointAxis) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    files: dict[str, str] = {}
    for relative in PROVENANCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime provenance file is missing: {path}")
        files[relative] = source_runner.sha256_file(path)
    return {
        "base_commit": (PROJECT_ROOT / "BASE_COMMIT.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "head_commit": head,
        "worktree_clean": not dirty,
        "dirty_entries": dirty,
        "runtime_file_sha256": files,
        "axis_config": {
            "path": str(axis.config_path.relative_to(PROJECT_ROOT)),
            "sha256": axis.config_sha256,
        },
        "training_protocol": {
            "path": str(axis.training_protocol_path.relative_to(PROJECT_ROOT)),
            "sha256": axis.training_protocol_sha256,
        },
    }


def _load_verified_model(payload: Mapping[str, Any]):
    model = source_runner.build_nsfpn_model()
    wrapper = load_checkpoint_into_model(model, payload)
    return model, wrapper


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    """Run one clean axis export and atomically publish a complete artifact."""

    started = time.perf_counter()
    config, axis = _load_run_contract(args)
    identifiers = read_split_ids(axis.split_path)
    if len(identifiers) != axis.expected_images:
        raise ValueError("fixed split count changed after axis resolution")
    checkpoint_payload = load_and_verify_checkpoint(axis)
    staging = prepare_staging(axis)
    try:
        source_runner.seed_everything(source_runner.SEED)
        device = source_runner.resolve_device(args.device)
        dataset = IRSTDResearchDataset(
            axis.dataset_root,
            split_file=axis.split_path,
            image_size=axis.image_size,
            dataset_name=axis.dataset,
            corruption="clean",
            severity=0,
            seed=source_runner.SEED,
        )
        evaluated_images = min(len(dataset), args.max_images or len(dataset))
        if evaluated_images < 1:
            raise RuntimeError("clean axis export requires at least one image")
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )

        model, checkpoint_wrapper = _load_verified_model(checkpoint_payload)
        model.to(device)
        adapter = source_runner.IRSTDModelAdapter(model, warm_flag=False)
        adapter.set_source_eval_mode()
        state_hash_before = source_runner.state_dict_sha256(model.state_dict())
        protocol = source_runner.build_unified_evaluation_protocol()
        if protocol.fixed_probability_threshold != 0.5:
            raise RuntimeError("runtime evaluator threshold drifted from strict 0.5")
        official_evaluator = OfficialMetricAdapter(image_size=axis.image_size)
        unified_evaluator = UnifiedResearchEvaluator(protocol)
        records: list[dict[str, Any]] = []
        visualizations: list[str] = []
        repeat_exact = False

        for index, batch in enumerate(loader):
            if index >= evaluated_images:
                break
            image = batch["image"].to(device, non_blocking=False)
            target_device = batch["mask"].to(device, non_blocking=False)
            metadata = source_runner.metadata_from_batch(batch)
            expected_id = Path(identifiers[index]).with_suffix("").as_posix()
            if metadata["image_id"] != expected_id:
                raise RuntimeError(
                    f"loader order drift at {index}: expected {expected_id!r}, "
                    f"got {metadata['image_id']!r}"
                )
            logits, repeated = source_runner.checked_source_forward(
                adapter,
                image,
                target_device,
                repeat_exact=index == 0,
            )
            if index == 0:
                repeat_exact = bool(repeated)
            logits_cpu = logits.detach().cpu()
            target_cpu = target_device.detach().cpu()
            source_runner._update_official_evaluator(
                official_evaluator, logits_cpu, target_cpu
            )
            unified_evaluator.update_logits(logits_cpu, target_cpu)

            probability = probabilities_from_logits(logits_cpu)[0, 0].astype(
                np.float32, copy=False
            )
            target = target_cpu.numpy()[0, 0]
            binary_mask = np.where(probability > 0.5, 255, 0).astype(np.uint8)
            identifier = str(metadata["image_id"])
            mask_relative = Path("prediction_masks_256") / _portable_prediction_path(
                identifier, ".png"
            )
            probability_relative = Path(
                "probability_maps_256"
            ) / _portable_prediction_path(identifier, ".npy")
            mask_path = staging / mask_relative
            probability_path = staging / probability_relative
            _write_png(mask_path, binary_mask)
            _write_npy(probability_path, probability)
            records.append(
                {
                    "index": index,
                    **metadata,
                    "input_shape": list(image.shape),
                    "logit_shape": list(logits.shape),
                    "prediction_mask": mask_relative.as_posix(),
                    "prediction_mask_sha256": source_runner.sha256_file(mask_path),
                    "probability_map": probability_relative.as_posix(),
                    "probability_map_sha256": source_runner.sha256_file(
                        probability_path
                    ),
                    "probability_map_dtype": "float32",
                    "probability_min": float(probability.min()),
                    "probability_max": float(probability.max()),
                    "pixel_metrics_at_probability_gt_0_5": _pixel_record(
                        probability, target
                    ),
                }
            )
            if index < min(args.visualization_count, evaluated_images):
                relative = (
                    Path("visualizations")
                    / f"{index:04d}_{source_runner.safe_artifact_stem(identifier)}.png"
                )
                source_runner.save_prediction_visualization(
                    image.detach().cpu(),
                    target_cpu,
                    logits_cpu,
                    staging / relative,
                )
                visualizations.append(relative.as_posix())

        if len(records) != evaluated_images:
            raise RuntimeError(
                f"expected {evaluated_images} predictions, wrote {len(records)}"
            )
        state_hash_after = source_runner.state_dict_sha256(model.state_dict())
        if state_hash_after != state_hash_before:
            raise RuntimeError("clean export mutated the locked Source model")
        official = official_evaluator.compute()
        unified = unified_evaluator.compute()
        reported = {
            "miou": float(official.mean_iou),
            "pd": float(official.detection_probability[0]),
            "fa_per_pixel_x1e6": float(official.false_alarm_pixel_rate[0] * 1e6),
        }
        complete_split = args.max_images is None
        metric_comparison = _checkpoint_metric_comparison(
            reported, axis, complete_split=complete_split
        )
        if metric_comparison["evaluated"] and not metric_comparison["passed"]:
            raise RuntimeError(
                "reloaded checkpoint metrics are not exactly equal to its recorded "
                f"test metrics: {metric_comparison['metrics']}"
            )

        metrics = {
            "schema_version": 2,
            "method": METHOD,
            "artifact_kind": "clean",
            "protocol_id": axis.protocol_id,
            "axis_config_sha256": axis.config_sha256,
            "dataset": axis.dataset,
            "dataset_root": str(axis.dataset_root),
            "split_file": str(axis.split_path),
            "split_sha256": axis.split_sha256,
            "ordered_image_ids_sha256": _sha256_ids(identifiers[:evaluated_images]),
            "available_images": len(dataset),
            "evaluated_images": evaluated_images,
            "complete_fixed_test_split": complete_split,
            "checkpoint_role": axis.role,
            "checkpoint": str(axis.checkpoint_path),
            "checkpoint_sha256": axis.checkpoint_sha256,
            "checkpoint_wrapper": checkpoint_wrapper,
            "checkpoint_metadata": _checkpoint_summary(axis, checkpoint_payload),
            "checkpoint_recorded_metric_comparison": metric_comparison,
            "scientific_eligibility": {
                "tier": "development_test_selected",
                "main_paper_table": False,
                "development_only": True,
            },
            "test_selected_checkpoint_disclosure": (
                f"{axis.role} was selected by repeated evaluation of this fixed "
                "test split during epochs 500-1000; this exporter performs no selection"
            ),
            "device": str(device),
            "seed": source_runner.SEED,
            "image_size": axis.image_size,
            "threshold_transform": "sigmoid",
            "threshold_comparison": "strict_greater_than",
            "threshold_value": 0.5,
            "threshold_rule": "sigmoid(logit) > 0.5",
            "prediction_mask_encoding": "uint8 PNG background=0 foreground=255",
            "probability_map_encoding": (
                "NumPy .npy float32 probability on the 256x256 evaluation grid"
            ),
            "official": official.to_dict(),
            "official_reported_operating_point": reported,
            "unified": unified.to_dict(),
            "artifact_counts": {
                "prediction_masks": len(records),
                "probability_maps": len(records),
                "per_image_records": len(records),
                "visualizations": len(visualizations),
            },
            "visualizations": visualizations,
            "checks": {
                "checkpoint_role_matches_selection_metric": True,
                "checkpoint_metadata_exact": True,
                "checkpoint_split_hash_matches": True,
                "strict_probability_gt_0_5": True,
                "ordered_ids_match_split": True,
                "repeat_logit_exact_first_image": repeat_exact,
                "model_state_sha256_before": state_hash_before,
                "model_state_sha256_after": state_hash_after,
                "model_state_unchanged": True,
                "all_prediction_masks_saved": len(records) == evaluated_images,
                "all_probability_maps_saved": len(records) == evaluated_images,
                "per_image_complete": len(records) == evaluated_images,
            },
            "runtime_seconds": time.perf_counter() - started,
        }
        run_config = {
            "schema_version": 2,
            "method": METHOD,
            "protocol_id": axis.protocol_id,
            "axis_config_sha256": axis.config_sha256,
            "dataset": axis.dataset,
            "checkpoint_role": axis.role,
            "checkpoint_path": str(axis.checkpoint_path),
            "checkpoint_sha256": axis.checkpoint_sha256,
            "checkpoint_epoch": axis.expected_epoch,
            "checkpoint_selection_metric": axis.expected_selection_metric,
            "checkpoint_selection_rule": axis.expected_selection_rule,
            "checkpoint_selection_value": axis.expected_selection_value,
            "checkpoint_recorded_test_metrics": dict(axis.recorded_test_metrics),
            "checkpoint_selection_split": "test",
            "test_selected": True,
            "development_only": True,
            "split_path": str(axis.split_path),
            "split_sha256": axis.split_sha256,
            "expected_images": axis.expected_images,
            "image_size": axis.image_size,
            "threshold": {
                "transform": "sigmoid",
                "rule": "strict_greater_than",
                "value": 0.5,
            },
            "device": str(device),
            "num_workers": args.num_workers,
            "max_images": args.max_images,
            "visualization_count": args.visualization_count,
            "output_dir": str(axis.output_dir),
            "parity_receipt_path": (
                str(axis.parity_receipt_path)
                if axis.parity_receipt_path is not None
                else None
            ),
            "parity_receipt_sha256": axis.parity_receipt_sha256,
        }
        provenance = _runtime_provenance(axis)
        axis_project_root = axis.config_path.parent.parent
        provenance["checkpoint"] = {
            "path": str(axis.checkpoint_path.relative_to(axis_project_root)),
            "sha256": axis.checkpoint_sha256,
            "epoch": axis.expected_epoch,
            "selection_metric": axis.expected_selection_metric,
            "selection_rule": axis.expected_selection_rule,
            "selection_value": axis.expected_selection_value,
            "recorded_test_metrics": dict(axis.recorded_test_metrics),
        }
        provenance["fixed_split"] = {
            "path": str(axis.split_path.relative_to(axis_project_root)),
            "sha256": axis.split_sha256,
            "ordered_ids_sha256": _sha256_ids(identifiers),
            "images": axis.expected_images,
        }
        _write_yaml(staging / "run_config.yaml", run_config)
        _write_json(staging / "provenance.json", provenance)
        _write_jsonl(staging / "per_image.jsonl", records)
        _write_json(staging / "metrics.json", metrics)
        required = (
            "run_config.yaml",
            "provenance.json",
            "metrics.json",
            "per_image.jsonl",
            *(record["prediction_mask"] for record in records),
            *(record["probability_map"] for record in records),
            *visualizations,
        )

        def prepublish_guard() -> None:
            verify_checkpoint_file(axis)
            if len(read_split_ids(axis.split_path)) != axis.expected_images:
                raise RuntimeError("split count drifted at publication boundary")

        publication = finalize_and_publish(
            staging=staging,
            final=axis.output_dir,
            axis=axis,
            required_payloads=required,
            manifest_extra={
                "method": METHOD,
                "complete_fixed_test_split": complete_split,
                "evaluated_images": evaluated_images,
                "all_masks_and_probabilities_required": True,
            },
            complete_extra={
                "method": METHOD,
                "complete_fixed_test_split": complete_split,
                "evaluated_images": evaluated_images,
            },
            prepublish_guard=prepublish_guard,
        )
        return {
            **metrics,
            "published_output_dir": str(axis.output_dir),
            "artifact_manifest_sha256": publication["manifest_sha256"],
            "complete_sha256": publication["complete_sha256"],
            "payload_tree_sha256": publication["payload_tree"]["sha256"],
        }
    except BaseException:
        remove_private_staging(staging, final=axis.output_dir)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_export(args)
    metrics = result["official_reported_operating_point"]
    print(f"Artifacts: {result['published_output_dir']}")
    print(
        f"{result['dataset']} {result['checkpoint_role']} clean: "
        f"mIoU={metrics['miou']:.9f}, Pd={metrics['pd']:.9f}, "
        f"Fa(x1e-6)={metrics['fa_per_pixel_x1e6']:.9f}"
    )
    print(
        f"Saved {result['artifact_counts']['prediction_masks']} masks and "
        f"{result['artifact_counts']['probability_maps']} probability maps."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "main",
    "run_export",
]
