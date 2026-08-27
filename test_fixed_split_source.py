"""Export clean fixed-test predictions from a locked ``best_miou`` checkpoint.

This runner does not train or select a checkpoint.  It reloads the already
selected weight, verifies its embedded split metadata, evaluates the complete
fixed test split, and writes one binary prediction mask plus one float32
probability map per image.  Aggregate metrics are checked against the values
recorded when ``best_miou.pth.tar`` was saved.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

from dataio.research_dataset import IRSTDResearchDataset, read_split_ids
from metrics.irstd_metrics import UnifiedResearchEvaluator, probabilities_from_logits
from metrics.official_metric_adapter import OfficialMetricAdapter
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "retrain_fixed_splits.yaml"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "retraining_fixed_split"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "baseline"
PROVENANCE_PATHS = (
    "configs/retrain_fixed_splits.yaml",
    "dataio/research_dataset.py",
    "metrics/connected_components.py",
    "metrics/irstd_metrics.py",
    "metrics/official_metric_adapter.py",
    "metrics/target_matching.py",
    "model/MSHNet_NSFPN.py",
    "test_fixed_split_source.py",
    "test_source.py",
    "tta/model_adapter.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a locked fixed-split NS-FPN best_miou checkpoint and "
            "export its clean prediction masks."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-images",
        type=source_runner.positive_integer,
        default=None,
        help="Deterministic-prefix smoke test only; omit for the final export.",
    )
    parser.add_argument(
        "--visualization-count",
        type=int,
        default=20,
        help="Number of input/GT/probability/prediction panels (default: 20).",
    )
    parser.add_argument(
        "--skip-probability-maps",
        action="store_true",
        help="Save binary masks only. The final protocol normally keeps float32 maps.",
    )
    return parser


def _project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_export_config(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.expanduser().resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"protocol does not exist: {protocol_path}")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    datasets = protocol.get("datasets", {})
    if args.dataset not in datasets:
        available = ", ".join(sorted(datasets))
        raise ValueError(f"unknown dataset {args.dataset!r}; expected one of {available}")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if args.visualization_count < 0:
        raise ValueError("visualization-count must be non-negative")

    dataset = datasets[args.dataset]
    image_size = int(protocol["training"]["base_size"])
    return {
        "schema_version": 1,
        "method": "locked_best_miou_clean_export",
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_path": str(protocol_path),
        "protocol_sha256": source_runner.sha256_file(protocol_path),
        "dataset": args.dataset,
        "root": str(_project_path(args.root or dataset["root"])),
        "split": str(_project_path(args.split or dataset["test_split"])),
        "expected_split_sha256": str(dataset["test_split_sha256"]),
        "expected_images": int(dataset["test_images"]),
        "checkpoint": str(
            _project_path(
                args.checkpoint
                or DEFAULT_RESULTS_ROOT / args.dataset / "best_miou.pth.tar"
            )
        ),
        "output_dir": str(
            _project_path(
                args.output_dir
                or DEFAULT_OUTPUT_ROOT / args.dataset / "best_miou"
            )
        ),
        "device": str(args.device),
        "num_workers": int(args.num_workers),
        "image_size": image_size,
        "max_images": args.max_images,
        "visualization_count": int(args.visualization_count),
        "save_probability_maps": not bool(args.skip_probability_maps),
        "probability_threshold": 0.5,
    }


def _load_checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("fixed-split checkpoint must be a metadata mapping")
    return payload


def validate_inputs(config: Mapping[str, Any]) -> tuple[tuple[str, ...], Mapping[str, Any]]:
    root = Path(config["root"])
    split = Path(config["split"])
    checkpoint = Path(config["checkpoint"])
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if not split.is_file():
        raise FileNotFoundError(f"test split does not exist: {split}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    identifiers = read_split_ids(split)
    split_hash = source_runner.sha256_file(split)
    if split_hash != config["expected_split_sha256"]:
        raise ValueError(
            "fixed test split hash drift: expected "
            f"{config['expected_split_sha256']}, got {split_hash}"
        )
    if len(identifiers) != config["expected_images"]:
        raise ValueError(
            f"fixed test count drift: expected {config['expected_images']}, "
            f"got {len(identifiers)}"
        )

    payload = _load_checkpoint_payload(checkpoint)
    required = {
        "dataset",
        "epoch",
        "selection_metric",
        "selection_rule",
        "test_metrics",
        "test_selected",
        "split_manifest",
        "state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise KeyError(f"checkpoint is missing fixed-split metadata: {missing}")
    if payload["dataset"] != config["dataset"]:
        raise ValueError(
            f"checkpoint dataset {payload['dataset']!r} does not match "
            f"{config['dataset']!r}"
        )
    if payload["selection_metric"] != "miou":
        raise ValueError(
            "this exporter only accepts the locked best_miou checkpoint; got "
            f"selection_metric={payload['selection_metric']!r}"
        )
    if payload["test_selected"] is not True:
        raise ValueError("checkpoint must explicitly disclose test_selected=true")
    checkpoint_split_hash = payload["split_manifest"].get("test_split_sha256")
    if checkpoint_split_hash != split_hash:
        raise ValueError(
            "checkpoint/test split mismatch: checkpoint records "
            f"{checkpoint_split_hash}, runtime split is {split_hash}"
        )
    if int(payload["test_metrics"]["images"]) != len(identifiers):
        raise ValueError("checkpoint test image count does not match runtime split")
    return identifiers, payload


def _checkpoint_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    run_config = payload.get("run_config", {})
    return {
        "schema_version": payload.get("schema_version"),
        "architecture": payload.get("architecture"),
        "dataset": payload["dataset"],
        "epoch": int(payload["epoch"]),
        "selection_metric": payload["selection_metric"],
        "selection_rule": payload["selection_rule"],
        "selection_value": float(payload["selection_value"]),
        "test_metrics": dict(payload["test_metrics"]),
        "test_selected": bool(payload["test_selected"]),
        "split_manifest": dict(payload["split_manifest"]),
        "training_protocol_id": run_config.get("protocol_id"),
        "training_protocol_sha256": run_config.get("protocol_sha256"),
    }


def _write_png_atomic(destination: Path, array: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    Image.fromarray(array, mode="L").save(temporary, format="PNG")
    os.replace(temporary, destination)


def _write_npy_atomic(destination: Path, array: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, destination)


def _relative_prediction_path(image_id: str, suffix: str) -> Path:
    relative = Path(image_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe image_id for prediction output: {image_id!r}")
    return relative.with_suffix(suffix)


def _pixel_record(probability: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    prediction = probability > 0.5
    foreground = target > 0
    true_positive = int(np.logical_and(prediction, foreground).sum())
    false_positive = int(np.logical_and(prediction, ~foreground).sum())
    false_negative = int(np.logical_and(~prediction, foreground).sum())
    true_negative = int(prediction.size - true_positive - false_positive - false_negative)
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


def _checkpoint_metric_comparison(
    measured: Mapping[str, float],
    checkpoint_metrics: Mapping[str, Any],
    *,
    complete_split: bool,
) -> dict[str, Any]:
    if not complete_split:
        return {
            "evaluated": False,
            "passed": None,
            "reason": "checkpoint comparison requires the complete fixed test split",
        }
    expected_by_name = {
        "miou": float(checkpoint_metrics["miou"]),
        "pd": float(checkpoint_metrics["pd"]),
        "fa_per_pixel_x1e6": float(checkpoint_metrics["fa_per_pixel_x1e6"]),
    }
    tolerance = 1e-12
    comparisons = {}
    for name, expected in expected_by_name.items():
        actual = float(measured[name])
        error = abs(actual - expected)
        comparisons[name] = {
            "measured": actual,
            "checkpoint_recorded": expected,
            "absolute_error": error,
            "absolute_tolerance": tolerance,
            "passed": error <= tolerance,
        }
    return {
        "evaluated": True,
        "passed": all(item["passed"] for item in comparisons.values()),
        "metrics": comparisons,
    }


def _runtime_provenance() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "base_commit": (PROJECT_ROOT / "BASE_COMMIT.txt")
        .read_text(encoding="utf-8")
        .strip(),
        "head_commit": head,
        "tracked_worktree_clean": not tracked_status,
        "tracked_dirty_entries": tracked_status,
        "file_sha256": {
            relative: source_runner.sha256_file(PROJECT_ROOT / relative)
            for relative in PROVENANCE_PATHS
        },
    }


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_export_config(args)
    identifiers, checkpoint_payload = validate_inputs(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    source_runner.seed_everything(source_runner.SEED)
    device = source_runner.resolve_device(config["device"])
    dataset = IRSTDResearchDataset(
        config["root"],
        split_file=config["split"],
        image_size=config["image_size"],
        dataset_name=config["dataset"],
        corruption="clean",
        severity=0,
        seed=source_runner.SEED,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["num_workers"] > 0,
    )
    evaluated_images = min(len(dataset), config["max_images"] or len(dataset))

    model = source_runner.build_nsfpn_model()
    checkpoint_wrapper = source_runner.load_trusted_checkpoint(
        model, Path(config["checkpoint"])
    )
    model.to(device)
    adapter = source_runner.IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    state_hash_before = source_runner.state_dict_sha256(model.state_dict())

    evaluation_protocol = source_runner.build_unified_evaluation_protocol()
    official_evaluator = OfficialMetricAdapter(image_size=config["image_size"])
    unified_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
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
                f"loader order drift at index {index}: expected {expected_id!r}, "
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
        mask_relative = Path("prediction_masks_256") / _relative_prediction_path(
            str(metadata["image_id"]), ".png"
        )
        mask_path = output_dir / mask_relative
        _write_png_atomic(mask_path, binary_mask)

        probability_relative: Path | None = None
        probability_hash: str | None = None
        if config["save_probability_maps"]:
            probability_relative = Path(
                "probability_maps_256"
            ) / _relative_prediction_path(str(metadata["image_id"]), ".npy")
            probability_path = output_dir / probability_relative
            _write_npy_atomic(probability_path, probability)
            probability_hash = source_runner.sha256_file(probability_path)

        record = {
            "index": index,
            **metadata,
            "input_shape": list(image.shape),
            "logit_shape": list(logits.shape),
            "prediction_mask": str(mask_relative),
            "prediction_mask_sha256": source_runner.sha256_file(mask_path),
            "probability_map": (
                str(probability_relative) if probability_relative is not None else None
            ),
            "probability_map_sha256": probability_hash,
            "probability_map_dtype": "float32" if probability_relative else None,
            "probability_min": float(probability.min()),
            "probability_max": float(probability.max()),
            "pixel_metrics_at_probability_gt_0_5": _pixel_record(
                probability, target
            ),
        }
        records.append(record)

        if index < min(config["visualization_count"], evaluated_images):
            visualization_relative = (
                Path("visualizations")
                / f"{index:04d}_{source_runner.safe_artifact_stem(metadata['image_id'])}.png"
            )
            source_runner.save_prediction_visualization(
                image.detach().cpu(),
                target_cpu,
                logits_cpu,
                output_dir / visualization_relative,
            )
            visualizations.append(str(visualization_relative))

    if len(records) != evaluated_images:
        raise RuntimeError(
            f"expected {evaluated_images} predictions, wrote {len(records)}"
        )
    state_hash_after = source_runner.state_dict_sha256(model.state_dict())
    if state_hash_after != state_hash_before:
        raise RuntimeError("clean export mutated the locked Source model state")

    official = official_evaluator.compute()
    unified = unified_evaluator.compute()
    reported = {
        "miou": float(official.mean_iou),
        "pd": float(official.detection_probability[0]),
        "fa_per_pixel_x1e6": float(official.false_alarm_pixel_rate[0] * 1e6),
    }
    complete_split = config["max_images"] is None
    checkpoint_comparison = _checkpoint_metric_comparison(
        reported,
        checkpoint_payload["test_metrics"],
        complete_split=complete_split,
    )
    aggregate = {
        "schema_version": 1,
        "method": config["method"],
        "dataset": config["dataset"],
        "dataset_root": config["root"],
        "split_file": config["split"],
        "split_sha256": source_runner.sha256_file(Path(config["split"])),
        "available_images": len(dataset),
        "evaluated_images": evaluated_images,
        "complete_fixed_test_split": complete_split,
        "checkpoint": config["checkpoint"],
        "checkpoint_sha256": source_runner.sha256_file(Path(config["checkpoint"])),
        "checkpoint_wrapper": checkpoint_wrapper,
        "checkpoint_metadata": _checkpoint_summary(checkpoint_payload),
        "checkpoint_recorded_metric_comparison": checkpoint_comparison,
        "test_selected_checkpoint_disclosure": (
            "best_miou was selected by repeated evaluation of this fixed test split "
            "during epochs 500-1000; this export performs no further selection"
        ),
        "device": str(device),
        "seed": source_runner.SEED,
        "image_size": config["image_size"],
        "threshold_rule": "sigmoid(logit) > 0.5",
        "prediction_mask_encoding": "uint8 PNG with background=0, foreground=255",
        "probability_map_encoding": (
            "NumPy .npy float32 probability on the 256x256 evaluation grid"
            if config["save_probability_maps"]
            else None
        ),
        "official": official.to_dict(),
        "official_reported_operating_point": reported,
        "unified": unified.to_dict(),
        "artifact_counts": {
            "prediction_masks": len(records),
            "probability_maps": len(records)
            if config["save_probability_maps"]
            else 0,
            "per_image_records": len(records),
            "visualizations": len(visualizations),
        },
        "visualizations": visualizations,
        "checks": {
            "checkpoint_selection_metric_is_miou": True,
            "checkpoint_split_hash_matches": True,
            "ordered_ids_match_split": True,
            "repeat_logit_exact_first_image": repeat_exact,
            "model_state_sha256_before": state_hash_before,
            "model_state_sha256_after": state_hash_after,
            "model_state_unchanged": True,
        },
        "runtime_seconds": time.perf_counter() - started,
        "repository_provenance": _runtime_provenance(),
    }
    source_runner.write_jsonl_atomic(output_dir / "per_image.jsonl", records)
    source_runner.write_json_atomic(output_dir / "metrics.json", aggregate)
    if checkpoint_comparison["evaluated"] and not checkpoint_comparison["passed"]:
        raise RuntimeError(
            "reloaded best_miou metrics do not match checkpoint-recorded values: "
            f"{checkpoint_comparison['metrics']}"
        )
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_export(args)
    output_dir = Path(load_export_config(args)["output_dir"])
    metrics = result["official_reported_operating_point"]
    print(f"Artifacts: {output_dir}")
    print(
        f"{result['dataset']} best_miou clean test: "
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
