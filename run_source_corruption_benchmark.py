"""Run Source on the immutable 13-condition fixed-test corruption cache."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
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
import test_fixed_split_source as clean_runner
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "source_corruption_benchmark_fixed_splits.yaml"
PROVENANCE_PATHS = (
    "configs/source_corruption_benchmark_fixed_splits.yaml",
    "dataio/corruption_cache.py",
    "metrics/connected_components.py",
    "metrics/irstd_metrics.py",
    "metrics/official_metric_adapter.py",
    "metrics/target_matching.py",
    "model/MSHNet_NSFPN.py",
    "run_source_corruption_benchmark.py",
    "test_fixed_split_source.py",
    "test_source.py",
    "tta/model_adapter.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True, choices=("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_protocol(path: Path, dataset_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("benchmark protocol must be a YAML mapping")
    protocol = dict(loaded)
    if protocol.get("protocol_id") != "nsfpn-source-corruption-benchmark-fixed-splits-v1":
        raise ValueError("unexpected source corruption benchmark protocol_id")
    datasets = protocol.get("datasets", {})
    if dataset_name not in datasets:
        raise ValueError(f"dataset {dataset_name!r} is absent from benchmark protocol")
    return protocol, dict(datasets[dataset_name])


def _conditions(protocol: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    actual = tuple(
        (str(corruption), int(severity))
        for corruption, severity in protocol["corruption"]["ordered_conditions"]
    )
    expected = (
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
    if actual != expected:
        raise ValueError("formal Source benchmark requires the exact 13 conditions")
    return actual


def _evaluation_protocol(protocol: Mapping[str, Any]) -> IRSTDEvaluationProtocol:
    config = protocol["evaluation"]
    if config["threshold_rule"] != "strict_greater_than":
        raise ValueError("unsupported threshold rule")
    matching = config["target_matching"]
    if (
        matching["assignment"] != "hungarian_minimum_centroid_distance"
        or matching["comparison"] != "strict_less_than"
        or matching["one_to_one"] is not True
    ):
        raise ValueError("target matching contract drifted")
    return IRSTDEvaluationProtocol(
        fixed_probability_threshold=float(config["fixed_probability_threshold"]),
        froc_probability_thresholds=tuple(
            float(value) for value in config["froc_probability_thresholds"]
        ),
        connectivity={4: 1, 8: 2}[int(config["foreground_connectivity_2d"])],
        max_centroid_distance=float(matching["max_centroid_distance_pixels"]),
        min_component_area=1,
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError(f"expected JSON object: {path}")
    return loaded


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            loaded = json.loads(line)
            if not isinstance(loaded, Mapping):
                raise TypeError(f"expected JSON object at {path}:{line_number}")
            records.append(loaded)
    return records


def _canonical_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON value trees exactly after normalising Python containers.

    ``UnifiedResearchEvaluator.to_dict()`` intentionally exposes its FROC
    sequence as a tuple, whereas loading the frozen JSON reference yields a
    list.  Both represent the same JSON array.  Serialising both operands also
    keeps this an exact comparison: no numeric tolerance or rounding is used.
    """

    options = {
        "ensure_ascii": False,
        "sort_keys": True,
        "separators": (",", ":"),
        "allow_nan": False,
    }
    return json.dumps(left, **options) == json.dumps(right, **options)


def _checkpoint_payload(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("best_miou checkpoint must be a metadata mapping")
    return payload


def _validate_checkpoint(
    path: Path,
    *,
    dataset_name: str,
    dataset_contract: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if sha256_file(path) != dataset_contract["checkpoint_sha256"]:
        raise ValueError("best_miou checkpoint SHA256 drifted")
    payload = _checkpoint_payload(path)
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
        raise KeyError(f"checkpoint metadata missing: {missing}")
    expected = {
        "schema_version": 1,
        "architecture": protocol["source"]["architecture"],
        "dataset": dataset_name,
        "epoch": int(dataset_contract["checkpoint_epoch"]),
        "selection_metric": "miou",
        "selection_rule": "maximize_miou_then_pd_then_minimize_fa",
        "test_selected": True,
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise ValueError(f"checkpoint {key} mismatch")
    split_manifest = payload["split_manifest"]
    if (
        split_manifest.get("test_split_sha256")
        != dataset_contract["test_split_sha256"]
        or int(split_manifest.get("test_count", -1)) != int(dataset_contract["test_images"])
        or split_manifest.get("corpus_manifest_sha256")
        != dataset_contract["corpus_manifest_sha256"]
        or split_manifest.get("overlap_count") != 0
    ):
        raise ValueError("checkpoint fixed-split manifest mismatch")
    run_config = payload["run_config"]
    if (
        run_config.get("protocol_sha256")
        != protocol["source"]["parent_training_protocol_sha256"]
        or run_config.get("max_train_batches") is not None
        or run_config.get("max_test_images") is not None
    ):
        raise ValueError("checkpoint parent training protocol mismatch")
    metrics = payload["test_metrics"]
    if (
        float(payload["selection_value"]) != float(metrics["miou"])
        or int(metrics["images"]) != int(dataset_contract["test_images"])
    ):
        raise ValueError("checkpoint selected metric metadata is inconsistent")
    return payload, clean_runner._checkpoint_summary(payload)


def _atomic_probability_memmap(
    destination: Path,
    count: int,
) -> tuple[Path, np.memmap]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    values = np.lib.format.open_memmap(
        partial,
        mode="w+",
        dtype=np.dtype("<f4"),
        shape=(count, 256, 256),
        fortran_order=False,
        version=(2, 0),
    )
    return partial, values


def _raw_array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _clean_references(
    dataset_contract: Mapping[str, Any],
    image_ids: Sequence[str],
) -> tuple[Path, Mapping[str, Any], list[Mapping[str, Any]]]:
    root = _project_path(dataset_contract["clean_reference"])
    metrics_path = root / "metrics.json"
    per_image_path = root / "per_image.jsonl"
    if sha256_file(metrics_path) != dataset_contract["clean_reference_metrics_sha256"]:
        raise ValueError("clean reference metrics.json SHA256 drifted")
    if sha256_file(per_image_path) != dataset_contract["clean_reference_per_image_sha256"]:
        raise ValueError("clean reference per_image.jsonl SHA256 drifted")
    metrics = _load_json(metrics_path)
    records = _load_jsonl(per_image_path)
    if len(records) != len(image_ids):
        raise ValueError("clean reference record count drifted")
    for index, (expected_id, record) in enumerate(zip(image_ids, records)):
        if int(record["index"]) != index or record["image_id"] != expected_id:
            raise ValueError("clean reference ID order drifted")
        for field in ("prediction_mask", "probability_map"):
            path = root / record[field]
            if sha256_file(path) != record[f"{field}_sha256"]:
                raise ValueError(f"clean reference {field} file hash drifted")
    return root, metrics, records


def _condition_summary(
    official: Any,
    unified: Any,
) -> dict[str, Any]:
    return {
        "legacy_mean_iou": float(official.mean_iou),
        "legacy_pd": float(official.detection_probability[0]),
        "legacy_fa_per_million_pixels": float(
            official.false_alarm_pixel_rate[0] * 1_000_000.0
        ),
        "unified_global_iou": float(unified.fixed.pixel.intersection_over_union),
        "unified_pd": float(unified.fixed.target.detection_probability),
        "unified_fa_per_million_pixels": float(
            unified.fixed.target.false_alarm_pixel_rate * 1_000_000.0
        ),
        "unified_false_positives_per_image": float(
            unified.fixed.target.false_positives_per_image
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    protocol_path = args.protocol.expanduser().resolve()
    protocol, dataset_contract = _load_protocol(protocol_path, args.dataset)
    protocol_hash = sha256_file(protocol_path)
    conditions = _conditions(protocol)
    corruption_contract = protocol["corruption"]
    if (
        sha256_file(_project_path(corruption_contract["severity_table"]))
        != corruption_contract["severity_table_sha256"]
    ):
        raise ValueError("frozen severity table SHA256 drifted")
    if (
        sha256_file(_project_path(corruption_contract["pilot_report"]))
        != corruption_contract["pilot_report_sha256"]
    ):
        raise ValueError("Pilot report SHA256 drifted")
    cache_contract = protocol["materialized_cache"]
    if (
        sha256_file(_project_path(cache_contract["generation_protocol_archive"]))
        != cache_contract["generation_protocol_sha256"]
    ):
        raise ValueError("cache generation protocol archive SHA256 drifted")
    cache_root = _project_path(protocol["materialized_cache"]["root"])
    cache_dir = (args.cache_dir or cache_root / args.dataset).expanduser().resolve()
    expected_cache = dataset_contract.get("materialized_cache")
    if not isinstance(expected_cache, Mapping):
        raise ValueError("consumer protocol has not frozen this dataset cache")
    cache_manifest, cache_audit = verify_cache_artifact(
        cache_dir,
        expected_protocol_sha256=str(expected_cache["generation_protocol_sha256"]),
        verify_file_hashes=True,
    )
    if (
        cache_audit["manifest_sha256"] != expected_cache["manifest_sha256"]
        or cache_manifest["cache_content_sha256"] != expected_cache["content_sha256"]
        or cache_manifest["dataset"] != args.dataset
        or cache_manifest["split_sha256"] != dataset_contract["test_split_sha256"]
        or cache_manifest["ordered_ids_sha256"]
        != dataset_contract["ordered_test_ids_sha256"]
        or int(cache_audit["verified_total_bytes"]) != int(expected_cache["bytes"])
    ):
        raise ValueError("materialized cache lineage does not match consumer protocol")
    cached_conditions = tuple(
        (record["corruption"], int(record["severity"]))
        for record in cache_manifest["conditions"]
    )
    if cached_conditions != conditions:
        raise ValueError("materialized cache condition order drifted")
    image_ids = tuple(str(value) for value in cache_manifest["image_ids"])
    if len(image_ids) != int(dataset_contract["test_images"]):
        raise ValueError("materialized cache image count drifted")
    if ordered_ids_sha256(image_ids) != dataset_contract["ordered_test_ids_sha256"]:
        raise ValueError("materialized cache ordered IDs drifted")

    checkpoint = _project_path(dataset_contract["checkpoint"])
    checkpoint_payload, checkpoint_summary = _validate_checkpoint(
        checkpoint,
        dataset_name=args.dataset,
        dataset_contract=dataset_contract,
        protocol=protocol,
    )
    reference_root, reference_metrics, reference_records = _clean_references(
        dataset_contract, image_ids
    )
    final_root = _project_path(protocol["outputs"]["root"]) / args.dataset / "best_miou"
    if args.output_dir is not None:
        final_root = args.output_dir.expanduser().resolve()
    if final_root.exists():
        raise FileExistsError(f"Source benchmark output already exists: {final_root}")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = final_root.with_name(f".{final_root.name}.build-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Source benchmark staging directory exists: {staging}")
    staging.mkdir(parents=False)

    source_runner.seed_everything(int(protocol["corruption"]["seed"]))
    device = source_runner.resolve_device(args.device)
    model = source_runner.build_nsfpn_model()
    checkpoint_wrapper = source_runner.load_trusted_checkpoint(model, checkpoint)
    model.to(device)
    adapter = source_runner.IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    state_hash_before = source_runner.state_dict_sha256(model.state_dict())
    evaluation_protocol = _evaluation_protocol(protocol)
    cache_conditions = {
        record["key"]: record for record in cache_manifest["conditions"]
    }
    condition_summaries = []
    all_files: dict[str, dict[str, Any]] = {}
    clean_parity: dict[str, Any] | None = None

    for condition_index, (corruption, severity) in enumerate(conditions):
        condition_started = time.perf_counter()
        key = condition_key(corruption, severity)
        cache_condition = cache_conditions[key]
        condition_dir = staging / "conditions" / key
        condition_dir.mkdir(parents=True)
        dataset = CachedCorruptionDataset(
            cache_dir,
            corruption=corruption,
            severity=severity,
            manifest=cache_manifest,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        probability_path = condition_dir / "probabilities_256.npy"
        probability_partial, probability_map = _atomic_probability_memmap(
            probability_path, len(dataset)
        )
        official_evaluator = OfficialMetricAdapter(image_size=256)
        unified_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
        input_hasher = TensorSequenceHasher()
        target_hasher = TensorSequenceHasher()
        probability_hasher = TensorSequenceHasher()
        records = []
        repeat_exact = False
        clean_probabilities_exact = True
        clean_masks_exact = True
        clean_mask_files_exact = True
        clean_probability_arrays_compared = 0
        clean_probability_values_compared = 0

        for index, batch in enumerate(loader):
            metadata = source_runner.metadata_from_batch(batch)
            image_id = str(metadata["image_id"])
            if image_id != image_ids[index]:
                raise RuntimeError("cached loader order drifted")
            image_cpu = batch["image"]
            target_cpu = batch["mask"]
            input_hasher.update(image_id, image_cpu[0])
            target_hasher.update(image_id, target_cpu[0])
            image = image_cpu.to(device, non_blocking=False)
            target_device = target_cpu.to(device, non_blocking=False)
            logits, repeated = source_runner.checked_source_forward(
                adapter,
                image,
                target_device,
                repeat_exact=index == 0,
            )
            if index == 0:
                repeat_exact = bool(repeated)
            logits_cpu = logits.detach().cpu()
            source_runner._update_official_evaluator(
                official_evaluator, logits_cpu, target_cpu
            )
            unified_evaluator.update_logits(logits_cpu, target_cpu)
            probability = probabilities_from_logits(logits_cpu)[0, 0].astype(
                np.float32, copy=False
            )
            binary_mask = np.where(probability > 0.5, 255, 0).astype(np.uint8)
            probability_map[index] = probability
            probability_hasher.update(image_id, probability)
            mask_relative = Path("prediction_masks_256") / clean_runner._relative_prediction_path(
                image_id, ".png"
            )
            mask_path = condition_dir / mask_relative
            clean_runner._write_png_atomic(mask_path, binary_mask)
            mask_file_hash = sha256_file(mask_path)

            if condition_index == 0:
                reference = reference_records[index]
                reference_probability = np.load(
                    reference_root / reference["probability_map"], allow_pickle=False
                )
                reference_mask = np.asarray(
                    Image.open(reference_root / reference["prediction_mask"]).convert("L")
                )
                probability_equal = np.array_equal(probability, reference_probability)
                mask_equal = np.array_equal(binary_mask, reference_mask)
                mask_file_equal = mask_file_hash == reference["prediction_mask_sha256"]
                clean_probability_arrays_compared += 1
                clean_probability_values_compared += int(probability.size)
                clean_probabilities_exact &= probability_equal
                clean_masks_exact &= mask_equal
                clean_mask_files_exact &= mask_file_equal
                if not probability_equal or not mask_equal or not mask_file_equal:
                    raise RuntimeError(
                        f"clean cached inference diverged from reference at {image_id}"
                    )

            target = target_cpu.numpy()[0, 0]
            records.append(
                {
                    "index": index,
                    **metadata,
                    "cache_content_sha256": cache_manifest["cache_content_sha256"],
                    "cache_condition_tensor_sha256": cache_condition[
                        "tensor_sequence_sha256"
                    ],
                    "probability_shard": "probabilities_256.npy",
                    "probability_shard_index": index,
                    "probability_tensor_raw_sha256": _raw_array_sha256(probability),
                    "probability_min": float(probability.min()),
                    "probability_max": float(probability.max()),
                    "prediction_mask": str(mask_relative),
                    "prediction_mask_sha256": mask_file_hash,
                    "pixel_metrics_at_probability_gt_0_5": clean_runner._pixel_record(
                        probability, target
                    ),
                }
            )

        probability_map.flush()
        del probability_map
        os.replace(probability_partial, probability_path)
        if len(records) != len(image_ids):
            raise RuntimeError("condition prediction count drifted")
        if input_hasher.hexdigest() != cache_condition["tensor_sequence_sha256"]:
            raise RuntimeError(f"consumer input tensor hash mismatch for {key}")
        if target_hasher.hexdigest() != cache_manifest["targets"]["tensor_sequence_sha256"]:
            raise RuntimeError(f"consumer target tensor hash mismatch for {key}")
        state_hash_after_condition = source_runner.state_dict_sha256(model.state_dict())
        if state_hash_after_condition != state_hash_before:
            raise RuntimeError(f"Source model state changed under {key}")
        official = official_evaluator.compute()
        unified = unified_evaluator.compute()
        summary = _condition_summary(official, unified)
        if condition_index == 0:
            official_equal = _canonical_json_equal(
                official.to_dict(), reference_metrics["official"]
            )
            unified_equal = _canonical_json_equal(
                unified.to_dict(), reference_metrics["unified"]
            )
            checkpoint_equal = (
                summary["legacy_mean_iou"]
                == float(checkpoint_payload["test_metrics"]["miou"])
                and summary["legacy_pd"]
                == float(checkpoint_payload["test_metrics"]["pd"])
                and summary["legacy_fa_per_million_pixels"]
                == float(checkpoint_payload["test_metrics"]["fa_per_pixel_x1e6"])
            )
            clean_parity = {
                "comparison_rule": "exact_after_json_container_canonicalization",
                "numeric_tolerance_used": False,
                "probability_arrays_compared": clean_probability_arrays_compared,
                "probability_values_compared": clean_probability_values_compared,
                "all_probability_arrays_exact": clean_probabilities_exact,
                "all_binary_mask_arrays_exact": clean_masks_exact,
                "all_binary_mask_file_hashes_exact": clean_mask_files_exact,
                "official_metrics_exact": official_equal,
                "unified_metrics_exact": unified_equal,
                "checkpoint_operating_point_exact": checkpoint_equal,
                "passed": all(
                    (
                        clean_probabilities_exact,
                        clean_masks_exact,
                        clean_mask_files_exact,
                        official_equal,
                        unified_equal,
                        checkpoint_equal,
                    )
                ),
            }
            if not clean_parity["passed"]:
                raise RuntimeError(f"clean parity gate failed: {clean_parity}")

        probability_file_hash = sha256_file(probability_path)
        for record in records:
            record["probability_shard_sha256"] = probability_file_hash
        per_image_path = condition_dir / "per_image.jsonl"
        source_runner.write_jsonl_atomic(per_image_path, records)
        metrics = {
            "schema_version": 1,
            "method": "Source",
            "dataset": args.dataset,
            "condition_index": condition_index,
            "condition_key": key,
            "corruption": corruption,
            "severity": severity,
            "evaluated_images": len(records),
            "cache_lineage": {
                "cache_dir": str(cache_dir),
                "cache_manifest_sha256": cache_audit["manifest_sha256"],
                "cache_content_sha256": cache_manifest["cache_content_sha256"],
                "condition_file_sha256": cache_condition["file_sha256"],
                "condition_tensor_sequence_sha256": cache_condition[
                    "tensor_sequence_sha256"
                ],
                "target_tensor_sequence_sha256": cache_manifest["targets"][
                    "tensor_sequence_sha256"
                ],
                "ordered_ids_sha256": cache_manifest["ordered_ids_sha256"],
            },
            "summary": summary,
            "official": official.to_dict(),
            "unified": unified.to_dict(),
            "probability_shard": "probabilities_256.npy",
            "probability_shard_sha256": probability_file_hash,
            "probability_tensor_sequence_sha256": probability_hasher.hexdigest(),
            "prediction_mask_count": len(records),
            "checks": {
                "repeat_logit_exact_first_image": repeat_exact,
                "input_hash_matches_cache": True,
                "target_hash_matches_cache": True,
                "ordered_ids_match_cache": True,
                "model_state_unchanged": True,
                "clean_parity": clean_parity if condition_index == 0 else None,
            },
            "runtime_seconds": time.perf_counter() - condition_started,
        }
        metrics_path = condition_dir / "metrics.json"
        source_runner.write_json_atomic(metrics_path, metrics)
        relative_condition = condition_dir.relative_to(staging)
        for path in (probability_path, per_image_path, metrics_path):
            relative = str(path.relative_to(staging))
            all_files[relative] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        for record in records:
            relative = str((condition_dir / record["prediction_mask"]).relative_to(staging))
            path = staging / relative
            all_files[relative] = {
                "sha256": record["prediction_mask_sha256"],
                "bytes": path.stat().st_size,
            }
        condition_summaries.append(
            {
                "condition_index": condition_index,
                "condition_key": key,
                "corruption": corruption,
                "severity": severity,
                **summary,
                "metrics": str((relative_condition / "metrics.json")),
                "metrics_sha256": all_files[
                    str(relative_condition / "metrics.json")
                ]["sha256"],
            }
        )

    state_hash_after = source_runner.state_dict_sha256(model.state_dict())
    if state_hash_after != state_hash_before:
        raise RuntimeError("Source model state changed during benchmark")
    if clean_parity is None or not clean_parity["passed"]:
        raise RuntimeError("clean parity was not established")
    benchmark = {
        "schema_version": 1,
        "method": "Source",
        "protocol_id": protocol["protocol_id"],
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_hash,
        "dataset": args.dataset,
        "condition_count": len(condition_summaries),
        "evaluated_images_per_condition": len(image_ids),
        "sample_condition_count": len(image_ids) * len(condition_summaries),
        "conditions": condition_summaries,
        "cache_audit": cache_audit,
        "cache_content_sha256": cache_manifest["cache_content_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_wrapper": checkpoint_wrapper,
        "checkpoint_metadata": checkpoint_summary,
        "checkpoint_test_selected_disclosure": protocol["source"][
            "checkpoint_disclosure"
        ],
        "clean_reference": str(reference_root),
        "clean_parity": clean_parity,
        "checks": {
            "full_fixed_test_split_all_conditions": True,
            "exact_13_condition_contract": True,
            "corruptions_consumed_only_from_materialized_cache": True,
            "all_prediction_masks_saved": True,
            "all_probability_maps_saved": True,
            "model_loaded_once": True,
            "model_state_sha256_before": state_hash_before,
            "model_state_sha256_after": state_hash_after,
            "model_state_unchanged": True,
        },
        "repository_provenance": source_runner.repository_provenance(PROVENANCE_PATHS),
        "runtime_seconds": time.perf_counter() - started,
    }
    benchmark_path = staging / "benchmark.json"
    source_runner.write_json_atomic(benchmark_path, benchmark)
    all_files["benchmark.json"] = {
        "sha256": sha256_file(benchmark_path),
        "bytes": benchmark_path.stat().st_size,
    }
    artifact_manifest = {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "dataset": args.dataset,
        "cache_manifest_sha256": cache_audit["manifest_sha256"],
        "cache_content_sha256": cache_manifest["cache_content_sha256"],
        "files": all_files,
    }
    artifact_manifest_path = staging / "artifact_manifest.json"
    source_runner.write_json_atomic(artifact_manifest_path, artifact_manifest)
    completion = {
        "complete": True,
        "dataset": args.dataset,
        "protocol_sha256": protocol_hash,
        "cache_content_sha256": cache_manifest["cache_content_sha256"],
        "benchmark_sha256": all_files["benchmark.json"]["sha256"],
        "artifact_manifest_sha256": sha256_file(artifact_manifest_path),
    }
    source_runner.write_json_atomic(staging / "COMPLETE.json", completion)
    os.replace(staging, final_root)
    benchmark["published_output_dir"] = str(final_root)
    return benchmark


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = run_benchmark(args)
    print(f"Artifacts: {benchmark['published_output_dir']}")
    print(
        f"{benchmark['dataset']}: {benchmark['condition_count']} conditions, "
        f"{benchmark['evaluated_images_per_condition']} images/condition, "
        f"clean parity={benchmark['clean_parity']['passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
