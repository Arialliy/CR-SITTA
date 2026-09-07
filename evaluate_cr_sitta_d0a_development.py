#!/usr/bin/env python3
"""Evaluate the completed NUDT D0-A checkpoint on the fixed development test.

This independent runner reuses the clean baseline's dataset, model adapter,
official/unified evaluators and prediction writers. It accepts a final-epoch
weights-only export, never best-checkpoint metadata, and never selects weights.
Default operation is read-only preflight; ``--execute`` publishes a new run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import export_cr_sitta_d0a_safe_checkpoint as safe_export
import test_fixed_split_source as clean_export
import test_source as source
from dataio.research_dataset import IRSTDResearchDataset, read_split_ids
from metrics.irstd_metrics import UnifiedResearchEvaluator, probabilities_from_logits
from metrics.official_metric_adapter import OfficialMetricAdapter


ROOT = Path(__file__).resolve().parent
DATASET = "NUDT-SIRST"
DATA_ROOT = ROOT / "datasets" / DATASET
SPLIT = DATA_ROOT / "img_idx" / "test_NUDT-SIRST.txt"
SPLIT_SHA256 = "a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e"
IMAGE_COUNT = 664
IMAGE_SIZE = 256
THRESHOLD_RULE = "sigmoid(logit) > 0.5"
TRAIN_RUN = ROOT / "results/cr_sitta/d0a_supervised_lfhf_train_v2" / DATASET
OUTPUT = ROOT / "results/cr_sitta/d0a_development_test_v1" / DATASET
EXECUTION_PLAN = ROOT / "results/cr_sitta/d0a_continuation_20260907/EXECUTION_PLAN.json"
BASELINES = {
    "best_miou": ROOT / "results/baseline/NUDT-SIRST/best_miou",
    "best_pd": ROOT / "results/baseline_checkpoint_axis_v2/best_pd/NUDT-SIRST",
}
EVALUATOR_PATHS = (
    "metrics/connected_components.py", "metrics/irstd_metrics.py",
    "metrics/official_metric_adapter.py", "metrics/target_matching.py",
    "test_source.py",
)
PREPROCESSING_PATHS = ("dataio/research_dataset.py", "tta/model_adapter.py")
RUNTIME_PATHS = EVALUATOR_PATHS + (
    "evaluate_cr_sitta_d0a_development.py", "export_cr_sitta_d0a_safe_checkpoint.py",
    "test_fixed_split_source.py", "dataio/research_dataset.py",
    "tta/model_adapter.py", "model/MSHNet_NSFPN.py", "utils/metric.py",
    "configs/protocol.yaml", "configs/retrain_fixed_splits.yaml",
    "corruptions/corruption_protocol.py",
)
ROLE = {
    "development_only": True, "paper_result": False,
    "evaluation_role": "development_test", "checkpoint_test_selected": False,
    "formal_test": False, "tta": False,
    "d0b_promotion_authorized": False, "d1_promotion_authorized": False,
}
METRIC_NAMES = ("miou", "pd", "fa_per_pixel_x1e6")


def _require(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve(strict=True)), "sha256": source.sha256_file(path)}


def _new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    source.write_json_atomic(path, value)


def reserve_output(path: Path) -> None:
    """An existing or partial run is never reused, even without COMPLETE."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)


def validate_checkpoint_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    provenance, state = safe_export._validate_safe_payload(payload)
    expected = {
        "dataset": DATASET, "epoch": 1000, "architecture": "MSHNet_NSFPN",
        "selection_rule": "fixed_final_epoch_train_only", "test_selected": False,
        "development_only": True, "state_dict_keys": 505,
    }
    for key, value in expected.items():
        _require(provenance.get(key), value, f"checkpoint.{key}")
    if len(state) != 505:
        raise ValueError("checkpoint must contain exactly 505 state keys")
    for field in ("split_manifest", "access_firewall"):
        for prefix in ("test", "validation"):
            for suffix in ("split_reads", "image_opens", "mask_opens"):
                key = f"{prefix}_{suffix}"
                _require(provenance.get(field, {}).get(key), 0, f"{field}.{key}")
    for key, tensor in state.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"non-finite checkpoint tensor: {key}")
    return provenance


def validate_baseline_protocol(
    metrics: Mapping[str, Any], runtime_hashes: Mapping[str, str], *,
    axis: str, expected_evaluator_hashes: Mapping[str, str],
) -> None:
    """Require the same complete split and actual evaluator implementations."""
    for key, value in {
        "dataset": DATASET, "split_sha256": SPLIT_SHA256,
        "available_images": IMAGE_COUNT, "evaluated_images": IMAGE_COUNT,
        "complete_fixed_test_split": True, "image_size": IMAGE_SIZE,
        "threshold_rule": THRESHOLD_RULE, "seed": source.SEED,
    }.items():
        _require(metrics.get(key), value, f"{axis}.{key}")
    _require(Path(metrics["split_file"]).resolve(), SPLIT.resolve(), f"{axis}.split_path")
    _require(Path(metrics["dataset_root"]).resolve(), DATA_ROOT.resolve(), f"{axis}.root")
    metadata = metrics["checkpoint_metadata"]
    for key, value in {
        "dataset": DATASET, "test_selected": True,
        "selection_metric": {"best_miou": "miou", "best_pd": "pd"}[axis],
    }.items():
        _require(metadata.get(key), value, f"{axis}.metadata.{key}")
    _require(metadata["split_manifest"].get("test_split_sha256"), SPLIT_SHA256,
             f"{axis}.checkpoint_split")
    _require(metadata["test_metrics"].get("images"), IMAGE_COUNT, f"{axis}.checkpoint_count")
    for relative, digest in expected_evaluator_hashes.items():
        _require(runtime_hashes.get(relative), digest, f"{axis}.evaluator.{relative}")
    _require(metrics["checkpoint_recorded_metric_comparison"].get("passed"), True,
             f"{axis}.checkpoint_metric_parity")
    for name in METRIC_NAMES:
        actual = float(metrics["official_reported_operating_point"][name])
        if not math.isfinite(actual) or actual != float(metadata["test_metrics"][name]):
            raise ValueError(f"{axis}.{name} does not match its recorded checkpoint")


def _validate_sealed_metadata(directory: Path) -> list[dict[str, str]]:
    complete_path, manifest_path = directory / "COMPLETE.json", directory / "artifact_manifest.json"
    complete, manifest = _read_json(complete_path), _read_json(manifest_path)
    _require(complete.get("complete"), True, "baseline completion")
    _require(source.sha256_file(manifest_path), complete["manifest_sha256"], "baseline manifest seal")
    files = {item["path"]: item for item in manifest["payload_tree"]["files"]}
    bindings = [_binding(complete_path), _binding(manifest_path)]
    for relative in ("metrics.json", "provenance.json", "per_image.jsonl", "run_config.yaml"):
        path = directory / relative
        _require(source.sha256_file(path), files[relative]["sha256"], f"sealed baseline {relative}")
        bindings.append(_binding(path))
    return bindings


def _best_miou_preprocessing_parity(metrics: Mapping[str, Any]) -> list[dict[str, str]]:
    receipt_path = ROOT / "results/checkpoint_axis_v2_parity/best_miou/PARITY_RECEIPT.json"
    candidate_dir = ROOT / "results/checkpoint_axis_v2_parity_candidates/best_miou/clean" / DATASET
    receipt = _read_json(receipt_path)
    _require(receipt.get("passed"), True, "best_miou parity")
    _require(receipt["clean"].get("passed"), True, "best_miou clean parity")
    _require(Path(receipt["reference_roots"]["clean"]).resolve(), BASELINES["best_miou"].parent.parent.resolve(),
             "best_miou parity reference root")
    _require(Path(receipt["candidate_roots"]["clean"]).resolve(), candidate_dir.parent.resolve(),
             "best_miou parity candidate root")
    seals = receipt["clean"]["datasets"][DATASET]["candidate_seals"]
    for relative in ("COMPLETE.json", "artifact_manifest.json"):
        _require(source.sha256_file(candidate_dir / relative), seals[relative], f"parity {relative}")
    candidate = _read_json(candidate_dir / "metrics.json")
    for field in ("checkpoint_sha256", "official_reported_operating_point", "split_sha256",
                  "evaluated_images", "image_size", "threshold_rule"):
        _require(candidate[field], metrics[field], f"best_miou current-runtime parity {field}")
    runtime = _read_json(candidate_dir / "provenance.json")["runtime_file_sha256"]
    validate_baseline_protocol(candidate, runtime, axis="best_miou", expected_evaluator_hashes={
        relative: source.sha256_file(ROOT / relative)
        for relative in EVALUATOR_PATHS + PREPROCESSING_PATHS})
    return [_binding(receipt_path), *_validate_sealed_metadata(candidate_dir)]


def load_baselines(identifiers: Sequence[str]) -> tuple[dict[str, Any], list[dict[str, str]]]:
    baselines: dict[str, Any] = {}
    bindings: list[dict[str, str]] = []
    evaluator_hashes = {p: source.sha256_file(ROOT / p) for p in EVALUATOR_PATHS}
    expected_ids = [Path(item).with_suffix("").as_posix() for item in identifiers]
    for axis, directory in BASELINES.items():
        metrics_path = directory / "metrics.json"
        metrics = _read_json(metrics_path)
        if axis == "best_miou":
            runtime = metrics["repository_provenance"]["file_sha256"]
            bindings.extend(_best_miou_preprocessing_parity(metrics))
        else:
            provenance_path = directory / "provenance.json"
            runtime = _read_json(provenance_path)["runtime_file_sha256"]
            bindings.append(_binding(provenance_path))
            complete_path = directory / "COMPLETE.json"
            complete = _read_json(complete_path)
            for key, value in {"complete": True, "dataset": DATASET,
                               "checkpoint_role": axis, "evaluated_images": IMAGE_COUNT}.items():
                _require(complete.get(key), value, f"{axis}.complete.{key}")
            bindings.append(_binding(complete_path))
            bindings.extend(_validate_sealed_metadata(directory))
            for relative in PREPROCESSING_PATHS:
                _require(runtime.get(relative), source.sha256_file(ROOT / relative),
                         f"best_pd preprocessing {relative}")
        validate_baseline_protocol(metrics, runtime, axis=axis,
                                  expected_evaluator_hashes=evaluator_hashes)
        records_path = directory / "per_image.jsonl"
        rows = [json.loads(line) for line in records_path.read_text().splitlines()]
        _require([row["image_id"] for row in rows], expected_ids, f"{axis}.ordered_ids")
        for index, row in enumerate(rows):
            for key, expected in {"index": index, "dataset": DATASET,
                                  "corruption": "clean", "severity": 0,
                                  "seed": source.SEED}.items():
                _require(row.get(key), expected, f"{axis}.record.{key}")
        checkpoint = Path(metrics["checkpoint"])
        _require(source.sha256_file(checkpoint), metrics["checkpoint_sha256"],
                 f"{axis}.checkpoint_sha256")
        bindings.extend((_binding(metrics_path), _binding(records_path), _binding(checkpoint)))
        baselines[axis] = {
            "metrics": metrics, "metrics_file": _binding(metrics_path),
            "evaluator_sha256": evaluator_hashes,
            "preprocessing": "RGB, PIL bilinear image / nearest mask resize to 256, ImageNet normalization",
            "baseline_checkpoint_test_selected": True,
        }
    return baselines, bindings


def build_comparison(measured: Mapping[str, float], baselines: Mapping[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for axis, baseline in baselines.items():
        reference = baseline["metrics"]["official_reported_operating_point"]
        deltas = {name: float(measured[name]) - float(reference[name]) for name in METRIC_NAMES}
        comparisons[axis] = {
            "baseline": dict(reference), "measured": dict(measured),
            "delta": deltas,
            "delta_percentage_points": {name: deltas[name] * 100 for name in ("miou", "pd")},
            "improved": {name: deltas[name] < 0 if name == "fa_per_pixel_x1e6"
                         else deltas[name] > 0 for name in METRIC_NAMES},
            "baseline_checkpoint_test_selected": True,
            "baseline_metrics_file": baseline["metrics_file"],
        }
    return {"schema_version": 1, "dataset": DATASET, "condition": "clean", **ROLE,
            "comparisons": comparisons,
            "interpretation": "Single fixed-final-epoch development comparison; no checkpoint selection, scientific gate or overall-method success is inferred."}


def _comparison_markdown(comparison: Mapping[str, Any]) -> str:
    text = ["# NUDT-SIRST D0-A epoch 1000 clean 开发评测", "",
            "固定 test 664 张，256×256，sigmoid(logit) > 0.5。", "",
            "| 权重 | mIoU (%) | PD (%) | Fa (×10⁻⁶) |",
            "|---|---:|---:|---:|"]
    first = next(iter(comparison["comparisons"].values()))["measured"]
    rows = [("CR-SITTA D0-A epoch 1000", first)] + [
        (f"NS-FPN {axis}", item["baseline"])
        for axis, item in comparison["comparisons"].items()]
    for name, metrics in rows:
        text.append(f"| {name} | {metrics['miou'] * 100:.6f} | {metrics['pd'] * 100:.6f} | {metrics['fa_per_pixel_x1e6']:.6f} |")
    text.extend(["", "| 相对基线 | ΔmIoU (百分点) | ΔPD (百分点) | ΔFa (×10⁻⁶) |",
                 "|---|---:|---:|---:|"])
    for axis, item in comparison["comparisons"].items():
        delta = item["delta"]
        text.append(f"| {axis} | {delta['miou'] * 100:+.6f} | {delta['pd'] * 100:+.6f} | {delta['fa_per_pixel_x1e6']:+.6f} |")
    text.extend(["", "此次为单数据集提前开发评测：development_only=true，paper_result=false。",
                 "新权重按固定第 1000 epoch 导出；两份 baseline 权重均由 test 指标选出。",
                 "本结果不授权 D0-B / D1 晋级，不构成 CR-SITTA 全方法或正式论文性能结论。", ""])
    return "\n".join(text)


def preflight(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    _require(checkpoint, (TRAIN_RUN / safe_export.SAFE_NAME).resolve(), "fixed checkpoint path")
    receipt_path = checkpoint.with_name(safe_export.RECEIPT_NAME)
    receipt = safe_export.verify_safe_export(receipt_path)
    _require(Path(receipt["artifacts"]["safe_checkpoint"]["path"]).resolve(), checkpoint,
             "receipt checkpoint path")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = validate_checkpoint_metadata(payload)
    _require(source.sha256_file(SPLIT), SPLIT_SHA256, "fixed test split SHA-256")
    identifiers = read_split_ids(SPLIT)
    _require(len(identifiers), IMAGE_COUNT, "fixed test count")
    evaluation_protocol = source.build_unified_evaluation_protocol()
    _require(evaluation_protocol.fixed_probability_threshold, 0.5, "unified fixed probability threshold")
    baselines, bindings = load_baselines(identifiers)
    bindings.extend((_binding(checkpoint), _binding(receipt_path), _binding(SPLIT),
                     _binding(EXECUTION_PLAN)))
    bindings.extend(receipt["artifacts"].values())
    bindings.extend(_binding(ROOT / relative) for relative in RUNTIME_PATHS)
    return {"ready": True, "identifiers": identifiers, "payload": payload,
            "provenance": provenance, "receipt": receipt, "baselines": baselines,
            "bindings": list({item["path"]: item for item in bindings}.values())}


def run_evaluation(checkpoint: Path, device_name: str) -> dict[str, Any]:
    started = time.perf_counter()
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing evaluation: {OUTPUT}")
    context = preflight(checkpoint)
    source.seed_everything(source.SEED)
    device = source.resolve_device(device_name)
    model = source.build_nsfpn_model()
    loaded = model.load_state_dict(context["payload"]["state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys or len(model.state_dict()) != 505:
        raise RuntimeError("original model strict 505-key checkpoint load failed")
    model.to(device)
    adapter = source.IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    state_before = source.state_dict_sha256(model.state_dict())
    evaluation_protocol = source.build_unified_evaluation_protocol()
    dataset = IRSTDResearchDataset(DATA_ROOT, split_file=SPLIT, image_size=IMAGE_SIZE,
                                  dataset_name=DATASET, corruption="clean", severity=0,
                                  seed=source.SEED)
    input_manifest = [{"image_id": item.image_id,
                       "image": _binding(item.image_path), "mask": _binding(item.mask_path)}
                      for item in dataset._records]
    context["bindings"].extend(binding for row in input_manifest
                                for binding in (row["image"], row["mask"]))
    contract = {
        "schema_version": 1, "protocol_id": "cr-sitta-d0a-nudt-development-test-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "method_name": "CR-SITTA",
        "method_stage": "D0-A", "dataset": DATASET, "condition": "clean", **ROLE,
        "checkpoint": _binding(checkpoint), "checkpoint_provenance": context["provenance"],
        "split": _binding(SPLIT), "ordered_image_ids": list(context["identifiers"]),
        "image_count": IMAGE_COUNT, "image_size": IMAGE_SIZE,
        "threshold_rule": THRESHOLD_RULE, "seed": source.SEED,
        "normalization": {"mean": source.IMAGENET_MEAN.tolist(), "std": source.IMAGENET_STD.tolist()},
        "preprocessing": "RGB; PIL bilinear image / nearest mask 256x256; mask GT unchanged semantically",
        "evaluators": "unchanged OfficialMetricAdapter and UnifiedResearchEvaluator from baseline",
        "device": str(device), "batch_size": 1, "num_workers": 0,
        "save_all_masks": True, "save_all_float32_probabilities": True,
        "authorization": {"source": "current user conversation", "date": "2026-09-07",
                          "user_message": "继续", "scope": "Evaluate already-completed NUDT epoch1000 and compare baseline performance"},
        "execution_plan": _binding(EXECUTION_PLAN),
        "protocol_deviation": {"type": "single_dataset_early_development_test",
                               "description": "User-authorized NUDT clean development evaluation before all datasets finish and before D0-B; independent from the frozen train-only pipeline.",
                               "training_contract_modified": False,
                               "training_access_counters_modified": False,
                               "no_validation_split": True},
        "comparison_axes": ["best_miou", "best_pd"],
        "input_manifest": input_manifest,
    }
    reserve_output(OUTPUT)
    _new_json(OUTPUT / "run_contract.json", contract)
    freeze = {"schema_version": 1, **ROLE, "run_contract": _binding(OUTPUT / "run_contract.json"),
              "input_bindings": context["bindings"], "frozen_before_first_inference": True}
    _new_json(OUTPUT / "EVALUATION_FREEZE.json", freeze)
    (OUTPUT / "run_contract.json").chmod(0o444)
    (OUTPUT / "EVALUATION_FREEZE.json").chmod(0o444)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    official_evaluator = OfficialMetricAdapter(image_size=IMAGE_SIZE)
    unified_evaluator = UnifiedResearchEvaluator(evaluation_protocol)
    records = []
    for index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=False)
        target_device = batch["mask"].to(device, non_blocking=False)
        metadata = source.metadata_from_batch(batch)
        expected_id = Path(context["identifiers"][index]).with_suffix("").as_posix()
        _require(metadata["image_id"], expected_id, "loader order")
        logits, repeated = source.checked_source_forward(adapter, image, target_device,
                                                         repeat_exact=index == 0)
        logits_cpu, target_cpu = logits.detach().cpu(), target_device.detach().cpu()
        source._update_official_evaluator(official_evaluator, logits_cpu, target_cpu)
        unified_evaluator.update_logits(logits_cpu, target_cpu)
        probability = probabilities_from_logits(logits_cpu)[0, 0].astype(np.float32, copy=False)
        prediction = np.where(probability > 0.5, 255, 0).astype(np.uint8)
        mask_relative = Path("prediction_masks_256") / clean_export._relative_prediction_path(expected_id, ".png")
        probability_relative = Path("probability_maps_256") / clean_export._relative_prediction_path(expected_id, ".npy")
        for relative in (mask_relative, probability_relative):
            if (OUTPUT / relative).exists():
                raise FileExistsError(f"duplicate prediction destination: {relative}")
        clean_export._write_png_atomic(OUTPUT / mask_relative, prediction)
        clean_export._write_npy_atomic(OUTPUT / probability_relative, probability)
        records.append({"index": index, **metadata, "input_shape": list(image.shape),
                        "logit_shape": list(logits.shape), "prediction_mask": str(mask_relative),
                        "prediction_mask_sha256": source.sha256_file(OUTPUT / mask_relative),
                        "probability_map": str(probability_relative),
                        "probability_map_sha256": source.sha256_file(OUTPUT / probability_relative),
                        "probability_map_dtype": "float32", "probability_min": float(probability.min()),
                        "probability_max": float(probability.max()),
                        "pixel_metrics_at_probability_gt_0_5": clean_export._pixel_record(probability, target_cpu.numpy()[0, 0]),
                        "input_source_sha256": input_manifest[index]["image"]["sha256"],
                        "gt_source_sha256": input_manifest[index]["mask"]["sha256"],
                        "repeat_logit_exact": repeated})
        if (index + 1) % 64 == 0 or index + 1 == IMAGE_COUNT:
            print(json.dumps({"evaluated": index + 1, "total": IMAGE_COUNT}), flush=True)
    _require(len(records), IMAGE_COUNT, "completed predictions")
    state_after = source.state_dict_sha256(model.state_dict())
    _require(state_after, state_before, "model state unchanged")
    for binding in context["bindings"]:
        _require(source.sha256_file(Path(binding["path"])), binding["sha256"],
                 f"input binding unchanged: {binding['path']}")
    official, unified = official_evaluator.compute(), unified_evaluator.compute()
    reported = {"miou": float(official.mean_iou), "pd": float(official.detection_probability[0]),
                "fa_per_pixel_x1e6": float(official.false_alarm_pixel_rate[0] * 1e6)}
    if not all(math.isfinite(value) for value in reported.values()):
        raise RuntimeError("non-finite aggregate metrics")
    comparison = build_comparison(reported, context["baselines"])
    metrics = {"schema_version": 1, "method": "CR-SITTA D0-A fixed epoch1000, no TTA",
               "dataset": DATASET, **ROLE, "checkpoint": str(checkpoint.resolve()),
               "checkpoint_sha256": source.sha256_file(checkpoint),
               "checkpoint_provenance": context["provenance"],
               "dataset_root": str(DATA_ROOT), "split_file": str(SPLIT), "split_sha256": SPLIT_SHA256,
               "available_images": IMAGE_COUNT, "evaluated_images": IMAGE_COUNT,
               "complete_fixed_test_split": True, "image_size": IMAGE_SIZE,
               "threshold_rule": THRESHOLD_RULE, "seed": source.SEED,
               "official": official.to_dict(), "unified": unified.to_dict(),
               "official_reported_operating_point": reported,
               "artifact_counts": {"prediction_masks": IMAGE_COUNT, "probability_maps": IMAGE_COUNT,
                                   "per_image_records": IMAGE_COUNT},
               "evaluation_access": {"test_images_inferred": IMAGE_COUNT, "test_masks_evaluated": IMAGE_COUNT,
                                     "validation_payload_opens": 0, "optimizer_steps": 0},
               "checks": {"strict_original_model_505_keys": True, "weights_only_loaded": True,
                          "model_state_sha256_before": state_before, "model_state_sha256_after": state_after,
                          "model_state_unchanged": True, "repeat_logit_exact_first_image": records[0]["repeat_logit_exact"],
                          "input_bindings_unchanged": True, "baseline_evaluator_hashes_match": True},
               "runtime_seconds": time.perf_counter() - started}
    source.write_jsonl_atomic(OUTPUT / "per_image.jsonl", records)
    _new_json(OUTPUT / "metrics.json", metrics)
    _new_json(OUTPUT / "comparison.json", comparison)
    with (OUTPUT / "comparison.md").open("x", encoding="utf-8") as handle:
        handle.write(_comparison_markdown(comparison))
    manifest = {str(path.relative_to(OUTPUT)): source.sha256_file(path)
                for path in sorted(OUTPUT.rglob("*")) if path.is_file()}
    _new_json(OUTPUT / "artifact_manifest.json", {"schema_version": 1, "files": manifest})
    _new_json(OUTPUT / "COMPLETE.json", {"schema_version": 1, "complete": True, "dataset": DATASET,
              **ROLE, "evaluated_images": IMAGE_COUNT, "checkpoint_sha256": metrics["checkpoint_sha256"],
              "artifact_manifest": _binding(OUTPUT / "artifact_manifest.json"),
              "artifact_counts": metrics["artifact_counts"], "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=TRAIN_RUN / safe_export.SAFE_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true", help="Publish the full 664-image development evaluation")
    args = parser.parse_args(argv)
    if args.execute:
        metrics = run_evaluation(args.checkpoint, args.device)
        print(json.dumps({"output": str(OUTPUT), "complete": True,
                          "metrics": metrics["official_reported_operating_point"]}), flush=True)
    else:
        context = preflight(args.checkpoint)
        print(json.dumps({"ready": context["ready"], "writes": 0, "dataset": DATASET,
                          "images": IMAGE_COUNT, "output_exists": OUTPUT.exists(), **ROLE}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
