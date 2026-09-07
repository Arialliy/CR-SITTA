"""Append-only CPU LF repair audit paired to the sealed v7 Pilot64 inputs.

This module never trains a detector or evaluates a test image.  L0 is a
read-only historical comparator, not a newly regenerated corruption.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import __version__ as pillow_version
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.audit_d0a_probe_label_preservation import (
    FORBIDDEN_ACCESS_FIELDS, VIEWS, audit_visibility, distribution,
    summarize_targets, target_regions, tensor_sha256,
    validate_access_counts, validate_diagnostic_parameters,
)
from tta.deteriorations.image_space import imagenet_denormalize

DEFAULT_CONFIG = ROOT / "configs/cr_sitta_lf_repair_audit_v2.yaml"
VARIANTS = ("clean", "L1", "L2", "L3", "L4a", "L4b")


def stable_probe_seed(namespace: str, global_seed: int, dataset: str,
                      image_id: str, view: str) -> int:
    """Variant is deliberately absent: every new variant shares one field."""
    data = json.dumps([namespace, global_seed, dataset, image_id, view],
                      ensure_ascii=False, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % (2**63)


def _key(row: Mapping[str, Any], *, target: bool = False) -> tuple[Any, ...]:
    fields = ("dataset", "image_id", "view") + (("target_id",) if target else ())
    return tuple(row[name] for name in fields)


def index_legacy_rows(image_rows: Iterable[Mapping[str, Any]],
                      target_rows: Iterable[Mapping[str, Any]]) -> tuple[dict, dict]:
    """Index only sealed LF rows; reject ambiguity rather than choosing one."""
    images, targets = {}, {}
    for rows, output, is_target in ((image_rows, images, False),
                                    (target_rows, targets, True)):
        for row in rows:
            if row["probe_id"] != "lf_mask":
                continue
            key = _key(row, target=is_target)
            if key in output:
                raise ValueError("duplicate sealed L0 pairing key")
            output[key] = dict(row)
    if not images:
        raise ValueError("sealed L0 image rows are missing")
    return images, targets


def assert_sealed_sample(metadata: Mapping[str, Any],
                         legacy_image: Mapping[str, Any]) -> None:
    expected = legacy_image["input_metadata"]
    for field in ("input_tensor_sha256", "target_tensor_sha256", "view",
                  "augmentation_seed", "historical_training_random_stream_replay"):
        if field not in metadata or metadata[field] != expected[field]:
            raise RuntimeError(f"sealed v7 input replay mismatch: {field}")


def enrich_visibility(artifacts: Mapping[str, Any], target: torch.Tensor,
                      visibility: Mapping[str, Any], descriptor: Mapping[str, Any],
                      legacy_targets: Mapping[tuple, Mapping[str, Any]]) -> tuple[dict, list[dict]]:
    """Retain the original E1 function and append explicitly named v2 fields."""
    image, rows = audit_visibility(artifacts, target, visibility=visibility)
    values = [value.detach().cpu()[0].numpy().astype(np.float64)
              for value in (artifacts["clean"], artifacts["preclip"], artifacts["postclip"])]
    regions = target_regions(target.detach().cpu().numpy().squeeze(),
        outer_radius=visibility["ring_outer_radius"], inner_radius=visibility["ring_inner_radius"])
    grays = [value.mean(axis=0) for value in values]
    image_key = _key(descriptor)
    expected_targets = {key for key in legacy_targets if key[:3] == image_key}
    observed_targets = {_key({**descriptor, **row}, target=True) for row in rows}
    if observed_targets != expected_targets:
        raise RuntimeError("target set differs from the sealed L0 image/view")
    enriched = []
    for row, region in zip(rows, regions, strict=True):
        key = _key({**descriptor, **row}, target=True)
        if key not in legacy_targets:
            raise RuntimeError(f"target missing from sealed L0: {key}")
        old = legacy_targets[key]
        for name in ("target_pixels", "ring_pixels", "bbox_yxyx_exclusive",
                     "ring_valid", "contrast_eligible_for_e1", "clean_contrast"):
            if row[name] != old[name]:
                raise RuntimeError(f"sealed target geometry/eligibility mismatch: {key} {name}")
        eligible = bool(row["contrast_eligible_for_e1"])
        fail = bool(row["postclip_contrast_below_ratio_threshold"])
        flip = bool(row["postclip_contrast_sign_flip"])
        old_fail = bool(old["postclip_contrast_below_ratio_threshold"])
        old_flip = bool(old["postclip_contrast_sign_flip"])
        core, ring = region["core"], region["ring"]
        outside = (values[1][:, core] < 0) | (values[1][:, core] > 1)
        peak = {}
        for phase, gray in zip(("clean", "preclip", "postclip"), grays, strict=True):
            peak[f"{phase}_target_peak_gray"] = float(gray[core].max())
            peak[f"{phase}_target_peak_above_ring_mean"] = (
                float(gray[core].max() - gray[ring].mean()) if row["ring_valid"] else None)
        enriched.append({**descriptor, **row, **peak,
            "source_image_id": descriptor["image_id"], "view_id": descriptor["view"],
            "core_pixels": row["target_pixels"], "original_e1_eligibility": eligible,
            "exclusion_reason": (None if eligible else "low_clean_contrast"
                                 if row["ring_valid"] else "empty_background_ring"),
            "contrast_clean": row["clean_contrast"],
            "contrast_preclip": row["preclip_contrast"],
            "contrast_postclip": row["postclip_contrast"],
            "retention_abs_preclip": row["preclip_contrast_retention_ratio"],
            "retention_abs_postclip": row["postclip_contrast_retention_ratio"],
            "sign_flip_preclip": row["preclip_contrast_sign_flip"],
            "sign_flip_postclip": row["postclip_contrast_sign_flip"],
            "original_E1_failure": fail if eligible else None,
            "compound_visibility_failure": (fail or flip) if eligible else None,
            "legacy_L0_original_E1_failure": old_fail if eligible else None,
            "legacy_L0_compound_visibility_failure": (old_fail or old_flip) if eligible else None,
            "actual_clipping_changed_fraction_core": float(outside.mean()),
            "actual_clipping_changed_any_channel_pixel_fraction_core": float(outside.any(axis=0).mean()),
            "endpoint_occupancy_fraction_core": row["postclip_target_clipping"]["rgb_channel_endpoint_fraction"],
        })
    if len(enriched) != len(regions):
        raise RuntimeError("target enumeration differs from original audit")
    image.update({
        "actual_clipping_changed_fraction_image": float(((values[1] < 0) | (values[1] > 1)).mean()),
        "endpoint_occupancy_fraction_image": image["postclip_image_clipping"]["rgb_channel_endpoint_fraction"],
    })
    return image, enriched


def cluster_bootstrap(rows: list[Mapping[str, Any]], image_ids: list[str], *,
                      replicates: int = 2000, seed: int = 42,
                      confidence_level: float = 0.95) -> dict[str, Any]:
    """Paired bootstrap of source IDs, including IDs with no eligible target."""
    if not image_ids or len(image_ids) != len(set(image_ids)):
        raise ValueError("bootstrap requires unique, nonempty image cluster IDs")
    if replicates < 1 or not 0 < confidence_level < 1:
        raise ValueError("invalid bootstrap specification")
    position = {name: i for i, name in enumerate(image_ids)}
    counts = np.zeros((len(image_ids), 5), dtype=np.int64)
    for row in rows:
        if row["image_id"] not in position:
            raise ValueError("target belongs to an undeclared image cluster")
        if row["contrast_eligible_for_e1"]:
            counts[position[row["image_id"]]] += np.array([1,
                bool(row["original_E1_failure"]), bool(row["compound_visibility_failure"]),
                bool(row["legacy_L0_original_E1_failure"]),
                bool(row["legacy_L0_compound_visibility_failure"])], dtype=np.int64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(image_ids), size=(replicates, len(image_ids)))
    totals = counts[samples].sum(axis=1)
    valid = totals[:, 0] > 0
    tail = (1.0 - confidence_level) / 2.0
    def interval(numerator: np.ndarray) -> list[float] | None:
        if not valid.any():
            return None
        return list(map(float, np.quantile(numerator[valid] / totals[valid, 0], [tail, 1-tail])))
    return {"unit": "source_image_id", "replicates": replicates, "seed": seed,
        "confidence_level": confidence_level, "image_clusters": len(image_ids),
        "valid_replicates": int(valid.sum()), "zero_eligible_replicates": int((~valid).sum()),
        "original_e1_fraction_ci": interval(totals[:, 1]),
        "compound_fraction_ci": interval(totals[:, 2]),
        "paired_l0_e1_fraction_delta_ci": interval(totals[:, 1]-totals[:, 3]),
        "paired_l0_compound_fraction_delta_ci": interval(totals[:, 2]-totals[:, 4])}


def summarize_cell(image_rows: list[Mapping[str, Any]], target_rows: list[Mapping[str, Any]], *,
                    visibility: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if not image_rows:
        raise ValueError("cannot summarize a missing cell")
    identities = {(row["dataset"], row["view"], row["probe_id"]) for row in image_rows}
    if len(identities) != 1 or any((r["dataset"], r["view"], r["probe_id"]) not in identities for r in target_rows):
        raise ValueError("mixed dataset/view/probe cell")
    dataset, view, probe_id = next(iter(identities))
    ids = [row["image_id"] for row in image_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate image in one cell")
    legacy = summarize_targets(target_rows, visibility)
    eligible = [row for row in target_rows if row["contrast_eligible_for_e1"]]
    compound = sum(bool(row["compound_visibility_failure"]) for row in eligible)
    old_e1 = sum(bool(row["legacy_L0_original_E1_failure"]) for row in eligible)
    old_compound = sum(bool(row["legacy_L0_compound_visibility_failure"]) for row in eligible)
    denominator = len(eligible)
    ratio = lambda n: n / denominator if denominator else None
    rms = [float(row["postclip_perturbation_rms_rgb"]) for row in image_rows]
    if not np.isfinite(rms).all():
        raise ValueError("non-finite perturbation")
    floor = config["gate"]["image_rms_floor"]
    above = sum(value > floor for value in rms)
    median = float(np.median(rms))
    informative = median > floor and above / len(rms) >= config["gate"]["image_rms_fraction_min"]
    bootstrap = config["bootstrap"]
    return {"dataset": dataset, "view": view, "probe_id": probe_id,
        "image_count": len(ids), "empty_target_view_count": sum(r["empty_target_view"] for r in image_rows),
        **legacy, "compound_visibility_failure_count": compound,
        "compound_visibility_failure_fraction": ratio(compound),
        "compound_denominator": "original_e1_eligible",
        "legacy_L0_e1_failure_count": old_e1,
        "legacy_L0_compound_failure_count": old_compound,
        "paired_L0_e1_fraction_delta": ratio(legacy["eligible_contrast_failure_count"] - old_e1),
        "paired_L0_compound_fraction_delta": ratio(compound - old_compound),
        "bootstrap": cluster_bootstrap(target_rows, ids, replicates=bootstrap["replicates"],
            seed=bootstrap["seed"], confidence_level=bootstrap["confidence_level"]),
        "postclip_perturbation_rms_rgb": distribution(rms),
        "operator_statistics": {name: distribution(row.get(name) for row in image_rows) for name in (
            "sampled_pair_keep_fraction", "sampled_all_pair_keep_fraction", "gain_mean", "gain_min",
            "gain_support_mean", "gain_support_min", "preclip_total_energy_retention",
            "postclip_total_energy_retention", "preclip_support_energy_retention",
            "postclip_support_energy_retention", "preclip_mean_drift_max", "postclip_mean_drift_max",
            "actual_clipping_changed_fraction_image", "endpoint_occupancy_fraction_image")},
        "nonidentity": {"image_rms_floor": floor, "median_rms": median,
            "images_above_floor": above, "image_count": len(rms), "fraction_above_floor": above / len(rms),
            "fraction_min": config["gate"]["image_rms_fraction_min"], "informative": informative,
            "constant_input_image_count": sum(bool(r["constant_input_image"]) for r in image_rows),
            "constant_images_excluded_from_denominator": False},
        "all_input_pairs_exact": all(r["sealed_input_pair_exact"] for r in image_rows),
        "all_gt_unchanged": all(r["gt_tensor_unchanged"] for r in image_rows),
        "all_probe_replays_exact": all(r["deterministic_repeat_exact"] for r in image_rows),
        "scientific_scope": "augmentation_visibility_screen_only", "paper_result": False}


def build_candidate_artifacts(normalized: torch.Tensor, *, candidate: Mapping[str, Any],
                               operator: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Label-free artifact construction; the target is not accepted here."""
    if normalized.device.type != "cpu":
        raise ValueError("LF repair audit is CPU-only")
    clean = imagenet_denormalize(normalized.unsqueeze(0) if normalized.ndim == 3 else normalized)
    clean_before = clean.clone()
    if candidate["probe_id"] == "clean":
        preclip, postclip = clean.clone(), clean.clone()
        diagnostics = {"probe_seed": None, "clean_control": True}
        random_field = None
    else:
        from tta.deteriorations.fourier_low_mask_v2 import LFConfig, lf_mask_diagnostic
        cfg = LFConfig(**operator, attenuation=candidate["attenuation"])
        arguments = {"protect_dc": candidate["protect_dc"], "shared_channels": candidate["shared_channels"]}
        generator = lambda: torch.Generator(device="cpu").manual_seed(seed)
        result = lf_mask_diagnostic(clean, cfg, generator=generator(), **arguments)
        replay = lf_mask_diagnostic(clean, cfg, generator=generator(), **arguments)
        if not (torch.equal(result.image, replay.image) and torch.equal(result.preclip, replay.preclip)
                and torch.equal(result.gain, replay.gain)
                and result.diagnostics == replay.diagnostics):
            raise RuntimeError("new LF deterministic replay mismatch")
        preclip, postclip = result.preclip, result.image
        diagnostics = dict(result.diagnostics)
        random_field = diagnostics.pop("random_field_values", None)
    if not torch.equal(clean, clean_before):
        raise RuntimeError("LF operator modified the input image")
    if not torch.equal(preclip.clamp(0, 1), postclip):
        raise RuntimeError("LF preclip/clamp output mismatch")
    if not all(torch.isfinite(t).all().item() for t in (clean, preclip, postclip)):
        raise RuntimeError("non-finite LF artifacts")
    diagnostics.update({"probe_seed": None if candidate["probe_id"] == "clean" else seed,
        "deterministic_repeat_exact": True, "preclip_clamp_parity_exact": True,
        "clean_physical_tensor_sha256": tensor_sha256(clean),
        "preclip_physical_tensor_sha256": tensor_sha256(preclip),
        "postclip_physical_tensor_sha256": tensor_sha256(postclip),
        "preclip_perturbation_rms_rgb": float((preclip-clean).square().mean().sqrt()),
        "postclip_perturbation_rms_rgb": float((postclip-clean).square().mean().sqrt()),
        "constant_input_image": bool(float(clean.max()-clean.min()) <= 1e-6),
        "hf_noise_rms_rgb": None})
    return {"clean": clean, "preclip": preclip, "postclip": postclip,
            "noise": None, "statistics": diagnostics, "random_field_values": random_field}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from analysis.d0a_v7_common import load_sample, write_jsonl_new
    from analysis.lf_repair_contract_v2 import prepare_run, freeze_dataset, complete_dataset
    if str(args.device) != "cpu":
        raise ValueError("LF repair runner only supports --device cpu")
    prepared = prepare_run(args.config, args.dataset)
    if not args.execute:
        return prepared["preflight"]
    config, legacy_config = prepared["new_config"], prepared["legacy_config"]
    if args.threads != config["threads"]:
        raise ValueError("--threads must equal the preregistered CPU thread count")
    validate_diagnostic_parameters(legacy_config)
    old_images, old_targets = index_legacy_rows(prepared["legacy_image_rows"], prepared["legacy_target_rows"])
    records = prepared["records"]
    expected_pairs = {(args.dataset, record["image_id"], view) for record in records for view in VIEWS}
    if set(old_images) != expected_pairs:
        raise RuntimeError("sealed L0 does not exactly cover requested Pilot64/views")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    prepared["contract"]["runtime"] = {"device": "cpu", "threads": torch.get_num_threads(),
        "python": platform.python_version(), "python_executable": sys.executable,
        "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "pillow": pillow_version,
        "physical_dtype": "float32", "fft_device": "cpu",
        "fft_implementation": "torch.fft.fft2/ifft2",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    freeze_dataset(prepared)
    output = prepared["output"]
    images, targets, fields = [], [], []
    access = {name: 0 for name in FORBIDDEN_ACCESS_FIELDS}
    access.update({"train_image_opens": 0, "train_mask_opens": 0})
    candidates = [{"probe_id": "clean"}, *config["variants"]]
    for index, record in enumerate(records, 1):
        for view in VIEWS:
            normalized, target, metadata = load_sample(record, view, legacy_config, access=access)
            old = old_images[(args.dataset, record["image_id"], view)]
            assert_sealed_sample(metadata, old)
            target_hash = tensor_sha256(target)
            seed = stable_probe_seed(config["probe_seed_namespace"], config["global_seed"],
                                     args.dataset, record["image_id"], view)
            field_payload = None
            for candidate in candidates:
                with torch.no_grad():
                    artifacts = build_candidate_artifacts(normalized, candidate=candidate,
                                                         operator=config["operator"], seed=seed)
                    descriptor = {"dataset": args.dataset, "image_id": record["image_id"], "view": view,
                        "probe_id": candidate["probe_id"],
                        "operator_id": artifacts["statistics"].get("operator_id", "identity_control"),
                        "operator_config_sha256": hashlib.sha256(json.dumps(
                            {**config["operator"], **candidate}, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                        "probe_seed": artifacts["statistics"]["probe_seed"],
                        "input_hash": metadata["input_tensor_sha256"], "target_hash": metadata["target_tensor_sha256"]}
                    descriptor["operator_config_hash"] = descriptor["operator_config_sha256"]
                    image_stats, target_stats = enrich_visibility(artifacts, target,
                        legacy_config["visibility"], descriptor, old_targets)
                if tensor_sha256(target) != target_hash:
                    raise RuntimeError("LF audit modified train GT")
                random_field = artifacts["random_field_values"]
                if random_field is not None:
                    if field_payload is not None and random_field != field_payload:
                        raise RuntimeError("new LF variants did not share the frozen random field")
                    field_payload = random_field
                images.append({**descriptor, "input_metadata": metadata, "sealed_input_pair_exact": True,
                               **artifacts["statistics"], **image_stats,
                               "postclip_rms": artifacts["statistics"]["postclip_perturbation_rms_rgb"]})
                targets.extend(target_stats)
            if field_payload is None:
                raise RuntimeError("LF operator did not expose the preregistered random field")
            fields.append({"dataset": args.dataset, "image_id": record["image_id"], "view": view,
                           "probe_seed": seed, "shared_by": list(VARIANTS[1:]),
                           "random_field_shape": artifacts["statistics"]["random_field_shape"],
                           "random_field_sha256": artifacts["statistics"]["random_field_sha256"],
                           "random_field_values": field_payload})
        if index % 8 == 0 or index == len(records):
            print(json.dumps({"dataset": args.dataset, "images_completed": index,
                              "images_total": len(records), "target_rows": len(targets)}), flush=True)
    validate_access_counts(access, len(records))
    write_jsonl_new(output / "per_image.jsonl", images)
    write_jsonl_new(output / "per_target.jsonl", targets)
    write_jsonl_new(output / "random_fields.jsonl", fields)
    # Keep historical rows verbatim with a provenance label, never regenerate L0.
    write_jsonl_new(output / "legacy_L0_per_image.jsonl", old_images.values())
    write_jsonl_new(output / "legacy_L0_per_target.jsonl", old_targets.values())
    cells = [summarize_cell([r for r in images if r["view"] == v and r["probe_id"] == p],
                [r for r in targets if r["view"] == v and r["probe_id"] == p],
                visibility=legacy_config["visibility"], config=config) for v in VIEWS for p in VARIANTS]
    summary = {"protocol_id": config["protocol_id"], "dataset": args.dataset,
        "scientific_scope": "augmentation_visibility_screen_only", "pilot_images": len(records),
        "per_image_rows": len(images), "per_target_rows": len(targets), "random_field_rows": len(fields),
        "cells": cells, "access": access, "device": "cpu", "threads": torch.get_num_threads(),
        "all_input_pairs_exact": all(r["sealed_input_pair_exact"] for r in images),
        "all_gt_unchanged": all(r["gt_tensor_unchanged"] for r in images),
        "all_probe_replays_exact": all(r["deterministic_repeat_exact"] for r in images),
        "legacy_L0_source": "sealed_v7_read_only_not_regenerated",
        "legacy_L0_compound_status": "derived_from_sealed_sign_flip_and_e1_same_original_denominator",
        "training_started": False, "test_payload_opens": 0, "paper_result": False,
        "ipma_meta_training_allowed": False, "full_source_training_allowed": False,
        "formal_test_allowed": False, "no_validation_split": True}
    complete_dataset(prepared, summary)
    return {"complete": True, "output_dir": str(output), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True, choices=("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    print(json.dumps(run(args), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
