#!/usr/bin/env python3
"""Read-only train8 mechanism diagnosis; GT oracle is not a deployed model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs/cr_sitta_o3_residual_reachability_v3.yaml"
CONFIG_SHA256 = "d48b134c14042157a5d24ed535022fcfae66168e3bb2248ee6d2ba698ab0429e"
OUTPUT = ROOT / "results/cr_sitta/o3_residual_reachability_v3/R0"
METHODS = ("source", "o3", "v1", "oracle_gt_diagnostic")
STATES = ("TP→TP", "TP→FN", "FN→TP", "FN→FN")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def checked_array(path, shape):
    result = np.load(path, mmap_mode="r", allow_pickle=False)
    if result.dtype != np.float32 or result.shape != tuple(shape) or not np.isfinite(result).all():
        raise ValueError(f"invalid parent finite float32 array: {path}")
    return result


def validate_head(head):
    if (not isinstance(head, torch.nn.Conv2d) or head.kernel_size != (1, 1)
            or head.stride != (1, 1) or head.padding != (0, 0)
            or head.dilation != (1, 1) or head.groups != 1
            or head.in_channels != 16 or head.out_channels != 1):
        raise ValueError("the proof requires the original affine pointwise output head")


def check_replay(logits, probabilities, expected, label):
    if (logits.shape != expected.shape or logits.dtype != np.float32
            or probabilities.dtype != np.float32 or not np.isfinite(logits).all()
            or not np.array_equal(probabilities, expected)):
        raise RuntimeError(f"{label}: saved probabilities are not bit-exact")
    if not np.array_equal(logits > 0, expected > .5):
        raise RuntimeError(f"{label}: ideal logit and original probability masks disagree")


def aggregate(records, cells):
    """Counts sum over image-condition observations; cell metrics are equal means."""
    if not records or not cells:
        raise ValueError("nonempty image records and condition cells required")
    integer_keys = [key for key, value in records[0]["counts"].items() if type(value) is int]
    counts = {key: sum(row["counts"][key] for row in records) for key in integer_keys}
    counts["zero_logit_counts"] = {
        key: sum(row["counts"]["zero_logit_counts"][key] for row in records)
        for key in records[0]["counts"]["zero_logit_counts"]}
    counts["oracle_gain_counts"] = {
        key: sum(row["counts"]["oracle_gain_counts"][key] for row in records)
        for key in ("0", "1", "2")}
    transitions = {key: sum(row["objects"]["transition_counts"][key] for row in records) for key in STATES}
    return {
        "image_condition_observations": len(records), "condition_count": len(cells),
        "pixel_counts": counts, "target_observations": sum(transitions.values()),
        "target_transition_counts": transitions,
        "v1_fn_unrepairable_fraction": counts["unrepairable_fn"] / counts["v1_fn"] if counts["v1_fn"] else None,
        "v1_fp_unrepairable_fraction": counts["unrepairable_fp"] / counts["v1_fp"] if counts["v1_fp"] else None,
        "gt_components_without_reachable_gt_pixel": sum(
            target["reachable_gt_positive_pixels"] == 0 for row in records for target in row["objects"]["targets"]),
        "equal_condition_macro": {
            method: {metric: sum(row[method][metric] for row in cells) / len(cells)
                     for metric in ("iou", "normalized_iou", "pd", "fa_per_million")}
            for method in METHODS},
    }


def prepare():
    from analysis import spatial_residual_contract_v1 as paths
    from analysis.o3_multiscale_guard_contract_v2 import verify_parent_run
    if CONFIG.is_symlink():
        raise ValueError("frozen reachability configuration hash differs")
    payload = CONFIG.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CONFIG_SHA256:
        raise ValueError("frozen reachability configuration hash differs")
    raw = yaml.safe_load(payload)
    if ROOT / raw["result_root"] != OUTPUT:
        raise ValueError("unexpected output root")
    parent = verify_parent_run(ROOT / raw["parent_run"])
    bindings = [paths._binding(CONFIG, CONFIG_SHA256)]
    bindings.extend(paths._binding(paths._path(name)) for name in
                    (raw["preregistration"], *raw["implementation_files"]))
    if parent["parent_ledger"]["step_0128.pth.tar"]["sha256"] != raw["model_binding"]["checkpoint_sha256"]:
        raise ValueError("wrong trained v1 checkpoint")
    return raw, parent, bindings


def run():
    from analysis import spatial_residual_contract_v1 as paths
    from analysis.o3_multiscale_guard_contract_v2 import verify_parent_run
    from analysis.o3_residual_reachability_v3 import analyze_reachability
    from analysis.o3_reachability_objects_v3 import object_transitions
    from metrics.connected_components import label_connected_components
    from model.o3_multiscale_residual_v1 import O3MultiScaleResidual
    from scripts import run_p3_stage_b_screen_v1 as b3
    from scripts import run_p3_stage_b4_full_pilot64_v1 as b4
    from scripts.run_o3_spatial_residual_v1 import save_mask

    raw, parent, bindings = prepare()
    paths._reject_symlinks(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    manager = None
    try:
        ids = list(parent["parent_manifest"]["image_ids"])
        write_json(OUTPUT / "manifest.json", {
            "protocol_id": raw["protocol_id"], "configuration": raw, "bindings": bindings,
            "image_ids": ids, "parent_verification": parent["receipt"],
            "oracle_uses_gt": True, "oracle_deployable": False,
            "new_model": False, "formal_test": False, "no_validation_split": True})
        parent_root = parent["parent_root"]
        contract = b4.load_contract(ROOT / parent["parent_manifest"]["configuration"]["parent_config"])
        _, host, _adapter, _film, manager, device, wrapper = b3._build_runtime(contract, "NUDT-SIRST", "cuda:0")
        validate_head(host.output_0)
        write_json(OUTPUT / "runtime.json", {
            **b3._runtime_environment_receipt(torch, device), "checkpoint_wrapper": wrapper,
            "torch_num_threads": torch.get_num_threads(), "optimizer_steps": 0})
        features = checked_array(parent_root / "o3_features.npy", (104, 16, 256, 256))
        targets = checked_array(parent_root / "train_targets.npy", (8, 1, 256, 256))
        if not np.logical_or(targets == 0, targets == 1).all():
            raise ValueError("training GT is not binary")
        probabilities = {name: checked_array(parent_root / filename, (104, 1, 256, 256)) for name, filename in (
            ("source", "source_probabilities.npy"), ("o3", "o3_probabilities.npy"), ("v1", "trained_probabilities.npy"))}
        if any(((value < 0) | (value > 1)).any() for value in probabilities.values()):
            raise ValueError("invalid saved probability range")
        checkpoint = torch.load(parent_root / "step_0128.pth.tar", map_location="cpu")
        if checkpoint["step"] != 128 or checkpoint["host_checkpoint"] != contract.raw["datasets"]["NUDT-SIRST"]:
            raise ValueError("trained v1 metadata or host differs")
        branch = O3MultiScaleResidual().to(device).eval()
        branch.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
        branch.requires_grad_(False)
        branch_before = {key: tensor.clone() for key, tensor in branch.state_dict().items()}
        z0 = np.empty((104, 1, 256, 256), dtype=np.float32)
        z1 = np.empty_like(z0)
        with torch.no_grad():
            for index in range(104):
                h = torch.from_numpy(np.array(features[index:index + 1], copy=True)).to(device)
                for name, result, destination in (("o3", h, z0), ("v1", branch(h), z1)):
                    logits = host.output_0(result)
                    array = logits[0].cpu().numpy()
                    prediction = torch.sigmoid(logits)[0].cpu().numpy()
                    check_replay(array, prediction, probabilities[name][index], f"{name}/{index}")
                    destination[index] = array
        manager.assert_source_state()
        write_json(OUTPUT / "replay.json", {
            "samples_checked": 104, "o3_probabilities_bit_exact": True, "v1_probabilities_bit_exact": True,
            "original_probability_mask_equals_ideal_logit_mask": True,
            "finite_logits_direct_from_head": True, "no_inverse_sigmoid": True})
        np.save(OUTPUT / "o3_logits.npy", z0, allow_pickle=False)
        np.save(OUTPUT / "v1_logits.npy", z1, allow_pickle=False)
        oracle = np.empty_like(z0)
        gains = np.empty(z0.shape, dtype=np.float64)
        reachability = {key: np.empty(z0.shape, dtype=bool) for key in (
            "can_positive", "can_negative", "repairable_fn", "repairable_fp", "unrepairable_fn", "unrepairable_fp")}
        records = []
        with (OUTPUT / "images.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(104):
                ci, ii = divmod(index, 8)
                family, severity = b4.CONDITIONS[ci]
                result = analyze_reachability(z0[index, 0], z1[index, 0], targets[ii, 0])
                arrays = result["arrays"]
                oracle[index, 0] = arrays["oracle_mask"]
                gains[index, 0] = arrays["oracle_gain"]
                for key in reachability:
                    reachability[key][index, 0] = arrays[key]
                objects = object_transitions(targets[ii, 0], probabilities["v1"][index, 0] > .5, arrays["oracle_mask"])
                labeled = label_connected_components(targets[ii, 0], connectivity=2, min_area=1)
                for target in objects["targets"]:
                    region = labeled.labels == target["id"]
                    if int(region.sum()) != target["area"]:
                        raise RuntimeError("target identity does not match fixed GT component labels")
                    target.update({
                        "v1_gt_positive_pixels": int(((z1[index, 0] > 0) & region).sum()),
                        "reachable_gt_positive_pixels": int((arrays["can_positive"] & region).sum()),
                        "unrepairable_fn_pixels": int((arrays["unrepairable_fn"] & region).sum())})
                record = {"index": index, "condition": b4._condition_key(family, severity),
                          "corruption": family, "severity": severity, "image_id": ids[ii],
                          "counts": result["counts"], "objects": objects}
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        np.save(OUTPUT / "oracle_binary_masks.npy", oracle, allow_pickle=False)
        np.save(OUTPUT / "oracle_gain.npy", gains, allow_pickle=False)
        np.savez_compressed(OUTPUT / "pixel_reachability.npz", **reachability)
        probabilities["oracle_gt_diagnostic"] = oracle
        cells = []
        with (OUTPUT / "cells.jsonl").open("x", encoding="utf-8") as stream:
            for ci, (family, severity) in enumerate(b4.CONDITIONS):
                condition = b4._condition_key(family, severity)
                saved = parent["parent_cells"][ci]
                if saved["condition"] != condition:
                    raise RuntimeError("parent cell order differs")
                sub = slice(ci * 8, (ci + 1) * 8)
                row = {"dataset": "NUDT-SIRST", "condition": condition, "corruption": family, "severity": severity}
                for name in METHODS:
                    destination = OUTPUT / "conditions" / condition / name
                    destination.mkdir(parents=True, exist_ok=False)
                    for image_id, probability in zip(ids, probabilities[name][sub], strict=True):
                        save_mask(destination / f"{image_id}.png", probability)
                    if name == "oracle_gt_diagnostic":
                        evaluated = b3._evaluation_result(oracle[sub], targets, ids)
                        row[name] = b3._endpoint_summary(evaluated)
                        if (row[name]["detected_targets"] != sum(r["objects"]["diagnostic"]["detected_targets"] for r in records[sub])
                                or row[name]["false_alarm_pixels"] != sum(r["objects"]["diagnostic"]["false_alarm_pixels"] for r in records[sub])):
                            raise RuntimeError("object diagnostics disagree with original evaluator")
                    else:
                        row[name] = copy.deepcopy(saved["trained" if name == "v1" else name])
                    write_json(destination / "fixed_metrics.json", {
                        "fixed_threshold_metrics": row[name], "oracle_uses_gt": name == "oracle_gt_diagnostic",
                        "continuous_oracle_froc_not_reported": True, "paper_result": False})
                if (row["v1"]["detected_targets"] != sum(r["objects"]["previous"]["detected_targets"] for r in records[sub])
                        or row["v1"]["false_alarm_pixels"] != sum(r["objects"]["previous"]["false_alarm_pixels"] for r in records[sub])):
                    raise RuntimeError("v1 object replay disagrees with original evaluator")
                if row["oracle_gt_diagnostic"]["iou"] < row["v1"]["iou"]:
                    raise RuntimeError("pixel oracle should not reduce fixed-threshold IoU")
                for name, prefix in (("v1", "v1"), ("oracle_gt_diagnostic", "oracle")):
                    for metric, count in (("intersection_pixels", "tp"), ("false_positive_pixels", "fp"), ("false_negative_pixels", "fn")):
                        if row[name][metric] != sum(r["counts"][f"{prefix}_{count}"] for r in records[sub]):
                            raise RuntimeError("oracle pixel counts disagree with original evaluator")
                cells.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                print(json.dumps({"phase": "reachability_diagnostic", "condition": condition,
                                  "v1_iou": row["v1"]["iou"], "gt_oracle_iou_bound": row["oracle_gt_diagnostic"]["iou"]}), flush=True)
        manager.assert_source_state()
        if any(not torch.equal(value, branch.state_dict()[key]) for key, value in branch_before.items()):
            raise RuntimeError("read-only v1 weights changed")
        verify_parent_run(parent_root)
        for binding in bindings:
            paths._binding(paths._path(binding["path"]), binding["sha256"])
        summary = {
            "scope": raw["scope"], "image_ids": ids, "samples": 104, "optimizer_steps": 0,
            "saved_masks_all_methods": 416, "oracle_uses_gt": True, "oracle_deployable": False,
            "bound_claim": "ideal_closed_interval_independent_pixel_binary_IoU_only",
            "pd_fa_bound_claim": False, "learnability_claim": False,
            "overall": aggregate(records, cells), "nonclean": aggregate(records[8:], cells[1:]),
            "clean": aggregate(records[:8], cells[:1]),
            "families": {family: aggregate([r for r in records if r["corruption"] == family],
                                          [r for r in cells if r["corruption"] == family])
                         for family in ("gaussian_noise", "gaussian_blur", "low_contrast", "stripe_noise")},
            "head_host_and_v1_unchanged": True, "parent_reverified": True,
            "new_raw_image_or_gt_decodes": 0, "test_payload_opens": 0, "new_o3_episodes": 0,
            "elapsed_seconds": time.perf_counter() - started}
        write_json(OUTPUT / "summary.json", summary)
        print(json.dumps({"phase": "diagnostic_done", "nonclean": summary["nonclean"], "optimizer_steps": 0}, ensure_ascii=False), flush=True)
        write_json(OUTPUT / "artifact_ledger.json", paths._file_ledger(OUTPUT))
        write_json(OUTPUT / "COMPLETE.json", {
            "complete": True, "samples": 104, "optimizer_steps": 0, "paper_result": False,
            "oracle_deployable": False, "automatic_training_allowed": False,
            "manifest_sha256": paths.sha256_file(OUTPUT / "manifest.json"),
            "ledger_sha256": paths.sha256_file(OUTPUT / "artifact_ledger.json")})
    except BaseException as exc:
        restored = False
        if manager is not None:
            try:
                manager.reset_to_source()
                manager.assert_source_state()
                restored = True
            except Exception:
                pass
        write_json(OUTPUT / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc),
                                          "optimizer_steps": 0, "host_restored": restored})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        raw, parent, bindings = prepare()
        print(json.dumps({"protocol_id": raw["protocol_id"], "parent_receipt": parent["receipt"],
                          "new_bound_files": len(bindings), "payloads_deserialized": 0}, ensure_ascii=False))
    else:
        run()


if __name__ == "__main__":
    main()
