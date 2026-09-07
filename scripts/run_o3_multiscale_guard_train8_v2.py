#!/usr/bin/env python3
"""One fixed source-training loss change on sealed O3 train8 features.

No new O3 episodes, image corruption, test access or inference-time GT rules.
All predecessor artifacts stay immutable. This is same-image fit evidence.
"""
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

CONFIG = ROOT / "configs/cr_sitta_o3_multiscale_guard_train8_v2.yaml"
CONFIG_SHA256 = "15fec709cad643ddef17ffb85066d39eeddb47d19f2c432a7243a7b0e98260b7"
OUTPUT = ROOT / "results/cr_sitta/o3_multiscale_guard_train8_v2/R0"
LOSS_KEYS = ("segmentation_full_fit_loss", "background_guard_full_fit_loss", "augmented_full_fit_loss")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def checked_array(path, shape):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or tuple(array.shape) != tuple(shape) or not np.isfinite(array).all():
        raise ValueError(f"invalid finite float32 array: {path}")
    return array


def require_exact_probability(actual, expected, label):
    if (actual.shape != expected.shape or actual.dtype != expected.dtype
            or not np.isfinite(actual).all() or not np.array_equal(actual, expected)):
        raise RuntimeError(f"{label}: replay is not bit-exact")


def checked_order(record):
    from scripts.run_o3_multiscale_train8_v1 import training_order
    if (not isinstance(record, dict) or record.get("sample_order") != "condition_major_then_image_id"
            or record.get("indices") != training_order(104)):
        raise ValueError("parent 128-step schedule differs")
    return record["indices"]


def loss_values(terms):
    if set(terms) != set(LOSS_KEYS):
        raise ValueError("three separate loss terms are required")
    values = {key: float(terms[key].detach()) for key in LOSS_KEYS}
    if not all(np.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("nonfinite or negative loss term")
    return values


def prepare():
    from analysis import spatial_residual_contract_v1 as paths
    from analysis.o3_multiscale_guard_contract_v2 import verify_parent_run
    if CONFIG.is_symlink() or hashlib.sha256(CONFIG.read_bytes()).hexdigest() != CONFIG_SHA256:
        raise ValueError("frozen v2 configuration byte hash differs")
    raw = yaml.safe_load(CONFIG.read_bytes())
    if ROOT / raw["result_root"] != OUTPUT:
        raise ValueError("unexpected result destination")
    parent = verify_parent_run(ROOT / raw["parent_run"])
    paths_to_bind = [str(CONFIG.relative_to(ROOT)), raw["preregistration"], *raw["implementation_files"]]
    bindings = [paths._binding(paths._path(path)) for path in paths_to_bind]
    return raw, parent, bindings


def run():
    from analysis import spatial_residual_contract_v1 as paths
    from analysis.o3_multiscale_guard_contract_v2 import verify_parent_run
    from analysis.o3_multiscale_guard_metrics_v2 import summarize
    from model.o3_multiscale_residual_v1 import O3MultiScaleResidual
    from scripts import run_p3_stage_b_screen_v1 as b3
    from scripts import run_p3_stage_b4_full_pilot64_v1 as b4
    from scripts.run_o3_spatial_residual_v1 import save_mask
    from training.o3_background_nonexpansion_v2 import guarded_supervised_loss

    raw, parent, bindings = prepare()
    paths._reject_symlinks(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=False)
    manifest = {"protocol_id": raw["protocol_id"], "configuration": raw, "bindings": bindings,
                "parent_verification": parent["receipt"], "image_ids": parent["parent_manifest"]["image_ids"],
                "source_training_gt_allowed": True, "no_new_o3_episodes": True,
                "formal_test": False, "no_validation_split": True, "paper_result": False}
    write_json(OUTPUT / "manifest.json", manifest)
    started = time.perf_counter()
    state_manager = None
    completed_steps = 0
    try:
        parent_root = parent["parent_root"]
        parent_config = parent["parent_manifest"]["configuration"]
        contract = b4.load_contract(ROOT / parent_config["parent_config"])
        _, host, _adapter, _film, state_manager, device, wrapper = b3._build_runtime(contract, "NUDT-SIRST", "cuda:0")
        write_json(OUTPUT / "runtime.json", {**b3._runtime_environment_receipt(torch, device),
                                           "checkpoint_wrapper": wrapper, "torch_num_threads": torch.get_num_threads()})
        features = checked_array(parent_root / "o3_features.npy", (104, 16, 256, 256))
        targets = checked_array(parent_root / "train_targets.npy", (8, 1, 256, 256))
        if not np.logical_or(targets == 0, targets == 1).all():
            raise ValueError("source training targets must be binary")
        references = {name: checked_array(parent_root / filename, (104, 1, 256, 256)) for name, filename in
                      (("source", "source_probabilities.npy"), ("o3", "o3_probabilities.npy"),
                       ("previous", "trained_probabilities.npy"))}
        if any(((value < 0) | (value > 1)).any() for value in references.values()):
            raise ValueError("invalid saved probability range")
        ids = list(parent["parent_manifest"]["image_ids"])
        target_all = np.tile(targets, (13, 1, 1, 1))
        if parent["parent_ledger"]["initial.pth.tar"]["sha256"] != raw["model"]["initialization_sha256"]:
            raise ValueError("initialization artifact does not match the declared fixed source")
        initial = torch.load(parent_root / "initial.pth.tar", map_location="cpu")
        prior = torch.load(parent_root / "step_0128.pth.tar", map_location="cpu")
        if initial["step"] != 0 or prior["step"] != 128 or initial["host_checkpoint"] != prior["host_checkpoint"]:
            raise ValueError("parent initialization/checkpoint metadata differs")
        if initial["host_checkpoint"] != contract.raw["datasets"]["NUDT-SIRST"]:
            raise ValueError("frozen output head checkpoint differs")
        branch, previous_branch = O3MultiScaleResidual().to(device), O3MultiScaleResidual().to(device).eval()
        branch.load_state_dict(initial["adapter_state_dict"], strict=True)
        previous_branch.load_state_dict(prior["adapter_state_dict"], strict=True)
        previous_branch.requires_grad_(False)
        if sum(p.numel() for p in branch.parameters()) != 1568:
            raise RuntimeError("model architecture changed")

        def batch(indices):
            return (torch.from_numpy(np.array(features[indices], copy=True)).to(device),
                    torch.from_numpy(np.array(target_all[indices], copy=True)).to(device))

        def forward_terms(h, target):
            with torch.no_grad():
                anchor = host.output_0(h).detach()
            return guarded_supervised_loss(host.output_0(branch(h)), anchor, target)

        def full_fit_losses():
            sums = {key: 0.0 for key in LOSS_KEYS}
            with torch.no_grad():
                for offset in range(0, 104, 4):
                    h, target = batch(list(range(offset, offset + 4)))
                    values = loss_values(forward_terms(h, target))
                    for key in LOSS_KEYS:
                        sums[key] += values[key] * 4
            return {key: value / 104 for key, value in sums.items()}

        # Read-only replay checks all 104 saved endpoints before any update.
        with torch.no_grad():
            for index in range(104):
                h, _ = batch([index])
                if not torch.equal(branch(h), h):
                    raise RuntimeError("initial module identity failed")
                for label, corrected in (("o3", branch(h)), ("previous", previous_branch(h))):
                    probability = torch.sigmoid(host.output_0(corrected))[0].cpu().numpy()
                    require_exact_probability(probability, references[label][index], label)
        initial_losses = full_fit_losses()
        if (initial_losses["background_guard_full_fit_loss"] != 0
                or initial_losses["segmentation_full_fit_loss"] != parent["parent_summary"]["initial_full_fit_loss"]):
            raise RuntimeError("initial full-fit loss does not exactly reproduce v1")
        replay = {"initial_identity": True, "o3_probabilities_bit_exact": True,
                  "v1_trained_probabilities_bit_exact": True, "samples_checked": 104,
                  "initial_losses": initial_losses}
        write_json(OUTPUT / "initial_replay.json", replay)
        del previous_branch, prior
        order_record = json.loads((parent_root / "training_order.json").read_text(encoding="utf-8"))
        order = checked_order(order_record)
        write_json(OUTPUT / "training_order.json", order_record)
        with (parent_root / "training.jsonl").open(encoding="utf-8") as stream:
            previous_first_step = json.loads(next(stream))
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        if optimizer.state:
            raise RuntimeError("new optimizer must start with empty state")
        with (OUTPUT / "training.jsonl").open("x", encoding="utf-8") as stream:
            for step, indices in enumerate(order, 1):
                h, target = batch(indices)
                optimizer.zero_grad(set_to_none=True)
                terms = forward_terms(h, target)
                values = loss_values(terms)
                terms["augmented_full_fit_loss"].backward()
                norms = {name: float(p.grad.norm()) for name, p in branch.named_parameters() if p.grad is not None}
                if len(norms) != 4 or not all(np.isfinite(value) for value in norms.values()):
                    raise RuntimeError("missing or nonfinite source training gradients")
                if step == 1:
                    if (values["background_guard_full_fit_loss"] != 0 or norms != previous_first_step["gradient_norms"]
                            or indices != previous_first_step["sample_indices"]
                            or values["segmentation_full_fit_loss"] != previous_first_step["loss"]):
                        raise RuntimeError("first update differs despite zero background increment")
                    write_json(OUTPUT / "first_step_replay.json", {"base_loss_exact": True,
                               "gradient_norms_exact": True, "indices_exact": True, "background_loss_zero": True})
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in branch.parameters()):
                    raise RuntimeError("nonfinite trained parameters")
                completed_steps = step
                minibatch_losses = {key.removesuffix("_full_fit_loss"): value for key, value in values.items()}
                row = {"step": step, "sample_indices": indices, "loss_scope": "current_minibatch",
                       "minibatch_losses": minibatch_losses, "gradient_norms": norms}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if step % 16 == 0:
                    print(json.dumps({"phase": "background_guard_fit", "step": step, "steps": 128,
                                      "loss_scope": "current_minibatch", **minibatch_losses}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        branch.eval()
        final_losses = full_fit_losses()
        state_manager.assert_source_state()
        torch.save({"model_label": raw["model"]["label"], "adapter_state_dict": branch.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "step": 128,
                    "host_checkpoint": initial["host_checkpoint"], "config_sha256": CONFIG_SHA256,
                    "parent_manifest_sha256": raw["parent_manifest_sha256"],
                    "initialization_sha256": raw["model"]["initialization_sha256"],
                    "selection_rule": "fixed_last_step", "trained_on": "source_train8",
                    "requires_original_o3_p2_endpoint": True, "inference_uses_gt": False}, OUTPUT / "step_0128.pth.tar")
        guarded = np.empty((104, 1, 256, 256), dtype=np.float32)
        with torch.no_grad():
            for index in range(104):
                h, _ = batch([index])
                corrected = branch(h)
                ratio = (corrected - h).square().mean().sqrt() / h.square().mean().clamp_min(1e-12).sqrt()
                if not bool(torch.isfinite(corrected).all()) or float(ratio) > .050001:
                    raise RuntimeError("nonfinite or out-of-bound residual")
                guarded[index] = torch.sigmoid(host.output_0(corrected))[0].cpu().numpy()
        np.save(OUTPUT / "guarded_probabilities.npy", guarded, allow_pickle=False)
        references["guarded"] = guarded
        cells = []
        old_by_condition = {row["condition"]: row for row in parent["parent_cells"]}
        with (OUTPUT / "cells.jsonl").open("x", encoding="utf-8") as stream:
            for ci, (corruption, severity) in enumerate(b4.CONDITIONS):
                condition = b4._condition_key(corruption, severity)
                saved = old_by_condition[condition]
                row = {key: saved[key] for key in ("dataset", "condition", "corruption", "severity")}
                sub = slice(ci * 8, (ci + 1) * 8)
                for name in ("source", "o3", "previous", "guarded"):
                    destination = OUTPUT / "conditions" / condition / name
                    destination.mkdir(parents=True, exist_ok=False)
                    for image_id, probability in zip(ids, references[name][sub], strict=True):
                        save_mask(destination / f"{image_id}.png", probability)
                    if name == "guarded":
                        result = b3._evaluation_result(guarded[sub], targets, ids)
                        row[name] = b3._endpoint_summary(result)
                        full_metrics = result.to_dict()
                    else:
                        old_name = "trained" if name == "previous" else name
                        row[name] = copy.deepcopy(saved[old_name])
                        full_metrics = json.loads((parent_root / "conditions" / condition / old_name / "metrics.json").read_text())
                    write_json(destination / "metrics.json", full_metrics)
                cells.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
        result_summary = summarize(cells, initial_losses, final_losses)
        result_summary.update({"scope": raw["scope"], "image_ids": ids, "samples": 104,
                               "optimizer_steps": 128, "new_prediction_masks": 104, "saved_masks_all_methods": 416,
                               "parent_feature_reuse": True, "same_initialization": True, "same_schedule": True,
                               "head_and_host_unchanged": True, "test_payload_opens": 0,
                               "new_o3_method_label_accesses": 0, "new_raw_image_or_mask_decodes": 0,
                               "elapsed_seconds": time.perf_counter() - started})
        state_manager.assert_source_state()
        verify_parent_run(parent_root)
        for binding in bindings:
            paths._binding(paths._path(binding["path"]), binding["sha256"])
        write_json(OUTPUT / "summary.json", result_summary)
        print(json.dumps(result_summary, sort_keys=True, ensure_ascii=False), flush=True)
        write_json(OUTPUT / "artifact_ledger.json", paths._file_ledger(OUTPUT))
        write_json(OUTPUT / "COMPLETE.json", {"complete": True, "steps": 128, "samples": 104,
                                             "manifest_sha256": paths.sha256_file(OUTPUT / "manifest.json"),
                                             "ledger_sha256": paths.sha256_file(OUTPUT / "artifact_ledger.json"),
                                             "paper_result": False, "automatic_full_training_allowed": False})
        return result_summary
    except BaseException as exc:
        restored = False
        if state_manager is not None:
            try:
                state_manager.reset_to_source()
                state_manager.assert_source_state()
                restored = True
            except Exception:
                pass
        write_json(OUTPUT / "FAILED.json", {"error_type": type(exc).__name__, "error": str(exc),
                                          "completed_optimizer_steps": completed_steps, "host_restored": restored})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        raw, parent, bindings = prepare()
        print(json.dumps({"protocol_id": raw["protocol_id"], "parent_receipt": parent["receipt"],
                          "new_bound_files": len(bindings), "payloads_deserialized": 0}, ensure_ascii=False, sort_keys=True))
    else:
        run()


if __name__ == "__main__":
    main()
