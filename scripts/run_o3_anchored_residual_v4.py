#!/usr/bin/env python3
"""Matched continuation: bounded v1 versus an anchored unsaturated correction."""
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

from scripts.run_o3_residual_reachability_v3 import checked_array, check_replay, validate_head, write_json

CONFIG = ROOT / "configs/cr_sitta_o3_anchored_residual_v4.yaml"
CONFIG_SHA256 = "0d47900c684a3e32923307de8551a4d2fea9e7179fb7200663e0f27f97592f88"
OUTPUT = ROOT / "results/cr_sitta/o3_anchored_residual_v4/R0"
ARMS = ("control", "candidate")
METHODS = ("source", "o3", "v1", *ARMS)


def box_capacity(features, z0, z1, targets, head_weights):
    """Ideal independent-channel closed box; GT-assisted diagnosis, not a model."""
    h = np.asarray(features)
    before, previous, truth, weights = (np.asarray(value) for value in (z0, z1, targets, head_weights))
    if (h.ndim != 3 or h.shape[0] != 16 or before.shape != h.shape[1:]
            or previous.shape != before.shape or truth.shape != before.shape or weights.size != 16):
        raise ValueError("box diagnostic input shape differs")
    if not all(np.isfinite(value).all() for value in (h, before, previous, truth, weights)):
        raise ValueError("box diagnostic requires finite values")
    if not np.logical_or(truth == 0, truth == 1).all():
        raise ValueError("box diagnostic requires binary source GT")
    with np.errstate(over="raise", invalid="raise"):
        scale = max(float(np.mean(np.asarray(h, dtype=np.float64) ** 2)), 1e-12) ** .5
        head_l1 = float(np.abs(np.asarray(weights, dtype=np.float64)).sum())
        margin = .05 * scale * head_l1
    if not np.isfinite(margin):
        raise ValueError("nonfinite box capacity")
    upper, lower = before.astype(np.float64) + margin, before.astype(np.float64) - margin
    fn, fp = (truth == 1) & (previous <= 0), (truth == 0) & (previous > 0)
    return {"rms_float64": scale, "head_l1": head_l1, "maximum_logit_increment": margin,
            "v1_fn": int(fn.sum()), "fn_upper_le_zero": int((fn & (upper <= 0)).sum()),
            "v1_fp": int(fp.sum()), "fp_lower_gt_zero": int((fp & (lower > 0)).sum()),
            "maximum_upper_logit_on_gt": float(upper[truth == 1].max()) if (truth == 1).any() else None}


def checked_schedule(record):
    from scripts.run_o3_multiscale_train8_v1 import training_order
    if (not isinstance(record, dict) or record.get("sample_order") != "condition_major_then_image_id"
            or record.get("indices") != training_order(104)):
        raise ValueError("both arms require the exact frozen parent schedule")
    return record["indices"]


def trained_transitions(target, previous_mask, prediction, arm):
    from analysis.o3_reachability_objects_v3 import object_transitions
    if arm not in ARMS:
        raise ValueError("only the two trained arms are allowed")
    result = object_transitions(target, previous_mask, prediction)
    # Reuse geometry/matching, not the old oracle-specific interpretation.
    result["comparison"] = f"v1→{arm}"
    result["interpretation"] = {
        "prediction_source": "fixed_step_source_supervised_trained_model",
        "oracle_mask": False, "evaluation_uses_gt": True, "inference_uses_gt": False,
        "fit_and_measure_on_same_images": True, "paper_result": False,
        "caveat": "Matched target states on source train8, not unseen-image generalization or an oracle bound."}
    return result


def prepare():
    from analysis import spatial_residual_contract_v1 as paths
    from analysis.o3_multiscale_guard_contract_v2 import verify_parent_run
    if CONFIG.is_symlink():
        raise ValueError("configuration hash differs")
    payload = CONFIG.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CONFIG_SHA256:
        raise ValueError("configuration hash differs")
    raw = yaml.safe_load(payload)
    if ROOT / raw["result_root"] != OUTPUT:
        raise ValueError("unexpected output root")
    parent = verify_parent_run(ROOT / raw["parent_run"])
    if parent["parent_ledger"]["step_0128.pth.tar"]["sha256"] != raw["initialization"]["checkpoint_sha256"]:
        raise ValueError("trained v1 checkpoint differs")
    bindings = [paths._binding(CONFIG, CONFIG_SHA256)]
    bindings.extend(paths._binding(paths._path(name)) for name in (raw["preregistration"], *raw["implementation_files"]))
    return raw, parent, bindings


def run():
    from analysis import spatial_residual_contract_v1 as paths
    from analysis.o3_multiscale_guard_contract_v2 import verify_parent_run
    from analysis.o3_anchored_metrics_v4 import summarize
    from model.o3_anchored_residual_v4 import O3AnchoredResidualV4
    from model.o3_multiscale_residual_v1 import O3MultiScaleResidual
    from scripts import run_p3_stage_b_screen_v1 as b3
    from scripts import run_p3_stage_b4_full_pilot64_v1 as b4
    from scripts.run_o3_multiscale_train8_v1 import supervised_loss
    from scripts.run_o3_spatial_residual_v1 import save_mask

    raw, parent, bindings = prepare()
    paths._reject_symlinks(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    manager = None
    steps_done = {arm: 0 for arm in ARMS}
    try:
        ids = list(parent["parent_manifest"]["image_ids"])
        write_json(OUTPUT / "manifest.json", {"protocol_id": raw["protocol_id"], "configuration": raw,
            "bindings": bindings, "parent_verification": parent["receipt"], "image_ids": ids,
            "initialization": "trained_v1_step128_both_arms", "source_training_gt_allowed": True,
            "no_validation_split": True, "formal_test": False, "paper_result": False})
        parent_root = parent["parent_root"]
        contract = b4.load_contract(ROOT / parent["parent_manifest"]["configuration"]["parent_config"])
        _, host, _adapter, _film, manager, device, wrapper = b3._build_runtime(contract, "NUDT-SIRST", "cuda:0")
        validate_head(host.output_0)
        write_json(OUTPUT / "runtime.json", {**b3._runtime_environment_receipt(torch, device),
            "checkpoint_wrapper": wrapper, "torch_num_threads": torch.get_num_threads()})
        features = checked_array(parent_root / "o3_features.npy", (104, 16, 256, 256))
        targets = checked_array(parent_root / "train_targets.npy", (8, 1, 256, 256))
        if not np.logical_or(targets == 0, targets == 1).all():
            raise ValueError("source GT is not binary")
        all_targets = np.tile(targets, (13, 1, 1, 1))
        probabilities = {name: checked_array(parent_root / filename, (104, 1, 256, 256)) for name, filename in (
            ("source", "source_probabilities.npy"), ("o3", "o3_probabilities.npy"), ("v1", "trained_probabilities.npy"))}
        if any(((value < 0) | (value > 1)).any() for value in probabilities.values()):
            raise ValueError("parent probability range differs")
        prior = torch.load(parent_root / "step_0128.pth.tar", map_location="cpu")
        if prior["step"] != 128 or prior["host_checkpoint"] != contract.raw["datasets"]["NUDT-SIRST"]:
            raise ValueError("v1 host metadata differs")
        control = O3MultiScaleResidual().to(device)
        control.load_state_dict(prior["adapter_state_dict"], strict=True)
        candidate = O3AnchoredResidualV4(prior["adapter_state_dict"]).to(device)
        branches = {"control": control, "candidate": candidate}
        parameters = {"control": list(control.parameters()), "candidate": list(candidate.learnable_parameters())}
        if any(sum(p.numel() for p in values) != 1568 for values in parameters.values()):
            raise RuntimeError("both arms must have 1568 trainable parameters")
        anchor_before = {key: value.clone() for key, value in candidate.anchor.state_dict().items()}
        def batch(indices):
            return (torch.from_numpy(np.array(features[indices], copy=True)).to(device),
                    torch.from_numpy(np.array(all_targets[indices], copy=True)).to(device))

        def full_fit_loss(branch):
            total = 0.0
            with torch.no_grad():
                for offset in range(0, 104, 4):
                    h, target = batch(list(range(offset, offset + 4)))
                    total += float(supervised_loss(host.output_0(branch(h)), target)) * 4
            value = total / 104
            if not np.isfinite(value) or value < 0:
                raise RuntimeError("nonfinite complete fitting loss")
            return value

        capacities = []
        with torch.no_grad():
            for index in range(104):
                h, _ = batch([index])
                z0 = host.output_0(h)
                check_replay(z0[0].cpu().numpy(), torch.sigmoid(z0)[0].cpu().numpy(), probabilities["o3"][index], "o3")
                baseline = control(h)
                if not torch.equal(candidate(h), baseline):
                    raise RuntimeError("candidate initial features differ from trained v1")
                for arm in ARMS:
                    z = host.output_0(branches[arm](h))
                    check_replay(z[0].cpu().numpy(), torch.sigmoid(z)[0].cpu().numpy(), probabilities["v1"][index], arm)
                z1 = host.output_0(baseline)
                family, severity = b4.CONDITIONS[index // 8]
                capacities.append({"condition": b4._condition_key(family, severity), "image_id": ids[index % 8],
                    **box_capacity(features[index], z0[0, 0].cpu().numpy(), z1[0, 0].cpu().numpy(),
                                   targets[index % 8, 0], host.output_0.weight.detach().cpu().numpy())})
        initial_losses = {arm: full_fit_loss(branches[arm]) for arm in ARMS}
        if any(value != parent["parent_summary"]["final_full_fit_loss"] for value in initial_losses.values()):
            raise RuntimeError("full initial loss does not exactly reproduce trained v1")
        write_json(OUTPUT / "initial_replay.json", {"samples": 104, "features_bit_exact_both_arms": True,
            "probabilities_bit_exact_o3_v1_and_both_arms": True, "initial_losses": initial_losses})
        write_json(OUTPUT / "amplitude_diagnostic.json", {"role": "gt_assisted_offline_model_class_bound",
            "formula": "M=.05*float64_RMS_floor(h)*sum_abs_head_weight", "not_a_performance_result": True,
            "not_a_pd_fa_bound": True, "closed_box_relaxation": True, "images": capacities})
        schedule_record = json.loads((parent_root / "training_order.json").read_text())
        order = checked_schedule(schedule_record)
        write_json(OUTPUT / "training_order.json", schedule_record)
        final_losses, fit_seconds = {}, {}
        for arm in ARMS:
            branch = branches[arm].train()
            before = {key: value.clone() for key, value in branch.state_dict().items()}
            optimizer = torch.optim.Adam(parameters[arm], lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
            if optimizer.state:
                raise RuntimeError("optimizer must be fresh in each arm")
            arm_started = time.perf_counter()
            with (OUTPUT / f"training_{arm}.jsonl").open("x", encoding="utf-8") as stream:
                for step, indices in enumerate(order, 1):
                    h, target = batch(indices)
                    optimizer.zero_grad(set_to_none=True)
                    loss = supervised_loss(host.output_0(branch(h)), target)
                    loss.backward()
                    norms = {name: float(p.grad.norm()) for name, p in branch.named_parameters() if p.requires_grad and p.grad is not None}
                    if len(norms) != 4 or not all(np.isfinite(value) for value in norms.values()):
                        raise RuntimeError("missing or nonfinite source training gradients")
                    optimizer.step()
                    if not all(bool(torch.isfinite(p).all()) for p in branch.parameters()):
                        raise RuntimeError("nonfinite model state")
                    steps_done[arm] = step
                    stream.write(json.dumps({"step": step, "arm": arm, "sample_indices": indices,
                        "loss_scope": "current_minibatch", "loss": float(loss.detach()), "gradient_norms": norms}, allow_nan=False) + "\n")
                    stream.flush()
                    if step % 32 == 0:
                        print(json.dumps({"phase": "fit", "arm": arm, "step": step, "steps": 128,
                                          "minibatch_loss": float(loss.detach())}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            branch.eval()
            fit_seconds[arm] = time.perf_counter() - arm_started
            final_losses[arm] = full_fit_loss(branch)
            manager.assert_source_state()
            if any(not torch.equal(value, candidate.anchor.state_dict()[key]) for key, value in anchor_before.items()):
                raise RuntimeError("frozen trained-v1 anchor changed")
            torch.save({"model_label": raw["model"]["label"] if arm == "candidate" else "CR-SITTA_O3_v1_ContinuedControl",
                "arm": arm, "model_state_dict": branch.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "step": 128, "inherited_source_fit_steps": 128, "host_checkpoint": prior["host_checkpoint"],
                "config_sha256": CONFIG_SHA256, "initialization_sha256": raw["initialization"]["checkpoint_sha256"],
                "selection_rule": "fixed_last_step", "trained_on": "source_train8", "inference_uses_gt": False,
                "requires_original_o3_p2_endpoint": True,
                "changed_state_keys": [key for key, value in branch.state_dict().items() if not torch.equal(before[key], value)]},
                OUTPUT / f"{arm}_step_0128.pth.tar")
            prediction = np.empty((104, 1, 256, 256), dtype=np.float32)
            residual_records = []
            with torch.no_grad():
                for index in range(104):
                    h, _ = batch([index])
                    updated = branch(h)
                    logits = host.output_0(updated)
                    if not bool(torch.isfinite(updated).all()) or not bool(torch.isfinite(logits).all()):
                        raise RuntimeError("nonfinite trained prediction")
                    prediction[index] = torch.sigmoid(logits)[0].cpu().numpy()
                    ratio = float((updated - h).square().mean().sqrt() / h.square().mean().clamp_min(1e-12).sqrt())
                    if not np.isfinite(ratio) or (arm == "control" and ratio > .050001):
                        raise RuntimeError("invalid residual ratio")
                    residual_records.append({"index": index, "rms_ratio_to_o3": ratio,
                        "logit_min": float(logits.min()), "logit_max": float(logits.max())})
            np.save(OUTPUT / f"{arm}_probabilities.npy", prediction, allow_pickle=False)
            probabilities[arm] = prediction
            write_json(OUTPUT / f"{arm}_residual_statistics.json", {"hard_bound_enforced": arm == "control",
                "records": residual_records, "maximum_rms_ratio": max(r["rms_ratio_to_o3"] for r in residual_records)})
        transitions = []
        with (OUTPUT / "transitions.jsonl").open("x", encoding="utf-8") as stream:
            for index in range(104):
                family, severity = b4.CONDITIONS[index // 8]
                row = {"condition": b4._condition_key(family, severity), "image_id": ids[index % 8]}
                for arm in ARMS:
                    row[arm] = trained_transitions(targets[index % 8, 0], probabilities["v1"][index, 0] > .5,
                                                  probabilities[arm][index, 0] > .5, arm)
                transitions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        cells = []
        with (OUTPUT / "cells.jsonl").open("x", encoding="utf-8") as stream:
            for ci, (family, severity) in enumerate(b4.CONDITIONS):
                condition = b4._condition_key(family, severity)
                saved = parent["parent_cells"][ci]
                if condition != saved["condition"]:
                    raise RuntimeError("parent cell order differs")
                sub = slice(ci * 8, (ci + 1) * 8)
                row = {"dataset": "NUDT-SIRST", "condition": condition, "corruption": family, "severity": severity}
                for method in METHODS:
                    destination = OUTPUT / "conditions" / condition / method
                    destination.mkdir(parents=True, exist_ok=False)
                    for image_id, probability in zip(ids, probabilities[method][sub], strict=True):
                        save_mask(destination / f"{image_id}.png", probability)
                    if method in ARMS:
                        result = b3._evaluation_result(probabilities[method][sub], targets, ids)
                        row[method], full = b3._endpoint_summary(result), result.to_dict()
                    else:
                        old = "trained" if method == "v1" else method
                        row[method] = copy.deepcopy(saved[old])
                        full = json.loads((parent_root / "conditions" / condition / old / "metrics.json").read_text())
                    write_json(destination / "metrics.json", full)
                cells.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        summary = summarize(cells, transitions, initial_losses, final_losses)
        summary.update({"scope": raw["scope"], "image_ids": ids, "samples": 104,
            "additional_optimizer_steps_per_arm": 128, "inherited_source_fit_steps": 128,
            "saved_masks_all_methods": 520, "fit_seconds": fit_seconds,
            "head_host_and_anchor_unchanged": True, "test_payload_opens": 0,
            "new_raw_image_or_gt_decodes": 0, "new_o3_episodes": 0,
            "elapsed_seconds": time.perf_counter() - started})
        manager.assert_source_state()
        verify_parent_run(parent_root)
        for binding in bindings:
            paths._binding(paths._path(binding["path"]), binding["sha256"])
        write_json(OUTPUT / "summary.json", summary)
        print(json.dumps({"phase": "paired_fit_complete", "nonclean_macro": summary["nonclean_macro"],
            "failed_goals": summary["failed_goals"], "initial_losses": initial_losses,
            "final_losses": final_losses, "fit_seconds": fit_seconds}, ensure_ascii=False, sort_keys=True), flush=True)
        write_json(OUTPUT / "artifact_ledger.json", paths._file_ledger(OUTPUT))
        write_json(OUTPUT / "COMPLETE.json", {"complete": True, "samples": 104,
            "steps_per_arm": 128, "paper_result": False, "automatic_full_training_allowed": False,
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
                                          "completed_steps": steps_done, "host_restored": restored})
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
