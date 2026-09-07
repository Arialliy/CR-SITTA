#!/usr/bin/env python3
"""Fixed-budget supervised fit smoke; original O3/P2 is evaluated unchanged.

This is source training on eight known train images, not validation, test,
meta-learning, or evidence of generalization. No frozen predecessor is edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs/cr_sitta_o3_multiscale_train8_v1.yaml"
CONFIG_SHA256 = "f82355e83569bdc8343118f91758700b76d8f03f2e4d3b77b6b196fa010839ee"
DATASET = "NUDT-SIRST"
OUTPUT = ROOT / "results/cr_sitta/o3_multiscale_train8_v1/R0"


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def supervised_loss(logits, targets):
    if (logits.shape != targets.shape or logits.ndim != 4 or logits.shape[1] != 1
            or min(logits.shape) < 1 or logits.dtype not in (torch.float32, torch.float64)
            or logits.dtype != targets.dtype or logits.device != targets.device
            or targets.requires_grad or targets.grad_fn is not None
            or not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all())
            or not bool(((targets == 0) | (targets == 1)).all())):
        raise ValueError("loss requires finite matching logits and detached binary GT [N,1,H,W]")
    probability = torch.sigmoid(logits)
    intersection = (probability * targets).sum((1, 2, 3))
    union = (probability + targets - probability * targets).sum((1, 2, 3))
    return F.binary_cross_entropy_with_logits(logits, targets) + (1 - (intersection + 1) / (union + 1)).mean()


def training_order(count, steps=128, batch_size=4, seed=42):
    if any(type(v) is not int or v <= 0 for v in (count, steps, batch_size)):
        raise ValueError("positive integer schedule dimensions required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = []
    while len(indices) < steps * batch_size:
        indices.extend(torch.randperm(count, generator=generator).tolist())
    return [indices[i * batch_size:(i + 1) * batch_size] for i in range(steps)]


def macro(rows, field):
    return {metric: float(np.mean([row[field][metric] for row in rows]))
            for metric in ("iou", "normalized_iou", "pd", "fa_per_million")}


def performance_signal(cells, initial_loss, final_loss, tolerance=1e-12):
    clean_rows = [row for row in cells if row["corruption"] == "clean"]
    nonclean = [row for row in cells if row["corruption"] != "clean"]
    if len(clean_rows) != 1 or len(nonclean) != 12 or len({row["condition"] for row in cells}) != 13:
        raise ValueError("exactly one clean and twelve unique nonclean conditions required")
    old, new = macro(nonclean, "o3"), macro(nonclean, "trained")
    clean = clean_rows[0]
    goals = {
        "full_fit_loss_decreased": final_loss < initial_loss - tolerance,
        "nonclean_iou_above_o3": new["iou"] > old["iou"] + tolerance,
        "nonclean_pd_not_below_o3": new["pd"] >= old["pd"] - tolerance,
        "nonclean_fa_not_above_o3": new["fa_per_million"] <= old["fa_per_million"] + tolerance,
        "clean_iou_not_below_o3": clean["trained"]["iou"] >= clean["o3"]["iou"] - tolerance,
        "clean_pd_not_below_o3": clean["trained"]["pd"] >= clean["o3"]["pd"] - tolerance,
        "clean_fa_not_above_o3": clean["trained"]["fa_per_million"] <= clean["o3"]["fa_per_million"] + tolerance,
    }
    return {"goals": goals, "failed_goals": [k for k, v in goals.items() if not v],
            "learning_check_passed": goals["full_fit_loss_decreased"],
            "fit_performance_signal_passed": all(v for k, v in goals.items() if k != "full_fit_loss_decreased"),
            "nonclean_macro": {name: macro(nonclean, name) for name in ("source", "o3", "trained")},
            "clean": clean, "generalization_claim": False, "automatic_full_training_allowed": False}


def prepare():
    from analysis import spatial_residual_contract_v1 as previous
    from scripts import run_p3_stage_b4_full_pilot64_v1 as b4
    payload = CONFIG.read_bytes()
    if CONFIG.is_symlink() or hashlib.sha256(payload).hexdigest() != CONFIG_SHA256:
        raise ValueError("frozen train8 configuration byte hash differs")
    raw = yaml.safe_load(payload)
    expected_training = {"optimizer": "Adam", "learning_rate": 0.001, "betas": [0.9, 0.999],
                         "epsilon": 1e-8, "weight_decay": 0.0, "batch_size": 4, "steps": 128,
                         "loss": "mean_binary_cross_entropy_with_logits_plus_mean_per_image_soft_iou_loss",
                         "soft_iou_smoothing": 1.0,
                         "order": "seeded_torch_cpu_permutations_repeated_until_budget",
                         "checkpoint_rule": "fixed_last_step_no_metric_selection", "evaluate_steps": [0, 128]}
    if (raw["protocol_id"] != "cr-sitta-o3-multiscale-train8-v1" or raw["dataset"] != DATASET
            or ROOT / raw["result_root"] != OUTPUT or raw["image_count"] != 8 or raw["seed"] != 42
            or raw["training"] != expected_training
            or raw["parent_config_sha256"] != b4.FROZEN_CONFIG_SHA256
            or raw["scope"] != {"role": "source_supervised_fit_smoke", "train_only": True,
                               "no_validation_split": True, "formal_test": False, "paper_result": False,
                               "full_training": False, "fit_and_measure_on_same_images": True}):
        raise ValueError("only the fixed source-supervised train8 smoke is supported")
    parent = b4.load_contract(ROOT / raw["parent_config"])
    bindings = previous._historical_bindings(parent, previous.read_config())
    ids, inputs = previous._input_bindings(parent, DATASET)
    bindings.extend(inputs)
    paths = [str(CONFIG.relative_to(ROOT)), raw["preregistration"], *raw["implementation_files"],
             "analysis/spatial_residual_contract_v1.py", "scripts/run_o3_spatial_residual_v1.py",
             "configs/cr_sitta_o3_spatial_residual_v1.yaml"]
    bindings.extend(previous._binding(previous._path(path)) for path in paths)
    unique = {item["path"]: item for item in bindings}
    return raw, parent, ids[:8], [unique[key] for key in sorted(unique)]


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def run():
    from analysis import spatial_residual_contract_v1 as previous
    from analysis.o3_multiscale_train_data_v1 import load_train_targets
    from model.o3_multiscale_residual_v1 import O3MultiScaleResidual
    from scripts import run_p3_stage_b_screen_v1 as b3
    from scripts import run_p3_stage_b4_full_pilot64_v1 as b4
    from scripts.run_o3_spatial_residual_v1 import save_mask, validate_sample
    from tta.o3_endpoint_features_v1 import capture_o3_endpoint
    from tta.views import validated_student_perturbations

    raw, parent, ids, bindings = prepare()
    previous._reject_symlinks(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(exist_ok=False)
    manifest = {"protocol_id": raw["protocol_id"], "configuration": raw, "image_ids": ids,
                "bindings": bindings, "phase": "source_supervised_fit_smoke",
                "original_o3_method_label_accesses": 0, "fit_gt_access_authorized": True,
                "formal_test": False, "no_validation_split": True, "paper_result": False}
    write_json(OUTPUT / "manifest.json", manifest)
    started = time.perf_counter()
    state_manager = None
    completed = 0
    try:
        _, host, adapter, film, state_manager, device, wrapper = b3._build_runtime(parent, DATASET, "cuda:0")
        write_json(OUTPUT / "runtime.json", {**b3._runtime_environment_receipt(torch, device),
                                           "checkpoint_wrapper": wrapper})
        branch = O3MultiScaleResidual().to(device)
        if sum(p.numel() for p in branch.parameters()) != 1568:
            raise RuntimeError("unexpected spatial branch parameter count")
        initial_state = {key: value.detach().cpu().clone() for key, value in branch.state_dict().items()}
        torch.save({"model_label": raw["model"]["label"], "adapter_state_dict": initial_state,
                    "step": 0, "host_checkpoint": parent.raw["datasets"][DATASET],
                    "config_sha256": previous.sha256_file(CONFIG)}, OUTPUT / "initial.pth.tar")
        conditions = tuple(b4.CONDITIONS)
        count = len(ids) * len(conditions)
        feature = np.lib.format.open_memmap(OUTPUT / "o3_features.npy", mode="w+", dtype="<f4", shape=(count, 16, 256, 256))
        source = np.empty((count, 1, 256, 256), dtype=np.float32)
        o3 = np.empty_like(source)
        teacher_manifest = b4._teacher_manifest(parent, DATASET)
        perturb = validated_student_perturbations((parent.raw["view_library"]["student_perturbation"],))[0]
        with (OUTPUT / "episodes.jsonl").open("x", encoding="utf-8") as episode_log:
            for ci, (corruption, severity) in enumerate(conditions):
                condition = b4._condition_key(corruption, severity)
                inputs = b3._method_input_dataset(parent, DATASET, condition)
                record = b3._teacher_condition_record(teacher_manifest, condition)
                teachers = b3._verified_teacher_array(parent, DATASET, teacher_manifest, record, "source_probabilities")
                uncertainty = b3._verified_teacher_array(parent, DATASET, teacher_manifest, record, "view_uncertainty")
                for ii, image_id in enumerate(ids):
                    sample = dict(inputs[ii])
                    validate_sample(sample, image_id=image_id, dataset=DATASET, corruption=corruption, severity=severity, seed=42)
                    observed = sample["image"].unsqueeze(0).to(device)
                    teacher = torch.from_numpy(np.array(teachers[ii], copy=True)).unsqueeze(0).to(device)
                    uncertain = torch.from_numpy(np.array(uncertainty[ii, 0], copy=True)).unsqueeze(0).to(device)
                    result = capture_o3_endpoint(contract=parent, model=host, adapter=adapter, film=film,
                                                 state_manager=state_manager, image=observed,
                                                 student_image=perturb.forward(observed), teacher=teacher, uncertainty=uncertain)
                    k = ci * len(ids) + ii
                    feature[k] = result["features"][0].numpy()
                    source[k] = result["source_probabilities"][0].numpy()
                    o3[k] = result["o3_probabilities"][0].numpy()
                    if not np.array_equal(source[k], teachers[ii]):
                        raise RuntimeError("source differs from frozen teacher")
                    with torch.no_grad():
                        h = result["features"].to(device)
                        if not torch.equal(branch(h), h):
                            raise RuntimeError("initial branch feature identity failed")
                        if not np.array_equal(torch.sigmoid(host.output_0(branch(h)))[0].cpu().numpy(), o3[k]):
                            raise RuntimeError("initial branch does not reproduce original O3")
                    episode_log.write(json.dumps({"condition": condition, "image_id": image_id,
                                                   "feature_sha256": tensor_sha(result["features"]),
                                                   "input_sha256": tensor_sha(observed),
                                                   "o3_diagnostics": result["diagnostics"]}, allow_nan=False) + "\n")
                    episode_log.flush()
                    completed += 1
                print(json.dumps({"phase": "o3_feature_capture", "condition": condition, "completed": completed, "total": count}), flush=True)
        feature.flush()
        np.save(OUTPUT / "source_probabilities.npy", source, allow_pickle=False)
        np.save(OUTPUT / "o3_probabilities.npy", o3, allow_pickle=False)
        state_manager.assert_source_state()
        # This independent API reads original, allowlisted TRAIN mask PNGs.
        # It does not pretend the old outer-only target cache is a training API.
        targets, target_receipt = load_train_targets(parent, ids)
        write_json(OUTPUT / "train_target_receipt.json", target_receipt)
        np.save(OUTPUT / "train_targets.npy", targets, allow_pickle=False)
        target_all = np.tile(targets, (len(conditions), 1, 1, 1))

        def batch(indices):
            return (torch.from_numpy(np.array(feature[indices], copy=True)).to(device),
                    torch.from_numpy(np.array(target_all[indices], copy=True)).to(device))

        def full_fit_loss():
            total = 0.0
            with torch.no_grad():
                for offset in range(0, count, 4):
                    h, target = batch(list(range(offset, min(offset + 4, count))))
                    total += float(supervised_loss(host.output_0(branch(h)), target)) * len(h)
            return total / count

        initial_loss = full_fit_loss()
        order = training_order(count)
        write_json(OUTPUT / "training_order.json", {"indices": order, "sample_order": "condition_major_then_image_id"})
        optimizer = torch.optim.Adam(branch.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        with (OUTPUT / "training.jsonl").open("x", encoding="utf-8") as training_log:
            for step, indices in enumerate(order, 1):
                h, target = batch(indices)
                optimizer.zero_grad(set_to_none=True)
                loss = supervised_loss(host.output_0(branch(h)), target)
                loss.backward()
                norms = {name: float(parameter.grad.norm()) for name, parameter in branch.named_parameters()
                         if parameter.grad is not None}
                if len(norms) != len(tuple(branch.parameters())) or not all(np.isfinite(v) for v in norms.values()):
                    raise RuntimeError("missing or nonfinite supervised gradient")
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in branch.parameters()):
                    raise RuntimeError("nonfinite trained weights")
                row = {"step": step, "loss": float(loss.detach()), "sample_indices": indices, "gradient_norms": norms}
                training_log.write(json.dumps(row, allow_nan=False) + "\n")
                training_log.flush()
                if step % 16 == 0:
                    print(json.dumps({"phase": "supervised_train", "step": step, "total_steps": 128, "loss": row["loss"]}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        branch.eval()
        final_loss = full_fit_loss()
        state_manager.assert_source_state()
        torch.save({"model_label": raw["model"]["label"], "adapter_state_dict": branch.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "step": 128,
                    "host_checkpoint": parent.raw["datasets"][DATASET], "config_sha256": previous.sha256_file(CONFIG),
                    "selection_rule": "fixed_last_step", "trained_on": "source_train8",
                    "requires_original_o3_p2_endpoint": True}, OUTPUT / "step_0128.pth.tar")
        trained = np.empty_like(o3)
        # Batch-one replay matches the eventual single-image inference setting.
        with torch.no_grad():
            for k in range(count):
                h, _ = batch([k])
                corrected = branch(h)
                ratio = (corrected - h).square().mean().sqrt() / h.square().mean().clamp_min(1e-12).sqrt()
                if not bool(torch.isfinite(corrected).all()) or float(ratio) > 0.050001:
                    raise RuntimeError("trained residual violates its finite RMS bound")
                trained[k] = torch.sigmoid(host.output_0(corrected))[0].cpu().numpy()
        np.save(OUTPUT / "trained_probabilities.npy", trained, allow_pickle=False)
        cells = []
        with (OUTPUT / "cells.jsonl").open("x", encoding="utf-8") as cell_log:
            for ci, (corruption, severity) in enumerate(conditions):
                condition = b4._condition_key(corruption, severity)
                sub = slice(ci * len(ids), (ci + 1) * len(ids))
                row = {"dataset": DATASET, "condition": condition, "corruption": corruption, "severity": severity}
                for name, probabilities in (("source", source[sub]), ("o3", o3[sub]), ("trained", trained[sub])):
                    destination = OUTPUT / "conditions" / condition / name
                    destination.mkdir(parents=True, exist_ok=False)
                    for image_id, probability in zip(ids, probabilities, strict=True):
                        save_mask(destination / f"{image_id}.png", probability)
                    result = b3._evaluation_result(probabilities, targets, ids)
                    row[name] = b3._endpoint_summary(result)
                    write_json(destination / "metrics.json", result.to_dict())
                cells.append(row)
                cell_log.write(json.dumps(row, allow_nan=False) + "\n")
                cell_log.flush()
        summary = {**performance_signal(cells, initial_loss, final_loss),
                   "scope": raw["scope"], "dataset": DATASET, "image_ids": ids,
                   "samples": count, "optimizer_steps": 128, "initial_full_fit_loss": initial_loss,
                   "final_full_fit_loss": final_loss, "mask_count_per_method": count,
                   "mask_count_total": count * 3, "host_restored": True,
                   "original_o3_method_label_accesses": 0, "test_payload_opens": 0,
                   "elapsed_seconds": time.perf_counter() - started}
        state_manager.assert_source_state()
        for binding in bindings:
            previous._binding(previous._path(binding["path"]), binding["sha256"])
        write_json(OUTPUT / "summary.json", summary)
        # No fallible stdout write after COMPLETE: an output pipe failure must
        # not retroactively label a sealed successful run as FAILED.
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
        write_json(OUTPUT / "artifact_ledger.json", previous._file_ledger(OUTPUT))
        write_json(OUTPUT / "COMPLETE.json", {"complete": True, "samples": count, "steps": 128,
                                             "manifest_sha256": previous.sha256_file(OUTPUT / "manifest.json"),
                                             "ledger_sha256": previous.sha256_file(OUTPUT / "artifact_ledger.json"),
                                             "paper_result": False, "automatic_full_training_allowed": False})
        return summary
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
                                          "captured_samples": completed, "host_restored": restored})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        raw, _, ids, bindings = prepare()
        print(json.dumps({"protocol": raw["protocol_id"], "image_ids": ids,
                          "bound_files": len(bindings), "payloads_deserialized": 0}, sort_keys=True))
    else:
        run()


if __name__ == "__main__":
    main()
