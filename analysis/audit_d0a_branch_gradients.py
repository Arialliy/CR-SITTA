"""Read-only endpoint clean/LF/HF gradient diagnosis on the frozen train Pilot64.

This does not train, update an optimizer, access test payloads, or promote a
scientific stage. The Adagrad calculation is a local counterfactual at the
completed endpoint, not reconstruction of historical training drift.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_cr_sitta_d0a import (  # noqa: E402
    batchnorm_batch_stats_no_running_update,
    build_deteriorated_view,
    compute_segmentation_loss,
)

DATASET = "NUDT-SIRST"
CHECKPOINT = ROOT / "results/cr_sitta/d0a_supervised_lfhf_train_v2/NUDT-SIRST/epoch_1000_train_only.pth.tar"
CHECKPOINT_SHA256 = "6909659608f0241d1cb8822200f73de267b8e620dd619d4423783a14c5f67a75"
TRAIN_CONTRACT = CHECKPOINT.parent / "run_contract.json"
TRAIN_CONTRACT_SHA256 = "2f004c142949481cd055d0ea7700c88d04045cf0ba0e6eec45f1dc394993044e"
OUTPUT = ROOT / "results/cr_sitta/d0a_v7_diagnostics_v1/branch_gradients/NUDT-SIRST"
STRUCTURAL_GROUPS = (
    "encoder", "fpn_lfp", "fpn_sfs", "decoder3_2", "decoder1", "decoder0", "head"
)
OVERLAPPING_GROUPS = ("bn_affine", "non_bn", "global")
BRANCHES = ("clean", "lf_mask", "hf_noise")
PAIRS = (("clean", "lf_mask"), ("clean", "hf_noise"), ("lf_mask", "hf_noise"))


def tensor_hash(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_hash(tensor).encode())
    return digest.hexdigest()


def structural_group(name: str) -> str:
    if name.startswith(("conv_init.", "encoder_", "middle_layer.")):
        return "encoder"
    if name.startswith("fpn.crossattn_list."):
        return "fpn_sfs"
    if name.startswith("fpn."):
        return "fpn_lfp"
    if name.startswith(("decoder_3.", "decoder_2.")):
        return "decoder3_2"
    if name.startswith("decoder_1."):
        return "decoder1"
    if name.startswith("decoder_0."):
        return "decoder0"
    if name.startswith(("output_", "final.")):
        return "head"
    raise ValueError(f"unmapped model parameter: {name}")


def parameter_mapping(model: nn.Module) -> list[dict[str, Any]]:
    bn_names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for child_name, _ in module.named_parameters(recurse=False):
                bn_names.add(f"{module_name}.{child_name}" if module_name else child_name)
    rows = []
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if not parameter.requires_grad:
            raise ValueError(f"endpoint audit requires the original trainable schema: {name}")
        rows.append({
            "parameter_index": index, "name": name, "shape": list(parameter.shape),
            "dtype": str(parameter.dtype), "numel": parameter.numel(),
            "structural_group": structural_group(name), "bn_affine": name in bn_names,
            "overlapping_groups": ["bn_affine" if name in bn_names else "non_bn", "global"],
        })
    if not rows or len(rows) != len({row["name"] for row in rows}):
        raise ValueError("invalid or empty parameter mapping")
    return rows


def names_for_group(mapping: Sequence[Mapping[str, Any]], group: str) -> list[str]:
    return [str(row["name"]) for row in mapping if
            row["structural_group"] == group or group in row["overlapping_groups"]]


def branch_gradient(
    model: nn.Module, image: Tensor, target: Tensor,
    loss_builder: Callable[[Any, Tensor], Tensor], *, degraded: bool,
    warm_flag: bool = False,
) -> tuple[float, dict[str, Tensor | None], dict[str, Any]]:
    """Compute autograd.grad and restore all buffers/modes even on exceptions."""
    initial_hash = state_hash(model)
    buffers = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}
    modes = {name: module.training for name, module in model.named_modules()}
    target_hash = tensor_hash(target)
    image_hash = tensor_hash(image)
    named = list(model.named_parameters())
    try:
        model.train()
        context = batchnorm_batch_stats_no_running_update(model) if degraded else nullcontext()
        with context:
            outputs = model(image, warm_flag)
            loss = loss_builder(outputs, target)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("branch loss is non-finite")
            gradients = torch.autograd.grad(loss, [parameter for _, parameter in named], allow_unused=True)
        changed_buffers = [name for name, tensor in model.named_buffers()
                           if not torch.equal(tensor.detach(), buffers[name])]
        if degraded and changed_buffers:
            raise RuntimeError(f"degraded branch persisted buffer updates: {changed_buffers}")
        detached = {}
        for (name, _), gradient in zip(named, gradients):
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"non-finite branch gradient: {name}")
            detached[name] = None if gradient is None else gradient.detach().cpu().clone()
        result = float(loss.detach().cpu()), detached, {
            "input_tensor_sha256": image_hash, "target_tensor_sha256": target_hash,
            "temporary_changed_buffers": changed_buffers,
            "batchnorm_policy": "train_batch_stats_no_running_update" if degraded else "train_batch_stats_temporary_running_update",
            "state_sha256_before": initial_hash,
        }
    finally:
        with torch.no_grad():
            for name, tensor in model.named_buffers():
                tensor.copy_(buffers[name])
        for name, module in model.named_modules():
            module.training = modes[name]
        if state_hash(model) != initial_hash:
            raise RuntimeError("branch audit changed model state despite buffer restoration")
        if tensor_hash(target) != target_hash or tensor_hash(image) != image_hash:
            raise RuntimeError("branch audit modified its image or target")
    result[2]["state_sha256_after"] = state_hash(model)
    result[2]["model_state_unchanged"] = True
    return result


def vector_statistics(
    first: Mapping[str, Tensor | None], second: Mapping[str, Tensor | None], names: Sequence[str],
) -> dict[str, Any]:
    aa = bb = dot = 0.0
    used_first = used_second = both = 0
    for name in names:
        left, right = first[name], second[name]
        if left is not None:
            used_first += 1
            aa += float(left.double().square().sum())
        if right is not None:
            used_second += 1
            bb += float(right.double().square().sum())
        if left is not None and right is not None:
            both += 1
            dot += float((left.double() * right.double()).sum())
    valid = aa > 0.0 and bb > 0.0
    return {
        "dot": dot, "first_squared_norm": aa, "second_squared_norm": bb,
        "first_norm": math.sqrt(aa), "second_norm": math.sqrt(bb),
        "cosine": dot / math.sqrt(aa * bb) if valid else None,
        "second_to_first_norm_ratio": math.sqrt(bb / aa) if aa > 0 else None,
        "valid_cosine": valid, "undefined_reason": None if valid else "one_or_both_gradient_norms_zero",
        "parameter_count": len(names), "first_used_parameters": used_first,
        "second_used_parameters": used_second, "both_used_parameters": both,
    }


def pool_pair_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aa = sum(float(row["first_squared_norm"]) for row in rows)
    bb = sum(float(row["second_squared_norm"]) for row in rows)
    dot = sum(float(row["dot"]) for row in rows)
    valid = aa > 0 and bb > 0
    return {
        "definition": "concatenate_all_batch_gradient_vectors_then_cosine_not_mean_of_cosines",
        "batches": len(rows), "valid_batches": sum(bool(row["valid_cosine"]) for row in rows),
        "dot": dot, "first_squared_norm": aa, "second_squared_norm": bb,
        "cosine": dot / math.sqrt(aa * bb) if valid else None,
        "second_to_first_norm_ratio": math.sqrt(bb / aa) if aa > 0 else None,
        "undefined_reason": None if valid else "one_or_both_gradient_norms_zero",
    }


def adagrad_state_by_name(
    model: nn.Module, optimizer_state: Mapping[str, Any], *, learning_rate: float,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Map the archived original optimizer order to the frozen original model."""
    groups = optimizer_state["param_groups"]
    if len(groups) != 1:
        raise ValueError("expected the original single Adagrad parameter group")
    group = groups[0]
    if float(group["lr"]) != learning_rate or any(float(group.get(k, 0)) != 0 for k in ("weight_decay", "lr_decay")):
        raise ValueError("Adagrad settings differ from the frozen endpoint")
    if bool(group.get("maximize", False)):
        raise ValueError("maximizing Adagrad is not the training optimizer")
    named = list(model.named_parameters())
    parameter_ids = group["params"]
    if len(parameter_ids) != len(named) or len(set(parameter_ids)) != len(named):
        raise ValueError("optimizer/model parameter coverage mismatch")
    if set(optimizer_state["state"]) != set(parameter_ids):
        raise ValueError("optimizer state coverage mismatch")
    sums = {}
    steps = {}
    for parameter_id, (name, parameter) in zip(parameter_ids, named):
        value = optimizer_state["state"][parameter_id]
        accumulator = value["sum"].detach().cpu()
        if accumulator.shape != parameter.shape or accumulator.dtype != parameter.dtype:
            raise ValueError(f"Adagrad accumulator schema mismatch: {name}")
        if not bool(torch.isfinite(accumulator).all()) or bool((accumulator < 0).any()):
            raise ValueError(f"invalid Adagrad accumulator: {name}")
        sums[name] = accumulator.clone()
        step = value["step"]
        steps[name] = float(step.item() if isinstance(step, Tensor) else step)
    return sums, {
        "optimizer": "Adagrad", "lr": learning_rate, "eps": float(group["eps"]),
        "parameter_count": len(named), "parameter_step_counts": steps,
        "mapping_basis": "frozen_training_runtime_Adagrad(model.parameters())_and_exact_original_model_parameter_order",
    }


def simulate_adagrad_step(
    clean: Mapping[str, Tensor | None], degraded: Mapping[str, Tensor | None],
    accumulators: Mapping[str, Tensor], names: Sequence[str], *, lr: float, eps: float,
) -> dict[str, Any]:
    """One local endpoint step; all operands remain unchanged and no step is applied."""
    gradient_sq = step_sq = clean_sq = dot_clean = accumulator_sum = increment_sq = 0.0
    rates_sum = 0.0
    rates_min = float("inf")
    rates_max = 0.0
    coordinates = 0
    updated_parameters = 0
    for name in names:
        gc, gd = clean[name], degraded[name]
        if gc is None and gd is None:
            continue
        updated_parameters += 1
        accumulator = accumulators[name].double()
        zero = torch.zeros_like(accumulator)
        gc64 = zero if gc is None else gc.double()
        gd64 = zero if gd is None else gd.double()
        combined = (gc64 + gd64) / 2.0
        increment = combined.square()
        effective_lr = lr / ((accumulator + increment).sqrt() + eps)
        step = -effective_lr * combined
        gradient_sq += float(increment.sum())
        increment_sq += float(increment.square().sum())
        step_sq += float(step.square().sum())
        clean_sq += float(gc64.square().sum())
        dot_clean += float((step * gc64).sum())
        accumulator_sum += float(accumulator.sum())
        rates_sum += float(effective_lr.sum())
        rates_min = min(rates_min, float(effective_lr.min()))
        rates_max = max(rates_max, float(effective_lr.max()))
        coordinates += combined.numel()
    return {
        "formula": "g=(g_clean+g_probe)/2; delta=-lr*g/(sqrt(G_endpoint+g^2)+eps)",
        "precision": "float64_diagnostic_arithmetic_on_frozen_float32_endpoint_tensors",
        "optimizer_steps_applied": 0, "updated_parameter_count_if_applied": updated_parameters,
        "coordinate_count_if_applied": coordinates, "combined_gradient_norm": math.sqrt(gradient_sq),
        "step_norm": math.sqrt(step_sq), "step_dot_clean_gradient": dot_clean,
        "step_cosine_with_clean_gradient": dot_clean / math.sqrt(step_sq * clean_sq) if step_sq > 0 and clean_sq > 0 else None,
        "linearized_clean_loss_change": dot_clean,
        "accumulator_endpoint_sum": accumulator_sum,
        "accumulator_increment_sum": gradient_sq,
        "accumulator_increment_l2": math.sqrt(increment_sq),
        "accumulator_relative_sum_increment": gradient_sq / accumulator_sum if accumulator_sum > 0 else None,
        "effective_lr_min": rates_min if coordinates else None,
        "effective_lr_max": rates_max if coordinates else None,
        "effective_lr_mean": rates_sum / coordinates if coordinates else None,
        "step_to_combined_gradient_norm_ratio": math.sqrt(step_sq / gradient_sq) if gradient_sq > 0 else None,
        "historical_drift_inference_permitted": False,
    }


def e2_diagnosis(pooled: Mapping[str, Any], *, required_negative: int = 4, minimum_valid_batches: int = 3) -> dict[str, Any]:
    probe_reports = {}
    for probe in ("lf_mask", "hf_noise"):
        key = f"clean__{probe}"
        valid = [group for group in STRUCTURAL_GROUPS if pooled[group][key]["valid_batches"] >= minimum_valid_batches
                 and pooled[group][key]["cosine"] is not None]
        negative = [group for group in valid if pooled[group][key]["cosine"] < 0.0]
        probe_reports[probe] = {
            "negative_groups": negative, "negative_group_count": len(negative),
            "eligible_groups": valid, "insufficient_groups": [group for group in STRUCTURAL_GROUPS if group not in valid],
            "triggered": len(negative) >= required_negative,
        }
    return {
        "rule": "per_probe_at_least_4_of_7_disjoint_structural_groups_pooled_cosine_lt_0_with_at_least_3_valid_batches_per_group",
        "required_negative_groups": required_negative, "minimum_valid_batches": minimum_valid_batches,
        "probes": probe_reports, "any_probe_triggered": any(report["triggered"] for report in probe_reports.values()),
        "diagnostic_only": True, "stage_promotion_authorized": False,
        "interpretation": "A trigger argues against unqualified equal-weight gradients locally; absence is not a safety or scientific gate pass.",
    }


def validate_access_counters(access: Mapping[str, Any]) -> None:
    for key in ("train_image_opens", "train_mask_opens", "test_image_opens",
                "test_mask_opens", "validation_image_opens", "validation_mask_opens"):
        expected = 64 if key.startswith("train_") else 0
        if access.get(key) != expected:
            raise RuntimeError(f"unexpected data access count: {key}={access.get(key)}, expected {expected}")


def checkpoint_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    from analysis.d0a_v7_common import sha256_file
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256 or sha256_file(TRAIN_CONTRACT) != TRAIN_CONTRACT_SHA256:
        raise ValueError("fixed NUDT endpoint/checkpoint contract hash mismatch")
    contract = json.loads(TRAIN_CONTRACT.read_text())
    for raw, expected in contract["runtime_sha256"].items():
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        if sha256_file(path) != expected:
            raise ValueError(f"original training runtime drift: {path}")
    settings = config["gradient"]
    if int(settings["batch_size"]) != 16 or int(settings["num_batches"]) != 4 or int(config["human_epoch"]) != 1000:
        raise ValueError("only the fixed 4x16 Pilot64 endpoint diagnosis is authorized")
    if float(settings["learning_rate"]) != 0.05:
        raise ValueError("frozen endpoint learning rate drift")
    for key, expected in (("dataset", DATASET), ("view", "train_crop_224"),
                          ("warm_epochs", 5), ("optimizer_steps", 0),
                          ("critical_structural_group_count", 7),
                          ("minimum_valid_batches", 3), ("conflict_group_count_threshold", 4),
                          ("conflict_cosine_threshold", 0.0),
                          ("aggregation", "concatenated_batch_gradients_cosine"),
                          ("checkpoint_sha256", CHECKPOINT_SHA256),
                          ("run_contract_sha256", TRAIN_CONTRACT_SHA256)):
        if settings[key] != expected:
            raise ValueError(f"frozen gradient setting drift: {key}")
    for key, expected in (("checkpoint", CHECKPOINT), ("run_contract", TRAIN_CONTRACT), ("output", OUTPUT)):
        if (ROOT / settings[key]).resolve() != expected:
            raise ValueError(f"frozen gradient path drift: {key}")
    if config["global_seed"] != 42 or config["probe_seed_namespace"] != contract["run_config"]["protocol_id"]:
        raise ValueError("frozen training probe seed/namespace drift")
    expected_probes = {"lf_mask": {"mask_ratio": 0.20, "keep_probability": 0.50},
                       "hf_noise": {"target_rms": 0.02, "low_cut_ratio": 0.20}}
    if config["probes"] != expected_probes:
        raise ValueError("diagnostic probe parameters differ from original training")
    return contract


def run(config_path: Path, *, device_name: str, execute: bool) -> dict[str, Any]:
    from analysis.d0a_v7_common import (
        read_config, load_pilot_records, load_sample, runtime_bindings,
        reserve_output, freeze_run, complete_run, write_json_new, write_jsonl_new,
    )
    config = read_config(config_path)
    training_contract = checkpoint_preflight(config)
    records = load_pilot_records(DATASET, config)
    if len(records) != 64:
        raise ValueError("requires all and only frozen Pilot64 train IDs")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing complete or partial existing output: {OUTPUT}")
    if not execute:
        return {"ready": True, "writes": 0, "execute": False, "dataset": DATASET, "images": 64,
                "checkpoint_sha256": CHECKPOINT_SHA256, "output": str(OUTPUT),
                "test_payload_opens": 0, "validation_payload_opens": 0}
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("actual NS-FPN branch diagnosis requires CUDA; CPU helpers are unit-tested separately")
    extra = [Path(__file__).resolve(), CHECKPOINT, TRAIN_CONTRACT]
    extra.extend(Path(raw) if Path(raw).is_absolute() else ROOT / raw for raw in training_contract["runtime_sha256"])
    bindings = runtime_bindings(config_path, extra, records)
    contract = {
        "schema_version": 1, "role": "P3_train_only_fixed_endpoint_branch_gradient_diagnosis", "dataset": DATASET,
        "development_only": True, "paper_result": False, "no_validation_split": True,
        "config_path": str(config_path.resolve()), "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_selection": "fixed_epoch_1000_train_only", "human_epoch": 1000, "seed": 42,
        "view": "train_crop_224", "batch_size": 16, "batches": 4,
        "batch_order": "contiguous_frozen_pilot_id_file_order_no_shuffle", "image_ids": [r["image_id"] for r in records],
        "branches": list(BRANCHES), "device": str(device), "no_optimizer_step": True,
        "structural_groups": list(STRUCTURAL_GROUPS), "overlapping_groups": list(OVERLAPPING_GROUPS),
        "fpn_lfp_scope": "all_fpn_parameters_except_crossattn_list_including_lateral_and_fpn_convs_not_pure_LFP",
        "e2_rule": "per_probe_4_of_7_groups_pooled_cos_lt_0_minimum_3_valid_batches",
        "historical_drift_inference_permitted": False, "test_payload_access_allowed": False,
        "validation_payload_access_allowed": False, "stage_promotion_authorized": False,
    }
    reserve_output(OUTPUT)
    freeze_run(OUTPUT, contract, bindings)
    # Exact file hash was anchored before this local, trusted original checkpoint
    # load. It contains numpy RNG metadata. No RNG restoration/resume is called.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 1000 or checkpoint["run_config"] != training_contract["run_config"]:
        raise ValueError("checkpoint endpoint or training configuration mismatch")
    if checkpoint.get("test_selected") is not False or checkpoint["selection_rule"] != "fixed_final_epoch_train_only":
        raise ValueError("unexpected checkpoint selection rule")
    if checkpoint["method_stage"] != "D0-A" or checkpoint["global_optimizer_step"] != 41000:
        raise ValueError("unexpected checkpoint training stage/completion")
    from model.MSHNet_NSFPN import MSHNet_NSFPN
    from model.loss import SLSIoULoss
    from train_fixed_split import seed_everything
    seed_everything(42)
    model = MSHNet_NSFPN(3).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if len(model.state_dict()) != 505:
        raise ValueError("expected 505-key original architecture")
    model.eval()
    mapping = parameter_mapping(model)
    accumulators, optimizer_metadata = adagrad_state_by_name(model, checkpoint["optimizer"], learning_rate=0.05)
    write_json_new(OUTPUT / "parameter_mapping.json", {"parameters": mapping, "optimizer": optimizer_metadata})
    del checkpoint
    initial_hash = state_hash(model)
    groups = STRUCTURAL_GROUPS + OVERLAPPING_GROUPS
    group_names = {group: names_for_group(mapping, group) for group in groups}
    loss_function = SLSIoULoss()
    def loss_builder(outputs: Any, target: Tensor) -> Tensor:
        return compute_segmentation_loss(outputs, target, loss_function, warm_epochs=5, epoch_index=999)
    batch_rows = []
    sample_rows = []
    access: dict[str, Any] = {}
    for batch_index in range(4):
        batch_records = records[16 * batch_index:16 * (batch_index + 1)]
        samples = [load_sample(record, "train_crop_224", config, access=access) for record in batch_records]
        image = torch.stack([item[0] for item in samples]).to(device)
        target = torch.stack([item[1] for item in samples]).to(device)
        identifiers = [record["image_id"] for record in batch_records]
        sample_rows.extend({"image_id": identifier, "batch_index": batch_index, **sample[2]}
                           for identifier, sample in zip(identifiers, samples))
        branch_results = {}
        for branch in BRANCHES:
            if branch == "clean":
                branch_image = image
            else:
                probes = config["probes"]
                with torch.no_grad():
                    branch_image = build_deteriorated_view(
                        image, identifiers, branch, str(config["probe_seed_namespace"]), 42, DATASET, 1000,
                        float(probes["lf_mask"]["mask_ratio"]), float(probes["lf_mask"]["keep_probability"]),
                        float(probes["hf_noise"]["target_rms"]), float(probes["hf_noise"]["low_cut_ratio"]),
                    )
            branch_results[branch] = branch_gradient(model, branch_image, target, loss_builder, degraded=branch != "clean")
            print(f"P3 dataset={DATASET} batch={batch_index + 1}/4 branch={branch} loss={branch_results[branch][0]:.8f}", flush=True)
        row = {"batch_index": batch_index, "image_ids": identifiers,
               "branch_losses": {branch: value[0] for branch, value in branch_results.items()},
               "branch_audits": {branch: value[2] for branch, value in branch_results.items()},
               "groups": {}, "parameter_gradient_coverage": []}
        for parameter in mapping:
            name = parameter["name"]
            row["parameter_gradient_coverage"].append({"name": name, "branches": {
                branch: {"autograd_used": value[1][name] is not None,
                         "norm": 0.0 if value[1][name] is None else float(value[1][name].double().norm())}
                for branch, value in branch_results.items()}})
        for group, names in group_names.items():
            row["groups"][group] = {
                "pairs": {f"{first}__{second}": vector_statistics(branch_results[first][1], branch_results[second][1], names)
                          for first, second in PAIRS},
                "endpoint_adagrad": {probe: simulate_adagrad_step(
                    branch_results["clean"][1], branch_results[probe][1], accumulators, names,
                    lr=optimizer_metadata["lr"], eps=optimizer_metadata["eps"])
                    for probe in ("lf_mask", "hf_noise")},
            }
        batch_rows.append(row)
        del branch_results, image, target, samples
    if state_hash(model) != initial_hash or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("audit mutated model state or persisted .grad buffers")
    validate_access_counters(access)
    pooled = {group: {f"{a}__{b}": pool_pair_statistics([row["groups"][group]["pairs"][f"{a}__{b}"] for row in batch_rows])
                      for a, b in PAIRS} for group in groups}
    summary = {
        **contract, "images": 64, "branch_backward_calls": 12,
        "model_state_sha256_before": initial_hash, "model_state_sha256_after": state_hash(model),
        "model_state_unchanged": True, "optimizer_steps_applied": 0, "persisted_parameter_gradients": 0,
        "pooled_group_statistics": pooled, "e2": e2_diagnosis(pooled), "access_counters": access,
        "test_payload_opens": 0, "validation_payload_opens": 0,
        "limitations": ["local_endpoint_only_not_historical_drift", "train_pilot_diagnostics_not_paper_metrics",
                        "four_pilot_batches_not_reconstruction_of_original_epoch1000_loader",
                        "unused_warmup_heads_have_zero_norm_and_undefined_cosine_not_artificial_zero_cosine"],
    }
    write_jsonl_new(OUTPUT / "per_batch.jsonl", batch_rows)
    write_jsonl_new(OUTPUT / "sample_transforms.jsonl", sample_rows)
    complete_run(OUTPUT, summary)
    return {"complete": True, "output": str(OUTPUT), "images": 64, "e2": summary["e2"], "model_state_unchanged": True}


def main() -> None:
    from analysis.d0a_v7_common import DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true", help="Freeze and execute train-only diagnostics; default is zero-write preflight.")
    args = parser.parse_args()
    print(json.dumps(run(args.config, device_name=args.device, execute=args.execute), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
