"""Bounded IPMA engineering checks on cached features; no training or file I/O.

The label-free API never accepts a label, identifier, path, or split. Only the
separate fit-only diagnostic accepts source-train GT. Native checks use the
unchanged float32 feature/head path. CPU float64 finite differences test their
own replay graph with fixed converted native teacher probabilities.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import math
from typing import Any

import torch
from torch import Tensor, nn

from model.ipma_d0_adapter_v1 import IdentityMetaAdapter
from model.loss import SLSIoULoss
from training.ipma_inner_v1 import inner_step, teacher_region_weights

INNER_LR = 0.05
INNER_RADIUS = 0.01
INNER_GRAD_FLOOR = 1.0e-8
PROBE_LOGIT_RMS_FLOOR = 1.0e-6
POST_LOGIT_RMS_FLOOR = 1.0e-6
META_GRAD_FLOOR = 1.0e-8
FD_EPSILON = 1.0e-4
FD_ABSOLUTE_TOLERANCE = 1.0e-6
FD_RELATIVE_TOLERANCE = 1.0e-3
FD_RANDOM_SEED = 20260907


def validate_engineering_configuration(config: Mapping[str, Any]) -> None:
    expected = {
        "inner": {"steps": 1, "learning_rate": INNER_LR, "delta_l2_radius": INNER_RADIUS},
        "engineering": {"proxy_gradient_norm_floor": INNER_GRAD_FLOOR,
            "probe_logit_rms_floor": PROBE_LOGIT_RMS_FLOOR, "post_logit_rms_floor": POST_LOGIT_RMS_FLOOR,
            "minimum_informative_episodes": 12, "meta_gradient_norm_floor": META_GRAD_FLOOR,
            "minimum_nonzero_meta_fit_episodes": 6, "require_all_finite": True,
            "require_all_identity_exact": True, "require_all_replays_exact": True, "require_all_fd_passed": True},
        "finite_difference": {"device": "cpu", "dtype": "float64", "epsilon": FD_EPSILON,
            "absolute_tolerance": FD_ABSOLUTE_TOLERANCE, "relative_tolerance": FD_RELATIVE_TOLERANCE,
            "direction_seed": FD_RANDOM_SEED, "directions": ["normalized_meta_gradient", "normalized_seeded_random"],
            "teacher": "fixed_native_float32_cast_double"},
        "task_loss": {"name": "SLSIoULoss", "warm_epoch": 5, "epoch": 999, "with_shape": True, "input": "raw_logits"},
    }
    for key, value in expected.items():
        if json.dumps(config.get(key), sort_keys=True, allow_nan=False) != json.dumps(value, sort_keys=True, allow_nan=False):
            raise ValueError(f"unregistered R4 engineering setting: {key}")


def tensor_sha256(value: Tensor) -> str:
    value = value.detach().cpu().contiguous()
    descriptor = json.dumps({"dtype": str(value.dtype), "shape": list(value.shape)}, sort_keys=True).encode()
    return hashlib.sha256(descriptor + b"\n" + value.numpy().tobytes()).hexdigest()


def module_signature(module: nn.Module) -> dict[str, Any]:
    return {"state": {name: tensor_sha256(value) for name, value in module.state_dict().items()},
            "modes": {name: child.training for name, child in module.named_modules()},
            "requires_grad": {name: parameter.requires_grad for name, parameter in module.named_parameters()},
            "grads": {name: None if parameter.grad is None else tensor_sha256(parameter.grad)
                      for name, parameter in module.named_parameters()}}


def _number(value: Tensor) -> float:
    if value.numel() != 1 or not bool(torch.isfinite(value.detach()).all()):
        raise ValueError("engineering scalar must be finite")
    return float(value.detach().cpu())


def _rms(value: Tensor) -> float:
    return _number(value.detach().double().square().mean().sqrt())


def _finite(values: Sequence[Tensor]) -> bool:
    return all(bool(torch.isfinite(value.detach()).all()) for value in values)


def _validate_native(adapter: IdentityMetaAdapter, head: nn.Module, observed: Tensor,
                     probe: Tensor, teacher: Tensor) -> None:
    if torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled():
        raise ValueError("native engineering checks require autocast disabled")
    for value in (observed, probe, teacher):
        if not isinstance(value, Tensor) or value.dtype != torch.float32:
            raise ValueError("native engineering checks require float32 tensors")
        if value.device != observed.device or value.requires_grad or value.grad_fn is not None:
            raise ValueError("native features/teacher must be detached on the same device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("non-finite native feature or teacher")
    if observed.shape != probe.shape:
        raise ValueError("observed/probe cached grids must match")
    adapter._validate_features(observed)
    adapter._validate_features(probe)
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("native source head must be frozen")
    for value in head.state_dict().values():
        if value.device != observed.device or value.dtype != observed.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError("native source-head precision/device/state mismatch")
    if set(dict(adapter.named_parameters())) != {"down.weight", "up.weight"}:
        raise ValueError("unexpected outer-parameter schema")
    if any(not parameter.requires_grad for parameter in adapter.parameters()):
        raise ValueError("adapter phi must retain the engineering meta graph")


def label_free_episode(
    adapter: IdentityMetaAdapter, head: nn.Module, h_observed: Tensor,
    h_probe: Tensor, teacher: Tensor, *, lr: float = INNER_LR, radius: float = INNER_RADIUS,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Identity, one-step signal and replay checks, without accepting GT."""
    _validate_native(adapter, head, h_observed, h_probe, teacher)
    before_adapter, before_head = module_signature(adapter), module_signature(head)
    input_hashes = [tensor_sha256(value) for value in (h_observed, h_probe, teacher)]
    weights = teacher_region_weights(teacher)
    with torch.no_grad():
        source_logits = head(h_observed)
        probe_logits = head(h_probe)
        if source_logits.shape != teacher.shape:
            raise ValueError("teacher/source logit grid mismatch")
        teacher_matches = torch.equal(source_logits.sigmoid(), teacher)
        if not teacher_matches:
            raise ValueError("teacher must be the fixed native observed-source probabilities")
        delta0 = h_observed.new_zeros(adapter.num_bases)
        identity_features = adapter(h_observed, delta0)
        identity_logits = head(identity_features)
    with torch.enable_grad():
        delta_true, proxy_true, gradient_true = inner_step(
            adapter, head, h_probe, teacher, weights,
            learning_rate=lr, radius=radius, create_graph=True)
        delta_false, proxy_false, gradient_false = inner_step(
            adapter, head, h_probe, teacher, weights,
            learning_rate=lr, radius=radius, create_graph=False)
        delta_repeat, proxy_repeat, gradient_repeat = inner_step(
            adapter, head, h_probe, teacher, weights,
            learning_rate=lr, radius=radius, create_graph=False)
    with torch.no_grad():
        adapted = adapter(h_observed, delta_false.detach())
        post_logits = head(adapted)
        repeat_features = adapter(h_observed, delta_repeat.detach())
        repeat_logits = head(repeat_features)
    all_values = [source_logits, probe_logits, identity_features, identity_logits, adapted, post_logits,
                  delta_true, proxy_true, gradient_true, delta_false, proxy_false, gradient_false,
                  delta_repeat, proxy_repeat, gradient_repeat]
    if any(value.dtype != torch.float32 for value in all_values):
        raise RuntimeError("native engineering computation left the fixed float32 path")
    if not _finite(all_values):
        raise RuntimeError("non-finite label-free engineering result")
    unchanged = before_adapter == module_signature(adapter) and before_head == module_signature(head)
    if not unchanged or input_hashes != [tensor_sha256(value) for value in (h_observed, h_probe, teacher)]:
        raise RuntimeError("label-free diagnostic mutated frozen state or its inputs")
    grad_norm = _number(gradient_false.double().norm())
    delta_norm = _number(delta_false.double().norm())
    proposed_norm = _number((float(lr) * gradient_false.detach()).norm())
    host_rms = _rms(h_observed)
    residual_rms = _rms(adapted - h_observed)
    probe_logit_rms = _rms(probe_logits - source_logits)
    post_logit_rms = _rms(post_logits - source_logits)
    numeric_exact = all(torch.equal(first.detach(), second.detach()) for first, second in (
        (delta_true, delta_false), (proxy_true, proxy_false), (gradient_true, gradient_false)))
    repeat_exact = all(torch.equal(first.detach(), second.detach()) for first, second in (
        (delta_false, delta_repeat), (proxy_false, proxy_repeat), (gradient_false, gradient_repeat),
        (adapted, repeat_features), (post_logits, repeat_logits)))
    identity_exact = tensor_sha256(h_observed) == tensor_sha256(identity_features) and tensor_sha256(source_logits) == tensor_sha256(identity_logits)
    receipt = {
        "role": "R4_label_free_cached_feature_engineering", "native_dtype": "float32",
        "native_device": str(h_observed.device), "finite": True,
        "identity_exact": identity_exact, "teacher_matches_observed_exact": teacher_matches,
        "create_graph_numeric_exact": numeric_exact, "repeat_exact": repeat_exact,
        "state_unchanged": unchanged,
        "no_persisted_gradients": all(parameter.grad is None for module in (adapter, head) for parameter in module.parameters()),
        "observed_feature_sha256": input_hashes[0], "probe_feature_sha256": input_hashes[1],
        "teacher_sha256": input_hashes[2], "source_logits_sha256": tensor_sha256(source_logits),
        "identity_logits_sha256": tensor_sha256(identity_logits), "post_logits_sha256": tensor_sha256(post_logits),
        "learning_rate": float(lr), "radius": float(radius), "proxy_loss": _number(proxy_false),
        "inner_gradient_norm": grad_norm, "delta_norm": delta_norm,
        "proposal_norm": proposed_norm, "projection_active": proposed_norm > radius,
        "delta_within_radius": delta_norm <= radius * (1.0 + 1.0e-6) + 1.0e-12,
        "probe_observed_logit_rms": probe_logit_rms,
        "probe_observed_probability_rms": _rms(probe_logits.sigmoid() - source_logits.sigmoid()),
        "post_observed_logit_rms": post_logit_rms,
        "feature_residual_rms": residual_rms, "host_feature_rms": host_rms,
        "host_feature_rms_floor": 1.0e-6,
        "host_feature_rms_floor_active": bool(h_observed.detach().square().mean() <= adapter.rms_squared_floor),
        "residual_over_floored_host_rms": residual_rms / max(host_rms, 1.0e-6),
        "label_free_signal_passed": grad_norm > INNER_GRAD_FLOOR and probe_logit_rms > PROBE_LOGIT_RMS_FLOOR and post_logit_rms > POST_LOGIT_RMS_FLOOR,
        "gt_accessed": False, "optimizer_steps_applied": 0,
        "ipma_meta_training_executed": False, "full_source_training_allowed": False,
        "formal_test_allowed": False, "paper_result": False,
    }
    tensors = {name: value.detach().clone() for name, value in (
        ("source_logits", source_logits), ("probe_logits", probe_logits),
        ("identity_logits", identity_logits), ("post_logits", post_logits),
        ("delta0", delta0), ("delta1", delta_false), ("inner_gradient", gradient_false))}
    return receipt, tensors


def _flatten(values: Sequence[Tensor]) -> Tensor:
    return torch.cat([value.reshape(-1) for value in values])


def _task_after_inner(adapter: IdentityMetaAdapter, head: nn.Module, observed: Tensor,
                      probe: Tensor, teacher: Tensor, target: Tensor, *,
                      lr: float, radius: float, create_graph: bool) -> tuple[Tensor, Tensor, Tensor]:
    delta, proxy, gradient = inner_step(adapter, head, probe, teacher, teacher_region_weights(teacher),
        learning_rate=lr, radius=radius, create_graph=create_graph)
    logits = head(adapter(observed, delta))
    loss = SLSIoULoss()(logits, target, 5, 999)
    return loss, logits, delta


def finite_difference_replay(
    adapter: IdentityMetaAdapter, head: nn.Module, observed: Tensor, probe: Tensor,
    teacher: Tensor, target: Tensor, *, lr: float = INNER_LR, radius: float = INNER_RADIUS,
    epsilon: float = FD_EPSILON, abs_tolerance: float = FD_ABSOLUTE_TOLERANCE,
    rel_tolerance: float = FD_RELATIVE_TOLERANCE, random_seed: int = FD_RANDOM_SEED,
) -> dict[str, Any]:
    """Two-direction exact-meta versus central-FD check on a CPU float64 copy."""
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0
           for value in (lr, radius, epsilon, abs_tolerance, rel_tolerance)):
        raise ValueError("FD scales and tolerances must be positive finite real values")
    replay_adapter = copy.deepcopy(adapter).cpu().double()
    replay_head = copy.deepcopy(head).cpu().double()
    for parameter in replay_head.parameters():
        parameter.requires_grad_(False)
    features = [value.detach().cpu().double().clone() for value in (observed, probe, teacher, target)]
    obs64, probe64, teacher64, target64 = features
    # Crucially teacher64 is a fixed conversion of the native teacher. It is
    # neither recomputed with the double head nor differentiated through phi.
    parameters = list(replay_adapter.parameters())
    initial = [parameter.detach().clone() for parameter in parameters]
    with torch.enable_grad():
        reference, _, delta = _task_after_inner(replay_adapter, replay_head, obs64, probe64, teacher64, target64,
            lr=lr, radius=radius, create_graph=True)
        gradients = torch.autograd.grad(reference, parameters, allow_unused=False)
    if not _finite([reference, delta, *gradients]):
        raise RuntimeError("non-finite double replay task or meta-gradient")
    flattened = _flatten(gradients).detach()
    gradient_norm = _number(flattened.norm())
    generator = torch.Generator(device="cpu").manual_seed(random_seed)
    random = torch.randn(flattened.shape, dtype=torch.float64, generator=generator)
    random = random / random.norm()
    directions = [("normalized_meta_gradient", flattened / gradient_norm if gradient_norm > 0 else None),
                  ("normalized_seeded_random", random)]
    results = []
    def displaced_loss(direction: Tensor, amount: float) -> Tensor:
        offset = 0
        with torch.no_grad():
            for parameter, original in zip(parameters, initial, strict=True):
                size = parameter.numel()
                parameter.copy_(original + amount * direction[offset:offset + size].view_as(parameter))
                offset += size
        with torch.enable_grad():
            loss, _, _ = _task_after_inner(replay_adapter, replay_head, obs64, probe64, teacher64, target64,
                lr=lr, radius=radius, create_graph=False)
        return loss.detach()
    try:
        for name, direction in directions:
            if direction is None:
                results.append({"direction": name, "defined": False, "finite": True,
                                "passed": False, "reason": "zero_meta_gradient_has_no_normalized_direction"})
                continue
            plus = _number(displaced_loss(direction, epsilon))
            minus = _number(displaced_loss(direction, -epsilon))
            finite_difference = (plus - minus) / (2.0 * epsilon)
            automatic = _number((flattened * direction).sum())
            error = abs(finite_difference - automatic)
            tolerance = abs_tolerance + rel_tolerance * max(abs(finite_difference), abs(automatic))
            results.append({"direction": name, "defined": True, "finite": True,
                "direction_norm": _number(direction.norm()), "plus_task_loss": plus, "minus_task_loss": minus,
                "autograd_directional_derivative": automatic, "central_finite_difference": finite_difference,
                "absolute_error": error, "tolerance": tolerance, "passed": error <= tolerance})
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, initial, strict=True):
                parameter.copy_(original)
    return {"device": "cpu", "dtype": "float64", "native_outputs_equivalence_claimed": False,
            "teacher_policy": "fixed_native_teacher_converted_once_never_recomputed",
            "teacher_sha256": tensor_sha256(teacher64), "task_loss": _number(reference),
            "meta_gradient_norm": gradient_norm, "epsilon": epsilon,
            "absolute_tolerance": abs_tolerance, "relative_tolerance": rel_tolerance,
            "random_direction_seed": random_seed, "direction_count": len(results),
            "directions": results, "all_directions_passed": all(row["passed"] for row in results),
            "finite": all(row["finite"] for row in results),
            "double_delta_norm": _number(delta.detach().norm()),
            "sfs_backbone_in_replay_graph": False, "optimizer_steps_applied": 0}


def supervised_fit_diagnostic(
    adapter: IdentityMetaAdapter, head: nn.Module, h_observed: Tensor,
    h_probe: Tensor, teacher: Tensor, target: Tensor, config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Source-fit-only true meta gradient; no phi update and no sample selection."""
    validate_engineering_configuration(config)
    _validate_native(adapter, head, h_observed, h_probe, teacher)
    if target.shape != teacher.shape or target.dtype != torch.float32 or target.device != teacher.device or target.requires_grad:
        raise ValueError("fit-only GT must be detached float32 on the native grid/device")
    if not bool(torch.isfinite(target).all()) or bool(((target < 0) | (target > 1)).any()):
        raise ValueError("fit-only target is non-finite or outside [0,1]")
    lr = float(config["inner"]["learning_rate"])
    radius = float(config["inner"]["delta_l2_radius"])
    before_adapter, before_head = module_signature(adapter), module_signature(head)
    inputs_before = [tensor_sha256(value) for value in (h_observed, h_probe, teacher, target)]
    with torch.no_grad():
        pre_logits = head(h_observed)
        if not torch.equal(pre_logits.sigmoid(), teacher):
            raise ValueError("fit diagnostic teacher must remain the native observed-source teacher")
        pre_loss = SLSIoULoss()(pre_logits, target, 5, 999)
    with torch.enable_grad():
        post_loss, post_logits, delta = _task_after_inner(adapter, head, h_observed, h_probe, teacher, target,
            lr=lr, radius=radius, create_graph=True)
        named = list(adapter.named_parameters())
        gradients = torch.autograd.grad(post_loss, [parameter for _, parameter in named], allow_unused=False)
    if not _finite([pre_loss, post_loss, post_logits, delta, *gradients]):
        raise RuntimeError("non-finite fit-only native meta-gradient diagnostic")
    grad_norm = _number(_flatten(gradients).double().norm())
    parameter_norm = _number(_flatten([parameter.detach() for _, parameter in named]).double().norm())
    fd = finite_difference_replay(adapter, head, h_observed, h_probe, teacher, target, lr=lr, radius=radius,
        epsilon=float(config["finite_difference"]["epsilon"]),
        abs_tolerance=float(config["finite_difference"]["absolute_tolerance"]),
        rel_tolerance=float(config["finite_difference"]["relative_tolerance"]),
        random_seed=int(config["finite_difference"]["direction_seed"]))
    unchanged = before_adapter == module_signature(adapter) and before_head == module_signature(head)
    if not unchanged or inputs_before != [tensor_sha256(value) for value in (h_observed, h_probe, teacher, target)]:
        raise RuntimeError("fit-only engineering diagnostic changed native state or inputs")
    receipt = {
        "role": "R4_source_fit_only_true_meta_gradient_diagnostic", "native_dtype": "float32",
        "native_device": str(h_observed.device), "fit_only": True,
        "target_empty": not bool((target > 0).any()), "target_sha256": inputs_before[3],
        "finite": True, "state_unchanged": unchanged,
        "no_persisted_gradients": all(parameter.grad is None for module in (adapter, head) for parameter in module.parameters()),
        "pre_sls": _number(pre_loss), "post_sls": _number(post_loss),
        "post_minus_pre_sls": _number(post_loss - pre_loss),
        "sls_input": "raw_logits_internal_sigmoid_exact_original_SLSIoULoss",
        "sls_warm_epoch": 5, "sls_epoch_index": 999,
        "meta_gradient_norm": grad_norm, "phi_parameter_norm": parameter_norm,
        "meta_gradient_to_parameter_norm": grad_norm / parameter_norm if parameter_norm > 0 else None,
        "parameter_meta_gradient_norms": {name: _number(gradient.double().norm()) for (name, _), gradient in zip(named, gradients, strict=True)},
        "delta_norm": _number(delta.detach().double().norm()), "finite_difference": fd,
        "meta_gradient_above_floor": grad_norm > META_GRAD_FLOOR,
        "optimizer_steps_applied": 0, "ipma_meta_training_executed": False,
        "full_source_training_allowed": False, "formal_test_allowed": False, "paper_result": False,
        "task_improvement_required_for_engineering_pass": False,
    }
    tensors = {"pre_logits": pre_logits.detach().clone(), "post_logits": post_logits.detach().clone(),
               "delta1": delta.detach().clone(), "meta_gradient_vector": _flatten(gradients).detach().clone()}
    return receipt, tensors


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def aggregate_engineering_gate(label_free: Sequence[Mapping[str, Any]],
                               fit: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A pure, fail-closed R4 engineering gate; no downstream work is executed."""
    reasons = []
    if len(label_free) != 16:
        reasons.append("requires_all_16_label_free_cases")
    if len(fit) != 8:
        reasons.append("requires_all_first_8_fit_diagnostics")
    if not _finite_tree(label_free) or not _finite_tree(fit):
        reasons.append("nonfinite_receipt_value")
    technical = ("finite", "identity_exact", "teacher_matches_observed_exact", "create_graph_numeric_exact",
                 "repeat_exact", "state_unchanged", "no_persisted_gradients", "delta_within_radius")
    def number(row: Mapping[str, Any], key: str) -> bool:
        value = row.get(key)
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    native_numbers = ("proxy_loss", "inner_gradient_norm", "delta_norm", "proposal_norm",
        "probe_observed_logit_rms", "probe_observed_probability_rms", "post_observed_logit_rms",
        "feature_residual_rms", "host_feature_rms", "residual_over_floored_host_rms")
    def technically_complete(row: Mapping[str, Any]) -> bool:
        return (all(row.get(key) is True for key in technical)
                and all(number(row, key) and row[key] >= 0 for key in native_numbers)
                and row.get("learning_rate") == INNER_LR and row.get("radius") == INNER_RADIUS
                and row.get("native_dtype") == "float32" and row.get("gt_accessed") is False
                and row["delta_norm"] <= INNER_RADIUS * (1 + 1e-6) + 1e-12)
    technical_count = sum(technically_complete(row) for row in label_free)
    if technical_count != 16:
        reasons.append("label_free_identity_replay_finite_or_state_check_failed")
    def above(row: Mapping[str, Any], key: str, floor: float) -> bool:
        value = row.get(key)
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > floor
    informative = sum(above(row, "inner_gradient_norm", INNER_GRAD_FLOOR)
                      and above(row, "probe_observed_logit_rms", PROBE_LOGIT_RMS_FLOOR)
                      and above(row, "post_observed_logit_rms", POST_LOGIT_RMS_FLOOR) for row in label_free)
    if informative < 12:
        reasons.append("fewer_than_12_of_16_informative_label_free_cases")
    fit_finite = sum(all(row.get(key) is True for key in ("finite", "state_unchanged", "no_persisted_gradients", "fit_only"))
                     and all(number(row, key) for key in ("pre_sls", "post_sls", "post_minus_pre_sls", "meta_gradient_norm", "phi_parameter_norm"))
                     and row.get("native_dtype") == "float32"
                     and row.get("sls_input") == "raw_logits_internal_sigmoid_exact_original_SLSIoULoss"
                     for row in fit)
    if fit_finite != 8:
        reasons.append("fit_finite_state_or_scope_check_failed")
    def fd_passed(row: Mapping[str, Any]) -> bool:
        fd = row.get("finite_difference")
        if not isinstance(fd, Mapping) or fd.get("finite") is not True or fd.get("all_directions_passed") is not True:
            return False
        fixed = {"device": "cpu", "dtype": "float64", "epsilon": FD_EPSILON,
                 "absolute_tolerance": FD_ABSOLUTE_TOLERANCE, "relative_tolerance": FD_RELATIVE_TOLERANCE,
                 "random_direction_seed": FD_RANDOM_SEED, "direction_count": 2,
                 "teacher_policy": "fixed_native_teacher_converted_once_never_recomputed",
                 "native_outputs_equivalence_claimed": False}
        if any(fd.get(key) != value for key, value in fixed.items()):
            return False
        directions = fd.get("directions", [])
        if not isinstance(directions, list) or len(directions) != 2 or any(not isinstance(d, Mapping) for d in directions):
            return False
        if {d.get("direction") for d in directions} != {"normalized_meta_gradient", "normalized_seeded_random"}:
            return False
        for direction in directions:
            if not all(direction.get(key) is True for key in ("defined", "finite", "passed")):
                return False
            if not all(number(direction, key) for key in ("direction_norm", "plus_task_loss", "minus_task_loss",
                "autograd_directional_derivative", "central_finite_difference", "absolute_error", "tolerance")):
                return False
            if not math.isclose(direction["direction_norm"], 1.0, abs_tol=1e-10, rel_tol=1e-10):
                return False
            central = (direction["plus_task_loss"] - direction["minus_task_loss"]) / (2 * FD_EPSILON)
            automatic = direction["autograd_directional_derivative"]
            error = abs(central - automatic)
            tolerance = FD_ABSOLUTE_TOLERANCE + FD_RELATIVE_TOLERANCE * max(abs(central), abs(automatic))
            if not all(math.isclose(direction[key], expected, rel_tol=1e-10, abs_tol=1e-12)
                       for key, expected in (("central_finite_difference", central), ("absolute_error", error), ("tolerance", tolerance))):
                return False
            if error > tolerance:
                return False
        return True
    all_fd = len(fit) == 8 and all(fd_passed(row) for row in fit)
    if not all_fd:
        reasons.append("every_fit_finite_difference_direction_must_pass")
    meta_count = sum(above(row, "meta_gradient_norm", META_GRAD_FLOOR) and fd_passed(row) for row in fit)
    if meta_count < 6:
        reasons.append("fewer_than_6_of_8_fit_meta_gradients_above_floor_with_fd_pass")
    for rows in (label_free, fit):
        if any(row.get("optimizer_steps_applied") != 0 or row.get("ipma_meta_training_executed") is not False for row in rows):
            reasons.append("unexpected_training_update_or_missing_no_update_evidence")
    passed = not reasons
    return {"engineering_passed": passed, "failure_reasons": reasons,
        "label_free_case_count": len(label_free), "technical_case_count": technical_count,
        "informative_label_free_case_count": informative, "required_informative_cases": 12,
        "fit_case_count": len(fit), "fit_meta_gradient_qualified_count": meta_count,
        "required_fit_meta_gradient_cases": 6, "all_fit_finite_differences_passed": all_fd,
        "bounded_meta_training_engineering_eligible": passed, "maximum_future_outer_steps": 64,
        "ipma_meta_training_allowed": False, "requires_frozen_r5_contract": True,
        "current_outer_optimizer_steps": 0,
        "ipma_meta_training_executed": False, "execution_started": False,
        "full_source_training_allowed": False, "formal_test_allowed": False, "paper_result": False,
        "task_improvement_used_for_gate": False,
        "authorization_note": "engineering eligibility only; no optimizer or next stage is launched by these checks"}
