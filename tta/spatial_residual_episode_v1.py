"""One label-free O3 spatial-residual episode on detached decoder features.

The frozen Stage-B4 O3 objective, region construction, diagnostics, Armijo
search and three safety budgets are reused verbatim. Only the parameter space
changes: a zero-initialized spatial residual with an explicit absolute radius.
There is no target/ID/path argument and no optimizer or cross-image state.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
from torch import Tensor, nn
import yaml

from tta.adapters.decoder_spatial_residual_v1 import DecoderSpatialResidual
from scripts import run_p3_stage_b4_full_pilot64_v1 as b4
from scripts import run_p3_stage_b_screen_v1 as b3
from tta.proposal_runner_v1 import propose_and_backtrack


class SpatialEpisodeError(RuntimeError):
    """A frozen contract, input, or episodic restoration check failed."""


def _validate_contract(contract: b4.FullPilotContract) -> None:
    """Check only immutable metadata; never open checkpoints or image caches."""

    if not isinstance(contract, b4.FullPilotContract):
        raise SpatialEpisodeError("contract must be the original FullPilotContract")
    expected = b4.REPOSITORY / "configs/p3_stage_b4_full_pilot64_proposal_gate_v1.yaml"
    if contract.repository != b4.REPOSITORY or contract.config_path != expected:
        raise SpatialEpisodeError("original Stage-B4 contract location differs")
    if expected.is_symlink() or not expected.is_file():
        raise SpatialEpisodeError("original Stage-B4 contract is not a regular file")
    payload = expected.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != b4.FROZEN_CONFIG_SHA256 or contract.config_sha256 != digest:
        raise SpatialEpisodeError("original Stage-B4 config bytes differ")
    if contract.raw != yaml.safe_load(payload.decode("utf-8")):
        raise SpatialEpisodeError("original Stage-B4 contract values differ")
    binding = contract.raw["frozen_parent_bindings"]["view_library"]
    view_path = contract.repository / binding["path"]
    if (
        view_path.is_symlink()
        or not view_path.is_file()
        or hashlib.sha256(view_path.read_bytes()).hexdigest() != binding["sha256"]
        or contract.raw["view_library"]["config_path"] != binding["path"]
    ):
        raise SpatialEpisodeError("original region-weight configuration differs")


def _validate_tensor(
    value: Tensor, name: str, *, shape: tuple[int, ...], device: torch.device
) -> None:
    if (
        not isinstance(value, Tensor)
        or value.layout != torch.strided
        or value.dtype != torch.float32
        or value.device != device
        or tuple(value.shape) != shape
        or value.requires_grad
        or value.grad_fn is not None
        or not bool(torch.isfinite(value).all().item())
    ):
        raise SpatialEpisodeError(
            f"{name} must be finite detached float32 with shape {shape} on {device}"
        )


def _snapshot_head(head: nn.Module) -> dict[str, Any]:
    return {
        "state": {key: value.detach().clone() for key, value in head.state_dict().items()},
        "modes": tuple((name, child.training) for name, child in head.named_modules()),
        "requires_grad": tuple((name, value.requires_grad) for name, value in head.named_parameters()),
        "grads": {
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in head.named_parameters()
        },
    }


def _assert_head_unchanged(head: nn.Module, snapshot: dict[str, Any]) -> None:
    after = _snapshot_head(head)
    if after["modes"] != snapshot["modes"] or after["requires_grad"] != snapshot["requires_grad"]:
        raise SpatialEpisodeError("frozen head modes or gradient policy changed")
    for field in ("state", "grads"):
        if after[field].keys() != snapshot[field].keys():
            raise SpatialEpisodeError("frozen head topology changed")
        for name, value in snapshot[field].items():
            current = after[field][name]
            if value is None:
                equal = current is None
            else:
                equal = (
                    current is not None
                    and current.dtype == value.dtype
                    and current.device == value.device
                    and current.shape == value.shape
                    and torch.equal(
                        current.contiguous().reshape(-1).view(torch.uint8),
                        value.contiguous().reshape(-1).view(torch.uint8),
                    )
                )
            if not equal:
                raise SpatialEpisodeError(f"frozen head {field} changed: {name}")


def _safety_decision(observed: dict[str, Any], proposal: dict[str, Any]) -> tuple[bool, str]:
    """The three comparisons are exactly the original B4 safety comparisons."""

    config = proposal["per_attempt_label_free_safety"]
    epsilon = float(proposal["epsilon"])
    failures: list[str] = []
    if (
        observed["post_reliable_background_mass"] - observed["source_reliable_background_mass"]
        > float(config["reliable_background_mass_delta_maximum"]) + epsilon
    ):
        failures.append("reliable_background_mass")
    if (
        observed["post_foreground_fraction"] - observed["source_foreground_fraction"]
        > float(config["predicted_positive_fraction_delta_maximum"]) + epsilon
    ):
        failures.append("predicted_positive_fraction")
    if observed["post_connected_component_count"] > observed["connected_component_count_maximum"]:
        failures.append("connected_component_count")
    return not failures, "passed" if not failures else "+".join(failures)


def run_episode(
    *,
    contract: b4.FullPilotContract,
    module: DecoderSpatialResidual,
    head: nn.Conv2d,
    observed_features: Tensor,
    student_features: Tensor,
    teacher: Tensor,
    uncertainty: Tensor,
    source_logits: Tensor,
    absolute_radius: float = 0.25,
) -> dict[str, Any]:
    """Propose one O3 update, snapshot its output, then reset even on errors.

    The caller extracts both 256-square decoder feature maps with a frozen
    host. The student is the unchanged B4 mild-contrast view; its provenance is
    bound by the caller's image/cache contract, never inferred from features.
    """

    if not isinstance(module, DecoderSpatialResidual):
        raise SpatialEpisodeError("module must be DecoderSpatialResidual")
    head_snapshot = None
    output: dict[str, Any] | None = None
    input_snapshots: tuple[tuple[Tensor, Tensor], ...] = ()
    try:
        head_snapshot = _snapshot_head(head) if isinstance(head, nn.Module) else None
        _validate_contract(contract)
        if isinstance(absolute_radius, bool) or absolute_radius != 0.25:
            raise SpatialEpisodeError("absolute radius must remain the frozen 0.25")
        named = tuple(module.named_parameters())
        if (
            len(named) != 1
            or named[0][0] != "kernel"
            or tuple(named[0][1].shape) != (16, 1, 3, 3)
            or named[0][1].dtype != torch.float32
            or not named[0][1].requires_grad
            or named[0][1].grad is not None
            or bool(torch.count_nonzero(named[0][1].detach()).item())
        ):
            raise SpatialEpisodeError("spatial module must start with fresh zero kernel and empty grad slots")
        if module.channels != 16 or module.max_residual_ratio != 0.05 or module.rms_floor != 1e-6:
            raise SpatialEpisodeError("spatial residual bounds differ from the frozen design")
        device = named[0][1].device
        if (
            type(head) is not nn.Conv2d
            or head.in_channels != 16
            or head.out_channels != 1
            or head.kernel_size != (1, 1)
            or head.stride != (1, 1)
            or head.padding != (0, 0)
            or head.groups != 1
            or any(child.training for child in head.modules())
            or any(value.requires_grad or value.grad is not None for value in head.parameters())
            or any(value.dtype != torch.float32 or value.device != device or not bool(torch.isfinite(value).all().item()) for value in head.state_dict().values())
        ):
            raise SpatialEpisodeError("output head must be the finite frozen eval float32 16-to-1 pointwise head")
        for name, value in (("observed_features", observed_features), ("student_features", student_features)):
            _validate_tensor(value, name, shape=(1, 16, 256, 256), device=device)
        for name, value in (("teacher", teacher), ("uncertainty", uncertainty), ("source_logits", source_logits)):
            _validate_tensor(value, name, shape=(1, 1, 256, 256), device=device)
        if bool(((teacher < 0) | (teacher > 1)).any().item()) or bool((uncertainty < 0).any().item()):
            raise SpatialEpisodeError("teacher or uncertainty range is invalid")
        input_snapshots = tuple((value, value.detach().clone()) for value in (observed_features, student_features, teacher, uncertainty, source_logits))
        with torch.no_grad():
            if not torch.equal(head(observed_features), source_logits):
                raise SpatialEpisodeError("cached head/features do not exactly reproduce source logits")
            source_probability = torch.sigmoid(source_logits)
            if not torch.equal(source_probability, teacher):
                raise SpatialEpisodeError("teacher is not bit-exact native Source sigmoid")
            if not torch.equal(module(observed_features), observed_features):
                raise SpatialEpisodeError("zero-kernel feature identity failed")
            if not torch.equal(head(module(observed_features)), source_logits):
                raise SpatialEpisodeError("zero-kernel output identity failed")

        region = b3._build_region_weights(contract, teacher, uncertainty)
        if float(region.background_weight.sum().item()) <= 0:
            raise SpatialEpisodeError("reliable-background region is empty")

        def loss_closure() -> Tensor:
            loss, _ = b3._objective_loss(contract, "O3", head(module(student_features)), teacher, region)
            return loss

        def diagnostics_for(logits: Tensor) -> dict[str, Any]:
            return b4._episode_diagnostics(
                contract=contract, source_logits=source_logits, post_logits=logits,
                teacher=teacher, foreground_weight=region.foreground_weight,
                background_weight=region.background_weight,
            )

        def safety_closure() -> tuple[bool, str]:
            with torch.no_grad():
                logits = head(module(observed_features))
                if not bool(torch.isfinite(logits).all().item()):
                    return False, "nonfinite_logits"
                if not bool(torch.isfinite(torch.sigmoid(logits)).all().item()):
                    return False, "nonfinite_probabilities"
                return _safety_decision(diagnostics_for(logits), contract.raw["proposal"])

        with torch.no_grad():
            _, terms_before = b3._objective_loss(contract, "O3", head(module(student_features)), teacher, region)
        proposal = contract.raw["proposal"]
        execution = propose_and_backtrack(
            named_parameters=named, loss_closure=loss_closure, safety_closure=safety_closure,
            relative_radius=None, absolute_radius=absolute_radius,
            coefficients=tuple(proposal["backtracking_coefficients"]),
            armijo_c=float(proposal["armijo_c"]), epsilon=float(proposal["epsilon"]),
            allow_cuda_nondeterministic_backward=False,
        )
        result = execution.result
        with torch.no_grad():
            post_features = module(observed_features)
            post_logits = head(post_features)
            post_probability = torch.sigmoid(post_logits)
            _, terms_after = b3._objective_loss(contract, "O3", head(module(student_features)), teacher, region)
            diagnostics = diagnostics_for(post_logits)
            residual_rms = (post_features - observed_features).square().mean().sqrt()
            host_rms = observed_features.square().mean().sqrt().clamp_min(module.rms_floor)
            residual_ratio = float((residual_rms / host_rms).item())
        endpoint_equal = b4._check_no_update_endpoint(
            source_probability=source_probability, post_probability=post_probability,
            accepted_update=bool(result.accepted),
        )
        gradient, direction = execution.proxy_gradient, execution.normalized_direction
        if any(vector.dtype != torch.float64 or vector.device.type != "cpu" or vector.shape != (144,) or not bool(torch.isfinite(vector).all().item()) for vector in (gradient, direction)):
            raise SpatialEpisodeError("audited proposal vector topology differs")
        if not math.isfinite(residual_ratio) or residual_ratio > module.max_residual_ratio + 1e-6:
            raise SpatialEpisodeError("observed feature residual exceeds its frozen bound")
        finite = (
            math.isfinite(float(result.loss_before)) and math.isfinite(float(result.loss_after))
            and not str(result.reason).startswith("nonfinite")
            and bool(torch.isfinite(post_probability).all().item())
        )
        output = {
            "post_probabilities": post_probability.detach().cpu().to(torch.float32).clone(),
            "source_probabilities": source_probability.detach().cpu().to(torch.float32).clone(),
            "proxy_gradient": gradient.clone(), "direction": direction.clone(),
            "endpoint_kernel": module.kernel.detach().cpu().to(torch.float32).clone(),
            "diagnostics": {
                "candidate_id": "O3_DecoderSpatialResidual", "objective": "O3",
                "parameter_space": "DecoderSpatialResidual", "teacher_candidate_id": "sealed_source_identity",
                "accepted_update": bool(result.accepted), "no_update": not bool(result.accepted),
                "finite": finite, "proposal_step": b4._serialise_step_result(result),
                "proposal_loss_before": float(result.loss_before) if finite else 0.0,
                "proposal_loss_after": float(result.loss_after) if finite else 0.0,
                "proposal_loss_strict_decrease": bool(result.loss_after < result.loss_before),
                "loss_terms_before": terms_before, "loss_terms_after": terms_after,
                "foreground_weight_sum": float(region.foreground_weight.sum().item()),
                "background_weight_sum": float(region.background_weight.sum().item()),
                "source_post_probability_bit_exact": endpoint_equal,
                "zero_kernel_identity_exact": True,
                "residual_over_floored_host_rms": residual_ratio,
                "absolute_radius": absolute_radius, "relative_radius": None,
                "parameter_scalar_count": 144, "method_label_accesses": 0,
                "optimizer_objects_created": 0, "optimizer_steps": 0,
                "candidate_proposals": 1, "cross_image_learning": False,
                "cached_features_detached": True, "sfs_backward_called": False,
                "cuda_nondeterministic_backward_override": False,
                **diagnostics,
            },
        }
        return output
    finally:
        module.reset_identity_()
        if any(bool(torch.count_nonzero(value.detach()).item()) or value.grad is not None for value in module.parameters()):
            raise SpatialEpisodeError("spatial module failed to reset exact zero state")
        if head_snapshot is not None:
            _assert_head_unchanged(head, head_snapshot)
        for value, before in input_snapshots:
            if not torch.equal(value, before):
                raise SpatialEpisodeError("detached episode input was mutated")
        if output is not None:
            output["diagnostics"]["episode_reset_exact"] = True
            output["diagnostics"]["head_state_unchanged"] = True
            output["diagnostics"]["cached_inputs_unchanged"] = True


__all__ = ["SpatialEpisodeError", "run_episode"]
