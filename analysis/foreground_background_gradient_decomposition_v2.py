"""Numerically closed Stage-B1 foreground/background gradient decomposition.

Version 2 preserves the region definitions and scientific metrics of v1 while
changing only how the three reported region gradients are evaluated.  Five
VJPs are taken from one shared Source graph:

``G_full, G_FG, G_FGsub, raw_G_FGsup, raw_G_BG``.

The scientific basis is the cumulative telescoping basis, evaluated on CPU in
float64::

    g_FGsub = G_FGsub
    g_FGsup = G_FG - G_FGsub
    g_BG    = G_full - G_FG

Consequently ``full_add`` is always the directly evaluated ``G_full`` rather
than the sum of three independent CUDA reductions.  The two independent raw
region VJPs are audit-only: their residuals against the derived vectors are
reported, but they never enter scientific metrics or returned basis vectors.

Like v1, this module performs no model forward, file access, optimizer
construction, parameter update, or ``Tensor.backward`` call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any, Final

import torch
from torch import Tensor

from analysis import foreground_background_gradient_decomposition_v1 as _v1
from analysis.d0_v3_outer_analyzer import FlatParameterLayout
from analysis.foreground_background_gradient_decomposition_v1 import (
    BACKGROUND_RULE,
    FOREGROUND_RULE,
    GROUP_IDS,
    SUBTHRESHOLD_RULE,
    SUPRATHRESHOLD_RULE,
    ForegroundBackgroundGradientError,
    GradientDecompositionConfig,
)
from tta.binary_tent import binary_entropy_map


SCHEMA_VERSION: Final = 2
ANALYSIS_TYPE: Final = "cr_sitta_stage_b1_fg_bg_gradient_decomposition_v2"

# Actual calls to torch.autograd.grad, in execution order.  The last two are
# deliberately retained only to quantify backend reduction non-closure.
BACKWARD_BASIS: Final = (
    "full_entropy_mean_direct",
    "foreground_entropy_add_direct",
    "foreground_subthreshold_entropy_add_direct",
    "foreground_suprathreshold_entropy_add_raw_audit",
    "background_entropy_add_raw_audit",
)
SCIENTIFIC_BASIS: Final = (
    "foreground_subthreshold_add",
    "foreground_suprathreshold_add",
    "background_add",
)
CUMULATIVE_ENDPOINTS: Final = (
    "foreground_subthreshold_add_direct",
    "foreground_add_direct",
    "full_add_direct",
)
AUDIT_ONLY_RAW_VJPS: Final = (
    "foreground_suprathreshold_add_raw",
    "background_add_raw",
)
DIRECT_VECTOR_FIELDS: Final = (
    "full_direct",
    "foreground_total_direct",
    "foreground_subthreshold_direct",
    "foreground_suprathreshold_raw_direct",
    "background_raw_direct",
)


@dataclass(frozen=True)
class GradientDecompositionVectors:
    """Detached CPU float64 direct, raw-audit, and scientific vectors.

    The first eleven fields preserve the v1 public scientific-vector API.
    The last five expose the exact direct VJPs in ``BACKWARD_BASIS`` order for
    runner-side storage and independent artifact verification.
    """

    foreground_subthreshold_add: Tensor
    foreground_suprathreshold_add: Tensor
    background_add: Tensor
    foreground_add: Tensor
    full_add: Tensor
    foreground_conditional_mean: Tensor | None
    background_conditional_mean: Tensor | None
    foreground_subthreshold_conditional_mean: Tensor | None
    foreground_suprathreshold_conditional_mean: Tensor | None
    parent_entropy: Tensor
    task: Tensor
    full_direct: Tensor
    foreground_total_direct: Tensor
    foreground_subthreshold_direct: Tensor
    foreground_suprathreshold_raw_direct: Tensor
    background_raw_direct: Tensor


@dataclass(frozen=True)
class ForegroundBackgroundGradientResult:
    """JSON-safe report and detached v2 vectors."""

    report: Mapping[str, Any]
    vectors: GradientDecompositionVectors

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.report))


def _within_tolerance(
    metrics: Mapping[str, float], config: GradientDecompositionConfig
) -> bool:
    return bool(
        metrics["max_abs"] <= config.parent_entropy_max_abs_tolerance
        and metrics["relative_l2"]
        <= config.parent_entropy_relative_l2_tolerance
    )


def _checked_metrics(
    reference: Tensor,
    reconstructed: Tensor,
    *,
    config: GradientDecompositionConfig,
    label: str,
) -> dict[str, Any]:
    metrics = _v1._residual_metrics(reference, reconstructed)
    verified = _within_tolerance(metrics, config)
    if not verified:
        raise ForegroundBackgroundGradientError(
            f"{label} failed; max_abs={metrics['max_abs']:.17g}, "
            "max_abs_tolerance="
            f"{config.parent_entropy_max_abs_tolerance:.17g}, "
            f"relative_l2={metrics['relative_l2']:.17g}, "
            "relative_l2_tolerance="
            f"{config.parent_entropy_relative_l2_tolerance:.17g}, "
            f"reference_l2={metrics['reference_l2']:.17g}"
        )
    return {
        **metrics,
        "max_abs_tolerance": config.parent_entropy_max_abs_tolerance,
        "relative_l2_tolerance": config.parent_entropy_relative_l2_tolerance,
        "verified": True,
    }


def _audit_metrics(
    raw_direct_reference: Tensor,
    derived_observed: Tensor,
    *,
    config: GradientDecompositionConfig,
) -> dict[str, Any]:
    # The frozen v2 protocol defines every raw audit's relative denominator as
    # max(||raw direct reference||_2, 1e-12).  Keep the argument names
    # directional so a seemingly symmetric residual comparison cannot silently
    # reverse that denominator.
    metrics = _v1._residual_metrics(raw_direct_reference, derived_observed)
    return {
        **metrics,
        "max_abs_tolerance": config.parent_entropy_max_abs_tolerance,
        "relative_l2_tolerance": config.parent_entropy_relative_l2_tolerance,
        "within_frozen_tolerance": _within_tolerance(metrics, config),
        "audit_only": True,
        "used_by_scientific_metrics": False,
    }


def _group_raw_audit(
    *,
    indices: Tensor,
    foreground_suprathreshold_add: Tensor,
    background_add: Tensor,
    full_add: Tensor,
    raw_foreground_suprathreshold_add: Tensor,
    raw_background_add: Tensor,
    foreground_subthreshold_add: Tensor,
    config: GradientDecompositionConfig,
) -> dict[str, Any]:
    derived_supra = foreground_suprathreshold_add[indices]
    derived_background = background_add[indices]
    direct_full = full_add[indices]
    raw_supra = raw_foreground_suprathreshold_add[indices]
    raw_background = raw_background_add[indices]
    raw_sum = foreground_subthreshold_add[indices] + raw_supra + raw_background
    return {
        "foreground_suprathreshold_raw_vs_derived": _audit_metrics(
            raw_supra, derived_supra, config=config
        ),
        "background_raw_vs_derived": _audit_metrics(
            raw_background, derived_background, config=config
        ),
        "raw_independent_three_sum_vs_direct_full": _audit_metrics(
            raw_sum, direct_full, config=config
        ),
    }


def analyze_foreground_background_gradient_decomposition(
    *,
    source_logits: Tensor,
    target: Tensor,
    named_parameters: Mapping[str, Tensor],
    parameter_layout: FlatParameterLayout,
    group_parameter_names: Mapping[str, Sequence[str]],
    parent_entropy_gradient_flat: Tensor,
    task_gradient_flat: Tensor,
    config: GradientDecompositionConfig | Mapping[str, Any] = (
        GradientDecompositionConfig()
    ),
) -> ForegroundBackgroundGradientResult:
    """Decompose one Source entropy gradient using cumulative linear closure.

    The call signature and parent/task evidence roles are compatible with v1.
    All returned scientific vectors are detached contiguous CPU float64
    tensors.  Raw independently evaluated supra/background vectors appear only
    as residual metrics in the JSON-safe report.
    """

    frozen_config = _v1._validated_config(config)
    _v1._validate_logits_target(source_logits, target)
    parameters = _v1._validate_named_parameters(
        named_parameters=named_parameters,
        parameter_layout=parameter_layout,
        device=source_logits.device,
    )
    normalized_groups, group_indices = _v1._validate_group_parameter_names(
        parameter_layout=parameter_layout,
        group_parameter_names=group_parameter_names,
    )
    parent_entropy = _v1._validate_external_vector(
        parent_entropy_gradient_flat,
        label="parent_entropy_gradient_flat",
        scalar_count=parameter_layout.scalar_count,
    )
    task = _v1._validate_external_vector(
        task_gradient_flat,
        label="task_gradient_flat",
        scalar_count=parameter_layout.scalar_count,
    )

    probability_for_partition = torch.sigmoid(source_logits.detach())
    foreground = target.detach() > 0
    background = target.detach() == 0
    foreground_subthreshold = foreground & (probability_for_partition <= 0.5)
    foreground_suprathreshold = foreground & (probability_for_partition > 0.5)
    total_pixels = int(target.numel())
    foreground_pixels = int(foreground.sum().item())
    background_pixels = int(background.sum().item())
    subthreshold_pixels = int(foreground_subthreshold.sum().item())
    suprathreshold_pixels = int(foreground_suprathreshold.sum().item())
    if foreground_pixels + background_pixels != total_pixels:
        raise ForegroundBackgroundGradientError(
            "foreground/background masks do not partition the target"
        )
    if subthreshold_pixels + suprathreshold_pixels != foreground_pixels:
        raise ForegroundBackgroundGradientError(
            "sub/suprathreshold masks do not partition foreground"
        )

    entropy = binary_entropy_map(source_logits, eps=frozen_config.entropy_eps)
    mask_dtype = entropy.dtype
    fg_sub_loss = (
        entropy * foreground_subthreshold.to(dtype=mask_dtype)
    ).sum() / total_pixels
    fg_supra_loss = (
        entropy * foreground_suprathreshold.to(dtype=mask_dtype)
    ).sum() / total_pixels
    foreground_direct_loss = (
        entropy * foreground.to(dtype=mask_dtype)
    ).sum() / total_pixels
    bg_loss = (entropy * background.to(dtype=mask_dtype)).sum() / total_pixels
    full_direct_loss = entropy.mean()
    loss_reconstruction = fg_sub_loss + fg_supra_loss + bg_loss
    foreground_loss_reconstruction = fg_sub_loss + fg_supra_loss
    loss_residual = abs(
        float((loss_reconstruction - full_direct_loss).detach().item())
    )
    foreground_loss_residual = abs(
        float(
            (foreground_loss_reconstruction - foreground_direct_loss)
            .detach()
            .item()
        )
    )
    if (
        not math.isfinite(loss_residual)
        or not math.isfinite(foreground_loss_residual)
        or loss_residual > frozen_config.parent_entropy_max_abs_tolerance
        or foreground_loss_residual
        > frozen_config.parent_entropy_max_abs_tolerance
    ):
        raise ForegroundBackgroundGradientError(
            "additive entropy losses do not reconstruct direct cumulative losses"
        )

    grad_slots_before = _v1._snapshot_grad_slots(parameters)
    versions_before = tuple(parameter._version for parameter in parameters)
    full_direct = _v1._autograd_flat(
        full_direct_loss,
        parameters,
        label=BACKWARD_BASIS[0],
        scalar_count=parameter_layout.scalar_count,
        retain_graph=True,
    )
    foreground_direct = _v1._autograd_flat(
        foreground_direct_loss,
        parameters,
        label=BACKWARD_BASIS[1],
        scalar_count=parameter_layout.scalar_count,
        retain_graph=True,
    )
    foreground_subthreshold_direct = _v1._autograd_flat(
        fg_sub_loss,
        parameters,
        label=BACKWARD_BASIS[2],
        scalar_count=parameter_layout.scalar_count,
        retain_graph=True,
    )
    raw_foreground_suprathreshold_add = _v1._autograd_flat(
        fg_supra_loss,
        parameters,
        label=BACKWARD_BASIS[3],
        scalar_count=parameter_layout.scalar_count,
        retain_graph=True,
    )
    raw_background_add = _v1._autograd_flat(
        bg_loss,
        parameters,
        label=BACKWARD_BASIS[4],
        scalar_count=parameter_layout.scalar_count,
        retain_graph=False,
    )
    if not _v1._grad_slots_unchanged(parameters, grad_slots_before):
        raise ForegroundBackgroundGradientError(
            "autograd.grad modified a parameter .grad slot"
        )
    versions_after = tuple(parameter._version for parameter in parameters)
    if versions_after != versions_before:
        raise ForegroundBackgroundGradientError(
            "parameter values changed during pure gradient decomposition"
        )

    # Cumulative endpoints and their adjacent differences are CPU float64.
    cumulative_direct = torch.stack(
        (
            foreground_subthreshold_direct,
            foreground_direct,
            full_direct,
        ),
        dim=0,
    )
    zero = torch.zeros_like(cumulative_direct[:1])
    scientific_basis = (
        cumulative_direct
        - torch.cat((zero, cumulative_direct[:-1]), dim=0)
    ).contiguous()
    fg_sub_add, fg_supra_add, bg_add = tuple(scientific_basis.unbind(0))
    foreground_add = foreground_direct
    full_add = full_direct

    reconstructed_endpoints = torch.cumsum(scientific_basis, dim=0)
    endpoint_closure: dict[str, Any] = {}
    for index, name in enumerate(CUMULATIVE_ENDPOINTS):
        endpoint_closure[name] = _checked_metrics(
            cumulative_direct[index],
            reconstructed_endpoints[index],
            config=frozen_config,
            label=f"cumulative closure for {name}",
        )

    # The artifact stores the five direct/raw VJPs, not the derived basis.
    # Reproduce that exact path before checking downstream telescoping.
    direct_vjp_stack = torch.stack(
        (
            full_direct,
            foreground_direct,
            foreground_subthreshold_direct,
            raw_foreground_suprathreshold_add,
            raw_background_add,
        ),
        dim=0,
    )
    direct_storage_roundtrip = direct_vjp_stack.to(dtype=torch.float32).to(
        dtype=torch.float64
    )
    direct_storage_drift: dict[str, Any] = {}
    for index, name in enumerate(DIRECT_VECTOR_FIELDS):
        direct_storage_drift[name] = _checked_metrics(
            direct_vjp_stack[index],
            direct_storage_roundtrip[index],
            config=frozen_config,
            label=f"float32 direct-VJP storage round-trip for {name}",
        )
    roundtrip_cumulative = torch.stack(
        (
            direct_storage_roundtrip[2],
            direct_storage_roundtrip[1],
            direct_storage_roundtrip[0],
        ),
        dim=0,
    )
    roundtrip_zero = torch.zeros_like(roundtrip_cumulative[:1])
    roundtrip_basis = roundtrip_cumulative - torch.cat(
        (roundtrip_zero, roundtrip_cumulative[:-1]), dim=0
    )
    storage_reconstruction = torch.cumsum(roundtrip_basis, dim=0)
    storage_roundtrip_closure: dict[str, Any] = {}
    storage_roundtrip_vs_live: dict[str, Any] = {}
    for index, name in enumerate(CUMULATIVE_ENDPOINTS):
        storage_roundtrip_closure[name] = _checked_metrics(
            roundtrip_cumulative[index],
            storage_reconstruction[index],
            config=frozen_config,
            label=f"float32 stored cumulative closure for {name}",
        )
        storage_roundtrip_vs_live[name] = _checked_metrics(
            cumulative_direct[index],
            storage_reconstruction[index],
            config=frozen_config,
            label=f"float32 stored reconstruction versus live {name}",
        )

    parent_metrics = _checked_metrics(
        parent_entropy,
        full_add,
        config=frozen_config,
        label="direct full entropy gradient differs from parent evidence",
    )

    foreground_conditional = _v1._optional_conditional(
        foreground_add,
        pixel_count=foreground_pixels,
        total_pixel_count=total_pixels,
    )
    background_conditional = _v1._optional_conditional(
        bg_add,
        pixel_count=background_pixels,
        total_pixel_count=total_pixels,
    )
    fg_sub_conditional = _v1._optional_conditional(
        fg_sub_add,
        pixel_count=subthreshold_pixels,
        total_pixel_count=total_pixels,
    )
    fg_supra_conditional = _v1._optional_conditional(
        fg_supra_add,
        pixel_count=suprathreshold_pixels,
        total_pixel_count=total_pixels,
    )

    weighted = torch.zeros_like(full_add)
    if foreground_conditional is not None:
        weighted = weighted + (
            foreground_pixels / total_pixels
        ) * foreground_conditional
    if background_conditional is not None:
        weighted = weighted + (
            background_pixels / total_pixels
        ) * background_conditional
    weighted_metrics = _checked_metrics(
        full_add,
        weighted,
        config=frozen_config,
        label="weighted conditional gradients do not reconstruct full entropy",
    )
    weighted_metrics.update(
        {
            "foreground_weight": foreground_pixels / total_pixels,
            "background_weight": background_pixels / total_pixels,
        }
    )

    conditional_vectors: dict[str, tuple[Tensor | None, str | None]] = {
        "full_entropy_mean": (full_add, None),
        "foreground_entropy_mean": (
            foreground_conditional,
            "empty_foreground" if foreground_conditional is None else None,
        ),
        "background_entropy_mean": (
            background_conditional,
            "empty_background" if background_conditional is None else None,
        ),
        "foreground_subthreshold_entropy_mean": (
            fg_sub_conditional,
            "empty_foreground_subthreshold"
            if fg_sub_conditional is None
            else None,
        ),
        "foreground_suprathreshold_entropy_mean": (
            fg_supra_conditional,
            "empty_foreground_suprathreshold"
            if fg_supra_conditional is None
            else None,
        ),
    }
    additive_basis = {
        "foreground_subthreshold_add": fg_sub_add,
        "foreground_suprathreshold_add": fg_supra_add,
        "background_add": bg_add,
        "foreground_add": foreground_add,
        "full_add": full_add,
    }
    additive_basis_present = {
        "foreground_subthreshold_add": subthreshold_pixels > 0,
        "foreground_suprathreshold_add": suprathreshold_pixels > 0,
        "background_add": background_pixels > 0,
        "foreground_add": foreground_pixels > 0,
        "full_add": True,
    }
    additive_alignment_vectors: dict[str, tuple[Tensor | None, str | None]] = {
        "full_entropy_mean": (full_add, None),
        "foreground_entropy_add": (
            foreground_add if foreground_pixels else None,
            "empty_foreground" if not foreground_pixels else None,
        ),
        "background_entropy_add": (
            bg_add if background_pixels else None,
            "empty_background" if not background_pixels else None,
        ),
        "foreground_subthreshold_entropy_add": (
            fg_sub_add if subthreshold_pixels else None,
            "empty_foreground_subthreshold" if not subthreshold_pixels else None,
        ),
        "foreground_suprathreshold_entropy_add": (
            fg_supra_add if suprathreshold_pixels else None,
            "empty_foreground_suprathreshold" if not suprathreshold_pixels else None,
        ),
    }

    raw_audit_per_group: dict[str, Any] = {}
    per_group: dict[str, Any] = {}
    for group_id in GROUP_IDS:
        indices = group_indices[group_id]
        task_part = task[indices]
        task_norm = float(torch.linalg.vector_norm(task_part).item())
        foreground_add_norm = (
            float(torch.linalg.vector_norm(foreground_add[indices]).item())
            if foreground_pixels
            else None
        )
        background_add_norm = (
            float(torch.linalg.vector_norm(bg_add[indices]).item())
            if background_pixels
            else None
        )
        foreground_conditional_norm = (
            float(torch.linalg.vector_norm(foreground_conditional[indices]).item())
            if foreground_conditional is not None
            else None
        )
        background_conditional_norm = (
            float(torch.linalg.vector_norm(background_conditional[indices]).item())
            if background_conditional is not None
            else None
        )
        foreground_dot = (
            float(torch.dot(foreground_add[indices], task_part).item())
            if foreground_pixels
            else None
        )
        background_dot = (
            float(torch.dot(bg_add[indices], task_part).item())
            if background_pixels
            else None
        )
        full_dot = float(torch.dot(full_add[indices], task_part).item())
        projection_denominator = (
            None
            if foreground_dot is None or background_dot is None
            else abs(foreground_dot) + abs(background_dot)
        )
        if projection_denominator is None:
            cancellation_value: float | None = None
            cancellation_status = "not_estimable_region_absent"
            cancellation_reason = (
                "empty_foreground" if not foreground_pixels else "empty_background"
            )
        elif task_norm <= frozen_config.cosine_zero_norm_tolerance:
            cancellation_value = None
            cancellation_status = "not_estimable_task_gradient_zero"
            cancellation_reason = "task_gradient_zero"
        elif projection_denominator <= frozen_config.cosine_zero_norm_tolerance:
            cancellation_value = None
            cancellation_status = "not_estimable_projection_denominator_zero"
            cancellation_reason = "foreground_and_background_task_dots_zero"
        else:
            cancellation_value = 1.0 - abs(full_dot) / projection_denominator
            cancellation_value = min(1.0, max(0.0, cancellation_value))
            cancellation_status = "estimable"
            cancellation_reason = None
        per_group[group_id] = {
            "parameter_tensor_count": len(normalized_groups[group_id]),
            "parameter_scalar_count": int(indices.numel()),
            "task_gradient_norm": task_norm,
            "additive_entropy_task_alignment": {
                name: _v1._alignment(
                    value,
                    task,
                    indices,
                    absent_reason=reason,
                    zero_tolerance=frozen_config.cosine_zero_norm_tolerance,
                )
                for name, (value, reason) in additive_alignment_vectors.items()
            },
            "conditional_entropy_task_alignment": {
                name: _v1._alignment(
                    value,
                    task,
                    indices,
                    absent_reason=reason,
                    zero_tolerance=frozen_config.cosine_zero_norm_tolerance,
                )
                for name, (value, reason) in conditional_vectors.items()
            },
            "additive_gradient_norms": {
                name: (
                    float(torch.linalg.vector_norm(value[indices]).item())
                    if additive_basis_present[name]
                    else None
                )
                for name, value in additive_basis.items()
            },
            "cross_region": {
                "foreground_background_additive_alignment": _v1._pair_metrics(
                    foreground_add if foreground_pixels else None,
                    bg_add if background_pixels else None,
                    indices,
                    left_absent_reason=(
                        "empty_foreground" if not foreground_pixels else None
                    ),
                    right_absent_reason=(
                        "empty_background" if not background_pixels else None
                    ),
                    zero_tolerance=frozen_config.cosine_zero_norm_tolerance,
                ),
                "background_to_foreground_additive_norm_ratio": _v1._ratio(
                    background_add_norm,
                    foreground_add_norm,
                    zero_tolerance=frozen_config.cosine_zero_norm_tolerance,
                    absent_reason=(
                        "empty_foreground_or_background"
                        if not foreground_pixels or not background_pixels
                        else None
                    ),
                ),
                "background_to_foreground_conditional_norm_ratio": _v1._ratio(
                    background_conditional_norm,
                    foreground_conditional_norm,
                    zero_tolerance=frozen_config.cosine_zero_norm_tolerance,
                    absent_reason=(
                        "empty_foreground_or_background"
                        if foreground_conditional is None
                        or background_conditional is None
                        else None
                    ),
                ),
                "projection_cancellation_ratio": {
                    "value": cancellation_value,
                    "status": cancellation_status,
                    "not_estimable_reason": cancellation_reason,
                    "formula": (
                        "1-abs(full_task_dot)/(abs(foreground_task_dot)+"
                        "abs(background_task_dot))"
                    ),
                    "foreground_task_dot": foreground_dot,
                    "background_task_dot": background_dot,
                    "full_task_dot": full_dot,
                },
            },
        }
        raw_audit_per_group[group_id] = _group_raw_audit(
            indices=indices,
            foreground_suprathreshold_add=fg_supra_add,
            background_add=bg_add,
            full_add=full_add,
            raw_foreground_suprathreshold_add=(
                raw_foreground_suprathreshold_add
            ),
            raw_background_add=raw_background_add,
            foreground_subthreshold_add=fg_sub_add,
            config=frozen_config,
        )

    raw_sum = (
        fg_sub_add
        + raw_foreground_suprathreshold_add
        + raw_background_add
    )
    raw_foreground_sum = fg_sub_add + raw_foreground_suprathreshold_add
    raw_audit = {
        "role": "audit_only_backend_reduction_nonclosure",
        "used_by_scientific_metrics": False,
        "used_by_returned_vectors": False,
        "failure_does_not_reject_scientific_cumulative_basis": True,
        "global": {
            "foreground_suprathreshold_raw_vs_derived": _audit_metrics(
                raw_foreground_suprathreshold_add,
                fg_supra_add,
                config=frozen_config,
            ),
            "background_raw_vs_derived": _audit_metrics(
                raw_background_add,
                bg_add,
                config=frozen_config,
            ),
            "raw_foreground_sum_vs_direct_foreground": _audit_metrics(
                raw_foreground_sum,
                foreground_add,
                config=frozen_config,
            ),
            "raw_independent_three_sum_vs_direct_full": _audit_metrics(
                raw_sum,
                full_add,
                config=frozen_config,
            ),
        },
        "per_group": raw_audit_per_group,
    }

    foreground_values = target.detach()[foreground]
    target_statistics = {
        "total_pixel_count": total_pixels,
        "foreground_pixel_count": foreground_pixels,
        "background_pixel_count": background_pixels,
        "foreground_subthreshold_pixel_count": subthreshold_pixels,
        "foreground_suprathreshold_pixel_count": suprathreshold_pixels,
        "foreground_fraction": foreground_pixels / total_pixels,
        "background_fraction": background_pixels / total_pixels,
        "foreground_subthreshold_fraction_of_full": subthreshold_pixels / total_pixels,
        "foreground_suprathreshold_fraction_of_full": suprathreshold_pixels / total_pixels,
        "target_min": float(target.detach().amin().item()),
        "target_max": float(target.detach().amax().item()),
        "foreground_value_min": (
            float(foreground_values.amin().item()) if foreground_pixels else None
        ),
        "foreground_value_max": (
            float(foreground_values.amax().item()) if foreground_pixels else None
        ),
        "foreground_value_mean": (
            float(foreground_values.mean().item()) if foreground_pixels else None
        ),
        "foreground_estimable": foreground_pixels > 0,
        "background_estimable": background_pixels > 0,
    }
    losses = {
        "full_entropy_mean": float(full_direct_loss.detach().item()),
        "additive": {
            "foreground_subthreshold_add": float(fg_sub_loss.detach().item()),
            "foreground_suprathreshold_add": float(fg_supra_loss.detach().item()),
            "foreground_add": float(foreground_loss_reconstruction.detach().item()),
            "background_add": float(bg_loss.detach().item()),
            "full_add": float(loss_reconstruction.detach().item()),
        },
        "conditional_mean": {
            "foreground": (
                float(
                    (
                        foreground_loss_reconstruction
                        * total_pixels
                        / foreground_pixels
                    )
                    .detach()
                    .item()
                )
                if foreground_pixels
                else None
            ),
            "background": (
                float((bg_loss * total_pixels / background_pixels).detach().item())
                if background_pixels
                else None
            ),
            "foreground_subthreshold": (
                float((fg_sub_loss * total_pixels / subthreshold_pixels).detach().item())
                if subthreshold_pixels
                else None
            ),
            "foreground_suprathreshold": (
                float((fg_supra_loss * total_pixels / suprathreshold_pixels).detach().item())
                if suprathreshold_pixels
                else None
            ),
        },
        "full_additive_scalar_residual": loss_residual,
        "foreground_additive_scalar_residual": foreground_loss_residual,
    }
    scalar_values = [
        value
        for value in (
            losses["full_entropy_mean"],
            *losses["additive"].values(),
            *(
                value
                for value in losses["conditional_mean"].values()
                if value is not None
            ),
            losses["full_additive_scalar_residual"],
            losses["foreground_additive_scalar_residual"],
        )
    ]
    if not all(math.isfinite(float(value)) for value in scalar_values):
        raise ForegroundBackgroundGradientError(
            "entropy loss report contains NaN/Inf"
        )

    group_descriptor = {
        group_id: list(normalized_groups[group_id]) for group_id in GROUP_IDS
    }
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": ANALYSIS_TYPE,
        "config": frozen_config.to_dict(),
        "conventions": {
            "source_forward_count_inside_module": 0,
            "autograd_backward_basis": list(BACKWARD_BASIS),
            "autograd_vjp_count": 5,
            "scientific_basis": list(SCIENTIFIC_BASIS),
            "cumulative_endpoints": list(CUMULATIVE_ENDPOINTS),
            "audit_only_raw_vjps": list(AUDIT_ONLY_RAW_VJPS),
            "cumulative_derivation": (
                "g_sub=G_FGsub;g_supra=G_FG-G_FGsub;g_BG=G_full-G_FG"
            ),
            "full_add_source": "direct_full_entropy_mean_vjp",
            "additive_denominator": "full_image_pixel_count",
            "conditional_means_are_derived": True,
            "foreground_rule": FOREGROUND_RULE,
            "background_rule": BACKGROUND_RULE,
            "foreground_subthreshold_rule": SUBTHRESHOLD_RULE,
            "foreground_suprathreshold_rule": SUPRATHRESHOLD_RULE,
            "parent_entropy_role": "sealed_label_free_full_image_mean_gradient",
            "task_gradient_role": (
                "external_existing_bce_plus_soft_iou_outer_oracle_gradient"
            ),
            "task_loss_recomputed_inside_module": False,
            "analysis_vector_dtype": "torch.float64_cpu",
            "artifact_storage_cast": "runner_may_cast_vectors_to_float32",
        },
        "parameter_layout": {
            "protocol": parameter_layout.to_dict()["protocol"],
            "layout_sha256": parameter_layout.layout_sha256,
            "parameter_tensor_count": len(parameter_layout.names),
            "parameter_scalar_count": parameter_layout.scalar_count,
            "group_ids": list(GROUP_IDS),
            "group_parameter_names_sha256": _v1._canonical_sha256(
                group_descriptor
            ),
        },
        "target_statistics": target_statistics,
        "losses": losses,
        "decomposition": {
            "foreground_add_is_sub_plus_supra": True,
            "full_add_is_foreground_plus_background": True,
            "weighted_conditional_reconstruction": weighted_metrics,
            "parent_full_entropy_reconstruction": parent_metrics,
        },
        "numeric_audit": {
            "evaluation_scheme": "cumulative_telescoping_cpu_float64_v2",
            "direct_vjp_order": list(BACKWARD_BASIS),
            "direct_vector_fields": list(DIRECT_VECTOR_FIELDS),
            "scientific_basis_order": list(SCIENTIFIC_BASIS),
            "direct_vjp_count": 5,
            "full_add_is_direct_vjp": True,
            "cumulative_float64_closure": endpoint_closure,
            "float32_direct_vjp_storage_drift": direct_storage_drift,
            "float32_storage_roundtrip_closure": storage_roundtrip_closure,
            "float32_storage_reconstruction_vs_live_direct": (
                storage_roundtrip_vs_live
            ),
            "raw_vjp_residual_audit": raw_audit,
        },
        "per_group": per_group,
        "finiteness": {
            "source_logits": True,
            "target": True,
            "entropy_map": True,
            "additive_gradients": True,
            "direct_cumulative_gradients": True,
            "raw_audit_gradients": True,
            "parent_entropy_gradient": True,
            "task_gradient": True,
            "reported_scalars": True,
        },
        "side_effects": {
            "uses_filesystem": False,
            "uses_optimizer": False,
            "calls_backward": False,
            "parameter_versions_unchanged": True,
            "parameter_grad_slots_unchanged": True,
        },
    }
    json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    vectors = GradientDecompositionVectors(
        foreground_subthreshold_add=fg_sub_add.clone(),
        foreground_suprathreshold_add=fg_supra_add.clone(),
        background_add=bg_add.clone(),
        foreground_add=foreground_add.clone(),
        full_add=full_add.clone(),
        foreground_conditional_mean=(
            None
            if foreground_conditional is None
            else foreground_conditional.clone()
        ),
        background_conditional_mean=(
            None
            if background_conditional is None
            else background_conditional.clone()
        ),
        foreground_subthreshold_conditional_mean=(
            None
            if fg_sub_conditional is None
            else fg_sub_conditional.clone()
        ),
        foreground_suprathreshold_conditional_mean=(
            None
            if fg_supra_conditional is None
            else fg_supra_conditional.clone()
        ),
        parent_entropy=parent_entropy.clone(),
        task=task.clone(),
        full_direct=full_direct.clone(),
        foreground_total_direct=foreground_direct.clone(),
        foreground_subthreshold_direct=(
            foreground_subthreshold_direct.clone()
        ),
        foreground_suprathreshold_raw_direct=(
            raw_foreground_suprathreshold_add.clone()
        ),
        background_raw_direct=raw_background_add.clone(),
    )
    return ForegroundBackgroundGradientResult(report=report, vectors=vectors)


compute_foreground_background_gradient_decomposition = (
    analyze_foreground_background_gradient_decomposition
)


__all__ = [
    "ANALYSIS_TYPE",
    "AUDIT_ONLY_RAW_VJPS",
    "BACKWARD_BASIS",
    "BACKGROUND_RULE",
    "CUMULATIVE_ENDPOINTS",
    "DIRECT_VECTOR_FIELDS",
    "FOREGROUND_RULE",
    "GROUP_IDS",
    "SCHEMA_VERSION",
    "SCIENTIFIC_BASIS",
    "GradientDecompositionConfig",
    "GradientDecompositionVectors",
    "ForegroundBackgroundGradientError",
    "ForegroundBackgroundGradientResult",
    "analyze_foreground_background_gradient_decomposition",
    "compute_foreground_background_gradient_decomposition",
]
