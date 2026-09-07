"""Regularizers for the Stage-C spatial low-rank FiLM coefficients.

Only the trainable ``scale_coeff`` and ``bias_coeff`` maps belong to this
regularizer.  The fixed channel basis is deliberately excluded.  Public
functions require stable, fully qualified parameter names so accidentally
passing Source-model weights fails closed instead of silently changing the
objective.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor

from ._validation import (
    StageBObjectiveError,
    differentiable_zero,
    ensure_finite_scalar,
)


@dataclass(frozen=True)
class RouterRegularizationOutput:
    """Live scalar terms plus topology counts for audit records."""

    adapter_l2: Tensor
    spatial_tv: Tensor
    coefficient_tensor_count: int
    coefficient_scalar_count: int
    spatial_axis_count: int


def _validated_router_parameters(
    router_parameters: Mapping[str, Tensor],
) -> tuple[tuple[str, Tensor], ...]:
    if not isinstance(router_parameters, Mapping):
        raise TypeError("router_parameters must be a mapping of names to tensors")
    items = tuple(router_parameters.items())
    if not items:
        raise StageBObjectiveError("router_parameters must not be empty")

    identities: set[int] = set()
    topology: dict[str, set[str]] = {}
    common_device: torch.device | None = None
    common_dtype: torch.dtype | None = None
    for name, parameter in items:
        if not isinstance(name, str) or not name:
            raise TypeError("router parameter names must be non-empty strings")
        prefix, separator, leaf = name.rpartition(".")
        if not separator:
            prefix, leaf = "", name
        if leaf not in {"scale_coeff", "bias_coeff"}:
            raise StageBObjectiveError(
                "router parameter leaf names must be scale_coeff or bias_coeff"
            )
        topology.setdefault(prefix, set()).add(leaf)

        if not isinstance(parameter, Tensor):
            raise TypeError(f"router parameter {name!r} must be a torch.Tensor")
        if id(parameter) in identities:
            raise StageBObjectiveError("router_parameters contain aliased tensors")
        identities.add(id(parameter))
        if parameter.ndim != 4 or int(parameter.shape[0]) != 1:
            raise StageBObjectiveError(
                f"router parameter {name!r} must have shape [1,rank,H,W]"
            )
        if any(int(size) <= 0 for size in parameter.shape):
            raise StageBObjectiveError(
                f"router parameter {name!r} dimensions must be positive"
            )
        if not parameter.is_floating_point() or parameter.is_complex():
            raise TypeError(
                f"router parameter {name!r} must be real floating-point"
            )
        if not bool(torch.isfinite(parameter).all().detach().item()):
            raise StageBObjectiveError(
                f"router parameter {name!r} must contain only finite values"
            )
        if common_device is None:
            common_device = parameter.device
            common_dtype = parameter.dtype
        elif parameter.device != common_device or parameter.dtype != common_dtype:
            raise StageBObjectiveError(
                "all router parameters must share one dtype and device"
            )

    expected = {"scale_coeff", "bias_coeff"}
    for prefix, leaves in topology.items():
        if leaves != expected:
            display = prefix if prefix else "<root>"
            raise StageBObjectiveError(
                f"router {display!r} must contain one scale_coeff and one bias_coeff"
            )
    return items


def router_coefficient_l2(
    router_parameters: Mapping[str, Tensor],
) -> Tensor:
    """Return ``sum ||coefficient||_2^2`` over trainable router maps."""

    items = _validated_router_parameters(router_parameters)
    result = torch.stack(
        [torch.square(parameter).sum() for _name, parameter in items]
    ).sum()
    ensure_finite_scalar(result, name="router coefficient L2")
    return result


def router_coefficient_total_variation(
    router_parameters: Mapping[str, Tensor],
) -> Tensor:
    """Return anisotropic TV with each existing spatial axis normalized.

    For every coefficient tensor, height and width differences are averaged
    independently and then added.  A size-one axis contributes no term.  If
    every map is 1x1, the result is an exact graph-connected zero.
    """

    items = _validated_router_parameters(router_parameters)
    terms: list[Tensor] = []
    for _name, parameter in items:
        parameter_has_axis = False
        if int(parameter.shape[-2]) > 1:
            height_difference = parameter[..., 1:, :] - parameter[..., :-1, :]
            terms.append(height_difference.abs().mean())
            parameter_has_axis = True
        if int(parameter.shape[-1]) > 1:
            width_difference = parameter[..., :, 1:] - parameter[..., :, :-1]
            terms.append(width_difference.abs().mean())
            parameter_has_axis = True
        if not parameter_has_axis:
            # Preserve a live, zero-gradient edge for every 1x1 map.
            terms.append(differentiable_zero(parameter))
    result = torch.stack(terms).sum()
    ensure_finite_scalar(result, name="router coefficient total variation")
    return result


def router_regularization(
    router_parameters: Mapping[str, Tensor],
) -> RouterRegularizationOutput:
    """Compute both unweighted router regularization terms."""

    items = _validated_router_parameters(router_parameters)
    adapter_l2 = router_coefficient_l2(router_parameters)
    spatial_tv = router_coefficient_total_variation(router_parameters)
    return RouterRegularizationOutput(
        adapter_l2=adapter_l2,
        spatial_tv=spatial_tv,
        coefficient_tensor_count=len(items),
        coefficient_scalar_count=sum(
            int(parameter.numel()) for _name, parameter in items
        ),
        spatial_axis_count=sum(
            int(parameter.shape[-2] > 1) + int(parameter.shape[-1] > 1)
            for _name, parameter in items
        ),
    )


__all__ = [
    "RouterRegularizationOutput",
    "router_coefficient_l2",
    "router_coefficient_total_variation",
    "router_regularization",
]
