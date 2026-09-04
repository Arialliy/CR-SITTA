"""Squared-distance anchor to the detached Source parameter state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import torch
from torch import Tensor

from ._validation import StageBObjectiveError, ensure_finite_scalar


ParameterCollection: TypeAlias = Mapping[str, Tensor] | Sequence[Tensor]


def _normalise_parameter_pairs(
    parameters: ParameterCollection,
    source_parameters: ParameterCollection,
) -> tuple[tuple[str, Tensor, Tensor], ...]:
    if isinstance(parameters, Mapping):
        if not isinstance(source_parameters, Mapping):
            raise TypeError(
                "source_parameters must be a mapping when parameters is a mapping"
            )
        names = tuple(parameters)
        if names != tuple(source_parameters):
            raise StageBObjectiveError(
                "parameter and Source parameter names/order must match exactly"
            )
        pairs = tuple(
            (str(name), parameters[name], source_parameters[name])
            for name in names
        )
    else:
        if isinstance(parameters, Tensor) or not isinstance(parameters, Sequence):
            raise TypeError("parameters must be a mapping or sequence of tensors")
        if isinstance(source_parameters, Tensor) or not isinstance(
            source_parameters, Sequence
        ):
            raise TypeError(
                "source_parameters must be a sequence when parameters is a sequence"
            )
        current_values = tuple(parameters)
        source_values = tuple(source_parameters)
        if len(current_values) != len(source_values):
            raise StageBObjectiveError(
                "parameter and Source parameter counts must match exactly"
            )
        pairs = tuple(
            (str(index), current, source)
            for index, (current, source) in enumerate(
                zip(current_values, source_values, strict=True)
            )
        )
    if not pairs:
        raise StageBObjectiveError("parameter anchor requires a non-empty space")
    return pairs


def parameter_anchor(
    parameters: ParameterCollection,
    source_parameters: ParameterCollection,
) -> Tensor:
    """Return ``sum_i ||phi_i - phi_i_source||_2^2``.

    Source values must already be detached snapshots.  Silent detachment is
    intentionally forbidden so an incorrectly wired anchor fails closed.
    """

    pairs = _normalise_parameter_pairs(parameters, source_parameters)
    current_identities: set[int] = set()
    source_identities: set[int] = set()
    common_device: torch.device | None = None
    common_dtype: torch.dtype | None = None
    components: list[Tensor] = []
    for name, current, source in pairs:
        if not isinstance(current, Tensor) or not isinstance(source, Tensor):
            raise TypeError(f"parameter pair {name!r} must contain tensors")
        if id(current) in current_identities:
            raise StageBObjectiveError("parameters contain duplicate tensor objects")
        if id(source) in source_identities:
            raise StageBObjectiveError(
                "source_parameters contain duplicate tensor objects"
            )
        current_identities.add(id(current))
        source_identities.add(id(source))
        for value, role in ((current, "parameter"), (source, "Source parameter")):
            if not torch.is_floating_point(value) or value.is_complex():
                raise TypeError(f"{role} {name!r} must be real floating-point")
            if not bool(torch.isfinite(value).all().detach().item()):
                raise StageBObjectiveError(f"{role} {name!r} must be finite")
        if source.requires_grad or source.grad_fn is not None:
            raise StageBObjectiveError(
                f"Source parameter {name!r} must be detached"
            )
        if current.shape != source.shape:
            raise StageBObjectiveError(
                f"parameter pair {name!r} shapes must match exactly"
            )
        if current.device != source.device or current.dtype != source.dtype:
            raise StageBObjectiveError(
                f"parameter pair {name!r} dtype/device must match exactly"
            )
        if common_device is None:
            common_device = current.device
            common_dtype = current.dtype
        elif current.device != common_device or current.dtype != common_dtype:
            raise StageBObjectiveError(
                "all anchored parameters must share one dtype and device"
            )
        components.append(torch.square(current - source).sum())
    loss = torch.stack(components).sum()
    ensure_finite_scalar(loss, name="parameter anchor")
    return loss


parameter_anchor_loss = parameter_anchor


__all__ = [
    "ParameterCollection",
    "parameter_anchor",
    "parameter_anchor_loss",
]
