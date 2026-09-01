"""Deterministic CPU vectors for named model parameters and gradients.

The outer diagnostic modules compare gradients and optimizer steps from many
named tensors.  This helper provides one strict topology/layout contract so a
silently missing, duplicated, reordered, or reshaped tensor cannot corrupt a
dot product.  Values are detached and converted to CPU float64 for analysis;
no CUDA API is called here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import TypeAlias

import torch
from torch import Tensor


NamedTensorValue: TypeAlias = Tensor | None
NamedTensorItems: TypeAlias = (
    Mapping[str, NamedTensorValue] | Sequence[tuple[str, NamedTensorValue]]
)


class ParameterVectorError(ValueError):
    """Named tensors do not satisfy the analysis-vector contract."""


def _items(values: NamedTensorItems, *, label: str) -> tuple[tuple[str, NamedTensorValue], ...]:
    if isinstance(values, Mapping):
        materialized = tuple((name, values[name]) for name in sorted(values))
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        materialized = tuple(values)
    else:
        raise ParameterVectorError(f"{label} must be a mapping or sequence of pairs")
    if not materialized:
        raise ParameterVectorError(f"{label} must not be empty")
    names: list[str] = []
    for index, item in enumerate(materialized):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ParameterVectorError(f"{label}[{index}] must be a (name, tensor) pair")
        name, _ = item
        if not isinstance(name, str) or not name:
            raise ParameterVectorError(f"{label}[{index}] has an invalid name")
        names.append(name)
    if len(set(names)) != len(names):
        raise ParameterVectorError(f"{label} contains duplicate parameter names")
    return materialized


def _validated_tensor(value: NamedTensorValue, *, label: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ParameterVectorError(f"{label} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise ParameterVectorError(f"{label} must use strided layout")
    if not torch.is_floating_point(value) or value.is_complex():
        raise ParameterVectorError(f"{label} must be a real floating-point tensor")
    if value.device.type != "cpu":
        raise ParameterVectorError(
            f"{label} must already reside on CPU; analysis never transfers CUDA tensors"
        )
    if value.numel() <= 0:
        raise ParameterVectorError(f"{label} must contain at least one scalar")
    detached = value.detach().to(dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(detached).all().item()):
        raise ParameterVectorError(f"{label} contains NaN or Inf")
    return detached


@dataclass(frozen=True)
class ParameterVectorLayout:
    """Frozen names, shapes, and offsets for one vectorized parameter space."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    numels: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.names or not (
            len(self.names) == len(self.shapes) == len(self.numels)
        ):
            raise ParameterVectorError("layout fields must have equal non-zero length")
        if len(set(self.names)) != len(self.names):
            raise ParameterVectorError("layout names must be unique")
        for index, (name, shape, numel) in enumerate(
            zip(self.names, self.shapes, self.numels, strict=True)
        ):
            if not isinstance(name, str) or not name:
                raise ParameterVectorError(f"layout.names[{index}] is invalid")
            if not isinstance(shape, tuple) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in shape
            ):
                raise ParameterVectorError(f"layout.shapes[{index}] is invalid")
            expected = math.prod(shape)
            if numel <= 0 or expected != numel:
                raise ParameterVectorError(
                    f"layout.numels[{index}] disagrees with shape {shape}"
                )

    @property
    def total_numel(self) -> int:
        return sum(self.numels)

    @property
    def parameter_names_sha256(self) -> str:
        return hashlib.sha256("\0".join(self.names).encode("utf-8")).hexdigest()

    @property
    def topology_sha256(self) -> str:
        encoded = "\n".join(
            f"{name}\0{','.join(str(value) for value in shape)}\0{numel}"
            for name, shape, numel in zip(
                self.names, self.shapes, self.numels, strict=True
            )
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def offsets(self) -> tuple[tuple[int, int], ...]:
        result: list[tuple[int, int]] = []
        start = 0
        for numel in self.numels:
            result.append((start, start + numel))
            start += numel
        return tuple(result)

    def flatten(
        self,
        values: NamedTensorItems,
        *,
        label: str,
        none_as_zero: bool = False,
    ) -> Tensor:
        """Align ``values`` by exact name/shape and return one CPU float64 vector."""

        materialized = _items(values, label=label)
        by_name = dict(materialized)
        observed_names = set(by_name)
        expected_names = set(self.names)
        if observed_names != expected_names:
            missing = [name for name in self.names if name not in observed_names]
            extra = sorted(observed_names - expected_names)
            raise ParameterVectorError(
                f"{label} name topology mismatch; missing={missing[:5]}, extra={extra[:5]}"
            )
        flattened: list[Tensor] = []
        for name, shape, numel in zip(
            self.names, self.shapes, self.numels, strict=True
        ):
            value = by_name[name]
            if value is None:
                if not none_as_zero:
                    raise ParameterVectorError(f"{label}.{name} is None")
                flattened.append(torch.zeros(numel, dtype=torch.float64))
                continue
            tensor = _validated_tensor(value, label=f"{label}.{name}")
            if tuple(tensor.shape) != shape:
                raise ParameterVectorError(
                    f"{label}.{name} shape mismatch: expected {shape}, "
                    f"observed {tuple(tensor.shape)}"
                )
            flattened.append(tensor.reshape(-1))
        return torch.cat(flattened)

    def unflatten(self, vector: Tensor, *, label: str = "vector") -> dict[str, Tensor]:
        value = _validated_tensor(vector, label=label).reshape(-1)
        if value.numel() != self.total_numel:
            raise ParameterVectorError(
                f"{label} length mismatch: expected {self.total_numel}, "
                f"observed {value.numel()}"
            )
        result: dict[str, Tensor] = {}
        for name, shape, (start, stop) in zip(
            self.names, self.shapes, self.offsets(), strict=True
        ):
            result[name] = value[start:stop].reshape(shape).clone()
        return result

    def validated_group_assignment(
        self,
        assignment: Mapping[str, str] | None,
    ) -> tuple[tuple[str, str], ...]:
        """Return exact ``(parameter_name, group)`` pairs in layout order."""

        if assignment is None:
            return tuple((name, "all") for name in self.names)
        if not isinstance(assignment, Mapping):
            raise ParameterVectorError(
                "parameter_groups must map every parameter name to one group"
            )
        observed = set(assignment)
        expected = set(self.names)
        if observed != expected:
            missing = [name for name in self.names if name not in observed]
            extra = sorted(observed - expected)
            raise ParameterVectorError(
                "parameter_groups must cover the layout exactly; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        result: list[tuple[str, str]] = []
        for name in self.names:
            group = assignment[name]
            if not isinstance(group, str) or not group:
                raise ParameterVectorError(
                    f"parameter_groups[{name!r}] must be a non-empty string"
                )
            result.append((name, group))
        return tuple(result)

    def group_indices(
        self,
        assignment: Mapping[str, str] | None,
    ) -> dict[str, Tensor]:
        pairs = self.validated_group_assignment(assignment)
        order: list[str] = []
        indices: dict[str, list[Tensor]] = {}
        for (name, group), (start, stop) in zip(
            pairs, self.offsets(), strict=True
        ):
            del name
            if group not in indices:
                order.append(group)
                indices[group] = []
            indices[group].append(torch.arange(start, stop, dtype=torch.int64))
        return {
            group: torch.cat(indices[group])
            for group in order
        }


def vectorize_named_tensors(
    values: NamedTensorItems,
    *,
    label: str,
) -> tuple[ParameterVectorLayout, Tensor]:
    """Create a layout and vector from non-null named tensors."""

    materialized = _items(values, label=label)
    names: list[str] = []
    shapes: list[tuple[int, ...]] = []
    tensors: list[Tensor] = []
    for name, value in materialized:
        tensor = _validated_tensor(value, label=f"{label}.{name}")
        names.append(name)
        shapes.append(tuple(tensor.shape))
        tensors.append(tensor.reshape(-1))
    layout = ParameterVectorLayout(
        names=tuple(names),
        shapes=tuple(shapes),
        numels=tuple(value.numel() for value in tensors),
    )
    return layout, torch.cat(tensors)


__all__ = [
    "NamedTensorItems",
    "NamedTensorValue",
    "ParameterVectorError",
    "ParameterVectorLayout",
    "vectorize_named_tensors",
]
