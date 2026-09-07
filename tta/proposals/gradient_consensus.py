"""Conservative LF/HF gradient consensus for Stage-C proposals.

The combiner deliberately does not resolve conflicting proxy gradients.  A
conflict is returned to the caller so the two Source-anchored steps can be
tried independently under the same label-free backtracking checks.  This is
the fail-closed rule specified for Stage C; it is not a PCGrad projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor


class GradientConsensusError(ValueError):
    """A structural input violates the Stage-C consensus contract."""


@dataclass(frozen=True)
class ConsensusResult:
    """Detached direction and scalar audit record for two proxy gradients."""

    gradient: Tensor | None
    cosine: float | None
    decision: str
    first_norm: float | None
    second_norm: float | None
    usable_branches: tuple[str, ...]
    trial_order: tuple[str, ...]


def _finite_real(
    value: Real,
    *,
    name: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool")
    checked = float(value)
    if not math.isfinite(checked):
        raise GradientConsensusError(f"{name} must be finite")
    if positive and checked <= 0.0:
        raise GradientConsensusError(f"{name} must be strictly positive")
    return checked


def _validate_gradient(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1 or value.numel() <= 0:
        raise GradientConsensusError(
            f"{name} must be a non-empty flattened 1D gradient"
        )
    if value.layout != torch.strided or value.is_sparse:
        raise GradientConsensusError(f"{name} must use strided dense layout")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError(f"{name} must be a real floating-point tensor")


def _detached_float64(value: Tensor) -> Tensor:
    """Use a detached high-precision view for stable norms and cosine."""

    return value.detach().to(dtype=torch.float64)


def _finite_tensor(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _sealed_direction(value: Tensor) -> Tensor:
    """Return a non-aliasing direction with no adaptation graph attached."""

    return value.detach().clone().contiguous()


def _branch_status(
    value: Tensor,
    *,
    min_norm: float,
) -> tuple[str, Tensor | None, float | None]:
    """Classify one detached branch without contaminating the other branch."""

    metric = _detached_float64(value)
    if not _finite_tensor(metric):
        return "nonfinite", None, None
    norm_tensor = torch.linalg.vector_norm(metric)
    if not _finite_tensor(norm_tensor):
        return "nonfinite", None, None
    norm = float(norm_tensor.item())
    return ("tiny" if norm < min_norm else "usable"), norm_tensor, norm


def combine_two_gradients(
    first: Tensor,
    second: Tensor,
    *,
    min_norm: float,
    eps: float = 1.0e-12,
) -> ConsensusResult:
    """Combine aligned LF/HF gradients and expose conflicts for separate trials.

    Both inputs must already use the same flattened parameter-vector layout.
    Non-finite and tiny branches are dropped independently.  When exactly one
    branch remains usable, its raw direction is retained.  When both are usable
    and aligned, their unit directions are summed; the later normalized
    proposal step determines the final magnitude.  A negative cosine never
    produces a synthesized direction here.
    """

    _validate_gradient(first, name="first")
    _validate_gradient(second, name="second")
    if first.shape != second.shape:
        raise GradientConsensusError(
            "flattened gradients must have identical 1D shape"
        )
    if first.device != second.device:
        raise GradientConsensusError(
            "flattened gradients must be on the same device"
        )
    if first.dtype != second.dtype:
        raise GradientConsensusError(
            "flattened gradients must have the same dtype"
        )
    checked_min_norm = _finite_real(min_norm, name="min_norm", positive=True)
    checked_eps = _finite_real(eps, name="eps", positive=True)

    first_status, norm_a_tensor, norm_a = _branch_status(
        first, min_norm=checked_min_norm
    )
    second_status, norm_b_tensor, norm_b = _branch_status(
        second, min_norm=checked_min_norm
    )
    first_usable = first_status == "usable"
    second_usable = second_status == "usable"

    if not first_usable and not second_usable:
        if first_status == second_status == "tiny":
            decision = "reject_both_tiny"
        elif first_status == second_status == "nonfinite":
            decision = "reject_both_nonfinite"
        else:
            decision = "reject_no_usable_branch"
        return ConsensusResult(
            gradient=None,
            cosine=None,
            decision=decision,
            first_norm=norm_a,
            second_norm=norm_b,
            usable_branches=(),
            trial_order=(),
        )
    if not first_usable:
        return ConsensusResult(
            gradient=_sealed_direction(second),
            cosine=None,
            decision=(
                "use_second_only_first_nonfinite"
                if first_status == "nonfinite"
                else "use_second_only"
            ),
            first_norm=norm_a,
            second_norm=norm_b,
            usable_branches=("second",),
            trial_order=("second",),
        )
    if not second_usable:
        return ConsensusResult(
            gradient=_sealed_direction(first),
            cosine=None,
            decision=(
                "use_first_only_second_nonfinite"
                if second_status == "nonfinite"
                else "use_first_only"
            ),
            first_norm=norm_a,
            second_norm=norm_b,
            usable_branches=("first",),
            trial_order=("first",),
        )

    if norm_a_tensor is None or norm_b_tensor is None or norm_a is None or norm_b is None:
        raise RuntimeError("usable gradient branch lacks a finite norm")
    first_metric = _detached_float64(first)
    second_metric = _detached_float64(second)
    cosine_tensor = torch.dot(first_metric, second_metric) / (
        norm_a_tensor * norm_b_tensor + checked_eps
    )
    if not _finite_tensor(cosine_tensor):
        return ConsensusResult(
            gradient=None,
            cosine=None,
            decision="reject_nonfinite_combination",
            first_norm=norm_a,
            second_norm=norm_b,
            usable_branches=("first", "second"),
            trial_order=(),
        )
    cosine_value = float(torch.clamp(cosine_tensor, -1.0, 1.0).item())
    if cosine_value >= 0.0:
        unit_a = first.detach() / norm_a
        unit_b = second.detach() / norm_b
        combined = _sealed_direction(unit_a + unit_b)
        if not _finite_tensor(combined):
            return ConsensusResult(
                gradient=None,
                cosine=cosine_value,
                decision="reject_nonfinite_combination",
                first_norm=norm_a,
                second_norm=norm_b,
                usable_branches=("first", "second"),
                trial_order=(),
            )
        return ConsensusResult(
            gradient=combined,
            cosine=cosine_value,
            decision="aligned_average",
            first_norm=norm_a,
            second_norm=norm_b,
            usable_branches=("first", "second"),
            trial_order=("consensus",),
        )

    # The runner must evaluate these two candidates independently using the
    # same Source snapshot, loss-decrease rule, and two-sided safety closure.
    return ConsensusResult(
        gradient=None,
        cosine=cosine_value,
        decision="conflict_try_separately",
        first_norm=norm_a,
        second_norm=norm_b,
        usable_branches=("first", "second"),
        trial_order=("first", "second"),
    )


__all__ = [
    "ConsensusResult",
    "GradientConsensusError",
    "combine_two_gradients",
]
