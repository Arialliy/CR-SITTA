"""Detached, label-free two-sided safety checks for Stage-C backtracking.

The first side bounds foreground-probability inflation on source-defined
reliable background.  The second side prevents each source-defined candidate
from losing either absolute core response or core-versus-ring contrast.  This
module only evaluates a proposed state; it never mutates model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor

from metrics.connected_components import label_connected_components


class TwoSidedSafetyError(ValueError):
    """A structural input violates the Stage-C safety contract."""


@dataclass(frozen=True)
class CandidateSafetyAudit:
    """Detached scalar observations for one candidate core/ring pair."""

    index: int
    core_pixel_count: int
    ring_pixel_count: int
    absolute_response_pre: float | None
    absolute_response_post: float | None
    absolute_drop: float | None
    ring_mean_pre: float | None
    ring_mean_post: float | None
    contrast_pre: float | None
    contrast_post: float | None
    contrast_drop: float | None
    absolute_passed: bool
    contrast_passed: bool


@dataclass(frozen=True)
class SafetyDecision:
    """Fail-closed decision plus serialization-ready detached audit fields.

    Component counts and largest-component fractions are audit-only in v1:
    v6 did not freeze structural hard margins.  ``background_passed`` therefore
    continues to represent the registered background-mass bound only.
    """

    passed: bool
    reasons: tuple[str, ...]
    background_mass_pre: float | None
    background_mass_post: float | None
    background_mass_increase: float | None
    background_weight_sum: float
    background_passed: bool
    positive_fraction_pre: float | None
    positive_fraction_post: float | None
    component_count_pre: int | None
    component_count_post: int | None
    largest_component_fraction_pre: float | None
    largest_component_fraction_post: float | None
    structure_metrics_are_audit_only: bool
    reliable_background_quantiles_pre: tuple[float, ...]
    reliable_background_quantiles_post: tuple[float, ...]
    quantile_levels: tuple[float, ...]
    candidates: tuple[CandidateSafetyAudit, ...]
    temperature: float
    max_background_mass_increase: float
    max_absolute_drop: float
    max_contrast_drop: float

    @property
    def reason(self) -> str:
        """A deterministic adapter for single-string safety-closure APIs."""

        return "passed" if self.passed else ";".join(self.reasons)


def _finite_real(
    value: Real,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool")
    checked = float(value)
    if not math.isfinite(checked):
        raise TwoSidedSafetyError(f"{name} must be finite")
    if positive and checked <= 0.0:
        raise TwoSidedSafetyError(f"{name} must be strictly positive")
    if nonnegative and checked < 0.0:
        raise TwoSidedSafetyError(f"{name} must be non-negative")
    return checked


def _validate_spatial_tensor(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or value.shape[1] != 1:
        raise TwoSidedSafetyError(f"{name} must have shape [B,1,H,W]")
    if any(int(size) <= 0 for size in value.shape):
        raise TwoSidedSafetyError(f"{name} dimensions must all be positive")
    if value.layout != torch.strided or value.is_sparse:
        raise TwoSidedSafetyError(f"{name} must use strided dense layout")
    if not torch.is_floating_point(value) or value.is_complex():
        raise TypeError(f"{name} must be a real floating-point tensor")


def _require_same_layout(value: Tensor, reference: Tensor, *, name: str) -> None:
    if value.shape != reference.shape:
        raise TwoSidedSafetyError(f"{name} shape must match source_logits")
    if value.device != reference.device:
        raise TwoSidedSafetyError(f"{name} device must match source_logits")
    if value.dtype != reference.dtype:
        raise TwoSidedSafetyError(f"{name} dtype must match source_logits")


def _validate_background_weight(value: Tensor, reference: Tensor) -> None:
    _validate_spatial_tensor(value, name="background_weight")
    _require_same_layout(value, reference, name="background_weight")
    if value.requires_grad or value.grad_fn is not None:
        raise TwoSidedSafetyError(
            "background_weight must be detached from the adaptation graph"
        )
    detached = value.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise TwoSidedSafetyError(
            "background_weight must contain only finite values"
        )
    if bool(((detached < 0.0) | (detached > 1.0)).any().item()):
        raise TwoSidedSafetyError("background_weight values must lie in [0,1]")


def _validate_mask(mask: Tensor, reference: Tensor, *, name: str) -> None:
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if mask.dtype is not torch.bool:
        raise TypeError(f"{name} must be a precomputed boolean mask")
    if mask.layout != torch.strided or mask.is_sparse:
        raise TwoSidedSafetyError(f"{name} must use strided dense layout")
    if mask.shape != reference.shape:
        raise TwoSidedSafetyError(f"{name} shape must match source_logits")
    if mask.device != reference.device:
        raise TwoSidedSafetyError(f"{name} device must match source_logits")


def _finite_tensor(value: Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _temperature_lse(values: Tensor, *, temperature: float) -> float:
    working = values.detach().to(dtype=torch.float64)
    response = temperature * torch.logsumexp(working / temperature, dim=0)
    if not _finite_tensor(response):
        raise TwoSidedSafetyError("candidate LSE response became non-finite")
    return float(response.item())


def _detached_mean(values: Tensor, *, name: str) -> float:
    result = values.detach().to(dtype=torch.float64).mean()
    if not _finite_tensor(result):
        raise TwoSidedSafetyError(f"{name} became non-finite")
    return float(result.item())


def _background_quantiles(
    probability: Tensor,
    background_weight: Tensor,
    *,
    levels: tuple[float, ...],
) -> tuple[float, ...]:
    values = probability.detach()[background_weight.detach() > 0.0]
    if values.numel() == 0:
        return ()
    working = values.to(dtype=torch.float64)
    quantiles = torch.quantile(
        working,
        torch.tensor(levels, device=working.device, dtype=working.dtype),
    )
    if not _finite_tensor(quantiles):
        raise TwoSidedSafetyError("background probability quantiles are non-finite")
    return tuple(float(value) for value in quantiles.cpu().tolist())


def _predicted_structure(probability: Tensor) -> tuple[int, float]:
    """Return 8-connected count and largest image-area fraction for audit.

    Stage-C episodes use batch size one.  For defensive support of a larger
    batch, counts are summed and the largest per-image area fraction is kept;
    no value from this helper participates in the v1 hard safety decision.
    """

    binary = (probability.detach() > 0.5).cpu()
    height, width = (int(value) for value in binary.shape[-2:])
    image_area = height * width
    component_count = 0
    largest_fraction = 0.0
    for batch_index in range(int(binary.shape[0])):
        extraction = label_connected_components(
            binary[batch_index, 0].numpy(), connectivity=2, min_area=1
        )
        component_count += len(extraction.components)
        largest_area = max(
            (component.area for component in extraction.components), default=0
        )
        largest_fraction = max(largest_fraction, largest_area / image_area)
    return component_count, largest_fraction


def _empty_candidate_audit(
    *,
    index: int,
    core_pixel_count: int,
    ring_pixel_count: int,
) -> CandidateSafetyAudit:
    return CandidateSafetyAudit(
        index=index,
        core_pixel_count=core_pixel_count,
        ring_pixel_count=ring_pixel_count,
        absolute_response_pre=None,
        absolute_response_post=None,
        absolute_drop=None,
        ring_mean_pre=None,
        ring_mean_post=None,
        contrast_pre=None,
        contrast_post=None,
        contrast_drop=None,
        absolute_passed=False,
        contrast_passed=False,
    )


def check_two_sided_safety(
    source_logits: Tensor,
    proposed_logits: Tensor,
    *,
    background_weight: Tensor,
    candidate_cores: tuple[Tensor, ...],
    candidate_rings: tuple[Tensor, ...],
    max_background_mass_increase: float,
    max_absolute_drop: float,
    max_contrast_drop: float,
    temperature: float = 0.25,
    eps: float = 1.0e-6,
) -> SafetyDecision:
    """Evaluate one proposed logit map against immutable Source predictions.

    Candidate masks must be precomputed boolean BCHW tensors.  Empty candidate
    tuples are valid and execute background safety only.  An empty core or ring
    is instead an explicit failed decision, never a NaN-producing reduction.
    Every returned observation is a Python scalar detached from autograd.
    Predicted component count and largest-component fraction are reported for
    audit only; no structural hard margin is invented before registration.
    """

    _validate_spatial_tensor(source_logits, name="source_logits")
    _validate_spatial_tensor(proposed_logits, name="proposed_logits")
    _require_same_layout(proposed_logits, source_logits, name="proposed_logits")
    _validate_background_weight(background_weight, source_logits)
    if not isinstance(candidate_cores, tuple):
        raise TypeError("candidate_cores must be a tuple of boolean masks")
    if not isinstance(candidate_rings, tuple):
        raise TypeError("candidate_rings must be a tuple of boolean masks")
    if len(candidate_cores) != len(candidate_rings):
        raise TwoSidedSafetyError(
            "candidate_cores and candidate_rings must have identical length"
        )
    for index, core in enumerate(candidate_cores):
        _validate_mask(core, source_logits, name=f"candidate_cores[{index}]")
    for index, ring in enumerate(candidate_rings):
        _validate_mask(ring, source_logits, name=f"candidate_rings[{index}]")

    checked_background_margin = _finite_real(
        max_background_mass_increase,
        name="max_background_mass_increase",
        nonnegative=True,
    )
    checked_absolute_margin = _finite_real(
        max_absolute_drop,
        name="max_absolute_drop",
        nonnegative=True,
    )
    checked_contrast_margin = _finite_real(
        max_contrast_drop,
        name="max_contrast_drop",
        nonnegative=True,
    )
    checked_temperature = _finite_real(
        temperature, name="temperature", positive=True
    )
    checked_eps = _finite_real(eps, name="eps", positive=True)

    reasons: list[str] = []
    candidate_audits: list[CandidateSafetyAudit] = []
    source_finite = _finite_tensor(source_logits.detach())
    proposed_finite = _finite_tensor(proposed_logits.detach())
    if not source_finite:
        reasons.append("nonfinite_source_logits")
    if not proposed_finite:
        reasons.append("nonfinite_proposed_logits")

    background_sum_tensor = background_weight.detach().to(dtype=torch.float64).sum()
    if not _finite_tensor(background_sum_tensor):
        # This is unreachable after weight validation, but remains explicit in
        # case a backend reduction itself overflows.
        reasons.append("nonfinite_background_weight_sum")
        background_sum = 0.0
    else:
        background_sum = float(background_sum_tensor.item())
    if background_sum <= 0.0:
        reasons.append("empty_reliable_background")

    quantile_levels = (0.5, 0.9, 0.99)
    background_pre: float | None = None
    background_post: float | None = None
    background_increase: float | None = None
    positive_fraction_pre: float | None = None
    positive_fraction_post: float | None = None
    component_count_pre: int | None = None
    component_count_post: int | None = None
    largest_component_fraction_pre: float | None = None
    largest_component_fraction_post: float | None = None
    quantiles_pre: tuple[float, ...] = ()
    quantiles_post: tuple[float, ...] = ()
    background_passed = False

    source = source_logits.detach().to(dtype=torch.float64)
    proposed = proposed_logits.detach().to(dtype=torch.float64)
    weight = background_weight.detach().to(dtype=torch.float64)
    if source_finite and proposed_finite:
        probability_pre = torch.sigmoid(source)
        probability_post = torch.sigmoid(proposed)
        positive_fraction_pre = float((probability_pre > 0.5).double().mean().item())
        positive_fraction_post = float(
            (probability_post > 0.5).double().mean().item()
        )
        component_count_pre, largest_component_fraction_pre = (
            _predicted_structure(probability_pre)
        )
        component_count_post, largest_component_fraction_post = (
            _predicted_structure(probability_post)
        )
        if background_sum > 0.0:
            denominator = background_sum + checked_eps
            mass_pre_tensor = (probability_pre * weight).sum() / denominator
            mass_post_tensor = (probability_post * weight).sum() / denominator
            increase_tensor = mass_post_tensor - mass_pre_tensor
            if not all(
                _finite_tensor(value)
                for value in (mass_pre_tensor, mass_post_tensor, increase_tensor)
            ):
                reasons.append("nonfinite_background_mass")
            else:
                background_pre = float(mass_pre_tensor.item())
                background_post = float(mass_post_tensor.item())
                background_increase = float(increase_tensor.item())
                background_passed = (
                    background_increase <= checked_background_margin
                )
                if not background_passed:
                    reasons.append("background_mass_inflation")
                quantiles_pre = _background_quantiles(
                    probability_pre,
                    weight,
                    levels=quantile_levels,
                )
                quantiles_post = _background_quantiles(
                    probability_post,
                    weight,
                    levels=quantile_levels,
                )

    for index, (core, ring) in enumerate(
        zip(candidate_cores, candidate_rings, strict=True)
    ):
        core_count = int(torch.count_nonzero(core).item())
        ring_count = int(torch.count_nonzero(ring).item())
        invalid_geometry = False
        if core_count == 0:
            reasons.append(f"candidate_{index}_empty_core")
            invalid_geometry = True
        if ring_count == 0:
            reasons.append(f"candidate_{index}_empty_ring")
            invalid_geometry = True
        if bool((core & ring).any().item()):
            reasons.append(f"candidate_{index}_core_ring_overlap")
            invalid_geometry = True
        if invalid_geometry or not source_finite or not proposed_finite:
            candidate_audits.append(
                _empty_candidate_audit(
                    index=index,
                    core_pixel_count=core_count,
                    ring_pixel_count=ring_count,
                )
            )
            continue

        absolute_pre = _temperature_lse(
            source[core], temperature=checked_temperature
        )
        absolute_post = _temperature_lse(
            proposed[core], temperature=checked_temperature
        )
        absolute_drop = absolute_pre - absolute_post
        ring_pre = _detached_mean(source[ring], name="source candidate ring mean")
        ring_post = _detached_mean(
            proposed[ring], name="proposed candidate ring mean"
        )
        contrast_pre = absolute_pre - ring_pre
        contrast_post = absolute_post - ring_post
        contrast_drop = contrast_pre - contrast_post
        absolute_passed = absolute_drop <= checked_absolute_margin
        contrast_passed = contrast_drop <= checked_contrast_margin
        if not absolute_passed:
            reasons.append(f"candidate_{index}_absolute_drop")
        if not contrast_passed:
            reasons.append(f"candidate_{index}_contrast_drop")
        candidate_audits.append(
            CandidateSafetyAudit(
                index=index,
                core_pixel_count=core_count,
                ring_pixel_count=ring_count,
                absolute_response_pre=absolute_pre,
                absolute_response_post=absolute_post,
                absolute_drop=absolute_drop,
                ring_mean_pre=ring_pre,
                ring_mean_post=ring_post,
                contrast_pre=contrast_pre,
                contrast_post=contrast_post,
                contrast_drop=contrast_drop,
                absolute_passed=absolute_passed,
                contrast_passed=contrast_passed,
            )
        )

    return SafetyDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        background_mass_pre=background_pre,
        background_mass_post=background_post,
        background_mass_increase=background_increase,
        background_weight_sum=background_sum,
        background_passed=background_passed,
        positive_fraction_pre=positive_fraction_pre,
        positive_fraction_post=positive_fraction_post,
        component_count_pre=component_count_pre,
        component_count_post=component_count_post,
        largest_component_fraction_pre=largest_component_fraction_pre,
        largest_component_fraction_post=largest_component_fraction_post,
        structure_metrics_are_audit_only=True,
        reliable_background_quantiles_pre=quantiles_pre,
        reliable_background_quantiles_post=quantiles_post,
        quantile_levels=quantile_levels,
        candidates=tuple(candidate_audits),
        temperature=checked_temperature,
        max_background_mass_increase=checked_background_margin,
        max_absolute_drop=checked_absolute_margin,
        max_contrast_drop=checked_contrast_margin,
    )


__all__ = [
    "CandidateSafetyAudit",
    "SafetyDecision",
    "TwoSidedSafetyError",
    "check_two_sided_safety",
]
