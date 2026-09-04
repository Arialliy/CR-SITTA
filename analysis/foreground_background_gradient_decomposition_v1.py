"""Pure Stage-B1 foreground/background entropy-gradient decomposition.

The caller supplies Source logits from one already-completed forward pass,
the frozen P0 parameter layout, the P0--P4 parameter-name subsets, the
label-free parent full-image entropy gradient, and the outer-oracle task
gradient.  This module performs no model forward, file access, optimizer
construction, parameter update, or ``Tensor.backward`` call.

The only autograd basis evaluated here is frozen as

``[fg_subthreshold_add, fg_suprathreshold_add, bg_add]``.

Every additive loss uses the *full-image* pixel count as denominator.  The
foreground and full-image vectors are sums of those basis vectors; regional
conditional means are derived by division by the corresponding pixel
fraction.  An empty region therefore has no conditional-mean gradient and is
reported as ``not_estimable`` rather than as a fabricated zero vector or
zero cosine.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Final, Literal

import torch
from torch import Tensor

from analysis.d0_v3_outer_analyzer import FlatParameterLayout
from tta.binary_tent import binary_entropy_map


SCHEMA_VERSION: Final = 1
ANALYSIS_TYPE: Final = "cr_sitta_stage_b1_fg_bg_gradient_decomposition"
GROUP_IDS: Final = ("P0", "P1", "P2", "P3", "P4")
BACKWARD_BASIS: Final = (
    "foreground_subthreshold_add",
    "foreground_suprathreshold_add",
    "background_add",
)
FOREGROUND_RULE: Final = "target>0"
BACKGROUND_RULE: Final = "target==0"
SUBTHRESHOLD_RULE: Final = "sigmoid(source_logits)<=0.5"
SUPRATHRESHOLD_RULE: Final = "sigmoid(source_logits)>0.5"


class ForegroundBackgroundGradientError(ValueError):
    """An input or computed decomposition violates the frozen B1 contract."""


def _finite_real(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ForegroundBackgroundGradientError(
            f"{field} must be a real number, not bool"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ForegroundBackgroundGradientError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise ForegroundBackgroundGradientError(f"{field} must be positive")
    if not positive and result < 0.0:
        raise ForegroundBackgroundGradientError(
            f"{field} must be non-negative"
        )
    return result


@dataclass(frozen=True)
class GradientDecompositionConfig:
    """Immutable numeric conventions for one B1 decomposition."""

    entropy_eps: float = 1.0e-6
    parent_entropy_max_abs_tolerance: float = 1.0e-7
    parent_entropy_relative_l2_tolerance: float = 1.0e-4
    cosine_zero_norm_tolerance: float = 0.0

    def __post_init__(self) -> None:
        eps = _finite_real(self.entropy_eps, field="entropy_eps", positive=True)
        if eps >= 0.5:
            raise ForegroundBackgroundGradientError(
                "entropy_eps must be strictly less than 0.5"
            )
        max_abs = _finite_real(
            self.parent_entropy_max_abs_tolerance,
            field="parent_entropy_max_abs_tolerance",
        )
        relative = _finite_real(
            self.parent_entropy_relative_l2_tolerance,
            field="parent_entropy_relative_l2_tolerance",
        )
        zero = _finite_real(
            self.cosine_zero_norm_tolerance,
            field="cosine_zero_norm_tolerance",
        )
        object.__setattr__(self, "entropy_eps", eps)
        object.__setattr__(self, "parent_entropy_max_abs_tolerance", max_abs)
        object.__setattr__(
            self, "parent_entropy_relative_l2_tolerance", relative
        )
        object.__setattr__(self, "cosine_zero_norm_tolerance", zero)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GradientDecompositionConfig":
        if not isinstance(value, Mapping):
            raise ForegroundBackgroundGradientError("config must be a mapping")
        expected = {
            "entropy_eps",
            "parent_entropy_max_abs_tolerance",
            "parent_entropy_relative_l2_tolerance",
            "cosine_zero_norm_tolerance",
        }
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected, key=str)
        if missing or unknown:
            raise ForegroundBackgroundGradientError(
                "config fields must be exact; "
                f"missing={missing}, unknown={unknown}"
            )
        return cls(
            entropy_eps=value["entropy_eps"],
            parent_entropy_max_abs_tolerance=value[
                "parent_entropy_max_abs_tolerance"
            ],
            parent_entropy_relative_l2_tolerance=value[
                "parent_entropy_relative_l2_tolerance"
            ],
            cosine_zero_norm_tolerance=value["cosine_zero_norm_tolerance"],
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "entropy_eps": self.entropy_eps,
            "parent_entropy_max_abs_tolerance": (
                self.parent_entropy_max_abs_tolerance
            ),
            "parent_entropy_relative_l2_tolerance": (
                self.parent_entropy_relative_l2_tolerance
            ),
            "cosine_zero_norm_tolerance": self.cosine_zero_norm_tolerance,
        }


@dataclass(frozen=True)
class GradientDecompositionVectors:
    """Detached CPU float64 vectors produced by one decomposition."""

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


@dataclass(frozen=True)
class ForegroundBackgroundGradientResult:
    """JSON-safe report plus detached vectors for later train-only aggregation."""

    report: Mapping[str, Any]
    vectors: GradientDecompositionVectors

    def to_dict(self) -> dict[str, Any]:
        """Return a defensive JSON-safe copy; raw vectors remain separate."""

        return deepcopy(dict(self.report))


def _validated_config(
    value: GradientDecompositionConfig | Mapping[str, Any],
) -> GradientDecompositionConfig:
    if isinstance(value, GradientDecompositionConfig):
        return GradientDecompositionConfig.from_mapping(value.to_dict())
    if isinstance(value, Mapping):
        return GradientDecompositionConfig.from_mapping(value)
    raise ForegroundBackgroundGradientError(
        "config must be GradientDecompositionConfig or an exact mapping"
    )


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ForegroundBackgroundGradientError(
            "group mapping is not canonical-JSON safe"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _validate_logits_target(source_logits: Tensor, target: Tensor) -> None:
    for value, name in ((source_logits, "source_logits"), (target, "target")):
        if not isinstance(value, Tensor):
            raise ForegroundBackgroundGradientError(
                f"{name} must be a torch.Tensor"
            )
        if value.ndim != 4 or tuple(value.shape[:2]) != (1, 1):
            raise ForegroundBackgroundGradientError(
                f"{name} must have shape [1,1,H,W]"
            )
        if value.shape[-2] <= 0 or value.shape[-1] <= 0:
            raise ForegroundBackgroundGradientError(
                f"{name} spatial dimensions must be positive"
            )
        if not torch.is_floating_point(value) or value.is_complex():
            raise ForegroundBackgroundGradientError(
                f"{name} must be real floating-point"
            )
        if value.dtype != torch.float32:
            raise ForegroundBackgroundGradientError(
                f"{name} must use the frozen torch.float32 dtype"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ForegroundBackgroundGradientError(
                f"{name} must contain only finite values"
            )
    if source_logits.shape != target.shape:
        raise ForegroundBackgroundGradientError(
            "source_logits and target shapes must match exactly"
        )
    if source_logits.device != target.device:
        raise ForegroundBackgroundGradientError(
            "source_logits and target devices must match"
        )
    if not source_logits.requires_grad or source_logits.grad_fn is None:
        raise ForegroundBackgroundGradientError(
            "source_logits must retain the Source forward autograd graph"
        )
    if target.requires_grad:
        raise ForegroundBackgroundGradientError("target must not require gradients")
    if bool((target < 0).any().item()) or bool((target > 1).any().item()):
        raise ForegroundBackgroundGradientError(
            "target values must lie in the closed interval [0,1]"
        )


def _validate_named_parameters(
    *,
    named_parameters: Mapping[str, Tensor],
    parameter_layout: FlatParameterLayout,
    device: torch.device,
) -> tuple[Tensor, ...]:
    if not isinstance(parameter_layout, FlatParameterLayout):
        raise ForegroundBackgroundGradientError(
            "parameter_layout must be FlatParameterLayout"
        )
    if not isinstance(named_parameters, Mapping):
        raise ForegroundBackgroundGradientError(
            "named_parameters must be an ordered mapping"
        )
    if tuple(named_parameters) != parameter_layout.names:
        raise ForegroundBackgroundGradientError(
            "named_parameters order/topology differs from parameter_layout"
        )
    parameters: list[Tensor] = []
    identities: set[int] = set()
    for name, expected_shape in zip(
        parameter_layout.names, parameter_layout.shapes, strict=True
    ):
        parameter = named_parameters[name]
        if not isinstance(parameter, Tensor):
            raise ForegroundBackgroundGradientError(
                f"named_parameters[{name!r}] must be a torch.Tensor"
            )
        if id(parameter) in identities:
            raise ForegroundBackgroundGradientError(
                "named_parameters contains duplicate tensor objects"
            )
        identities.add(id(parameter))
        if tuple(parameter.shape) != expected_shape:
            raise ForegroundBackgroundGradientError(
                f"named_parameters[{name!r}] shape differs from parameter_layout"
            )
        if (
            parameter.dtype != torch.float32
            or parameter.device != device
            or not parameter.requires_grad
            or not parameter.is_leaf
        ):
            raise ForegroundBackgroundGradientError(
                f"named_parameters[{name!r}] must be a leaf float32 tensor "
                "requiring gradients on the Source-logit device"
            )
        if not bool(torch.isfinite(parameter).all().item()):
            raise ForegroundBackgroundGradientError(
                f"named_parameters[{name!r}] contains NaN/Inf"
            )
        parameters.append(parameter)
    if sum(parameter.numel() for parameter in parameters) != parameter_layout.scalar_count:
        raise ForegroundBackgroundGradientError(
            "named parameter scalar count differs from parameter_layout"
        )
    return tuple(parameters)


def _validate_group_parameter_names(
    *,
    parameter_layout: FlatParameterLayout,
    group_parameter_names: Mapping[str, Sequence[str]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, Tensor]]:
    if not isinstance(group_parameter_names, Mapping):
        raise ForegroundBackgroundGradientError(
            "group_parameter_names must be a mapping"
        )
    if tuple(group_parameter_names) != GROUP_IDS:
        raise ForegroundBackgroundGradientError(
            "group_parameter_names keys/order must be exactly P0,P1,P2,P3,P4"
        )
    layout_names = parameter_layout.names
    layout_set = set(layout_names)
    normalized: dict[str, tuple[str, ...]] = {}
    indices: dict[str, Tensor] = {}
    offsets = parameter_layout.offsets
    ends = (*offsets[1:], parameter_layout.scalar_count)
    spans = dict(zip(layout_names, zip(offsets, ends, strict=True), strict=True))
    for group_id in GROUP_IDS:
        raw = group_parameter_names[group_id]
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise ForegroundBackgroundGradientError(
                f"group_parameter_names[{group_id!r}] must be a sequence"
            )
        names = tuple(raw)
        if not names or not all(isinstance(name, str) and name for name in names):
            raise ForegroundBackgroundGradientError(
                f"group_parameter_names[{group_id!r}] must be non-empty names"
            )
        if len(set(names)) != len(names) or not set(names).issubset(layout_set):
            raise ForegroundBackgroundGradientError(
                f"group_parameter_names[{group_id!r}] has duplicate/unknown names"
            )
        canonical = tuple(name for name in layout_names if name in set(names))
        if names != canonical:
            raise ForegroundBackgroundGradientError(
                f"group_parameter_names[{group_id!r}] is not in layout order"
            )
        normalized[group_id] = names
        indices[group_id] = torch.cat(
            [torch.arange(*spans[name], dtype=torch.int64) for name in names]
        )
    if normalized["P0"] != layout_names:
        raise ForegroundBackgroundGradientError(
            "P0 must contain every parameter in the frozen layout"
        )
    for smaller, larger in zip(GROUP_IDS[1:-1], GROUP_IDS[2:], strict=True):
        if not set(normalized[smaller]).issubset(normalized[larger]):
            raise ForegroundBackgroundGradientError(
                f"frozen nested-group invariant failed: {smaller} is not in {larger}"
            )
    if not set(normalized["P4"]).issubset(normalized["P0"]):
        raise ForegroundBackgroundGradientError("P4 must be a subset of P0")
    return normalized, indices


def _validate_external_vector(
    value: Tensor,
    *,
    label: str,
    scalar_count: int,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ForegroundBackgroundGradientError(f"{label} must be a torch.Tensor")
    if (
        value.ndim != 1
        or value.numel() != scalar_count
        or not torch.is_floating_point(value)
        or value.is_complex()
    ):
        raise ForegroundBackgroundGradientError(
            f"{label} must be a real floating vector [{scalar_count}]"
        )
    if value.requires_grad:
        raise ForegroundBackgroundGradientError(
            f"{label} must be detached external evidence"
        )
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise ForegroundBackgroundGradientError(f"{label} contains NaN/Inf")
    return result


def _snapshot_grad_slots(parameters: Sequence[Tensor]) -> tuple[Tensor | None, ...]:
    return tuple(
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    )


def _grad_slots_unchanged(
    parameters: Sequence[Tensor], before: Sequence[Tensor | None]
) -> bool:
    for parameter, old in zip(parameters, before, strict=True):
        current = parameter.grad
        if old is None:
            if current is not None:
                return False
        elif current is None or not torch.equal(current.detach(), old):
            return False
    return True


def _flatten_gradients(
    gradients: Sequence[Tensor | None],
    *,
    label: str,
    scalar_count: int,
) -> Tensor:
    if any(gradient is None for gradient in gradients):
        raise ForegroundBackgroundGradientError(
            f"{label} has an unused P0 parameter"
        )
    materialized = tuple(gradient for gradient in gradients if gradient is not None)
    if any(not bool(torch.isfinite(value).all().item()) for value in materialized):
        raise ForegroundBackgroundGradientError(f"{label} contains NaN/Inf")
    result = torch.cat(
        [value.detach().reshape(-1).to(device="cpu", dtype=torch.float64) for value in materialized]
    ).contiguous()
    if result.numel() != scalar_count:
        raise ForegroundBackgroundGradientError(
            f"{label} scalar count differs from parameter_layout"
        )
    return result


def _autograd_flat(
    loss: Tensor,
    parameters: Sequence[Tensor],
    *,
    label: str,
    scalar_count: int,
    retain_graph: bool,
) -> Tensor:
    try:
        gradients = torch.autograd.grad(
            loss,
            tuple(parameters),
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=False,
            materialize_grads=False,
        )
    except (RuntimeError, ValueError) as exc:
        raise ForegroundBackgroundGradientError(
            f"unable to compute {label} from the shared Source graph"
        ) from exc
    return _flatten_gradients(
        gradients, label=label, scalar_count=scalar_count
    )


def _residual_metrics(
    reference: Tensor,
    reconstructed: Tensor,
    *,
    relative_denominator_floor: float = 1.0e-12,
) -> dict[str, float]:
    residual = reconstructed - reference
    max_abs = float(torch.max(torch.abs(residual)).item())
    l2 = float(torch.linalg.vector_norm(residual).item())
    reference_l2 = float(torch.linalg.vector_norm(reference).item())
    reconstructed_l2 = float(torch.linalg.vector_norm(reconstructed).item())
    # The frozen parent comparison is relative to the sealed parent vector,
    # never to whichever of the two vectors happens to be larger.
    denominator = max(reference_l2, relative_denominator_floor)
    relative_l2 = l2 / denominator
    values = (max_abs, l2, reference_l2, reconstructed_l2, relative_l2)
    if not all(math.isfinite(value) for value in values):
        raise ForegroundBackgroundGradientError(
            "gradient decomposition residual is non-finite"
        )
    return {
        "max_abs": max_abs,
        "l2": l2,
        "reference_l2": reference_l2,
        "reconstructed_l2": reconstructed_l2,
        "relative_l2_denominator": denominator,
        "relative_l2_denominator_floor": relative_denominator_floor,
        "relative_l2": relative_l2,
    }


def _alignment(
    entropy_gradient: Tensor | None,
    task_gradient: Tensor,
    indices: Tensor,
    *,
    absent_reason: str | None,
    zero_tolerance: float,
) -> dict[str, Any]:
    task = task_gradient[indices]
    task_norm = float(torch.linalg.vector_norm(task).item())
    if entropy_gradient is None:
        return {
            "estimable": False,
            "not_estimable_reason": absent_reason,
            "entropy_gradient_norm": None,
            "task_gradient_norm": task_norm,
            "entropy_task_dot": None,
            "entropy_task_cosine": None,
            "task_projection": None,
            "unit_descent_task_change": None,
            "direction": "undefined",
            "cosine_status": "not_estimable_region_absent",
        }
    entropy = entropy_gradient[indices]
    entropy_norm = float(torch.linalg.vector_norm(entropy).item())
    dot = float(torch.dot(entropy, task).item())
    if task_norm <= zero_tolerance:
        task_projection: float | None = None
    else:
        task_projection = dot / task_norm
    if entropy_norm <= zero_tolerance:
        unit_descent_task_change: float | None = None
    else:
        unit_descent_task_change = -dot / entropy_norm

    if entropy_norm <= zero_tolerance and task_norm <= zero_tolerance:
        cosine: float | None = None
        status = "not_estimable_both_gradients_zero"
    elif entropy_norm <= zero_tolerance:
        cosine = None
        status = "not_estimable_entropy_gradient_zero"
    elif task_norm <= zero_tolerance:
        cosine = None
        status = "not_estimable_task_gradient_zero"
    else:
        raw = torch.dot(entropy, task) / (entropy_norm * task_norm)
        cosine = float(torch.clamp(raw, -1.0, 1.0).item())
        status = "estimable"
    if entropy_norm <= zero_tolerance or task_norm <= zero_tolerance:
        direction: Literal["beneficial", "harmful", "neutral", "undefined"] = (
            "undefined"
        )
    elif unit_descent_task_change is not None and unit_descent_task_change < 0.0:
        direction = "beneficial"
    elif unit_descent_task_change is not None and unit_descent_task_change > 0.0:
        direction = "harmful"
    else:
        direction = "neutral"
    values = (
        entropy_norm,
        task_norm,
        dot,
        *(tuple() if task_projection is None else (task_projection,)),
        *(
            tuple()
            if unit_descent_task_change is None
            else (unit_descent_task_change,)
        ),
    )
    if not all(math.isfinite(value) for value in values) or (
        cosine is not None and not math.isfinite(cosine)
    ):
        raise ForegroundBackgroundGradientError(
            "gradient alignment produced NaN/Inf"
        )
    return {
        "estimable": True,
        "not_estimable_reason": None,
        "entropy_gradient_norm": entropy_norm,
        "task_gradient_norm": task_norm,
        "entropy_task_dot": dot,
        "entropy_task_cosine": cosine,
        "task_projection": task_projection,
        "unit_descent_task_change": unit_descent_task_change,
        "direction": direction,
        "cosine_status": status,
    }


def _pair_metrics(
    left: Tensor | None,
    right: Tensor | None,
    indices: Tensor,
    *,
    left_absent_reason: str | None,
    right_absent_reason: str | None,
    zero_tolerance: float,
) -> dict[str, Any]:
    if left is None or right is None:
        reasons = [
            value
            for value in (left_absent_reason, right_absent_reason)
            if value is not None
        ]
        return {
            "estimable": False,
            "not_estimable_reason": "+".join(reasons),
            "foreground_gradient_norm": None,
            "background_gradient_norm": None,
            "foreground_background_dot": None,
            "foreground_background_cosine": None,
            "cosine_status": "not_estimable_region_absent",
        }
    left_part = left[indices]
    right_part = right[indices]
    left_norm = float(torch.linalg.vector_norm(left_part).item())
    right_norm = float(torch.linalg.vector_norm(right_part).item())
    dot = float(torch.dot(left_part, right_part).item())
    if left_norm <= zero_tolerance and right_norm <= zero_tolerance:
        cosine = None
        status = "not_estimable_both_gradients_zero"
    elif left_norm <= zero_tolerance:
        cosine = None
        status = "not_estimable_foreground_gradient_zero"
    elif right_norm <= zero_tolerance:
        cosine = None
        status = "not_estimable_background_gradient_zero"
    else:
        cosine = float(
            torch.clamp(
                torch.dot(left_part, right_part) / (left_norm * right_norm),
                -1.0,
                1.0,
            ).item()
        )
        status = "estimable"
    values = (left_norm, right_norm, dot)
    if not all(math.isfinite(value) for value in values) or (
        cosine is not None and not math.isfinite(cosine)
    ):
        raise ForegroundBackgroundGradientError(
            "foreground/background alignment produced NaN/Inf"
        )
    return {
        "estimable": True,
        "not_estimable_reason": None,
        "foreground_gradient_norm": left_norm,
        "background_gradient_norm": right_norm,
        "foreground_background_dot": dot,
        "foreground_background_cosine": cosine,
        "cosine_status": status,
    }


def _ratio(
    numerator: float | None,
    denominator: float | None,
    *,
    zero_tolerance: float,
    absent_reason: str | None,
) -> dict[str, Any]:
    if numerator is None or denominator is None:
        return {
            "value": None,
            "status": "not_estimable_region_absent",
            "not_estimable_reason": absent_reason,
        }
    if denominator <= zero_tolerance:
        return {
            "value": None,
            "status": "not_estimable_denominator_zero",
            "not_estimable_reason": "foreground_gradient_zero",
        }
    value = numerator / denominator
    if not math.isfinite(value):
        raise ForegroundBackgroundGradientError("gradient norm ratio is non-finite")
    return {"value": value, "status": "estimable", "not_estimable_reason": None}


def _optional_conditional(
    additive: Tensor,
    *,
    pixel_count: int,
    total_pixel_count: int,
) -> Tensor | None:
    if pixel_count == 0:
        return None
    return (additive * (float(total_pixel_count) / float(pixel_count))).contiguous()


def analyze_foreground_background_gradient_decomposition(
    *,
    source_logits: Tensor,
    target: Tensor,
    named_parameters: Mapping[str, Tensor],
    parameter_layout: FlatParameterLayout,
    group_parameter_names: Mapping[str, Sequence[str]],
    parent_entropy_gradient_flat: Tensor,
    task_gradient_flat: Tensor,
    config: GradientDecompositionConfig | Mapping[str, Any] = GradientDecompositionConfig(),
) -> ForegroundBackgroundGradientResult:
    """Decompose one Source entropy gradient without updating any parameter.

    ``parent_entropy_gradient_flat`` is the sealed label-free full-image mean
    entropy gradient from the parent episode.  ``task_gradient_flat`` is the
    existing outer-oracle BCE+soft-IoU task gradient; this function never
    reconstructs or differentiates that supervised loss.
    """

    frozen_config = _validated_config(config)
    _validate_logits_target(source_logits, target)
    parameters = _validate_named_parameters(
        named_parameters=named_parameters,
        parameter_layout=parameter_layout,
        device=source_logits.device,
    )
    normalized_groups, group_indices = _validate_group_parameter_names(
        parameter_layout=parameter_layout,
        group_parameter_names=group_parameter_names,
    )
    parent_entropy = _validate_external_vector(
        parent_entropy_gradient_flat,
        label="parent_entropy_gradient_flat",
        scalar_count=parameter_layout.scalar_count,
    )
    task = _validate_external_vector(
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
    bg_loss = (entropy * background.to(dtype=mask_dtype)).sum() / total_pixels
    full_direct_loss = entropy.mean()
    loss_reconstruction = fg_sub_loss + fg_supra_loss + bg_loss
    loss_residual = abs(
        float((loss_reconstruction - full_direct_loss).detach().item())
    )
    if (
        not math.isfinite(loss_residual)
        or loss_residual > frozen_config.parent_entropy_max_abs_tolerance
    ):
        raise ForegroundBackgroundGradientError(
            "additive entropy losses do not reconstruct full-image mean"
        )

    grad_slots_before = _snapshot_grad_slots(parameters)
    versions_before = tuple(parameter._version for parameter in parameters)
    fg_sub_add = _autograd_flat(
        fg_sub_loss,
        parameters,
        label="foreground_subthreshold_add",
        scalar_count=parameter_layout.scalar_count,
        retain_graph=True,
    )
    fg_supra_add = _autograd_flat(
        fg_supra_loss,
        parameters,
        label="foreground_suprathreshold_add",
        scalar_count=parameter_layout.scalar_count,
        retain_graph=True,
    )
    bg_add = _autograd_flat(
        bg_loss,
        parameters,
        label="background_add",
        scalar_count=parameter_layout.scalar_count,
        retain_graph=False,
    )
    if not _grad_slots_unchanged(parameters, grad_slots_before):
        raise ForegroundBackgroundGradientError(
            "autograd.grad modified a parameter .grad slot"
        )
    versions_after = tuple(parameter._version for parameter in parameters)
    if versions_after != versions_before:
        raise ForegroundBackgroundGradientError(
            "parameter values changed during pure gradient decomposition"
        )

    foreground_add = fg_sub_add + fg_supra_add
    full_add = foreground_add + bg_add
    foreground_conditional = _optional_conditional(
        foreground_add,
        pixel_count=foreground_pixels,
        total_pixel_count=total_pixels,
    )
    background_conditional = _optional_conditional(
        bg_add,
        pixel_count=background_pixels,
        total_pixel_count=total_pixels,
    )
    fg_sub_conditional = _optional_conditional(
        fg_sub_add,
        pixel_count=subthreshold_pixels,
        total_pixel_count=total_pixels,
    )
    fg_supra_conditional = _optional_conditional(
        fg_supra_add,
        pixel_count=suprathreshold_pixels,
        total_pixel_count=total_pixels,
    )

    weighted = torch.zeros_like(full_add)
    if foreground_conditional is not None:
        weighted = weighted + (foreground_pixels / total_pixels) * foreground_conditional
    if background_conditional is not None:
        weighted = weighted + (background_pixels / total_pixels) * background_conditional
    weighted_metrics = _residual_metrics(full_add, weighted)
    weighted_verified = (
        weighted_metrics["max_abs"]
        <= frozen_config.parent_entropy_max_abs_tolerance
        and weighted_metrics["relative_l2"]
        <= frozen_config.parent_entropy_relative_l2_tolerance
    )
    if not weighted_verified:
        raise ForegroundBackgroundGradientError(
            "weighted conditional gradients do not reconstruct full entropy"
        )

    parent_metrics = _residual_metrics(parent_entropy, full_add)
    parent_verified = (
        parent_metrics["max_abs"]
        <= frozen_config.parent_entropy_max_abs_tolerance
        and parent_metrics["relative_l2"]
        <= frozen_config.parent_entropy_relative_l2_tolerance
    )
    if not parent_verified:
        raise ForegroundBackgroundGradientError(
            "reconstructed full entropy gradient differs from parent evidence; "
            f"max_abs={parent_metrics['max_abs']:.17g}, "
            f"max_abs_tolerance={frozen_config.parent_entropy_max_abs_tolerance:.17g}, "
            f"relative_l2={parent_metrics['relative_l2']:.17g}, "
            "relative_l2_tolerance="
            f"{frozen_config.parent_entropy_relative_l2_tolerance:.17g}, "
            f"reference_l2={parent_metrics['reference_l2']:.17g}"
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
            float(
                torch.linalg.vector_norm(foreground_conditional[indices]).item()
            )
            if foreground_conditional is not None
            else None
        )
        background_conditional_norm = (
            float(
                torch.linalg.vector_norm(background_conditional[indices]).item()
            )
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
            # Triangle inequality bounds this expression in [0,1].  Clamp only
            # sub-ULP noise so the serialized diagnostic remains interpretable.
            cancellation_value = min(1.0, max(0.0, cancellation_value))
            cancellation_status = "estimable"
            cancellation_reason = None
        per_group[group_id] = {
            "parameter_tensor_count": len(normalized_groups[group_id]),
            "parameter_scalar_count": int(indices.numel()),
            "task_gradient_norm": task_norm,
            "additive_entropy_task_alignment": {
                name: _alignment(
                    value,
                    task,
                    indices,
                    absent_reason=reason,
                    zero_tolerance=frozen_config.cosine_zero_norm_tolerance,
                )
                for name, (value, reason) in additive_alignment_vectors.items()
            },
            "conditional_entropy_task_alignment": {
                name: _alignment(
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
                "foreground_background_additive_alignment": _pair_metrics(
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
                "background_to_foreground_additive_norm_ratio": _ratio(
                    background_add_norm,
                    foreground_add_norm,
                    zero_tolerance=frozen_config.cosine_zero_norm_tolerance,
                    absent_reason=(
                        "empty_foreground_or_background"
                        if not foreground_pixels or not background_pixels
                        else None
                    ),
                ),
                "background_to_foreground_conditional_norm_ratio": _ratio(
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
            "foreground_add": float((fg_sub_loss + fg_supra_loss).detach().item()),
            "background_add": float(bg_loss.detach().item()),
            "full_add": float(loss_reconstruction.detach().item()),
        },
        "conditional_mean": {
            "foreground": (
                float(
                    ((fg_sub_loss + fg_supra_loss) * total_pixels / foreground_pixels)
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
            "additive_denominator": "full_image_pixel_count",
            "conditional_means_are_derived": True,
            "foreground_rule": FOREGROUND_RULE,
            "background_rule": BACKGROUND_RULE,
            "foreground_subthreshold_rule": SUBTHRESHOLD_RULE,
            "foreground_suprathreshold_rule": SUPRATHRESHOLD_RULE,
            "parent_entropy_role": "sealed_label_free_full_image_mean_gradient",
            "task_gradient_role": "external_existing_bce_plus_soft_iou_outer_oracle_gradient",
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
            "group_parameter_names_sha256": _canonical_sha256(group_descriptor),
        },
        "target_statistics": target_statistics,
        "losses": losses,
        "decomposition": {
            "foreground_add_is_sub_plus_supra": True,
            "full_add_is_foreground_plus_background": True,
            "weighted_conditional_reconstruction": {
                **weighted_metrics,
                "max_abs_tolerance": frozen_config.parent_entropy_max_abs_tolerance,
                "relative_l2_tolerance": (
                    frozen_config.parent_entropy_relative_l2_tolerance
                ),
                "verified": weighted_verified,
                "foreground_weight": foreground_pixels / total_pixels,
                "background_weight": background_pixels / total_pixels,
            },
            "parent_full_entropy_reconstruction": {
                **parent_metrics,
                "max_abs_tolerance": frozen_config.parent_entropy_max_abs_tolerance,
                "relative_l2_tolerance": (
                    frozen_config.parent_entropy_relative_l2_tolerance
                ),
                "verified": parent_verified,
            },
        },
        "per_group": per_group,
        "finiteness": {
            "source_logits": True,
            "target": True,
            "entropy_map": True,
            "additive_gradients": True,
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
    # Enforce the advertised JSON-safe schema before returning it.
    json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    vectors = GradientDecompositionVectors(
        foreground_subthreshold_add=fg_sub_add.clone(),
        foreground_suprathreshold_add=fg_supra_add.clone(),
        background_add=bg_add.clone(),
        foreground_add=foreground_add.clone(),
        full_add=full_add.clone(),
        foreground_conditional_mean=(
            None if foreground_conditional is None else foreground_conditional.clone()
        ),
        background_conditional_mean=(
            None if background_conditional is None else background_conditional.clone()
        ),
        foreground_subthreshold_conditional_mean=(
            None if fg_sub_conditional is None else fg_sub_conditional.clone()
        ),
        foreground_suprathreshold_conditional_mean=(
            None if fg_supra_conditional is None else fg_supra_conditional.clone()
        ),
        parent_entropy=parent_entropy.clone(),
        task=task.clone(),
    )
    return ForegroundBackgroundGradientResult(report=report, vectors=vectors)


# Descriptive alias for callers that prefer a compute-style API.
compute_foreground_background_gradient_decomposition = (
    analyze_foreground_background_gradient_decomposition
)


__all__ = [
    "ANALYSIS_TYPE",
    "BACKWARD_BASIS",
    "BACKGROUND_RULE",
    "FOREGROUND_RULE",
    "GROUP_IDS",
    "SCHEMA_VERSION",
    "GradientDecompositionConfig",
    "GradientDecompositionVectors",
    "ForegroundBackgroundGradientError",
    "ForegroundBackgroundGradientResult",
    "analyze_foreground_background_gradient_decomposition",
    "compute_foreground_background_gradient_decomposition",
]
