"""Train-only outer-oracle task loss for D0-v2 diagnostics.

This loss exists only to measure supervised task-gradient alignment on the
frozen source-training Pilot64.  Ground-truth targets passed here must never
be visible to, or used to update, an adaptation method.  The mandatory
``SourceTrainAnalysisProvenance`` receipt makes that boundary explicit and
rejects validation/test evidence.

The frozen loss definition is

``lambda_bce * BCEWithLogits(mean) + lambda_soft_iou * (1 - soft IoU)``.

Soft IoU is computed per sample and then averaged.  D0-v2 admits only a
single image, so this reduction is unambiguous.  For an empty target the same
epsilon-smoothed formula is used: ``IoU = eps / (sum(sigmoid(logits)) + eps)``.
Thus predicted foreground is penalized while an asymptotically empty
prediction approaches zero IoU loss; there is no special-case constant that
would remove the logit gradient.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn

from analysis.source_train_provenance import (
    OUTER_ORACLE_ROLE,
    AnalysisProvenanceError,
    SourceTrainAnalysisProvenance,
)


TASK_LOSS_SCHEMA_VERSION = 1
TASK_LOSS_TYPE = "d0_v2_outer_oracle_bce_soft_iou"
BCE_REDUCTION = "mean_over_all_pixels"
SOFT_IOU_REDUCTION = "mean_over_sample"
EMPTY_TARGET_CONVENTION = "epsilon_smoothed_union_penalizes_foreground"


class D0V2TaskLossError(ValueError):
    """The frozen task-loss contract or one of its tensors is invalid."""


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise D0V2TaskLossError(f"{field} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise D0V2TaskLossError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class D0V2TaskLossConfig:
    """Immutable, exact-schema definition of the D0-v2 oracle loss."""

    lambda_bce: float
    lambda_soft_iou: float
    eps: float
    bce_reduction: Literal["mean_over_all_pixels"] = BCE_REDUCTION
    soft_iou_reduction: Literal["mean_over_sample"] = SOFT_IOU_REDUCTION
    empty_target_convention: Literal[
        "epsilon_smoothed_union_penalizes_foreground"
    ] = EMPTY_TARGET_CONVENTION

    def __post_init__(self) -> None:
        lambda_bce = _finite_float(self.lambda_bce, "lambda_bce")
        lambda_soft_iou = _finite_float(
            self.lambda_soft_iou, "lambda_soft_iou"
        )
        eps = _finite_float(self.eps, "eps")
        if lambda_bce < 0.0:
            raise D0V2TaskLossError("lambda_bce must be non-negative")
        if lambda_soft_iou < 0.0:
            raise D0V2TaskLossError("lambda_soft_iou must be non-negative")
        if lambda_bce == 0.0 and lambda_soft_iou == 0.0:
            raise D0V2TaskLossError(
                "at least one of lambda_bce and lambda_soft_iou must be positive"
            )
        if eps <= 0.0:
            raise D0V2TaskLossError("eps must be positive")
        if self.bce_reduction != BCE_REDUCTION:
            raise D0V2TaskLossError(
                f"bce_reduction must be exactly {BCE_REDUCTION!r}"
            )
        if self.soft_iou_reduction != SOFT_IOU_REDUCTION:
            raise D0V2TaskLossError(
                "soft_iou_reduction must be exactly "
                f"{SOFT_IOU_REDUCTION!r}"
            )
        if self.empty_target_convention != EMPTY_TARGET_CONVENTION:
            raise D0V2TaskLossError(
                "empty_target_convention must be exactly "
                f"{EMPTY_TARGET_CONVENTION!r}"
            )

        # Normalize accepted integer inputs while retaining a frozen dataclass.
        object.__setattr__(self, "lambda_bce", lambda_bce)
        object.__setattr__(self, "lambda_soft_iou", lambda_soft_iou)
        object.__setattr__(self, "eps", eps)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "D0V2TaskLossConfig":
        """Parse an exact configuration mapping; missing/unknown keys fail."""

        if not isinstance(value, Mapping):
            raise D0V2TaskLossError("task-loss config must be a mapping")
        expected_fields = {
            "lambda_bce",
            "lambda_soft_iou",
            "eps",
            "bce_reduction",
            "soft_iou_reduction",
            "empty_target_convention",
        }
        observed_fields = set(value)
        missing = sorted(expected_fields - observed_fields)
        unknown = sorted(observed_fields - expected_fields, key=str)
        if missing or unknown:
            raise D0V2TaskLossError(
                "task-loss config fields must be exact; "
                f"missing={missing}, unknown={unknown}"
            )
        return cls(
            lambda_bce=value["lambda_bce"],
            lambda_soft_iou=value["lambda_soft_iou"],
            eps=value["eps"],
            bce_reduction=value["bce_reduction"],
            soft_iou_reduction=value["soft_iou_reduction"],
            empty_target_convention=value["empty_target_convention"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lambda_bce": self.lambda_bce,
            "lambda_soft_iou": self.lambda_soft_iou,
            "eps": self.eps,
            "bce_reduction": self.bce_reduction,
            "soft_iou_reduction": self.soft_iou_reduction,
            "empty_target_convention": self.empty_target_convention,
        }


def _validated_config(
    value: D0V2TaskLossConfig | Mapping[str, Any],
) -> D0V2TaskLossConfig:
    if isinstance(value, D0V2TaskLossConfig):
        # Reconstruct to protect against deliberate frozen-dataclass mutation.
        return D0V2TaskLossConfig.from_mapping(value.to_dict())
    if isinstance(value, Mapping):
        return D0V2TaskLossConfig.from_mapping(value)
    raise D0V2TaskLossError(
        "config must be D0V2TaskLossConfig or an exact config mapping"
    )


def _validated_provenance(
    provenance: SourceTrainAnalysisProvenance,
) -> SourceTrainAnalysisProvenance:
    if not isinstance(provenance, SourceTrainAnalysisProvenance):
        raise D0V2TaskLossError(
            "provenance must be SourceTrainAnalysisProvenance"
        )
    try:
        # Reparse the complete receipt so a mutated instance cannot bypass the
        # source-train/outer-oracle isolation contract.
        return SourceTrainAnalysisProvenance.from_mapping(
            provenance.to_dict(), require_outer_oracle=True
        )
    except AnalysisProvenanceError as exc:
        raise D0V2TaskLossError(
            "task loss requires train-only outer-oracle provenance"
        ) from exc


def _validate_tensors(logits: Tensor, target: Tensor) -> None:
    for tensor, name in ((logits, "logits"), (target, "target")):
        if not isinstance(tensor, Tensor):
            raise D0V2TaskLossError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 1:
            raise D0V2TaskLossError(
                f"{name} shape must be exactly [1, 1, H, W]"
            )
        if tensor.shape[2] <= 0 or tensor.shape[3] <= 0:
            raise D0V2TaskLossError(f"{name} H and W must be positive")
        if not torch.is_floating_point(tensor):
            raise D0V2TaskLossError(f"{name} must be a floating-point tensor")
        if not bool(torch.isfinite(tensor).all().item()):
            raise D0V2TaskLossError(f"{name} must contain only finite values")

    if logits.shape != target.shape:
        raise D0V2TaskLossError("logits and target shapes must match exactly")
    if logits.dtype != target.dtype:
        raise D0V2TaskLossError("logits and target dtypes must match exactly")
    if logits.device != target.device:
        raise D0V2TaskLossError("logits and target devices must match exactly")
    if target.requires_grad:
        raise D0V2TaskLossError("target must not require gradients")
    target_min = float(target.amin().item())
    target_max = float(target.amax().item())
    if target_min < 0.0 or target_max > 1.0:
        raise D0V2TaskLossError("target values must lie in the closed interval [0, 1]")


def compute_d0_v2_task_loss(
    *,
    logits: Tensor,
    target: Tensor,
    config: D0V2TaskLossConfig | Mapping[str, Any],
    provenance: SourceTrainAnalysisProvenance,
) -> tuple[Tensor, dict[str, Any]]:
    """Return a differentiable total loss and a JSON-safe audit receipt.

    Only scalar copies stored in ``audit`` are detached.  The returned
    ``total_loss`` is the original autograd-connected tensor and is suitable
    for computing the *diagnostic* supervised gradient.  It must not be used
    as an adaptation objective.
    """

    frozen_config = _validated_config(config)
    frozen_provenance = _validated_provenance(provenance)
    _validate_tensors(logits, target)

    bce_loss = nn.BCEWithLogitsLoss(reduction="mean")(logits, target)
    probability = torch.sigmoid(logits)
    reduce_dims = (1, 2, 3)
    intersection = torch.sum(probability * target, dim=reduce_dims)
    union = torch.sum(
        probability + target - probability * target, dim=reduce_dims
    )
    soft_iou_score_per_sample = (
        intersection + frozen_config.eps
    ) / (union + frozen_config.eps)
    soft_iou_score = soft_iou_score_per_sample.mean()
    soft_iou_loss = 1.0 - soft_iou_score
    weighted_bce = frozen_config.lambda_bce * bce_loss
    weighted_soft_iou = frozen_config.lambda_soft_iou * soft_iou_loss
    total_loss = weighted_bce + weighted_soft_iou

    components = {
        "bce_with_logits": float(bce_loss.detach().item()),
        "soft_iou_score": float(soft_iou_score.detach().item()),
        "soft_iou_loss": float(soft_iou_loss.detach().item()),
        "weighted_bce": float(weighted_bce.detach().item()),
        "weighted_soft_iou": float(weighted_soft_iou.detach().item()),
        "total_loss": float(total_loss.detach().item()),
    }
    if not all(math.isfinite(value) for value in components.values()):
        raise D0V2TaskLossError("computed loss components must all be finite")

    target_is_empty = bool(torch.count_nonzero(target).item() == 0)
    audit: dict[str, Any] = {
        "schema_version": TASK_LOSS_SCHEMA_VERSION,
        "task_loss_type": TASK_LOSS_TYPE,
        "scope": frozen_provenance.to_dict(),
        "label_isolation": {
            "role": OUTER_ORACLE_ROLE,
            "source_train_only": True,
            "outer_oracle_analysis": True,
            "used_by_adaptation": False,
            "adaptation_gradient_uses_labels": False,
            "method_label_accesses": 0,
            "outer_evaluator_label_accesses": (
                frozen_provenance.outer_evaluator_label_accesses
            ),
        },
        "config": frozen_config.to_dict(),
        "input": {
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
            "device_type": logits.device.type,
            "target_is_empty": target_is_empty,
            "empty_target_convention_applied": target_is_empty,
        },
        "components": components,
    }
    return total_loss, audit


# Descriptive public alias for callers that do not encode the phase name.
compute_outer_oracle_task_loss = compute_d0_v2_task_loss


__all__ = [
    "BCE_REDUCTION",
    "D0V2TaskLossConfig",
    "D0V2TaskLossError",
    "EMPTY_TARGET_CONVENTION",
    "SOFT_IOU_REDUCTION",
    "TASK_LOSS_SCHEMA_VERSION",
    "TASK_LOSS_TYPE",
    "compute_d0_v2_task_loss",
    "compute_outer_oracle_task_loss",
]
