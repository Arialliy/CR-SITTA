"""Source-supervised background probability-increase penalty, training only.

This module is separate from the model, inference, and the original O3 update.
The fixed reference is computed as ``head(h)`` under ``no_grad`` on the SAME
batch of detached O3 features, not taken from batch-one cached probabilities.
The runner starts from the original initial weights, not a trained endpoint.

The penalty is a soft objective, not a guarantee of non-increasing false-alarm
counts or target preservation. Its denominator is GT foreground area, NOT
background area: tiny or empty targets can give background changes substantial
weight. At exact identity the ReLU penalty and its gradient are both zero;
pre-existing O3 false alarms, decreases, and GT foreground changes are not
penalized by this term. Source segmentation supervision remains unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor

from scripts.run_o3_multiscale_train8_v1 import supervised_loss as _original_supervised_loss


def _validate(logits: Tensor, reference_logits: Tensor, targets: Tensor) -> None:
    for name, value in (("logits", logits), ("reference_logits", reference_logits), ("targets", targets)):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{name} must be a dense strided tensor")
        if value.ndim != 4 or value.shape[1] != 1 or any(size == 0 for size in value.shape):
            raise ValueError(f"{name} must have nonempty shape [N,1,H,W]")
        if value.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must have dtype float32 or float64")
        if value.device.type == "meta":
            raise ValueError(f"{name} must be on a materialized non-meta device")
        if value.shape != logits.shape or value.dtype != logits.dtype or value.device != logits.device:
            raise ValueError(f"{name} shape, dtype, and device must match logits")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name} must be finite")
        if name != "logits" and (value.requires_grad or value.grad_fn is not None):
            raise ValueError(f"{name} must be detached")
    if not bool(((targets == 0) | (targets == 1)).all().item()):
        raise ValueError("targets must be binary source-training GT")


def background_nonexpansion_loss(
    logits: Tensor, reference_logits: Tensor, targets: Tensor
) -> Tensor:
    """Mean per-image positive background probability increase / max(GT area, 1).

    ``reference_logits`` and ``targets`` must be detached. This function never
    constructs labels, adapts a teacher, clamps inference probabilities, or
    modifies predictions. PyTorch's zero ReLU derivative at equality fixes the
    intended first-step behavior. Saturated sigmoid probabilities can have
    zero derivatives even when the positive-increase penalty is nonzero.
    """
    _validate(logits, reference_logits, targets)
    probability = torch.sigmoid(logits)
    reference_probability = torch.sigmoid(reference_logits).detach()
    positive_increase = torch.relu(probability - reference_probability)
    dimensions = (1, 2, 3)
    background_increase = ((1 - targets) * positive_increase).sum(dim=dimensions)
    target_area = targets.sum(dim=dimensions).clamp_min(1)
    loss = (background_increase / target_area).mean()
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("background nonexpansion loss must be finite")
    return loss


def guarded_supervised_loss(
    logits: Tensor, reference_logits: Tensor, targets: Tensor
) -> dict[str, Tensor]:
    """Return the original segmentation loss, guard, and fixed-weight sum.

    The added weight is exactly 1.0; there is no tunable keyword argument.
    Values remain differentiable scalar tensors, including the exact-zero
    guard at identity. Log all three separately because their scales differ.
    """
    guard = background_nonexpansion_loss(logits, reference_logits, targets)
    segmentation = _original_supervised_loss(logits, targets)
    augmented = segmentation + guard
    if not bool(torch.isfinite(augmented).item()):
        raise ValueError("augmented source-supervised loss must be finite")
    return {
        "segmentation_full_fit_loss": segmentation,
        "background_guard_full_fit_loss": guard,
        "augmented_full_fit_loss": augmented,
    }


__all__ = ["background_nonexpansion_loss", "guarded_supervised_loss"]
