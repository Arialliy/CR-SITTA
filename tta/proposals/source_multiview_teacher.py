"""Frozen, source-anchored multi-view teachers for single-image TTA.

Only exactly invertible flips are used as full-image geometric views.  The
optional context-tile view is reconstructed on the original 256 x 256 canvas
before it participates in aggregation.  Every inference and aggregation
result produced by this module is detached by construction.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Literal, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


GeometricView = Literal["identity", "hflip", "vflip", "hvflip"]
AggregationMethod = Literal[
    "mean",
    "trimmed_mean",
    "disagreement_weighted_mean",
    "source_anchor",
]

GEOMETRIC_VIEWS: tuple[GeometricView, ...] = (
    "identity",
    "hflip",
    "vflip",
    "hvflip",
)
CONTEXT_TILE_INPUT_SIZE = 256
CONTEXT_TILE_CROP_SIZE = 224
CONTEXT_TILE_ORIGINS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 32),
    (32, 0),
    (32, 32),
)


@runtime_checkable
class LogitsAdapter(Protocol):
    """Minimal adapter protocol used by the teacher builder."""

    def forward_logits(self, image: Tensor) -> Tensor:
        """Return logits with shape ``[B, 1, H, W]``."""


def _validate_bchw(tensor: Tensor, *, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W]")
    if any(int(size) <= 0 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must all be positive")


def _validate_image(image: Tensor, *, require_256: bool = False) -> None:
    _validate_bchw(image, name="image")
    if not image.is_floating_point():
        raise TypeError("image must have a floating-point dtype")
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError("image must contain only finite values")
    if require_256 and image.shape[-2:] != (
        CONTEXT_TILE_INPUT_SIZE,
        CONTEXT_TILE_INPUT_SIZE,
    ):
        raise ValueError("context-tile inference requires a 256 x 256 image")


def _validate_probability(probability: Tensor, *, name: str) -> None:
    _validate_bchw(probability, name=name)
    if probability.shape[1] != 1:
        raise ValueError(f"{name} must have exactly one channel")
    if not probability.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(probability).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    if bool((probability < 0).any().item()) or bool((probability > 1).any().item()):
        raise ValueError(f"{name} values must lie in [0, 1]")


def _validate_view(view: str) -> GeometricView:
    if view not in GEOMETRIC_VIEWS:
        choices = ", ".join(GEOMETRIC_VIEWS)
        raise ValueError(f"unknown geometric view {view!r}; expected one of {choices}")
    return view  # type: ignore[return-value]


def apply_geometric_view(tensor: Tensor, view: GeometricView) -> Tensor:
    """Apply one of the four exact full-image views to a BCHW tensor."""

    _validate_bchw(tensor, name="tensor")
    checked = _validate_view(view)
    if checked == "identity":
        return tensor
    if checked == "hflip":
        return torch.flip(tensor, dims=(-1,))
    if checked == "vflip":
        return torch.flip(tensor, dims=(-2,))
    return torch.flip(tensor, dims=(-2, -1))


def inverse_geometric_view(tensor: Tensor, view: GeometricView) -> Tensor:
    """Invert a geometric view exactly (all supported views are involutions)."""

    return apply_geometric_view(tensor, view)


def _underlying_module(predictor: object) -> nn.Module | None:
    if isinstance(predictor, nn.Module):
        return predictor
    wrapped = getattr(predictor, "model", None)
    return wrapped if isinstance(wrapped, nn.Module) else None


@contextmanager
def _temporary_eval(predictor: object):
    """Use eval mode for inference and restore every module's prior mode."""

    module = _underlying_module(predictor)
    if module is None:
        yield
        return

    modes = tuple((child, bool(child.training)) for child in module.modules())
    module.eval()
    try:
        yield
    finally:
        # Assign each flag directly.  Calling train() recursively here could
        # overwrite heterogeneous child modes that existed before inference.
        for child, training in modes:
            child.training = training


def _extract_logits(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and len(output) == 2:
        logits = output[1]
        if isinstance(logits, Tensor):
            return logits
    raise TypeError("predictor must return logits or an (auxiliary, logits) pair")


def _call_logits(predictor: object, image: Tensor) -> Tensor:
    forward_logits = getattr(predictor, "forward_logits", None)
    if callable(forward_logits):
        return _extract_logits(forward_logits(image))
    if callable(predictor):
        return _extract_logits(predictor(image))
    raise TypeError("predictor must be callable or implement forward_logits(image)")


def _convert_logits_to_probability(predictor: object, logits: Tensor) -> Tensor:
    converter: Callable[[Tensor], Tensor] | None = getattr(
        predictor, "logits_to_prob", None
    )
    probability = converter(logits) if callable(converter) else torch.sigmoid(logits)
    if not isinstance(probability, Tensor):
        raise TypeError("logits_to_prob must return a torch.Tensor")
    return probability


def frozen_probability_forward(predictor: object, image: Tensor) -> Tensor:
    """Run one mutation-free source inference and return detached probabilities.

    The underlying model, when discoverable either directly or as
    ``predictor.model``, is temporarily put in evaluation mode.  Its original
    heterogeneous train/eval flags are restored even if inference raises.
    Parameter ``requires_grad`` flags are never changed.
    """

    _validate_image(image)
    with _temporary_eval(predictor), torch.inference_mode():
        logits = _call_logits(predictor, image)
        _validate_bchw(logits, name="logits")
        expected = (image.shape[0], 1, image.shape[-2], image.shape[-1])
        if tuple(logits.shape) != expected:
            raise ValueError(
                "logits must have shape [B, 1, H, W] matching the input; "
                f"expected {expected}, got {tuple(logits.shape)}"
            )
        if not logits.is_floating_point():
            raise TypeError("logits must have a floating-point dtype")
        if not bool(torch.isfinite(logits).all().item()):
            raise ValueError("logits must contain only finite values")
        probability = _convert_logits_to_probability(predictor, logits)
        if tuple(probability.shape) != expected:
            raise ValueError("probability shape must match the validated logits shape")
        _validate_probability(probability, name="probability")
        return probability.detach()


def extract_context_tiles(image: Tensor) -> Tensor:
    """Return four resized context crops with shape ``[4, B, C, 256, 256]``."""

    _validate_image(image, require_256=True)
    crops: list[Tensor] = []
    with torch.inference_mode():
        for top, left in CONTEXT_TILE_ORIGINS:
            crop = image[
                :,
                :,
                top : top + CONTEXT_TILE_CROP_SIZE,
                left : left + CONTEXT_TILE_CROP_SIZE,
            ]
            if crop.shape[-2:] != (CONTEXT_TILE_CROP_SIZE, CONTEXT_TILE_CROP_SIZE):
                raise RuntimeError("context-tile crop geometry is inconsistent")
            crops.append(
                F.interpolate(
                    crop,
                    size=(CONTEXT_TILE_INPUT_SIZE, CONTEXT_TILE_INPUT_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.stack(crops, dim=0).detach()


def stitch_context_tile_probabilities(tile_probabilities: Tensor) -> Tensor:
    """Overlap-average four 224 x 224 tile probabilities on a 256 canvas.

    ``tile_probabilities`` must have shape ``[4, B, 1, 224, 224]`` and already
    be resized back from the model's 256 x 256 output.
    """

    if not isinstance(tile_probabilities, Tensor):
        raise TypeError("tile_probabilities must be a torch.Tensor")
    if tile_probabilities.ndim != 5:
        raise ValueError(
            "tile_probabilities must have shape [4, B, 1, 224, 224]"
        )
    expected_tail = (1, CONTEXT_TILE_CROP_SIZE, CONTEXT_TILE_CROP_SIZE)
    if (
        tile_probabilities.shape[0] != len(CONTEXT_TILE_ORIGINS)
        or tile_probabilities.shape[1] <= 0
        or tuple(tile_probabilities.shape[2:]) != expected_tail
    ):
        raise ValueError(
            "tile_probabilities must have shape [4, B, 1, 224, 224]"
        )
    flat = tile_probabilities.reshape(
        -1, 1, CONTEXT_TILE_CROP_SIZE, CONTEXT_TILE_CROP_SIZE
    )
    _validate_probability(flat, name="tile_probabilities")

    batch_size = int(tile_probabilities.shape[1])
    with torch.inference_mode():
        total = tile_probabilities.new_zeros(
            (batch_size, 1, CONTEXT_TILE_INPUT_SIZE, CONTEXT_TILE_INPUT_SIZE)
        )
        count = tile_probabilities.new_zeros((1, 1, 256, 256))
        for index, (top, left) in enumerate(CONTEXT_TILE_ORIGINS):
            total[
                :,
                :,
                top : top + CONTEXT_TILE_CROP_SIZE,
                left : left + CONTEXT_TILE_CROP_SIZE,
            ] += tile_probabilities[index]
            count[
                :,
                :,
                top : top + CONTEXT_TILE_CROP_SIZE,
                left : left + CONTEXT_TILE_CROP_SIZE,
            ] += 1
        if bool((count == 0).any().item()):
            raise RuntimeError("context-tile origins do not cover the output canvas")
        reconstructed = total / count
        _validate_probability(reconstructed, name="reconstructed tile probability")
        return reconstructed.detach()


def infer_context_tile_probability(predictor: object, image: Tensor) -> Tensor:
    """Infer and overlap-average the four fixed context tiles.

    Each tile is forwarded separately with the original image batch size.  In
    particular, the formal single-image path makes four ``B=1`` calls rather
    than silently turning the four views into one ``B=4`` batch.  This keeps
    view semantics independent of any batch-sensitive detector behavior.
    """

    _validate_image(image, require_256=True)
    with torch.inference_mode():
        tiles = extract_context_tiles(image)
        restored: list[Tensor] = []
        for tile in tiles.unbind(dim=0):
            probability = frozen_probability_forward(predictor, tile)
            restored.append(
                F.interpolate(
                    probability,
                    size=(CONTEXT_TILE_CROP_SIZE, CONTEXT_TILE_CROP_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return stitch_context_tile_probabilities(
            torch.stack(restored, dim=0)
        ).detach()


def build_aligned_view_probabilities(
    predictor: object,
    image: Tensor,
    *,
    include_context_tile: bool = False,
) -> tuple[Tensor, tuple[str, ...]]:
    """Build source-aligned views as ``[V, B, 1, H, W]``.

    View zero is always the unmodified source prediction.  The four base views
    are ordered as ``identity, hflip, vflip, hvflip``.  ``context_tile`` is
    appended when requested and is valid only for 256 x 256 inputs.
    """

    _validate_image(image, require_256=include_context_tile)
    names: list[str] = list(GEOMETRIC_VIEWS)
    aligned: list[Tensor] = []
    with torch.inference_mode():
        for view in GEOMETRIC_VIEWS:
            transformed = apply_geometric_view(image, view)
            probability = frozen_probability_forward(predictor, transformed)
            aligned.append(inverse_geometric_view(probability, view))
        if include_context_tile:
            aligned.append(infer_context_tile_probability(predictor, image))
            names.append("context_tile")

        stacked = torch.stack(aligned, dim=0)
        if stacked.shape[1:] != (image.shape[0], 1, image.shape[-2], image.shape[-1]):
            raise RuntimeError("aligned probability shapes are inconsistent")
        flat = stacked.reshape(-1, 1, image.shape[-2], image.shape[-1])
        _validate_probability(flat, name="aligned probabilities")
        return stacked.detach(), tuple(names)


def _validate_probability_stack(probabilities: Tensor) -> None:
    if not isinstance(probabilities, Tensor):
        raise TypeError("probabilities must be a torch.Tensor")
    if probabilities.ndim != 5:
        raise ValueError("probabilities must have shape [V, B, 1, H, W]")
    if probabilities.shape[0] <= 0 or probabilities.shape[1] <= 0:
        raise ValueError("probabilities must contain at least one view and one image")
    if probabilities.shape[2] != 1 or min(probabilities.shape[3:]) <= 0:
        raise ValueError("probabilities must have shape [V, B, 1, H, W]")
    flat = probabilities.reshape(
        probabilities.shape[0] * probabilities.shape[1],
        1,
        probabilities.shape[3],
        probabilities.shape[4],
    )
    _validate_probability(flat, name="probabilities")


def aggregate_aligned_probabilities(
    probabilities: Tensor,
    method: AggregationMethod = "mean",
    *,
    trim_each_side: int = 1,
    tau: float = 0.05,
    beta: float = 0.5,
    source_anchor_aggregate: Literal[
        "mean", "trimmed_mean", "disagreement_weighted_mean"
    ] = "mean",
) -> Tensor:
    """Aggregate aligned probabilities into one detached teacher probability.

    For ``source_anchor``, view zero is the identity source probability and the
    formula is ``identity + beta * (aggregate - identity)``.  The inner
    aggregate defaults to ``mean`` and can be selected explicitly.
    """

    _validate_probability_stack(probabilities)
    allowed = {
        "mean",
        "trimmed_mean",
        "disagreement_weighted_mean",
        "source_anchor",
    }
    if method not in allowed:
        raise ValueError(f"unknown aggregation method {method!r}")
    if isinstance(trim_each_side, bool) or not isinstance(trim_each_side, int):
        raise TypeError("trim_each_side must be an integer")
    if trim_each_side < 0:
        raise ValueError("trim_each_side must be non-negative")
    if not isinstance(tau, (int, float)) or isinstance(tau, bool):
        raise TypeError("tau must be a real number")
    if not torch.isfinite(torch.tensor(float(tau))) or float(tau) <= 0:
        raise ValueError("tau must be finite and greater than zero")
    if not isinstance(beta, (int, float)) or isinstance(beta, bool):
        raise TypeError("beta must be a real number")
    if not torch.isfinite(torch.tensor(float(beta))) or not 0 <= float(beta) <= 1:
        raise ValueError("beta must be finite and lie in [0, 1]")
    if source_anchor_aggregate not in {
        "mean",
        "trimmed_mean",
        "disagreement_weighted_mean",
    }:
        raise ValueError("source_anchor_aggregate must be a non-anchor method")

    detached = probabilities.detach()

    def non_anchor(which: str) -> Tensor:
        if which == "mean":
            return detached.mean(dim=0)
        if which == "trimmed_mean":
            remaining = detached.shape[0] - 2 * trim_each_side
            if remaining <= 0:
                raise ValueError(
                    "trim_each_side removes every view; at least one must remain"
                )
            ordered = detached.sort(dim=0).values
            stop = ordered.shape[0] - trim_each_side
            selected = ordered[trim_each_side:stop] if trim_each_side else ordered
            return selected.mean(dim=0)
        mean = detached.mean(dim=0, keepdim=True)
        log_weights = -detached.sub(mean).square().div(float(tau))
        # Normalizing after subtracting the largest log-weight implements the
        # specified exponential weighting while preventing all weights from
        # underflowing when a valid, very small tau is selected.
        log_weights = log_weights - log_weights.max(dim=0, keepdim=True).values
        weights = torch.exp(log_weights)
        denominator = weights.sum(dim=0)
        if bool((denominator <= 0).any().item()):
            raise RuntimeError("disagreement weights have a zero denominator")
        return (weights * detached).sum(dim=0) / denominator

    with torch.inference_mode():
        if method == "source_anchor":
            aggregate = non_anchor(source_anchor_aggregate)
            identity = detached[0]
            teacher = identity + float(beta) * (aggregate - identity)
        else:
            teacher = non_anchor(method)
        _validate_probability(teacher, name="teacher probability")
        return teacher.detach()


def build_aligned_multiview_teacher(
    predictor: object,
    image: Tensor,
    *,
    include_context_tile: bool = False,
    aggregation: AggregationMethod = "mean",
    trim_each_side: int = 1,
    tau: float = 0.05,
    beta: float = 0.5,
    source_anchor_aggregate: Literal[
        "mean", "trimmed_mean", "disagreement_weighted_mean"
    ] = "mean",
) -> tuple[Tensor, Tensor]:
    """Return ``(teacher_probability, view_uncertainty)`` without gradients.

    Uncertainty is the population variance across aligned views.  Both outputs
    have shape ``[B, 1, H, W]``, are finite, lie in ``[0, 1]``, and are
    detached regardless of the caller's ambient gradient mode.
    """

    with torch.inference_mode():
        probabilities, _ = build_aligned_view_probabilities(
            predictor,
            image,
            include_context_tile=include_context_tile,
        )
        teacher = aggregate_aligned_probabilities(
            probabilities,
            aggregation,
            trim_each_side=trim_each_side,
            tau=tau,
            beta=beta,
            source_anchor_aggregate=source_anchor_aggregate,
        )
        uncertainty = probabilities.var(dim=0, unbiased=False)
        _validate_probability(uncertainty, name="view uncertainty")
        return teacher.detach(), uncertainty.detach()


__all__ = [
    "AggregationMethod",
    "CONTEXT_TILE_CROP_SIZE",
    "CONTEXT_TILE_INPUT_SIZE",
    "CONTEXT_TILE_ORIGINS",
    "GEOMETRIC_VIEWS",
    "GeometricView",
    "LogitsAdapter",
    "aggregate_aligned_probabilities",
    "apply_geometric_view",
    "build_aligned_multiview_teacher",
    "build_aligned_view_probabilities",
    "extract_context_tiles",
    "frozen_probability_forward",
    "infer_context_tile_probability",
    "inverse_geometric_view",
    "stitch_context_tile_probabilities",
]
