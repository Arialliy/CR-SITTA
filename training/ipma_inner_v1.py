"""Label-free, one-step functional IPMA; no file paths, GT, or optimizer state.

The production region weights are teacher probabilities q and 1-q.  Their
positive mass is not a reliable-target/candidate test.  In particular, an
all-background teacher is legal and does not cause reliability abstention.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.ipma_d0_adapter_v1 import IdentityMetaAdapter


class NoProxyMassError(ValueError):
    """Both supplied regions have zero mass; caller must record abstention."""


def _validate_teacher(teacher: Tensor) -> None:
    if not isinstance(teacher, Tensor) or teacher.ndim != 4 or teacher.shape[:2] != (1, 1):
        raise ValueError("teacher must be one probability map [1,1,H,W]")
    if min(teacher.shape[-2:]) < 1 or teacher.dtype not in (torch.float32, torch.float64):
        raise ValueError("teacher requires a nonempty float32/float64 grid")
    if teacher.requires_grad or teacher.grad_fn is not None:
        raise ValueError("teacher must already be detached")
    if not bool(torch.isfinite(teacher).all()) or bool(((teacher < 0) | (teacher > 1)).any()):
        raise ValueError("teacher probabilities must be finite and inside [0,1]")


def teacher_region_weights(teacher: Tensor) -> Tensor:
    """The fixed q/(1-q) soft regions; no confidence-based filtering is done."""
    _validate_teacher(teacher)
    return torch.stack((teacher, 1.0 - teacher), dim=0).detach()


def balanced_proxy(logits: Tensor, teacher: Tensor, weights: Tensor) -> Tensor:
    """Soft BCE normalized separately over positive-mass supplied regions.

    Explicit weights allow tests of the zero-mass boundary.  The production
    caller uses teacher_region_weights(), not GT or a reliability detector.
    """
    _validate_teacher(teacher)
    if not isinstance(logits, Tensor) or logits.shape != teacher.shape:
        raise ValueError("logits/teacher shape mismatch")
    if not isinstance(weights, Tensor) or weights.shape != (2,) + logits.shape:
        raise ValueError("weights must have shape [2,1,1,H,W]")
    if any(t.device != logits.device or t.dtype != logits.dtype for t in (teacher, weights)):
        raise ValueError("logits/teacher/weights device or dtype mismatch")
    if weights.requires_grad or weights.grad_fn is not None:
        raise ValueError("region weights must already be detached")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("non-finite proxy logits")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("region weights must be finite and nonnegative")
    per_pixel = F.binary_cross_entropy_with_logits(logits, teacher, reduction="none")
    terms = []
    for weight in weights:
        denominator = weight.sum()
        if not bool(torch.isfinite(denominator)):
            raise ValueError("non-finite proxy region mass")
        if float(denominator) > 0:
            terms.append((per_pixel * weight).sum() / denominator)
    if not terms:
        raise NoProxyMassError("both proxy regions have zero mass; record abstain")
    result = torch.stack(terms).mean()
    if not bool(torch.isfinite(result)):
        raise RuntimeError("non-finite balanced proxy loss")
    return result


def inner_step(adapter: IdentityMetaAdapter, head: nn.Module,
               h_probe: Tensor, teacher: Tensor, weights: Tensor, *,
               learning_rate: float = 0.05, radius: float = 0.01,
               create_graph: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Create delta0=0, take one differentiable gradient step, project to L2.

    create_graph=True keeps the full mixed derivative needed by the outer
    loss.  False gives the same numerical update without that meta graph.
    No model state or .grad field is updated by this function.
    """
    if not isinstance(adapter, IdentityMetaAdapter) or not isinstance(head, nn.Module):
        raise TypeError("inner_step needs the IPMA adapter and a frozen source head")
    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be bool")
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0
           for v in (learning_rate, radius)):
        raise ValueError("learning_rate/radius must be positive finite real numbers")
    adapter._validate_features(h_probe)
    if any(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError("source output head must be frozen")
    for name, value in head.state_dict(keep_vars=True).items():
        if value.device != h_probe.device or (value.is_floating_point() and value.dtype != h_probe.dtype):
            raise ValueError(f"head/features device or dtype mismatch: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"non-finite source head state: {name}")
    # A local scope permits use inside a caller's source-evaluation no_grad
    # block without silently cutting the adaptation graph.
    with torch.enable_grad():
        delta0 = h_probe.new_zeros(adapter.num_bases, requires_grad=True)
        loss = balanced_proxy(head(adapter(h_probe, delta0)), teacher, weights)
        (gradient,) = torch.autograd.grad(loss, delta0, create_graph=create_graph)
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("non-finite inner gradient")
        proposal = delta0 - float(learning_rate) * gradient
        scale = torch.clamp(float(radius) / proposal.norm().clamp_min(1e-12), max=1.0)
        delta1 = proposal * scale
        if not bool(torch.isfinite(delta1).all()):
            raise RuntimeError("non-finite projected episode delta")
    return delta1, loss, gradient


__all__ = ["NoProxyMassError", "teacher_region_weights", "balanced_proxy", "inner_step"]
