"""Synthetic CPU exact-meta, SLS compatibility and proxy contract tests."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from model.ipma_d0_adapter_v1 import IdentityMetaAdapter
from model.loss import SLSIoULoss
from training.ipma_inner_v1 import NoProxyMassError, balanced_proxy, inner_step, teacher_region_weights


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        yield
    torch.set_num_threads(previous)


def components():
    adapter = IdentityMetaAdapter().double()
    head = torch.nn.Conv2d(16, 1, 1).double().requires_grad_(False)
    observed = torch.randn(1, 16, 8, 8, dtype=torch.float64)
    probe = observed + 0.2 * torch.randn_like(observed)
    teacher = head(observed).sigmoid().detach()
    weights = teacher_region_weights(teacher)
    target = (torch.rand_like(teacher) > 0.9).double()
    return adapter, head, observed, probe, teacher, weights, target


@pytest.mark.parametrize("radius", [0.01, 1e-6])
def test_create_graph_flags_have_identical_numeric_updates(radius):
    adapter, head, _, probe, teacher, weights, _ = components()
    yes = inner_step(adapter, head, probe, teacher, weights, radius=radius, create_graph=True)
    no = inner_step(adapter, head, probe, teacher, weights, radius=radius, create_graph=False)
    assert all(torch.equal(a, b) for a, b in zip(yes, no, strict=True))
    assert yes[2].grad_fn is not None
    assert no[2].grad_fn is None
    assert yes[0].norm() <= radius + 1e-15


@pytest.mark.parametrize("radius", [0.01, 1e-6])
@pytest.mark.parametrize("loss_name", ["sls", "bce"])
def test_full_meta_directional_derivative_matches_finite_difference(radius, loss_name):
    adapter, head, observed, probe, teacher, weights, target = components()
    parameters = list(adapter.parameters())
    direction = [torch.randn_like(parameter) for parameter in parameters]
    norm = sum(d.square().sum() for d in direction).sqrt()
    direction = [d / norm for d in direction]
    original = [parameter.detach().clone() for parameter in parameters]
    def loss():
        delta, _, _ = inner_step(adapter, head, probe, teacher, weights, radius=radius, create_graph=True)
        logits = head(adapter(observed, delta))
        return (SLSIoULoss()(logits, target, 5, 999) if loss_name == "sls"
                else F.binary_cross_entropy_with_logits(logits, target))
    objective = loss()
    gradients = torch.autograd.grad(objective, parameters)
    autograd = sum((g*d).sum() for g, d in zip(gradients, direction, strict=True)).item()
    epsilon = 1e-5
    values = []
    try:
        for sign in (1, -1):
            with torch.no_grad():
                for parameter, base, d in zip(parameters, original, direction, strict=True):
                    parameter.copy_(base + sign*epsilon*d)
            values.append(loss().item())
    finally:
        with torch.no_grad():
            for parameter, base in zip(parameters, original, strict=True): parameter.copy_(base)
    numerical = (values[0] - values[1]) / (2 * epsilon)
    assert abs(autograd) > 1e-12
    assert abs(autograd - numerical) < 5e-9
    assert all(torch.isfinite(g).all() for g in gradients)


def test_exact_meta_is_not_the_detached_inner_gradient_shortcut():
    adapter, head, observed, probe, teacher, weights, target = components()
    grads = []
    for create_graph in (True, False):
        delta, _, _ = inner_step(adapter, head, probe, teacher, weights, create_graph=create_graph)
        loss = SLSIoULoss()(head(adapter(observed, delta)), target, 5, 999)
        grads.append(torch.cat([g.flatten() for g in torch.autograd.grad(loss, list(adapter.parameters()))]))
    assert torch.isfinite(grads[0]).all() and torch.isfinite(grads[1]).all()
    assert float((grads[0] - grads[1]).norm()) > 1e-9


def test_frozen_head_preserves_input_gradient_and_sls_handles_empty_gt():
    adapter, head, observed, probe, teacher, weights, target = components()
    target.zero_()
    delta, _, gradient = inner_step(adapter, head, probe, teacher, weights, create_graph=True)
    loss = SLSIoULoss()(head(adapter(observed, delta)), target, 5, 999)
    meta = torch.autograd.grad(loss, list(adapter.parameters()))
    assert torch.isfinite(loss) and gradient.norm() > 0
    assert all(torch.isfinite(value).all() for value in meta)
    assert all(parameter.grad is None for parameter in head.parameters())


def test_identity_probe_zero_gradient_is_finite_not_false_success():
    adapter, head, observed, _, teacher, weights, target = components()
    delta, _, gradient = inner_step(adapter, head, observed, teacher, weights, create_graph=True)
    assert torch.count_nonzero(gradient) == 0
    assert torch.count_nonzero(delta) == 0
    loss = SLSIoULoss()(head(adapter(observed, delta)), target, 5, 999)
    meta = torch.autograd.grad(loss, list(adapter.parameters()))
    assert all(torch.isfinite(value).all() for value in meta)
    assert all(torch.count_nonzero(value) == 0 for value in meta)


def test_inner_does_not_silently_lose_gradient_in_source_no_grad_scope():
    adapter, head, _, probe, teacher, weights, _ = components()
    with torch.no_grad():
        delta, _, gradient = inner_step(adapter, head, probe, teacher, weights, create_graph=True)
    assert delta.grad_fn is not None and gradient.grad_fn is not None
    assert gradient.norm() > 0


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_background_uncertain_and_foreground_teachers_are_mass_valid(value):
    teacher = torch.full((1, 1, 3, 4), value, dtype=torch.float64)
    logits = torch.zeros_like(teacher, requires_grad=True)
    weights = teacher_region_weights(teacher)
    assert torch.equal(weights[0], teacher)
    assert torch.equal(weights[1], 1 - teacher)
    loss = balanced_proxy(logits, teacher, weights)
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, logits)[0]).all()


def test_zero_region_mass_uses_dedicated_abstain_exception():
    teacher = torch.zeros(1, 1, 3, 4, dtype=torch.float64)
    with pytest.raises(NoProxyMassError, match="zero mass"):
        balanced_proxy(torch.zeros_like(teacher), teacher, torch.zeros(2, *teacher.shape, dtype=teacher.dtype))


@pytest.mark.parametrize("fault", ["teacher_grad", "weights_grad", "teacher_nan", "teacher_range", "weights_nan",
                                  "negative_weights", "logit_nan", "logit_shape", "weights_shape", "weights_dtype", "teacher_dtype"])
def test_proxy_invalid_inputs_fail_closed(fault):
    teacher = torch.full((1, 1, 3, 4), 0.5, dtype=torch.float64)
    logits = torch.zeros_like(teacher, requires_grad=True)
    weights = teacher_region_weights(teacher)
    if fault == "teacher_grad": teacher.requires_grad_(True)
    if fault == "weights_grad": weights.requires_grad_(True)
    if fault == "teacher_nan": teacher[0, 0, 0, 0] = float("nan")
    if fault == "teacher_range": teacher[0, 0, 0, 0] = 1.1
    if fault == "weights_nan": weights[0, 0, 0, 0, 0] = float("nan")
    if fault == "negative_weights": weights[0, 0, 0, 0, 0] = -0.1
    if fault == "logit_nan": logits = torch.full_like(logits, float("nan"))
    if fault == "logit_shape": logits = logits[:, :, :2]
    if fault == "weights_shape": weights = weights[:1]
    if fault == "weights_dtype": weights = weights.float()
    if fault == "teacher_dtype": teacher = teacher.float()
    with pytest.raises(ValueError):
        balanced_proxy(logits, teacher, weights)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan"), True])
@pytest.mark.parametrize("parameter", ["learning_rate", "radius"])
def test_invalid_inner_hyperparameters_rejected(value, parameter):
    adapter, head, _, probe, teacher, weights, _ = components()
    with pytest.raises(ValueError, match="positive finite"):
        inner_step(adapter, head, probe, teacher, weights, **{parameter: value}, create_graph=True)


def test_unfrozen_head_or_nondetached_feature_is_rejected():
    adapter, head, _, probe, teacher, weights, _ = components()
    head.requires_grad_(True)
    with pytest.raises(ValueError, match="head must be frozen"):
        inner_step(adapter, head, probe, teacher, weights, create_graph=True)
    head.requires_grad_(False)
    with pytest.raises(ValueError, match="detached"):
        inner_step(adapter, head, probe.requires_grad_(), teacher, weights, create_graph=True)


def test_head_and_teacher_device_mismatch_rejected_without_gpu():
    adapter, head, _, probe, teacher, weights, _ = components()
    with pytest.raises(ValueError, match="device"):
        inner_step(adapter, head.to("meta"), probe, teacher, weights, create_graph=True)
    with pytest.raises(ValueError, match="device"):
        balanced_proxy(torch.zeros_like(teacher, device="meta"), teacher, weights)
