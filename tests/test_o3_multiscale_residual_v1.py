"""Synthetic CPU tests only; no image dataset, source weights, or accelerator."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from model.o3_multiscale_residual_v1 import O3MultiScaleResidual


@pytest.fixture(autouse=True)
def deterministic_cpu():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(704)
        yield
    torch.set_num_threads(old_threads)


def fixture_components(dtype=torch.float32, shape=(2, 16, 9, 12)):
    return O3MultiScaleResidual().to(dtype=dtype), torch.randn(shape, dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(1, 16, 1, 1), (1, 16, 7, 8), (2, 16, 8, 7), (3, 16, 9, 13), (2, 16, 10, 12)])
def test_initial_identity_for_batches_odd_even_and_nonsquare(dtype, shape):
    branch, h = fixture_components(dtype, shape)
    before = h.clone()
    output = branch(h)
    assert output.shape == h.shape
    assert output.dtype == h.dtype
    assert torch.equal(output, h)
    assert torch.equal(output.contiguous().view(torch.uint8), h.contiguous().view(torch.uint8))
    assert torch.equal(h, before)
    assert output.requires_grad


def test_fixed_architecture_parameter_count_and_no_stateful_layers():
    branch = O3MultiScaleResidual()
    assert [(name, tuple(p.shape)) for name, p in branch.named_parameters()] == [
        ("expand.weight", (32, 16, 1, 1)),
        ("dw3.weight", (16, 1, 3, 3)),
        ("dw5.weight", (16, 1, 5, 5)),
        ("project.weight", (16, 32, 1, 1)),
    ]
    assert sum(p.numel() for p in branch.parameters()) == 1568
    assert all(p.requires_grad for p in branch.parameters())
    assert all(child.bias is None for child in branch.modules() if isinstance(child, nn.Conv2d))
    assert not list(branch.buffers())
    assert not any(isinstance(child, (nn.Dropout, nn.modules.batchnorm._BatchNorm)) for child in branch.modules())
    assert not branch.relu3.inplace and not branch.relu5.inplace
    assert torch.count_nonzero(branch.project.weight) == 0
    for name in ("expand", "dw3", "dw5"):
        assert torch.count_nonzero(getattr(branch, name).weight) > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_two_step_gradient_project_first_then_upstream(dtype):
    branch, h = fixture_components(dtype)
    objective_weight = torch.randn_like(h)
    optimizer = torch.optim.SGD(branch.parameters(), lr=0.5)
    ((branch(h) * objective_weight).mean()).backward()
    assert branch.project.weight.grad.norm() > 0
    for layer in (branch.expand, branch.dw3, branch.dw5):
        assert layer.weight.grad is not None
        assert torch.count_nonzero(layer.weight.grad) == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(branch(h), h)
    ((branch(h) * objective_weight).mean()).backward()
    for parameter in branch.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0
    assert h.grad is None


def test_frozen_head_transmits_supervised_gradient_without_host_graph():
    branch, h = fixture_components()
    head = nn.Conv2d(16, 1, 1).eval().requires_grad_(False)
    head_snapshot = {name: value.clone() for name, value in head.state_dict().items()}
    target = torch.zeros(2, 1, 9, 12)
    target[:, :, 3:5, 5:7] = 1.0
    loss = nn.functional.binary_cross_entropy_with_logits(head(branch(h)), target)
    assert loss.requires_grad
    loss.backward()
    assert branch.project.weight.grad.norm() > 0
    assert all(p.grad is None for p in head.parameters())
    assert all(torch.equal(value, head_snapshot[name]) for name, value in head.state_dict().items())
    assert h.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.0, 1e-20])
def test_zero_and_tiny_inputs_are_finite(dtype, scale):
    branch, h = fixture_components(dtype)
    h.mul_(scale)
    with torch.no_grad():
        branch.project.weight.normal_()
    output = branch(h)
    assert torch.isfinite(output).all()
    output.sum().backward()
    for p in branch.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    if scale == 0.0:
        assert torch.equal(output, h)
        assert all(torch.count_nonzero(p.grad) == 0 for p in branch.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bound_is_per_image_even_for_saturating_projection(dtype):
    branch, h = fixture_components(dtype, (3, 16, 9, 12))
    h[1].mul_(100)
    h[2].mul_(1e-9)
    with torch.no_grad():
        branch.project.weight.normal_(std=1000)
    scale = h.square().mean((1, 2, 3), keepdim=True).clamp_min(1e-12).sqrt()
    output = branch(h)
    tolerance = 4 * torch.finfo(dtype).eps * (h.abs() + scale)
    assert ((output - h).abs() <= 0.05 * scale + tolerance).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_independence_and_train_eval_agreement(dtype):
    branch, h = fixture_components(dtype, (3, 16, 9, 12))
    with torch.no_grad():
        branch.project.weight.normal_(std=0.1)
    together = branch(h)
    separate = torch.cat([branch(x[None]) for x in h], dim=0)
    assert torch.allclose(together, separate, rtol=2e-6, atol=2e-7)
    assert torch.equal(branch.train()(h), branch.eval()(h))
    changed = h.clone()
    changed[2].mul_(100)
    assert torch.equal(branch(h)[:2], branch(changed)[:2])


def test_trained_state_dict_round_trip_preserves_outputs_and_nonzero_projection():
    branch, h = fixture_components()
    with torch.no_grad():
        branch.project.weight.normal_(std=0.2)
    stream = io.BytesIO()
    torch.save(branch.state_dict(), stream)
    stream.seek(0)
    replica = O3MultiScaleResidual()
    replica.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    assert torch.count_nonzero(replica.project.weight) > 0
    assert torch.equal(branch(h), replica(h))


def test_noncontiguous_features_are_supported():
    branch, h = fixture_components()
    h = h.transpose(2, 3)
    assert not h.is_contiguous()
    assert torch.equal(branch(h), h)


@pytest.mark.parametrize("shape", [(1, 16, 9), (1, 15, 9, 12), (0, 16, 9, 12), (1, 16, 0, 12), (1, 16, 9, 0)])
def test_invalid_shapes(shape):
    with pytest.raises(ValueError, match="shape"):
        O3MultiScaleResidual()(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.complex64])
def test_unsupported_dtypes(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        O3MultiScaleResidual()(torch.zeros(1, 16, 9, 12, dtype=dtype))


@pytest.mark.parametrize("fault", ["leaf", "host_graph"])
def test_host_gradient_connection_is_rejected(fault):
    branch, h = fixture_components()
    h.requires_grad_(True)
    if fault == "host_graph":
        h = h * 2
    with pytest.raises(ValueError, match="detached"):
        branch(h)


@pytest.mark.parametrize("fault", ["input_nan", "input_inf", "weight_nan", "weight_inf", "scale_overflow", "branch_overflow"])
def test_nonfinite_data_parameters_and_intermediates_fail_closed(fault):
    branch, h = fixture_components()
    with torch.no_grad():
        if fault.startswith("input_"):
            h.flatten()[0] = float(fault.split("_")[1])
        elif fault.startswith("weight_"):
            branch.project.weight.flatten()[0] = float(fault.split("_")[1])
        elif fault == "scale_overflow":
            h.fill_(torch.finfo(h.dtype).max)
        else:
            h.fill_(1)
            branch.expand.weight.fill_(torch.finfo(h.dtype).max)
    with pytest.raises(ValueError, match="finite|overflow"):
        branch(h)


def test_dtype_mismatch():
    branch, h = fixture_components(torch.float64)
    with pytest.raises(TypeError, match="dtype mismatch"):
        branch(h.float())


def test_device_mismatch_without_accelerator():
    branch, h = fixture_components()
    branch.to("meta")
    with pytest.raises(ValueError, match="device mismatch"):
        branch(h)


def test_meta_features_rejected():
    with pytest.raises(ValueError, match="materialized"):
        O3MultiScaleResidual().to("meta")(torch.empty(1, 16, 9, 12, device="meta"))


def test_sparse_features_rejected():
    branch, h = fixture_components()
    with pytest.raises(ValueError, match="dense"):
        branch(h.to_sparse())


def test_nontensor_rejected():
    with pytest.raises(TypeError, match="torch.Tensor"):
        O3MultiScaleResidual()(None)
