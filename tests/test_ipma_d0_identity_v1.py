"""CPU-only synthetic identity/state tests; no source dataset or checkpoint."""
from __future__ import annotations

import inspect
import math

import pytest
import torch

from model.ipma_d0_adapter_v1 import IdentityMetaAdapter
from training.ipma_inner_v1 import inner_step, teacher_region_weights


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        yield
    torch.set_num_threads(previous)


def components(dtype=torch.float64):
    adapter = IdentityMetaAdapter().to(dtype=dtype)
    head = torch.nn.Conv2d(16, 1, 1).to(dtype=dtype).requires_grad_(False)
    features = torch.randn(1, 16, 6, 7, dtype=dtype)
    return adapter, head, features


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_delta_identity_before_and_after_phi_changes(dtype):
    adapter, head, features = components(dtype)
    delta = features.new_zeros(5, requires_grad=True)
    before = head(features)
    assert torch.equal(adapter(features, delta), features)
    assert torch.equal(head(adapter(features, delta)), before)
    with torch.no_grad():
        adapter.down.weight.add_(torch.randn_like(adapter.down.weight) * 0.02)
        adapter.up.weight.mul_(1.3)
    assert torch.equal(adapter(features, delta), features)
    assert torch.equal(head(adapter(features, delta)), before)


def test_nonzero_initial_weights_and_zero_delta_have_nonzero_jacobian():
    adapter, head, features = components()
    assert torch.count_nonzero(adapter.down.weight) > 0
    assert torch.count_nonzero(adapter.up.weight) > 0
    delta = features.new_zeros(5, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(lambda d: head(adapter(features, d)), delta)
    assert jacobian.shape == (1, 1, 6, 7, 5)
    assert torch.isfinite(jacobian).all()
    assert all(float(jacobian[..., k].norm()) > 0 for k in range(5))


def test_double_zero_counterexample_has_no_delta_jacobian():
    adapter, head, features = components()
    with torch.no_grad():
        adapter.up.weight.zero_()
    delta = features.new_zeros(5, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(lambda d: head(adapter(features, d)), delta)
    assert torch.count_nonzero(jacobian) == 0
    assert torch.equal(adapter(features, delta), features)


def test_rms_normalization_retains_global_bound_and_not_an_identity_branch():
    adapter, _, features = components()
    delta = features.new_tensor([0.001, -0.003, 0.002, 0.004, -0.005])
    basis = adapter.bases(features)
    source_rms = features.square().mean().sqrt()
    each_rms = basis.square().mean((2, 3, 4)).sqrt()
    assert torch.allclose(each_rms, source_rms.expand_as(each_rms), atol=1e-12)
    residual_rms = (adapter(features, delta) - features).square().mean().sqrt()
    assert residual_rms <= math.sqrt(5) * delta.norm() * source_rms + 1e-12
    assert residual_rms > 0


@pytest.mark.parametrize("scale", [0.0, 1e-20])
def test_zero_and_tiny_features_remain_finite_at_rms_floor(scale):
    adapter, _, features = components()
    features *= scale
    delta = features.new_zeros(5, requires_grad=True)
    output = adapter(features, delta)
    assert torch.equal(output, features)
    assert torch.isfinite(adapter.bases(features)).all()
    (gradient,) = torch.autograd.grad(output.sum(), delta, create_graph=True)
    assert torch.isfinite(gradient).all()


def test_episode_delta_never_becomes_parameter_buffer_or_optimizer_state():
    adapter, head, features = components()
    teacher = head(features).sigmoid().detach()
    delta, _, _ = inner_step(adapter, head, features + 0.2,
        teacher, teacher_region_weights(teacher), create_graph=True)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.001)
    parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(delta) not in parameter_ids
    assert set(dict(adapter.named_parameters())) == {"down.weight", "up.weight"}
    assert set(dict(adapter.named_buffers())) == {"kernels"}
    assert all("delta" not in key for key in adapter.state_dict())
    assert len(optimizer.state_dict()["param_groups"][0]["params"]) == 2
    assert optimizer.state_dict()["state"] == {}


def test_a_b_a_functional_reset_preserves_all_module_state_and_grad_fields():
    adapter, head, observed = components()
    frozen = {name: value.clone() for name, value in adapter.state_dict().items()}
    head_before = {name: value.clone() for name, value in head.state_dict().items()}
    def episode(features):
        teacher = head(features).sigmoid().detach()
        return inner_step(adapter, head, features + 0.1, teacher,
                          teacher_region_weights(teacher), create_graph=True)
    first = episode(observed)
    episode(torch.randn_like(observed))
    third = episode(observed)
    assert all(torch.equal(a, b) for a, b in zip(first, third, strict=True))
    assert all(torch.equal(adapter.state_dict()[k], v) for k, v in frozen.items())
    assert all(torch.equal(head.state_dict()[k], v) for k, v in head_before.items())
    assert all(p.grad is None for p in adapter.parameters())
    assert all(p.grad is None for p in head.parameters())


@pytest.mark.parametrize("channels,rank", [(8, 4), (16, 2), (True, 4), (16, 4.0)])
def test_nonregistered_adapter_dimensions_are_rejected(channels, rank):
    with pytest.raises(ValueError, match="fixes"):
        IdentityMetaAdapter(channels, rank)


@pytest.mark.parametrize("fault", ["shape", "dtype", "nondetached", "nan", "delta_shape", "delta_nan", "delta_dtype", "state_nan"])
def test_adapter_rejects_invalid_features_delta_or_state(fault):
    adapter, _, features = components()
    delta = features.new_zeros(5)
    if fault == "shape": features = features[:, :8]
    if fault == "dtype": features, delta = features.half(), delta.half()
    if fault == "nondetached": features.requires_grad_(True)
    if fault == "nan": features[0, 0, 0, 0] = float("nan")
    if fault == "delta_shape": delta = delta[:4]
    if fault == "delta_nan": delta[0] = float("nan")
    if fault == "delta_dtype": delta = delta.float()
    if fault == "state_nan":
        with torch.no_grad(): adapter.down.weight[0, 0] = float("nan")
    with pytest.raises((TypeError, ValueError), match="mismatch|float32|detached|finite|episode"):
        adapter(features, delta)


def test_meta_device_mismatch_rejected_without_any_gpu():
    adapter, _, features = components()
    with pytest.raises(ValueError, match="device"):
        adapter(features, torch.zeros(5, dtype=features.dtype, device="meta"))


def test_public_adapter_and_inner_interfaces_do_not_accept_labels_or_paths():
    allowed = {"self", "h", "delta"}
    assert set(inspect.signature(IdentityMetaAdapter.forward).parameters) == allowed
    assert set(inspect.signature(inner_step).parameters) == {
        "adapter", "head", "h_probe", "teacher", "weights", "learning_rate", "radius", "create_graph"}
