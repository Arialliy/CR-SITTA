"""Synthetic CPU-only checks; no checkpoint, image, label, or GPU loading."""
from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from model.o3_anchored_residual_v4 import O3AnchoredResidualV4
from model.o3_multiscale_residual_v1 import O3MultiScaleResidual


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def trained_fixture(dtype=torch.float32):
    # Nonzero, deterministic synthetic weights stand in for a trained v1.
    # No CUDA RNG snapshot or global CPU RNG consumption is needed.
    with torch.random.fork_rng(devices=[]):
        reference = O3MultiScaleResidual().to(dtype=dtype)
    with torch.no_grad():
        reference.expand.weight.fill_(0.25)
        reference.dw3.weight.fill_(0.05)
        reference.dw5.weight.fill_(0.05)
        reference.project.weight.fill_(0.01)
    return reference.eval()


def feature(dtype=torch.float32):
    return torch.linspace(0.2, 1.2, 16 * 7 * 9, dtype=dtype).reshape(1, 16, 7, 9)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonzero_trained_anchor_is_exact_initial_identity(dtype):
    reference = trained_fixture(dtype)
    model = O3AnchoredResidualV4(reference.state_dict())
    h = feature(dtype)
    expected = reference(h)
    assert not torch.equal(expected, h)
    assert torch.equal(model(h), expected)
    assert next(model.parameters()).dtype == dtype
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 1568
    assert sum(p.numel() for p in model.parameters() if not p.requires_grad) == 1568
    assert [id(p) for p in model.learnable_parameters()] == [id(p) for p in model.live.parameters()]
    assert set(model.state_dict()) == {f"{prefix}.{name}" for prefix in ("anchor", "live")
                                      for name in reference.state_dict()}
    assert not any(isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.modules.dropout._DropoutNd))
                   for m in model.modules())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_first_step_all_live_gradients_nonzero_and_frozen_anchor_unchanged(dtype):
    model = O3AnchoredResidualV4(trained_fixture(dtype).state_dict())
    anchor_before = {k: v.clone() for k, v in model.anchor.state_dict().items()}
    live_before = {k: v.clone() for k, v in model.live.state_dict().items()}
    h = feature(dtype)
    h_before = h.clone()
    optimizer = torch.optim.SGD(model.learnable_parameters(), lr=0.01)
    model(h).square().mean().backward()
    for parameter in model.live.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    assert all(p.grad is None and not p.requires_grad for p in model.anchor.parameters())
    optimizer.step()
    assert all(torch.equal(value, model.anchor.state_dict()[name]) for name, value in anchor_before.items())
    assert all(not torch.equal(value, model.live.state_dict()[name]) for name, value in live_before.items())
    assert torch.equal(h, h_before) and h.grad is None


def test_positive_and_negative_new_directions_can_exceed_original_bound():
    model = O3AnchoredResidualV4(trained_fixture().state_dict())
    h = torch.ones(1, 16, 9, 9)
    anchored = model.anchor(h)
    with torch.no_grad():
        model.live.project.weight[:8].add_(1.0)
        model.live.project.weight[8:].sub_(1.0)
    actual = model(h)
    assert ((actual - anchored)[:, :8] > 0.05).all()
    assert ((actual - anchored)[:, 8:] < -0.05).all()
    assert ((actual - h)[:, :8] > 0.05).all()
    assert ((actual - h)[:, 8:] < -0.05).all()
    # Old v1's residual is positive in every channel; the new correction is not
    # a nonnegative scalar rescaling of that existing direction.
    assert ((anchored - h) > 0).all()
    assert torch.isfinite(actual).all()


def test_exact_requested_preactivation_formula_after_live_update():
    model = O3AnchoredResidualV4(trained_fixture(torch.float64).state_dict())
    h = feature(torch.float64)
    with torch.no_grad():
        model.live.project.weight.add_(0.2)
    scale = h.square().mean((1, 2, 3), keepdim=True).clamp_min(1e-12).sqrt()
    def u(network):
        left, right = network.expand(h / scale).chunk(2, dim=1)
        return network.project(torch.cat((network.relu3(network.dw3(left)),
                                          network.relu5(network.dw5(right))), dim=1))
    expected = model.anchor(h) + 0.05 * scale * (u(model.live) - u(model.anchor))
    assert torch.equal(model(h), expected)


def test_train_and_eval_never_enable_anchor_training():
    model = O3AnchoredResidualV4(trained_fixture().state_dict())
    for mode in (True, False, True):
        assert model.train(mode) is model
        assert model.training is mode and model.live.training is mode
        assert not any(m.training for m in model.anchor.modules())
        assert all(not p.requires_grad for p in model.anchor.parameters())
    with pytest.raises(ValueError):
        model.train("yes")


@pytest.mark.parametrize("tamper", ["training", "requires_grad"])
def test_direct_anchor_tampering_fails_closed(tamper):
    model = O3AnchoredResidualV4(trained_fixture().state_dict())
    if tamper == "training":
        model.anchor.train()
    else:
        model.anchor.requires_grad_(True)
    with pytest.raises(RuntimeError, match="anchor"):
        model(feature())


def test_full_state_roundtrip_and_no_alias_to_input_or_anchor():
    reference = trained_fixture()
    supplied = reference.state_dict()
    model = O3AnchoredResidualV4(supplied)
    for name, live in model.live.state_dict().items():
        assert live.data_ptr() != model.anchor.state_dict()[name].data_ptr()
        assert live.data_ptr() != supplied[name].data_ptr()
        assert model.anchor.state_dict()[name].data_ptr() != supplied[name].data_ptr()
    with torch.no_grad():
        model.live.project.weight.add_(0.03)
    state = copy.deepcopy(model.state_dict())
    restored = O3AnchoredResidualV4(trained_fixture().state_dict())
    restored.load_state_dict(state, strict=True)
    assert torch.equal(model(feature()), restored(feature()))
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in state.items())
    assert all(not p.requires_grad for p in restored.anchor.parameters())


@pytest.mark.parametrize("updated", [False, True])
def test_each_image_is_independent_of_batch_peers(updated):
    model = O3AnchoredResidualV4(trained_fixture(torch.float64).state_dict())
    if updated:
        with torch.no_grad():
            model.live.project.weight.add_(0.02)
    h = feature(torch.float64)
    peer = torch.flip(h, (-1,)) * 11.0
    together = model(torch.cat((h, peer), dim=0))
    separate = torch.cat((model(h), model(peer)), dim=0)
    torch.testing.assert_close(together, separate, rtol=1e-12, atol=1e-12)


def test_constructor_preserves_cpu_rng_without_cuda_rng_access(monkeypatch):
    state = trained_fixture().state_dict()
    before = torch.get_rng_state().clone()
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic CPU construction must not access CUDA RNG")
    monkeypatch.setattr(torch.cuda, "get_rng_state", forbidden)
    monkeypatch.setattr(torch.cuda, "set_rng_state", forbidden)
    model = O3AnchoredResidualV4(state)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(model(feature()), model.anchor(feature()))


@pytest.mark.parametrize("case", ["not_tensor", "sparse", "rank", "channels", "empty", "half", "integer",
                                  "requires_grad", "nonleaf", "nan", "inf", "meta", "dtype_mismatch", "rms_overflow"])
def test_invalid_feature_inputs_rejected(case):
    model = O3AnchoredResidualV4(trained_fixture().state_dict())
    h = feature()
    if case == "not_tensor": h = h.tolist()
    elif case == "sparse": h = h.to_sparse()
    elif case == "rank": h = h[0]
    elif case == "channels": h = h[:, :15]
    elif case == "empty": h = h[:, :, :0]
    elif case == "half": h = h.half()
    elif case == "integer": h = h.long()
    elif case == "requires_grad": h.requires_grad_()
    elif case == "nonleaf": h = h.requires_grad_() * 2
    elif case == "nan": h[0, 0, 0, 0] = float("nan")
    elif case == "inf": h[0, 0, 0, 0] = float("inf")
    elif case == "meta": h = torch.empty(h.shape, device="meta")
    elif case == "dtype_mismatch": h = h.double()
    elif case == "rms_overflow": h.fill_(1e30)
    with pytest.raises((ValueError, TypeError)):
        model(h)


@pytest.mark.parametrize("which", ["anchor", "live"])
def test_nonfinite_branch_parameters_rejected(which):
    model = O3AnchoredResidualV4(trained_fixture().state_dict())
    with torch.no_grad():
        getattr(model, which).project.weight[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(feature())


def test_live_device_mismatch_rejected_without_reading_meta_parameters():
    model = O3AnchoredResidualV4(trained_fixture().state_dict())
    model.live.to("meta")
    with pytest.raises(ValueError, match="device mismatch"):
        model(feature())


def test_finite_parameters_with_overflowing_preactivations_rejected():
    model = O3AnchoredResidualV4(trained_fixture().state_dict())
    with torch.no_grad():
        model.live.expand.weight.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError, match="finite"):
        model(torch.ones(1, 16, 7, 9))


@pytest.mark.parametrize("case", ["empty", "not_mapping", "not_tensor", "missing", "extra", "shape",
                                  "nan", "half", "mixed_dtype", "meta", "sparse"])
def test_invalid_anchor_state_rejected(case):
    state = copy.deepcopy(trained_fixture().state_dict())
    key = "project.weight"
    if case == "empty": state = {}
    elif case == "not_mapping": state = list(state.values())
    elif case == "not_tensor": state[key] = 1
    elif case == "missing": del state[key]
    elif case == "extra": state["unexpected"] = torch.ones(1)
    elif case == "shape": state[key] = torch.zeros(1)
    elif case == "nan": state[key][0, 0, 0, 0] = float("nan")
    elif case == "half": state = {k: v.half() for k, v in state.items()}
    elif case == "mixed_dtype": state[key] = state[key].double()
    elif case == "meta": state[key] = torch.empty(state[key].shape, device="meta")
    elif case == "sparse": state[key] = state[key].to_sparse()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        O3AnchoredResidualV4(state)
