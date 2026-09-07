"""CPU synthetic tests only: no dataset, pretrained weights, or accelerator."""

from __future__ import annotations

import inspect

import pytest
import torch

from tta.adapters.decoder_spatial_residual_v1 import DecoderSpatialResidual


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        yield
    torch.set_num_threads(previous)


def components(dtype=torch.float64, batch=1):
    return DecoderSpatialResidual().to(dtype=dtype), torch.randn(
        batch, 16, 7, 9, dtype=dtype
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 3])
def test_zero_kernel_is_exact_identity_without_changing_input(dtype, batch):
    adapter, x = components(dtype, batch)
    original = x.clone()
    result = adapter(x)
    assert torch.equal(result, original)
    assert torch.equal(result.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8))
    assert torch.equal(x, original)
    assert result.requires_grad


def test_exact_parameter_contract_no_rng_or_buffers():
    before = torch.random.get_rng_state()
    adapter = DecoderSpatialResidual()
    assert torch.equal(before, torch.random.get_rng_state())
    assert list(dict(adapter.named_parameters())) == ["kernel"]
    assert adapter.kernel.shape == (16, 1, 3, 3)
    assert adapter.kernel.numel() == 144
    assert torch.count_nonzero(adapter.kernel) == 0
    assert dict(adapter.named_buffers()) == {}
    assert len(list(adapter.modules())) == 1
    assert set(inspect.signature(adapter.forward).parameters) == {"x"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_first_step_gradient_is_finite_nonzero_and_updates_output(dtype):
    adapter, x = components(dtype)
    target_weight = torch.randn_like(x)
    (gradient,) = torch.autograd.grad((adapter(x) * target_weight).sum(), adapter.kernel)
    assert torch.isfinite(gradient).all()
    assert bool(torch.all(gradient.flatten(1).norm(dim=1) > 0))
    assert x.grad is None
    with torch.no_grad():
        adapter.kernel.add_(gradient, alpha=-0.001)
    assert not torch.equal(adapter(x), x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [0.0, 1e-20])
def test_zero_and_tiny_inputs_remain_finite(dtype, scale):
    adapter, x = components(dtype)
    x.mul_(scale)
    with torch.no_grad():
        adapter.kernel.fill_(0.7)
    result = adapter(x)
    assert torch.isfinite(result).all()
    result.sum().backward()
    assert torch.isfinite(adapter.kernel.grad).all()
    if scale == 0.0:
        assert torch.equal(result, x)
        assert torch.count_nonzero(adapter.kernel.grad) == 0


def test_zero_input_preserves_numerical_identity_including_negative_zero():
    adapter = DecoderSpatialResidual()
    x = torch.full((1, 16, 2, 2), -0.0)
    assert torch.equal(adapter(x), x)


def test_response_depends_on_local_content_and_does_not_mix_channels():
    adapter = DecoderSpatialResidual().double()
    x = torch.zeros(1, 16, 7, 9, dtype=torch.float64)
    x[0, 3, 3, 4] = 1.0
    with torch.no_grad():
        adapter.kernel[3].fill_(1.0)
    difference = adapter(x) - x
    assert torch.count_nonzero(difference[:, :3]) == 0
    assert torch.count_nonzero(difference[:, 4:]) == 0
    assert torch.count_nonzero(difference[:, 3]) == 9
    assert difference[0, 3, 2, 3] > 0
    assert difference[0, 3, 0, 0] == 0
    assert torch.unique(difference[:, 3]).numel() > 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_per_pixel_bound_with_saturating_kernel_and_per_image_rms(dtype):
    adapter, x = components(dtype, batch=3)
    x[1].mul_(100.0)
    x[2].mul_(1e-9)
    with torch.no_grad():
        adapter.kernel.copy_(torch.randn_like(adapter.kernel) * 1e4)
    scale = x.square().mean((1, 2, 3), keepdim=True).clamp_min(1e-12).sqrt()
    result = adapter(x)
    tolerance = 4 * torch.finfo(dtype).eps * (x.abs() + scale)
    assert bool(torch.all((result - x).abs() <= 0.05 * scale + tolerance))


def test_batch_processing_matches_independent_images():
    adapter, x = components(batch=3)
    with torch.no_grad():
        adapter.kernel.copy_(torch.randn_like(adapter.kernel))
    separate = torch.cat([adapter(sample[None]) for sample in x])
    assert torch.equal(adapter(x), separate)


def test_reset_restores_identity_clears_gradient_and_preserves_parameter_object():
    adapter, x = components()
    kernel = adapter.kernel
    adapter(x).sum().backward()
    with torch.no_grad():
        adapter.kernel.fill_(0.1)
    assert not torch.equal(adapter(x), x)
    assert adapter.reset_identity_() is adapter
    assert adapter.kernel is kernel
    assert adapter.kernel.grad is None
    assert torch.count_nonzero(adapter.kernel) == 0
    assert torch.equal(adapter(x), x)


def test_state_dict_round_trip_and_eval_train_parity():
    adapter, x = components()
    with torch.no_grad():
        adapter.kernel.normal_()
    replica = DecoderSpatialResidual().double()
    replica.load_state_dict(adapter.state_dict(), strict=True)
    assert torch.equal(replica(x), adapter(x))
    assert torch.equal(adapter.eval()(x), adapter.train()(x))


def test_float64_directional_finite_difference_at_identity():
    adapter, x = components()
    objective_weights = torch.randn_like(x)
    direction = torch.randn_like(adapter.kernel)
    direction /= direction.norm()
    objective = (adapter(x) * objective_weights).sum()
    (gradient,) = torch.autograd.grad(objective, adapter.kernel)
    analytical = (gradient * direction).sum()
    epsilon = 1e-5
    with torch.no_grad():
        adapter.kernel.copy_(epsilon * direction)
        plus = (adapter(x) * objective_weights).sum()
        adapter.kernel.copy_(-epsilon * direction)
        minus = (adapter(x) * objective_weights).sum()
    numerical = (plus - minus) / (2 * epsilon)
    assert torch.allclose(analytical, numerical, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("channels", [0, 1, 15, 17, True, 16.0])
def test_only_registered_channel_count_is_accepted(channels):
    with pytest.raises((TypeError, ValueError), match="16"):
        DecoderSpatialResidual(channels=channels)


@pytest.mark.parametrize("name", ["max_residual_ratio", "rms_floor"])
@pytest.mark.parametrize("value", [0, -1.0, float("nan"), float("inf"), True, "0.05"])
def test_invalid_hyperparameters_are_rejected(name, value):
    with pytest.raises((TypeError, ValueError), match=name):
        DecoderSpatialResidual(**{name: value})


@pytest.mark.parametrize("shape", [(1, 16, 7), (1, 15, 7, 9), (0, 16, 7, 9), (1, 16, 0, 9), (1, 16, 7, 0)])
def test_invalid_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="shape"):
        DecoderSpatialResidual()(torch.zeros(shape))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64, torch.complex64])
def test_unsupported_dtype_is_rejected(dtype):
    with pytest.raises(TypeError, match="float32 or float64"):
        DecoderSpatialResidual()(torch.zeros(1, 16, 7, 9, dtype=dtype))


@pytest.mark.parametrize("fault", ["input_nan", "input_inf", "kernel_nan", "kernel_inf", "rms_overflow", "conv_overflow"])
def test_nonfinite_inputs_parameters_or_intermediates_are_rejected(fault):
    adapter, x = components(torch.float32)
    with torch.no_grad():
        if fault.startswith("input_"):
            x[0, 0, 0, 0] = float(fault.split("_")[1])
        elif fault.startswith("kernel_"):
            adapter.kernel[0, 0, 0, 0] = float(fault.split("_")[1])
        elif fault == "rms_overflow":
            x.fill_(torch.finfo(x.dtype).max)
        else:
            x.fill_(1.0)
            adapter.kernel.fill_(torch.finfo(x.dtype).max)
    with pytest.raises(ValueError, match="finite|overflow"):
        adapter(x)


def test_differentiable_source_input_is_rejected():
    adapter, x = components()
    with pytest.raises(ValueError, match="detached"):
        adapter(x.requires_grad_())


def test_kernel_dtype_mismatch_is_rejected():
    adapter, x = components()
    with pytest.raises(TypeError, match="dtype mismatch"):
        adapter(x.float())


def test_kernel_device_mismatch_is_rejected_without_gpu():
    adapter, x = components()
    with pytest.raises(ValueError, match="device mismatch"):
        adapter(x.to("meta"))


def test_meta_device_is_rejected_even_if_kernel_device_matches():
    adapter, x = components()
    with pytest.raises(ValueError, match="materialized"):
        adapter.to("meta")(x.to("meta"))


def test_sparse_input_is_rejected():
    adapter, x = components()
    with pytest.raises(ValueError, match="dense"):
        adapter(x.to_sparse())


def test_non_tensor_input_is_rejected():
    with pytest.raises(TypeError, match="torch.Tensor"):
        DecoderSpatialResidual()(None)


def test_underflowing_rms_floor_is_rejected_for_feature_dtype():
    adapter = DecoderSpatialResidual(rms_floor=1e-30)
    with pytest.raises(ValueError, match="representable"):
        adapter(torch.zeros(1, 16, 7, 9))


@pytest.mark.parametrize("floor", [1e-200, 1e200])
def test_underflowing_or_overflowing_squared_floor_is_rejected_at_construction(floor):
    with pytest.raises(ValueError, match="squared"):
        DecoderSpatialResidual(rms_floor=floor)


def test_extra_repr_contains_numerical_contract():
    result = repr(DecoderSpatialResidual())
    assert "channels=16" in result
    assert "max_residual_ratio=0.05" in result
    assert "rms_floor=1e-06" in result
