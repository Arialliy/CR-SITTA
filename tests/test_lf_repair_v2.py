"""CPU/synthetic-only correctness tests for the append-only LF-v2 operator."""

from dataclasses import FrozenInstanceError
import inspect
import json
import math

import pytest
import torch

from tta.deteriorations.fourier_low_mask_v2 import (
    LFConfig, LFOutput, lf_mask_diagnostic, lf_soft_mask,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def gen(seed=42):
    return torch.Generator(device="cpu").manual_seed(seed)


def sample(height=32, width=32, *, batch=2, channels=3, dtype=torch.float64):
    return torch.rand(batch, channels, height, width, dtype=dtype, generator=gen(17))


def run(image, *, attenuation=0.25, keep=0.5, protect_dc=True, shared_channels=True, seed=42):
    return lf_mask_diagnostic(
        image, LFConfig(attenuation=attenuation, pair_keep_probability=keep),
        generator=gen(seed), protect_dc=protect_dc, shared_channels=shared_channels,
    )


@pytest.mark.parametrize("shape", [(2, 2), (3, 5), (31, 32), (32, 31), (224, 224), (256, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_geometry_mean_dc_mixing_parseval(shape, dtype):
    height, width = shape
    x = sample(height, width, dtype=dtype)
    before = x.clone()
    out = run(x)
    hard = run(x, attenuation=1)
    iy = (2 * (height // 2) - torch.arange(height)) % height
    ix = (2 * (width // 2) - torch.arange(width)) % width
    assert torch.equal(out.gain, out.gain.index_select(-2, iy).index_select(-1, ix))
    assert torch.equal(out.support, out.support.index_select(0, iy).index_select(1, ix))
    assert torch.equal(x, before)
    assert not out.support[height // 2, width // 2]
    assert bool((out.gain[..., height // 2, width // 2] == 1).all())
    eps = torch.finfo(dtype).eps
    assert out.diagnostics["imaginary_max"] <= 64 * eps
    torch.testing.assert_close(out.preclip.mean((-2, -1)), x.mean((-2, -1)), rtol=0, atol=8 * eps)
    torch.testing.assert_close(out.preclip, 0.75 * x + 0.25 * hard.preclip, rtol=0, atol=8 * eps)
    spectrum = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
    bound = 0.25**2 * spectrum[..., out.support].abs().square().sum()
    actual = (out.preclip - x).square().sum()
    assert float(actual) <= float(bound) + 64 * eps * max(1.0, float(bound))
    assert bool(((out.image >= 0) & (out.image <= 1)).all())


def test_full_random_field_is_paired_across_all_controls_and_identity():
    x = sample()
    controls = [
        dict(attenuation=1, protect_dc=False, shared_channels=False),
        dict(attenuation=1, protect_dc=True, shared_channels=False),
        dict(attenuation=1, protect_dc=True, shared_channels=True),
        dict(attenuation=0.25), dict(attenuation=0.5),
        dict(attenuation=0), dict(keep=1),
    ]
    outputs = [run(x, **control) for control in controls]
    random_hashes = {output.diagnostics["random_field_sha256"] for output in outputs}
    assert len(random_hashes) == 1
    assert len({output.diagnostics["full_support_sha256"] for output in outputs}) == 1
    assert len({output.diagnostics["orbit_representatives_sha256"] for output in outputs}) == 1
    assert outputs[0].diagnostics["random_field_shape"] == [2, 3, outputs[0].diagnostics["all_orbit_count"]]
    assert outputs[0].diagnostics["active_orbit_count"] == outputs[1].diagnostics["active_orbit_count"] + 1
    without_dc = outputs[1].support
    assert torch.equal(outputs[0].gain[..., without_dc], outputs[1].gain[..., without_dc])
    assert torch.equal(outputs[1].gain[:, :1], outputs[2].gain)
    assert torch.equal(outputs[3].gain, 0.75 + 0.25 * outputs[2].gain)
    assert torch.equal(outputs[4].gain, 0.50 + 0.50 * outputs[2].gain)
    assert outputs[0].support[x.shape[-2] // 2, x.shape[-1] // 2]


def test_field_is_actual_float64_cpu_draw_with_sorted_canonical_orbits():
    x = sample()
    out = run(x, protect_dc=False, shared_channels=False)
    stats = out.diagnostics
    actual = torch.tensor(stats["random_field_values"], dtype=torch.float64)
    expected = torch.rand(stats["random_field_shape"], dtype=torch.float64, generator=gen())
    assert torch.equal(actual, expected)
    reps = stats["orbit_representatives"]
    assert reps == sorted(set(reps))
    height, width = x.shape[-2:]
    for orbit, rep in enumerate(reps):
        y, z = divmod(rep, width)
        expected_gain = 1 - 0.25 * (actual[..., orbit] >= 0.5).to(x.dtype)
        assert torch.equal(out.gain[..., y, z], expected_gain)
        partner_y, partner_x = (2 * (height // 2) - y) % height, (2 * (width // 2) - z) % width
        assert rep <= partner_y * width + partner_x
        assert torch.equal(out.gain[..., y, z], out.gain[..., partner_y, partner_x])


def test_pair_keep_probability_is_q_not_q_squared():
    out = run(sample(16, 16, batch=2048, channels=1), attenuation=1)
    stats = out.diagnostics
    active_reps = [rep for rep in stats["orbit_representatives"] if rep != 8 * 16 + 8]
    kept = torch.stack([out.gain[..., rep // 16, rep % 16] == 1 for rep in active_reps], dim=-1)
    fraction = float(kept.double().mean())
    n = kept.numel()
    assert abs(fraction - 0.5) < 6 * math.sqrt(0.25 / n)
    assert fraction > 0.4
    assert fraction == stats["sampled_pair_keep_fraction"]


@pytest.mark.parametrize("attenuation,keep", [(0.0, 0.5), (0.25, 1.0)])
def test_identity_exact_independently_stored_and_random_field_consumed(attenuation, keep):
    x = sample()
    cfg = LFConfig(attenuation=attenuation, pair_keep_probability=keep)
    generator = gen()
    output = lf_soft_mask(x, cfg, generator=generator)
    assert torch.equal(output.image, x)
    assert torch.equal(output.preclip, x)
    assert output.image.data_ptr() != x.data_ptr()
    assert output.preclip.data_ptr() != x.data_ptr()
    assert output.diagnostics["identity_control"] is True
    assert output.diagnostics["preclip_rms"] == 0
    assert output.diagnostics["postclip_rms"] == 0
    assert bool((output.gain == 1).all())
    expected_generator = gen()
    torch.rand(output.diagnostics["random_field_shape"], dtype=torch.float64, generator=expected_generator)
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_same_seed_exact_replay_order_independence_and_global_rng_unchanged():
    x = sample()
    global_before = torch.random.get_rng_state().clone()
    first = run(x, seed=4)
    run(x, seed=8, protect_dc=False, shared_channels=False)
    repeated = run(x, seed=4)
    assert torch.equal(first.image, repeated.image)
    assert torch.equal(first.preclip, repeated.preclip)
    assert torch.equal(first.gain, repeated.gain)
    assert first.diagnostics == repeated.diagnostics
    assert torch.equal(global_before, torch.random.get_rng_state())
    assert run(x, seed=5).diagnostics["random_field_sha256"] != first.diagnostics["random_field_sha256"]


def test_production_exactly_matches_shared_protected_diagnostic():
    x = sample()
    cfg = LFConfig()
    production = lf_soft_mask(x, cfg, generator=gen())
    diagnostic = lf_mask_diagnostic(x, cfg, generator=gen(), protect_dc=True, shared_channels=True)
    assert isinstance(production, LFOutput)
    assert torch.equal(production.image, diagnostic.image)
    assert torch.equal(production.preclip, diagnostic.preclip)
    assert torch.equal(production.gain, diagnostic.gain)
    assert production.diagnostics == diagnostic.diagnostics


def test_physical_gray_rgb_sharing_and_independent_control():
    x = sample(channels=1).expand(-1, 3, -1, -1).clone()
    shared = run(x, attenuation=1)
    separate = run(x, attenuation=1, shared_channels=False)
    assert torch.equal(shared.preclip[:, 0], shared.preclip[:, 1])
    assert torch.equal(shared.image[:, 0], shared.image[:, 2])
    assert not torch.equal(separate.gain[:, 0], separate.gain[:, 1])
    assert not torch.equal(separate.preclip[:, 0], separate.preclip[:, 1])


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_constant_images_not_cleared_and_zero_energy_is_explicit(value):
    x = torch.full((1, 3, 32, 32), value, dtype=torch.float64)
    output = run(x)
    torch.testing.assert_close(output.image, x, rtol=0, atol=1e-14)
    assert output.diagnostics["preclip_mean_drift_max"] < 1e-14
    assert output.diagnostics["preclip_support_energy_retention"] is None
    if value == 0:
        assert output.diagnostics["preclip_total_energy_retention"] is None
    json.dumps(output.diagnostics, allow_nan=False)


def test_soft_gain_is_not_recorded_as_keep_probability():
    output = run(sample())
    stats = output.diagnostics
    assert 0 < stats["sampled_pair_keep_fraction"] < 1
    assert stats["gain_min"] == 0.75
    assert stats["gain_mean"] > stats["sampled_pair_keep_fraction"]
    assert not any("gain" in key and "keep" in key for key in stats)


def test_energy_retention_matches_image_weighted_spectral_energy():
    x = sample()
    output = run(x)
    spectrum = torch.fft.fftshift(torch.fft.fft2(x, norm="ortho"), dim=(-2, -1))
    energy = spectrum.abs().square()
    expected = float((energy * output.gain.square()).sum() / energy.sum())
    assert output.diagnostics["preclip_total_energy_retention"] == pytest.approx(expected, abs=2e-14)
    expected_support = float((energy * output.gain.square())[..., output.support].sum() / energy[..., output.support].sum())
    assert output.diagnostics["preclip_support_energy_retention"] == pytest.approx(expected_support, abs=2e-14)
    assert abs(expected - float(output.gain.square().mean())) > 1e-4


def test_clip_once_not_hard_branch_clamp_before_mix():
    x = sample()
    hard = run(x, attenuation=1)
    soft = run(x, attenuation=0.25)
    expected = (0.75 * x + 0.25 * hard.preclip).clamp(0, 1)
    wrong = 0.75 * x + 0.25 * hard.image
    torch.testing.assert_close(soft.image, expected, rtol=0, atol=1e-14)
    assert float((soft.image - wrong).abs().max()) > 1e-5
    assert soft.diagnostics["actual_clipped_fraction"] == float(((soft.preclip < 0) | (soft.preclip > 1)).double().mean())
    assert soft.diagnostics["postclip_endpoint_fraction"] == float(((soft.image <= 0) | (soft.image >= 1)).double().mean())


def test_endpoint_occupancy_is_not_actual_clipping():
    x = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    output = run(x)
    assert output.diagnostics["actual_clipped_fraction"] == 0
    assert output.diagnostics["postclip_endpoint_fraction"] == 1


def test_signed_contrast_mixes_linearly_before_clamp():
    x = sample()
    soft, hard = run(x), run(x, attenuation=1)
    def contrast(value):
        gray = value.mean(1)
        return gray[:, 11:14, 11:14].mean() - gray[:, 2:7, 2:7].mean()
    assert float(contrast(soft.preclip)) == pytest.approx(float(0.75 * contrast(x) + 0.25 * contrast(hard.preclip)), abs=2e-14)


def test_noncontiguous_input_is_supported_and_not_mutated():
    x = sample(31, 32).transpose(-2, -1)
    assert not x.is_contiguous()
    original = x.clone()
    output = run(x)
    assert torch.equal(x, original)
    assert output.image.shape == x.shape


def test_input_gradient_available_without_inplace_modification():
    x = sample(batch=1).requires_grad_()
    output = run(x)
    gradient, = torch.autograd.grad(output.image.square().mean(), x)
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0
    assert output.gain.requires_grad is False


@pytest.mark.parametrize("name,value,error", [
    ("mask_ratio", 0, ValueError), ("mask_ratio", 0.5, ValueError),
    ("pair_keep_probability", 0, ValueError), ("pair_keep_probability", 1.01, ValueError),
    ("attenuation", -0.1, ValueError), ("attenuation", 1.1, ValueError),
    ("mask_ratio", float("nan"), ValueError), ("attenuation", float("inf"), ValueError),
    ("pair_keep_probability", True, TypeError), ("mask_ratio", "0.2", TypeError),
    ("attenuation", None, TypeError),
])
def test_invalid_config_rejected(name, value, error):
    with pytest.raises(error):
        LFConfig(**{name: value})


@pytest.mark.parametrize("x,error", [
    (None, TypeError), (torch.zeros(3, 16, 16), ValueError),
    (torch.zeros(0, 3, 16, 16), ValueError), (torch.zeros(1, 3, 1, 16), ValueError),
    (torch.zeros(1, 3, 16, 16, dtype=torch.float16), TypeError),
    (torch.zeros(1, 3, 16, 16, dtype=torch.int64), TypeError),
    (torch.zeros(1, 3, 16, 16, dtype=torch.complex64), TypeError),
    (torch.full((1, 3, 16, 16), -0.1), ValueError),
    (torch.full((1, 3, 16, 16), 1.1), ValueError),
    (torch.full((1, 3, 16, 16), float("nan")), ValueError),
    (torch.full((1, 3, 16, 16), float("inf")), ValueError),
])
def test_invalid_input_rejected(x, error):
    with pytest.raises(error):
        run(x)


def test_cfg_generator_and_boolean_flags_are_typed():
    x = sample()
    with pytest.raises(TypeError):
        lf_soft_mask(x, {}, generator=gen())
    with pytest.raises(TypeError):
        lf_soft_mask(x, LFConfig(), generator=None)
    with pytest.raises(TypeError):
        lf_mask_diagnostic(x, LFConfig(), generator=gen(), protect_dc=1, shared_channels=True)
    with pytest.raises(TypeError):
        lf_mask_diagnostic(x, LFConfig(), generator=gen(), protect_dc=True, shared_channels="yes")


def test_only_label_free_image_config_generator_interfaces_and_frozen_config():
    assert list(inspect.signature(lf_soft_mask).parameters) == ["image", "cfg", "generator"]
    assert list(inspect.signature(lf_mask_diagnostic).parameters) == ["image", "cfg", "generator", "protect_dc", "shared_channels"]
    cfg = LFConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.attenuation = 0.5
    with pytest.raises(TypeError):
        lf_soft_mask(sample(), cfg, generator=gen(), target=torch.zeros(1))
