from __future__ import annotations

import inspect

import pytest
import torch

from tta.deteriorations import (
    identity_deterioration,
    imagenet_denormalize,
    imagenet_normalize,
    inject_high_frequency_noise,
    inject_high_frequency_noise_components,
    mask_low_frequency_amplitude,
)


def _generator(seed: int) -> torch.Generator:
    result = torch.Generator(device="cpu")
    result.manual_seed(seed)
    return result


def test_clean_identity_is_bit_exact_and_independently_stored() -> None:
    image = torch.linspace(0.0, 1.0, 3 * 16 * 18).reshape(1, 3, 16, 18)
    clean = identity_deterioration(image)
    assert torch.equal(clean, image)
    assert clean.data_ptr() != image.data_ptr()

    fft_identity = mask_low_frequency_amplitude(
        image,
        mask_ratio=0.2,
        keep_probability=1.0,
        generator=_generator(3),
    )
    assert torch.equal(fft_identity.amplitude_mask, torch.ones_like(image))
    assert torch.allclose(fft_identity.image, image, atol=2.0e-6, rtol=0.0)


def test_explicit_imagenet_round_trip_preserves_physical_image() -> None:
    image = torch.rand(2, 3, 16, 18, generator=_generator(4))
    normalized = imagenet_normalize(image)
    recovered = imagenet_denormalize(normalized)
    assert torch.allclose(recovered, image, atol=6.0e-8, rtol=0.0)


def test_denormalization_rejects_nonphysical_preprocessing_drift() -> None:
    with pytest.raises(ValueError, match=r"physical \[0,1\]"):
        imagenet_denormalize(torch.full((1, 3, 8, 8), 100.0))


def test_low_frequency_mask_is_seed_exact_shape_preserving_and_symmetric() -> None:
    image = torch.rand(2, 3, 18, 20, generator=_generator(9))
    first = mask_low_frequency_amplitude(
        image,
        mask_ratio=0.3,
        keep_probability=0.4,
        generator=_generator(77),
    )
    second = mask_low_frequency_amplitude(
        image.clone(),
        mask_ratio=0.3,
        keep_probability=0.4,
        generator=_generator(77),
    )
    assert torch.equal(first.image, second.image)
    assert torch.equal(first.amplitude_mask, second.amplitude_mask)
    assert first.image.shape == image.shape
    assert first.image.dtype == image.dtype
    assert first.image.device == image.device
    assert (first.image >= 0).all() and (first.image <= 1).all()

    height, width = image.shape[-2:]
    cy, cx = height // 2, width // 2
    iy = torch.tensor([(2 * cy - i) % height for i in range(height)])
    ix = torch.tensor([(2 * cx - i) % width for i in range(width)])
    mirrored = first.amplitude_mask.index_select(-2, iy).index_select(-1, ix)
    assert torch.equal(first.amplitude_mask, mirrored)


def test_low_frequency_mask_reduces_central_amplitude_and_keeps_retained_phase() -> None:
    yy, xx = torch.meshgrid(
        torch.arange(32, dtype=torch.float32),
        torch.arange(32, dtype=torch.float32),
        indexing="ij",
    )
    image = (0.5 + 0.15 * torch.sin(2 * torch.pi * xx / 16)).reshape(1, 1, 32, 32)
    result = mask_low_frequency_amplitude(
        image,
        mask_ratio=0.25,
        keep_probability=0.1,
        generator=_generator(1),
    )
    spectrum = torch.fft.fftshift(torch.fft.fft2(image, norm="ortho"))
    masked = spectrum * result.amplitude_mask
    assert masked.abs().sum() < spectrum.abs().sum()
    retained = result.amplitude_mask.bool() & (spectrum.abs() > 1.0e-6)
    assert torch.allclose(torch.angle(masked[retained]), torch.angle(spectrum[retained]))


def test_high_frequency_noise_is_seed_exact_and_has_requested_preclip_rms() -> None:
    image = torch.full((2, 3, 32, 32), 0.5)
    first = inject_high_frequency_noise_components(
        image,
        target_rms=0.02,
        low_cut_ratio=0.2,
        generator=_generator(123),
    )
    second = inject_high_frequency_noise_components(
        image,
        target_rms=0.02,
        low_cut_ratio=0.2,
        generator=_generator(123),
    )
    assert torch.equal(first.image, second.image)
    assert torch.equal(first.noise, second.noise)
    achieved = first.noise.square().mean(dim=(-2, -1)).sqrt()
    assert torch.allclose(achieved, torch.full_like(achieved, 0.02), atol=2e-6, rtol=0)
    assert torch.equal(first.image, image + first.noise)

    spectrum = torch.fft.fft2(first.noise, norm="ortho")
    blocked = first.high_pass_mask == 0
    # FFT -> iFFT -> FFT round trips leave float32 round-off at frequencies
    # that were set to zero.  Bound that residue instead of requiring a
    # mathematically exact zero representation.
    blocked_residue = (spectrum * blocked).abs().amax()
    retained_scale = (spectrum * ~blocked).abs().amax().clamp_min(1.0e-12)
    assert blocked_residue <= 1.0e-6 * retained_scale


def test_public_probe_signatures_do_not_accept_condition_or_label() -> None:
    for function in (mask_low_frequency_amplitude, inject_high_frequency_noise):
        names = set(inspect.signature(function).parameters)
        assert not names.intersection(
            {"condition", "corruption", "severity", "label", "mask", "target"}
        )


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(1, 1, 8),
        torch.zeros(1, 1, 8, 8, dtype=torch.int64),
        torch.full((1, 1, 8, 8), float("nan")),
        torch.full((1, 1, 8, 8), -0.01),
        torch.full((1, 1, 8, 8), 1.01),
    ],
)
def test_deteriorations_reject_nonphysical_inputs(bad: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError)):
        mask_low_frequency_amplitude(
            bad,
            mask_ratio=0.2,
            keep_probability=0.5,
            generator=_generator(1),
        )
    with pytest.raises((TypeError, ValueError)):
        inject_high_frequency_noise(
            bad,
            target_rms=0.02,
            low_cut_ratio=0.2,
            generator=_generator(1),
        )
