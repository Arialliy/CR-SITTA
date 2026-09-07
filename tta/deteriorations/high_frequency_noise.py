"""Deterministic high-frequency-noise probes for Stage-C ASB-SFR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .fourier_low_mask import (
    DeteriorationContractError,
    _finite_float,
    _random_on_generator_device,
    validate_physical_image,
)


@dataclass(frozen=True)
class HighFrequencyNoiseOutput:
    """Auditable pre-clipping high-pass noise and deteriorated image."""

    image: Tensor
    noise: Tensor
    high_pass_mask: Tensor
    target_rms: float


def inject_high_frequency_noise_components(
    image_01: Tensor,
    *,
    target_rms: float,
    low_cut_ratio: float,
    generator: torch.Generator,
    eps: float = 1.0e-8,
) -> HighFrequencyNoiseOutput:
    """Add RMS-normalized high-pass Gaussian noise in physical image space."""

    validate_physical_image(image_01)
    checked_rms = _finite_float(
        target_rms,
        name="target_rms",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
        upper_inclusive=True,
    )
    checked_cut = _finite_float(
        low_cut_ratio,
        name="low_cut_ratio",
        lower=0.0,
        upper=1.0,
        lower_inclusive=True,
        upper_inclusive=False,
    )
    checked_eps = _finite_float(
        eps,
        name="eps",
        lower=0.0,
        upper=0.5,
        lower_inclusive=False,
        upper_inclusive=False,
    )
    random = _random_on_generator_device(
        tuple(int(value) for value in image_01.shape),
        generator=generator,
        destination=image_01,
    )
    # Inverse-normal-CDF is unnecessary: Box-Muller gives deterministic
    # Gaussian samples using only the caller-owned generator stream.
    second = _random_on_generator_device(
        tuple(int(value) for value in image_01.shape),
        generator=generator,
        destination=image_01,
    )
    tiny = torch.finfo(image_01.dtype).tiny
    gaussian = torch.sqrt(-2.0 * torch.log(random.clamp_min(tiny))) * torch.cos(
        2.0 * torch.pi * second
    )

    spectrum = torch.fft.fft2(gaussian, dim=(-2, -1), norm="ortho")
    height, width = (int(value) for value in image_01.shape[-2:])
    fy = torch.fft.fftfreq(height, device=image_01.device, dtype=image_01.dtype) * 2.0
    fx = torch.fft.fftfreq(width, device=image_01.device, dtype=image_01.dtype) * 2.0
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    high_pass_2d = (radius >= checked_cut).to(dtype=image_01.dtype)
    if not bool((high_pass_2d > 0).any().item()):
        raise DeteriorationContractError("low_cut_ratio removed every frequency")
    high_pass = high_pass_2d.reshape(1, 1, height, width)
    high_noise = torch.fft.ifft2(
        spectrum * high_pass,
        dim=(-2, -1),
        norm="ortho",
    ).real
    rms = high_noise.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    if bool((~torch.isfinite(rms) | (rms <= checked_eps)).any().item()):
        raise RuntimeError("high-frequency noise has zero or non-finite RMS")
    scaled_noise = high_noise * (checked_rms / rms)
    deteriorated = (image_01 + scaled_noise).clamp(0.0, 1.0)
    if not bool(torch.isfinite(deteriorated.detach()).all().item()):
        raise RuntimeError("high-frequency deterioration produced NaN or Inf")
    return HighFrequencyNoiseOutput(
        image=deteriorated,
        noise=scaled_noise.detach(),
        high_pass_mask=high_pass.detach(),
        target_rms=checked_rms,
    )


def inject_high_frequency_noise(
    image_01: Tensor,
    *,
    target_rms: float,
    low_cut_ratio: float,
    generator: torch.Generator,
    eps: float = 1.0e-8,
) -> Tensor:
    """Return only the deteriorated image for model-facing code."""

    return inject_high_frequency_noise_components(
        image_01,
        target_rms=target_rms,
        low_cut_ratio=low_cut_ratio,
        generator=generator,
        eps=eps,
    ).image


__all__ = [
    "HighFrequencyNoiseOutput",
    "inject_high_frequency_noise",
    "inject_high_frequency_noise_components",
]
