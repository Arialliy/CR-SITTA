"""Low-frequency amplitude masking on physical ``[0, 1]`` images.

The transform never consumes a benchmark corruption label.  Its only source
of randomness is the caller-owned :class:`torch.Generator`, which makes the
per-image Stage-C probe reproducible and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import torch
from torch import Tensor


class DeteriorationContractError(ValueError):
    """A Stage-C deterioration input violates the frozen contract."""


@dataclass(frozen=True)
class FourierMaskOutput:
    """The deteriorated image and the Hermitian-symmetric amplitude mask."""

    image: Tensor
    amplitude_mask: Tensor


def _finite_float(
    value: Real,
    *,
    name: str,
    lower: float,
    upper: float,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise DeteriorationContractError(f"{name} must be finite")
    lower_ok = converted >= lower if lower_inclusive else converted > lower
    upper_ok = converted <= upper if upper_inclusive else converted < upper
    if not lower_ok or not upper_ok:
        left = "[" if lower_inclusive else "("
        right = "]" if upper_inclusive else ")"
        raise DeteriorationContractError(
            f"{name} must lie in {left}{lower}, {upper}{right}"
        )
    return converted


def validate_physical_image(image_01: Tensor) -> None:
    """Require a finite real BCHW tensor whose intensities lie in ``[0, 1]``."""

    if not isinstance(image_01, Tensor):
        raise TypeError("image_01 must be a torch.Tensor")
    if image_01.ndim != 4 or any(int(size) <= 0 for size in image_01.shape):
        raise DeteriorationContractError(
            "image_01 must have non-empty shape [B,C,H,W]"
        )
    if not image_01.is_floating_point() or image_01.is_complex():
        raise TypeError("image_01 must be a real floating-point tensor")
    detached = image_01.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise DeteriorationContractError("image_01 contains NaN or Inf")
    if bool(((detached < 0.0) | (detached > 1.0)).any().item()):
        raise DeteriorationContractError(
            "image_01 must contain physical intensities in [0, 1]"
        )


def _random_on_generator_device(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    destination: Tensor,
) -> Tensor:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator")
    generator_device = torch.device(generator.device)
    random = torch.rand(
        shape,
        generator=generator,
        device=generator_device,
        dtype=destination.dtype,
    )
    return random.to(device=destination.device)


def _conjugate_partner_indices(size: int, center: int, device: torch.device) -> Tensor:
    return torch.tensor(
        [(2 * center - index) % size for index in range(size)],
        device=device,
        dtype=torch.long,
    )


def identity_deterioration(image_01: Tensor) -> Tensor:
    """Return an exact, independently stored clean branch."""

    validate_physical_image(image_01)
    return image_01.clone()


def mask_low_frequency_amplitude(
    image_01: Tensor,
    *,
    mask_ratio: float,
    keep_probability: float,
    generator: torch.Generator,
) -> FourierMaskOutput:
    """Randomly suppress central Fourier amplitudes without changing geometry.

    The random mask is made Hermitian symmetric before application.  This
    preserves the conjugate symmetry required for a real inverse FFT while
    leaving the phase of every retained Fourier coefficient unchanged.
    """

    validate_physical_image(image_01)
    checked_ratio = _finite_float(
        mask_ratio,
        name="mask_ratio",
        lower=0.0,
        upper=0.5,
        lower_inclusive=False,
        upper_inclusive=False,
    )
    checked_keep = _finite_float(
        keep_probability,
        name="keep_probability",
        lower=0.0,
        upper=1.0,
        lower_inclusive=False,
        upper_inclusive=True,
    )

    spectrum = torch.fft.fftshift(
        torch.fft.fft2(image_01, dim=(-2, -1), norm="ortho"),
        dim=(-2, -1),
    )
    height, width = (int(value) for value in image_01.shape[-2:])
    half_h = max(1, int(round(height * checked_ratio / 2.0)))
    half_w = max(1, int(round(width * checked_ratio / 2.0)))
    center_y, center_x = height // 2, width // 2
    top, bottom = max(0, center_y - half_h), min(height, center_y + half_h)
    left, right = max(0, center_x - half_w), min(width, center_x + half_w)

    mask = torch.ones_like(image_01)
    random_keep = _random_on_generator_device(
        (*image_01.shape[:-2], bottom - top, right - left),
        generator=generator,
        destination=image_01,
    )
    mask[..., top:bottom, left:right] = (
        random_keep < checked_keep
    ).to(dtype=image_01.dtype)

    # A real spatial image requires M(k) == M(-k).  Logical AND is
    # deliberately conservative: a sampled dropout removes the whole pair.
    partner_y = _conjugate_partner_indices(height, center_y, image_01.device)
    partner_x = _conjugate_partner_indices(width, center_x, image_01.device)
    mirrored = mask.index_select(-2, partner_y).index_select(-1, partner_x)
    mask = torch.minimum(mask, mirrored)

    deteriorated_spectrum = spectrum * mask
    spatial_complex = torch.fft.ifft2(
        torch.fft.ifftshift(deteriorated_spectrum, dim=(-2, -1)),
        dim=(-2, -1),
        norm="ortho",
    )
    # The symmetric mask should make the imaginary residue numerical only.
    real_scale = spatial_complex.real.detach().abs().amax().clamp_min(1.0)
    imaginary_residue = spatial_complex.imag.detach().abs().amax()
    tolerance = 64.0 * torch.finfo(image_01.dtype).eps * real_scale
    if bool((imaginary_residue > tolerance).item()):
        raise RuntimeError("low-frequency mask broke Fourier conjugate symmetry")

    deteriorated = spatial_complex.real.clamp(0.0, 1.0)
    if not bool(torch.isfinite(deteriorated.detach()).all().item()):
        raise RuntimeError("low-frequency deterioration produced NaN or Inf")
    return FourierMaskOutput(
        image=deteriorated,
        amplitude_mask=mask.detach(),
    )


__all__ = [
    "DeteriorationContractError",
    "FourierMaskOutput",
    "identity_deterioration",
    "mask_low_frequency_amplitude",
    "validate_physical_image",
]
