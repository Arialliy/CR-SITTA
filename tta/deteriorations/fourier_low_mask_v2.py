"""Versioned, label-free LF repair with paired conjugate-orbit randomness.

The historical LF implementation and package exports are intentionally untouched.
All controls draw the same CPU float64 ``[B,C,all_orbits]`` field, including DC.
Protection, channel sharing, and attenuation are applied *after* that draw.  Thus
restarting a caller-owned generator at the same seed produces paired L1--L4
controls, irrespective of which control is evaluated first.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from typing import Any

import torch
from torch import Tensor


OPERATOR_ID = "lf_pair_shared_dc_soft_v2"
RANDOM_FIELD_POLICY = "cpu_float64_B_C_full_sorted_conjugate_orbits_including_dc_v1"
TENSOR_HASH_SCHEMA = "lf_v2_dtype_shape_contiguous_bytes_v1"


@dataclass(frozen=True)
class LFConfig:
    mask_ratio: float = 0.20
    pair_keep_probability: float = 0.50
    attenuation: float = 0.25

    def __post_init__(self) -> None:
        for name in ("mask_ratio", "pair_keep_probability", "attenuation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, float(value))
        if not 0 < self.mask_ratio < 0.5:
            raise ValueError("mask_ratio must be in (0, 0.5)")
        if not 0 < self.pair_keep_probability <= 1:
            raise ValueError("pair_keep_probability must be in (0, 1]")
        if not 0 <= self.attenuation <= 1:
            raise ValueError("attenuation must be in [0, 1]")


@dataclass(frozen=True)
class LFOutput:
    image: Tensor
    preclip: Tensor
    gain: Tensor  # [B,1,H,W] when shared, otherwise [B,C,H,W].
    support: Tensor  # CPU [H,W]; eligible gain support, excluding protected DC.
    diagnostics: dict[str, Any]


def _tensor_sha256(value: Tensor) -> str:
    """A new, explicit hash schema; not interchangeable with old v7 hashes."""
    value = value.detach().cpu().contiguous()
    descriptor = json.dumps(
        {"schema": TENSOR_HASH_SCHEMA, "dtype": str(value.dtype), "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\n" + value.numpy().tobytes()).hexdigest()


def _full_conjugate_geometry(height: int, width: int, ratio: float):
    """Return the old rectangle's closure and sorted orbits, *including* DC."""
    cy, cx = height // 2, width // 2
    iy = (2 * cy - torch.arange(height)) % height
    ix = (2 * cx - torch.arange(width)) % width
    ids = torch.arange(height * width).reshape(height, width)
    partners = ids.index_select(0, iy).index_select(1, ix)
    half_h = max(1, round(height * ratio / 2))
    half_w = max(1, round(width * ratio / 2))
    rect = torch.zeros(height, width, dtype=torch.bool)
    rect[max(0, cy - half_h):min(height, cy + half_h),
         max(0, cx - half_w):min(width, cx + half_w)] = True
    support = rect | rect.index_select(0, iy).index_select(1, ix)
    canonical = torch.minimum(ids, partners)
    representatives, inverse = torch.unique(
        canonical[support], sorted=True, return_inverse=True
    )
    return support, representatives, inverse


def _validate_input(image: Tensor, cfg: LFConfig, generator: torch.Generator) -> None:
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor")
    if not isinstance(cfg, LFConfig):
        raise TypeError("cfg must be an LFConfig")
    if image.ndim != 4 or any(n <= 0 for n in image.shape):
        raise ValueError("expected nonempty BCHW")
    if image.layout != torch.strided or image.dtype not in (torch.float32, torch.float64):
        raise TypeError("use dense float32/float64 physical inputs")
    if min(image.shape[-2:]) < 2:
        raise ValueError("spatial dimensions must be at least 2")
    x = image.detach()
    if not bool(torch.isfinite(x).all()) or bool(((x < 0) | (x > 1)).any()):
        raise ValueError("physical input must be finite and inside [0,1]")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise TypeError("a caller-owned CPU generator is required")


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def lf_mask_diagnostic(
    image: Tensor, cfg: LFConfig, *, generator: torch.Generator,
    protect_dc: bool, shared_channels: bool,
) -> LFOutput:
    """Apply paired L1--L4 controls without accepting labels, paths, or IDs.

    L1: alpha=1, no protection or sharing; L2: alpha=1, DC protected;
    L3: alpha=1, protected and shared; L4a/b: same, alpha=0.25/0.50.
    Identity controls still consume the complete registered random field, but
    bypass FFT reconstruction to return the exact input values.
    """
    _validate_input(image, cfg, generator)
    if type(protect_dc) is not bool or type(shared_channels) is not bool:
        raise TypeError("protect_dc and shared_channels must be bool")
    batch, channels, height, width = image.shape
    cy, cx = height // 2, width // 2
    full_support, representatives, inverse = _full_conjugate_geometry(
        height, width, cfg.mask_ratio
    )
    # Deliberately unchanged by protection/sharing/alpha, including identity.
    random_field = torch.rand(
        batch, channels, representatives.numel(), generator=generator,
        dtype=torch.float64, device="cpu",
    )
    full_keep = random_field < cfg.pair_keep_probability
    selected_keep = full_keep[:, :1] if shared_channels else full_keep
    gain_cpu = torch.ones(batch, selected_keep.shape[1], height, width, dtype=image.dtype)
    gain_cpu[..., full_support] = (
        1 - cfg.attenuation * (~selected_keep[..., inverse]).to(image.dtype)
    )
    support = full_support.clone()
    active_orbits = torch.ones(representatives.numel(), dtype=torch.bool)
    dc_orbit = representatives == cy * width + cx
    if int(dc_orbit.sum()) != 1:
        raise RuntimeError("expected exactly one DC orbit")
    if protect_dc:
        gain_cpu[..., cy, cx] = 1
        support[cy, cx] = False
        active_orbits[dc_orbit] = False
    gain = gain_cpu.to(image.device)
    spectrum = torch.fft.fftshift(torch.fft.fft2(image, norm="ortho"), dim=(-2, -1))
    identity_control = cfg.attenuation == 0 or cfg.pair_keep_probability == 1
    if identity_control:
        preclip = image.clone()
        imaginary_max = 0.0
    else:
        spatial = torch.fft.ifft2(
            torch.fft.ifftshift(spectrum * gain, dim=(-2, -1)), norm="ortho"
        )
        imaginary_max = float(spatial.imag.detach().abs().max())
        tolerance = 64 * torch.finfo(image.dtype).eps * max(
            1.0, float(spatial.real.detach().abs().max())
        )
        if not math.isfinite(imaginary_max) or imaginary_max > tolerance:
            raise RuntimeError("non-numerical imaginary residue")
        preclip = spatial.real
    # No hard-branch clamp: attenuate/mix first, clamp exactly once.
    output = preclip.clamp(0, 1)
    if not bool(torch.isfinite(preclip.detach()).all()) or not bool(torch.isfinite(output.detach()).all()):
        raise RuntimeError("non-finite LF output")

    with torch.no_grad():
        x = image.detach()
        before, after = preclip.detach(), output.detach()
        clean_spectrum = spectrum.detach()
        before_spectrum = torch.fft.fftshift(torch.fft.fft2(before, norm="ortho"), dim=(-2, -1))
        after_spectrum = torch.fft.fftshift(torch.fft.fft2(after, norm="ortho"), dim=(-2, -1))
        energies = [value.abs().square() for value in (clean_spectrum, before_spectrum, after_spectrum)]
        support_device = support.to(image.device)
        totals = [float(value.sum()) for value in energies]
        support_totals = [float(value[..., support_device].sum()) for value in energies]
        selected_dc = selected_keep[..., dc_orbit].squeeze(-1).expand(batch, channels)
        stats: dict[str, Any] = {
            "operator_id": OPERATOR_ID if protect_dc and shared_channels else "lf_pair_diagnostic_v2",
            "random_field_policy": RANDOM_FIELD_POLICY,
            "tensor_hash_schema": TENSOR_HASH_SCHEMA,
            "mask_ratio": cfg.mask_ratio,
            "pair_keep_probability": cfg.pair_keep_probability,
            "attenuation": cfg.attenuation,
            "protect_dc": protect_dc,
            "shared_channels": shared_channels,
            "identity_control": identity_control,
            "random_field_shape": list(random_field.shape),
            "random_field_sha256": _tensor_sha256(random_field),
            "random_field_values": random_field.tolist(),
            "orbit_representatives_sha256": _tensor_sha256(representatives),
            "orbit_representatives": representatives.tolist(),
            "all_orbit_count": int(representatives.numel()),
            "active_orbit_count": int(active_orbits.sum()),
            "orbit_count": int(active_orbits.sum()),
            "sampled_all_pair_keep_fraction": float(full_keep.double().mean()),
            "sampled_pair_keep_fraction": float(selected_keep[..., active_orbits].double().mean()),
            "sampled_dc_keep_per_channel": selected_dc.tolist(),
            "gain_shape": list(gain_cpu.shape),
            "gain_mean": float(gain_cpu.double().mean()),
            "gain_min": float(gain_cpu.min()),
            "gain_support_mean": float(gain_cpu[..., support].double().mean()),
            "gain_support_min": float(gain_cpu[..., support].min()),
            "gain_sha256": _tensor_sha256(gain_cpu),
            "support_frequency_bins": int(support.sum()),
            "support_sha256": _tensor_sha256(support),
            "full_support_frequency_bins": int(full_support.sum()),
            "full_support_sha256": _tensor_sha256(full_support),
            "dc_gain_per_channel": gain_cpu[..., cy, cx].expand(batch, channels).tolist(),
            "preclip_rms": float((before - x).square().mean().sqrt()),
            "postclip_rms": float((after - x).square().mean().sqrt()),
            "actual_clipped_fraction": float(((before < 0) | (before > 1)).double().mean()),
            "postclip_endpoint_fraction": float(((after <= 0) | (after >= 1)).double().mean()),
            "preclip_mean_drift_max": float((before.mean((-2, -1)) - x.mean((-2, -1))).abs().max()),
            "postclip_mean_drift_max": float((after.mean((-2, -1)) - x.mean((-2, -1))).abs().max()),
            "imaginary_max": imaginary_max,
            "clean_total_energy": totals[0],
            "preclip_total_energy": totals[1],
            "postclip_total_energy": totals[2],
            "preclip_total_energy_retention": _ratio(totals[1], totals[0]),
            "postclip_total_energy_retention": _ratio(totals[2], totals[0]),
            "clean_support_energy": support_totals[0],
            "preclip_support_energy": support_totals[1],
            "postclip_support_energy": support_totals[2],
            "preclip_support_energy_retention": _ratio(support_totals[1], support_totals[0]),
            "postclip_support_energy_retention": _ratio(support_totals[2], support_totals[0]),
        }
    return LFOutput(output, preclip, gain.detach(), support, stats)


def lf_soft_mask(image: Tensor, cfg: LFConfig, *, generator: torch.Generator) -> LFOutput:
    """Production LF-v2: pair-sampled, DC-protected, physically channel-shared."""
    return lf_mask_diagnostic(image, cfg, generator=generator, protect_dc=True, shared_channels=True)


__all__ = ["LFConfig", "LFOutput", "lf_soft_mask", "lf_mask_diagnostic"]
