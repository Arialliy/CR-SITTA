"""Identity-initialized spatial low-rank FiLM modulation for Stage-C."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _positive_finite_float(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return converted


def _grid_size(value: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("grid_size must be a (height, width) tuple")
    return (
        _positive_integer(value[0], name="grid_size[0]"),
        _positive_integer(value[1], name="grid_size[1]"),
    )


class SpatialLowRankFiLM(nn.Module):
    """Spatially varying channel modulation with a fixed orthogonal basis.

    The channel basis is a non-trainable ``[rank, channels]`` buffer.  Only
    the two low-resolution coefficient maps are trainable.  Both maps start
    at exact zero, making the initial transformation ``feature * 1 + 0``
    while preserving a live first-step gradient through the fixed basis.
    """

    def __init__(
        self,
        channels: int,
        *,
        rank: int,
        grid_size: tuple[int, int] = (8, 8),
        max_scale_delta: float = 0.05,
        max_bias: float = 0.05,
        seed: int = 3407,
    ) -> None:
        super().__init__()
        checked_channels = _positive_integer(channels, name="channels")
        checked_rank = _positive_integer(rank, name="rank")
        if checked_rank > checked_channels:
            raise ValueError("rank must satisfy 1 <= rank <= channels")
        checked_grid = _grid_size(grid_size)
        checked_scale = _positive_finite_float(
            max_scale_delta, name="max_scale_delta"
        )
        checked_bias = _positive_finite_float(max_bias, name="max_bias")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        # A private CPU generator makes basis construction reproducible without
        # consuming or changing the experiment's process-wide RNG stream.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        random_basis = torch.randn(
            checked_channels, checked_rank, generator=generator
        )
        orthonormal_columns, _ = torch.linalg.qr(
            random_basis, mode="reduced"
        )
        basis = orthonormal_columns.T.contiguous()
        self.register_buffer("channel_basis", basis)

        grid_height, grid_width = checked_grid
        self.scale_coeff = nn.Parameter(
            torch.zeros(1, checked_rank, grid_height, grid_width)
        )
        self.bias_coeff = nn.Parameter(
            torch.zeros(1, checked_rank, grid_height, grid_width)
        )

        self.channels = checked_channels
        self.rank = checked_rank
        self.grid_size = checked_grid
        self.max_scale_delta = checked_scale
        self.max_bias = checked_bias
        self.seed = seed

    def reset_identity_(self, *, clear_gradients: bool = True) -> None:
        """Restore the exact zero-coefficient identity state in place."""

        if type(clear_gradients) is not bool:
            raise TypeError("clear_gradients must be bool")
        with torch.no_grad():
            self.scale_coeff.zero_()
            self.bias_coeff.zero_()
        if clear_gradients:
            self.scale_coeff.grad = None
            self.bias_coeff.grad = None

    def is_identity(self) -> bool:
        """Return whether both trainable coefficient maps are exactly zero."""

        return bool(
            torch.count_nonzero(self.scale_coeff.detach()).item() == 0
            and torch.count_nonzero(self.bias_coeff.detach()).item() == 0
        )

    def forward(self, feature: Tensor) -> Tensor:
        if not isinstance(feature, Tensor):
            raise TypeError("feature must be a torch.Tensor")
        if feature.ndim != 4:
            raise ValueError("feature must have shape [B, C, H, W]")
        if int(feature.shape[1]) != self.channels:
            raise ValueError(
                "unexpected feature channel count: "
                f"expected {self.channels}, got {int(feature.shape[1])}"
            )
        if not feature.is_floating_point() or feature.is_complex():
            raise TypeError("feature must have a real floating-point dtype")
        if feature.device != self.channel_basis.device:
            raise ValueError("feature and router must be on the same device")
        if feature.dtype != self.channel_basis.dtype:
            raise ValueError("feature and router must have the same dtype")

        spatial_scale = F.interpolate(
            self.scale_coeff,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        spatial_bias = F.interpolate(
            self.bias_coeff,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        scale_map = torch.einsum(
            "brhw,rc->bchw", spatial_scale, self.channel_basis
        )
        bias_map = torch.einsum(
            "brhw,rc->bchw", spatial_bias, self.channel_basis
        )
        scale = 1.0 + self.max_scale_delta * torch.tanh(scale_map)
        bias = self.max_bias * torch.tanh(bias_map)
        return feature * scale + bias

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, rank={self.rank}, "
            f"grid_size={self.grid_size}, "
            f"max_scale_delta={self.max_scale_delta}, "
            f"max_bias={self.max_bias}, seed={self.seed}"
        )


__all__ = ["SpatialLowRankFiLM"]
