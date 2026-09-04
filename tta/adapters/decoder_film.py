"""Identity-initialized channel-wise modulation for decoder features."""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn


def _positive_finite_float(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return converted


class DecoderFiLM(nn.Module):
    """Apply bounded per-channel scale and bias to ``decoder_0`` features.

    Both raw parameter vectors are zero initialized.  Consequently the
    initial transform is exactly ``x * 1 + 0`` while both vectors remain on a
    live first-step gradient path.
    """

    def __init__(
        self,
        channels: int = 16,
        max_scale_delta: float = 0.05,
        max_bias: float = 0.05,
    ) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int):
            raise TypeError("channels must be an integer")
        if channels <= 0:
            raise ValueError("channels must be strictly positive")

        self.channels = channels
        self.max_scale_delta = _positive_finite_float(
            max_scale_delta, name="max_scale_delta"
        )
        self.max_bias = _positive_finite_float(max_bias, name="max_bias")
        self.raw_scale = nn.Parameter(torch.zeros(channels))
        self.raw_bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("decoder feature must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError("decoder feature must have shape [B, C, H, W]")
        if int(x.shape[1]) != self.channels:
            raise ValueError(
                "unexpected decoder channel count: "
                f"expected {self.channels}, got {int(x.shape[1])}"
            )
        if not x.is_floating_point():
            raise TypeError("decoder feature must have a floating-point dtype")

        scale = 1.0 + self.max_scale_delta * torch.tanh(self.raw_scale)
        bias = self.max_bias * torch.tanh(self.raw_bias)
        return x * scale[None, :, None, None] + bias[None, :, None, None]

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, "
            f"max_scale_delta={self.max_scale_delta}, max_bias={self.max_bias}"
        )
