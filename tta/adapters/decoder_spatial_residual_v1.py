"""Bounded, identity-initialized local correction for frozen D0 features.

This module does not classify degradations, denoise an image explicitly, or
provide a performance guarantee.  Its 144 trainable scalars make a spatially
varying correction from the local content of each feature channel.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _positive_finite(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


class DecoderSpatialResidual(nn.Module):
    """Apply ``x + ratio * s * tanh(depthwise_conv(x / s, kernel))``.

    ``s`` is the detached per-image feature RMS with a positive floor.  There
    is one zero-initialized 3-by-3 kernel per channel, with no bias, RNG, BN,
    persistent episode state, or trainable source features.  Zero kernels
    preserve numerical identity while remaining on the first-step gradient
    path.  As usual for floating-point addition, signed-zero bit patterns
    are not an identity guarantee.

    The real-arithmetic residual bound is ``ratio * s`` at every location.
    When auditing ``output - x``, allow rounding error from the final add.
    """

    def __init__(
        self,
        channels: int = 16,
        max_residual_ratio: float = 0.05,
        rms_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int):
            raise TypeError("channels must be the integer 16")
        if channels != 16:
            raise ValueError("DecoderSpatialResidual v1 fixes channels to 16")
        self.channels = channels
        self.max_residual_ratio = _positive_finite(
            max_residual_ratio, "max_residual_ratio"
        )
        self.rms_floor = _positive_finite(rms_floor, "rms_floor")
        if not 0.0 < self.rms_floor * self.rms_floor < float("inf"):
            raise ValueError("rms_floor squared must be finite and positive")
        self.kernel = nn.Parameter(torch.zeros(channels, 1, 3, 3))

    def _validate(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            raise TypeError("D0 features must be a torch.Tensor")
        if x.layout != torch.strided:
            raise ValueError("D0 features must be a dense strided tensor")
        if x.ndim != 4 or x.shape[1] != self.channels or any(n == 0 for n in x.shape):
            raise ValueError("D0 features must have nonempty shape [B, 16, H, W]")
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("D0 features must have dtype float32 or float64")
        if x.requires_grad or x.grad_fn is not None:
            raise ValueError("D0 source features must be detached")
        if tuple(self.kernel.shape) != (16, 1, 3, 3):
            raise ValueError("kernel must have shape [16, 1, 3, 3]")
        if self.kernel.layout != torch.strided:
            raise ValueError("kernel must be a dense strided tensor")
        if x.device != self.kernel.device:
            raise ValueError("D0 features and kernel device mismatch")
        if x.dtype != self.kernel.dtype:
            raise TypeError("D0 features and kernel dtype mismatch")
        if x.device.type == "meta":
            raise ValueError("D0 features require a materialized non-meta device")
        if not torch.isfinite(x).all() or not torch.isfinite(self.kernel).all():
            raise ValueError("D0 features and kernel must be finite")

    def forward(self, x: Tensor) -> Tensor:
        self._validate(x)
        floor_squared = x.new_tensor(self.rms_floor * self.rms_floor)
        if not torch.isfinite(floor_squared) or floor_squared <= 0:
            raise ValueError("rms_floor squared is not representable in feature dtype")
        scale = x.square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(
            floor_squared
        ).sqrt().detach()
        if not torch.isfinite(scale).all():
            raise ValueError("feature RMS overflow: intermediate scale must be finite")
        response = F.conv2d(x / scale, self.kernel, padding=1, groups=self.channels)
        if not torch.isfinite(response).all():
            raise ValueError("depthwise convolution response must be finite")
        output = x + self.max_residual_ratio * scale * torch.tanh(response)
        if not torch.isfinite(output).all():
            raise ValueError("corrected D0 features must be finite")
        return output

    @torch.no_grad()
    def reset_identity_(self) -> DecoderSpatialResidual:
        """Restore the episode's identity kernel and clear its gradient field."""
        self.kernel.zero_()
        self.kernel.grad = None
        return self

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, max_residual_ratio={self.max_residual_ratio}, "
            f"rms_floor={self.rms_floor}"
        )
