"""Identity-initialized low-rank residual mixing for decoder features."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LowRankResidualMixer(nn.Module):
    """Mix channels through a fixed down basis and trainable up projection.

    The rows of ``down_basis`` are orthonormal and stored as a buffer.  Only
    ``up.weight`` is trainable.  Its zero initialization makes the complete
    residual branch exactly zero without blocking its first-step gradient.
    """

    def __init__(
        self,
        channels: int = 16,
        rank: int = 4,
        residual_scale: float = 0.1,
        seed: int = 3407,
    ) -> None:
        super().__init__()
        if isinstance(channels, bool) or not isinstance(channels, int):
            raise TypeError("channels must be an integer")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be an integer")
        if channels <= 1:
            raise ValueError("channels must be greater than one")
        if not 1 <= rank < channels:
            raise ValueError("rank must satisfy 1 <= rank < channels")
        if isinstance(residual_scale, bool) or not isinstance(residual_scale, Real):
            raise TypeError("residual_scale must be a real number")
        checked_scale = float(residual_scale)
        if not math.isfinite(checked_scale) or checked_scale <= 0.0:
            raise ValueError("residual_scale must be finite and strictly positive")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        random_basis = torch.randn(rank, channels, generator=generator)
        orthonormal_columns, _ = torch.linalg.qr(
            random_basis.T, mode="reduced"
        )
        basis = orthonormal_columns.T.contiguous()
        self.register_buffer("down_basis", basis[:, :, None, None])

        # Conv2d initialization consumes the process RNG even though the
        # weights are immediately zeroed.  fork_rng keeps construction local
        # and prevents adapter creation from perturbing experiment randomness.
        with torch.random.fork_rng(devices=[]):
            self.up = nn.Conv2d(rank, channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.up.weight)

        self.channels = channels
        self.rank = rank
        self.residual_scale = checked_scale
        self.seed = seed

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

        low_rank = F.conv2d(x, self.down_basis)
        return x + self.residual_scale * self.up(low_rank)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, rank={self.rank}, "
            f"residual_scale={self.residual_scale}, seed={self.seed}"
        )
