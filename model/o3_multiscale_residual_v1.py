"""Source-trained, bounded local multiscale correction of frozen D0 features.

This independently written branch takes inspiration from the parallel 3x3/5x5
local paths of SCTransNet's CFN, not its complete architecture or training:
https://arxiv.org/abs/2401.15583
https://github.com/xdFai/SCTransNet/blob/main/model/SCTransNet.py

There is deliberately no ECA, cross-attention, BN, dropout, or test-time update
rule here. A zero output projection starts at numerical identity. Its gradient
can be nonzero immediately; upstream convolution gradients start at zero and
become available after the projection learns. The residual amplitude bound is
not a guarantee of target preservation, lower false alarms, or improved IoU.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class O3MultiScaleResidual(nn.Module):
    """Correct detached ``[N,16,H,W]`` features using 1,568 learned scalars.

    ``s = detach(sqrt(clamp(mean(h*h), min=1e-12)))`` is computed per image,
    not across the batch. The two local paths are concatenated, projected,
    and bounded as ``h + 0.05*s*tanh(branch(h/s))``. All convolutions have no
    bias. Float32 and float64 are supported; source features must be detached.

    A trained branch is persistent source-model state. If a later algorithm
    adapts it episodically, restore that trained snapshot, not an all-zero
    projection. No episodic reset is implemented by this class.
    """

    channels = 16
    hidden_channels = 32
    max_residual_ratio = 0.05
    rms_floor = 1e-6

    def __init__(self) -> None:
        super().__init__()
        self.expand = nn.Conv2d(16, 32, kernel_size=1, bias=False)
        self.dw3 = nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False)
        self.dw5 = nn.Conv2d(16, 16, kernel_size=5, padding=2, groups=16, bias=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.relu5 = nn.ReLU(inplace=False)
        self.project = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        nn.init.zeros_(self.project.weight)

    def _validate(self, h: Tensor) -> None:
        if not isinstance(h, Tensor):
            raise TypeError("D0 features must be a torch.Tensor")
        if h.layout != torch.strided:
            raise ValueError("D0 features must be dense strided tensors")
        if h.ndim != 4 or h.shape[1] != 16 or any(size == 0 for size in h.shape):
            raise ValueError("D0 features must have nonempty shape [N,16,H,W]")
        if h.dtype not in (torch.float32, torch.float64):
            raise TypeError("D0 features must have dtype float32 or float64")
        if h.requires_grad or h.grad_fn is not None:
            raise ValueError("D0 source features must be detached from the host graph")
        if h.device.type == "meta":
            raise ValueError("D0 features require a materialized non-meta device")
        if not bool(torch.isfinite(h).all().item()):
            raise ValueError("D0 features must be finite")
        for name, parameter in self.named_parameters():
            if parameter.device != h.device:
                raise ValueError(f"parameter and feature device mismatch: {name}")
            if parameter.dtype != h.dtype:
                raise TypeError(f"parameter and feature dtype mismatch: {name}")
            if not bool(torch.isfinite(parameter).all().item()):
                raise ValueError(f"branch parameters must be finite: {name}")

    def forward(self, h: Tensor) -> Tensor:
        self._validate(h)
        scale = h.square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12).sqrt().detach()
        if not bool(torch.isfinite(scale).all().item()):
            raise ValueError("feature RMS overflow: scale must be finite")
        expanded3, expanded5 = self.expand(h / scale).chunk(2, dim=1)
        local3 = self.relu3(self.dw3(expanded3))
        local5 = self.relu5(self.dw5(expanded5))
        correction = self.project(torch.cat((local3, local5), dim=1))
        if not bool(torch.isfinite(correction).all().item()):
            raise ValueError("branch correction must be finite")
        result = h + 0.05 * scale * torch.tanh(correction)
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError("corrected D0 features must be finite")
        return result

    def extra_repr(self) -> str:
        return "channels=16, hidden_channels=32, residual_ratio=0.05, rms_floor=1e-6"
