"""A frozen trained-v1 anchor with an independently learnable logit-free increment.

The live network starts as an exact copy of the anchor. Its *pre-tanh* output
difference supplies the increment, so the original v1 tanh amplitude limit no
longer restricts subsequent corrections. This is neither a scalar gate over
the old residual nor a guarantee of improved detection or target preservation.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
import copy

import torch
from torch import Tensor, nn

from model.o3_multiscale_residual_v1 import O3MultiScaleResidual


class O3AnchoredResidualV4(nn.Module):
    """Return ``anchor(h) + 0.05*s*(u_live(h/s) - u_anchor(h/s))``.

    ``s`` is the detached, per-image RMS with the same floor as v1. ``u`` is
    the existing expand / parallel depthwise+ReLU / project computation before
    tanh. Only the 1,568 live parameters are trainable; the complete state dict
    contains both the frozen anchor and the live network. Inputs are detached
    float32/float64 ``[N,16,H,W]`` features, never labels or dataset metadata.

    The supplied state is copied, preserving its dtype/device. Construction
    does not advance the CPU RNG or request CUDA RNG state. Calling train()
    changes the live network's mode but always keeps the anchor in eval mode.
    """

    channels = 16
    residual_scale = 0.05
    rms_floor = 1e-6

    def __init__(self, anchor_state_dict: Mapping[str, Tensor]) -> None:
        super().__init__()
        if not isinstance(anchor_state_dict, Mapping) or not anchor_state_dict:
            raise TypeError("anchor_state_dict must be a nonempty tensor mapping")
        state = dict(anchor_state_dict)
        first = next(iter(state.values()))
        if not isinstance(first, Tensor):
            raise TypeError("anchor state values must be tensors")
        for name, value in state.items():
            if not isinstance(name, str) or not isinstance(value, Tensor):
                raise TypeError("anchor state must map string names to tensors")
            if value.layout != torch.strided or value.device.type == "meta":
                raise ValueError("anchor state must contain materialized dense tensors")
            if value.dtype not in (torch.float32, torch.float64):
                raise TypeError("anchor state must use float32 or float64")
            if value.dtype != first.dtype or value.device != first.device:
                raise ValueError("anchor state must have one common dtype and device")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("anchor state must be finite")
        with torch.random.fork_rng(devices=[]):
            self.anchor = O3MultiScaleResidual()
        self.anchor.to(device=first.device, dtype=first.dtype)
        self.anchor.load_state_dict(state, strict=True)
        self.anchor.requires_grad_(False)
        self.anchor.eval()
        self.live = copy.deepcopy(self.anchor)
        self.live.requires_grad_(True)
        self.train(True)

    def train(self, mode: bool = True) -> O3AnchoredResidualV4:
        super().train(mode)
        self.anchor.eval()
        return self

    def learnable_parameters(self) -> Iterator[nn.Parameter]:
        return self.live.parameters()

    @staticmethod
    def _preactivation(network: O3MultiScaleResidual, h: Tensor, scale: Tensor) -> Tensor:
        expanded3, expanded5 = network.expand(h / scale).chunk(2, dim=1)
        local3 = network.relu3(network.dw3(expanded3))
        local5 = network.relu5(network.dw5(expanded5))
        result = network.project(torch.cat((local3, local5), dim=1))
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError("pre-tanh branch output must be finite")
        return result

    def forward(self, h: Tensor) -> Tensor:
        self.anchor._validate(h)
        self.live._validate(h)
        if any(module.training for module in self.anchor.modules()):
            raise RuntimeError("the frozen anchor must remain in eval mode")
        if any(parameter.requires_grad for parameter in self.anchor.parameters()):
            raise RuntimeError("the anchor parameters must remain frozen")
        scale = h.square().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12).sqrt().detach()
        if not bool(torch.isfinite(scale).all().item()):
            raise ValueError("feature RMS overflow: scale must be finite")
        with torch.no_grad():
            anchored = self.anchor(h)
            anchor_output = self._preactivation(self.anchor, h, scale)
        live_output = self._preactivation(self.live, h, scale)
        result = anchored + self.residual_scale * scale * (live_output - anchor_output)
        if not bool(torch.isfinite(result).all().item()):
            raise ValueError("corrected D0 features must be finite")
        return result

    def extra_repr(self) -> str:
        return "channels=16, trainable=1568, frozen=1568, residual_scale=0.05, hard_bound=False"


__all__ = ["O3AnchoredResidualV4"]
