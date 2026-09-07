"""Identity-preserving, five-basis IPMA on detached NS-FPN D0 features.

Only the down/up weights are source-side learnable state.  Episode-local delta
is an explicit input: it is never a parameter, buffer, or cached attribute.
The frozen backbone and SFS are intentionally absent from this module.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class IdentityMetaAdapter(nn.Module):
    """The preregistered D0 adapter: C=16, rank=4, five spatial bases."""

    num_bases = 5
    rms_squared_floor = 1.0e-12

    def __init__(self, channels: int = 16, rank: int = 4):
        super().__init__()
        if type(channels) is not int or channels != 16 or type(rank) is not int or rank != 4:
            raise ValueError("IPMA v1 fixes D0 channels=16 and rank=4")
        self.channels = channels
        self.rank = rank
        self.down = nn.Conv2d(channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank * self.num_bases, channels, 1, bias=False)
        # PyTorch's default nonzero initialization is retained.  The caller
        # freezes the seed/initial weights; only episode-local delta is zero.
        kernels = torch.tensor([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        ], dtype=torch.float32)
        kernels /= torch.tensor([1, 16, 4, 8, 8], dtype=torch.float32)[:, None, None]
        self.register_buffer("kernels", kernels[:, None])

    def _validate_features(self, h: Tensor) -> None:
        if not isinstance(h, Tensor) or h.ndim != 4 or h.shape[:2] != (1, self.channels):
            raise ValueError("one detached D0 episode [1,16,H,W] is required")
        if min(h.shape[-2:]) < 1 or h.layout != torch.strided:
            raise ValueError("D0 spatial grid must be nonempty and dense")
        if h.dtype not in (torch.float32, torch.float64):
            raise TypeError("IPMA requires float32/float64; no half-precision RMS floor")
        if h.requires_grad or h.grad_fn is not None:
            raise ValueError("D0 features must be detached from the frozen host")
        for name, tensor in self.state_dict(keep_vars=True).items():
            if tensor.device != h.device or tensor.dtype != h.dtype:
                raise ValueError(f"adapter/features device or dtype mismatch: {name}")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"non-finite adapter state: {name}")
        if not bool(torch.isfinite(h).all()):
            raise ValueError("non-finite D0 features")

    def bases(self, h: Tensor) -> Tensor:
        """Return [1,5,16,H,W], with differentiable per-basis RMS scaling.

        The RMS bound is global, not a per-pixel, logit, or target-preservation
        guarantee.  Its denominator is max(RMS(h), 1e-6).
        """
        self._validate_features(h)
        u = self.down(h)
        scale = h.detach().square().mean((1, 2, 3), keepdim=True).clamp_min(self.rms_squared_floor).sqrt()
        outputs = []
        for k in range(self.num_bases):
            kernel = self.kernels[k:k+1].repeat(self.rank, 1, 1, 1)
            filtered = F.conv2d(u, kernel, padding=1, groups=self.rank)
            weight = self.up.weight[:, k*self.rank:(k+1)*self.rank]
            basis = F.conv2d(filtered, weight)
            rms = basis.square().mean((1, 2, 3), keepdim=True).clamp_min(self.rms_squared_floor).sqrt()
            outputs.append(basis / rms * scale)
        result = torch.stack(outputs, dim=1)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("non-finite normalized IPMA basis")
        return result

    def forward(self, h: Tensor, delta: Tensor) -> Tensor:
        if not isinstance(h, Tensor) or not isinstance(delta, Tensor):
            raise TypeError("features and episode delta must be tensors")
        if delta.shape != (self.num_bases,) or delta.device != h.device or delta.dtype != h.dtype:
            raise ValueError("delta shape/device/dtype mismatch")
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("non-finite episode delta")
        basis = self.bases(h)
        # Never branch on delta == 0: the identity value must retain its
        # nonzero delta Jacobian for the first inner update.
        result = h + (basis * delta.view(1, -1, 1, 1, 1)).sum(dim=1)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("non-finite adapted D0 features")
        return result


__all__ = ["IdentityMetaAdapter"]
