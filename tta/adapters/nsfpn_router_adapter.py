"""Explicit parameter-space control for Stage-C NS-FPN spatial routers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .spatial_low_rank_film import SpatialLowRankFiLM


RouterSpace: TypeAlias = Literal["R-E1", "R-D0", "R-E1+D0"]
ROUTER_E1_SPACE: RouterSpace = "R-E1"
ROUTER_D0_SPACE: RouterSpace = "R-D0"
ROUTER_E1_D0_SPACE: RouterSpace = "R-E1+D0"
ROUTER_SPACES = (
    ROUTER_E1_SPACE,
    ROUTER_D0_SPACE,
    ROUTER_E1_D0_SPACE,
)

_SPACE_ALIASES = {
    "E1": ROUTER_E1_SPACE,
    "D0": ROUTER_D0_SPACE,
    "E1+D0": ROUTER_E1_D0_SPACE,
    **{space: space for space in ROUTER_SPACES},
}


class NSFPNRouterAdapterError(RuntimeError):
    """The explicit NS-FPN router parameter-space contract was violated."""


def _canonical_space(space: str) -> RouterSpace:
    if not isinstance(space, str):
        raise TypeError("space must be a string")
    try:
        return _SPACE_ALIASES[space]
    except KeyError as exc:
        raise ValueError(
            f"unsupported router space {space!r}; expected one of {ROUTER_SPACES}"
        ) from exc


def build_default_nsfpn_routers(
    space: str,
    *,
    grid_size: tuple[int, int] = (8, 8),
    max_scale_delta: float = 0.05,
    max_bias: float = 0.05,
    seed: int = 3407,
) -> tuple[nn.Module, nn.Module]:
    """Build the frozen v6 E1/D0 router topology for one parameter space."""

    canonical = _canonical_space(space)
    router_e1: nn.Module = nn.Identity()
    router_d0: nn.Module = nn.Identity()
    if canonical in (ROUTER_E1_SPACE, ROUTER_E1_D0_SPACE):
        router_e1 = SpatialLowRankFiLM(
            64,
            rank=4,
            grid_size=grid_size,
            max_scale_delta=max_scale_delta,
            max_bias=max_bias,
            seed=seed,
        )
    if canonical in (ROUTER_D0_SPACE, ROUTER_E1_D0_SPACE):
        router_d0 = SpatialLowRankFiLM(
            16,
            rank=2,
            grid_size=grid_size,
            max_scale_delta=max_scale_delta,
            max_bias=max_bias,
            seed=seed,
        )
    return router_e1, router_d0


class NSFPNRouterAdapter:
    """Expose exactly one of R-E1, R-D0, or R-E1+D0 for adaptation.

    The wrapper deliberately is not an ``nn.Module``: the adaptable NS-FPN
    remains the sole owner of both Source and router parameters, avoiding a
    second checkpoint prefix.  Source weights are frozen and only the selected
    router coefficient maps receive gradients.
    """

    def __init__(self, model: nn.Module, *, space: str) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.space = _canonical_space(space)
        self._router_e1 = self._require_router("router_e1")
        self._router_d0 = self._require_router("router_d0")

        # Keep the immutable Source-buffer anchor outside the live model and on
        # CPU.  This covers BN running statistics/counters as well as every
        # other registered buffer, including the fixed router bases.  A tuple
        # is used deliberately so callers cannot replace entries through a
        # public mutable mapping.
        self._source_buffer_snapshot = tuple(
            (name, buffer.detach().to(device="cpu").clone())
            for name, buffer in self.model.named_buffers()
        )

        if self.space in (ROUTER_E1_SPACE, ROUTER_E1_D0_SPACE):
            if not isinstance(self._router_e1, SpatialLowRankFiLM):
                raise NSFPNRouterAdapterError(
                    "R-E1 requires model.router_e1 to be SpatialLowRankFiLM"
                )
        if self.space in (ROUTER_D0_SPACE, ROUTER_E1_D0_SPACE):
            if not isinstance(self._router_d0, SpatialLowRankFiLM):
                raise NSFPNRouterAdapterError(
                    "R-D0 requires model.router_d0 to be SpatialLowRankFiLM"
                )
        self.set_router_mode()

    def _current_named_buffers(self) -> tuple[tuple[str, Tensor], ...]:
        current = tuple(self.model.named_buffers())
        expected_names = tuple(name for name, _buffer in self._source_buffer_snapshot)
        actual_names = tuple(name for name, _buffer in current)
        if actual_names != expected_names:
            raise NSFPNRouterAdapterError(
                "model buffer topology differs from the immutable Source snapshot"
            )
        return current

    def _restore_source_buffers_(self) -> None:
        current = self._current_named_buffers()
        with torch.no_grad():
            for (name, buffer), (snapshot_name, snapshot) in zip(
                current, self._source_buffer_snapshot, strict=True
            ):
                if name != snapshot_name or buffer.shape != snapshot.shape:
                    raise NSFPNRouterAdapterError(
                        "model buffer layout differs from the immutable Source snapshot"
                    )
                buffer.copy_(snapshot.to(device=buffer.device, dtype=buffer.dtype))
        self._assert_source_buffers_restored()

    def _assert_source_buffers_restored(self) -> None:
        for (name, buffer), (snapshot_name, snapshot) in zip(
            self._current_named_buffers(), self._source_buffer_snapshot, strict=True
        ):
            if name != snapshot_name or buffer.shape != snapshot.shape:
                raise NSFPNRouterAdapterError(
                    "model buffer layout differs from the immutable Source snapshot"
                )
            expected = snapshot.to(device=buffer.device, dtype=buffer.dtype)
            if not torch.equal(buffer.detach(), expected):
                raise NSFPNRouterAdapterError(
                    f"model buffer {name!r} differs from the immutable Source snapshot"
                )

    def _assert_registered_routers_identity(self) -> None:
        for name, router in (
            ("router_e1", self._router_e1),
            ("router_d0", self._router_d0),
        ):
            if isinstance(router, SpatialLowRankFiLM):
                if not router.is_identity():
                    raise NSFPNRouterAdapterError(
                        f"{name} is not identity; refusing to expose adapted state as Source"
                    )
            elif any(True for _parameter in router.parameters()):
                raise NSFPNRouterAdapterError(
                    f"{name} has parameters but no auditable identity contract"
                )

    def _require_router(self, name: str) -> nn.Module:
        router = getattr(self.model, name, None)
        if not isinstance(router, nn.Module):
            raise NSFPNRouterAdapterError(
                f"model.{name} must be an explicitly registered nn.Module"
            )
        return router

    def _selected_routers(self) -> tuple[tuple[str, SpatialLowRankFiLM], ...]:
        result: list[tuple[str, SpatialLowRankFiLM]] = []
        if self.space in (ROUTER_E1_SPACE, ROUTER_E1_D0_SPACE):
            if not isinstance(self._router_e1, SpatialLowRankFiLM):
                raise NSFPNRouterAdapterError("router_e1 topology changed")
            result.append(("router_e1", self._router_e1))
        if self.space in (ROUTER_D0_SPACE, ROUTER_E1_D0_SPACE):
            if not isinstance(self._router_d0, SpatialLowRankFiLM):
                raise NSFPNRouterAdapterError("router_d0 topology changed")
            result.append(("router_d0", self._router_d0))
        return tuple(result)

    def named_adaptable_parameters(
        self,
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        named: list[tuple[str, nn.Parameter]] = []
        for router_name, router in self._selected_routers():
            local = tuple(router.named_parameters())
            if tuple(name for name, _ in local) != (
                "scale_coeff",
                "bias_coeff",
            ):
                raise NSFPNRouterAdapterError(
                    f"{router_name} trainable topology changed"
                )
            named.extend(
                (f"{router_name}.{name}", parameter)
                for name, parameter in local
            )
        if not named or len({id(parameter) for _, parameter in named}) != len(named):
            raise NSFPNRouterAdapterError(
                "router parameter topology is empty or aliases parameters"
            )
        return tuple(named)

    def collect_adaptable_params(self) -> tuple[list[nn.Parameter], list[str]]:
        """Return selected coefficient maps and stable fully-qualified names."""

        named = self.named_adaptable_parameters()
        return (
            [parameter for _, parameter in named],
            [name for name, _ in named],
        )

    def set_source_eval_mode(self) -> None:
        """Use the exact Source endpoint or fail closed on residual state."""

        self._assert_registered_routers_identity()
        self._assert_source_buffers_restored()
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def set_router_mode(self) -> None:
        """Freeze Source and enable only the selected router coefficients."""

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for _name, parameter in self.named_adaptable_parameters():
            parameter.requires_grad_(True)
        self.assert_only_router_trainable()

    def assert_only_router_trainable(self) -> None:
        expected = {
            id(parameter) for _, parameter in self.named_adaptable_parameters()
        }
        actual = {
            id(parameter)
            for parameter in self.model.parameters()
            if parameter.requires_grad
        }
        if actual != expected:
            raise NSFPNRouterAdapterError(
                "requires_grad is not restricted to the selected router space"
            )

    def _validate_optimizer(self, optimizer: Optimizer) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be torch.optim.Optimizer")
        expected = tuple(
            parameter for _, parameter in self.named_adaptable_parameters()
        )
        actual = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        if len(actual) != len(expected) or any(
            current is not wanted
            for current, wanted in zip(actual, expected, strict=True)
        ):
            raise NSFPNRouterAdapterError(
                "optimizer parameter layout differs from selected routers"
            )

    def reset_identity_(self, *, optimizer: Optimizer | None = None) -> None:
        """Restore Source buffers and reset routers, gradients, and optimizer."""

        if optimizer is not None:
            self._validate_optimizer(optimizer)
        for router in (self._router_e1, self._router_d0):
            if isinstance(router, SpatialLowRankFiLM):
                router.reset_identity_(clear_gradients=True)
        self._restore_source_buffers_()
        if optimizer is not None:
            optimizer.state.clear()
        self.set_router_mode()
        self._assert_registered_routers_identity()
        self._assert_source_buffers_restored()

    # Explicit episode terminology for callers that manage per-image state.
    reset_episode_ = reset_identity_

    def forward_logits(self, image: Tensor, *, warm_flag: bool = False) -> Tensor:
        if not isinstance(image, Tensor) or image.ndim != 4:
            raise ValueError("image must have shape [B, C, H, W]")
        output = self.model(image, warm_flag)
        if (
            not isinstance(output, Sequence)
            or len(output) != 2
            or not isinstance(output[1], Tensor)
        ):
            raise TypeError("adaptable NS-FPN must return (auxiliary, logits)")
        logits = output[1]
        if logits.ndim != 4 or logits.shape[:2] != (image.shape[0], 1):
            raise ValueError("final logits must have shape [B, 1, H, W]")
        return logits


# Readable alias for callers that spell out the spatial nature of the adapter.
NSFPNSpatialRouterAdapter = NSFPNRouterAdapter


__all__ = [
    "NSFPNRouterAdapter",
    "NSFPNRouterAdapterError",
    "NSFPNSpatialRouterAdapter",
    "ROUTER_D0_SPACE",
    "ROUTER_E1_D0_SPACE",
    "ROUTER_E1_SPACE",
    "ROUTER_SPACES",
    "RouterSpace",
    "build_default_nsfpn_routers",
]
