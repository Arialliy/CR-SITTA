"""NS-FPN with explicit Stage-C spatial-router insertion points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .MSHNet_NSFPN import MSHNet_NSFPN, ResNet


_ROUTER_PREFIXES = ("router_e1.", "router_d0.")


class MSHNetNSFPNAdaptable(MSHNet_NSFPN):
    """Explicitly route the FPN E1 and full-resolution D0 features.

    ``None`` selects ``nn.Identity`` at either insertion point.  A legacy
    Source checkpoint can still be loaded with ``strict=True``: its complete
    original key set is checked exactly, while the newly registered routers
    are reset to their identity state and retain their fixed basis buffers.
    Full adaptable state dictionaries continue to use ordinary strict PyTorch
    semantics.
    """

    def __init__(
        self,
        input_channels: int,
        block: type[nn.Module] = ResNet,
        *,
        router_e1: nn.Module | None = None,
        router_d0: nn.Module | None = None,
    ) -> None:
        super().__init__(input_channels=input_channels, block=block)
        if router_e1 is not None and not isinstance(router_e1, nn.Module):
            raise TypeError("router_e1 must be an nn.Module or None")
        if router_d0 is not None and not isinstance(router_d0, nn.Module):
            raise TypeError("router_d0 must be an nn.Module or None")
        self.router_e1 = router_e1 if router_e1 is not None else nn.Identity()
        self.router_d0 = router_d0 if router_d0 is not None else nn.Identity()

    def _reset_registered_routers_identity_(self) -> None:
        for name, router in (
            ("router_e1", self.router_e1),
            ("router_d0", self.router_d0),
        ):
            reset = getattr(router, "reset_identity_", None)
            if callable(reset):
                reset()
            elif any(True for _ in router.parameters()):
                raise RuntimeError(
                    f"{name} has parameters but no reset_identity_ contract"
                )

    def _legacy_source_keys(self) -> frozenset[str]:
        return frozenset(
            key
            for key in self.state_dict()
            if not key.startswith(_ROUTER_PREFIXES)
        )

    def load_source_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        assign: bool = False,
    ):
        """Strictly load an original ``MSHNet_NSFPN`` state dictionary."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        expected = self._legacy_source_keys()
        actual = frozenset(state_dict)
        if actual != expected:
            missing = tuple(sorted(expected - actual))
            unexpected = tuple(sorted(actual - expected))
            raise RuntimeError(
                "legacy Source state_dict keys do not exactly match; "
                f"missing={missing}, unexpected={unexpected}"
            )

        self._reset_registered_routers_identity_()
        augmented = self.state_dict()
        for key in expected:
            augmented[key] = state_dict[key]
        return super().load_state_dict(augmented, strict=True, assign=assign)

    def load_state_dict(  # type: ignore[override]
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load a full adaptable state or an exact legacy Source state."""

        if isinstance(state_dict, Mapping):
            incoming = frozenset(state_dict)
            legacy = self._legacy_source_keys()
            full = frozenset(self.state_dict())
            if strict and incoming == legacy and legacy != full:
                return self.load_source_state_dict(state_dict, assign=assign)
        return super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )

    def forward(
        self, x: Tensor, warm_flag: bool = False
    ) -> tuple[list[Tensor], Tensor]:
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))

        x_e1, x_e2, x_e3, x_m = self.fpn([x_e1, x_e2, x_e3, x_m])
        x_e1 = self.router_e1(x_e1)

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))
        x_d0 = self.router_d0(x_d0)

        if warm_flag:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(
                torch.cat(
                    [
                        mask0,
                        self.up(mask1),
                        self.up_4(mask2),
                        self.up_8(mask3),
                    ],
                    dim=1,
                )
            )
            return [mask0, mask1, mask2, mask3], output

        return [], self.output_0(x_d0)


# A conservative alias accommodates the repository's underscore-heavy model
# naming without creating a second implementation.
MSHNet_NSFPN_Adaptable = MSHNetNSFPNAdaptable


__all__ = ["MSHNetNSFPNAdaptable", "MSHNet_NSFPN_Adaptable"]
