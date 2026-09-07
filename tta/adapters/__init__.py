"""Lightweight, source-identity feature adapters for CR-SITTA."""

from .decoder_film import DecoderFiLM
from .decoder_hook import Decoder0HookError, decoder0_modulation
from .low_rank_feature_mixer import LowRankResidualMixer
from .nsfpn_router_adapter import (
    NSFPNRouterAdapter,
    NSFPNRouterAdapterError,
    NSFPNSpatialRouterAdapter,
    ROUTER_D0_SPACE,
    ROUTER_E1_D0_SPACE,
    ROUTER_E1_SPACE,
    ROUTER_SPACES,
    RouterSpace,
    build_default_nsfpn_routers,
)
from .spatial_low_rank_film import SpatialLowRankFiLM

__all__ = [
    "Decoder0HookError",
    "DecoderFiLM",
    "LowRankResidualMixer",
    "NSFPNRouterAdapter",
    "NSFPNRouterAdapterError",
    "NSFPNSpatialRouterAdapter",
    "ROUTER_D0_SPACE",
    "ROUTER_E1_D0_SPACE",
    "ROUTER_E1_SPACE",
    "ROUTER_SPACES",
    "RouterSpace",
    "SpatialLowRankFiLM",
    "build_default_nsfpn_routers",
    "decoder0_modulation",
]
