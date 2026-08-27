"""Test-time adaptation building blocks for CR-SITTA."""

from .adabn import AdaBNMethod
from .episodic_runner import (
    AdaptationOutcome,
    EpisodeProtocolError,
    EpisodeResult,
    EpisodicMethod,
    EpisodicRunner,
    NoUpdateMethod,
)
from .model_adapter import IRSTDModelAdapter
from .state_manager import (
    EpisodicStateManager,
    SourceSnapshotError,
    SourceStateMismatchError,
    StateFingerprint,
    StatefulHooks,
)

__all__ = [
    "AdaBNMethod",
    "AdaptationOutcome",
    "EpisodeProtocolError",
    "EpisodeResult",
    "EpisodicMethod",
    "EpisodicRunner",
    "EpisodicStateManager",
    "IRSTDModelAdapter",
    "NoUpdateMethod",
    "SourceSnapshotError",
    "SourceStateMismatchError",
    "StateFingerprint",
    "StatefulHooks",
]
