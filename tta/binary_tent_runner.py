"""Three-prediction audit wrapper for the frozen episodic runner.

The shared :mod:`tta.episodic_runner` is part of the completed AdaBN artifact
and must remain byte-identical.  This subclass adds source-stat BN validation
and exposes Source, update-time, and post-update logits without changing that
frozen implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from tta.binary_tent import (
    BN_PROTOCOL_BATCH_STATS,
    BN_PROTOCOL_SOURCE_STATS,
    BinaryTentOutcome,
    BinaryTentProtocolError,
    binary_entropy_map,
)
from tta.episodic_runner import EpisodeResult, EpisodicMethod, EpisodicRunner


ENTROPY_REDUCTION_REL_TOL = 1e-6
ENTROPY_REDUCTION_ABS_TOL = 1e-7


def _cpu_tensor_sha256(value: Tensor) -> str:
    value = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


@dataclass(frozen=True)
class BinaryTentEpisodeResult:
    """A completed episode with the three scientifically distinct outputs."""

    episode: EpisodeResult
    logits_source_pre: Tensor
    logits_tent_pre: Tensor
    logits_tent_post: Tensor
    entropy_tent_pre: float
    entropy_tent_post: float
    source_tent_pre_bit_exact: bool
    tent_pre_post_bit_exact: bool
    diagnostics: MappingProxyType


class BinaryTentEpisodicRunner(EpisodicRunner):
    """Use the frozen generic reset flow with both registered BN protocols."""

    def _assert_method_bn_protocol(self, method: EpisodicMethod) -> None:
        protocol = getattr(method, "bn_protocol", None)
        if protocol == BN_PROTOCOL_BATCH_STATS:
            super()._assert_method_bn_protocol(method)
            return
        if protocol != BN_PROTOCOL_SOURCE_STATS:
            raise BinaryTentProtocolError(
                f"unsupported Binary TENT BN protocol: {protocol!r}"
            )

        count = 0
        for name, module in self.adapter.model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            count += 1
            if module.training or not module.track_running_stats:
                raise BinaryTentProtocolError(
                    f"source-stat BatchNorm must be eval/tracked at {name!r}"
                )
            if (
                module.running_mean is None
                or module.running_var is None
                or module.num_batches_tracked is None
            ):
                raise BinaryTentProtocolError(
                    f"source-stat BatchNorm is missing buffers at {name!r}"
                )
        if count == 0:
            raise BinaryTentProtocolError(
                "source-stat Binary TENT requires at least one BatchNorm2d"
            )

    def run_one_image(self, **kwargs: Any) -> BinaryTentEpisodeResult:
        episode = super().run_one_image(**kwargs)
        outcome = episode.outcome
        if not isinstance(outcome, BinaryTentOutcome):
            raise BinaryTentProtocolError(
                "BinaryTentEpisodicRunner requires BinaryTentOutcome"
            )

        source = episode.logits_pre.detach().cpu().clone()
        tent_pre = outcome.logits_tent_pre.detach().cpu().clone()
        tent_post = episode.logits_post.detach().cpu().clone()
        eps = float(outcome.diagnostics["entropy_eps"])
        entropy_pre = float(binary_entropy_map(tent_pre, eps=eps).mean().item())
        entropy_post = float(binary_entropy_map(tent_post, eps=eps).mean().item())
        if not math.isfinite(entropy_pre) or not math.isfinite(entropy_post):
            raise BinaryTentProtocolError("episode entropy diagnostics are non-finite")
        reported_pre = float(
            outcome.diagnostics["optimization_entropy_pre_device"]
        )
        entropy_pre_reduction_abs_error = abs(entropy_pre - reported_pre)
        if not math.isclose(
            entropy_pre,
            reported_pre,
            rel_tol=ENTROPY_REDUCTION_REL_TOL,
            abs_tol=ENTROPY_REDUCTION_ABS_TOL,
        ):
            raise BinaryTentProtocolError(
                "stored update-time entropy differs from returned TENT-pre logits"
            )

        source_tent_exact = bool(torch.equal(source, tent_pre))
        if (
            outcome.diagnostics["bn_protocol"] == BN_PROTOCOL_SOURCE_STATS
            and not source_tent_exact
        ):
            raise BinaryTentProtocolError(
                "source-stat TENT-pre logits must be bit-exact with Source logits"
            )

        if episode.state_changes_after_prepare != ("runtime",):
            raise BinaryTentProtocolError(
                "Binary TENT prepare must change exactly the runtime component"
            )
        if (
            episode.state_after_adapt_fingerprint
            != episode.state_after_post_fingerprint
        ):
            differences = episode.state_after_post_fingerprint.differing_components(
                episode.state_after_adapt_fingerprint
            )
            raise BinaryTentProtocolError(
                "post-update inference changed state after adaptation: "
                + ", ".join(differences)
            )

        merged = dict(outcome.diagnostics)
        merged.update(
            {
                "entropy_pre": entropy_pre,
                "entropy_post": entropy_post,
                "entropy_delta": entropy_post - entropy_pre,
                "entropy_pre_reduction_abs_error": entropy_pre_reduction_abs_error,
                "entropy_reduction_rel_tolerance": ENTROPY_REDUCTION_REL_TOL,
                "entropy_reduction_abs_tolerance": ENTROPY_REDUCTION_ABS_TOL,
                "source_tent_pre_bit_exact": source_tent_exact,
                "tent_pre_post_bit_exact": bool(torch.equal(tent_pre, tent_post)),
                "source_pre_raw_sha256": _cpu_tensor_sha256(source),
                "tent_pre_raw_sha256": _cpu_tensor_sha256(tent_pre),
                "tent_post_raw_sha256": _cpu_tensor_sha256(tent_post),
                "post_forward_state_unchanged": True,
            }
        )
        return BinaryTentEpisodeResult(
            episode=episode,
            logits_source_pre=source,
            logits_tent_pre=tent_pre,
            logits_tent_post=tent_post,
            entropy_tent_pre=entropy_pre,
            entropy_tent_post=entropy_post,
            source_tent_pre_bit_exact=source_tent_exact,
            tent_pre_post_bit_exact=bool(torch.equal(tent_pre, tent_post)),
            diagnostics=MappingProxyType(merged),
        )


__all__ = [
    "BinaryTentEpisodeResult",
    "BinaryTentEpisodicRunner",
    "ENTROPY_REDUCTION_ABS_TOL",
    "ENTROPY_REDUCTION_REL_TOL",
]
