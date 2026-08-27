"""Single-image episodic AdaBN / Test-Time Norm baseline.

This baseline isolates the effect of replacing Source BatchNorm statistics
with the current image's spatial statistics.  It performs no optimisation and
owns no learnable or persistent state.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from tta.episodic_runner import AdaptationOutcome
from tta.model_adapter import IRSTDModelAdapter


class AdaBNMethod:
    """Current-image BN statistics with frozen affine parameters and buffers."""

    name = "adabn"
    optimizer = None
    requires_grad = False
    bn_protocol = "single_image_spatial_batch_stats"
    allowed_state_changes = frozenset({"runtime"})
    exact_state_changes = frozenset({"runtime"})
    state_frozen_after_prepare = True

    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        adapter.set_adabn_mode()

    def adapt_one_image(
        self,
        *,
        adapter: IRSTDModelAdapter,
        image: Tensor,
        logits_pre: Tensor,
        metadata: Mapping[str, Any],
    ) -> AdaptationOutcome:
        # The runner's post prediction is the sole current-statistics forward.
        # Keeping this hook forward-free prevents accidental double exposure.
        del image, logits_pre, metadata
        batchnorm_count = sum(
            isinstance(module, nn.BatchNorm2d)
            for module in adapter.model.modules()
        )
        if batchnorm_count == 0:
            raise RuntimeError("AdaBN was prepared without any BatchNorm2d modules")
        return AdaptationOutcome(
            decision="adapted",
            optimizer_steps=0,
            diagnostics={
                "bn_protocol": "single_image_spatial_batch_stats",
                "batchnorm2d_modules": batchnorm_count,
                "learnable_update": False,
                "running_statistics_accumulated": False,
                "statistics_forward": "runner_post_prediction",
            },
        )


__all__ = ["AdaBNMethod"]
