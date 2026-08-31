"""SS-calibration-only Binary TENT fast runner with zero-effect support.

The historical :mod:`tta.binary_tent_fast_runner` deliberately rejects an
optimizer step when no BatchNorm affine tensor changes byte-for-byte.  That
strict behaviour is part of already sealed artifact lineage and must remain
unchanged.  This v2 runner keeps the same Binary TENT method and the same
single ``optimizer.step()`` implementation, but accepts a numerically
zero-effect step only when temporary finite optimizer state exists and every
other episodic invariant/reset gate passes.

This module is intentionally narrow: it is used only by the source-train SS
calibration v2 protocol and is bound separately by that protocol's runtime
seal.  Zero/nonzero update activity is diagnostic evidence, never a selector
gate or ranking input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Any

import torch
from torch import Tensor

from tta.binary_tent import BinaryTentOutcome
from tta.binary_tent_fast_runner import (
    BinaryTentFastEpisodeResult,
    BinaryTentFastProtocolError,
    BinaryTentFastRunner,
    ResidentStateCheck,
)
from tta.episodic_runner import EpisodeProtocolError, _safe_metadata


RELATIVE_STEP_NORM_EPSILON = 1e-12
ZERO_UPDATE_POLICY = (
    "allow_only_with_one_finite_optimizer_step_temporary_state_and_exact_reset"
)


class BinaryTentFastRunnerV2(BinaryTentFastRunner):
    """Allow a truthful zero-effect result without weakening v1 semantics."""

    def __init__(
        self,
        adapter: Any,
        state_manager: Any,
        method: Any,
        *,
        full_audit_cadence: int | None = None,
        require_nonzero_parameter_update: bool = False,
    ) -> None:
        if not isinstance(require_nonzero_parameter_update, bool):
            raise TypeError("require_nonzero_parameter_update must be boolean")
        if require_nonzero_parameter_update:
            raise ValueError(
                "BinaryTentFastRunnerV2 is the allow-zero SS-v2 runner; "
                "use BinaryTentFastRunner for strict nonzero enforcement"
            )
        super().__init__(
            adapter,
            state_manager,
            method,
            full_audit_cadence=full_audit_cadence,
        )
        self.require_nonzero_parameter_update = False
        component_norms = tuple(
            float(
                torch.linalg.vector_norm(
                    parameter.detach().to(dtype=torch.float64)
                ).item()
            )
            for parameter in self._adaptable_objects
        )
        if not component_norms or not all(
            math.isfinite(value) and value >= 0.0 for value in component_norms
        ):
            raise BinaryTentFastProtocolError(
                "v2 Source BN-affine parameter norm is invalid"
            )
        self._source_parameter_norm = math.sqrt(
            math.fsum(value * value for value in component_norms)
        )
        if not math.isfinite(self._source_parameter_norm):
            raise BinaryTentFastProtocolError(
                "v2 Source BN-affine parameter norm is NaN/Inf"
            )

    def _run_body(
        self,
        *,
        image: Tensor,
        metadata: Mapping[str, Any],
        audit_due: bool,
    ) -> dict[str, Any]:
        """Run the v1 body with only its nonzero-delta premise generalized."""

        if not isinstance(image, Tensor) or image.ndim != 4:
            raise ValueError("image must be a tensor with shape [1,C,H,W]")
        if image.shape[0] != 1 or image.shape[1] < 1:
            raise ValueError(
                "Binary TENT fast runner requires a batch of exactly one image"
            )
        if not torch.is_floating_point(image) or not torch.isfinite(image).all():
            raise ValueError("image must be a finite floating-point tensor")
        try:
            safe_metadata = _safe_metadata(metadata)
        except EpisodeProtocolError as error:
            raise BinaryTentFastProtocolError(str(error)) from error

        checks: dict[str, ResidentStateCheck] = {}
        self._assert_canonical_source("episode_start")
        self._assert_optimizer_source("episode_start")
        checks["episode_start"] = self._assert_resident_state(
            "episode_start", include_adaptable=True
        )

        canonical_image = image.detach().clone()
        input_reference = canonical_image.clone()
        logits_source = self._checked_forward(canonical_image, grad=False)
        self._assert_canonical_source("source_forward")
        self._assert_optimizer_source("source_forward")
        checks["source_forward"] = self._assert_resident_state(
            "source_forward", include_adaptable=True
        )

        self.method.prepare_episode(self.adapter)
        self._assert_tent_runtime("prepare_episode")
        self._assert_optimizer_source("prepare_episode")
        checks["prepare_episode"] = self._assert_resident_state(
            "prepare_episode"
        )

        if torch.is_inference_mode_enabled():
            raise BinaryTentFastProtocolError(
                "Binary TENT adaptation cannot run inside torch.inference_mode()"
            )
        with torch.enable_grad():
            outcome = self.method.adapt_one_image(
                adapter=self.adapter,
                image=canonical_image.clone(),
                logits_pre=logits_source.clone(),
                metadata=safe_metadata,
            )
        if not isinstance(outcome, BinaryTentOutcome):
            raise BinaryTentFastProtocolError(
                "Binary TENT fast runner requires BinaryTentOutcome"
            )
        if outcome.decision != "adapted" or outcome.optimizer_steps != 1:
            raise BinaryTentFastProtocolError(
                "Binary TENT fast runner requires exactly one accepted optimizer step"
            )

        self._assert_tent_runtime("adapt_one_image")
        checks["adapt_one_image"] = self._assert_resident_state(
            "adapt_one_image"
        )
        changed_adaptable_tensors = self._adaptable_change_count()
        optimizer_after_adapt = self._capture_optimizer_state("adapt_one_image")
        if not optimizer_after_adapt.parameter_names_with_state:
            raise BinaryTentFastProtocolError(
                "Binary TENT step did not create temporary optimizer state"
            )
        adaptable_after_adapt = self._adaptable_values()

        adapt_fingerprint = None
        adapt_differences = None
        if audit_due:
            adapt_fingerprint = self.state.current_fingerprint()
            adapt_differences = adapt_fingerprint.differing_components(
                self.state.source_fingerprint
            )
            expected_differences = (
                ("model", "optimizer", "runtime")
                if changed_adaptable_tensors > 0
                else ("optimizer", "runtime")
            )
            if adapt_differences != expected_differences:
                raise BinaryTentFastProtocolError(
                    "v2 full adaptation audit expected components "
                    f"{expected_differences} to differ from Source, got "
                    f"{adapt_differences}"
                )

        logits_post = self._checked_forward(canonical_image, grad=False)
        self._assert_tent_runtime("post_forward")
        checks["post_forward"] = self._assert_resident_state("post_forward")
        self._assert_adaptable_exact(adaptable_after_adapt, "post_forward")
        self._assert_optimizer_state_exact(optimizer_after_adapt, "post_forward")

        post_fingerprint = None
        if audit_due:
            post_fingerprint = self.state.current_fingerprint()
            if post_fingerprint != adapt_fingerprint:
                differences = post_fingerprint.differing_components(
                    adapt_fingerprint
                )
                raise BinaryTentFastProtocolError(
                    "post-update forward changed state after adaptation: "
                    + ", ".join(differences)
                )

        if not torch.equal(canonical_image, input_reference):
            raise BinaryTentFastProtocolError("episode modified the canonical input")

        return {
            "metadata": safe_metadata,
            "logits_source": logits_source,
            "outcome": outcome,
            "logits_post": logits_post,
            "resident_checks": checks,
            "changed_adaptable_tensors": changed_adaptable_tensors,
            "optimizer_state_parameter_count": len(
                optimizer_after_adapt.parameter_names_with_state
            ),
            "adapt_fingerprint": adapt_fingerprint,
            "post_fingerprint": post_fingerprint,
            "adapt_differences": adapt_differences,
        }

    def run_one_image(
        self,
        *,
        image: Tensor,
        metadata: Mapping[str, Any],
        force_full_audit: bool = False,
    ) -> BinaryTentFastEpisodeResult:
        """Return v1 evidence plus truthful, ranking-excluded strength fields."""

        result = super().run_one_image(
            image=image,
            metadata=metadata,
            force_full_audit=force_full_audit,
        )
        diagnostics = dict(result.diagnostics)
        changed = diagnostics.get("changed_bn_affine_tensors_fast_gate")
        if isinstance(changed, bool) or not isinstance(changed, int) or changed < 0:
            self._aborted = True
            raise BinaryTentFastProtocolError(
                "v2 changed BN-affine tensor count is invalid"
            )
        step_norm = diagnostics.get("step_norm")
        if isinstance(step_norm, bool) or not isinstance(step_norm, (int, float)):
            self._aborted = True
            raise BinaryTentFastProtocolError("v2 parameter step norm is invalid")
        step_norm = float(step_norm)
        relative_step_norm = step_norm / (
            self._source_parameter_norm + RELATIVE_STEP_NORM_EPSILON
        )
        if not math.isfinite(relative_step_norm) or relative_step_norm < 0.0:
            self._aborted = True
            raise BinaryTentFastProtocolError(
                "v2 relative parameter step norm is invalid"
            )
        diagnostics.update(
            {
                "require_nonzero_parameter_update": False,
                "zero_parameter_update_allowed": True,
                "zero_parameter_update_observed": changed == 0,
                "numerically_zero_parameter_delta": changed == 0,
                "source_parameter_norm": self._source_parameter_norm,
                "relative_step_norm": relative_step_norm,
                "relative_step_norm_epsilon": RELATIVE_STEP_NORM_EPSILON,
                "zero_parameter_update_policy": ZERO_UPDATE_POLICY,
            }
        )
        return replace(result, diagnostics=diagnostics)


__all__ = [
    "BinaryTentFastRunnerV2",
    "RELATIVE_STEP_NORM_EPSILON",
    "ZERO_UPDATE_POLICY",
]
