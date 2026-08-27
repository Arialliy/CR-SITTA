"""Label-free single-image episodic execution with fail-closed reset gates.

The runner deliberately contains no adaptation objective and no evaluator.
It exposes only an image, frozen non-label metadata, and source logits to a
method.  Ground-truth masks remain in the outer benchmark/evaluator process.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import random
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager, StateFingerprint


SAFE_METADATA_KEYS = frozenset(
    {
        "image_id",
        "original_size",
        "dataset",
        "corruption",
        "severity",
        "seed",
    }
)


class EpisodeProtocolError(RuntimeError):
    """Raised when a method violates the frozen episodic execution contract."""


@dataclass(frozen=True)
class AdaptationOutcome:
    """Method-reported result of one adaptation attempt.

    ``decision='adapted'`` also covers a statistics-only method such as AdaBN,
    for which ``optimizer_steps`` is zero.  The actual full-state hashes are
    measured by the runner and never trusted from this self-report.
    """

    decision: Literal["no_update", "adapted"]
    optimizer_steps: int
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.decision not in ("no_update", "adapted"):
            raise ValueError("decision must be 'no_update' or 'adapted'")
        if isinstance(self.optimizer_steps, bool) or not isinstance(
            self.optimizer_steps, int
        ):
            raise TypeError("optimizer_steps must be an integer")
        if self.optimizer_steps < 0:
            raise ValueError("optimizer_steps must be non-negative")
        if self.decision == "no_update" and self.optimizer_steps != 0:
            raise ValueError("a no-update outcome cannot report optimizer steps")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("diagnostics must be a mapping")
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(deepcopy(dict(self.diagnostics))),
        )


class EpisodicMethod(Protocol):
    """Minimal method interface shared later by no-update, AdaBN, and TENT."""

    name: str
    requires_grad: bool
    allowed_state_changes: frozenset[str]

    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        """Configure model modes/gradients after the Source pre-forward."""

    def adapt_one_image(
        self,
        *,
        adapter: IRSTDModelAdapter,
        image: Tensor,
        logits_pre: Tensor,
        metadata: Mapping[str, Any],
    ) -> AdaptationOutcome:
        """Adapt from private tensor copies without access to any label."""


class NoUpdateMethod:
    """Strict Source control used to validate the episodic runner itself."""

    name = "no_update"
    optimizer = None
    requires_grad = False
    allowed_state_changes: frozenset[str] = frozenset()

    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        adapter.set_source_eval_mode()

    def adapt_one_image(
        self,
        *,
        adapter: IRSTDModelAdapter,
        image: Tensor,
        logits_pre: Tensor,
        metadata: Mapping[str, Any],
    ) -> AdaptationOutcome:
        del adapter, image, logits_pre, metadata
        return AdaptationOutcome(
            decision="no_update",
            optimizer_steps=0,
            diagnostics={"reason": "forced_no_update"},
        )


@dataclass(frozen=True)
class EpisodeResult:
    """Detached outputs and audit evidence from one completed episode."""

    method: str
    metadata: Mapping[str, Any]
    logits_pre: Tensor
    logits_post: Tensor
    outcome: AdaptationOutcome
    pre_post_bit_exact: bool
    input_unchanged: bool
    source_fingerprint: StateFingerprint
    state_after_prepare_fingerprint: StateFingerprint
    state_after_adapt_fingerprint: StateFingerprint
    state_after_post_fingerprint: StateFingerprint
    reset_fingerprint: StateFingerprint
    state_changes_after_prepare: tuple[str, ...]
    state_changes_after_adapt: tuple[str, ...]
    state_changes_after_post: tuple[str, ...]
    source_state_sha256: str
    state_after_adapt_sha256: str
    state_after_post_sha256: str
    reset_state_sha256: str


def _safe_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and recursively freeze the complete label-free schema."""

    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    keys = set(metadata)
    unknown = sorted(str(key) for key in keys - SAFE_METADATA_KEYS)
    if unknown:
        raise EpisodeProtocolError(
            "metadata contains fields outside the label-free allowlist: "
            + ", ".join(unknown)
        )
    if "image_id" not in metadata:
        raise EpisodeProtocolError("metadata must contain image_id")

    required = SAFE_METADATA_KEYS
    missing = sorted(required - keys)
    if missing:
        raise EpisodeProtocolError(
            "metadata is missing required fields: " + ", ".join(missing)
        )

    strings: dict[str, str] = {}
    for key in ("image_id", "dataset", "corruption"):
        value = metadata[key]
        if not isinstance(value, str) or not value.strip():
            raise EpisodeProtocolError(f"metadata {key} must be a non-empty string")
        strings[key] = value

    original_size = metadata["original_size"]
    if not isinstance(original_size, (tuple, list)) or len(original_size) != 2:
        raise EpisodeProtocolError(
            "metadata original_size must contain exactly two integers"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in original_size
    ):
        raise EpisodeProtocolError(
            "metadata original_size must contain two positive integers"
        )

    severity = metadata["severity"]
    seed = metadata["seed"]
    if isinstance(severity, bool) or not isinstance(severity, int):
        raise EpisodeProtocolError("metadata severity must be an integer")
    if severity < 0 or severity > 5:
        raise EpisodeProtocolError("metadata severity must lie in [0, 5]")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EpisodeProtocolError("metadata seed must be an integer")
    if strings["corruption"] == "clean" and severity != 0:
        raise EpisodeProtocolError("clean metadata requires severity=0")
    if strings["corruption"] != "clean" and severity == 0:
        raise EpisodeProtocolError("corrupted metadata requires severity in [1, 5]")

    # Every value is immutable after normalisation, so a method cannot smuggle
    # or mutate a label through an otherwise allowed metadata field.
    return MappingProxyType(
        {
            "image_id": strings["image_id"],
            "original_size": tuple(original_size),
            "dataset": strings["dataset"],
            "corruption": strings["corruption"],
            "severity": severity,
            "seed": seed,
        }
    )


@dataclass(frozen=True)
class _RNGState:
    python: object
    numpy: tuple[Any, ...]
    torch_cpu: Tensor
    torch_cuda: tuple[Tensor, ...] | None


def _capture_rng_state() -> _RNGState:
    numpy_state = np.random.get_state()
    frozen_numpy_state = (
        numpy_state[0],
        numpy_state[1].copy(),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    cuda_state = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_initialized()
        else None
    )
    return _RNGState(
        python=random.getstate(),
        numpy=frozen_numpy_state,
        torch_cpu=torch.get_rng_state().clone(),
        torch_cuda=cuda_state,
    )


def _restore_rng_state(state: _RNGState) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.set_rng_state(state.torch_cpu)
    if state.torch_cuda is not None:
        if not torch.cuda.is_initialized():
            raise EpisodeProtocolError(
                "CUDA RNG was initialized at episode start but is now unavailable"
            )
        if len(state.torch_cuda) != torch.cuda.device_count():
            raise EpisodeProtocolError("CUDA device topology changed during episode")
        torch.cuda.set_rng_state_all(list(state.torch_cuda))


class EpisodicRunner:
    """Run one image from Source, through a temporary method state, then reset."""

    def __init__(
        self,
        adapter: IRSTDModelAdapter,
        state_manager: EpisodicStateManager,
    ) -> None:
        if not isinstance(adapter, IRSTDModelAdapter):
            raise TypeError("adapter must be an IRSTDModelAdapter")
        if not isinstance(state_manager, EpisodicStateManager):
            raise TypeError("state_manager must be an EpisodicStateManager")
        if state_manager.model is not adapter.model:
            raise ValueError("adapter and state_manager must own the same model")
        self.adapter = adapter
        self.state = state_manager
        # Fail early if the source snapshot was captured before source eval mode
        # was configured.  The correct construction order is part of the API.
        self.state.assert_source_state()
        self._assert_source_contract()
        self._assert_managed_optimizer_topology()

    def _batchnorm_affine_parameter_ids(self) -> set[int]:
        allowed: set[int] = set()
        for module in self.adapter.model.modules():
            if not isinstance(module, nn.BatchNorm2d) or not module.affine:
                continue
            if module.weight is not None:
                allowed.add(id(module.weight))
            if module.bias is not None:
                allowed.add(id(module.bias))
        return allowed

    def _assert_source_contract(self) -> None:
        training = [
            name or "<root>"
            for name, module in self.adapter.model.named_modules()
            if module.training
        ]
        if training:
            raise EpisodeProtocolError(
                "Source snapshot requires every module in eval mode: "
                + ", ".join(training[:10])
            )
        trainable = [
            name
            for name, parameter in self.adapter.model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable:
            raise EpisodeProtocolError(
                "Source snapshot requires every parameter frozen: "
                + ", ".join(trainable[:10])
            )
        gradients = [
            name
            for name, parameter in self.adapter.model.named_parameters()
            if parameter.grad is not None
        ]
        if gradients:
            raise EpisodeProtocolError(
                "Source snapshot requires every parameter gradient cleared: "
                + ", ".join(gradients[:10])
            )
        for name, module in self.adapter.model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if not module.track_running_stats:
                raise EpisodeProtocolError(
                    f"Source BatchNorm {name!r} must track running statistics"
                )
            if (
                module.running_mean is None
                or module.running_var is None
                or module.num_batches_tracked is None
            ):
                raise EpisodeProtocolError(
                    f"Source BatchNorm {name!r} is missing running-stat buffers"
                )

    def _assert_non_bn_modules_are_eval(self) -> None:
        offenders = [
            name or "<root>"
            for name, module in self.adapter.model.named_modules()
            if not isinstance(module, nn.BatchNorm2d) and module.training
        ]
        if offenders:
            raise EpisodeProtocolError(
                "non-BatchNorm modules entered training mode: "
                + ", ".join(offenders[:10])
            )

    def _assert_adaptation_contract(self) -> None:
        self._assert_non_bn_modules_are_eval()
        allowed = self._batchnorm_affine_parameter_ids()
        offenders = [
            name
            for name, parameter in self.adapter.model.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed
        ]
        if offenders:
            raise EpisodeProtocolError(
                "only BatchNorm2d affine parameters may require gradients: "
                + ", ".join(offenders[:10])
            )
        for name, module in self.adapter.model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if module.training and module.track_running_stats:
                raise EpisodeProtocolError(
                    f"training BatchNorm {name!r} must not accumulate running stats"
                )
            if not module.training:
                if not module.track_running_stats:
                    raise EpisodeProtocolError(
                        f"eval BatchNorm {name!r} must use Source running stats"
                    )
                if module.running_mean is None or module.running_var is None:
                    raise EpisodeProtocolError(
                        f"eval BatchNorm {name!r} is missing running statistics"
                    )

    def _managed_optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        optimizer = self.state.optimizer
        if optimizer is None:
            return ()
        return tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )

    def _assert_managed_optimizer_topology(self) -> None:
        parameters = self._managed_optimizer_parameters()
        if not parameters:
            return
        allowed = self._batchnorm_affine_parameter_ids()
        parameter_ids = {id(parameter) for parameter in parameters}
        if len(parameter_ids) != len(parameters):
            raise EpisodeProtocolError(
                "managed optimizer contains duplicate parameter bindings"
            )
        offenders = [
            name
            for name, parameter in self.adapter.model.named_parameters()
            if id(parameter) in parameter_ids and id(parameter) not in allowed
        ]
        if offenders:
            raise EpisodeProtocolError(
                "managed optimizer may contain only BatchNorm2d affine parameters: "
                + ", ".join(offenders[:10])
            )

    def _assert_method_optimizer(self, method: EpisodicMethod) -> Optimizer | None:
        optimizer = getattr(method, "optimizer", None)
        if optimizer is not None and not isinstance(optimizer, Optimizer):
            raise EpisodeProtocolError("method optimizer must be a Torch Optimizer")
        if optimizer is not self.state.optimizer:
            raise EpisodeProtocolError(
                "method optimizer must be identical to the optimizer managed by "
                "the state manager, including None"
            )
        if optimizer is not None:
            optimizer_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            trainable_ids = {
                id(parameter)
                for parameter in self.adapter.model.parameters()
                if parameter.requires_grad
            }
            if optimizer_ids != trainable_ids:
                raise EpisodeProtocolError(
                    "method optimizer parameters must exactly match the trainable "
                    "BatchNorm2d affine parameters"
                )
        elif any(
            parameter.requires_grad for parameter in self.adapter.model.parameters()
        ):
            raise EpisodeProtocolError(
                "trainable parameters require the state-manager-owned optimizer"
            )
        return optimizer

    @staticmethod
    def _method_requires_grad(method: EpisodicMethod) -> bool:
        value = getattr(method, "requires_grad", None)
        if not isinstance(value, bool):
            raise EpisodeProtocolError(
                "method must declare a boolean requires_grad policy"
            )
        return value

    @staticmethod
    def _allowed_state_changes(method: EpisodicMethod) -> frozenset[str]:
        value = getattr(method, "allowed_state_changes", None)
        if not isinstance(value, frozenset) or not all(
            isinstance(item, str) for item in value
        ):
            raise EpisodeProtocolError(
                "method must declare allowed_state_changes as frozenset[str]"
            )
        valid = {
            "model",
            "optimizer",
            "runtime",
            "topology",
            "gradients",
            "extras",
        }
        unknown = sorted(value - valid)
        if unknown:
            raise EpisodeProtocolError(
                "method declares unknown state components: " + ", ".join(unknown)
            )
        return value

    def _assert_state_change_policy(
        self,
        *,
        method: EpisodicMethod,
        fingerprint: StateFingerprint,
        stage: str,
    ) -> tuple[str, ...]:
        differences = fingerprint.differing_components(
            self.state.source_fingerprint
        )
        allowed = self._allowed_state_changes(method)
        forbidden = tuple(item for item in differences if item not in allowed)
        if forbidden:
            raise EpisodeProtocolError(
                f"method {method.name!r} changed forbidden state components "
                f"after {stage}: {', '.join(forbidden)}"
            )
        exact = getattr(method, "exact_state_changes", None)
        if exact is not None:
            if not isinstance(exact, frozenset) or not all(
                isinstance(item, str) for item in exact
            ):
                raise EpisodeProtocolError(
                    "method exact_state_changes must be frozenset[str] or None"
                )
            if frozenset(differences) != exact:
                raise EpisodeProtocolError(
                    f"method {method.name!r} must change exactly "
                    f"{sorted(exact)} after {stage}, got {list(differences)}"
                )
        return differences

    def _assert_method_bn_protocol(self, method: EpisodicMethod) -> None:
        protocol = getattr(method, "bn_protocol", None)
        if protocol is None:
            return
        if protocol != "single_image_spatial_batch_stats":
            raise EpisodeProtocolError(f"unsupported method BN protocol: {protocol!r}")
        batchnorm_count = 0
        for name, module in self.adapter.model.named_modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            batchnorm_count += 1
            if not module.training or module.track_running_stats:
                raise EpisodeProtocolError(
                    f"method {method.name!r} requires every BatchNorm2d to use "
                    f"non-accumulating batch statistics; violation at {name!r}"
                )
            if (
                module.running_mean is None
                or module.running_var is None
                or module.num_batches_tracked is None
            ):
                raise EpisodeProtocolError(
                    f"method {method.name!r} requires intact Source BN buffers at "
                    f"{name!r}"
                )
        if batchnorm_count == 0:
            raise EpisodeProtocolError(
                f"method {method.name!r} requires at least one BatchNorm2d"
            )

    @staticmethod
    def _assert_state_frozen_after_prepare(
        *,
        method: EpisodicMethod,
        prepared: StateFingerprint,
        current: StateFingerprint,
        stage: str,
    ) -> None:
        frozen = getattr(method, "state_frozen_after_prepare", False)
        if not isinstance(frozen, bool):
            raise EpisodeProtocolError(
                "method state_frozen_after_prepare must be boolean"
            )
        if frozen and current != prepared:
            differences = current.differing_components(prepared)
            raise EpisodeProtocolError(
                f"method {method.name!r} must keep state frozen after prepare; "
                f"changed after {stage}: {', '.join(differences)}"
            )

    def _checked_forward(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.adapter.forward_logits(image)
        if logits.ndim != 4 or logits.shape[:2] != (1, 1):
            raise EpisodeProtocolError(
                f"logits must have shape [1,1,H,W], got {tuple(logits.shape)}"
            )
        if logits.shape[-2:] != image.shape[-2:]:
            raise EpisodeProtocolError("logit and image spatial sizes differ")
        if not torch.isfinite(logits).all():
            raise EpisodeProtocolError("model produced NaN/Inf logits")
        return logits.detach().clone()

    def run_one_image(
        self,
        *,
        image: Tensor,
        metadata: Mapping[str, Any],
        method: EpisodicMethod,
    ) -> EpisodeResult:
        """Run one label-free episode and restore Source even after exceptions."""
        source_fingerprint = self.state.source_fingerprint
        source_hash = source_fingerprint.full_sha256
        rng_state = _capture_rng_state()
        reset_fingerprint = None

        try:
            self.state.reset_to_source()
            self.state.assert_source_state()
            self._assert_source_contract()

            if not isinstance(image, Tensor) or image.ndim != 4:
                raise ValueError("image must be a tensor with shape [1,C,H,W]")
            if image.shape[0] != 1 or image.shape[1] < 1:
                raise ValueError(
                    "episodic runner requires a batch of exactly one image"
                )
            if not torch.is_floating_point(image) or not torch.isfinite(image).all():
                raise ValueError("image must be a finite floating-point tensor")
            safe_metadata = _safe_metadata(metadata)
            method_name = getattr(method, "name", None)
            if not isinstance(method_name, str) or not method_name:
                raise TypeError("method must expose a non-empty string name")
            if not callable(getattr(method, "prepare_episode", None)) or not callable(
                getattr(method, "adapt_one_image", None)
            ):
                raise TypeError(
                    "method must implement prepare_episode() and adapt_one_image()"
                )

            canonical_image = image.detach().clone()
            input_reference = canonical_image.clone()

            logits_pre = self._checked_forward(canonical_image)
            # Source inference itself must not mutate buffers or runtime state.
            self.state.assert_source_state()
            if not torch.equal(canonical_image, input_reference):
                raise EpisodeProtocolError("Source forward modified its input tensor")

            method.prepare_episode(self.adapter)
            self._assert_adaptation_contract()
            self._assert_method_bn_protocol(method)
            method_optimizer = self._assert_method_optimizer(method)
            prepared_fingerprint = self.state.current_fingerprint()
            state_changes_after_prepare = self._assert_state_change_policy(
                method=method,
                fingerprint=prepared_fingerprint,
                stage="prepare_episode",
            )
            # Formal evaluators commonly run inference under torch.no_grad().
            # Adaptation must explicitly override that outer context.
            requires_grad = self._method_requires_grad(method)
            if requires_grad and torch.is_inference_mode_enabled():
                raise EpisodeProtocolError(
                    "episodic adaptation cannot run inside torch.inference_mode()"
                )
            grad_context = torch.enable_grad() if requires_grad else torch.no_grad()
            with grad_context:
                outcome = method.adapt_one_image(
                    adapter=self.adapter,
                    image=canonical_image.clone(),
                    logits_pre=logits_pre.clone(),
                    metadata=safe_metadata,
                )
            if not isinstance(outcome, AdaptationOutcome):
                raise TypeError("adapt_one_image() must return AdaptationOutcome")
            if outcome.optimizer_steps > 0 and method_optimizer is None:
                raise EpisodeProtocolError(
                    "optimizer steps require the state-manager-owned optimizer"
                )
            self._assert_adaptation_contract()
            self._assert_method_bn_protocol(method)
            state_after_adapt_fingerprint = self.state.current_fingerprint()
            self._assert_state_frozen_after_prepare(
                method=method,
                prepared=prepared_fingerprint,
                current=state_after_adapt_fingerprint,
                stage="adapt_one_image",
            )
            state_changes_after_adapt = self._assert_state_change_policy(
                method=method,
                fingerprint=state_after_adapt_fingerprint,
                stage="adapt_one_image",
            )
            state_after_adapt = state_after_adapt_fingerprint.full_sha256

            if outcome.decision == "no_update":
                # A method cannot claim no update while changing modes, params,
                # buffers, optimizer state, gradients, or registered extras.
                self.state.assert_source_state()
            logits_post = self._checked_forward(canonical_image)
            self._assert_adaptation_contract()
            self._assert_method_bn_protocol(method)
            state_after_post_fingerprint = self.state.current_fingerprint()
            self._assert_state_frozen_after_prepare(
                method=method,
                prepared=prepared_fingerprint,
                current=state_after_post_fingerprint,
                stage="post_forward",
            )
            state_changes_after_post = self._assert_state_change_policy(
                method=method,
                fingerprint=state_after_post_fingerprint,
                stage="post_forward",
            )
            state_after_post = state_after_post_fingerprint.full_sha256
            input_unchanged = bool(torch.equal(canonical_image, input_reference))
            if not input_unchanged:
                raise EpisodeProtocolError("episode modified the canonical input")

            pre_post_exact = bool(torch.equal(logits_pre, logits_post))
            if outcome.decision == "no_update":
                self.state.assert_source_state()
                if not pre_post_exact:
                    raise EpisodeProtocolError(
                        "no-update pre/post logits are not bit-exact"
                    )
        finally:
            try:
                reset_fingerprint = self.state.reset_to_source()
            finally:
                _restore_rng_state(rng_state)

        assert reset_fingerprint is not None
        reset_hash = reset_fingerprint.full_sha256
        if reset_hash != source_hash:
            raise EpisodeProtocolError("post-episode reset hash differs from Source")
        return EpisodeResult(
            method=method_name,
            metadata=safe_metadata,
            logits_pre=logits_pre.detach().cpu().clone(),
            logits_post=logits_post.detach().cpu().clone(),
            outcome=outcome,
            pre_post_bit_exact=pre_post_exact,
            input_unchanged=input_unchanged,
            source_fingerprint=source_fingerprint,
            state_after_prepare_fingerprint=prepared_fingerprint,
            state_after_adapt_fingerprint=state_after_adapt_fingerprint,
            state_after_post_fingerprint=state_after_post_fingerprint,
            reset_fingerprint=reset_fingerprint,
            state_changes_after_prepare=state_changes_after_prepare,
            state_changes_after_adapt=state_changes_after_adapt,
            state_changes_after_post=state_changes_after_post,
            source_state_sha256=source_hash,
            state_after_adapt_sha256=state_after_adapt,
            state_after_post_sha256=state_after_post,
            reset_state_sha256=reset_hash,
        )


__all__ = [
    "AdaptationOutcome",
    "EpisodeProtocolError",
    "EpisodeResult",
    "EpisodicMethod",
    "EpisodicRunner",
    "NoUpdateMethod",
    "SAFE_METADATA_KEYS",
]
