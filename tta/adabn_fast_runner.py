"""Fast, fail-closed execution for the statistics-only episodic AdaBN baseline.

The generic :mod:`tta.episodic_runner` deliberately computes several complete
CPU SHA-256 fingerprints per episode.  That is useful as a reference audit,
but unnecessarily expensive for AdaBN: this method is allowed to change only
module runtime flags, never a parameter, buffer, gradient, or optimiser.

This runner therefore checks every parameter and every registered buffer by
exact tensor-byte equality on its resident device.  Persistent,
non-persistent, and ``None`` buffers are all covered.  A single small boolean
vector is transferred per device; model tensors are not copied to the CPU for
each check.  Runtime values and Python object topology are checked separately.
Complete cryptographic fingerprints remain available at an explicit cadence
and are represented by ``None`` when they were not computed.

This class is intentionally AdaBN-specific.  It must not be used for TENT or
any method that updates learnable state.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn

from tta.episodic_runner import (
    EpisodeProtocolError,
    _capture_rng_state,
    _restore_rng_state,
    _safe_metadata,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager, StateFingerprint


CHECK_KIND = "exact_resident_tensor_bytes_and_runtime_values"


class AdaBNFastProtocolError(EpisodeProtocolError):
    """Raised when the specialised AdaBN execution contract is violated."""


class AdaBNFastRecoveryError(AdaBNFastProtocolError):
    """Raised when both an episode and its emergency full reset fail."""

    def __init__(self, original: BaseException, recovery: BaseException) -> None:
        self.original_error = original
        self.recovery_error = recovery
        super().__init__(
            "AdaBN fast episode failed and the complete StateManager reset also "
            f"failed: episode={original!r}; reset={recovery!r}"
        )


@dataclass(frozen=True)
class ExactTensorCheck:
    """Evidence returned only after an exact value/topology gate passes."""

    stage: str
    check_kind: str
    parameter_tensors: int
    persistent_buffer_tensors: int
    nonpersistent_buffer_tensors: int
    none_buffers: int
    batchnorm_running_mean_buffers: int
    batchnorm_running_var_buffers: int
    batchnorm_num_batches_tracked_buffers: int
    parameter_bytes_exact: bool = True
    all_registered_buffer_bytes_exact: bool = True
    buffer_none_and_persistence_topology_exact: bool = True


@dataclass(frozen=True)
class AdaBNFastEpisodeResult:
    """One completed episode with explicit fast-gate and optional SHA evidence."""

    method: Literal["adabn"]
    episode_number: int
    metadata: Mapping[str, Any]
    logits_pre: Tensor
    logits_post: Tensor
    input_unchanged: bool
    check_kind: str
    exact_checks: Mapping[str, ExactTensorCheck]
    checks: Mapping[str, bool]
    source_state_sha256: str
    full_audit_performed: bool
    full_audit_reason: Literal["forced", "cadence"] | None
    post_full_fingerprint: StateFingerprint | None
    reset_full_fingerprint: StateFingerprint | None
    post_full_state_differences: tuple[str, ...] | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exact_checks",
            MappingProxyType(dict(self.exact_checks)),
        )
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))


@dataclass(frozen=True)
class _ParameterSource:
    name: str
    object: nn.Parameter
    value: Tensor


@dataclass(frozen=True)
class _BufferSource:
    module_name: str
    name: str
    object: Tensor | None
    value: Tensor | None
    persistent: bool
    bn_role: str | None


@dataclass(frozen=True)
class _ModuleSource:
    name: str
    object: nn.Module
    qualified_type: str
    training: bool


def _qualified_type(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def full_audit_due(
    episode_number: int,
    cadence: int | None,
    *,
    force: bool = False,
) -> tuple[bool, Literal["forced", "cadence"] | None]:
    """Return the exact 1-based full-audit schedule decision.

    The cadence formula is ``episode_number % cadence == 0``.  Condition
    boundaries can be audited independently with ``force=True``.
    """

    if isinstance(episode_number, bool) or not isinstance(episode_number, int):
        raise TypeError("episode_number must be an integer")
    if episode_number <= 0:
        raise ValueError("episode_number must be positive")
    if cadence is not None:
        if isinstance(cadence, bool) or not isinstance(cadence, int):
            raise TypeError("full_audit_cadence must be an integer or None")
        if cadence <= 0:
            raise ValueError("full_audit_cadence must be positive")
    if not isinstance(force, bool):
        raise TypeError("force must be boolean")
    if force:
        return True, "forced"
    if cadence is not None and episode_number % cadence == 0:
        return True, "cadence"
    return False, None


class AdaBNFastRunner:
    """Run label-free, batch-one AdaBN episodes with exact resident-state gates.

    The runner owns the model for its lifetime.  Any failed episode permanently
    aborts the runner even if the emergency full reset succeeds; callers must
    discard it and fail the enclosing condition rather than continue with
    potentially incomplete evidence.
    """

    def __init__(
        self,
        adapter: IRSTDModelAdapter,
        state_manager: EpisodicStateManager,
        *,
        full_audit_cadence: int | None = None,
    ) -> None:
        if not isinstance(adapter, IRSTDModelAdapter):
            raise TypeError("adapter must be an IRSTDModelAdapter")
        if not isinstance(state_manager, EpisodicStateManager):
            raise TypeError("state_manager must be an EpisodicStateManager")
        if adapter.model is not state_manager.model:
            raise ValueError("adapter and state manager must own the same model")
        if state_manager.optimizer is not None:
            raise AdaBNFastProtocolError("AdaBN fast runner forbids an optimizer")
        if state_manager.extra_state_names:
            raise AdaBNFastProtocolError(
                "AdaBN fast runner forbids method-specific extra state: "
                + ", ".join(state_manager.extra_state_names)
            )
        # Reuse the schedule validator without consuming an episode.
        if full_audit_cadence is not None:
            full_audit_due(1, full_audit_cadence)

        self.adapter = adapter
        self.state = state_manager
        self.full_audit_cadence = full_audit_cadence
        self._completed_episodes = 0
        self._aborted = False

        # Seal against a stale or incorrectly configured caller before taking
        # the fast runner's private immutable, device-resident value copies.
        self.state.assert_source_state()
        self._modules = tuple(
            _ModuleSource(
                name=name,
                object=module,
                qualified_type=_qualified_type(module),
                training=bool(module.training),
            )
            for name, module in self.adapter.model.named_modules()
        )
        self._parameters = tuple(
            _ParameterSource(
                name=name,
                object=parameter,
                value=parameter.detach().clone(),
            )
            for name, parameter in self.adapter.model.named_parameters()
        )
        self._source_requires_grad = tuple(
            (entry.name, bool(entry.object.requires_grad))
            for entry in self._parameters
        )
        self._buffers = self._capture_buffers()
        self._source_nonpersistent = tuple(
            (
                entry.name,
                tuple(sorted(entry.object._non_persistent_buffers_set)),
            )
            for entry in self._modules
        )
        # ``named_modules`` and ``named_parameters`` remove duplicate objects by
        # default.  Seal the direct registration maps as well, so adding an
        # alias to an existing module/parameter cannot evade the topology gate.
        self._source_direct_modules = tuple(
            (
                entry.name,
                tuple(
                    (name, child, None if child is None else _qualified_type(child))
                    for name, child in sorted(entry.object._modules.items())
                ),
            )
            for entry in self._modules
        )
        self._source_direct_parameters = tuple(
            (
                entry.name,
                tuple(
                    (name, parameter)
                    for name, parameter in sorted(entry.object._parameters.items())
                ),
            )
            for entry in self._modules
        )
        self._source_bn_runtime = tuple(
            (
                entry.name,
                bool(entry.object.track_running_stats),
                entry.object.momentum,
                entry.object.eps,
            )
            for entry in self._modules
            if isinstance(entry.object, nn.BatchNorm2d)
        )

        if not self._source_bn_runtime:
            raise AdaBNFastProtocolError(
                "AdaBN fast runner requires at least one BatchNorm2d"
            )
        if any(flag for _name, flag in self._source_requires_grad):
            raise AdaBNFastProtocolError(
                "AdaBN Source snapshot requires every parameter to be frozen"
            )
        if any(parameter.object.grad is not None for parameter in self._parameters):
            raise AdaBNFastProtocolError(
                "AdaBN Source snapshot requires every gradient to be None"
            )
        self._assert_canonical_source_runtime("construction")
        self._assert_source_runtime("construction")
        self._assert_exact_values("construction")

    @property
    def completed_episodes(self) -> int:
        return self._completed_episodes

    @property
    def aborted(self) -> bool:
        return self._aborted

    def _capture_buffers(self) -> tuple[_BufferSource, ...]:
        buffers: list[_BufferSource] = []
        for module_name, module in self.adapter.model.named_modules():
            nonpersistent = module._non_persistent_buffers_set
            for name, value in sorted(module._buffers.items()):
                role = None
                if isinstance(module, nn.BatchNorm2d) and name in {
                    "running_mean",
                    "running_var",
                    "num_batches_tracked",
                }:
                    role = name
                buffers.append(
                    _BufferSource(
                        module_name=module_name,
                        name=name,
                        object=value,
                        value=None if value is None else value.detach().clone(),
                        persistent=name not in nonpersistent,
                        bn_role=role,
                    )
                )
        return tuple(buffers)

    def _assert_topology(self, stage: str) -> None:
        current_modules = tuple(self.adapter.model.named_modules())
        if tuple(name for name, _module in current_modules) != tuple(
            entry.name for entry in self._modules
        ):
            raise AdaBNFastProtocolError(
                f"module topology changed after {stage}"
            )
        for (name, module), source in zip(
            current_modules, self._modules, strict=True
        ):
            if module is not source.object or _qualified_type(module) != source.qualified_type:
                raise AdaBNFastProtocolError(
                    f"module object/type topology changed at {name!r} after {stage}"
                )

        modules_by_name = dict(current_modules)
        for module_name, expected_children in self._source_direct_modules:
            current_children = tuple(
                (name, child, None if child is None else _qualified_type(child))
                for name, child in sorted(
                    modules_by_name[module_name]._modules.items()
                )
            )
            if len(current_children) != len(expected_children) or any(
                current_name != expected_name
                or current_child is not expected_child
                or current_type != expected_type
                for (
                    current_name,
                    current_child,
                    current_type,
                ), (
                    expected_name,
                    expected_child,
                    expected_type,
                ) in zip(current_children, expected_children, strict=False)
            ):
                raise AdaBNFastProtocolError(
                    f"direct child-module topology changed at {module_name!r} "
                    f"after {stage}"
                )

        for module_name, expected_parameters in self._source_direct_parameters:
            current_parameters_direct = tuple(
                (name, parameter)
                for name, parameter in sorted(
                    modules_by_name[module_name]._parameters.items()
                )
            )
            if len(current_parameters_direct) != len(expected_parameters) or any(
                current_name != expected_name
                or current_parameter is not expected_parameter
                for (
                    current_name,
                    current_parameter,
                ), (
                    expected_name,
                    expected_parameter,
                ) in zip(
                    current_parameters_direct,
                    expected_parameters,
                    strict=False,
                )
            ):
                raise AdaBNFastProtocolError(
                    f"direct parameter topology changed at {module_name!r} "
                    f"after {stage}"
                )

        current_parameters = tuple(self.adapter.model.named_parameters())
        if tuple(name for name, _parameter in current_parameters) != tuple(
            entry.name for entry in self._parameters
        ):
            raise AdaBNFastProtocolError(
                f"parameter topology changed after {stage}"
            )
        for (name, parameter), source in zip(
            current_parameters, self._parameters, strict=True
        ):
            if parameter is not source.object:
                raise AdaBNFastProtocolError(
                    f"parameter object topology changed at {name!r} after {stage}"
                )

        modules = modules_by_name
        source_nonpersistent = dict(self._source_nonpersistent)
        for module_name, module in current_modules:
            if tuple(sorted(module._non_persistent_buffers_set)) != source_nonpersistent[
                module_name
            ]:
                raise AdaBNFastProtocolError(
                    f"buffer persistence topology changed at {module_name!r} "
                    f"after {stage}"
                )
        source_by_module: dict[str, list[_BufferSource]] = defaultdict(list)
        for source in self._buffers:
            source_by_module[source.module_name].append(source)
        for module_name, module in modules.items():
            expected = tuple(entry.name for entry in source_by_module[module_name])
            if tuple(sorted(module._buffers)) != expected:
                raise AdaBNFastProtocolError(
                    f"registered buffer topology changed at {module_name!r} "
                    f"after {stage}"
                )
            for source in source_by_module[module_name]:
                if module._buffers[source.name] is not source.object:
                    raise AdaBNFastProtocolError(
                        f"buffer object topology changed at "
                        f"{module_name}.{source.name} after {stage}"
                    )

    def _assert_frozen_parameters_and_gradients(self, stage: str) -> None:
        current_requires_grad = tuple(
            (entry.name, bool(entry.object.requires_grad))
            for entry in self._parameters
        )
        if current_requires_grad != self._source_requires_grad:
            raise AdaBNFastProtocolError(
                f"parameter requires_grad changed after {stage}"
            )
        offenders = [
            entry.name for entry in self._parameters if entry.object.grad is not None
        ]
        if offenders:
            raise AdaBNFastProtocolError(
                "AdaBN created or retained parameter gradients after "
                f"{stage}: {', '.join(offenders[:10])}"
            )
        if self.state.optimizer is not None:
            raise AdaBNFastProtocolError(
                f"AdaBN acquired an optimizer after {stage}"
            )

    @staticmethod
    def _byte_flag(current: Tensor, source: Tensor, label: str) -> Tensor:
        current_metadata = (
            tuple(current.shape),
            current.dtype,
            current.device,
            current.layout,
        )
        source_metadata = (
            tuple(source.shape),
            source.dtype,
            source.device,
            source.layout,
        )
        if current_metadata != source_metadata:
            raise AdaBNFastProtocolError(
                f"tensor metadata changed at {label}: "
                f"current={current_metadata}, source={source_metadata}"
            )
        if current.layout != torch.strided:
            raise AdaBNFastProtocolError(
                f"fast exact-byte gate does not support non-strided tensor {label}"
            )
        current_bytes = current.detach().contiguous().reshape(-1).view(torch.uint8)
        source_bytes = source.detach().contiguous().reshape(-1).view(torch.uint8)
        return torch.eq(current_bytes, source_bytes).all()

    @staticmethod
    def _resolve_byte_flags(
        flags: list[tuple[str, Tensor]], stage: str
    ) -> None:
        by_device: dict[torch.device, list[tuple[str, Tensor]]] = defaultdict(list)
        for label, flag in flags:
            by_device[flag.device].append((label, flag))
        mismatches: list[str] = []
        for entries in by_device.values():
            resolved = torch.stack([flag for _label, flag in entries]).detach().cpu()
            for (label, _flag), exact in zip(
                entries, resolved.tolist(), strict=True
            ):
                if not exact:
                    mismatches.append(label)
        if mismatches:
            raise AdaBNFastProtocolError(
                "exact tensor-byte mismatch after "
                f"{stage}: {', '.join(mismatches[:10])}"
            )

    def _assert_exact_values(self, stage: str) -> ExactTensorCheck:
        self._assert_topology(stage)
        self._assert_frozen_parameters_and_gradients(stage)
        flags: list[tuple[str, Tensor]] = []
        for source in self._parameters:
            flags.append(
                (
                    f"parameter:{source.name}",
                    self._byte_flag(source.object, source.value, source.name),
                )
            )

        modules = dict(self.adapter.model.named_modules())
        persistent = 0
        nonpersistent = 0
        none_buffers = 0
        bn_counts = {
            "running_mean": 0,
            "running_var": 0,
            "num_batches_tracked": 0,
        }
        for source in self._buffers:
            current = modules[source.module_name]._buffers[source.name]
            if source.value is None:
                none_buffers += 1
                if current is not None:
                    raise AdaBNFastProtocolError(
                        "None buffer gained a tensor after "
                        f"{stage}: {source.module_name}.{source.name}"
                    )
                continue
            if current is None:
                raise AdaBNFastProtocolError(
                    "tensor buffer became None after "
                    f"{stage}: {source.module_name}.{source.name}"
                )
            if source.persistent:
                persistent += 1
            else:
                nonpersistent += 1
            if source.bn_role is not None:
                bn_counts[source.bn_role] += 1
            label = f"buffer:{source.module_name}.{source.name}"
            flags.append((label, self._byte_flag(current, source.value, label)))

        self._resolve_byte_flags(flags, stage)
        return ExactTensorCheck(
            stage=stage,
            check_kind=CHECK_KIND,
            parameter_tensors=len(self._parameters),
            persistent_buffer_tensors=persistent,
            nonpersistent_buffer_tensors=nonpersistent,
            none_buffers=none_buffers,
            batchnorm_running_mean_buffers=bn_counts["running_mean"],
            batchnorm_running_var_buffers=bn_counts["running_var"],
            batchnorm_num_batches_tracked_buffers=bn_counts[
                "num_batches_tracked"
            ],
        )

    def _assert_source_runtime(self, stage: str) -> None:
        self._assert_topology(stage)
        self._assert_frozen_parameters_and_gradients(stage)
        actual_training = tuple(
            (entry.name, bool(entry.object.training)) for entry in self._modules
        )
        expected_training = tuple(
            (entry.name, entry.training) for entry in self._modules
        )
        if actual_training != expected_training:
            raise AdaBNFastProtocolError(
                f"module training runtime differs from Source after {stage}"
            )
        actual_bn = tuple(
            (
                entry.name,
                bool(entry.object.track_running_stats),
                entry.object.momentum,
                entry.object.eps,
            )
            for entry in self._modules
            if isinstance(entry.object, nn.BatchNorm2d)
        )
        if actual_bn != self._source_bn_runtime:
            raise AdaBNFastProtocolError(
                f"BatchNorm runtime differs from Source after {stage}"
            )

    def _assert_canonical_source_runtime(self, stage: str) -> None:
        """Require the project-wide deterministic Source inference contract."""

        training_modules = [
            entry.name or "<root>"
            for entry in self._modules
            if entry.object.training
        ]
        if training_modules:
            raise AdaBNFastProtocolError(
                "AdaBN Source snapshot requires every module in eval mode after "
                f"{stage}: {', '.join(training_modules[:10])}"
            )
        for entry in self._modules:
            module = entry.object
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if (
                not module.track_running_stats
                or module.running_mean is None
                or module.running_var is None
                or module.num_batches_tracked is None
            ):
                raise AdaBNFastProtocolError(
                    "AdaBN Source snapshot requires tracked BatchNorm statistics "
                    f"at {entry.name!r} after {stage}"
                )

    def _assert_adabn_runtime(self, stage: str) -> None:
        self._assert_topology(stage)
        self._assert_frozen_parameters_and_gradients(stage)
        source_bn = {name: (momentum, eps) for name, _track, momentum, eps in self._source_bn_runtime}
        batchnorm_count = 0
        for entry in self._modules:
            module = entry.object
            if isinstance(module, nn.BatchNorm2d):
                batchnorm_count += 1
                if not module.training or module.track_running_stats:
                    raise AdaBNFastProtocolError(
                        f"BatchNorm {entry.name!r} is outside the single-image "
                        f"non-accumulating AdaBN mode after {stage}"
                    )
                momentum, eps = source_bn[entry.name]
                if module.momentum != momentum or module.eps != eps:
                    raise AdaBNFastProtocolError(
                        f"BatchNorm eps/momentum changed at {entry.name!r} "
                        f"after {stage}"
                    )
                if (
                    module.running_mean is None
                    or module.running_var is None
                    or module.num_batches_tracked is None
                ):
                    raise AdaBNFastProtocolError(
                        f"BatchNorm Source buffers disappeared at {entry.name!r} "
                        f"after {stage}"
                    )
            elif module.training:
                raise AdaBNFastProtocolError(
                    f"non-BatchNorm module {entry.name!r} entered training mode "
                    f"after {stage}"
                )
        if batchnorm_count != len(self._source_bn_runtime):
            raise AdaBNFastProtocolError("BatchNorm topology count changed")

    def _restore_source_runtime_only(self) -> None:
        # Match StateManager's source-mode restoration order: parent modules
        # first, then their children, so child-specific values win.
        for entry in self._modules:
            entry.object.train(entry.training)
        for name, requires_grad in self._source_requires_grad:
            parameter = dict(self.adapter.model.named_parameters())[name]
            parameter.requires_grad_(requires_grad)
        source_bn = {
            name: (track, momentum, eps)
            for name, track, momentum, eps in self._source_bn_runtime
        }
        for entry in self._modules:
            if not isinstance(entry.object, nn.BatchNorm2d):
                continue
            track, momentum, eps = source_bn[entry.name]
            entry.object.track_running_stats = track
            entry.object.momentum = momentum
            entry.object.eps = eps

    def _checked_forward(self, image: Tensor) -> Tensor:
        with torch.no_grad():
            logits = self.adapter.forward_logits(image)
        if logits.ndim != 4 or logits.shape[:2] != (1, 1):
            raise AdaBNFastProtocolError(
                f"logits must have shape [1,1,H,W], got {tuple(logits.shape)}"
            )
        if logits.shape[-2:] != image.shape[-2:]:
            raise AdaBNFastProtocolError("logit and image spatial sizes differ")
        if not torch.isfinite(logits).all():
            raise AdaBNFastProtocolError("model produced NaN/Inf logits")
        return logits.detach().clone()

    def _emergency_full_reset(self, original: BaseException) -> None:
        try:
            reset = self.state.reset_to_source()
            if reset != self.state.source_fingerprint:
                raise AdaBNFastProtocolError(
                    "emergency reset fingerprint differs from Source"
                )
        except BaseException as recovery:
            raise AdaBNFastRecoveryError(original, recovery) from original

    def _run_body(
        self,
        *,
        image: Tensor,
        metadata: Mapping[str, Any],
        audit_due: bool,
    ) -> dict[str, Any]:
        if not isinstance(image, Tensor) or image.ndim != 4:
            raise ValueError("image must be a tensor with shape [1,C,H,W]")
        if image.shape[0] != 1 or image.shape[1] < 1:
            raise ValueError("AdaBN fast runner requires a batch of one image")
        if not torch.is_floating_point(image) or not torch.isfinite(image).all():
            raise ValueError("image must be a finite floating-point tensor")
        try:
            safe_metadata = _safe_metadata(metadata)
        except EpisodeProtocolError as error:
            raise AdaBNFastProtocolError(str(error)) from error

        exact_checks: dict[str, ExactTensorCheck] = {}
        self._assert_source_runtime("episode_start")
        exact_checks["episode_start"] = self._assert_exact_values("episode_start")

        canonical_image = image.detach().clone()
        input_reference = canonical_image.clone()
        logits_pre = self._checked_forward(canonical_image)
        self._assert_source_runtime("source_forward")
        exact_checks["source_forward"] = self._assert_exact_values(
            "source_forward"
        )

        self.adapter.set_adabn_mode()
        self._assert_adabn_runtime("prepare_adabn")
        exact_checks["prepare_adabn"] = self._assert_exact_values(
            "prepare_adabn"
        )

        logits_post = self._checked_forward(canonical_image)
        self._assert_adabn_runtime("adabn_forward")
        exact_checks["adabn_forward"] = self._assert_exact_values(
            "adabn_forward"
        )
        input_unchanged = bool(torch.equal(canonical_image, input_reference))
        if not input_unchanged:
            raise AdaBNFastProtocolError("episode modified the canonical input")

        post_fingerprint = None
        post_differences = None
        if audit_due:
            post_fingerprint = self.state.current_fingerprint()
            post_differences = post_fingerprint.differing_components(
                self.state.source_fingerprint
            )
            if post_differences != ("runtime",):
                raise AdaBNFastProtocolError(
                    "full post audit expected only runtime to differ from Source, "
                    f"got {post_differences}"
                )

        return {
            "metadata": safe_metadata,
            "logits_pre": logits_pre,
            "logits_post": logits_post,
            "input_unchanged": input_unchanged,
            "exact_checks": exact_checks,
            "post_fingerprint": post_fingerprint,
            "post_differences": post_differences,
        }

    def run_one_image(
        self,
        *,
        image: Tensor,
        metadata: Mapping[str, Any],
        force_full_audit: bool = False,
    ) -> AdaBNFastEpisodeResult:
        """Run one episode, restore Source runtime, and return explicit evidence."""

        if self._aborted:
            raise AdaBNFastProtocolError(
                "AdaBN fast runner was aborted by a previous failed episode"
            )
        if not isinstance(force_full_audit, bool):
            raise TypeError("force_full_audit must be boolean")
        episode_number = self._completed_episodes + 1
        audit_due, audit_reason = full_audit_due(
            episode_number,
            self.full_audit_cadence,
            force=force_full_audit,
        )
        rng_state = _capture_rng_state()
        body: dict[str, Any] | None = None
        reset_exact: ExactTensorCheck | None = None
        reset_fingerprint: StateFingerprint | None = None
        candidate: AdaBNFastEpisodeResult | None = None

        try:
            try:
                body = self._run_body(
                    image=image,
                    metadata=metadata,
                    audit_due=audit_due,
                )
            except BaseException as original:
                self._aborted = True
                self._emergency_full_reset(original)
                raise

            # Success path: restore only the runtime components AdaBN is
            # allowed to change, then verify every Source value again.  Any
            # restoration failure falls back to the complete manager reset and
            # aborts the condition.
            try:
                self._restore_source_runtime_only()
                self._assert_source_runtime("runtime_reset")
                reset_exact = self._assert_exact_values("runtime_reset")
                if audit_due:
                    reset_fingerprint = self.state.assert_source_state()
                    if reset_fingerprint != self.state.source_fingerprint:
                        raise AdaBNFastProtocolError(
                            "full reset audit differs from Source"
                        )
                assert body is not None
                exact_checks = dict(body["exact_checks"])
                exact_checks["runtime_reset"] = reset_exact
                checks = {
                    "safe_label_free_metadata": True,
                    "batch_size_one": True,
                    "source_pre_no_grad": not body["logits_pre"].requires_grad,
                    "adabn_post_no_grad": not body["logits_post"].requires_grad,
                    "optimizer_absent": self.state.optimizer is None,
                    "all_parameters_exact_every_gate": True,
                    "all_persistent_nonpersistent_none_buffers_exact_every_gate": True,
                    "all_bn_running_buffers_exact_every_gate": True,
                    "parameter_and_module_topology_exact_every_gate": True,
                    "gradients_absent_every_gate": True,
                    "requires_grad_source_exact_every_gate": True,
                    "non_batchnorm_modules_eval_during_adabn": True,
                    "source_runtime_restored_per_image": True,
                    "input_unchanged": bool(body["input_unchanged"]),
                    "rng_restored": True,
                }
                # Build the complete return value inside the fail-closed region.
                # In particular, a device-to-CPU copy failure must abort the
                # runner instead of leaving a reusable, half-completed episode.
                candidate = AdaBNFastEpisodeResult(
                    method="adabn",
                    episode_number=episode_number,
                    metadata=body["metadata"],
                    logits_pre=body["logits_pre"].detach().cpu().clone(),
                    logits_post=body["logits_post"].detach().cpu().clone(),
                    input_unchanged=bool(body["input_unchanged"]),
                    check_kind=CHECK_KIND,
                    exact_checks=exact_checks,
                    checks=checks,
                    source_state_sha256=self.state.source_fingerprint.full_sha256,
                    full_audit_performed=audit_due,
                    full_audit_reason=audit_reason,
                    post_full_fingerprint=body["post_fingerprint"],
                    reset_full_fingerprint=reset_fingerprint,
                    post_full_state_differences=body["post_differences"],
                )
            except BaseException as original:
                self._aborted = True
                self._emergency_full_reset(original)
                raise
        finally:
            try:
                _restore_rng_state(rng_state)
            except BaseException as original:
                self._aborted = True
                self._emergency_full_reset(original)
                raise

        assert candidate is not None
        self._completed_episodes = episode_number
        return candidate


__all__ = [
    "AdaBNFastEpisodeResult",
    "AdaBNFastProtocolError",
    "AdaBNFastRecoveryError",
    "AdaBNFastRunner",
    "CHECK_KIND",
    "ExactTensorCheck",
    "full_audit_due",
]
