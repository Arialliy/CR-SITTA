"""Fast, fail-closed execution for one-step episodic Binary TENT.

The generic Binary TENT runner cryptographically fingerprints the complete
model and optimiser several times per image.  That is the reference audit
path, but it is too expensive for the 49,920-episode source calibration grid.
This specialised runner replaces the per-episode SHA work with exact,
resident-device tensor comparisons and reserves complete StateManager SHA-256
audits for an explicit cadence or a forced condition boundary.

The fast path is deliberately narrow.  A Source optimiser must be empty and
only BatchNorm2d affine parameters plus the optimiser's temporary moment state
may change.  Every other parameter, every registered buffer (persistent,
non-persistent, and ``None``), runtime mode, gradient, input, and RNG stream is
guarded.  A failed episode permanently aborts the runner and invokes the full
StateManager reset as an emergency recovery path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from tta.adabn_fast_runner import full_audit_due
from tta.binary_tent import (
    BN_PROTOCOL_BATCH_STATS,
    BN_PROTOCOL_SOURCE_STATS,
    BinaryTentMethod,
    BinaryTentOutcome,
    BinaryTentProtocolError,
    binary_entropy_map,
)
from tta.binary_tent_runner import (
    ENTROPY_REDUCTION_ABS_TOL,
    ENTROPY_REDUCTION_REL_TOL,
)
from tta.episodic_runner import (
    EpisodeProtocolError,
    _capture_rng_state,
    _restore_rng_state,
    _safe_metadata,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager, StateFingerprint


CHECK_KIND = "exact_resident_forbidden_state_and_fast_tent_reset"


class BinaryTentFastProtocolError(BinaryTentProtocolError):
    """Raised when the specialised fast execution contract is violated."""


class BinaryTentFastRecoveryError(BinaryTentFastProtocolError):
    """Raised when an episode and its emergency complete reset both fail."""

    def __init__(self, original: BaseException, recovery: BaseException) -> None:
        self.original_error = original
        self.recovery_error = recovery
        super().__init__(
            "Binary TENT fast episode failed and the complete StateManager "
            f"reset also failed: episode={original!r}; reset={recovery!r}"
        )


@dataclass(frozen=True)
class ResidentStateCheck:
    """Evidence emitted after one exact topology/value gate passes."""

    stage: str
    check_kind: str
    non_adaptable_parameter_tensors: int
    adaptable_parameter_tensors: int
    persistent_buffer_tensors: int
    nonpersistent_buffer_tensors: int
    none_buffers: int
    batchnorm_running_mean_buffers: int
    batchnorm_running_var_buffers: int
    batchnorm_num_batches_tracked_buffers: int
    non_adaptable_parameter_bytes_exact: bool = True
    all_registered_buffer_bytes_exact: bool = True
    buffer_none_and_persistence_topology_exact: bool = True
    parameter_module_and_optimizer_topology_exact: bool = True


@dataclass(frozen=True)
class BinaryTentFastEpisodeResult:
    """Three predictions plus explicit fast-gate and optional SHA evidence."""

    method: Literal["binary_episodic_tent"]
    episode_number: int
    metadata: Mapping[str, Any]
    logits_source_pre: Tensor
    logits_tent_pre: Tensor
    logits_tent_post: Tensor
    entropy_tent_pre: float
    entropy_tent_post: float
    source_tent_pre_bit_exact: bool
    tent_pre_post_bit_exact: bool
    input_unchanged: bool
    check_kind: str
    resident_checks: Mapping[str, ResidentStateCheck]
    checks: Mapping[str, bool]
    diagnostics: Mapping[str, Any]
    source_state_sha256: str
    full_audit_performed: bool
    full_audit_reason: Literal["forced", "cadence"] | None
    adapt_full_fingerprint: StateFingerprint | None
    post_full_fingerprint: StateFingerprint | None
    reset_full_fingerprint: StateFingerprint | None
    adapt_full_state_differences: tuple[str, ...] | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )
        object.__setattr__(
            self, "resident_checks", MappingProxyType(dict(self.resident_checks))
        )
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))
        object.__setattr__(
            self, "diagnostics", MappingProxyType(deepcopy(dict(self.diagnostics)))
        )


@dataclass(frozen=True)
class _ModuleSource:
    name: str
    object: nn.Module
    qualified_type: str
    training: bool


@dataclass(frozen=True)
class _ParameterSource:
    name: str
    object: nn.Parameter
    value: Tensor
    adaptable: bool
    requires_grad: bool


@dataclass(frozen=True)
class _BufferSource:
    module_name: str
    name: str
    object: Tensor | None
    value: Tensor | None
    persistent: bool
    bn_role: str | None


@dataclass(frozen=True)
class _OptimizerTensor:
    parameter_name: str
    state_name: str
    value: Tensor


@dataclass(frozen=True)
class _OptimizerScalar:
    parameter_name: str
    state_name: str
    value: Any


@dataclass(frozen=True)
class _OptimizerStateSnapshot:
    tensors: tuple[_OptimizerTensor, ...]
    scalars: tuple[_OptimizerScalar, ...]
    parameter_names_with_state: tuple[str, ...]


def _qualified_type(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _cpu_tensor_sha256(value: Tensor) -> str:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(raw.tobytes()).hexdigest()


def _option_equal(current: Any, expected: Any) -> bool:
    """Exact recursive comparison for fixed optimiser configuration values."""

    if type(current) is not type(expected):
        return False
    if isinstance(current, Tensor):
        return bool(torch.equal(current, expected))
    if isinstance(current, Mapping):
        return tuple(current) == tuple(expected) and all(
            _option_equal(current[key], expected[key]) for key in current
        )
    if isinstance(current, (tuple, list)):
        return len(current) == len(expected) and all(
            _option_equal(left, right)
            for left, right in zip(current, expected, strict=True)
        )
    return bool(current == expected)


class BinaryTentFastRunner:
    """Run one-step Binary TENT episodes without per-image complete SHA copies.

    ``method`` is bound for the runner lifetime.  The optimiser must have an
    empty Source state, which is the canonical construction used by
    :class:`BinaryTentMethod`; this permits exact restoration via clearing its
    temporary state instead of loading a deep-copied state dict every image.
    """

    def __init__(
        self,
        adapter: IRSTDModelAdapter,
        state_manager: EpisodicStateManager,
        method: BinaryTentMethod,
        *,
        full_audit_cadence: int | None = None,
    ) -> None:
        if not isinstance(adapter, IRSTDModelAdapter):
            raise TypeError("adapter must be an IRSTDModelAdapter")
        if not isinstance(state_manager, EpisodicStateManager):
            raise TypeError("state_manager must be an EpisodicStateManager")
        if not isinstance(method, BinaryTentMethod):
            raise TypeError("method must be a BinaryTentMethod")
        if adapter.model is not state_manager.model:
            raise ValueError("adapter and state manager must own the same model")
        if method.optimizer is not state_manager.optimizer:
            raise BinaryTentFastProtocolError(
                "method optimizer must be the StateManager-owned optimizer"
            )
        if state_manager.optimizer is None:
            raise BinaryTentFastProtocolError(
                "Binary TENT fast runner requires an optimizer"
            )
        if state_manager.extra_state_names:
            raise BinaryTentFastProtocolError(
                "Binary TENT fast runner forbids method-specific extra state: "
                + ", ".join(state_manager.extra_state_names)
            )
        if full_audit_cadence is not None:
            full_audit_due(1, full_audit_cadence)

        self.adapter = adapter
        self.state = state_manager
        self.method = method
        self.optimizer = method.optimizer
        self.full_audit_cadence = full_audit_cadence
        self._completed_episodes = 0
        self._aborted = False

        # Seal the complete Source once.  All private tensor copies stay on the
        # resident device; no model tensor is copied to CPU by a fast gate.
        self.state.assert_source_state()
        adaptable, adaptable_names = self.adapter.collect_adaptable_params()
        self._adaptable_names = tuple(adaptable_names)
        self._adaptable_objects = tuple(adaptable)
        if self._adaptable_names != tuple(self.method.parameter_names):
            raise BinaryTentFastProtocolError(
                "method parameter names differ from adapter BN affine order"
            )
        if tuple(map(id, self._optimizer_parameters())) != tuple(
            map(id, self._adaptable_objects)
        ):
            raise BinaryTentFastProtocolError(
                "optimizer parameters differ from adapter BN affine order"
            )

        adaptable_ids = {id(parameter) for parameter in self._adaptable_objects}
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
                adaptable=id(parameter) in adaptable_ids,
                requires_grad=bool(parameter.requires_grad),
            )
            for name, parameter in self.adapter.model.named_parameters()
        )
        self._buffers = self._capture_buffers()
        self._source_nonpersistent = tuple(
            (
                entry.name,
                tuple(sorted(entry.object._non_persistent_buffers_set)),
            )
            for entry in self._modules
        )
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
            raise BinaryTentFastProtocolError(
                "Binary TENT fast runner requires at least one BatchNorm2d"
            )

        # Optimizer configuration is immutable; only ``optimizer.state`` may
        # be populated temporarily by the one step.
        self._optimizer_object = self.optimizer
        self._optimizer_type = _qualified_type(self.optimizer)
        self._optimizer_param_groups_object = self.optimizer.param_groups
        self._optimizer_group_objects = tuple(self.optimizer.param_groups)
        self._optimizer_group_parameters = tuple(
            tuple(group["params"]) for group in self.optimizer.param_groups
        )
        self._optimizer_group_options = tuple(
            deepcopy({key: value for key, value in group.items() if key != "params"})
            for group in self.optimizer.param_groups
        )
        self._optimizer_defaults = deepcopy(self.optimizer.defaults)
        self._optimizer_state_object = self.optimizer.state
        if self.optimizer.state:
            raise BinaryTentFastProtocolError(
                "fast restoration requires an empty Source optimizer state"
            )

        self._assert_canonical_source("construction")
        self._assert_optimizer_source("construction")
        self._assert_resident_state("construction", include_adaptable=True)

    @property
    def completed_episodes(self) -> int:
        return self._completed_episodes

    @property
    def aborted(self) -> bool:
        return self._aborted

    def _optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )

    def _capture_buffers(self) -> tuple[_BufferSource, ...]:
        values: list[_BufferSource] = []
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
                values.append(
                    _BufferSource(
                        module_name=module_name,
                        name=name,
                        object=value,
                        value=None if value is None else value.detach().clone(),
                        persistent=name not in nonpersistent,
                        bn_role=role,
                    )
                )
        return tuple(values)

    def _assert_topology(self, stage: str) -> None:
        current_modules = tuple(self.adapter.model.named_modules())
        if tuple(name for name, _module in current_modules) != tuple(
            entry.name for entry in self._modules
        ):
            raise BinaryTentFastProtocolError(
                f"module topology changed after {stage}"
            )
        modules_by_name = dict(current_modules)
        for (name, module), source in zip(
            current_modules, self._modules, strict=True
        ):
            if module is not source.object or _qualified_type(module) != source.qualified_type:
                raise BinaryTentFastProtocolError(
                    f"module object/type topology changed at {name!r} after {stage}"
                )

        for module_name, expected_children in self._source_direct_modules:
            current_children = tuple(
                (name, child, None if child is None else _qualified_type(child))
                for name, child in sorted(modules_by_name[module_name]._modules.items())
            )
            if len(current_children) != len(expected_children) or any(
                current_name != expected_name
                or current_child is not expected_child
                or current_type != expected_type
                for (current_name, current_child, current_type), (
                    expected_name,
                    expected_child,
                    expected_type,
                ) in zip(current_children, expected_children, strict=False)
            ):
                raise BinaryTentFastProtocolError(
                    f"direct child-module topology changed at {module_name!r} "
                    f"after {stage}"
                )

        for module_name, expected_parameters in self._source_direct_parameters:
            current_parameters = tuple(
                (name, parameter)
                for name, parameter in sorted(
                    modules_by_name[module_name]._parameters.items()
                )
            )
            if len(current_parameters) != len(expected_parameters) or any(
                current_name != expected_name
                or current_parameter is not expected_parameter
                for (current_name, current_parameter), (
                    expected_name,
                    expected_parameter,
                ) in zip(current_parameters, expected_parameters, strict=False)
            ):
                raise BinaryTentFastProtocolError(
                    f"direct parameter topology changed at {module_name!r} "
                    f"after {stage}"
                )

        current_parameters = tuple(self.adapter.model.named_parameters())
        if tuple(name for name, _parameter in current_parameters) != tuple(
            entry.name for entry in self._parameters
        ):
            raise BinaryTentFastProtocolError(
                f"parameter topology changed after {stage}"
            )
        for (name, parameter), source in zip(
            current_parameters, self._parameters, strict=True
        ):
            if parameter is not source.object:
                raise BinaryTentFastProtocolError(
                    f"parameter object topology changed at {name!r} after {stage}"
                )

        source_nonpersistent = dict(self._source_nonpersistent)
        source_buffers: dict[str, list[_BufferSource]] = defaultdict(list)
        for source in self._buffers:
            source_buffers[source.module_name].append(source)
        for module_name, module in current_modules:
            if tuple(sorted(module._non_persistent_buffers_set)) != source_nonpersistent[
                module_name
            ]:
                raise BinaryTentFastProtocolError(
                    f"buffer persistence topology changed at {module_name!r} "
                    f"after {stage}"
                )
            expected = tuple(entry.name for entry in source_buffers[module_name])
            if tuple(sorted(module._buffers)) != expected:
                raise BinaryTentFastProtocolError(
                    f"registered buffer topology changed at {module_name!r} "
                    f"after {stage}"
                )
            for source in source_buffers[module_name]:
                if module._buffers[source.name] is not source.object:
                    raise BinaryTentFastProtocolError(
                        "buffer object topology changed at "
                        f"{module_name}.{source.name} after {stage}"
                    )

        self._assert_optimizer_configuration(stage)

    def _assert_optimizer_configuration(self, stage: str) -> None:
        if self.state.optimizer is not self._optimizer_object:
            raise BinaryTentFastProtocolError(
                f"managed optimizer object changed after {stage}"
            )
        if _qualified_type(self.optimizer) != self._optimizer_type:
            raise BinaryTentFastProtocolError(
                f"optimizer type changed after {stage}"
            )
        if self.optimizer.param_groups is not self._optimizer_param_groups_object:
            raise BinaryTentFastProtocolError(
                f"optimizer param_groups object changed after {stage}"
            )
        if self.optimizer.state is not self._optimizer_state_object:
            raise BinaryTentFastProtocolError(
                f"optimizer state mapping object changed after {stage}"
            )
        if len(self.optimizer.param_groups) != len(self._optimizer_group_objects):
            raise BinaryTentFastProtocolError(
                f"optimizer param-group count changed after {stage}"
            )
        for index, (group, expected_object, expected_parameters, expected_options) in enumerate(
            zip(
                self.optimizer.param_groups,
                self._optimizer_group_objects,
                self._optimizer_group_parameters,
                self._optimizer_group_options,
                strict=True,
            )
        ):
            if group is not expected_object:
                raise BinaryTentFastProtocolError(
                    f"optimizer param-group object changed at {index} after {stage}"
                )
            current_parameters = tuple(group.get("params", ()))
            if len(current_parameters) != len(expected_parameters) or any(
                current is not expected
                for current, expected in zip(
                    current_parameters, expected_parameters, strict=False
                )
            ):
                raise BinaryTentFastProtocolError(
                    f"optimizer parameter order/binding changed at group {index} "
                    f"after {stage}"
                )
            current_options = {key: value for key, value in group.items() if key != "params"}
            if not _option_equal(current_options, expected_options):
                raise BinaryTentFastProtocolError(
                    f"optimizer options changed at group {index} after {stage}"
                )
        if not _option_equal(self.optimizer.defaults, self._optimizer_defaults):
            raise BinaryTentFastProtocolError(
                f"optimizer defaults changed after {stage}"
            )

    @staticmethod
    def _byte_flag(current: Tensor, expected: Tensor, label: str) -> Tensor:
        metadata = (
            tuple(current.shape),
            current.dtype,
            current.device,
            current.layout,
        )
        expected_metadata = (
            tuple(expected.shape),
            expected.dtype,
            expected.device,
            expected.layout,
        )
        if metadata != expected_metadata:
            raise BinaryTentFastProtocolError(
                f"tensor metadata changed at {label}: "
                f"current={metadata}, source={expected_metadata}"
            )
        if current.layout != torch.strided:
            raise BinaryTentFastProtocolError(
                f"fast exact-byte gate does not support non-strided tensor {label}"
            )
        current_bytes = current.detach().resolve_conj().resolve_neg().contiguous()
        expected_bytes = expected.detach().resolve_conj().resolve_neg().contiguous()
        return torch.eq(
            current_bytes.reshape(-1).view(torch.uint8),
            expected_bytes.reshape(-1).view(torch.uint8),
        ).all()

    @staticmethod
    def _false_flag_labels(flags: Sequence[tuple[str, Tensor]]) -> list[str]:
        by_device: dict[torch.device, list[tuple[str, Tensor]]] = defaultdict(list)
        for label, flag in flags:
            by_device[flag.device].append((label, flag))
        mismatches: list[str] = []
        for entries in by_device.values():
            resolved = torch.stack([flag for _label, flag in entries]).detach().cpu()
            mismatches.extend(
                label
                for (label, _flag), exact in zip(
                    entries, resolved.tolist(), strict=True
                )
                if not exact
            )
        return mismatches

    @classmethod
    def _resolve_byte_flags(
        cls, flags: Sequence[tuple[str, Tensor]], stage: str
    ) -> None:
        mismatches = cls._false_flag_labels(flags)
        if mismatches:
            raise BinaryTentFastProtocolError(
                "exact resident tensor-byte mismatch after "
                f"{stage}: {', '.join(mismatches[:10])}"
            )

    def _assert_resident_state(
        self, stage: str, *, include_adaptable: bool = False
    ) -> ResidentStateCheck:
        self._assert_topology(stage)
        flags: list[tuple[str, Tensor]] = []
        for source in self._parameters:
            if source.adaptable and not include_adaptable:
                continue
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
                    raise BinaryTentFastProtocolError(
                        "None buffer gained a tensor after "
                        f"{stage}: {source.module_name}.{source.name}"
                    )
                continue
            if current is None:
                raise BinaryTentFastProtocolError(
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
        return ResidentStateCheck(
            stage=stage,
            check_kind=CHECK_KIND,
            non_adaptable_parameter_tensors=sum(
                not source.adaptable for source in self._parameters
            ),
            adaptable_parameter_tensors=sum(
                source.adaptable for source in self._parameters
            ),
            persistent_buffer_tensors=persistent,
            nonpersistent_buffer_tensors=nonpersistent,
            none_buffers=none_buffers,
            batchnorm_running_mean_buffers=bn_counts["running_mean"],
            batchnorm_running_var_buffers=bn_counts["running_var"],
            batchnorm_num_batches_tracked_buffers=bn_counts[
                "num_batches_tracked"
            ],
        )

    def _assert_all_gradients_none(self, stage: str) -> None:
        offenders = [
            source.name
            for source in self._parameters
            if source.object.grad is not None
        ]
        if offenders:
            raise BinaryTentFastProtocolError(
                f"parameter gradients remain after {stage}: "
                + ", ".join(offenders[:10])
            )

    def _assert_canonical_source(self, stage: str) -> None:
        self._assert_topology(stage)
        training = [
            entry.name or "<root>"
            for entry in self._modules
            if entry.object.training != entry.training
        ]
        if training:
            raise BinaryTentFastProtocolError(
                f"module training runtime differs from Source after {stage}: "
                + ", ".join(training[:10])
            )
        if any(entry.training for entry in self._modules):
            raise BinaryTentFastProtocolError(
                "Binary TENT Source snapshot requires every module in eval mode"
            )
        requires_grad = tuple(
            (entry.name, bool(entry.object.requires_grad))
            for entry in self._parameters
        )
        expected = tuple(
            (entry.name, entry.requires_grad) for entry in self._parameters
        )
        if requires_grad != expected or any(flag for _name, flag in expected):
            raise BinaryTentFastProtocolError(
                f"parameter requires_grad differs from frozen Source after {stage}"
            )
        current_bn = tuple(
            (
                entry.name,
                bool(entry.object.track_running_stats),
                entry.object.momentum,
                entry.object.eps,
            )
            for entry in self._modules
            if isinstance(entry.object, nn.BatchNorm2d)
        )
        if current_bn != self._source_bn_runtime:
            raise BinaryTentFastProtocolError(
                f"BatchNorm runtime differs from Source after {stage}"
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
                raise BinaryTentFastProtocolError(
                    f"Source BatchNorm is incomplete at {entry.name!r} after {stage}"
                )
        self._assert_all_gradients_none(stage)

    def _assert_tent_runtime(self, stage: str) -> None:
        self._assert_topology(stage)
        adaptable_ids = {id(parameter) for parameter in self._adaptable_objects}
        for entry in self._parameters:
            expected = id(entry.object) in adaptable_ids
            if bool(entry.object.requires_grad) != expected:
                raise BinaryTentFastProtocolError(
                    f"requires_grad violates BN-affine-only policy at "
                    f"{entry.name!r} after {stage}"
                )
        source_bn = {
            name: (momentum, eps)
            for name, _track, momentum, eps in self._source_bn_runtime
        }
        for entry in self._modules:
            module = entry.object
            if not isinstance(module, nn.BatchNorm2d):
                if module.training:
                    raise BinaryTentFastProtocolError(
                        f"non-BatchNorm module entered training mode at "
                        f"{entry.name!r} after {stage}"
                    )
                continue
            momentum, eps = source_bn[entry.name]
            if module.momentum != momentum or module.eps != eps:
                raise BinaryTentFastProtocolError(
                    f"BatchNorm eps/momentum changed at {entry.name!r} after {stage}"
                )
            if (
                module.running_mean is None
                or module.running_var is None
                or module.num_batches_tracked is None
            ):
                raise BinaryTentFastProtocolError(
                    f"BatchNorm Source buffers disappeared at {entry.name!r} "
                    f"after {stage}"
                )
            if self.method.bn_protocol == BN_PROTOCOL_BATCH_STATS:
                if not module.training or module.track_running_stats:
                    raise BinaryTentFastProtocolError(
                        f"batch-stat TENT mode violation at {entry.name!r} "
                        f"after {stage}"
                    )
            elif self.method.bn_protocol == BN_PROTOCOL_SOURCE_STATS:
                if module.training or not module.track_running_stats:
                    raise BinaryTentFastProtocolError(
                        f"source-stat TENT mode violation at {entry.name!r} "
                        f"after {stage}"
                    )
            else:  # constructor validation in BinaryTentMethod should make this unreachable
                raise BinaryTentFastProtocolError(
                    f"unsupported BN protocol: {self.method.bn_protocol!r}"
                )
        self._assert_all_gradients_none(stage)

    def _assert_optimizer_source(self, stage: str) -> None:
        self._assert_optimizer_configuration(stage)
        if self.optimizer.state:
            raise BinaryTentFastProtocolError(
                f"optimizer state differs from empty Source after {stage}"
            )

    def _capture_optimizer_state(self, stage: str) -> _OptimizerStateSnapshot:
        self._assert_optimizer_configuration(stage)
        entries_by_id = {
            id(parameter): (name, parameter)
            for name, parameter in zip(
                self._adaptable_names, self._adaptable_objects, strict=True
            )
        }
        tensors: list[_OptimizerTensor] = []
        scalars: list[_OptimizerScalar] = []
        names_with_state: list[str] = []
        finite_flags: list[tuple[str, Tensor]] = []
        for parameter, state in self.optimizer.state.items():
            expected_entry = entries_by_id.get(id(parameter))
            if expected_entry is None or parameter is not expected_entry[1]:
                raise BinaryTentFastProtocolError(
                    f"optimizer state is bound outside BN affine parameters after {stage}"
                )
            parameter_name = expected_entry[0]
            if not isinstance(state, Mapping):
                raise BinaryTentFastProtocolError(
                    f"optimizer state for {parameter_name!r} is not a mapping "
                    f"after {stage}"
                )
            names_with_state.append(parameter_name)
            if not all(isinstance(state_name, str) for state_name in state):
                raise BinaryTentFastProtocolError(
                    f"optimizer state key is not a string after {stage}"
                )
            for state_name in sorted(state):
                value = state[state_name]
                if isinstance(value, Tensor):
                    finite_flags.append(
                        (
                            f"{parameter_name}.{state_name}",
                            torch.isfinite(value).all(),
                        )
                    )
                    tensors.append(
                        _OptimizerTensor(
                            parameter_name=parameter_name,
                            state_name=state_name,
                            value=value.detach().clone(),
                        )
                    )
                elif isinstance(value, (bool, int, float, str)) or value is None:
                    if isinstance(value, float) and not math.isfinite(value):
                        raise BinaryTentFastProtocolError(
                            f"optimizer scalar {parameter_name}.{state_name} is "
                            f"non-finite after {stage}"
                        )
                    scalars.append(
                        _OptimizerScalar(parameter_name, state_name, value)
                    )
                else:
                    raise BinaryTentFastProtocolError(
                        "fast runner does not support optimizer state value "
                        f"{_qualified_type(value)} at "
                        f"{parameter_name}.{state_name}"
                    )
        nonfinite = self._false_flag_labels(finite_flags)
        if nonfinite:
            raise BinaryTentFastProtocolError(
                f"optimizer tensors contain NaN/Inf after {stage}: "
                + ", ".join(nonfinite[:10])
            )
        return _OptimizerStateSnapshot(
            tensors=tuple(tensors),
            scalars=tuple(scalars),
            parameter_names_with_state=tuple(names_with_state),
        )

    def _assert_optimizer_state_exact(
        self, expected: _OptimizerStateSnapshot, stage: str
    ) -> None:
        current = self._capture_optimizer_state(stage)
        if current.parameter_names_with_state != expected.parameter_names_with_state:
            raise BinaryTentFastProtocolError(
                f"optimizer state topology changed after {stage}"
            )
        if current.scalars != expected.scalars:
            raise BinaryTentFastProtocolError(
                f"optimizer scalar state changed after {stage}"
            )
        current_keys = tuple(
            (value.parameter_name, value.state_name) for value in current.tensors
        )
        expected_keys = tuple(
            (value.parameter_name, value.state_name) for value in expected.tensors
        )
        if current_keys != expected_keys:
            raise BinaryTentFastProtocolError(
                f"optimizer tensor state topology changed after {stage}"
            )
        flags = [
            (
                f"optimizer:{current_value.parameter_name}.{current_value.state_name}",
                self._byte_flag(
                    current_value.value,
                    expected_value.value,
                    f"optimizer:{current_value.parameter_name}.{current_value.state_name}",
                ),
            )
            for current_value, expected_value in zip(
                current.tensors, expected.tensors, strict=True
            )
        ]
        self._resolve_byte_flags(flags, stage)

    def _adaptable_values(self) -> tuple[Tensor, ...]:
        return tuple(
            parameter.detach().clone() for parameter in self._adaptable_objects
        )

    def _assert_adaptable_exact(
        self, expected: Sequence[Tensor], stage: str
    ) -> None:
        if len(expected) != len(self._adaptable_objects):
            raise BinaryTentFastProtocolError(
                f"adaptable snapshot length changed after {stage}"
            )
        self._resolve_byte_flags(
            [
                (
                    f"adaptable:{name}",
                    self._byte_flag(parameter, value, f"adaptable:{name}"),
                )
                for name, parameter, value in zip(
                    self._adaptable_names,
                    self._adaptable_objects,
                    expected,
                    strict=True,
                )
            ],
            stage,
        )

    def _adaptable_change_count(self) -> int:
        source_by_name = {entry.name: entry for entry in self._parameters}
        flags = [
            self._byte_flag(
                parameter,
                source_by_name[name].value,
                f"adaptable:{name}",
            )
            for name, parameter in zip(
                self._adaptable_names, self._adaptable_objects, strict=True
            )
        ]
        if not flags:
            return 0
        by_device: dict[torch.device, list[Tensor]] = defaultdict(list)
        for flag in flags:
            by_device[flag.device].append(flag)
        exact_values: list[bool] = []
        for device_flags in by_device.values():
            exact_values.extend(torch.stack(device_flags).detach().cpu().tolist())
        return sum(not exact for exact in exact_values)

    def _restore_source_fast(self) -> None:
        # BN affine values are the only model tensors allowed to differ.
        source_by_name = {entry.name: entry for entry in self._parameters}
        with torch.no_grad():
            for name, parameter in zip(
                self._adaptable_names, self._adaptable_objects, strict=True
            ):
                parameter.copy_(source_by_name[name].value)
        self.optimizer.state.clear()
        for entry in self._parameters:
            entry.object.grad = None

        # Restore parent modes first and child-specific values last, matching
        # StateManager's source restoration order.
        for entry in self._modules:
            entry.object.train(entry.training)
        for entry in self._parameters:
            entry.object.requires_grad_(entry.requires_grad)
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

    def _checked_forward(self, image: Tensor, *, grad: bool) -> Tensor:
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            logits = self.adapter.forward_logits(image)
        if logits.ndim != 4 or logits.shape[:2] != (1, 1):
            raise BinaryTentFastProtocolError(
                f"logits must have shape [1,1,H,W], got {tuple(logits.shape)}"
            )
        if logits.shape[-2:] != image.shape[-2:]:
            raise BinaryTentFastProtocolError("logit and image spatial sizes differ")
        if not torch.isfinite(logits).all():
            raise BinaryTentFastProtocolError("model produced NaN/Inf logits")
        return logits if grad else logits.detach().clone()

    def _emergency_full_reset(self, original: BaseException) -> None:
        try:
            reset = self.state.reset_to_source()
            if reset != self.state.source_fingerprint:
                raise BinaryTentFastProtocolError(
                    "emergency reset fingerprint differs from Source"
                )
        except BaseException as recovery:
            raise BinaryTentFastRecoveryError(original, recovery) from original

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
        if changed_adaptable_tensors <= 0:
            raise BinaryTentFastProtocolError(
                "Binary TENT step did not change any BN affine tensor"
            )
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
            if adapt_differences != ("model", "optimizer", "runtime"):
                raise BinaryTentFastProtocolError(
                    "full adaptation audit expected model, optimizer, and runtime "
                    f"to differ from Source, got {adapt_differences}"
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
                differences = post_fingerprint.differing_components(adapt_fingerprint)
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
        """Run one episode, fast-reset Source, and return three predictions."""

        if self._aborted:
            raise BinaryTentFastProtocolError(
                "Binary TENT fast runner was aborted by a previous failed episode"
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
        reset_fingerprint: StateFingerprint | None = None
        candidate: BinaryTentFastEpisodeResult | None = None

        try:
            try:
                body = self._run_body(
                    image=image,
                    metadata=metadata,
                    audit_due=audit_due,
                )
            except BaseException as original:
                exposed: BaseException = original
                if (
                    isinstance(original, EpisodeProtocolError)
                    and not isinstance(original, BinaryTentFastProtocolError)
                ):
                    exposed = BinaryTentFastProtocolError(str(original))
                self._aborted = True
                self._emergency_full_reset(exposed)
                if exposed is not original:
                    raise exposed from original
                raise

            try:
                self._restore_source_fast()
                self._assert_canonical_source("fast_reset")
                self._assert_optimizer_source("fast_reset")
                assert body is not None
                resident_checks = dict(body["resident_checks"])
                resident_checks["fast_reset"] = self._assert_resident_state(
                    "fast_reset", include_adaptable=True
                )
                if audit_due:
                    reset_fingerprint = self.state.assert_source_state()
                    if reset_fingerprint != self.state.source_fingerprint:
                        raise BinaryTentFastProtocolError(
                            "full reset audit differs from Source"
                        )

                source = body["logits_source"].detach().cpu().clone()
                outcome = body["outcome"]
                assert isinstance(outcome, BinaryTentOutcome)
                tent_pre = outcome.logits_tent_pre.detach().cpu().clone()
                tent_post = body["logits_post"].detach().cpu().clone()
                eps = float(outcome.diagnostics["entropy_eps"])
                entropy_pre = float(binary_entropy_map(tent_pre, eps=eps).mean())
                entropy_post = float(binary_entropy_map(tent_post, eps=eps).mean())
                if not math.isfinite(entropy_pre) or not math.isfinite(entropy_post):
                    raise BinaryTentFastProtocolError(
                        "episode entropy diagnostics are non-finite"
                    )
                reported_pre = float(
                    outcome.diagnostics["optimization_entropy_pre_device"]
                )
                entropy_error = abs(entropy_pre - reported_pre)
                if not math.isclose(
                    entropy_pre,
                    reported_pre,
                    rel_tol=ENTROPY_REDUCTION_REL_TOL,
                    abs_tol=ENTROPY_REDUCTION_ABS_TOL,
                ):
                    raise BinaryTentFastProtocolError(
                        "stored update-time entropy differs from returned TENT-pre logits"
                    )
                source_tent_exact = bool(torch.equal(source, tent_pre))
                if (
                    self.method.bn_protocol == BN_PROTOCOL_SOURCE_STATS
                    and not source_tent_exact
                ):
                    raise BinaryTentFastProtocolError(
                        "source-stat TENT-pre logits must be bit-exact with Source logits"
                    )
                tent_pre_post_exact = bool(torch.equal(tent_pre, tent_post))

                diagnostics = dict(outcome.diagnostics)
                diagnostics.update(
                    {
                        "entropy_pre": entropy_pre,
                        "entropy_post": entropy_post,
                        "entropy_delta": entropy_post - entropy_pre,
                        "entropy_pre_reduction_abs_error": entropy_error,
                        "entropy_reduction_rel_tolerance": ENTROPY_REDUCTION_REL_TOL,
                        "entropy_reduction_abs_tolerance": ENTROPY_REDUCTION_ABS_TOL,
                        "source_tent_pre_bit_exact": source_tent_exact,
                        "tent_pre_post_bit_exact": tent_pre_post_exact,
                        "source_pre_raw_sha256": _cpu_tensor_sha256(source),
                        "tent_pre_raw_sha256": _cpu_tensor_sha256(tent_pre),
                        "tent_post_raw_sha256": _cpu_tensor_sha256(tent_post),
                        "changed_bn_affine_tensors_fast_gate": body[
                            "changed_adaptable_tensors"
                        ],
                        "temporary_optimizer_state_parameter_count": body[
                            "optimizer_state_parameter_count"
                        ],
                        "post_forward_adaptable_params_unchanged": True,
                        "post_forward_optimizer_state_unchanged": True,
                        "fast_source_reset_complete": True,
                        "full_sha_audit_performed": audit_due,
                        "full_sha_audit_reason": audit_reason,
                    }
                )
                result_checks = {
                    "safe_label_free_metadata": True,
                    "batch_size_one": True,
                    "source_pre_no_grad": not source.requires_grad,
                    "tent_pre_detached": not tent_pre.requires_grad,
                    "tent_post_no_grad": not tent_post.requires_grad,
                    "only_bn_affine_temporarily_trainable": True,
                    "non_bn_parameters_exact_every_gate": True,
                    "all_registered_buffers_exact_every_gate": True,
                    "parameter_module_optimizer_topology_exact_every_gate": True,
                    "bn_affine_changed_by_one_step": (
                        body["changed_adaptable_tensors"] > 0
                    ),
                    "optimizer_changed_by_one_step": (
                        body["optimizer_state_parameter_count"] > 0
                    ),
                    "post_forward_state_unchanged": True,
                    "all_gradients_cleared": True,
                    "source_runtime_restored_per_image": True,
                    "bn_affine_restored_per_image": True,
                    "optimizer_restored_per_image": True,
                    "input_unchanged": True,
                    "rng_restored": True,
                }
                candidate = BinaryTentFastEpisodeResult(
                    method="binary_episodic_tent",
                    episode_number=episode_number,
                    metadata=body["metadata"],
                    logits_source_pre=source,
                    logits_tent_pre=tent_pre,
                    logits_tent_post=tent_post,
                    entropy_tent_pre=entropy_pre,
                    entropy_tent_post=entropy_post,
                    source_tent_pre_bit_exact=source_tent_exact,
                    tent_pre_post_bit_exact=tent_pre_post_exact,
                    input_unchanged=True,
                    check_kind=CHECK_KIND,
                    resident_checks=resident_checks,
                    checks=result_checks,
                    diagnostics=diagnostics,
                    source_state_sha256=self.state.source_fingerprint.full_sha256,
                    full_audit_performed=audit_due,
                    full_audit_reason=audit_reason,
                    adapt_full_fingerprint=body["adapt_fingerprint"],
                    post_full_fingerprint=body["post_fingerprint"],
                    reset_full_fingerprint=reset_fingerprint,
                    adapt_full_state_differences=body["adapt_differences"],
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
    "BinaryTentFastEpisodeResult",
    "BinaryTentFastProtocolError",
    "BinaryTentFastRecoveryError",
    "BinaryTentFastRunner",
    "CHECK_KIND",
    "ResidentStateCheck",
]
