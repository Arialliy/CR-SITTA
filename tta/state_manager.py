"""Exact source-state restoration for single-image episodic adaptation.

The manager is deliberately method agnostic: it does not know how AdaBN,
TENT, or a future CR-SITTA method updates a model.  Its only responsibility is
to make every episode start from, and return to, the same immutable source
state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import struct
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class SourceStateMismatchError(AssertionError):
    """Raised when the live episodic state differs from the source snapshot."""


class SourceSnapshotError(RuntimeError):
    """Raised when the source snapshot cannot be created or restored safely."""


@dataclass(frozen=True)
class StatefulHooks:
    """Snapshot/restore callbacks for method-specific Python state.

    ``snapshot`` must return a value supported by :func:`deepcopy` and by the
    deterministic fingerprint encoder below.  ``restore`` receives a defensive
    copy, so it cannot mutate the manager's private source snapshot.
    """

    snapshot: Callable[[], Any]
    restore: Callable[[Any], None]


@dataclass(frozen=True)
class StateFingerprint:
    """Component hashes for one complete episodic state."""

    model_sha256: str
    optimizer_sha256: str | None
    runtime_sha256: str
    topology_sha256: str
    gradients_sha256: str
    extras_sha256: str
    full_sha256: str

    def differing_components(self, other: "StateFingerprint") -> tuple[str, ...]:
        """Return stable names for components that differ from ``other``."""

        fields = (
            ("model", self.model_sha256, other.model_sha256),
            ("optimizer", self.optimizer_sha256, other.optimizer_sha256),
            ("runtime", self.runtime_sha256, other.runtime_sha256),
            ("topology", self.topology_sha256, other.topology_sha256),
            ("gradients", self.gradients_sha256, other.gradients_sha256),
            ("extras", self.extras_sha256, other.extras_sha256),
        )
        return tuple(name for name, current, expected in fields if current != expected)


@dataclass(frozen=True)
class _SourceSnapshot:
    model_state: Any
    buffer_state: tuple[
        tuple[str, tuple[str, ...], tuple[tuple[str, Tensor | None], ...]], ...
    ]
    optimizer_state: Any | None
    optimizer_parameter_names: tuple[tuple[str, ...], ...] | None
    module_training: tuple[tuple[str, str, bool], ...]
    parameter_requires_grad: tuple[tuple[str, bool], ...]
    bn_runtime: tuple[tuple[str, bool, float | None, float], ...]
    parameter_topology: tuple[tuple[str, str, bool], ...]
    parameter_gradients: tuple[tuple[str, Tensor | None], ...]
    extra_states: tuple[tuple[str, Any], ...]
    fingerprint: StateFingerprint


def _qualified_type(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _update_bytes(digest: "hashlib._Hash", value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


def _update_tensor(digest: "hashlib._Hash", tensor: Tensor) -> None:
    digest.update(b"tensor")
    _update_bytes(digest, str(tensor.dtype).encode("utf-8"))
    _update_bytes(digest, str(tensor.device).encode("utf-8"))
    _update_bytes(digest, str(tensor.layout).encode("utf-8"))
    _update_object(digest, tuple(tensor.shape))

    detached = tensor.detach()
    if detached.layout != torch.strided:
        # Model and optimizer states in this project are dense.  Supporting a
        # dense representation here still gives deterministic coverage should
        # a future method add a sparse tensor to its optimizer state.
        detached = detached.to_dense()
    detached = detached.resolve_conj().resolve_neg().cpu().contiguous()
    if detached.numel() == 0:
        raw = b""
    else:
        raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
    _update_bytes(digest, raw)


def _object_digest(value: Any) -> bytes:
    digest = hashlib.sha256()
    _update_object(digest, value)
    return digest.digest()


def _update_object(digest: "hashlib._Hash", value: Any) -> None:
    """Update ``digest`` with a deterministic, type-aware representation."""

    if value is None:
        digest.update(b"none")
    elif isinstance(value, bool):
        digest.update(b"bool1" if value else b"bool0")
    elif isinstance(value, int):
        digest.update(b"int")
        _update_bytes(digest, str(value).encode("ascii"))
    elif isinstance(value, float):
        digest.update(b"float")
        digest.update(struct.pack(">d", value))
    elif isinstance(value, str):
        digest.update(b"str")
        _update_bytes(digest, value.encode("utf-8"))
    elif isinstance(value, bytes):
        digest.update(b"bytes")
        _update_bytes(digest, value)
    elif isinstance(value, Tensor):
        _update_tensor(digest, value)
    elif isinstance(value, torch.device):
        digest.update(b"device")
        _update_bytes(digest, str(value).encode("utf-8"))
    elif isinstance(value, torch.dtype):
        digest.update(b"dtype")
        _update_bytes(digest, str(value).encode("utf-8"))
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        digest.update(struct.pack(">Q", len(value)))
        # Mappings are semantic key/value sets.  Sorting by the canonical key
        # digest avoids false mismatches caused only by insertion order.
        items = sorted(
            ((_object_digest(key), key, item) for key, item in value.items()),
            key=lambda entry: entry[0],
        )
        for key_digest, _key, item in items:
            digest.update(key_digest)
            _update_object(digest, item)
    elif isinstance(value, tuple):
        digest.update(b"tuple")
        digest.update(struct.pack(">Q", len(value)))
        for item in value:
            _update_object(digest, item)
    elif isinstance(value, list):
        digest.update(b"list")
        digest.update(struct.pack(">Q", len(value)))
        for item in value:
            _update_object(digest, item)
    elif isinstance(value, (set, frozenset)):
        digest.update(b"frozenset" if isinstance(value, frozenset) else b"set")
        item_digests = sorted(_object_digest(item) for item in value)
        digest.update(struct.pack(">Q", len(item_digests)))
        for item_digest in item_digests:
            digest.update(item_digest)
    else:
        raise TypeError(
            "episodic state contains an unsupported fingerprint value of type "
            f"{_qualified_type(value)}"
        )


def _sha256(value: Any) -> str:
    return _object_digest(value).hex()


def _module_runtime_state(
    model: nn.Module,
) -> tuple[
    tuple[tuple[str, str, bool], ...],
    tuple[tuple[str, bool], ...],
    tuple[tuple[str, bool, float | None, float], ...],
]:
    module_training = tuple(
        (name, _qualified_type(module), bool(module.training))
        for name, module in model.named_modules()
    )
    parameter_requires_grad = tuple(
        (name, bool(parameter.requires_grad))
        for name, parameter in model.named_parameters()
    )
    bn_runtime = tuple(
        (
            name,
            bool(module.track_running_stats),
            module.momentum,
            module.eps,
        )
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    )
    return module_training, parameter_requires_grad, bn_runtime


def _buffer_state(
    model: nn.Module,
) -> tuple[
    tuple[str, tuple[str, ...], tuple[tuple[str, Tensor | None], ...]], ...
]:
    """Capture every registered buffer, including None and non-persistent ones."""

    modules: list[
        tuple[str, tuple[str, ...], tuple[tuple[str, Tensor | None], ...]]
    ] = []
    for module_name, module in model.named_modules():
        nonpersistent = tuple(sorted(module._non_persistent_buffers_set))
        buffers = tuple(
            (
                name,
                None if value is None else value.detach().clone(),
            )
            for name, value in sorted(module._buffers.items())
        )
        modules.append((module_name, nonpersistent, buffers))
    return tuple(modules)


def _parameter_topology(
    model: nn.Module,
    source_parameters: tuple[tuple[str, nn.Parameter], ...],
) -> tuple[tuple[str, str, bool], ...]:
    source_by_name = dict(source_parameters)
    return tuple(
        (
            name,
            _qualified_type(parameter),
            source_by_name.get(name) is parameter,
        )
        for name, parameter in model.named_parameters()
    )


def _optimizer_parameter_names(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    strict: bool = True,
) -> tuple[tuple[str, ...], ...]:
    name_by_id = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    groups: list[tuple[str, ...]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        names: list[str] = []
        for parameter_index, parameter in enumerate(group["params"]):
            name = name_by_id.get(id(parameter))
            if name is None:
                if strict:
                    raise SourceSnapshotError(
                        "optimizer parameter is not bound to the managed model at "
                        f"group {group_index}, position {parameter_index}"
                    )
                name = (
                    f"<unbound:{group_index}:{parameter_index}:"
                    f"{_qualified_type(parameter)}>"
                )
            names.append(name)
        groups.append(tuple(names))
    return tuple(groups)


def _parameter_gradients(model: nn.Module) -> tuple[tuple[str, Tensor | None], ...]:
    return tuple(
        (
            name,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
        for name, parameter in model.named_parameters()
    )


def _runtime_payload(
    module_training: tuple[tuple[str, str, bool], ...],
    parameter_requires_grad: tuple[tuple[str, bool], ...],
    bn_runtime: tuple[tuple[str, bool, float | None, float], ...],
) -> tuple[Any, ...]:
    return (
        ("module_training", module_training),
        ("parameter_requires_grad", parameter_requires_grad),
        ("bn_runtime", bn_runtime),
    )


def _build_fingerprint(
    *,
    model_state: Any,
    buffer_state: tuple[
        tuple[str, tuple[str, ...], tuple[tuple[str, Tensor | None], ...]], ...
    ],
    optimizer_state: Any | None,
    optimizer_parameter_names: tuple[tuple[str, ...], ...] | None,
    module_training: tuple[tuple[str, str, bool], ...],
    parameter_requires_grad: tuple[tuple[str, bool], ...],
    bn_runtime: tuple[tuple[str, bool, float | None, float], ...],
    parameter_topology: tuple[tuple[str, str, bool], ...],
    parameter_gradients: tuple[tuple[str, Tensor | None], ...],
    extra_states: tuple[tuple[str, Any], ...],
) -> StateFingerprint:
    model_hash = _sha256(
        (("state_dict", model_state), ("registered_buffers", buffer_state))
    )
    optimizer_hash = (
        None
        if optimizer_state is None
        else _sha256(
            (
                ("state_dict", optimizer_state),
                ("model_parameter_names", optimizer_parameter_names),
            )
        )
    )
    runtime_hash = _sha256(
        _runtime_payload(
            module_training,
            parameter_requires_grad,
            bn_runtime,
        )
    )
    topology_hash = _sha256(parameter_topology)
    gradients_hash = _sha256(parameter_gradients)
    extras_hash = _sha256(extra_states)
    components = (
        ("model", model_hash),
        ("optimizer", optimizer_hash),
        ("runtime", runtime_hash),
        ("topology", topology_hash),
        ("gradients", gradients_hash),
        ("extras", extras_hash),
    )
    return StateFingerprint(
        model_sha256=model_hash,
        optimizer_sha256=optimizer_hash,
        runtime_sha256=runtime_hash,
        topology_sha256=topology_hash,
        gradients_sha256=gradients_hash,
        extras_sha256=extras_hash,
        full_sha256=_sha256(components),
    )


class EpisodicStateManager:
    """Own an immutable source snapshot and restore it after every episode.

    The source state is captured once at construction by default.  It includes
    model parameters, every persistent/non-persistent/None buffer, optimizer
    values and model-parameter bindings, all module train/eval flags, parameter
    ``requires_grad`` flags, BatchNorm ``track_running_stats``/``momentum``/
    ``eps``, and parameter gradients.  Additional method state can be
    registered only at construction through named :class:`StatefulHooks`.

    The class is intentionally not thread-safe: one instance belongs to one
    model/optimizer pair and one sequential episodic runner.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        *,
        extra_stateful: Mapping[str, StatefulHooks | object] | None = None,
        capture_on_init: bool = True,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if optimizer is not None and not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer or None")

        self.model = model
        self.optimizer = optimizer
        self._extra_hooks = self._normalise_extra_hooks(extra_stateful or {})
        self._snapshot: _SourceSnapshot | None = None
        self._source_parameter_objects: tuple[tuple[str, nn.Parameter], ...] = ()
        self._source_optimizer_groups: tuple[tuple[nn.Parameter, ...], ...] | None = (
            None
        )
        if capture_on_init:
            self.save_source_state()

    @staticmethod
    def _normalise_extra_hooks(
        extra_stateful: Mapping[str, StatefulHooks | object],
    ) -> tuple[tuple[str, StatefulHooks], ...]:
        hooks: list[tuple[str, StatefulHooks]] = []
        for name, stateful in sorted(extra_stateful.items()):
            if not isinstance(name, str) or not name:
                raise ValueError("extra state names must be non-empty strings")
            if isinstance(stateful, StatefulHooks):
                hook = stateful
            else:
                snapshot = getattr(stateful, "state_dict", None)
                restore = getattr(stateful, "load_state_dict", None)
                if not callable(snapshot) or not callable(restore):
                    raise TypeError(
                        f"extra state {name!r} must provide state_dict/load_state_dict "
                        "or be a StatefulHooks instance"
                    )
                hook = StatefulHooks(snapshot=snapshot, restore=restore)
            if not callable(hook.snapshot) or not callable(hook.restore):
                raise TypeError(f"extra state callbacks for {name!r} must be callable")
            hooks.append((name, hook))
        return tuple(hooks)

    def _require_snapshot(self) -> _SourceSnapshot:
        if self._snapshot is None:
            raise SourceSnapshotError("source state has not been saved")
        return self._snapshot

    def _capture_extra_states(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            (name, deepcopy(hook.snapshot())) for name, hook in self._extra_hooks
        )

    def save_source_state(self) -> StateFingerprint:
        """Capture source state exactly once; later calls fail closed.

        ``capture_on_init=False`` exists for callers that must finish optimizer
        initialization before explicitly sealing the source snapshot.
        """

        if self._snapshot is not None:
            raise SourceSnapshotError(
                "source state is already saved and cannot be overwritten"
            )

        model_state = deepcopy(self.model.state_dict())
        optimizer_state = (
            None if self.optimizer is None else deepcopy(self.optimizer.state_dict())
        )
        source_parameter_objects = tuple(self.model.named_parameters())
        source_optimizer_groups = (
            None
            if self.optimizer is None
            else tuple(tuple(group["params"]) for group in self.optimizer.param_groups)
        )
        buffer_state = _buffer_state(self.model)
        optimizer_parameter_names = (
            None
            if self.optimizer is None
            else _optimizer_parameter_names(self.model, self.optimizer)
        )
        module_training, parameter_requires_grad, bn_runtime = _module_runtime_state(
            self.model
        )
        parameter_topology = _parameter_topology(
            self.model, source_parameter_objects
        )
        parameter_gradients = _parameter_gradients(self.model)
        extra_states = self._capture_extra_states()
        fingerprint = _build_fingerprint(
            model_state=model_state,
            buffer_state=buffer_state,
            optimizer_state=optimizer_state,
            optimizer_parameter_names=optimizer_parameter_names,
            module_training=module_training,
            parameter_requires_grad=parameter_requires_grad,
            bn_runtime=bn_runtime,
            parameter_topology=parameter_topology,
            parameter_gradients=parameter_gradients,
            extra_states=extra_states,
        )

        # Parameter objects and optimizer membership are structural bindings,
        # not serialised values.  Keep their identities so an accidental module
        # or param-group replacement fails closed instead of appearing reset.
        self._source_parameter_objects = source_parameter_objects
        self._source_optimizer_groups = source_optimizer_groups
        self._snapshot = _SourceSnapshot(
            model_state=model_state,
            buffer_state=buffer_state,
            optimizer_state=optimizer_state,
            optimizer_parameter_names=optimizer_parameter_names,
            module_training=module_training,
            parameter_requires_grad=parameter_requires_grad,
            bn_runtime=bn_runtime,
            parameter_topology=parameter_topology,
            parameter_gradients=parameter_gradients,
            extra_states=extra_states,
            fingerprint=fingerprint,
        )
        return fingerprint

    @property
    def source_fingerprint(self) -> StateFingerprint:
        """Return the immutable, hash-only description of the source state."""

        return self._require_snapshot().fingerprint

    @property
    def extra_state_names(self) -> tuple[str, ...]:
        """Return the stable names of method-specific state owned by the manager.

        Specialised runners can use this read-only contract to fail closed when
        they do not implement the generic manager's extra-state restore path.
        """

        return tuple(name for name, _hook in self._extra_hooks)

    def current_fingerprint(self) -> StateFingerprint:
        """Fingerprint the complete live episodic state."""

        self._require_snapshot()
        module_training, parameter_requires_grad, bn_runtime = _module_runtime_state(
            self.model
        )
        optimizer_state = None if self.optimizer is None else self.optimizer.state_dict()
        optimizer_parameter_names = (
            None
            if self.optimizer is None
            else _optimizer_parameter_names(
                self.model,
                self.optimizer,
                strict=False,
            )
        )
        return _build_fingerprint(
            model_state=self.model.state_dict(),
            buffer_state=_buffer_state(self.model),
            optimizer_state=optimizer_state,
            optimizer_parameter_names=optimizer_parameter_names,
            module_training=module_training,
            parameter_requires_grad=parameter_requires_grad,
            bn_runtime=bn_runtime,
            parameter_topology=_parameter_topology(
                self.model, self._source_parameter_objects
            ),
            parameter_gradients=_parameter_gradients(self.model),
            extra_states=self._capture_extra_states(),
        )

    def state_hash(self) -> str:
        """Return the complete live-state SHA-256 for concise gate logging."""

        return self.current_fingerprint().full_sha256

    def assert_source_state(self) -> StateFingerprint:
        """Assert that every managed component exactly matches the source."""

        self._require_snapshot()
        binding_issue = self._parameter_binding_issue()
        if binding_issue is not None:
            raise SourceStateMismatchError(
                "episodic state differs from source in: topology ("
                + binding_issue
                + ")"
            )
        current = self.current_fingerprint()
        source = self.source_fingerprint
        if current != source:
            differences = ", ".join(current.differing_components(source))
            raise SourceStateMismatchError(
                "episodic state differs from source in: " + differences
            )
        return current

    def _parameter_binding_issue(self) -> str | None:
        current = tuple(self.model.named_parameters())
        source_names = tuple(name for name, _parameter in self._source_parameter_objects)
        current_names = tuple(name for name, _parameter in current)
        if current_names != source_names:
            return "model parameter names changed after the source snapshot"
        if any(
            current_parameter is not source_parameter
            for (_name, current_parameter), (_source_name, source_parameter) in zip(
                current, self._source_parameter_objects, strict=True
            )
        ):
            return "model parameter objects changed after the source snapshot"
        return None

    def _preflight_reset(
        self,
        snapshot: _SourceSnapshot,
    ) -> tuple[dict[str, nn.Module], dict[str, nn.Parameter]]:
        """Validate every restorable topology before performing any write."""

        binding_issue = self._parameter_binding_issue()
        if binding_issue is not None:
            raise SourceSnapshotError(binding_issue)

        modules = dict(self.model.named_modules())
        expected_module_names = tuple(
            name for name, _type, _training in snapshot.module_training
        )
        if tuple(modules) != expected_module_names:
            raise SourceSnapshotError(
                "model module topology changed after the source snapshot"
            )
        for name, expected_type, _training in snapshot.module_training:
            if _qualified_type(modules[name]) != expected_type:
                raise SourceSnapshotError(
                    f"model module type changed at {name!r} after the source snapshot"
                )

        parameters = dict(self.model.named_parameters())
        expected_parameter_names = tuple(
            name for name, _flag in snapshot.parameter_requires_grad
        )
        if tuple(parameters) != expected_parameter_names:
            raise SourceSnapshotError(
                "model parameter topology changed after the source snapshot"
            )
        # Parameter identities are unchanged, but direct .data assignment can
        # still alter shape/dtype/device.  Reject it before load_state_dict can
        # partially copy earlier parameters and then fail on a later one.
        for name, parameter in parameters.items():
            source_value = snapshot.model_state.get(name)
            if not isinstance(source_value, Tensor):
                raise SourceSnapshotError(
                    f"source state is missing model parameter {name!r}"
                )
            current_metadata = (
                tuple(parameter.shape),
                parameter.dtype,
                parameter.device,
            )
            source_metadata = (
                tuple(source_value.shape),
                source_value.dtype,
                source_value.device,
            )
            if current_metadata != source_metadata:
                raise SourceSnapshotError(
                    f"model parameter metadata changed at {name!r} after the "
                    "source snapshot"
                )

        expected_buffer_modules = tuple(
            name for name, _nonpersistent, _buffers in snapshot.buffer_state
        )
        if expected_buffer_modules != tuple(modules):
            raise SourceSnapshotError(
                "model buffer module topology changed after the source snapshot"
            )
        for module_name, source_nonpersistent, source_buffers in snapshot.buffer_state:
            module = modules[module_name]
            current_nonpersistent = tuple(
                sorted(module._non_persistent_buffers_set)
            )
            if current_nonpersistent != source_nonpersistent:
                raise SourceSnapshotError(
                    "buffer persistence topology changed at module "
                    f"{module_name!r} after the source snapshot"
                )
            current_buffer_names = tuple(sorted(module._buffers))
            source_buffer_names = tuple(name for name, _value in source_buffers)
            if current_buffer_names != source_buffer_names:
                raise SourceSnapshotError(
                    f"registered buffer topology changed at module {module_name!r} "
                    "after the source snapshot"
                )

        if self.optimizer is not None:
            if snapshot.optimizer_state is None or self._source_optimizer_groups is None:
                raise SourceSnapshotError("optimizer source state is unavailable")
            if len(self.optimizer.param_groups) != len(self._source_optimizer_groups):
                raise SourceSnapshotError(
                    "optimizer param-group topology changed after the source snapshot"
                )
            # This validates every current optimizer parameter binding without
            # requiring source order; a legal reorder is restored below.
            _optimizer_parameter_names(self.model, self.optimizer, strict=True)
        elif snapshot.optimizer_state is not None:
            raise SourceSnapshotError("managed optimizer was removed after snapshot")

        return modules, parameters

    @staticmethod
    def _restore_registered_buffers(
        modules: Mapping[str, nn.Module],
        buffer_state: tuple[
            tuple[str, tuple[str, ...], tuple[tuple[str, Tensor | None], ...]], ...
        ],
    ) -> None:
        for module_name, nonpersistent, buffers in buffer_state:
            module = modules[module_name]
            module._non_persistent_buffers_set = set(nonpersistent)
            for name, source_value in buffers:
                module._buffers[name] = (
                    None
                    if source_value is None
                    else source_value.detach().clone()
                )

    def reset_to_source(self, *, verify: bool = True) -> StateFingerprint:
        """Restore every managed component to the sealed source state."""

        snapshot = self._require_snapshot()
        modules, parameters = self._preflight_reset(snapshot)

        # Restore registered buffers first.  This makes official TENT-style
        # running_mean/running_var/num_batches_tracked=None mutations safe for
        # strict state_dict loading, while non-persistent buffers are restored
        # even though state_dict deliberately excludes them.
        self._restore_registered_buffers(modules, snapshot.buffer_state)
        self.model.load_state_dict(deepcopy(snapshot.model_state), strict=True)

        if self.optimizer is not None:
            assert snapshot.optimizer_state is not None
            assert self._source_optimizer_groups is not None
            # Restore group parameter membership before load_state_dict restores
            # group hyperparameters and moment/momentum tensors.
            for group, source_parameters in zip(
                self.optimizer.param_groups,
                self._source_optimizer_groups,
                strict=True,
            ):
                group["params"] = list(source_parameters)
            self.optimizer.load_state_dict(deepcopy(snapshot.optimizer_state))

        for name, expected_type, training in snapshot.module_training:
            module = modules[name]
            assert _qualified_type(module) == expected_type
            module.train(training)

        for name, requires_grad in snapshot.parameter_requires_grad:
            parameters[name].requires_grad_(requires_grad)

        for name, track_running_stats, momentum, eps in snapshot.bn_runtime:
            module = modules.get(name)
            assert isinstance(module, nn.modules.batchnorm._BatchNorm)
            module.track_running_stats = track_running_stats
            module.momentum = momentum
            module.eps = eps

        for name, source_gradient in snapshot.parameter_gradients:
            parameters[name].grad = (
                None if source_gradient is None else source_gradient.detach().clone()
            )

        live_hooks = dict(self._extra_hooks)
        for name, source_state in snapshot.extra_states:
            live_hooks[name].restore(deepcopy(source_state))

        return self.assert_source_state() if verify else self.current_fingerprint()


__all__ = [
    "EpisodicStateManager",
    "SourceSnapshotError",
    "SourceStateMismatchError",
    "StateFingerprint",
    "StatefulHooks",
]
