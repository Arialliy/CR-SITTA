"""One-step binary TENT for single-image episodic IRSTD adaptation.

This module is intentionally isolated from the frozen AdaBN implementation.
It follows the upstream TENT optimisation pattern while replacing multiclass
softmax entropy with sigmoid binary entropy and exposing the update-time
prediction separately from the Source and post-update predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer, SGD

from tta.episodic_runner import AdaptationOutcome, EpisodeProtocolError
from tta.model_adapter import IRSTDModelAdapter


BN_PROTOCOL_BATCH_STATS = "single_image_spatial_batch_stats"
BN_PROTOCOL_SOURCE_STATS = "source_running_statistics"
SUPPORTED_BN_PROTOCOLS = frozenset(
    {BN_PROTOCOL_BATCH_STATS, BN_PROTOCOL_SOURCE_STATS}
)
CUDA_BACKWARD_REQUIRE_DISABLED = "require_disabled"
CUDA_BACKWARD_TEMPORARILY_DISABLE = "temporarily_disable"
SUPPORTED_CUDA_BACKWARD_POLICIES = frozenset(
    {CUDA_BACKWARD_REQUIRE_DISABLED, CUDA_BACKWARD_TEMPORARILY_DISABLE}
)


class BinaryTentProtocolError(EpisodeProtocolError):
    """Raised when the one-step Binary TENT contract is violated."""


def binary_entropy_map(logits: Tensor, *, eps: float = 1e-6) -> Tensor:
    """Return elementwise Bernoulli entropy for ``[B,1,H,W]`` logits.

    A one-channel softmax is identically one and therefore invalid for this
    task.  The clamp follows the pre-registered v1 loss definition and keeps
    logarithms finite for saturated predictions.
    """

    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("binary TENT logits must have shape [B,1,H,W]")
    if not torch.is_floating_point(logits):
        raise TypeError("binary TENT logits must be floating point")
    if not math.isfinite(eps) or eps <= 0.0 or eps >= 0.5:
        raise ValueError("eps must be finite and lie strictly between 0 and 0.5")
    if not torch.isfinite(logits).all():
        raise BinaryTentProtocolError("binary TENT received NaN/Inf logits")

    probability = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )
    if not torch.isfinite(entropy).all():
        raise BinaryTentProtocolError("binary entropy produced NaN/Inf")
    return entropy


def _normalise_parameters(
    parameters: Sequence[nn.Parameter],
) -> tuple[nn.Parameter, ...]:
    values = tuple(parameters)
    if not values:
        raise ValueError("binary TENT requires at least one adaptable parameter")
    if not all(isinstance(parameter, nn.Parameter) for parameter in values):
        raise TypeError("all adaptable values must be torch Parameters")
    if len({id(parameter) for parameter in values}) != len(values):
        raise ValueError("adaptable parameter sequence contains duplicates")
    if not all(torch.is_floating_point(parameter) for parameter in values):
        raise TypeError("all adaptable parameters must be floating point")
    return values


def build_binary_tent_optimizer(
    parameters: Sequence[nn.Parameter],
    *,
    name: Literal["Adam", "SGD"],
    learning_rate: float,
) -> Optimizer:
    """Build one of the two fully specified source-calibration candidates."""

    values = _normalise_parameters(parameters)
    if isinstance(learning_rate, bool) or not isinstance(
        learning_rate, (int, float)
    ):
        raise TypeError("learning_rate must be numeric")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")

    if name == "Adam":
        return Adam(
            values,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
            amsgrad=False,
            foreach=False,
            maximize=False,
            capturable=False,
            differentiable=False,
            fused=False,
        )
    if name == "SGD":
        return SGD(
            values,
            lr=learning_rate,
            momentum=0.9,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=True,
            maximize=False,
            foreach=False,
            differentiable=False,
        )
    raise ValueError("optimizer name must be exactly 'Adam' or 'SGD'")


def _tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _norm_report(
    names: Sequence[str],
    values: Sequence[Tensor | None],
) -> tuple[float, dict[str, float], int]:
    """Compute all per-tensor norms with one device-to-host synchronization."""

    if len(names) != len(values):
        raise ValueError("norm names and tensors must have the same length")
    if not values:
        return 0.0, {}, 0

    devices = {value.device for value in values if value is not None}
    if len(devices) > 1:
        raise BinaryTentProtocolError(
            "adaptable tensors unexpectedly span multiple devices"
        )
    device = next(iter(devices), torch.device("cpu"))
    components = torch.stack(
        [
            torch.zeros((), dtype=torch.float64, device=device)
            if value is None
            else torch.linalg.vector_norm(value.detach().to(dtype=torch.float64))
            for value in values
        ]
    )
    components_cpu = components.detach().cpu()
    component_values = [float(value) for value in components_cpu.tolist()]
    by_name = dict(zip(names, component_values, strict=True))
    global_norm = float(torch.linalg.vector_norm(components_cpu).item())
    nonzero_count = sum(value > 0.0 for value in component_values)
    return global_norm, by_name, nonzero_count


def _registered_buffers(
    model: nn.Module,
) -> tuple[tuple[str, Tensor | None], ...]:
    """Return every registered buffer, including None and non-persistent ones."""

    values: list[tuple[str, Tensor | None]] = []
    for module_name, module in model.named_modules():
        prefix = f"{module_name}." if module_name else ""
        for local_name, value in sorted(module._buffers.items()):
            values.append((f"{prefix}{local_name}", value))
    return tuple(values)


def _clone_named_tensors(
    values: Sequence[tuple[str, Tensor | None]],
) -> tuple[tuple[str, Tensor | None], ...]:
    return tuple(
        (name, None if value is None else value.detach().clone())
        for name, value in values
    )


def _byte_equal_flag(current: Tensor, expected: Tensor) -> Tensor:
    if (
        current.shape != expected.shape
        or current.dtype != expected.dtype
        or current.device != expected.device
        or current.layout != expected.layout
    ):
        return torch.zeros((), dtype=torch.bool, device=current.device)
    if current.layout != torch.strided:
        current = current.to_dense()
        expected = expected.to_dense()
    current_bytes = current.detach().resolve_conj().resolve_neg().contiguous()
    expected_bytes = expected.detach().resolve_conj().resolve_neg().contiguous()
    return torch.eq(
        current_bytes.reshape(-1).view(torch.uint8),
        expected_bytes.reshape(-1).view(torch.uint8),
    ).all()


def _assert_named_tensors_unchanged(
    *,
    current: Sequence[tuple[str, Tensor | None]],
    expected: Sequence[tuple[str, Tensor | None]],
    label: str,
) -> None:
    """Fail closed if a named resident tensor differs byte-for-byte."""

    current_by_name = dict(current)
    expected_by_name = dict(expected)
    if tuple(current_by_name) != tuple(expected_by_name):
        raise BinaryTentProtocolError(f"{label} topology changed during adaptation")

    flags_by_device: dict[torch.device, list[Tensor]] = {}
    topology_failures: list[str] = []
    for name, expected_value in expected:
        current_value = current_by_name[name]
        if expected_value is None or current_value is None:
            if expected_value is not current_value:
                topology_failures.append(name)
            continue
        if (
            current_value.shape != expected_value.shape
            or current_value.dtype != expected_value.dtype
            or current_value.device != expected_value.device
            or current_value.layout != expected_value.layout
        ):
            topology_failures.append(name)
            continue
        flags_by_device.setdefault(current_value.device, []).append(
            _byte_equal_flag(current_value, expected_value)
        )

    if topology_failures:
        raise BinaryTentProtocolError(
            f"{label} topology/value kind changed: "
            + ", ".join(topology_failures[:10])
        )
    for flags in flags_by_device.values():
        if flags and not bool(torch.stack(flags).all().item()):
            changed = []
            for name, expected_value in expected:
                current_value = current_by_name[name]
                if expected_value is None or current_value is None:
                    continue
                if not bool(_byte_equal_flag(current_value, expected_value).item()):
                    changed.append(name)
            raise BinaryTentProtocolError(
                f"{label} changed during adaptation: " + ", ".join(changed[:10])
            )


def _autocast_enabled() -> bool:
    return bool(torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled())


def _json_safe_optimizer_group(group: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sorted(group.items()):
        if key == "params":
            continue
        if isinstance(value, tuple):
            result[key] = list(value)
        elif isinstance(value, (str, bool, int, float)) or value is None:
            result[key] = value
        else:
            result[key] = repr(value)
    return result


@dataclass(frozen=True)
class BinaryTentOutcome(AdaptationOutcome):
    """Adaptation outcome carrying the actual update-time TENT prediction."""

    logits_tent_pre: Tensor

    def __post_init__(self) -> None:
        super().__post_init__()
        value = self.logits_tent_pre
        if not isinstance(value, Tensor):
            raise TypeError("logits_tent_pre must be a torch.Tensor")
        if value.ndim != 4 or value.shape[:2] != (1, 1):
            raise ValueError("logits_tent_pre must have shape [1,1,H,W]")
        if not torch.isfinite(value).all():
            raise BinaryTentProtocolError("logits_tent_pre contains NaN/Inf")
        object.__setattr__(self, "logits_tent_pre", value.detach().cpu().clone())


class BinaryTentMethod:
    """Exactly one unconditional binary-entropy update of all BN affine terms."""

    name = "binary_episodic_tent"
    requires_grad = True
    allowed_state_changes = frozenset({"model", "optimizer", "runtime"})
    state_frozen_after_prepare = False

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        parameter_names: Sequence[str],
        bn_protocol: Literal[
            "single_image_spatial_batch_stats", "source_running_statistics"
        ],
        entropy_eps: float = 1e-6,
        steps: int = 1,
        diagnostic_detail: Literal["global", "per_parameter"] = "global",
        cuda_backward_determinism_policy: Literal[
            "require_disabled", "temporarily_disable"
        ] = CUDA_BACKWARD_REQUIRE_DISABLED,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch Optimizer")
        if bn_protocol not in SUPPORTED_BN_PROTOCOLS:
            raise ValueError(f"unsupported BN protocol: {bn_protocol!r}")
        if steps != 1:
            raise ValueError("Binary TENT v1 is frozen to exactly one step")
        if not math.isfinite(entropy_eps) or entropy_eps <= 0.0 or entropy_eps >= 0.5:
            raise ValueError("entropy_eps must lie strictly between 0 and 0.5")
        if diagnostic_detail not in ("global", "per_parameter"):
            raise ValueError(
                "diagnostic_detail must be exactly 'global' or 'per_parameter'"
            )
        if cuda_backward_determinism_policy not in SUPPORTED_CUDA_BACKWARD_POLICIES:
            raise ValueError(
                "unsupported CUDA backward determinism policy: "
                f"{cuda_backward_determinism_policy!r}"
            )

        optimizer_parameters = tuple(
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        _normalise_parameters(optimizer_parameters)
        names = tuple(parameter_names)
        if len(names) != len(optimizer_parameters):
            raise ValueError("parameter_names must match optimizer parameter count")
        if not all(isinstance(name, str) and name for name in names):
            raise ValueError("parameter_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("parameter_names must be unique")

        self.optimizer = optimizer
        self.parameter_names = names
        self.bn_protocol = bn_protocol
        self.entropy_eps = float(entropy_eps)
        self.steps = steps
        self.diagnostic_detail = diagnostic_detail
        self.cuda_backward_determinism_policy = cuda_backward_determinism_policy
        self._bound_model_id: int | None = None
        self._source_non_adaptable_parameters: tuple[
            tuple[str, Tensor | None], ...
        ] = ()
        self._source_registered_buffers: tuple[tuple[str, Tensor | None], ...] = ()

    @classmethod
    def from_adapter(
        cls,
        adapter: IRSTDModelAdapter,
        *,
        optimizer_name: Literal["Adam", "SGD"],
        learning_rate: float,
        bn_protocol: Literal[
            "single_image_spatial_batch_stats", "source_running_statistics"
        ],
        entropy_eps: float = 1e-6,
        diagnostic_detail: Literal["global", "per_parameter"] = "global",
        cuda_backward_determinism_policy: Literal[
            "require_disabled", "temporarily_disable"
        ] = CUDA_BACKWARD_REQUIRE_DISABLED,
    ) -> "BinaryTentMethod":
        parameters, names = adapter.collect_adaptable_params()
        optimizer = build_binary_tent_optimizer(
            parameters,
            name=optimizer_name,
            learning_rate=learning_rate,
        )
        return cls(
            optimizer,
            parameter_names=names,
            bn_protocol=bn_protocol,
            entropy_eps=entropy_eps,
            steps=1,
            diagnostic_detail=diagnostic_detail,
            cuda_backward_determinism_policy=cuda_backward_determinism_policy,
        )

    @property
    def use_batch_stats(self) -> bool:
        return self.bn_protocol == BN_PROTOCOL_BATCH_STATS

    def _optimizer_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        )

    def _validate_and_bind_adapter(self, adapter: IRSTDModelAdapter) -> None:
        parameters, names = adapter.collect_adaptable_params()
        expected_parameters = tuple(parameters)
        expected_names = tuple(names)
        optimizer_parameters = self._optimizer_parameters()
        if expected_names != self.parameter_names:
            raise BinaryTentProtocolError(
                "parameter_names do not exactly match adapter BN affine order"
            )
        if tuple(map(id, expected_parameters)) != tuple(map(id, optimizer_parameters)):
            raise BinaryTentProtocolError(
                "optimizer parameters do not exactly match adapter BN affine order"
            )

        model_id = id(adapter.model)
        adaptable_ids = {id(parameter) for parameter in expected_parameters}
        current_non_adaptable = tuple(
            (name, parameter)
            for name, parameter in adapter.model.named_parameters()
            if id(parameter) not in adaptable_ids
        )
        current_buffers = _registered_buffers(adapter.model)
        if self._bound_model_id is None:
            self._bound_model_id = model_id
            self._source_non_adaptable_parameters = _clone_named_tensors(
                current_non_adaptable
            )
            self._source_registered_buffers = _clone_named_tensors(current_buffers)
        elif self._bound_model_id != model_id:
            raise BinaryTentProtocolError(
                "BinaryTentMethod cannot be rebound to a different model"
            )
        else:
            _assert_named_tensors_unchanged(
                current=current_non_adaptable,
                expected=self._source_non_adaptable_parameters,
                label="non-adaptable Source parameters",
            )
            _assert_named_tensors_unchanged(
                current=current_buffers,
                expected=self._source_registered_buffers,
                label="registered Source buffers",
            )

    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        self._validate_and_bind_adapter(adapter)
        adapter.set_tent_mode(use_batch_stats=self.use_batch_stats)
        trainable_ids = {
            id(parameter)
            for parameter in adapter.model.parameters()
            if parameter.requires_grad
        }
        optimizer_ids = {id(parameter) for parameter in self._optimizer_parameters()}
        if trainable_ids != optimizer_ids:
            raise BinaryTentProtocolError(
                "exactly the optimizer-bound BN affine parameters must be trainable"
            )

    def adapt_one_image(
        self,
        *,
        adapter: IRSTDModelAdapter,
        image: Tensor,
        logits_pre: Tensor,
        metadata: Mapping[str, Any],
    ) -> BinaryTentOutcome:
        del metadata
        if logits_pre.ndim != 4 or logits_pre.shape[:2] != (1, 1):
            raise BinaryTentProtocolError("Source logits must have shape [1,1,H,W]")
        if image.ndim != 4 or image.shape[0] != 1:
            raise BinaryTentProtocolError("Binary TENT requires exactly one image")
        if _autocast_enabled():
            raise BinaryTentProtocolError(
                "Binary TENT v1 forbids CUDA and CPU autocast/AMP"
            )
        if image.dtype != torch.float32 or logits_pre.dtype != torch.float32:
            raise BinaryTentProtocolError(
                "Binary TENT v1 requires float32 image and Source logits"
            )
        if image.device != logits_pre.device:
            raise BinaryTentProtocolError(
                "image and Source logits must reside on the same device"
            )
        deterministic_before = bool(torch.are_deterministic_algorithms_enabled())
        warn_only_before = bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        if image.is_cuda:
            if (
                self.cuda_backward_determinism_policy
                == CUDA_BACKWARD_REQUIRE_DISABLED
                and deterministic_before
            ):
                raise BinaryTentProtocolError(
                    "CUDA Binary TENT requires deterministic algorithms to be "
                    "disabled under the require_disabled policy"
                )
            if (
                self.cuda_backward_determinism_policy
                == CUDA_BACKWARD_TEMPORARILY_DISABLE
                and (not deterministic_before or warn_only_before)
            ):
                raise BinaryTentProtocolError(
                    "the temporarily_disable policy requires strict deterministic "
                    "algorithms for Source/TENT-pre/TENT-post forwards"
                )

        parameters = self._optimizer_parameters()
        if any(parameter.dtype != torch.float32 for parameter in parameters):
            raise BinaryTentProtocolError(
                "all adaptable BN affine parameters must be float32"
            )
        if any(parameter.device != image.device for parameter in parameters):
            raise BinaryTentProtocolError(
                "all adaptable BN affine parameters must share the image device"
            )
        if not all(parameter.requires_grad for parameter in parameters):
            raise BinaryTentProtocolError(
                "all optimizer-bound BN affine parameters must require gradients"
            )
        if any(parameter.grad is not None for parameter in parameters):
            raise BinaryTentProtocolError("episode began with uncleared gradients")

        before = tuple(parameter.detach().clone() for parameter in parameters)
        self.optimizer.zero_grad(set_to_none=True)
        logits_tent_pre = adapter.forward_logits(image)
        if (
            logits_tent_pre.dtype != torch.float32
            or logits_tent_pre.device != image.device
        ):
            raise BinaryTentProtocolError(
                "TENT-pre logits must be float32 on the image device"
            )
        entropy_pre_tensor = binary_entropy_map(
            logits_tent_pre,
            eps=self.entropy_eps,
        ).mean()
        if not torch.isfinite(entropy_pre_tensor):
            raise BinaryTentProtocolError("mean binary entropy is NaN/Inf")
        backward_deterministic = deterministic_before
        backward_warn_only = warn_only_before
        if (
            image.is_cuda
            and self.cuda_backward_determinism_policy
            == CUDA_BACKWARD_TEMPORARILY_DISABLE
        ):
            torch.use_deterministic_algorithms(False)
            try:
                backward_deterministic = bool(
                    torch.are_deterministic_algorithms_enabled()
                )
                backward_warn_only = bool(
                    torch.is_deterministic_algorithms_warn_only_enabled()
                )
                if backward_deterministic:
                    raise BinaryTentProtocolError(
                        "failed to disable deterministic algorithms for CUDA backward"
                    )
                entropy_pre_tensor.backward()
            finally:
                torch.use_deterministic_algorithms(
                    deterministic_before,
                    warn_only=warn_only_before,
                )
        else:
            entropy_pre_tensor.backward()
        deterministic_restored = (
            bool(torch.are_deterministic_algorithms_enabled())
            == deterministic_before
            and bool(torch.is_deterministic_algorithms_warn_only_enabled())
            == warn_only_before
        )
        if not deterministic_restored:
            raise BinaryTentProtocolError(
                "CUDA backward determinism policy was not restored exactly"
            )

        gradient_values: list[Tensor | None] = []
        for name, parameter in zip(self.parameter_names, parameters, strict=True):
            gradient = parameter.grad
            if gradient is None:
                gradient_values.append(None)
                continue
            if not torch.isfinite(gradient).all():
                raise BinaryTentProtocolError(f"gradient contains NaN/Inf: {name}")
            gradient_values.append(gradient)
        gradient_norm, gradient_norms, nonzero_gradient_tensors = _norm_report(
            self.parameter_names,
            gradient_values,
        )
        if not math.isfinite(gradient_norm):
            raise BinaryTentProtocolError("global gradient norm is NaN/Inf")

        self.optimizer.step()

        deltas: list[Tensor] = []
        for name, parameter, source in zip(
            self.parameter_names, parameters, before, strict=True
        ):
            if not torch.isfinite(parameter).all():
                raise BinaryTentProtocolError(
                    f"updated parameter contains NaN/Inf: {name}"
                )
            delta = parameter.detach() - source
            deltas.append(delta)
        step_norm, step_norms, changed_parameter_tensors = _norm_report(
            self.parameter_names,
            deltas,
        )
        if not math.isfinite(step_norm):
            raise BinaryTentProtocolError("parameter step norm is NaN/Inf")

        adaptable_ids = {id(parameter) for parameter in parameters}
        _assert_named_tensors_unchanged(
            current=tuple(
                (name, parameter)
                for name, parameter in adapter.model.named_parameters()
                if id(parameter) not in adaptable_ids
            ),
            expected=self._source_non_adaptable_parameters,
            label="non-adaptable parameters",
        )
        _assert_named_tensors_unchanged(
            current=_registered_buffers(adapter.model),
            expected=self._source_registered_buffers,
            label="registered buffers",
        )

        self.optimizer.zero_grad(set_to_none=True)
        if any(parameter.grad is not None for parameter in parameters):
            raise BinaryTentProtocolError("optimizer did not clear gradients to None")

        optimizer_groups = tuple(
            _json_safe_optimizer_group(group) for group in self.optimizer.param_groups
        )
        diagnostics = {
            "bn_protocol": self.bn_protocol,
            "loss": "full_image_mean_binary_sigmoid_entropy",
            "entropy_eps": self.entropy_eps,
            "optimization_entropy_pre_device": float(
                entropy_pre_tensor.detach().cpu().item()
            ),
            "gradient_norm": gradient_norm,
            "step_norm": step_norm,
            "number_trainable_tensors": len(parameters),
            "number_trainable_scalars": sum(
                parameter.numel() for parameter in parameters
            ),
            "number_gradient_tensors": sum(
                gradient is not None for gradient in gradient_values
            ),
            "number_nonzero_gradient_tensors": nonzero_gradient_tensors,
            "number_updated_bn_affine_tensors": changed_parameter_tensors,
            "parameter_names_sha256": hashlib.sha256(
                "\0".join(self.parameter_names).encode("utf-8")
            ).hexdigest(),
            "optimizer": type(self.optimizer).__name__,
            "optimizer_param_groups": optimizer_groups,
            "optimizer_steps": 1,
            "update_accepted": True,
            "actual_parameter_delta_nonzero": step_norm > 0.0,
            "gradient_nonzero": gradient_norm > 0.0,
            "finite": True,
            "amp_enabled": False,
            "numeric_precision": "float32",
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "forward_deterministic_algorithms_enabled": deterministic_before,
            "forward_deterministic_algorithms_warn_only": warn_only_before,
            "backward_deterministic_algorithms_enabled": backward_deterministic,
            "backward_deterministic_algorithms_warn_only": backward_warn_only,
            "episode_device_type": image.device.type,
            "cuda_backward_determinism_policy": (
                self.cuda_backward_determinism_policy
            ),
            "deterministic_policy_restored_after_backward": (
                deterministic_restored
            ),
            "diagnostic_detail": self.diagnostic_detail,
            "optimizer_params_exact_all_bn_affine": True,
            "only_bn_affine_requires_grad": True,
            "non_adaptable_parameters_resident_bit_exact": True,
            "registered_buffers_resident_bit_exact": True,
            "logits_tent_pre_raw_sha256": _tensor_sha256(logits_tent_pre),
        }
        if self.diagnostic_detail == "per_parameter":
            diagnostics.update(
                {
                    "trainable_parameter_names": self.parameter_names,
                    "gradient_norm_by_parameter": gradient_norms,
                    "step_norm_by_parameter": step_norms,
                }
            )
        return BinaryTentOutcome(
            decision="adapted",
            optimizer_steps=1,
            diagnostics=diagnostics,
            logits_tent_pre=logits_tent_pre,
        )


__all__ = [
    "BN_PROTOCOL_BATCH_STATS",
    "BN_PROTOCOL_SOURCE_STATS",
    "CUDA_BACKWARD_REQUIRE_DISABLED",
    "CUDA_BACKWARD_TEMPORARILY_DISABLE",
    "BinaryTentMethod",
    "BinaryTentOutcome",
    "BinaryTentProtocolError",
    "SUPPORTED_BN_PROTOCOLS",
    "SUPPORTED_CUDA_BACKWARD_POLICIES",
    "binary_entropy_map",
    "build_binary_tent_optimizer",
]
