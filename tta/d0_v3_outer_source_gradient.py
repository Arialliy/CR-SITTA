"""Source-task gradient worker for the train-only formal P3 outer evaluator.

This module never constructs an adaptation method or optimizer.  It builds a
fresh frozen NS-FPN Source model, enables gradients only for the frozen set of
BatchNorm affine tensors, computes one BCE+soft-IoU diagnostic gradient from a
Pilot64 train target, and restores the complete Source state.  The returned
gradient is evidence for alignment only and cannot update model parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from analysis.d0_v2_task_loss import D0V2TaskLossConfig, compute_d0_v2_task_loss
from analysis.d0_v3_outer_analyzer import FlatParameterLayout
from analysis.source_train_provenance import SourceTrainAnalysisProvenance
from tta.d0_secure_io import read_stable_regular_file
from tta.d0_v2_parameter_groups import (
    build_d0_v2_fine_group_inventory,
    verify_frozen_d0_v2_fine_inventory,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class D0V3OuterSourceGradientError(RuntimeError):
    """The formal Source-task gradient could not be produced safely."""


@dataclass(frozen=True)
class D0V3OuterSourceModel:
    model: nn.Module
    adapter: IRSTDModelAdapter
    state_manager: EpisodicStateManager
    parameters: tuple[nn.Parameter, ...]
    parameter_names: tuple[str, ...]
    layout: FlatParameterLayout
    source_parameters_flat: Tensor
    fine_group_assignment: Mapping[str, str]
    checkpoint_wrapper: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class D0V3OuterGradientResult:
    source_logits: Tensor
    supervised_gradient_flat: Tensor
    task_loss_audit: Mapping[str, Any]
    source_state_sha256: str
    reset_source_state_sha256: str


def _same_snapshot(left: Any, right: Any) -> bool:
    fields = (
        "sha256",
        "device",
        "inode",
        "mode",
        "link_count",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _load_checkpoint(
    model: nn.Module,
    *,
    path: Path,
    expected_sha256: str,
) -> tuple[str, str]:
    import test_source as source_runner

    before = read_stable_regular_file(path)
    if before.sha256 != expected_sha256:
        raise D0V3OuterSourceGradientError("checkpoint SHA-256 differs")
    buffer = io.BytesIO(before.data)
    try:
        try:
            checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)
        except TypeError:
            buffer.seek(0)
            checkpoint = torch.load(buffer, map_location="cpu")
        state, wrapper = source_runner.extract_state_dict(checkpoint)
        expected = model.state_dict()
        if set(state) != set(expected):
            raise D0V3OuterSourceGradientError("checkpoint key topology differs")
        mismatches = {
            key: (tuple(state[key].shape), tuple(expected[key].shape))
            for key in expected
            if tuple(state[key].shape) != tuple(expected[key].shape)
        }
        if mismatches:
            raise D0V3OuterSourceGradientError(
                f"checkpoint tensor shapes differ: {mismatches}"
            )
        model.load_state_dict(state, strict=True)
    except D0V3OuterSourceGradientError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise D0V3OuterSourceGradientError("checkpoint load failed") from exc
    finally:
        buffer.close()
    after = read_stable_regular_file(path)
    if not _same_snapshot(before, after):
        raise D0V3OuterSourceGradientError("checkpoint changed while loading")
    return wrapper, before.sha256


def build_d0_v3_outer_source_model(
    *,
    project_root: Path,
    checkpoint_path: str,
    checkpoint_sha256: str,
    device: torch.device,
) -> D0V3OuterSourceModel:
    """Build a Source-only oracle-gradient model with no optimizer."""

    if not isinstance(project_root, Path) or not project_root.is_absolute():
        raise D0V3OuterSourceGradientError("project_root must be absolute")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise D0V3OuterSourceGradientError("checkpoint_path must be non-empty")
    relative = Path(checkpoint_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise D0V3OuterSourceGradientError(
            "checkpoint_path must be canonical project-relative"
        )
    path = project_root / relative
    if not path.absolute().is_relative_to(project_root):
        raise D0V3OuterSourceGradientError("checkpoint_path escapes project root")
    if not isinstance(device, torch.device) or device.type not in {"cpu", "cuda"}:
        raise D0V3OuterSourceGradientError("device must be CPU or CUDA")

    import test_source as source_runner

    model = source_runner.build_nsfpn_model()
    wrapper, observed_sha256 = _load_checkpoint(
        model, path=path, expected_sha256=checkpoint_sha256
    )
    model.to(device)
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    state_manager = EpisodicStateManager(model)
    state_manager.assert_source_state()
    inventory = build_d0_v2_fine_group_inventory(model)
    verify_frozen_d0_v2_fine_inventory(inventory)

    adapter.set_tent_mode(use_batch_stats=False)
    parameters_list, names_list = adapter.collect_adaptable_params()
    parameters = tuple(parameters_list)
    names = tuple(names_list)
    if len(parameters) != 106 or len(names) != 106:
        raise D0V3OuterSourceGradientError(
            "outer Source model must expose exactly 106 BN affine tensors"
        )
    named = tuple(zip(names, parameters, strict=True))
    layout = FlatParameterLayout.from_named_tensors(named)
    if layout.scalar_count != 8736:
        raise D0V3OuterSourceGradientError(
            "outer Source BN affine scalar count differs from 8736"
        )
    source_parameters_flat = layout.pack(
        {name: parameter.detach().cpu() for name, parameter in named}
    )
    assignment: dict[str, str] = {}
    for group in inventory.groups:
        for name in group.parameter_names:
            if name in assignment:
                raise D0V3OuterSourceGradientError(
                    f"fine-group parameter assigned twice: {name}"
                )
            assignment[name] = group.group_id
    if tuple(assignment) != names:
        raise D0V3OuterSourceGradientError(
            "fine-group assignment order differs from BN parameter order"
        )
    # Restore the actual Source runtime before returning.  Each gradient call
    # enters source-stat TENT mode itself and resets back to this fingerprint.
    reset = state_manager.reset_to_source()
    if reset != state_manager.source_fingerprint:
        raise D0V3OuterSourceGradientError("initial Source reset differs")
    return D0V3OuterSourceModel(
        model=model,
        adapter=adapter,
        state_manager=state_manager,
        parameters=parameters,
        parameter_names=names,
        layout=layout,
        source_parameters_flat=source_parameters_flat,
        fine_group_assignment=assignment,
        checkpoint_wrapper=wrapper,
        checkpoint_sha256=observed_sha256,
    )


def compute_d0_v3_outer_source_gradient(
    worker: D0V3OuterSourceModel,
    *,
    image: Tensor,
    target: Tensor,
    task_loss_config: D0V2TaskLossConfig | Mapping[str, Any],
    provenance: SourceTrainAnalysisProvenance,
) -> D0V3OuterGradientResult:
    """Compute one train-only Source task gradient and restore Source exactly."""

    if not isinstance(worker, D0V3OuterSourceModel):
        raise D0V3OuterSourceGradientError("worker has an invalid type")
    if (
        not isinstance(image, Tensor)
        or image.ndim != 4
        or tuple(image.shape[:2]) != (1, 3)
        or image.dtype != torch.float32
    ):
        raise D0V3OuterSourceGradientError(
            "image must be float32 [1,3,H,W]"
        )
    if (
        not isinstance(target, Tensor)
        or target.ndim != 4
        or tuple(target.shape[:2]) != (1, 1)
        or target.dtype != torch.float32
        or target.shape[-2:] != image.shape[-2:]
    ):
        raise D0V3OuterSourceGradientError(
            "target must be float32 [1,1,H,W] aligned to image"
        )
    device = next(worker.model.parameters()).device
    if image.device != device or target.device != device:
        raise D0V3OuterSourceGradientError("image/target/model devices differ")
    if not bool(torch.isfinite(image).all().item()) or not bool(
        torch.isfinite(target).all().item()
    ):
        raise D0V3OuterSourceGradientError("image/target contains NaN/Inf")

    source_fingerprint = worker.state_manager.assert_source_state()
    worker.adapter.set_tent_mode(use_batch_stats=False)
    worker.model.zero_grad(set_to_none=True)
    logits = worker.adapter.forward_logits(image)
    if (
        logits.dtype != torch.float32
        or logits.shape != target.shape
        or not bool(torch.isfinite(logits).all().item())
    ):
        raise D0V3OuterSourceGradientError("outer Source logits are invalid")
    total_loss, audit = compute_d0_v2_task_loss(
        logits=logits,
        target=target,
        config=task_loss_config,
        provenance=provenance,
    )
    deterministic_before = bool(torch.are_deterministic_algorithms_enabled())
    warn_only_before = bool(
        torch.is_deterministic_algorithms_warn_only_enabled()
    )
    try:
        if device.type == "cuda":
            torch.use_deterministic_algorithms(False)
        total_loss.backward()
    finally:
        torch.use_deterministic_algorithms(
            deterministic_before, warn_only=warn_only_before
        )
    if (
        bool(torch.are_deterministic_algorithms_enabled()) != deterministic_before
        or bool(torch.is_deterministic_algorithms_warn_only_enabled())
        != warn_only_before
    ):
        raise D0V3OuterSourceGradientError(
            "determinism policy was not restored after outer backward"
        )
    gradients: dict[str, Tensor] = {}
    for name, parameter in zip(
        worker.parameter_names, worker.parameters, strict=True
    ):
        gradient = parameter.grad
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise D0V3OuterSourceGradientError(
                f"outer supervised gradient missing/non-finite: {name}"
            )
        gradients[name] = gradient.detach().cpu().contiguous().clone()
    flat = worker.layout.pack(gradients)
    source_logits = logits.detach().cpu().contiguous().clone()
    reset = worker.state_manager.reset_to_source()
    if reset != source_fingerprint or worker.state_manager.assert_source_state() != source_fingerprint:
        raise D0V3OuterSourceGradientError(
            "outer Source model did not reset exactly after diagnostic backward"
        )
    return D0V3OuterGradientResult(
        source_logits=source_logits,
        supervised_gradient_flat=flat,
        task_loss_audit=dict(audit),
        source_state_sha256=source_fingerprint.full_sha256,
        reset_source_state_sha256=reset.full_sha256,
    )


__all__ = [
    "D0V3OuterGradientResult",
    "D0V3OuterSourceGradientError",
    "D0V3OuterSourceModel",
    "build_d0_v3_outer_source_model",
    "compute_d0_v3_outer_source_gradient",
]
