"""Fresh all-BN candidate construction for the D0-v2 engineering smoke.

Every call builds a new NS-FPN model, securely loads the frozen checkpoint
bytes, creates a new Binary TENT method/optimizer, and seals a new episodic
state manager.  The module performs no dataset access and no publication.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any

import torch
from torch import nn

from analysis.d0_v2_independent_candidate_contract import CandidateSpec
from tta.binary_tent import (
    CUDA_BACKWARD_TEMPORARILY_DISABLE,
    BinaryTentMethod,
)
from tta.binary_tent_fast_runner_v2 import BinaryTentFastRunnerV2
from tta.d0_secure_io import StableFileSnapshot, read_stable_regular_file
from tta.d0_v2_parameter_groups import (
    build_d0_v2_fine_group_inventory,
    verify_frozen_d0_v2_fine_inventory,
)
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class D0V2CandidateBuildError(RuntimeError):
    """A fresh candidate could not be bound to the frozen Source model."""


@dataclass(frozen=True)
class D0V2FreshCandidate:
    candidate: CandidateSpec
    model: nn.Module
    adapter: IRSTDModelAdapter
    method: BinaryTentMethod
    state_manager: EpisodicStateManager
    runner: BinaryTentFastRunnerV2
    parameter_names: tuple[str, ...]
    checkpoint_wrapper: str
    checkpoint_sha256: str
    source_model_sha256: str
    source_runtime_sha256: str
    source_topology_sha256: str


def _same_snapshot_identity(
    left: StableFileSnapshot, right: StableFileSnapshot
) -> bool:
    return (
        left.sha256,
        left.device,
        left.inode,
        left.mode,
        left.link_count,
        left.size_bytes,
        left.mtime_ns,
        left.ctime_ns,
    ) == (
        right.sha256,
        right.device,
        right.inode,
        right.mode,
        right.link_count,
        right.size_bytes,
        right.mtime_ns,
        right.ctime_ns,
    )


def _load_checkpoint_bytes_exact(
    model: nn.Module,
    *,
    checkpoint_path: Path,
    expected_sha256: str,
) -> tuple[str, StableFileSnapshot]:
    """Stable no-follow read followed by exact key/shape checkpoint loading."""

    import test_source as source_runner

    try:
        before = read_stable_regular_file(checkpoint_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise D0V2CandidateBuildError(
            f"cannot securely read checkpoint: {checkpoint_path}"
        ) from exc
    if before.sha256 != expected_sha256:
        raise D0V2CandidateBuildError(
            "checkpoint SHA-256 differs from the D0-v2 config"
        )

    buffer = io.BytesIO(before.data)
    try:
        try:
            value = torch.load(buffer, map_location="cpu", weights_only=True)
        except TypeError:
            buffer.seek(0)
            value = torch.load(buffer, map_location="cpu")
        state_dict, wrapper = source_runner.extract_state_dict(value)
        expected_state = model.state_dict()
        missing = sorted(set(expected_state) - set(state_dict))
        unexpected = sorted(set(state_dict) - set(expected_state))
        if missing or unexpected:
            raise D0V2CandidateBuildError(
                "checkpoint keys differ from NS-FPN; "
                f"missing={missing}, unexpected={unexpected}"
            )
        shape_mismatches = {
            name: (tuple(state_dict[name].shape), tuple(expected_state[name].shape))
            for name in expected_state
            if tuple(state_dict[name].shape) != tuple(expected_state[name].shape)
        }
        if shape_mismatches:
            raise D0V2CandidateBuildError(
                f"checkpoint shapes differ from NS-FPN: {shape_mismatches}"
            )
        model.load_state_dict(state_dict, strict=True)
    except D0V2CandidateBuildError:
        raise
    except (RuntimeError, TypeError, ValueError) as exc:
        raise D0V2CandidateBuildError("checkpoint load failed closed") from exc
    finally:
        buffer.close()

    try:
        after = read_stable_regular_file(checkpoint_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise D0V2CandidateBuildError(
            "checkpoint could not be revalidated after loading"
        ) from exc
    if not _same_snapshot_identity(before, after):
        raise D0V2CandidateBuildError("checkpoint changed while being loaded")
    return wrapper, before


def build_fresh_d0_v2_candidate(
    *,
    project_root: Path,
    dataset_config: Mapping[str, Any],
    candidate: CandidateSpec,
    device: torch.device,
    entropy_eps: float,
    diagnostic_detail: str,
) -> D0V2FreshCandidate:
    """Build one independent Source model/method/optimizer/runner tuple."""

    if not isinstance(project_root, Path) or not project_root.is_absolute():
        raise D0V2CandidateBuildError("project_root must be an absolute Path")
    if not isinstance(dataset_config, Mapping):
        raise D0V2CandidateBuildError("dataset_config must be a mapping")
    if not isinstance(candidate, CandidateSpec):
        raise D0V2CandidateBuildError("candidate must be CandidateSpec")
    if not isinstance(device, torch.device) or device.type != "cuda":
        raise D0V2CandidateBuildError(
            "formal engineering smoke candidate requires a CUDA device"
        )
    expected_fields = {
        "train_split_sha256",
        "checkpoint_role",
        "checkpoint_path",
        "checkpoint_sha256",
    }
    missing = sorted(expected_fields - set(dataset_config))
    unknown = sorted(set(dataset_config) - expected_fields)
    if missing or unknown:
        raise D0V2CandidateBuildError(
            "dataset config fields must be exact; "
            f"missing={missing}, unknown={unknown}"
        )
    if dataset_config["checkpoint_role"] != "best_miou":
        raise D0V2CandidateBuildError("smoke checkpoint role must be best_miou")
    relative = dataset_config["checkpoint_path"]
    if not isinstance(relative, str) or not relative:
        raise D0V2CandidateBuildError("checkpoint_path must be non-empty")
    checkpoint_path = Path(project_root, relative)
    if not checkpoint_path.absolute().is_relative_to(project_root):
        raise D0V2CandidateBuildError("checkpoint path escapes project root")
    expected_sha256 = dataset_config["checkpoint_sha256"]
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise D0V2CandidateBuildError("checkpoint_sha256 is invalid")

    import test_source as source_runner

    model = source_runner.build_nsfpn_model()
    wrapper, checkpoint_snapshot = _load_checkpoint_bytes_exact(
        model,
        checkpoint_path=checkpoint_path,
        expected_sha256=expected_sha256,
    )
    model.to(device)
    inventory = build_d0_v2_fine_group_inventory(model)
    verify_frozen_d0_v2_fine_inventory(inventory)

    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise D0V2CandidateBuildError(
            "Source model must begin with every parameter frozen"
        )
    method = BinaryTentMethod.from_adapter(
        adapter,
        optimizer_name=candidate.optimizer,
        learning_rate=float(candidate.learning_rate),
        bn_protocol="source_running_statistics",
        entropy_eps=float(entropy_eps),
        diagnostic_detail=diagnostic_detail,
        cuda_backward_determinism_policy=CUDA_BACKWARD_TEMPORARILY_DISABLE,
    )
    if method.optimizer.state:
        raise D0V2CandidateBuildError(
            "fresh candidate optimizer state must be empty"
        )
    parameters = tuple(
        parameter
        for group in method.optimizer.param_groups
        for parameter in group["params"]
    )
    if len(parameters) != 106 or len(method.parameter_names) != 106:
        raise D0V2CandidateBuildError(
            "fresh candidate must bind exactly 106 all-BN affine tensors"
        )
    state_manager = EpisodicStateManager(model, optimizer=method.optimizer)
    source = state_manager.source_fingerprint
    runner = BinaryTentFastRunnerV2(
        adapter,
        state_manager,
        method,
        full_audit_cadence=None,
        require_nonzero_parameter_update=False,
    )
    state_manager.assert_source_state()
    return D0V2FreshCandidate(
        candidate=candidate,
        model=model,
        adapter=adapter,
        method=method,
        state_manager=state_manager,
        runner=runner,
        parameter_names=tuple(method.parameter_names),
        checkpoint_wrapper=wrapper,
        checkpoint_sha256=checkpoint_snapshot.sha256,
        source_model_sha256=source.model_sha256,
        source_runtime_sha256=source.runtime_sha256,
        source_topology_sha256=source.topology_sha256,
    )


__all__ = [
    "D0V2CandidateBuildError",
    "D0V2FreshCandidate",
    "build_fresh_d0_v2_candidate",
]
