#!/usr/bin/env python3
"""Fail-closed Stage-2 authorization boundary for Binary TENT-SS v3.

The public authorization command accepts no evidence path or evidence digest.
Its only trust anchor is the reviewed digest of the frozen v3 configuration in
this source file.  That configuration either blocks Stage 2 explicitly or
binds one formal Stage-1 aggregate manifest.  The aggregate verifier then
recomputes the selector-v3 receipt from the bound gate manifest, cell records,
and diagnostic evidence before any Stage-2 backend operation is reachable.

This module is deliberately CPU-only and creates no files by itself.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Protocol, TypeVar

from tta.binary_tent_ss_calibration_selector_v3 import (
    Candidate,
    STAGE1_RECEIPT_FILENAME,
    STAGE1_RECEIPT_SCHEMA_VERSION,
    STAGE1_RECEIPT_TYPE,
)
from tta.binary_tent_ss_stage1_gate_artifact_v3 import (
    FrozenGateConfigBinding,
    Stage1GateArtifactError,
    Stage1GateNotFormalError,
    VerifiedFormalStage1Gate,
    verify_formal_stage1_gate,
)


EXIT_EXECUTION_ERROR = 2
EXIT_SCIENTIFIC_GATE_BLOCKED = 3
EXIT_PROTOCOL_GATE_BLOCKED = 4

PROJECT_ROOT = Path(__file__).resolve().parent
FROZEN_AUTHORIZATION_CONFIG_PATH = (
    PROJECT_ROOT / "configs/binary_tent_ss_calibration_v3.yaml"
)
# Independent, reviewed trust anchor.  It is intentionally not read from the
# configuration, receipt directory, CLI, environment, or a mutable sidecar.
FROZEN_AUTHORIZATION_CONFIG_SHA256 = (
    "2a1524f75dd3cb945c1f1819943db34f74b1b6ae7f800754426b432a774848c7"
)


class Stage2AuthorizationError(RuntimeError):
    """Base class for failures before the Stage-2 side-effect boundary."""


class ReceiptVerificationError(Stage2AuthorizationError):
    """The frozen artifact chain or recomputed receipt is invalid."""


class Stage1ProtocolBlockedError(Stage2AuthorizationError):
    """Stage 1 did not pass its protocol gate."""


class ScientificGateBlockedError(Stage2AuthorizationError):
    """Stage 1 did not grant a formal scientific Stage-2 authorization."""


@dataclass(frozen=True, slots=True)
class Stage2Authorization:
    """Capability passed to a backend only after the full evidence chain passes."""

    config_path: Path
    config_sha256: str
    aggregate_manifest_path: Path
    aggregate_manifest_sha256: str
    frozen_gate_manifest_path: Path
    frozen_gate_manifest_sha256: str
    stage1_records_path: Path
    stage1_records_sha256: str
    diagnostic_evidence_path: Path
    diagnostic_evidence_sha256: str
    receipt_path: Path
    receipt_sha256: str
    receipt_byte_count: int
    selected_for_stage2: tuple[Candidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorization_type": "binary_tent_ss_stage2_authorization_v3",
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "aggregate_manifest_path": str(self.aggregate_manifest_path),
            "aggregate_manifest_sha256": self.aggregate_manifest_sha256,
            "frozen_gate_manifest_path": str(self.frozen_gate_manifest_path),
            "frozen_gate_manifest_sha256": self.frozen_gate_manifest_sha256,
            "stage1_records_path": str(self.stage1_records_path),
            "stage1_records_sha256": self.stage1_records_sha256,
            "diagnostic_evidence_path": str(self.diagnostic_evidence_path),
            "diagnostic_evidence_sha256": self.diagnostic_evidence_sha256,
            "receipt_path": str(self.receipt_path),
            "receipt_sha256": self.receipt_sha256,
            "receipt_byte_count": self.receipt_byte_count,
            "protocol_status": "passed",
            "scientific_status": "passed",
            "stage2_allowed": True,
            "selected_candidate_count": len(self.selected_for_stage2),
            "selected_for_stage2": [
                candidate.to_dict() for candidate in self.selected_for_stage2
            ],
        }


PathsT = TypeVar("PathsT")
LeaseT = TypeVar("LeaseT")
CudaT = TypeVar("CudaT")
ProcessIdsT = TypeVar("ProcessIdsT")
CommandsT = TypeVar("CommandsT")
LaunchResultT = TypeVar("LaunchResultT")


class Stage2LaunchBackend(
    Protocol[PathsT, LeaseT, CudaT, ProcessIdsT, CommandsT, LaunchResultT]
):
    """Fixed ordering of operations that are forbidden before authorization."""

    def create_stage2_paths(self, authorization: Stage2Authorization) -> PathsT:
        ...

    def acquire_gpu_lease(self, authorization: Stage2Authorization) -> LeaseT:
        ...

    def initialize_cuda(
        self, authorization: Stage2Authorization, lease: LeaseT
    ) -> CudaT:
        ...

    def create_process_ids(
        self, authorization: Stage2Authorization
    ) -> ProcessIdsT:
        ...

    def build_worker_commands(
        self,
        authorization: Stage2Authorization,
        paths: PathsT,
        lease: LeaseT,
        cuda_runtime: CudaT,
        process_ids: ProcessIdsT,
    ) -> CommandsT:
        ...

    def launch_workers(
        self,
        authorization: Stage2Authorization,
        paths: PathsT,
        lease: LeaseT,
        cuda_runtime: CudaT,
        process_ids: ProcessIdsT,
        commands: CommandsT,
    ) -> LaunchResultT:
        ...


def _production_config_binding() -> FrozenGateConfigBinding:
    """Return the sole production trust anchor; callers cannot replace it."""

    return FrozenGateConfigBinding(
        path=FROZEN_AUTHORIZATION_CONFIG_PATH,
        expected_sha256=FROZEN_AUTHORIZATION_CONFIG_SHA256,
        project_root=PROJECT_ROOT,
    )


def _verify_trusted_chain(
    binding: FrozenGateConfigBinding,
) -> VerifiedFormalStage1Gate:
    try:
        return verify_formal_stage1_gate(binding)
    except Stage1GateNotFormalError as exc:
        raise ScientificGateBlockedError(str(exc)) from exc
    except (Stage1GateArtifactError, OSError, RuntimeError, ValueError) as exc:
        raise ReceiptVerificationError(
            f"formal Stage-1 artifact-chain verification failed: {exc}"
        ) from exc


def _assert_stage2_gate(
    verified: VerifiedFormalStage1Gate,
) -> Stage2Authorization:
    """Apply the explicit protocol/science/permission/cardinality gates."""

    receipt: Mapping[str, Any] = verified.receipt
    if receipt.get("protocol_status") != "passed":
        raise Stage1ProtocolBlockedError(
            "Stage 1 protocol did not pass; Stage 2 is forbidden"
        )
    if receipt.get("scientific_status") != "passed":
        raise ScientificGateBlockedError(
            "Stage 1 executed successfully but failed the scientific gate; "
            "Stage 2 is forbidden"
        )
    if receipt.get("formal_fully_frozen_gate") is not True:
        raise ScientificGateBlockedError(
            "Stage-1 receipt is not backed by a formal fully frozen gate"
        )
    if receipt.get("unresolved_gate_thresholds") is not False:
        raise ScientificGateBlockedError(
            "Stage-1 receipt has unresolved gate thresholds"
        )
    if receipt.get("retrospective_negative_replay") is not False:
        raise ScientificGateBlockedError(
            "a retrospective replay cannot authorize Stage 2"
        )
    if receipt.get("stage2_allowed") is not True:
        raise ScientificGateBlockedError(
            "Stage-1 receipt does not explicitly set stage2_allowed=true"
        )
    selected = verified.selected_for_stage2
    if not 1 <= len(selected) <= 3:
        raise ScientificGateBlockedError(
            "Stage-1 receipt must authorize between one and three eligible "
            f"candidates; observed {len(selected)}"
        )
    return Stage2Authorization(
        config_path=verified.config_path,
        config_sha256=verified.config_sha256,
        aggregate_manifest_path=verified.aggregate_manifest_path,
        aggregate_manifest_sha256=verified.aggregate_manifest_sha256,
        frozen_gate_manifest_path=verified.frozen_gate_manifest_path,
        frozen_gate_manifest_sha256=verified.frozen_gate_manifest_sha256,
        stage1_records_path=verified.stage1_records_path,
        stage1_records_sha256=verified.stage1_records_sha256,
        diagnostic_evidence_path=verified.diagnostic_evidence_path,
        diagnostic_evidence_sha256=verified.diagnostic_evidence_sha256,
        receipt_path=verified.receipt_path,
        receipt_sha256=verified.receipt_sha256,
        receipt_byte_count=verified.receipt_byte_count,
        selected_for_stage2=selected,
    )


def authorize_stage2() -> Stage2Authorization:
    """Authorize from the code-pinned production configuration only."""

    return _assert_stage2_gate(
        _verify_trusted_chain(_production_config_binding())
    )


def _run_authorized_backend(
    authorization: Stage2Authorization,
    backend: Stage2LaunchBackend[
        PathsT, LeaseT, CudaT, ProcessIdsT, CommandsT, LaunchResultT
    ],
) -> LaunchResultT:
    paths = backend.create_stage2_paths(authorization)
    lease = backend.acquire_gpu_lease(authorization)
    cuda_runtime = backend.initialize_cuda(authorization, lease)
    process_ids = backend.create_process_ids(authorization)
    commands = backend.build_worker_commands(
        authorization, paths, lease, cuda_runtime, process_ids
    )
    return backend.launch_workers(
        authorization, paths, lease, cuda_runtime, process_ids, commands
    )


def launch_stage2(
    backend: Stage2LaunchBackend[
        PathsT, LeaseT, CudaT, ProcessIdsT, CommandsT, LaunchResultT
    ],
) -> LaunchResultT:
    """Enter the backend only after production-chain authorization succeeds."""

    authorization = authorize_stage2()
    return _run_authorized_backend(authorization, backend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="role", required=True)
    subparsers.add_parser(
        "authorize-stage2",
        help="verify the frozen formal Stage-1 chain without side effects",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.role != "authorize-stage2":
        raise Stage2AuthorizationError(f"unsupported role: {args.role!r}")
    authorization = authorize_stage2()
    return {
        **authorization.to_dict(),
        "authorization_only": True,
        "stage2_side_effects_performed": False,
        "next_action": "pass this capability to the reviewed v3 Stage-2 backend",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except ScientificGateBlockedError as exc:
        print(f"SCIENTIFIC_GATE_BLOCKED: {exc}", file=sys.stderr)
        return EXIT_SCIENTIFIC_GATE_BLOCKED
    except Stage1ProtocolBlockedError as exc:
        print(f"PROTOCOL_GATE_BLOCKED: {exc}", file=sys.stderr)
        return EXIT_PROTOCOL_GATE_BLOCKED
    except (Stage2AuthorizationError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_EXECUTION_ERROR
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
