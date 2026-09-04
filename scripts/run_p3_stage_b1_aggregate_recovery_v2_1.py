#!/usr/bin/env python3
"""Run the immutable CPU-only Stage-B1 v2.1 aggregate recovery.

This runner consumes only the 39 completed Stage-B1 v2 cell artifacts.  It
never opens an image/target payload, builds a model or optimizer, evaluates a
test/validation split, selects a candidate, or authorizes a later stage.  The
recovery is deliberately published under a new fixed root and cannot replace
the failed v2 aggregate or any completed cell.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Final, Mapping, Sequence


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.p3_stage_b1_aggregate_recovery_v2_1 import (  # noqa: E402
    CONFIG_RELATIVE_PATH,
    EXPECTED_CELL_COUNT,
    MEMBERS,
    TOTAL_EPISODE_COUNT,
    P3StageB1AggregateRecoveryV21Contract,
    VerifiedStageB1AggregateRecoveryV21,
    build_cell_manifest_complete_ledger,
    build_incident_payload,
    build_recovery_code_seal,
    build_recovery_payloads,
    collect_live_parent_preflight,
    load_bound_parent_contract,
    load_recovery_contract,
    thaw_mechanism_gate_with_proof,
    verify_parent_code_seals,
    verify_recovery_aggregate,
)
from analysis.p3_stage_b1_contract_v2 import P3StageB1ContractV2  # noqa: E402
from tta.d0_secure_io import (  # noqa: E402
    ensure_directory_chain_nofollow,
    publish_file_noreplace,
    read_stable_regular_file,
)
from tta.d0_v3_atomic_shard import publish_flat_directory_noreplace  # noqa: E402


DEFAULT_CONFIG: Final = PROJECT_ROOT / CONFIG_RELATIVE_PATH
SCHEMA_VERSION: Final = 2
_STATUS_IDS: Final = frozenset(
    {
        "background_norm_dominance",
        "background_cancellation",
        "subthreshold_erasure",
    }
)
_STATUS_VALUES: Final = frozenset(
    {"supported", "not_supported", "not_estimable"}
)


class P3StageB1AggregateRecoveryRunnerV21Error(RuntimeError):
    """The aggregate-only recovery runner failed closed."""


@dataclass(frozen=True, slots=True)
class _Context:
    contract: P3StageB1AggregateRecoveryV21Contract
    parent: P3StageB1ContractV2
    cell_code_seal: Mapping[str, Any]
    recovery_code_seal: Mapping[str, Any]


def _assert_cpu_only(*, boundary: str) -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "":
        observed = "<unset>" if visible is None else repr(visible)
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "aggregate recovery requires CUDA_VISIBLE_DEVICES='' "
            f"({boundary}); observed={observed}"
        )
    # Do not import torch here.  Some public schema modules import it for CPU
    # tensor types; if present, its CUDA runtime must still be untouched.
    torch_module = sys.modules.get("torch")
    cuda = None if torch_module is None else getattr(torch_module, "cuda", None)
    if cuda is not None and bool(cuda.is_initialized()):
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            f"aggregate recovery initialized CUDA ({boundary})"
        )


def _project_relative_path(relative: str, *, label: str) -> Path:
    value = Path(relative)
    if (
        value.is_absolute()
        or not value.parts
        or any(part in {"", ".", ".."} for part in value.parts)
    ):
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            f"{label} is not canonical project-relative"
        )
    absolute = Path(os.path.abspath(os.fspath(PROJECT_ROOT / value)))
    try:
        absolute.relative_to(PROJECT_ROOT)
    except ValueError as exc:  # pragma: no cover - also guarded by parts.
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            f"{label} escapes the project root"
        ) from exc
    return absolute


def _destination(contract: P3StageB1AggregateRecoveryV21Contract) -> Path:
    root = _project_relative_path(contract.output_root, label="recovery output root")
    relative = Path(contract.aggregate_relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "aggregate relative path is not canonical"
        )
    destination = Path(os.path.abspath(os.fspath(root / relative)))
    try:
        destination.relative_to(root)
    except ValueError as exc:  # pragma: no cover - also guarded by parts.
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "aggregate destination escapes the recovery root"
        ) from exc
    return destination


def _incident_path(contract: P3StageB1AggregateRecoveryV21Contract) -> Path:
    root = _project_relative_path(contract.output_root, label="recovery output root")
    relative = Path(contract.incident_relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "incident relative path is not canonical"
        )
    incident = Path(os.path.abspath(os.fspath(root / relative)))
    try:
        incident.relative_to(root)
    except ValueError as exc:  # pragma: no cover - also guarded by parts.
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "incident path escapes the recovery root"
        ) from exc
    if incident == _destination(contract):
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "incident and aggregate destinations must differ"
        )
    return incident


def _canonical_config_path(config_path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(config_path)))
    repository = Path(os.path.abspath(os.fspath(PROJECT_ROOT)))
    try:
        absolute.relative_to(repository)
    except ValueError as exc:
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "aggregate recovery config path must remain inside the repository"
        ) from exc
    return absolute


def _load_context(config_path: Path = DEFAULT_CONFIG) -> _Context:
    _assert_cpu_only(boundary="before contract load")
    contract = load_recovery_contract(_canonical_config_path(config_path))
    parent = load_bound_parent_contract(contract, repository_root=PROJECT_ROOT)
    cell_code_seal = verify_parent_code_seals(
        contract, parent, repository_root=PROJECT_ROOT
    )
    recovery_code_seal = build_recovery_code_seal(
        contract, repository_root=PROJECT_ROOT
    )
    _destination(contract)
    _incident_path(contract)
    _assert_cpu_only(boundary="after contract load")
    return _Context(contract, parent, cell_code_seal, recovery_code_seal)


def _assert_context_unchanged(context: _Context, *, boundary: str) -> None:
    _assert_cpu_only(boundary=boundary)
    observed = build_recovery_code_seal(
        context.contract, repository_root=PROJECT_ROOT
    )
    if observed != context.recovery_code_seal:
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            f"recovery implementation changed ({boundary})"
        )
    current_cell_seal = verify_parent_code_seals(
        context.contract, context.parent, repository_root=PROJECT_ROOT
    )
    if current_cell_seal != context.cell_code_seal:
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            f"parent implementation changed ({boundary})"
        )


def _read_exact_incident(
    path: Path, expected: bytes, *, required: bool
) -> str | None:
    if not path.exists() and not path.is_symlink():
        if required:
            raise FileNotFoundError(f"recovery incident is missing: {path}")
        return None
    try:
        stable = read_stable_regular_file(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            f"recovery incident is not a stable regular file: {path}"
        ) from exc
    if stable.data != expected:
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "existing recovery incident differs from the frozen payload"
        )
    if stable.mode & 0o222 or stable.link_count != 1:
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "published recovery incident is not immutable and singly linked"
        )
    return stable.sha256


def _write_fd_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:  # pragma: no cover - defensive kernel boundary.
            raise P3StageB1AggregateRecoveryRunnerV21Error("short staging write")
        written += count
    os.fsync(descriptor)


def _publish_incident_noreplace(
    context: _Context, *, path: Path, payload: bytes
) -> str:
    existing = _read_exact_incident(path, payload, required=False)
    if existing is not None:
        return existing
    parent_relative = path.parent.relative_to(PROJECT_ROOT)
    ensure_directory_chain_nofollow(PROJECT_ROOT, parent_relative.parts)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".staging", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        _write_fd_all(descriptor, payload)
    finally:
        os.close(descriptor)
    os.chmod(temporary, 0o444, follow_symlinks=False)

    def pre_rename_guard() -> None:
        _assert_context_unchanged(context, boundary="incident pre-rename")
        if read_stable_regular_file(temporary).data != payload:
            raise P3StageB1AggregateRecoveryRunnerV21Error(
                "incident staging bytes changed"
            )

    try:
        published = publish_file_noreplace(
            temporary, path, pre_rename_guard=pre_rename_guard
        )
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
        raise
    observed = _read_exact_incident(published, payload, required=True)
    assert observed is not None
    return observed


def _write_member(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        _write_fd_all(descriptor, payload)
    finally:
        os.close(descriptor)
    if read_stable_regular_file(path).data != payload:
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            f"aggregate staging member differs after write: {path.name}"
        )


def _result(
    verified: VerifiedStageB1AggregateRecoveryV21,
    *, incident_sha256: str,
    ledger_sha256: str,
) -> dict[str, Any]:
    statuses = dict(verified.mechanism_statuses)
    if set(statuses) != _STATUS_IDS or any(
        value not in _STATUS_VALUES for value in statuses.values()
    ):
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "recovery did not report the exact three mechanism statuses"
        )
    if (
        verified.cell_count != EXPECTED_CELL_COUNT
        or verified.episode_count != TOTAL_EPISODE_COUNT
        or verified.candidate_selection_performed
        or verified.stage_b3_authorized
        or verified.p5_authorized
    ):
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "recovery count or authorization boundary differs"
        )
    return {
        "path": str(verified.path),
        "cell_count": verified.cell_count,
        "episode_count": verified.episode_count,
        "manifest_sha256": verified.manifest_sha256,
        "complete_sha256": verified.complete_sha256,
        "mechanism_evidence_sha256": verified.mechanism_evidence_sha256,
        "raw_numeric_audit_sha256": verified.raw_numeric_audit_sha256,
        "recovery_receipt_sha256": verified.recovery_receipt_sha256,
        "incident_sha256": incident_sha256,
        "cell_manifest_complete_ledger_sha256": ledger_sha256,
        "mechanism_statuses": statuses,
        "candidate_selection_performed": False,
        "stage_b3_authorized": False,
        "p5_authorized": False,
        "test_payload_opened": False,
        "validation_payload_opened": False,
        "cuda_initialized": False,
    }


def validate_only(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate all frozen recovery bindings without creating any output."""

    context = _load_context(config_path)
    rows, ledger_sha256 = build_cell_manifest_complete_ledger(
        context.contract, context.parent, repository_root=PROJECT_ROOT
    )
    _, adapter_proof = thaw_mechanism_gate_with_proof(
        context.parent.raw["mechanism_evidence_flags"]
    )
    incident = build_incident_payload(context.contract)
    _assert_context_unchanged(context, boundary="validate return")
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "p3_stage_b1_aggregate_recovery_v2_1_validate_only",
        "valid": True,
        "mode": "cpu_only_no_output",
        "config_sha256": context.contract.config_file_sha256,
        "parent_config_sha256": context.parent.config_file_sha256,
        "required_cell_count": EXPECTED_CELL_COUNT,
        "required_episode_count": TOTAL_EPISODE_COUNT,
        "cell_ledger_file_count": len(rows),
        "cell_manifest_complete_ledger_sha256": ledger_sha256,
        "incident_sha256": hashlib.sha256(incident).hexdigest(),
        "adapter_semantics_equal": adapter_proof["canonical_semantics_equal"],
        "recovery_code_seal_bundle_sha256": context.recovery_code_seal[
            "bundle_sha256"
        ],
        "destination": str(_destination(context.contract)),
        "filesystem_created": False,
        "raw_image_or_target_opened": False,
        "test_payload_opened": False,
        "validation_payload_opened": False,
        "cuda_initialized": False,
        "candidate_selection_performed": False,
        "stage_b3_authorized": False,
        "p5_authorized": False,
    }


def run_formal_aggregate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Build and atomically publish the fixed recovery aggregate."""

    context = _load_context(config_path)
    destination = _destination(context.contract)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"recovery aggregate destination already exists: {destination}"
        )
    incident_path = _incident_path(context.contract)
    incident_payload = build_incident_payload(context.contract)
    # A pre-existing incident is accepted only when byte-identical, allowing a
    # safe retry after a prepublication aggregate failure.
    _read_exact_incident(incident_path, incident_payload, required=False)
    _, ledger_sha256 = build_cell_manifest_complete_ledger(
        context.contract, context.parent, repository_root=PROJECT_ROOT
    )
    preflight = collect_live_parent_preflight(
        context.contract,
        context.parent,
        repository_root=PROJECT_ROOT,
        expected_cell_code_seal=context.cell_code_seal,
    )
    if (
        len(preflight.lineage) != EXPECTED_CELL_COUNT
        or len(preflight.records) != TOTAL_EPISODE_COUNT
    ):
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "recovery preflight is not exact 39 x 64"
        )
    incident_sha256 = hashlib.sha256(incident_payload).hexdigest()
    payloads = build_recovery_payloads(
        context.contract,
        context.parent,
        preflight,
        repository_root=PROJECT_ROOT,
        incident_sha256=incident_sha256,
        cell_ledger_sha256=ledger_sha256,
    )
    if set(payloads) != MEMBERS:
        raise P3StageB1AggregateRecoveryRunnerV21Error(
            "recovery builder returned a different member set"
        )
    _assert_context_unchanged(context, boundary="before publication")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"recovery aggregate destination already exists: {destination}"
        )

    incident_sha256 = _publish_incident_noreplace(
        context, path=incident_path, payload=incident_payload
    )
    destination_parent = destination.parent.relative_to(PROJECT_ROOT)
    ensure_directory_chain_nofollow(PROJECT_ROOT, destination_parent.parts)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.recovery-build-", dir=destination.parent
        )
    )
    for name in sorted(payloads):
        _write_member(staging / name, payloads[name])

    verified_by_path: dict[Path, VerifiedStageB1AggregateRecoveryV21] = {}

    def semantic_verifier(path: Path) -> VerifiedStageB1AggregateRecoveryV21:
        _assert_context_unchanged(context, boundary="semantic verification")
        _read_exact_incident(incident_path, incident_payload, required=True)
        absolute = Path(os.path.abspath(os.fspath(path)))
        verified = verify_recovery_aggregate(
            absolute,
            contract=context.contract,
            parent=context.parent,
            repository_root=PROJECT_ROOT,
            verify_live_cells=(absolute == destination),
            expected_cell_code_seal=context.cell_code_seal,
        )
        _result(
            verified,
            incident_sha256=incident_sha256,
            ledger_sha256=ledger_sha256,
        )
        verified_by_path[absolute] = verified
        return verified

    published = publish_flat_directory_noreplace(
        staging,
        destination,
        expected_members=sorted(MEMBERS),
        semantic_verifier=semantic_verifier,
    )
    published = Path(os.path.abspath(os.fspath(published)))
    verified = verified_by_path.get(published)
    if verified is None:  # pragma: no cover - publisher verifies canonical path.
        verified = semantic_verifier(published)
    _assert_context_unchanged(context, boundary="aggregate return")
    result = _result(
        verified,
        incident_sha256=incident_sha256,
        ledger_sha256=ledger_sha256,
    )
    result.update(
        {
            "published": True,
            "immutable": True,
            "atomic_no_replace": True,
            "live_39_cell_rebuild_passed_at_publication": True,
        }
    )
    return result


def verify_formal_aggregate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Rebuild and verify the fixed recovery aggregate and all 39 live cells."""

    context = _load_context(config_path)
    incident_payload = build_incident_payload(context.contract)
    incident_sha256 = _read_exact_incident(
        _incident_path(context.contract), incident_payload, required=True
    )
    assert incident_sha256 is not None
    _, ledger_sha256 = build_cell_manifest_complete_ledger(
        context.contract, context.parent, repository_root=PROJECT_ROOT
    )
    verified = verify_recovery_aggregate(
        _destination(context.contract),
        contract=context.contract,
        parent=context.parent,
        repository_root=PROJECT_ROOT,
        verify_live_cells=True,
        expected_cell_code_seal=context.cell_code_seal,
    )
    _assert_context_unchanged(context, boundary="verify return")
    result = _result(
        verified,
        incident_sha256=incident_sha256,
        ledger_sha256=ledger_sha256,
    )
    result.update({"valid": True, "live_39_cell_rebuild_passed": True})
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("aggregate")
    commands.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        result = validate_only(arguments.config)
    elif arguments.command == "aggregate":
        result = run_formal_aggregate(arguments.config)
    else:
        result = verify_formal_aggregate(arguments.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
