#!/usr/bin/env python3
"""Build, preflight, or CPU-verify the formal P3 Stage-A R0 aggregate.

``preflight`` is read-only and refuses any missing/invalid cell.  ``run`` does
the same full preflight before creating a staging directory, then publishes
the canonical five-file aggregate atomically with no replacement.  ``verify``
rebuilds the evidence and science decision from all 39 live outer shards.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Final


PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.d0_v3_formal_contract import (  # noqa: E402
    CONFIG_FILE_SHA256,
    CONFIG_RELATIVE_PATH,
    load_d0_v3_formal_contract,
    verify_frozen_parent_bindings,
)
from analysis.d0_v3_r0_aggregate_shard import (  # noqa: E402
    MEMBERS,
    R0AggregatePreflight,
    build_r0_aggregate_payloads,
    collect_r0_preflight,
    verify_r0_aggregate_shard,
)
from tta.d0_secure_io import (  # noqa: E402
    ensure_directory_chain_nofollow,
    read_stable_regular_file,
)
from tta.d0_v3_atomic_shard import publish_flat_directory_noreplace  # noqa: E402


DEFAULT_CONFIG: Final = PROJECT_ROOT / CONFIG_RELATIVE_PATH


class D0V3R0AggregateRunnerError(RuntimeError):
    """The CPU-only R0 aggregate runner failed closed."""


def _load_contract(config_path: Path = DEFAULT_CONFIG) -> Any:
    config = Path(os.path.abspath(os.fspath(config_path)))
    try:
        config.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise D0V3R0AggregateRunnerError(
            "formal config must remain inside the repository"
        ) from exc
    contract = load_d0_v3_formal_contract(config)
    verify_frozen_parent_bindings(contract, repository_root=PROJECT_ROOT)
    if contract.config_file_sha256 != CONFIG_FILE_SHA256:
        raise D0V3R0AggregateRunnerError("formal config SHA differs")
    if contract.stage2_authorized:
        raise D0V3R0AggregateRunnerError(
            "formal config unexpectedly authorizes Stage 2"
        )
    return contract


def _assert_cuda_not_initialized() -> None:
    # The imported public shard verifiers depend on torch for CPU tensor
    # schemas, but no aggregate path may initialize or execute CUDA.
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.cuda.is_initialized():
        raise D0V3R0AggregateRunnerError(
            "R0 aggregation requires a fresh CPU-only process; CUDA is initialized"
        )


def _destination(contract: Any) -> Path:
    relative = Path(contract.output_root)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise D0V3R0AggregateRunnerError("formal output root is not canonical")
    return PROJECT_ROOT / relative / "aggregate_phase" / "R0"


def validate_contract_only(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate frozen routing without checking cells or creating output."""

    _assert_cuda_not_initialized()
    contract = _load_contract(config_path)
    destination = _destination(contract)
    return {
        "schema_version": 3,
        "role": "d0_v3_formal_stage_a_r0_aggregate_validate_only",
        "valid": True,
        "mode": "cpu_only_no_output",
        "config_sha256": contract.config_file_sha256,
        "cell_count_required": 39,
        "outer_records_per_cell": 640,
        "replicate_evidence_count": 10,
        "destination": str(destination),
        "filesystem_created": False,
        "raw_gt_opened": False,
        "test_payload_opened": False,
        "validation_payload_opened": False,
        "gpu_initialized": False,
        "paper_result": False,
        "formal_protocol_complete": False,
        "stage2_authorized": False,
    }


def _preflight(config_path: Path = DEFAULT_CONFIG) -> R0AggregatePreflight:
    _assert_cuda_not_initialized()
    contract = _load_contract(config_path)
    return collect_r0_preflight(
        repository_root=PROJECT_ROOT,
        output_root_relative=contract.output_root,
        config_sha256=str(contract.config_file_sha256),
    )


def preflight_only(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Verify/rebuild the complete R0 grid without writing any artifact."""

    value = _preflight(config_path)
    decision = value.decision_receipt
    return {
        "schema_version": 3,
        "role": "d0_v3_formal_stage_a_r0_aggregate_preflight",
        "valid": True,
        "complete_input_grid": True,
        "config_sha256": value.config_sha256,
        "cell_count": len(value.lineage),
        "outer_record_count": 39 * 640,
        "replicate_evidence_count": len(value.evidence),
        "scientific_status": decision["scientific_status"],
        "formal_stage_a_protocol_complete": decision[
            "formal_stage_a_protocol_complete"
        ],
        "eligible_candidate_ids": [
            item["candidate_id"] for item in decision["eligible_candidates"]
        ],
        "required_followup_replicates": decision[
            "required_followup_replicates"
        ],
        "filesystem_created": False,
        "raw_gt_opened": False,
        "test_payload_opened": False,
        "validation_payload_opened": False,
        "gpu_initialized": False,
        "paper_result": False,
        "stage2_authorized": False,
    }


def _write_member_noreplace(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:  # pragma: no cover - defensive kernel boundary.
                raise D0V3R0AggregateRunnerError("short aggregate member write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    stable = read_stable_regular_file(path)
    if stable.data != payload:
        raise D0V3R0AggregateRunnerError(
            f"aggregate staging member differs after write: {path.name}"
        )


def run_formal_aggregate(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Publish the formal aggregate only after a full successful preflight."""

    # Crucially, no directory is created before all 39 public verifiers and
    # the complete 24,960-record evidence rebuild have succeeded.
    preflight = _preflight(config_path)
    contract = _load_contract(config_path)
    destination = _destination(contract)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"R0 aggregate destination already exists: {destination}")
    parent_relative = destination.parent.relative_to(PROJECT_ROOT)
    ensure_directory_chain_nofollow(PROJECT_ROOT, parent_relative.parts)
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.staging"
    os.mkdir(staging, mode=0o700)
    payloads = build_r0_aggregate_payloads(preflight)
    if set(payloads) != MEMBERS:
        raise D0V3R0AggregateRunnerError("aggregate payload member set differs")
    for name in sorted(payloads):
        _write_member_noreplace(staging / name, payloads[name])

    def semantic_verifier(path: Path) -> object:
        # The staging check is byte/schema complete.  At the canonical name,
        # re-run all 39 live verifiers and the evidence derivation.  The atomic
        # publisher rolls the rename back if this post-rename check fails.
        return verify_r0_aggregate_shard(
            path,
            repository_root=PROJECT_ROOT,
            output_root_relative=contract.output_root,
            expected_config_sha256=str(contract.config_file_sha256),
            verify_live_cells=(path == destination),
        )

    published = publish_flat_directory_noreplace(
        staging,
        destination,
        expected_members=tuple(MEMBERS),
        semantic_verifier=semantic_verifier,
    )
    verified = verify_r0_aggregate_shard(
        published,
        repository_root=PROJECT_ROOT,
        output_root_relative=contract.output_root,
        expected_config_sha256=str(contract.config_file_sha256),
        verify_live_cells=False,
    )
    result = asdict(verified)
    result["path"] = str(verified.path)
    result.update(
        {
            "published": True,
            "immutable": True,
            "atomic_no_replace": True,
            "live_39_cell_rebuild_passed_at_publication": True,
            "raw_gt_opened": False,
            "test_payload_opened": False,
            "validation_payload_opened": False,
            "gpu_initialized": False,
            "paper_result": False,
            "stage2_authorized": False,
        }
    )
    return result


def verify_formal_aggregate(
    path: Path | None = None,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Re-run the public, live 39-cell CPU verifier."""

    _assert_cuda_not_initialized()
    contract = _load_contract(config_path)
    target = _destination(contract) if path is None else Path(path)
    verified = verify_r0_aggregate_shard(
        target,
        repository_root=PROJECT_ROOT,
        output_root_relative=contract.output_root,
        expected_config_sha256=str(contract.config_file_sha256),
        verify_live_cells=True,
    )
    result = asdict(verified)
    result["path"] = str(verified.path)
    result.update(
        {
            "valid": True,
            "live_39_cell_rebuild_passed": True,
            "raw_gt_opened": False,
            "test_payload_opened": False,
            "validation_payload_opened": False,
            "gpu_initialized": False,
            "paper_result": False,
            "stage2_authorized": False,
        }
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("preflight")
    subparsers.add_parser("run")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        result = validate_contract_only(arguments.config)
    elif arguments.command == "preflight":
        result = preflight_only(arguments.config)
    elif arguments.command == "run":
        result = run_formal_aggregate(arguments.config)
    else:
        result = verify_formal_aggregate(
            arguments.path, config_path=arguments.config
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
