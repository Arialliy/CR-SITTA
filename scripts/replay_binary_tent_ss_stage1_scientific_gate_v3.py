#!/usr/bin/env python3
"""Certify the completed TENT-SS v2 Stage 1 as a v3 negative result.

This is a retrospective, necessary-condition replay.  It validates the
immutable v2 aggregate, recomputes all selector metrics from integer sufficient
statistics, and writes a v3 scientific receipt.  Because the remaining v3
thresholds have not yet been preregistered, this command can only certify a
negative result; it can never authorize Stage 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_CONFIG = PROJECT_ROOT / "configs/binary_tent_ss_calibration_v3.yaml"
DEFAULT_AGGREGATE = (
    PROJECT_ROOT
    / "results/binary_tent/ss_calibration_v2/stage1/aggregate"
)
JSON_SEPARATORS = (",", ":")


from tta.binary_tent_ss_calibration_selector_v3 import (  # noqa: E402
    RETROSPECTIVE_MODE,
    ScientificGateSpec,
    select_stage1_candidates,
    validate_stage1_scientific_receipt,
)


class ReplayVerificationError(RuntimeError):
    """Raised when frozen input identity or replay semantics do not match."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    unresolved = path.absolute()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ReplayVerificationError(
            f"{label} must be a regular non-symlink file: {unresolved}"
        )
    return unresolved.resolve()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayVerificationError(f"{label} must be a mapping")
    return value


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayVerificationError(f"cannot load {label} {path}: {exc}") from exc
    return _mapping(value, label)


def _load_config(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ReplayVerificationError(f"cannot load v3 config {path}: {exc}") from exc
    return _mapping(value, "v3 config")


def _project_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReplayVerificationError(f"{label} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReplayVerificationError(f"{label} must remain under the project root")
    path = (PROJECT_ROOT / candidate).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ReplayVerificationError(f"{label} escapes the project root")
    return path


def _verify_config_bindings(config: Mapping[str, Any]) -> None:
    frozen = _mapping(
        config.get("v2_execution_protocol_frozen_unchanged"),
        "v2_execution_protocol_frozen_unchanged",
    )
    for prefix in ("config", "selector"):
        path = _regular_file(
            _project_path(frozen.get(f"{prefix}_path"), f"{prefix}_path"),
            f"frozen v2 {prefix}",
        )
        expected = frozen.get(f"{prefix}_sha256")
        observed = _sha256(path)
        if observed != expected:
            raise ReplayVerificationError(
                f"frozen v2 {prefix} SHA-256 mismatch: expected {expected}, observed {observed}"
            )


def _verify_aggregate(aggregate: Path) -> tuple[Path, Path, Path, Path]:
    unresolved = aggregate.absolute()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ReplayVerificationError(
            f"v2 Stage-1 aggregate must be a real directory: {unresolved}"
        )
    aggregate = unresolved.resolve()
    manifest_path = _regular_file(
        aggregate / "artifact_manifest.json", "v2 aggregate manifest"
    )
    complete_path = _regular_file(
        aggregate / "COMPLETE.json", "v2 aggregate completion sentinel"
    )
    manifest = _load_json(manifest_path, "v2 aggregate manifest")
    if manifest.get("artifact_type") != "binary_tent_ss_calibration_v2_stage1_aggregate":
        raise ReplayVerificationError("unexpected v2 aggregate artifact_type")
    files = _mapping(manifest.get("files"), "v2 aggregate manifest.files")
    expected_names = set(files) | {manifest_path.name, complete_path.name}
    observed_names = {path.name for path in aggregate.iterdir()}
    if observed_names != expected_names:
        raise ReplayVerificationError(
            "v2 aggregate file set differs from its frozen manifest: "
            f"missing={sorted(expected_names - observed_names)}, "
            f"extra={sorted(observed_names - expected_names)}"
        )
    for name, raw_record in files.items():
        record = _mapping(raw_record, f"manifest.files[{name!r}]")
        path = _regular_file(aggregate / name, f"manifest file {name}")
        expected_size = record.get("bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ReplayVerificationError(f"manifest size for {name} is invalid")
        if path.stat().st_size != expected_size:
            raise ReplayVerificationError(f"manifest size mismatch for {name}")
        expected_sha = record.get("sha256")
        observed_sha = _sha256(path)
        if observed_sha != expected_sha:
            raise ReplayVerificationError(
                f"manifest SHA-256 mismatch for {name}: expected {expected_sha}, observed {observed_sha}"
            )
    return (
        manifest_path,
        complete_path,
        _regular_file(aggregate / "stage1_records.jsonl", "v2 Stage-1 records"),
        _regular_file(
            aggregate / "lr_strength_diagnostics.jsonl", "v2 diagnostics"
        ),
    )


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ReplayVerificationError(
                        f"blank JSONL record at {path}:{line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReplayVerificationError(
                        f"invalid JSONL record at {path}:{line_number}: {exc}"
                    ) from exc
                records.append(_mapping(value, f"records[{line_number - 1}]"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ReplayVerificationError(f"cannot read v2 Stage-1 records: {exc}") from exc
    return records


def _file_binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_replay_receipt(
    *, config_path: Path = DEFAULT_CONFIG, aggregate: Path = DEFAULT_AGGREGATE
) -> dict[str, Any]:
    config_path = _regular_file(config_path, "v3 scientific-gate config")
    config = _load_config(config_path)
    if config.get("schema_version") != 3:
        raise ReplayVerificationError("v3 config schema_version must be 3")
    _verify_config_bindings(config)

    active = _mapping(config.get("active_profile"), "active_profile")
    if active.get("mode") != RETROSPECTIVE_MODE:
        raise ReplayVerificationError("active profile is not retrospective replay")
    if active.get("stage2_authorization") is not False:
        raise ReplayVerificationError("retrospective profile must forbid Stage 2")

    manifest, complete, records_path, diagnostics_path = _verify_aggregate(aggregate)
    evidence = _mapping(active.get("evidence"), "active_profile.evidence")
    configured_records = _regular_file(
        _project_path(evidence.get("records_path"), "active records_path"),
        "configured v2 records",
    )
    if configured_records != records_path:
        raise ReplayVerificationError("configured records path is not the verified aggregate")
    if _sha256(records_path) != evidence.get("records_sha256"):
        raise ReplayVerificationError("active-profile records SHA-256 mismatch")

    gate_spec = ScientificGateSpec.retrospective_v2_negative_replay()
    if gate_spec.profile_id != active.get("id"):
        raise ReplayVerificationError("selector replay profile differs from v3 config")
    receipt = select_stage1_candidates(
        _load_jsonl(records_path), diagnostics=None, gate_spec=gate_spec
    )

    expected = _mapping(active.get("expected_decision"), "expected_decision")
    for field in (
        "protocol_status",
        "scientific_status",
        "stage2_allowed",
        "eligible_candidates",
        "selected_for_stage2",
    ):
        if receipt.get(field) != expected.get(field):
            raise ReplayVerificationError(
                f"replayed {field} differs from the frozen expected decision"
            )
    if receipt.get("stage3_allowed") is not False:
        raise ReplayVerificationError("negative replay unexpectedly allows Stage 3")

    original_receipt = _regular_file(
        records_path.parent / "stage1_ss_top3_receipt.json", "original v2 receipt"
    )
    selector_v3 = _regular_file(
        PROJECT_ROOT / "tta/binary_tent_ss_calibration_selector_v3.py",
        "selector v3 source",
    )
    receipt["source_evidence_bindings"] = {
        "binding_policy": "sha256_of_verified_immutable_inputs",
        "v3_config": _file_binding(config_path),
        "selector_v3": _file_binding(selector_v3),
        "v2_aggregate_manifest": _file_binding(manifest),
        "v2_aggregate_complete": _file_binding(complete),
        "v2_stage1_records": _file_binding(records_path),
        "v2_strength_diagnostics": _file_binding(diagnostics_path),
        "v2_stage1_top3_receipt": _file_binding(original_receipt),
    }
    receipt["replay_scope"] = {
        "paper_result": False,
        "source_train_derived": True,
        "uses_test_images": False,
        "uses_test_labels": False,
        "authorizes_stage2": False,
        "claim": "necessary_condition_failure_certificate_only",
    }
    validate_stage1_scientific_receipt(receipt)
    return receipt


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=JSON_SEPARATORS,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_atomic(path: Path, payload: bytes) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite replay receipt: {path}")
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    receipt = build_replay_receipt(
        config_path=args.config.absolute(), aggregate=args.aggregate.absolute()
    )
    payload = _canonical_bytes(receipt)
    output = args.output.absolute()
    if args.role == "replay":
        _write_new_atomic(output, payload)
        status = "created"
    else:
        if not output.is_file() or output.is_symlink():
            raise ReplayVerificationError(f"replay receipt is absent: {output}")
        observed = output.read_bytes()
        if observed != payload:
            raise ReplayVerificationError(
                "existing replay receipt is not byte-identical to recomputed evidence"
            )
        status = "verified"
    return {
        "role": args.role,
        "status": status,
        "output": str(output),
        "receipt_sha256": hashlib.sha256(payload).hexdigest(),
        "protocol_status": receipt["protocol_status"],
        "scientific_status": receipt["scientific_status"],
        "stage2_allowed": receipt["stage2_allowed"],
        "selected_for_stage2": receipt["selected_for_stage2"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("replay", "verify"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(build_parser().parse_args(argv))
    except (ReplayVerificationError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
