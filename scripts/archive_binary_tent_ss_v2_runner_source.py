#!/usr/bin/env python3
"""Archive the byte-exact legacy SS-v2 runner before disabling Stage 2.

The canonical negative-result archive is never modified.  This supplement is
bound to its sealed Stage-1 runtime ledger and is for isolated historical
restoration only; it is not an executable Stage-2 authorization artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_ROOT / "results/binary_tent/ss_calibration_v2_negative_archive"
DEFAULT_DESTINATION = (
    PROJECT_ROOT
    / "results/binary_tent/ss_calibration_v2_negative_archive_code_supplement_v1"
)
RUNNER_RELATIVE = Path("run_binary_tent_ss_calibration_v2.py")
RUNTIME_SEAL_RELATIVE = Path("stage1/aggregate/runtime_seal.json")
RUNNER_ROLE = "critical_code:run_binary_tent_ss_calibration_v2.py"
ARTIFACT_TYPE = "binary_tent_ss_v2_negative_archive_code_supplement_v1"
SNAPSHOT_RELATIVE = Path("code_snapshot/run_binary_tent_ss_calibration_v2.py")


class CodeSupplementError(RuntimeError):
    """The historical runner cannot be archived or verified safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CodeSupplementError(f"required regular JSON file missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CodeSupplementError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CodeSupplementError(f"JSON root must be an object: {path}")
    return value


def _verify_canonical_archive(archive: Path) -> dict[str, str]:
    if archive.is_symlink() or not archive.is_dir():
        raise CodeSupplementError(f"canonical negative archive missing: {archive}")
    checksum_path = archive / "SHA256SUMS"
    if checksum_path.is_symlink() or not checksum_path.is_file():
        raise CodeSupplementError("canonical archive SHA256SUMS is missing")
    ledger: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name
            or name in ledger
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            raise CodeSupplementError("canonical archive SHA256SUMS is invalid")
        member = archive / name
        if member.is_symlink() or not member.is_file() or sha256_file(member) != digest:
            raise CodeSupplementError(f"canonical archive member mismatch: {name}")
        ledger[name] = digest
    if not ledger:
        raise CodeSupplementError("canonical archive checksum ledger is empty")
    return ledger


def _sealed_runner_binding(
    *, project_root: Path, archive: Path
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    ledger = _verify_canonical_archive(archive)
    runtime_relative = RUNTIME_SEAL_RELATIVE.as_posix()
    if runtime_relative not in ledger:
        raise CodeSupplementError("runtime seal is absent from canonical archive ledger")
    runtime_path = archive / RUNTIME_SEAL_RELATIVE
    runtime = _load_json(runtime_path)
    bindings = runtime.get("bindings")
    if not isinstance(bindings, list):
        raise CodeSupplementError("runtime seal bindings must be a list")
    matches = [
        value
        for value in bindings
        if isinstance(value, Mapping) and value.get("role") == RUNNER_ROLE
    ]
    if len(matches) != 1:
        raise CodeSupplementError("runtime seal must bind exactly one legacy runner")
    binding = dict(matches[0])
    expected_path = (project_root / RUNNER_RELATIVE).resolve()
    if Path(str(binding.get("path"))).resolve() != expected_path:
        raise CodeSupplementError("sealed runner path differs from active project runner")
    runner = project_root / RUNNER_RELATIVE
    if runner.is_symlink() or not runner.is_file():
        raise CodeSupplementError("active legacy runner is missing or a symlink")
    observed_sha = sha256_file(runner)
    observed_bytes = runner.stat().st_size
    if binding.get("sha256") != observed_sha or binding.get("bytes") != observed_bytes:
        raise CodeSupplementError(
            "active legacy runner bytes differ from the sealed historical runner"
        )
    return runner, binding, ledger


def verify_supplement(destination: Path) -> dict[str, Any]:
    root = destination.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CodeSupplementError(f"code supplement missing: {root}")
    expected = {
        SNAPSHOT_RELATIVE.as_posix(),
        "artifact_manifest.json",
        "COMPLETE.json",
        "SHA256SUMS",
    }
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed != expected or any(path.is_symlink() for path in root.rglob("*")):
        raise CodeSupplementError("code supplement member set is not exact")
    checksums: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if not separator or name in checksums:
            raise CodeSupplementError("invalid supplement SHA256SUMS")
        checksums[name] = digest
    if set(checksums) != expected - {"SHA256SUMS"}:
        raise CodeSupplementError("supplement checksum member set differs")
    for name, digest in checksums.items():
        if sha256_file(root / name) != digest:
            raise CodeSupplementError(f"supplement hash mismatch: {name}")
    manifest = _load_json(root / "artifact_manifest.json")
    complete = _load_json(root / "COMPLETE.json")
    snapshot = root / SNAPSHOT_RELATIVE
    source = manifest.get("historical_runner_source")
    if (
        manifest.get("artifact_type") != ARTIFACT_TYPE
        or manifest.get("paper_result") is not False
        or manifest.get("stage2_authorization") is not False
        or manifest.get("restore_in_isolated_historical_environment_only") is not True
        or not isinstance(source, Mapping)
        or source.get("sha256") != sha256_file(snapshot)
        or source.get("bytes") != snapshot.stat().st_size
        or source.get("sealed_role") != RUNNER_ROLE
        or complete.get("artifact_type") != ARTIFACT_TYPE
        or complete.get("complete") is not True
        or complete.get("manifest_sha256") != sha256_file(root / "artifact_manifest.json")
    ):
        raise CodeSupplementError("code supplement contract failed")
    return {
        "status": "verified",
        "path": str(root),
        "runner_sha256": source["sha256"],
        "paper_result": False,
        "stage2_authorization": False,
    }


def create_or_verify_supplement(
    *, project_root: Path, archive: Path, destination: Path
) -> dict[str, Any]:
    project_root = project_root.expanduser().resolve()
    archive = archive.expanduser().resolve()
    destination = destination.expanduser().resolve()
    runner, binding, ledger = _sealed_runner_binding(
        project_root=project_root, archive=archive
    )
    if destination.exists() or destination.is_symlink():
        verified = verify_supplement(destination)
        if verified["runner_sha256"] != binding["sha256"]:
            raise CodeSupplementError("existing supplement binds another runner")
        return verified
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        snapshot = staging / SNAPSHOT_RELATIVE
        snapshot.parent.mkdir(parents=True)
        shutil.copyfile(runner, snapshot)
        if sha256_file(snapshot) != binding["sha256"]:
            raise CodeSupplementError("runner changed while it was copied")
        manifest = {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "paper_result": False,
            "stage2_authorization": False,
            "restore_in_isolated_historical_environment_only": True,
            "canonical_negative_archive": {
                "path": str(archive),
                "negative_result_sha256": ledger["NEGATIVE_RESULT.json"],
                "sha256sums_sha256": sha256_file(archive / "SHA256SUMS"),
                "runtime_seal_sha256": ledger[RUNTIME_SEAL_RELATIVE.as_posix()],
            },
            "historical_runner_source": {
                "archive_path": SNAPSHOT_RELATIVE.as_posix(),
                "restore_project_relative_path": RUNNER_RELATIVE.as_posix(),
                "sealed_role": RUNNER_ROLE,
                "sha256": binding["sha256"],
                "bytes": binding["bytes"],
            },
        }
        (staging / "artifact_manifest.json").write_bytes(_json_bytes(manifest))
        complete = {
            "schema_version": 1,
            "artifact_type": ARTIFACT_TYPE,
            "complete": True,
            "paper_result": False,
            "stage2_authorization": False,
            "manifest_sha256": sha256_file(staging / "artifact_manifest.json"),
        }
        (staging / "COMPLETE.json").write_bytes(_json_bytes(complete))
        members = [SNAPSHOT_RELATIVE.as_posix(), "artifact_manifest.json", "COMPLETE.json"]
        (staging / "SHA256SUMS").write_text(
            "".join(f"{sha256_file(staging / name)}  {name}\n" for name in sorted(members)),
            encoding="utf-8",
        )
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify_supplement(destination)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = (
            verify_supplement(args.destination)
            if args.verify_only
            else create_or_verify_supplement(
                project_root=args.project_root,
                archive=args.archive,
                destination=args.destination,
            )
        )
    except (CodeSupplementError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_TYPE",
    "CodeSupplementError",
    "create_or_verify_supplement",
    "main",
    "sha256_file",
    "verify_supplement",
]
