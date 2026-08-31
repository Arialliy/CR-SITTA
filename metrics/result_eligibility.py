"""Scientific-eligibility registry for immutable experiment artifacts.

Artifact integrity and scientific eligibility are separate gates. Full-tree
artifacts are bound byte-for-byte. Manifest-backed artifacts are accepted only
after every declared payload, recursively referenced manifest, completion seal,
and the exact artifact tree have been verified from stable regular-file
snapshots. The registry itself never edits an experiment artifact directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any

import yaml


TREE_ALGORITHM = "sorted-root-relative-path-tab-sha256-tab-size-lf-v1"
SEALED_CHAIN_ALGORITHM = (
    "verified-exact-tree-with-recursive-manifest-completion-chain-v2"
)
EXPECTED_REGISTRY_RELATIVE = Path("results/artifact_eligibility_registry_v1.json")
ALLOWED_TIERS = {
    "engineering_smoke",
    "development_test_selected",
    "protocol_asset_nonperformance",
    "final_paper_benchmark",
}


class EligibilityError(RuntimeError):
    """Raised when an artifact or eligibility claim violates the contract."""


@dataclass(frozen=True)
class _FileSnapshot:
    sha256: str
    size_bytes: int
    data: bytes | None = None


def _stable_regular_file_snapshot(
    path: Path,
    *,
    retain_bytes: bool = False,
    chunk_size: int = 1024 * 1024,
) -> _FileSnapshot:
    """Read one non-symlink regular file from a single stable descriptor."""

    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise EligibilityError(f"refusing symlink file: {path}") from error
        raise EligibilityError(f"cannot open artifact file: {path}: {error}") from error

    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain_bytes else None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise EligibilityError(f"artifact member is not a regular file: {path}")
        total = 0
        while True:
            chunk = os.read(descriptor, chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, key) != getattr(after, key) for key in stable_fields):
        raise EligibilityError(f"artifact file changed while being read: {path}")
    if total != before.st_size:
        raise EligibilityError(
            f"artifact file size changed while being read: {path}: "
            f"read={total}, stat={before.st_size}"
        )
    return _FileSnapshot(
        sha256=digest.hexdigest(),
        size_bytes=total,
        data=b"".join(chunks) if chunks is not None else None,
    )


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    return _stable_regular_file_snapshot(path, chunk_size=chunk_size).sha256


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EligibilityError(f"{label} must be a mapping")
    return value


def _require_sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EligibilityError(f"{label} must be a sequence")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(value)
    if observed != expected:
        raise EligibilityError(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EligibilityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EligibilityError(f"{label} must be a non-empty string")
    return value


def _assert_no_symlink_components(root: Path, path: Path, *, label: str) -> None:
    if root.is_symlink():
        raise EligibilityError(f"{label} root is a symlink: {root}")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise EligibilityError(f"{label} escapes root: {path}") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise EligibilityError(f"{label} contains a symlink: {current}")


def _resolve_within(project_root: Path, relative: str, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise EligibilityError(f"{label} must be a non-empty project-relative path")
    if any(part in ("", ".", "..") for part in Path(relative).parts):
        raise EligibilityError(f"{label} is not canonical: {relative}")
    root = project_root.resolve()
    candidate = root / relative
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise EligibilityError(f"{label} escapes the project root: {relative}") from error
    _assert_no_symlink_components(root, candidate, label=label)
    return candidate


def _project_path(project_root: Path, path: Path, *, label: str) -> Path:
    root = project_root.resolve()
    candidate = path if path.is_absolute() else root / path
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as error:
        raise EligibilityError(f"{label} must be inside the project root: {path}") from error
    return _resolve_within(root, relative, label=label)


def _resolve_reference(
    artifact_root: Path,
    declaring_directory: Path,
    raw: Any,
    *,
    label: str,
) -> Path:
    relative = _require_nonempty_string(raw, label)
    if Path(relative).is_absolute():
        raise EligibilityError(f"{label} must be relative: {relative}")
    candidate = Path(os.path.abspath(declaring_directory / relative))
    try:
        candidate.relative_to(artifact_root)
    except ValueError as error:
        raise EligibilityError(f"{label} escapes artifact root: {relative}") from error
    _assert_no_symlink_components(artifact_root, candidate, label=label)
    return candidate


@dataclass(frozen=True)
class ArtifactIdentity:
    algorithm: str
    sha256: str
    file_count: int
    total_size_bytes: int
    bound_files: tuple[Mapping[str, Any], ...]
    verification: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "sha256": self.sha256,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "bound_files": [dict(value) for value in self.bound_files],
            "verification": dict(self.verification),
        }


def _identity_from_snapshots(
    root: Path,
    snapshots: Mapping[Path, _FileSnapshot],
    *,
    algorithm: str,
    verification: Mapping[str, Any],
) -> ArtifactIdentity:
    if not snapshots:
        raise EligibilityError(f"artifact identity has no files: {root}")
    digest = hashlib.sha256()
    records: list[Mapping[str, Any]] = []
    total_size = 0
    for path in sorted(snapshots, key=lambda value: value.relative_to(root).as_posix()):
        snapshot = snapshots[path]
        relative = path.relative_to(root).as_posix()
        digest.update(
            f"{relative}\t{snapshot.sha256}\t{snapshot.size_bytes}\n".encode("utf-8")
        )
        total_size += snapshot.size_bytes
        records.append(
            {
                "path": relative,
                "sha256": snapshot.sha256,
                "size_bytes": snapshot.size_bytes,
            }
        )
    return ArtifactIdentity(
        algorithm=algorithm,
        sha256=digest.hexdigest(),
        file_count=len(records),
        total_size_bytes=total_size,
        bound_files=tuple(records),
        verification=dict(verification),
    )


def full_tree_identity(root: Path) -> ArtifactIdentity:
    if root.is_symlink() or not root.is_dir():
        raise EligibilityError(f"artifact root must be a real directory: {root}")
    snapshots: dict[Path, _FileSnapshot] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise EligibilityError(f"artifact tree contains a symlink: {path}")
        if path.is_file():
            snapshots[path] = _stable_regular_file_snapshot(path)
        elif not path.is_dir():
            raise EligibilityError(f"artifact tree contains a special entry: {path}")
    return _identity_from_snapshots(
        root,
        snapshots,
        algorithm=TREE_ALGORITHM,
        verification={
            "mode": "full_tree",
            "exact_tree_verified": True,
            "completion_chain_verified": False,
            "manifest_count": 0,
            "completion_count": 0,
            "payload_file_count": len(snapshots),
        },
    )


class _SealedManifestChainVerifier:
    def __init__(self, root: Path) -> None:
        if root.is_symlink() or not root.is_dir():
            raise EligibilityError(f"artifact root must be a real directory: {root}")
        self.root = root
        self.snapshots: dict[Path, _FileSnapshot] = {}
        self.roles: dict[Path, set[str]] = {}
        self.manifests: dict[Path, Mapping[str, Any]] = {}
        self.completions: dict[Path, Mapping[str, Any]] = {}
        self.processed_seals: set[tuple[Path, Path]] = set()
        self.root_seal_count = 0

    def _snapshot(
        self,
        path: Path,
        *,
        expected_sha256: Any,
        role: str,
        expected_size: int | None = None,
        retain_bytes: bool = False,
    ) -> _FileSnapshot:
        expected = _require_sha256(expected_sha256, f"{role} SHA-256")
        _assert_no_symlink_components(self.root, path, label=role)
        snapshot = self.snapshots.get(path)
        if snapshot is None or (retain_bytes and snapshot.data is None):
            snapshot = _stable_regular_file_snapshot(path, retain_bytes=retain_bytes)
            previous = self.snapshots.get(path)
            if previous is not None and (
                previous.sha256 != snapshot.sha256
                or previous.size_bytes != snapshot.size_bytes
            ):
                raise EligibilityError(f"artifact file changed between references: {path}")
            self.snapshots[path] = snapshot
        if snapshot.sha256 != expected:
            raise EligibilityError(
                f"{role} SHA-256 drift for {path}: "
                f"{snapshot.sha256} != {expected}"
            )
        if expected_size is not None and snapshot.size_bytes != expected_size:
            raise EligibilityError(
                f"{role} size drift for {path}: "
                f"{snapshot.size_bytes} != {expected_size}"
            )
        self.roles.setdefault(path, set()).add(role)
        return snapshot

    @staticmethod
    def _parse_json(snapshot: _FileSnapshot, path: Path, *, label: str) -> Mapping[str, Any]:
        if snapshot.data is None:
            raise AssertionError("JSON snapshot bytes were not retained")
        try:
            value = json.loads(snapshot.data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EligibilityError(f"invalid {label} JSON: {path}") from error
        return _require_mapping(value, label)

    def _verify_payload_ledgers(
        self, manifest_path: Path, manifest: Mapping[str, Any]
    ) -> None:
        ledgers_found = False
        files = manifest.get("files")
        if files is not None:
            ledgers_found = True
            files_mapping = _require_mapping(files, f"{manifest_path} files")
            for raw_path, raw_spec in files_mapping.items():
                path = _resolve_reference(
                    self.root,
                    manifest_path.parent,
                    raw_path,
                    label=f"{manifest_path} files path",
                )
                if isinstance(raw_spec, Mapping):
                    expected = raw_spec.get("sha256")
                    raw_size = raw_spec.get("bytes", raw_spec.get("size_bytes"))
                    if raw_size is not None and (type(raw_size) is not int or raw_size < 0):
                        raise EligibilityError(
                            f"invalid payload size for {path}: {raw_size!r}"
                        )
                else:
                    expected = raw_spec
                    raw_size = None
                self._snapshot(
                    path,
                    expected_sha256=expected,
                    expected_size=raw_size,
                    role="payload",
                )

        files_sha256 = manifest.get("files_sha256")
        if files_sha256 is not None:
            ledgers_found = True
            digest_mapping = _require_mapping(
                files_sha256, f"{manifest_path} files_sha256"
            )
            for raw_path, expected in digest_mapping.items():
                path = _resolve_reference(
                    self.root,
                    manifest_path.parent,
                    raw_path,
                    label=f"{manifest_path} files_sha256 path",
                )
                self._snapshot(path, expected_sha256=expected, role="payload")

        if not ledgers_found:
            raise EligibilityError(f"manifest has no payload ledger: {manifest_path}")

    def _process_manifest(self, path: Path, expected_sha256: Any) -> Mapping[str, Any]:
        expected = _require_sha256(expected_sha256, "manifest SHA-256")
        existing = self.manifests.get(path)
        if existing is not None:
            self._snapshot(
                path,
                expected_sha256=expected,
                role="manifest",
                retain_bytes=True,
            )
            return existing
        snapshot = self._snapshot(
            path,
            expected_sha256=expected,
            role="manifest",
            retain_bytes=True,
        )
        manifest = self._parse_json(snapshot, path, label="manifest")
        self.manifests[path] = manifest
        self._verify_payload_ledgers(path, manifest)
        self._discover_child_seals(manifest, declaring_manifest=path)
        return manifest

    def _process_completion(self, path: Path, expected_sha256: Any) -> Mapping[str, Any]:
        expected = _require_sha256(expected_sha256, "completion SHA-256")
        existing = self.completions.get(path)
        if existing is not None:
            self._snapshot(
                path,
                expected_sha256=expected,
                role="completion",
                retain_bytes=True,
            )
            return existing
        snapshot = self._snapshot(
            path,
            expected_sha256=expected,
            role="completion",
            retain_bytes=True,
        )
        completion = self._parse_json(snapshot, path, label="completion")
        if completion.get("complete") is not True:
            raise EligibilityError(f"completion seal is not complete=true: {path}")
        self.completions[path] = completion
        return completion

    def _process_seal(
        self,
        *,
        declaring_directory: Path,
        manifest_raw: Any,
        manifest_sha256: Any,
        completion_raw: Any,
        completion_sha256: Any,
        root_declaration: bool = False,
        linked_manifests: Sequence[Any] = (),
    ) -> None:
        manifest_path = _resolve_reference(
            self.root,
            declaring_directory,
            manifest_raw,
            label="manifest declaration",
        )
        completion_path = _resolve_reference(
            self.root,
            declaring_directory,
            completion_raw,
            label="completion declaration",
        )
        manifest_digest = _require_sha256(manifest_sha256, "manifest declaration SHA-256")
        pair = (manifest_path, completion_path)
        if pair in self.processed_seals:
            self._snapshot(
                manifest_path,
                expected_sha256=manifest_digest,
                role="manifest",
                retain_bytes=True,
            )
            self._snapshot(
                completion_path,
                expected_sha256=completion_sha256,
                role="completion",
                retain_bytes=True,
            )
            return
        self.processed_seals.add(pair)
        if root_declaration:
            self.root_seal_count += 1

        manifest = self._process_manifest(manifest_path, manifest_digest)
        completion = self._process_completion(completion_path, completion_sha256)
        primary_links = {
            completion.get("artifact_manifest_sha256"),
            completion.get("manifest_sha256"),
        }
        if manifest_digest not in primary_links:
            raise EligibilityError(
                f"completion does not bind its primary manifest: {completion_path}"
            )

        for index, raw in enumerate(linked_manifests):
            spec = _require_mapping(raw, f"linked_manifest[{index}]")
            _require_exact_keys(
                spec,
                {"path", "sha256", "completion_hash_field"},
                f"linked_manifest[{index}]",
            )
            linked_path = _resolve_reference(
                self.root,
                declaring_directory,
                spec["path"],
                label=f"linked_manifest[{index}].path",
            )
            linked_digest = _require_sha256(
                spec["sha256"], f"linked_manifest[{index}].sha256"
            )
            field = _require_nonempty_string(
                spec["completion_hash_field"],
                f"linked_manifest[{index}].completion_hash_field",
            )
            if completion.get(field) != linked_digest:
                raise EligibilityError(
                    f"completion field {field} does not bind {linked_path}"
                )
            linked_manifest = self._process_manifest(linked_path, linked_digest)
            outer_digest = linked_manifest.get("outer_manifest_sha256")
            if outer_digest is not None and outer_digest != manifest_digest:
                raise EligibilityError(
                    f"linked manifest does not bind outer manifest: {linked_path}"
                )
        _require_mapping(manifest, "primary manifest")

    def _discover_child_seals(self, value: Any, *, declaring_manifest: Path) -> None:
        if isinstance(value, Mapping):
            manifest_key = None
            digest_key = None
            if "artifact_manifest" in value or "artifact_manifest_sha256" in value:
                manifest_key = "artifact_manifest"
                digest_key = "artifact_manifest_sha256"
            elif "manifest" in value or "manifest_sha256" in value:
                manifest_key = "manifest"
                digest_key = "manifest_sha256"
            has_completion_reference = "completion" in value or "completion_sha256" in value
            if manifest_key is not None or has_completion_reference:
                required = {manifest_key, digest_key, "completion", "completion_sha256"}
                if None in required or any(key not in value for key in required):
                    raise EligibilityError(
                        f"partial child manifest/completion seal in {declaring_manifest}"
                    )
                self._process_seal(
                    declaring_directory=declaring_manifest.parent,
                    manifest_raw=value[manifest_key],
                    manifest_sha256=value[digest_key],
                    completion_raw=value["completion"],
                    completion_sha256=value["completion_sha256"],
                )
            for key, child in value.items():
                if key not in {"files", "files_sha256"}:
                    self._discover_child_seals(child, declaring_manifest=declaring_manifest)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                self._discover_child_seals(child, declaring_manifest=declaring_manifest)

    def verify(
        self,
        seals: Sequence[Any],
        *,
        exact_tree: bool,
        supplemental_files: Sequence[Any] = (),
    ) -> ArtifactIdentity:
        if exact_tree is not True:
            raise EligibilityError("sealed_manifest_chain requires exact_tree=true")
        if not seals:
            raise EligibilityError("sealed_manifest_chain requires at least one root seal")
        for index, raw in enumerate(seals):
            seal = _require_mapping(raw, f"seal[{index}]")
            allowed = {
                "manifest",
                "manifest_sha256",
                "completion",
                "completion_sha256",
                "linked_manifests",
            }
            required = allowed - {"linked_manifests"}
            observed = set(seal)
            if not required <= observed or observed - allowed:
                raise EligibilityError(
                    f"seal[{index}] keys differ: missing={sorted(required - observed)}, "
                    f"extra={sorted(observed - allowed)}"
                )
            linked = seal.get("linked_manifests", ())
            linked_sequence = _require_sequence(linked, f"seal[{index}].linked_manifests")
            self._process_seal(
                declaring_directory=self.root,
                manifest_raw=seal["manifest"],
                manifest_sha256=seal["manifest_sha256"],
                completion_raw=seal["completion"],
                completion_sha256=seal["completion_sha256"],
                root_declaration=True,
                linked_manifests=linked_sequence,
            )

        for index, raw in enumerate(supplemental_files):
            spec = _require_mapping(raw, f"supplemental_file[{index}]")
            _require_exact_keys(
                spec,
                {"path", "sha256", "size_bytes"},
                f"supplemental_file[{index}]",
            )
            raw_size = spec["size_bytes"]
            if type(raw_size) is not int or raw_size < 0:
                raise EligibilityError(
                    f"supplemental_file[{index}].size_bytes must be non-negative int"
                )
            path = _resolve_reference(
                self.root,
                self.root,
                spec["path"],
                label=f"supplemental_file[{index}].path",
            )
            self._snapshot(
                path,
                expected_sha256=spec["sha256"],
                expected_size=raw_size,
                role="supplemental_payload",
            )

        actual_paths: set[Path] = set()
        for path in sorted(self.root.rglob("*")):
            if path.is_symlink():
                raise EligibilityError(f"artifact tree contains a symlink: {path}")
            if path.is_file():
                actual_paths.add(path)
            elif not path.is_dir():
                raise EligibilityError(f"artifact tree contains a special entry: {path}")
        bound_paths = set(self.snapshots)
        if actual_paths != bound_paths:
            missing = sorted(
                path.relative_to(self.root).as_posix() for path in bound_paths - actual_paths
            )
            unbound = sorted(
                path.relative_to(self.root).as_posix() for path in actual_paths - bound_paths
            )
            raise EligibilityError(
                "sealed manifest chain does not cover the exact artifact tree: "
                f"missing={missing[:10]}, unbound={unbound[:10]}"
            )

        manifest_paths = {path for path, roles in self.roles.items() if "manifest" in roles}
        completion_paths = {
            path for path, roles in self.roles.items() if "completion" in roles
        }
        payload_paths = {
            path
            for path, roles in self.roles.items()
            if roles & {"payload", "supplemental_payload"}
        }
        return _identity_from_snapshots(
            self.root,
            self.snapshots,
            algorithm=SEALED_CHAIN_ALGORITHM,
            verification={
                "mode": "sealed_manifest_chain",
                "exact_tree_verified": True,
                "completion_chain_verified": True,
                "root_seal_count": self.root_seal_count,
                "recursive_seal_count": len(self.processed_seals),
                "manifest_count": len(manifest_paths),
                "completion_count": len(completion_paths),
                "payload_file_count": len(payload_paths),
            },
        )


def sealed_manifest_chain_identity(
    root: Path,
    seals: Sequence[Any],
    *,
    exact_tree: bool,
    supplemental_files: Sequence[Any] = (),
) -> ArtifactIdentity:
    return _SealedManifestChainVerifier(root).verify(
        seals,
        exact_tree=exact_tree,
        supplemental_files=supplemental_files,
    )


def compute_artifact_identity(
    project_root: Path,
    artifact: Mapping[str, Any],
    *,
    verify_expected: bool = True,
) -> ArtifactIdentity:
    root_raw = _require_nonempty_string(artifact["root"], "artifact.root")
    root = _resolve_within(project_root, root_raw, label="artifact.root")
    try:
        root.relative_to(project_root.resolve() / "results")
    except ValueError as error:
        raise EligibilityError(f"artifact root must be inside results/: {root}") from error
    identity = _require_mapping(artifact["identity"], "artifact.identity")
    mode = identity.get("mode")
    if mode == "full_tree":
        _require_exact_keys(identity, {"mode", "expected_sha256"}, "identity")
        observed = full_tree_identity(root)
    elif mode == "sealed_manifest_chain":
        allowed = {
            "mode",
            "expected_sha256",
            "exact_tree",
            "seals",
            "supplemental_files",
        }
        required = allowed - {"supplemental_files"}
        observed_keys = set(identity)
        if not required <= observed_keys or observed_keys - allowed:
            raise EligibilityError(
                "identity keys differ: "
                f"missing={sorted(required - observed_keys)}, "
                f"extra={sorted(observed_keys - allowed)}"
            )
        observed = sealed_manifest_chain_identity(
            root,
            _require_sequence(identity["seals"], "identity.seals"),
            exact_tree=identity["exact_tree"],
            supplemental_files=_require_sequence(
                identity.get("supplemental_files", ()),
                "identity.supplemental_files",
            ),
        )
    else:
        raise EligibilityError(f"unsupported artifact identity mode: {mode!r}")
    expected = _require_sha256(identity["expected_sha256"], "identity.expected_sha256")
    if verify_expected and observed.sha256 != expected:
        raise EligibilityError(
            f"artifact identity drift for {artifact['artifact_id']}: "
            f"{observed.sha256} != {expected}"
        )
    return observed


def _validate_artifact_completeness(
    artifact: Mapping[str, Any], identity: ArtifactIdentity
) -> None:
    artifact_complete = artifact["artifact_complete"]
    formal_complete = artifact["formal_protocol_complete"]
    for label, value in (
        ("artifact_complete", artifact_complete),
        ("formal_protocol_complete", formal_complete),
    ):
        if not isinstance(value, bool):
            raise EligibilityError(f"{artifact['artifact_id']}.{label} must be bool")
    if formal_complete and not artifact_complete:
        raise EligibilityError("formal protocol completion requires artifact_complete=true")
    verification = _require_mapping(identity.verification, "identity.verification")
    if artifact_complete and verification.get("exact_tree_verified") is not True:
        raise EligibilityError("artifact_complete requires an exact verified artifact tree")
    if formal_complete and verification.get("completion_chain_verified") is not True:
        raise EligibilityError(
            "formal_protocol_complete requires a verified completion chain"
        )


def _validate_scientific_eligibility(
    artifact: Mapping[str, Any], policies: Mapping[str, Any]
) -> None:
    eligibility = _require_mapping(
        artifact["scientific_eligibility"], "scientific_eligibility"
    )
    _require_exact_keys(
        eligibility,
        {"tier", "main_paper_table", "reason_codes"},
        "scientific_eligibility",
    )
    tier = eligibility["tier"]
    if tier not in ALLOWED_TIERS:
        raise EligibilityError(f"unsupported scientific tier: {tier!r}")
    main_paper = eligibility["main_paper_table"]
    if not isinstance(main_paper, bool):
        raise EligibilityError("main_paper_table must be bool")
    reasons = _require_sequence(eligibility["reason_codes"], "reason_codes")
    if any(not isinstance(value, str) or not value for value in reasons):
        raise EligibilityError("reason_codes must contain non-empty strings")
    if len(reasons) != len(set(reasons)):
        raise EligibilityError("reason_codes must not contain duplicates")
    if not main_paper and not reasons:
        raise EligibilityError("ineligible artifacts require at least one reason code")

    selection = _require_mapping(artifact["selection"], "selection")
    _require_exact_keys(
        selection,
        {
            "source_checkpoint_role",
            "checkpoint_selection_split",
            "tta_calibration_split",
            "test_used_for_checkpoint_selection",
            "test_used_for_tta_selection",
        },
        "selection",
    )
    for key in (
        "source_checkpoint_role",
        "checkpoint_selection_split",
        "tta_calibration_split",
    ):
        _require_nonempty_string(selection[key], f"selection.{key}")
    for key in ("test_used_for_checkpoint_selection", "test_used_for_tta_selection"):
        if not isinstance(selection[key], bool):
            raise EligibilityError(f"selection.{key} must be bool")

    if tier == "development_test_selected" and not (
        selection["test_used_for_checkpoint_selection"]
        or selection["test_used_for_tta_selection"]
    ):
        raise EligibilityError(
            "development_test_selected tier requires an explicit test-selection flag"
        )
    if tier in {"engineering_smoke", "protocol_asset_nonperformance"} and main_paper:
        raise EligibilityError(f"{tier} artifacts cannot enter a main-paper table")
    if tier == "final_paper_benchmark" or main_paper:
        if artifact["artifact_complete"] is not True:
            raise EligibilityError("final/main-paper evidence requires artifact_complete=true")
        if artifact["formal_protocol_complete"] is not True:
            raise EligibilityError(
                "final/main-paper evidence requires formal_protocol_complete=true"
            )
    if main_paper:
        if tier != "final_paper_benchmark":
            raise EligibilityError("main-paper eligibility requires final_paper_benchmark tier")
        if policies["main_paper_requires_no_test_selection"] is not True:
            raise EligibilityError("main-paper no-test-selection policy must be enabled")
        if selection["test_used_for_checkpoint_selection"]:
            raise EligibilityError("test-selected checkpoint cannot enter the main paper table")
        if selection["test_used_for_tta_selection"]:
            raise EligibilityError("test-selected TTA cannot enter the main paper table")
        if reasons:
            raise EligibilityError("eligible main-paper artifacts cannot carry reason codes")


@dataclass(frozen=True)
class LoadedRegistryConfig:
    path: Path
    sha256: str
    output: Path
    value: Mapping[str, Any]


def load_config(config_path: Path, project_root: Path) -> LoadedRegistryConfig:
    root = project_root.resolve()
    path = _project_path(root, config_path, label="config_path")
    snapshot = _stable_regular_file_snapshot(path, retain_bytes=True)
    if snapshot.data is None:
        raise AssertionError("config bytes were not retained")
    try:
        value = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EligibilityError(f"invalid YAML config: {path}") from error
    config = _require_mapping(value, "config")
    _require_exact_keys(
        config,
        {"schema_version", "registry_id", "output", "policies", "artifacts"},
        "config",
    )
    if config["schema_version"] != 1:
        raise EligibilityError("unsupported registry config schema")
    if config["registry_id"] != "cr-sitta-artifact-eligibility-registry-v1":
        raise EligibilityError("unexpected registry_id")
    policies = _require_mapping(config["policies"], "policies")
    _require_exact_keys(
        policies,
        {
            "official_split_model",
            "validation_split_created",
            "main_paper_requires_no_test_selection",
            "existing_artifacts_are_immutable",
        },
        "policies",
    )
    if policies["official_split_model"] != "train_test_only":
        raise EligibilityError("official split model must remain train_test_only")
    if policies["validation_split_created"] is not False:
        raise EligibilityError("this protocol forbids creation of a validation split")
    if policies["main_paper_requires_no_test_selection"] is not True:
        raise EligibilityError("main-paper no-test-selection policy must be enabled")
    if policies["existing_artifacts_are_immutable"] is not True:
        raise EligibilityError("existing artifact immutability must be required")
    output = _resolve_within(root, str(config["output"]), label="output")
    expected_output = root / EXPECTED_REGISTRY_RELATIVE
    if output != expected_output:
        raise EligibilityError(f"registry output must be {expected_output}")
    return LoadedRegistryConfig(
        path=path,
        sha256=snapshot.sha256,
        output=output,
        value=config,
    )


def build_registry(
    config_path: Path,
    *,
    project_root: Path,
    generator_path: Path,
    eligibility_module_path: Path | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    loaded = load_config(config_path, project_root)
    config = loaded.value
    generator = _project_path(project_root, generator_path, label="generator_path")
    module = _project_path(
        project_root,
        eligibility_module_path or Path(__file__),
        label="eligibility_module_path",
    )
    generator_snapshot = _stable_regular_file_snapshot(generator)
    module_snapshot = _stable_regular_file_snapshot(module)
    policies = _require_mapping(config["policies"], "policies")
    raw_artifacts = _require_sequence(config["artifacts"], "artifacts")
    entries: dict[str, Any] = {}
    artifact_ids: set[str] = set()
    for index, raw in enumerate(raw_artifacts):
        artifact = _require_mapping(raw, f"artifact[{index}]")
        _require_exact_keys(
            artifact,
            {
                "artifact_id",
                "root",
                "artifact_role",
                "identity",
                "artifact_complete",
                "formal_protocol_complete",
                "scientific_eligibility",
                "selection",
                "notes",
            },
            f"artifact[{index}]",
        )
        artifact_id = _require_nonempty_string(artifact["artifact_id"], "artifact_id")
        _require_nonempty_string(artifact["artifact_role"], f"{artifact_id}.artifact_role")
        _require_nonempty_string(artifact["notes"], f"{artifact_id}.notes")
        if artifact_id in artifact_ids:
            raise EligibilityError(f"duplicate artifact_id: {artifact_id}")
        artifact_ids.add(artifact_id)
        identity = compute_artifact_identity(project_root, artifact)
        _validate_artifact_completeness(artifact, identity)
        _validate_scientific_eligibility(artifact, policies)
        if identity.sha256 in entries:
            raise EligibilityError(
                f"duplicate artifact identity SHA-256: {identity.sha256}"
            )
        entries[identity.sha256] = {
            "artifact_id": artifact_id,
            "root": artifact["root"],
            "artifact_role": artifact["artifact_role"],
            "artifact_complete": artifact["artifact_complete"],
            "formal_protocol_complete": artifact["formal_protocol_complete"],
            "scientific_eligibility": dict(
                _require_mapping(artifact["scientific_eligibility"], "eligibility")
            ),
            "selection": dict(_require_mapping(artifact["selection"], "selection")),
            "notes": artifact["notes"],
            "identity": identity.to_dict(),
        }

    relative_config = loaded.path.relative_to(project_root).as_posix()
    relative_generator = generator.relative_to(project_root).as_posix()
    relative_module = module.relative_to(project_root).as_posix()
    return {
        "schema_version": 1,
        "registry_id": config["registry_id"],
        "protocol_config": {
            "path": relative_config,
            "sha256": loaded.sha256,
            "single_snapshot_verified": True,
        },
        "generator": {
            "path": relative_generator,
            "sha256": generator_snapshot.sha256,
            "eligibility_module": relative_module,
            "eligibility_module_sha256": module_snapshot.sha256,
        },
        "policies": dict(policies),
        "artifact_count": len(entries),
        "artifacts_by_identity_sha256": dict(sorted(entries.items())),
        "checks": {
            "existing_artifacts_mutated": False,
            "all_artifact_identities_verified": True,
            "all_artifact_completeness_claims_verified": True,
            "scientific_eligibility_gate_verified": True,
            "all_ineligible_artifacts_have_reason_codes": True,
            "validation_split_created": False,
        },
    }


def registry_bytes(registry: Mapping[str, Any]) -> bytes:
    return json.dumps(
        registry,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _read_open_descriptor(descriptor: int) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise EligibilityError("registry output is not a regular file")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise EligibilityError("registry output changed while being read")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise EligibilityError("registry output size changed while being read")
    return payload


def materialize_registry(
    output: Path,
    payload: bytes,
    *,
    replace_existing: bool = False,
) -> str:
    output = Path(os.path.abspath(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise EligibilityError(f"registry output parent cannot be a symlink: {output.parent}")
    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(output.parent, parent_flags)
    temporary_name = f".{output.name}.build-{os.getpid()}-{secrets.token_hex(8)}"
    try:
        try:
            descriptor = os.open(
                output.name,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            existing = None
        except OSError as error:
            raise EligibilityError(f"registry output is unsafe: {output}: {error}") from error
        else:
            try:
                existing = _read_open_descriptor(descriptor)
            finally:
                os.close(descriptor)
        if existing == payload:
            return "already_current"
        if existing is not None and not replace_existing:
            raise FileExistsError(f"refusing to overwrite a different registry: {output}")

        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=parent_fd,
        )
        temporary_exists = True
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if replace_existing:
                os.replace(
                    temporary_name,
                    output.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                temporary_exists = False
                status = "replaced"
            else:
                os.link(
                    temporary_name,
                    output.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                os.unlink(temporary_name, dir_fd=parent_fd)
                temporary_exists = False
                status = "created"
            os.fsync(parent_fd)
        finally:
            if temporary_exists:
                try:
                    os.unlink(temporary_name, dir_fd=parent_fd)
                except FileNotFoundError:
                    pass
        return status
    finally:
        os.close(parent_fd)


def validate_materialized_registry(output: Path, payload: bytes) -> None:
    snapshot = _stable_regular_file_snapshot(output, retain_bytes=True)
    if snapshot.data != payload:
        raise EligibilityError(f"materialized registry differs from expected: {output}")


__all__ = [
    "ALLOWED_TIERS",
    "ArtifactIdentity",
    "EXPECTED_REGISTRY_RELATIVE",
    "EligibilityError",
    "LoadedRegistryConfig",
    "build_registry",
    "compute_artifact_identity",
    "full_tree_identity",
    "load_config",
    "materialize_registry",
    "registry_bytes",
    "sealed_manifest_chain_identity",
    "sha256_file",
    "validate_materialized_registry",
]
