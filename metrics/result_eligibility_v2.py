"""Append-only scientific-eligibility registry v2.

Version 1 is an immutable, materialized registry whose config, generator and
eligibility engine are themselves part of its provenance.  This module never
rewrites those files.  It verifies v1, appends exactly the three completed
``best_pd`` development artifacts, and records that the source-validation
stage was skipped under the user-mandated train/test-only constraint.

The schema supports a pre-publication state for inspecting an intended
contract before all seals exist; such a config is deliberately not buildable.
The checked-in config is frozen only after every declared exact-tree identity
and completion chain has been independently verified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

import result_eligibility as v1


REGISTRY_ID = "cr-sitta-artifact-eligibility-registry-v2"
EXPECTED_REGISTRY_RELATIVE = Path("results/artifact_eligibility_registry_v2.json")
PENDING_STATUS = "pending_artifact_seals"
FROZEN_STATUS = "frozen"
EXPECTED_NEW_ARTIFACTS = {
    "clean_source_best_pd_standardized_export": {
        "root": "results/baseline_checkpoint_axis_v2/best_pd",
        "root_seals": 3,
        "kind": "clean",
    },
    "source_corruption_39_cell_best_pd": {
        "root": "results/source_corruption_checkpoint_axis_v2/best_pd",
        "root_seals": 3,
        "kind": "source39",
    },
    "adabn_batch_stats_39_cell_best_pd": {
        "root": "results/adabn/adabn_checkpoint_axis_v2/best_pd",
        "root_seals": 1,
        "kind": "adabn39",
    },
}
EXPECTED_POLICIES = {
    "official_split_model": "train_test_only",
    "validation_split_created": False,
    "main_paper_requires_no_test_selection": True,
    "existing_artifacts_are_immutable": True,
    "append_only_parent_required": True,
}
EXPECTED_P2_DECISION = {
    "status": "skipped_by_user_constraint",
    "artifact_created": False,
    "validation_split_created": False,
    "source_val_ids_created": False,
    "final_paper_benchmark_authorized": False,
    "reason_codes": ["user_requires_train_test_only", "no_validation_split"],
}
EXPECTED_SELECTION = {
    "source_checkpoint_role": "best_pd",
    "checkpoint_selection_split": "test",
    "tta_calibration_split": "none",
    "test_used_for_checkpoint_selection": True,
    "test_used_for_tta_selection": False,
}
EXPECTED_ELIGIBILITY = {
    "tier": "development_test_selected",
    "main_paper_table": False,
    "reason_codes": ["source_checkpoint_test_selected"],
}
V2_SEALED_CHAIN_ALGORITHM = (
    "verified-exact-tree-with-payload-tree-and-recursive-manifest-chain-v3"
)
P1_PAYLOAD_TREE_ALGORITHM = "sorted-relative-path-tab-sha256-size-lf-v1"


class EligibilityV2Error(v1.EligibilityError):
    """Raised when the append-only v2 contract is not publishable."""


class _V2SealedManifestChainVerifier(v1._SealedManifestChainVerifier):
    """Verify both legacy ledgers and checkpoint-axis ``payload_tree`` ledgers.

    P1 clean artifacts intentionally store their payload ledger as a sorted
    record list below ``payload_tree.files``.  AdaBN condition manifests also
    carry an external Source lineage containing digests but no in-tree paths.
    The v1 verifier predates both shapes, so v2 extends only those two parsing
    rules while retaining v1's stable-file, no-symlink and exact-tree gates.
    """

    def _verify_payload_tree(self, manifest_path: Path, raw: Any) -> None:
        tree = _mapping(raw, f"{manifest_path} payload_tree")
        _exact_keys(
            tree,
            {"algorithm", "file_count", "files", "sha256"},
            f"{manifest_path} payload_tree",
        )
        if tree["algorithm"] != P1_PAYLOAD_TREE_ALGORITHM:
            raise EligibilityV2Error(
                f"unsupported payload_tree algorithm in {manifest_path}: "
                f"{tree['algorithm']!r}"
            )
        raw_count = tree["file_count"]
        if type(raw_count) is not int or raw_count < 0:
            raise EligibilityV2Error(f"invalid payload_tree file_count: {manifest_path}")
        expected_tree_sha = _sha256(tree["sha256"], f"{manifest_path} payload_tree.sha256")
        records = _sequence(tree["files"], f"{manifest_path} payload_tree.files")
        if len(records) != raw_count:
            raise EligibilityV2Error(
                f"payload_tree file_count differs in {manifest_path}: "
                f"{len(records)} != {raw_count}"
            )
        digest = hashlib.sha256()
        previous: str | None = None
        for index, raw_record in enumerate(records):
            record = _mapping(raw_record, f"payload_tree.files[{index}]")
            _exact_keys(
                record,
                {"path", "sha256", "size_bytes"},
                f"payload_tree.files[{index}]",
            )
            raw_path = record["path"]
            if not isinstance(raw_path, str) or not raw_path:
                raise EligibilityV2Error("payload_tree path must be non-empty string")
            relative_path = Path(raw_path)
            if (
                relative_path.is_absolute()
                or relative_path.as_posix() != raw_path
                or any(part in {"", ".", ".."} for part in relative_path.parts)
            ):
                raise EligibilityV2Error(
                    f"payload_tree path is not canonical: {raw_path!r}"
                )
            if previous is not None and raw_path <= previous:
                raise EligibilityV2Error(
                    f"payload_tree paths must be strictly sorted and unique: {manifest_path}"
                )
            previous = raw_path
            expected_sha = _sha256(
                record["sha256"], f"payload_tree.files[{index}].sha256"
            )
            size = record["size_bytes"]
            if type(size) is not int or size < 0:
                raise EligibilityV2Error(
                    f"payload_tree.files[{index}].size_bytes must be non-negative int"
                )
            path = v1._resolve_reference(
                self.root,
                manifest_path.parent,
                raw_path,
                label=f"payload_tree.files[{index}].path",
            )
            self._snapshot(
                path,
                expected_sha256=expected_sha,
                expected_size=size,
                role="payload_tree_payload",
            )
            digest.update(f"{raw_path}\t{expected_sha}\t{size}\n".encode("utf-8"))
        if digest.hexdigest() != expected_tree_sha:
            raise EligibilityV2Error(
                f"payload_tree digest differs in {manifest_path}: "
                f"{digest.hexdigest()} != {expected_tree_sha}"
            )

    def _verify_payload_ledgers(
        self, manifest_path: Path, manifest: Mapping[str, Any]
    ) -> None:
        has_legacy = manifest.get("files") is not None or manifest.get("files_sha256") is not None
        has_payload_tree = manifest.get("payload_tree") is not None
        if has_legacy:
            super()._verify_payload_ledgers(manifest_path, manifest)
        if has_payload_tree:
            self._verify_payload_tree(manifest_path, manifest["payload_tree"])
        if not has_legacy and not has_payload_tree:
            raise EligibilityV2Error(f"manifest has no supported payload ledger: {manifest_path}")

    def _discover_child_seals(self, value: Any, *, declaring_manifest: Path) -> None:
        if isinstance(value, Mapping):
            artifact_path = "artifact_manifest" in value
            generic_path = "manifest" in value
            completion_path = "completion" in value
            completion_digest = "completion_sha256" in value
            # Digest-only external lineage (for example source_artifact with
            # artifact_manifest_sha256 + complete_sha256) is not an in-tree
            # child seal.  Any in-tree path or the canonical completion digest
            # key, however, declares internal-seal intent and must be complete.
            internal_intent = artifact_path or generic_path or completion_path or completion_digest
            if internal_intent:
                if artifact_path and generic_path:
                    raise EligibilityV2Error(
                        f"ambiguous child manifest path in {declaring_manifest}"
                    )
                manifest_key = "artifact_manifest" if artifact_path else "manifest"
                digest_key = (
                    "artifact_manifest_sha256" if artifact_path else "manifest_sha256"
                )
                required = {manifest_key, digest_key, "completion", "completion_sha256"}
                if not (artifact_path or generic_path) or any(key not in value for key in required):
                    raise EligibilityV2Error(
                        f"partial internal child manifest/completion seal in {declaring_manifest}"
                    )
                self._process_seal(
                    declaring_directory=declaring_manifest.parent,
                    manifest_raw=value[manifest_key],
                    manifest_sha256=value[digest_key],
                    completion_raw=value["completion"],
                    completion_sha256=value["completion_sha256"],
                )
            for key, child in value.items():
                if key not in {"files", "files_sha256", "payload_tree"}:
                    self._discover_child_seals(child, declaring_manifest=declaring_manifest)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for child in value:
                self._discover_child_seals(child, declaring_manifest=declaring_manifest)

    def _process_seal(self, **kwargs: Any) -> None:
        super()._process_seal(**kwargs)
        declaring_directory = kwargs["declaring_directory"]
        manifest_path = v1._resolve_reference(
            self.root,
            declaring_directory,
            kwargs["manifest_raw"],
            label="manifest declaration",
        )
        completion_path = v1._resolve_reference(
            self.root,
            declaring_directory,
            kwargs["completion_raw"],
            label="completion declaration",
        )
        manifest = self.manifests[manifest_path]
        completion = self.completions[completion_path]
        tree = manifest.get("payload_tree")
        if tree is not None:
            tree_mapping = _mapping(tree, f"{manifest_path} payload_tree")
            if completion.get("payload_tree_sha256") != tree_mapping.get("sha256"):
                raise EligibilityV2Error(
                    f"completion does not bind payload_tree SHA-256: {completion_path}"
                )
            if completion.get("payload_file_count") != tree_mapping.get("file_count"):
                raise EligibilityV2Error(
                    f"completion does not bind payload_tree file_count: {completion_path}"
                )

    def verify(self, *args: Any, **kwargs: Any) -> v1.ArtifactIdentity:
        identity = super().verify(*args, **kwargs)
        verification = dict(identity.verification)
        verification["payload_file_count"] = sum(
            1
            for roles in self.roles.values()
            if roles & {"payload", "payload_tree_payload", "supplemental_payload"}
        )
        return replace(
            identity,
            algorithm=V2_SEALED_CHAIN_ALGORITHM,
            verification=verification,
        )


def sealed_manifest_chain_identity_v2(
    root: Path,
    seals: Sequence[Any],
    *,
    exact_tree: bool,
    supplemental_files: Sequence[Any] = (),
) -> v1.ArtifactIdentity:
    """Return a strict v2 identity without modifying the artifact tree."""
    try:
        return _V2SealedManifestChainVerifier(root).verify(
            seals,
            exact_tree=exact_tree,
            supplemental_files=supplemental_files,
        )
    except EligibilityV2Error:
        raise
    except v1.EligibilityError as error:
        raise EligibilityV2Error(str(error)) from error


def compute_artifact_identity_v2(
    project_root: Path,
    artifact: Mapping[str, Any],
    *,
    verify_expected: bool = True,
) -> v1.ArtifactIdentity:
    """Compute a registry-v2 identity using the extended sealed verifier."""

    identity_spec = _mapping(artifact.get("identity"), "artifact.identity")
    if identity_spec.get("mode") != "sealed_manifest_chain":
        return v1.compute_artifact_identity(
            project_root, artifact, verify_expected=verify_expected
        )
    root = _project_path(project_root, artifact.get("root"), "artifact.root")
    observed = sealed_manifest_chain_identity_v2(
        root,
        _sequence(identity_spec.get("seals"), "identity.seals"),
        exact_tree=identity_spec.get("exact_tree"),
        supplemental_files=_sequence(
            identity_spec.get("supplemental_files", ()), "identity.supplemental_files"
        ),
    )
    expected = _sha256(identity_spec.get("expected_sha256"), "identity.expected_sha256")
    if verify_expected and observed.sha256 != expected:
        raise EligibilityV2Error(
            f"artifact identity drift for {artifact.get('artifact_id')}: "
            f"{observed.sha256} != {expected}"
        )
    return observed


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    observed = set(value)
    if observed != expected:
        raise EligibilityV2Error(
            f"{label} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EligibilityV2Error(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EligibilityV2Error(f"{label} must be a sequence")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EligibilityV2Error(f"{label} must be a lowercase SHA-256 digest")
    return value


def _project_path(project_root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise EligibilityV2Error(f"{label} must be a project-relative path")
    if any(part in {"", ".", ".."} for part in Path(raw).parts):
        raise EligibilityV2Error(f"{label} is not canonical: {raw!r}")
    try:
        return v1._resolve_within(project_root.resolve(), raw, label=label)
    except v1.EligibilityError as error:
        raise EligibilityV2Error(str(error)) from error


def _snapshot_yaml(path: Path) -> tuple[Mapping[str, Any], str]:
    snapshot = v1._stable_regular_file_snapshot(path, retain_bytes=True)
    if snapshot.data is None:
        raise AssertionError("config bytes were not retained")
    try:
        parsed = yaml.safe_load(snapshot.data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise EligibilityV2Error(f"invalid v2 YAML: {path}") from error
    return _mapping(parsed, "v2 config"), snapshot.sha256


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    snapshot = v1._stable_regular_file_snapshot(path, retain_bytes=True)
    if snapshot.data is None:
        raise AssertionError("JSON bytes were not retained")
    try:
        return _mapping(json.loads(snapshot.data.decode("utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EligibilityV2Error(f"invalid {label}: {path}") from error


def _validate_parent_shape(parent: Mapping[str, Any]) -> None:
    _exact_keys(
        parent,
        {
            "registry_id",
            "config_path",
            "config_sha256",
            "materialized_path",
            "materialized_sha256",
            "generator_path",
            "generator_sha256",
            "eligibility_module_path",
            "eligibility_module_sha256",
            "artifact_count",
            "append_only",
        },
        "parent_registry",
    )
    if parent["registry_id"] != "cr-sitta-artifact-eligibility-registry-v1":
        raise EligibilityV2Error("v2 parent must be the frozen v1 registry")
    if parent["artifact_count"] != 11 or parent["append_only"] is not True:
        raise EligibilityV2Error("v2 parent count/append-only contract differs")
    expected_paths = {
        "config_path": "configs/artifact_eligibility_registry_v1.yaml",
        "materialized_path": "results/artifact_eligibility_registry_v1.json",
        "generator_path": "scripts/validate_result_eligibility.py",
        "eligibility_module_path": "metrics/result_eligibility.py",
    }
    for key, expected in expected_paths.items():
        if parent[key] != expected:
            raise EligibilityV2Error(f"parent_registry.{key} must be {expected}")
    for key in (
        "config_sha256",
        "materialized_sha256",
        "generator_sha256",
        "eligibility_module_sha256",
    ):
        _sha256(parent[key], f"parent_registry.{key}")


def _validate_stage_decisions(value: Mapping[str, Any]) -> None:
    _exact_keys(value, {"P2_source_val"}, "stage_decisions")
    decision = _mapping(value["P2_source_val"], "stage_decisions.P2_source_val")
    if dict(decision) != EXPECTED_P2_DECISION:
        raise EligibilityV2Error(
            "P2_source_val must remain skipped_by_user_constraint with no artifact, "
            "validation split, source-val IDs, or final-paper authorization"
        )


def _validate_artifact_declaration(
    artifact: Mapping[str, Any], *, status: str
) -> None:
    _exact_keys(
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
        "new artifact",
    )
    artifact_id = artifact.get("artifact_id")
    if artifact_id not in EXPECTED_NEW_ARTIFACTS:
        raise EligibilityV2Error(f"unexpected v2 artifact_id: {artifact_id!r}")
    expected = EXPECTED_NEW_ARTIFACTS[str(artifact_id)]
    if artifact.get("root") != expected["root"]:
        raise EligibilityV2Error(f"{artifact_id} must use canonical root {expected['root']}")
    if dict(_mapping(artifact.get("scientific_eligibility"), "eligibility")) != EXPECTED_ELIGIBILITY:
        raise EligibilityV2Error(f"{artifact_id} eligibility contract differs")
    if dict(_mapping(artifact.get("selection"), "selection")) != EXPECTED_SELECTION:
        raise EligibilityV2Error(f"{artifact_id} selection contract differs")
    identity = _mapping(artifact.get("identity"), "identity")
    _exact_keys(
        identity,
        {"mode", "expected_sha256", "exact_tree", "seals"},
        f"{artifact_id}.identity",
    )
    if identity.get("mode") != "sealed_manifest_chain" or identity.get("exact_tree") is not True:
        raise EligibilityV2Error(f"{artifact_id} requires exact sealed_manifest_chain identity")
    seals = _sequence(identity.get("seals"), f"{artifact_id}.identity.seals")
    if status == PENDING_STATUS:
        if artifact.get("artifact_complete") is not False:
            raise EligibilityV2Error("pending artifact declarations must be incomplete")
        if artifact.get("formal_protocol_complete") is not False:
            raise EligibilityV2Error("pending artifact declarations cannot be formal-complete")
        if identity.get("expected_sha256") is not None or len(seals) != 0:
            raise EligibilityV2Error("pending artifact declarations cannot freeze identity/seals")
    else:
        if artifact.get("artifact_complete") is not True:
            raise EligibilityV2Error("frozen v2 artifacts require artifact_complete=true")
        if artifact.get("formal_protocol_complete") is not True:
            raise EligibilityV2Error("frozen v2 artifacts require formal_protocol_complete=true")
        _sha256(identity.get("expected_sha256"), f"{artifact_id}.identity.expected_sha256")
        if len(seals) != expected["root_seals"]:
            raise EligibilityV2Error(
                f"{artifact_id} requires exactly {expected['root_seals']} root seals"
            )


def load_v2_config(config_path: Path, project_root: Path) -> tuple[Mapping[str, Any], str]:
    root = project_root.resolve()
    path = config_path if config_path.is_absolute() else root / config_path
    path = _project_path(root, path.relative_to(root).as_posix(), "v2 config path")
    config, digest = _snapshot_yaml(path)
    _exact_keys(
        config,
        {
            "schema_version",
            "registry_id",
            "output",
            "status",
            "parent_registry",
            "policies",
            "stage_decisions",
            "artifacts",
        },
        "v2 config",
    )
    if config.get("schema_version") != 2 or config.get("registry_id") != REGISTRY_ID:
        raise EligibilityV2Error("unexpected v2 schema/registry_id")
    if config.get("output") != EXPECTED_REGISTRY_RELATIVE.as_posix():
        raise EligibilityV2Error("v2 output path differs from its immutable versioned path")
    status = config.get("status")
    if status not in {PENDING_STATUS, FROZEN_STATUS}:
        raise EligibilityV2Error(f"unsupported v2 publication status: {status!r}")
    _validate_parent_shape(_mapping(config.get("parent_registry"), "parent_registry"))
    if dict(_mapping(config.get("policies"), "policies")) != EXPECTED_POLICIES:
        raise EligibilityV2Error("v2 policies differ from the frozen train/test-only policy")
    _validate_stage_decisions(_mapping(config.get("stage_decisions"), "stage_decisions"))
    artifacts = _sequence(config.get("artifacts"), "artifacts")
    ids = []
    for raw in artifacts:
        artifact = _mapping(raw, "new artifact")
        _validate_artifact_declaration(artifact, status=str(status))
        ids.append(artifact["artifact_id"])
    if len(ids) != 3 or set(ids) != set(EXPECTED_NEW_ARTIFACTS) or len(set(ids)) != 3:
        raise EligibilityV2Error("v2 must declare exactly the three best_pd artifacts")
    return config, digest


def _verify_parent(config: Mapping[str, Any], project_root: Path) -> Mapping[str, Any]:
    root = project_root.resolve()
    parent = _mapping(config["parent_registry"], "parent_registry")
    paths = {
        key: _project_path(root, parent[key], f"parent_registry.{key}")
        for key in (
            "config_path",
            "materialized_path",
            "generator_path",
            "eligibility_module_path",
        )
    }
    sha_keys = {
        "config_path": "config_sha256",
        "materialized_path": "materialized_sha256",
        "generator_path": "generator_sha256",
        "eligibility_module_path": "eligibility_module_sha256",
    }
    for path_key, sha_key in sha_keys.items():
        observed = v1.sha256_file(paths[path_key])
        if observed != parent[sha_key]:
            raise EligibilityV2Error(
                f"immutable v1 {path_key} drift: {observed} != {parent[sha_key]}"
            )
    expected_parent = v1.build_registry(
        paths["config_path"],
        project_root=root,
        generator_path=paths["generator_path"],
        eligibility_module_path=paths["eligibility_module_path"],
    )
    expected_bytes = v1.registry_bytes(expected_parent)
    v1.validate_materialized_registry(paths["materialized_path"], expected_bytes)
    materialized = _load_json(paths["materialized_path"], "parent registry JSON")
    if materialized.get("artifact_count") != parent["artifact_count"]:
        raise EligibilityV2Error("parent artifact count differs")
    return materialized


def _verify_kind_gates(
    artifact: Mapping[str, Any], identity: v1.ArtifactIdentity, project_root: Path
) -> None:
    artifact_id = str(artifact["artifact_id"])
    spec = EXPECTED_NEW_ARTIFACTS[artifact_id]
    if identity.verification.get("root_seal_count") != spec["root_seals"]:
        raise EligibilityV2Error(f"{artifact_id} root seal count differs")
    root = _project_path(project_root, artifact["root"], f"{artifact_id}.root")
    if spec["kind"] in {"clean", "source39"}:
        dataset_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
        if dataset_dirs != ["IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"]:
            raise EligibilityV2Error(f"{artifact_id} dataset set differs: {dataset_dirs}")
        for dataset in dataset_dirs:
            completion = _load_json(root / dataset / "COMPLETE.json", "dataset COMPLETE")
            if completion.get("checkpoint_role") != "best_pd":
                raise EligibilityV2Error(f"{artifact_id}/{dataset} is not best_pd")
            if spec["kind"] == "source39" and completion.get("condition_count") != 13:
                raise EligibilityV2Error(f"{artifact_id}/{dataset} is not a 13-condition shard")
    else:
        completion = _load_json(root / "COMPLETE.json", "AdaBN global COMPLETE")
        aggregate = _load_json(root / "aggregate_metrics.json", "AdaBN aggregate")
        if completion.get("complete") is not True or completion.get("scope") != "global":
            raise EligibilityV2Error("AdaBN global completion seal is absent or invalid")
        for value, label in ((completion, "AdaBN COMPLETE"), (aggregate, "AdaBN aggregate")):
            if value.get("checkpoint_role") != "best_pd":
                raise EligibilityV2Error(f"{label} is not best_pd")
            if value.get("extra_best_pd_tuning_episodes") != 0:
                raise EligibilityV2Error(f"{label} reports extra best_pd tuning")
        if aggregate.get("dataset_count") != 3 or aggregate.get("condition_count_per_dataset") != 13:
            raise EligibilityV2Error("AdaBN aggregate is not exactly 3 datasets x 13 conditions")
        checks = _mapping(aggregate.get("checks"), "AdaBN aggregate.checks")
        if not checks or any(value is not True for value in checks.values()):
            raise EligibilityV2Error("AdaBN aggregate contains a failed completion check")


def inspect_v2_config(config_path: Path, project_root: Path) -> Mapping[str, Any]:
    config, digest = load_v2_config(config_path, project_root)
    root = project_root.resolve()
    records = []
    for raw in _sequence(config["artifacts"], "artifacts"):
        artifact = _mapping(raw, "artifact")
        artifact_root = _project_path(root, artifact["root"], "artifact.root")
        dataset_completions = sorted(
            path.relative_to(artifact_root).as_posix()
            for path in artifact_root.glob("*/COMPLETE.json")
            if path.is_file() and not path.is_symlink()
        ) if artifact_root.is_dir() else []
        global_complete = artifact_root / "COMPLETE.json"
        identity = _mapping(artifact["identity"], "identity")
        records.append(
            {
                "artifact_id": artifact["artifact_id"],
                "root": artifact["root"],
                "root_exists": artifact_root.is_dir() and not artifact_root.is_symlink(),
                "dataset_completion_count": len(dataset_completions),
                "global_complete_exists": global_complete.is_file() and not global_complete.is_symlink(),
                "identity_frozen": identity.get("expected_sha256") is not None and bool(identity.get("seals")),
                "publication_ready": config["status"] == FROZEN_STATUS,
            }
        )
    return {
        "inspection_only": True,
        "registry_id": REGISTRY_ID,
        "status": config["status"],
        "config_sha256": digest,
        "materialization_authorized": config["status"] == FROZEN_STATUS,
        "stage_decisions": dict(config["stage_decisions"]),
        "artifacts": records,
    }


def build_v2_registry(
    config_path: Path,
    *,
    project_root: Path,
    generator_path: Path,
    eligibility_module_path: Path | None = None,
) -> Mapping[str, Any]:
    root = project_root.resolve()
    config, config_sha = load_v2_config(config_path, root)
    if config["status"] != FROZEN_STATUS:
        raise EligibilityV2Error(
            "v2 registry is pending_artifact_seals; materialize/validate are forbidden"
        )
    parent = _verify_parent(config, root)
    policies_for_v1 = {
        key: config["policies"][key]
        for key in (
            "official_split_model",
            "validation_split_created",
            "main_paper_requires_no_test_selection",
            "existing_artifacts_are_immutable",
        )
    }
    entries = dict(_mapping(parent["artifacts_by_identity_sha256"], "parent artifacts"))
    parent_ids = {value["artifact_id"] for value in entries.values()}
    parent_roots = {value["root"] for value in entries.values()}
    new_ids: list[str] = []
    for raw in _sequence(config["artifacts"], "artifacts"):
        artifact = _mapping(raw, "artifact")
        identity = compute_artifact_identity_v2(root, artifact)
        v1._validate_artifact_completeness(artifact, identity)
        v1._validate_scientific_eligibility(artifact, policies_for_v1)
        _verify_kind_gates(artifact, identity, root)
        artifact_id = str(artifact["artifact_id"])
        if artifact_id in parent_ids or artifact["root"] in parent_roots or identity.sha256 in entries:
            raise EligibilityV2Error("v2 append collides with an immutable v1 artifact")
        new_ids.append(artifact_id)
        entries[identity.sha256] = {
            "artifact_id": artifact_id,
            "root": artifact["root"],
            "artifact_role": artifact["artifact_role"],
            "artifact_complete": artifact["artifact_complete"],
            "formal_protocol_complete": artifact["formal_protocol_complete"],
            "scientific_eligibility": dict(artifact["scientific_eligibility"]),
            "selection": dict(artifact["selection"]),
            "notes": artifact["notes"],
            "identity": identity.to_dict(),
        }
    if set(new_ids) != set(EXPECTED_NEW_ARTIFACTS):
        raise EligibilityV2Error("effective v2 additions differ from the three best_pd artifacts")
    if any(
        value["scientific_eligibility"]["main_paper_table"] is True
        or value["scientific_eligibility"]["tier"] == "final_paper_benchmark"
        for value in entries.values()
    ):
        raise EligibilityV2Error("P2 skipped status forbids final-paper benchmark entries")

    generator = generator_path if generator_path.is_absolute() else root / generator_path
    module = eligibility_module_path or Path(__file__)
    module = module if module.is_absolute() else root / module
    generator_sha = v1.sha256_file(generator)
    module_sha = v1.sha256_file(module)
    return {
        "schema_version": 2,
        "registry_id": REGISTRY_ID,
        "status": FROZEN_STATUS,
        "protocol_config": {
            "path": Path(config_path).resolve().relative_to(root).as_posix(),
            "sha256": config_sha,
            "single_snapshot_verified": True,
        },
        "generator": {
            "path": Path(generator).resolve().relative_to(root).as_posix(),
            "sha256": generator_sha,
            "eligibility_module": Path(module).resolve().relative_to(root).as_posix(),
            "eligibility_module_sha256": module_sha,
            "base_v1_eligibility_module": config["parent_registry"]["eligibility_module_path"],
            "base_v1_eligibility_module_sha256": config["parent_registry"]["eligibility_module_sha256"],
        },
        "parent_registry": dict(config["parent_registry"]),
        "policies": dict(config["policies"]),
        "stage_decisions": dict(config["stage_decisions"]),
        "parent_artifact_count": parent["artifact_count"],
        "new_artifact_count": len(new_ids),
        "artifact_count": len(entries),
        "new_artifact_ids": sorted(new_ids),
        "artifacts_by_identity_sha256": dict(sorted(entries.items())),
        "checks": {
            "immutable_parent_v1_byte_validated": True,
            "append_only_no_collisions": True,
            "all_new_artifact_identities_verified": True,
            "all_new_completion_chains_verified": True,
            "all_new_artifacts_development_test_selected": True,
            "all_new_artifacts_excluded_from_main_paper": True,
            "p2_source_val_status_recorded": True,
            "validation_split_created": False,
            "final_paper_benchmark_authorized": False,
        },
    }


def registry_bytes(registry: Mapping[str, Any]) -> bytes:
    return json.dumps(
        registry, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"


__all__ = [
    "EligibilityV2Error",
    "EXPECTED_REGISTRY_RELATIVE",
    "FROZEN_STATUS",
    "PENDING_STATUS",
    "build_v2_registry",
    "compute_artifact_identity_v2",
    "inspect_v2_config",
    "load_v2_config",
    "registry_bytes",
    "sealed_manifest_chain_identity_v2",
]
