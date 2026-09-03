from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = PROJECT_ROOT / "metrics"
if str(METRICS_ROOT) not in sys.path:
    sys.path.insert(0, str(METRICS_ROOT))

from result_eligibility_v2 import (  # noqa: E402
    EligibilityV2Error,
    EXPECTED_NEW_ARTIFACTS,
    EXPECTED_P2_DECISION,
    FROZEN_STATUS,
    PENDING_STATUS,
    build_v2_registry,
    inspect_v2_config,
    load_v2_config,
    sealed_manifest_chain_identity_v2,
)


CONFIG = PROJECT_ROOT / "configs" / "artifact_eligibility_registry_v2.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, value: object) -> Path:
    # load_v2_config requires the config to stay inside its declared project.
    project = tmp_path / "project"
    path = project / "configs" / "artifact_eligibility_registry_v2.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _as_pending(value: dict[str, object]) -> dict[str, object]:
    pending = copy.deepcopy(value)
    pending["status"] = PENDING_STATUS
    for artifact in pending["artifacts"]:
        artifact["artifact_complete"] = False
        artifact["formal_protocol_complete"] = False
        artifact["identity"]["expected_sha256"] = None
        artifact["identity"]["seals"] = []
    return pending


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return _sha256(path)


def _tree_record(relative: str, payload: bytes) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _tree_sha(records: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['path']}\t{record['sha256']}\t{record['size_bytes']}\n".encode()
        )
    return digest.hexdigest()


def _payload_tree_fixture(
    tmp_path: Path, *, external_lineage: bool = False, partial_child: bool = False
) -> tuple[Path, list[dict[str, str]]]:
    root = tmp_path / "artifact"
    root.mkdir()
    payload = b"payload-v1"
    (root / "payload.bin").write_bytes(payload)
    records = [_tree_record("payload.bin", payload)]
    manifest: dict[str, object] = {
        "payload_tree": {
            "algorithm": "sorted-relative-path-tab-sha256-size-lf-v1",
            "file_count": len(records),
            "files": records,
            "sha256": _tree_sha(records),
        }
    }
    if external_lineage:
        manifest["source_artifact"] = {
            "root": "/external/source",
            "artifact_manifest_sha256": "1" * 64,
            "complete_sha256": "2" * 64,
        }
    if partial_child:
        manifest["children"] = {
            "broken": {
                "artifact_manifest": "child/artifact_manifest.json",
                "artifact_manifest_sha256": "3" * 64,
                "completion_sha256": "4" * 64,
            }
        }
    manifest_sha = _write_json(root / "artifact_manifest.json", manifest)
    complete_sha = _write_json(
        root / "COMPLETE.json",
        {
            "complete": True,
            "manifest_sha256": manifest_sha,
            "payload_tree_sha256": _tree_sha(records),
            "payload_file_count": len(records),
        },
    )
    seals = [
        {
            "manifest": "artifact_manifest.json",
            "manifest_sha256": manifest_sha,
            "completion": "COMPLETE.json",
            "completion_sha256": complete_sha,
        }
    ]
    return root, seals


def test_checked_in_config_is_inspectable_and_frozen() -> None:
    config, _ = load_v2_config(CONFIG, PROJECT_ROOT)
    inspection = inspect_v2_config(CONFIG, PROJECT_ROOT)

    assert config["status"] == FROZEN_STATUS
    assert inspection["inspection_only"] is True
    assert inspection["materialization_authorized"] is True
    assert len(inspection["artifacts"]) == 3
    assert {row["artifact_id"] for row in inspection["artifacts"]} == set(
        EXPECTED_NEW_ARTIFACTS
    )
    assert all(row["identity_frozen"] is True for row in inspection["artifacts"])


def test_pending_config_build_fails_before_parent_or_artifact_scan(
    tmp_path: Path,
) -> None:
    value = _as_pending(_raw_config())
    path = _write_config(tmp_path, value)
    with pytest.raises(EligibilityV2Error, match="pending_artifact_seals"):
        build_v2_registry(
            path,
            project_root=path.parents[1],
            generator_path=path.parents[1] / "scripts" / "validate_result_eligibility_v2.py",
        )


def test_cli_expected_payload_routes_frozen_config(monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = PROJECT_ROOT / "scripts" / "validate_result_eligibility_v2.py"
    spec = importlib.util.spec_from_file_location("eligibility_v2_cli", module_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    observed: dict[str, object] = {}

    def fake_build(config_path: Path, **kwargs: object) -> dict[str, object]:
        observed["config_path"] = config_path
        observed.update(kwargs)
        return {"registry_id": "fixture"}

    monkeypatch.setattr(cli, "build_v2_registry", fake_build)
    output, payload = cli.expected_payload(CONFIG)
    assert observed["config_path"] == CONFIG
    assert observed["project_root"] == PROJECT_ROOT
    assert output == PROJECT_ROOT / "results/artifact_eligibility_registry_v2.json"
    assert b'"registry_id": "fixture"' in payload


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_v1_four_immutable_files_match_parent_contract() -> None:
    config = _raw_config()
    parent = config["parent_registry"]
    pairs = {
        "config_path": "config_sha256",
        "materialized_path": "materialized_sha256",
        "generator_path": "generator_sha256",
        "eligibility_module_path": "eligibility_module_sha256",
    }
    for path_key, hash_key in pairs.items():
        assert _sha256(PROJECT_ROOT / parent[path_key]) == parent[hash_key]


def test_p2_skip_is_explicit_and_blocks_final_paper_authorization() -> None:
    config, _ = load_v2_config(CONFIG, PROJECT_ROOT)
    decision = config["stage_decisions"]["P2_source_val"]
    assert decision == EXPECTED_P2_DECISION
    assert decision["status"] == "skipped_by_user_constraint"
    assert decision["final_paper_benchmark_authorized"] is False
    assert all(
        artifact["scientific_eligibility"]["main_paper_table"] is False
        for artifact in config["artifacts"]
    )


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda value: value["artifacts"][0].update({"artifact_complete": False}),
            "artifact_complete=true",
        ),
        (
            lambda value: value["artifacts"][0]["scientific_eligibility"].update(
                {"main_paper_table": True}
            ),
            "eligibility contract differs",
        ),
        (
            lambda value: value["stage_decisions"]["P2_source_val"].update(
                {"source_val_ids_created": True}
            ),
            "P2_source_val",
        ),
        (
            lambda value: value["artifacts"][0].update(
                {"root": "results/noncanonical-best-pd"}
            ),
            "canonical root",
        ),
        (
            lambda value: value["artifacts"][1].update(
                {"artifact_id": value["artifacts"][0]["artifact_id"]}
            ),
            "canonical root|exactly the three",
        ),
        (
            lambda value: value["parent_registry"].update(
                {"materialized_path": "results/another-v1.json"}
            ),
            "parent_registry.materialized_path",
        ),
    ],
)
def test_frozen_contract_rejects_unsafe_claims(
    tmp_path: Path, mutation: object, message: str
) -> None:
    value = copy.deepcopy(_raw_config())
    mutation(value)
    path = _write_config(tmp_path, value)
    with pytest.raises(EligibilityV2Error, match=message):
        load_v2_config(path, path.parents[1])


def test_frozen_status_requires_all_identity_seals(tmp_path: Path) -> None:
    value = copy.deepcopy(_raw_config())
    value["artifacts"][0]["identity"]["seals"] = []
    path = _write_config(tmp_path, value)
    with pytest.raises(EligibilityV2Error, match="requires exactly 3 root seals"):
        load_v2_config(path, path.parents[1])


def test_payload_tree_list_ledger_and_completion_binding_pass(tmp_path: Path) -> None:
    root, seals = _payload_tree_fixture(tmp_path)
    identity = sealed_manifest_chain_identity_v2(root, seals, exact_tree=True)

    assert identity.file_count == 3
    assert identity.verification["exact_tree_verified"] is True
    assert identity.verification["completion_chain_verified"] is True
    assert identity.verification["payload_file_count"] == 1


def test_external_source_lineage_is_not_a_child_seal(tmp_path: Path) -> None:
    root, seals = _payload_tree_fixture(tmp_path, external_lineage=True)
    identity = sealed_manifest_chain_identity_v2(root, seals, exact_tree=True)

    assert identity.verification["recursive_seal_count"] == 1
    assert identity.verification["manifest_count"] == 1


def test_true_partial_internal_child_seal_is_rejected(tmp_path: Path) -> None:
    root, seals = _payload_tree_fixture(tmp_path, partial_child=True)
    with pytest.raises(EligibilityV2Error, match="partial internal child"):
        sealed_manifest_chain_identity_v2(root, seals, exact_tree=True)


@pytest.mark.parametrize("tamper", ["payload", "size", "digest", "tree_sha", "count"])
def test_payload_tree_tamper_is_rejected(tmp_path: Path, tamper: str) -> None:
    root, seals = _payload_tree_fixture(tmp_path)
    manifest_path = root / "artifact_manifest.json"
    complete_path = root / "COMPLETE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if tamper == "payload":
        (root / "payload.bin").write_bytes(b"tampered")
    elif tamper == "size":
        manifest["payload_tree"]["files"][0]["size_bytes"] += 1
    elif tamper == "digest":
        manifest["payload_tree"]["files"][0]["sha256"] = "0" * 64
    elif tamper == "tree_sha":
        manifest["payload_tree"]["sha256"] = "0" * 64
    else:
        complete["payload_file_count"] += 1
    if tamper in {"size", "digest", "tree_sha"}:
        manifest_sha = _write_json(manifest_path, manifest)
        complete["manifest_sha256"] = manifest_sha
        seals[0]["manifest_sha256"] = manifest_sha
    complete_sha = _write_json(complete_path, complete)
    seals[0]["completion_sha256"] = complete_sha

    with pytest.raises(EligibilityV2Error):
        sealed_manifest_chain_identity_v2(root, seals, exact_tree=True)


def test_payload_tree_path_escape_is_rejected(tmp_path: Path) -> None:
    root, seals = _payload_tree_fixture(tmp_path)
    manifest_path = root / "artifact_manifest.json"
    complete_path = root / "COMPLETE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest["payload_tree"]["files"][0]["path"] = "../escape.bin"
    manifest["payload_tree"]["sha256"] = _tree_sha(manifest["payload_tree"]["files"])
    manifest_sha = _write_json(manifest_path, manifest)
    complete["manifest_sha256"] = manifest_sha
    complete["payload_tree_sha256"] = manifest["payload_tree"]["sha256"]
    seals[0]["manifest_sha256"] = manifest_sha
    seals[0]["completion_sha256"] = _write_json(complete_path, complete)

    with pytest.raises(EligibilityV2Error, match="not canonical|escapes"):
        sealed_manifest_chain_identity_v2(root, seals, exact_tree=True)


def test_payload_tree_noncanonical_path_is_rejected(tmp_path: Path) -> None:
    root, seals = _payload_tree_fixture(tmp_path)
    manifest_path = root / "artifact_manifest.json"
    complete_path = root / "COMPLETE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    manifest["payload_tree"]["files"][0]["path"] = "./payload.bin"
    manifest["payload_tree"]["sha256"] = _tree_sha(manifest["payload_tree"]["files"])
    manifest_sha = _write_json(manifest_path, manifest)
    complete["manifest_sha256"] = manifest_sha
    complete["payload_tree_sha256"] = manifest["payload_tree"]["sha256"]
    seals[0]["manifest_sha256"] = manifest_sha
    seals[0]["completion_sha256"] = _write_json(complete_path, complete)

    with pytest.raises(EligibilityV2Error, match="not canonical"):
        sealed_manifest_chain_identity_v2(root, seals, exact_tree=True)


def test_payload_tree_rejects_unbound_extra_file_and_symlink(tmp_path: Path) -> None:
    root, seals = _payload_tree_fixture(tmp_path)
    (root / "extra.bin").write_bytes(b"extra")
    with pytest.raises(EligibilityV2Error, match="unbound"):
        sealed_manifest_chain_identity_v2(root, seals, exact_tree=True)

    (root / "extra.bin").unlink()
    (root / "link.bin").symlink_to(root / "payload.bin")
    with pytest.raises(EligibilityV2Error, match="symlink"):
        sealed_manifest_chain_identity_v2(root, seals, exact_tree=True)
