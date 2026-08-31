from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = PROJECT_ROOT / "metrics"
if str(METRICS_ROOT) not in sys.path:
    sys.path.insert(0, str(METRICS_ROOT))

from result_eligibility import (  # noqa: E402
    EligibilityError,
    build_registry,
    full_tree_identity,
    materialize_registry,
    registry_bytes,
    sealed_manifest_chain_identity,
    validate_materialized_registry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return _sha256(path)


def _sealed_fixture(project: Path) -> tuple[Path, list[dict[str, object]], str]:
    artifact = project / "results" / "artifact"
    child = artifact / "child"
    child.mkdir(parents=True)
    payload = child / "payload.bin"
    payload.write_bytes(b"payload-v1")
    child_manifest = child / "artifact_manifest.json"
    child_manifest_sha = _write_json(
        child_manifest,
        {
            "files": {
                "payload.bin": {
                    "bytes": payload.stat().st_size,
                    "sha256": _sha256(payload),
                }
            }
        },
    )
    child_complete = child / "COMPLETE.json"
    child_complete_sha = _write_json(
        child_complete,
        {"complete": True, "artifact_manifest_sha256": child_manifest_sha},
    )

    summary = artifact / "summary.json"
    summary.write_text('{"summary":true}\n', encoding="utf-8")
    manifest = artifact / "artifact_manifest.json"
    manifest_sha = _write_json(
        manifest,
        {
            "files": {
                "summary.json": {
                    "bytes": summary.stat().st_size,
                    "sha256": _sha256(summary),
                }
            },
            "children": {
                "child": {
                    "artifact_manifest": "child/artifact_manifest.json",
                    "artifact_manifest_sha256": child_manifest_sha,
                    "completion": "child/COMPLETE.json",
                    "completion_sha256": child_complete_sha,
                }
            },
        },
    )
    complete = artifact / "COMPLETE.json"
    complete_sha = _write_json(
        complete,
        {"complete": True, "artifact_manifest_sha256": manifest_sha},
    )
    seals: list[dict[str, object]] = [
        {
            "manifest": "artifact_manifest.json",
            "manifest_sha256": manifest_sha,
            "completion": "COMPLETE.json",
            "completion_sha256": complete_sha,
        }
    ]
    identity = sealed_manifest_chain_identity(artifact, seals, exact_tree=True)
    return artifact, seals, identity.sha256


def _artifact_entry(
    project: Path,
    *,
    sealed: bool = False,
    artifact_complete: bool = True,
    formal_complete: bool = False,
    main_paper: bool = False,
    test_selected: bool = True,
) -> dict[str, object]:
    if sealed:
        artifact, seals, identity_sha = _sealed_fixture(project)
        identity: dict[str, object] = {
            "mode": "sealed_manifest_chain",
            "expected_sha256": identity_sha,
            "exact_tree": True,
            "seals": seals,
        }
    else:
        artifact = project / "results" / "artifact"
        artifact.mkdir(parents=True)
        (artifact / "payload.json").write_text('{"value":1}\n', encoding="utf-8")
        identity = {
            "mode": "full_tree",
            "expected_sha256": full_tree_identity(artifact).sha256,
        }
    return {
        "artifact_id": "fixture",
        "root": "results/artifact",
        "artifact_role": "test_fixture",
        "identity": identity,
        "artifact_complete": artifact_complete,
        "formal_protocol_complete": formal_complete,
        "scientific_eligibility": {
            "tier": "final_paper_benchmark" if main_paper else "development_test_selected",
            "main_paper_table": main_paper,
            "reason_codes": [] if main_paper else ["source_checkpoint_test_selected"],
        },
        "selection": {
            "source_checkpoint_role": "best_miou",
            "checkpoint_selection_split": "test" if test_selected else "train",
            "tta_calibration_split": "train",
            "test_used_for_checkpoint_selection": test_selected,
            "test_used_for_tta_selection": False,
        },
        "notes": "fixture",
    }


def _config(project: Path, artifact: dict[str, object]) -> Path:
    config = {
        "schema_version": 1,
        "registry_id": "cr-sitta-artifact-eligibility-registry-v1",
        "output": "results/artifact_eligibility_registry_v1.json",
        "policies": {
            "official_split_model": "train_test_only",
            "validation_split_created": False,
            "main_paper_requires_no_test_selection": True,
            "existing_artifacts_are_immutable": True,
        },
        "artifacts": [artifact],
    }
    path = project / "configs" / "artifact_eligibility_registry_v1.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _provenance_files(project: Path) -> tuple[Path, Path]:
    generator = project / "generator.py"
    module = project / "result_eligibility.py"
    generator.write_text("# generator fixture\n", encoding="utf-8")
    module.write_text("# module fixture\n", encoding="utf-8")
    return generator, module


def _build(project: Path, config: Path, generator: Path, module: Path) -> dict[str, object]:
    return build_registry(
        config,
        project_root=project,
        generator_path=generator,
        eligibility_module_path=module,
    )


def test_full_tree_identity_is_deterministic_and_detects_bytes(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "b").write_bytes(b"b")
    (root / "a").write_bytes(b"a")
    first = full_tree_identity(root)
    second = full_tree_identity(root)
    assert first == second
    assert first.verification["exact_tree_verified"] is True
    (root / "a").write_bytes(b"changed")
    assert full_tree_identity(root).sha256 != first.sha256


def test_full_tree_identity_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"target")
    (root / "link").symlink_to(target)
    with pytest.raises(EligibilityError, match="symlink"):
        full_tree_identity(root)


def test_recursive_sealed_chain_verifies_payload_size_hash_and_exact_tree(
    tmp_path: Path,
) -> None:
    artifact, seals, identity_sha = _sealed_fixture(tmp_path)

    identity = sealed_manifest_chain_identity(artifact, seals, exact_tree=True)

    assert identity.sha256 == identity_sha
    assert identity.file_count == 6
    assert identity.verification == {
        "mode": "sealed_manifest_chain",
        "exact_tree_verified": True,
        "completion_chain_verified": True,
        "root_seal_count": 1,
        "recursive_seal_count": 2,
        "manifest_count": 2,
        "completion_count": 2,
        "payload_file_count": 2,
    }


def test_recursive_sealed_chain_rejects_payload_drift(tmp_path: Path) -> None:
    artifact, seals, _ = _sealed_fixture(tmp_path)
    (artifact / "child" / "payload.bin").write_bytes(b"payload-v2")

    with pytest.raises(EligibilityError, match="payload SHA-256 drift"):
        sealed_manifest_chain_identity(artifact, seals, exact_tree=True)


def test_recursive_sealed_chain_rejects_payload_size_drift(tmp_path: Path) -> None:
    artifact, seals, _ = _sealed_fixture(tmp_path)
    child_manifest = artifact / "child" / "artifact_manifest.json"
    value = json.loads(child_manifest.read_text(encoding="utf-8"))
    value["files"]["payload.bin"]["bytes"] += 1
    child_sha = _write_json(child_manifest, value)
    child_complete = artifact / "child" / "COMPLETE.json"
    child_complete_sha = _write_json(
        child_complete,
        {"complete": True, "artifact_manifest_sha256": child_sha},
    )
    top_manifest = artifact / "artifact_manifest.json"
    top_value = json.loads(top_manifest.read_text(encoding="utf-8"))
    top_value["children"]["child"]["artifact_manifest_sha256"] = child_sha
    top_value["children"]["child"]["completion_sha256"] = child_complete_sha
    top_sha = _write_json(top_manifest, top_value)
    top_complete_sha = _write_json(
        artifact / "COMPLETE.json",
        {"complete": True, "artifact_manifest_sha256": top_sha},
    )
    seals[0]["manifest_sha256"] = top_sha
    seals[0]["completion_sha256"] = top_complete_sha

    with pytest.raises(EligibilityError, match="payload size drift"):
        sealed_manifest_chain_identity(artifact, seals, exact_tree=True)


def test_recursive_sealed_chain_rejects_false_completion(tmp_path: Path) -> None:
    artifact, seals, _ = _sealed_fixture(tmp_path)
    complete = artifact / "COMPLETE.json"
    value = json.loads(complete.read_text(encoding="utf-8"))
    value["complete"] = False
    seals[0]["completion_sha256"] = _write_json(complete, value)

    with pytest.raises(EligibilityError, match="complete=true"):
        sealed_manifest_chain_identity(artifact, seals, exact_tree=True)


def test_recursive_sealed_chain_rejects_unbound_extra_file(tmp_path: Path) -> None:
    artifact, seals, _ = _sealed_fixture(tmp_path)
    (artifact / "unbound.txt").write_text("unbound\n", encoding="utf-8")

    with pytest.raises(EligibilityError, match="unbound=.*unbound.txt"):
        sealed_manifest_chain_identity(artifact, seals, exact_tree=True)


def test_successful_registry_build_uses_single_config_snapshot(tmp_path: Path) -> None:
    artifact = _artifact_entry(tmp_path)
    config = _config(tmp_path, artifact)
    generator, module = _provenance_files(tmp_path)

    registry = _build(tmp_path, config, generator, module)

    assert registry["artifact_count"] == 1
    assert registry["protocol_config"]["sha256"] == _sha256(config)
    assert registry["protocol_config"]["single_snapshot_verified"] is True
    assert registry["checks"]["all_artifact_completeness_claims_verified"] is True
    assert registry["checks"]["scientific_eligibility_gate_verified"] is True
    assert registry_bytes(registry).endswith(b"\n")


@pytest.mark.parametrize(
    ("artifact_complete", "formal_complete", "message"),
    [
        (False, False, "artifact_complete=true"),
        (True, False, "formal_protocol_complete=true"),
    ],
)
def test_main_paper_requires_complete_and_formal(
    tmp_path: Path,
    artifact_complete: bool,
    formal_complete: bool,
    message: str,
) -> None:
    artifact = _artifact_entry(
        tmp_path,
        sealed=True,
        artifact_complete=artifact_complete,
        formal_complete=formal_complete,
        main_paper=True,
        test_selected=False,
    )
    config = _config(tmp_path, artifact)
    generator, module = _provenance_files(tmp_path)

    with pytest.raises(EligibilityError, match=message):
        _build(tmp_path, config, generator, module)


def test_formal_completion_requires_verified_completion_chain(tmp_path: Path) -> None:
    artifact = _artifact_entry(tmp_path, formal_complete=True)
    config = _config(tmp_path, artifact)
    generator, module = _provenance_files(tmp_path)

    with pytest.raises(EligibilityError, match="verified completion chain"):
        _build(tmp_path, config, generator, module)


def test_main_paper_rejects_test_selected_checkpoint(tmp_path: Path) -> None:
    artifact = _artifact_entry(
        tmp_path,
        sealed=True,
        formal_complete=True,
        main_paper=True,
        test_selected=True,
    )
    config = _config(tmp_path, artifact)
    generator, module = _provenance_files(tmp_path)

    with pytest.raises(EligibilityError, match="test-selected checkpoint"):
        _build(tmp_path, config, generator, module)


def test_materialize_is_idempotent_refuses_drift_and_atomically_replaces(
    tmp_path: Path,
) -> None:
    output = tmp_path / "results" / "registry.json"
    first = b'{"stable":true}\n'
    second = b'{"stable":"v2"}\n'
    assert materialize_registry(output, first) == "created"
    assert materialize_registry(output, first) == "already_current"
    with pytest.raises(FileExistsError, match="overwrite"):
        materialize_registry(output, second)

    assert materialize_registry(output, second, replace_existing=True) == "replaced"
    validate_materialized_registry(output, second)
    assert not list(output.parent.glob(f".{output.name}.build-*"))


def test_validate_and_materialize_reject_symlink_output(tmp_path: Path) -> None:
    output = tmp_path / "results" / "registry.json"
    output.parent.mkdir()
    target = tmp_path / "target.json"
    target.write_bytes(b"target\n")
    output.symlink_to(target)

    with pytest.raises(EligibilityError, match="unsafe|symlink"):
        materialize_registry(output, b"new\n", replace_existing=True)
    with pytest.raises(EligibilityError, match="symlink"):
        validate_materialized_registry(output, b"target\n")
    assert target.read_bytes() == b"target\n"


def test_cli_expected_payload_never_reloads_config_for_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = PROJECT_ROOT / "scripts" / "validate_result_eligibility.py"
    spec = importlib.util.spec_from_file_location("eligibility_cli_fixture", module_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    config = tmp_path / "config.yaml"
    config.write_text("output: /unsafe/second-read\n", encoding="utf-8")
    monkeypatch.setattr(cli, "build_registry", lambda *args, **kwargs: {"ok": True})

    output, payload = cli.expected_payload(config)

    assert output == PROJECT_ROOT / "results/artifact_eligibility_registry_v1.json"
    assert payload == registry_bytes({"ok": True})


def test_real_ss_cache_v2_is_train_side_nonperformance_asset() -> None:
    config_path = PROJECT_ROOT / "configs" / "artifact_eligibility_registry_v1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    artifacts = {
        artifact["artifact_id"]: artifact for artifact in config["artifacts"]
    }
    artifact = artifacts["binary_tent_ss_calibration_cache_v2"]

    assert artifact["root"] == "results/binary_tent/ss_calibration_cache_v2"
    assert artifact["artifact_role"] == (
        "train_side_protocol_calibration_asset_nonperformance"
    )
    assert artifact["artifact_complete"] is True
    assert artifact["formal_protocol_complete"] is True
    assert artifact["scientific_eligibility"] == {
        "tier": "protocol_asset_nonperformance",
        "main_paper_table": False,
        "reason_codes": [
            "nonperformance_train_side_calibration_asset",
            "development_only_method_calibration_input",
        ],
    }
    assert artifact["selection"] == {
        "source_checkpoint_role": "none",
        "checkpoint_selection_split": "none",
        "tta_calibration_split": "train",
        "test_used_for_checkpoint_selection": False,
        "test_used_for_tta_selection": False,
    }
    assert artifact["identity"]["mode"] == "sealed_manifest_chain"
    assert artifact["identity"]["exact_tree"] is True
    assert len(artifact["identity"]["seals"]) == 3
