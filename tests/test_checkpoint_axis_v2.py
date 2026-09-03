from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

import benchmark.checkpoint_axis as checkpoint_axis_module
from benchmark.checkpoint_axis import (
    ARTIFACT_CONTRACT,
    PARITY_CONTRACT,
    CheckpointAxis,
    artifact_tree_ledger,
    canonical_output_dir,
    finalize_and_publish,
    frozen_reference_seal,
    load_and_verify_checkpoint,
    load_axis_config,
    prepare_staging,
    resolve_axis,
    validate_checkpoint_payload,
    verify_parity_receipt,
    verify_published_artifact,
)
import export_fixed_split_source_axis_v2 as clean_exporter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_axis(tmp_path: Path, *, role: str = "best_miou") -> CheckpointAxis:
    project = tmp_path / "project"
    configs = project / "configs"
    dataset = project / "datasets" / "D"
    results = project / "results"
    configs.mkdir(parents=True)
    dataset.mkdir(parents=True)
    results.mkdir()
    config = configs / "axis.yaml"
    training = configs / "training.yaml"
    split = dataset / "test.txt"
    checkpoint = project / "checkpoint.bin"
    config.write_bytes(b"axis\n")
    training.write_bytes(b"training\n")
    split.write_bytes(b"sample\n")
    checkpoint.write_bytes(b"checkpoint\n")
    return CheckpointAxis(
        role=role,  # type: ignore[arg-type]
        expected_selection_metric="miou" if role == "best_miou" else "pd",
        checkpoint_path=checkpoint,
        checkpoint_sha256=_sha256(checkpoint),
        development_only=True,
        dataset="D",
        expected_selection_rule="rule",
        expected_selection_value=0.5,
        expected_epoch=1,
        recorded_test_metrics=MappingProxyType(
            {
                "duration_seconds": 1.0,
                "epoch": 1,
                "fa_per_pixel_x1e6": 0.0,
                "images": 1,
                "miou": 0.0,
                "pd": 0.0,
            }
        ),
        checkpoint_selection_split="test",
        test_selected=True,
        dataset_root=dataset,
        split_path=split,
        split_sha256=_sha256(split),
        expected_images=1,
        image_size=8,
        artifact_kind="clean",
        output_dir=results / "candidate" / "D",
        protocol_id="axis-v1",
        config_path=config,
        config_sha256=_sha256(config),
        training_protocol_path=training,
        training_protocol_sha256=_sha256(training),
        training_protocol_id="training-v1",
        threshold_transform="sigmoid",
        threshold_rule="strict_greater_than",
        threshold_value=0.5,
        parity_receipt_path=None,
        parity_receipt_sha256=None,
    )


def test_real_config_freezes_both_roles_and_role_first_outputs() -> None:
    config = load_axis_config()

    assert config["scope"]["development_only"] is True
    assert config["scope"]["checkpoint_selection_split"] == "test"
    assert config["threshold"] == {
        "transform": "sigmoid",
        "rule": "strict_greater_than",
        "value": 0.5,
    }
    assert config["parity_gate"]["status"] == "pending_runtime_receipt"
    assert set(config["parity_anchor"]["checkpoints"]) == {
        "IRSTD-1K",
        "NUAA-SIRST",
        "NUDT-SIRST",
    }
    assert set(config["development_axis"]["checkpoints"]) == {
        "IRSTD-1K",
        "NUAA-SIRST",
        "NUDT-SIRST",
    }

    axis = resolve_axis(
        config,
        dataset="NUAA-SIRST",
        role="best_pd",
        artifact_kind="clean",
        verify_files=False,
        verify_parity_gate=False,
    )
    assert axis.output_dir == (
        PROJECT_ROOT
        / "results"
        / "baseline_checkpoint_axis_v2"
        / "best_pd"
        / "NUAA-SIRST"
    )
    assert axis.expected_epoch == 504
    assert axis.expected_selection_metric == "pd"
    assert axis.expected_selection_rule == "maximize_pd_then_minimize_fa_then_miou"
    assert axis.checkpoint_sha256 == (
        "4882f405feebf66e34e6832aadaea19fbad38459f2afedfe3fc261dd75f02286"
    )
    assert dict(axis.recorded_test_metrics)["pd"] == 0.9809885931558935


def test_best_miou_requires_exact_configured_parity_destination() -> None:
    config = load_axis_config()
    expected = (
        PROJECT_ROOT
        / "results"
        / "checkpoint_axis_v2_parity_candidates"
        / "best_miou"
        / "clean"
        / "IRSTD-1K"
    )

    with pytest.raises(ValueError, match="explicit --output-dir"):
        canonical_output_dir(
            config,
            artifact_kind="clean",
            role="best_miou",
            dataset="IRSTD-1K",
        )
    assert canonical_output_dir(
        config,
        artifact_kind="clean",
        role="best_miou",
        dataset="IRSTD-1K",
        output_override=expected,
    ) == expected
    with pytest.raises(ValueError, match="parity destination drift"):
        canonical_output_dir(
            config,
            artifact_kind="clean",
            role="best_miou",
            dataset="IRSTD-1K",
            output_override=PROJECT_ROOT / "results" / "somewhere-else",
        )
    with pytest.raises(ValueError, match="formal checkpoint-axis root"):
        canonical_output_dir(
            config,
            artifact_kind="clean",
            role="best_miou",
            dataset="IRSTD-1K",
            output_override=(
                PROJECT_ROOT
                / "results"
                / "baseline_checkpoint_axis_v2"
                / "best_miou"
                / "IRSTD-1K"
            ),
        )


def test_best_pd_override_cannot_pollute_formal_roots() -> None:
    config = load_axis_config()
    with pytest.raises(ValueError, match="best_pd output override"):
        canonical_output_dir(
            config,
            artifact_kind="clean",
            role="best_pd",
            dataset="IRSTD-1K",
            output_override=(
                PROJECT_ROOT
                / "results"
                / "baseline_checkpoint_axis_v2"
                / "smoke"
                / "IRSTD-1K"
            ),
        )


def test_best_pd_default_resolver_is_blocked_without_real_global_receipt() -> None:
    config = load_axis_config()
    receipt = (
        PROJECT_ROOT
        / "results"
        / "checkpoint_axis_v2_parity"
        / "best_miou"
        / "PARITY_RECEIPT.json"
    )
    if receipt.exists():
        pytest.skip("a real parity receipt has since been published")
    with pytest.raises(FileNotFoundError):
        resolve_axis(
            config,
            dataset="IRSTD-1K",
            role="best_pd",
            artifact_kind="clean",
            verify_files=False,
        )


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_real_checkpoint_bytes_and_all_frozen_metadata_are_exact() -> None:
    config = load_axis_config()
    output = (
        PROJECT_ROOT
        / "results"
        / "checkpoint_axis_v2_parity_candidates"
        / "best_miou"
        / "clean"
        / "IRSTD-1K"
    )
    axis = resolve_axis(
        config,
        dataset="IRSTD-1K",
        role="best_miou",
        artifact_kind="clean",
        output_override=output,
    )
    payload = load_and_verify_checkpoint(axis)

    validate_checkpoint_payload(axis, payload)
    assert payload["epoch"] == 535
    assert payload["selection_metric"] == "miou"
    assert payload["selection_rule"] == "maximize_miou_then_pd_then_minimize_fa"
    assert dict(payload["test_metrics"]) == dict(axis.recorded_test_metrics)
    changed = dict(payload)
    changed["epoch"] = 536
    with pytest.raises(ValueError, match="checkpoint epoch drift"):
        validate_checkpoint_payload(axis, changed)


def test_recursive_manifest_complete_and_noreplace_publication(tmp_path: Path) -> None:
    axis = _synthetic_axis(tmp_path)
    staging = prepare_staging(axis)
    (staging / "nested").mkdir()
    (staging / "nested" / "payload.bin").write_bytes(b"payload")
    (staging / "metrics.json").write_text("{}\n", encoding="utf-8")

    result = finalize_and_publish(
        staging=staging,
        final=axis.output_dir,
        axis=axis,
        required_payloads=("nested/payload.bin", "metrics.json"),
    )

    assert result["complete"]["complete"] is True
    assert result["manifest"]["artifact_contract"] == ARTIFACT_CONTRACT
    assert [record["path"] for record in result["payload_tree"]["files"]] == [
        "metrics.json",
        "nested/payload.bin",
    ]
    assert (axis.output_dir / "artifact_manifest.json").is_file()
    assert (axis.output_dir / "COMPLETE.json").is_file()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        prepare_staging(axis)


def test_atomic_publish_race_never_replaces_destination(tmp_path: Path) -> None:
    axis = _synthetic_axis(tmp_path)
    staging = prepare_staging(axis)
    (staging / "payload.bin").write_bytes(b"candidate")
    axis.output_dir.mkdir()
    (axis.output_dir / "owner.bin").write_bytes(b"preexisting")

    with pytest.raises(FileExistsError, match="destination exists"):
        finalize_and_publish(
            staging=staging,
            final=axis.output_dir,
            axis=axis,
            required_payloads=("payload.bin",),
        )

    assert (axis.output_dir / "owner.bin").read_bytes() == b"preexisting"
    assert not (axis.output_dir / "payload.bin").exists()
    assert staging.is_dir()


def test_recursive_verifier_rejects_tampering_and_symlinks(tmp_path: Path) -> None:
    axis = _synthetic_axis(tmp_path)
    staging = prepare_staging(axis)
    payload = staging / "payload.bin"
    payload.write_bytes(b"first")
    finalize_and_publish(
        staging=staging,
        final=axis.output_dir,
        axis=axis,
        required_payloads=("payload.bin",),
    )
    payload = axis.output_dir / "payload.bin"
    payload.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="recursive payload tree drift"):
        verify_published_artifact(axis.output_dir)

    second = replace(axis, output_dir=axis.output_dir.parent / "symlink-case")
    staging = prepare_staging(second)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (staging / "linked.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe file"):
        artifact_tree_ledger(staging)


def test_global_parity_receipt_revalidates_envelope_and_live_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_axis = _synthetic_axis(tmp_path)
    project = base_axis.config_path.parent.parent
    config = deepcopy(load_axis_config())
    config["_runtime"] = {
        "config_path": str(base_axis.config_path),
        "config_sha256": base_axis.config_sha256,
        "project_root": str(project),
        "training_protocol_path": str(base_axis.training_protocol_path),
        "training_protocol_sha256": base_axis.training_protocol_sha256,
    }
    config["parity_gate"]["receipt_path"] = (
        "results/checkpoint_axis_v2_parity/best_miou/PARITY_RECEIPT.json"
    )
    config["parity_gate"]["candidate_roots"] = {
        kind: f"results/candidates/best_miou/{kind}"
        for kind in ("clean", "source", "adabn")
    }
    for dataset in ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"):
        config["parity_anchor"]["checkpoints"][dataset][
            "sha256"
        ] = base_axis.checkpoint_sha256
        config["parity_anchor"]["checkpoints"][dataset][
            "epoch"
        ] = base_axis.expected_epoch
        config["datasets"][dataset]["test_split_sha256"] = base_axis.split_sha256
    candidate_roots = {
        kind: project / "results" / "candidates" / "best_miou" / kind
        for kind in ("clean", "source", "adabn")
    }
    sections: dict[str, object] = {}
    for kind, candidate_root in candidate_roots.items():
        datasets: dict[str, object] = {}
        for dataset in ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"):
            axis = replace(
                base_axis,
                artifact_kind=kind,  # type: ignore[arg-type]
                dataset=dataset,
                output_dir=candidate_root / dataset,
            )
            staging = prepare_staging(axis)
            (staging / "payload.txt").write_text(
                f"{kind}/{dataset}\n", encoding="utf-8"
            )
            verified = finalize_and_publish(
                staging=staging,
                final=axis.output_dir,
                axis=axis,
                required_payloads=("payload.txt",),
            )
            datasets[dataset] = {
                "candidate_seals": {
                    "artifact_manifest.json": verified["manifest_sha256"],
                    "COMPLETE.json": verified["complete_sha256"],
                    "payload_tree_sha256": verified["payload_tree"]["sha256"],
                    "payload_file_count": verified["payload_tree"]["file_count"],
                }
            }
        section: dict[str, object] = {"passed": True, "datasets": datasets}
        if kind in {"source", "adabn"}:
            section["totals"] = {"conditions": 39}
        sections[kind] = section

    gate_root = (
        project / "results" / "checkpoint_axis_v2_parity" / "best_miou"
    )
    gate_root.mkdir(parents=True)
    producer_path = project / "scripts" / "verify_checkpoint_axis_v2_parity.py"
    producer_path.parent.mkdir(parents=True)
    producer_path.write_text("# synthetic parity producer\n", encoding="utf-8")
    producer_sha256 = _sha256(producer_path)
    candidate_observation = {
        "path": "results/synthetic-observation",
        "capture_timing": "post_run_pre_patch",
        "full_runtime_dependency_sealed": False,
        "implementation_identity_asserted": False,
        "adabn_v2_orchestrator_runtime_bound": False,
    }
    guard_only_patch_audit = {
        "status": "passed",
        "test_fixture": True,
    }
    implementation_seal = {
        "files": [
            {
                "path": "scripts/verify_checkpoint_axis_v2_parity.py",
                "sha256": producer_sha256,
                "bytes": producer_path.stat().st_size,
            }
        ]
    }
    monkeypatch.setattr(
        checkpoint_axis_module,
        "verify_candidate_producer_observation",
        lambda *, project_root: candidate_observation,
    )
    monkeypatch.setattr(
        checkpoint_axis_module,
        "checkpoint_axis_guard_only_patch_audit",
        lambda *, project_root: guard_only_patch_audit,
    )
    monkeypatch.setattr(
        checkpoint_axis_module,
        "verify_implementation_dependency_seal",
        lambda seal, project_root: dict(seal),
    )
    receipt = {
        "schema_version": 1,
        "receipt_type": "checkpoint_axis_v2_best_miou_exact_parity",
        "artifact_contract": PARITY_CONTRACT,
        "status": "passed",
        "passed": True,
        "checkpoint_role": "best_miou",
        "axis_config_sha256": base_axis.config_sha256,
        "numeric_tolerance_used": False,
        "producer": {
            "path": "scripts/verify_checkpoint_axis_v2_parity.py",
            "sha256": producer_sha256,
        },
        "candidate_producer_provenance": {
            "capture_timing": "post_run_pre_patch",
            "full_runtime_dependency_sealed": False,
            "implementation_identity_asserted": False,
            "adabn_v2_orchestrator_runtime_bound": False,
            "observation": candidate_observation,
        },
        "parity_assertion": {
            "subject": "legacy_best_miou_candidate_scientific_payloads",
            "scientific_payload_bit_exact": True,
            "implementation_identity_asserted": False,
            "applies_to_best_pd_runtime": False,
        },
        "receipt_verifier_implementation": {
            "purpose": "verify_and_publish_exact_output_parity_receipt",
            "seal": implementation_seal,
        },
        "authorized_best_pd_live_implementation": {
            "purpose": "authorize_frozen_best_pd_development_axis_without_retuning",
            "direct_parity_status": "not_run",
            "guard_only_patch_continuity": True,
            "guard_only_patch_audit": guard_only_patch_audit,
            "seal": implementation_seal,
        },
        "comparison_contract": {
            "ordered_ids_exact": True,
            "float32_probability_arrays_bit_exact": True,
            "probability_file_sha256_exact": True,
            "binary_png_bytes_exact": True,
            "integer_sufficient_statistics_exact": True,
            "official_metrics_exact": True,
            "unified_metrics_exact": True,
        },
        "frozen_reference_seal": frozen_reference_seal(config),
        "candidate_roots": {
            kind: str(path) for kind, path in candidate_roots.items()
        },
        **sections,
    }
    receipt_path = gate_root / "PARITY_RECEIPT.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = artifact_tree_ledger(
        gate_root, exclude=("artifact_manifest.json", "COMPLETE.json")
    )
    manifest = {
        "schema_version": 1,
        "artifact_contract": PARITY_CONTRACT,
        "receipt_type": "checkpoint_axis_v2_best_miou_exact_parity",
        "axis_config_sha256": base_axis.config_sha256,
        "verifier_sha256": producer_sha256,
        "payload_tree": ledger,
    }
    manifest_path = gate_root / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete = {
        "schema_version": 1,
        "complete": True,
        "artifact_contract": PARITY_CONTRACT,
        "axis_config_sha256": base_axis.config_sha256,
        "manifest_sha256": _sha256(manifest_path),
        "parity_receipt_sha256": _sha256(receipt_path),
        "payload_tree_sha256": ledger["sha256"],
        "payload_file_count": ledger["file_count"],
    }
    (gate_root / "COMPLETE.json").write_text(
        json.dumps(complete, sort_keys=True) + "\n", encoding="utf-8"
    )

    verified_receipt = verify_parity_receipt(config=config)
    assert verified_receipt["passed"] is True
    assert verified_receipt["_receipt_sha256"] == _sha256(receipt_path)

    tampered = candidate_roots["clean"] / "IRSTD-1K" / "payload.txt"
    tampered.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="recursive payload tree"):
        verify_parity_receipt(config=config)


def test_checkpoint_metric_comparison_uses_exact_equality(tmp_path: Path) -> None:
    axis = _synthetic_axis(tmp_path)
    measured = {
        "miou": 0.0,
        "pd": 0.0,
        "fa_per_pixel_x1e6": 0.0,
    }
    exact = clean_exporter._checkpoint_metric_comparison(
        measured, axis, complete_split=True
    )
    assert exact["passed"] is True
    measured["pd"] = float(np.nextafter(0.0, 1.0))
    drift = clean_exporter._checkpoint_metric_comparison(
        measured, axis, complete_split=True
    )
    assert drift["passed"] is False
    assert drift["metrics"]["pd"]["comparison"] == "exact_float_equality"


class _FakeNSFPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, 1, kernel_size=1)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self, image, warm_flag):
        return ([image] if warm_flag else []), self.head(image)


def test_clean_exporter_writes_every_mask_probability_and_sealed_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    axis = _synthetic_axis(tmp_path)
    images = axis.dataset_root / "images"
    masks = axis.dataset_root / "masks"
    images.mkdir()
    masks.mkdir()
    identifiers = ("sample", "second")
    axis.split_path.write_text("sample\nsecond\n", encoding="utf-8")
    for index, identifier in enumerate(identifiers):
        image = np.full((10, 12, 3), 70 + index * 20, dtype=np.uint8)
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[4:6, 5:7] = 255
        Image.fromarray(image).save(images / f"{identifier}.png")
        Image.fromarray(mask).save(masks / f"{identifier}.png")
    axis = replace(
        axis,
        split_sha256=_sha256(axis.split_path),
        expected_images=2,
        recorded_test_metrics=MappingProxyType(
            {
                "duration_seconds": 1.0,
                "epoch": 1,
                "fa_per_pixel_x1e6": 0.0,
                "images": 2,
                "miou": 0.0,
                "pd": 0.0,
            }
        ),
    )
    payload = {
        "schema_version": 1,
        "architecture": "fake",
        "split_manifest": {
            "test_split_sha256": axis.split_sha256,
            "test_count": 2,
        },
    }
    monkeypatch.setattr(clean_exporter, "_load_run_contract", lambda args: ({}, axis))
    monkeypatch.setattr(
        clean_exporter, "load_and_verify_checkpoint", lambda value: payload
    )
    monkeypatch.setattr(
        clean_exporter,
        "_load_verified_model",
        lambda value: (_FakeNSFPN(), "state_dict"),
    )
    monkeypatch.setattr(
        clean_exporter,
        "_runtime_provenance",
        lambda value: {"test_fixture": True},
    )
    args = SimpleNamespace(
        dataset="D",
        checkpoint_role="best_miou",
        axis_config=axis.config_path,
        output_dir=axis.output_dir,
        device="cpu",
        num_workers=0,
        max_images=1,
        visualization_count=0,
    )

    result = clean_exporter.run_export(args)

    assert result["artifact_counts"] == {
        "prediction_masks": 1,
        "probability_maps": 1,
        "per_image_records": 1,
        "visualizations": 0,
    }
    assert result["complete_fixed_test_split"] is False
    probability = np.load(
        axis.output_dir / "probability_maps_256" / "sample.npy",
        allow_pickle=False,
    )
    assert np.array_equal(probability, np.full((8, 8), 0.5, dtype=np.float32))
    mask = np.asarray(Image.open(axis.output_dir / "prediction_masks_256" / "sample.png"))
    assert not mask.any(), "strict probability > 0.5 must reject exactly 0.5"
    verified = verify_published_artifact(axis.output_dir, expected_axis=axis)
    assert verified["complete"]["complete"] is True
    assert len((axis.output_dir / "per_image.jsonl").read_text().splitlines()) == 1


def test_cli_defaults_to_best_pd() -> None:
    args = clean_exporter.build_parser().parse_args(["--dataset", "IRSTD-1K"])
    assert args.checkpoint_role == "best_pd"
