from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.archive_binary_tent_ss_stage1_negative_v2 import (
    DATASETS,
    DEFAULT_DESTINATION,
    DEFAULT_SOURCE,
    FROZEN_CONFIG_PATHS,
    NegativeArchiveError,
    build_archive_plan,
    create_or_verify_archive,
    main,
    sha256_file,
    verify_archive,
)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _candidate(index: int) -> dict[str, Any]:
    rates = (
        (0.00001, "1e-5"),
        (0.00003, "3e-5"),
        (0.0001, "1e-4"),
        (0.0003, "3e-4"),
        (0.001, "1e-3"),
    )
    optimizer = "Adam" if index < 5 else "SGD"
    rate, decimal = rates[index % 5]
    return {
        "learning_rate": rate,
        "learning_rate_decimal": decimal,
        "optimizer": optimizer,
    }


def _cell_record(candidate: Mapping[str, Any], cell: int) -> dict[str, Any]:
    return {
        "candidate": dict(candidate),
        "corruption": f"condition-{cell}",
        "dataset": DATASETS[cell % len(DATASETS)],
        "hard_gates": {
            "exact_reset": True,
            "finite_values": True,
            "label_firewall": True,
        },
        "image_count": 64,
        "method_label_accesses": 0,
        "severity": cell,
        "stage": 1,
        "test_image_opens": 0,
        "test_label_opens": 0,
    }


def _artifact_manifest(
    directory: Path,
    names: tuple[str, ...],
    **fields: Any,
) -> dict[str, Any]:
    return {
        **fields,
        "files": {
            name: {
                "bytes": (directory / name).stat().st_size,
                "sha256": sha256_file(directory / name),
            }
            for name in names
        },
        "scope": {"paper_result": False, "source_train_derived": True},
    }


def _complete_directory(
    directory: Path,
    manifest: Mapping[str, Any],
    **fields: Any,
) -> None:
    _write_json(directory / "artifact_manifest.json", manifest)
    _write_json(
        directory / "COMPLETE.json",
        {
            **fields,
            "artifact_manifest_sha256": sha256_file(
                directory / "artifact_manifest.json"
            ),
            "complete": True,
            "scope": {"paper_result": False, "source_train_derived": True},
        },
    )


def _make_project(project: Path) -> Path:
    source = project / DEFAULT_SOURCE
    stage1 = source / "stage1"
    aggregate_records: list[dict[str, Any]] = []
    shard_index: list[dict[str, Any]] = []
    ranking: list[dict[str, Any]] = []

    for index in range(10):
        candidate = _candidate(index)
        slug = f"{candidate['optimizer']}-{candidate['learning_rate_decimal']}"
        shard = stage1 / "shards" / slug
        records = [_cell_record(candidate, cell) for cell in range(39)]
        aggregate_records.extend(records)
        _write_jsonl(shard / "records.jsonl", records)
        _write_jsonl(shard / "lr_strength_diagnostics.jsonl", [{}] * 39)
        _write_json(shard / "provenance.json", {"candidate": candidate})
        _write_json(shard / "run_summary.json", {"record_count": 39})
        _write_json(shard / "runtime_seal.json", {"sealed": True})
        process_id = f"process-{index}"
        manifest = _artifact_manifest(
            shard,
            (
                "records.jsonl",
                "lr_strength_diagnostics.jsonl",
                "provenance.json",
                "run_summary.json",
                "runtime_seal.json",
            ),
            candidate=candidate,
            episode_count=2496,
            record_count=39,
            stage=1,
        )
        _complete_directory(
            shard,
            manifest,
            candidate=candidate,
            episode_count=2496,
            process_id=process_id,
            record_count=39,
            stage=1,
        )
        shard_index.append(
            {
                "artifact_manifest_sha256": sha256_file(
                    shard / "artifact_manifest.json"
                ),
                "candidate": candidate,
                "process_id": process_id,
            }
        )
        ranking.append(
            {
                "candidate": candidate,
                "primary_mean_over_runs_macro_global_iou_delta": {
                    "denominator": 1 if index < 5 else 1000,
                    "exact": "0/1" if index < 5 else "-1/1000",
                    "numerator": 0 if index < 5 else -1,
                    "value": 0.0 if index < 5 else -0.001,
                },
            }
        )

    aggregate = stage1 / "aggregate"
    _write_jsonl(aggregate / "stage1_records.jsonl", aggregate_records)
    _write_jsonl(aggregate / "lr_strength_diagnostics.jsonl", [{}] * 390)
    _write_json(aggregate / "provenance.json", {"source_train_derived": True})
    _write_json(aggregate / "runtime_seal.json", {"sealed": True})
    _write_json(aggregate / "shard_index.json", {"shards": shard_index})
    _write_json(
        aggregate / "stage1_ss_top3_receipt.json",
        {
            "ranking": ranking,
            "receipt_type": "stage1_ss_top3",
            "schema_version": 2,
            "scope": {"paper_result": False, "source_train_derived": True},
            "top3": [_candidate(index) for index in range(3)],
        },
    )
    aggregate_manifest = _artifact_manifest(
        aggregate,
        (
            "stage1_records.jsonl",
            "lr_strength_diagnostics.jsonl",
            "provenance.json",
            "runtime_seal.json",
            "shard_index.json",
            "stage1_ss_top3_receipt.json",
        ),
        candidate_count=10,
        episode_count=24960,
        record_count=390,
        stage=1,
    )
    _complete_directory(
        aggregate,
        aggregate_manifest,
        episode_count=24960,
        global_runtime_seal_sha256="a" * 64,
        record_count=390,
        stage=1,
    )

    for relative in FROZEN_CONFIG_PATHS:
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("paper_result: false\n", encoding="utf-8")
    pilot_root = project / "configs/tta_train_side_pilot_v2"
    _write_json(pilot_root / "manifest.json", {"paper_result": False})
    for dataset in DATASETS:
        (pilot_root / f"{dataset}.txt").write_text("image-001\n", encoding="utf-8")
        split_root = project / "datasets" / dataset / "img_idx"
        split_root.mkdir(parents=True, exist_ok=True)
        (split_root / f"train_{dataset}.txt").write_text(
            "image-001\n", encoding="utf-8"
        )
        (split_root / f"test_{dataset}.txt").write_text(
            "image-002\n", encoding="utf-8"
        )
        cache = project / "results/binary_tent/ss_calibration_cache_v2" / dataset
        _write_json(cache / "manifest.json", {"dataset": dataset})
        _write_json(cache / "method_input_manifest.json", {"dataset": dataset})
        _write_json(cache / "COMPLETE.json", {"complete": True})
        _write_json(
            project
            / "results/corruption_pilot_fixed_split/round_02"
            / dataset
            / "artifact_manifest.json",
            {"dataset": dataset},
        )
    return source


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_dry_run_validates_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project = tmp_path / "project"
    source = _make_project(project)
    before = _tree_hashes(source)

    exit_code = main(["--project-root", str(project), "--dry-run"])

    assert exit_code == 0
    assert not (project / DEFAULT_DESTINATION).exists()
    assert _tree_hashes(source) == before
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry_run_validated_no_writes"
    assert payload["stage1_cells"] == 390
    assert payload["stage1_episodes"] == 24960
    assert payload["paper_result"] is False
    assert payload["scientific_gate_v3_replay"] == "absent_not_provided"


def test_create_is_copy_only_content_addressed_and_idempotent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = _make_project(project)
    before = _tree_hashes(source)
    plan = build_archive_plan(project_root=project)

    first = create_or_verify_archive(plan)
    second = create_or_verify_archive(plan)

    destination = project / DEFAULT_DESTINATION
    assert first["status"] == "created_and_verified"
    assert second["status"] == "verified_existing"
    assert _tree_hashes(source) == before
    assert verify_archive(destination)["paper_result"] is False
    negative = json.loads((destination / "NEGATIVE_RESULT.json").read_text())
    assert negative["archive_scope"]["paper_result"] is False
    assert negative["eligible_candidates"] is None
    assert negative["scientific_gate_evidence"]["scientific_gate_v3_replay"] == {
        "receipt": None,
        "receipt_type": None,
        "status": "absent_not_provided",
    }
    aborted = json.loads(
        (
            destination
            / "stage2_aborted_partial/ABORTED_INCOMPLETE.json"
        ).read_text()
    )
    assert aborted["paper_result"] is False
    assert aborted["formal_output_audit"]["formal_shards_published"] is False
    assert aborted["formal_output_audit"]["formal_aggregate_published"] is False
    assert aborted["partial_progress"]["completed_cells_per_slot"] is None
    assert aborted["partial_progress"]["claim_intentionally_omitted"] is True
    assert "9/39" not in json.dumps(aborted)


def test_tampered_existing_archive_fails_closed_without_overwrite(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _make_project(project)
    plan = build_archive_plan(project_root=project)
    create_or_verify_archive(plan)
    destination = project / DEFAULT_DESTINATION
    copied = destination / "stage1/aggregate/stage1_records.jsonl"
    copied.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(NegativeArchiveError, match="checksum mismatch"):
        create_or_verify_archive(plan)

    assert copied.read_text(encoding="utf-8") == "tampered\n"


def test_formal_stage2_path_blocks_before_destination_creation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = _make_project(project)
    formal = source / "stage2/shards/slot-1"
    formal.mkdir(parents=True)
    (formal / "records.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(NegativeArchiveError, match="formal Stage-2/final paths exist"):
        build_archive_plan(project_root=project)

    assert not (project / DEFAULT_DESTINATION).exists()


def test_unlisted_archive_file_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _make_project(project)
    plan = build_archive_plan(project_root=project)
    create_or_verify_archive(plan)
    destination = project / DEFAULT_DESTINATION
    (destination / "unlisted.txt").write_text("unsafe\n", encoding="utf-8")

    with pytest.raises(NegativeArchiveError, match="membership mismatch"):
        verify_archive(destination)


def test_optional_v3_replay_is_validated_and_archived(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _make_project(project)
    receipt = project / "results/scientific_gate_v3_replay.json"
    _write_json(
        receipt,
        {
            "eligible_candidates": [],
            "formal_fully_frozen_gate": False,
            "paper_result": False,
            "protocol_status": "passed",
            "receipt_type": "stage1_ss_scientific_selection",
            "replay_scope": {
                "authorizes_stage2": False,
                "paper_result": False,
                "uses_test_images": False,
                "uses_test_labels": False,
            },
            "retrospective_negative_replay": True,
            "route_decision": "stop_before_stage2",
            "schema_version": 3,
            "scientific_status": "failed",
            "scientific_gate": {
                "authorizes_stage2": False,
                "mode": "retrospective_negative_replay",
                "threshold_resolution_status": "unresolved_retrospective_only",
                "unresolved_thresholds": ["clean_safety.margin"],
            },
            "selector_protocol_id": (
                "cr-sitta-binary-tent-ss-calibration-selector-v3"
            ),
            "selected_for_stage2": [],
            "source_evidence_bindings": {
                name: {"bytes": 1, "path": f"evidence/{name}", "sha256": "a" * 64}
                for name in (
                    "v3_config",
                    "selector_v3",
                    "v2_aggregate_manifest",
                    "v2_aggregate_complete",
                    "v2_stage1_records",
                    "v2_strength_diagnostics",
                    "v2_stage1_top3_receipt",
                )
            },
            "stage2_allowed": False,
            "stage3_allowed": False,
            "unresolved_gate_thresholds": True,
        },
    )
    plan = build_archive_plan(
        project_root=project, scientific_gate_receipt=receipt
    )

    create_or_verify_archive(plan)

    destination = project / DEFAULT_DESTINATION
    negative = json.loads((destination / "NEGATIVE_RESULT.json").read_text())
    assert negative["eligible_candidates"] == []
    assert (
        negative["scientific_gate_evidence"]["scientific_gate_v3_replay"]["status"]
        == "provided_and_verified"
    )
    assert (
        destination
        / "scientific_gate_v3_replay/stage1_ss_scientific_selection_receipt.json"
    ).is_file()
    checksums = (destination / "SHA256SUMS").read_text(encoding="utf-8")
    assert (
        "scientific_gate_v3_replay/stage1_ss_scientific_selection_receipt.json"
        in checksums
    )
    assert "scientific_gate_v3_replay/VERIFICATION.json" in checksums


def test_invalid_v3_replay_fails_closed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    _make_project(project)
    receipt = project / "results/invalid_scientific_gate_v3_replay.json"
    _write_json(
        receipt,
        {
            "eligible_candidates": [],
            "paper_result": False,
            "protocol_status": "passed",
            "receipt_type": "stage1_scientific_selection_v3_replay",
            "route_decision": "stop_before_stage2",
            "scientific_status": "passed",
            "selected_for_stage2": [],
            "stage2_allowed": False,
        },
    )

    with pytest.raises(NegativeArchiveError, match="scientific_status"):
        build_archive_plan(
            project_root=project, scientific_gate_receipt=receipt
        )

    assert not (project / DEFAULT_DESTINATION).exists()


def test_explicit_stage2_log_is_quarantined_without_progress_inference(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _make_project(project)
    raw_log = project / "logs/stage2-interrupted.stderr"
    raw_log.parent.mkdir(parents=True)
    raw_log.write_text("unstructured worker output; 9/39\n", encoding="utf-8")
    plan = build_archive_plan(project_root=project, stage2_logs=(raw_log,))

    create_or_verify_archive(plan)

    destination = project / DEFAULT_DESTINATION
    copied = (
        destination
        / "stage2_aborted_partial/raw_logs_only/logs/stage2-interrupted.stderr"
    )
    assert copied.read_bytes() == raw_log.read_bytes()
    aborted = json.loads(
        (
            destination
            / "stage2_aborted_partial/ABORTED_INCOMPLETE.json"
        ).read_text()
    )
    assert aborted["raw_log_evidence"]["status"] == "archived_uninterpreted_raw_logs"
    assert aborted["raw_log_evidence"]["file_count"] == 1
    assert aborted["partial_progress"]["completed_cells_per_slot"] is None
