from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmark.checkpoint_axis as checkpoint_axis
from analysis import summarize_checkpoint_axis_development_v1 as summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_cells() -> tuple[summary.MethodCell, ...]:
    cells: list[summary.MethodCell] = []
    for role_index, role in enumerate(summary.ROLES):
        for dataset_index, dataset in enumerate(summary.DATASETS):
            checkpoint_sha = _digest(f"checkpoint/{role}/{dataset}")
            epoch = 100 + 10 * role_index + dataset_index
            for condition_index, (condition_key, corruption, severity) in enumerate(
                summary.CONDITIONS
            ):
                base = 0.30 + 0.04 * role_index + 0.01 * dataset_index + 0.001 * condition_index
                source_metrics = summary.Metrics(
                    legacy_mean_iou=base,
                    legacy_pd=base + 0.10,
                    legacy_fa_per_million_pixels=10.0 + condition_index,
                    unified_global_iou=base + 0.01,
                    unified_pd=base + 0.11,
                    unified_fa_per_million_pixels=11.0 + condition_index,
                    unified_false_positives_per_image=0.1 + 0.01 * condition_index,
                )
                adabn_metrics = summary.Metrics(
                    legacy_mean_iou=base + 0.02,
                    legacy_pd=base + 0.13,
                    legacy_fa_per_million_pixels=20.0 + condition_index,
                    unified_global_iou=base + 0.03,
                    unified_pd=base + 0.14,
                    unified_fa_per_million_pixels=22.0 + condition_index,
                    unified_false_positives_per_image=0.2 + 0.01 * condition_index,
                )
                for method, metrics in (
                    ("Source", source_metrics),
                    ("AdaBN", adabn_metrics),
                ):
                    cells.append(
                        summary.MethodCell(
                            dataset=dataset,
                            checkpoint_role=role,
                            checkpoint_epoch=epoch,
                            checkpoint_sha256=checkpoint_sha,
                            method=method,
                            condition_index=condition_index,
                            condition_key=condition_key,
                            corruption=corruption,
                            severity=severity,
                            metrics=metrics,
                            metrics_path=f"/verified/{role}/{dataset}/{method}/{condition_key}.json",
                            metrics_sha256=_digest(
                                f"metrics/{role}/{dataset}/{method}/{condition_key}"
                            ),
                        )
                    )
    return tuple(cells)


def _synthetic_lineage() -> dict[str, object]:
    return {
        "axis_config": {"path": "/verified/config.yaml", "sha256": _digest("config")},
        "parity_receipt": {
            "path": "/verified/PARITY_RECEIPT.json",
            "sha256": _digest("receipt"),
            "status": "passed",
            "passed": True,
            "numeric_tolerance_used": False,
            "public_verifier": "fixture.public_verifier",
        },
        "artifacts": [],
    }


def test_build_tables_has_exact_lattice_and_keeps_metric_families_separate() -> None:
    tables = summary.build_tables(_synthetic_cells())

    assert {name: len(records) for name, records in tables.items()} == summary.EXPECTED_TABLE_ROWS
    first = tables["cell"][0]
    assert {field: first[field] for field in summary.ELIGIBILITY_FIELDS} == (
        summary.ELIGIBILITY_FIELDS
    )
    assert first["source_legacy_mean_iou"] == pytest.approx(0.30)
    assert first["source_unified_global_iou"] == pytest.approx(0.31)
    assert first["delta_adabn_minus_source_legacy_mean_iou"] == pytest.approx(0.02)
    assert first["delta_adabn_minus_source_unified_global_iou"] == pytest.approx(0.02)
    assert first["delta_adabn_minus_source_legacy_fa_per_million_pixels"] == pytest.approx(10.0)
    assert first["delta_adabn_minus_source_unified_fa_per_million_pixels"] == pytest.approx(11.0)
    assert "source_miou" not in first
    assert summary.METRIC_DEFINITIONS["legacy_fa_per_million_pixels"]["unit"] == (
        "false_alarm_pixels_per_1e6_image_pixels"
    )
    assert summary.METRIC_DEFINITIONS["unified_global_iou"]["display_name"] == (
        "unified GlobalIoU"
    )


def test_macro_tables_are_equal_cell_means_and_checkpoint_delta_is_role_difference() -> None:
    tables = summary.build_tables(_synthetic_cells())
    dataset_row = next(
        row
        for row in tables["dataset"]
        if row["dataset"] == "IRSTD-1K"
        and row["checkpoint_role"] == "best_miou"
        and row["aggregation_group"] == "corrupt12"
    )
    # condition indices 1..12 are equally weighted.
    assert dataset_row["source_legacy_mean_iou"] == pytest.approx(
        sum(0.30 + 0.001 * index for index in range(1, 13)) / 12
    )
    assert dataset_row["condition_count"] == 12

    severity_row = next(
        row
        for row in tables["severity"]
        if row["dataset"] == "IRSTD-1K"
        and row["checkpoint_role"] == "best_miou"
        and row["severity_group"] == "S3"
    )
    assert severity_row["condition_count"] == 4

    global_row = next(
        row
        for row in tables["global"]
        if row["checkpoint_role"] == "best_miou"
        and row["aggregation_group"] == "all39"
    )
    assert global_row["condition_count"] == 39

    checkpoint_row = next(
        row
        for row in tables["checkpoint"]
        if row["scope"] == "cell"
        and row["method"] == "Source"
        and row["dataset"] == "IRSTD-1K"
        and row["condition_key"] == "clean_S0"
    )
    assert checkpoint_row["delta_best_pd_minus_best_miou_legacy_mean_iou"] == pytest.approx(0.04)
    assert checkpoint_row["delta_best_pd_minus_best_miou_unified_global_iou"] == pytest.approx(0.04)
    assert checkpoint_row["best_miou_checkpoint_sha256s"].startswith("IRSTD-1K=")
    assert checkpoint_row["best_pd_checkpoint_sha256s"].startswith("IRSTD-1K=")


def test_missing_cell_fails_closed() -> None:
    cells = _synthetic_cells()[:-1]
    with pytest.raises(summary.SummaryContractError, match="lattice is incomplete"):
        summary.build_tables(cells)


def test_adabn_extractor_accepts_signed_exact_delta_and_rejects_schema_drift(
    tmp_path: Path,
) -> None:
    source_cells = tuple(
        cell
        for cell in _synthetic_cells()
        if cell.checkpoint_role == "best_miou"
        and cell.dataset == "IRSTD-1K"
        and cell.method == "Source"
    )
    records: list[dict[str, object]] = []
    for source_cell in source_cells:
        source_values = source_cell.metrics.to_dict()
        adabn_values = {
            "legacy_mean_iou": source_values["legacy_mean_iou"] - 0.01,
            "legacy_pd": source_values["legacy_pd"] + 0.01,
            "legacy_fa_per_million_pixels": source_values["legacy_fa_per_million_pixels"] + 1.0,
            "unified_global_iou": source_values["unified_global_iou"] - 0.02,
            "unified_pd": source_values["unified_pd"] + 0.01,
            "unified_fa_per_million_pixels": source_values["unified_fa_per_million_pixels"] + 2.0,
            "unified_false_positives_per_image": source_values["unified_false_positives_per_image"] + 0.1,
        }
        records.append(
            {
                "condition_index": source_cell.condition_index,
                "condition_key": source_cell.condition_key,
                "corruption": source_cell.corruption,
                "severity": source_cell.severity,
                **adabn_values,
                "source_summary": source_values,
                "deltas_from_source": {
                    field: adabn_values[field] - source_values[field]
                    for field in summary.METRIC_FIELDS
                },
                "metrics": f"conditions/{source_cell.condition_key}/metrics.json",
                "metrics_sha256": _digest(f"adabn/{source_cell.condition_key}"),
            }
        )
    aggregate_dataset = {"dataset": "IRSTD-1K", "conditions": records}
    axis = SimpleNamespace(
        dataset="IRSTD-1K",
        role="best_miou",
        expected_epoch=100,
        checkpoint_sha256=source_cells[0].checkpoint_sha256,
    )
    for record in records:
        metrics_path = tmp_path / "IRSTD-1K" / str(record["metrics"])
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        condition_payload = {
            "schema_version": 2,
            "method": "AdaBN",
            "dataset": "IRSTD-1K",
            "checkpoint_role": "best_miou",
            "checkpoint_sha256": axis.checkpoint_sha256,
            "condition_key": record["condition_key"],
            "corruption": record["corruption"],
            "severity": record["severity"],
            "development_only": True,
            "main_paper_table": False,
            "paper_result": False,
            "extra_best_pd_tuning_episodes": 0,
            "summary": {field: record[field] for field in summary.METRIC_FIELDS},
            "source_summary": record["source_summary"],
            "deltas_from_source": record["deltas_from_source"],
        }
        metrics_path.write_text(
            json.dumps(condition_payload, sort_keys=True), encoding="utf-8"
        )
        record["metrics_sha256"] = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    extracted = summary._adabn_cells(
        aggregate_dataset,
        axis=axis,
        global_root=tmp_path,
        source_by_key={cell.condition_key: cell for cell in source_cells},
    )
    assert len(extracted) == 13
    assert extracted[0].metrics.legacy_mean_iou < source_cells[0].metrics.legacy_mean_iou

    drifted = deepcopy(aggregate_dataset)
    drifted["conditions"][0]["deltas_from_source"]["unregistered_metric"] = 0.0
    with pytest.raises(summary.SummaryContractError, match="delta keys"):
        summary._adabn_cells(
            drifted,
            axis=axis,
            global_root=tmp_path,
            source_by_key={cell.condition_key: cell for cell in source_cells},
        )


def test_hidden_or_noncanonical_input_root_is_rejected(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    hidden = results / ".source.build-123" / "IRSTD-1K"
    with pytest.raises(summary.SummaryContractError, match="private/hidden"):
        summary._reject_hidden_path(hidden, results_root=results, label="fixture")

    expected = results / "source" / "best_pd" / "IRSTD-1K"
    actual = results / "other" / "best_pd" / "IRSTD-1K"
    with pytest.raises(summary.SummaryContractError, match="canonical root"):
        summary._require_canonical_public_root(
            actual, expected, results_root=results, label="fixture"
        )

    public = results / "source" / "best_pd" / "IRSTD-1K"
    public.mkdir(parents=True)
    hidden_child = public / ".condition.build-123"
    hidden_child.mkdir()
    (hidden_child / "must_not_be_opened.bin").write_bytes(b"private")
    with pytest.raises(summary.SummaryContractError, match="private members"):
        summary._assert_no_hidden_descendants(public, label="fixture")


def test_summary_bundle_is_atomic_no_replace_and_development_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = summary.VerifiedInputs(
        cells=_synthetic_cells(), lineage=_synthetic_lineage()
    )
    output = tmp_path / "summary"
    monkeypatch.setattr(summary, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(
        summary, "collect_verified_inputs", lambda **_: verified
    )
    audit = summary.generate_summary(output_dir=output)

    assert audit["complete"]["complete"] is True
    assert audit["complete"]["development_only"] is True
    assert audit["complete"]["main_paper_table"] is False
    saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert saved["scientific_eligibility"] == {
        "development_only": True,
        "main_paper": False,
        "main_paper_table": False,
        "reason": "best_miou and best_pd checkpoints were selected on the fixed test split",
        "tier": "development_test_selected",
    }
    assert saved["table_row_counts"] == summary.EXPECTED_TABLE_ROWS
    cell_envelope = json.loads(
        (output / "cell_metrics.json").read_text(encoding="utf-8")
    )
    assert cell_envelope["scientific_eligibility_tier"] == (
        "development_test_selected"
    )
    cell_header = (output / "cell_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert cell_header.startswith(
        "scientific_eligibility_tier,development_only,main_paper,main_paper_table,"
    )
    assert not list(tmp_path.glob(".summary.build-*"))

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        summary.generate_summary(output_dir=output)


def test_prepublish_guard_failure_leaves_no_output_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = summary.VerifiedInputs(
        cells=_synthetic_cells(), lineage=_synthetic_lineage()
    )
    output = tmp_path / "summary"
    calls = 0

    def collector(**_: object) -> summary.VerifiedInputs:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise summary.SummaryContractError("live verifier failed")
        return verified

    monkeypatch.setattr(summary, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(summary, "collect_verified_inputs", collector)
    with pytest.raises(summary.SummaryContractError, match="live verifier failed"):
        summary.generate_summary(output_dir=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".summary.build-*"))


def test_postrename_verifier_failure_rolls_back_canonical_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = summary.VerifiedInputs(
        cells=_synthetic_cells(), lineage=_synthetic_lineage()
    )
    output = tmp_path / "summary"
    real_verifier = summary.verify_summary_artifact

    def verifier(root: Path) -> dict[str, object]:
        if Path(root) == output:
            raise summary.SummaryContractError("post-rename verifier failed")
        return real_verifier(root)

    monkeypatch.setattr(summary, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(
        summary, "collect_verified_inputs", lambda **_: verified
    )
    monkeypatch.setattr(summary, "verify_summary_artifact", verifier)
    with pytest.raises(summary.SummaryContractError, match="post-rename verifier failed"):
        summary.generate_summary(output_dir=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".summary.build-*"))


def test_formal_writer_has_no_caller_supplied_payload_or_authorization() -> None:
    parameters = inspect.signature(summary._publish_summary_bundle).parameters
    assert set(parameters) == {"axis_config_path", "output_dir"}
    assert "payload" not in parameters
    assert "authorization" not in parameters


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_real_best_miou_and_best_pd_clean_fixtures_pass_public_verifier() -> None:
    config = checkpoint_axis.load_axis_config()
    receipt = checkpoint_axis.verify_parity_receipt(config=config)
    assert receipt["passed"] is True
    expected = {
        ("best_miou", "IRSTD-1K"): {
            "legacy_mean_iou": 0.6870056497175141,
            "legacy_pd": 0.9319727891156463,
            "legacy_fa_per_million_pixels": 10.703926655783583,
            "unified_global_iou": 0.6874576271186441,
            "unified_fa_per_million_pixels": 11.690813510572138,
        },
        ("best_miou", "NUAA-SIRST"): {
            "legacy_mean_iou": 0.7097310208744418,
            "legacy_pd": 0.9581749049429658,
            "legacy_fa_per_million_pixels": 25.668990946261683,
            "unified_global_iou": 0.7097310208744418,
            "unified_fa_per_million_pixels": 26.453321225175234,
        },
        ("best_miou", "NUDT-SIRST"): {
            "legacy_mean_iou": 0.8021570182394925,
            "legacy_pd": 0.9767195767195768,
            "legacy_fa_per_million_pixels": 18.912625600056476,
            "unified_global_iou": 0.8021570182394925,
            "unified_fa_per_million_pixels": 18.912625600056476,
        },
        ("best_pd", "IRSTD-1K"): {
            "legacy_mean_iou": 0.6715891558298435,
            "legacy_pd": 0.9523809523809523,
            "legacy_fa_per_million_pixels": 17.30847714552239,
            "unified_global_iou": 0.6720299757549041,
            "unified_fa_per_million_pixels": 18.295364000310943,
        },
        ("best_pd", "NUAA-SIRST"): {
            "legacy_mean_iou": 0.6929541678882049,
            "legacy_pd": 0.9809885931558935,
            "legacy_fa_per_million_pixels": 46.70330297167056,
            "unified_global_iou": 0.6929541678882049,
            "unified_fa_per_million_pixels": 48.628477292640184,
        },
        ("best_pd", "NUDT-SIRST"): {
            "legacy_mean_iou": 0.7906652194138626,
            "legacy_pd": 0.982010582010582,
            "legacy_fa_per_million_pixels": 24.106129106268828,
            "unified_global_iou": 0.7906652194138626,
            "unified_fa_per_million_pixels": 24.106129106268828,
        },
    }
    for role in summary.ROLES:
        for dataset in summary.DATASETS:
            root = summary._artifact_root(
                config, artifact_kind="clean", role=role, dataset=dataset
            )
            axis = checkpoint_axis.resolve_axis(
                config,
                dataset=dataset,
                role=role,  # type: ignore[arg-type]
                artifact_kind="clean",
                output_override=(root if role == "best_miou" else None),
                verify_files=True,
                verify_parity_gate=True,
            )
            metrics, lineage = summary.verify_and_extract_clean_artifact(
                root, axis=axis
            )
            values = metrics.to_dict()
            for field, value in expected[(role, dataset)].items():
                assert values[field] == value
            assert lineage["public_verifier"] == (
                "benchmark.checkpoint_axis.verify_published_artifact"
            )
            assert Path(str(lineage["root"])) == root
