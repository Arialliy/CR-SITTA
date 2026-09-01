from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

import analysis.analyze_tent_ss_noop as analyzer
from analysis.analyze_tent_ss_noop import (
    AnalysisInputError,
    INPUT_ARTIFACT_TYPE,
    analyze_input_manifest,
    publish_analysis,
    sha256_file,
    verify_output,
)
from tta.diagnostics import (
    FUNCTIONAL_NOOP,
    METRIC_NOOP,
    NUMERIC_NOOP,
    TASK_EFFECTIVE_CHANGE,
    THRESHOLD_NOOP,
    DiagnosticError,
    NoOpThresholds,
    analyze_noop_episode,
    parameter_change_report,
    threshold_margin_report,
)


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    return np.log(probability / (1.0 - probability))


def _params(*, changed: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    before = {"bn.weight": np.array([1.0, 2.0], dtype=np.float32)}
    after = {
        "bn.weight": np.array(
            [1.1 if changed else 1.0, 2.0], dtype=np.float32
        )
    }
    return before, after


def _analyze(
    pre_probability: np.ndarray,
    post_probability: np.ndarray,
    *,
    changed_parameter: bool = True,
    target: np.ndarray | None = None,
    thresholds: NoOpThresholds | None = None,
) -> dict[str, Any]:
    parameter_pre, parameter_post = _params(changed=changed_parameter)
    if target is None:
        target = np.zeros_like(pre_probability)
    return analyze_noop_episode(
        parameter_pre=parameter_pre,
        parameter_post=parameter_post,
        logits_pre=_logit(pre_probability),
        logits_post=_logit(post_probability),
        target=target,
        thresholds=thresholds,
    )


def test_noop_priority_numeric_precedes_observed_function_change() -> None:
    report = _analyze(
        np.full((2, 2), 0.1),
        np.full((2, 2), 0.9),
        changed_parameter=False,
    )

    assert report["classification"] == NUMERIC_NOOP
    assert report["parameter_change"]["numerically_zero_at_floor"] is True
    assert report["binary_transitions"]["binary_pixel_xor_count"] == 4


def test_noop_priority_functional_uses_both_frozen_null_floors() -> None:
    thresholds = NoOpThresholds(
        logit_null_floor=0.01,
        probability_null_floor=0.01,
    )
    pre = np.full((2, 2), 0.4)
    post = 1.0 / (1.0 + np.exp(-(_logit(pre) + 0.001)))

    report = _analyze(pre, post, thresholds=thresholds)

    assert report["classification"] == FUNCTIONAL_NOOP
    assert report["functional_change"]["functionally_zero_at_floors"] is True
    assert report["parameter_change"]["max_abs_delta"] > 0.0


def test_threshold_noop_has_function_change_without_binary_crossing() -> None:
    report = _analyze(
        np.array([[0.1, 0.2], [0.8, 0.9]]),
        np.array([[0.2, 0.3], [0.7, 0.8]]),
    )

    assert report["classification"] == THRESHOLD_NOOP
    assert report["binary_transitions"] == {
        "BG_to_FG_pixel_count": 0,
        "FG_to_BG_pixel_count": 0,
        "binary_pixel_xor_count": 0,
    }
    assert report["functional_change"]["probability"]["absolute"]["q99"] > 0
    assert report["parameter_max_abs_delta"] > 0
    assert report["relative_step_norm"] > 0
    assert report["logit_delta_q99"] is not None
    assert report["prob_delta_q99"] is not None
    assert report["component_count_pre"] == report["state"]["pre"]["component_count"]
    assert report["foreground_probability_mass_pre"] == report["state"]["pre"][
        "foreground_probability_mass_mean"
    ]


def test_metric_noop_detects_mask_xor_with_identical_official_counts() -> None:
    pre = np.full((3, 3), 0.1)
    post = np.full((3, 3), 0.1)
    pre[0, 0] = 0.9
    post[2, 2] = 0.9

    report = _analyze(pre, post)

    assert report["classification"] == METRIC_NOOP
    assert report["binary_transitions"]["binary_pixel_xor_count"] == 2
    assert report["metric_counts"]["identical"] is True
    assert report["state"]["pre"]["component_count"] == 1
    assert report["state"]["post"]["component_count"] == 1


def test_task_effective_change_reports_directional_transitions_and_metrics() -> None:
    pre = np.full((2, 2), 0.1)
    post = np.full((2, 2), 0.1)
    pre[0, 0] = 0.9

    report = _analyze(pre, post)

    assert report["classification"] == TASK_EFFECTIVE_CHANGE
    assert report["binary_transitions"]["FG_to_BG_pixel_count"] == 1
    assert report["metric_counts"]["identical"] is False
    assert report["state"]["pre"]["foreground_fraction"] == pytest.approx(0.25)
    assert report["state"]["post"]["foreground_fraction"] == 0.0
    assert report["state"]["pre"]["entropy_mean"] > 0.0


def test_parameter_report_has_global_per_tensor_and_quantile_evidence() -> None:
    before = {
        "a": np.array([0.0, 1.0], dtype=np.float32),
        "b": np.array([2.0], dtype=np.float32),
    }
    after = {
        "a": np.array([0.0, 1.5], dtype=np.float32),
        "b": np.array([1.0], dtype=np.float32),
    }

    report = parameter_change_report(before, after)

    assert report["tensor_count"] == 2
    assert report["scalar_count"] == 3
    assert report["changed_tensor_count"] == 2
    assert report["max_abs_delta"] == 1.0
    assert report["absolute_delta_quantiles"]["count"] == 3
    assert [item["name"] for item in report["per_tensor"]] == ["a", "b"]


def test_threshold_margin_contains_all_required_oracle_strata() -> None:
    pre = np.array([[0.60, 0.40, 0.49], [0.51, 0.90, 0.10]])
    post = np.array([[0.40, 0.60, 0.52], [0.48, 0.80, 0.20]])
    target = np.array([[1, 1, 1], [0, 0, 1]], dtype=np.float32)

    report = threshold_margin_report(pre, post, target)
    strata = report["strata"]

    assert set(strata) == {
        "all",
        "near_threshold",
        "gt_target",
        "gt_background",
        "source_false_positive",
        "source_missed_target",
    }
    assert strata["all"]["pixel_count"] == 6
    assert strata["all"]["binary_xor_count"] == 4
    assert strata["near_threshold"]["pixel_count"] == 2
    assert strata["source_false_positive"]["pixel_count"] == 2
    assert strata["source_missed_target"]["pixel_count"] == 3
    assert strata["gt_target"]["threshold_margin_pre"]["q99"] is not None


def test_provided_probability_must_match_logits() -> None:
    before, after = _params(changed=True)
    with pytest.raises(DiagnosticError, match="do not match sigmoid"):
        analyze_noop_episode(
            parameter_pre=before,
            parameter_post=after,
            logits_pre=np.zeros((2, 2)),
            logits_post=np.zeros((2, 2)),
            probability_pre=np.zeros((2, 2)),
            probability_post=np.zeros((2, 2)),
            target=np.zeros((2, 2)),
        )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _reference(project: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(project).as_posix(),
        "sha256": sha256_file(path),
    }


def _make_analysis_input(project: Path) -> Path:
    dataset = "TOY-SIRST"
    image_id = "train-001"
    train_split = project / f"datasets/{dataset}/img_idx/train_{dataset}.txt"
    pilot_ids = project / f"configs/tta_train_side_pilot_v2/{dataset}.txt"
    train_split.parent.mkdir(parents=True, exist_ok=True)
    pilot_ids.parent.mkdir(parents=True, exist_ok=True)
    train_split.write_text(f"{image_id}\ntrain-002\n", encoding="utf-8")
    pilot_ids.write_text(f"{image_id}\n", encoding="utf-8")

    pilot_manifest = project / "configs/tta_train_side_pilot_v2/manifest.json"
    _write_json(
        pilot_manifest,
        {
            "datasets": {
                dataset: {
                    "checks": {
                        "output_ids_contained_in_train": True,
                        "output_ids_unique": True,
                        "output_test_overlap_count": 0,
                        "train_ids_unique": True,
                        "train_test_overlap_count": 0,
                    },
                    "output": {
                        "file_sha256": sha256_file(pilot_ids),
                        "path": pilot_ids.relative_to(project).as_posix(),
                    },
                    "train_split": {
                        "path": train_split.relative_to(project).as_posix(),
                        "sha256": sha256_file(train_split),
                    },
                }
            },
            "metadata_io_boundary": {"test_pixels_opened": 0},
            "no_validation_split": True,
            "paper_result": False,
        },
    )

    cache_root = project / f"results/binary_tent/ss_calibration_cache_v2/{dataset}"
    cache_manifest = cache_root / "manifest.json"
    _write_json(
        cache_manifest,
        {
            "calibration_ids_file_sha256": sha256_file(pilot_ids),
            "calibration_ids_path": pilot_ids.relative_to(project).as_posix(),
            "dataset": dataset,
            "image_ids": [image_id],
            "label_firewall": {
                "method_received_labels": False,
                "targets_for_outer_evaluator_only": True,
            },
            "source_open_scope": {"test_images": 0, "test_masks": 0},
            "split_role": "train_side_pilot_v2_derived_64",
            "train_split": train_split.relative_to(project).as_posix(),
            "train_split_sha256": sha256_file(train_split),
        },
    )
    method_manifest = cache_root / "method_input_manifest.json"
    _write_json(
        method_manifest,
        {
            "dataset": dataset,
            "forbidden_fields": [
                "ground_truth",
                "gt",
                "label",
                "mask",
                "target",
            ],
            "outer_manifest_sha256": sha256_file(cache_manifest),
            "targets_exposed": False,
        },
    )
    complete = cache_root / "COMPLETE.json"
    _write_json(
        complete,
        {
            "complete": True,
            "dataset": dataset,
            "manifest_sha256": sha256_file(cache_manifest),
            "method_input_manifest_sha256": sha256_file(method_manifest),
            "method_received_labels": False,
            "test_images_opened": 0,
            "test_masks_opened": 0,
        },
    )

    array_path = project / "diagnostic_inputs/episode-001.npz"
    array_path.parent.mkdir(parents=True)
    pre_probability = np.array([[0.1, 0.49], [0.9, 0.2]], dtype=np.float32)
    post_probability = np.array([[0.1, 0.51], [0.8, 0.2]], dtype=np.float32)
    np.savez(
        array_path,
        parameter_pre_0=np.array([1.0, 2.0], dtype=np.float32),
        parameter_post_0=np.array([1.1, 2.0], dtype=np.float32),
        logits_pre=_logit(pre_probability).astype(np.float32),
        logits_post=_logit(post_probability).astype(np.float32),
        target=np.array([[0, 1], [0, 0]], dtype=np.float32),
    )

    input_manifest = project / "diagnostic_inputs/manifest.json"
    _write_json(
        input_manifest,
        {
            "artifact_type": INPUT_ARTIFACT_TYPE,
            "episodes": [
                {
                    "arrays": _reference(project, array_path),
                    "dataset": dataset,
                    "episode_id": "episode-001",
                    "image_id": image_id,
                    "logits_post_key": "logits_post",
                    "logits_pre_key": "logits_pre",
                    "metadata": {
                        "corruption": "gaussian_noise",
                        "severity": 3,
                    },
                    "parameters": [
                        {
                            "name": "bn.weight",
                            "post_key": "parameter_post_0",
                            "pre_key": "parameter_pre_0",
                        }
                    ],
                    "split_role": "train",
                    "target_key": "target",
                }
            ],
            "label_boundary": {
                "method_label_accesses": 0,
                "targets_outer_evaluator_only": True,
            },
            "provenance": {
                "datasets": {
                    dataset: {
                        "cache_complete": _reference(project, complete),
                        "cache_manifest": _reference(project, cache_manifest),
                        "method_input_manifest": _reference(
                            project, method_manifest
                        ),
                        "pilot_ids": _reference(project, pilot_ids),
                        "pilot_manifest": _reference(project, pilot_manifest),
                        "train_split": _reference(project, train_split),
                    }
                },
                "train_provenance_complete": True,
            },
            "schema_version": 1,
            "scope": {
                "oracle_analysis": True,
                "paper_test_result": False,
                "source_train_derived": True,
                "split_role": "train",
                "use_test_images": False,
                "use_test_labels": False,
            },
            "thresholds": NoOpThresholds().to_dict(),
        },
    )
    return input_manifest


def test_train_side_analyzer_publishes_mandatory_oracle_metadata(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    input_manifest = _make_analysis_input(project)

    run = analyze_input_manifest(input_manifest, project_root=project)
    output = project / "results/binary_tent/tent_ss_noop_diagnostics_v1"
    result = publish_analysis(run, output, project_root=project)

    assert result["status"] == "created_and_verified"
    assert verify_output(output)["status"] == "verified"
    for filename in (
        "summary.json",
        "provenance.json",
        "artifact_manifest.json",
        "COMPLETE.json",
    ):
        value = json.loads((output / filename).read_text(encoding="utf-8"))
        assert value["oracle_analysis"] is True
        assert value["paper_test_result"] is False
        assert value["method_label_accesses"] == 0
        assert value["outer_evaluator_label_accesses"] == 1
        assert value["split_role"] == "train"
    record = json.loads(
        (output / "episode_diagnostics.jsonl").read_text(encoding="utf-8")
    )
    assert record["oracle_analysis"] is True
    assert record["paper_test_result"] is False
    assert record["method_label_accesses"] == 0
    assert record["outer_evaluator_label_accesses"] == 1
    assert record["threshold_margin"]["strata"]["gt_target"]["pixel_count"] == 1


def test_analyzer_rejects_test_role_before_loading_episode_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    input_manifest = _make_analysis_input(project)
    value = json.loads(input_manifest.read_text(encoding="utf-8"))
    value["scope"]["split_role"] = "test"
    _write_json(input_manifest, value)
    called = False

    def forbidden_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("np.load must not be called")

    monkeypatch.setattr(analyzer.np, "load", forbidden_load)
    with pytest.raises(AnalysisInputError, match="source-train-only"):
        analyze_input_manifest(input_manifest, project_root=project)
    assert called is False


def test_analyzer_fails_closed_when_train_provenance_is_missing(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    input_manifest = _make_analysis_input(project)
    value = json.loads(input_manifest.read_text(encoding="utf-8"))
    del value["provenance"]["datasets"]["TOY-SIRST"]["cache_complete"]
    _write_json(input_manifest, value)

    with pytest.raises(AnalysisInputError, match="fields must be exact"):
        analyze_input_manifest(input_manifest, project_root=project)


def test_analyzer_rejects_image_outside_frozen_train_pilot(tmp_path: Path) -> None:
    project = tmp_path / "project"
    input_manifest = _make_analysis_input(project)
    value = json.loads(input_manifest.read_text(encoding="utf-8"))
    value["episodes"][0]["image_id"] = "not-in-train"
    _write_json(input_manifest, value)

    with pytest.raises(AnalysisInputError, match="not bound to frozen train"):
        analyze_input_manifest(input_manifest, project_root=project)
