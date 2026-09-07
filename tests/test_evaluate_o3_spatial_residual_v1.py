"""CPU synthetic tests only: no repository images, checkpoints or train GT."""

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from analysis import evaluate_o3_spatial_residual_v1 as outer


def _counts():
    return {
        "intersection_pixels": 100, "false_positive_pixels": 20,
        "false_negative_pixels": 10, "true_negative_pixels": 64 * 256 * 256 - 130,
        "predicted_positive_pixels": 120, "target_positive_pixels": 110,
        "detected_targets": 8, "total_targets": 10, "false_alarm_pixels": 5,
        "total_image_pixels": 64 * 256 * 256, "image_count": 64,
    }


def _cells():
    records = []
    for dataset in outer.b4.DATASETS:
        for corruption, severity in outer.b4.CONDITIONS:
            source = {"iou": 0.7, "pd": 0.8, "fa_per_million": 10.0}
            previous = {"iou": 0.701, "pd": 0.8, "fa_per_million": 10.1}
            adapted = {"iou": 0.71, "pd": 0.8, "fa_per_million": 9.0}
            records.append({"dataset": dataset, "condition": outer.b4._condition_key(corruption, severity),
                            "corruption_family": corruption, "severity": severity,
                            "source": source, "previous_o3": previous, "adapted": adapted})
    return records


def _historical_records():
    return [{"dataset": r["dataset"], "condition": r["condition"],
             "corruption_family": r["corruption_family"], "severity": f"S{r['severity']}",
             "candidate_id": "O3_P2", "episode_count": 64,
             "source_counts": _counts(), "adapted_counts": _counts()}
            for r in _cells()]


def _jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_aggregate_metrics_and_clean_exclusion():
    cells = _cells()
    for row in cells:
        if row["corruption_family"] == "clean":
            row["adapted"]["iou"] = 0.9
    result = outer.summarize_performance(cells)
    assert result["nonclean_macro"]["cell_count"] == 36
    assert result["nonclean_macro"]["adapted"]["iou"] == pytest.approx(0.71)
    assert result["nonclean_macro"]["delta_vs_source"]["iou_pp"] == pytest.approx(1.0)
    assert result["nonclean_macro"]["delta_vs_previous_o3"]["iou_pp"] == pytest.approx(0.9)
    assert result["nonclean_macro"]["delta_vs_source"]["fa_per_million"] == pytest.approx(-1.0)
    assert all(v["cell_count"] == 9 for v in result["families_nonclean"].values())
    assert all(v["cell_count"] == 12 for v in result["datasets_nonclean"].values())
    assert result["all_advancement_goals_met"] is True
    assert result["paper_result"] is False
    assert result["formal_test"] is False


@pytest.mark.parametrize("family", ["gaussian_blur", "gaussian_noise"])
def test_family_must_actually_improve_source(family):
    cells = _cells()
    for row in cells:
        if row["corruption_family"] == family:
            row["adapted"]["iou"] = row["source"]["iou"]
    result = outer.summarize_performance(cells)
    assert result["advancement_goals"][f"{family}_iou_above_source"] is False
    assert result["all_advancement_goals_met"] is False


@pytest.mark.parametrize("family", ["low_contrast", "stripe_noise"])
def test_existing_family_gain_must_be_preserved(family):
    cells = _cells()
    for row in cells:
        if row["corruption_family"] == family:
            row["adapted"]["iou"] = row["previous_o3"]["iou"] - 0.001
    result = outer.summarize_performance(cells)
    assert result["advancement_goals"][f"{family}_iou_preserves_previous_o3"] is False


def test_fa_is_not_excused_by_iou_improvement():
    cells = _cells()
    for row in cells:
        row["adapted"]["fa_per_million"] = 10.01
    result = outer.summarize_performance(cells)
    assert result["advancement_goals"]["nonclean_iou_above_previous_o3"] is True
    assert result["advancement_goals"]["nonclean_fa_not_above_source"] is False
    assert result["all_advancement_goals_met"] is False


def test_clean_absolute_drop_boundary_and_strict_fa():
    cells = _cells()
    clean = cells[0]
    clean["adapted"] = {"iou": 0.698, "pd": 0.798, "fa_per_million": 10.0}
    result = outer.summarize_performance(cells)
    assert result["all_advancement_goals_met"]
    clean["adapted"]["iou"] -= 0.000001
    result = outer.summarize_performance(cells)
    assert not result["advancement_goals"]["IRSTD-1K_clean_iou_within_0p002"]
    clean["adapted"]["fa_per_million"] += 0.000001
    result = outer.summarize_performance(cells)
    assert not result["advancement_goals"]["IRSTD-1K_clean_fa_not_above_source"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "identity", "nonfinite"])
def test_summary_rejects_invalid_cell_set(change):
    cells = _cells()
    if change == "missing":
        cells.pop()
    elif change == "duplicate":
        cells[-1] = copy.deepcopy(cells[0])
    elif change == "identity":
        cells[0]["severity"] = 5
    else:
        cells[1]["adapted"]["iou"] = float("nan")
    with pytest.raises(outer.PerformanceEvaluationError):
        outer.summarize_performance(cells)


def test_historical_metrics_come_only_from_counts(tmp_path):
    path = tmp_path / "old.jsonl"
    records = _historical_records()
    _jsonl(path, records)
    result = outer.load_historical_cells(path)
    endpoint = result[("IRSTD-1K", "clean_S0")]["previous_o3"]
    assert endpoint["iou"] == pytest.approx(100 / 130)
    assert endpoint["pd"] == 0.8
    assert endpoint["fa_per_million"] == pytest.approx(5 / (64 * 256 * 256) * 1e6)
    assert "normalized_iou" not in endpoint


@pytest.mark.parametrize("change", ["missing", "duplicate", "bad_count", "wrong_severity", "wrong_size"])
def test_historical_cells_fail_closed(tmp_path, change):
    records = _historical_records()
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records.append(records[0])
    elif change == "bad_count":
        records[0]["source_counts"]["intersection_pixels"] += 1
    elif change == "wrong_severity":
        records[0]["severity"] = "S1"
    else:
        records[0]["episode_count"] = 16
    path = tmp_path / "old.jsonl"
    _jsonl(path, records)
    with pytest.raises((outer.PerformanceEvaluationError, outer.b4.StageB4ProtocolError)):
        outer.load_historical_cells(path)


def test_source_requires_exact_all_counts():
    historical = outer.endpoint_from_counts(_counts())
    source = dict(historical)
    outer.assert_paired_source(source, historical)
    source["false_alarm_pixels"] += 1
    with pytest.raises(outer.PerformanceEvaluationError, match="false_alarm_pixels"):
        outer.assert_paired_source(source, historical)


def test_bad_candidate_barrier_never_loads_gt(monkeypatch, tmp_path):
    def barrier(**kwargs):
        raise RuntimeError("candidate incomplete")

    def forbidden(*args, **kwargs):
        pytest.fail("target access before global barrier")

    monkeypatch.setitem(sys.modules, "analysis.spatial_residual_contract_v1",
                        SimpleNamespace(verify_all_candidates=barrier))
    monkeypatch.setattr(outer.b3, "_load_outer_targets", forbidden)
    with pytest.raises(RuntimeError, match="candidate incomplete"):
        outer.evaluate(tmp_path / "unused.yaml")


def _mock_run(monkeypatch, tmp_path):
    monkeypatch.setattr(outer, "REPOSITORY", tmp_path)
    cfg = {"result_root": "results/new", "parent_config": "old.yaml",
           "parent_aggregate": {"path": "results/old"}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    old = tmp_path / "results/old"
    old.mkdir(parents=True)
    _jsonl(old / "science_gate_cells.jsonl", _historical_records())
    for name in ("manifest.json", "COMPLETE.json"):
        (old / name).write_text("{}", encoding="utf-8")
    candidate = tmp_path / "results/new/candidate"
    for dataset in outer.b4.DATASETS:
        root = candidate / dataset
        root.mkdir(parents=True)
        (root / "summary.json").write_text(json.dumps({"image_ids": [f"id{i}" for i in range(64)]}), encoding="utf-8")
        for name in ("manifest.json", "COMPLETE.json"):
            (root / name).write_text("{}", encoding="utf-8")
    events = []

    def barrier(**kwargs):
        events.append("barrier")
        return {d: {} for d in outer.b4.DATASETS}

    def targets(parent, dataset):
        assert events[0] == "barrier"
        events.append(f"targets:{dataset}")
        return None

    endpoint = {**outer.endpoint_from_counts(_counts()), "normalized_iou": 0.75}
    fake_result = SimpleNamespace(to_dict=lambda: {"fixed": endpoint})
    monkeypatch.setitem(sys.modules, "analysis.spatial_residual_contract_v1",
                        SimpleNamespace(verify_all_candidates=barrier))
    monkeypatch.setattr(outer.b4, "load_contract", lambda path: object())
    monkeypatch.setattr(outer.b3, "_load_outer_targets", targets)
    monkeypatch.setattr(outer.b3, "_evaluation_result", lambda *args: fake_result)
    monkeypatch.setattr(outer.b3, "_endpoint_summary", lambda result: dict(endpoint))
    monkeypatch.setattr(outer, "_probabilities", lambda path: None)
    return config, events, endpoint


def test_outer_writes_all_cells_complete_and_reuse_does_not_load_gt(monkeypatch, tmp_path):
    config, events, _ = _mock_run(monkeypatch, tmp_path)
    result = outer.evaluate(config)
    root = Path(result["path"])
    assert events[0] == "barrier" and events[-1] == "barrier"
    assert sum(v.startswith("targets:") for v in events) == 3
    assert len((root / "cells.jsonl").read_text().splitlines()) == 39
    assert len(list((root / "conditions").rglob("*_metrics.json"))) == 78
    assert (root / "COMPLETE.json").is_file()
    assert result["summary"]["outer_target_loader_calls"] == 3
    assert not result["summary"]["all_advancement_goals_met"]
    before = len(events)
    reused = outer.evaluate(config)
    assert reused["status"] == "existing_verified_complete_no_op"
    assert events[before:] == ["barrier"]


def test_source_mismatch_preserves_partial_and_has_no_complete(monkeypatch, tmp_path):
    config, events, endpoint = _mock_run(monkeypatch, tmp_path)
    endpoint["false_alarm_pixels"] += 1
    with pytest.raises(outer.PerformanceEvaluationError, match="Source count differs"):
        outer.evaluate(config)
    output = tmp_path / "results/new/outer/R0"
    assert (output / "FAILED.json").is_file()
    assert not (output / "COMPLETE.json").exists()
    before = len(events)
    with pytest.raises(outer.PerformanceEvaluationError, match="incomplete outer artifact"):
        outer.evaluate(config)
    assert events[before:] == ["barrier"]


def test_existing_summary_tamper_detected_without_target_reload(monkeypatch, tmp_path):
    config, events, _ = _mock_run(monkeypatch, tmp_path)
    result = outer.evaluate(config)
    (Path(result["path"]) / "summary.json").write_text("{}", encoding="utf-8")
    before = len(events)
    with pytest.raises(outer.PerformanceEvaluationError, match="existing outer artifact differs"):
        outer.evaluate(config)
    assert events[before:] == ["barrier"]


@pytest.mark.parametrize("dtype,shape", [(np.float64, (64, 1, 256, 256)), (np.float32, (1, 1, 2, 2))])
def test_probability_shape_and_dtype_rejected(tmp_path, dtype, shape):
    path = tmp_path / "probabilities.npy"
    np.save(path, np.zeros(shape, dtype=dtype))
    with pytest.raises(outer.PerformanceEvaluationError, match="shape/dtype"):
        outer._probabilities(path)


def test_probability_values_and_read_only_memmap(tmp_path):
    path = tmp_path / "probabilities.npy"
    value = np.zeros((64, 1, 256, 256), dtype=np.float32)
    np.save(path, value)
    result = outer._probabilities(path)
    assert not result.flags.writeable
    value[0, 0, 0, 0] = np.nan
    np.save(path, value)
    with pytest.raises(outer.PerformanceEvaluationError, match="values invalid"):
        outer._probabilities(path)
