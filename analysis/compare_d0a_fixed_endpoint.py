"""P0/P1: derive a new report from existing NUDT test metrics, never re-test.

Original sealed reports and their scripts are not modified. The new naming and
interpretation receipt live outside the original evaluation artifact tree.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import d0a_v7_common as common

ROOT = common.ROOT
SOURCE_RUN = ROOT / "results/retraining_fixed_split/NUDT-SIRST"
D0A_RUN = ROOT / "results/cr_sitta/d0a_supervised_lfhf_train_v2/NUDT-SIRST"
EVALUATION = ROOT / "results/cr_sitta/d0a_development_test_v1/NUDT-SIRST"
BASELINES = {
    "best_miou": ROOT / "results/baseline/NUDT-SIRST/best_miou/metrics.json",
    "best_pd": ROOT / "results/baseline_checkpoint_axis_v2/best_pd/NUDT-SIRST/metrics.json",
}
NAMES = ("miou", "pd", "fa_per_pixel_x1e6")
DISPLAY_NAME = "D0-A prepared source, fixed epoch1000, no TTA"


def read_epoch(path: Path, epoch: int) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    matches = [row for row in rows if row.get("epoch") == epoch]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one epoch={epoch}, found {len(matches)}")
    if [row["epoch"] for row in rows] != list(range(500, 1001)):
        raise ValueError("expected the complete, unique historical 500..1000 log")
    return matches[0]


def extract_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    result = {key: float(row[key]) for key in NAMES}
    if not all(math.isfinite(x) for x in result.values()):
        raise ValueError("metrics must be finite")
    if not all(0 <= result[key] <= 1 for key in ("miou", "pd")) or result[NAMES[2]] < 0:
        raise ValueError("metric units or range invalid")
    return result


def compare(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    candidate, reference = extract_metrics(candidate), extract_metrics(reference)
    delta = {key: candidate[key] - reference[key] for key in NAMES}
    harmful = delta["miou"] < 0 and delta["pd"] < 0 and delta[NAMES[2]] > 0
    better = delta["miou"] > 0 and delta["pd"] > 0 and delta[NAMES[2]] < 0
    status = "all_three_worse" if harmful else "all_three_better" if better else "mixed_or_tied"
    return {"candidate": candidate, "reference": reference, "delta": delta,
            "delta_percentage_points": {key: delta[key] * 100 for key in ("miou", "pd")},
            "relative_fa_change_percent": delta[NAMES[2]] / reference[NAMES[2]] * 100
                if reference[NAMES[2]] else None,
            "E0_all_three_harm_condition": harmful, "tradeoff": status}


def preflight(config_path: Path = common.DEFAULT_CONFIG) -> dict[str, Any]:
    config = common.read_config(config_path)
    output = ROOT / config["result_root"] / "fixed_endpoint/NUDT-SIRST"
    complete = json.loads((EVALUATION / "COMPLETE.json").read_text())
    manifest_path = EVALUATION / "artifact_manifest.json"
    if not complete["complete"] or common.sha256_file(manifest_path) != complete["artifact_manifest"]["sha256"]:
        raise ValueError("original evaluation completion seal mismatch")
    manifest = json.loads(manifest_path.read_text())["files"]
    paths = [Path(__file__), SOURCE_RUN / "test_metrics.jsonl", SOURCE_RUN / "run_contract.json",
             D0A_RUN / "run_contract.json", EVALUATION / "COMPLETE.json", manifest_path]
    for name in ("metrics.json", "comparison.json", "comparison.md", "run_contract.json", "EVALUATION_FREEZE.json"):
        path = EVALUATION / name
        if common.sha256_file(path) != manifest[name]:
            raise ValueError(f"original sealed metadata changed: {name}")
        paths.append(path)
    old_freeze = json.loads((EVALUATION / "EVALUATION_FREEZE.json").read_text())
    # Check the original evaluation entrypoint without reading test payloads.
    entry = next(x for x in old_freeze["input_bindings"]
                 if Path(x["path"]).name == "evaluate_cr_sitta_d0a_development.py")
    if common.sha256_file(entry["path"]) != entry["sha256"]:
        raise ValueError("original frozen evaluation code changed")
    paths.append(Path(entry["path"]))
    metrics = json.loads((EVALUATION / "metrics.json").read_text())
    original_comparison = json.loads((EVALUATION / "comparison.json").read_text())
    if metrics.get("tta") is not False or metrics.get("checkpoint_test_selected") is not False:
        raise ValueError("D0-A report is not a fixed-epoch, no-TTA evaluation")
    row = read_epoch(SOURCE_RUN / "test_metrics.jsonl", 1000)
    if row["images"] != 664 or metrics["evaluated_images"] != 664:
        raise ValueError("incomplete fixed test comparison")
    source_contract = json.loads((SOURCE_RUN / "run_contract.json").read_text())
    prepared_contract = json.loads((D0A_RUN / "run_contract.json").read_text())
    source_config, prepared_config = source_contract["run_config"], prepared_contract["run_config"]
    if source_config["expected_test_split_sha256"] != metrics["split_sha256"]:
        raise ValueError("fixed test split hash differs")
    if source_config["expected_train_split_sha256"] != prepared_config["expected_train_split_sha256"]:
        raise ValueError("source training splits differ")
    baselines = {}
    for axis, path in BASELINES.items():
        anchor = original_comparison["comparisons"][axis]["baseline_metrics_file"]
        if common.sha256_file(path) != anchor["sha256"]:
            raise ValueError(f"sealed {axis} comparator changed")
        baselines[axis] = json.loads(path.read_text())
        paths.append(path)
    shared = ("seed", "batch_size", "learning_rate", "epochs", "warm_epochs", "base_size", "crop_size")
    matched = {key: {"source": source_config[key], "d0a": prepared_config[key],
                     "equal": source_config[key] == prepared_config[key]} for key in shared}
    differences = {"num_workers": {"source": source_config["num_workers"], "d0a": prepared_config["num_workers"]},
                   "trainer_sha256": {"source": source_contract["runtime_sha256"]["train_fixed_split.py"],
                                       "d0a": prepared_contract["runtime_sha256"]["train_fixed_split.py"]}}
    return {"ready": True, "output": output, "config": config, "source": row, "metrics": metrics,
            "baselines": baselines, "matched": matched, "differences": differences,
            "bindings": common.runtime_bindings(config_path, paths, [])}


def markdown_table(title: str, rows: list[tuple[str, Mapping[str, float]]]) -> str:
    lines = [f"## {title}", "", "| 权重 | mIoU (%) ↑ | PD (%) ↑ | Fa (×10⁻⁶) ↓ |",
             "|---|---:|---:|---:|"]
    for name, value in rows:
        lines.append(f"| {name} | {value['miou']*100:.6f} | {value['pd']*100:.6f} | {value[NAMES[2]]:.6f} |")
    return "\n".join(lines)


def execute(config_path: Path = common.DEFAULT_CONFIG) -> dict[str, Any]:
    context = preflight(config_path)
    output = context["output"]
    common.reserve_output(output)
    role = {"stage": "D0-A", "role": "prepared_source_clean_development_evaluation",
            "tta_executed": False, "development_only": True, "paper_result": False,
            "historical_test_metrics_reused": True, "train_side_evidence": False,
            "new_test_image_opens": 0, "new_test_mask_opens": 0,
            "validation_payload_opens": 0, "authorizes_tuning": False,
            "new_training_authorized": False}
    common.freeze_run(output, {**role, "protocol": context["config"],
        "method_display_name": DISPLAY_NAME, "schema_mapping": {"original_tta": "tta_executed"}}, context["bindings"])
    candidate = context["metrics"]["official_reported_operating_point"]
    fixed = compare(candidate, context["source"])
    fronts = {axis: compare(candidate, item["official_reported_operating_point"])
              for axis, item in context["baselines"].items()}
    pixel_evidence = {}
    for axis, item in context["baselines"].items():
        ref = item["unified"]["fixed"]["pixel"]
        current = context["metrics"]["unified"]["fixed"]["pixel"]
        pixel_evidence[axis] = {key: {"source": ref[key], "d0a": current[key], "delta": current[key]-ref[key]}
            for key in ("predicted_positive_pixels", "false_positive_pixels", "false_negative_pixels")}
    summary = {**role, "dataset": "NUDT-SIRST", "fixed_endpoint_effect": fixed,
        "test_selected_front_reference": fronts,
        "protocol_comparison": {"matched_settings": context["matched"], "differences": context["differences"],
            "causal_objective_effect_identified": False,
            "limitation": "Same endpoint is not a paired single-variable ablation; worker random streams and historical trainer hashes differ."},
        "existing_aggregate_pixel_evidence": pixel_evidence,
        "interpretation": "Fixed-endpoint trade-off, not all-three harm. No Pareto improvement over the existing best_miou comparator. Mechanism remains unproven."}
    receipt = {**role, "method_display_name": DISPLAY_NAME,
        "checkpoint_rule": "fixed_epoch1000_train_only",
        "original_evaluation_complete": common.binding(EVALUATION / "COMPLETE.json"),
        "original_evaluation_manifest": common.binding(EVALUATION / "artifact_manifest.json"),
        "original_metrics": common.binding(EVALUATION / "metrics.json"),
        "posthoc_interpretation_only": True, "original_sealed_artifacts_modified": False,
        "scientific_interpretation": "clean_pareto_not_improved",
        "does_not_authorize": ["formal_test", "checkpoint_selection", "hyperparameter_tuning", "full_training", "IPMA", "D0-B"]}
    common.write_json_new(output / "CLEAN_SAFETY_DEVELOPMENT_RECEIPT.json", receipt)
    text = "# D0-A prepared source：已有 clean 结果的派生诊断\n\n"
    text += "仅重用已有 NUDT fixed-test 日志，无新测试推理，无 TTA；不是训练侧独立证据。原封存结果不变。\n\n"
    text += markdown_table("A. 固定终点比较", [("NS-FPN clean-only epoch1000", fixed["reference"]), (DISPLAY_NAME, candidate)])
    d = fixed["delta_percentage_points"]
    text += f"\n\nΔmIoU={d['miou']:+.6f} 个百分点；ΔPD={d['pd']:+.6f} 个百分点；ΔFa={fixed['delta'][NAMES[2]]:+.6f}（{fixed['relative_fa_change_percent']:+.4f}%）。\n"
    text += "E0 三指标全部恶化条件未触发。这是 mixed trade-off，不是 objective 的因果隔离结论。\n\n"
    text += markdown_table("B. 已有 test-selected 开发前沿参考", [(f"NS-FPN {axis}", x["reference"]) for axis, x in fronts.items()] + [(DISPLAY_NAME, candidate)])
    text += "\n\n两份 baseline 由重复 test 选出；D0-A 固定 epoch1000。不能混称为同选择规则的方法提升表。\n\n"
    text += "## 解释边界\n\n训练 workers 为 12 与 8，历史 trainer 哈希不同；seed 相同不代表增强随机流相同。\n\n"
    text += "相对 best_miou，预测前景总像素 28498→27656，像素 FP 3210→2932，FN 3027→3591。Fa 增加不能直接解释为全局前景膨胀；目标匹配 Fa 与像素 FP 也不能混用。尚未执行逐图归因或置信区间分析。\n\n"
    text += "下一步仅允许独立 P2/P3 train Pilot64 诊断；不会自动恢复全训练、启动 IPMA 或绕过 D0-B。\n"
    with (output / "REPORT.md").open("x", encoding="utf-8") as handle:
        handle.write(text)
    common.complete_run(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        result = execute(args.config)
        print(json.dumps({"complete": True, "fixed_endpoint": result["fixed_endpoint_effect"]}))
    else:
        result = preflight(args.config)
        print(json.dumps({"ready": result["ready"], "writes": 0, "output_exists": result["output"].exists()}))


if __name__ == "__main__":
    main()
