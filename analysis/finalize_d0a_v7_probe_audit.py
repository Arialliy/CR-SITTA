"""Finalize completed P2 numerical records after the UTF-8 metadata-read error.

No images are decoded, no probes are generated, and no original file is
overwritten. A separately bound recovery receipt records this metadata-only
continuation. Run with PYTHONUTF8=1 to protect the frozen legacy helper's reads.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import d0a_v7_common as common
from analysis import audit_d0a_probe_label_preservation as audit

DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
LOG_TAGS = {"IRSTD-1K": "irstd", "NUAA-SIRST": "nuaa", "NUDT-SIRST": "nudt"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_records(image_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]],
                     dataset: str, image_ids: list[str]) -> None:
    expected = {(identifier, view, probe) for identifier in image_ids
                for view in audit.VIEWS for probe in audit.PROBES}
    keys = [(row["image_id"], row["view"], row["probe_id"]) for row in image_rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("incomplete or duplicate per-image records")
    images = dict(zip(keys, image_rows))
    grouped = defaultdict(list)
    for row in target_rows:
        key = (row["image_id"], row["view"], row["probe_id"])
        if row["dataset"] != dataset or key not in expected:
            raise ValueError("foreign per-target record")
        grouped[key].append(row)
    for key, row in images.items():
        if row["dataset"] != dataset:
            raise ValueError("foreign per-image record")
        if not all(row.get(name) is True for name in (
            "gt_tensor_unchanged", "deterministic_repeat_exact", "original_runner_parity_exact", "preclip_clamp_parity_exact")):
            raise ValueError("original numerical audit did not pass its checks")
        targets = grouped[key]
        if len(targets) != row["target_count"] or sorted(r["target_id"] for r in targets) != list(range(1, len(targets)+1)):
            raise ValueError("incomplete or duplicate target-component records")
    for identifier in image_ids:
        for view in audit.VIEWS:
            records = [images[(identifier, view, probe)] for probe in audit.PROBES]
            if len({r["target_count"] for r in records}) != 1:
                raise ValueError("GT component counts differ across probes")
            metadata = [r["input_metadata"] for r in records]
            if not all(value == metadata[0] for value in metadata):
                raise ValueError("input crop or GT changed across probes")


def prepare(dataset: str, config_path: Path) -> dict[str, Any]:
    config = common.read_config(config_path)
    root = common.absolute(config["result_root"])
    output = root / "probe_label_preservation" / dataset
    for name in ("summary.json", "artifact_manifest.json", "COMPLETE.json", "FINALIZATION_RECOVERY.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing existing finalization artifact: {output/name}")
    contract = read_json(output / "RUN_CONTRACT.json")
    freeze = read_json(output / "PRE_RUN_FREEZE.json")
    for item in freeze["input_bindings"] + [freeze["contract"]]:
        if common.sha256_file(item["path"]) != item["sha256"]:
            raise ValueError(f"original frozen input changed: {item['path']}")
    config_binding = next(item for item in freeze["input_bindings"] if item["path"] == str(config_path.resolve()))
    if config_binding["sha256"] != common.sha256_file(config_path):
        raise ValueError("configuration differs from original numerical run")
    image_rows = [json.loads(line) for line in (output / "per_image.jsonl").read_text(encoding="utf-8").splitlines()]
    target_rows = [json.loads(line) for line in (output / "per_target.jsonl").read_text(encoding="utf-8").splitlines()]
    pilot = common.load_pilot_records(dataset, config)
    if len(pilot) != 64 or contract["pilot_images"] != 64 or contract["expected_per_image_rows"] != 384:
        raise ValueError("unexpected original diagnostic scope")
    validate_records(image_rows, target_rows, dataset, [r["image_id"] for r in pilot])
    log_path = root / f"p2_{LOG_TAGS[dataset]}.log"
    log = log_path.read_text(encoding="utf-8")
    if "UnicodeDecodeError: 'ascii'" not in log or '"images_completed": 64' not in log:
        raise ValueError("log does not prove the expected post-computation encoding failure")
    return locals()


def finalize(dataset: str, config_path: Path = common.DEFAULT_CONFIG) -> dict[str, Any]:
    if not sys.flags.utf8_mode:
        raise RuntimeError("run finalizer with PYTHONUTF8=1; do not alter frozen original helpers")
    context = prepare(dataset, config_path)
    config, output, contract = (context[name] for name in ("config", "output", "contract"))
    image_rows, target_rows = context["image_rows"], context["target_rows"]
    grouped = defaultdict(list)
    for row in target_rows:
        grouped[(row["view"], row["probe_id"])].append(row)
    cells = []
    for view in audit.VIEWS:
        for probe_id in audit.PROBES:
            images = [r for r in image_rows if r["view"] == view and r["probe_id"] == probe_id]
            cells.append({"view": view, "probe_id": probe_id, "image_count": len(images),
                "empty_target_view_count": sum(r["empty_target_view"] for r in images),
                "image_statistics": audit.summarize_images(images),
                **audit.summarize_targets(grouped[(view, probe_id)], config["visibility"])})
    access = {key: 0 for key in audit.FORBIDDEN_ACCESS_FIELDS}
    access.update({"train_image_opens": 128, "train_mask_opens": 128})
    audit.validate_access_counts(access, 64)
    receipt = {
        "schema_version": 1, "type": "metadata_only_completion_recovery", "dataset": dataset,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "failure": "UnicodeDecodeError reading UTF-8 PRE_RUN_FREEZE.json with default ASCII encoding after all numerical records were written",
        "finalizer": common.binding(Path(__file__)),
        "original_bindings": [common.binding(output / name) for name in (
            "RUN_CONTRACT.json", "PRE_RUN_FREEZE.json", "per_image.jsonl", "per_target.jsonl")],
        "failure_log": common.binding(context["log_path"]),
        "numerical_aggregators": common.binding(Path(audit.__file__)),
        "original_numerical_records_overwritten": False, "original_code_modified": False,
        "new_image_decodes": 0, "new_mask_decodes": 0, "probe_generations": 0, "model_forwards": 0,
        "access_counts_evidence": "Reconstructed from the original runner's successful validate_access_counts assertion, executed before both numerical JSONL files were published; the original in-memory counter object was not separately persisted.",
        "python_utf8_mode": bool(sys.flags.utf8_mode), "paper_result": False,
    }
    common.write_json_new(output / "FINALIZATION_RECOVERY.json", receipt)
    summary = {"diagnostic_id": contract["diagnostic_id"], "dataset": dataset,
        "role": contract["role"], "pilot_images": 64,
        "development_only": True, "paper_result": False, "no_validation_split": True,
        "per_image_rows": len(image_rows), "per_target_rows": len(target_rows),
        "cells": cells, "access": access,
        "access_counts_evidence": receipt["access_counts_evidence"],
        "all_probe_replays_exact": True, "all_original_runner_parity_exact": True,
        "all_gt_unchanged": True, "label_semantics_changed_proven": False,
        "training_started": False, "test_payload_opens": 0,
        "completion_recovery": "FINALIZATION_RECOVERY.json",
        "e1_visibility_risk_triggered": any(r["e1_visibility_risk_triggered"] is True for r in cells if r["probe_id"] != "clean")}
    common.complete_run(output, summary)
    return {"dataset": dataset, "complete": True, "output": str(output),
            "per_image_rows": len(image_rows), "per_target_rows": len(target_rows),
            "e1_visibility_risk_triggered": summary["e1_visibility_risk_triggered"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--config", type=Path, default=common.DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for dataset in args.dataset:
        if args.execute:
            print(json.dumps(finalize(dataset, args.config), ensure_ascii=False))
        else:
            prepare(dataset, args.config)
            print(json.dumps({"dataset": dataset, "ready": True, "writes": 0}))


if __name__ == "__main__":
    main()
