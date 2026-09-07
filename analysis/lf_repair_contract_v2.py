"""Append-only, train-allowlisted contracts for v9 LF R0--R3 only.

Old v7 helpers/config remain unchanged. Metadata preflight hashes only the
allowlisted train payloads; it does not decode images or open test split files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from analysis import d0a_v7_common as legacy

ROOT = legacy.ROOT
DEFAULT_CONFIG = ROOT / "configs/cr_sitta_lf_repair_audit_v2.yaml"
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
VIEWS = ("full_256", "train_crop_224")
VARIANTS = [
    {"probe_id": "L1", "attenuation": 1.0, "protect_dc": False, "shared_channels": False},
    {"probe_id": "L2", "attenuation": 1.0, "protect_dc": True, "shared_channels": False},
    {"probe_id": "L3", "attenuation": 1.0, "protect_dc": True, "shared_channels": True},
    {"probe_id": "L4a", "attenuation": 0.25, "protect_dc": True, "shared_channels": True},
    {"probe_id": "L4b", "attenuation": 0.50, "protect_dc": True, "shared_channels": True},
]
NEW_SOURCES = (
    "analysis/lf_repair_contract_v2.py", "analysis/audit_lf_repair_v2.py",
    "analysis/finalize_lf_repair_v2.py", "tta/deteriorations/fourier_low_mask_v2.py",
    "tests/test_lf_repair_contract_v2.py", "tests/test_lf_repair_v2.py",
    "tests/test_audit_lf_repair_v2.py", "tests/test_lf_repair_gates_v2.py",
)
PROTECTED_SOURCES = (
    "analysis/compare_d0a_fixed_endpoint.py", "analysis/audit_d0a_probe_label_preservation.py",
    "analysis/audit_d0a_branch_gradients.py", "analysis/d0a_v7_common.py",
    "configs/cr_sitta_d0a_v7_diagnostics_v1.yaml", "tta/deteriorations/fourier_low_mask.py",
    "tta/deteriorations/__init__.py", "train_cr_sitta_d0a.py",
)


def _nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def read_json(path: str | Path) -> Any:
    return json.loads(legacy.absolute(path).read_text(encoding="utf-8"), parse_constant=_nonfinite)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with legacy.absolute(path).open(encoding="utf-8") as handle:
        return [json.loads(line, parse_constant=_nonfinite) for line in handle if line.strip()]


def _expect(value: Any, expected: Any, label: str) -> None:
    # JSON equality rejects bool-as-number and pins the exact declared shape.
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"unregistered LF protocol change: {label}")


def read_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(legacy.absolute(path).read_text(encoding="utf-8"))
    fixed = {
        "schema_version": 1, "protocol_id": "cr-sitta-lf-repair-audit-v2",
        "design_document": "CR-SITTA_P0-P3诊断后的LF修复与IPMA微型验证_v9.md",
        "implementation_addendum": "configs/cr_sitta_lf_repair_audit_v2_preregistration.md",
        "parent_code_anchor": "9bccebb92aaabcd4c97e1556de96d1a643af69b8",
        "parent_execution_receipt_sha256": "96a76ecff017e81b8bf76e69401267d7ff91faf12578c076b84beebdb592e340",
        "parent_manifests": {
            "fixed_endpoint/NUDT-SIRST": "9a80f082202090acfaaf6b29f7c42ecaf784088ba26a0712040d489b153cee97",
            "probe_label_preservation/IRSTD-1K": "94ce20b2a3e6ff3adb49134371b1f11d682396bd73d9bc849c8dccd1a0bbb7e8",
            "probe_label_preservation/NUAA-SIRST": "1065264aa82dd042ed7e6d878856cefbd65a993bab98986797447bc0aa3a7313",
            "probe_label_preservation/NUDT-SIRST": "4916e556584fb06acb2e498b7c19768fe95623b7e7f95388da234b121fdd5f0c",
            "branch_gradients/NUDT-SIRST": "6882e8a1903e419748ab143b4900ca974c267695337eb8cfe62e39d881844027",
        },
        "legacy_config": "configs/cr_sitta_d0a_v7_diagnostics_v1.yaml",
        "legacy_config_sha256": "f68ad21683b811e46bf864eb8293419db2227eae3e5a759e6815f586009e2844",
        "result_root": "results/cr_sitta/lf_repair_audit_v2",
        "parent_result_root": "results/cr_sitta/d0a_v7_diagnostics_v1",
        "datasets": list(DATASETS), "views": list(VIEWS), "pilot_images_per_dataset": 64,
        "device": "cpu", "threads": 2, "global_seed": 42,
        "probe_seed_namespace": "cr-sitta-lf-repair-audit-v2-paired-orbits",
        "operator": {"mask_ratio": 0.20, "pair_keep_probability": 0.50}, "variants": VARIANTS,
        "scientific_scope": "augmentation_visibility_screen_only",
        "stage_scope": {key: False for key in (
            "full_source_training_allowed", "ipma_meta_training_allowed", "formal_test_allowed",
            "new_test_payload_access", "new_validation_split", "paper_result")},
        "gate": {"candidates": ["L4a", "L4b"], "risk_fraction_max": 0.10,
            "compound_denominator": "original_e1_eligible", "image_rms_floor": 1.0e-6,
            "image_rms_fraction_min": 0.90,
            "image_rms_aggregation": "each_dataset_view_median_and_coverage",
            "empty_eligible_status": "insufficient_evidence",
            "constant_images": "retain_in_denominator_and_report_noninformative"},
        "selection": {"preferred": "L4a", "r3_selected_probe_must_be": "L4a",
            "stronger_only_pass": "alternate_eligible_no_automatic_promotion",
            "l4b_promotion_requires": "separate_unlabeled_teacher_signal_receipt_not_implemented_in_R3"},
        "bootstrap": {"replicates": 2000, "seed": 42, "confidence_level": 0.95,
            "unit": "source_image_id", "interval": "percentile",
            "paired_to": "sealed_L0_same_image_and_view",
            "zero_denominator_replicates": "omit_and_report_count_not_fill_zero",
            "interval_used_for_gate": False},
        "random_field": {"shape": "batch_physical_channels_all_orbits_including_dc",
            "dtype": "float64", "device": "cpu", "orbit_order": "ascending_canonical_flat_index",
            "shared_channel_selection": "channel_0", "candidate_id_in_seed": False,
            "include_view_in_seed": True, "seed_replicates": 1,
            "legacy_new_frequency_masks_identical_claimed": False},
        "evaluation": {"visibility_definition": "unchanged_v7_config",
            "target_components": "keep_all_including_low_contrast_invalid_ring_and_empty_crops",
            "mask_rule": "GT_gt_0", "detection_threshold_reserved": "strict_sigmoid_gt_0.5",
            "detector_forward_enabled": False, "detector_backward_enabled": False,
            "checkpoint_loaded": False, "legacy_L0": "sealed_records_read_only_no_rerun"},
    }
    if set(config) != set(fixed):
        raise ValueError("unknown or missing LF protocol keys")
    for key, value in fixed.items():
        _expect(config.get(key), value, key)
    if legacy.sha256_file(config["legacy_config"]) != config["legacy_config_sha256"]:
        raise ValueError("old diagnostic config drift")
    return config


def assert_bindings(bindings: list[dict[str, str]]) -> None:
    for item in bindings:
        if legacy.sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {item['path']}")


def verify_complete_dir(output: str | Path, expected_manifest_sha: str | None = None) -> dict[str, Any]:
    output = legacy.absolute(output)
    complete = read_json(output / "COMPLETE.json")
    if complete.get("complete") is not True or complete.get("paper_result") is not False:
        raise ValueError("missing complete development-only receipt")
    manifest_path = output / "artifact_manifest.json"
    manifest_hash = legacy.sha256_file(manifest_path)
    if expected_manifest_sha is not None and manifest_hash != expected_manifest_sha:
        raise ValueError("parent artifact manifest drift")
    manifest_binding = complete.get("artifact_manifest", {})
    if manifest_binding.get("sha256") != manifest_hash or legacy.absolute(manifest_binding["path"]) != manifest_path:
        raise ValueError("COMPLETE does not bind local manifest")
    manifest = read_json(manifest_path)
    files = manifest.get("files", {})
    if not {"RUN_CONTRACT.json", "PRE_RUN_FREEZE.json", "summary.json"} <= set(files):
        raise ValueError("required frozen artifacts missing")
    bindings = [legacy.binding(output / "COMPLETE.json"), legacy.binding(manifest_path)]
    for relative, digest in files.items():
        path = output / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink() or not path.resolve().is_relative_to(output):
            raise ValueError("artifact escapes sealed output directory")
        if legacy.sha256_file(path) != digest:
            raise ValueError(f"sealed artifact changed: {path}")
        bindings.append({"path": str(path), "sha256": digest})
    return {"complete": complete, "manifest": manifest, "bindings": bindings,
            "summary": read_json(output / "summary.json")}


def parent_evidence(config: dict[str, Any]) -> dict[str, Any]:
    parent = legacy.absolute(config["parent_result_root"])
    expected_names = {"fixed_endpoint/NUDT-SIRST", "branch_gradients/NUDT-SIRST",
                      *(f"probe_label_preservation/{d}" for d in DATASETS)}
    if set(config["parent_manifests"]) != expected_names:
        raise ValueError("all five old diagnostic manifests are required")
    receipt = legacy.binding(parent / "EXECUTION_RECEIPT.json")
    if receipt["sha256"] != config["parent_execution_receipt_sha256"]:
        raise ValueError("parent execution receipt drift")
    bindings = [receipt]
    frozen_old: dict[str, str] = {}
    parents = []
    for name, digest in config["parent_manifests"].items():
        verified = verify_complete_dir(parent / name, digest)
        bindings.extend(verified["bindings"])
        freeze = read_json(parent / name / "PRE_RUN_FREEZE.json")
        assert_bindings([freeze["contract"]])
        for item in freeze["input_bindings"]:
            p = legacy.absolute(item["path"])
            if p.is_relative_to(ROOT) and str(p.relative_to(ROOT)) in PROTECTED_SOURCES:
                if str(p) in frozen_old and frozen_old[str(p)] != item["sha256"]:
                    raise ValueError("old diagnostics disagree on a protected source")
                frozen_old[str(p)] = item["sha256"]
        parents.append({"relative_path": name, "manifest_sha256": digest})
    protected = [{"path": p, "sha256": s} for p, s in frozen_old.items()]
    assert_bindings(protected)
    bindings.extend(protected)
    # The package export may not have been included in an old freeze. The
    # archived user commit still binds it without modifying the working tree.
    anchor = config["parent_code_anchor"]
    for relative in PROTECTED_SOURCES:
        archived = subprocess.run(["git", "show", f"{anchor}:{relative}"], cwd=ROOT,
                                  check=True, capture_output=True).stdout
        if hashlib.sha256(archived).hexdigest() != legacy.sha256_file(ROOT / relative):
            raise ValueError(f"protected source differs from parent commit: {relative}")
        bindings.append(legacy.binding(ROOT / relative))
    return {"schema_version": 1, "parent_code_anchor": anchor,
            "five_complete_manifests_verified": True, "parents": parents,
            "old_sources_unchanged": True, "old_results_rerun": False,
            "new_test_payload_access": False, "input_bindings": deduplicate(bindings)}


def deduplicate(bindings: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = {}
    for item in bindings:
        if item["path"] in seen and seen[item["path"]] != item["sha256"]:
            raise ValueError("conflicting file bindings")
        seen[item["path"]] = item["sha256"]
    return [{"path": p, "sha256": seen[p]} for p in sorted(seen)]


def assert_pilot_raw_bindings(config: dict[str, Any], records: list[dict[str, str]]) -> None:
    """Compare only explicit train PNGs against their original P2 freeze."""
    by_dataset: dict[str, dict[str, str]] = {}
    for record in records:
        dataset = record["dataset"]
        if dataset not in by_dataset:
            parent = legacy.absolute(config["parent_result_root"]) / "probe_label_preservation" / dataset
            by_dataset[dataset] = {item["path"]: item["sha256"]
                                  for item in read_json(parent / "PRE_RUN_FREEZE.json")["input_bindings"]}
        for field in ("image_path", "mask_path"):
            path = str(legacy.absolute(record[field]))
            if path not in by_dataset[dataset] or legacy.sha256_file(path) != by_dataset[dataset][path]:
                raise ValueError(f"train Pilot64 raw payload differs from sealed P2: {path}")


def initialize(config_path: str | Path = DEFAULT_CONFIG, *, execute: bool = False) -> dict[str, Any]:
    config_path = legacy.absolute(config_path)
    config = read_config(config_path)
    old = legacy.read_config(config["legacy_config"])
    evidence = parent_evidence(config)
    records = [record for dataset in DATASETS for record in legacy.load_pilot_records(dataset, old)]
    assert_pilot_raw_bindings(config, records)
    extra = [config_path, config["design_document"], config["implementation_addendum"], *NEW_SOURCES]
    bindings = legacy.runtime_bindings(config["legacy_config"], extra, records)
    bindings += evidence["input_bindings"]
    # Bind the future host axis, without loading weights or doing inference.
    for key in ("checkpoint", "run_contract"):
        item = legacy.binding(old["gradient"][key])
        if item["sha256"] != old["gradient"][f"{key}_sha256"]:
            raise ValueError(f"fixed D0-A host {key} drift")
        bindings.append(item)
    output = legacy.absolute(config["result_root"])
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                check=True, text=True, capture_output=True).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT,
                                check=True, text=True, capture_output=True).stdout.splitlines()
    contract = {"protocol_id": config["protocol_id"], "role": "source_train_pilot_visibility_only",
        "output_dir": str(output), "code_commit": git_commit, "git_status_at_freeze": git_status,
        "uncommitted_new_sources_bound_individually": True,
        "pilot_images": len(records), "train_payload_files_hashed": len(records) * 2,
        "train_payload_image_decodes": 0, "test_payload_opens": 0,
        "test_split_reads": 0, "validation_payload_opens": 0,
        "scientific_scope": config["scientific_scope"], "stage_scope": config["stage_scope"],
        "new_training_authorized": False, "paper_result": False,
        "frozen_before_new_visibility_results": True,
        "input_bindings": deduplicate(bindings)}
    preflight = {"ready": not output.exists(), "execute": execute, "writes": 0,
                 "output_dir": str(output), "input_bindings": len(contract["input_bindings"]),
                 "train_payload_files_hashed": len(records) * 2,
                 "train_payload_image_decodes": 0, "test_payload_opens": 0,
                 "five_complete_manifests_verified": True}
    if not execute:
        return preflight
    legacy.reserve_output(output)
    evidence["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    legacy.write_json_new(output / "P0_P3_EVIDENCE_BINDING.json", evidence)
    contract["input_bindings"].append(legacy.binding(output / "P0_P3_EVIDENCE_BINDING.json"))
    contract["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    legacy.write_json_new(output / "PREREGISTRATION.json", contract)
    for name in ("P0_P3_EVIDENCE_BINDING.json", "PREREGISTRATION.json"):
        (output / name).chmod(0o444)
    return {**preflight, "writes": 2, "preregistered": True}


def validate_preregistration(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_config(config_path)
    path = legacy.absolute(config["result_root"]) / "PREREGISTRATION.json"
    contract = read_json(path)
    if contract.get("protocol_id") != config["protocol_id"] or contract.get("frozen_before_new_visibility_results") is not True:
        raise ValueError("missing LF preregistration")
    _expect(contract.get("stage_scope"), config["stage_scope"], "preregistration stage scope")
    assert_bindings(contract["input_bindings"])
    config_binding = legacy.binding(config_path)
    if config_binding not in contract["input_bindings"]:
        raise ValueError("unbound LF config")
    for source in NEW_SOURCES:
        if legacy.binding(source) not in contract["input_bindings"]:
            raise ValueError(f"unbound LF source: {source}")
    return contract


def prepare_run(config_path: str | Path, dataset: str) -> dict[str, Any]:
    config_path = legacy.absolute(config_path)
    config = read_config(config_path)
    if dataset not in DATASETS:
        raise ValueError("unknown train dataset")
    old = legacy.read_config(config["legacy_config"])
    from analysis.audit_d0a_probe_label_preservation import validate_diagnostic_parameters
    validate_diagnostic_parameters(old)
    records = legacy.load_pilot_records(dataset, old)
    evidence = parent_evidence(config)
    assert_pilot_raw_bindings(config, records)
    root = legacy.absolute(config["result_root"])
    preregistered = (root / "PREREGISTRATION.json").exists()
    bindings = evidence["input_bindings"]
    if preregistered:
        prereg = validate_preregistration(config_path)
        bindings += prereg["input_bindings"] + [legacy.binding(root / "PREREGISTRATION.json")]
    else:
        bindings += legacy.runtime_bindings(config["legacy_config"],
            [config_path, config["design_document"], config["implementation_addendum"], *NEW_SOURCES], records)
    parent = legacy.absolute(config["parent_result_root"]) / "probe_label_preservation" / dataset
    output = root / dataset
    preflight = {"ready": preregistered and not output.exists(), "writes": 0, "dataset": dataset,
        "preregistered": preregistered, "output_dir": str(output), "device": "cpu",
        "pilot_images": 64, "views": list(VIEWS), "variants": ["clean", *[v["probe_id"] for v in VARIANTS]],
        "expected_per_image_rows": 64 * 2 * 6, "test_payload_opens": 0,
        "train_payload_image_decodes": 0, "checkpoint_loaded": False,
        "training_enabled": False, "input_bindings": len(deduplicate(bindings))}
    contract = {**preflight, "protocol_id": config["protocol_id"], "dataset": dataset,
        "no_validation_split": True, "development_only": True, "paper_result": False,
        "scientific_scope": config["scientific_scope"], "stage_scope": config["stage_scope"],
        "operator": config["operator"], "variants": config["variants"],
        "gate": config["gate"], "bootstrap": config["bootstrap"], "random_field": config["random_field"],
        "visibility": old["visibility"], "checkpoint_used": None,
        "legacy_L0": "sealed_records_read_only", "historical_training_random_stream_replay": False,
        "data_access_note": "hash reads and decoded payload opens counted separately; allowlisted train only"}
    return {"new_config": config, "legacy_config": old, "config_path": config_path,
        "records": records, "legacy_image_rows": read_jsonl(parent / "per_image.jsonl"),
        "legacy_target_rows": read_jsonl(parent / "per_target.jsonl"), "output": output,
        "bindings": deduplicate(bindings), "preflight": preflight, "contract": contract}


def freeze_dataset(prepared: dict[str, Any]) -> None:
    validate_preregistration(prepared["config_path"])
    if prepared["output"] != legacy.absolute(prepared["new_config"]["result_root"]) / prepared["contract"]["dataset"]:
        raise ValueError("unexpected dataset output target")
    assert_bindings(prepared["bindings"])
    legacy.reserve_output(prepared["output"])
    legacy.freeze_run(prepared["output"], prepared["contract"], prepared["bindings"])


def complete_output(output: str | Path, summary: dict[str, Any]) -> None:
    output = legacy.absolute(output)
    if (output / "COMPLETE.json").exists() or (output / "summary.json").exists():
        raise FileExistsError("refusing to overwrite LF completion")
    freeze = read_json(output / "PRE_RUN_FREEZE.json")
    assert_bindings(freeze["input_bindings"] + [freeze["contract"]])
    legacy.write_json_new(output / "summary.json", summary)
    files = {str(p.relative_to(output)): legacy.sha256_file(p)
             for p in sorted(output.rglob("*")) if p.is_file()}
    legacy.write_json_new(output / "artifact_manifest.json", {"schema_version": 1, "files": files})
    legacy.write_json_new(output / "COMPLETE.json", {"schema_version": 1, "complete": True,
        "development_only": True, "paper_result": False, "new_training_authorized": False,
        "artifact_manifest": legacy.binding(output / "artifact_manifest.json"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})


def complete_dataset(prepared: dict[str, Any], summary: dict[str, Any]) -> None:
    validate_preregistration(prepared["config_path"])
    complete_output(prepared["output"], summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(initialize(args.config, execute=args.execute), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
