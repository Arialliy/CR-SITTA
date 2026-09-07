"""Metadata-only, append-only contracts for the frozen v9 R4 engineering run.

This module does not import torch, load a checkpoint, decode an image, consult
a test split, or initialize CUDA.  Parent train byte hashes may be verified;
that is distinct from decoding or using a held-out train mask.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from analysis import d0a_v7_common as legacy
from analysis import lf_repair_contract_v2 as lf

ROOT = legacy.ROOT
DEFAULT_CONFIG = ROOT / "configs/cr_sitta_ipma_micro16_v1.yaml"
CONFIG_SHA256 = "7dfd1618e13c28e498e52bfc1622c85c94782b1a6d8459983824c5e136fe9538"
OUTPUT_RELATIVE = "results/cr_sitta/ipma_micro16_v1/engineering"
NEW_SOURCES = (
    "model/ipma_d0_adapter_v1.py", "training/ipma_inner_v1.py",
    "analysis/ipma_micro16_data_v1.py", "analysis/ipma_engineering_checks_v1.py",
    "analysis/ipma_micro16_contract_v1.py", "analysis/audit_ipma_micro16_v1.py",
    "tests/test_ipma_d0_identity_v1.py", "tests/test_ipma_inner_meta_equivalence_v1.py",
    "tests/test_ipma_micro16_data_v1.py", "tests/test_ipma_engineering_checks_v1.py",
    "tests/test_ipma_micro16_contract_v1.py", "tests/test_audit_ipma_micro16_v1.py",
)
EXTRA_DEPENDENCIES = (
    "analysis/d0a_v7_common.py", "analysis/lf_repair_contract_v2.py",
    "analysis/audit_lf_repair_v2.py", "tta/deteriorations/fourier_low_mask_v2.py",
    "tta/deteriorations/image_space.py", "train_fixed_split.py", "model/loss.py",
    "metrics/irstd_metrics.py", "metrics/connected_components.py", "metrics/target_matching.py",
    "metrics/target_transitions.py", "export_cr_sitta_d0a_safe_checkpoint.py",
    "model/MSHNet_NSFPN.py", "model/NS_FPN.py", "metrics/official_metric_adapter.py",
    "utils/metric.py", "tta/deteriorations/__init__.py",
    "SFS_MSDeformAttn/ops/modules/__init__.py", "SFS_MSDeformAttn/ops/functions/__init__.py",
    "SFS_MSDeformAttn/ops/__init__.py", "metrics/__init__.py",
)
NO_AUTHORIZATION = ("ipma_meta_training_allowed", "full_source_training_allowed",
                    "formal_test_allowed", "paper_result")


def _same(value: Any, expected: Any, name: str) -> None:
    if json.dumps(value, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"R4 frozen contract mismatch: {name}")


def read_config(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(config_path)
    path = path if path.is_absolute() else ROOT / path
    # Reject unknown paths before opening or hashing them (especially test).
    if path != DEFAULT_CONFIG or path.is_symlink():
        raise ValueError("R4 only accepts the canonical preregistered config path")
    if legacy.sha256_file(path) != CONFIG_SHA256:
        raise ValueError("R4 configuration byte hash differs from preregistration")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _same(config["protocol_id"], "cr-sitta-ipma-micro16-v1", "protocol")
    _same(config["stage"], "R4_engineering_only", "stage")
    _same(config["result_root"], OUTPUT_RELATIVE, "output")
    for key in NO_AUTHORIZATION:
        _same(config["scope"].get(key), False, f"scope.{key}")
    _same(config["scope"].get("outer_optimizer_steps"), 0, "outer optimizer steps")
    _same(config["sampling"].get("check_gt_decode_allowed"), False, "check GT decode")
    return config


def _bound(path: str | Path, expected: str, label: str) -> dict[str, str]:
    item = legacy.binding(path)
    if item["sha256"] != expected:
        raise ValueError(f"R4 {label} hash mismatch")
    return item


def _verify_freeze(output: Path) -> list[dict[str, str]]:
    freeze = lf.read_json(output / "PRE_RUN_FREEZE.json")
    bindings = freeze["input_bindings"] + [freeze["contract"]]
    lf.assert_bindings(bindings)
    return bindings


def validate_lf_approval(config: dict[str, Any], lf_config: dict[str, Any]) -> dict[str, Any]:
    root = legacy.absolute(lf_config["result_root"])
    finalization = root / "finalization"
    verified = lf.verify_complete_dir(finalization, config["lf_finalization_manifest_sha256"])
    bindings = verified["bindings"] + _verify_freeze(finalization)
    gate_path = finalization / "GATE_LF_RECEIPT.json"
    approved_path = finalization / "APPROVED_LF_CONFIG.json"
    bindings += [_bound(gate_path, config["lf_gate_sha256"], "LF gate"),
                 _bound(approved_path, config["lf_approved_config_sha256"], "approved LF")]
    gate, approved = lf.read_json(gate_path), lf.read_json(approved_path)
    _same(gate.get("selected_probe_id"), "L4a", "LF selected probe")
    _same(gate.get("signal_smoke_16_allowed"), True, "LF signal smoke authorization")
    _same(gate.get("scientific_scope"), "augmentation_visibility_screen_only", "LF scientific scope")
    _same(gate.get("approved_probe"), approved, "LF approved descriptor")
    _same(verified["summary"], gate, "LF summary/receipt")
    for key in NO_AUTHORIZATION:
        _same(gate.get(key), False, f"LF {key}")
        _same(approved.get(key), False, f"approved LF {key}")
    for key, expected in {"probe_id": "L4a", "mask_ratio": 0.2, "pair_keep_probability": 0.5,
                          "attenuation": 0.25, "protect_dc": True, "shared_channels": True,
                          "signal_smoke_16_allowed": True}.items():
        _same(approved.get(key), expected, f"approved LF {key}")
    bindings.append(_bound(ROOT / "tta/deteriorations/fourier_low_mask_v2.py",
                           approved["operator_source_sha256"], "approved LF implementation"))
    _same(gate["approved_operator_sha256"], approved["operator_source_sha256"], "LF operator source")
    return {"approved_lf": approved, "gate": gate, "bindings": bindings}


def validate_host_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Verify the weights-only export's existing attestation; never torch.load."""
    host = config["host"]
    bindings = [_bound(host[key], host[f"{key}_sha256"], f"host {key}")
                for key in ("source_checkpoint", "safe_checkpoint", "safe_export", "run_contract")]
    receipt = lf.read_json(host["safe_export"])
    _same(receipt.get("receipt_type"), "cr_sitta_d0a_safe_export_receipt_v1", "safe-export type")
    _same(receipt.get("status"), "published_weights_only_verified", "safe-export status")
    _same(receipt.get("dataset"), "NUDT-SIRST", "safe-export dataset")
    cc = receipt["checkpoint_contract"]
    for key, value in {"architecture": "MSHNet_NSFPN", "epoch": 1000, "state_dict_keys": 505,
                       "torch_load_weights_only": True, "all_state_dict_tensors_cpu": True,
                       "test_selected": False, "selection_rule": "fixed_final_epoch_train_only",
                       "repository_model_loads_verified": True}.items():
        _same(cc.get(key), value, f"safe-export {key}")
    for key in ("source_checkpoint", "safe_checkpoint", "run_contract"):
        item = receipt["artifacts"][key]
        _same(str(legacy.absolute(item["path"])), str(legacy.absolute(host[key])), f"safe-export {key} path")
        _same(item["sha256"], host[f"{key}_sha256"], f"safe-export {key} hash")
    for item in receipt["artifacts"].values():
        bindings.append(_bound(item["path"], item["sha256"], "safe-export provenance"))
    original = lf.read_json(host["run_contract"])
    original_config = original["run_config"]
    for key, value in {"dataset": "NUDT-SIRST", "epochs": 1000, "host_architecture": "MSHNet_NSFPN",
                       "checkpoint_selection": "fixed_final_epoch_train_only"}.items():
        _same(original_config.get(key), value, f"original host {key}")
    runtime = original["runtime_sha256"]
    if not runtime or not any(Path(path).name.startswith("MultiScaleDeformableAttention")
                              and Path(path).suffix == ".so" for path in runtime):
        raise ValueError("original runtime must bind the SFS extension bytes")
    for path, digest in runtime.items():
        bindings.append(_bound(path, digest, "original host runtime"))
    return {"bindings": bindings, "safe_export": receipt, "original_run_contract": original,
            "weights_loaded": False, "original_runtime_files_verified": len(runtime)}


def validate_replay_rows(config: dict[str, Any], records: list[dict[str, str]],
                         new_rows: list[dict[str, Any]], old_rows: list[dict[str, Any]],
                         approved: dict[str, Any]) -> list[dict[str, Any]]:
    ids = [record["image_id"] for record in records]
    _same(ids, config["sampling"]["ids"], "original Pilot64 prefix order")
    wanted = set(ids)
    def index(rows: list[dict[str, Any]], probe: str) -> dict[str, dict[str, Any]]:
        output = {}
        for row in rows:
            if row["dataset"] == "NUDT-SIRST" and row["view"] == "train_crop_224" and row["probe_id"] == probe and row["image_id"] in wanted:
                if row["image_id"] in output:
                    raise ValueError("duplicate frozen input pairing row")
                output[row["image_id"]] = row
        if set(output) != wanted:
            raise ValueError("missing fixed micro16 input pairing row")
        return output
    current, previous = index(new_rows, "L4a"), index(old_rows, "lf_mask")
    for identifier in ids:
        row, old = current[identifier], previous[identifier]
        _same(row["input_metadata"], old["input_metadata"], "P2/R2 input and GT metadata")
        _same(row["input_hash"], old["input_metadata"]["input_tensor_sha256"], "R2 input hash")
        _same(row["target_hash"], old["input_metadata"]["target_tensor_sha256"], "R2 GT hash")
        _same(row["operator_config_sha256"], approved["operator_config_sha256"], "R2 approved operator hash")
        _same(row["sealed_input_pair_exact"], True, "R2 exact pairing")
        _same(row["gt_tensor_unchanged"], True, "R2 GT unchanged")
        _same(row["target_count"], old["target_count"], "P2/R2 target count")
        for key in ("clean_physical_tensor_sha256", "preclip_physical_tensor_sha256",
                    "postclip_physical_tensor_sha256", "random_field_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"missing canonical R2 replay digest: {key}")
    return [current[identifier] for identifier in ids]


def prepare_run(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_config(config_path)
    lf_config = lf.read_config(config["lf_config"])
    preregistration = lf.validate_preregistration(config["lf_config"])
    approved = validate_lf_approval(config, lf_config)
    host = validate_host_metadata(config)
    old = legacy.read_config(lf_config["legacy_config"])
    records = legacy.load_pilot_records("NUDT-SIRST", old)[:16]
    _same([row["image_id"] for row in records], config["sampling"]["ids"], "Pilot64 first 16 IDs")
    lf.assert_pilot_raw_bindings(lf_config, records)
    lf_root = legacy.absolute(lf_config["result_root"])
    r2_output = lf_root / "NUDT-SIRST"
    r2 = lf.verify_complete_dir(r2_output, approved["gate"]["r2_dataset_manifest_sha256"]["NUDT-SIRST"])
    r2_bindings = r2["bindings"] + _verify_freeze(r2_output)
    p2_output = legacy.absolute(lf_config["parent_result_root"]) / "probe_label_preservation/NUDT-SIRST"
    p2 = lf.verify_complete_dir(p2_output, lf_config["parent_manifests"]["probe_label_preservation/NUDT-SIRST"])
    selected_rows = validate_replay_rows(config, records,
        lf.read_jsonl(r2_output / "per_image.jsonl"), lf.read_jsonl(p2_output / "per_image.jsonl"),
        approved["approved_lf"])
    paths = [DEFAULT_CONFIG, ROOT / config["design_document"], ROOT / config["addendum"],
             ROOT / config["lf_config"], *(ROOT / path for path in NEW_SOURCES + EXTRA_DEPENDENCIES)]
    bindings = [legacy.binding(path) for path in paths]
    bindings += preregistration["input_bindings"] + [legacy.binding(lf_root / "PREREGISTRATION.json")]
    bindings += approved["bindings"] + host["bindings"] + r2_bindings + p2["bindings"]
    bindings += [legacy.binding(record[key]) for record in records for key in ("image_path", "mask_path")]
    bindings = lf.deduplicate(bindings)
    output = ROOT / OUTPUT_RELATIVE
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                check=True, text=True, capture_output=True).stdout.strip()
    preflight = {"ready": not output.exists() and not output.is_symlink(), "writes": 0,
        "output_dir": str(output), "stage": "R4_engineering_only", "dataset": "NUDT-SIRST",
        "images_total": 16, "fit_images": 8, "check_images": 8,
        "image_decodes": 0, "mask_decodes": 0, "weights_loaded": False, "cuda_initialized": False,
        "test_split_reads": 0, "test_payload_opens": 0, "validation_payload_opens": 0,
        "new_train_files_hashed": 32, "inherited_parent_train_hash_checks": True,
        "input_bindings": len(bindings), "outer_optimizer_steps": 0,
        "approved_probe_id": "L4a", "formal_test_allowed": False}
    contract = {"protocol_id": config["protocol_id"], "stage": config["stage"],
        "role": "source_train_micro16_engineering_only_not_meta_training",
        "configuration": copy.deepcopy(config), "scope": dict(config["scope"]),
        "sampling": dict(config["sampling"]), "host": dict(config["host"]),
        "approved_lf": approved["approved_lf"], "runtime_requested": dict(config["runtime"]),
        "output_dir": str(output), "code_commit": git_commit,
        "config_binding": legacy.binding(DEFAULT_CONFIG), "new_sources_bound_individually": True,
        "input_replay_rows": selected_rows, "original_runtime_files_verified": host["original_runtime_files_verified"],
        "safe_export_weights_only_metadata_verified": True, "weights_loaded_before_freeze": False,
        "image_decodes_before_freeze": 0, "mask_decodes_before_freeze": 0,
        "train_raw_byte_hash_checks_are_not_gt_decodes": True,
        "no_validation_split": True, "development_only": True, "paper_result": False,
        "outer_optimizer_steps": 0, "new_training_authorized": False,
        "ipma_meta_training_allowed": False, "full_source_training_allowed": False,
        "formal_test_allowed": False, "r5_requires_separate_frozen_contract": True}
    return {"config_path": DEFAULT_CONFIG, "new_config": config, "legacy_config": old,
        "lf_config": lf_config, "approved_lf": approved["approved_lf"], "records": records,
        "legacy_lf_image_rows": selected_rows, "output": output, "bindings": bindings,
        "preflight": preflight, "contract": contract}


def _assert_engineering_scope(value: dict[str, Any]) -> None:
    for key in NO_AUTHORIZATION + ("new_training_authorized", "new_validation_split"):
        if key in value:
            _same(value[key], False, key)
    if "outer_optimizer_steps" in value:
        _same(value["outer_optimizer_steps"], 0, "outer optimizer steps")
    if "scope" in value:
        _assert_engineering_scope(value["scope"])


def freeze_run(prepared: dict[str, Any]) -> None:
    config = read_config(prepared["config_path"])
    _same(prepared["new_config"], config, "prepared configuration")
    _same(prepared["contract"]["configuration"], config, "run configuration snapshot")
    _same([row["image_id"] for row in prepared["records"]], config["sampling"]["ids"], "prepared IDs")
    _assert_engineering_scope(prepared["contract"])
    output = ROOT / OUTPUT_RELATIVE
    if prepared["output"] != output or output.is_symlink():
        raise ValueError("R4 output must be the canonical append-only engineering directory")
    lf.validate_preregistration(config["lf_config"])
    lf.assert_bindings(prepared["bindings"])
    # Even an empty pre-existing directory is a partial run, not a reusable
    # reservation.  reserve_output makes the append-only reservation itself.
    legacy.reserve_output(output)
    legacy.freeze_run(output, prepared["contract"], prepared["bindings"])


def complete_run(prepared: dict[str, Any], summary: dict[str, Any]) -> None:
    config = read_config(prepared["config_path"])
    if prepared["output"] != ROOT / OUTPUT_RELATIVE:
        raise ValueError("unexpected R4 completion directory")
    frozen_contract = lf.read_json(prepared["output"] / "RUN_CONTRACT.json")
    _same(frozen_contract["configuration"], config, "completion configuration snapshot")
    _assert_engineering_scope(frozen_contract)
    _assert_engineering_scope(summary)
    payload = {**summary, "protocol_id": config["protocol_id"], "stage": config["stage"],
        "scope": dict(config["scope"]), "outer_optimizer_steps": 0,
        "new_training_authorized": False, "ipma_meta_training_allowed": False,
        "full_source_training_allowed": False, "formal_test_allowed": False,
        "development_only": True, "paper_result": False, "r5_requires_separate_frozen_contract": True}
    lf.complete_output(prepared["output"], payload)


def verify_run(path: str | Path) -> dict[str, Any]:
    output = Path(path)
    output = output if output.is_absolute() else ROOT / output
    if output != ROOT / OUTPUT_RELATIVE or output.is_symlink():
        raise ValueError("only the canonical R4 engineering output can be verified")
    config = read_config()
    verified = lf.verify_complete_dir(output)
    _verify_freeze(output)
    contract = lf.read_json(output / "RUN_CONTRACT.json")
    _same(contract["configuration"], config, "verified configuration snapshot")
    _same(contract["config_binding"], legacy.binding(DEFAULT_CONFIG), "verified config binding")
    _same(verified["complete"].get("new_training_authorized"), False, "COMPLETE training authorization")
    _same(verified["summary"].get("protocol_id"), config["protocol_id"], "verified summary protocol")
    _same(verified["summary"].get("stage"), config["stage"], "verified summary stage")
    _same(verified["summary"].get("scope"), config["scope"], "verified summary scope")
    _assert_engineering_scope(contract)
    _assert_engineering_scope(verified["summary"])
    return verified["summary"]


__all__ = ["DEFAULT_CONFIG", "read_config", "prepare_run", "freeze_run", "complete_run", "verify_run"]
