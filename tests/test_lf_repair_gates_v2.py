from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from analysis import finalize_lf_repair_v2 as gate


def config():
    return {
        "datasets": list(gate.DATASETS), "views": list(gate.VIEWS),
        "global_seed": 42, "probe_seed_namespace": "synthetic-paired-orbits",
        "operator": {"mask_ratio": 0.20, "pair_keep_probability": 0.50},
        "variants": [
            {"probe_id": "L1", "attenuation": 1.0, "protect_dc": False, "shared_channels": False},
            {"probe_id": "L2", "attenuation": 1.0, "protect_dc": True, "shared_channels": False},
            {"probe_id": "L3", "attenuation": 1.0, "protect_dc": True, "shared_channels": True},
            {"probe_id": "L4a", "attenuation": 0.25, "protect_dc": True, "shared_channels": True},
            {"probe_id": "L4b", "attenuation": 0.50, "protect_dc": True, "shared_channels": True},
        ],
    }


def target(*, clean=0.2, post=0.12, ring=10, target_id=1, image_id="image"):
    valid = ring > 0
    eligible = valid and abs(clean) >= gate.MIN_CONTRAST
    ratio = (abs(post) + gate.EPSILON) / (abs(clean) + gate.EPSILON) if valid else None
    below = ratio < gate.RETENTION_MIN if valid else None
    flip = clean * post < 0 if valid else None
    return {
        "image_id": image_id, "target_id": target_id, "target_pixels": 1,
        "ring_pixels": ring, "bbox_yxyx_exclusive": [0, 0, 1, 1], "ring_valid": valid,
        "clean_contrast": clean if valid else None,
        "preclip_contrast": post if valid else None,
        "postclip_contrast": post if valid else None,
        "preclip_contrast_retention_ratio": ratio, "postclip_contrast_retention_ratio": ratio,
        "preclip_contrast_sign_flip": flip, "postclip_contrast_sign_flip": flip,
        "postclip_contrast_below_ratio_threshold": below,
        "contrast_eligible_for_e1": eligible,
        "original_E1_failure": below if eligible else None,
        "compound_visibility_failure": bool(below or flip) if eligible else None,
    }


def cell_images(*, rms=0.01):
    return [{"image_id": f"image-{i}", "postclip_perturbation_rms_rgb": rms} for i in range(64)]


def access():
    return {"train_image_opens": 128, "train_mask_opens": 128,
            **{field: 0 for field in gate.FORBIDDEN_ACCESS}}


def all_cells():
    return [{"dataset": dataset, "view": view, "probe_id": probe,
             **gate.evaluate_cell(cell_images(rms=0.0 if probe == "clean" else 0.01), [target()])}
            for dataset in gate.DATASETS for view in gate.VIEWS for probe in gate.PROBES]


@pytest.fixture
def raw_evidence():
    cfg = config()
    images, targets, old_images, old_targets = [], [], [], []
    for dataset in gate.DATASETS:
        for view in gate.VIEWS:
            for index in range(64):
                image_id = f"image-{index}"
                metadata = {"input_tensor_sha256": "a" * 64, "target_tensor_sha256": "b" * 64,
                    "view": view, "augmentation_seed": 3 if view == "train_crop_224" else None,
                    "historical_training_random_stream_replay": False}
                base = {"dataset": dataset, "image_id": image_id, "view": view}
                old_images.append({**base, "probe_id": "lf_mask", "input_metadata": metadata, "target_count": 1})
                old_targets.append({**base, "probe_id": "lf_mask", **target(image_id=image_id)})
                for probe in gate.PROBES:
                    descriptor = {**base, "probe_id": probe,
                        "operator_config_sha256": gate.expected_operator_hash(cfg, probe),
                        "input_hash": "a" * 64, "target_hash": "b" * 64,
                        "probe_seed": None if probe == "clean" else gate.expected_probe_seed(cfg, dataset, image_id, view)}
                    images.append({**descriptor, "input_metadata": metadata, "target_count": 1,
                        "sealed_input_pair_exact": True, "gt_tensor_unchanged": True,
                        "deterministic_repeat_exact": True, "preclip_clamp_parity_exact": True,
                        "postclip_perturbation_rms_rgb": 0.0 if probe == "clean" else 0.01,
                        "postclip_rms": 0.0 if probe == "clean" else 0.01,
                        "clean_physical_tensor_sha256": "c" * 64,
                        "preclip_physical_tensor_sha256": "c" * 64 if probe == "clean" else "d" * 64,
                        "postclip_physical_tensor_sha256": "c" * 64 if probe == "clean" else "d" * 64})
                    targets.append({**descriptor, **target(image_id=image_id, post=0.2 if probe == "clean" else 0.12)})
    return {"image_rows": images, "target_rows": targets,
            "legacy_images": old_images, "legacy_targets": old_targets,
            "config": cfg, "access_by_dataset": {dataset: access() for dataset in gate.DATASETS}}


def test_contrast_inversion_is_compound_failure_but_not_original_e1():
    result = gate.recompute_target(target(clean=0.2, post=-0.2))
    assert result["eligible"]
    assert not result["e1_failure"]
    assert result["compound_failure"]


def test_original_e1_eligibility_and_same_compound_denominator_are_retained():
    targets = [target(), target(clean=0.001, post=-0.001), target(ring=0)]
    cell = gate.evaluate_cell(cell_images(), targets)
    assert cell["target_count"] == 3
    assert cell["original_e1_denominator"] == cell["compound_denominator"] == 1
    assert cell["low_clean_contrast_target_count"] == cell["invalid_ring_target_count"] == 1


def test_exact_ten_percent_is_allowed_but_more_is_not():
    exactly = gate.evaluate_cell(cell_images(), [target(post=0)] + [target()] * 9)
    assert exactly["original_e1_failure_fraction"] == 0.1
    assert exactly["candidate_cell_passed"]
    above = gate.evaluate_cell(cell_images(), [target(post=0)] + [target()] * 8)
    assert not above["candidate_cell_passed"]


def test_empty_eligible_denominator_is_scientific_failure_not_fake_zero_risk():
    result = gate.evaluate_cell(cell_images(), [target(clean=0.001)])
    assert result["original_e1_failure_fraction"] is None
    assert result["compound_failure_fraction"] is None
    assert not result["candidate_cell_passed"]
    assert "insufficient_evidence_empty_original_e1_denominator" in result["failure_reasons"]


def test_exact_rms_floor_is_not_informative_and_constant_images_are_not_dropped():
    assert not gate.evaluate_cell(cell_images(rms=1e-6), [target()])["image_signal_floor_passed"]
    images = cell_images()
    for row in images[:6]:
        row["postclip_perturbation_rms_rgb"] = 0.0
    assert gate.evaluate_cell(images, [target()])["image_signal_floor_passed"]
    images[6]["postclip_perturbation_rms_rgb"] = 0.0
    result = gate.evaluate_cell(images, [target()])
    assert result["image_count"] == 64 and result["images_above_rms_floor"] == 57
    assert not result["image_signal_floor_passed"]


def test_any_one_bad_dataset_view_blocks_candidate_without_macro_averaging():
    cells = all_cells()
    bad = next(row for row in cells if row["dataset"] == "NUDT-SIRST" and row["view"] == "train_crop_224" and row["probe_id"] == "L4a")
    bad.update(gate.evaluate_cell(cell_images(), [target(post=0)] * 2 + [target()] * 8))
    receipt = gate.select_candidate(cells)
    assert receipt["selected_probe_id"] is None
    assert not receipt["signal_smoke_16_allowed"]
    assert receipt["candidates"]["L4b"]["alternate_eligible"]
    assert "no_selection_l4b_alternate" in receipt["status"]


def test_both_pass_always_selects_mild_a_and_never_authorizes_training_or_test():
    result = gate.select_candidate(all_cells())
    assert result["selected_probe_id"] == "L4a" and result["signal_smoke_16_allowed"]
    for key in ("ipma_meta_training_allowed", "full_source_training_allowed", "formal_test_allowed", "paper_result"):
        assert result[key] is False


def test_bad_diagnostic_controls_do_not_override_qualified_l4_candidate():
    cells = all_cells()
    for row in cells:
        if row["probe_id"] in ("L1", "L2", "L3"):
            row.update(gate.evaluate_cell(cell_images(), [target(post=0)]))
    assert gate.select_candidate(cells)["selected_probe_id"] == "L4a"


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_all_36_cells_are_required_even_noneligible_controls(change):
    cells = all_cells()
    if change == "missing":
        cells = cells[1:]
    else:
        cells.append(cells[0])
    with pytest.raises(gate.GateInputError):
        gate.select_candidate(cells)


def test_complete_raw_records_recompute_gate_without_summary_flags(raw_evidence):
    cells, receipt = gate.evaluate_records(**raw_evidence)
    assert len(cells) == 36
    assert receipt["selected_probe_id"] == "L4a"
    assert all(row["original_e1_denominator"] == 64 for row in cells)


@pytest.mark.parametrize("field", ["test_split_reads", "test_image_opens", "validation_mask_opens", "train_image_opens"])
def test_access_missing_or_nonzero_forbidden_fails_closed(field):
    counts = access()
    del counts[field]
    with pytest.raises(gate.GateInputError):
        gate.validate_access(counts)
    counts = access()
    counts[field] += 1
    with pytest.raises(gate.GateInputError):
        gate.validate_access(counts)


@pytest.mark.parametrize("mutation", ["missing_image", "duplicate_image", "missing_target", "duplicate_target", "input_hash", "target_hash", "operator_hash", "seed", "nonfinite", "target_geometry", "eligibility"])
def test_raw_record_coverage_hash_and_numeric_drift_fail_closed(raw_evidence, mutation):
    if mutation == "missing_image":
        raw_evidence["image_rows"].pop()
    elif mutation == "duplicate_image":
        raw_evidence["image_rows"].append(raw_evidence["image_rows"][0])
    elif mutation == "missing_target":
        raw_evidence["target_rows"].pop()
    elif mutation == "duplicate_target":
        raw_evidence["target_rows"].append(raw_evidence["target_rows"][0])
    elif mutation in ("input_hash", "target_hash"):
        raw_evidence["image_rows"][0][mutation] = "f" * 64
    elif mutation == "operator_hash":
        raw_evidence["image_rows"][0]["operator_config_sha256"] = "f" * 64
    elif mutation == "seed":
        raw_evidence["image_rows"][1]["probe_seed"] += 1
    elif mutation == "nonfinite":
        raw_evidence["image_rows"][0]["nested_diagnostics"] = {"bad": float("nan")}
    elif mutation == "target_geometry":
        raw_evidence["target_rows"][0]["ring_pixels"] += 1
    elif mutation == "eligibility":
        raw_evidence["target_rows"][0]["contrast_eligible_for_e1"] = False
    with pytest.raises(gate.GateInputError):
        gate.evaluate_records(**raw_evidence)


def test_fabricated_low_risk_flag_is_rejected_from_raw_contrasts():
    row = target(post=0)
    row["postclip_contrast_below_ratio_threshold"] = False
    row["original_E1_failure"] = False
    with pytest.raises(gate.GateInputError):
        gate.recompute_target(row)


def test_empty_ring_missing_instead_of_explicit_null_is_rejected():
    row = target(ring=0)
    del row["postclip_contrast"]
    with pytest.raises(gate.GateInputError):
        gate.recompute_target(row)


def test_jsonl_rejects_nonfinite_and_missing_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"rms": NaN}\n', encoding="utf-8")
    with pytest.raises(gate.GateInputError):
        gate.read_jsonl(path)
    with pytest.raises(FileNotFoundError):
        gate.read_jsonl(tmp_path / "missing.jsonl")


def sealed_fixture(path):
    from analysis import d0a_v7_common as artifacts
    path.mkdir()
    files = {}
    for name in ("RUN_CONTRACT.json", "PRE_RUN_FREEZE.json", "summary.json"):
        artifacts.write_json_new(path / name, {})
        files[name] = artifacts.sha256_file(path / name)
    artifacts.write_json_new(path / "artifact_manifest.json", {"files": files})
    artifacts.write_json_new(path / "COMPLETE.json", {"complete": True, "paper_result": False,
        "artifact_manifest": artifacts.binding(path / "artifact_manifest.json")})
    return path


def test_manifest_hash_drift_cannot_become_scientific_pass(tmp_path):
    from analysis import lf_repair_contract_v2 as contract
    sealed = sealed_fixture(tmp_path / "sealed")
    with pytest.raises(ValueError, match="manifest drift"):
        contract.verify_complete_dir(sealed, "f" * 64)
    (sealed / "summary.json").write_text('{"status":"PASS"}', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        contract.verify_complete_dir(sealed)


def test_default_cli_has_explicit_execute_and_no_image_or_model_loader():
    source = inspect.getsource(gate)
    assert "--execute" in source
    assert "def finalize(config_path: Path, *, execute: bool = False)" in source
    for forbidden in ("Image.open", "load_sample(", "torch.load", "optimizer.step", "subprocess.run"):
        assert forbidden not in source


def freeze_case():
    cfg = config()
    cfg.update({"protocol_id": "cr-sitta-lf-repair-audit-v2", "gate": {"risk_fraction_max": 0.1},
                "stage_scope": {"paper_result": False}, "scientific_scope": "augmentation_visibility_screen_only"})
    prereg_binding = {"path": "/synthetic/PREREGISTRATION.json", "sha256": "a" * 64}
    config_binding = {"path": "/synthetic/config.yaml", "sha256": "b" * 64}
    operator_binding = {"path": "/synthetic/operator.py", "sha256": "c" * 64}
    own_binding = {"path": "/synthetic/RUN_CONTRACT.json", "sha256": "d" * 64}
    prereg = {"input_bindings": [config_binding, operator_binding]}
    visibility = {"min_clean_contrast": 1 / 255}
    run_contract = {"dataset": "NUDT-SIRST", "protocol_id": cfg["protocol_id"],
        "operator": cfg["operator"], "variants": cfg["variants"], "gate": cfg["gate"],
        "visibility": visibility, "stage_scope": cfg["stage_scope"],
        "scientific_scope": cfg["scientific_scope"], "paper_result": False, "checkpoint_used": None}
    freeze = {"contract": own_binding, "input_bindings": [config_binding, operator_binding, prereg_binding]}
    return {"freeze": freeze, "run_contract": run_contract, "config": cfg,
            "legacy_visibility": visibility, "dataset": "NUDT-SIRST", "preregistration": prereg,
            "preregistration_binding": prereg_binding, "run_contract_binding": own_binding}


def test_r2_freeze_must_bind_same_preregistration_operator_and_configuration():
    case = freeze_case()
    gate.validate_r2_freeze(**case)
    case["freeze"]["input_bindings"] = case["freeze"]["input_bindings"][:-1]
    with pytest.raises(gate.GateInputError, match="preregistered inputs"):
        gate.validate_r2_freeze(**case)


@pytest.mark.parametrize("mutation", ["foreign_contract", "operator_drift", "dataset", "gate", "variant", "duplicate_binding"])
def test_self_consistent_unrelated_r2_seal_does_not_establish_protocol_eligibility(mutation):
    case = copy.deepcopy(freeze_case())
    if mutation == "foreign_contract":
        case["freeze"]["contract"] = {"path": "/elsewhere/contract.json", "sha256": "f" * 64}
    elif mutation == "operator_drift":
        case["freeze"]["input_bindings"][1] = {"path": "/synthetic/operator.py", "sha256": "f" * 64}
    elif mutation == "dataset":
        case["run_contract"]["dataset"] = "IRSTD-1K"
    elif mutation == "gate":
        case["run_contract"]["gate"] = {"risk_fraction_max": 0.9}
    elif mutation == "variant":
        case["run_contract"]["variants"] = []
    else:
        case["freeze"]["input_bindings"].append(case["freeze"]["input_bindings"][0])
    with pytest.raises(gate.GateInputError):
        gate.validate_r2_freeze(**case)


def field_case():
    import torch
    from tta.deteriorations.fourier_low_mask_v2 import _tensor_sha256
    values = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]], dtype=torch.float64)
    cfg = config()
    base = {"dataset": "NUDT-SIRST", "image_id": "synthetic", "view": "full_256"}
    field = {**base, "probe_seed": gate.expected_probe_seed(cfg, **base), "shared_by": list(gate.VARIANTS),
        "random_field_shape": list(values.shape), "random_field_sha256": _tensor_sha256(values),
        "random_field_values": values.tolist()}
    images = [{**base, "probe_id": probe, "random_field_sha256": field["random_field_sha256"],
               "random_field_shape": field["random_field_shape"],
               "tensor_hash_schema": "lf_v2_dtype_shape_contiguous_bytes_v1"} for probe in gate.PROBES]
    return [field], images, cfg


def test_random_field_hash_recomputed_matches_operator_torch_hash_schema():
    gate.validate_random_fields(*field_case())


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "hash", "value", "shape", "dtype", "nonfinite", "candidate_hash", "candidate_missing", "seed"])
def test_random_field_sidecar_and_every_variant_must_agree(mutation):
    fields, images, cfg = copy.deepcopy(field_case())
    if mutation == "missing":
        fields.clear()
    elif mutation == "duplicate":
        fields.append(fields[0])
    elif mutation == "hash":
        fields[0]["random_field_sha256"] = "f" * 64
    elif mutation == "value":
        fields[0]["random_field_values"][0][0][0] = 0.7
    elif mutation == "shape":
        fields[0]["random_field_shape"] = [1, 3, 3]
    elif mutation == "dtype":
        fields[0]["random_field_values"] = [[[0, 0], [0, 0], [0, 0]]]
    elif mutation == "nonfinite":
        fields[0]["random_field_values"][0][0][0] = float("nan")
    elif mutation == "candidate_hash":
        images[1]["random_field_sha256"] = "f" * 64
    elif mutation == "candidate_missing":
        images.pop()
    else:
        fields[0]["probe_seed"] += 1
    with pytest.raises(gate.GateInputError):
        gate.validate_random_fields(fields, images, cfg)


def test_complete_scientific_failure_can_be_sealed_without_training_authority(tmp_path):
    from analysis import d0a_v7_common as artifacts
    from analysis import lf_repair_contract_v2 as contract
    output = tmp_path / "negative"
    artifacts.freeze_run(output, {"paper_result": False}, [])
    cells = all_cells()
    for cell in cells:
        if cell["probe_id"] in ("L4a", "L4b"):
            cell.update(gate.evaluate_cell(cell_images(), [target(post=0)]))
    result = gate.select_candidate(cells)
    contract.complete_output(output, result)
    sealed = contract.verify_complete_dir(output)
    assert sealed["complete"]["complete"]
    assert not sealed["complete"]["new_training_authorized"]
    assert not sealed["summary"]["signal_smoke_16_allowed"]
    assert sealed["summary"]["selected_probe_id"] is None


def test_preregistration_failure_propagates_before_any_finalization_writes(monkeypatch, tmp_path):
    from analysis import lf_repair_contract_v2 as contract
    monkeypatch.setattr(contract, "read_config", lambda path: {})
    def mismatch(path):
        raise RuntimeError("frozen operator changed")
    monkeypatch.setattr(contract, "validate_preregistration", mismatch)
    with pytest.raises(RuntimeError, match="frozen operator"):
        gate.finalize(tmp_path / "config.yaml", execute=True)
    assert list(tmp_path.iterdir()) == []
