"""CPU synthetic tests only: no real dataset files or detector inference."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import types

import pytest
import torch

from analysis import audit_lf_repair_v2 as audit
from analysis.audit_d0a_probe_label_preservation import VISIBILITY_DEFAULTS, audit_visibility
from tta.deteriorations.image_space import imagenet_denormalize, imagenet_normalize


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configuration():
    return {"protocol_id": "cr-sitta-lf-repair-audit-v2", "global_seed": 42, "threads": 1,
        "probe_seed_namespace": "cr-sitta-lf-repair-audit-v2-paired-orbits",
        "operator": {"mask_ratio": 0.2, "pair_keep_probability": 0.5},
        "variants": [{"probe_id": name, "attenuation": alpha, "protect_dc": dc, "shared_channels": shared}
                     for name, alpha, dc, shared in [("L1", 1.0, False, False),
                     ("L2", 1.0, True, False), ("L3", 1.0, True, True),
                     ("L4a", 0.25, True, True), ("L4b", 0.5, True, True)]],
        "gate": {"image_rms_floor": 1e-6, "image_rms_fraction_min": 0.9},
        "bootstrap": {"replicates": 2000, "seed": 42, "confidence_level": 0.95}}


def synthetic_sample(*, empty=False, contrast=0.4, empty_ring=False):
    physical = torch.full((1, 3, 24, 24), 0.3, dtype=torch.float64)
    target = torch.zeros((1, 24, 24), dtype=torch.float64)
    if not empty:
        target[:, 10:13, 10:13] = 1
        physical[:, :, 10:13, 10:13] += contrast
    if empty_ring:
        target.fill_(1)
    normalized = imagenet_normalize(physical)[0]
    return normalized, target


def legacy_for(normalized, target, *, image_id="a", view="full_256"):
    physical = imagenet_denormalize(normalized[None])
    artifacts = {"clean": physical, "preclip": physical, "postclip": physical,
                 "noise": None, "statistics": {}}
    image, targets = audit_visibility(artifacts, target, visibility=VISIBILITY_DEFAULTS)
    descriptor = {"dataset": "NUDT-SIRST", "image_id": image_id, "view": view, "probe_id": "lf_mask"}
    metadata = {"input_tensor_sha256": "synthetic_input", "target_tensor_sha256": "synthetic_target",
        "view": view, "augmentation_seed": 123 if view == "train_crop_224" else None,
        "historical_training_random_stream_replay": False}
    return {**descriptor, **image, "input_metadata": metadata}, [{**descriptor, **r} for r in targets]


def enrich_case(*, post=None, pre=None, empty=False, contrast=0.4, empty_ring=False):
    normalized, target = synthetic_sample(empty=empty, contrast=contrast, empty_ring=empty_ring)
    clean = imagenet_denormalize(normalized[None])
    old_image, old_targets = legacy_for(normalized, target)
    _, old_target_index = audit.index_legacy_rows([old_image], old_targets)
    descriptor = {"dataset": "NUDT-SIRST", "image_id": "a", "view": "full_256", "probe_id": "L4a"}
    artifacts = {"clean": clean, "preclip": clean if pre is None else pre,
                 "postclip": clean if post is None else post, "noise": None, "statistics": {}}
    image, rows = audit.enrich_visibility(artifacts, target, VISIBILITY_DEFAULTS, descriptor, old_target_index)
    return {**descriptor, **image, "postclip_perturbation_rms_rgb": 0.01,
            "constant_input_image": False, "sealed_input_pair_exact": True,
            "deterministic_repeat_exact": True}, rows


def test_seed_is_stable_order_independent_and_view_distinct():
    args = ("namespace", 42, "NUDT-SIRST", "abc")
    expected = audit.stable_probe_seed(*args, "full_256")
    for _ in range(3):
        assert audit.stable_probe_seed(*args, "full_256") == expected
    assert audit.stable_probe_seed(*args, "train_crop_224") != expected
    assert audit.stable_probe_seed("other", *args[1:], "full_256") != expected


@pytest.mark.parametrize("field", ["input_tensor_sha256", "target_tensor_sha256", "augmentation_seed", "view"])
def test_sealed_replay_rejects_any_hash_or_crop_drift(field):
    sample = legacy_for(*synthetic_sample())[0]
    replay = copy.deepcopy(sample["input_metadata"])
    audit.assert_sealed_sample(replay, sample)
    replay[field] = "changed"
    with pytest.raises(RuntimeError, match="replay mismatch"):
        audit.assert_sealed_sample(replay, sample)


def test_legacy_duplicate_and_missing_l0_rejected():
    image, rows = legacy_for(*synthetic_sample())
    with pytest.raises(ValueError, match="duplicate"):
        audit.index_legacy_rows([image, image], rows)
    with pytest.raises(ValueError, match="missing"):
        audit.index_legacy_rows([{**image, "probe_id": "clean"}], rows)


def test_missing_target_is_not_silently_dropped_from_paired_evidence():
    normalized, target = synthetic_sample()
    clean = imagenet_denormalize(normalized[None])
    old_image, old_rows = legacy_for(normalized, target)
    _, old_index = audit.index_legacy_rows([old_image], old_rows)
    descriptor = {"dataset": "NUDT-SIRST", "image_id": "a", "view": "full_256", "probe_id": "L4a"}
    artifacts = {"clean": clean, "preclip": clean, "postclip": clean, "noise": None, "statistics": {}}
    with pytest.raises(RuntimeError, match="target set differs"):
        audit.enrich_visibility(artifacts, torch.zeros_like(target), VISIBILITY_DEFAULTS, descriptor, old_index)


def test_identity_retains_original_eligibility_and_has_zero_risk():
    image, rows = enrich_case()
    assert image["target_count"] == 1
    assert rows[0]["original_E1_failure"] is False
    assert rows[0]["compound_visibility_failure"] is False
    assert rows[0]["retention_abs_postclip"] == 1
    assert rows[0]["exclusion_reason"] is None


def test_sign_flip_compound_is_not_redefined_as_original_e1():
    normalized, _ = synthetic_sample()
    post = imagenet_denormalize(normalized[None])
    post[:, :, 10:13, 10:13] = -0.1
    _, rows = enrich_case(post=post, pre=post)
    assert rows[0]["original_E1_failure"] is False
    assert rows[0]["compound_visibility_failure"] is True
    assert rows[0]["sign_flip_postclip"] is True


@pytest.mark.parametrize("mode,reason", [("low", "low_clean_contrast"), ("ring", "empty_background_ring")])
def test_excluded_targets_are_retained_with_peak_and_reason(mode, reason):
    _, rows = enrich_case(contrast=0.0001 if mode == "low" else 0.4, empty_ring=mode == "ring")
    assert len(rows) == 1
    assert rows[0]["exclusion_reason"] == reason
    assert rows[0]["original_E1_failure"] is None
    assert rows[0]["compound_visibility_failure"] is None
    assert rows[0]["clean_target_peak_gray"] is not None


def test_empty_crop_stays_in_image_denominator_and_missing_ci_is_explicit():
    image, rows = enrich_case(empty=True)
    summary = audit.summarize_cell([image], rows, visibility=VISIBILITY_DEFAULTS, config=configuration())
    assert summary["image_count"] == summary["empty_target_view_count"] == 1
    assert summary["eligible_target_count"] == 0
    assert summary["compound_visibility_failure_fraction"] is None
    assert summary["bootstrap"]["valid_replicates"] == 0
    assert summary["bootstrap"]["original_e1_fraction_ci"] is None


def test_actual_clipping_is_distinct_from_endpoint_occupancy():
    normalized, _ = synthetic_sample()
    pre = imagenet_denormalize(normalized[None])
    pre[:, :, 10:13, 10:13] = 1.0
    _, rows = enrich_case(pre=pre, post=pre)
    assert rows[0]["endpoint_occupancy_fraction_core"] == 1.0
    assert rows[0]["actual_clipping_changed_fraction_core"] == 0.0
    pre[:, :, 10:13, 10:13] = 1.2
    _, rows = enrich_case(pre=pre, post=pre.clamp(0, 1))
    assert rows[0]["actual_clipping_changed_fraction_core"] == 1.0


def test_bootstrap_clusters_by_image_includes_empty_and_pairs_old_comparator():
    _, rows = enrich_case()
    rows = [{**rows[0], "target_id": i, "original_E1_failure": True,
             "compound_visibility_failure": True} for i in range(20)]
    result = audit.cluster_bootstrap(rows, ["a", "empty"], replicates=2000, seed=42)
    assert result["image_clusters"] == 2
    assert result["zero_eligible_replicates"] > 0
    assert result["original_e1_fraction_ci"] == [1.0, 1.0]
    assert result["paired_l0_e1_fraction_delta_ci"] == [1.0, 1.0]
    assert result == audit.cluster_bootstrap(rows, ["a", "empty"], replicates=2000, seed=42)


def test_nonidentity_requires_median_and_coverage_including_constant_images():
    image, _ = enrich_case(empty=True)
    images = [{**image, "image_id": str(i), "constant_input_image": i == 0,
               "postclip_perturbation_rms_rgb": 0 if i < 2 else 0.01} for i in range(10)]
    summary = audit.summarize_cell(images, [], visibility=VISIBILITY_DEFAULTS, config=configuration())
    assert summary["nonidentity"]["informative"] is False
    assert summary["nonidentity"]["constant_input_image_count"] == 1
    images[1]["postclip_perturbation_rms_rgb"] = 0.01
    summary = audit.summarize_cell(images, [], visibility=VISIBILITY_DEFAULTS, config=configuration())
    assert summary["nonidentity"]["informative"] is True
    assert summary["nonidentity"]["fraction_above_floor"] == 0.9


def test_new_variants_share_full_field_without_reusing_legacy_probe():
    config = configuration()
    normalized, _ = synthetic_sample()
    artifacts = [audit.build_candidate_artifacts(normalized, candidate=candidate,
                 operator=config["operator"], seed=43) for candidate in config["variants"]]
    assert all(value["random_field_values"] == artifacts[0]["random_field_values"] for value in artifacts)
    assert len({value["statistics"]["random_field_sha256"] for value in artifacts}) == 1
    assert all("random_field_values" not in value["statistics"] for value in artifacts)
    assert all(value["statistics"]["deterministic_repeat_exact"] for value in artifacts)


def test_synthetic_clean_control_is_exact_identity_and_no_noise_field():
    normalized, _ = synthetic_sample()
    value = audit.build_candidate_artifacts(normalized, candidate={"probe_id": "clean"}, operator={}, seed=7)
    assert torch.equal(value["clean"], value["postclip"])
    assert value["random_field_values"] is None
    assert value["statistics"]["postclip_perturbation_rms_rgb"] == 0


def fake_contract(monkeypatch, prepared, calls):
    module = types.ModuleType("analysis.lf_repair_contract_v2")
    module.prepare_run = lambda *args: prepared
    module.freeze_dataset = lambda item: calls.append("freeze")
    module.complete_dataset = lambda item, summary: calls.append(("complete", summary))
    monkeypatch.setitem(sys.modules, "analysis.lf_repair_contract_v2", module)


def test_default_dry_run_never_loads_sample_or_writes(monkeypatch):
    import analysis.d0a_v7_common as common
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not decode or write")
    monkeypatch.setattr(common, "load_sample", forbidden)
    monkeypatch.setattr(common, "write_jsonl_new", forbidden)
    calls = []
    fake_contract(monkeypatch, {"preflight": {"ready": True, "writes": 0}}, calls)
    args = argparse.Namespace(config=Path("synthetic.yaml"), dataset="NUDT-SIRST", execute=False, device="cpu", threads=1)
    assert audit.run(args) == {"ready": True, "writes": 0}
    assert calls == []


def test_execute_cpu_synthetic_loads_each_image_view_once_and_never_rebuilds_l0(monkeypatch, tmp_path):
    import analysis.d0a_v7_common as common
    normalized, target = synthetic_sample()
    records = [{"dataset": "NUDT-SIRST", "image_id": name} for name in ["a", "b"]]
    old_images, old_targets = [], []
    for record in records:
        for view in audit.VIEWS:
            old_image, rows = legacy_for(normalized, target, image_id=record["image_id"], view=view)
            old_images.append(old_image); old_targets.extend(rows)
    legacy_config = {"visibility": VISIBILITY_DEFAULTS}
    prepared = {"new_config": configuration(), "legacy_config": legacy_config, "records": records,
        "legacy_image_rows": old_images, "legacy_target_rows": old_targets,
        "output": tmp_path, "contract": {}, "preflight": {"ready": True, "writes": 0}}
    calls, loaded, writes = [], [], {}
    fake_contract(monkeypatch, prepared, calls)
    monkeypatch.setattr(audit, "validate_diagnostic_parameters", lambda c: None)
    def load(record, view, config, *, access):
        loaded.append((record["image_id"], view))
        access["train_image_opens"] += 1; access["train_mask_opens"] += 1
        metadata = legacy_for(normalized, target, image_id=record["image_id"], view=view)[0]["input_metadata"]
        return normalized.clone(), target.clone(), metadata
    monkeypatch.setattr(common, "load_sample", load)
    monkeypatch.setattr(common, "write_jsonl_new", lambda path, rows: writes.setdefault(path.name, list(rows)))
    args = argparse.Namespace(config=Path("synthetic.yaml"), dataset="NUDT-SIRST", execute=True, device="cpu", threads=1)
    result = audit.run(args)
    assert len(loaded) == len(set(loaded)) == 4
    assert result["per_image_rows"] == 24
    assert len(result["cells"]) == 12
    assert result["access"]["train_image_opens"] == result["access"]["train_mask_opens"] == 4
    assert all(result["access"][field] == 0 for field in audit.FORBIDDEN_ACCESS_FIELDS)
    assert writes["legacy_L0_per_image.jsonl"] == old_images
    assert writes["legacy_L0_per_target.jsonl"] == old_targets
    assert len(writes["random_fields.jsonl"]) == 4
    assert calls[0] == "freeze" and calls[-1][0] == "complete"
    json.dumps(result, allow_nan=False)


def test_gpu_device_is_rejected_before_preflight(monkeypatch):
    fake_contract(monkeypatch, {}, [])
    with pytest.raises(ValueError, match="CPU|cpu"):
        audit.run(argparse.Namespace(device="cuda:0"))


def test_thread_drift_rejected_before_output_reservation(monkeypatch):
    calls = []
    fake_contract(monkeypatch, {"new_config": configuration(), "legacy_config": {}}, calls)
    args = argparse.Namespace(config=Path("synthetic.yaml"), dataset="NUDT-SIRST", execute=True, device="cpu", threads=2)
    with pytest.raises(ValueError, match="thread count"):
        audit.run(args)
    assert calls == []
