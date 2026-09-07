from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from analysis import audit_d0a_probe_label_preservation as audit
from tta.deteriorations import imagenet_normalize


@pytest.fixture
def config() -> dict:
    return {
        "global_seed": 42,
        "human_epoch": 1000,
        "probe_seed_namespace": "cr-sitta-d0a-supervised-lfhf-train-1000e-v2",
        "probes": {"lf_mask": {"mask_ratio": 0.2, "keep_probability": 0.5},
                   "hf_noise": {"target_rms": 0.02, "low_cut_ratio": 0.2}},
        "visibility": dict(audit.VISIBILITY_DEFAULTS),
    }


def artifacts(clean: torch.Tensor, post: torch.Tensor, pre: torch.Tensor | None = None) -> dict:
    return {"clean": clean, "preclip": post if pre is None else pre,
            "postclip": post, "noise": None, "statistics": {}}


def target_and_image() -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.zeros(1, 24, 24)
    target[:, 11:13, 11:13] = 1
    image = torch.full((1, 3, 24, 24), 0.4)
    image[:, :, 11:13, 11:13] = 0.9
    return target, image


@pytest.mark.parametrize("probe_id", audit.PROBES)
def test_probe_reuses_original_exactly_and_has_no_target_argument(config: dict, probe_id: str) -> None:
    image = torch.rand((1, 3, 24, 26), generator=torch.Generator().manual_seed(7))
    normalized = imagenet_normalize(image)
    original = normalized.clone()
    output = audit.build_probe_artifacts(normalized, probe_id=probe_id,
        dataset="NUDT-SIRST", image_id="synthetic", config=config)
    assert torch.equal(normalized, original)
    assert output["statistics"]["original_runner_parity_exact"]
    assert output["statistics"]["deterministic_repeat_exact"]
    assert torch.equal(output["preclip"].clamp(0, 1), output["postclip"])
    assert output["postclip"].shape == image.shape
    assert torch.isfinite(output["postclip"]).all()


def test_hf_preclip_rms_is_frozen_and_differs_from_postclip(config: dict) -> None:
    image = torch.ones(1, 3, 24, 26)
    output = audit.build_probe_artifacts(imagenet_normalize(image), probe_id="hf_noise",
        dataset="NUDT-SIRST", image_id="synthetic", config=config)
    assert output["statistics"]["hf_noise_rms_rgb"] == pytest.approx(0.02, abs=1e-7)
    assert output["statistics"]["preclip_perturbation_rms_rgb"] == pytest.approx(0.02, abs=1e-7)
    assert output["statistics"]["postclip_perturbation_rms_rgb"] < 0.02
    assert (output["preclip"] > 1).any()


def test_lf_reports_region_keep_not_full_mask_keep(config: dict) -> None:
    image = torch.full((1, 3, 64, 64), 0.5)
    output = audit.build_probe_artifacts(imagenet_normalize(image), probe_id="lf_mask",
        dataset="NUDT-SIRST", image_id="synthetic", config=config)
    stats = output["statistics"]
    assert stats["lf_nominal_region_frequency_bins"] == 144
    assert 0.10 < stats["lf_nominal_region_keep_fraction"] < 0.40
    assert stats["lf_full_mask_keep_fraction"] > 0.95
    assert len(stats["lf_dc_mask_per_channel"]) == 3
    for before, after, keep in zip(stats["clean_dc_per_channel"], stats["preclip_dc_per_channel"], stats["lf_dc_mask_per_channel"]):
        assert after == pytest.approx(before * keep, abs=1e-5)


def test_connected_components_use_eight_connectivity_and_ring_excludes_all_targets() -> None:
    target = np.zeros((24, 24), dtype=np.uint8)
    target[10, 10] = 1
    target[11, 11] = 2
    target[10, 14] = 255
    regions = audit.target_regions(target)
    assert len(regions) == 2
    assert regions[0]["target_pixels"] == 2
    for region in regions:
        assert not (region["ring"] & (target > 0)).any()
        assert not (region["ring"] & region["core"]).any()
    assert not regions[0]["ring"][10, 11]  # inner radius 1 is excluded
    assert regions[0]["ring"][10, 13]


def test_empty_target_view_is_retained(config: dict) -> None:
    image = torch.full((1, 3, 16, 16), 0.5)
    stats, rows = audit.audit_visibility(artifacts(image, image), torch.zeros(1, 16, 16),
                                         visibility=config["visibility"])
    assert stats["empty_target_view"]
    assert stats["target_count"] == 0
    assert rows == []
    summary = audit.summarize_targets(rows, config["visibility"])
    assert summary["e1_visibility_risk_triggered"] is None
    assert summary["e1_insufficient_eligible_targets"]


def test_invalid_ring_is_reported_not_removed(config: dict) -> None:
    image = torch.full((1, 3, 16, 16), 0.5)
    stats, rows = audit.audit_visibility(artifacts(image, image), torch.ones(1, 16, 16),
                                         visibility=config["visibility"])
    assert len(rows) == 1
    assert not rows[0]["ring_valid"]
    assert rows[0]["invalid_reason"] == "empty_background_ring"
    assert rows[0]["postclip_contrast_retention_ratio"] is None
    assert stats["ring_invalid_target_count"] == 1


def test_unchanged_mask_does_not_imply_visibility_preserved(config: dict) -> None:
    target, image = target_and_image()
    gt_before = target.clone()
    invisible = torch.full_like(image, 0.4)
    stats, rows = audit.audit_visibility(artifacts(image, invisible), target,
                                         visibility=config["visibility"])
    assert stats["gt_tensor_unchanged"] and torch.equal(target, gt_before)
    assert rows[0]["postclip_contrast_retention_ratio"] < 0.3
    summary = audit.summarize_targets(rows, config["visibility"])
    assert summary["e1_visibility_risk_triggered"]
    assert not summary["label_semantics_changed_proven"]


def test_gray_contrast_uses_rgb_mean_and_records_sign_flip(config: dict) -> None:
    target, image = target_and_image()
    flipped = image.clone()
    flipped[:, 0, 11:13, 11:13] = 0.0
    flipped[:, 1, 11:13, 11:13] = 0.1
    flipped[:, 2, 11:13, 11:13] = 0.2
    _, rows = audit.audit_visibility(artifacts(image, flipped), target, visibility=config["visibility"])
    assert rows[0]["postclip_contrast"] == pytest.approx(-0.3, abs=1e-6)
    assert rows[0]["postclip_contrast_sign_flip"]
    assert rows[0]["postclip_contrast_retention_ratio"] == pytest.approx(0.6, abs=1e-6)


def test_low_contrast_targets_retained_but_not_e1_eligible(config: dict) -> None:
    target, image = target_and_image()
    image[:, :, 11:13, 11:13] = 0.401
    invisible = torch.full_like(image, 0.4)
    _, rows = audit.audit_visibility(artifacts(image, invisible), target, visibility=config["visibility"])
    summary = audit.summarize_targets(rows, config["visibility"])
    assert summary["target_count"] == 1
    assert summary["eligible_target_count"] == 0
    assert summary["raw_all_defined_contrast_failure_fraction"] == 1
    assert summary["e1_visibility_risk_triggered"] is None


def test_e1_fraction_threshold_is_strict_and_independent_of_clipping(config: dict) -> None:
    target, image = target_and_image()
    _, valid = audit.audit_visibility(artifacts(image, image), target, visibility=config["visibility"])
    rows = [deepcopy(valid[0]) for _ in range(10)]
    rows[0]["postclip_contrast_below_ratio_threshold"] = True
    rows[0]["target_clipping_warning"] = True
    assert audit.summarize_targets(rows, config["visibility"])["e1_visibility_risk_triggered"] is False
    rows[1]["postclip_contrast_below_ratio_threshold"] = True
    result = audit.summarize_targets(rows, config["visibility"])
    assert result["e1_visibility_risk_triggered"] is True
    assert result["target_clipping_warning_count"] == 1


def test_clipping_reports_preclip_excursion_and_rgb_endpoint_delta(config: dict) -> None:
    target, clean = target_and_image()
    preclip = clean.clone()
    preclip[:, 0, 11:13, 11:13] = 1.3
    clipped = preclip.clamp(0, 1)
    stats, rows = audit.audit_visibility(artifacts(clean, clipped, preclip), target,
                                         visibility=config["visibility"])
    assert rows[0]["target_rgb_channel_endpoint_fraction_increase"] == pytest.approx(1 / 3)
    assert rows[0]["target_clipping_warning"]
    assert rows[0]["preclip_target_clipping"]["rgb_channel_out_of_range_fraction"] == pytest.approx(1 / 3)
    assert rows[0]["postclip_target_clipping"]["gray_endpoint_pixel_fraction"] == 0
    assert stats["preclip_max"] > 1 and stats["postclip_max"] == 1


def test_preflight_is_zero_writes_and_does_not_load_pixels(tmp_path: Path, config: dict, monkeypatch) -> None:
    from analysis import d0a_v7_common as common
    config["result_root"] = str(tmp_path / "not_created")
    monkeypatch.setattr(common, "read_config", lambda _: config)
    monkeypatch.setattr(common, "load_pilot_records", lambda dataset, cfg: [{"image_id": str(i)} for i in range(64)])
    monkeypatch.setattr(common, "runtime_bindings", lambda *args: [])

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not decode images, write, or initialize CUDA")

    monkeypatch.setattr(common, "load_sample", forbidden)
    monkeypatch.setattr(common, "reserve_output", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    result = audit.run(argparse.Namespace(config=tmp_path / "config.yaml", dataset="NUDT-SIRST",
        execute=False, device="cuda:0", threads=1))
    assert result["writes"] == 0 and result["ready"]
    assert result["expected_per_image_rows"] == 384
    assert list(tmp_path.iterdir()) == []


def test_config_drift_is_rejected_before_diagnostics(config: dict) -> None:
    audit.validate_diagnostic_parameters(config)
    config["probes"]["lf_mask"]["keep_probability"] = 0.9
    with pytest.raises(ValueError, match="unchanged frozen"):
        audit.validate_diagnostic_parameters(config)


def test_clipping_warning_strictly_exceeds_point_one(config: dict) -> None:
    image = torch.full((1, 3, 24, 24), 0.5)
    target = torch.zeros((1, 24, 24))
    target[:, 10:12, 10:15] = 1
    after = image.clone()
    after[:, 0, 10, 10:13] = 1
    _, rows = audit.audit_visibility(artifacts(image, after), target, visibility=config["visibility"])
    assert rows[0]["target_rgb_channel_endpoint_fraction_increase"] == 0.1
    assert not rows[0]["target_clipping_warning"]


def test_synthetic_execution_seals_all_six_conditions_and_refuses_overwrite(
    tmp_path: Path, config: dict, monkeypatch
) -> None:
    from analysis import d0a_v7_common as common
    config["result_root"] = str(tmp_path / "diagnostics")
    monkeypatch.setattr(common, "read_config", lambda _: config)
    monkeypatch.setattr(common, "load_pilot_records", lambda dataset, cfg: [{"image_id": "synthetic"}])
    monkeypatch.setattr(common, "runtime_bindings", lambda *args: [])
    target, image = target_and_image()
    normalized = imagenet_normalize(image)[0]

    def sample(record, view, cfg, access):
        access["synthetic_only"] = access.get("synthetic_only", 0) + 1
        access["train_image_opens"] += 1
        access["train_mask_opens"] += 1
        return normalized.clone(), target.clone(), {"view": view, "synthetic_only": True}

    monkeypatch.setattr(common, "load_sample", sample)
    args = argparse.Namespace(config=tmp_path / "config.yaml", dataset="NUDT-SIRST",
                              execute=True, device="cpu", threads=1)
    result = audit.run(args)
    output = Path(result["output_dir"])
    assert result["complete"]
    assert result["per_image_rows"] == 6 and result["per_target_rows"] == 6
    assert len(result["cells"]) == 6
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["complete"]
    assert not complete["paper_result"] and not complete["new_training_authorized"]
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert common.sha256_file(output / name) == digest
    with pytest.raises(FileExistsError, match="existing diagnostic output"):
        audit.run(args)


def test_payload_access_assertion_requires_128_train_opens_and_zero_test_validation() -> None:
    access = {name: 0 for name in audit.FORBIDDEN_ACCESS_FIELDS}
    access.update({"train_image_opens": 128, "train_mask_opens": 128})
    audit.validate_access_counts(access, 64)
    access["train_image_opens"] = 127
    with pytest.raises(RuntimeError, match="expected 128"):
        audit.validate_access_counts(access, 64)
    access["train_image_opens"] = 128
    access["test_mask_opens"] = 1
    with pytest.raises(RuntimeError, match="nonzero or missing"):
        audit.validate_access_counts(access, 64)
