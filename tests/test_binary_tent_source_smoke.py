from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import run_binary_tent_source_smoke as smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _prediction(logits: np.ndarray) -> smoke.PredictionArrays:
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    probability = 1.0 / (1.0 + np.exp(-logits))
    probability = np.ascontiguousarray(probability, dtype=np.float32)
    mask = np.where(probability > 0.5, 255, 0).astype(np.uint8)
    return smoke.PredictionArrays(logits, probability, mask)


def _episode(
    source: np.ndarray,
    tent_pre: np.ndarray,
    tent_post: np.ndarray,
    *,
    step_norm: float,
) -> smoke.EpisodeArrays:
    return smoke.EpisodeArrays(
        predictions={
            "source_pre": _prediction(source),
            "tent_pre": _prediction(tent_pre),
            "tent_post": _prediction(tent_post),
        },
        step_norm=step_norm,
        diagnostics={},
        state={},
    )


def test_default_config_is_strictly_train_side_and_provisional() -> None:
    path, config = smoke.load_smoke_config(
        PROJECT_ROOT / "configs" / "binary_tent_source_smoke_v1.yaml"
    )
    assert path.is_file()
    assert config["scope"]["use_test_images"] is False
    assert config["scope"]["use_test_labels"] is False
    assert config["scope"]["tuning_allowed"] is False
    assert config["method"]["optimizer"] == {
        "name": "Adam",
        "learning_rate": 1e-3,
        "status": "provisional_implementation_smoke_only",
        "selected_by_this_smoke": False,
    }
    assert config["execution"]["canonical_repeat_runs"] == 3
    assert config["execution"]["device_type"] == "cuda"
    deterministic = config["execution"]["deterministic_algorithms"]
    assert deterministic["policy"] == "strict_forwards_temporary_backward_disable"
    assert deterministic["forward_enabled"] is True
    assert deterministic["backward_enabled"] is False
    assert deterministic["post_forward_enabled"] is True
    scope_checks = smoke.protocol_scope_gate_checks(config)
    assert scope_checks
    assert all(value is True for value in scope_checks.values())
    assert "paper_result_is_false" in scope_checks
    assert "scientific_result_is_not_frozen" in scope_checks


def test_frozen_adabn_reference_contract_validates_without_pixel_execution() -> None:
    config_path, config = smoke.load_smoke_config(
        PROJECT_ROOT / "configs" / "binary_tent_source_smoke_v1.yaml"
    )
    paths = smoke.resolve_smoke_paths(config_path, config)
    source_config, _, selected_ids, _, contract, reference = (
        smoke.validate_frozen_source_reference(config, paths)
    )
    assert source_config["dataset"]["name"] == "IRSTD-1K"
    assert len(selected_ids) == 32
    assert contract["selected_ids_absent_from_fixed_test"] is True
    assert contract["test_images_opened"] == 0
    assert contract["test_masks_opened"] == 0
    assert reference["all_64_referenced_probability_files_hash_verified"] is True


def test_validate_only_does_not_publish_or_materialize(tmp_path: Path) -> None:
    result = smoke.run_source_smoke(
        Namespace(
            config=PROJECT_ROOT / "configs" / "binary_tent_source_smoke_v1.yaml",
            output_dir=tmp_path / "must_not_exist",
            device="cpu",
            validate_only=True,
        )
    )
    assert result["validate_only"] is True
    assert result["validation"]["test_images_opened"] == 0
    assert not (tmp_path / "must_not_exist").exists()


def test_runtime_policy_keeps_strict_forwards_and_is_exactly_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = smoke._capture_runtime_policy()

    def fake_seed_everything(_seed: int) -> None:
        os.environ["PYTHONHASHSEED"] = str(_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    monkeypatch.setattr(smoke.source_runner, "seed_everything", fake_seed_everything)
    try:
        with smoke.binary_tent_cuda_policy(
            seed=17, workspace_config=":4096:8"
        ) as record:
            assert record["active"]["deterministic_algorithms_enabled"] is True
            assert record["active"]["deterministic_algorithms_warn_only"] is False
            assert record["active"]["cudnn_deterministic"] is True
            assert record["active"]["cudnn_benchmark"] is False
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
            assert record["episode_backward_policy"] == "temporarily_disable"
            assert record["backward_disable_scope"] == "entropy_backward_only"
        assert smoke.runtime_policy_is_restored(record["before"])
        assert smoke._capture_runtime_policy() == before
    finally:
        smoke._restore_runtime_policy(before)


def test_compare_passes_measures_post_drift_and_step_norm_only() -> None:
    zeros = np.zeros((2, 2), dtype=np.float32)
    negative = np.full((2, 2), -0.25, dtype=np.float32)
    canonical = {
        "a": _episode(negative, negative, negative, step_norm=0.50),
        "b": _episode(negative, negative, negative, step_norm=0.25),
    }
    changed = negative.copy()
    changed[0, 0] = 0.25
    candidate = {
        "a": _episode(negative, negative, changed, step_norm=0.55),
        "b": _episode(negative, negative, negative + 0.01, step_norm=0.20),
    }
    comparison = smoke.compare_episode_passes(
        canonical, candidate, selected_ids=("a", "b")
    )
    assert comparison["aggregate"]["source_pre"]["logits_bit_exact_images"] == 2
    assert comparison["aggregate"]["tent_pre"]["probability_bit_exact_images"] == 2
    post = comparison["aggregate"]["tent_post"]
    assert post["logits_bit_exact_images"] == 0
    assert post["logits_global_max_abs_difference"] == pytest.approx(0.5)
    assert post["mask_disagreement_pixels"] == 1
    assert post["mask_disagreement_rate"] == pytest.approx(1 / 8)
    assert comparison["aggregate"]["optimizer_step_norm"][
        "global_max_absolute_difference"
    ] == pytest.approx(0.05)
    assert zeros.shape == (2, 2)


def test_tolerance_candidate_is_labelled_non_gating_and_covers_observed() -> None:
    comparison = {
        "aggregate": {
            "tent_post": {
                "logits_global_max_abs_difference": 0.01,
                "probability_global_mean_abs_difference": 0.001,
                "probability_global_max_abs_difference": 0.005,
                "mask_disagreement_rate": 0.002,
            },
            "optimizer_step_norm": {"global_max_absolute_difference": 0.003},
        }
    }
    result = smoke.derive_train_tolerance_candidate([comparison])
    assert result["formal_acceptance_gate"] is False
    assert result["scientific_result_frozen"] is False
    assert result["selection_or_tuning_performed"] is False
    for key, observed in result["observed_envelope"].items():
        assert result["candidate_envelope"][key] >= observed


def test_metric_differences_are_signed_candidate_minus_canonical() -> None:
    def payload(offset: float) -> dict[str, object]:
        return {
            name: {
                evaluator: {
                    "iou": 0.5 + offset,
                    "pd": 0.6 + offset,
                    "fa_per_million_pixels": 10.0 + offset,
                }
                for evaluator in (
                    "official_nsfpn_operating_point",
                    "unified_fixed_probability_0_5",
                )
            }
            for name in smoke.PREDICTION_NAMES
        }

    result = smoke.metric_differences(payload(0.02), payload(0.0))
    assert result["tent_post"]["official_nsfpn_operating_point"][
        "iou"
    ] == pytest.approx(0.02)
    assert result["source_pre"]["unified_fixed_probability_0_5"][
        "fa_per_million_pixels"
    ] == pytest.approx(0.02)


def test_finalize_publishes_only_after_required_files_and_gates(tmp_path: Path) -> None:
    staging = tmp_path / ".artifact.build"
    final = tmp_path / "artifact"
    required = (
        "run_config.yaml",
        "provenance.json",
        "aggregate_metrics.json",
        "batch_stats/metrics.json",
        "batch_stats/per_image.jsonl",
        "batch_stats/adaptation_diagnostics.jsonl",
        "source_stats/metrics.json",
        "source_stats/per_image.jsonl",
        "source_stats/adaptation_diagnostics.jsonl",
    )
    for relative in required:
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    complete = smoke.finalize_and_publish(
        staging=staging,
        final_root=final,
        protocol_id=smoke.EXPECTED_PROTOCOL_ID,
        protocol_sha256="a" * 64,
        checks={"gate": True},
    )
    assert complete["complete"] is True
    assert final.is_dir()
    assert (final / "artifact_manifest.json").is_file()
    assert (final / "COMPLETE.json").is_file()
    assert not staging.exists()


def test_finalize_refuses_failed_gate(tmp_path: Path) -> None:
    staging = tmp_path / ".failed.build"
    staging.mkdir()
    with pytest.raises(RuntimeError, match="hard gates failed"):
        smoke.finalize_and_publish(
            staging=staging,
            final_root=tmp_path / "must_not_publish",
            protocol_id=smoke.EXPECTED_PROTOCOL_ID,
            protocol_sha256="b" * 64,
            checks={"gate": False},
        )
    assert not (tmp_path / "must_not_publish").exists()
