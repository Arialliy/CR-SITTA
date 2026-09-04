from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from analysis.proxy_objective_space_screen_v1 import CELL_FIELDS
from scripts import run_p3_stage_b_screen_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3_stage_b_objective_space_screen_v1.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _schema_contract() -> runner.ScreenContract:
    payload = CONFIG.read_bytes()
    config_sha256 = hashlib.sha256(payload).hexdigest()
    assert config_sha256 == runner.FROZEN_CONFIG_SHA256
    raw = yaml.safe_load(payload.decode("utf-8"))
    assert isinstance(raw, dict)
    return runner.ScreenContract(ROOT, CONFIG, config_sha256, raw)


def _step_contract() -> SimpleNamespace:
    return SimpleNamespace(
        raw={
            "parameter_spaces": {
                "P2": {"trust_radius": 0.1},
                "DecoderFiLM": {"trust_radius": 0.25},
            }
        }
    )


def test_frozen_contract_is_train_only_and_has_exact_candidate_roster() -> None:
    contract = _schema_contract()
    assert contract.candidate_ids == runner.CANDIDATES
    assert contract.raw["scope"]["split_name"] == "train"
    assert contract.raw["scope"]["use_validation_payload"] is False
    assert contract.raw["scope"]["use_test_payload"] is False
    assert contract.raw["scope"]["method_label_accesses"] == 0
    assert contract.raw["stage_transition"]["stage_b4_allowed_before_b3_gate"] is False


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local Stage-B parent artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_live_contract_and_all_frozen_parent_bindings_verify() -> None:
    contract = runner.load_contract(CONFIG)
    assert contract.config_sha256 == runner.FROZEN_CONFIG_SHA256
    assert contract.candidate_ids == runner.CANDIDATES


def test_b3_p2_step_uses_fixed_relative_normalized_radius() -> None:
    parameter = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    gradient = torch.tensor([3.0e-8, 4.0e-8])
    step, gradient_norm, step_norm = runner._normalized_step(
        _step_contract(), "P2", (("p", parameter),), gradient
    )
    expected_radius = 0.1 * (5.0 + 1.0e-12)
    assert gradient_norm == pytest.approx(5.0e-8)
    assert step_norm == pytest.approx(expected_radius, rel=3.0e-5)
    assert torch.dot(step, gradient).item() < 0.0
    # This distinguishes B3 fixed normalization from B4 min-clipping.
    assert step_norm > 1_000_000.0 * gradient_norm


def test_b3_film_step_uses_fixed_absolute_normalized_radius() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    gradient = torch.tensor([3.0, 4.0])
    step, gradient_norm, step_norm = runner._normalized_step(
        _step_contract(), "DecoderFiLM", (("p", parameter),), gradient
    )
    assert gradient_norm == pytest.approx(5.0)
    assert step_norm == pytest.approx(0.25)
    assert torch.equal(step, torch.tensor([-0.15, -0.20]))


def test_gate_projection_is_exact_and_normalizes_severity() -> None:
    rich = {
        "candidate_id": "O2_P2",
        "dataset": "IRSTD-1K",
        "condition": "gaussian_noise_S3",
        "corruption_family": "gaussian_noise",
        "severity": 3,
        "valid_alignment_episode_count": 16,
        "normalized_task_directional_derivative": -0.25,
        "source_iou": 0.4,
        "adapted_iou": 0.41,
        "delta_pd": 0.01,
        "delta_fa_per_million": -2.0,
        "source_foreground_fraction": 0.001,
        "adapted_foreground_fraction": 0.0011,
        "episode_count": 16,
        "functional_changed_episode_count": 16,
        "threshold_crossing_count": 3,
        "ignored_rich_field": "audit-only",
    }
    projected = runner._gate_cell_summary(rich)
    assert set(projected) == CELL_FIELDS
    assert projected["severity"] == "S3"
    assert projected["delta_fa"] == -2.0


def test_zero_valid_alignment_projects_to_zero_not_missing() -> None:
    rich = {
        "candidate_id": "O4_DecoderFiLM",
        "dataset": "NUDT-SIRST",
        "condition": "clean_S0",
        "corruption_family": "clean",
        "severity": 0,
        "valid_alignment_episode_count": 0,
        "normalized_task_directional_derivative": None,
        "source_iou": 0.5,
        "adapted_iou": 0.5,
        "delta_pd": 0.0,
        "delta_fa_per_million": 0.0,
        "source_foreground_fraction": 0.0,
        "adapted_foreground_fraction": 0.0,
        "episode_count": 16,
        "functional_changed_episode_count": 0,
        "threshold_crossing_count": 0,
    }
    projected = runner._gate_cell_summary(rich)
    assert projected["severity"] == "S0"
    assert projected["normalized_task_directional_derivative"] == 0.0


def test_nonzero_valid_alignment_cannot_omit_derivative() -> None:
    with pytest.raises(runner.StageB3ProtocolError, match="cannot omit"):
        runner._gate_cell_summary(
            {
                "valid_alignment_episode_count": 1,
                "normalized_task_directional_derivative": None,
            }
        )
