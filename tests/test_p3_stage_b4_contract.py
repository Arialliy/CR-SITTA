from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch
import yaml

from analysis.p3_stage_b4_science_gate_v1 import CELL_FIELDS, EPISODE_FIELDS
from scripts import run_p3_stage_b4_full_pilot64_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3_stage_b4_full_pilot64_proposal_gate_v1.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _schema_contract() -> runner.FullPilotContract:
    payload = CONFIG.read_bytes()
    config_sha256 = hashlib.sha256(payload).hexdigest()
    assert config_sha256 == runner.FROZEN_CONFIG_SHA256
    raw = yaml.safe_load(payload.decode("utf-8"))
    assert isinstance(raw, dict)
    return runner.FullPilotContract(ROOT, CONFIG, config_sha256, raw)


def _counts() -> dict[str, int]:
    return {
        "intersection_pixels": 3,
        "false_positive_pixels": 1,
        "false_negative_pixels": 2,
        "true_negative_pixels": 65530,
        "predicted_positive_pixels": 4,
        "target_positive_pixels": 5,
        "detected_targets": 1,
        "total_targets": 1,
        "false_alarm_pixels": 1,
        "total_image_pixels": 65536,
        "image_count": 1,
    }


def test_frozen_contract_is_train_only_and_exactly_b3_authorized() -> None:
    contract = _schema_contract()
    assert contract.config_sha256 == runner.FROZEN_CONFIG_SHA256
    assert contract.candidate_ids == runner.CANDIDATES
    assert tuple(contract.raw["datasets"]) == runner.DATASETS
    assert contract.raw["scope"]["split_name"] == "train"
    assert contract.raw["scope"]["no_validation_split"] is True
    assert contract.raw["scope"]["image_count_per_condition"] == 64
    assert contract.raw["scope"]["use_validation_payload"] is False
    assert contract.raw["scope"]["use_test_payload"] is False
    assert contract.raw["scope"]["method_label_accesses"] == 0
    assert contract.raw["stage_transition"]["formal_test_allowed_by_b4"] is False


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


def test_proposal_and_safety_values_are_fully_numeric_and_frozen() -> None:
    contract = _schema_contract()
    proposal = contract.raw["proposal"]
    assert proposal["optimizer_object_forbidden"] is True
    assert proposal["backtracking_coefficients"] == [1.0, 0.5, 0.25, 0.125]
    assert proposal["armijo_c"] == 1.0e-4
    assert proposal["all_rejected_action"] == "exact_no_update"
    safety = proposal["per_attempt_label_free_safety"]
    assert safety["reliable_background_mass_delta_maximum"] == 0.0001
    assert safety["predicted_positive_fraction_delta_maximum"] == 0.001
    components = safety["connected_component_count"]
    assert components["maximum_formula"] == (
        "max_source_plus_absolute_allowance_or_ceil_source_times_multiplier"
    )
    assert components["absolute_allowance"] == 3
    assert components["source_multiplier"] == 2.0
    assert contract.raw["parameter_spaces"]["P2"]["relative_radius"] == 0.0005
    assert contract.raw["parameter_spaces"]["DecoderFiLM"]["absolute_radius"] == 0.25


def test_gate_projections_have_exact_schemas_and_integer_counts() -> None:
    cell = runner._gate_cell_summary(
        {
            "candidate_id": "O3_P2",
            "dataset": "IRSTD-1K",
            "condition": "gaussian_noise_S3",
            "corruption_family": "gaussian_noise",
            "severity": 3,
            "episode_count": 64,
            "source_counts": _counts(),
            "adapted_counts": _counts(),
            "ignored_audit_field": "not forwarded",
        }
    )
    assert set(cell) == set(CELL_FIELDS)
    assert cell["severity"] == "S3"
    assert all(type(value) is int for value in cell["source_counts"].values())
    episode = runner._gate_episode_summary(
        {
            "candidate_id": "O3_P2",
            "dataset": "IRSTD-1K",
            "condition": "gaussian_noise_S3",
            "episode_index": 7,
            "proxy_gradient_nonzero": True,
            "task_gradient_nonzero": False,
            "gradient_cosine": 0.0,
            "accepted_update": False,
            "finite": True,
            "maximum_absolute_logit_delta": 0.0,
            "proposal_loss_before": 1.0,
            "proposal_loss_after": 1.0,
            "threshold_crossing_count": 0,
            "image_id": "outer-audit-only",
        }
    )
    assert set(episode) == set(EPISODE_FIELDS)
    assert "image_id" not in episode
    assert episode["gradient_cosine"] == 0.0


def test_label_free_diagnostics_use_strict_threshold_and_component_formula() -> None:
    contract = _schema_contract()
    source = torch.full((1, 1, 256, 256), -1.0)
    post = source.clone()
    post[0, 0, 20, 20] = 1.0
    teacher = torch.sigmoid(source)
    background = torch.ones_like(source)
    value = runner._episode_diagnostics(
        contract=contract,
        source_logits=source,
        post_logits=post,
        teacher=teacher,
        foreground_weight=torch.ones_like(source),
        background_weight=background,
    )
    assert value["source_connected_component_count"] == 0
    assert value["post_connected_component_count"] == 1
    assert value["connected_component_count_maximum"] == 3
    assert value["threshold_crossing_count"] == 1
    assert value["functional_logit_change_above_threshold"] is True
