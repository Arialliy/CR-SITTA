from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path

import pytest
import torch

import run_p3_stage_c0_signal_audit_v1 as runner
from analysis.stage_c0_active_support import audit_active_support
from analysis.stage_c0_group_alignment import (
    analyze_group_alignment,
    clipped_descent_direction,
)
from analysis.stage_c0_teacher_student_gap import measure_teacher_student_gap
from analysis.stage_c_science_gate_v1 import (
    authorize_stage_c_followup,
    default_gate_config,
    evaluate_stage_c0_science_gate,
)
from tta.deteriorations.image_space import imagenet_normalize


LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def contract(monkeypatch: pytest.MonkeyPatch):
    if not LOCAL_ARTIFACT_TESTS_ENABLED:
        pytest.skip(
            "requires the frozen local Stage-C environment; set "
            f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
        )
    digest = runner.current_config_sha256()
    monkeypatch.setattr(runner, "FROZEN_CONFIG_SHA256", digest)
    return runner.load_contract()


def test_frozen_contract_is_train_only_and_has_two_unique_probes(contract) -> None:
    assert contract.raw["scope"]["split_name"] == "train"
    assert contract.raw["scope"]["no_validation_split"] is True
    assert contract.raw["scope"]["use_validation_payload"] is False
    assert contract.raw["scope"]["use_test_payload"] is False
    assert contract.raw["scope"]["label_free_probe_episode_count_per_dataset"] == 1664
    assert contract.raw["scope"]["target_payload_deserialization_count_per_dataset"] == 1
    assert contract.raw["scope"]["unique_image_condition_target_uses_per_dataset"] == 832
    assert contract.raw["scope"]["probe_episode_target_uses_per_dataset"] == 1664
    assert tuple(value["id"] for value in contract.probe_records) == (
        "lf_mask",
        "hf_noise",
    )
    assert contract.raw["parameter_spaces"]["ordered"] == [
        "R-E1",
        "R-D0",
        "P2",
    ]
    assert "test-selected" in contract.raw["scope"][
        "upstream_checkpoint_selection_disclosure"
    ]
    gate = contract.raw["stage_c0_signal_gate"]
    assert gate["eligibility_evaluated_before_followup_authorization"] is True
    assert gate["ranking_at_c0"] == "forbidden"
    assert "filter_before_ranking" not in gate


def test_pre_run_freeze_receipt_contract_is_complete_and_ordered(contract) -> None:
    expected = runner._expected_pre_run_freeze_receipt(contract)
    assert set(expected) == {
        "schema_version",
        "artifact_type",
        "protocol",
        "science_gate",
        "critical_code_sha256",
        "datasets",
        "created_before_formal_execution",
        "formal_phase_directories_absent_at_publication",
        "development_only",
        "paper_result",
        "validation_payload_opens",
        "test_payload_opens",
    }
    assert expected["protocol"]["config_sha256"] == contract.config_sha256
    assert tuple(expected["critical_code_sha256"]) == tuple(
        contract.raw["implementation"]["critical_code_paths"]
    )
    assert tuple(expected["datasets"]) == runner.DATASETS
    assert expected["created_before_formal_execution"] is True
    assert expected["validation_payload_opens"] == 0
    assert expected["test_payload_opens"] == 0
    assert "freeze" in runner._parser()._subparsers._group_actions[0].choices


def test_teacher_student_gap_is_finite_label_free_and_exact_for_match() -> None:
    teacher = torch.tensor([[[[0.2, 0.8], [0.5, 0.1]]]], dtype=torch.float32)
    student = torch.logit(teacher).requires_grad_(True)
    result = measure_teacher_student_gap(teacher, student)
    assert result.finite
    assert result.teacher_student_gap_l1 == pytest.approx(0.0, abs=1e-7)
    assert result.teacher_student_logit_gap_max == pytest.approx(0.0, abs=1e-6)
    attached_teacher = teacher.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="detached"):
        measure_teacher_student_gap(attached_teacher, student)


def test_active_support_counts_candidate_near_and_far_without_gt() -> None:
    active = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])
    target_weight = torch.tensor([[[[0.5, 0.0], [0.0, 0.0]]]])
    background_weight = torch.tensor([[[[0.0, 0.8], [1.0, 0.9]]]])
    near = torch.tensor([[[[True, False], [False, False]]]])
    result = audit_active_support(
        active.detach(),
        target_weight.detach(),
        background_weight.detach(),
        near.detach(),
    )
    assert result.active_episode
    assert result.active_pixel_count == 3
    assert result.active_near_candidate_pixel_count == 1
    assert result.active_far_background_pixel_count == 2
    assert not result.active_only_far_background
    assert result.active_target_weight == pytest.approx(0.5)
    assert result.active_background_weight == pytest.approx(1.7)


def test_clipped_virtual_step_and_directional_derivatives_follow_frozen_formula() -> None:
    proxy = torch.tensor([3.0, 4.0])
    task = torch.tensor([3.0, 4.0])
    direction = clipped_descent_direction(proxy, radius=2.0, nonzero_epsilon=1e-12)
    assert direction.tolist() == pytest.approx([-1.2, -1.6])
    result = analyze_group_alignment(
        proxy,
        task,
        candidate_absolute_gradient=torch.tensor([1.0, 0.0]),
        candidate_contrast_gradient=torch.tensor([0.0, 1.0]),
        virtual_step_radius=2.0,
    )
    assert result.outer_task_gradient_cosine == pytest.approx(1.0)
    assert result.virtual_step_norm == pytest.approx(2.0)
    assert result.normalized_virtual_step_task_directional_derivative == pytest.approx(-10.0)
    assert result.normalized_virtual_step_task_directional_derivative == pytest.approx(
        -result.outer_task_gradient_cosine
        * result.task_gradient_norm
        * result.virtual_step_norm
    )
    assert result.candidate_absolute_response_derivative == pytest.approx(-1.2)
    assert result.candidate_local_contrast_derivative == pytest.approx(-1.6)
    # A gradient already inside the radius is not enlarged.
    inside = clipped_descent_direction(torch.tensor([0.3, 0.4]), radius=2.0)
    assert inside.tolist() == pytest.approx([-0.3, -0.4])


def test_probe_seed_and_images_are_reproducible_without_condition_routing(contract) -> None:
    input_hash = hashlib.sha256(b"image-a").hexdigest()
    seed_a = runner.derive_probe_seed(
        global_seed=42,
        image_id="image-a",
        probe_id="lf_mask",
        input_tensor_sha256=input_hash,
    )
    seed_b = runner.derive_probe_seed(
        global_seed=42,
        image_id="image-a",
        probe_id="lf_mask",
        input_tensor_sha256=input_hash,
    )
    assert seed_a == seed_b
    assert "condition" not in inspect.signature(runner.derive_probe_seed).parameters
    assert "corruption" not in inspect.signature(runner.build_probe_image).parameters
    physical = torch.linspace(0.0, 1.0, 3 * 16 * 16).reshape(1, 3, 16, 16)
    normalized = imagenet_normalize(physical)
    left = runner.build_probe_image(
        normalized,
        contract.probe_records[0],
        generator=torch.Generator().manual_seed(seed_a),
    )
    right = runner.build_probe_image(
        normalized,
        contract.probe_records[0],
        generator=torch.Generator().manual_seed(seed_b),
    )
    assert torch.equal(left, right)
    assert left.shape == normalized.shape


class _FakeTorch:
    def __init__(self) -> None:
        self.enabled = True
        self.warn_only = True
        self.calls: list[tuple[bool, bool]] = []

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.enabled

    def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
        return self.warn_only

    def use_deterministic_algorithms(
        self, enabled: bool, *, warn_only: bool = False
    ) -> None:
        self.enabled = enabled
        self.warn_only = warn_only
        self.calls.append((enabled, warn_only))


def test_scoped_cuda_backward_allowance_restores_on_exception() -> None:
    fake = _FakeTorch()
    tensor = type("CudaTensor", (), {"is_cuda": True})()
    with pytest.raises(RuntimeError, match="boom"):
        with runner._scoped_cuda_backward_allowance(fake, tensor=tensor):
            assert fake.enabled is False
            raise RuntimeError("boom")
    assert fake.enabled is True
    assert fake.warn_only is True
    assert fake.calls == [(False, False), (True, True)]


def _toy_outer_records(dataset: str) -> list[dict]:
    records: list[dict] = []
    for condition_index, (corruption, severity) in enumerate(runner.CONDITIONS):
        for image_index in range(64):
            for probe_index, probe in enumerate(runner.PROBE_IDS):
                spaces = {
                    space: {
                        "proxy_gradient_norm": 1.0,
                        "task_gradient_norm": 1.0,
                        "both_gradients_nonzero": True,
                        "outer_task_gradient_cosine": 0.2,
                        "normalized_virtual_step_task_directional_derivative": -0.1,
                        "candidate_absolute_response_derivative": 0.01,
                        "candidate_local_contrast_derivative": 0.01,
                        "virtual_step_norm": 0.1,
                        "absolute_l2_radius": 0.1,
                        "threshold_crossing_episode": image_index % 10 == 0,
                    }
                    for space in runner.PARAMETER_SPACES
                }
                records.append(
                    {
                        "dataset": dataset,
                        "condition": runner._condition_key(corruption, severity),
                        "corruption": corruption,
                        "severity": severity,
                        "image_index": image_index,
                        "probe_index": probe_index,
                        "probe_id": probe,
                        "teacher_student_gap_l1": 0.1,
                        "teacher_student_logit_gap_mean": 0.2,
                        "teacher_student_logit_gap_max": 0.3,
                        "active_episode": True,
                        "active_pixel_count": 10,
                        "active_pixel_fraction": 10 / 65536,
                        "active_target_weight": 2.0,
                        "active_background_weight": 3.0,
                        "candidate_proximal_active_episode": True,
                        "candidate_proximal_active_pixel_count": 2,
                        "total_pixel_count": 65536,
                        "parameter_space_evidence": spaces,
                    }
                )
    return records


def test_toy_aggregate_matches_pure_gate_schema_and_authorizes_only_c1() -> None:
    gate = default_gate_config()
    records = {dataset: _toy_outer_records(dataset) for dataset in runner.DATASETS}
    identities = {
        dataset: {
            "comparison_count": 64,
            "exact_match_count": 64,
            "mismatch_count": 0,
            "maximum_absolute_output_difference": 0.0,
        }
        for dataset in runner.DATASETS
    }
    evidence = runner.build_aggregate_evidence(
        records, identities, gate_config=gate
    )
    assert evidence["mechanism_aggregate"]["nonclean_probe_episode_count"] == 4608
    receipt = evaluate_stage_c0_science_gate(evidence, gate)
    assert receipt.eligible_space_ids == ("R-E1", "R-D0", "P2")
    authorization = authorize_stage_c_followup(receipt)
    assert authorization.stage_c1_allowed
    assert not authorization.stage_c_r1_r2_allowed
    assert not authorization.formal_test_allowed
