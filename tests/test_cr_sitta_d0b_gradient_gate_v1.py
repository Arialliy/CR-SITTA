from __future__ import annotations

import ast
import copy
import os
from pathlib import Path

import pytest
import yaml

import run_cr_sitta_d0b_gradient_gate_v1 as runner
from analysis.cr_sitta_d0b_gate import authorize_d1, build_d0b_science_receipt
from analysis.stage_c_science_gate_v1 import evaluate_stage_c0_science_gate


LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = REPOSITORY / "configs/cr_sitta_d0b_gradient_gate_v1.yaml"
DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST")
FAMILIES = (
    "gaussian_noise",
    "gaussian_blur",
    "low_contrast",
    "stripe_noise",
)
SPACES = ("R-E1", "R-D0", "P2")


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _step(space: str) -> dict:
    return {
        "radius_mode": (
            "relative_to_source_parameter_l2" if space == "P2" else "absolute_l2"
        ),
        "radius_value": "0.0005" if space == "P2" else "0.25",
        "nonzero_epsilon": "1e-12",
        "direction": "negative_proxy_gradient",
        "clip_rule": "min_1_radius_over_norm_plus_epsilon",
    }


def _space(space: str) -> dict:
    return {
        "parameter_space": space,
        "nonclean_probe_episode_count": 4608,
        "nonclean_nonfinite_measurement_episode_count": 0,
        "nonclean_finite_nonzero_proxy_gradient_episode_count": 4000,
        "nonclean_threshold_crossing_episode_count": 14,
        "proxy_gradient_norm_mean": "0.20",
        "task_gradient_norm_mean": "0.30",
        "candidate_absolute_response_derivative_mean": "-0.01",
        "candidate_local_contrast_derivative_mean": "-0.02",
        "dataset_aggregates": [
            {
                "dataset_id": dataset,
                "nonclean_probe_episode_count": 1536,
                "both_gradients_nonzero_episode_count": 1400,
                "macro_outer_gradient_cosine": "0.10",
            }
            for dataset in DATASETS
        ],
        "family_aggregates": [
            {
                "family_id": family,
                "nonclean_probe_episode_count": 1152,
                "normalized_virtual_task_loss_directional_derivative": derivative,
            }
            for family, derivative in zip(
                FAMILIES, ("-0.03", "-0.02", "-0.01", "0.01"), strict=True
            )
        ],
        "macro_outer_gradient_cosine": "0.10",
        "virtual_step_contract": _step(space),
        "identity": {
            "comparison_count": 192,
            "exact_match_count": 192,
            "mismatch_count": 0,
            "maximum_absolute_output_difference": "0",
        },
    }


def _eligible_evidence() -> dict:
    return {
        "schema_version": "stage_c0_aggregate_evidence_v1",
        "replicate_id": "R0",
        "scope": {
            "data_role": "train",
            "pilot_role": "fixed_Pilot64",
            "pilot_image_count_per_dataset": 64,
            "development_only": True,
            "paper_result": False,
            "thresholds_frozen_before_run": True,
            "validation_access_count": 0,
            "test_access_count": 0,
        },
        "mechanism_aggregate": {
            "mechanism_id": "ASB-SFR_C0",
            "nonclean_probe_episode_count": 4608,
            "nonclean_nonfinite_signal_episode_count": 0,
            "nonclean_active_support_episode_count": 2000,
            "nonclean_active_pixel_count": 100_000,
            "nonclean_candidate_proximal_active_episode_count": 200,
            "nonclean_candidate_proximal_active_pixel_count": 5_000,
            "nonclean_total_pixel_count": 4608 * 256 * 256,
            "signal_summary": {
                "teacher_student_probability_l1_mean": "0.03",
                "teacher_student_logit_gap_mean": "0.10",
                "teacher_student_logit_gap_max": "1.0",
                "active_pixel_fraction_lf_mean": "0.10",
                "active_pixel_fraction_hf_mean": "0.08",
                "active_target_weight_lf_mean": "0.40",
                "active_background_weight_lf_mean": "0.60",
                "active_target_weight_hf_mean": "0.35",
                "active_background_weight_hf_mean": "0.65",
            },
            "space_aggregates": [_space(space) for space in SPACES],
            "o4_activity": {
                "configured": False,
                "active_episode_count": 0,
                "contribution_episode_count": 0,
            },
        },
    }


def test_repository_contract_is_train_only_checkpoint_rebound() -> None:
    contract = runner.load_contract(CONFIG)
    raw = contract.raw
    assert raw["scope"]["method_stage"] == "D0-B"
    assert raw["scope"]["data_role"] == "train"
    assert raw["scope"]["no_validation_split"] is True
    assert raw["scope"]["use_validation_payload"] is False
    assert raw["scope"]["use_test_payload"] is False
    assert raw["artifact_policy"]["formal_test_allowed"] is False
    assert raw["artifact_policy"]["may_authorize_only"] == "D1_train_internal_OOF"
    for dataset in DATASETS:
        binding = raw["datasets"][dataset]
        assert binding["checkpoint_role"] == "epoch_1000_train_only_safe"
        assert binding["checkpoint_path"].endswith("epoch_1000_train_only_safe.pth.tar")
        assert binding["safe_export_receipt"].endswith("SAFE_EXPORT.json")
        assert binding["teacher_root"].startswith(raw["output"]["root"])
        assert "best_miou" not in binding["checkpoint_path"]
        assert "best_pd" not in binding["checkpoint_path"]


def test_every_checkpoint_dependent_artifact_is_declared_rebuilt() -> None:
    raw = _config()
    assert set(raw["artifact_policy"]["checkpoint_dependent_artifacts_must_be_rebuilt"]) == {
        "strong_teacher",
        "candidate_masks_and_region_weights",
        "source_probabilities",
        "proxy_gradients",
        "outer_task_gradients",
        "aggregate_evidence",
    }
    assert raw["artifact_policy"]["permitted_reuse"] == {
        "artifact": "image_only_13_condition_Pilot64_cache",
        "checkpoint_independent_proof_required": True,
        "rehash_and_rebind_in_pre_run_freeze": True,
    }
    for dataset in DATASETS:
        active_inputs = (
            raw["datasets"][dataset]["checkpoint_path"],
            raw["datasets"][dataset]["safe_export_receipt"],
            raw["datasets"][dataset]["cache_root"],
        )
        assert not any(
            value.startswith(forbidden)
            for value in active_inputs
            for forbidden in raw["artifact_policy"]["forbidden_input_roots"]
        )
    assert "export_cr_sitta_d0a_safe_checkpoint.py" in raw["implementation"][
        "critical_code_paths"
    ]


def test_scientific_thresholds_are_exactly_v7_and_crossing_is_report_only() -> None:
    gate = _config()["stage_c0_signal_gate"]
    assert gate["minimum_nonclean_finite_nonzero_proxy_gradient_fraction"] == 0.80
    assert gate["minimum_nonclean_active_support_episode_fraction"] == 0.30
    assert gate["minimum_candidate_proximal_active_episode_fraction_among_active"] == 0.05
    assert gate["minimum_positive_dataset_cosines"] == 2
    assert gate["macro_cosine_strictly_greater_than"] == 0.08
    assert gate["minimum_improving_families"] == 3
    assert "minimum_nonclean_threshold_crossing_episode_fraction" not in gate


def test_contract_rejects_test_selected_checkpoint_path() -> None:
    raw = _config()
    raw["datasets"]["IRSTD-1K"]["checkpoint_path"] = (
        "results/baseline/IRSTD-1K/best_miou.pth.tar"
    )
    with pytest.raises(runner.D0BProtocolError):
        runner._validate_contract_semantics(raw)


def test_preflight_requires_all_three_safe_exports_without_writes(monkeypatch) -> None:
    contract = runner.load_contract(CONFIG)
    calls: list[str] = []

    monkeypatch.setattr(runner, "_validate_contract_semantics", lambda _raw: None)
    monkeypatch.setattr(runner, "_verify_static_bindings", lambda _contract: None)
    monkeypatch.setattr(
        runner,
        "_verify_cache_and_split",
        lambda _contract, dataset: {"cache": dataset},
    )

    def safe(_contract, dataset):
        calls.append(dataset)
        if dataset == "NUAA-SIRST":
            raise runner.D0BProtocolError("fixture safe export missing")
        return {"checkpoint": {"path": dataset, "sha256": "a" * 64}}

    monkeypatch.setattr(runner, "_verify_safe_export", safe)
    before = set(contract.output_root.parent.glob("*")) if contract.output_root.parent.exists() else set()
    result = runner.inspect_preflight(contract)
    after = set(contract.output_root.parent.glob("*")) if contract.output_root.parent.exists() else set()
    assert calls == list(DATASETS)
    assert result["ready"] is False
    assert result["datasets"]["NUAA-SIRST"]["ready"] is False
    assert "fixture safe export missing" in result["datasets"]["NUAA-SIRST"]["error"]
    assert result["writes_performed"] == 0
    assert before == after


def test_preflight_success_requires_exactly_three_ready_exports(monkeypatch) -> None:
    contract = runner.load_contract(CONFIG)
    monkeypatch.setattr(runner, "_validate_contract_semantics", lambda _raw: None)
    monkeypatch.setattr(runner, "_verify_static_bindings", lambda _contract: None)
    monkeypatch.setattr(
        runner,
        "_verify_cache_and_split",
        lambda _contract, dataset: {"image_only_cache": {"dataset": dataset}},
    )
    monkeypatch.setattr(
        runner,
        "_verify_safe_export",
        lambda _contract, dataset: {
            "checkpoint": {"path": dataset, "sha256": "b" * 64}
        },
    )
    result = runner.inspect_preflight(contract)
    assert result["ready"] is True
    assert result["errors"] == []
    assert tuple(result["datasets"]) == DATASETS
    assert all(value["ready"] is True for value in result["datasets"].values())


def test_current_missing_safe_export_preflight_does_not_publish_freeze() -> None:
    """Stable until all real exports exist; skip automatically once they do."""

    contract = runner.load_contract(CONFIG)
    if all(
        (REPOSITORY / contract.raw["datasets"][dataset]["safe_export_receipt"]).is_file()
        for dataset in DATASETS
    ):
        pytest.skip("all three real safe exports now exist")
    freeze = runner._freeze_path(contract)
    existed = freeze.exists()
    result = runner.inspect_preflight(contract)
    assert result["ready"] is False
    assert freeze.exists() is existed


def test_d0b_gate_projects_positive_result_to_d1_only() -> None:
    raw = _eligible_evidence()
    before = copy.deepcopy(raw)
    receipt = evaluate_stage_c0_science_gate(
        raw, _config()["stage_c0_signal_gate"]
    )
    science = build_d0b_science_receipt(receipt)
    authorization = authorize_d1(receipt).to_receipt()
    assert raw == before
    assert science["scientific_status"] == "scientific_eligible"
    assert science["d1_train_internal_oof_allowed"] is True
    assert authorization["d1_train_internal_oof_allowed"] is True
    assert authorization["authorization_scope"] == "D1_train_internal_OOF_only"
    assert science["formal_test_allowed"] is False
    assert authorization["formal_test_allowed"] is False
    forbidden = {"stage_c1_allowed", "stage_c_r1_r2_allowed", "formal_test_authorized"}
    assert not forbidden.intersection(science)
    assert not forbidden.intersection(authorization)


def test_d0b_gate_negative_result_stops_before_d1_and_test() -> None:
    evidence = _eligible_evidence()
    for space in evidence["mechanism_aggregate"]["space_aggregates"]:
        space["nonclean_finite_nonzero_proxy_gradient_episode_count"] = 1
    receipt = evaluate_stage_c0_science_gate(
        evidence, _config()["stage_c0_signal_gate"]
    )
    science = build_d0b_science_receipt(receipt)
    authorization = authorize_d1(receipt).to_receipt()
    assert science["scientific_status"] == "scientific_no_eligible"
    assert science["d1_train_internal_oof_allowed"] is False
    assert authorization["d1_train_internal_oof_allowed"] is False
    assert authorization["reason"] == "scientific_no_eligible_stop"
    assert authorization["formal_test_allowed"] is False


def test_runner_has_no_formal_test_command_and_uses_no_replace_publishers() -> None:
    source = (REPOSITORY / "run_cr_sitta_d0b_gradient_gate_v1.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    subcommands = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_parser"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert "test" not in subcommands
    assert "formal-test" not in subcommands
    assert "publish_file_noreplace" in source
    assert "publish_directory_noreplace" in source
    assert "old_checkpoint_dependent_numeric_artifact_inputs" in source


def test_legacy_logic_reuse_is_hash_bound_and_numeric_artifacts_are_not_inputs() -> None:
    raw = _config()
    prior = raw["lineage"]["prior_c0_logic"]
    assert prior["reuse_scope"] == "formulas_and_verified_execution_logic_only"
    assert prior["numeric_artifact_reuse"] == "forbidden"
    assert runner.sha256_file(REPOSITORY / prior["path"]) == prior["sha256"]
    configured_text = CONFIG.read_text(encoding="utf-8")
    # Old roots occur only in the explicit denylist/lineage declaration, never
    # as dataset teacher/candidate/gradient inputs.
    for dataset in DATASETS:
        record = raw["datasets"][dataset]
        assert record["teacher_root"].startswith(raw["output"]["root"])
        assert "p3_stage_c0_signal_audit_v2" not in record["teacher_root"]
    assert "forbidden_input_roots" in configured_text


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local Stage-C/D0 provenance artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_static_preflight_never_opens_old_checkpoint_dependent_result(
    monkeypatch,
) -> None:
    contract = runner.load_contract(CONFIG)
    forbidden = (
        REPOSITORY
        / contract.raw["lineage"]["old_c0_negative_result"]["path"]
    ).resolve()
    real_verify = runner._verify_exact_file

    def reject_old_result(path, expected_sha256, label):
        assert Path(path).resolve() != forbidden
        return real_verify(path, expected_sha256, label)

    monkeypatch.setattr(runner, "_verify_exact_file", reject_old_result)
    runner._verify_static_bindings(contract)


def test_teacher_per_image_provenance_is_exact_and_label_free() -> None:
    record = {
        "condition": "gaussian_noise_S3",
        "image_index": 7,
        "image_id": "pilot-007",
        "input_tensor_sha256": "a" * 64,
        "base_view_probabilities_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
        "method_label_accesses": 0,
        "validation_payload_opens": 0,
        "test_payload_opens": 0,
    }
    expected = {
        "condition": "gaussian_noise_S3",
        "image_index": 7,
        "image_id": "pilot-007",
        "input_tensor_sha256": "a" * 64,
        "probability_sha256": "b" * 64,
        "checkpoint_sha256": "c" * 64,
    }
    runner._verify_teacher_per_image_record(record, **expected)
    for field in tuple(record):
        changed = dict(record)
        changed[field] = 1 if record[field] == 0 else "changed"
        with pytest.raises(runner.D0BProtocolError):
            runner._verify_teacher_per_image_record(changed, **expected)


def test_aggregate_provenance_requires_development_only_full_rebuild() -> None:
    required = (
        "strong_teacher",
        "candidate_masks_and_region_weights",
        "source_probabilities",
        "proxy_gradients",
        "outer_task_gradients",
        "aggregate_evidence",
    )
    manifest = {
        "development_only": True,
        "paper_result": False,
        "checkpoint_dependent_artifacts_rebuilt": list(required),
        "old_checkpoint_dependent_numeric_artifact_inputs": [],
    }
    runner._verify_aggregate_provenance(manifest, required)
    mutations = (
        ("development_only", False),
        ("paper_result", True),
        ("checkpoint_dependent_artifacts_rebuilt", list(required[:-1])),
        ("old_checkpoint_dependent_numeric_artifact_inputs", ["old"]),
    )
    for field, value in mutations:
        changed = dict(manifest)
        changed[field] = value
        with pytest.raises(runner.D0BProtocolError):
            runner._verify_aggregate_provenance(changed, required)


def test_legacy_logic_binding_is_scoped_and_restored(monkeypatch) -> None:
    contract = runner.load_contract(CONFIG)
    fake_freeze = {
        "datasets": {
            dataset: {
                "checkpoint": {
                    "path": contract.raw["datasets"][dataset]["checkpoint_path"],
                    "sha256": str(index) * 64,
                    "role": "epoch_1000_train_only_safe",
                }
            }
            for index, dataset in enumerate(DATASETS, start=1)
        }
    }
    monkeypatch.setattr(
        runner, "verify_freeze", lambda _contract: (fake_freeze, "f" * 64)
    )
    legacy = __import__("run_p3_stage_c0_signal_audit_v2")
    original = (
        legacy.PROTOCOL_ID,
        legacy.PREDECESSOR_PROTOCOL_ID,
        legacy.PROBE_SEED_NAMESPACE,
        legacy._verify_pre_run_freeze,
    )
    with pytest.raises(RuntimeError, match="fixture exit"):
        with runner._bound_legacy_c0_logic(contract) as (bound, projected):
            assert bound is legacy
            assert bound.PROTOCOL_ID == runner.PROTOCOL_ID
            assert bound.PREDECESSOR_PROTOCOL_ID == runner.LEGACY_C0_PROTOCOL_ID
            assert bound.PROBE_SEED_NAMESPACE == runner.PROBE_SEED_NAMESPACE
            assert projected.raw["freeze"]["pre_run_freeze_receipt"] == (
                contract.raw["freeze"]["pre_run_receipt"]
            )
            for dataset in DATASETS:
                assert projected.raw["datasets"][dataset]["checkpoint_sha256"] == (
                    fake_freeze["datasets"][dataset]["checkpoint"]["sha256"]
                )
            raise RuntimeError("fixture exit")
    assert (
        legacy.PROTOCOL_ID,
        legacy.PREDECESSOR_PROTOCOL_ID,
        legacy.PROBE_SEED_NAMESPACE,
        legacy._verify_pre_run_freeze,
    ) == original
