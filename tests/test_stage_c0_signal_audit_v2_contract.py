from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import run_p3_stage_c0_signal_audit_v1 as predecessor
import run_p3_stage_c0_signal_audit_v2 as runner


ROOT = Path(__file__).resolve().parents[1]
V1_SMOKE = (
    ROOT
    / "results/cr_sitta/p3_stage_c0_signal_audit_v1"
    / "engineering_smoke/gpu_smoke_v1/candidate/IRSTD-1K"
)
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"
REQUIRES_LOCAL_ARTIFACTS = pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local Stage-C predecessor artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)


@pytest.fixture()
def contract(monkeypatch: pytest.MonkeyPatch):
    if not LOCAL_ARTIFACT_TESTS_ENABLED:
        pytest.skip(
            "requires the frozen local Stage-C environment; set "
            f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
        )
    monkeypatch.setattr(
        runner, "FROZEN_CONFIG_SHA256", runner.current_config_sha256()
    )
    return runner.load_contract()


def _canonical_round_trip(value):
    return json.loads(runner._canonical_json_bytes(value))


@REQUIRES_LOCAL_ARTIFACTS
def test_v1_frozen_bytes_and_abort_evidence_are_unchanged() -> None:
    expected = {
        "run_p3_stage_c0_signal_audit_v1.py": (
            "115aaa85fc584c41f857ea5b09a3bda860589dd72b685ac5cb6932edb3c612a3"
        ),
        "configs/p3_stage_c0_signal_audit_v1.yaml": (
            "785efe100797e25af776361e048ba3569349a1b925b50993a4f96abdec92358c"
        ),
        "results/cr_sitta/p3_stage_c0_signal_audit_v1/PRE_RUN_FREEZE.json": (
            "171755ea6df470227c98da11e2f358cf7dce486e3c45506ce38ae7ced91531cf"
        ),
        "results/cr_sitta/p3_stage_c0_signal_audit_v1/ABORTED_ENGINEERING_RECEIPT.json": (
            "9c12255a7e650509dc1505369cd22b8a5197cd63fff295d690e82840168b1b52"
        ),
    }
    assert {
        relative: runner.sha256_file(ROOT / relative) for relative in expected
    } == expected
    abort = json.loads(
        (ROOT / next(reversed(expected))).read_text(encoding="utf-8")
    )
    assert abort["status"] == "aborted_before_formal_execution"
    assert abort["formal_execution"]["formal_execution_started"] is False
    assert abort["scope"]["gate_evaluated"] is False
    assert abort["scope"]["scientific_result"] is False


def test_v2_revision_lineage_is_projected_into_pre_run_receipt(contract) -> None:
    assert contract.raw["protocol_id"] == runner.PROTOCOL_ID
    lineage = contract.raw["revision_lineage"]
    assert lineage["predecessor_protocol_id"] == predecessor.PROTOCOL_ID
    assert lineage["science_contract_changed"] is False
    assert lineage["science_gate_changed"] is False
    assert lineage["formal_execution_started"] is False
    expected = runner._expected_pre_run_freeze_receipt(contract)
    assert expected["revision_lineage"] == lineage
    assert expected["probe_seed_contract"] == {
        "artifact_protocol_id": runner.PROTOCOL_ID,
        "descriptor_protocol_field_name": "protocol_id",
        "descriptor_protocol_field_value": predecessor.PROTOCOL_ID,
        "scientific_inputs_preserved_from_predecessor": True,
    }


@pytest.mark.parametrize(
    "probe_id,golden",
    (("lf_mask", 6705630015044588659), ("hf_noise", 4812635961954743907)),
)
def test_v2_probe_seed_is_bit_exact_to_v1(probe_id: str, golden: int) -> None:
    kwargs = {
        "global_seed": 42,
        "image_id": "XDU102",
        "probe_id": probe_id,
        "input_tensor_sha256": (
            "5e1eda46b71a2c4ff90b7bc8949de6903181de83ad70230d434d0f42aa58dde7"
        ),
    }
    old = predecessor.derive_probe_seed(**kwargs)
    new = runner.derive_probe_seed(
        probe_seed_namespace=predecessor.PROTOCOL_ID, **kwargs
    )
    assert old == new == golden


@REQUIRES_LOCAL_ARTIFACTS
def test_canonical_layout_object_uses_key_roster_not_json_key_order() -> None:
    serialized = json.loads((V1_SMOKE / "parameter_layouts.json").read_text())
    assert tuple(serialized) == ("P2", "R-D0", "R-E1")
    validated = runner._validate_layouts(serialized)
    assert tuple(validated) == runner.PARAMETER_SPACES


@pytest.mark.parametrize(
    "label",
    ("candidate gradient evidence", "outer space evidence"),
)
def test_canonical_nested_space_objects_use_exact_roster_not_key_order(
    label: str,
) -> None:
    original = {space: {"parameter_space": space} for space in runner.PARAMETER_SPACES}
    serialized = _canonical_round_trip(original)
    assert tuple(serialized) == ("P2", "R-D0", "R-E1")
    accepted = runner._mapping_with_exact_keys(
        serialized, runner.PARAMETER_SPACES, label
    )
    assert [accepted[space]["parameter_space"] for space in runner.PARAMETER_SPACES] == list(
        runner.PARAMETER_SPACES
    )
    missing = dict(serialized)
    missing.pop("P2")
    with pytest.raises(runner.StageC0ProtocolError, match="key roster differs"):
        runner._mapping_with_exact_keys(missing, runner.PARAMETER_SPACES, label)
    extra = dict(serialized, unexpected={})
    with pytest.raises(runner.StageC0ProtocolError, match="key roster differs"):
        runner._mapping_with_exact_keys(extra, runner.PARAMETER_SPACES, label)
