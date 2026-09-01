from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest

import run_binary_tent_ss_calibration_v2 as legacy
import run_binary_tent_ss_calibration_v3 as v3


def _forbid(calls: list[str], name: str):
    def fail(*_args, **_kwargs):
        calls.append(name)
        raise AssertionError(f"legacy Stage-2 crossed side-effect boundary: {name}")

    return fail


def test_every_legacy_stage2_function_blocks_before_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(legacy, "load_contract", _forbid(calls, "load_contract"))
    monkeypatch.setattr(
        legacy.core,
        "_verify_inherited_gpu_lease",
        _forbid(calls, "verify_gpu_lease"),
    )
    monkeypatch.setattr(
        legacy.core,
        "_with_stage_launcher_lock",
        _forbid(calls, "stage_launcher_lock"),
    )
    monkeypatch.setattr(legacy.core, "_run_parallel", _forbid(calls, "run_parallel"))

    stage2_args = argparse.Namespace(
        role="worker-stage2",
        execution_config=tmp_path / "missing.yaml",
        top3_receipt=tmp_path / "old-top3.json",
        process_id="legacy",
        slot_index=1,
        device="cuda:0",
        gpu_ids="0",
    )
    calls_to_block = (
        lambda: legacy.run_stage2_worker(stage2_args),
        lambda: legacy.aggregate_final(stage2_args),
        lambda: legacy.launch_stage2(stage2_args),
        lambda: legacy._launch_stage2_locked(stage2_args, None, ("0",), -1),
        lambda: legacy.run(stage2_args),
    )
    for invoke in calls_to_block:
        with pytest.raises(
            legacy.LegacyV2Stage2BlockedError, match="permanently blocked"
        ):
            invoke()
    assert calls == []
    assert not (tmp_path / "stage2").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["launch-stage2", "--gpu-ids", "0"],
        [
            "worker-stage2",
            "--top3-receipt",
            "/tmp/old-top3.json",
            "--process-id",
            "legacy",
            "--slot-index",
            "1",
        ],
        ["aggregate-final"],
    ],
)
def test_legacy_stage2_cli_has_dedicated_exit_3(argv: list[str]) -> None:
    assert legacy.main(argv) == legacy.EXIT_SCIENTIFIC_GATE_BLOCKED == 3


def test_legacy_stage1_dispatch_is_not_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = {"stage1_behavior": "unchanged"}
    monkeypatch.setattr(legacy, "validate_only", lambda _args: expected)
    assert legacy.run(argparse.Namespace(role="validate")) is expected


def test_parser_retains_disabled_roles_for_explicit_failure_message() -> None:
    parser = legacy.build_parser()
    assert parser.parse_args(["launch-stage2"]).role == "launch-stage2"
    assert parser.parse_args(
        [
            "worker-stage2",
            "--top3-receipt",
            "/tmp/old-top3.json",
            "--process-id",
            "legacy",
            "--slot-index",
            "1",
        ]
    ).role == "worker-stage2"
    assert parser.parse_args(["aggregate-final"]).role == "aggregate-final"


def test_synthetic_legacy_top3_receipt_cannot_be_relabelled_as_v3_authority(
    tmp_path: Path,
) -> None:
    raw = (
        b'{"receipt_type":"stage1_ss_top3","schema_version":2,'
        b'"top3":[]}\n'
    )
    relabelled = tmp_path / v3.STAGE1_RECEIPT_FILENAME
    relabelled.write_bytes(raw)
    # A receipt and its same-caller digest are no longer an authorization API.
    assert not hasattr(v3, "ReceiptBinding")
    with pytest.raises(TypeError):
        v3.authorize_stage2(  # type: ignore[call-arg]
            relabelled,
            hashlib.sha256(raw).hexdigest(),
        )


def test_real_negative_v3_receipt_returns_scientific_block_exit_3() -> None:
    # The code-pinned production config records that the formal gate remains
    # unresolved, so it blocks without accepting the retrospective receipt.
    assert v3.main(["authorize-stage2"]) == v3.EXIT_SCIENTIFIC_GATE_BLOCKED == 3
