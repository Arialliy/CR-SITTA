from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis.d0_v3_label_free_shard import VerifiedLabelFreeShard
from scripts import run_d0_v3_formal_stage_a_outer as runner


def test_dry_run_preflight_has_no_route_to_real_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = SimpleNamespace(config_file_sha256="1" * 64)
    verified = VerifiedLabelFreeShard(
        path=tmp_path / "dry",
        dataset="NUAA-SIRST",
        condition="clean_S0",
        replicate="R0",
        formal=False,
        dry_run=True,
        image_count=1,
        candidate_count=10,
        episode_count=10,
        manifest_sha256="2" * 64,
        complete_sha256="3" * 64,
        phase_receipt_sha256=None,
    )
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(
        runner, "verify_label_free_shard", lambda *_args, **_kwargs: verified
    )
    monkeypatch.setattr(
        runner,
        "guarded_load_outer_targets",
        lambda *_args, **_kwargs: pytest.fail("dry-run reached real target guard"),
    )
    monkeypatch.setattr(
        runner,
        "build_d0_v3_outer_source_model",
        lambda *_args, **_kwargs: pytest.fail("dry-run built an outer model"),
    )
    result = runner.dry_run_preflight(
        label_free_path=tmp_path / "dry",
        config_path=tmp_path / "config.yaml",
    )
    assert result["real_target_opened"] is False
    assert result["outer_target_loader_calls"] == 0
    assert result["outer_source_model_builds"] == 0
    assert result["stage2_authorized"] is False


def test_formal_runner_source_orders_full_preflight_before_target_guard() -> None:
    source = inspect.getsource(runner.run_formal_outer_cell)
    preflight = source.index("_formal_label_free_preflight(")
    worker = source.index("build_d0_v3_outer_source_model(")
    guard = source.index("guarded_load_outer_targets(")
    assert preflight < worker < guard
    assert "load_outer_evaluator_targets_v2" not in source
    assert source.count("guarded_load_outer_targets(") == 1
    assert source.count("build_d0_v3_outer_source_model(") == 1
    assert "outer_optimizer" not in source


def test_formal_runner_is_fixed_to_r0_and_64_without_replicate_override() -> None:
    signature = inspect.signature(runner.run_formal_outer_cell)
    assert "replicate" not in signature.parameters
    source = inspect.getsource(runner.run_formal_outer_cell)
    assert '"replicate_id": "R0"' in source
    assert "range(FORMAL_IMAGE_COUNT)" in source
    assert "CANDIDATE_COUNT" in source


def test_source_parameter_bit_exact_gate_accepts_broadcast_identity() -> None:
    source = torch.arange(12, dtype=torch.float32)
    repeated = np.broadcast_to(source.numpy(), (64, 10, 12)).copy().astype("<f4")
    runner._source_parameter_match(source, repeated)
    repeated[63, 9, 11] += 1.0
    with pytest.raises(runner.D0V3FormalOuterRunnerError, match="not bit-exact"):
        runner._source_parameter_match(source, repeated)


def test_fixed_output_layout_separates_candidate_and_outer_phases() -> None:
    contract = SimpleNamespace(
        output_root="results/cr_sitta/tent_failure_diagnostics_v3_formal_stage_a"
    )
    label = runner._fixed_label_free_path(contract, "NUAA-SIRST", "clean_S0")
    outer = runner._fixed_outer_path(contract, "NUAA-SIRST", "clean_S0")
    assert label.parts[-5:] == (
        "candidate_phase",
        "shards",
        "R0",
        "NUAA-SIRST",
        "clean_S0",
    )
    assert outer.parts[-5:] == (
        "outer_phase",
        "shards",
        "R0",
        "NUAA-SIRST",
        "clean_S0",
    )
    assert label != outer


def test_cpu_formal_forward_runtime_is_strict_and_reseeds() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    first = runner._configure_formal_forward_runtime(
        device=torch.device("cpu"), seed=42
    )
    sample_one = torch.rand(4)
    runner._configure_formal_forward_runtime(device=torch.device("cpu"), seed=42)
    sample_two = torch.rand(4)
    assert torch.equal(sample_one, sample_two)
    assert first["deterministic_algorithms"] is True
    assert first["deterministic_warn_only"] is False
    assert first["cudnn_benchmark"] is False
    assert first["cudnn_deterministic"] is True


def test_cuda_formal_forward_runtime_rejects_missing_frozen_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(
        runner.D0V3FormalOuterRunnerError,
        match="environment differs from candidate runtime",
    ):
        runner._configure_formal_forward_runtime(
            device=torch.device("cuda:0"), seed=42
        )


def test_formal_runtime_is_configured_before_model_build_and_target_guard() -> None:
    source = inspect.getsource(runner.run_formal_outer_cell)
    preflight = source.index("_formal_label_free_preflight(")
    runtime = source.index("_configure_formal_forward_runtime(")
    worker = source.index("build_d0_v3_outer_source_model(")
    guard = source.index("guarded_load_outer_targets(")
    assert preflight < runtime < worker < guard


def test_fine_group_order_is_explicitly_frozen_without_losing_values() -> None:
    reverse = tuple(reversed(runner.FINE_ALIGNMENT_GROUP_IDS))
    analysis = {
        "entropy_task_alignment": {
            "global": {"cosine": 0.25},
            "per_group": {group: {"marker": index} for index, group in enumerate(reverse)},
        }
    }
    sealed = runner._freeze_fine_group_order(analysis)
    per_group = sealed["entropy_task_alignment"]["per_group"]
    assert tuple(per_group) == runner.FINE_ALIGNMENT_GROUP_IDS
    assert {key: value["marker"] for key, value in per_group.items()} == {
        key: reverse.index(key) for key in runner.FINE_ALIGNMENT_GROUP_IDS
    }


def test_fine_group_order_rejects_missing_or_unknown_group() -> None:
    groups = {group: {} for group in runner.FINE_ALIGNMENT_GROUP_IDS[:-1]}
    groups["not_frozen"] = {}
    with pytest.raises(
        runner.D0V3FormalOuterRunnerError,
        match="keys differ from frozen",
    ):
        runner._freeze_fine_group_order(
            {"entropy_task_alignment": {"per_group": groups}}
        )
