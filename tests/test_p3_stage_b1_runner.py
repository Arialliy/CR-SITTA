from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
import torch

from analysis.foreground_background_gradient_decomposition_v1 import GROUP_IDS
from analysis.p3_stage_b1_contract import load_p3_stage_b1_contract
from analysis.p3_stage_b1_outer_cell_shard import BASIS_ORDER
from scripts import run_p3_stage_b_gradient_decomposition_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3_stage_b_gradient_decomposition_v1.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"
DATASET = "NUAA-SIRST"
CONDITION = "clean_S0"


class _InjectedFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class _Fingerprint:
    full_sha256: str


class _FakeStateManager:
    def __init__(self, *, reset_matches: bool = True) -> None:
        self.source = _Fingerprint("1" * 64)
        self.reset_value = self.source if reset_matches else _Fingerprint("2" * 64)
        self.reset_calls = 0

    def assert_source_state(self) -> _Fingerprint:
        return self.source

    def reset_to_source(self) -> _Fingerprint:
        self.reset_calls += 1
        return self.reset_value


class _FakeAdapter:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def set_tent_mode(self, *, use_batch_stats: bool) -> None:
        assert use_batch_stats is False

    def forward_logits(self, _image: torch.Tensor) -> torch.Tensor:
        return self.logits.clone()


class _FakeModel:
    def zero_grad(self, *, set_to_none: bool) -> None:
        assert set_to_none is True


class _FakeLayout:
    names: tuple[str, ...] = ()
    offsets: tuple[int, ...] = ()
    scalar_count = runner.SCALAR_COUNT
    layout_sha256 = "3" * 64

    def to_dict(self) -> dict[str, Any]:
        return {"layout": "frozen"}


class _FakeMethodInputs:
    def __init__(self, image_ids: tuple[str, ...]) -> None:
        self.image_ids = image_ids

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        assert index == 0
        return {"image": torch.zeros((3, 256, 256), dtype=torch.float32)}


def _failing(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("forbidden side effect was reached")


def _install_compute_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    analyzer: Callable[..., Any],
    reset_matches: bool = True,
    candidate_logit_value: float = 0.0,
    outer_logit_value: float = 0.0,
    parent_entropy: np.ndarray | None = None,
) -> tuple[Any, _FakeStateManager, list[str]]:
    """Install a one-image, no-filesystem/no-CUDA harness around ``_compute_cell``."""

    contract = load_p3_stage_b1_contract(CONFIG)
    image_ids = tuple(f"pilot_{index:03d}" for index in range(64))
    layout = _FakeLayout()
    parent = SimpleNamespace(
        candidate_path=ROOT / "mock-parent-candidate",
        outer_path=ROOT / "mock-parent-outer",
        candidate_verified=SimpleNamespace(phase_receipt_sha256="4" * 64),
        outer_verified=SimpleNamespace(),
        candidate_manifest={},
        outer_manifest={},
        phase_receipt=SimpleNamespace(code_files=()),
        ordered_image_ids=image_ids,
        layout=layout,
    )
    state = _FakeStateManager(reset_matches=reset_matches)
    source_logits = torch.full(
        (1, 1, 256, 256), candidate_logit_value, dtype=torch.float32
    )
    worker = SimpleNamespace(
        layout=layout,
        parameter_names=(),
        parameters=(),
        model=_FakeModel(),
        adapter=_FakeAdapter(source_logits),
        state_manager=state,
    )
    candidate_logits = np.full(
        (1, 1, 256, 256), candidate_logit_value, dtype="<f4"
    )
    outer_logits = np.full(
        (1, 1, 256, 256), outer_logit_value, dtype="<f4"
    )
    entropy = (
        np.zeros((1, 10, runner.SCALAR_COUNT), dtype="<f4")
        if parent_entropy is None
        else np.ascontiguousarray(parent_entropy, dtype="<f4")
    )
    task = np.zeros((1, runner.SCALAR_COUNT), dtype="<f4")
    target = np.zeros((1, 1, 256, 256), dtype="<f4")
    events: list[str] = []

    monkeypatch.setattr(
        runner,
        "_preflight_parent_cell",
        lambda **_kwargs: events.append("parent_preflight") or parent,
    )
    monkeypatch.setattr(
        runner,
        "_code_seal",
        lambda _contract: events.append("code_seal")
        or {"files": [{"path": "mock.py", "sha256": "5" * 64}], "bundle_sha256": "6" * 64},
    )
    monkeypatch.setattr(
        runner,
        "_configure_runtime",
        lambda **_kwargs: events.append("runtime") or {"device": "cpu"},
    )
    monkeypatch.setattr(
        runner,
        "build_d0_v3_outer_source_model",
        lambda **_kwargs: events.append("model") or worker,
    )
    monkeypatch.setattr(
        runner,
        "_group_parameter_names",
        lambda _worker: {group: () for group in GROUP_IDS},
    )
    monkeypatch.setattr(
        runner,
        "build_coarse_group_layout",
        lambda **_kwargs: {"group_order": list(GROUP_IDS)},
    )

    def load_array(path: Path, *, shape: tuple[int, ...]) -> np.ndarray:
        del shape
        if path.name == runner.SOURCE_LOGITS_FILENAME:
            return candidate_logits
        if path.name == runner.OUTER_SOURCE_LOGITS_FILENAME:
            return outer_logits
        if path.name == runner.ENTROPY_GRADIENTS_FILENAME:
            return entropy
        if path.name == runner.SUPERVISED_GRADIENTS_FILENAME:
            return task
        raise AssertionError(f"unexpected array path: {path}")

    monkeypatch.setattr(runner, "_load_array", load_array)
    monkeypatch.setattr(
        runner,
        "read_stable_regular_file",
        lambda _path: SimpleNamespace(data=b"{}"),
    )

    class CountingTargets:
        def __getitem__(self, index: int) -> np.ndarray:
            events.append("target_index")
            return target[index]

    def guarded(*_args: Any, **_kwargs: Any) -> Any:
        events.append("target_guard")
        return SimpleNamespace(access_receipt_bytes=b"{}", targets=CountingTargets())

    monkeypatch.setattr(runner, "guarded_load_outer_targets", guarded)
    monkeypatch.setattr(
        runner,
        "SourceCalibrationMethodInputDatasetV2",
        lambda *_args, **_kwargs: _FakeMethodInputs(image_ids),
    )
    monkeypatch.setattr(
        runner, "analyze_foreground_background_gradient_decomposition", analyzer
    )
    return contract, state, events


def _successful_analysis(*_args: Any, **_kwargs: Any) -> Any:
    sub = torch.full((runner.SCALAR_COUNT,), 1.0, dtype=torch.float64)
    supra = torch.full((runner.SCALAR_COUNT,), 2.0, dtype=torch.float64)
    background = torch.full((runner.SCALAR_COUNT,), 3.0, dtype=torch.float64)
    return SimpleNamespace(
        vectors=SimpleNamespace(
            foreground_subthreshold_add=sub,
            foreground_suprathreshold_add=supra,
            background_add=background,
            full_add=sub + supra + background,
        ),
        report={
            "target_statistics": {
                "total_pixel_count": 256 * 256,
                "foreground_pixel_count": 0,
                "background_pixel_count": 256 * 256,
                "foreground_subthreshold_pixel_count": 0,
                "foreground_suprathreshold_pixel_count": 0,
            },
            "per_group": {group: {} for group in GROUP_IDS},
        },
    )


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local Stage-A aggregate artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_validate_only_has_no_directory_cuda_target_or_dataset_side_effect() -> None:
    cuda_initialized_before = torch.cuda.is_initialized()
    script = Path(runner.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "--config", str(CONFIG), "validate"],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["filesystem_created"] is False
    assert result["dataset_payload_opened"] is False
    assert result["cuda_initialized"] is False
    assert result["paper_result"] is False
    assert result["stage_b3_authorized"] is False
    assert torch.cuda.is_initialized() is cuda_initialized_before


def test_validate_only_rejects_preinitialized_cuda_before_contract_or_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(runner, "_load_contract", _failing)
    monkeypatch.setattr(runner, "_configure_runtime", _failing)
    monkeypatch.setattr(runner, "guarded_load_outer_targets", _failing)
    monkeypatch.setattr(runner, "SourceCalibrationMethodInputDatasetV2", _failing)
    monkeypatch.setattr(runner, "build_d0_v3_outer_source_model", _failing)

    with pytest.raises(
        runner.P3StageB1RunnerError,
        match="before any CUDA context is initialized",
    ):
        runner.validate_only(CONFIG)


def test_config_failure_blocks_code_cuda_target_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "_load_contract",
        lambda _path: (_ for _ in ()).throw(_InjectedFailure("config")),
    )
    monkeypatch.setattr(runner, "_code_seal", _failing)
    monkeypatch.setattr(runner, "_configure_runtime", _failing)
    monkeypatch.setattr(runner, "guarded_load_outer_targets", _failing)
    destination = tmp_path / "never-created" / "cell"

    with pytest.raises(_InjectedFailure, match="config"):
        runner.run_formal_cell(
            dataset=DATASET,
            condition=CONDITION,
            config_path=tmp_path / "bad.yaml",
            device=torch.device("cuda:0"),
            output_path=destination,
        )
    assert not destination.parent.exists()


def test_formal_cpu_is_rejected_before_contract_or_any_side_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_load_contract", _failing)
    monkeypatch.setattr(runner, "_code_seal", _failing)
    monkeypatch.setattr(runner, "_configure_runtime", _failing)
    monkeypatch.setattr(runner, "guarded_load_outer_targets", _failing)
    destination = tmp_path / "never-created" / "cell"

    with pytest.raises(
        runner.P3StageB1RunnerError,
        match="formal Stage-B1 cells require the sole visible device cuda:0",
    ):
        runner.run_formal_cell(
            dataset=DATASET,
            condition=CONDITION,
            config_path=tmp_path / "config.yaml",
            device=torch.device("cpu"),
            output_path=destination,
        )
    assert not destination.parent.exists()


def test_parent_receipt_failure_blocks_cuda_target_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_p3_stage_b1_contract(CONFIG)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(
        runner,
        "_preflight_parent_cell",
        lambda **_kwargs: (_ for _ in ()).throw(_InjectedFailure("parent receipt")),
    )
    monkeypatch.setattr(runner, "_code_seal", lambda _contract: {})
    monkeypatch.setattr(runner, "_configure_runtime", _failing)
    monkeypatch.setattr(runner, "guarded_load_outer_targets", _failing)
    destination = tmp_path / "never-created" / "cell"

    with pytest.raises(_InjectedFailure, match="parent receipt"):
        runner.run_formal_cell(
            dataset=DATASET,
            condition=CONDITION,
            config_path=tmp_path / "config.yaml",
            device=torch.device("cuda:0"),
            output_path=destination,
        )
    assert not destination.parent.exists()


def test_code_seal_failure_blocks_cuda_target_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_p3_stage_b1_contract(CONFIG)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "_preflight_parent_cell", lambda **_kwargs: object())
    monkeypatch.setattr(
        runner,
        "_code_seal",
        lambda _contract: (_ for _ in ()).throw(_InjectedFailure("code seal")),
    )
    monkeypatch.setattr(runner, "_configure_runtime", _failing)
    monkeypatch.setattr(runner, "guarded_load_outer_targets", _failing)
    monkeypatch.setattr(runner, "build_d0_v3_outer_source_model", _failing)
    destination = tmp_path / "never-created" / "cell"

    with pytest.raises(_InjectedFailure, match="code seal"):
        runner.run_formal_cell(
            dataset=DATASET,
            condition=CONDITION,
            config_path=tmp_path / "config.yaml",
            device=torch.device("cuda:0"),
            output_path=destination,
        )
    assert not destination.parent.exists()


def test_runtime_failure_occurs_after_seal_but_before_model_target_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_p3_stage_b1_contract(CONFIG)
    events: list[str] = []
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(
        runner,
        "_preflight_parent_cell",
        lambda **_kwargs: events.append("parent") or object(),
    )
    monkeypatch.setattr(
        runner, "_code_seal", lambda _contract: events.append("seal") or {}
    )

    def fail_runtime(**_kwargs: Any) -> Any:
        events.append("runtime")
        raise _InjectedFailure("runtime")

    monkeypatch.setattr(runner, "_configure_runtime", fail_runtime)
    monkeypatch.setattr(runner, "build_d0_v3_outer_source_model", _failing)
    monkeypatch.setattr(runner, "guarded_load_outer_targets", _failing)
    destination = tmp_path / "never-created" / "cell"

    with pytest.raises(_InjectedFailure, match="runtime"):
        runner.run_formal_cell(
            dataset=DATASET,
            condition=CONDITION,
            config_path=tmp_path / "config.yaml",
            device=torch.device("cuda:0"),
            output_path=destination,
        )
    assert events == ["seal", "parent", "runtime"]
    assert not destination.parent.exists()


def test_receipt_validator_is_fail_closed_before_code_cuda_and_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_ids = tuple(f"pilot_{index:03d}" for index in range(64))
    expected_ids_sha = "7" * 64
    layout = _FakeLayout()
    contract = SimpleNamespace(
        datasets=(DATASET,),
        conditions=(CONDITION,),
        pilot64_ordered_id_sha256=((DATASET, expected_ids_sha),),
        raw={
            "frozen_parent_bindings": {"cache_protocol": {"sha256": "8" * 64}},
            "datasets": {DATASET: {"checkpoint_sha256": "9" * 64}},
            "parent_cell_artifacts": {
                "candidate_shard_template": "candidate/{dataset}/{condition}",
                "outer_shard_template": "outer/{dataset}/{condition}",
                "parameter_layout": {
                    "layout_sha256": layout.layout_sha256,
                    "parameter_tensor_count": 0,
                    "scalar_parameter_count": runner.SCALAR_COUNT,
                },
            },
        },
    )
    candidate = SimpleNamespace(
        dataset=DATASET,
        condition=CONDITION,
        replicate="R0",
        formal=True,
        dry_run=False,
        image_count=64,
        candidate_count=10,
        phase_receipt_sha256="a" * 64,
    )
    outer = SimpleNamespace(
        dataset=DATASET,
        condition=CONDITION,
        replicate="R0",
        image_count=64,
    )
    candidate_manifest = {
        "ordered_image_ids": list(image_ids),
        "ordered_image_ids_sha256": expected_ids_sha,
    }
    outer_manifest = {
        "ordered_image_ids": list(image_ids),
        "ordered_image_ids_sha256": expected_ids_sha,
    }
    monkeypatch.setattr(
        runner,
        "_fixed_parent_paths",
        lambda *_args: (ROOT / "candidate", ROOT / "outer"),
    )
    monkeypatch.setattr(runner, "verify_label_free_shard", lambda *_a, **_k: candidate)
    monkeypatch.setattr(runner, "verify_outer_cell_shard", lambda *_a, **_k: outer)

    def read_canonical(path: Path, *, label: str) -> Any:
        del label
        if path.name == runner.PARENT_CANDIDATE_MANIFEST_FILENAME:
            return candidate_manifest
        if path.name == runner.PARENT_OUTER_MANIFEST_FILENAME:
            return outer_manifest
        if path.name == runner.PARENT_LAYOUT_FILENAME:
            return {"layout": "frozen"}
        raise AssertionError(path)

    monkeypatch.setattr(runner, "_read_canonical", read_canonical)
    monkeypatch.setattr(runner, "validate_parameter_layout", lambda _value: layout)
    monkeypatch.setattr(
        runner, "read_stable_regular_file", lambda _path: SimpleNamespace(data=b"phase")
    )

    def reject_receipt(_source: bytes, **kwargs: Any) -> Any:
        assert kwargs == {
            "expected_dataset": DATASET,
            "expected_condition": CONDITION,
            "expected_replicate": 0,
            "expected_cache_protocol_sha256": "8" * 64,
            "expected_checkpoint_sha256": "9" * 64,
            "expected_config_sha256": runner.PARENT_CONFIG_SHA256,
            "expected_ordered_image_ids": image_ids,
            "expected_receipt_sha256": "a" * 64,
        }
        raise _InjectedFailure("receipt validation")

    monkeypatch.setattr(runner, "validate_label_free_cell_receipt", reject_receipt)
    monkeypatch.setattr(runner, "_code_seal", lambda _contract: {})
    monkeypatch.setattr(runner, "_configure_runtime", _failing)
    monkeypatch.setattr(runner, "guarded_load_outer_targets", _failing)

    with pytest.raises(_InjectedFailure, match="receipt validation"):
        runner._compute_cell(
            contract=contract,
            dataset=DATASET,
            condition=CONDITION,
            device=torch.device("cuda:0"),
            image_count=1,
        )


def test_candidate_outer_source_logit_mismatch_blocks_target_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, _state, events = _install_compute_harness(
        monkeypatch,
        analyzer=_successful_analysis,
        candidate_logit_value=0.0,
        outer_logit_value=1.0,
    )
    with pytest.raises(runner.P3StageB1RunnerError, match="Source logits differ"):
        runner._compute_cell(
            contract=contract,
            dataset=DATASET,
            condition=CONDITION,
            device=torch.device("cuda:0"),
            image_count=1,
        )
    assert events == ["code_seal", "parent_preflight", "runtime", "model"]


def test_recomputed_source_logit_mismatch_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract, state, events = _install_compute_harness(
        monkeypatch,
        analyzer=_successful_analysis,
        candidate_logit_value=0.0,
        outer_logit_value=0.0,
    )
    # Parent arrays remain zero while the live model returns a different value.
    monkeypatch.setattr(
        runner,
        "build_d0_v3_outer_source_model",
        lambda **_kwargs: events.append("model")
        or SimpleNamespace(
            layout=_FakeLayout(),
            parameter_names=(),
            parameters=(),
            model=_FakeModel(),
            adapter=_FakeAdapter(torch.ones((1, 1, 256, 256), dtype=torch.float32)),
            state_manager=state,
        ),
    )
    with pytest.raises(runner.P3StageB1RunnerError, match="Source logits differ"):
        runner._compute_cell(
            contract=contract,
            dataset=DATASET,
            condition=CONDITION,
            device=torch.device("cpu"),
            image_count=1,
        )
    assert state.reset_calls == 1
    assert "target_guard" in events


def test_all_ten_parent_entropy_slices_are_checked_not_only_slice_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entropy = np.full((1, 10, runner.SCALAR_COUNT), 6.0, dtype="<f4")
    entropy[0, 9, 0] += 1.0e-3
    contract, state, _events = _install_compute_harness(
        monkeypatch,
        analyzer=_successful_analysis,
        parent_entropy=entropy,
    )
    with pytest.raises(runner.P3StageB1RunnerError, match="all-candidate"):
        runner._compute_cell(
            contract=contract,
            dataset=DATASET,
            condition=CONDITION,
            device=torch.device("cpu"),
            image_count=1,
        )
    assert state.reset_calls == 1


def test_basis_storage_and_group_order_are_exact() -> None:
    contract = load_p3_stage_b1_contract(CONFIG)
    assert runner.STORAGE_BASIS == BASIS_ORDER == (
        "foreground_subthreshold",
        "foreground_suprathreshold",
        "background",
    )
    assert tuple(contract.raw["gradient_basis"]["basis_order"]) == BASIS_ORDER
    assert GROUP_IDS == ("P0", "P1", "P2", "P3", "P4")
    assert tuple(contract.raw["parameter_spaces"]["nested_order"]) == GROUP_IDS


def test_basis_array_is_stored_in_frozen_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entropy = np.full((1, 10, runner.SCALAR_COUNT), 6.0, dtype="<f4")
    contract, state, events = _install_compute_harness(
        monkeypatch,
        analyzer=_successful_analysis,
        parent_entropy=entropy,
    )
    basis, records, _layout, _receipt, _parent, _runtime, _seal = runner._compute_cell(
        contract=contract,
        dataset=DATASET,
        condition=CONDITION,
        device=torch.device("cuda:0"),
        image_count=1,
    )
    assert basis.shape == (1, 3, runner.SCALAR_COUNT)
    assert np.all(basis[0, 0] == 1.0)
    assert np.all(basis[0, 1] == 2.0)
    assert np.all(basis[0, 2] == 3.0)
    assert records[0]["gradient_integrity"]["basis_order"] == list(BASIS_ORDER)
    assert state.reset_calls == 1
    assert events.count("target_index") == 1
    assert events.index("code_seal") < events.index("runtime") < events.index(
        "target_guard"
    )


def test_rng_is_restored_when_episode_analysis_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_analysis(*_args: Any, **_kwargs: Any) -> Any:
        random.random()
        np.random.random()
        torch.rand(1)
        raise _InjectedFailure("analysis")

    contract, state, _events = _install_compute_harness(
        monkeypatch, analyzer=failing_analysis
    )
    before = runner._capture_rng(torch.device("cpu"))
    try:
        with pytest.raises(_InjectedFailure, match="analysis"):
            runner._compute_cell(
                contract=contract,
                dataset=DATASET,
                condition=CONDITION,
                device=torch.device("cpu"),
                image_count=1,
            )
        after = runner._capture_rng(torch.device("cpu"))
        assert runner._rng_equal(before, after)
        assert state.reset_calls == 1
    finally:
        runner._restore_rng(before, torch.device("cpu"))


def test_state_reset_mismatch_is_a_hard_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entropy = np.full((1, 10, runner.SCALAR_COUNT), 6.0, dtype="<f4")
    contract, state, _events = _install_compute_harness(
        monkeypatch,
        analyzer=_successful_analysis,
        reset_matches=False,
        parent_entropy=entropy,
    )
    with pytest.raises(runner.P3StageB1RunnerError, match="state|reset"):
        runner._compute_cell(
            contract=contract,
            dataset=DATASET,
            condition=CONDITION,
            device=torch.device("cpu"),
            image_count=1,
        )
    assert state.reset_calls == 1


def test_code_seal_is_rechecked_before_any_output_directory_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_p3_stage_b1_contract(CONFIG)
    initial_seal = {
        "files": [{"path": "analysis/mock.py", "sha256": "a" * 64}],
        "bundle_sha256": "b" * 64,
    }
    drifted_seal = {
        "files": [{"path": "analysis/mock.py", "sha256": "c" * 64}],
        "bundle_sha256": "d" * 64,
    }
    parent = SimpleNamespace()
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(
        runner,
        "_compute_cell",
        lambda **_kwargs: (
            np.zeros((64, 3, runner.SCALAR_COUNT), dtype="<f4"),
            [],
            {},
            b"{}",
            parent,
            {},
            initial_seal,
        ),
    )
    monkeypatch.setattr(runner, "_code_seal", lambda _contract: drifted_seal)
    monkeypatch.setattr(runner, "build_stage_b1_cell_payloads", _failing)
    destination = tmp_path / "never-created" / "cell"

    with pytest.raises(runner.P3StageB1RunnerError, match="changed during"):
        runner.run_formal_cell(
            dataset=DATASET,
            condition=CONDITION,
            config_path=tmp_path / "config.yaml",
            device=torch.device("cuda:0"),
            output_path=destination,
        )
    assert not destination.parent.exists()


def test_formal_publication_uses_final_builder_and_verifier_interfaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = load_p3_stage_b1_contract(CONFIG)
    code_seal = {
        "files": [{"path": "analysis/mock.py", "sha256": "a" * 64}],
        "bundle_sha256": "b" * 64,
    }
    layout = SimpleNamespace(
        names=(), scalar_count=runner.SCALAR_COUNT, layout_sha256="c" * 64
    )
    parent = SimpleNamespace(
        candidate_path=tmp_path / "parent-candidate",
        outer_path=tmp_path / "parent-outer",
        ordered_image_ids=tuple(f"pilot_{index:03d}" for index in range(64)),
        layout=layout,
        candidate_verified=SimpleNamespace(
            manifest_sha256="d" * 64,
            complete_sha256="e" * 64,
            phase_receipt_sha256="f" * 64,
        ),
        outer_verified=SimpleNamespace(
            manifest_sha256="1" * 64,
            complete_sha256="2" * 64,
            outer_access_receipt_sha256="3" * 64,
        ),
        candidate_manifest={
            "arrays": {
                runner.ENTROPY_GRADIENTS_FILENAME: {"sha256": "4" * 64}
            }
        },
        outer_manifest={
            "arrays": {
                runner.SUPERVISED_GRADIENTS_FILENAME: {"sha256": "5" * 64}
            }
        },
    )
    basis = np.zeros((64, 3, runner.SCALAR_COUNT), dtype="<f4")
    records = [{} for _ in range(64)]
    group_layout = {"group_order": list(GROUP_IDS)}
    builder_calls: list[dict[str, Any]] = []
    verifier_calls: list[dict[str, Any]] = []
    destination = tmp_path / "result" / "cell"

    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_load_contract", lambda _path: contract)
    monkeypatch.setattr(
        runner,
        "_compute_cell",
        lambda **_kwargs: (
            basis,
            records,
            group_layout,
            b"{}",
            parent,
            {"device": "cpu"},
            code_seal,
        ),
    )
    monkeypatch.setattr(runner, "_code_seal", lambda _contract: code_seal)
    monkeypatch.setattr(runner, "_sha256_file", lambda _path: "6" * 64)

    def build_payloads(**kwargs: Any) -> dict[str, bytes]:
        builder_calls.append(kwargs)
        return {name: name.encode("ascii") for name in runner.CELL_MEMBERS}

    monkeypatch.setattr(runner, "build_stage_b1_cell_payloads", build_payloads)

    def verify_cell(path: Path, **kwargs: Any) -> Any:
        verifier_calls.append({"path": path, **kwargs})
        return SimpleNamespace(dataset=DATASET, condition=CONDITION)

    monkeypatch.setattr(runner, "verify_stage_b1_cell_shard", verify_cell)

    def publish(
        staging: Path,
        final: Path,
        *,
        expected_members: Any,
        semantic_verifier: Callable[[Path], Any],
    ) -> Path:
        assert tuple(expected_members) == tuple(sorted(runner.CELL_MEMBERS))
        assert {path.name for path in staging.iterdir()} == runner.CELL_MEMBERS
        semantic_verifier(staging)
        staging.rename(final)
        semantic_verifier(final)
        return final

    monkeypatch.setattr(runner, "publish_flat_directory_noreplace", publish)

    published = runner.run_formal_cell(
        dataset=DATASET,
        condition=CONDITION,
        config_path=tmp_path / "config.yaml",
        device=torch.device("cuda:0"),
        output_path=destination,
    )

    assert published == destination
    assert len(builder_calls) == 1
    call = builder_calls[0]
    assert call["region_gradient_basis"] is basis
    assert call["episode_records"] is records
    assert call["code_seal"] == code_seal
    assert call["data_boundary"] == runner.MANIFEST_DATA_BOUNDARY
    assert call["authorization"] == runner.CELL_AUTHORIZATION
    assert call["execution"]["optimizer_build_count"] == 0
    assert call["execution"]["optimizer_step_count"] == 0
    assert call["execution"]["model_weight_update_count"] == 0
    assert call["execution"]["validation_payload_access_count"] == 0
    assert call["execution"]["test_payload_access_count"] == 0
    assert [value["verify_live_parents"] for value in verifier_calls] == [
        False,
        True,
        True,
    ]
    assert all(value["config"] is contract.raw for value in verifier_calls)
    assert all(value["expected_code_seal"] == code_seal for value in verifier_calls)


def test_runner_uses_final_artifact_builder_and_verifier_public_api() -> None:
    source = inspect.getsource(runner)
    assert "build_stage_b1_cell_payloads(" in source
    assert "verify_stage_b1_cell_shard(" in source
    assert "verify_stage_b1_outer_cell_shard" not in source


def test_runner_has_no_optimizer_or_direct_test_validation_loader() -> None:
    source = inspect.getsource(runner._compute_cell)
    assert "guarded_load_outer_targets(" in source
    assert source.count("guarded_load_outer_targets(") == 1
    assert "load_outer_evaluator_targets_v2" not in source
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
    assert "test_source" not in source
    assert "validation" not in source.lower()
