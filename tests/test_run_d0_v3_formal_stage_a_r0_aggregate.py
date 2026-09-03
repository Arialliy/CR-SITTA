from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import run_d0_v3_formal_stage_a_r0_aggregate as runner


LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_validate_is_cpu_only_read_only_and_never_authorizes_stage2() -> None:
    script = Path(runner.__file__).resolve()
    completed = subprocess.run(
        [sys.executable, str(script), "validate"],
        cwd=script.parents[1],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["valid"] is True
    assert result["mode"] == "cpu_only_no_output"
    assert result["cell_count_required"] == 39
    assert result["outer_records_per_cell"] == 640
    assert result["replicate_evidence_count"] == 10
    assert result["filesystem_created"] is False
    assert result["raw_gt_opened"] is False
    assert result["test_payload_opened"] is False
    assert result["validation_payload_opened"] is False
    assert result["gpu_initialized"] is False
    assert result["paper_result"] is False
    assert result["stage2_authorized"] is False


def test_incomplete_preflight_fails_before_contract_reload_or_any_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "untouched"
    sentinel.mkdir()
    monkeypatch.setattr(
        runner,
        "_preflight",
        lambda _config: (_ for _ in ()).throw(
            runner.D0V3R0AggregateRunnerError("missing fixed cell")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_load_contract",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("must not load/create after failed preflight")
        ),
    )
    with pytest.raises(
        runner.D0V3R0AggregateRunnerError, match="missing fixed cell"
    ):
        runner.run_formal_aggregate(tmp_path / "config.yaml")
    assert list(sentinel.iterdir()) == []


def test_runner_refuses_a_process_that_already_initialized_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = sys.modules.get("torch")
    fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(
        runner.D0V3R0AggregateRunnerError, match="CUDA is initialized"
    ):
        runner._assert_cuda_not_initialized()
    if original is not None:
        monkeypatch.setitem(sys.modules, "torch", original)
