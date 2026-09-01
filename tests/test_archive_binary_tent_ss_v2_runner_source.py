from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.archive_binary_tent_ss_v2_runner_source import (
    CodeSupplementError,
    RUNNER_ROLE,
    create_or_verify_supplement,
    sha256_file,
    verify_supplement,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "results/binary_tent/ss_calibration_v2_negative_archive"
SUPPLEMENT = (
    ROOT
    / "results/binary_tent/ss_calibration_v2_negative_archive_code_supplement_v1"
)
HISTORICAL_RUNNER_SHA256 = (
    "16b883e8effe2f264c49fef387c0c175cfbcd95dcaa3648b2770992ba4c61c48"
)
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    runner = project / "run_binary_tent_ss_calibration_v2.py"
    runner.write_bytes(b"historical-v2-runner\n")
    archive = project / "canonical"
    seal = archive / "stage1/aggregate/runtime_seal.json"
    _write_json(
        seal,
        {
            "bindings": [
                {
                    "role": RUNNER_ROLE,
                    "path": str(runner.resolve()),
                    "sha256": sha256_file(runner),
                    "bytes": runner.stat().st_size,
                }
            ]
        },
    )
    negative = archive / "NEGATIVE_RESULT.json"
    _write_json(negative, {"result_type": "negative"})
    (archive / "SHA256SUMS").write_text(
        f"{sha256_file(negative)}  NEGATIVE_RESULT.json\n"
        f"{sha256_file(seal)}  stage1/aggregate/runtime_seal.json\n",
        encoding="utf-8",
    )
    return project, archive, project / "supplement"


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_real_supplement_preserves_sealed_historical_runner() -> None:
    result = verify_supplement(SUPPLEMENT)
    assert result["runner_sha256"] == HISTORICAL_RUNNER_SHA256
    assert result["paper_result"] is False
    assert result["stage2_authorization"] is False
    runtime = json.loads(
        (CANONICAL / "stage1/aggregate/runtime_seal.json").read_text(
            encoding="utf-8"
        )
    )
    binding = next(item for item in runtime["bindings"] if item["role"] == RUNNER_ROLE)
    assert binding["sha256"] == HISTORICAL_RUNNER_SHA256


def test_create_is_idempotent_verify_only_and_refuses_tamper(tmp_path: Path) -> None:
    project, archive, destination = _fixture(tmp_path)
    first = create_or_verify_supplement(
        project_root=project, archive=archive, destination=destination
    )
    second = create_or_verify_supplement(
        project_root=project, archive=archive, destination=destination
    )
    assert first == second
    snapshot = destination / "code_snapshot/run_binary_tent_ss_calibration_v2.py"
    snapshot.write_bytes(b"tampered\n")
    with pytest.raises(CodeSupplementError, match="hash mismatch"):
        verify_supplement(destination)
    with pytest.raises(CodeSupplementError):
        create_or_verify_supplement(
            project_root=project, archive=archive, destination=destination
        )


def test_unsealed_runner_drift_blocks_before_copy(tmp_path: Path) -> None:
    project, archive, destination = _fixture(tmp_path)
    (project / "run_binary_tent_ss_calibration_v2.py").write_bytes(b"new runner\n")
    with pytest.raises(CodeSupplementError, match="differ"):
        create_or_verify_supplement(
            project_root=project, archive=archive, destination=destination
        )
    assert not destination.exists()


def test_extra_supplement_member_is_rejected(tmp_path: Path) -> None:
    project, archive, destination = _fixture(tmp_path)
    create_or_verify_supplement(
        project_root=project, archive=archive, destination=destination
    )
    (destination / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(CodeSupplementError, match="member set"):
        verify_supplement(destination)
