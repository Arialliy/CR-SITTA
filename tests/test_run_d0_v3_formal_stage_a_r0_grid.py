from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from analysis.d0_v3_formal_contract import CONDITIONS, CONFIG_FILE_SHA256, DATASETS
from scripts import run_d0_v3_formal_stage_a_r0_grid as grid


LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _definition(root: Path) -> grid.GridDefinition:
    config = root / "configs" / "formal.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("frozen: true\n", encoding="utf-8")
    output = root / "results" / "formal"
    cells: list[grid.GridCell] = []
    for dataset in DATASETS:
        for condition in CONDITIONS:
            index = len(cells)
            cells.append(
                grid.GridCell(
                    index=index,
                    dataset=dataset,
                    condition=condition,
                    candidate_path=(
                        output
                        / "candidate_phase"
                        / "shards"
                        / "R0"
                        / dataset
                        / condition
                    ),
                    outer_path=(
                        output
                        / "outer_phase"
                        / "shards"
                        / "R0"
                        / dataset
                        / condition
                    ),
                )
            )
    return grid.GridDefinition(
        config_path=config,
        config_sha256="a" * 64,
        output_root=output,
        cells=tuple(cells),
    )


def _fake_repo(tmp_path: Path) -> tuple[Path, grid.GridDefinition]:
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for relative in (
        grid.CANDIDATE_SCRIPT_RELATIVE,
        grid.OUTER_SCRIPT_RELATIVE,
        grid.LAUNCHER_SCRIPT_RELATIVE,
    ):
        path = root / relative
        path.write_text(f"# immutable test worker: {relative}\n", encoding="utf-8")
    return root, _definition(root)


def _argument(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


class FakeWorkers:
    def __init__(
        self,
        definition: grid.GridDefinition,
        *,
        fail_candidate_verify: tuple[str, str] | None = None,
    ) -> None:
        self.definition = definition
        self.fail_candidate_verify = fail_candidate_verify
        self.calls: list[tuple[tuple[str, ...], dict[str, str]]] = []
        self.by_key = {
            (cell.dataset, cell.condition): cell for cell in definition.cells
        }

    def __call__(
        self, command: tuple[str, ...], cwd: Path, environment: Any
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == self.definition.config_path.parents[1]
        env = dict(environment)
        self.calls.append((command, env))
        script = Path(command[1]).name
        if script == Path(grid.CANDIDATE_SCRIPT_RELATIVE).name:
            action = command[2]
            if action == "validate":
                value = {
                    "valid": True,
                    "config_sha256": self.definition.config_sha256,
                    "gpu_initialized": False,
                    "target_payload_opened": False,
                    "validation_payload_opened": False,
                    "test_payload_opened": False,
                    "formal_protocol_complete": False,
                    "stage2_authorized": False,
                }
                return subprocess.CompletedProcess(
                    command, 0, json.dumps(value), ""
                )
            if action == "run":
                key = (_argument(command, "--dataset"), _argument(command, "--condition"))
                self.by_key[key].candidate_path.mkdir(parents=True)
                return subprocess.CompletedProcess(command, 0, "{}\n", "")
            assert action == "verify"
            candidate_path = Path(_argument(command, "--path"))
            cell = next(
                value
                for value in self.definition.cells
                if value.candidate_path == candidate_path
            )
            if self.fail_candidate_verify == (cell.dataset, cell.condition):
                return subprocess.CompletedProcess(command, 7, "", "invalid shard")
            assert candidate_path.is_dir()
            value = {
                "valid": True,
                "formal": True,
                "dry_run": False,
                "episode_count": 640,
                "formal_protocol_complete": False,
                "stage2_authorized": False,
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(value), "")

        assert script == Path(grid.OUTER_SCRIPT_RELATIVE).name
        # The outer parser receives --config before its subcommand.
        action = command[4]
        key = (_argument(command, "--dataset"), _argument(command, "--condition"))
        cell = self.by_key[key]
        if action == "run":
            cell.outer_path.mkdir(parents=True)
            return subprocess.CompletedProcess(command, 0, str(cell.outer_path), "")
        assert action == "verify"
        assert cell.outer_path.is_dir()
        value = {
            "path": str(cell.outer_path),
            "dataset": cell.dataset,
            "condition": cell.condition,
            "replicate": "R0",
            "image_count": 64,
            "candidate_count": 10,
            "record_count": 640,
            "manifest_sha256": "b" * 64,
            "complete_sha256": "c" * 64,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(value), "")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_cpu_validate_freezes_exact_3_by_13_plan_without_output() -> None:
    result = grid.validate_grid_only()
    assert result["valid"] is True
    assert result["config_sha256"] == CONFIG_FILE_SHA256
    assert result["dataset_count"] == 3
    assert result["condition_count_per_dataset"] == 13
    assert result["cell_count"] == 39
    assert result["gpu_initialized"] is False
    assert result["filesystem_created"] is False
    assert result["paper_result"] is False
    assert result["formal_protocol_complete"] is False
    assert result["stage2_authorized"] is False


def test_serial_grid_resumes_only_verified_artifacts_and_records_immutable_status(
    tmp_path: Path,
) -> None:
    root, definition = _fake_repo(tmp_path)
    # Cell 0 is complete; cell 1 has only a candidate shard; all others are absent.
    definition.cells[0].candidate_path.mkdir(parents=True)
    definition.cells[0].outer_path.mkdir(parents=True)
    definition.cells[1].candidate_path.mkdir(parents=True)
    workers = FakeWorkers(definition)
    launcher = grid.R0GridLauncher(
        definition=definition,
        project_root=root,
        python_executable=Path(sys.executable),
        cuda_visible_device="1",
        executor=workers,
        session_id="test_serial_resume",
    )
    result = launcher.run()

    assert result["complete"] is True
    assert result["r0_grid_complete"] is True
    assert result["verified_cell_count"] == 39
    assert result["counts"] == {
        "candidate_ran_verified": 37,
        "candidate_existing_verified": 2,
        "outer_ran_verified": 38,
        "outer_existing_verified": 1,
    }
    assert result["command_count"] == 154
    assert result["paper_result"] is False
    assert result["scientific_selection_performed"] is False
    assert result["formal_protocol_complete"] is False
    assert result["stage2_authorized"] is False

    session = Path(result["path"])
    assert len(list((session / "cells").glob("*.json"))) == 39
    assert len(list((session / "commands").glob("*.json"))) == 154
    assert _json(session / "COMPLETE.json")["stage2_authorized"] is False
    manifest = _json(session / "MANIFEST.json")
    assert manifest["cell_count"] == 39
    assert manifest["overwrite_policy"] == "forbidden"
    assert manifest["paper_result"] is False
    assert manifest["formal_protocol_complete"] is False
    assert [value["dataset"] for value in manifest["cells"][:13]] == [
        "IRSTD-1K"
    ] * 13
    assert [value["condition"] for value in manifest["cells"][:13]] == list(
        CONDITIONS
    )

    first = _json(session / "cells" / f"{definition.cells[0].slug}.json")
    second = _json(session / "cells" / f"{definition.cells[1].slug}.json")
    third = _json(session / "cells" / f"{definition.cells[2].slug}.json")
    assert first["candidate_phase"]["disposition"] == "existing_verified"
    assert first["outer_phase"]["disposition"] == "existing_verified"
    assert second["candidate_phase"]["disposition"] == "existing_verified"
    assert second["outer_phase"]["disposition"] == "ran_verified"
    assert third["candidate_phase"]["disposition"] == "ran_verified"
    assert third["outer_phase"]["disposition"] == "ran_verified"

    # Every verifier is explicitly CPU-only, while every compute command sees
    # exactly the one requested physical device.  No worker receives an output
    # override, so only its frozen canonical destination can be published.
    for command, environment in workers.calls:
        is_verify = "verify" in command or "validate" in command
        if is_verify:
            assert environment["CUDA_VISIBLE_DEVICES"] == ""
        else:
            assert environment["CUDA_VISIBLE_DEVICES"] == "1"
            assert "--output" not in command
            assert "--label-free" not in command
        if (
            Path(command[1]).name
            == Path(grid.CANDIDATE_SCRIPT_RELATIVE).name
            and command[2] == "run"
        ):
            assert _argument(command, "--replicate") == "R0"
            assert _argument(command, "--max-images") == "64"
            assert _argument(command, "--cuda-visible-device") == "1"
        if Path(command[1]).name == Path(grid.OUTER_SCRIPT_RELATIVE).name and "run" in command:
            assert _argument(command, "--device") == "cuda:0"


def test_existing_invalid_candidate_fails_closed_without_run_delete_or_outer(
    tmp_path: Path,
) -> None:
    root, definition = _fake_repo(tmp_path)
    first = definition.cells[0]
    first.candidate_path.mkdir(parents=True)
    sentinel = first.candidate_path / "PARTIAL"
    sentinel.write_text("preserve me", encoding="utf-8")
    workers = FakeWorkers(
        definition,
        fail_candidate_verify=(first.dataset, first.condition),
    )
    launcher = grid.R0GridLauncher(
        definition=definition,
        project_root=root,
        python_executable=Path(sys.executable),
        cuda_visible_device="1",
        executor=workers,
        session_id="test_fail_closed",
    )
    with pytest.raises(grid.D0V3R0GridLauncherError, match="exit 7"):
        launcher.run()

    assert sentinel.read_text(encoding="utf-8") == "preserve me"
    assert not first.outer_path.exists()
    assert len(workers.calls) == 2  # CPU preflight, then CPU candidate verify.
    assert workers.calls[1][0][2] == "verify"
    assert all("candidate_gpu_run" not in path.name for path in (
        root / "results" / "formal" / "orchestration" / "r0_grid" / "runs"
        / "test_fail_closed" / "commands"
    ).glob("*.json"))
    session = (
        root
        / "results"
        / "formal"
        / "orchestration"
        / "r0_grid"
        / "runs"
        / "test_fail_closed"
    )
    failure = _json(session / "FAILED.json")
    assert failure["complete"] is False
    assert failure["r0_grid_complete"] is False
    assert failure["no_artifact_was_removed_or_replaced"] is True
    assert failure["paper_result"] is False
    assert failure["stage2_authorized"] is False
    assert not (session / "COMPLETE.json").exists()


def test_atomic_record_publication_never_replaces_existing_bytes(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "status.json"
    first = {"complete": False, "stage2_authorized": False}
    grid._atomic_write_json_noreplace(destination, first)
    original = destination.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        grid._atomic_write_json_noreplace(
            destination, {"complete": True, "stage2_authorized": True}
        )
    assert destination.read_bytes() == original
    assert _json(destination) == first


@pytest.mark.parametrize("value", ["", "0,1", "../1", "1 2", ".", ".."])
def test_visible_device_must_name_exactly_one_safe_device(value: str) -> None:
    with pytest.raises(grid.D0V3R0GridLauncherError, match="exactly one"):
        grid._validate_visible_device(value)


def test_launcher_rejects_anything_other_than_full_39_cell_grid(
    tmp_path: Path,
) -> None:
    root, definition = _fake_repo(tmp_path)
    shortened = grid.GridDefinition(
        config_path=definition.config_path,
        config_sha256=definition.config_sha256,
        output_root=definition.output_root,
        cells=definition.cells[:-1],
    )
    with pytest.raises(grid.D0V3R0GridLauncherError, match="full 39-cell"):
        grid.R0GridLauncher(
            definition=shortened,
            project_root=root,
            python_executable=Path(sys.executable),
            cuda_visible_device="1",
            executor=FakeWorkers(definition),
            session_id="short_grid",
        )
