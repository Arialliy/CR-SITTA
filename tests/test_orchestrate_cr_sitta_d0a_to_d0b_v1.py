from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.orchestrate_cr_sitta_d0a_to_d0b_v1 as orchestrator


def _empty_proc(tmp_path: Path) -> Path:
    proc = tmp_path / "proc"
    proc.mkdir()
    return proc


def test_default_status_is_read_only_and_fixed_to_three_datasets(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    before = tuple(repository.rglob("*"))
    report = orchestrator.build_status(repository, proc_root=_empty_proc(tmp_path))
    after = tuple(repository.rglob("*"))
    assert before == after
    assert tuple(report["datasets"]) == orchestrator.DATASETS
    assert {value["state"] for value in report["datasets"].values()} == {
        "waiting_not_started"
    }
    assert report["writes_performed"] == 0
    assert report["d1_launched"] is False
    assert report["formal_test_allowed"] is False
    assert orchestrator.build_parser().parse_args([]).command is None


def test_trainer_process_match_requires_runner_and_exact_output_dir(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = repository / orchestrator.D0A_RESULT_RELATIVE / "IRSTD-1K"
    work = tmp_path / "work"
    work.mkdir()
    proc = tmp_path / "proc"
    proc.mkdir()

    def process(pid: int, argv: list[str]) -> None:
        root = proc / str(pid)
        root.mkdir()
        (root / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")
        (root / "cwd").symlink_to(work, target_is_directory=True)

    process(
        101,
        ["python", "train_cr_sitta_d0a.py", "--output-dir", str(output)],
    )
    process(
        102,
        ["python", "other.py", "--output-dir", str(output)],
    )
    process(
        103,
        ["python", "train_cr_sitta_d0a.py", "--output-dir", str(output) + "-other"],
    )
    assert orchestrator.trainer_processes_for_output(output, proc_root=proc) == (101,)


def test_new_empty_queued_directory_gets_stale_grace(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    output = repository / orchestrator.D0A_RESULT_RELATIVE / "IRSTD-1K"
    output.mkdir(parents=True)
    mtime = output.stat().st_mtime
    proc = _empty_proc(tmp_path)
    recent = orchestrator.inspect_dataset_state(
        repository,
        "IRSTD-1K",
        proc_root=proc,
        now=mtime + 10,
        stale_grace_seconds=300,
    )
    stale = orchestrator.inspect_dataset_state(
        repository,
        "IRSTD-1K",
        proc_root=proc,
        now=mtime + 301,
        stale_grace_seconds=300,
    )
    assert recent["state"] == "waiting_transient"
    assert stale["state"] == "failed_or_stale"


def _metrics_rows(*, batches: int = 3) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0
    for epoch in range(1, 1001):
        end = start + batches
        rows.append(
            {
                "epoch": epoch,
                "epoch_index": epoch - 1,
                "warm_flag": epoch <= 5,
                "batches": batches,
                "starting_optimizer_step": start,
                "ending_optimizer_step": end,
                "mean_clean_loss": 1.0,
                "mean_degraded_loss": 1.1,
                "mean_combined_loss": 1.05,
                "last_gradient_l2": 0.1,
                "duration_seconds": 0.2,
                "probe_counts": {
                    "lf_mask": sum(value % 2 == 0 for value in range(start, end)),
                    "hf_noise": sum(value % 2 == 1 for value in range(start, end)),
                },
                "probe_tensor_sha256s": [],
                "learning_rate": 0.05,
            }
        )
        start = end
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_metrics_requires_exact_1000_epoch_and_step_chain(tmp_path: Path) -> None:
    path = tmp_path / "train_metrics.jsonl"
    rows = _metrics_rows()
    _write_jsonl(path, rows)
    observed, digest = orchestrator._validate_metrics(
        path, expected_batches=3, warm_epochs=5, learning_rate=0.05
    )
    assert len(observed) == 1000
    assert observed[-1]["ending_optimizer_step"] == 3000
    assert len(digest) == 64

    rows[500]["starting_optimizer_step"] += 1
    _write_jsonl(path, rows)
    with pytest.raises(orchestrator.OrchestrationError, match="discontinuity"):
        orchestrator._validate_metrics(
            path, expected_batches=3, warm_epochs=5, learning_rate=0.05
        )


@pytest.mark.parametrize("mutation", ["missing_row", "duplicate_epoch", "nan"])
def test_metrics_fail_closed_on_incomplete_or_nonfinite(
    tmp_path: Path, mutation: str
) -> None:
    rows = _metrics_rows()
    if mutation == "missing_row":
        rows.pop()
    elif mutation == "duplicate_epoch":
        rows[4]["epoch"] = 4
    else:
        rows[7]["mean_combined_loss"] = float("nan")
    path = tmp_path / "train_metrics.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(orchestrator.OrchestrationError):
        orchestrator._validate_metrics(
            path, expected_batches=3, warm_epochs=5, learning_rate=0.05
        )


def test_export_command_uses_only_persisted_anchor_hashes(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    anchor = {
        "artifacts": {
            "final_checkpoint": {"path": "final.pth.tar", "sha256": "a" * 64},
            "run_contract": {"path": "run_contract.json", "sha256": "b" * 64},
            "full_train_freeze": {"path": "freeze.json", "sha256": "c" * 64},
        }
    }
    command = orchestrator.build_export_command(
        repository,
        "IRSTD-1K",
        anchor,
        python_executable=Path("/python"),
    )
    assert command[0] == "/python"
    assert "--trust-local-source" in command
    assert command[command.index("--expected-source-sha256") + 1] == "a" * 64
    assert command[command.index("--expected-run-contract-sha256") + 1] == "b" * 64
    assert command[command.index("--expected-full-train-freeze-sha256") + 1] == "c" * 64
    assert "--resume" not in command


def test_anchor_artifact_paths_cannot_escape_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    assert orchestrator._artifact_path(repository, "inside/file", "fixture") == (
        repository / "inside/file"
    )
    with pytest.raises(orchestrator.OrchestrationError, match="outside"):
        orchestrator._artifact_path(repository, str(tmp_path / "outside"), "fixture")
    with pytest.raises(orchestrator.OrchestrationError, match="contains '..'"):
        orchestrator._artifact_path(repository, "inside/../../outside", "fixture")


@pytest.mark.parametrize("wrong_zero", [False, 0.0, "0", None])
def test_access_counters_require_integer_zero(wrong_zero: Any) -> None:
    values = {field: 0 for field in orchestrator.ZERO_ACCESS_FIELDS}
    values[orchestrator.ZERO_ACCESS_FIELDS[0]] = wrong_zero
    with pytest.raises(orchestrator.OrchestrationError, match="integer zero"):
        orchestrator._require_zero_access(values, "fixture")


def test_safe_export_rejects_extra_artifact_before_d0b(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repository"
    paths = orchestrator._dataset_paths(repository, "IRSTD-1K")
    paths.output_dir.mkdir(parents=True)
    paths.safe_checkpoint.write_bytes(b"safe")
    expected = {
        "source_checkpoint",
        "safe_checkpoint",
        "exporter",
        "run_contract",
        "full_train_freeze",
        "protocol_config",
        "train_split",
        "archived_train_split",
        "completion_summary",
        "training_runner",
        "original_model_implementation",
        "adaptable_model_implementation",
        "smoke_gate",
    }
    receipt = {
        "dataset": "IRSTD-1K",
        "artifacts": {
            **{name: {"path": "placeholder", "sha256": "a" * 64} for name in expected},
            "test_metrics": {"path": "forbidden", "sha256": "b" * 64},
        },
    }
    paths.safe_export_receipt.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    import export_cr_sitta_d0a_safe_checkpoint as exporter

    monkeypatch.setattr(exporter, "verify_safe_export", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(exporter, "validate_repository_models", lambda state: None)
    with pytest.raises(orchestrator.OrchestrationError, match="roster differs"):
        orchestrator.verify_safe_export_against_anchor(
            repository,
            "IRSTD-1K",
            {"artifacts": {}},
        )


def test_runtime_roster_rejects_extra_path_before_it_can_be_opened(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    freeze_keys = {
        "train_cr_sitta_d0a.py",
        "configs/cr_sitta_d0a_train_v2.yaml",
        "train_fixed_split.py",
        "model/loss.py",
        "model/MSHNet_NSFPN.py",
        "model/NS_FPN.py",
        "model/diff_cross_attns.py",
        "tta/deteriorations/__init__.py",
        "tta/deteriorations/fourier_low_mask.py",
        "tta/deteriorations/high_frequency_noise.py",
        "tta/deteriorations/image_space.py",
        "SFS_MSDeformAttn/ops/functions/ms_deform_attn_func.py",
        "SFS_MSDeformAttn/ops/modules/ms_deform_attn.py",
    }
    freeze = {key: "a" * 64 for key in freeze_keys}
    run = {key: "a" * 64 for key in freeze_keys if key != "tta/deteriorations/__init__.py"}
    extension = repository / ".conda/lib/python3.10/site-packages/MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so"
    run[str(extension.resolve())] = "b" * 64
    poison = tmp_path / "test_payload_should_not_be_opened.bin"
    run[str(poison)] = "c" * 64
    with pytest.raises(orchestrator.OrchestrationError, match="roster differs"):
        orchestrator._validate_runtime_rosters(repository, run, freeze)
    assert not poison.exists()


def test_anchor_set_requires_identical_runtime_bytes_across_all_three(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    anchors = {
        dataset: ({"runtime_sha256": {"runtime.py": "a" * 64}}, chr(97 + index) * 64)
        for index, dataset in enumerate(orchestrator.DATASETS)
    }
    anchors["NUDT-SIRST"][0]["runtime_sha256"]["runtime.py"] = "b" * 64
    with pytest.raises(orchestrator.OrchestrationError, match="not identical"):
        orchestrator.publish_anchor_set(repository, anchors)


def test_safe_export_rejects_coordinated_but_different_weights(
    tmp_path: Path, monkeypatch
) -> None:
    import torch
    import export_cr_sitta_d0a_safe_checkpoint as exporter

    repository = tmp_path / "repository"
    paths = orchestrator._dataset_paths(repository, "IRSTD-1K")
    paths.output_dir.mkdir(parents=True)
    paths.safe_checkpoint.write_bytes(b"placeholder")
    anchor_names = {
        "final_checkpoint": "final.pth.tar",
        "safe_exporter": str(orchestrator.EXPORTER_RELATIVE),
        "run_contract": "run_contract.json",
        "full_train_freeze": str(orchestrator.D0A_FREEZE_RELATIVE),
        "protocol_config": str(orchestrator.D0A_CONFIG_RELATIVE),
        "official_train_split": "datasets/IRSTD-1K/img_idx/train_IRSTD-1K.txt",
        "archived_train_split": str(paths.archived_train_split.relative_to(repository)),
        "completion_summary": str(paths.summary.relative_to(repository)),
        "training_runner": "train_cr_sitta_d0a.py",
    }
    anchor = {
        "artifacts": {
            name: {"path": value, "sha256": chr(97 + index) * 64}
            for index, (name, value) in enumerate(anchor_names.items())
        },
        "frozen_runtime_sha256": {"model/MSHNet_NSFPN.py": "d" * 64},
    }
    freeze = repository / orchestrator.D0A_FREEZE_RELATIVE
    freeze.parent.mkdir(parents=True, exist_ok=True)
    smoke_path = repository / "results/cr_sitta/smoke/SMOKE_GATE.json"
    freeze.write_text(
        json.dumps(
            {"smoke_gate": {"path": str(smoke_path.relative_to(repository)), "sha256": "e" * 64}}
        )
        + "\n",
        encoding="utf-8",
    )
    receipt_paths = {
        "source_checkpoint": anchor_names["final_checkpoint"],
        "safe_checkpoint": str(paths.safe_checkpoint.relative_to(repository)),
        "exporter": anchor_names["safe_exporter"],
        "run_contract": anchor_names["run_contract"],
        "full_train_freeze": anchor_names["full_train_freeze"],
        "protocol_config": anchor_names["protocol_config"],
        "train_split": anchor_names["official_train_split"],
        "archived_train_split": anchor_names["archived_train_split"],
        "completion_summary": anchor_names["completion_summary"],
        "training_runner": anchor_names["training_runner"],
        "original_model_implementation": "model/MSHNet_NSFPN.py",
        "adaptable_model_implementation": "model/MSHNet_NSFPN_adaptable.py",
        "smoke_gate": str(smoke_path.relative_to(repository)),
    }
    receipt_hashes = {
        receipt_name: anchor["artifacts"][anchor_name]["sha256"]
        for receipt_name, anchor_name in {
            "source_checkpoint": "final_checkpoint",
            "exporter": "safe_exporter",
            "run_contract": "run_contract",
            "full_train_freeze": "full_train_freeze",
            "protocol_config": "protocol_config",
            "train_split": "official_train_split",
            "archived_train_split": "archived_train_split",
            "completion_summary": "completion_summary",
            "training_runner": "training_runner",
        }.items()
    }
    receipt_hashes.update(
        {
            "safe_checkpoint": "f" * 64,
            "original_model_implementation": "d" * 64,
            "adaptable_model_implementation": "1" * 64,
            "smoke_gate": "e" * 64,
        }
    )
    receipt = {
        "dataset": "IRSTD-1K",
        "artifacts": {
            name: {"path": path, "sha256": receipt_hashes[name]}
            for name, path in receipt_paths.items()
        },
    }
    paths.safe_export_receipt.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    monkeypatch.setattr(exporter, "verify_safe_export", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(exporter, "validate_repository_models", lambda state: None)
    monkeypatch.setattr(exporter, "_load_trusted_checkpoint", lambda *args, **kwargs: {"state_dict": {"w": torch.tensor([1.0])}})
    monkeypatch.setattr(exporter, "_cpu_tensor_state_dict", lambda value: value)
    monkeypatch.setattr(exporter, "_weights_only_load", lambda path: {"state_dict": {"w": torch.tensor([2.0])}})
    monkeypatch.setattr(exporter, "_validate_safe_payload", lambda value: ({}, value["state_dict"]))
    with pytest.raises(orchestrator.OrchestrationError, match="state_dict differs"):
        orchestrator.verify_safe_export_against_anchor(repository, "IRSTD-1K", anchor)


def test_d0b_gpu_commands_are_list_argv_and_fixed_cuda_zero(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    command = orchestrator._d0b_command(
        repository, Path("/python"), "teacher", "NUDT-SIRST"
    )
    assert isinstance(command, tuple)
    assert command[-4:] == ("--dataset", "NUDT-SIRST", "--device", "cuda:0")
    with pytest.raises(orchestrator.OrchestrationError):
        orchestrator._d0b_command(repository, Path("/python"), "test")
    with pytest.raises(orchestrator.OrchestrationError):
        orchestrator._d0b_command(repository, Path("/python"), "D1")


def test_gpu2_lease_waits_only_for_physical_gpu2(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    process_queries = 0
    sleeps: list[float] = []

    def runner(command, environment):
        nonlocal process_queries
        if "--query-gpu=index,uuid" in command:
            return orchestrator.CommandResult(
                tuple(command), 0, "0, GPU-zero\n1, GPU-one\n2, GPU-two\n", ""
            )
        if "--query-compute-apps=gpu_uuid,pid" in command:
            process_queries += 1
            payload = (
                "GPU-zero, 111\nGPU-two, 222\n"
                if process_queries == 1
                else "GPU-zero, 111\n"
            )
            return orchestrator.CommandResult(tuple(command), 0, payload, "")
        raise AssertionError(command)

    with orchestrator.gpu2_exclusive_lease(
        repository, command_runner=runner, sleep=sleeps.append, poll_seconds=0.25
    ) as gpu_uuid:
        assert gpu_uuid == "GPU-two"
    assert sleeps == [0.25]
    assert (repository / orchestrator.ORCHESTRATOR_RESULT_RELATIVE / ".physical_gpu2.execution.lock").is_file()


def test_gpu2_lock_leaf_symlink_fails_closed_before_gpu_query(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    root = repository / orchestrator.ORCHESTRATOR_RESULT_RELATIVE
    root.mkdir(parents=True)
    target = tmp_path / "outside.lock"
    target.write_text("do not open\n", encoding="utf-8")
    (root / ".physical_gpu2.execution.lock").symlink_to(target)
    called = False

    def runner(command, environment):
        nonlocal called
        called = True
        raise AssertionError("GPU query must not run")

    with pytest.raises(OSError):
        with orchestrator.gpu2_exclusive_lease(repository, command_runner=runner):
            pass
    assert called is False
    assert target.read_text(encoding="utf-8") == "do not open\n"


def test_logged_gpu_command_sets_exact_visibility_and_device(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    session = repository / "results" / "cr_sitta" / "session"
    session.mkdir(parents=True)
    monkeypatch.setattr(orchestrator, "verify_handoff_freeze", lambda _repository: ({}, "d" * 64))
    observed: dict[str, Any] = {}

    def runner(command, environment):
        observed["command"] = tuple(command)
        observed["environment"] = dict(environment)
        return orchestrator.CommandResult(tuple(command), 0, "{}\n", "")

    command = orchestrator._d0b_command(
        repository, Path("/python"), "candidate", "IRSTD-1K"
    )
    orchestrator._run_logged_command(
        repository,
        session,
        "gpu_step",
        command,
        gpu=True,
        command_runner=runner,
    )
    assert observed["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert observed["environment"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert observed["command"][-2:] == ("--device", "cuda:0")


def _patch_execute_prerequisites(monkeypatch, repository: Path, session: Path) -> None:
    anchors = {
        dataset: {"dataset": dataset, "artifacts": {}} for dataset in orchestrator.DATASETS
    }
    monkeypatch.setattr(orchestrator, "_validate_d0b_static_contract", lambda _repo: {})
    monkeypatch.setattr(orchestrator, "_wait_for_d0a", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orchestrator,
        "validate_dataset_completion",
        lambda _repo, dataset, **kwargs: anchors[dataset],
    )
    monkeypatch.setattr(
        orchestrator,
        "publish_completion_anchor",
        lambda _repo, dataset, payload, **kwargs: (payload, dataset[0].lower() * 64),
    )
    monkeypatch.setattr(
        orchestrator,
        "publish_anchor_set",
        lambda *args, **kwargs: ({}, "e" * 64),
    )
    monkeypatch.setattr(orchestrator, "publish_handoff_freeze", lambda *args, **kwargs: ({}, "f" * 64))
    monkeypatch.setattr(orchestrator, "verify_handoff_freeze", lambda *args, **kwargs: ({}, "f" * 64))
    monkeypatch.setattr(orchestrator, "_new_session_dir", lambda _repo: session)
    monkeypatch.setattr(
        orchestrator,
        "verify_completion_anchor",
        lambda _repo, dataset, **kwargs: (anchors[dataset], dataset[0].lower() * 64),
    )
    monkeypatch.setattr(orchestrator, "verify_safe_export_against_anchor", lambda *args, **kwargs: {})
    monkeypatch.setattr(orchestrator, "build_export_command", lambda _repo, dataset, anchor, **kwargs: ("export", dataset))
    monkeypatch.setattr(orchestrator, "_publish_failure_receipt", lambda *args, **kwargs: None)


def test_execute_has_global_phase_barriers_and_stops_at_d0b(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    session = repository / "session"
    session.mkdir()
    _patch_execute_prerequisites(monkeypatch, repository, session)
    calls: list[str] = []

    def logged(_repo, _session, name, command, **kwargs):
        calls.append(name)
        return orchestrator.CommandResult(tuple(command), 0, "{}", "")

    @contextmanager
    def lease(*args, **kwargs):
        yield "GPU-two"

    monkeypatch.setattr(orchestrator, "_run_logged_command", logged)
    monkeypatch.setattr(orchestrator, "gpu2_exclusive_lease", lease)
    monkeypatch.setattr(orchestrator, "physical_gpu_compute_pids", lambda *args, **kwargs: ())
    monkeypatch.setattr(
        orchestrator,
        "_publish_pipeline_complete",
        lambda _repo: ({"status": "protocol_complete_at_d0b"}, "a" * 64),
    )
    result = orchestrator.execute_pipeline(
        repository,
        proc_root=_empty_proc(tmp_path),
        python_executable=Path(sys.executable),
    )
    assert result["status"] == "protocol_complete_at_d0b"
    phase_calls = [name for name in calls if name.startswith("d0b_")]
    assert phase_calls == [
        "d0b_preflight",
        "d0b_freeze",
        *(f"d0b_{dataset}_teacher" for dataset in orchestrator.DATASETS),
        *(f"d0b_{dataset}_candidate" for dataset in orchestrator.DATASETS),
        *(f"d0b_{dataset}_outer" for dataset in orchestrator.DATASETS),
        "d0b_aggregate",
        "d0b_verify",
    ]
    assert all("D1" not in name and "test" not in name for name in calls)


def test_command_failure_stops_all_later_phases(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    session = repository / "session"
    session.mkdir()
    _patch_execute_prerequisites(monkeypatch, repository, session)
    calls: list[str] = []

    def logged(_repo, _session, name, command, **kwargs):
        calls.append(name)
        if name == "d0b_IRSTD-1K_candidate":
            raise orchestrator.OrchestrationError("injected failure")
        return orchestrator.CommandResult(tuple(command), 0, "{}", "")

    @contextmanager
    def lease(*args, **kwargs):
        yield "GPU-two"

    monkeypatch.setattr(orchestrator, "_run_logged_command", logged)
    monkeypatch.setattr(orchestrator, "gpu2_exclusive_lease", lease)
    monkeypatch.setattr(orchestrator, "physical_gpu_compute_pids", lambda *args, **kwargs: ())
    monkeypatch.setattr(orchestrator, "_publish_pipeline_complete", lambda _repo: pytest.fail("must not complete"))
    with pytest.raises(orchestrator.OrchestrationError, match="injected"):
        orchestrator.execute_pipeline(
            repository,
            proc_root=_empty_proc(tmp_path),
            python_executable=Path(sys.executable),
        )
    assert "d0b_IRSTD-1K_candidate" in calls
    assert not any("_outer" in name for name in calls)
    assert "d0b_aggregate" not in calls
    assert "d0b_verify" not in calls


def test_partial_safe_export_stops_before_any_d0b_phase(tmp_path: Path, monkeypatch) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    session = repository / "session"
    session.mkdir()
    _patch_execute_prerequisites(monkeypatch, repository, session)
    first = orchestrator._dataset_paths(repository, orchestrator.DATASETS[0])
    first.output_dir.mkdir(parents=True)
    first.safe_checkpoint.write_bytes(b"partial")
    calls: list[str] = []
    monkeypatch.setattr(
        orchestrator,
        "_run_logged_command",
        lambda *args, **kwargs: calls.append(str(args[2])),
    )
    with pytest.raises(orchestrator.OrchestrationError, match="partial SAFE_EXPORT"):
        orchestrator.execute_pipeline(
            repository,
            proc_root=_empty_proc(tmp_path),
            python_executable=Path(sys.executable),
        )
    assert not any(name.startswith("d0b_") for name in calls)


@pytest.mark.parametrize(
    ("scientific_status", "eligible", "allowed"),
    [
        ("scientific_no_eligible", [], False),
        ("scientific_eligible", ["R-E1"], True),
    ],
)
def test_pipeline_complete_records_science_but_never_launches_d1_or_test(
    tmp_path: Path,
    monkeypatch,
    scientific_status: str,
    eligible: list[str],
    allowed: bool,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    root = repository / orchestrator.ORCHESTRATOR_RESULT_RELATIVE
    aggregate = repository / orchestrator.D0B_RESULT_RELATIVE / "aggregate_phase" / "R0"
    aggregate.mkdir(parents=True)
    root.mkdir(parents=True)
    artifacts = {
        root / "D0A_COMPLETION_ANCHOR_SET.json": {},
        root / "HANDOFF_EXECUTION_FREEZE.json": {},
        repository / orchestrator.D0B_RESULT_RELATIVE / "PRE_RUN_FREEZE.json": {},
        aggregate / "manifest.json": {"formal_test_allowed": False},
        aggregate / "D0B_SCIENCE_DECISION.json": {
            "protocol_status": "protocol_complete",
            "scientific_status": scientific_status,
            "eligible_parameter_space_ids": eligible,
            "d1_train_internal_oof_allowed": allowed,
            "formal_test_allowed": False,
        },
        aggregate / "D1_AUTHORIZATION.json": {
            "protocol_status": "protocol_complete",
            "scientific_status": scientific_status,
            "eligible_parameter_space_ids": eligible,
            "d1_train_internal_oof_allowed": allowed,
            "formal_test_allowed": False,
        },
    }
    for path, payload in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "verify_handoff_freeze", lambda _repo: ({}, "f" * 64))
    payload, _digest = orchestrator._publish_pipeline_complete(repository)
    assert payload["scientific_status"] == scientific_status
    assert payload["d1_train_internal_oof_allowed"] is allowed
    assert payload["scientific_negative_is_normal_completion"] is (
        scientific_status == "scientific_no_eligible"
    )
    assert payload["terminal_action"] == "stop_after_d0b_verify"
    assert payload["d1_launched"] is False
    assert payload["formal_test_allowed"] is False
    assert (root / "PIPELINE_COMPLETE.json").stat().st_mode & 0o222 == 0


def test_reviewed_real_d0b_contract_proves_pilot64_train_subset() -> None:
    result = orchestrator._validate_d0b_static_contract(orchestrator.REPOSITORY)
    assert tuple(result) == orchestrator.DATASETS
    assert all(record["pilot_count"] == 64 for record in result.values())
    assert all(record["pilot_subset_of_official_train"] is True for record in result.values())
