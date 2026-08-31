from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import shutil
import threading
from types import SimpleNamespace

import pytest
import torch

import run_binary_tent_source_calibration as runner
from tta.binary_tent_calibration_selector import (
    ALL_CANDIDATES,
    BN_PROTOCOLS,
    CONDITIONS,
    DATASETS,
    REQUIRED_HARD_GATES,
    REQUIRED_PROTOCOL_AUDIT,
)


def _counts(*, intersection: int, union: int = 30) -> dict[str, int]:
    return {
        "intersection_pixels": intersection,
        "union_pixels": union,
        "false_alarm_pixels": 3,
        "total_image_pixels": 256 * 256 * 64,
        "detected_targets": 7,
        "total_targets": 8,
    }


def _candidate_records(
    candidate,
    *,
    stage: int,
    process_id: str,
    score: int,
) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for dataset in DATASETS:
        for protocol in BN_PROTOCOLS:
            for corruption, severity in CONDITIONS:
                values.append(
                    runner.build_cell_record(
                        stage=stage,
                        process_id=process_id,
                        candidate=candidate,
                        dataset=dataset,
                        bn_protocol=protocol,
                        corruption=corruption,
                        severity=severity,
                        tent_pre=_counts(intersection=10),
                        tent_post=_counts(intersection=10 + score),
                        hard_gates={key: True for key in REQUIRED_HARD_GATES},
                        protocol_audit={
                            key: True for key in REQUIRED_PROTOCOL_AUDIT
                        },
                    )
                )
    return values


def _seal(tmp_path: Path) -> runner.RuntimeSeal:
    path = tmp_path / "seal-source.txt"
    path.write_text("immutable", encoding="utf-8")
    stat = path.stat()
    binding = runner.BoundFile(
        path=str(path),
        role="critical_code:test",
        sha256=runner.sha256_file(path),
        bytes=stat.st_size,
        device=stat.st_dev,
        inode=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
    )
    runtime_environment = runner._runtime_environment_contract()
    body = {
        "schema_version": 1,
        "algorithm": "calibration-runtime-seal-v1",
        "bindings": [runner.asdict(binding)],
        "cache_lineage": {},
        "runtime_environment": runtime_environment,
    }
    return runner.RuntimeSeal(
        schema_version=1,
        algorithm="calibration-runtime-seal-v1",
        bindings=(binding,),
        cache_lineage={},
        runtime_environment=runtime_environment,
        global_runtime_seal_sha256=runner._canonical_json_sha256(body),
    )


def _gpu_lease_receipt(tmp_path: Path, gpu_id: str = "1") -> dict[str, object]:
    lease_path = (
        tmp_path
        / "results/binary_tent/.source_calibration_physical_gpu_leases"
        / f"physical-gpu-{gpu_id}.lock"
    )
    descriptor = runner._acquire_recoverable_flock(
        lease_path, owner="synthetic-test-durable-gpu-lease"
    )
    os.close(descriptor)
    lease_stat = lease_path.stat()
    return {
        "launcher_managed": True,
        "physical_gpu_id": gpu_id,
        "lease_path": str(lease_path),
        "lease_device": int(lease_stat.st_dev),
        "lease_inode": int(lease_stat.st_ino),
        "launcher_pid": os.getpid(),
        "linux_parent_death_signal": "SIGTERM",
        "worker_parent_pid_guard_verified": True,
        "worker_holds_lease_for_process_lifetime": True,
        "stage_launcher_claim_fd_inherited_for_process_lifetime": True,
    }


def _worker_environment(gpu_id: str = "1") -> dict[str, object]:
    return {
        "environment": {
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "42",
        },
        "device": {"logical_index": 0, "visible_device_count": 1},
    }


def _tamper_json_and_reseal_artifact(
    artifact: Path, relative: str, mutate
) -> None:
    path = artifact / relative
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    runner._write_json(path, value)
    manifest_path = artifact / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = runner._artifact_files(
        artifact, excluded=("artifact_manifest.json", "COMPLETE.json")
    )
    runner._write_json(manifest_path, manifest)
    complete_path = artifact / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["artifact_manifest_sha256"] = runner.sha256_file(manifest_path)
    runner._write_json(complete_path, complete)


def _tamper_jsonl_and_reseal_artifact(
    artifact: Path, relative: str, mutate
) -> None:
    path = artifact / relative
    values = runner._load_jsonl(path)
    mutate(values)
    runner._write_jsonl(path, values)
    manifest_path = artifact / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = runner._artifact_files(
        artifact, excluded=("artifact_manifest.json", "COMPLETE.json")
    )
    runner._write_json(manifest_path, manifest)
    complete_path = artifact / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["artifact_manifest_sha256"] = runner.sha256_file(manifest_path)
    runner._write_json(complete_path, complete)


def _fake_contract(tmp_path: Path):
    return SimpleNamespace(
        output_root=tmp_path / "results" / "binary_tent" / "source_calibration_v1",
        publication_work_root=(
            tmp_path / "results" / "binary_tent" / ".source_calibration_publication_work"
        ),
        execution={
            "outputs": {
                "stage1_aggregate_directory": "stage1/aggregate",
                "stage2_aggregate_directory": "final/aggregate",
            },
            "launcher": {
                "failure_termination_timeout_seconds": 0.1,
                "lease_wait_heartbeat_seconds": 30,
                "physical_gpu_lease_directory": str(
                    tmp_path
                    / "results/binary_tent/.source_calibration_physical_gpu_leases"
                ),
            },
        },
        execution_path=tmp_path / "execution.yaml",
        scientific_path=tmp_path / "scientific.yaml",
        cache_protocol_path=tmp_path / "cache.yaml",
        scientific={
            "inherited_source_limitation": {
                "checkpoint_role": "best_miou",
                "checkpoint_selection": "test_selected_during_source_training",
                "disclosure": "Synthetic test fixture for the inherited source limitation.",
            }
        },
    )


def _candidate_summary(
    candidate,
    *,
    process_id: str | None = None,
    stage: int | None = None,
    gpu_lease: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "candidate": candidate.to_dict(),
        "fresh_process": True,
        "cell_record_count": 78,
        "episode_count": 4992,
        "model_method_optimizer_build_count": 6,
        "fresh_build_object_id_triples_unique": True,
        "checkpoint_loads": [
            {
                "dataset": dataset,
                "bn_protocol": bn_protocol,
                "checkpoint_sha256": "0" * 64,
                "checkpoint_wrapper": "synthetic-test-fixture",
                "checkpoint_state_dict_strict_load_verified": True,
                "checkpoint_matches_dataset": dataset,
            }
            for dataset in DATASETS
            for bn_protocol in BN_PROTOCOLS
        ],
        "fixed_seed_audit": {
            "seed": 42,
            "python_seed_applied": True,
            "numpy_seed_applied": True,
            "torch_seed_applied": True,
            "cuda_seed_applied": True,
        },
        "runtime_seconds": 0.0,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "target_access_order": "outer_evaluator_after_episode_return_only",
        "target_access_audit": runner._target_access_audit(
            tensor_evaluation_completed=True
        ),
        **({"process_id": process_id} if process_id is not None else {}),
        **({"stage": stage} if stage is not None else {}),
        **(
            {
                "physical_gpu_lease": gpu_lease,
                "worker_environment": _worker_environment(
                    str(gpu_lease["physical_gpu_id"])
                ),
            }
            if gpu_lease is not None
            else {}
        ),
    }


def _publish_stage1_shard(contract, seal, candidate, index: int) -> None:
    process_id = f"stage1-process-{index}"
    gpu_lease = _gpu_lease_receipt(contract.output_root.parents[2])
    records = _candidate_records(
        candidate,
        stage=1,
        process_id=process_id,
        score=index,
    )
    scope_provenance = runner._scope_provenance(contract)
    provenance = {
        "schema_version": 1,
        "stage": 1,
        "fresh_process": True,
        "process_id": process_id,
        "pid": 1000 + index,
        "candidate": candidate.to_dict(),
        "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        "runtime_audits": [],
        "protocol_audit": {key: True for key in REQUIRED_PROTOCOL_AUDIT},
        "hard_gate_set": list(REQUIRED_HARD_GATES),
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "physical_gpu_lease": gpu_lease,
        "worker_environment": _worker_environment(),
        "target_access_audit": runner._target_access_audit(
            tensor_evaluation_completed=True
        ),
        "paper_result": False,
        **scope_provenance,
    }
    runner._publish_directory(
        final=runner._stage1_shard_path(contract, candidate),
        work_root=contract.publication_work_root,
        primary_files={
            "records.jsonl": records,
            "run_summary.json": _candidate_summary(
                candidate,
                process_id=process_id,
                stage=1,
                gpu_lease=gpu_lease,
            ),
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_source_calibration_stage1_candidate_shard",
            "stage": 1,
            "fresh_process": True,
            "physical_gpu_lease": gpu_lease,
            "worker_environment": _worker_environment(),
            "process_id": process_id,
            "candidate": candidate.to_dict(),
            "record_count": 78,
            "episode_count": 4992,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            **scope_provenance,
        },
        completion_metadata={
            "stage": 1,
            "process_id": process_id,
            "candidate": candidate.to_dict(),
            "record_count": 78,
            "episode_count": 4992,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
            "scope": dict(runner.SCOPE),
        },
    )


def _publish_stage2_shard(
    contract,
    *,
    base_seal,
    stage2_seal,
    receipt_binding,
    top3,
    process_index: int,
) -> None:
    process_id = runner._stage2_slot_process_id(
        receipt_binding.sha256, process_index
    )
    gpu_lease = _gpu_lease_receipt(
        contract.output_root.parents[2], str(process_index)
    )
    records: list[dict[str, object]] = []
    candidate_summaries: list[dict[str, object]] = []
    for rank, candidate in enumerate(top3, start=1):
        records.extend(
            _candidate_records(
                candidate,
                stage=2,
                process_id=process_id,
                score=4 - rank,
            )
        )
        candidate_summaries.append(
            {"frozen_top3_rank": rank, **_candidate_summary(candidate)}
        )
    frozen_top3 = [value.to_dict() for value in top3]
    scope_provenance = runner._scope_provenance(contract)
    summary = {
        "stage": 2,
        "process_id": process_id,
        "slot_index": process_index,
        "fresh_process": True,
        "top3_in_frozen_order": frozen_top3,
        "candidate_summaries": candidate_summaries,
        "record_count": 234,
        "episode_count": 14976,
        "candidate_model_method_optimizer_rebuilt_before_each_candidate": True,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "physical_gpu_lease": gpu_lease,
        "worker_environment": _worker_environment(str(process_index)),
        "target_access_audit": runner._target_access_audit(
            tensor_evaluation_completed=True
        ),
    }
    provenance = {
        "schema_version": 1,
        "stage": 2,
        "fresh_process": True,
        "process_id": process_id,
        "slot_index": process_index,
        "pid": 2000 + process_index,
        "top3_receipt": receipt_binding.path,
        "top3_receipt_sha256": receipt_binding.sha256,
        "global_runtime_seal_sha256": stage2_seal.global_runtime_seal_sha256,
        "runtime_audits": [],
        "protocol_audit": {key: True for key in REQUIRED_PROTOCOL_AUDIT},
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "physical_gpu_lease": gpu_lease,
        "worker_environment": _worker_environment(str(process_index)),
        "target_access_audit": runner._target_access_audit(
            tensor_evaluation_completed=True
        ),
        "paper_result": False,
        **scope_provenance,
    }
    runner._publish_directory(
        final=runner._stage2_shard_path(contract, process_id),
        work_root=contract.publication_work_root,
        primary_files={
            "records.jsonl": records,
            "run_summary.json": summary,
            "provenance.json": provenance,
            "runtime_seal.json": stage2_seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_source_calibration_stage2_process_shard",
            "stage": 2,
            "fresh_process": True,
            "physical_gpu_lease": gpu_lease,
            "worker_environment": _worker_environment(str(process_index)),
            "process_id": process_id,
            "slot_index": process_index,
            "top3_in_frozen_order": frozen_top3,
            "record_count": 234,
            "episode_count": 14976,
            "global_runtime_seal_sha256": stage2_seal.global_runtime_seal_sha256,
            "stage1_runtime_seal_sha256": base_seal.global_runtime_seal_sha256,
            "top3_receipt_sha256": receipt_binding.sha256,
            **scope_provenance,
        },
        completion_metadata={
            "stage": 2,
            "process_id": process_id,
            "slot_index": process_index,
            "record_count": 234,
            "episode_count": 14976,
            "global_runtime_seal_sha256": stage2_seal.global_runtime_seal_sha256,
            "top3_receipt_sha256": receipt_binding.sha256,
            "scope": dict(runner.SCOPE),
        },
    )


def test_default_execution_contract_is_static_and_source_train_derived() -> None:
    contract = runner.load_contract()

    assert contract.output_root == (
        runner.PROJECT_ROOT / "results/binary_tent/source_calibration_v1"
    )
    assert contract.publication_work_root == (
        runner.PROJECT_ROOT
        / "results/binary_tent/.source_calibration_publication_work"
    )
    assert contract.execution["execution"]["cuda_device_order"] == "PCI_BUS_ID"
    assert tuple(contract.checkpoints) == DATASETS
    assert contract.execution["evaluation"]["entropy_hard_gate"] is False
    assert contract.execution["evaluation"]["performance_hard_gate"] is False
    assert (
        contract.cache_protocol["materialized_cache"]["official_consumers"][
            "method_facing"
        ]
        == "SourceCalibrationMethodInputDataset"
    )
    critical_relative = {
        path.relative_to(runner.PROJECT_ROOT).as_posix()
        for path in contract.critical_code_paths
    }
    assert {
        "corruptions/__init__.py",
        "corruptions/infrared_corruptions.py",
        "dataio/__init__.py",
        "dataio/research_dataset.py",
        "metrics/__init__.py",
        "metrics/official_metric_adapter.py",
        "run_corruption_pilot.py",
        "tta/__init__.py",
        "tta/adabn.py",
        "utils/metric.py",
    } <= critical_relative


def test_method_boundary_has_no_target_parameter_and_rejects_unsafe_metadata() -> None:
    assert tuple(inspect.signature(runner.run_label_free_episode).parameters) == (
        "runner",
        "image",
        "metadata",
    )

    class Recorder:
        def __init__(self) -> None:
            self.received = None

        def run_one_image(self, *, image, metadata):
            self.received = (image, metadata)
            return "returned"

    recorder = Recorder()
    image = torch.zeros(1, 3, 4, 4)
    safe = {"image_id": "a", "dataset": "toy", "severity": 0}
    assert (
        runner.run_label_free_episode(recorder, image=image, metadata=safe)
        == "returned"
    )
    assert recorder.received == (image, safe)

    with pytest.raises(runner.CalibrationExecutionError, match="unsafe"):
        runner.run_label_free_episode(
            recorder,
            image=image,
            metadata={**safe, "target": torch.zeros(1)},
        )


def test_cell_record_is_exact_selector_contract_without_transition_or_entropy() -> None:
    record = runner.build_cell_record(
        stage=1,
        process_id="fresh-1",
        candidate=ALL_CANDIDATES[0],
        dataset=DATASETS[0],
        bn_protocol=BN_PROTOCOLS[0],
        corruption="clean",
        severity=0,
        tent_pre=_counts(intersection=10),
        tent_post=_counts(intersection=11),
        hard_gates={key: True for key in REQUIRED_HARD_GATES},
        protocol_audit={key: True for key in REQUIRED_PROTOCOL_AUDIT},
    )

    assert set(record["hard_gates"]) == set(REQUIRED_HARD_GATES)
    assert set(record["protocol_audit"]) == set(REQUIRED_PROTOCOL_AUDIT)
    assert record["optimizer_steps_total"] == 64
    serialized = json.dumps(record).lower()
    assert "entropy" not in serialized
    assert "transition" not in serialized
    assert "erasure" not in serialized


def test_runtime_seal_monitor_detects_identity_and_byte_drift(tmp_path: Path) -> None:
    seal = _seal(tmp_path)
    monitor = runner.RuntimeSealMonitor(seal)
    assert monitor.assert_unchanged(stage="before")["verified"] is True

    bound = Path(seal.bindings[0].path)
    bound.write_text("changed", encoding="utf-8")
    with pytest.raises(runner.CalibrationExecutionError, match="drift"):
        monitor.assert_unchanged(stage="after", full_byte_rehash=True)


def test_shard_publish_is_atomic_verified_and_refuses_overwrite(tmp_path: Path) -> None:
    seal = _seal(tmp_path)
    destination = tmp_path / "stage1" / "shard"
    records = _candidate_records(
        ALL_CANDIDATES[0], stage=1, process_id="p1", score=1
    )
    runner._publish_directory(
        final=destination,
        work_root=tmp_path / "publication-work",
        primary_files={"records.jsonl": records},
        manifest_metadata={
            "record_count": 78,
            "episode_count": 4992,
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        },
        completion_metadata={
            "global_runtime_seal_sha256": seal.global_runtime_seal_sha256,
        },
    )
    verified = runner.verify_shard(
        destination, expected_seal_sha256=seal.global_runtime_seal_sha256
    )
    assert len(verified["records"]) == 78
    assert not tuple((tmp_path / "publication-work").glob("*.build-*"))

    with pytest.raises(FileExistsError, match="overwrite"):
        runner._publish_directory(
            final=destination,
            work_root=tmp_path / "publication-work",
            primary_files={"records.jsonl": records},
            manifest_metadata={},
            completion_metadata={},
        )
    extra = destination / "unexpected-empty-directory"
    extra.mkdir()
    with pytest.raises(
        runner.CalibrationExecutionError, match="exact directory tree"
    ):
        runner.verify_shard(
            destination, expected_seal_sha256=seal.global_runtime_seal_sha256
        )
    extra.rmdir()
    linked_parent = tmp_path / "linked-stage1"
    linked_parent.symlink_to(destination.parent, target_is_directory=True)
    with pytest.raises(runner.CalibrationExecutionError, match="ancestor.*symlink"):
        runner.verify_shard(
            linked_parent / destination.name,
            expected_seal_sha256=seal.global_runtime_seal_sha256,
        )


def test_stage1_aggregator_validates_ten_shards_and_calls_exact_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _fake_contract(tmp_path)
    seal = _seal(tmp_path)
    for index, candidate in enumerate(ALL_CANDIDATES, start=1):
        _publish_stage1_shard(contract, seal, candidate, index)

    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(
        runner, "capture_runtime_seal", lambda _contract: (seal, {})
    )
    args = SimpleNamespace(execution_config=tmp_path / "unused.yaml")
    result = runner.aggregate_stage1(args)

    destination = Path(result["destination"])
    receipt = json.loads(
        (destination / "stage1_top3_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["validation"]["episode_count"] == 49920
    assert receipt["top3"][0] == ALL_CANDIDATES[-1].to_dict()
    assert (destination / "COMPLETE.json").is_file()

    top3 = tuple(
        runner.Candidate.from_values(value["optimizer"], value["learning_rate"])
        for value in receipt["top3"]
    )
    receipt_binding = runner._bound_file(
        destination / "stage1_top3_receipt.json",
        role="stage1_top3_receipt",
    )
    stage2_seal = runner._extend_runtime_seal(seal, receipt_binding)
    for process_index in (1, 2):
        _publish_stage2_shard(
            contract,
            base_seal=seal,
            stage2_seal=stage2_seal,
            receipt_binding=receipt_binding,
            top3=top3,
            process_index=process_index,
        )

    final = runner.aggregate_final(args)
    final_destination = Path(final["destination"])
    final_receipt = json.loads(
        (final_destination / "final_selection_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert final_receipt["validation"]["total_calibration_episode_count"] == 79872
    assert final_receipt["both_bn_protocols_retained_for_formal_evaluation"] is True
    assert (final_destination / "COMPLETE.json").is_file()
    assert final["post_publish_verified"] is True
    final_manifest = json.loads(
        (final_destination / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert final_manifest["episode_count"] == 29952
    assert "stage2_episode_count" not in final_manifest

    _tamper_json_and_reseal_artifact(
        final_destination,
        "final_selection_receipt.json",
        lambda value: value.__setitem__("selected_candidate", {"tampered": True}),
    )
    with pytest.raises(
        runner.CalibrationExecutionError, match="selection receipt"
    ):
        runner._verify_final_aggregate(contract, seal, stage2_seal)


def test_parser_keeps_validate_launcher_and_worker_roles_separate() -> None:
    parser = runner.build_parser()
    assert parser.parse_args(["validate"]).role == "validate"
    args = parser.parse_args(
        [
            "worker-stage1",
            "--optimizer",
            "Adam",
            "--learning-rate",
            "1e-5",
            "--process-id",
            "fresh-process",
        ]
    )
    assert args.role == "worker-stage1"
    assert args.device == "cuda:0"


def test_parallel_scheduler_reuses_only_the_gpu_slot_that_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[object] = []
    remaining_polls = {"slow": 3, "fast": 0, "next": 0}

    class FakeProcess:
        def __init__(self, command: list[str], env: dict[str, str], **kwargs) -> None:
            self.command = command
            self.gpu = env["CUDA_VISIBLE_DEVICES"]
            self.pid = 100 + len(processes)
            self.reaped = False
            processes.append(self)

        def poll(self):
            name = self.command[0]
            if remaining_polls[name] > 0:
                remaining_polls[name] -= 1
                return None
            return 0

        def wait(self, timeout=None):
            self.reaped = True
            return 0

        def terminate(self):
            raise AssertionError("successful scheduler must not terminate workers")

        def kill(self):
            raise AssertionError("successful scheduler must not kill workers")

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    assignments = runner._run_parallel(
        [["slow"], ["fast"], ["next"]],
        ("1", "2"),
        poll_interval_seconds=0,
    )

    assert [value["gpu_id"] for value in assignments] == ["1", "2", "2"]
    assert [value.gpu for value in processes] == ["1", "2", "2"]
    assert all(value.reaped for value in processes)


def test_scheduler_emits_assignment_and_rate_limited_lease_wait_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class CompletedProcess:
        pid = 777

        def __init__(self, command, **kwargs) -> None:
            pass

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    real_acquire = runner._acquire_recoverable_flock
    attempts = 0

    def briefly_blocked(path: Path, *, owner: str) -> int:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise BlockingIOError("synthetic external GPU lease holder")
        return real_acquire(path, owner=owner)

    monkeypatch.setattr(runner.subprocess, "Popen", CompletedProcess)
    monkeypatch.setattr(runner, "_acquire_recoverable_flock", briefly_blocked)
    runner._run_parallel(
        [["worker"]],
        ("1",),
        poll_interval_seconds=0,
        slot_lock_directory=tmp_path / "gpu-leases",
        lease_wait_heartbeat_seconds=30,
    )
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    wait_events = [
        value
        for value in events
        if value["event"] == "binary_tent_source_calibration_gpu_lease_wait"
    ]
    assignment_events = [
        value
        for value in events
        if value["event"]
        == "binary_tent_source_calibration_worker_assigned"
    ]
    assert len(wait_events) == 1
    assert wait_events[0]["pending_worker_count"] == 1
    assert wait_events[0]["slots"] == {
        "configured": ["1"],
        "active": [],
        "lease_blocked": ["1"],
    }
    assert wait_events[0]["elapsed_seconds"] >= 0
    assert len(assignment_events) == 1
    assert assignment_events[0]["physical_gpu_id"] == "1"
    assert assignment_events[0]["pid"] == 777
    assert assignment_events[0]["pending_worker_count"] == 0
    assert assignment_events[0]["slots"] == {
        "configured": ["1"],
        "active": ["1"],
    }
    assert assignment_events[0]["elapsed_seconds"] >= 0


def test_parallel_scheduler_failure_terminates_kills_and_waits_every_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[object] = []

    class FakeProcess:
        def __init__(self, command: list[str], env: dict[str, str], **kwargs) -> None:
            self.command = command
            self.pid = 200 + len(processes)
            self.return_code = 7 if command[0] == "fail" else None
            self.terminated = False
            self.killed = False
            self.waited = False
            processes.append(self)

        def poll(self):
            return self.return_code

        def wait(self, timeout=None):
            if self.return_code is None and timeout is not None:
                raise runner.subprocess.TimeoutExpired(self.command, timeout)
            self.waited = True
            return 0 if self.return_code is None else self.return_code

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.return_code = -9

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    with pytest.raises(runner.CalibrationExecutionError, match="worker exited 7"):
        runner._run_parallel(
            [["fail"], ["hung"]],
            ("1", "2"),
            termination_timeout_seconds=0.001,
            poll_interval_seconds=0,
        )

    failed, hung = processes
    assert failed.waited is True
    assert hung.terminated is True
    assert hung.killed is True
    assert hung.waited is True
    assert all(value.poll() is not None for value in processes)


def test_publish_directory_exclusive_lock_and_atomic_noreplace_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact"
    entered_rename = threading.Event()
    release_rename = threading.Event()
    real_rename = runner._rename_directory_noreplace

    def paused_rename(source: Path, final: Path) -> None:
        entered_rename.set()
        assert release_rename.wait(timeout=5)
        real_rename(source, final)

    monkeypatch.setattr(runner, "_rename_directory_noreplace", paused_rename)
    failures: list[BaseException] = []

    def publish_first() -> None:
        try:
            runner._publish_directory(
                final=destination,
                work_root=tmp_path / "publication-work",
                primary_files={"payload.json": {"builder": 1}},
                manifest_metadata={},
                completion_metadata={},
            )
        except BaseException as error:  # pragma: no cover - assertion reports it
            failures.append(error)

    thread = threading.Thread(target=publish_first)
    thread.start()
    assert entered_rename.wait(timeout=5)
    with pytest.raises(FileExistsError, match="publish lock already exists"):
        runner._publish_directory(
            final=destination,
            work_root=tmp_path / "publication-work",
            primary_files={"payload.json": {"builder": 2}},
            manifest_metadata={},
            completion_metadata={},
        )
    release_rename.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []
    assert json.loads((destination / "payload.json").read_text())["builder"] == 1
    publication_locks = tuple(
        (tmp_path / "publication-work").glob("*.publish.lock")
    )
    assert len(publication_locks) == 1
    assert publication_locks[0].is_file() and not publication_locks[0].is_symlink()

    raced_destination = tmp_path / "appears-during-rename"

    def destination_appears(source: Path, final: Path) -> None:
        final.mkdir()
        (final / "winner.txt").write_text("external winner", encoding="utf-8")
        real_rename(source, final)

    monkeypatch.setattr(runner, "_rename_directory_noreplace", destination_appears)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        runner._publish_directory(
            final=raced_destination,
            work_root=tmp_path / "publication-work",
            primary_files={"payload.json": {"builder": "loser"}},
            manifest_metadata={},
            completion_metadata={},
        )
    assert (raced_destination / "winner.txt").read_text() == "external winner"
    assert not tuple((tmp_path / "publication-work").glob("*.build-*"))
    publication_locks = tuple(
        (tmp_path / "publication-work").glob("*.publish.lock")
    )
    assert len(publication_locks) == 2
    assert all(path.is_file() and not path.is_symlink() for path in publication_locks)


def test_scheduler_passes_lease_fd_and_parent_death_guard_to_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class FakeProcess:
        def __init__(self, command, **kwargs) -> None:
            self.pid = 901
            calls.append(kwargs)

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(runner.subprocess, "Popen", FakeProcess)
    lock_directory = tmp_path / "gpu-leases"
    stage_guard = runner._acquire_recoverable_flock(
        tmp_path / "publication-work/.stage1.launcher.lock",
        owner="stage-guard",
    )
    try:
        assignments = runner._run_parallel(
            [["worker"]],
            ("1", "2"),
            poll_interval_seconds=0,
            slot_lock_directory=lock_directory,
            inherited_guard_fds=(stage_guard,),
        )
    finally:
        os.close(stage_guard)

    assert assignments[0]["worker_inherits_gpu_lease"] is True
    assert assignments[0]["linux_parent_death_signal"] == "SIGTERM"
    assert len(calls[0]["pass_fds"]) == 2
    assert stage_guard in calls[0]["pass_fds"]
    inherited_fd = int(calls[0]["env"]["CALIBRATION_GPU_LEASE_FD"])
    assert calls[0]["env"]["CALIBRATION_GPU_LEASE_FD"] == str(inherited_fd)
    assert calls[0]["env"]["CALIBRATION_LAUNCHER_PID"] == str(os.getpid())
    assert calls[0]["env"]["CALIBRATION_STAGE_CLAIM_FDS"] == str(stage_guard)
    assert callable(calls[0]["preexec_fn"])
    assert calls[0]["stdout"] is runner.subprocess.DEVNULL

    reacquired = runner._acquire_recoverable_flock(
        lock_directory / "physical-gpu-1.lock", owner="test-reacquire"
    )
    os.close(reacquired)


def test_real_child_observes_inherited_flock_and_sigterm_parent_guard(
    tmp_path: Path,
) -> None:
    child = (
        "import ctypes, os, signal; "
        "fd=int(os.environ['CALIBRATION_GPU_LEASE_FD']); os.fstat(fd); "
        "stage_fd=int(os.environ['CALIBRATION_STAGE_CLAIM_FDS']); os.fstat(stage_fd); "
        "value=ctypes.c_int(0); libc=ctypes.CDLL(None); "
        "result=libc.prctl(2, ctypes.byref(value), 0, 0, 0); "
        "assert result == 0 and value.value == signal.SIGTERM; "
        "assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'"
    )
    stage_guard = runner._acquire_recoverable_flock(
        tmp_path / "publication-work/.stage1.launcher.lock", owner="stage-guard"
    )
    try:
        assignments = runner._run_parallel(
            [[runner.sys.executable, "-c", child]],
            ("1", "2"),
            poll_interval_seconds=0.001,
            slot_lock_directory=tmp_path / "gpu-leases",
            inherited_guard_fds=(stage_guard,),
        )
    finally:
        os.close(stage_guard)
    assert len(assignments) == 1
    assert assignments[0]["worker_inherits_gpu_lease"] is True


def test_physical_gpu_ids_and_inherited_env_require_canonical_ascii_decimal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert runner._gpu_ids("1,2") == ("1", "2")
    for raw in ("1,01", "01,2", "1,１"):
        with pytest.raises(runner.argparse.ArgumentTypeError, match="canonical"):
            runner._gpu_ids(raw)
    with pytest.raises(runner.CalibrationExecutionError, match="canonical"):
        runner._run_parallel([], ("1", "01"), poll_interval_seconds=0)

    monkeypatch.setenv("CALIBRATION_GPU_LEASE_FD", "0")
    monkeypatch.setenv("CALIBRATION_LAUNCHER_PID", str(os.getppid()))
    monkeypatch.setenv("CALIBRATION_PHYSICAL_GPU_ID", "01")
    with pytest.raises(runner.CalibrationExecutionError, match="canonical"):
        runner._verify_inherited_gpu_lease(required=True)


def test_posthoc_gpu_lease_verifier_requires_durable_inode_and_real_ancestors(
    tmp_path: Path,
) -> None:
    contract = _fake_contract(tmp_path)
    receipt = _gpu_lease_receipt(tmp_path)
    embedded = {
        "physical_gpu_lease": receipt,
        "worker_environment": _worker_environment(),
    }
    runner._verify_embedded_gpu_lease(
        contract=contract,
        manifest=embedded,
        provenance=embedded,
        summary=embedded,
        label="durable fixture",
    )

    wrong_inode = dict(receipt)
    wrong_inode["lease_inode"] = int(receipt["lease_inode"]) + 1
    wrong_embedded = {
        "physical_gpu_lease": wrong_inode,
        "worker_environment": _worker_environment(),
    }
    with pytest.raises(runner.CalibrationExecutionError, match="durable GPU lease inode"):
        runner._verify_embedded_gpu_lease(
            contract=contract,
            manifest=wrong_embedded,
            provenance=wrong_embedded,
            summary=wrong_embedded,
            label="wrong inode",
        )

    lease_directory = Path(str(receipt["lease_path"])).parent
    real_directory = lease_directory.with_name("lease-directory-backing")
    lease_directory.rename(real_directory)
    lease_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(runner.CalibrationExecutionError, match="ancestor.*symlink"):
        runner._verify_embedded_gpu_lease(
            contract=contract,
            manifest=embedded,
            provenance=embedded,
            summary=embedded,
            label="symlinked lease directory",
        )


def test_scheduler_releases_leases_even_when_reap_cleanup_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedProcess:
        pid = 902

        def __init__(self, command, **kwargs) -> None:
            pass

        def poll(self):
            return 9

        def wait(self, timeout=None):
            return 9

    monkeypatch.setattr(runner.subprocess, "Popen", FailedProcess)
    monkeypatch.setattr(
        runner,
        "_terminate_kill_and_reap",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("reap fault")),
    )
    lock_directory = tmp_path / "gpu-leases"
    with pytest.raises(
        runner.CalibrationExecutionError, match="cleanup also failed.*reap fault"
    ):
        runner._run_parallel(
            [["failed-worker"]],
            ("1", "2"),
            poll_interval_seconds=0,
            slot_lock_directory=lock_directory,
        )
    reacquired = runner._acquire_recoverable_flock(
        lock_directory / "physical-gpu-1.lock", owner="after-cleanup-fault"
    )
    os.close(reacquired)


def test_formal_worker_without_inherited_gpu_lease_fails_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "CALIBRATION_GPU_LEASE_FD",
        "CALIBRATION_LAUNCHER_PID",
        "CALIBRATION_PHYSICAL_GPU_ID",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        runner,
        "load_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("contract must not load before lease rejection")
        ),
    )
    args = SimpleNamespace(
        execution_config=Path("unused"),
        optimizer="Adam",
        learning_rate="1e-5",
        process_id="manual-bypass",
        device="cuda:0",
    )
    with pytest.raises(runner.CalibrationExecutionError, match="requires.*lease"):
        runner.run_stage1_worker(args)


def test_stage_launcher_claim_prevents_concurrent_scan_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _fake_contract(tmp_path)
    lock_path = contract.publication_work_root / ".stage1.launcher.lock"
    held = runner._acquire_recoverable_flock(lock_path, owner="first-launcher")
    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(
        runner,
        "capture_runtime_seal",
        lambda _contract: (_ for _ in ()).throw(
            AssertionError("concurrent launcher must fail before scan")
        ),
    )
    try:
        with pytest.raises(runner.CalibrationExecutionError, match="another stage1"):
            runner.launch_stage1(
                SimpleNamespace(
                    execution_config=tmp_path / "unused.yaml", gpu_ids="1,2"
                )
            )
    finally:
        os.close(held)


def test_stage1_safe_resume_runs_only_missing_and_rejects_semantic_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _fake_contract(tmp_path)
    seal = _seal(tmp_path)
    for index, candidate in enumerate(ALL_CANDIDATES[:-1], start=1):
        _publish_stage1_shard(contract, seal, candidate, index)
    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "capture_runtime_seal", lambda _contract: (seal, {}))
    missing_destination = runner._stage1_shard_path(
        contract, ALL_CANDIDATES[-1]
    )
    control_key = runner._publication_control_key(missing_destination)
    contract.publication_work_root.mkdir(parents=True, exist_ok=True)
    (contract.publication_work_root / f".{control_key}.publish.lock").write_text(
        "stale after SIGKILL\n", encoding="utf-8"
    )
    stale_build = contract.publication_work_root / f".{control_key}.build-crashed"
    stale_build.mkdir()

    def complete_missing(commands, gpu_ids, **kwargs):
        assert len(commands) == 1
        assert commands[0][commands[0].index("--optimizer") + 1] == "SGD"
        _publish_stage1_shard(contract, seal, ALL_CANDIDATES[-1], 10)
        return [{"gpu_id": "1", "pid": 910, "command": commands[0]}]

    monkeypatch.setattr(runner, "_run_parallel", complete_missing)
    args = SimpleNamespace(
        execution_config=tmp_path / "unused.yaml", gpu_ids="1,2"
    )
    result = runner.launch_stage1(args)
    assert result["launched"] == 1
    assert result["skipped_current_runtime_seal"] == 9
    assert missing_destination.is_dir()
    assert stale_build.is_dir()

    tampered = runner._stage1_shard_path(contract, ALL_CANDIDATES[0])
    _tamper_json_and_reseal_artifact(
        tampered,
        "run_summary.json",
        lambda value: value.__setitem__("fresh_process", False),
    )
    with pytest.raises(runner.CalibrationExecutionError, match="fresh process"):
        runner.launch_stage1(args)
    assert tampered.is_dir()


def test_stage1_safe_resume_rejects_self_resealed_completion_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _fake_contract(tmp_path)
    seal = _seal(tmp_path)
    for index, candidate in enumerate(ALL_CANDIDATES, start=1):
        _publish_stage1_shard(contract, seal, candidate, index)
    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "capture_runtime_seal", lambda _contract: (seal, {}))
    monkeypatch.setattr(
        runner,
        "_run_parallel",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("all candidate shards should have been resumable")
        ),
    )
    tampered = runner._stage1_shard_path(contract, ALL_CANDIDATES[0])
    _tamper_json_and_reseal_artifact(
        tampered,
        "COMPLETE.json",
        lambda value: value.__setitem__("episode_count", 4991),
    )
    with pytest.raises(runner.CalibrationExecutionError, match="completion episode count"):
        runner.launch_stage1(
            SimpleNamespace(
                execution_config=tmp_path / "unused.yaml", gpu_ids="1,2"
            )
        )
    assert tampered.is_dir()


def test_stage1_launcher_rejects_self_resealed_record_access_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _fake_contract(tmp_path)
    seal = _seal(tmp_path)
    for index, candidate in enumerate(ALL_CANDIDATES, start=1):
        _publish_stage1_shard(contract, seal, candidate, index)
    tampered = runner._stage1_shard_path(contract, ALL_CANDIDATES[0])

    def add_test_open(values) -> None:
        values[0]["test_image_opens"] = 1

    _tamper_jsonl_and_reseal_artifact(tampered, "records.jsonl", add_test_open)
    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "capture_runtime_seal", lambda _contract: (seal, {}))
    with pytest.raises(Exception, match="test/label access firewall"):
        runner.launch_stage1(
            SimpleNamespace(
                execution_config=tmp_path / "unused.yaml", gpu_ids="1,2"
            )
        )
    assert tampered.is_dir()


def test_stage2_receipt_bound_safe_resume_runs_only_missing_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _fake_contract(tmp_path)
    seal = _seal(tmp_path)
    for index, candidate in enumerate(ALL_CANDIDATES, start=1):
        _publish_stage1_shard(contract, seal, candidate, index)
    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "capture_runtime_seal", lambda _contract: (seal, {}))
    args = SimpleNamespace(execution_config=tmp_path / "unused.yaml")
    runner.aggregate_stage1(args)
    receipt_path = runner._stage1_receipt_path(contract)
    receipt_binding = runner._bound_file(
        receipt_path, role="stage1_top3_receipt"
    )
    stage2_seal = runner._extend_runtime_seal(seal, receipt_binding)
    top3 = runner._load_top3(receipt_path)
    _publish_stage2_shard(
        contract,
        base_seal=seal,
        stage2_seal=stage2_seal,
        receipt_binding=receipt_binding,
        top3=top3,
        process_index=1,
    )
    missing_process_id = runner._stage2_slot_process_id(receipt_binding.sha256, 2)
    missing_destination = runner._stage2_shard_path(contract, missing_process_id)
    control_key = runner._publication_control_key(missing_destination)
    (contract.publication_work_root / f".{control_key}.publish.lock").write_text(
        "stale after SIGKILL\n", encoding="utf-8"
    )
    stale_build = contract.publication_work_root / f".{control_key}.build-crashed"
    stale_build.mkdir()

    def complete_missing(commands, gpu_ids, **kwargs):
        assert len(commands) == 1
        assert commands[0][commands[0].index("--slot-index") + 1] == "2"
        _publish_stage2_shard(
            contract,
            base_seal=seal,
            stage2_seal=stage2_seal,
            receipt_binding=receipt_binding,
            top3=top3,
            process_index=2,
        )
        return [{"gpu_id": "1", "pid": 920, "command": commands[0]}]

    monkeypatch.setattr(runner, "_run_parallel", complete_missing)
    launch_args = SimpleNamespace(
        execution_config=tmp_path / "unused.yaml",
        gpu_ids="1,2",
        top3_receipt=None,
    )
    result = runner.launch_stage2(launch_args)
    assert result["launched"] == 1
    assert result["skipped_current_runtime_seal"] == 1
    assert result["top3_receipt"] == str(receipt_path.resolve())
    assert missing_destination.is_dir()
    assert stale_build.is_dir()

    slot1 = runner._stage2_shard_path(
        contract, runner._stage2_slot_process_id(receipt_binding.sha256, 1)
    )
    manifest_path = slot1 / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["top3_receipt_sha256"] = "0" * 64
    runner._write_json(manifest_path, manifest)
    complete_path = slot1 / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["artifact_manifest_sha256"] = runner.sha256_file(manifest_path)
    runner._write_json(complete_path, complete)
    with pytest.raises(runner.CalibrationExecutionError, match="receipt SHA256"):
        runner.launch_stage2(launch_args)
    assert slot1.is_dir()

    aggregate = runner._stage1_aggregate_path(contract)

    def drift_denominator(values) -> None:
        values[0]["endpoints"]["tent_pre"]["total_image_pixels"] -= 1

    _tamper_jsonl_and_reseal_artifact(
        aggregate, "stage1_records.jsonl", drift_denominator
    )
    with pytest.raises(runner.CalibrationExecutionError, match="total_image_pixels"):
        runner.launch_stage2(launch_args)
    assert aggregate.is_dir()


def test_stage2_launcher_rejects_self_resealed_endpoint_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _fake_contract(tmp_path)
    seal = _seal(tmp_path)
    for index, candidate in enumerate(ALL_CANDIDATES, start=1):
        _publish_stage1_shard(contract, seal, candidate, index)
    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "capture_runtime_seal", lambda _contract: (seal, {}))
    args = SimpleNamespace(execution_config=tmp_path / "unused.yaml")
    runner.aggregate_stage1(args)
    receipt_path = runner._stage1_receipt_path(contract)
    receipt_binding = runner._bound_file(
        receipt_path, role="stage1_top3_receipt"
    )
    stage2_seal = runner._extend_runtime_seal(seal, receipt_binding)
    top3 = runner._load_top3(receipt_path)
    for process_index in (1, 2):
        _publish_stage2_shard(
            contract,
            base_seal=seal,
            stage2_seal=stage2_seal,
            receipt_binding=receipt_binding,
            top3=top3,
            process_index=process_index,
        )
    slot1 = runner._stage2_shard_path(
        contract, runner._stage2_slot_process_id(receipt_binding.sha256, 1)
    )

    def break_endpoint(values) -> None:
        values[0]["endpoints"]["tent_post"]["intersection_pixels"] = 31

    _tamper_jsonl_and_reseal_artifact(slot1, "records.jsonl", break_endpoint)
    with pytest.raises(Exception, match="intersection_pixels exceeds union_pixels"):
        runner.launch_stage2(
            SimpleNamespace(
                execution_config=tmp_path / "unused.yaml",
                gpu_ids="1,2",
                top3_receipt=None,
            )
        )
    assert slot1.is_dir()


def test_receipt_path_is_exact_and_receipt_binding_detects_tamper(
    tmp_path: Path,
) -> None:
    contract = _fake_contract(tmp_path)
    receipt = runner._stage1_receipt_path(contract)
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"receipt_type":"stage1_top3"}\n', encoding="utf-8")
    wrong = tmp_path / "stage1_top3_receipt.json"
    with pytest.raises(runner.CalibrationExecutionError, match="output_root"):
        runner._resolve_top3_receipt(contract, wrong)
    seal = runner._extend_runtime_seal(
        _seal(tmp_path), runner._bound_file(receipt, role="stage1_top3_receipt")
    )
    monitor = runner.RuntimeSealMonitor(seal)
    monitor.assert_unchanged(stage="entry", active_paths=(receipt,))
    receipt.write_text('{"receipt_type":"tampered"}\n', encoding="utf-8")
    with pytest.raises(runner.CalibrationExecutionError, match="drift"):
        monitor.assert_unchanged(stage="later", active_paths=(receipt,))


def test_cross_run_invariants_and_selector_reject_tampered_evidence() -> None:
    stage1_records = [
        record
        for index, candidate in enumerate(ALL_CANDIDATES, start=1)
        for record in _candidate_records(
            candidate, stage=1, process_id=f"p-{index}", score=index
        )
    ]
    bad_denominator = [dict(value) for value in stage1_records]
    bad_denominator[0] = json.loads(json.dumps(bad_denominator[0]))
    bad_denominator[0]["endpoints"]["tent_pre"]["total_image_pixels"] -= 1
    with pytest.raises(runner.CalibrationExecutionError, match="total_image_pixels"):
        runner._verify_cross_run_endpoint_invariants(bad_denominator)

    false_fresh = json.loads(json.dumps(stage1_records))
    false_fresh[0]["fresh_process"] = False
    with pytest.raises(Exception, match="fresh process"):
        runner.select_stage1_top3(false_fresh)


def test_cache_external_provenance_and_parent_tensor_hash_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = DATASETS[0]
    ids = tuple(f"image-{index:02d}" for index in range(64))
    ids_sha = runner.ordered_ids_sha256(ids)
    protocol = tmp_path / "cache-protocol.yaml"
    protocol.write_text("fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    train = tmp_path / "train.txt"
    test = tmp_path / "test.txt"
    train.write_text("train\n", encoding="utf-8")
    test.write_text("test\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    root = cache_root / dataset
    root.mkdir(parents=True)
    runner._write_json(root / "COMPLETE.json", {})
    runner._write_json(root / "method_input_manifest.json", {})
    source_manifests = {
        "algorithm": "sorted-id-image_sha256-mask_sha256-lf-v1",
        "combined_sha256": "1" * 64,
        "images_sha256": "2" * 64,
        "masks_sha256": "3" * 64,
    }
    gt_hash = "4" * 64
    conditions = [
        {
            "corruption": corruption,
            "severity": severity,
            "tensor_sequence_sha256": "5" * 64,
            "parent_pilot_tensor_sequence_sha256": "5" * 64,
            "gt_mask_tensor_sequence_sha256": gt_hash,
        }
        for corruption, severity in CONDITIONS
    ]
    manifest = {
        "dataset": dataset,
        "image_ids": list(ids),
        "ordered_ids_sha256": ids_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": "6" * 64,
        "train_split": str(train),
        "test_split_metadata": str(test),
        "train_split_sha256": "7" * 64,
        "test_split_sha256": "8" * 64,
        "split_role": "fixed_train_derived_sha256_ranked_64",
        "source_file_manifest_sha256": "tampered",
        "source_manifests": source_manifests,
        "fixed_test_boundary": {
            "test_dataset_constructed": False,
            "test_images_opened": 0,
            "test_masks_opened": 0,
        },
        "label_firewall": {
            "method_received_labels": False,
            "targets_written_for_outer_evaluator_only": True,
        },
        "targets": {
            "method_facing_access": "forbidden",
            "role": "outer_source_side_calibration_evaluator_only",
            "path": "outer_evaluator/targets.npy",
            "file_sha256": "9" * 64,
            "tensor_sequence_sha256": gt_hash,
            "parent_pilot_tensor_sequence_sha256": gt_hash,
        },
        "conditions": conditions,
        "files": {"outer_evaluator/targets.npy": {"sha256": "9" * 64}},
    }
    dataset_contract = {
        "train_split_sha256": "7" * 64,
        "test_split_sha256": "8" * 64,
        "pilot_source_manifest_sha256": "1" * 64,
        "expected_gt_tensor_sequence_sha256": gt_hash,
    }
    anchor = {
        "dataset_contract": dataset_contract,
        "selected_ids": ids,
        "train_path": train.resolve(),
        "test_path": test.resolve(),
        "source_manifests": source_manifests,
        "pilot_by_condition": {
            pair: {
                "model_input_tensor_sha256": "5" * 64,
                "gt_mask_tensor_sha256": gt_hash,
            }
            for pair in CONDITIONS
        },
    }
    contract = SimpleNamespace(
        cache_root=cache_root,
        cache_protocol_path=protocol,
        scientific={
            "source_train_subsets": {
                "datasets": {
                    dataset: {
                        "ordered_ids_sha256": ids_sha,
                        "checkpoint_sha256": "6" * 64,
                    }
                }
            }
        },
        checkpoints={dataset: checkpoint.resolve()},
        execution={"cache_protocol": {"sha256": runner.sha256_file(protocol)}},
    )
    monkeypatch.setattr(
        runner, "_verify_parent_pilot_anchor", lambda *_args: (anchor, ())
    )
    monkeypatch.setattr(
        runner,
        "verify_cache_artifact",
        lambda *_args, **_kwargs: (manifest, {"manifest_sha256": "a" * 64}),
    )
    with pytest.raises(runner.CalibrationExecutionError, match="source-file manifest"):
        runner._verify_cache_context(contract, dataset)

    manifest["source_file_manifest_sha256"] = "1" * 64
    conditions[0]["parent_pilot_tensor_sequence_sha256"] = "b" * 64
    with pytest.raises(
        runner.CalibrationExecutionError, match="cache/declared-parent tensor hash"
    ):
        runner._verify_cache_context(contract, dataset)


def test_validate_only_never_loads_arrays_models_cuda_or_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _fake_contract(tmp_path)
    seal = _seal(tmp_path)
    caches = {
        dataset: SimpleNamespace(manifest_sha256=str(index) * 64)
        for index, dataset in enumerate(DATASETS, start=1)
    }
    monkeypatch.setattr(runner, "load_contract", lambda _path: contract)
    monkeypatch.setattr(runner, "capture_runtime_seal", lambda _contract: (seal, caches))
    monkeypatch.setattr(runner.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        runner.np,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("np.load forbidden")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_build_fast_runner",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model construction forbidden")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_publish_directory",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("output publication forbidden")
        ),
    )
    result = runner.validate_only(
        SimpleNamespace(execution_config=tmp_path / "unused.yaml")
    )
    assert result["cache_tensor_arrays_loaded"] == 0
    assert result["target_access_audit"][
        "target_integrity_bytes_hashed_before_adaptation"
    ] is True
    assert result["target_access_audit"][
        "target_tensor_deserialized_and_indexed_after_all_cell_episodes_complete"
    ] is False
    assert not contract.output_root.exists()


def test_progress_event_and_gpu_smoke_cli_are_fixed(capsys) -> None:
    runner._emit_cell_progress(
        stage=1,
        process_id="p1",
        candidate=ALL_CANDIDATES[0],
        dataset=DATASETS[0],
        bn_protocol=BN_PROTOCOLS[0],
        corruption="clean",
        severity=0,
        completed_cells=1,
    )
    event = json.loads(capsys.readouterr().err)
    assert event["cell_images"] == 64
    assert event["completed_episodes_for_candidate"] == 64
    smoke = runner.build_parser().parse_args(["gpu-smoke", "--full-cell"])
    assert smoke.role == "gpu-smoke"
    assert smoke.full_cell_smoke is True
    assert all(hasattr(runner, name) for name in runner.__all__)
