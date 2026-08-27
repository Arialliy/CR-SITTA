from __future__ import annotations

import inspect
import json
from pathlib import Path
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
    body = {
        "schema_version": 1,
        "algorithm": "calibration-runtime-seal-v1",
        "bindings": [runner.asdict(binding)],
        "cache_lineage": {},
    }
    return runner.RuntimeSeal(
        schema_version=1,
        algorithm="calibration-runtime-seal-v1",
        bindings=(binding,),
        cache_lineage={},
        global_runtime_seal_sha256=runner._canonical_json_sha256(body),
    )


def _fake_contract(tmp_path: Path):
    return SimpleNamespace(
        output_root=tmp_path / "results" / "binary_tent" / "source_calibration_v1",
        execution={
            "outputs": {
                "stage1_aggregate_directory": "stage1/aggregate",
                "stage2_aggregate_directory": "final/aggregate",
            }
        },
        scientific={
            "inherited_source_limitation": {
                "checkpoint_role": "best_miou",
                "checkpoint_selection": "test_selected_during_source_training",
                "disclosure": "Synthetic test fixture for the inherited source limitation.",
            }
        },
    )


def _candidate_summary(candidate) -> dict[str, object]:
    return {
        "candidate": candidate.to_dict(),
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
    }


def _publish_stage1_shard(contract, seal, candidate, index: int) -> None:
    process_id = f"stage1-process-{index}"
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
        "paper_result": False,
        **scope_provenance,
    }
    runner._publish_directory(
        final=runner._stage1_shard_path(contract, candidate),
        primary_files={
            "records.jsonl": records,
            "run_summary.json": _candidate_summary(candidate),
            "provenance.json": provenance,
            "runtime_seal.json": seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_source_calibration_stage1_candidate_shard",
            "stage": 1,
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
    process_id = f"stage2-fresh-{process_index}"
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
        "fresh_process": True,
        "top3_in_frozen_order": frozen_top3,
        "candidate_summaries": candidate_summaries,
        "record_count": 234,
        "episode_count": 14976,
        "candidate_model_method_optimizer_rebuilt_before_each_candidate": True,
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
    }
    provenance = {
        "schema_version": 1,
        "stage": 2,
        "fresh_process": True,
        "process_id": process_id,
        "pid": 2000 + process_index,
        "top3_receipt": receipt_binding.path,
        "top3_receipt_sha256": receipt_binding.sha256,
        "global_runtime_seal_sha256": stage2_seal.global_runtime_seal_sha256,
        "runtime_audits": [],
        "protocol_audit": {key: True for key in REQUIRED_PROTOCOL_AUDIT},
        "test_image_opens": 0,
        "test_label_opens": 0,
        "method_label_accesses": 0,
        "paper_result": False,
        **scope_provenance,
    }
    runner._publish_directory(
        final=runner._stage2_shard_path(contract, process_id),
        primary_files={
            "records.jsonl": records,
            "run_summary.json": summary,
            "provenance.json": provenance,
            "runtime_seal.json": stage2_seal.to_dict(),
        },
        manifest_metadata={
            "artifact_type": "binary_tent_source_calibration_stage2_process_shard",
            "stage": 2,
            "process_id": process_id,
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
    assert tuple(contract.checkpoints) == DATASETS
    assert contract.execution["evaluation"]["entropy_hard_gate"] is False
    assert contract.execution["evaluation"]["performance_hard_gate"] is False
    assert (
        contract.cache_protocol["materialized_cache"]["official_consumers"][
            "method_facing"
        ]
        == "SourceCalibrationMethodInputDataset"
    )


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
    assert not tuple(destination.parent.glob(".shard.build-*"))

    with pytest.raises(FileExistsError, match="overwrite"):
        runner._publish_directory(
            final=destination,
            primary_files={"records.jsonl": records},
            manifest_metadata={},
            completion_metadata={},
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
        def __init__(self, command: list[str], env: dict[str, str]) -> None:
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


def test_parallel_scheduler_failure_terminates_kills_and_waits_every_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: list[object] = []

    class FakeProcess:
        def __init__(self, command: list[str], env: dict[str, str]) -> None:
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
            primary_files={"payload.json": {"builder": 2}},
            manifest_metadata={},
            completion_metadata={},
        )
    release_rename.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert failures == []
    assert json.loads((destination / "payload.json").read_text())["builder"] == 1
    assert not destination.with_name(".artifact.publish.lock").exists()

    raced_destination = tmp_path / "appears-during-rename"

    def destination_appears(source: Path, final: Path) -> None:
        final.mkdir()
        (final / "winner.txt").write_text("external winner", encoding="utf-8")
        real_rename(source, final)

    monkeypatch.setattr(runner, "_rename_directory_noreplace", destination_appears)
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        runner._publish_directory(
            final=raced_destination,
            primary_files={"payload.json": {"builder": "loser"}},
            manifest_metadata={},
            completion_metadata={},
        )
    assert (raced_destination / "winner.txt").read_text() == "external winner"
    assert not tuple(tmp_path.glob(".appears-during-rename.build-*"))
    assert not raced_destination.with_name(
        ".appears-during-rename.publish.lock"
    ).exists()
