from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

import recover_cr_sitta_d0a_v2 as recovery


RUNNER_SOURCE = '''\
def derive_probe_seed(protocol_id, global_seed, dataset_id, human_epoch, image_id, probe_id):
    return hash((protocol_id, global_seed, dataset_id, human_epoch, image_id, probe_id))

def _restore_rng_state(payload):
    state = payload.get("rng_state")
    if not state:
        return
    raise RuntimeError("fixture must never restore RNG")
'''

LOADER_SOURCE = '''\
class Generator:
    def manual_seed(self, value):
        return value

def make_train_loader(dataset, config, human_epoch):
    generator = Generator()
    generator.manual_seed(config["seed"] + human_epoch)
    return generator
'''


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _fixture(tmp_path: Path, *, epoch: int = 3) -> dict[str, Any]:
    repository = tmp_path / "repository"
    run_dir = repository / "results" / "cr_sitta" / "d0a_v2" / "TOY-SIRST"
    run_dir.mkdir(parents=True)
    (run_dir / "splits").mkdir()

    for relative in recovery.REQUIRED_FROZEN_RUNTIME:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "train_cr_sitta_d0a.py":
            path.write_text(RUNNER_SOURCE, encoding="utf-8")
        elif relative == "train_fixed_split.py":
            path.write_text(LOADER_SOURCE, encoding="utf-8")
        else:
            path.write_text("# frozen fixture runtime\n", encoding="utf-8")

    split_path = repository / "datasets" / "TOY-SIRST" / "img_idx" / "train.txt"
    split_path.parent.mkdir(parents=True)
    split_path.write_text("a\nb\nc\nd\n", encoding="utf-8")
    (run_dir / "splits" / "train.txt").write_bytes(split_path.read_bytes())
    split_sha = recovery.sha256_file(split_path)
    protocol_path = repository / "configs" / "cr_sitta_d0a_train_v2.yaml"
    protocol_sha = recovery.sha256_file(protocol_path)

    run_config = {
        "schema_version": 1,
        "protocol_id": recovery.PROTOCOL_ID,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "host_architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "data_mode": "full_train_only",
        "dataset": "TOY-SIRST",
        "root": str(repository / "datasets" / "TOY-SIRST"),
        "train_split": str(split_path),
        "expected_train_split_sha256": split_sha,
        "expected_train_images": 4,
        "expected_train_corpus_manifest_sha256": "c" * 64,
        "known_train_size_mismatches": [],
        "epochs": 1000,
        "batch_size": 2,
        "seed": 42,
        "expected_state_dict_keys": 1,
        "output_dir": str(run_dir),
        "train_only_smoke": False,
        "max_train_batches": None,
        "probe_schedule": "deterministic_alternating_per_optimizer_step",
        "checkpoint_selection": "fixed_final_epoch_train_only",
        "validation_payload_access_allowed": False,
        "test_payload_access_allowed": False,
    }
    split_manifest = {
        "role": "official_train_only",
        "train_count": 4,
        "train_split_sha256": split_sha,
        "train_corpus_manifest_sha256": "c" * 64,
        "known_train_size_mismatches": [],
        "test_split_reads": 0,
        "test_image_opens": 0,
        "test_mask_opens": 0,
        "validation_split_reads": 0,
        "validation_image_opens": 0,
        "validation_mask_opens": 0,
    }
    firewall = {
        "implementation_has_test_loader": False,
        "implementation_has_validation_loader": False,
        "test_split_reads": 0,
        "test_image_opens": 0,
        "test_mask_opens": 0,
        "validation_split_reads": 0,
        "validation_image_opens": 0,
        "validation_mask_opens": 0,
    }
    runtime_hashes = {
        relative: recovery.sha256_file(repository / relative)
        for relative in recovery.REQUIRED_FROZEN_RUNTIME
    }
    contract_runtime = {
        key: value
        for key, value in runtime_hashes.items()
        if key != "tta/deteriorations/__init__.py"
    }
    contract = {
        "run_config": run_config,
        "split_manifest": split_manifest,
        "access_firewall": firewall,
        "runtime_sha256": contract_runtime,
    }
    contract_path = run_dir / "run_contract.json"
    _write_json(contract_path, contract)
    freeze = {
        "schema_version": 1,
        "protocol_id": recovery.PROTOCOL_ID,
        "status": "runtime_bytes_frozen_full_training_in_progress",
        "science_result": False,
        "formal_test_authorized": False,
        "tta_authorized": False,
        "runtime_sha256": runtime_hashes,
        "started_run_contract": {
            "dataset": "TOY-SIRST",
            "path": str(contract_path.relative_to(repository)),
            "sha256": recovery.sha256_file(contract_path),
        },
    }
    freeze_path = run_dir.parent / "FULL_TRAIN_FREEZE.json"
    _write_json(freeze_path, freeze)

    steps_per_epoch = 2
    rows = [
        {
            "epoch": human_epoch,
            "batches": steps_per_epoch,
            "starting_optimizer_step": steps_per_epoch * (human_epoch - 1),
            "ending_optimizer_step": steps_per_epoch * human_epoch,
            "mean_clean_loss": 1.0 / human_epoch,
            "mean_degraded_loss": 2.0 / human_epoch,
            "mean_combined_loss": 1.5 / human_epoch,
            "last_gradient_l2": 0.25,
        }
        for human_epoch in range(1, epoch + 1)
    ]
    metrics_path = run_dir / "train_metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    checkpoint = {
        "schema_version": 1,
        "architecture": "MSHNet_NSFPN",
        "method_name": "CR-SITTA",
        "method_stage": "D0-A",
        "development_only": True,
        "test_selected": False,
        "selection_rule": "fixed_final_epoch_train_only",
        "epoch": epoch,
        "global_optimizer_step": steps_per_epoch * epoch,
        "state_dict": {"weight": torch.tensor([1.25])},
        "optimizer": {"state": {0: {"sum": torch.tensor([2.0])}}, "param_groups": []},
        "latest_train_metrics": rows[-1],
        "split_manifest": split_manifest,
        "run_config": run_config,
        "rng_state": {"torch_cpu": torch.get_rng_state()},
    }
    source_path = run_dir / recovery.SOURCE_NAME
    _save_checkpoint(source_path, checkpoint)
    return {
        "repository": repository,
        "run_dir": run_dir,
        "freeze_path": freeze_path,
        "contract_path": contract_path,
        "metrics_path": metrics_path,
        "source_path": source_path,
        "checkpoint": checkpoint,
        "rows": rows,
    }


def _recovery_kwargs(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_source_sha256": recovery.sha256_file(fixture["source_path"]),
        "expected_run_contract_sha256": recovery.sha256_file(
            fixture["contract_path"]
        ),
        "expected_full_train_freeze_sha256": recovery.sha256_file(
            fixture["freeze_path"]
        ),
        "original_process_stopped": True,
        "trusted_local_source": True,
        "full_train_freeze": fixture["freeze_path"],
        "repository": fixture["repository"],
        "resume_validator": _toy_resume_validator,
    }


def _recover(fixture: dict[str, Any]) -> dict[str, Any]:
    return recovery.recover_interrupted_run(
        fixture["run_dir"],
        **_recovery_kwargs(fixture),
    )


def _toy_resume_validator(
    checkpoint: dict[str, Any], run_config: dict[str, Any]
) -> None:
    assert checkpoint["state_dict"].keys() == {"weight"}
    assert checkpoint["optimizer"]["state"]
    assert run_config["method_stage"] == "D0-A"


def test_interrupted_run_publishes_only_rng_sanitized_noreplace_artifacts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    source_sha_before = recovery.sha256_file(fixture["source_path"])

    receipt = _recover(fixture)

    resume_path = fixture["run_dir"] / "recovery" / "resume_epoch_3_rng_sanitized.pth.tar"
    receipt_path = fixture["run_dir"] / "recovery" / recovery.RECOVERY_RECEIPT_NAME
    assert receipt["status"] == "resume_checkpoint_rng_sanitized"
    assert receipt["resume_semantics"]["cuda_continuation_bit_exact_guaranteed"] is False
    assert receipt["resume_semantics"]["frozen_v2_model_optimizer_load_validated"] is False
    assert receipt["publication"]["artifact_pair_published_as_one_directory_rename"] is True
    assert receipt["sanitization"]["torch_load_weights_only_verified"] is True
    assert "custom CUDA" in receipt["resume_semantics"]["cuda_disclosure"]
    assert recovery.sha256_file(fixture["source_path"]) == source_sha_before
    assert receipt_path.is_file()
    assert sorted(path.name for path in receipt_path.parent.iterdir()) == [
        recovery.RECOVERY_RECEIPT_NAME,
        "resume_epoch_3_rng_sanitized.pth.tar",
    ]
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt

    sanitized = torch.load(resume_path, map_location="cpu", weights_only=False)
    weights_only_sanitized = torch.load(
        resume_path, map_location="cpu", weights_only=True
    )
    assert set(sanitized) == set(fixture["checkpoint"]) - {"rng_state"}
    assert "rng_state" not in sanitized
    assert set(weights_only_sanitized) == set(sanitized)
    assert sanitized["run_config"] == fixture["checkpoint"]["run_config"]
    assert sanitized["split_manifest"] == fixture["checkpoint"]["split_manifest"]
    assert sanitized["global_optimizer_step"] == 6
    assert torch.equal(sanitized["state_dict"]["weight"], torch.tensor([1.25]))
    assert receipt["output_checkpoint"]["sha256"] == recovery.sha256_file(resume_path)
    assert receipt["frozen_bindings"]["run_contract"]["sha256"] == recovery.sha256_file(
        fixture["contract_path"]
    )
    assert receipt["frozen_bindings"]["FULL_TRAIN_FREEZE"]["sha256"] == recovery.sha256_file(
        fixture["freeze_path"]
    )

    with pytest.raises(recovery.RecoveryError, match="refusing to replace") as error:
        _recover(fixture)
    assert error.value.code == "D0A_NO_REPLACE"


def test_sanitized_payload_short_circuits_original_v2_rng_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    _recover(fixture)
    resume_path = fixture["run_dir"] / "recovery" / "resume_epoch_3_rng_sanitized.pth.tar"
    sanitized = torch.load(resume_path, map_location="cpu", weights_only=False)
    assert "rng_state" not in sanitized

    import train_cr_sitta_d0a as original_v2

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("RNG restoration must not be called")

    monkeypatch.setattr(original_v2.random, "setstate", forbidden)
    monkeypatch.setattr(original_v2.np.random, "set_state", forbidden)
    monkeypatch.setattr(original_v2.torch, "set_rng_state", forbidden)
    monkeypatch.setattr(original_v2.torch.cuda, "set_rng_state_all", forbidden)
    original_v2._restore_rng_state(sanitized)


def test_all_recovery_checkpoint_loads_use_cpu_map_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    original_load = recovery.torch.load
    map_locations: list[Any] = []

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        map_locations.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(recovery.torch, "load", recording_load)
    _recover(fixture)
    assert map_locations
    assert set(map_locations) == {"cpu"}


def test_recovery_requires_explicit_confirmation_original_process_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    arguments = _recovery_kwargs(fixture)
    arguments["original_process_stopped"] = False
    monkeypatch.setattr(
        recovery.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("active-run refusal must precede pickle load"),
    )
    with pytest.raises(recovery.RecoveryError) as error:
        recovery.recover_interrupted_run(fixture["run_dir"], **arguments)
    assert error.value.code == "D0A_ACTIVE_RUN_NOT_EXCLUDED"
    assert not (fixture["run_dir"] / "recovery").exists()


def test_recovery_requires_trusted_local_pickle_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    arguments = _recovery_kwargs(fixture)
    arguments["trusted_local_source"] = False
    monkeypatch.setattr(
        recovery.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("untrusted source must not be unpickled"),
    )
    with pytest.raises(recovery.RecoveryError) as error:
        recovery.recover_interrupted_run(fixture["run_dir"], **arguments)
    assert error.value.code == "D0A_UNTRUSTED_PICKLE"
    assert not (fixture["run_dir"] / "recovery").exists()


@pytest.mark.parametrize(
    "anchor",
    [
        "expected_source_sha256",
        "expected_run_contract_sha256",
        "expected_full_train_freeze_sha256",
    ],
)
def test_external_hash_anchor_mismatch_precedes_pickle_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    anchor: str,
) -> None:
    fixture = _fixture(tmp_path)
    arguments = _recovery_kwargs(fixture)
    arguments[anchor] = "0" * 64
    monkeypatch.setattr(
        recovery.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("unanchored source must not be unpickled"),
    )
    with pytest.raises(recovery.RecoveryError) as error:
        recovery.recover_interrupted_run(fixture["run_dir"], **arguments)
    assert error.value.code == "D0A_EXTERNAL_HASH_MISMATCH"
    assert not (fixture["run_dir"] / "recovery").exists()


def test_directory_publication_failure_leaves_no_partial_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def fail_publish(*args: Any, **kwargs: Any) -> None:
        raise recovery.SecureIOError("injected rename failure")

    monkeypatch.setattr(recovery, "publish_directory_noreplace", fail_publish)
    with pytest.raises(recovery.RecoveryError) as error:
        _recover(fixture)
    assert error.value.code == "D0A_ATOMIC_PUBLICATION_FAILED"
    assert not (fixture["run_dir"] / "recovery").exists()
    assert not list(fixture["run_dir"].glob(".recovery-staging-*"))


def test_resume_validator_failure_publishes_nothing(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    arguments = _recovery_kwargs(fixture)

    def reject(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected v2 load failure")

    arguments["resume_validator"] = reject
    with pytest.raises(RuntimeError, match="injected v2 load failure"):
        recovery.recover_interrupted_run(fixture["run_dir"], **arguments)
    assert not (fixture["run_dir"] / "recovery").exists()
    assert not list(fixture["run_dir"].glob(".recovery-staging-*"))


@pytest.mark.parametrize("metrics_fault", ["behind", "ahead", "partial"])
def test_metrics_checkpoint_mismatch_fails_closed_without_truncation(
    tmp_path: Path, metrics_fault: str
) -> None:
    fixture = _fixture(tmp_path)
    metrics_path = fixture["metrics_path"]
    if metrics_fault == "behind":
        rows = fixture["rows"][:-1]
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    elif metrics_fault == "ahead":
        extra = dict(fixture["rows"][-1])
        extra["epoch"] = 4
        extra["starting_optimizer_step"] = 6
        extra["ending_optimizer_step"] = 8
        metrics_path.write_text(
            metrics_path.read_text(encoding="utf-8") + json.dumps(extra) + "\n",
            encoding="utf-8",
        )
    else:
        metrics_path.write_bytes(metrics_path.read_bytes() + b'{"epoch":4')

    with pytest.raises(recovery.RecoveryError):
        _recover(fixture)
    assert not (fixture["run_dir"] / "recovery").exists()


def test_runtime_or_run_contract_hash_drift_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["repository"] / "model" / "NS_FPN.py").write_text(
        "# drifted runtime\n", encoding="utf-8"
    )
    with pytest.raises(recovery.RecoveryError) as error:
        _recover(fixture)
    assert error.value.code == "D0A_RUNTIME_HASH_DRIFT"
    assert not (fixture["run_dir"] / "recovery").exists()


def test_wrong_method_or_full_train_mode_is_refused(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    source = torch.load(fixture["source_path"], map_location="cpu", weights_only=False)
    source["method_stage"] = "D0-B"
    _save_checkpoint(fixture["source_path"], source)
    with pytest.raises(recovery.RecoveryError) as error:
        _recover(fixture)
    assert error.value.code == "D0A_CONTRACT_MISMATCH"


def test_epoch_1000_missing_final_emits_only_independent_refusal(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, epoch=1000)

    outcome = _recover(fixture)

    recovery_dir = fixture["run_dir"] / "recovery"
    refusal_path = recovery_dir / recovery.FINAL_RECOVERY_RECEIPT_NAME
    assert outcome["status"] == "resume_refused_final_checkpoint_recovery_required"
    assert outcome["refusal_code"] == "D0A_EPOCH_1000_FINAL_EXPORT_REQUIRED"
    assert outcome["process_exit_status"] == recovery.FINAL_EXPORT_REQUIRED_EXIT_STATUS
    assert refusal_path.is_file()
    assert [path.name for path in recovery_dir.iterdir()] == [
        recovery.FINAL_RECOVERY_RECEIPT_NAME
    ]
    assert not (recovery_dir / recovery.RECOVERY_RECEIPT_NAME).exists()
    assert list(recovery_dir.glob("resume_epoch_*_rng_sanitized.pth.tar")) == []
    assert outcome["output_checkpoint"] is None


def test_epoch_1000_cli_returns_dedicated_refusal_status(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, epoch=1000)
    exit_status = recovery.main(
        [
            "--run-dir",
            str(fixture["run_dir"]),
            "--expected-source-sha256",
            recovery.sha256_file(fixture["source_path"]),
            "--expected-run-contract-sha256",
            recovery.sha256_file(fixture["contract_path"]),
            "--expected-full-train-freeze-sha256",
            recovery.sha256_file(fixture["freeze_path"]),
            "--confirm-original-process-stopped",
            "--trust-local-source",
            "--full-train-freeze",
            str(fixture["freeze_path"]),
            "--repository",
            str(fixture["repository"]),
        ]
    )
    assert exit_status == recovery.FINAL_EXPORT_REQUIRED_EXIT_STATUS
