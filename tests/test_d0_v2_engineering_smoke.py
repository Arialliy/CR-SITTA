from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from analysis.d0_v2_independent_candidate_contract import (
    FROZEN_CANDIDATES,
    IndependentCandidateReceipt,
)
from analysis.d0_v2_smoke_repro_contract import (
    FreshSubprocessIdentity,
    aggregate_three,
    build_d0_v2_smoke_process_receipt,
    canonical_d0_v2_smoke_process_receipt_bytes,
    canonical_d0_v2_smoke_repro_aggregate_bytes,
)
import scripts.run_d0_v2_engineering_smoke as smoke_cli
from tta.d0_v2_candidate_worker import (
    D0V2CandidateBuildError,
    build_fresh_d0_v2_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_d0_v2_engineering_smoke.py"
CONFIG = ROOT / "configs/tent_failure_diagnostics_v2_independent_candidates.yaml"
LOCAL_ARTIFACT_TEST_ENV = "NS_FPN_RUN_LOCAL_ARTIFACT_TESTS"
LOCAL_ARTIFACT_TESTS_ENABLED = os.environ.get(LOCAL_ARTIFACT_TEST_ENV) == "1"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate_receipt(process_index: int, candidate_index: int) -> dict:
    candidate = FROZEN_CANDIDATES[candidate_index]
    return IndependentCandidateReceipt(
        candidate_index=candidate_index,
        candidate=candidate,
        config_sha256=_sha("config"),
        dataset="NUAA-SIRST",
        condition="clean_S0",
        sample_index=0,
        sample_id="Misc_421",
        split_sha256=_sha("split"),
        checkpoint_sha256=_sha("checkpoint"),
        source_state_sha256=_sha("source"),
        runtime_sha256=_sha("runtime"),
        determinism_sha256=_sha("determinism"),
        input_sha256=_sha("input"),
        selected_parameter_names_sha256=_sha("selected"),
        pre_logits_sha256=_sha("pre"),
        post_logits_sha256=_sha(
            f"post:{process_index}:{candidate_index}"
        ),
        entropy_gradient_bundle_sha256=_sha(
            f"gradient:{process_index}:{candidate_index}"
        ),
        parameter_delta_bundle_sha256=_sha(
            f"delta:{process_index}:{candidate_index}"
        ),
        model_instance_id=f"replicate-{process_index}:model:{candidate_index}",
        method_instance_id=f"replicate-{process_index}:method:{candidate_index}",
        optimizer_instance_id=(
            f"replicate-{process_index}:optimizer:{candidate_index}"
        ),
        autograd_graph_id=f"replicate-{process_index}:graph:{candidate_index}",
        backward_execution_id=(
            f"replicate-{process_index}:backward:{candidate_index}"
        ),
        gradient_buffer_owner_id=(
            f"replicate-{process_index}:gradient:{candidate_index}"
        ),
        gradient_tensor_count=106,
        changed_parameter_tensor_count=100,
        optimizer_state_entry_count_after_step=106,
        native_reference_parameter_tensor_count=106,
        native_reference_optimizer_state_tensor_count=(
            318 if candidate.optimizer == "Adam" else 106
        ),
        step_norm_l2=(process_index + 1) * (candidate_index + 1) * 1.0e-6,
    ).to_dict()


def _process_receipt(process_index: int) -> dict:
    process_id = smoke_cli.PROCESS_IDS[process_index]
    return build_d0_v2_smoke_process_receipt(
        [_candidate_receipt(process_index, index) for index in range(10)],
        parent_run_nonce=_sha("parent"),
        child_launch_nonce=_sha(f"child:{process_index}"),
        command_sha256=_sha(f"command:{process_index}"),
        process_identity=FreshSubprocessIdentity(
            process_id=process_id,
            os_process_id=5000 + process_index,
            process_start_time_ticks=9000 + process_index,
        ),
    )


def _write_synthetic_publication(directory: Path) -> dict:
    directory.mkdir()
    processes = [_process_receipt(index) for index in range(3)]
    for filename, receipt in zip(
        smoke_cli.PROCESS_FILENAMES, processes, strict=True
    ):
        (directory / filename).write_bytes(
            canonical_d0_v2_smoke_process_receipt_bytes(receipt)
        )
    aggregate = aggregate_three(processes)
    aggregate_bytes = canonical_d0_v2_smoke_repro_aggregate_bytes(aggregate)
    (directory / smoke_cli.AGGREGATE_FILENAME).write_bytes(aggregate_bytes)
    process_hashes = {
        filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        for filename in smoke_cli.PROCESS_FILENAMES
    }
    complete = smoke_cli._complete_manifest(
        config_sha256=_sha("config"),
        process_file_sha256s=process_hashes,
        aggregate_sha256=hashlib.sha256(aggregate_bytes).hexdigest(),
    )
    (directory / smoke_cli.COMPLETE_FILENAME).write_bytes(
        smoke_cli._canonical_json_bytes(complete)
    )
    return aggregate


def test_validate_command_is_cpu_only_and_non_materializing() -> None:
    cuda_initialized_before = torch.cuda.is_initialized()
    destination = (
        ROOT
        / "results/cr_sitta/tent_failure_diagnostics_v2_independent_candidates"
        / "engineering_smoke"
    )
    existed_before = destination.exists() or destination.is_symlink()

    value = smoke_cli.validate_contract_only(CONFIG)

    assert value["valid"] is True
    assert value["sample"] == {
        "dataset": "NUAA-SIRST",
        "condition": "clean_S0",
        "image_index": 0,
        "image_id": "Misc_421",
        "original_size": (225, 334),
        "split_name": "train",
        "split_role": "frozen_pilot64",
    }
    assert value["candidate_count_per_process"] == 10
    assert value["fresh_process_count"] == 3
    assert value["filesystem_created"] is False
    assert value["gpu_initialized"] is False
    assert value["paper_result"] is False
    assert value["formal_p3_complete"] is False
    assert value["stage2_authorized"] is False
    assert (destination.exists() or destination.is_symlink()) is existed_before
    assert torch.cuda.is_initialized() is cuda_initialized_before


def test_public_cli_help_hides_internal_worker_and_discloses_boundaries() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "{validate,run,verify}" in completed.stdout
    assert "_worker" not in completed.stdout
    assert "never a paper result" in completed.stdout
    assert "never Stage-2 authority" in completed.stdout


def test_worker_without_parent_binding_fails_before_cuda_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda_initialized_before = torch.cuda.is_initialized()
    for name in (
        smoke_cli.ENV_PROCESS_ID,
        smoke_cli.ENV_PARENT_NONCE,
        smoke_cli.ENV_CHILD_NONCE,
        smoke_cli.ENV_COMMAND_SHA256,
        smoke_cli.ENV_CONFIG_SHA256,
    ):
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "replicate_0.json"

    code = smoke_cli.main(
        [
            "_worker",
            "--config",
            str(CONFIG),
            "--process-id",
            "replicate_0",
            "--parent-run-nonce",
            _sha("parent"),
            "--child-launch-nonce",
            _sha("child"),
            "--output",
            str(output),
        ]
    )

    assert code == 2
    assert not output.exists()
    assert torch.cuda.is_initialized() is cuda_initialized_before


def test_candidate_builder_rejects_cpu_before_model_or_checkpoint_access() -> None:
    cuda_initialized_before = torch.cuda.is_initialized()
    with pytest.raises(D0V2CandidateBuildError, match="requires a CUDA device"):
        build_fresh_d0_v2_candidate(
            project_root=ROOT,
            dataset_config={},
            candidate=FROZEN_CANDIDATES[0],
            device=torch.device("cpu"),
            entropy_eps=1.0e-6,
            diagnostic_detail="global",
        )
    assert torch.cuda.is_initialized() is cuda_initialized_before


def test_source_state_hash_excludes_only_candidate_optimizer_configuration() -> None:
    common = {
        "model_sha256": _sha("model"),
        "runtime_sha256": _sha("source-runtime"),
        "topology_sha256": _sha("topology"),
        "gradients_sha256": _sha("gradients"),
        "extras_sha256": _sha("extras"),
    }
    adam = SimpleNamespace(
        **common,
        optimizer_sha256=_sha("adam-optimizer"),
        full_sha256=_sha("adam-full"),
    )
    sgd = SimpleNamespace(
        **common,
        optimizer_sha256=_sha("sgd-optimizer"),
        full_sha256=_sha("sgd-full"),
    )
    runtime_drift = SimpleNamespace(
        **{**common, "runtime_sha256": _sha("drifted-runtime")},
        optimizer_sha256=_sha("adam-optimizer"),
        full_sha256=_sha("other-full"),
    )

    adam_hash = smoke_cli._candidate_invariant_source_state_sha256(adam)
    assert adam_hash == smoke_cli._candidate_invariant_source_state_sha256(sgd)
    assert adam_hash != smoke_cli._candidate_invariant_source_state_sha256(
        runtime_drift
    )


def test_zero_update_is_diagnostic_not_a_failed_runner_gate() -> None:
    smoke_cli._validate_runner_result_checks(
        {"invariant": True, "bn_affine_changed_by_one_step": False},
        changed_parameter_tensor_count=0,
        candidate_slug="Adam_lr_1em5",
    )
    smoke_cli._validate_runner_result_checks(
        {"invariant": True, "bn_affine_changed_by_one_step": True},
        changed_parameter_tensor_count=1,
        candidate_slug="Adam_lr_1em5",
    )
    with pytest.raises(
        smoke_cli.D0V2EngineeringSmokeError, match="update-activity"
    ):
        smoke_cli._validate_runner_result_checks(
            {"invariant": True, "bn_affine_changed_by_one_step": False},
            changed_parameter_tensor_count=1,
            candidate_slug="Adam_lr_1em5",
        )


@pytest.mark.skipif(
    not LOCAL_ARTIFACT_TESTS_ENABLED,
    reason=(
        "requires ignored local result artifacts; set "
        f"{LOCAL_ARTIFACT_TEST_ENV}=1 to opt in"
    ),
)
def test_narrow_input_seal_opens_no_target_or_test_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda_initialized_before = torch.cuda.is_initialized()
    contract = smoke_cli._load_contract(CONFIG)
    observed: list[Path] = []
    original = smoke_cli.read_stable_regular_file

    def recording_read(path: object):
        observed.append(Path(path))
        return original(path)

    monkeypatch.setattr(smoke_cli, "read_stable_regular_file", recording_read)
    seal = smoke_cli._capture_method_facing_input_seal(
        contract, config_path=CONFIG
    )

    relative = [path.relative_to(ROOT).as_posix() for path in observed]
    assert seal.complete["method_received_labels"] is False
    assert seal.complete["test_images_opened"] == 0
    assert seal.complete["test_masks_opened"] == 0
    assert any(path.endswith("conditions/clean_S0.npy") for path in relative)
    assert any("train_NUAA-SIRST.txt" in path for path in relative)
    assert all("outer_evaluator/targets.npy" not in path for path in relative)
    assert all("test_NUAA-SIRST.txt" not in path for path in relative)
    assert all("IRSTD-1K" not in path for path in relative)
    assert all("NUDT-SIRST" not in path for path in relative)
    assert torch.cuda.is_initialized() is cuda_initialized_before


def test_worker_command_and_visible_device_contract_are_exact(tmp_path: Path) -> None:
    command = smoke_cli._worker_command(
        config_path=CONFIG,
        process_id="replicate_0",
        parent_run_nonce=_sha("parent"),
        child_launch_nonce=_sha("child"),
        output=tmp_path / "replicate_0.json",
    )
    assert command[0] == sys.executable
    assert command[2] == "_worker"
    assert smoke_cli._command_sha256(command) == smoke_cli._command_sha256(
        tuple(command)
    )
    assert smoke_cli._validate_visible_device("0") == "0"
    assert smoke_cli._validate_visible_device("GPU-abcd:1") == "GPU-abcd:1"
    for unsafe in ("", "0,1", "../../0", "0 1"):
        with pytest.raises(smoke_cli.D0V2EngineeringSmokeError):
            smoke_cli._validate_visible_device(unsafe)


def test_flat_publication_verifier_rebuilds_all_three_receipts(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "engineering_smoke"
    expected = _write_synthetic_publication(directory)

    observed = smoke_cli.verify_engineering_smoke_directory(directory)

    assert observed == expected
    assert observed["fresh_process_count"] == 3
    assert observed["total_candidate_receipt_count"] == 30
    assert observed["paper_result"] is False
    assert observed["formal_p3_complete"] is False
    assert observed["stage2_authorized"] is False


def test_publication_verifier_rejects_unknown_file_and_symlink(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "unknown"
    _write_synthetic_publication(unknown)
    (unknown / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(
        smoke_cli.D0V2EngineeringSmokeError, match="member set"
    ):
        smoke_cli.verify_engineering_smoke_directory(unknown)

    unsafe = tmp_path / "unsafe"
    _write_synthetic_publication(unsafe)
    aggregate_path = unsafe / smoke_cli.AGGREGATE_FILENAME
    aggregate_path.unlink()
    aggregate_path.symlink_to(unsafe / smoke_cli.PROCESS_FILENAMES[0])
    with pytest.raises(
        smoke_cli.D0V2EngineeringSmokeError, match="securely snapshot"
    ):
        smoke_cli.verify_engineering_smoke_directory(unsafe)


def test_publication_verifier_rejects_noncanonical_or_tampered_aggregate(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "engineering_smoke"
    _write_synthetic_publication(directory)
    path = directory / smoke_cli.AGGREGATE_FILENAME
    value = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    with pytest.raises(
        smoke_cli.D0V2EngineeringSmokeError, match="not canonical bytes"
    ):
        smoke_cli.verify_engineering_smoke_directory(directory)


def test_file_publication_is_atomic_no_replace(tmp_path: Path) -> None:
    destination = tmp_path / "receipt.json"
    smoke_cli._publish_bytes_noreplace(destination, b"first\n")
    with pytest.raises(FileExistsError):
        smoke_cli._publish_bytes_noreplace(destination, b"second\n")
    assert destination.read_bytes() == b"first\n"
    assert not tuple(tmp_path.glob(".*.tmp"))
