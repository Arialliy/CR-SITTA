from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import run_p3_stage_b4_full_pilot64_v1 as runner


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p3_stage_b4_full_pilot64_proposal_gate_v1.yaml"


def _schema_contract() -> runner.FullPilotContract:
    payload = CONFIG.read_bytes()
    config_sha256 = hashlib.sha256(payload).hexdigest()
    assert config_sha256 == runner.FROZEN_CONFIG_SHA256
    raw = yaml.safe_load(payload.decode("utf-8"))
    assert isinstance(raw, dict)
    return runner.FullPilotContract(ROOT, CONFIG, config_sha256, raw)


@dataclass(frozen=True)
class _Nested:
    value: float


@dataclass(frozen=True)
class _Result:
    finite: float
    nonfinite: float
    nested: tuple[_Nested, ...]


def test_nonfinite_proposal_diagnostics_are_json_safe_and_reason_preserving() -> None:
    value = runner._serialise_step_result(
        _Result(finite=1.0, nonfinite=float("nan"), nested=(_Nested(float("inf")),))
    )
    assert value == {
        "finite": 1.0,
        "nonfinite": None,
        "nested": [{"value": None}],
    }


def _engineering_candidate_manifest(contract, *, formal: bool = False):
    extension_fixture = Path(runner.__file__).resolve()
    return {
        **runner._runtime_manifest_base(
            contract,
            artifact_type="cr_sitta_p3_stage_b4_candidate_dataset",
            phase="candidate",
            dataset="IRSTD-1K",
            formal=formal,
        ),
        "method_boundary": {
            "method_label_accesses": 0,
            "outer_target_loader_calls": 0,
            "validation_payload_opens": 0,
            "test_payload_opens": 0,
        },
        "candidate_ids": list(runner.CANDIDATES),
        "teacher_probability_role": "sealed_source_identity",
        "b2_candidate_selected": False,
        "source_state_restored": True,
        "source_teacher_post_output_hashes_recorded": True,
        "rejected_no_update_bit_exact_source_enforced": True,
        "runtime_environment": {
            "python": "3.test",
            "numpy": "test",
            "torch": "test",
            "cuda_runtime": None,
            "cudnn": None,
            "device": "cpu",
            "gpu_name": None,
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "pythonhashseed": "42",
            "cublas_workspace_config": ":4096:8",
            "cuda_device_order": "PCI_BUS_ID",
            "cuda_visible_devices": None,
            "sfs_extension_file": str(extension_fixture),
            "sfs_extension_sha256": runner.sha256_file(extension_fixture),
        },
        "conditions": ["clean_S0"],
        "condition_count": 1,
        "image_count_per_condition": 1,
        "episode_count": 4,
        "code_sha256": {},
    }


def test_atomic_publish_runs_guard_and_ledger_detects_tampering(tmp_path: Path) -> None:
    real = _schema_contract()
    contract = runner.FullPilotContract(
        repository=real.repository,
        config_path=real.config_path,
        config_sha256=real.config_sha256,
        raw=real.raw,
    )
    destination = tmp_path / "published"
    staging = runner._new_staging(destination)
    (staging / "payload.txt").write_text("sealed\n", encoding="utf-8")
    calls = []
    runner._publish(
        staging,
        destination,
        _engineering_candidate_manifest(contract),
        pre_rename_guard=lambda: calls.append("guard"),
    )
    assert calls == ["guard"]
    runner.verify_artifact(
        destination,
        contract=contract,
        phase="candidate",
        dataset="IRSTD-1K",
        expected_formal=False,
    )
    (destination / "payload.txt").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(runner.StageB4ProtocolError, match="ledger differs"):
        runner.verify_artifact(
            destination,
            contract=contract,
            phase="candidate",
            dataset="IRSTD-1K",
            expected_formal=False,
        )


def test_outer_refuses_nonformal_candidate_before_target_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def forbidden_target_access(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("target loader must not be reached")

    monkeypatch.setattr(runner, "_capture_code_hashes", lambda _contract: {})
    monkeypatch.setattr(runner.b3, "_load_outer_targets", forbidden_target_access)
    with pytest.raises(runner.StageB4ProtocolError, match="refuses engineering"):
        runner._execute_outer_payload(
            tmp_path,
            contract=SimpleNamespace(),
            dataset="IRSTD-1K",
            device_name="cpu",
            candidate_root=tmp_path,
            candidate_manifest={"formal": False},
        )
    assert called is False


def test_candidate_execution_has_no_outer_target_loader_call() -> None:
    candidate_source = "\n".join(
        (
            inspect.getsource(runner.run_candidate),
            inspect.getsource(runner._execute_candidate_payload),
            inspect.getsource(runner._candidate_episode),
        )
    )
    assert "_load_outer_targets" not in candidate_source


def test_rejected_proposal_requires_bit_exact_source_endpoint() -> None:
    source = torch.tensor([0.25, 0.75], dtype=torch.float32)
    assert runner._check_no_update_endpoint(
        source_probability=source,
        post_probability=source.clone(),
        accepted_update=False,
    ) is True
    changed = source.clone()
    changed[0] = torch.nextafter(changed[0], torch.tensor(float("inf")))
    with pytest.raises(runner.StageB4ProtocolError, match="differs from Source"):
        runner._check_no_update_endpoint(
            source_probability=source,
            post_probability=changed,
            accepted_update=False,
        )


def test_candidate_episode_restores_source_when_postprocessing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PostprocessingFailure(RuntimeError):
        pass

    class StateManager:
        def __init__(self) -> None:
            self.adapted = False
            self.reset_count = 0
            self.assert_count = 0

        def reset_to_source(self) -> None:
            self.adapted = False
            self.reset_count += 1

        def assert_source_state(self) -> None:
            self.assert_count += 1
            assert self.adapted is False

    state = StateManager()

    def fail_after_acceptance(**_kwargs):
        state.adapted = True
        raise PostprocessingFailure("post forward failed")

    monkeypatch.setattr(runner, "_candidate_episode_from_source", fail_after_acceptance)
    with pytest.raises(PostprocessingFailure, match="post forward failed"):
        runner._candidate_episode(
            contract=None,
            model=None,
            adapter=None,
            film=None,
            state_manager=state,
            image=None,
            student_image=None,
            teacher=None,
            uncertainty=None,
            source_logits=None,
            candidate_id="O3_P2",
        )
    assert state.adapted is False
    assert state.reset_count == 2
    assert state.assert_count == 2


def _runtime_environment() -> dict[str, object]:
    return {
        "python": "3.test",
        "numpy": "test",
        "torch": "test",
        "cuda_runtime": "test",
        "cudnn": 1,
        "device": "cuda:0",
        "gpu_name": "test-gpu",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "pythonhashseed": "42",
        "cublas_workspace_config": ":4096:8",
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_visible_devices": "2",
        "sfs_extension_file": "/tmp/test-extension.so",
        "sfs_extension_sha256": "0" * 64,
    }


def test_candidate_outer_runtime_environment_is_exactly_bound() -> None:
    candidate = _runtime_environment()
    runner._assert_runtime_environment_match(candidate, dict(candidate))
    changed = dict(candidate)
    changed["sfs_extension_sha256"] = "1" * 64
    with pytest.raises(runner.StageB4ProtocolError, match="environments differ"):
        runner._assert_runtime_environment_match(candidate, changed)
    changed = dict(candidate)
    changed["cuda_visible_devices"] = "3"
    with pytest.raises(runner.StageB4ProtocolError, match="environments differ"):
        runner._assert_runtime_environment_match(candidate, changed)


def test_runtime_environment_rehashes_dynamic_extension(tmp_path: Path) -> None:
    extension = tmp_path / "MultiScaleDeformableAttention.so"
    extension.write_bytes(b"frozen-extension")
    environment = _runtime_environment()
    environment["sfs_extension_file"] = str(extension)
    environment["sfs_extension_sha256"] = runner.sha256_file(extension)
    runner._validated_runtime_environment(
        environment, "runtime", verify_extension_file=True
    )
    extension.write_bytes(b"changed-extension")
    with pytest.raises(runner.StageB4ProtocolError, match="file hash differs"):
        runner._validated_runtime_environment(
            environment, "runtime", verify_extension_file=True
        )


def _all_background_endpoint(image_count: int) -> dict[str, object]:
    pixels = image_count * 256 * 256
    return {
        "iou": 1.0,
        "normalized_iou": 1.0,
        "pd": 0.0,
        "fa_per_million": 0.0,
        "foreground_fraction": 0.0,
        "intersection_pixels": 0,
        "false_positive_pixels": 0,
        "false_negative_pixels": 0,
        "true_negative_pixels": pixels,
        "predicted_positive_pixels": 0,
        "target_positive_pixels": 0,
        "detected_targets": 0,
        "total_targets": 0,
        "false_alarm_pixels": 0,
        "total_image_pixels": pixels,
        "image_count": image_count,
    }


def _outer_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cells: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    for corruption, severity in runner.CONDITIONS:
        condition = runner._condition_key(corruption, severity)
        for candidate_id in runner.CANDIDATES:
            endpoint = _all_background_endpoint(64)
            cells.append(
                {
                    "candidate_id": candidate_id,
                    "dataset": "IRSTD-1K",
                    "condition": condition,
                    "corruption_family": corruption,
                    "severity": severity,
                    "episode_count": 64,
                    "source_counts": dict(endpoint),
                    "adapted_counts": dict(endpoint),
                    "accepted_episode_count": 0,
                    "no_update_episode_count": 64,
                    "gain_attribution": {
                        "accepted": {
                            "episode_count": 0,
                            "source": None,
                            "adapted": None,
                        },
                        "no_update": {
                            "episode_count": 64,
                            "source": dict(endpoint),
                            "adapted": dict(endpoint),
                        },
                    },
                }
            )
            for episode_index in range(64):
                episodes.append(
                    {
                        "candidate_id": candidate_id,
                        "dataset": "IRSTD-1K",
                        "condition": condition,
                        "episode_index": episode_index,
                        "proxy_gradient_nonzero": True,
                        "task_gradient_nonzero": True,
                        "gradient_cosine": 0.0,
                        "accepted_update": False,
                        "finite": True,
                        "maximum_absolute_logit_delta": 0.0,
                        "proposal_loss_before": 1.0,
                        "proposal_loss_after": 1.0,
                        "threshold_crossing_count": 0,
                    }
                )
    return cells, episodes


def test_outer_record_verifier_rejects_gain_attribution_tampering(
    tmp_path: Path,
) -> None:
    cells, episodes = _outer_records()
    valid = tmp_path / "valid"
    valid.mkdir()
    runner._write_jsonl(valid / "cell_summaries.jsonl", cells)
    runner._write_jsonl(valid / "episode_summaries.jsonl", episodes)
    runner._load_and_validate_outer_records(valid, dataset="IRSTD-1K")

    cells[0]["gain_attribution"]["no_update"]["episode_count"] = 63
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    runner._write_jsonl(tampered / "cell_summaries.jsonl", cells)
    runner._write_jsonl(tampered / "episode_summaries.jsonl", episodes)
    with pytest.raises(runner.StageB4ProtocolError, match="count differs"):
        runner._load_and_validate_outer_records(tampered, dataset="IRSTD-1K")
    assert "_load_and_validate_outer_records" in inspect.getsource(
        runner.verify_artifact
    )


def _candidate_diagnostic_records(image_ids: list[str]) -> list[dict[str, object]]:
    digest = "a" * 64
    records: list[dict[str, object]] = []
    for corruption, severity in runner.CONDITIONS:
        condition = runner._condition_key(corruption, severity)
        for image_index, image_id in enumerate(image_ids):
            for candidate_id in runner.CANDIDATES:
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "teacher_candidate_id": "sealed_source_identity",
                        "objective": candidate_id.split("_", 1)[0],
                        "parameter_space": candidate_id.split("_", 1)[1],
                        "dataset": "IRSTD-1K",
                        "condition": condition,
                        "corruption_family": corruption,
                        "severity": severity,
                        "image_index": image_index,
                        "image_id": image_id,
                        "seed": 42,
                        "accepted_update": False,
                        "no_update": True,
                        "no_update_bit_exact_source": True,
                        "source_output_sha256": digest,
                        "teacher_probability_sha256": digest,
                        "post_output_sha256": digest,
                        "source_state_sha256": digest,
                        "post_reset_state_sha256": digest,
                        "input_tensor_sha256": digest,
                        "teacher_uncertainty_sha256": digest,
                        "method_label_accesses": 0,
                        "validation_payload_opens": 0,
                        "test_payload_opens": 0,
                    }
                )
    return records


def test_candidate_diagnostic_proofs_and_cartesian_keyset_are_pre_target(
    tmp_path: Path,
) -> None:
    image_ids = [f"image-{index:02d}" for index in range(64)]
    records = _candidate_diagnostic_records(image_ids)
    valid = tmp_path / "valid-candidate"
    valid.mkdir()
    runner._write_jsonl(valid / "episode_diagnostics.jsonl", records)
    result = runner._candidate_diagnostics_by_key(
        valid,
        dataset="IRSTD-1K",
        image_ids=image_ids,
        expected_source_state_sha256="a" * 64,
        seed=42,
    )
    assert len(result) == 13 * 64 * 4

    incomplete = tmp_path / "incomplete-candidate"
    incomplete.mkdir()
    runner._write_jsonl(incomplete / "episode_diagnostics.jsonl", records[:-1])
    with pytest.raises(runner.StageB4ProtocolError, match="Cartesian keyset"):
        runner._candidate_diagnostics_by_key(
            incomplete,
            dataset="IRSTD-1K",
            image_ids=image_ids,
            expected_source_state_sha256="a" * 64,
            seed=42,
        )
    source = inspect.getsource(runner._execute_outer_payload)
    assert source.index("_candidate_diagnostics_by_key") < source.index(
        "_load_outer_targets"
    )
    assert source.index("_verify_candidate_output_hash_proofs") < source.index(
        "_load_outer_targets"
    )
    assert source.index("_assert_runtime_environment_match") < source.index(
        "_load_outer_targets"
    )
    assert "_verify_formal_candidate_payload" in inspect.getsource(
        runner.verify_artifact
    )


def test_aggregate_parent_paths_and_field_set_are_exact(tmp_path: Path) -> None:
    repository = tmp_path
    candidate = repository / "candidate" / "IRSTD-1K"
    outer = repository / "outer" / "IRSTD-1K"
    parent = {field: "a" * 64 for field in runner._AGGREGATE_PARENT_FIELDS}
    parent["candidate_path"] = "candidate/IRSTD-1K"
    parent["outer_path"] = "outer/IRSTD-1K"
    runner._assert_aggregate_parent_paths(
        parent,
        repository=repository,
        candidate_root=candidate,
        outer_root=outer,
        dataset="IRSTD-1K",
    )
    changed = dict(parent)
    changed["outer_path"] = "outer/NUDT-SIRST"
    with pytest.raises(runner.StageB4ProtocolError, match="parent path differs"):
        runner._assert_aggregate_parent_paths(
            changed,
            repository=repository,
            candidate_root=candidate,
            outer_root=outer,
            dataset="IRSTD-1K",
        )
    changed = dict(parent)
    changed["unexpected"] = "field"
    with pytest.raises(runner.StageB4ProtocolError, match="field set differs"):
        runner._assert_aggregate_parent_paths(
            changed,
            repository=repository,
            candidate_root=candidate,
            outer_root=outer,
            dataset="IRSTD-1K",
        )


def test_smoke_publishes_only_below_engineering_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = SimpleNamespace(
        output_root=tmp_path / "results_root",
        raw={"output": {"engineering_phase": "engineering_dry_runs"}},
    )
    observed: dict[str, Path] = {}

    def fake_execute(staging, **_kwargs):
        return {"code_sha256": {}}

    def fake_publish(staging, destination, _manifest, *, pre_rename_guard):
        pre_rename_guard()
        observed["destination"] = destination
        shutil.rmtree(staging)

    monkeypatch.setattr(runner, "_execute_candidate_payload", fake_execute)
    monkeypatch.setattr(runner, "_publish", fake_publish)
    monkeypatch.setattr(runner, "verify_artifact", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "_assert_contract_unchanged", lambda _contract: None)
    monkeypatch.setattr(runner, "_capture_code_hashes", lambda _contract: {})
    result = runner.run_candidate(
        contract,
        dataset="IRSTD-1K",
        device_name="cpu",
        max_images=1,
        condition="clean_S0",
    )
    engineering = contract.output_root / "engineering_dry_runs"
    formal = contract.output_root / "candidate_phase" / "R0" / "IRSTD-1K"
    assert observed["destination"].is_relative_to(engineering)
    assert observed["destination"] != formal
    assert result["formal"] is False


@pytest.mark.parametrize("maximum", [0, 65])
def test_smoke_image_limit_fails_before_any_runtime(
    maximum: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _schema_contract()
    monkeypatch.setattr(
        runner,
        "_execute_candidate_payload",
        lambda *_args, **_kwargs: pytest.fail("runtime must not be reached"),
    )
    with pytest.raises(runner.StageB4ProtocolError, match=r"\[1,64\]"):
        runner.run_candidate(
            contract,
            dataset="IRSTD-1K",
            device_name="cpu",
            max_images=maximum,
            condition="clean_S0",
        )


def test_cli_contains_all_protocol_phases() -> None:
    parser = runner._build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {
        "validate",
        "smoke",
        "candidate",
        "outer",
        "aggregate",
        "verify",
    }
