from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import run_binary_tent_cross_process_repro as repro
import test_source as source_runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "binary_tent_cross_process_repro_candidate_v1.yaml"


def _metrics(offset: float = 0.0) -> dict[str, object]:
    return {
        prediction: {
            evaluator: {
                "iou": 0.50 + offset,
                "pd": 0.75 + offset,
                "fa_per_million_pixels": 10.0 + offset,
            }
            for evaluator in repro.EVALUATOR_NAMES
        }
        for prediction in repro.PREDICTION_NAMES
    }


def _write_run(
    root: Path,
    *,
    run_id: str,
    image_ids: tuple[str, ...],
    tent_post_delta: float,
    step_delta: float,
    worker: bool,
) -> repro.RunDescriptor:
    root.mkdir(parents=True)
    for protocol in ("batch_stats", "source_stats"):
        protocol_root = root / protocol
        records: list[dict[str, object]] = []
        diagnostics: list[dict[str, object]] = []
        for index, image_id in enumerate(image_ids):
            prediction_records: dict[str, object] = {}
            for prediction_name in repro.PREDICTION_NAMES:
                probability = np.full((256, 256), 0.25, dtype=np.float32)
                if prediction_name == "tent_post":
                    probability[0, 0] = np.float32(0.25 + tent_post_delta)
                mask = np.where(probability > 0.5, 255, 0).astype(np.uint8)
                probability_relative = (
                    Path("predictions") / prediction_name / f"{image_id}.npy"
                )
                mask_relative = Path("masks") / prediction_name / f"{image_id}.png"
                probability_path = protocol_root / probability_relative
                mask_path = protocol_root / mask_relative
                probability_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(probability_path, probability, allow_pickle=False)
                Image.fromarray(mask, mode="L").save(mask_path)
                prediction_records[prediction_name] = {
                    "probability_map": str(probability_relative),
                    "prediction_mask": str(mask_relative),
                }
            records.append(
                {
                    "index": index,
                    "image_id": image_id,
                    "optimizer_step_norm": 0.1 + step_delta,
                    "predictions": prediction_records,
                }
            )
            fingerprint = {"full_sha256": f"state-{image_id}"}
            diagnostics.append(
                {
                    "index": index,
                    "image_id": image_id,
                    "source_fingerprint": fingerprint,
                    "reset_fingerprint": fingerprint,
                    "method_diagnostics": {
                        "episode_invariants": {"reset_exact_source": True}
                    },
                }
            )
        source_runner.write_jsonl_atomic(protocol_root / "per_image.jsonl", records)
        source_runner.write_jsonl_atomic(
            protocol_root / "adaptation_diagnostics.jsonl", diagnostics
        )
        source_runner.write_json_atomic(
            protocol_root / "metrics.json",
            {"canonical_metrics_outer_train_diagnostic_only": _metrics(tent_post_delta)},
        )
    provenance: dict[str, object] = {
        "fixed_test_boundary": {"test_images_opened": 0, "test_masks_opened": 0}
    }
    if worker:
        provenance["process"] = {
            "pid": 100 + int(run_id[-1]),
            "process_uuid": f"uuid-{run_id}",
        }
    source_runner.write_json_atomic(root / "provenance.json", provenance)
    if not worker:
        source_runner.write_json_atomic(
            root / "artifact_manifest.json", {"synthetic_reference": True}
        )
    return repro.load_run_descriptor(
        run_id=run_id,
        root=root,
        origin="fresh_worker" if worker else "published_historical_reference",
        selected_ids=image_ids,
    )


def test_config_requires_two_fresh_workers_three_runs_and_candidate_only() -> None:
    path, config = repro.load_audit_config(CONFIG)
    assert path == CONFIG
    assert config["execution"]["worker_count"] == 2
    assert config["execution"]["episodes_per_worker"] == 64
    assert config["execution"]["minimum_independent_processes"] == 3
    assert config["execution"]["expected_pairwise_comparisons"] == 3
    assert config["comparison"]["all_unordered_run_pairs_required"] is True
    assert config["tolerance_candidate"]["automatic_output"] == "candidate_only"
    assert config["tolerance_candidate"]["scientific_tolerance_frozen"] is False
    assert config["scope"]["use_test_images"] is False
    assert config["scope"]["use_test_labels"] is False


def test_validate_only_verifies_reference_without_output_or_workers(tmp_path: Path) -> None:
    output = tmp_path / "must_not_exist"
    result = repro.run(
        Namespace(
            config=CONFIG,
            device="cuda:0",
            output_dir=output,
            validate_only=True,
            worker_id=None,
            worker_output=None,
            expected_parent_pid=None,
            expected_runtime_seal_sha256=None,
        )
    )
    assert result["published_reference_verified"] is True
    assert result["published_reference_files_verified"] == 393
    assert result["published_runtime_files_verified"] == len(
        repro.source_smoke.PROVENANCE_PATHS
    )
    assert result["published_extension_binary_verified"] is True
    assert result["planned_fresh_workers"] == 2
    assert result["planned_independent_runs_including_reference"] == 3
    assert result["planned_unordered_pairwise_comparisons"] == 3
    assert result["worker_processes_started"] == 0
    assert result["gpu_execution_started"] is False
    assert result["dataset_pixels_opened"] == 0
    assert result["test_images_opened"] == 0
    assert result["test_masks_opened"] == 0
    assert result["tolerance_decision"] == "candidate_only_not_frozen"
    assert not output.exists()


def test_all_three_unordered_pairs_include_worker_to_worker(tmp_path: Path) -> None:
    image_ids = ("a", "b")
    reference = _write_run(
        tmp_path / "reference",
        run_id="published_reference",
        image_ids=image_ids,
        tent_post_delta=0.0,
        step_delta=0.0,
        worker=False,
    )
    worker_1 = _write_run(
        tmp_path / "worker_01",
        run_id="worker_01",
        image_ids=image_ids,
        tent_post_delta=0.01,
        step_delta=0.001,
        worker=True,
    )
    worker_2 = _write_run(
        tmp_path / "worker_02",
        run_id="worker_02",
        image_ids=image_ids,
        tent_post_delta=0.02,
        step_delta=0.003,
        worker=True,
    )
    pairs = repro.all_unordered_pairs((reference, worker_1, worker_2))
    assert [(left.run_id, right.run_id) for left, right in pairs] == [
        ("published_reference", "worker_01"),
        ("published_reference", "worker_02"),
        ("worker_01", "worker_02"),
    ]
    comparisons = [
        repro.compare_run_pair(left, right, selected_ids=image_ids, threshold=0.5)
        for left, right in pairs
    ]
    assert all(repro.comparison_hard_gate(value, len(image_ids)) for value in comparisons)
    worker_pair = comparisons[-1]["protocols"]["batch_stats"]
    assert worker_pair["aggregate"]["source_pre"]["probability_bit_exact_images"] == 2
    assert worker_pair["aggregate"]["tent_pre"]["mask_bit_exact_images"] == 2
    assert worker_pair["aggregate"]["tent_post"][
        "probability_global_max_abs_difference"
    ] == pytest.approx(0.01, abs=1e-7)
    assert worker_pair["aggregate"]["optimizer_step_norm"][
        "global_max_absolute_difference"
    ] == pytest.approx(0.002)
    assert worker_pair["metric_absolute_differences"]["tent_post"][
        "official_nsfpn_operating_point"
    ]["iou"] == pytest.approx(0.01)


def test_candidate_uses_every_pair_but_explicitly_refuses_freeze() -> None:
    def comparison(probability: float, step: float, metric: float) -> dict[str, object]:
        protocol = {
            "aggregate": {
                "tent_post": {
                    "probability_global_mean_abs_difference": probability / 10,
                    "probability_global_max_abs_difference": probability,
                    "mask_disagreement_pixels": 1,
                    "mask_disagreement_rate": 1 / 65536,
                },
                "optimizer_step_norm": {"global_max_absolute_difference": step},
            },
            "metric_absolute_differences": {
                "tent_post": {
                    evaluator: {name: metric for name in repro.METRIC_NAMES}
                    for evaluator in repro.EVALUATOR_NAMES
                }
            },
        }
        return {"protocols": {name: protocol for name in ("batch_stats", "source_stats")}}

    comparisons = [
        comparison(0.01, 0.001, 0.02),
        comparison(0.03, 0.004, 0.01),
        comparison(0.02, 0.002, 0.04),
    ]
    candidate = repro.derive_candidate(comparisons, safety_multiplier=1.5)
    assert candidate["pairwise_comparisons_used"] == 3
    assert candidate["global_observed_max"]["probability_max_abs"] == pytest.approx(0.03)
    assert candidate["global_heuristic_candidate_envelope"][
        "probability_max_abs"
    ] == pytest.approx(0.045)
    assert candidate["global_heuristic_candidate_envelope"][
        "mask_disagreement_pixels"
    ] == 2
    assert candidate["scientific_tolerance_frozen"] is False
    assert candidate["formal_acceptance_gate"] is False
    assert candidate["used_to_accept_or_reject_tent_post_in_this_audit"] is False
    assert candidate["freeze_decision"]["decision"] == "do_not_freeze_automatically"


def test_worker_command_is_a_new_python_process_with_64_episode_worker_mode(
    tmp_path: Path,
) -> None:
    command = repro.build_worker_command(
        config_path=CONFIG,
        device="cuda:0",
        worker_id="worker_01",
        worker_output=tmp_path / "worker_01",
        parent_pid=123,
        expected_runtime_seal_sha256="d" * 64,
        python_executable="/exact/python",
    )
    assert command[0] == "/exact/python"
    assert command[1].endswith("run_binary_tent_cross_process_repro.py")
    assert command[command.index("--worker-id") + 1] == "worker_01"
    assert command[command.index("--expected-parent-pid") + 1] == "123"
    assert command[command.index("--expected-runtime-seal-sha256") + 1] == "d" * 64
    assert "--validate-only" not in command


def test_runtime_seal_detects_a_config_edit(tmp_path: Path) -> None:
    config = tmp_path / "protocol.yaml"
    config.write_text("value: 1\n", encoding="utf-8")
    seal = repro._runtime_seal(config)
    assert len(repro._runtime_seal_sha256(seal)) == 64
    repro._assert_runtime_seal(seal, config, stage="unchanged")

    config.write_text("value: 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after process entry"):
        repro._assert_runtime_seal(seal, config, stage="mutated")


def test_atomic_finalize_is_fail_closed_and_marks_candidate_not_frozen(
    tmp_path: Path,
) -> None:
    staging = tmp_path / ".candidate.build"
    final = tmp_path / "candidate"
    staging.mkdir()
    for relative in ("run_config.yaml", "provenance.json", "aggregate_metrics.json"):
        (staging / relative).write_text("{}\n", encoding="utf-8")
    complete = repro.finalize_artifact(
        staging=staging,
        final_root=final,
        protocol_id=repro.EXPECTED_PROTOCOL_ID,
        protocol_sha256="a" * 64,
        scope="test_candidate",
        checks={"gate": True},
        required_files=("run_config.yaml", "provenance.json", "aggregate_metrics.json"),
    )
    assert complete["complete"] is True
    assert complete["scientific_result_frozen"] is False
    assert complete["tolerance_status"] == "candidate_only_not_frozen"
    assert final.is_dir()
    assert not staging.exists()
    assert source_runner.sha256_file(final / "artifact_manifest.json") == complete[
        "artifact_manifest_sha256"
    ]

    failed_staging = tmp_path / ".failed.build"
    failed_staging.mkdir()
    with pytest.raises(RuntimeError, match="hard gates failed"):
        repro.finalize_artifact(
            staging=failed_staging,
            final_root=tmp_path / "must_not_publish",
            protocol_id=repro.EXPECTED_PROTOCOL_ID,
            protocol_sha256="b" * 64,
            scope="test_candidate",
            checks={"gate": False},
            required_files=(),
        )
    assert not (tmp_path / "must_not_publish").exists()


def test_manifest_verifier_rejects_tampering(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    payload = root / "payload.json"
    payload.write_text("{}\n", encoding="utf-8")
    manifest = {
        "protocol_id": "p",
        "protocol_sha256": "c" * 64,
        "files": {
            "payload.json": {
                "sha256": source_runner.sha256_file(payload),
                "bytes": payload.stat().st_size,
            }
        },
    }
    source_runner.write_json_atomic(root / "artifact_manifest.json", manifest)
    source_runner.write_json_atomic(
        root / "COMPLETE.json",
        {
            "complete": True,
            "protocol_id": "p",
            "protocol_sha256": "c" * 64,
            "artifact_manifest_sha256": source_runner.sha256_file(
                root / "artifact_manifest.json"
            ),
        },
    )
    assert repro.verify_artifact_manifest(
        root, expected_protocol_id="p", expected_protocol_sha256="c" * 64
    )["verified_files"] == 1
    payload.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bytes mismatch|SHA256 mismatch"):
        repro.verify_artifact_manifest(
            root, expected_protocol_id="p", expected_protocol_sha256="c" * 64
        )
