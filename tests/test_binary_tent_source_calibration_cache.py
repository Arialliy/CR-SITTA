from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dataio.corruption_cache import sha256_file
import materialize_binary_tent_source_calibration_cache as cache


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "binary_tent_source_calibration_cache_v1.yaml"


def _toy_outer_manifest(staging: Path) -> dict[str, object]:
    image_ids = [f"image-{index:03d}" for index in range(64)]
    records: list[dict[str, object]] = []
    files: dict[str, dict[str, object]] = {}
    for index, (corruption, severity) in enumerate(cache.EXPECTED_CONDITIONS):
        key = cache.condition_key(corruption, severity)
        relative = f"conditions/{key}.npy"
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"condition-{index}".encode("utf-8"))
        file_hash = sha256_file(path)
        files[relative] = {"sha256": file_hash, "bytes": path.stat().st_size}
        records.append(
            {
                "index": index,
                "key": key,
                "corruption": corruption,
                "severity": severity,
                "path": relative,
                "shape": [64, 3, 256, 256],
                "dtype": "little_endian_float32",
                "contiguous_order": "C",
                "tensor_sequence_sha256": f"{index + 1:064x}",
                "file_sha256": file_hash,
            }
        )
    return {
        "schema_version": 1,
        "cache_format": "nsfpn-materialized-source-calibration-cache-v1",
        "protocol_sha256": "a" * 64,
        "runtime_seal_sha256": "c" * 64,
        "dataset": "toy",
        "seed": 42,
        "image_ids": image_ids,
        "ordered_ids_sha256": cache.ordered_ids_sha256(image_ids),
        "original_sizes": [[256, 256] for _ in image_ids],
        "condition_count": 13,
        "conditions": records,
        "targets": {"path": cache.TARGET_RELATIVE_PATH},
        "files": files,
        "label_firewall": {
            "official_method_facing_consumer": (
                "SourceCalibrationMethodInputDataset"
            ),
            "outer_evaluator_target_loader": "load_outer_evaluator_targets",
            "contract_evidence": cache._label_firewall_contract_evidence(),
        },
        "cache_content_sha256": "b" * 64,
        "runtime_seconds": 0.0,
    }


def test_versioned_config_freezes_train_derived_cache_contract() -> None:
    path, protocol = cache.load_protocol(CONFIG)
    assert path == CONFIG
    assert protocol["protocol_id"] == (
        "cr-sitta-binary-tent-source-calibration-cache-v1"
    )
    assert protocol["scope"]["independent_validation_set"] is False
    assert protocol["scope"]["use_test_images"] is False
    assert protocol["scope"]["use_test_labels"] is False
    assert protocol["scope"]["adaptation_interface_invoked"] is False
    assert protocol["scope"]["adaptation_receives_labels"] is False
    assert (
        protocol["scope"]["target_transition_hyperparameter_selection_allowed"]
        is False
    )
    assert tuple(
        (str(name), int(severity))
        for name, severity in protocol["input_protocol"]["ordered_conditions"]
    ) == cache.EXPECTED_CONDITIONS
    assert protocol["input_protocol"]["subset_size_per_dataset"] == 64
    assert protocol["input_protocol"]["seed"] == 42
    assert protocol["materialized_cache"]["root"] == (
        "results/binary_tent/source_calibration_cache_v1"
    )
    assert protocol["materialized_cache"]["images"]["shape"] == [64, 3, 256, 256]
    assert protocol["materialized_cache"]["targets"]["shape"] == [64, 1, 256, 256]
    assert (
        protocol["materialized_cache"]["targets"]["method_facing_access"]
        == "forbidden"
    )
    assert protocol["materialized_cache"]["runtime_seal"]["enabled"] is True
    assert protocol["materialized_cache"]["runtime_seal"]["drift_policy"] == (
        "fail_closed_and_do_not_publish"
    )
    assert protocol["materialized_cache"]["official_consumers"][
        "method_facing_manifest"
    ] == "method_input_manifest.json"
    assert protocol["materialized_cache"]["official_consumers"][
        "expected_protocol_sha256_required"
    ] is True
    assert "method_input_manifest_sha256" in protocol["materialized_cache"][
        "required_hashes"
    ]
    assert "test_source.py" in protocol["provenance_chain"]["input_pipeline_files"]
    assert "test_source.py" in cache.PROVENANCE_PATHS
    assert tuple(protocol["datasets"]) == cache.DATASET_NAMES


@pytest.mark.parametrize("dataset_name", cache.DATASET_NAMES)
def test_validate_only_real_contract_decodes_no_pixels_and_creates_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
) -> None:
    class ForbiddenPixelDataset:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("validate-only constructed a pixel dataset")

    monkeypatch.setattr(cache, "IRSTDResearchDataset", ForbiddenPixelDataset)
    output = tmp_path / dataset_name / "must_not_exist"
    result = cache.run(
        Namespace(
            dataset=dataset_name,
            protocol=CONFIG,
            output_dir=output,
            validate_only=True,
        )
    )
    assert result["validate_only"] is True
    assert result["selected_count"] == 64
    assert result["condition_count"] == 13
    assert len(result["expected_condition_tensor_sequence_sha256"]) == 13
    assert all(
        len(value) == 64
        for value in result["expected_condition_tensor_sequence_sha256"].values()
    )
    assert len(result["expected_gt_tensor_sequence_sha256"]) == 64
    assert result["validation"]["dataset_pixel_arrays_decoded"] == 0
    assert result["validation"]["dataset_image_files_hashed_for_source_manifest"] == 0
    assert result["validation"]["dataset_mask_files_hashed_for_source_manifest"] == 0
    assert (
        result["validation"]["dataset_image_file_bytes_read_for_source_manifest"]
        == 0
    )
    assert (
        result["validation"]["dataset_mask_file_bytes_read_for_source_manifest"]
        == 0
    )
    assert result["validation"]["selected_source_manifests_verified"] is False
    assert (
        result["validation"]["selected_source_manifest_byte_verification_deferred"]
        is True
    )
    assert result["validation"]["test_dataset_constructed"] is False
    assert result["validation"]["test_images_opened"] == 0
    assert result["validation"]["test_masks_opened"] == 0
    assert result["validation"]["selected_ids_in_fixed_train"] is True
    assert result["validation"]["selected_ids_absent_from_fixed_test"] is True
    assert result["validation"]["method_received_labels"] is False
    assert result["validation"]["checkpoint_state_dict_strict_load_verified"] is True
    assert result["validation"]["runtime_seal_verified"] is True
    assert len(result["runtime_seal_sha256"]) == 64
    assert result["runtime_seal"]["protocol_sha256"] == sha256_file(CONFIG)
    assert set(result["runtime_seal"]["provenance_files_sha256"]) == set(
        cache.PROVENANCE_PATHS
    )
    assert not output.exists()
    assert not output.parent.exists()


def test_validate_only_refuses_existing_destination(tmp_path: Path) -> None:
    output = tmp_path / "already_complete"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        cache.run(
            Namespace(
                dataset="IRSTD-1K",
                protocol=CONFIG,
                output_dir=output,
                validate_only=True,
            )
        )


def test_little_endian_contiguous_tensor_guard() -> None:
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)[:, :, ::-1]
    assert not source.flags.c_contiguous
    value = cache._require_float32_c(source, (2, 3, 4), "toy")
    assert value.dtype.str == "<f4"
    assert value.flags.c_contiguous
    with pytest.raises(TypeError, match="float32"):
        cache._require_float32_c(source.astype(np.float64), (2, 3, 4), "toy")
    with pytest.raises(ValueError, match="shape mismatch"):
        cache._require_float32_c(source, (3, 2, 4), "toy")


def test_small_memmap_helper_is_little_endian_contiguous_and_hashable(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "condition.npy"
    partial, mapped = cache._write_memmap_atomic(destination, shape=(2, 1, 2, 3))
    mapped[0] = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    mapped[1] = -1.0
    mapped.flush()
    del mapped
    partial.replace(destination)
    loaded = np.load(destination, mmap_mode="r", allow_pickle=False)
    assert loaded.dtype.str == "<f4"
    assert loaded.flags.c_contiguous
    assert len(cache._hash_memmap_records(destination, ("a", "b"))) == 64


def test_atomic_publish_and_refuse_overwrite_without_large_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "cache" / "toy"
    context = {
        "final_output": final,
        "dataset_name": "toy",
        "protocol_sha256": "a" * 64,
        "runtime_seal": {"runtime_seal_sha256": "c" * 64},
    }

    monkeypatch.setattr(
        cache,
        "_assert_context_runtime_seal",
        lambda *_args, **_kwargs: {"verified": True},
    )

    def fake_materialize_staging(
        received: object, staging: Path
    ) -> dict[str, object]:
        assert received is context
        assert staging.parent == final.parent
        return _toy_outer_manifest(staging)

    monkeypatch.setattr(cache, "_materialize_staging", fake_materialize_staging)
    result = cache.materialize(context)
    assert result["published_output_dir"] == str(final)
    assert final.is_dir()
    assert not final.with_name(f".{final.name}.build-{cache.os.getpid()}").exists()
    complete = json.loads((final / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["complete"] is True
    assert complete["manifest_sha256"] == sha256_file(final / "manifest.json")
    assert complete["label_firewall_verified"] is True
    assert complete["method_input_manifest_sha256"] == sha256_file(
        final / cache.METHOD_INPUT_MANIFEST_NAME
    )
    assert complete["runtime_seal_sha256"] == "c" * 64
    assert complete["test_images_opened"] == 0
    assert complete["test_masks_opened"] == 0
    method_manifest_text = (final / cache.METHOD_INPUT_MANIFEST_NAME).read_text(
        encoding="utf-8"
    )
    assert cache.TARGET_RELATIVE_PATH not in method_manifest_text
    assert '"targets"' not in method_manifest_text
    assert not final.with_name(f".{final.name}.publish.lock").exists()
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        cache.materialize(context)


def test_failed_staging_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "cache" / "toy"
    context = {
        "final_output": final,
        "dataset_name": "toy",
        "protocol_sha256": "a" * 64,
        "runtime_seal": {"runtime_seal_sha256": "c" * 64},
    }

    monkeypatch.setattr(
        cache,
        "_assert_context_runtime_seal",
        lambda *_args, **_kwargs: {"verified": True},
    )

    def fail_after_partial(_context: object, staging: Path) -> dict[str, object]:
        (staging / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(cache, "_materialize_staging", fail_after_partial)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        cache.materialize(context)
    assert not final.exists()
    assert not final.with_name(f".{final.name}.build-{cache.os.getpid()}").exists()
    assert not final.with_name(f".{final.name}.publish.lock").exists()


def test_runtime_seal_mutation_fails_closed_before_publish_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = tmp_path / "protocol.yaml"
    tracked = tmp_path / "tracked.py"
    protocol.write_text("schema_version: 1\n", encoding="utf-8")
    tracked.write_text("VERSION = 1\n", encoding="utf-8")
    monkeypatch.setattr(cache, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cache, "PROVENANCE_PATHS", ("tracked.py",))
    seal = cache.capture_runtime_seal(protocol)
    final = tmp_path / "output" / "toy"
    context = {
        "final_output": final,
        "dataset_name": "toy",
        "protocol_sha256": seal["protocol_sha256"],
        "runtime_seal": seal,
    }

    def mutate_during_build(_context: object, staging: Path) -> dict[str, object]:
        (staging / "partial.bin").write_bytes(b"partial")
        tracked.write_text("VERSION = 2\n", encoding="utf-8")
        return {
            "schema_version": 1,
            "cache_content_sha256": "b" * 64,
            "condition_count": 0,
            "image_ids": [],
            "runtime_seconds": 0.0,
        }

    monkeypatch.setattr(cache, "_materialize_staging", mutate_during_build)
    with pytest.raises(RuntimeError, match="runtime provenance seal changed"):
        cache.materialize(context)
    assert not final.exists()
    assert not final.with_name(f".{final.name}.build-{cache.os.getpid()}").exists()
    assert not final.with_name(f".{final.name}.publish.lock").exists()


def test_exclusive_publish_lock_refuses_concurrent_builder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "cache" / "toy"
    final.parent.mkdir(parents=True)
    lock = final.with_name(f".{final.name}.publish.lock")
    lock.write_text("other builder\n", encoding="utf-8")
    context = {
        "final_output": final,
        "dataset_name": "toy",
        "protocol_sha256": "a" * 64,
        "runtime_seal": {"runtime_seal_sha256": "c" * 64},
    }
    monkeypatch.setattr(
        cache,
        "_assert_context_runtime_seal",
        lambda *_args, **_kwargs: {"verified": True},
    )
    with pytest.raises(FileExistsError, match="publish lock already exists"):
        cache.materialize(context)
    assert lock.read_text(encoding="utf-8") == "other builder\n"
    assert not final.exists()


def test_method_consumer_uses_only_sanitized_manifest_and_image_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = _toy_outer_manifest(tmp_path)
    method_manifest = cache._build_method_input_manifest(
        outer, outer_manifest_sha256="d" * 64
    )
    method_path = tmp_path / cache.METHOD_INPUT_MANIFEST_NAME
    method_path.write_text(
        json.dumps(method_manifest, sort_keys=True), encoding="utf-8"
    )
    complete = {
        "complete": True,
        "manifest_sha256": "d" * 64,
        "method_input_manifest_sha256": sha256_file(method_path),
        "runtime_seal_sha256": "c" * 64,
        "label_firewall_verified": True,
        "label_firewall_contract": method_manifest["label_firewall_contract"][
            "contract_id"
        ],
        "label_firewall_evidence_sha256": hashlib.sha256(
            json.dumps(
                method_manifest["label_firewall_contract"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "adaptation_interface_invoked": False,
    }
    (tmp_path / "COMPLETE.json").write_text(
        json.dumps(complete, sort_keys=True), encoding="utf-8"
    )

    def forbidden_outer_loader(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("method consumer parsed the outer evaluator manifest")

    monkeypatch.setattr(cache, "_load_completed_cache_metadata", forbidden_outer_loader)
    opened: list[Path] = []
    real_safe = cache._safe_cache_file

    def recording_safe(root: Path, relative: str, *, label: str) -> Path:
        path = real_safe(root, relative, label=label)
        opened.append(path)
        return path

    class FakeFlags:
        c_contiguous = True
        writeable = False

    class FakeImages:
        shape = (64, 3, 256, 256)
        dtype = np.dtype("<f4")
        flags = FakeFlags()

    loaded_arrays: list[Path] = []

    def fake_np_load(path: Path, **_kwargs: object) -> FakeImages:
        loaded_arrays.append(Path(path))
        return FakeImages()

    monkeypatch.setattr(cache, "_safe_cache_file", recording_safe)
    monkeypatch.setattr(cache.np, "load", fake_np_load)
    consumer = cache.SourceCalibrationMethodInputDataset(
        tmp_path, corruption="clean", severity=0, expected_protocol_sha256="a" * 64
    )
    assert len(consumer) == 64
    assert consumer.targets_opened is False
    assert set(consumer.method_metadata) == {
        "dataset",
        "condition",
        "seed",
        "ordered_ids_sha256",
        "protocol_sha256",
        "runtime_seal_sha256",
        "targets_opened",
        "sample_fields",
    }
    assert loaded_arrays == [tmp_path / "conditions" / "clean_S0.npy"]
    assert all(cache.TARGET_RELATIVE_PATH not in str(path) for path in opened)
    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "outer_evaluator").exists()


def test_firewall_sample_schema_and_delayed_target_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = cache._label_firewall_contract_evidence()
    assert evidence["verified"] is True
    assert tuple(evidence["sample_fields"]) == cache.METHOD_FACING_SAMPLE_FIELDS
    assert evidence["forbidden_fields_exposed"] == []
    sample = cache._build_method_facing_sample(
        image=cache.torch.zeros((3, 2, 2)),
        image_id="a",
        original_size=(2, 2),
        dataset_name="toy",
        corruption="clean",
        severity=0,
        seed=42,
    )
    assert tuple(sample) == cache.METHOD_FACING_SAMPLE_FIELDS
    assert not (set(sample) & cache.METHOD_FACING_FORBIDDEN_FIELDS)

    def forbidden_file_access(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("target gate touched cache files before episodes completed")

    monkeypatch.setattr(cache, "_safe_cache_file", forbidden_file_access)
    with pytest.raises(TypeError, match="expected_protocol_sha256"):
        cache.SourceCalibrationMethodInputDataset(
            tmp_path, corruption="clean", severity=0
        )
    with pytest.raises(TypeError, match="expected protocol SHA256"):
        cache.SourceCalibrationMethodInputDataset(
            tmp_path,
            corruption="clean",
            severity=0,
            expected_protocol_sha256=None,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="episodes_complete mismatch"):
        cache.load_outer_evaluator_targets(
            tmp_path,
            episodes_complete=False,
            expected_protocol_sha256="a" * 64,
        )
    with pytest.raises(TypeError, match="must be a boolean"):
        cache.load_outer_evaluator_targets(
            tmp_path,
            episodes_complete=1,  # type: ignore[arg-type]
            expected_protocol_sha256="a" * 64,
        )
    with pytest.raises(TypeError, match="expected protocol SHA256"):
        cache.load_outer_evaluator_targets(
            tmp_path,
            episodes_complete=True,
            expected_protocol_sha256=None,  # type: ignore[arg-type]
        )


def test_config_types_and_required_hashes_fail_closed() -> None:
    _path, protocol = cache.load_protocol(CONFIG)

    bad_severity = copy.deepcopy(protocol)
    bad_severity["input_protocol"]["ordered_conditions"][0][1] = False
    with pytest.raises(TypeError, match="severity must be an integer"):
        cache._conditions(bad_severity)

    bad_bool = copy.deepcopy(protocol)
    bad_bool["materialized_cache"]["exclusive_sibling_publish_lock"] = 1
    with pytest.raises(TypeError, match="must be a boolean"):
        cache._validate_cache_layout(bad_bool)

    bad_shape = copy.deepcopy(protocol)
    bad_shape["materialized_cache"]["images"]["shape"][1] = True
    with pytest.raises(TypeError, match=r"images shape\[1\] must be an integer"):
        cache._validate_cache_layout(bad_shape)

    missing_hash = copy.deepcopy(protocol)
    missing_hash["materialized_cache"]["required_hashes"].pop()
    with pytest.raises(ValueError, match="required cache hashes mismatch"):
        cache._validate_cache_layout(missing_hash)
