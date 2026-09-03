from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from analysis import d0_v3_phase_receipt as phase
from tta.d0_secure_io import SecureIOError


DATASET = "NUAA-SIRST"
CONDITION = "clean_S0"
PROTOCOL_SHA = "1" * 64
CHECKPOINT_SHA = "2" * 64
CONFIG_SHA = "3" * 64
CODE_SEALS = {
    "analysis/formal_runner.py": "4" * 64,
    "tta/formal_observer.py": "5" * 64,
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _image_ids() -> list[str]:
    return [f"pilot_{index:02d}" for index in range(phase.IMAGE_COUNT)]


def _pilot_sha(image_ids: list[str]) -> str:
    payload = json.dumps(
        image_ids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _episodes(image_ids: list[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    serial = 1
    for image_index, image_id in enumerate(image_ids):
        for candidate_index, candidate in enumerate(phase.FROZEN_CANDIDATES):
            records.append(
                {
                    "image_index": image_index,
                    "image_id": image_id,
                    "candidate_index": candidate_index,
                    "candidate_slug": candidate.slug,
                    "complete": True,
                    "episode_receipt_sha256": f"{serial:064x}",
                }
            )
            serial += 1
    return records


def _method_manifest(image_ids: list[str]) -> dict[str, object]:
    return {
        "schema_version": 2,
        "protocol_sha256": PROTOCOL_SHA,
        "dataset": DATASET,
        "image_ids": image_ids,
        "ordered_ids_sha256": _pilot_sha(image_ids),
        "conditions": [{"key": value} for value in phase.CONDITIONS],
        "targets_exposed": False,
    }


def _cache(tmp_path: Path) -> tuple[Path, str]:
    cache_root = tmp_path / "cache" / DATASET
    cache_root.mkdir(parents=True)
    payload = _canonical(_method_manifest(_image_ids()))
    (cache_root / "method_input_manifest.json").write_bytes(payload)
    return cache_root, hashlib.sha256(payload).hexdigest()


def _receipt(method_manifest_sha256: str) -> dict[str, object]:
    image_ids = _image_ids()
    return phase.build_label_free_cell_receipt(
        dataset=DATASET,
        condition=CONDITION,
        replicate=0,
        ordered_image_ids=image_ids,
        episode_records=_episodes(image_ids),
        cache_protocol_sha256=PROTOCOL_SHA,
        cache_method_manifest_sha256=method_manifest_sha256,
        checkpoint_sha256=CHECKPOINT_SHA,
        config_sha256=CONFIG_SHA,
        code_seals=CODE_SEALS,
        candidate_phase_data_boundary=phase.zero_candidate_phase_data_boundary(),
    )


def test_build_is_canonical_complete_and_order_independent(tmp_path: Path) -> None:
    _, method_sha = _cache(tmp_path)
    first = _receipt(method_sha)
    image_ids = _image_ids()
    second = phase.build_label_free_cell_receipt(
        dataset=DATASET,
        condition=CONDITION,
        replicate=0,
        ordered_image_ids=image_ids,
        episode_records=list(reversed(_episodes(image_ids))),
        cache_protocol_sha256=PROTOCOL_SHA,
        cache_method_manifest_sha256=method_sha,
        checkpoint_sha256=CHECKPOINT_SHA,
        config_sha256=CONFIG_SHA,
        code_seals=dict(reversed(list(CODE_SEALS.items()))),
        candidate_phase_data_boundary=phase.zero_candidate_phase_data_boundary(),
    )

    first_bytes = phase.canonical_label_free_cell_receipt_bytes(first)
    assert first_bytes == phase.canonical_label_free_cell_receipt_bytes(second)
    assert phase.label_free_cell_receipt_sha256(first) == hashlib.sha256(
        first_bytes
    ).hexdigest()
    validated = phase.validate_label_free_cell_receipt(
        first_bytes,
        expected_dataset=DATASET,
        expected_condition=CONDITION,
        expected_replicate=0,
        expected_cache_protocol_sha256=PROTOCOL_SHA,
        expected_cache_method_manifest_sha256=method_sha,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_config_sha256=CONFIG_SHA,
        expected_code_seals=CODE_SEALS,
        expected_ordered_image_ids=image_ids,
    )
    assert len(validated.ordered_image_ids) == 64
    assert len(validated.episode_receipt_sha256s) == 640
    assert validated.to_dict()["completion"] == {
        "candidate_grid_complete": True,
        "complete_episode_count": 640,
        "episode_receipts_complete": True,
        "label_free_phase_complete": True,
        "ordered_images_complete": True,
    }


def test_less_than_640_or_duplicate_episode_records_fail_closed(tmp_path: Path) -> None:
    _, method_sha = _cache(tmp_path)
    image_ids = _image_ids()
    with pytest.raises(phase.D0V3PhaseReceiptError, match="exactly 640"):
        phase.build_label_free_cell_receipt(
            dataset=DATASET,
            condition=CONDITION,
            replicate=0,
            ordered_image_ids=image_ids,
            episode_records=_episodes(image_ids)[:-1],
            cache_protocol_sha256=PROTOCOL_SHA,
            cache_method_manifest_sha256=method_sha,
            checkpoint_sha256=CHECKPOINT_SHA,
            config_sha256=CONFIG_SHA,
            code_seals=CODE_SEALS,
            candidate_phase_data_boundary=phase.zero_candidate_phase_data_boundary(),
        )

    duplicate = _episodes(image_ids)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(phase.D0V3PhaseReceiptError, match="duplicate episode"):
        phase.build_label_free_cell_receipt(
            dataset=DATASET,
            condition=CONDITION,
            replicate=0,
            ordered_image_ids=image_ids,
            episode_records=duplicate,
            cache_protocol_sha256=PROTOCOL_SHA,
            cache_method_manifest_sha256=method_sha,
            checkpoint_sha256=CHECKPOINT_SHA,
            config_sha256=CONFIG_SHA,
            code_seals=CODE_SEALS,
            candidate_phase_data_boundary=phase.zero_candidate_phase_data_boundary(),
        )


def test_episode_tamper_and_noncanonical_bytes_fail_closed(tmp_path: Path) -> None:
    _, method_sha = _cache(tmp_path)
    receipt = _receipt(method_sha)
    receipt["episodes"]["records"][0]["episode_receipt_sha256"] = "f" * 64
    with pytest.raises(phase.D0V3PhaseReceiptError, match="hash-list seal|ledger seal"):
        phase.validate_label_free_cell_receipt(_canonical(receipt))

    valid = phase.canonical_label_free_cell_receipt_bytes(_receipt(method_sha))
    with pytest.raises(phase.D0V3PhaseReceiptError, match="canonical representation"):
        phase.validate_label_free_cell_receipt(valid + b"\n")


@pytest.mark.parametrize(
    "field",
    [
        "train_target_payload_bytes_opened",
        "train_target_payload_deserialization_count",
        "train_target_indexing_count",
        "test_split_files_opened",
        "test_images_opened",
        "test_masks_opened",
        "test_labels_opened",
        "method_label_accesses",
    ],
)
def test_any_gt_or_test_access_count_fails_closed(
    tmp_path: Path, field: str
) -> None:
    _, method_sha = _cache(tmp_path)
    receipt = _receipt(method_sha)
    receipt["candidate_phase_data_boundary"][field] = 1
    with pytest.raises(phase.D0V3PhaseReceiptError, match=field):
        phase.validate_label_free_cell_receipt(receipt)


def test_guarded_loader_binds_cell_and_emits_non_adaptation_access_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, method_sha = _cache(tmp_path)
    receipt = _receipt(method_sha)
    receipt_bytes = phase.canonical_label_free_cell_receipt_bytes(receipt)
    calls: list[tuple[Path, str, bool]] = []

    def fake_loader(
        root: Path, *, expected_protocol_sha256: str, episodes_complete: bool
    ) -> np.ndarray:
        calls.append((root, expected_protocol_sha256, episodes_complete))
        return np.zeros((64, 1, 256, 256), dtype=np.float32)

    monkeypatch.setattr(phase, "load_outer_evaluator_targets_v2", fake_loader)
    result = phase.guarded_load_outer_targets(
        cache_root,
        PROTOCOL_SHA,
        receipt_bytes,
        dataset=DATASET,
        condition=CONDITION,
        replicate=0,
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        expected_config_sha256=CONFIG_SHA,
        expected_code_seals=CODE_SEALS,
        expected_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
    )

    assert calls == [(cache_root, PROTOCOL_SHA, True)]
    access = result.access_receipt()
    assert access["phase_evidence"]["complete_candidate_episode_count"] == 640
    assert access["phase_evidence"]["candidate_phase_target_access_count"] == 0
    assert access["outer_access"]["used_by_adaptation"] is False
    assert access["outer_access"]["adaptation_target_indexing_count"] == 0
    assert access["authorization"]["stage2_authorized"] is False
    assert result.label_free_receipt_sha256 == hashlib.sha256(receipt_bytes).hexdigest()
    assert result.access_receipt_sha256 == hashlib.sha256(
        result.access_receipt_bytes
    ).hexdigest()


def test_wrong_current_condition_rejected_before_target_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, method_sha = _cache(tmp_path)
    calls = 0

    def forbidden_loader(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("target loader must remain unreachable")

    monkeypatch.setattr(phase, "load_outer_evaluator_targets_v2", forbidden_loader)
    with pytest.raises(phase.D0V3PhaseReceiptError, match="condition mismatch"):
        phase.guarded_load_outer_targets(
            cache_root,
            PROTOCOL_SHA,
            phase.canonical_label_free_cell_receipt_bytes(_receipt(method_sha)),
            dataset=DATASET,
            condition="gaussian_noise_S1",
            replicate=0,
        )
    assert calls == 0


def test_live_method_manifest_tamper_rejected_before_target_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, method_sha = _cache(tmp_path)
    receipt_bytes = phase.canonical_label_free_cell_receipt_bytes(_receipt(method_sha))
    (cache_root / "method_input_manifest.json").write_bytes(b"{}")
    calls = 0

    def forbidden_loader(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("target loader must remain unreachable")

    monkeypatch.setattr(phase, "load_outer_evaluator_targets_v2", forbidden_loader)
    with pytest.raises(phase.D0V3PhaseReceiptError, match="manifest SHA-256"):
        phase.guarded_load_outer_targets(
            cache_root,
            PROTOCOL_SHA,
            receipt_bytes,
            dataset=DATASET,
            condition=CONDITION,
            replicate=0,
        )
    assert calls == 0


def test_symlink_receipt_is_rejected_without_loader_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, method_sha = _cache(tmp_path)
    real = tmp_path / "receipt.json"
    real.write_bytes(phase.canonical_label_free_cell_receipt_bytes(_receipt(method_sha)))
    link = tmp_path / "receipt-link.json"
    link.symlink_to(real.name)
    calls = 0

    def forbidden_loader(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("target loader must remain unreachable")

    monkeypatch.setattr(phase, "load_outer_evaluator_targets_v2", forbidden_loader)
    with pytest.raises(ValueError, match="symlink|regular non-symlink"):
        phase.guarded_load_outer_targets(
            cache_root,
            PROTOCOL_SHA,
            link,
            dataset=DATASET,
            condition=CONDITION,
            replicate=0,
        )
    assert calls == 0


def test_unstable_receipt_read_is_rejected_without_loader_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, method_sha = _cache(tmp_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(
        phase.canonical_label_free_cell_receipt_bytes(_receipt(method_sha))
    )
    calls = 0

    def unstable_read(_path: object):
        raise SecureIOError("file changed while being read")

    def forbidden_loader(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("target loader must remain unreachable")

    monkeypatch.setattr(phase, "read_stable_regular_file", unstable_read)
    monkeypatch.setattr(phase, "load_outer_evaluator_targets_v2", forbidden_loader)
    with pytest.raises(SecureIOError, match="changed while being read"):
        phase.guarded_load_outer_targets(
            cache_root,
            PROTOCOL_SHA,
            receipt_path,
            dataset=DATASET,
            condition=CONDITION,
            replicate=0,
        )
    assert calls == 0


def test_mapping_and_boolean_are_not_accepted_as_guard_receipt_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root, method_sha = _cache(tmp_path)
    receipt = _receipt(method_sha)
    monkeypatch.setattr(
        phase,
        "load_outer_evaluator_targets_v2",
        lambda *_args, **_kwargs: pytest.fail("target loader must remain unreachable"),
    )
    for unsafe in (receipt, True):
        with pytest.raises(phase.D0V3PhaseReceiptError, match="immutable bytes|stable file"):
            phase.guarded_load_outer_targets(
                cache_root,
                PROTOCOL_SHA,
                unsafe,  # type: ignore[arg-type]
                dataset=DATASET,
                condition=CONDITION,
                replicate=0,
            )
