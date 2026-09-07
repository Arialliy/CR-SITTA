"""Synthetic train-mask fixtures only; never load real project GT or weights."""

import copy
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from analysis import o3_multiscale_train_data_v1 as data


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    ids = [f"sample_{index:04d}" for index in range(663)]
    train_path = tmp_path / data.TRAIN_SPLIT
    train_path.parent.mkdir(parents=True)
    train_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    train_sha = _sha(train_path.read_bytes())
    protocol = {"datasets": {data.DATASET: {"root": data.DATASET_ROOT,
                "train_split": data.TRAIN_SPLIT, "train_split_sha256": train_sha}},
                "preprocessing": {"mask_resize": {"size": [256, 256], "interpolation": "nearest"},
                                  "mask_corrupted": False}}
    protocol_path = tmp_path / "cache_protocol.yaml"
    protocol_path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    spec = {"train_split_sha256": train_sha, "cache_root": "synthetic_cache",
            "checkpoint_sha256": "a" * 64, "cache_manifest_sha256": "b" * 64}
    config_path = tmp_path / "parent.yaml"
    config_path.write_text("synthetic parent", encoding="utf-8")
    contract = data.b4.FullPilotContract(tmp_path, config_path, _sha(config_path.read_bytes()),
        {"datasets": {data.DATASET: spec}, "frozen_parent_bindings": {"cache_protocol": {
            "path": protocol_path.name, "sha256": _sha(protocol_path.read_bytes())}}})
    masks = tmp_path / data.DATASET_ROOT / "masks"
    masks.mkdir()
    source_files = {}
    originals = []
    for index, identifier in enumerate(ids[:8]):
        values = np.array([[0, 128, 255], [255, 0, index]], dtype=np.uint8)
        originals.append(values)
        path = masks / f"{identifier}.png"
        Image.fromarray(values).save(path)
        source_files[identifier] = {"mask_sha256": _sha(path.read_bytes())}
    manifest = {"dataset": data.DATASET, "image_ids": ids[:64],
                "train_split": data.TRAIN_SPLIT, "train_split_sha256": train_sha,
                "source_files": source_files}
    calls = []
    monkeypatch.setattr(data.b4, "load_contract", lambda path: contract)
    monkeypatch.setattr(data.b4, "_teacher_manifest", lambda parent, dataset: {"image_ids": ids[:64]})
    monkeypatch.setattr(data.cache_api, "_consumer_metadata", lambda *args, **kwargs:
                        ({}, manifest, {"image_ids": ids[:64]}))

    def consumed(parent, dataset, *, include_outer_target):
        assert dataset == data.DATASET
        assert include_outer_target is False
        calls.append("verified_image_and_teacher_payloads")

    def forbidden(*args, **kwargs):
        pytest.fail("sealed outer target API must not be called")

    monkeypatch.setattr(data.b4, "_verify_consumed_payloads", consumed)
    monkeypatch.setattr(data.cache_api, "load_outer_evaluator_targets_v2", forbidden)
    monkeypatch.setattr(data.cache_api, "_load_verified_numpy_payload", forbidden)
    return {"parent": contract, "ids": ids[:8], "all_ids": ids, "manifest": manifest,
            "masks": masks, "originals": originals, "calls": calls,
            "train_path": train_path, "protocol_path": protocol_path}


def test_train8_exact_original_preprocessing_and_receipt(fixture):
    values, receipt = data.load_train_targets(fixture["parent"], fixture["ids"])
    expected = np.stack([np.asarray(Image.fromarray(v).convert("L").resize(
        (256, 256), Image.Resampling.NEAREST), dtype=np.float32)[None] / 255.0
        for v in fixture["originals"]])
    assert values.shape == (8, 1, 256, 256)
    assert values.dtype == np.float32
    assert np.array_equal(values, expected)
    assert not values.flags.writeable
    assert np.float32(128 / 255) in values
    assert receipt["role"] == "source_supervised_residual_training_train8"
    assert receipt["image_ids"] == fixture["ids"]
    assert receipt["train_mask_png_decodes"] == 8
    assert receipt["unique_train_mask_files"] == 8
    assert receipt["sealed_outer_target_payload_opens"] == 0
    assert receipt["outer_target_loader_calls"] == 0
    assert receipt["other_pilot_mask_decodes"] == 0
    assert receipt["test_mask_decodes"] == receipt["validation_mask_decodes"] == 0
    assert receipt["supervised_labels_enter_original_o3_update"] is False
    assert receipt["paper_result"] is receipt["formal_test"] is False
    assert receipt["target_tensor_sha256"] == _sha(values.tobytes())
    assert fixture["calls"] == ["verified_image_and_teacher_payloads"]
    for record in receipt["masks"]:
        original = fixture["masks"] / f"{record['image_id']}.png"
        assert record["file_sha256"] == _sha(original.read_bytes())


def test_repeat_load_is_exact_and_does_not_modify_original_masks(fixture):
    paths = list(fixture["masks"].glob("*.png"))
    before = {p: p.read_bytes() for p in paths}
    first, receipt1 = data.load_train_targets(fixture["parent"], tuple(fixture["ids"]))
    second, receipt2 = data.load_train_targets(fixture["parent"], fixture["ids"])
    assert np.array_equal(first, second)
    assert receipt1 == receipt2
    assert {p: p.read_bytes() for p in paths} == before


@pytest.mark.parametrize("kind", ["wrong_order", "later_pilot", "missing", "duplicate", "string", "bad_type"])
def test_only_exact_first8_is_accepted(fixture, monkeypatch, kind):
    ids = list(fixture["ids"])
    if kind == "wrong_order":
        ids.reverse()
    elif kind == "later_pilot":
        ids[-1] = fixture["all_ids"][8]
    elif kind == "missing":
        ids.pop()
    elif kind == "duplicate":
        ids[-1] = ids[0]
    elif kind == "string":
        ids = ids[0]
    else:
        ids[0] = 0
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("mask decoded before ID validation"))
    with pytest.raises(data.SourceSupervisedDataError):
        data.load_train_targets(fixture["parent"], ids)


def test_train_membership_is_explicitly_checked(fixture, monkeypatch):
    rogue = list(fixture["all_ids"][:64])
    rogue[-1] = "not_official_train"
    monkeypatch.setattr(data.b4, "_teacher_manifest", lambda *args: {"image_ids": rogue})
    with pytest.raises(data.SourceSupervisedDataError, match="subset"):
        data.load_train_targets(fixture["parent"], fixture["ids"])


def test_changed_train_text_fails_before_mask_decode(fixture, monkeypatch):
    fixture["train_path"].write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("mask decoded before hash check"))
    with pytest.raises(data.SourceSupervisedDataError, match="actual train split SHA256"):
        data.load_train_targets(fixture["parent"], fixture["ids"])


def test_changed_cache_protocol_fails_before_mask_decode(fixture, monkeypatch):
    fixture["protocol_path"].write_text("changed\n", encoding="utf-8")
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("mask decoded before hash check"))
    with pytest.raises(data.SourceSupervisedDataError, match="cache protocol SHA256"):
        data.load_train_targets(fixture["parent"], fixture["ids"])


def test_any_source_mask_hash_mismatch_stops_before_any_png_decode(fixture, monkeypatch):
    bad = fixture["masks"] / f"{fixture['ids'][-1]}.png"
    bad.write_bytes(b"not the frozen original")
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("decoded before all raw masks hash-verified"))
    with pytest.raises(data.SourceSupervisedDataError, match="source train mask SHA256"):
        data.load_train_targets(fixture["parent"], fixture["ids"])


def test_source_mask_symlink_rejected(fixture, tmp_path):
    path = fixture["masks"] / f"{fixture['ids'][0]}.png"
    original = tmp_path / "outside.png"
    original.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(original)
    with pytest.raises((ValueError, OSError)):
        data.load_train_targets(fixture["parent"], fixture["ids"])


def test_parent_must_match_reloaded_frozen_contract(fixture):
    altered = copy.deepcopy(fixture["parent"])
    altered.raw["datasets"][data.DATASET]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(data.SourceSupervisedDataError, match="parent.raw"):
        data.load_train_targets(altered, fixture["ids"])


def test_parent_wrong_type_rejected(fixture):
    with pytest.raises(data.SourceSupervisedDataError, match="parent contract"):
        data.load_train_targets({}, fixture["ids"])


def test_frozen_image_or_teacher_hash_failure_prevents_mask_decode(fixture, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("frozen image cache changed")
    monkeypatch.setattr(data.b4, "_verify_consumed_payloads", fail)
    monkeypatch.setattr(Image, "open", lambda *args, **kwargs: pytest.fail("decoded before cache checks"))
    with pytest.raises(RuntimeError, match="frozen image cache changed"):
        data.load_train_targets(fixture["parent"], fixture["ids"])


def test_teacher_and_cache_order_must_match(fixture):
    fixture["manifest"]["image_ids"] = list(reversed(fixture["all_ids"][:64]))
    with pytest.raises(data.SourceSupervisedDataError, match="cache/teacher ordered IDs"):
        data.load_train_targets(fixture["parent"], fixture["ids"])


def test_no_image_or_sealed_target_path_is_opened(fixture, monkeypatch):
    read = data.cache_api.secure_io._read_stable_bytes
    opened = []
    def track(path, *, label):
        path = Path(path)
        assert "outer_evaluator" not in path.parts
        assert "images" not in path.parts
        assert not path.name.startswith("test_")
        opened.append(path)
        return read(path, label=label)
    monkeypatch.setattr(data.cache_api.secure_io, "_read_stable_bytes", track)
    data.load_train_targets(fixture["parent"], fixture["ids"])
    assert len([p for p in opened if p.suffix == ".png"]) == 8
