"""Synthetic PNG fixtures and fake CPU hosts only; no research payload access."""
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from analysis import d0a_v7_common as legacy
from analysis import ipma_micro16_data_v1 as data


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def synthetic_pilot(tmp_path_factory):
    root = tmp_path_factory.mktemp("ipma_synthetic_pilot")
    image_dir, mask_dir = root / "images", root / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    ids = [f"synthetic_{index:03d}" for index in range(64)]
    for index, identifier in enumerate(ids):
        width, height = 44 + index % 3, 32 + index % 5
        y, x = np.mgrid[:height, :width]
        gray = ((x * 3 + y * 7 + index) % 256).astype(np.uint8)
        Image.fromarray(np.stack((gray, gray, gray), axis=-1)).save(image_dir / f"{identifier}.png")
        # A native image/mask mismatch is intentional: geometry must follow
        # the real image size, just as the legacy transform does.
        mw, mh = (width - 3, height - 2) if index == 0 else (width, height)
        mask = np.zeros((mh, mw), dtype=np.uint8)
        if index % 5:
            mask[mh // 2:mh // 2 + 2, mw // 2:mw // 2 + 2] = 255
        Image.fromarray(mask).save(mask_dir / f"{identifier}.png")
    train, pilot = root / "train_ids.txt", root / "pilot_ids.txt"
    train.write_text("\n".join(ids) + "\n", encoding="utf-8")
    pilot.write_text("\n".join(ids) + "\n", encoding="utf-8")
    train_sha, pilot_sha = legacy.sha256_file(train), legacy.sha256_file(pilot)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"datasets": {data.DATASET: {
        "checks": {"output_test_overlap_count": 0},
        "output": {"file_sha256": pilot_sha}, "train_split": {"sha256": train_sha},
    }}}), encoding="utf-8")
    cfg = deepcopy(legacy.read_config())
    cfg["sampling"].update(pilot_manifest=str(manifest), pilot_manifest_sha256=legacy.sha256_file(manifest))
    cfg["datasets"][data.DATASET] = {
        "root": str(root), "train_split": str(train), "train_split_sha256": train_sha,
        "pilot_ids": str(pilot), "pilot_ids_sha256": pilot_sha, "train_images": 64,
    }
    records = legacy.load_pilot_records(data.DATASET, cfg)
    expected = [legacy.load_sample(record, data.VIEW, cfg) for record in records[:16]]
    return cfg, records, expected


def test_image_only_all16_match_original_p2_without_gt_opens(synthetic_pilot, monkeypatch):
    cfg, records, expected = synthetic_pilot
    opened = []
    original_open = Image.open
    def image_only(path, *args, **kwargs):
        resolved = Path(path)
        assert resolved.parent.name == "images", "observation attempted a GT open"
        opened.append(str(resolved))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(data.Image, "open", image_only)
    access = {}
    rng_state = random.getstate()
    for index, record in enumerate(records[:16]):
        image, meta = data.load_observation(record, cfg, access)
        assert torch.equal(image, expected[index][0])
        assert meta["input_tensor_sha256"] == expected[index][2]["input_tensor_sha256"]
        assert meta["augmentation_seed"] == expected[index][2]["augmentation_seed"]
        assert meta["view"] == data.VIEW
        assert meta["source_role"] == ("meta_fit_train8" if index < 8 else "meta_check_train8")
        assert meta["ground_truth_loaded"] is False
        assert not any("mask" in key or "target" in key or "path" in key for key in meta)
        assert random.getstate() == rng_state
    assert len(opened) == access["train_image_opens"] == 16
    assert all(access[field] == 0 for field in data.ACCESS_FIELDS if field != "train_image_opens")


def test_fit_target_all8_match_original_gt_and_only_open_fit_masks(synthetic_pilot, monkeypatch):
    cfg, records, expected = synthetic_pilot
    metadata = [data.load_observation(record, cfg, {})[1] for record in records[:8]]
    opened = []
    original_open = Image.open
    allowed = {record["mask_path"] for record in records[:8]}
    def fit_mask_only(path, *args, **kwargs):
        assert str(path) in allowed, "fit label load opened a non-fit mask/image"
        opened.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(data.Image, "open", fit_mask_only)
    access = {}
    rng_state = random.getstate()
    for index, record in enumerate(records[:8]):
        target, meta = data.load_fit_target(record, cfg, access, native_image_size=metadata[index]["native_image_size"])
        assert target.shape == (1, 224, 224)
        assert torch.equal(target, expected[index][1])
        assert meta["target_tensor_sha256"] == expected[index][2]["target_tensor_sha256"]
        assert meta["augmentation_seed"] == expected[index][2]["augmentation_seed"]
        assert meta["source_role"] == "meta_fit_train8"
        assert random.getstate() == rng_state
    assert len(opened) == access["fit_mask_opens"] == access["train_mask_opens"] == 8
    assert access["train_image_opens"] == 0
    assert all(access[field] == 0 for field in data.FORBIDDEN_ACCESS_FIELDS)


@pytest.mark.parametrize("index", [8, 15, 16, 63])
def test_check8_and_other_train_labels_rejected_before_open(synthetic_pilot, monkeypatch, index):
    cfg, records, _ = synthetic_pilot
    def forbidden_open(*args, **kwargs):
        pytest.fail("rejected ID reached Image.open")
    monkeypatch.setattr(data.Image, "open", forbidden_open)
    access = {}
    with pytest.raises(ValueError, match="meta-fit train8"):
        data.load_fit_target(records[index], cfg, access, native_image_size=[44, 32])
    assert all(value == 0 for value in access.values())


@pytest.mark.parametrize("mutation", ["dataset", "unknown_id", "after16", "image_path", "mask_path"])
def test_observation_allowlist_and_both_paths_checked_before_open(synthetic_pilot, monkeypatch, mutation):
    cfg, records, _ = synthetic_pilot
    record = dict(records[0])
    if mutation == "dataset": record["dataset"] = "IRSTD-1K"
    elif mutation == "unknown_id": record["image_id"] = "unlisted"
    elif mutation == "after16": record = dict(records[16])
    else: record[mutation] = records[1][mutation]
    def forbidden_open(*args, **kwargs): pytest.fail("invalid record reached Image.open")
    monkeypatch.setattr(data.Image, "open", forbidden_open)
    with pytest.raises(ValueError):
        data.load_observation(record, cfg, {})


@pytest.mark.parametrize("field", ["train_split_sha256", "pilot_ids_sha256"])
def test_split_and_pilot_hash_drift_rejected_before_decode(synthetic_pilot, monkeypatch, field):
    cfg, records, _ = synthetic_pilot
    changed = deepcopy(cfg)
    changed["datasets"][data.DATASET][field] = "0" * 64
    def forbidden_open(*args, **kwargs): pytest.fail("hash drift reached Image.open")
    monkeypatch.setattr(data.Image, "open", forbidden_open)
    with pytest.raises(ValueError, match="hash mismatch"):
        data.load_observation(records[0], changed, {})


def test_replay_random_state_restored_even_when_transform_raises(synthetic_pilot, monkeypatch):
    cfg, records, _ = synthetic_pilot
    state = random.getstate()
    def fail(*args):
        random.random()
        raise RuntimeError("synthetic transform failure")
    monkeypatch.setattr(data.FixedSplitIRSTDDataset, "_train_transform", fail)
    ledger = {}
    with pytest.raises(RuntimeError, match="synthetic transform failure"):
        data.load_observation(records[0], cfg, ledger)
    assert random.getstate() == state
    assert ledger["train_image_opens"] == 1
    assert ledger["train_mask_opens"] == 0


@pytest.mark.parametrize("size", [None, [], [0, 4], [True, 4], [4.0, 4], [4, 4, 4], "44x32"])
def test_invalid_native_size_rejected_before_mask_open(synthetic_pilot, monkeypatch, size):
    cfg, records, _ = synthetic_pilot
    def forbidden_open(*args, **kwargs): pytest.fail("invalid size reached Image.open")
    monkeypatch.setattr(data.Image, "open", forbidden_open)
    with pytest.raises(ValueError, match="native_image_size"):
        data.load_fit_target(records[0], cfg, {}, native_image_size=size)


@pytest.mark.parametrize("field", list(data.FORBIDDEN_ACCESS_FIELDS))
def test_contaminated_access_ledger_rejected(synthetic_pilot, field):
    cfg, records, _ = synthetic_pilot
    with pytest.raises(ValueError, match="forbidden"):
        data.load_observation(records[0], cfg, {field: 1})


class FakeHost(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 16, 1)
        self.bn = nn.BatchNorm2d(16)
        self.output_0 = nn.Conv2d(16, 1, 1)
        self.last_feature = None

    def forward(self, image, warm_flag):
        assert warm_flag is False
        self.last_feature = self.bn(self.encoder(image))
        return [], self.output_0(self.last_feature)


def frozen_host(kind=FakeHost):
    host = kind().eval()
    host.requires_grad_(False)
    return host


def fake_image():
    return torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(19))


def test_frozen_d0_features_exact_head_replay_detached_clone_and_unchanged_host():
    host, image = frozen_host(), fake_image().requires_grad_()
    before = data.state_digest(host)
    feature, logits = data.extract_d0_source_features(host, image)
    assert feature.shape == (1, 16, 8, 8)
    assert logits.shape == (1, 1, 8, 8)
    assert not feature.requires_grad and feature.grad_fn is None
    assert not logits.requires_grad and logits.grad_fn is None
    assert feature.data_ptr() != host.last_feature.data_ptr()
    assert torch.equal(host.output_0(feature), logits)
    assert data.state_digest(host) == before
    assert not host.output_0._forward_pre_hooks
    assert image.grad is None


@pytest.mark.parametrize("mode", ["all_train", "mixed_bn", "unfrozen_parameter"])
def test_frozen_eval_precondition_without_silent_model_change(mode):
    host = frozen_host()
    if mode == "all_train": host.train()
    elif mode == "mixed_bn": host.bn.train()
    else: host.output_0.weight.requires_grad_(True)
    before = data.state_digest(host)
    with pytest.raises(ValueError):
        data.extract_d0_source_features(host, fake_image())
    assert data.state_digest(host) == before
    assert not host.output_0._forward_pre_hooks


@pytest.mark.parametrize("failure", ["zero_calls", "two_calls", "throw", "wrong_logits", "nonfinite"])
def test_bad_capture_or_replay_rejected_and_hook_always_removed(failure):
    class BadHost(FakeHost):
        def forward(self, image, warm_flag):
            h = self.bn(self.encoder(image))
            if failure == "zero_calls": return [], h[:, :1]
            logits = self.output_0(h)
            if failure == "two_calls": logits = self.output_0(h)
            if failure == "throw": raise RuntimeError("synthetic host failure")
            if failure == "wrong_logits": logits = logits + 1
            if failure == "nonfinite": logits = logits * float("nan")
            return [], logits
    host = frozen_host(BadHost)
    before = data.state_digest(host)
    with pytest.raises(RuntimeError):
        data.extract_d0_source_features(host, fake_image())
    assert not host.output_0._forward_pre_hooks
    assert data.state_digest(host) == before


@pytest.mark.parametrize("mutation", ["parameter", "bn_buffer", "mode", "requires_grad", "nonpersistent_buffer"])
def test_state_digest_detects_all_required_mutations(mutation):
    host = frozen_host()
    host.register_buffer("ephemeral", torch.tensor(0), persistent=False)
    before = data.state_digest(host)
    if mutation == "parameter": host.output_0.weight.add_(1)
    elif mutation == "bn_buffer": host.bn.running_mean.add_(1)
    elif mutation == "mode": host.bn.train()
    elif mutation == "requires_grad": host.output_0.weight.requires_grad_(True)
    else: host.ephemeral.add_(1)
    with pytest.raises(RuntimeError, match="frozen host"):
        data.assert_state_unchanged(host, before)


def test_forward_buffer_mutation_detected_and_temporary_hook_removed():
    class MutatingHost(FakeHost):
        def forward(self, image, warm_flag):
            self.bn.running_mean.add_(1)
            return super().forward(image, warm_flag)
    host = frozen_host(MutatingHost)
    with pytest.raises(RuntimeError, match="frozen host"):
        data.extract_d0_source_features(host, fake_image())
    assert not host.output_0._forward_pre_hooks


def test_existing_user_hook_preserved_after_temporary_hook_cleanup():
    host = frozen_host()
    calls = []
    original = host.output_0.register_forward_pre_hook(lambda module, args: calls.append("called"))
    try:
        existing_keys = set(host.output_0._forward_pre_hooks)
        data.extract_d0_source_features(host, fake_image())
        assert set(host.output_0._forward_pre_hooks) == existing_keys
        assert len(calls) == 2  # actual frozen forward and direct head replay
    finally:
        original.remove()


@pytest.mark.parametrize("image", [torch.zeros(3, 8, 8), torch.zeros(2, 3, 8, 8), torch.full((1, 3, 8, 8), float("nan"))])
def test_bad_image_rejected_without_hook_installation(image):
    host = frozen_host()
    with pytest.raises(ValueError):
        data.extract_d0_source_features(host, image)
    assert not host.output_0._forward_pre_hooks
