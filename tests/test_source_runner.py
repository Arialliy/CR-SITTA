from __future__ import annotations

from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

import test_source as source_runner
from tta.model_adapter import IRSTDModelAdapter


class FakeNSFPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, image, warm_flag):
        return ([image] if warm_flag else []), self.head(image)


def test_runner_import_does_not_import_real_nsfpn_model() -> None:
    assert "model.MSHNet_NSFPN" not in sys.modules


@pytest.mark.parametrize("wrapper", [None, "state_dict", "net"])
def test_loads_direct_and_standard_wrapped_exact_state_dict(tmp_path, wrapper) -> None:
    source = FakeNSFPN()
    expected = deepcopy(source.state_dict())
    payload = expected if wrapper is None else {wrapper: expected, "epoch": 4}
    checkpoint = tmp_path / "weight.pkl"
    torch.save(payload, checkpoint)

    target = FakeNSFPN()
    with torch.no_grad():
        target.head.weight.zero_()
        target.head.bias.zero_()
    loaded_wrapper = source_runner.load_trusted_checkpoint(target, checkpoint)

    assert loaded_wrapper == ("direct" if wrapper is None else wrapper)
    for key, value in target.state_dict().items():
        assert torch.equal(value, expected[key])


def test_checkpoint_loader_rejects_missing_or_rewritten_keys(tmp_path) -> None:
    model = FakeNSFPN()
    bad_state = {
        f"module.{key}": value for key, value in model.state_dict().items()
    }
    checkpoint = tmp_path / "bad.pkl"
    torch.save(bad_state, checkpoint)

    with pytest.raises(RuntimeError, match="do not exactly match"):
        source_runner.load_trusted_checkpoint(model, checkpoint)


def test_state_dict_hash_is_stable_and_detects_one_value_change() -> None:
    model = FakeNSFPN()
    first = source_runner.state_dict_sha256(model.state_dict())
    second = source_runner.state_dict_sha256(model.state_dict())
    assert first == second

    with torch.no_grad():
        model.head.bias.add_(1.0)
    assert source_runner.state_dict_sha256(model.state_dict()) != first


def test_state_dict_hash_supports_batchnorm_scalar_buffers() -> None:
    model = nn.BatchNorm2d(3)

    digest = source_runner.state_dict_sha256(model.state_dict())

    assert len(digest) == 64


def test_repository_provenance_hashes_requested_runtime_files() -> None:
    provenance = source_runner.repository_provenance(("test_source.py",))

    assert len(provenance["base_commit"]) == 40
    assert len(provenance["head_commit"]) == 40
    assert provenance["file_sha256"]["test_source.py"] == (
        source_runner.sha256_file(source_runner.PROJECT_ROOT / "test_source.py")
    )
    assert isinstance(provenance["worktree_clean"], bool)


def test_full_source_reference_comparison_uses_frozen_protocol() -> None:
    result = source_runner.compare_with_frozen_source_reference(
        "IRSTD-1k",
        source_runner.DATASET_DEFAULTS["IRSTD-1k"]["split"].resolve(),
        None,
        {
            "mean_iou": 0.6934,
            "detection_probability": 0.9558,
            "false_alarm_per_million_pixels": 8.35,
        },
    )

    assert result["evaluated"] is True
    assert result["passed"] is True
    assert all(metric["passed"] for metric in result["metrics"].values())


def test_unified_evaluator_protocol_is_built_from_versioned_contract() -> None:
    protocol = source_runner.build_unified_evaluation_protocol()

    assert protocol.fixed_probability_threshold == 0.5
    assert protocol.connectivity == 2
    assert protocol.max_centroid_distance == 3.0
    assert protocol.froc_probability_thresholds == tuple(i / 20 for i in range(21))


def test_resolve_cuda_device_sets_current_context_for_sfs(monkeypatch) -> None:
    selected = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)

    device = source_runner.resolve_device("cuda:2")

    assert device == torch.device("cuda:2")
    assert selected == [2]


def test_checked_forward_is_exact_and_enforces_spatial_shape() -> None:
    model = FakeNSFPN().eval()
    adapter = IRSTDModelAdapter(model, warm_flag=False)
    adapter.set_source_eval_mode()
    image = torch.randn(1, 3, 6, 7)
    mask = torch.zeros(1, 1, 6, 7)

    logits, repeat_ok = source_runner.checked_source_forward(
        adapter, image, mask, repeat_exact=True
    )

    assert logits.shape == (1, 1, 6, 7)
    assert repeat_ok is True

    with pytest.raises(ValueError, match="spatial sizes differ"):
        source_runner.checked_source_forward(
            adapter, image, torch.zeros(1, 1, 5, 7)
        )


def test_metadata_uncollation_matches_batch_size_one_default_collation() -> None:
    batch = {
        "image_id": ["XDU189"],
        "original_size": [torch.tensor([512]), torch.tensor([512])],
        "dataset": ["IRSTD-1k"],
        "corruption": ["clean"],
        "severity": torch.tensor([0]),
        "seed": torch.tensor([42]),
    }

    metadata = source_runner.metadata_from_batch(batch)

    assert metadata == {
        "image_id": "XDU189",
        "original_size": [512, 512],
        "dataset": "IRSTD-1k",
        "corruption": "clean",
        "severity": 0,
        "seed": 42,
    }


def test_visualization_and_atomic_json_artifacts(tmp_path) -> None:
    image = torch.zeros(1, 3, 4, 5)
    mask = torch.zeros(1, 1, 4, 5)
    logits = torch.zeros(1, 1, 4, 5)
    visualization = tmp_path / "visualizations" / "sample.png"

    source_runner.save_prediction_visualization(
        image, mask, logits, visualization
    )
    source_runner.write_json_atomic(tmp_path / "metrics.json", {"ok": True})
    source_runner.write_jsonl_atomic(
        tmp_path / "per_image.jsonl", [{"image_id": "sample"}]
    )

    assert visualization.is_file()
    assert (tmp_path / "metrics.json").read_text().endswith("\n")
    assert (tmp_path / "per_image.jsonl").read_text() == (
        '{"image_id": "sample"}\n'
    )


def test_normalised_image_conversion_has_expected_shape_and_dtype() -> None:
    image = torch.from_numpy(
        ((np.full((3, 2, 4), 0.5, dtype=np.float32) - 0.5) / 0.25)
    )
    restored = source_runner.normalised_image_to_uint8(image)

    assert restored.shape == (2, 4, 3)
    assert restored.dtype == np.uint8


def test_fake_model_end_to_end_run_writes_complete_artifacts(
    tmp_path, monkeypatch
) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    image_ids = ("sample_a", "sample_b")
    for index, image_id in enumerate(image_ids):
        image = np.full((10, 12, 3), 80 + 40 * index, dtype=np.uint8)
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[4:6, 5:7] = 255
        Image.fromarray(image).save(images_dir / f"{image_id}.png")
        Image.fromarray(mask).save(masks_dir / f"{image_id}.png")
    split = tmp_path / "test.txt"
    split.write_text("\n".join(image_ids), encoding="utf-8")

    checkpoint = tmp_path / "fake_weight.pkl"
    torch.save(FakeNSFPN().state_dict(), checkpoint)
    output_dir = tmp_path / "results"
    monkeypatch.setattr(source_runner, "build_nsfpn_model", FakeNSFPN)
    args = SimpleNamespace(
        dataset="IRSTD-1k",
        root=dataset_root,
        split=split,
        checkpoint=checkpoint,
        device="cpu",
        max_images=None,
        output_dir=output_dir,
        image_size=8,
    )

    aggregate = source_runner.run_source_reproduction(args)

    assert aggregate["evaluated_images"] == 2
    assert aggregate["official"]["image_count"] == 2
    assert aggregate["checks"]["repeat_logit_exact_first_image"] is True
    assert aggregate["checks"]["model_state_unchanged"] is True
    assert aggregate["frozen_reference_comparison"]["evaluated"] is False
    assert "test_source.py" in aggregate["repository_provenance"]["file_sha256"]
    assert aggregate["visualization_count"] == 2
    assert (output_dir / "metrics.json").is_file()
    assert len((output_dir / "per_image.jsonl").read_text().splitlines()) == 2
    assert len(list((output_dir / "visualizations").glob("*.png"))) == 2
