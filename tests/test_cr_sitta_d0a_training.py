from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader, Dataset

import train_cr_sitta_d0a as runner
from train_fixed_split import corpus_manifest, sha256_file, train_one_epoch
from tta.deteriorations.image_space import imagenet_normalize


@pytest.mark.parametrize(
    "step, expected",
    [
        (0, "lf_mask"),
        (1, "hf_noise"),
        (2, "lf_mask"),
        (3, "hf_noise"),
        (100, "lf_mask"),
        (101, "hf_noise"),
    ],
)
def test_probe_kind_alternates_lf_then_hf(step: int, expected: str) -> None:
    assert runner.probe_kind_for_step(step) == expected


def test_probe_seed_is_stable_and_binds_every_declared_field() -> None:
    fields: dict[str, object] = {
        "protocol_id": "cr-sitta-d0a-v1",
        "global_seed": 42,
        "dataset_id": "IRSTD-1K",
        "human_epoch": 17,
        "image_id": "XDU001",
        "probe_id": "lf_mask",
    }

    def derive(values: dict[str, object]) -> int:
        return runner.derive_probe_seed(
            str(values["protocol_id"]),
            int(values["global_seed"]),
            str(values["dataset_id"]),
            int(values["human_epoch"]),
            str(values["image_id"]),
            str(values["probe_id"]),
        )

    baseline = derive(fields)
    assert derive(dict(fields)) == baseline
    assert isinstance(baseline, int)

    replacements: dict[str, object] = {
        "protocol_id": "cr-sitta-d0a-v2",
        "global_seed": 43,
        "dataset_id": "NUAA-SIRST",
        "human_epoch": 18,
        "image_id": "XDU002",
        "probe_id": "hf_noise",
    }
    changed_seeds = []
    for name, replacement in replacements.items():
        changed = dict(fields)
        changed[name] = replacement
        changed_seeds.append(derive(changed))
    assert all(value != baseline for value in changed_seeds)
    assert len(set(changed_seeds)) == len(changed_seeds)


def _build_view(
    image: Tensor,
    *,
    probe_id: str,
    epoch: int = 5,
) -> Tensor:
    return runner.build_deteriorated_view(
        image,
        ("sample-a",),
        probe_id,
        "cr-sitta-d0a-view-test-v1",
        3407,
        "NUDT-SIRST",
        epoch,
        0.20,
        0.50,
        0.02,
        0.20,
    )


@pytest.mark.parametrize("probe_id", ["lf_mask", "hf_noise"])
def test_deteriorated_view_is_bit_exact_finite_shape_preserving_and_pure(
    probe_id: str,
) -> None:
    physical = torch.linspace(
        0.15, 0.85, 3 * 16 * 18, dtype=torch.float32
    ).reshape(1, 3, 16, 18)
    normalized = imagenet_normalize(physical)
    original = normalized.clone()

    first = _build_view(normalized, probe_id=probe_id)
    second = _build_view(normalized.clone(), probe_id=probe_id)

    assert isinstance(first, Tensor)
    assert torch.equal(first, second)
    assert first.shape == normalized.shape
    assert first.dtype == normalized.dtype
    assert first.device == normalized.device
    assert bool(torch.isfinite(first).all())
    assert torch.equal(normalized, original)
    assert first.data_ptr() != normalized.data_ptr()


def test_label_free_view_api_has_no_target_or_benchmark_condition_channel() -> None:
    forbidden = {
        "target",
        "targets",
        "mask",
        "masks",
        "label",
        "labels",
        "condition",
        "corruption",
        "severity",
    }
    view_parameters = set(
        inspect.signature(runner.build_deteriorated_view).parameters
    )
    seed_parameters = set(inspect.signature(runner.derive_probe_seed).parameters)
    assert not view_parameters.intersection(forbidden)
    assert not seed_parameters.intersection(forbidden)


class _BNContextToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=False)
        self.bn_training = nn.BatchNorm2d(3)
        self.branch = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout2d(0.25))

    def forward(self, image: Tensor) -> Tensor:
        value = self.bn_training(self.conv(image))
        return self.branch(value)


def test_batchnorm_eval_only_preserves_running_state_and_restores_all_modes() -> None:
    model = _BNContextToy().train()
    # Exercise restoration of a pre-existing non-uniform module-mode layout.
    model.branch[0].eval()
    modes_before = tuple(
        (name, bool(module.training)) for name, module in model.named_modules()
    )
    bn_state_before = {
        name: (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone(),
        )
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    }

    with runner.batchnorm_eval_only(model):
        assert model.training
        assert model.conv.training
        assert model.branch[1].training
        assert all(
            not module.training
            for module in model.modules()
            if isinstance(module, nn.BatchNorm2d)
        )
        output = model(torch.randn(2, 3, 8, 8))
        assert bool(torch.isfinite(output).all())

    assert tuple(
        (name, bool(module.training)) for name, module in model.named_modules()
    ) == modes_before
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            mean, variance, count = bn_state_before[name]
            assert torch.equal(module.running_mean, mean)
            assert torch.equal(module.running_var, variance)
            assert torch.equal(module.num_batches_tracked, count)


def test_batchnorm_batch_stats_policy_is_finite_and_does_not_persist_buffers() -> None:
    model = _BNContextToy().train()
    snapshots = {
        name: (
            module.running_mean.detach().clone(),
            module.running_var.detach().clone(),
            module.num_batches_tracked.detach().clone(),
        )
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    }
    modes = tuple((name, module.training) for name, module in model.named_modules())
    with runner.batchnorm_batch_stats_no_running_update(model):
        assert all(
            module.training and not module.track_running_stats
            for module in model.modules()
            if isinstance(module, nn.BatchNorm2d)
        )
        output = model(torch.randn(2, 3, 8, 8) * 100.0)
        assert bool(torch.isfinite(output).all())
    assert tuple((name, module.training) for name, module in model.named_modules()) == modes
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            mean, variance, count = snapshots[name]
            assert module.track_running_stats
            assert torch.equal(module.running_mean, mean)
            assert torch.equal(module.running_var, variance)
            assert torch.equal(module.num_batches_tracked, count)


class _RecordingSegmentationLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        warm_epochs: int,
        epoch_index: int,
    ) -> Tensor:
        self.calls.append(
            (
                tuple(prediction.shape),
                tuple(target.shape),
                int(warm_epochs),
                int(epoch_index),
            )
        )
        return prediction.mean()


def test_compute_segmentation_loss_matches_nsfpn_multiscale_contract() -> None:
    auxiliary = [
        torch.full((1, 1, 8, 8), 2.0, requires_grad=True),
        torch.full((1, 1, 4, 4), 3.0, requires_grad=True),
        torch.full((1, 1, 2, 2), 4.0, requires_grad=True),
        torch.full((1, 1, 1, 1), 5.0, requires_grad=True),
    ]
    final = torch.ones((1, 1, 8, 8), requires_grad=True)
    target = torch.zeros((1, 1, 8, 8))
    loss_function = _RecordingSegmentationLoss()

    loss = runner.compute_segmentation_loss(
        (auxiliary, final),
        target,
        loss_function,
        warm_epochs=5,
        epoch_index=2,
    )

    assert loss.item() == pytest.approx(3.0)
    assert [call[1][-2:] for call in loss_function.calls] == [
        (8, 8),
        (8, 8),
        (4, 4),
        (2, 2),
        (1, 1),
    ]
    assert all(call[2:] == (5, 2) for call in loss_function.calls)
    loss.backward()
    assert final.grad is not None
    assert all(prediction.grad is not None for prediction in auxiliary)


def _write_train_fixture(tmp_path: Path) -> dict[str, object]:
    dataset_root = tmp_path / "source_train_dataset"
    image_root = dataset_root / "images"
    mask_root = dataset_root / "masks"
    image_root.mkdir(parents=True)
    mask_root.mkdir()
    identifiers = ["sample-a", "sample-b"]
    for index, identifier in enumerate(identifiers):
        Image.new(
            "RGB", (6, 5), color=(20 + index, 40 + index, 60 + index)
        ).save(image_root / f"{identifier}.png")
        Image.new("L", (6, 5), color=255 if index else 0).save(
            mask_root / f"{identifier}.png"
        )
    train_split = dataset_root / "train_ids.txt"
    train_split.write_text("sample-a\nsample-b\n", encoding="utf-8")
    corpus_sha256, mismatches = corpus_manifest(dataset_root, identifiers)
    assert mismatches == []
    return {
        "root": str(dataset_root),
        "train_split": str(train_split),
        "expected_train_split_sha256": sha256_file(train_split),
        "expected_train_images": len(identifiers),
        "expected_train_corpus_manifest_sha256": corpus_sha256,
        "known_train_size_mismatches": [],
    }


def test_train_only_split_validation_needs_and_opens_no_test_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _write_train_fixture(tmp_path)
    sentinel = Path(str(config["root"])) / "test_ids.txt"
    sentinel.write_text("must-not-be-opened\n", encoding="utf-8")
    # Supplying a poison test path proves the train-only validator ignores it;
    # omitting it must also remain valid because it is outside this API.
    config["test_split"] = str(sentinel)

    original_open = Path.open
    opened: list[Path] = []

    def audited_open(path: Path, *args: object, **kwargs: object):
        lowered = path.name.lower()
        if "test" in lowered or "validation" in lowered:
            raise AssertionError(f"train-only validator opened {path}")
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audited_open)
    result = runner.validate_train_only_split(config)

    assert result["train_count"] == 2
    assert result["train_split_sha256"] == config[
        "expected_train_split_sha256"
    ]
    assert result["train_corpus_manifest_sha256"] == config[
        "expected_train_corpus_manifest_sha256"
    ]
    assert sentinel not in opened


def test_train_only_split_validation_rejects_train_hash_drift(
    tmp_path: Path,
) -> None:
    config = _write_train_fixture(tmp_path)
    config["expected_train_split_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="train split hash"):
        runner.validate_train_only_split(config)


class _OneBatchDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(81)
        self.image = torch.rand(2, 3, 8, 8, generator=generator)
        self.target = (torch.rand(2, 1, 8, 8, generator=generator) > 0.8).float()

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        return self.image[index], self.target[index], f"sample-{index}"


class _TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.output = nn.Conv2d(3, 1, 1)

    def forward(self, image: Tensor, warm_flag: bool):
        del warm_flag
        return [], self.output(self.bn(image))


def test_lambda_zero_is_exactly_the_baseline_single_branch_update() -> None:
    torch.manual_seed(12)
    baseline = _TinySegmentationModel()
    candidate = _TinySegmentationModel()
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    baseline_optimizer = Adagrad(baseline.parameters(), lr=0.05)
    candidate_optimizer = Adagrad(candidate.parameters(), lr=0.05)
    dataset = _OneBatchDataset()
    baseline_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    candidate_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    baseline_metrics = train_one_epoch(
        baseline,
        baseline_optimizer,
        baseline_loader,
        runner.SLSIoULoss(),
        torch.device("cpu"),
        human_epoch=1,
        warm_epochs=5,
        max_batches=1,
    )
    candidate_metrics = runner.train_one_epoch_d0a(
        candidate,
        candidate_optimizer,
        candidate_loader,
        runner.SLSIoULoss(),
        torch.device("cpu"),
        {
            "warm_epochs": 5,
            "lambda_degraded": 0.0,
            "train_only_smoke": False,
            "max_train_batches": 1,
        },
        human_epoch=1,
        starting_optimizer_step=0,
    )

    assert candidate_metrics["mean_degraded_loss"] is None
    assert candidate_metrics["mean_clean_loss"] == baseline_metrics["mean_loss"]
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name
