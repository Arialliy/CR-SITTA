from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import pytest
import torch

from dataio.research_dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IRSTDResearchDataset,
    read_split_ids,
    resolve_dataset_layout,
)


def _make_dataset_root(
    tmp_path: Path,
    *,
    layout: tuple[str, str] = ("img", "label"),
    identifiers: tuple[str, ...] = ("sample",),
    split_ids: tuple[str, ...] | None = None,
) -> tuple[Path, Path]:
    root = tmp_path / "Synthetic-IRSTD"
    image_dir = root / layout[0]
    mask_dir = root / layout[1]
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for index, identifier in enumerate(identifiers):
        image = np.zeros((3, 5, 3), dtype=np.uint8)
        image[..., 0] = 32 + index
        image[..., 1] = 96 + index
        image[..., 2] = 224 - index
        mask = np.zeros((3, 5), dtype=np.uint8)
        mask[0, 0] = 255
        Image.fromarray(image, mode="RGB").save(image_dir / f"{identifier}.png")
        Image.fromarray(mask, mode="L").save(mask_dir / f"{identifier}.png")

    split_path = root / "chosen.txt"
    selected = identifiers if split_ids is None else split_ids
    split_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return root, split_path


@pytest.mark.parametrize("layout", [("img", "label"), ("images", "masks")])
def test_supported_layouts_return_standard_sample(
    tmp_path: Path, layout: tuple[str, str]
) -> None:
    root, _ = _make_dataset_root(tmp_path, layout=layout)
    dataset = IRSTDResearchDataset(
        root,
        split_file="chosen.txt",
        image_size=(6, 10),
        dataset_name="unit-dataset",
    )

    sample = dataset[0]

    assert set(sample) == {
        "image",
        "mask",
        "image_id",
        "original_size",
        "dataset",
        "corruption",
        "severity",
        "seed",
    }
    assert sample["image"].shape == (3, 6, 10)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].shape == (1, 6, 10)
    assert sample["mask"].dtype == torch.float32
    assert set(torch.unique(sample["mask"]).tolist()) <= {0.0, 1.0}
    assert sample["original_size"] == (3, 5)
    assert sample["image_id"] == "sample"
    assert sample["dataset"] == "unit-dataset"
    assert sample["corruption"] == "clean"
    assert sample["severity"] == 0
    assert sample["seed"] == 42


def test_mask_resize_is_nearest_and_remains_binary(tmp_path: Path) -> None:
    root, _ = _make_dataset_root(tmp_path)
    dataset = IRSTDResearchDataset(root, split_file="chosen.txt", image_size=(6, 10))

    mask = dataset[0]["mask"]

    # The single source pixel grows by exactly 2x along each dimension.
    assert mask.sum().item() == pytest.approx(4.0)
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}


def test_nonbinary_mask_values_are_preserved_for_official_metric_parity(
    tmp_path: Path,
) -> None:
    root, _ = _make_dataset_root(tmp_path)
    mask = np.zeros((3, 5), dtype=np.uint8)
    mask[1, 2] = 128
    Image.fromarray(mask, mode="L").save(root / "label" / "sample.png")
    dataset = IRSTDResearchDataset(root, split_file="chosen.txt", image_size=(3, 5))

    loaded = dataset[0]["mask"]

    assert loaded[0, 1, 2].item() == pytest.approx(128.0 / 255.0)


def test_corruption_receives_resized_physical_rgb_before_normalisation(
    tmp_path: Path,
) -> None:
    root, _ = _make_dataset_root(tmp_path)
    observed: dict[str, object] = {}

    def transform(
        image_01: np.ndarray,
        corruption: str,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        observed["shape"] = image_01.shape
        observed["range"] = (float(image_01.min()), float(image_01.max()))
        observed["metadata"] = (corruption, severity)
        observed["rng_type"] = type(rng)
        return np.full_like(image_01, 0.5)

    dataset = IRSTDResearchDataset(
        root,
        split_file="chosen.txt",
        image_size=(4, 7),
        corruption="gaussian_noise",
        severity=2,
        seed=17,
        corruption_transform=transform,
    )

    image = dataset[0]["image"]
    expected_channel_values = torch.from_numpy((0.5 - IMAGENET_MEAN) / IMAGENET_STD)

    assert observed["shape"] == (4, 7, 3)
    minimum, maximum = observed["range"]  # type: ignore[misc]
    assert 0.0 <= minimum <= maximum <= 1.0
    assert observed["metadata"] == ("gaussian_noise", 2)
    assert observed["rng_type"] is np.random.Generator
    assert torch.allclose(image[:, 0, 0], expected_channel_values)
    assert torch.allclose(image, expected_channel_values[:, None, None].expand_as(image))


def test_injected_rng_is_repeatable_per_protocol_tuple(tmp_path: Path) -> None:
    root, _ = _make_dataset_root(tmp_path)

    def stochastic_transform(
        image_01: np.ndarray,
        corruption: str,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return np.full_like(image_01, rng.uniform(0.1, 0.9))

    common = dict(
        dataset_root=root,
        split_file="chosen.txt",
        image_size=4,
        corruption="stripe_noise",
        severity=3,
        corruption_transform=stochastic_transform,
    )
    first = IRSTDResearchDataset(**common, seed=123)
    same = IRSTDResearchDataset(**common, seed=123)
    different = IRSTDResearchDataset(**common, seed=124)

    assert torch.equal(first[0]["image"], first[0]["image"])
    assert torch.equal(first[0]["image"], same[0]["image"])
    assert not torch.equal(first[0]["image"], different[0]["image"])


def test_non_clean_metadata_cannot_silently_skip_transform(tmp_path: Path) -> None:
    root, _ = _make_dataset_root(tmp_path)
    with pytest.raises(ValueError, match="requires corruption_transform"):
        IRSTDResearchDataset(
            root,
            split_file="chosen.txt",
            image_size=4,
            corruption="gaussian_noise",
            severity=1,
        )


@pytest.mark.parametrize(
    ("transform", "message"),
    [
        (lambda image, *_: image + 2.0, r"\[0, 1\]"),
        (lambda image, *_: image[..., 0], "shape"),
        (lambda image, *_: np.full_like(image, np.nan), "NaN or Inf"),
    ],
)
def test_invalid_corruption_outputs_fail_loudly(
    tmp_path: Path,
    transform: Callable[..., np.ndarray],
    message: str,
) -> None:
    root, _ = _make_dataset_root(tmp_path)
    dataset = IRSTDResearchDataset(
        root,
        split_file="chosen.txt",
        image_size=4,
        corruption="gaussian_blur",
        severity=1,
        corruption_transform=transform,
    )

    with pytest.raises((TypeError, ValueError), match=message):
        _ = dataset[0]


def test_duplicate_and_alias_ids_are_rejected(tmp_path: Path) -> None:
    duplicate_root, duplicate_split = _make_dataset_root(
        tmp_path / "duplicate",
        split_ids=("sample", "sample"),
    )
    with pytest.raises(ValueError, match="Duplicate image ID"):
        IRSTDResearchDataset(
            duplicate_root, split_file=duplicate_split, image_size=4
        )

    alias_root, alias_split = _make_dataset_root(
        tmp_path / "alias",
        split_ids=("sample", "sample.png"),
    )
    with pytest.raises(ValueError, match="same image file"):
        IRSTDResearchDataset(alias_root, split_file=alias_split, image_size=4)


def test_missing_image_or_mask_is_reported_during_initialisation(tmp_path: Path) -> None:
    missing_image_root, split_path = _make_dataset_root(
        tmp_path / "missing-image", split_ids=("absent",)
    )
    with pytest.raises(FileNotFoundError, match="Missing image"):
        IRSTDResearchDataset(
            missing_image_root, split_file=split_path, image_size=4
        )

    missing_mask_root, split_path = _make_dataset_root(tmp_path / "missing-mask")
    (missing_mask_root / "label" / "sample.png").unlink()
    with pytest.raises(FileNotFoundError, match="Missing mask"):
        IRSTDResearchDataset(missing_mask_root, split_file=split_path, image_size=4)


def test_image_and_mask_are_independently_resized_like_official_loader(
    tmp_path: Path,
) -> None:
    root, _ = _make_dataset_root(tmp_path)
    Image.fromarray(np.zeros((4, 5), dtype=np.uint8), mode="L").save(
        root / "label" / "sample.png"
    )
    dataset = IRSTDResearchDataset(root, split_file="chosen.txt", image_size=4)

    sample = dataset[0]
    assert sample["image"].shape == (3, 4, 4)
    assert sample["mask"].shape == (1, 4, 4)
    assert sample["original_size"] == (3, 5)


def test_layout_must_be_complete_and_unambiguous(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    (incomplete / "img").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="img/label"):
        resolve_dataset_layout(incomplete)

    ambiguous = tmp_path / "ambiguous"
    for directory in ("img", "label", "images", "masks"):
        (ambiguous / directory).mkdir(parents=True)
    with pytest.raises(ValueError, match="both supported layouts"):
        resolve_dataset_layout(ambiguous)


@pytest.mark.parametrize(
    ("corruption", "severity", "seed"),
    [
        ("clean", 1, 0),
        ("unknown", 1, 0),
        ("gaussian_noise", 0, 0),
        ("gaussian_noise", 6, 0),
        ("gaussian_noise", 1, -1),
    ],
)
def test_invalid_protocol_ranges_are_rejected(
    tmp_path: Path, corruption: str, severity: int, seed: int
) -> None:
    root, _ = _make_dataset_root(tmp_path)
    with pytest.raises(ValueError):
        IRSTDResearchDataset(
            root,
            split_file="chosen.txt",
            image_size=4,
            corruption=corruption,
            severity=severity,
            seed=seed,
        )


@pytest.mark.parametrize(
    ("relative_path", "expected_count"),
    [
        ("datasets/IRSTD-1K/img_idx/train_IRSTD-1K.txt", 800),
        ("datasets/IRSTD-1K/img_idx/test_IRSTD-1K.txt", 201),
        ("datasets/NUAA-SIRST/img_idx/train_NUAA-SIRST.txt", 213),
        ("datasets/NUAA-SIRST/img_idx/test_NUAA-SIRST.txt", 214),
        ("datasets/NUDT-SIRST/img_idx/train_NUDT-SIRST.txt", 663),
        ("datasets/NUDT-SIRST/img_idx/test_NUDT-SIRST.txt", 664),
    ],
)
def test_checked_in_official_split_coverage_is_unique_and_read_only(
    relative_path: str, expected_count: int
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    split_path = repository_root / relative_path
    contents_before = split_path.read_bytes()

    identifiers = read_split_ids(split_path)

    assert len(identifiers) == expected_count
    assert len(set(identifiers)) == expected_count
    assert split_path.read_bytes() == contents_before
