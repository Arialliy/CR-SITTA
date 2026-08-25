from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

import run_corruption_pilot as pilot


class FakeNSFPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, image: torch.Tensor, warm_flag: bool):
        return ([image] if warm_flag else []), self.head(image)


def _make_source_dataset(tmp_path: Path) -> tuple[Path, Path, tuple[str, ...]]:
    root = tmp_path / "source-data"
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    image_ids = ("source_c", "source_a", "source_d", "source_b")
    for index, image_id in enumerate(image_ids):
        row_gradient = np.linspace(30, 190, 9, dtype=np.uint8)[None, :, None]
        image = np.broadcast_to(row_gradient, (7, 9, 3)).copy()
        image = np.clip(image.astype(np.int16) + 5 * index, 0, 255).astype(np.uint8)
        mask = np.zeros((7, 9), dtype=np.uint8)
        mask[3, 3 + (index % 2)] = 255
        Image.fromarray(image, mode="RGB").save(images / f"{image_id}.png")
        Image.fromarray(mask, mode="L").save(masks / f"{image_id}.png")
    split = tmp_path / "source_trainval.txt"
    split.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    return root, split, image_ids


def _pilot_args(
    root: Path,
    split: Path,
    checkpoint: Path,
    output_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="IRSTD-1k",
        root=root,
        split=split,
        checkpoint=checkpoint,
        device="cpu",
        subset_size=2,
        image_size=8,
        seed=42,
        output_dir=output_dir,
    )


def test_cli_defaults_to_bounded_official_trainval_protocol() -> None:
    parser = pilot.build_argument_parser()
    args = parser.parse_args(
        ["--dataset", "IRSTD-1k", "--root", "/nonexistent-for-parse-only"]
    )

    assert args.split is None
    assert args.subset_size == 64
    assert pilot.PILOT_DATASET_DEFAULTS["IRSTD-1k"]["split"].name == "trainval.txt"
    assert all(
        defaults["split"].name == "trainval.txt"
        for defaults in pilot.PILOT_DATASET_DEFAULTS.values()
    )
    assert len(pilot.DEFAULT_CONDITIONS) == 21


def test_sha256_subset_is_exact_and_independent_of_input_order() -> None:
    image_ids = ("gamma", "alpha", "beta", "delta")
    expected = sorted(
        image_ids,
        key=lambda image_id: (hashlib.sha256(image_id.encode()).hexdigest(), image_id),
    )[:3]

    first = pilot.sha256_ranked_subset(image_ids, 3)
    second = pilot.sha256_ranked_subset(tuple(reversed(image_ids)), 3)

    assert [item.image_id for item in first] == expected
    assert first == second
    assert all(
        item.image_id_sha256 == hashlib.sha256(item.image_id.encode()).hexdigest()
        for item in first
    )


def test_condition_contract_requires_one_leading_clean_and_no_duplicates() -> None:
    assert pilot.normalise_conditions(
        (("clean", 0), ("gaussian_noise", 1))
    ) == (("clean", 0), ("gaussian_noise", 1))
    with pytest.raises(ValueError, match="clean/0 exactly once and first"):
        pilot.normalise_conditions((("gaussian_noise", 1),))
    with pytest.raises(ValueError, match="unique"):
        pilot.normalise_conditions((("clean", 0), ("clean", 0)))


@pytest.mark.parametrize(
    ("values", "classification"),
    [
        ([3.0, 2.0, 2.0, 1.0], "non_increasing"),
        ([1.0, 2.0, 2.0, 3.0], "non_decreasing"),
        ([1.0, 1.0], "constant"),
        ([1.0, 3.0, 2.0], "mixed"),
    ],
)
def test_monotonic_summary_reports_without_enforcing(
    values: list[float], classification: str
) -> None:
    summary = pilot.monotonic_summary(values)

    assert summary["classification"] == classification
    assert len(summary["adjacent_deltas"]) == len(values) - 1


def test_runner_refuses_any_test_txt_even_when_explicit(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    split = tmp_path / "test.txt"
    split.write_text("sample\n", encoding="utf-8")
    checkpoint = tmp_path / "weight.pkl"
    checkpoint.write_bytes(b"placeholder")
    args = _pilot_args(root, split, checkpoint, tmp_path / "out")

    with pytest.raises(ValueError, match="refuses test.txt"):
        pilot.resolve_pilot_paths(args)


def test_official_test_ids_are_detected_even_if_split_was_renamed() -> None:
    known_test_id = pilot.read_split_ids(
        pilot.PILOT_DATASET_DEFAULTS["IRSTD-1k"]["test_split"]
    )[0]

    assert pilot.official_test_id_overlap(
        "IRSTD-1k", ("safe_custom_source_id", known_test_id)
    ) == (known_test_id,)
    assert pilot.official_test_id_overlap(
        "IRSTD-1k", ("safe_custom_source_id",)
    ) == ()


def test_fake_model_bounded_pilot_writes_metrics_hashes_trends_and_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, split, image_ids = _make_source_dataset(tmp_path)
    checkpoint = tmp_path / "fake.pkl"
    torch.manual_seed(7)
    torch.save(FakeNSFPN().state_dict(), checkpoint)
    output_dir = tmp_path / "pilot-output"
    args = _pilot_args(root, split, checkpoint, output_dir)

    build_calls = 0

    def build_fake_model() -> FakeNSFPN:
        nonlocal build_calls
        build_calls += 1
        return FakeNSFPN()

    real_apply_corruption = pilot.apply_corruption
    observed_physical_ranges: list[tuple[float, float, str, int]] = []

    def observed_corruption(
        image_01: np.ndarray,
        corruption: str,
        severity: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        observed_physical_ranges.append(
            (float(image_01.min()), float(image_01.max()), corruption, severity)
        )
        return real_apply_corruption(image_01, corruption, severity, rng)

    monkeypatch.setattr(pilot.source_runner, "build_nsfpn_model", build_fake_model)
    monkeypatch.setattr(pilot, "apply_corruption", observed_corruption)
    conditions = (
        ("clean", 0),
        *(("gaussian_noise", severity) for severity in range(1, 6)),
    )
    severity_path = pilot.get_default_severity_table().source_path
    severity_before = severity_path.read_bytes()

    result = pilot.run_corruption_pilot(args, conditions=conditions)

    assert build_calls == 1
    assert result["scope"] == "source_domain_only"
    assert len(result["repository_provenance"]["head_commit"]) == 40
    assert "run_corruption_pilot.py" in result["repository_provenance"]["file_sha256"]
    assert result["split_role"] == "explicit_source_holdout_without_official_test_ids"
    assert result["condition_count"] == 6
    assert result["full_clean_plus_4x5_protocol"] is False
    expected_selected = pilot.sha256_ranked_subset(image_ids, 2)
    assert result["selection"]["selected_ids"] == [
        item.image_id for item in expected_selected
    ]
    assert result["selection"]["selected"] == [
        {
            "rank": rank,
            "image_id": item.image_id,
            "image_id_sha256": item.image_id_sha256,
        }
        for rank, item in enumerate(expected_selected)
    ]
    selected_digest = result["selection"]["ordered_selected_ids_sha256"]
    assert len(selected_digest) == 64
    assert all(
        condition["evaluated_ids_sha256"] == selected_digest
        for condition in result["conditions"]
    )
    assert all(
        condition["ids_identical_to_selected"] for condition in result["conditions"]
    )
    assert all(condition["evaluated_images"] == 2 for condition in result["conditions"])
    assert all(
        set(condition["metrics"]["fixed"]) >= {
            "iou",
            "pd",
            "fa_pixel_rate",
            "fa_per_million_pixels",
        }
        for condition in result["conditions"]
    )
    assert all(len(condition["metrics"]["froc"]) == 21 for condition in result["conditions"])
    assert result["checks"]["model_state_unchanged"] is True
    assert result["checks"]["same_ordered_ids_all_conditions"] is True
    assert result["checks"]["official_test_ids_absent"] is True
    assert result["checks"]["official_test_id_overlap_count"] == 0
    assert result["checks"]["severity_table_unchanged"] is True
    assert result["checks"]["severity_table_automatically_frozen"] is False
    assert set(result["trends"]) == {"gaussian_noise"}
    assert result["trends"]["gaussian_noise"]["pilot_only_no_automatic_freeze"] is True
    assert result["runtime"]["total_seconds"] > 0.0
    assert result["runtime"]["model_load_seconds"] >= 0.0

    assert observed_physical_ranges
    assert all(
        0.0 <= minimum <= maximum <= 1.0
        for minimum, maximum, _, _ in observed_physical_ranges
    )
    assert {
        (corruption, severity)
        for _, _, corruption, severity in observed_physical_ranges
    } == set(conditions)

    assert len(result["visualizations"]) == 1
    visualization = result["visualizations"][0]
    assert visualization["panels"] == ["clean", "S1", "S2", "S3", "S4", "S5"]
    grid_path = output_dir / visualization["path"]
    assert grid_path.is_file()
    assert len(visualization["sha256"]) == 64
    with Image.open(grid_path) as grid:
        assert grid.width > grid.height

    artifact_path = output_dir / "pilot.json"
    assert artifact_path.is_file()
    written = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert written["selection"]["selected_ids"] == result["selection"]["selected_ids"]
    assert written["conditions"][0]["corruption"] == "clean"
    assert severity_path.read_bytes() == severity_before
