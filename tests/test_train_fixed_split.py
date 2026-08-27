from pathlib import Path

import pytest

from train_fixed_split import (
    load_run_configuration,
    read_split,
    selection_key,
    validate_fixed_splits,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_read_split_rejects_duplicate_ids(tmp_path: Path) -> None:
    split = tmp_path / "split.txt"
    _write(split, "a\na\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_split(split)


def test_validate_fixed_splits_rejects_overlap(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    train = root / "train.txt"
    test = root / "test.txt"
    _write(train, "shared\n")
    _write(test, "shared\n")
    config = {
        "root": str(root),
        "train_split": str(train),
        "test_split": str(test),
        "expected_train_split_sha256": "unused",
        "expected_test_split_sha256": "unused",
        "expected_train_images": 1,
        "expected_test_images": 1,
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_fixed_splits(config)


def test_resume_rejects_smoke_to_full_contract_mixing(tmp_path: Path) -> None:
    # Contract equality is intentionally exact.  This regression test captures
    # the fields that previously allowed prefix-test best values to leak into a
    # full run after resume.
    parser_namespace = type(
        "Args",
        (),
        {
            "dataset": "IRSTD-1K",
            "protocol": Path("configs/retrain_fixed_splits.yaml"),
            "root": None,
            "train_split": None,
            "test_split": None,
            "output_dir": tmp_path / "run",
            "device": "cuda:0",
            "epochs": None,
            "eval_start_epoch": None,
            "batch_size": None,
            "num_workers": 0,
            "seed": None,
            "max_train_batches": None,
            "max_test_images": None,
        },
    )()
    full = load_run_configuration(parser_namespace)
    parser_namespace.max_train_batches = 1
    parser_namespace.max_test_images = 2
    smoke = load_run_configuration(parser_namespace)
    assert full != smoke
    assert full["max_train_batches"] is None
    assert smoke["max_train_batches"] == 1


def test_best_pd_uses_fa_then_miou_as_frozen_tie_breakers() -> None:
    first = {"miou": 0.7, "pd": 1.0, "fa_per_pixel_x1e6": 8.0}
    lower_fa = {"miou": 0.6, "pd": 1.0, "fa_per_pixel_x1e6": 7.0}
    higher_miou = {"miou": 0.8, "pd": 1.0, "fa_per_pixel_x1e6": 8.0}
    assert selection_key("pd", lower_fa) > selection_key("pd", first)
    assert selection_key("pd", higher_miou) > selection_key("pd", first)
