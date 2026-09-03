from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "verify_checkpoint_axis_v2_parity.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_axis_v2_parity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, values: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _clean_fixture(root: Path, probability: np.ndarray) -> None:
    directory = root / "D"
    mask = directory / "prediction_masks_256" / "a.png"
    prob = directory / "probability_maps_256" / "a.npy"
    mask.parent.mkdir(parents=True)
    prob.parent.mkdir(parents=True)
    mask.write_bytes(b"same-png")
    np.save(prob, probability, allow_pickle=False)
    record = {
        "index": 0,
        "image_id": "a",
        "prediction_mask": "prediction_masks_256/a.png",
        "probability_map": "probability_maps_256/a.npy",
        "pixel_metrics_at_probability_gt_0_5": {"tp": 1, "fp": 0},
    }
    _jsonl(directory / "per_image.jsonl", [record])


def test_clean_record_comparison_is_exact(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    values = np.array([[0.25, 0.75]], dtype=np.float32)
    _clean_fixture(old, values)
    _clean_fixture(new, values)

    result = parity._compare_records(
        old / "D", new / "D", probability_mode="per_image", label="fixture"
    )

    assert result == {
        "images": 1,
        "mask_files": 1,
        "probability_values": 2,
        "integer_records": 1,
    }


def test_probability_one_ulp_drift_is_rejected(tmp_path: Path) -> None:
    old = tmp_path / "old.npy"
    new = tmp_path / "new.npy"
    value = np.array([0.5], dtype=np.float32)
    drift = np.nextafter(value, np.array([1.0], dtype=np.float32))
    np.save(old, value, allow_pickle=False)
    np.save(new, drift, allow_pickle=False)

    with pytest.raises(parity.ParityError, match="not bit-exact"):
        parity._compare_array_files(old, new, "fixture")


def test_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    parity._write_json(output, {"passed": True})
    with pytest.raises(FileExistsError):
        parity._write_json(output, {"passed": False})


def test_safe_child_rejects_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(parity.ParityError, match="unsafe"):
        parity._safe_child(tmp_path, "../outside.npy", "probability")
