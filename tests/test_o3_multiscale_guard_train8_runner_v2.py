"""Synthetic helpers only; no real feature/checkpoint/test payload access."""
import copy
import json

import numpy as np
import pytest
import torch

from scripts.run_o3_multiscale_train8_v1 import training_order
from scripts.run_o3_multiscale_guard_train8_v2 import (
    LOSS_KEYS, checked_array, checked_order, loss_values, require_exact_probability, write_json,
)


def test_exact_replay_accepts_identical_and_rejects_single_ulp():
    saved = np.full((1, 2, 3), .5, dtype=np.float32)
    require_exact_probability(saved.copy(), saved, "fixture")
    changed = saved.copy()
    changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.float32(1.))
    with pytest.raises(RuntimeError, match="bit-exact"):
        require_exact_probability(changed, saved, "fixture")


@pytest.mark.parametrize("kind", ["dtype", "shape", "nan"])
def test_replay_rejects_bad_input(kind):
    saved = np.zeros((1, 2, 3), dtype=np.float32)
    actual = saved.copy()
    if kind == "dtype": actual = actual.astype(np.float64)
    elif kind == "shape": actual = actual.reshape(1, 3, 2)
    else: actual[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        require_exact_probability(actual, saved, "fixture")


def test_saved_training_order_exact():
    order = {"sample_order": "condition_major_then_image_id", "indices": training_order(104)}
    assert checked_order(order) == training_order(104)
    changed = copy.deepcopy(order)
    changed["indices"][0][0] = (changed["indices"][0][0] + 1) % 104
    with pytest.raises(ValueError):
        checked_order(changed)


@pytest.mark.parametrize("record", [None, {}, {"sample_order": "other", "indices": []},
                                    {"sample_order": "condition_major_then_image_id", "indices": []}])
def test_invalid_order(record):
    with pytest.raises(ValueError):
        checked_order(record)


def test_loss_values_detached_and_three_terms_preserved():
    inputs = {key: torch.tensor(float(index), requires_grad=True) for index, key in enumerate(LOSS_KEYS)}
    actual = loss_values(inputs)
    assert actual == dict(zip(LOSS_KEYS, (0., 1., 2.)))
    assert all(value.grad is None for value in inputs.values())


@pytest.mark.parametrize("kind", ["missing", "extra", "nan", "negative"])
def test_loss_terms_reject_malformed(kind):
    values = {key: torch.tensor(0.) for key in LOSS_KEYS}
    if kind == "missing": values.pop(LOSS_KEYS[0])
    elif kind == "extra": values["not_declared"] = torch.tensor(0.)
    elif kind == "nan": values[LOSS_KEYS[0]] = torch.tensor(float("nan"))
    else: values[LOSS_KEYS[0]] = torch.tensor(-1.)
    with pytest.raises(ValueError):
        loss_values(values)


def test_array_read_is_readonly_and_shape_bound(tmp_path):
    path = tmp_path / "fixture.npy"
    expected = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    np.save(path, expected, allow_pickle=False)
    actual = checked_array(path, (2, 2, 3))
    assert np.array_equal(actual, expected) and not actual.flags.writeable
    with pytest.raises(ValueError):
        checked_array(path, (2, 3, 2))


@pytest.mark.parametrize("kind", ["float64", "nan", "object"])
def test_invalid_array(tmp_path, kind):
    path = tmp_path / "fixture.npy"
    if kind == "float64": value = np.zeros(2, dtype=np.float64)
    elif kind == "nan": value = np.full(2, np.nan, dtype=np.float32)
    else: value = np.array([{"not": "allowed"}], dtype=object)
    np.save(path, value)
    with pytest.raises(ValueError):
        checked_array(path, (2,))


def test_json_never_overwrites(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}
