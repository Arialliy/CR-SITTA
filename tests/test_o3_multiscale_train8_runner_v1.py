"""Synthetic tests only: no dataset, pretrained checkpoint or real metric access."""
import copy
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from model.o3_multiscale_residual_v1 import O3MultiScaleResidual
from scripts.run_o3_multiscale_train8_v1 import (
    performance_signal, supervised_loss, training_order, write_json,
)


def test_loss_definition_and_gradient():
    z = torch.tensor([[[[0., 1.], [-1., 2.]]]], requires_grad=True)
    y = torch.tensor([[[[0., 1.], [0., 1.]]]])
    p = z.sigmoid()
    expected = F.binary_cross_entropy_with_logits(z, y) + 1 - ((p*y).sum()+1)/((p+y-p*y).sum()+1)
    loss = supervised_loss(z, y)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.norm() > 0


@pytest.mark.parametrize("positive", [False, True])
def test_empty_or_full_target_finite(positive):
    z = torch.zeros(2, 1, 3, 5, requires_grad=True)
    y = torch.full_like(z, float(positive))
    loss = supervised_loss(z, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


@pytest.mark.parametrize("invalid", ["soft", "nan", "grad", "shape", "dtype", "empty"])
def test_invalid_targets_rejected(invalid):
    z, y = torch.zeros(2, 1, 3, 5), torch.zeros(2, 1, 3, 5)
    if invalid == "soft": y[0, 0, 0, 0] = 0.5
    elif invalid == "nan": y[0, 0, 0, 0] = float("nan")
    elif invalid == "grad": y.requires_grad_(True)
    elif invalid == "shape": y = y[:1]
    elif invalid == "dtype": y = y.double()
    elif invalid == "empty": z, y = z[:0], y[:0]
    with pytest.raises(ValueError):
        supervised_loss(z, y)


def test_seeded_schedule_is_complete_repeatable_without_global_rng_mutation():
    rng_before = torch.get_rng_state().clone()
    order = training_order(104)
    assert len(order) == 128 and all(len(batch) == 4 for batch in order)
    flat = sum(order, [])
    for offset in range(0, len(flat)-103, 104):
        assert sorted(flat[offset:offset+104]) == list(range(104))
    assert order == training_order(104)
    assert order != training_order(104, seed=43)
    assert torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize("kwargs", [{"count": 0}, {"count": 1, "steps": -1},
                                    {"count": 1, "batch_size": False}, {"count": 1, "seed": -1}])
def test_invalid_schedule(kwargs):
    with pytest.raises(ValueError):
        training_order(**kwargs)


def cells():
    metric = {"iou": .5, "normalized_iou": .49, "pd": .8, "fa_per_million": 3.}
    result = []
    for index in range(13):
        result.append({"condition": str(index), "corruption": "clean" if index == 0 else "noise",
                       "source": dict(metric), "o3": dict(metric), "trained": dict(metric)})
    return result


def test_fit_loss_decrease_does_not_imply_performance_signal():
    result = performance_signal(cells(), 1., .9)
    assert result["learning_check_passed"]
    assert not result["fit_performance_signal_passed"]
    assert result["failed_goals"] == ["nonclean_iou_above_o3"]
    assert not result["automatic_full_training_allowed"]
    assert not result["generalization_claim"]


def test_clean_not_pooled_and_fa_increase_fails():
    rows = cells()
    for row in rows[1:]: row["trained"]["iou"] += .01
    rows[0]["trained"]["fa_per_million"] += .01
    result = performance_signal(rows, 1., .9)
    assert result["nonclean_macro"]["trained"]["iou"] == pytest.approx(.51)
    assert not result["fit_performance_signal_passed"]
    assert result["failed_goals"] == ["clean_fa_not_above_o3"]


def test_fit_positive_still_not_generalization_or_automatic_training():
    rows = cells()
    for row in rows[1:]: row["trained"]["iou"] += .01
    result = performance_signal(rows, 1., .9)
    assert result["fit_performance_signal_passed"] and result["learning_check_passed"]
    assert not result["automatic_full_training_allowed"] and not result["generalization_claim"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra_clean"])
def test_incomplete_metric_grid_rejected(mutation):
    rows = cells()
    if mutation == "missing": rows.pop()
    elif mutation == "duplicate": rows[1]["condition"] = rows[0]["condition"]
    else: rows[1]["corruption"] = "clean"
    with pytest.raises(ValueError):
        performance_signal(rows, 1., .9)


def test_multistep_source_fit_keeps_frozen_head_and_features():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        branch = O3MultiScaleResidual()
        head = torch.nn.Conv2d(16, 1, 1).eval().requires_grad_(False)
        head_before = copy.deepcopy(head.state_dict())
        h = torch.randn(2, 16, 9, 11)
        h_before = h.clone()
        y = (torch.rand(2, 1, 9, 11) > .8).float()
        before = float(supervised_loss(head(branch(h)), y))
        optimizer = torch.optim.Adam(branch.parameters(), lr=.001)
        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            supervised_loss(head(branch(h)), y).backward()
            optimizer.step()
        after = float(supervised_loss(head(branch(h)), y))
        assert after < before
        assert torch.equal(h, h_before) and h.grad is None
        assert all(p.grad is None for p in head.parameters())
        assert all(torch.equal(head_before[k], v) for k, v in head.state_dict().items())


def test_json_output_never_overwrites(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}
