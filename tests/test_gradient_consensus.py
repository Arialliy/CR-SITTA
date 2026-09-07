from __future__ import annotations

import inspect

import pytest
import torch

from tta.proposals.gradient_consensus import (
    GradientConsensusError,
    combine_two_gradients,
)


def test_aligned_gradients_combine_unit_directions_and_detach() -> None:
    first = torch.tensor([3.0, 0.0], requires_grad=True)
    second = torch.tensor([4.0, 4.0], requires_grad=True)

    output = combine_two_gradients(first, second, min_norm=1.0e-8)

    expected = first.detach() / first.detach().norm() + second.detach() / second.detach().norm()
    assert output.decision == "aligned_average"
    assert output.cosine == pytest.approx(2.0**-0.5)
    assert output.first_norm == pytest.approx(3.0)
    assert output.second_norm == pytest.approx(32.0**0.5)
    assert output.usable_branches == ("first", "second")
    assert output.trial_order == ("consensus",)
    assert output.gradient is not None
    assert torch.allclose(output.gradient, expected)
    assert not output.gradient.requires_grad
    assert output.gradient.grad_fn is None
    assert output.gradient.data_ptr() != first.data_ptr()
    assert output.gradient.data_ptr() != second.data_ptr()


def test_tiny_branches_are_handled_without_amplification() -> None:
    tiny = torch.tensor([1.0e-10, 0.0])
    intact = torch.tensor([2.0, -3.0])

    use_second = combine_two_gradients(tiny, intact, min_norm=1.0e-6)
    assert use_second.decision == "use_second_only"
    assert use_second.cosine is None
    assert use_second.gradient is not None
    assert torch.equal(use_second.gradient, intact)
    assert use_second.gradient.data_ptr() != intact.data_ptr()

    use_first = combine_two_gradients(intact, tiny, min_norm=1.0e-6)
    assert use_first.decision == "use_first_only"
    assert use_first.gradient is not None
    assert torch.equal(use_first.gradient, intact)

    neither = combine_two_gradients(tiny, -tiny, min_norm=1.0e-6)
    assert neither.decision == "reject_both_tiny"
    assert neither.gradient is None
    assert neither.trial_order == ()


def test_nonfinite_branch_is_dropped_while_finite_branch_is_retained() -> None:
    intact = torch.tensor([1.0, 2.0])
    output = combine_two_gradients(
        torch.tensor([1.0, float("nan")]),
        intact,
        min_norm=1.0e-8,
    )
    assert output.decision == "use_second_only_first_nonfinite"
    assert output.gradient is not None
    assert torch.equal(output.gradient, intact)
    assert output.cosine is None
    assert output.first_norm is None
    assert output.usable_branches == ("second",)
    assert output.trial_order == ("second",)

    symmetric = combine_two_gradients(
        intact,
        torch.tensor([float("inf"), 2.0]),
        min_norm=1.0e-8,
    )
    assert symmetric.decision == "use_first_only_second_nonfinite"
    assert symmetric.gradient is not None
    assert torch.equal(symmetric.gradient, intact)

    both_bad = combine_two_gradients(
        torch.tensor([float("nan"), 0.0]),
        torch.tensor([float("inf"), 0.0]),
        min_norm=1.0e-8,
    )
    assert both_bad.decision == "reject_both_nonfinite"
    assert both_bad.gradient is None


def test_conflict_requires_two_separate_source_anchored_trials() -> None:
    first = torch.tensor([1.0, 0.0])
    second = torch.tensor([-1.0, 0.1])

    output = combine_two_gradients(first, second, min_norm=1.0e-8)

    assert output.decision == "conflict_try_separately"
    assert output.cosine is not None and output.cosine < 0.0
    assert output.gradient is None
    assert output.usable_branches == ("first", "second")
    assert output.trial_order == ("first", "second")


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        (torch.zeros(1, 2), torch.zeros(1, 2), "flattened 1D"),
        (torch.zeros(2), torch.zeros(3), "identical 1D shape"),
        (torch.zeros(2), torch.zeros(2, dtype=torch.float64), "same dtype"),
        (torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64), "floating-point"),
    ],
)
def test_structural_gradient_errors_fail_closed(
    first: torch.Tensor,
    second: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, GradientConsensusError), match=message):
        combine_two_gradients(first, second, min_norm=1.0e-8)


@pytest.mark.parametrize(("name", "value"), [("min_norm", 0.0), ("eps", float("inf"))])
def test_scalar_contract_is_strict(name: str, value: float) -> None:
    kwargs = {"min_norm": 1.0e-8, "eps": 1.0e-12}
    kwargs[name] = value
    with pytest.raises((TypeError, GradientConsensusError)):
        combine_two_gradients(torch.ones(2), torch.ones(2), **kwargs)


def test_consensus_signature_has_no_label_or_condition_channel() -> None:
    names = set(inspect.signature(combine_two_gradients).parameters)
    forbidden = {
        "ground_truth",
        "gt",
        "label",
        "mask",
        "outer_mask",
        "target",
        "condition",
        "corruption",
        "severity",
    }
    assert names.isdisjoint(forbidden)
