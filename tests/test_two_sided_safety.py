from __future__ import annotations

import inspect

import pytest
import torch

from tta.proposals.two_sided_safety import (
    TwoSidedSafetyError,
    check_two_sided_safety,
)


def _case() -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    source = torch.full((1, 1, 5, 5), -4.0)
    core = torch.zeros_like(source, dtype=torch.bool)
    core[..., 2, 2] = True
    ring = torch.zeros_like(core)
    ring[..., 1:4, 1:4] = True
    ring &= ~core
    source[core] = 4.0
    source[ring] = -2.0
    background = torch.ones_like(source)
    background[core | ring] = 0.0
    return source, background, (core,), (ring,)


def _check(source: torch.Tensor, proposed: torch.Tensor, **overrides: object):
    _, background, cores, rings = _case()
    kwargs: dict[str, object] = {
        "background_weight": background,
        "candidate_cores": cores,
        "candidate_rings": rings,
        "max_background_mass_increase": 0.01,
        "max_absolute_drop": 0.25,
        "max_contrast_drop": 0.25,
        "temperature": 0.5,
    }
    kwargs.update(overrides)
    return check_two_sided_safety(source, proposed, **kwargs)


def test_unchanged_proposal_passes_and_returns_detached_audit() -> None:
    source, _, _, _ = _case()
    source = source.requires_grad_()
    proposed = source.detach().clone().requires_grad_()

    output = _check(source, proposed)

    assert output.passed
    assert output.reasons == ()
    assert output.reason == "passed"
    assert output.background_passed
    assert output.background_mass_increase == pytest.approx(0.0)
    assert output.positive_fraction_pre == output.positive_fraction_post
    assert output.component_count_pre == output.component_count_post == 1
    assert output.largest_component_fraction_pre == pytest.approx(1.0 / 25.0)
    assert output.largest_component_fraction_post == pytest.approx(1.0 / 25.0)
    assert output.structure_metrics_are_audit_only is True
    assert output.quantile_levels == (0.5, 0.9, 0.99)
    assert len(output.reliable_background_quantiles_pre) == 3
    assert output.candidates[0].absolute_drop == pytest.approx(0.0)
    assert output.candidates[0].contrast_drop == pytest.approx(0.0)
    assert isinstance(output.candidates[0].absolute_response_pre, float)
    assert source.grad is None and proposed.grad is None


def test_background_inflation_is_rejected() -> None:
    source, background, _, _ = _case()
    proposed = source.clone()
    proposed[background.bool()] = 0.0

    output = _check(source, proposed)

    assert not output.passed
    assert "background_mass_inflation" in output.reasons
    assert output.background_mass_increase is not None
    assert output.background_mass_increase > output.max_background_mass_increase


def test_candidate_absolute_response_drop_is_rejected() -> None:
    source, _, cores, _ = _case()
    proposed = source.clone()
    proposed[cores[0]] = 2.0

    output = _check(source, proposed, max_contrast_drop=10.0)

    assert not output.passed
    assert "candidate_0_absolute_drop" in output.reasons
    assert "candidate_0_contrast_drop" not in output.reasons
    assert output.candidates[0].absolute_drop == pytest.approx(2.0)
    assert not output.candidates[0].absolute_passed


def test_candidate_local_contrast_drop_is_rejected_independently() -> None:
    source, _, _, rings = _case()
    proposed = source.clone()
    proposed[rings[0]] = 1.0

    output = _check(source, proposed, max_absolute_drop=10.0)

    assert not output.passed
    assert "candidate_0_absolute_drop" not in output.reasons
    assert "candidate_0_contrast_drop" in output.reasons
    assert output.candidates[0].contrast_drop == pytest.approx(3.0)
    assert not output.candidates[0].contrast_passed


def test_empty_candidate_tuple_runs_background_safety_only() -> None:
    source, background, _, _ = _case()
    output = check_two_sided_safety(
        source,
        source.clone(),
        background_weight=background,
        candidate_cores=(),
        candidate_rings=(),
        max_background_mass_increase=0.0,
        max_absolute_drop=0.0,
        max_contrast_drop=0.0,
    )
    assert output.passed
    assert output.candidates == ()
    assert output.background_passed
    assert output.temperature == pytest.approx(0.25)


def test_predicted_structure_changes_are_reported_but_not_unregistered_gates() -> None:
    source, background, cores, rings = _case()
    proposed = source.clone()
    proposed[..., 0, 0:2] = 4.0
    output = check_two_sided_safety(
        source,
        proposed,
        background_weight=background,
        candidate_cores=cores,
        candidate_rings=rings,
        max_background_mass_increase=1.0,
        max_absolute_drop=0.0,
        max_contrast_drop=0.0,
    )

    assert output.passed
    assert output.component_count_pre == 1
    assert output.component_count_post == 2
    assert output.largest_component_fraction_pre == pytest.approx(1.0 / 25.0)
    assert output.largest_component_fraction_post == pytest.approx(2.0 / 25.0)
    assert output.structure_metrics_are_audit_only is True


def test_empty_ring_is_an_explicit_failed_decision() -> None:
    source, background, cores, _ = _case()
    empty_ring = torch.zeros_like(cores[0])
    output = check_two_sided_safety(
        source,
        source.clone(),
        background_weight=background,
        candidate_cores=cores,
        candidate_rings=(empty_ring,),
        max_background_mass_increase=0.0,
        max_absolute_drop=0.0,
        max_contrast_drop=0.0,
    )
    assert not output.passed
    assert output.reasons == ("candidate_0_empty_ring",)
    assert output.candidates[0].ring_pixel_count == 0
    assert output.candidates[0].absolute_response_pre is None


def test_backtracking_can_turn_rejection_into_acceptance() -> None:
    source, background, _, _ = _case()
    full_step = source.clone()
    full_step[background.bool()] += 4.0
    small_step = source.clone()
    small_step[background.bool()] += 0.1

    rejected = _check(source, full_step, max_background_mass_increase=0.001)
    accepted = _check(source, small_step, max_background_mass_increase=0.01)

    assert not rejected.passed
    assert "background_mass_inflation" in rejected.reasons
    assert accepted.passed


def test_all_safety_failures_are_no_update_and_input_immutable() -> None:
    source, background, cores, rings = _case()
    source_before = source.clone()
    proposed = source.clone()
    proposed[background.bool()] = 2.0
    proposed[cores[0]] = -2.0
    proposed[rings[0]] = 2.0
    proposed_before = proposed.clone()

    output = _check(source, proposed)

    assert not output.passed
    assert {
        "background_mass_inflation",
        "candidate_0_absolute_drop",
        "candidate_0_contrast_drop",
    }.issubset(output.reasons)
    assert torch.equal(source, source_before)
    assert torch.equal(proposed, proposed_before)


def test_temperature_scaled_lse_is_used_for_candidate_response() -> None:
    source, background, cores, rings = _case()
    second_core_pixel = cores[0].clone()
    second_core_pixel[..., 2, 3] = True
    adjusted_ring = rings[0] & ~second_core_pixel
    source[second_core_pixel] = 0.0
    source[..., 2, 2] = 2.0
    temperature = 0.5

    output = check_two_sided_safety(
        source,
        source.clone(),
        background_weight=background,
        candidate_cores=(second_core_pixel,),
        candidate_rings=(adjusted_ring,),
        max_background_mass_increase=0.0,
        max_absolute_drop=0.0,
        max_contrast_drop=0.0,
        temperature=temperature,
    )
    expected = temperature * torch.logsumexp(
        source[second_core_pixel].double() / temperature, dim=0
    )
    assert output.passed
    assert output.candidates[0].absolute_response_pre == pytest.approx(
        expected.item()
    )


def test_nonfinite_proposal_and_empty_background_fail_closed() -> None:
    source, background, cores, rings = _case()
    proposed = source.clone()
    proposed[..., 0, 0] = float("nan")
    nonfinite = _check(source, proposed)
    assert not nonfinite.passed
    assert "nonfinite_proposed_logits" in nonfinite.reasons
    assert nonfinite.background_mass_post is None

    empty = check_two_sided_safety(
        source,
        source.clone(),
        background_weight=torch.zeros_like(background),
        candidate_cores=cores,
        candidate_rings=rings,
        max_background_mass_increase=0.0,
        max_absolute_drop=0.0,
        max_contrast_drop=0.0,
    )
    assert not empty.passed
    assert "empty_reliable_background" in empty.reasons


def test_malformed_candidate_metadata_and_learned_weight_are_rejected() -> None:
    source, background, cores, rings = _case()
    with pytest.raises(TypeError, match="boolean mask"):
        check_two_sided_safety(
            source,
            source.clone(),
            background_weight=background,
            candidate_cores=(cores[0].float(),),
            candidate_rings=rings,
            max_background_mass_increase=0.0,
            max_absolute_drop=0.0,
            max_contrast_drop=0.0,
        )
    with pytest.raises(TwoSidedSafetyError, match="identical length"):
        check_two_sided_safety(
            source,
            source.clone(),
            background_weight=background,
            candidate_cores=cores,
            candidate_rings=(),
            max_background_mass_increase=0.0,
            max_absolute_drop=0.0,
            max_contrast_drop=0.0,
        )
    with pytest.raises(TwoSidedSafetyError, match="detached"):
        check_two_sided_safety(
            source,
            source.clone(),
            background_weight=background.requires_grad_(),
            candidate_cores=cores,
            candidate_rings=rings,
            max_background_mass_increase=0.0,
            max_absolute_drop=0.0,
            max_contrast_drop=0.0,
        )


def test_safety_signature_has_no_label_outer_or_condition_channel() -> None:
    names = set(inspect.signature(check_two_sided_safety).parameters)
    forbidden = {
        "ground_truth",
        "gt",
        "label",
        "target",
        "outer_mask",
        "condition",
        "corruption",
        "severity",
    }
    assert names.isdisjoint(forbidden)
