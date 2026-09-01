from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.analyze_tent_optimizer_geometry import (
    ANALYSIS_SCHEMA_VERSION,
    CONTINUOUS_IDEAL_REFERENCE,
    CROSS_BACKEND_CPU_STORAGE_REPLAY_REFERENCE,
    OptimizerFirstStepSpec,
    OptimizerGeometryError,
    analyze_optimizer_first_step,
    analyze_payload,
    main,
    pytorch_first_step_reference,
    pytorch_first_step_reference_after,
)
from analysis.source_train_provenance import (
    AnalysisProvenanceError,
    SourceTrainAnalysisProvenance,
)
from tta.binary_tent import build_binary_tent_optimizer
from tta.parameter_vector import ParameterVectorError


def _provenance_mapping() -> dict:
    return {
        "dataset": "IRSTD-1K",
        "split_name": "train",
        "split_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "seed": 0,
        "source_train_derived": True,
        "paper_test_result": False,
        "use_test_images": False,
        "use_test_labels": False,
        "oracle_analysis": False,
        "method_label_accesses": 0,
        "outer_evaluator_label_accesses": 0,
        "adaptation_gradient_uses_labels": False,
        "supervised_gradient_role": "none",
    }


def _provenance() -> SourceTrainAnalysisProvenance:
    return SourceTrainAnalysisProvenance.from_mapping(
        _provenance_mapping(), require_outer_oracle=False
    )


def _sgd_config(lr: float) -> dict:
    return {
        "name": "SGD",
        "learning_rate": lr,
        "momentum": 0.9,
        "dampening": 0.0,
        "weight_decay": 0.0,
        "nesterov": True,
        "maximize": False,
        "initial_optimizer_state_empty": True,
    }


def _adam_config(lr: float) -> dict:
    return {
        "name": "Adam",
        "learning_rate": lr,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "decoupled_weight_decay": False,
        "initial_optimizer_state_empty": True,
    }


def test_current_sgd_first_step_is_exact_negative_1_point_9_lr_gradient() -> None:
    lr = 1e-3
    parameters = [
        nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64)),
        nn.Parameter(torch.tensor([0.5], dtype=torch.float64)),
    ]
    gradients = [
        torch.tensor([0.2, -0.4], dtype=torch.float64),
        torch.tensor([1.5], dtype=torch.float64),
    ]
    optimizer = build_binary_tent_optimizer(parameters, name="SGD", learning_rate=lr)
    assert optimizer.state == {}
    before = [parameter.detach().clone() for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.clone()
    optimizer.step()

    report = analyze_optimizer_first_step(
        parameters_before={"bn.bias": before[1], "bn.weight": before[0]},
        gradients={"bn.bias": gradients[1], "bn.weight": gradients[0]},
        parameters_after={
            "bn.bias": parameters[1].detach(),
            "bn.weight": parameters[0].detach(),
        },
        optimizer_config=_sgd_config(lr),
        provenance=_provenance(),
        parameter_groups={"bn.bias": "decoder", "bn.weight": "decoder"},
    )
    expected_weight = -1.9 * lr * gradients[0]
    torch.testing.assert_close(
        parameters[0].detach() - before[0], expected_weight, rtol=1e-12, atol=1e-15
    )
    assert report["current_binary_tent_optimizer_configuration"] is True
    assert "-1.9" in report["first_step_formula"]
    assert report["global"]["cpu_storage_replay_within_frozen_tolerance"] is True
    assert report["global"][
        "cross_backend_cpu_storage_replay_cos_actual_expected"
    ] == pytest.approx(1.0)
    assert report["global"][
        "cross_backend_cpu_storage_replay_actual_to_expected_norm_ratio"
    ] == pytest.approx(1.0)
    assert report["per_group"]["decoder"]["parameter_tensor_count"] == 2
    assert report["per_group"]["decoder"]["relative_actual_step_norm"] > 0.0


def test_current_adam_first_step_matches_formula_and_reports_near_sign_fraction() -> None:
    lr = 1e-4
    parameter = nn.Parameter(
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    )
    gradient = torch.tensor([2.0, -3.0, 1e-10, 0.0], dtype=torch.float64)
    optimizer = build_binary_tent_optimizer([parameter], name="Adam", learning_rate=lr)
    assert optimizer.state == {}
    before = parameter.detach().clone()
    parameter.grad = gradient.clone()
    optimizer.step()

    report = analyze_optimizer_first_step(
        parameters_before={"bn.weight": before},
        gradients={"bn.weight": gradient},
        parameters_after={"bn.weight": parameter.detach()},
        optimizer_config=_adam_config(lr),
        provenance=_provenance(),
        small_gradient_threshold=1e-8,
    )
    expected = -lr * gradient / (gradient.abs() + 1e-8)
    torch.testing.assert_close(
        parameter.detach() - before, expected, rtol=1e-9, atol=1e-15
    )
    assert report["current_binary_tent_optimizer_configuration"] is True
    assert report["global"]["cpu_storage_replay_within_frozen_tolerance"] is True
    assert report["global"]["small_gradient_scalar_count"] == 2
    assert report["global"]["small_gradient_fraction"] == pytest.approx(0.5)
    assert report["global"]["near_sign_step_scalar_count"] == 2
    assert report["global"]["near_sign_step_fraction"] == pytest.approx(0.5)


@pytest.mark.parametrize("lr", [1e-5, 3e-5])
def test_float32_adam_storage_reference_handles_ulp_boundary(lr: float) -> None:
    parameter = nn.Parameter(
        torch.tensor([2.1058685779571533], dtype=torch.float32)
    )
    gradient = torch.tensor([3.7e-11], dtype=torch.float32)
    before = parameter.detach().clone()
    parameter.grad = gradient.clone()
    optimizer = build_binary_tent_optimizer(
        [parameter], name="Adam", learning_rate=lr
    )
    optimizer.step()

    report = analyze_optimizer_first_step(
        parameters_before={"decoder_0.bn.weight": before},
        gradients={"decoder_0.bn.weight": gradient},
        parameters_after={"decoder_0.bn.weight": parameter.detach()},
        optimizer_config=_adam_config(lr),
        provenance=_provenance(),
        verification_rtol=1e-5,
        verification_atol=1e-7,
    )
    assert report["schema_version"] == ANALYSIS_SCHEMA_VERSION == 3
    assert report["references"]["cross_backend_cpu_storage_replay"]["name"] == (
        CROSS_BACKEND_CPU_STORAGE_REPLAY_REFERENCE
    )
    assert report["references"]["continuous_ideal"]["name"] == (
        CONTINUOUS_IDEAL_REFERENCE
    )
    assert report["references"]["continuous_ideal"][
        "used_for_hard_gate"
    ] is False
    assert report["parameter_storage"]["storage_dtype_counts"] == {
        "torch.float32": 1
    }
    assert report["global"]["cpu_storage_replay_within_frozen_tolerance"] is True
    assert report["global"][
        "cross_backend_cpu_storage_replay_max_abs_residual"
    ] == 0.0
    if lr == 3e-5:
        assert report["global"]["continuous_ideal_max_abs_residual"] > 1e-7
    else:
        assert report["global"]["continuous_ideal_max_abs_residual"] < 1e-7


@pytest.mark.parametrize(
    ("name", "config_factory"),
    [("Adam", _adam_config), ("SGD", _sgd_config)],
)
@pytest.mark.parametrize("lr", [1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
def test_native_reference_is_bit_exact_with_actual_float32_first_step(
    name: str, config_factory, lr: float
) -> None:
    before = {
        "bn.bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
        "bn.weight": torch.tensor([1.0, 2.1058686], dtype=torch.float32),
    }
    gradients = {
        "bn.bias": torch.tensor([1e-10, -0.25], dtype=torch.float32),
        "bn.weight": torch.tensor([0.5, 3.7e-11], dtype=torch.float32),
    }
    parameters = [nn.Parameter(before[key].clone()) for key in before]
    for parameter, value in zip(parameters, gradients.values(), strict=True):
        parameter.grad = value.clone()
    optimizer = build_binary_tent_optimizer(
        parameters, name=name, learning_rate=lr
    )
    reference = pytorch_first_step_reference(
        parameters_before=before,
        gradients=gradients,
        optimizer=OptimizerFirstStepSpec.from_mapping(config_factory(lr)),
    )
    optimizer.step()
    assert all(
        torch.equal(parameter.detach(), reference.parameters_after[key])
        for key, parameter in zip(before, parameters, strict=True)
    )
    for key, parameter in zip(before, parameters, strict=True):
        actual_state = optimizer.state[parameter]
        expected_state = reference.optimizer_state[key]
        assert set(actual_state) == set(expected_state)
        assert all(
            torch.equal(actual_state[field], expected_state[field])
            for field in expected_state
        )


def test_native_reference_keeps_inactive_parameter_unchanged() -> None:
    before = {
        "active": torch.tensor([2.0], dtype=torch.float32),
        "inactive": torch.tensor([5.0], dtype=torch.float32),
    }
    config = _sgd_config(0.1)
    config.update(weight_decay=0.1)
    reference = pytorch_first_step_reference(
        parameters_before=before,
        gradients={"active": torch.tensor([1.0]), "inactive": None},
        optimizer=OptimizerFirstStepSpec.from_mapping(config),
    )
    assert torch.equal(
        reference.parameters_after["inactive"], before["inactive"]
    )
    assert not torch.equal(
        reference.parameters_after["active"], before["active"]
    )
    assert reference.optimizer_state["inactive"] == {}
    assert set(reference.optimizer_state["active"]) == {"momentum_buffer"}


@pytest.mark.parametrize("field", ["gradient", "after"])
def test_storage_reference_rejects_dtype_drift(field: str) -> None:
    before = {"bn.weight": torch.tensor([1.0], dtype=torch.float32)}
    gradient = {"bn.weight": torch.tensor([0.2], dtype=torch.float32)}
    after = {"bn.weight": torch.tensor([0.9], dtype=torch.float32)}
    if field == "gradient":
        gradient["bn.weight"] = gradient["bn.weight"].to(torch.float64)
    else:
        after["bn.weight"] = after["bn.weight"].to(torch.float64)
    with pytest.raises(OptimizerGeometryError, match="dtype"):
        analyze_optimizer_first_step(
            parameters_before=before,
            gradients=gradient,
            parameters_after=after,
            optimizer_config=_sgd_config(0.1),
            provenance=_provenance(),
        )


def test_storage_reference_rejects_named_topology_drift() -> None:
    with pytest.raises(OptimizerGeometryError, match="topology"):
        pytorch_first_step_reference_after(
            parameters_before={"bn.weight": torch.tensor([1.0])},
            gradients={"other": torch.tensor([0.2])},
            optimizer=OptimizerFirstStepSpec.from_mapping(_sgd_config(0.1)),
        )


def test_none_gradient_stays_inactive_even_with_weight_decay() -> None:
    report = analyze_optimizer_first_step(
        parameters_before={"active": torch.tensor([2.0]), "inactive": torch.tensor([5.0])},
        gradients={"active": torch.tensor([1.0]), "inactive": None},
        parameters_after={
            "active": torch.tensor([2.0 - 0.1 * 1.2]),
            "inactive": torch.tensor([5.0]),
        },
        optimizer_config={
            "name": "SGD",
            "learning_rate": 0.1,
            "momentum": 0.0,
            "dampening": 0.0,
            "weight_decay": 0.1,
            "nesterov": False,
            "maximize": False,
            "initial_optimizer_state_empty": True,
        },
        provenance=_provenance(),
        verification_atol=1e-7,
    )
    assert report["global"]["active_gradient_scalar_count"] == 1
    assert report["global"]["cpu_storage_replay_within_frozen_tolerance"] is True


def test_nonempty_state_bad_groups_and_bad_train_provenance_fail_closed() -> None:
    config = _sgd_config(1e-3)
    config["initial_optimizer_state_empty"] = False
    with pytest.raises(OptimizerGeometryError, match="empty optimizer state"):
        OptimizerFirstStepSpec.from_mapping(config)

    with pytest.raises(ParameterVectorError, match="cover the layout exactly"):
        analyze_optimizer_first_step(
            parameters_before={"a": torch.tensor([1.0]), "b": torch.tensor([2.0])},
            gradients={"a": torch.tensor([1.0]), "b": torch.tensor([1.0])},
            parameters_after={"a": torch.tensor([0.9]), "b": torch.tensor([1.9])},
            optimizer_config={
                "name": "SGD",
                "learning_rate": 0.1,
                "momentum": 0.0,
                "dampening": 0.0,
                "weight_decay": 0.0,
                "nesterov": False,
                "maximize": False,
                "initial_optimizer_state_empty": True,
            },
            provenance=_provenance(),
            parameter_groups={"a": "one"},
        )

    bad = _provenance_mapping()
    bad["split_name"] = "test"
    with pytest.raises(AnalysisProvenanceError, match="train split"):
        SourceTrainAnalysisProvenance.from_mapping(
            bad, require_outer_oracle=False
        )
    with pytest.raises(AnalysisProvenanceError, match="train split"):
        SourceTrainAnalysisProvenance(
            dataset="IRSTD-1K",
            split_name="test",  # type: ignore[arg-type]
            split_sha256="a" * 64,
            checkpoint_sha256="b" * 64,
            seed=0,
            oracle_analysis=False,
            outer_evaluator_label_accesses=0,
            supervised_gradient_role="none",
        )


def test_geometry_json_payload_and_cli_emit_fixed_nonpaper_scope(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {
        "provenance": _provenance_mapping(),
        "parameters_before": {"a": [1.0]},
        "gradients": {"a": [2.0]},
        "parameters_after": {"a": [0.62]},
        "optimizer": {
            "name": "SGD",
            "learning_rate": 0.1,
            "momentum": 0.9,
            "dampening": 0.0,
            "weight_decay": 0.0,
            "nesterov": True,
            "maximize": False,
            "initial_optimizer_state_empty": True,
        },
    }
    direct = analyze_payload(payload)
    assert direct["scope"]["paper_test_result"] is False
    assert direct["scope"]["method_label_accesses"] == 0
    input_path = tmp_path / "geometry.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["--input-json", str(input_path)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["global"]["cpu_storage_replay_within_frozen_tolerance"] is True


def test_geometry_analysis_does_not_change_cuda_initialization_state() -> None:
    initialized_before = torch.cuda.is_initialized()
    analyze_optimizer_first_step(
        parameters_before={"a": torch.tensor([1.0])},
        gradients={"a": torch.tensor([1.0])},
        parameters_after={"a": torch.tensor([0.81])},
        optimizer_config=_sgd_config(0.1),
        provenance=_provenance(),
        verification_atol=1e-7,
    )
    assert torch.cuda.is_initialized() is initialized_before
