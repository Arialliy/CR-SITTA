from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import torch
from torch import nn

from analysis import audit_d0a_branch_gradients as audit


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_init = nn.Conv2d(3, 2, 1)
        self.encoder_0 = nn.Sequential(nn.BatchNorm2d(2), nn.Conv2d(2, 2, 1))
        self.output_0 = nn.Conv2d(2, 1, 1)
        self.output_1 = nn.Conv2d(2, 1, 1)  # unused outside warmup
        self.final = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, image, warm_flag):
        assert not warm_flag
        return [], self.output_0(self.encoder_0(self.conv_init(image)))


def loss(outputs, target):
    return (outputs[1] - target).square().mean()


@pytest.mark.parametrize("name,expected", [
    ("conv_init.weight", "encoder"), ("encoder_3.0.bn1.weight", "encoder"),
    ("middle_layer.1.conv1.weight", "encoder"),
    ("fpn.wavenhance_list.0.attention.conv1.weight", "fpn_lfp"),
    ("fpn.lateral_convs.0.module.0.weight", "fpn_lfp"),
    ("fpn.fpn_convs.0.module.0.weight", "fpn_lfp"),
    ("fpn.crossattn_list.0.norm1.weight", "fpn_sfs"),
    ("decoder_3.0.conv1.weight", "decoder3_2"), ("decoder_2.0.conv1.weight", "decoder3_2"),
    ("decoder_1.0.conv1.weight", "decoder1"), ("decoder_0.0.conv1.weight", "decoder0"),
    ("output_0.weight", "head"), ("final.bias", "head"),
])
def test_structural_mapping(name, expected):
    assert audit.structural_group(name) == expected


def test_unknown_parameter_fails_closed():
    with pytest.raises(ValueError, match="unmapped"):
        audit.structural_group("new_module.weight")


def test_mapping_partitions_parameters_and_tracks_overlapping_bn_affine():
    model = Toy()
    mapping = audit.parameter_mapping(model)
    structural_names = [name for group in audit.STRUCTURAL_GROUPS for name in audit.names_for_group(mapping, group)]
    assert len(structural_names) == len(set(structural_names)) == len(list(model.parameters()))
    assert set(audit.names_for_group(mapping, "global")) == set(dict(model.named_parameters()))
    assert set(audit.names_for_group(mapping, "bn_affine")) == {"encoder_0.0.weight", "encoder_0.0.bias"}
    assert not set(audit.names_for_group(mapping, "bn_affine")) & set(audit.names_for_group(mapping, "non_bn"))


@pytest.mark.parametrize("degraded", [False, True])
def test_branch_grad_restores_all_state_modes_and_does_not_persist_gradients(degraded):
    torch.manual_seed(7)
    model = Toy().eval()
    model.encoder_0[0].train()
    modes = {name: module.training for name, module in model.named_modules()}
    before = audit.state_hash(model)
    image, target = torch.rand(2, 3, 8, 8), torch.rand(2, 1, 8, 8)
    value, gradients, receipt = audit.branch_gradient(model, image, target, loss, degraded=degraded)
    assert value > 0
    assert audit.state_hash(model) == before == receipt["state_sha256_after"]
    assert {name: module.training for name, module in model.named_modules()} == modes
    assert all(parameter.grad is None for parameter in model.parameters())
    assert gradients["output_1.weight"] is None and gradients["final.bias"] is None
    assert gradients["conv_init.weight"] is not None
    assert all(value is None or (value.device.type == "cpu" and not value.requires_grad) for value in gradients.values())
    assert bool(receipt["temporary_changed_buffers"]) is not degraded
    assert model.encoder_0[0].track_running_stats


def test_exception_still_restores_clean_bn_updates():
    model = Toy().eval()
    before = audit.state_hash(model)
    def fail(*args):
        raise RuntimeError("intentional failure")
    with pytest.raises(RuntimeError, match="intentional"):
        audit.branch_gradient(model, torch.rand(2, 3, 8, 8), torch.rand(2, 1, 8, 8), fail, degraded=False)
    assert audit.state_hash(model) == before
    assert not model.training


def test_clean_and_degraded_on_same_input_have_same_gradients_with_batch_statistics():
    torch.manual_seed(7)
    model = Toy().eval()
    image, target = torch.rand(2, 3, 8, 8), torch.rand(2, 1, 8, 8)
    clean = audit.branch_gradient(model, image, target, loss, degraded=False)
    degraded = audit.branch_gradient(model, image, target, loss, degraded=True)
    assert clean[0] == degraded[0]
    for name in clean[1]:
        if clean[1][name] is None:
            assert degraded[1][name] is None
        else:
            assert torch.equal(clean[1][name], degraded[1][name])


def test_zero_and_unused_norms_are_undefined_not_zero_cosine():
    left = {"unused": None, "zero": torch.zeros(2)}
    right = {"unused": None, "zero": torch.ones(2)}
    result = audit.vector_statistics(left, right, ["unused", "zero"])
    assert result["cosine"] is None
    assert result["second_to_first_norm_ratio"] is None
    assert result["first_used_parameters"] == 1
    assert result["first_norm"] == 0
    assert result["undefined_reason"] == "one_or_both_gradient_norms_zero"


def test_gradient_cosines_and_norm_ratio_match_manual():
    left, right = {"a": torch.tensor([1., 2.]), "b": None}, {"a": torch.tensor([-2., -4.]), "b": None}
    result = audit.vector_statistics(left, right, ["a", "b"])
    assert result["cosine"] == pytest.approx(-1)
    assert result["second_to_first_norm_ratio"] == pytest.approx(2)
    assert result["dot"] == -10


def test_pooling_concatenates_batches_not_average_cosines():
    one = audit.vector_statistics({"a": torch.tensor([100.])}, {"a": torch.tensor([100.])}, ["a"])
    two = audit.vector_statistics({"a": torch.tensor([1.])}, {"a": torch.tensor([-1.])}, ["a"])
    result = audit.pool_pair_statistics([one, two])
    assert result["cosine"] == pytest.approx(9999 / 10001)
    assert result["valid_batches"] == 2


def test_adagrad_mapping_is_complete_and_does_not_change_original_accumulators():
    model = Toy()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.05)
    archived = optimizer.state_dict()
    sums, metadata = audit.adagrad_state_by_name(model, archived, learning_rate=0.05)
    assert len(sums) == len(list(model.parameters()))
    assert metadata["eps"] == 1e-10
    sums["conv_init.weight"].fill_(5)
    assert torch.count_nonzero(archived["state"][0]["sum"]) == 0


def test_adagrad_mapping_rejects_schema_or_settings_drift():
    model = Toy()
    original = torch.optim.Adagrad(model.parameters(), lr=0.05).state_dict()
    wrong = copy.deepcopy(original)
    wrong["state"][0]["sum"] = torch.zeros(5)
    with pytest.raises(ValueError, match="schema"):
        audit.adagrad_state_by_name(model, wrong, learning_rate=0.05)
    wrong = copy.deepcopy(original)
    wrong["param_groups"][0]["lr_decay"] = 0.1
    with pytest.raises(ValueError, match="settings"):
        audit.adagrad_state_by_name(model, wrong, learning_rate=0.05)


def test_adagrad_counterfactual_includes_current_gradient_in_accumulator():
    clean, degraded, accumulators = {"a": torch.tensor([2.])}, {"a": torch.tensor([4.])}, {"a": torch.tensor([16.])}
    before = accumulators["a"].clone()
    result = audit.simulate_adagrad_step(clean, degraded, accumulators, ["a"], lr=0.05, eps=0)
    # combined=3, new accumulator=25, delta=-0.03, delta dot gc=-0.06.
    assert result["step_norm"] == pytest.approx(0.03)
    assert result["step_dot_clean_gradient"] == pytest.approx(-0.06)
    assert result["accumulator_increment_sum"] == 9
    assert result["effective_lr_mean"] == pytest.approx(0.01)
    assert result["optimizer_steps_applied"] == 0
    assert not result["historical_drift_inference_permitted"]
    assert torch.equal(accumulators["a"], before)


def test_unused_adagrad_parameters_do_not_get_fake_steps():
    result = audit.simulate_adagrad_step({"a": None}, {"a": None}, {"a": torch.zeros(3)}, ["a"], lr=0.05, eps=1e-10)
    assert result["updated_parameter_count_if_applied"] == 0
    assert result["step_norm"] == 0
    assert result["effective_lr_mean"] is None
    assert result["step_cosine_with_clean_gradient"] is None


def make_pooled(negative_groups=4, valid_batches=3):
    return {group: {f"clean__{probe}": {"valid_batches": valid_batches,
                "cosine": -0.1 if index < negative_groups and probe == "lf_mask" else 0.1}
            for probe in ("lf_mask", "hf_noise")}
            for index, group in enumerate(audit.STRUCTURAL_GROUPS)}


def test_e2_requires_four_structural_groups_and_three_valid_batches_per_group():
    assert audit.e2_diagnosis(make_pooled())["probes"]["lf_mask"]["triggered"]
    assert not audit.e2_diagnosis(make_pooled(3))["any_probe_triggered"]
    assert not audit.e2_diagnosis(make_pooled(4, 2))["any_probe_triggered"]
    result = audit.e2_diagnosis(make_pooled())
    assert not result["probes"]["hf_noise"]["triggered"]
    assert not result["stage_promotion_authorized"]
    assert result["diagnostic_only"]


def test_no_optimizer_update_or_test_loader_in_runner():
    source = inspect.getsource(audit)
    assert ".step(" not in source
    assert "test_split" not in source
    assert "FixedSplitIRSTDDataset(" not in source
    assert "_restore_rng_state" not in source


def test_access_counters_require_all_pilot_and_zero_test_validation():
    access = {"train_image_opens": 64, "train_mask_opens": 64,
              "test_image_opens": 0, "test_mask_opens": 0,
              "validation_image_opens": 0, "validation_mask_opens": 0}
    audit.validate_access_counters(access)
    for key in access:
        invalid = dict(access)
        invalid[key] += 1
        with pytest.raises(RuntimeError, match="data access count"):
            audit.validate_access_counters(invalid)
    with pytest.raises(RuntimeError):
        audit.validate_access_counters({})


def test_preflight_writes_nothing_and_opens_no_payload(monkeypatch, tmp_path):
    from analysis import d0a_v7_common as common
    monkeypatch.setattr(audit, "OUTPUT", tmp_path / "missing")
    monkeypatch.setattr(audit, "checkpoint_preflight", lambda config: {})
    monkeypatch.setattr(common, "read_config", lambda path: {})
    monkeypatch.setattr(common, "load_pilot_records", lambda dataset, config: [{"image_id": str(i)} for i in range(64)])
    def forbidden(*args, **kwargs):
        raise AssertionError("must not load payloads or create artifacts during preflight")
    for name in ("load_sample", "reserve_output", "freeze_run", "complete_run"):
        monkeypatch.setattr(common, name, forbidden)
    result = audit.run(Path("unused"), device_name="cpu", execute=False)
    assert result["ready"] and result["writes"] == 0
    assert not (tmp_path / "missing").exists()


def test_preflight_refuses_existing_partial_output(monkeypatch, tmp_path):
    from analysis import d0a_v7_common as common
    monkeypatch.setattr(audit, "OUTPUT", tmp_path)
    monkeypatch.setattr(audit, "checkpoint_preflight", lambda config: {})
    monkeypatch.setattr(common, "read_config", lambda path: {})
    monkeypatch.setattr(common, "load_pilot_records", lambda dataset, config: [{}] * 64)
    with pytest.raises(FileExistsError):
        audit.run(Path("unused"), device_name="cpu", execute=False)
