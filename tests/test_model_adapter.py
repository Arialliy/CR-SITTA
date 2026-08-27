import pytest
import torch
from torch import nn

from tta.model_adapter import IRSTDModelAdapter


class TinyNSFPNLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout2d(p=0.9)
        self.head = nn.Conv2d(4, 1, 1)

    def forward(self, image, warm_flag):
        features = self.dropout(self.bn(self.conv(image)))
        logits = self.head(features)
        return ([features] if warm_flag else []), logits


def test_forward_returns_raw_single_channel_logits() -> None:
    torch.manual_seed(1)
    adapter = IRSTDModelAdapter(TinyNSFPNLike())
    image = torch.randn(2, 3, 8, 9)
    logits = adapter.forward_logits(image)
    assert logits.shape == (2, 1, 8, 9)
    assert torch.isfinite(logits).all()


def test_zero_logit_maps_to_half_probability() -> None:
    logits = torch.zeros(1, 1, 2, 3)
    probability = IRSTDModelAdapter.logits_to_prob(logits)
    assert torch.equal(probability, torch.full_like(logits, 0.5))


def test_collects_only_batchnorm_affine_in_named_module_order() -> None:
    model = TinyNSFPNLike()
    parameters, names = IRSTDModelAdapter(model).collect_adaptable_params()
    assert names == ["bn.weight", "bn.bias"]
    assert parameters == [model.bn.weight, model.bn.bias]


@pytest.mark.parametrize("use_batch_stats", [True, False])
def test_tent_mode_freezes_non_bn_and_disables_dropout(use_batch_stats: bool) -> None:
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    running_mean = model.bn.running_mean.clone()
    adapter.set_tent_mode(use_batch_stats)

    assert model.dropout.training is False
    assert model.bn.training is use_batch_stats
    assert model.bn.track_running_stats is (not use_batch_stats)
    assert model.bn.weight.requires_grad
    assert model.bn.bias.requires_grad
    assert not model.conv.weight.requires_grad
    assert not model.head.weight.requires_grad
    assert torch.equal(model.bn.running_mean, running_mean)


def test_source_mode_is_deterministic_and_frozen() -> None:
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    image = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        first = adapter.forward_logits(image)
        second = adapter.forward_logits(image)
    assert torch.equal(first, second)
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_source_mode_restores_bn_flags_after_batch_stat_mode() -> None:
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)

    adapter.set_tent_mode(use_batch_stats=True)
    assert model.bn.training
    assert not model.bn.track_running_stats

    adapter.set_source_eval_mode()
    assert not model.bn.training
    assert model.bn.track_running_stats


def test_adabn_mode_uses_batch_stats_without_updating_buffers_or_parameters() -> None:
    model = TinyNSFPNLike()
    adapter = IRSTDModelAdapter(model)
    running_mean = model.bn.running_mean.clone()
    running_var = model.bn.running_var.clone()
    batches = model.bn.num_batches_tracked.clone()
    parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    adapter.set_adabn_mode()
    with torch.no_grad():
        adapter.forward_logits(torch.randn(1, 3, 8, 8))

    assert not model.training
    assert not model.dropout.training
    assert model.bn.training
    assert not model.bn.track_running_stats
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert torch.equal(model.bn.running_mean, running_mean)
    assert torch.equal(model.bn.running_var, running_var)
    assert torch.equal(model.bn.num_batches_tracked, batches)
    assert all(
        torch.equal(parameter, parameters[name])
        for name, parameter in model.named_parameters()
    )


def test_adabn_requires_at_least_one_batchnorm2d() -> None:
    class NoBatchNormModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Conv2d(3, 1, 1)

        def forward(self, image, warm_flag):
            del warm_flag
            return image, self.head(image)

    with pytest.raises(ValueError, match="at least one BatchNorm2d"):
        IRSTDModelAdapter(NoBatchNormModel()).set_adabn_mode()


def test_rejects_probability_shape_or_invalid_model_return() -> None:
    with pytest.raises(ValueError):
        IRSTDModelAdapter.logits_to_prob(torch.zeros(1, 2, 3, 3))

    class BadModel(nn.Module):
        def forward(self, image, warm_flag):
            return "not logits"

    with pytest.raises(TypeError):
        IRSTDModelAdapter(BadModel()).forward_logits(torch.zeros(1, 3, 3, 3))
