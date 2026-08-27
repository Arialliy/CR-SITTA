from __future__ import annotations

import pytest
import torch
from torch import nn

from tta.adabn import AdaBNMethod
from tta.episodic_runner import AdaptationOutcome, EpisodeProtocolError, EpisodicRunner
from tta.model_adapter import IRSTDModelAdapter
from tta.state_manager import EpisodicStateManager


class ControlledAdaBNModel(nn.Module):
    """One-channel model whose Source and per-image BN outputs are analytic."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.1)
        self.dropout = nn.Dropout2d(p=0.95)
        self.head = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1.0)
            self.bn.weight.fill_(1.75)
            self.bn.bias.fill_(-0.25)
            self.bn.running_mean.fill_(7.0)
            self.bn.running_var.fill_(9.0)
            self.head.weight.fill_(1.0)

    def forward(self, image: torch.Tensor, warm_flag: bool):
        features = self.dropout(self.bn(self.conv(image)))
        return ([features] if warm_flag else []), self.head(features)


class TwoBatchNormModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)

    def forward(self, image: torch.Tensor, warm_flag: bool):
        del warm_flag
        logits = self.bn2(self.bn1(image))
        return [], logits


def _metadata(image_id: str) -> dict[str, object]:
    return {
        "image_id": image_id,
        "original_size": [4, 4],
        "dataset": "source-pilot",
        "corruption": "gaussian_noise",
        "severity": 3,
        "seed": 42,
    }


def _make_runner(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[IRSTDModelAdapter, EpisodicStateManager, EpisodicRunner]:
    adapter = IRSTDModelAdapter(model)
    adapter.set_source_eval_mode()
    state = EpisodicStateManager(model, optimizer=optimizer)
    return adapter, state, EpisodicRunner(adapter, state)


def test_adabn_post_logits_match_current_image_batchnorm_formula() -> None:
    model = ControlledAdaBNModel()
    _adapter, state, runner = _make_runner(model)
    image = torch.tensor(
        [
            [
                [
                    [-3.0, -1.0, 0.0, 1.0],
                    [2.0, 4.0, 7.0, 11.0],
                    [12.0, 13.0, 15.0, 18.0],
                    [21.0, 25.0, 30.0, 36.0],
                ]
            ]
        ]
    )
    source_mean = model.bn.running_mean.detach().reshape(1, 1, 1, 1).clone()
    source_var = model.bn.running_var.detach().reshape(1, 1, 1, 1).clone()
    weight = model.bn.weight.detach().reshape(1, 1, 1, 1).clone()
    bias = model.bn.bias.detach().reshape(1, 1, 1, 1).clone()
    eps = model.bn.eps

    result = runner.run_one_image(
        image=image,
        metadata=_metadata("manual-current-stats"),
        method=AdaBNMethod(),
    )

    current_mean = image.mean(dim=(0, 2, 3), keepdim=True)
    # torch.nn.BatchNorm2d uses the biased variance for the current forward.
    current_var = image.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    expected_post = (image - current_mean) / torch.sqrt(current_var + eps)
    expected_post = expected_post * weight + bias
    expected_source = (image - source_mean) / torch.sqrt(source_var + eps)
    expected_source = expected_source * weight + bias

    torch.testing.assert_close(result.logits_post, expected_post)
    torch.testing.assert_close(result.logits_pre, expected_source)
    assert not torch.equal(result.logits_pre, result.logits_post)
    assert result.state_changes_after_prepare == ("runtime",)
    assert result.state_changes_after_adapt == ("runtime",)
    assert result.state_changes_after_post == ("runtime",)
    assert (
        result.state_after_prepare_fingerprint
        == result.state_after_adapt_fingerprint
        == result.state_after_post_fingerprint
    )
    assert (
        result.state_after_post_fingerprint.model_sha256
        == result.source_fingerprint.model_sha256
    )
    assert (
        result.state_after_post_fingerprint.gradients_sha256
        == result.source_fingerprint.gradients_sha256
    )
    assert result.source_fingerprint.optimizer_sha256 is None
    assert result.state_after_post_fingerprint.optimizer_sha256 is None
    assert result.reset_fingerprint == result.source_fingerprint
    state.assert_source_state()

class TamperingAdaBNMethod(AdaBNMethod):
    def __init__(self, mutation: str) -> None:
        self.mutation = mutation

    def adapt_one_image(self, *, adapter, image, logits_pre, metadata):
        batchnorm = adapter.model.bn
        if self.mutation == "parameter":
            batchnorm.weight.add_(0.25)
        elif self.mutation == "buffer":
            batchnorm.running_mean.add_(3.0)
        elif self.mutation == "gradient":
            batchnorm.weight.grad = torch.ones_like(batchnorm.weight)
        elif self.mutation == "eps":
            batchnorm.eps = 0.5
        elif self.mutation == "momentum":
            batchnorm.momentum = 0.75
        else:  # pragma: no cover - helper misuse
            raise ValueError(self.mutation)
        return super().adapt_one_image(
            adapter=adapter,
            image=image,
            logits_pre=logits_pre,
            metadata=metadata,
        )


@pytest.mark.parametrize(
    ("mutation", "component"),
    [
        ("parameter", "model"),
        ("buffer", "model"),
        ("gradient", "gradients"),
        ("eps", "runtime"),
        ("momentum", "runtime"),
    ],
)
def test_adabn_rejects_state_tampering_before_reset(
    mutation: str,
    component: str,
) -> None:
    model = ControlledAdaBNModel()
    _adapter, state, runner = _make_runner(model)

    with pytest.raises(
        EpisodeProtocolError,
        match=rf"must keep state frozen after prepare.*{component}",
    ):
        runner.run_one_image(
            image=torch.randn(1, 1, 4, 4),
            metadata=_metadata(f"tamper-{mutation}"),
            method=TamperingAdaBNMethod(mutation),
        )

    state.assert_source_state()
    assert model.bn.eps == 1e-5
    assert model.bn.momentum == 0.1
    assert model.bn.weight.grad is None


class PostForwardBufferTamperModel(ControlledAdaBNModel):
    def forward(self, image: torch.Tensor, warm_flag: bool):
        output = super().forward(image, warm_flag)
        if self.bn.training:
            self.bn.running_var.add_(2.0)
        return output


def test_adabn_rejects_running_buffer_mutation_during_post_forward() -> None:
    model = PostForwardBufferTamperModel()
    _adapter, state, runner = _make_runner(model)

    with pytest.raises(
        EpisodeProtocolError,
        match=r"must keep state frozen after prepare.*post_forward: model",
    ):
        runner.run_one_image(
            image=torch.randn(1, 1, 4, 4),
            metadata=_metadata("post-forward-buffer-tamper"),
            method=AdaBNMethod(),
        )

    state.assert_source_state()


class PartialBatchStatsAdaBNMethod(AdaBNMethod):
    def prepare_episode(self, adapter: IRSTDModelAdapter) -> None:
        adapter.set_source_eval_mode()
        adapter.model.bn1.train()
        adapter.model.bn1.track_running_stats = False


def test_adabn_rejects_partial_batchnorm_mode_switch() -> None:
    model = TwoBatchNormModel()
    _adapter, state, runner = _make_runner(model)

    with pytest.raises(
        EpisodeProtocolError,
        match=r"requires every BatchNorm2d.*violation at 'bn2'",
    ):
        runner.run_one_image(
            image=torch.randn(1, 1, 4, 4),
            metadata=_metadata("partial-bn-mode"),
            method=PartialBatchStatsAdaBNMethod(),
        )

    state.assert_source_state()


def test_adabn_rejects_a_state_manager_owned_optimizer() -> None:
    model = ControlledAdaBNModel()
    optimizer = torch.optim.SGD(
        [model.bn.weight, model.bn.bias],
        lr=0.1,
        momentum=0.9,
    )
    _adapter, state, runner = _make_runner(model, optimizer)

    with pytest.raises(
        EpisodeProtocolError,
        match=r"optimizer.*identical.*including None",
    ):
        runner.run_one_image(
            image=torch.randn(1, 1, 4, 4),
            metadata=_metadata("managed-optimizer"),
            method=AdaBNMethod(),
        )

    state.assert_source_state()
    assert optimizer.state_dict()["state"] == {}


class GradContextSpyAdaBNMethod(AdaBNMethod):
    def __init__(self) -> None:
        self.grad_enabled: bool | None = None

    def adapt_one_image(self, *, adapter, image, logits_pre, metadata):
        self.grad_enabled = torch.is_grad_enabled()
        return super().adapt_one_image(
            adapter=adapter,
            image=image,
            logits_pre=logits_pre,
            metadata=metadata,
        )


def test_adabn_hook_runs_without_grad_and_cannot_receive_labels() -> None:
    model = ControlledAdaBNModel()
    _adapter, state, runner = _make_runner(model)
    method = GradContextSpyAdaBNMethod()

    result = runner.run_one_image(
        image=torch.randn(1, 1, 4, 4),
        metadata=_metadata("no-grad"),
        method=method,
    )
    assert method.grad_enabled is False
    assert result.outcome.optimizer_steps == 0
    assert all(parameter.grad is None for parameter in model.parameters())

    unsafe_metadata = {
        **_metadata("label-firewall"),
        "mask": torch.ones(1, 1, 4, 4),
    }
    with pytest.raises(
        EpisodeProtocolError,
        match="outside the label-free allowlist",
    ):
        runner.run_one_image(
            image=torch.randn(1, 1, 4, 4),
            metadata=unsafe_metadata,
            method=method,
        )
    state.assert_source_state()
