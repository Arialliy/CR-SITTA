from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor, nn

from model.MSHNet_NSFPN_adaptable import MSHNetNSFPNAdaptable
from tta.adapters import (
    NSFPNRouterAdapter,
    NSFPNRouterAdapterError,
    ROUTER_D0_SPACE,
    ROUTER_E1_D0_SPACE,
    ROUTER_E1_SPACE,
    SpatialLowRankFiLM,
    build_default_nsfpn_routers,
)


class _ToyFPN(nn.Module):
    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, ...]:
        return tuple(
            feature.mean(dim=1, keepdim=True).repeat(1, 64, 1, 1)
            for feature in features
        )


def _conv(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=False),
    )


def _toy_model(
    *, router_e1: nn.Module | None = None, router_d0: nn.Module | None = None
) -> MSHNetNSFPNAdaptable:
    # Exercise the real adaptable forward/load implementation with a small,
    # CPU-only module inventory.  The production NS-FPN construction is not
    # needed for these architecture-contract tests and its deformable operator
    # is CUDA-only.
    model = MSHNetNSFPNAdaptable.__new__(MSHNetNSFPNAdaptable)
    nn.Module.__init__(model)
    model.pool = nn.MaxPool2d(2, 2)
    model.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    model.up_4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
    model.up_8 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
    model.conv_init = nn.Conv2d(3, 16, kernel_size=1)
    model.encoder_0 = _conv(16, 16)
    model.encoder_1 = _conv(16, 32)
    model.encoder_2 = _conv(32, 64)
    model.encoder_3 = _conv(64, 128)
    model.middle_layer = _conv(128, 256)
    model.fpn = _ToyFPN()
    model.decoder_3 = _conv(128, 128)
    model.decoder_2 = _conv(192, 64)
    model.decoder_1 = _conv(128, 32)
    model.decoder_0 = _conv(48, 16)
    model.output_0 = nn.Conv2d(16, 1, kernel_size=1)
    model.output_1 = nn.Conv2d(32, 1, kernel_size=1)
    model.output_2 = nn.Conv2d(64, 1, kernel_size=1)
    model.output_3 = nn.Conv2d(128, 1, kernel_size=1)
    model.final = nn.Conv2d(4, 1, kernel_size=3, padding=1)
    model.router_e1 = router_e1 if router_e1 is not None else nn.Identity()
    model.router_d0 = router_d0 if router_d0 is not None else nn.Identity()
    return model


def _router_pair() -> tuple[SpatialLowRankFiLM, SpatialLowRankFiLM]:
    return (
        SpatialLowRankFiLM(64, rank=4, grid_size=(8, 8), seed=3407),
        SpatialLowRankFiLM(16, rank=2, grid_size=(8, 8), seed=3407),
    )


def test_spatial_router_is_exact_identity_with_fixed_orthogonal_basis() -> None:
    generator_state = torch.random.get_rng_state().clone()
    router = SpatialLowRankFiLM(16, rank=2, grid_size=(8, 8), seed=3407)
    assert torch.equal(torch.random.get_rng_state(), generator_state)

    feature = torch.randn(2, 16, 11, 13)
    output = router(feature)
    basis = router.channel_basis
    expected_identity = torch.eye(2, dtype=basis.dtype)

    assert torch.equal(output, feature)
    assert router.is_identity()
    assert tuple(name for name, _ in router.named_parameters()) == (
        "scale_coeff",
        "bias_coeff",
    )
    assert sum(parameter.numel() for parameter in router.parameters()) == 256
    assert "channel_basis" in dict(router.named_buffers())
    assert "channel_basis" not in dict(router.named_parameters())
    assert torch.count_nonzero(basis) > 0
    assert torch.allclose(
        basis @ basis.T, expected_identity, atol=1e-6, rtol=1e-6
    )


def test_spatial_router_has_live_first_gradient_and_location_dependent_map() -> None:
    router = SpatialLowRankFiLM(4, rank=2, grid_size=(2, 2), seed=11)
    feature = torch.arange(1, 1 + 4 * 6 * 8, dtype=torch.float32).reshape(
        1, 4, 6, 8
    )
    spatial_weight = torch.linspace(0.25, 1.75, 6 * 8).reshape(1, 1, 6, 8)
    loss = (router(feature) * spatial_weight).sum()
    loss.backward()

    assert router.scale_coeff.grad is not None
    assert router.bias_coeff.grad is not None
    assert torch.linalg.vector_norm(router.scale_coeff.grad) > 0
    assert torch.linalg.vector_norm(router.bias_coeff.grad) > 0

    router.reset_identity_()
    with torch.no_grad():
        router.scale_coeff[0, 0, 0, 0] = 1.0
        modulated = router(torch.ones_like(feature))
    changed = (modulated - 1.0).abs()
    channel = int(router.channel_basis[0].abs().argmax().item())
    assert changed[0, channel, 0, 0] > changed[0, channel, -1, -1]
    assert not torch.equal(modulated, feature)


@pytest.mark.parametrize(
    "space, expected_e1, expected_d0, expected_count",
    [
        (ROUTER_E1_SPACE, True, False, 512),
        (ROUTER_D0_SPACE, False, True, 256),
        (ROUTER_E1_D0_SPACE, True, True, 768),
    ],
)
def test_default_router_spaces_have_exact_v6_topology(
    space: str,
    expected_e1: bool,
    expected_d0: bool,
    expected_count: int,
) -> None:
    router_e1, router_d0 = build_default_nsfpn_routers(space)
    assert isinstance(router_e1, SpatialLowRankFiLM) is expected_e1
    assert isinstance(router_d0, SpatialLowRankFiLM) is expected_d0
    if isinstance(router_e1, SpatialLowRankFiLM):
        assert (router_e1.channels, router_e1.rank, router_e1.grid_size) == (
            64,
            4,
            (8, 8),
        )
    if isinstance(router_d0, SpatialLowRankFiLM):
        assert (router_d0.channels, router_d0.rank, router_d0.grid_size) == (
            16,
            2,
            (8, 8),
        )
    assert sum(parameter.numel() for parameter in router_e1.parameters()) + sum(
        parameter.numel() for parameter in router_d0.parameters()
    ) == expected_count


def test_adaptable_model_strictly_loads_legacy_source_and_is_bit_exact() -> None:
    torch.manual_seed(17)
    source = _toy_model().eval()
    router_e1, router_d0 = _router_pair()
    adaptable = _toy_model(
        router_e1=router_e1, router_d0=router_d0
    ).eval()

    # This is the exact legacy checkpoint API: no router keys are present and
    # strict=True must still verify every original NS-FPN key.
    incompatible = adaptable.load_state_dict(source.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert router_e1.is_identity() and router_d0.is_identity()

    image = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        source_cold = source(image, False)
        adaptable_cold = adaptable(image, False)
        source_warm = source(image, True)
        adaptable_warm = adaptable(image, True)

    assert torch.equal(adaptable_cold[1], source_cold[1])
    assert len(adaptable_warm[0]) == len(source_warm[0]) == 4
    assert all(
        torch.equal(current, expected)
        for current, expected in zip(
            adaptable_warm[0], source_warm[0], strict=True
        )
    )
    assert torch.equal(adaptable_warm[1], source_warm[1])
    assert len(adaptable.router_e1._forward_hooks) == 0
    assert len(adaptable.router_d0._forward_hooks) == 0

    missing = dict(source.state_dict())
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="Missing key"):
        adaptable.load_state_dict(missing, strict=True)


@pytest.mark.parametrize(
    "space, expected_names",
    [
        (
            ROUTER_E1_SPACE,
            ("router_e1.scale_coeff", "router_e1.bias_coeff"),
        ),
        (
            ROUTER_D0_SPACE,
            ("router_d0.scale_coeff", "router_d0.bias_coeff"),
        ),
        (
            ROUTER_E1_D0_SPACE,
            (
                "router_e1.scale_coeff",
                "router_e1.bias_coeff",
                "router_d0.scale_coeff",
                "router_d0.bias_coeff",
            ),
        ),
    ],
)
def test_only_selected_router_requires_grad_and_receives_gradient(
    space: str, expected_names: tuple[str, ...]
) -> None:
    router_e1, router_d0 = _router_pair()
    model = _toy_model(router_e1=router_e1, router_d0=router_d0)
    adapter = NSFPNRouterAdapter(model, space=space)
    parameters, names = adapter.collect_adaptable_params()

    assert tuple(names) == expected_names
    assert tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ) == expected_names

    image = torch.randn(1, 3, 16, 16)
    loss = adapter.forward_logits(image).square().sum()
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.linalg.vector_norm(gradient) > 0 for gradient in gradients)
    assert all(
        parameter.grad is None for parameter in model.parameters()
    )


def test_episode_reset_restores_router_optimizer_and_output_exactly() -> None:
    router_e1, router_d0 = _router_pair()
    model = _toy_model(router_e1=router_e1, router_d0=router_d0)
    adapter = NSFPNRouterAdapter(model, space=ROUTER_E1_D0_SPACE)
    parameters, _names = adapter.collect_adaptable_params()
    optimizer = torch.optim.SGD(parameters, lr=0.1, momentum=0.9)
    image = torch.randn(1, 3, 16, 16)

    with torch.no_grad():
        source_output = adapter.forward_logits(image).clone()
    optimizer.zero_grad(set_to_none=True)
    adapter.forward_logits(image).square().sum().backward()
    optimizer.step()
    with torch.no_grad():
        adapted_output = adapter.forward_logits(image).clone()

    assert optimizer.state
    assert not torch.equal(adapted_output, source_output)
    adapter.reset_episode_(optimizer=optimizer)
    with torch.no_grad():
        restored_output = adapter.forward_logits(image)

    assert router_e1.is_identity() and router_d0.is_identity()
    assert optimizer.state == {}
    assert all(parameter.grad is None for parameter in parameters)
    assert torch.equal(restored_output, source_output)
    adapter.assert_only_router_trainable()


def test_source_mode_rejects_adapted_router_instead_of_mislabeling_source() -> None:
    router_e1, router_d0 = _router_pair()
    model = _toy_model(router_e1=router_e1, router_d0=router_d0)
    adapter = NSFPNRouterAdapter(model, space=ROUTER_E1_D0_SPACE)
    with torch.no_grad():
        router_e1.scale_coeff.fill_(0.25)

    with pytest.raises(NSFPNRouterAdapterError, match="not identity"):
        adapter.set_source_eval_mode()


def test_episode_reset_restores_all_source_buffers_and_output() -> None:
    router_e1, router_d0 = _router_pair()
    model = _toy_model(router_e1=router_e1, router_d0=router_d0)
    model.audit_bn = nn.BatchNorm2d(1)
    model.register_buffer("source_audit_buffer", torch.tensor([3.0, 7.0]))
    adapter = NSFPNRouterAdapter(model, space=ROUTER_E1_D0_SPACE)
    expected_buffers = {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }
    image = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        source_output = adapter.forward_logits(image).clone()
        router_e1.scale_coeff.fill_(0.5)
        router_d0.bias_coeff.fill_(-0.5)
        router_e1.channel_basis.add_(0.125)
        model.audit_bn.running_mean.fill_(9.0)
        model.audit_bn.running_var.fill_(4.0)
        model.audit_bn.num_batches_tracked.fill_(11)
        model.source_audit_buffer.zero_()
        changed_output = adapter.forward_logits(image).clone()

    assert not torch.equal(changed_output, source_output)
    adapter.reset_episode_()

    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, expected_buffers[name])
    with torch.no_grad():
        restored_output = adapter.forward_logits(image)
    assert torch.equal(restored_output, source_output)
    adapter.set_source_eval_mode()
    assert all(not parameter.requires_grad for parameter in model.parameters())
