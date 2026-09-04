from __future__ import annotations

import pytest
import torch
from torch import nn

from tta.adapters import (
    Decoder0HookError,
    DecoderFiLM,
    LowRankResidualMixer,
    decoder0_modulation,
)


class TinyDecoderModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder_0 = nn.Conv2d(3, 16, kernel_size=1)
        self.output_0 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.output_0(self.decoder_0(image))


def _trainable_scalar_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def _hook_count(module: nn.Module) -> int:
    return len(module._forward_hooks)


def test_decoder_film_is_exact_source_identity_and_has_32_trainable_scalars() -> None:
    feature = torch.randn(2, 16, 7, 9)
    adapter = DecoderFiLM()

    assert torch.equal(adapter(feature), feature)
    assert _trainable_scalar_count(adapter) == 32
    assert [name for name, _ in adapter.named_parameters()] == [
        "raw_scale",
        "raw_bias",
    ]


def test_decoder_film_has_live_first_step_gradients_for_both_vectors() -> None:
    feature = torch.arange(1, 1 + 16 * 2 * 3, dtype=torch.float32).reshape(
        1, 16, 2, 3
    )
    adapter = DecoderFiLM()

    adapter(feature).sum().backward()

    assert adapter.raw_scale.grad is not None
    assert adapter.raw_bias.grad is not None
    assert torch.count_nonzero(adapter.raw_scale.grad) == 16
    assert torch.count_nonzero(adapter.raw_bias.grad) == 16


@pytest.mark.parametrize("rank, expected_scalars", [(2, 32), (4, 64)])
def test_low_rank_mixer_is_identity_orthonormal_and_exact_size(
    rank: int,
    expected_scalars: int,
) -> None:
    feature = torch.randn(2, 16, 5, 7)
    adapter = LowRankResidualMixer(rank=rank)

    assert torch.equal(adapter(feature), feature)
    assert _trainable_scalar_count(adapter) == expected_scalars
    assert [name for name, _ in adapter.named_parameters()] == ["up.weight"]
    assert torch.count_nonzero(adapter.up.weight) == 0
    assert "down_basis" in dict(adapter.named_buffers())
    assert "down_basis" not in dict(adapter.named_parameters())

    basis = adapter.down_basis[:, :, 0, 0]
    identity = torch.eye(rank, dtype=basis.dtype, device=basis.device)
    assert torch.allclose(basis @ basis.T, identity, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("rank", [2, 4])
def test_low_rank_mixer_zero_up_still_has_nonzero_first_step_gradient(
    rank: int,
) -> None:
    feature = torch.arange(1, 1 + 16 * 3 * 3, dtype=torch.float32).reshape(
        1, 16, 3, 3
    )
    adapter = LowRankResidualMixer(rank=rank)

    adapter(feature).sum().backward()

    assert adapter.up.weight.grad is not None
    assert torch.linalg.vector_norm(adapter.up.weight.grad) > 0


def test_decoder_hook_preserves_source_identity_and_cleans_up_normally() -> None:
    torch.manual_seed(31)
    model = TinyDecoderModel().eval()
    image = torch.randn(1, 3, 8, 6)
    adapter = DecoderFiLM()
    hooks_before = _hook_count(model.decoder_0)

    with torch.no_grad():
        source = model(image)
        with decoder0_modulation(model, adapter):
            assert _hook_count(model.decoder_0) == hooks_before + 1
            adapted = model(image)
        restored = model(image)

    assert torch.equal(adapted, source)
    assert torch.equal(restored, source)
    assert _hook_count(model.decoder_0) == hooks_before
    assert not hasattr(model.decoder_0, "_cr_sitta_decoder0_modulation_token")


def test_decoder_hook_cleans_up_on_exception() -> None:
    model = TinyDecoderModel()
    adapter = LowRankResidualMixer(rank=2)
    hooks_before = _hook_count(model.decoder_0)

    with pytest.raises(RuntimeError, match="sentinel"):
        with decoder0_modulation(model, adapter):
            assert _hook_count(model.decoder_0) == hooks_before + 1
            raise RuntimeError("sentinel")

    assert _hook_count(model.decoder_0) == hooks_before
    assert not hasattr(model.decoder_0, "_cr_sitta_decoder0_modulation_token")


def test_decoder_hook_rejects_nested_cr_sitta_modulation_and_keeps_outer_alive() -> None:
    model = TinyDecoderModel()
    adapter = DecoderFiLM()
    hooks_before = _hook_count(model.decoder_0)

    with decoder0_modulation(model, adapter):
        with pytest.raises(Decoder0HookError, match="already has an active"):
            with decoder0_modulation(model, adapter):
                pass
        assert _hook_count(model.decoder_0) == hooks_before + 1

    assert _hook_count(model.decoder_0) == hooks_before
