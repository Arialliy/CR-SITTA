"""Controlled lifecycle for modulation immediately after ``decoder_0``."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from torch import Tensor, nn


class Decoder0HookError(RuntimeError):
    """Raised when the decoder modulation lifecycle contract is violated."""


_ACTIVE_TOKEN_ATTRIBUTE = "_cr_sitta_decoder0_modulation_token"


def _validate_decoder0(model: nn.Module) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    decoder = getattr(model, "decoder_0", None)
    if not isinstance(decoder, nn.Module):
        raise Decoder0HookError("model.decoder_0 must be a torch.nn.Module")
    return decoder


@contextmanager
def decoder0_modulation(
    model: nn.Module,
    modulator: nn.Module,
) -> Iterator[None]:
    """Temporarily insert ``modulator`` after exactly ``model.decoder_0``.

    A decoder may have at most one active CR-SITTA modulation context.  The
    hook and its lifecycle marker are removed in ``finally``, including when
    model inference or the body of the context raises.
    """

    decoder = _validate_decoder0(model)
    if not isinstance(modulator, nn.Module):
        raise TypeError("modulator must be a torch.nn.Module")
    if getattr(decoder, _ACTIVE_TOKEN_ATTRIBUTE, None) is not None:
        raise Decoder0HookError(
            "decoder_0 already has an active CR-SITTA modulation hook"
        )

    token = object()
    setattr(decoder, _ACTIVE_TOKEN_ATTRIBUTE, token)

    def hook(module: nn.Module, _inputs: tuple[object, ...], output: object) -> Tensor:
        if module is not decoder:
            raise Decoder0HookError("decoder hook fired for an unexpected module")
        if getattr(decoder, _ACTIVE_TOKEN_ATTRIBUTE, None) is not token:
            raise Decoder0HookError("decoder modulation lifecycle token changed")
        if not isinstance(output, Tensor):
            raise TypeError("model.decoder_0 must return a torch.Tensor")

        modulated = modulator(output)
        if not isinstance(modulated, Tensor):
            raise TypeError("decoder modulator must return a torch.Tensor")
        if modulated.shape != output.shape:
            raise Decoder0HookError(
                "decoder modulator must preserve the decoder feature shape"
            )
        if modulated.device != output.device:
            raise Decoder0HookError(
                "decoder modulator must preserve the decoder feature device"
            )
        if modulated.dtype != output.dtype:
            raise Decoder0HookError(
                "decoder modulator must preserve the decoder feature dtype"
            )
        return modulated

    try:
        handle = decoder.register_forward_hook(hook)
    except BaseException:
        if getattr(decoder, _ACTIVE_TOKEN_ATTRIBUTE, None) is token:
            delattr(decoder, _ACTIVE_TOKEN_ATTRIBUTE)
        raise

    try:
        yield
    finally:
        handle.remove()
        if getattr(decoder, _ACTIVE_TOKEN_ATTRIBUTE, None) is token:
            delattr(decoder, _ACTIVE_TOKEN_ATTRIBUTE)
