"""Capture detached D0 features *after* the unchanged per-image O3/P2 update.

The capture hook exists only for the final observed-image forward, never for
the original O3 gradient, Armijo or safety forwards. The original episodic
state manager restores the entire host on every exit. No label argument or
target loader is part of this bridge.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from scripts import run_p3_stage_b4_full_pilot64_v1 as b4


class O3EndpointFeatureError(RuntimeError):
    """The original O3 endpoint could not be captured without changing it."""


def _tensor(value: Any, *, name: str, shape: tuple[int, ...], device: torch.device) -> Tensor:
    if (not isinstance(value, Tensor) or value.layout != torch.strided
            or value.dtype != torch.float32 or tuple(value.shape) != shape
            or value.device != device or value.requires_grad or value.grad_fn is not None
            or not bool(torch.isfinite(value).all().item())):
        raise O3EndpointFeatureError(f"invalid detached float32 {name}")
    return value


def _cpu(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu").clone().contiguous()


def _cleanup_note(error: BaseException, note: str) -> None:
    """Keep the primary error, including on Python 3.10 without add_note."""
    try:
        add_note = getattr(error, "add_note", None)
        if callable(add_note):
            add_note(note)
        else:
            notes = getattr(error, "__notes__", None)
            error.__notes__ = [*(notes if isinstance(notes, list) else []), note]
    except Exception:
        # Error annotation must never replace the original execution failure.
        pass


def capture_o3_endpoint(
    *, contract: Any, model: nn.Module, adapter: Any, film: nn.Module,
    state_manager: Any, image: Tensor, student_image: Tensor,
    teacher: Tensor, uncertainty: Tensor,
) -> dict[str, Any]:
    """Return CPU float32 features/probabilities and the original O3 diagnostics.

    ``features`` is [1,16,256,256], probabilities are [1,1,256,256]. The learned
    residual branch must be applied by the caller *after* this function; it is
    deliberately absent from the original O3 objective and endpoint selection.
    """
    handle = None
    active_error: BaseException | None = None
    output: dict[str, Any] | None = None
    try:
        state_manager.reset_to_source()
        state_manager.assert_source_state()
        if torch.is_inference_mode_enabled():
            raise O3EndpointFeatureError("O3 requires autograd, not inference_mode")
        raw = getattr(contract, "raw", None)
        if (getattr(contract, "config_sha256", None) != b4.FROZEN_CONFIG_SHA256
                or not isinstance(raw, Mapping) or raw.get("protocol_id") != b4.PROTOCOL_ID):
            raise O3EndpointFeatureError("expected the original frozen B4 contract")
        if not isinstance(model, nn.Module) or getattr(adapter, "model", None) is not model:
            raise O3EndpointFeatureError("adapter must own the supplied original host")
        head = getattr(model, "output_0", None)
        if (not isinstance(head, nn.Conv2d) or head.in_channels != 16
                or head.out_channels != 1 or head.kernel_size != (1, 1)):
            raise O3EndpointFeatureError("expected the original 16-to-1 output_0 head")
        device = head.weight.device
        for name, value, shape in (
            ("image", image, (1, 3, 256, 256)),
            ("student_image", student_image, (1, 3, 256, 256)),
            ("teacher", teacher, (1, 1, 256, 256)),
            ("uncertainty", uncertainty, (1, 1, 256, 256)),
        ):
            _tensor(value, name=name, shape=shape, device=device)
        if bool(((teacher < 0) | (teacher > 1)).any().item()) or bool((uncertainty < 0).any().item()):
            raise O3EndpointFeatureError("teacher range or uncertainty is invalid")
        adapter.set_source_eval_mode()
        state_manager.assert_source_state()
        with torch.no_grad():
            source_logits = adapter.forward_logits(image)
            _tensor(source_logits, name="source logits", shape=(1, 1, 256, 256), device=device)
            source_probabilities = torch.sigmoid(source_logits)
        if not torch.equal(source_probabilities, teacher):
            raise O3EndpointFeatureError("native Source probabilities are not bit-exact to teacher")

        # Unlike b4._candidate_episode, this internal entry point leaves an
        # accepted P2 endpoint applied. Our finally block owns the reset.
        with torch.enable_grad():
            post, _proxy_gradient, _direction, old_diagnostics = b4._candidate_episode_from_source(
                contract=contract, model=model, adapter=adapter, film=film,
                state_manager=state_manager, image=image, student_image=student_image,
                teacher=teacher, uncertainty=uncertainty, source_logits=source_logits,
                candidate_id="O3_P2",
            )
        _tensor(post, name="original O3 probabilities", shape=(1, 1, 256, 256),
                device=torch.device("cpu"))
        if (not isinstance(old_diagnostics, Mapping) or old_diagnostics.get("finite") is not True
                or old_diagnostics.get("candidate_id") != "O3_P2"):
            raise O3EndpointFeatureError("original O3 diagnostics are invalid")
        if any(child.training for child in model.modules()):
            raise O3EndpointFeatureError("O3 endpoint must retain eval-mode host modules")
        captures: list[Tensor] = []

        def capture(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if len(args) != 1 or not isinstance(args[0], Tensor):
                raise O3EndpointFeatureError("invalid output_0 feature input")
            captures.append(args[0].detach().clone())

        handle = head.register_forward_pre_hook(capture)
        try:
            # P2 requires_grad flags may still be true. no_grad is sufficient;
            # do not reset/freeze parameters before capturing their endpoint.
            with torch.no_grad():
                endpoint_logits = adapter.forward_logits(image)
        finally:
            handle.remove()
            handle = None
        if len(captures) != 1:
            raise O3EndpointFeatureError("expected exactly one endpoint D0 capture")
        features = _tensor(captures[0], name="endpoint D0 features", shape=(1, 16, 256, 256), device=device)
        _tensor(endpoint_logits, name="endpoint logits", shape=(1, 1, 256, 256), device=device)
        with torch.no_grad():
            replay_logits = head(features)
            if not torch.equal(replay_logits, endpoint_logits):
                raise O3EndpointFeatureError("D0/head replay is not bit-exact to endpoint logits")
            endpoint_probabilities = _cpu(torch.sigmoid(endpoint_logits))
        if not torch.equal(endpoint_probabilities, post):
            raise O3EndpointFeatureError("captured endpoint is not bit-exact to original O3 probabilities")
        b4._check_no_update_endpoint(
            source_probability=_cpu(source_probabilities), post_probability=post,
            accepted_update=bool(old_diagnostics["accepted_update"]),
        )
        output = {
            "features": _cpu(features), "source_probabilities": _cpu(source_probabilities),
            "o3_probabilities": _cpu(post),
            "diagnostics": {**dict(old_diagnostics), "bridge_teacher_bit_exact": True,
                            "bridge_endpoint_bit_exact": True, "bridge_head_replay_bit_exact": True,
                            "bridge_capture_forward_count": 1},
        }
        return output
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if handle is not None:
            try:
                handle.remove()
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            state_manager.reset_to_source()
            state_manager.assert_source_state()
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            if active_error is None:
                raise O3EndpointFeatureError("O3 endpoint bridge failed exact Source cleanup") from cleanup_errors[0]
            for error in cleanup_errors:
                _cleanup_note(active_error, f"endpoint bridge cleanup also failed: {type(error).__name__}: {error}")
        elif output is not None:
            output["diagnostics"]["source_state_restored"] = True


__all__ = ["O3EndpointFeatureError", "capture_o3_endpoint"]
