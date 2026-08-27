"""A stable logits and BatchNorm interface for infrared target detectors."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class IRSTDModelAdapter:
    """Wrap NS-FPN-like models behind a method-independent interface.

    The adapter deliberately owns no model parameters. This avoids changing
    checkpoint key prefixes and makes source, baseline, Oracle, and CR-SITTA
    paths call exactly the same forward implementation.
    """

    def __init__(self, model: nn.Module, *, warm_flag: bool = False) -> None:
        self.model = model
        self.warm_flag = bool(warm_flag)

    def forward_logits(self, image: Tensor) -> Tensor:
        """Return raw logits with shape ``[B, 1, H, W]``.

        NS-FPN returns ``(auxiliary_outputs, final_logits)``. A tensor return is
        also accepted so the same adapter can support a second backbone later.
        No sigmoid, thresholding, detaching, or no-grad context is applied here.
        """

        if not isinstance(image, Tensor) or image.ndim != 4:
            raise ValueError("image must be a tensor with shape [B, C, H, W]")

        output = self.model(image, self.warm_flag)
        if isinstance(output, Tensor):
            logits = output
        elif isinstance(output, Sequence) and len(output) == 2:
            logits = output[1]
        else:
            raise TypeError(
                "model must return logits or an (auxiliary, logits) pair"
            )

        if not isinstance(logits, Tensor):
            raise TypeError("the model's final output must be a torch.Tensor")
        if logits.ndim != 4 or logits.shape[0] != image.shape[0] or logits.shape[1] != 1:
            raise ValueError(
                "final logits must have shape [B, 1, H, W] with the input batch size"
            )
        return logits

    @staticmethod
    def logits_to_prob(logits: Tensor) -> Tensor:
        """Convert raw logits to probabilities at the single canonical site."""

        if not isinstance(logits, Tensor) or logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("logits must have shape [B, 1, H, W]")
        return torch.sigmoid(logits)

    def collect_adaptable_params(self) -> tuple[list[nn.Parameter], list[str]]:
        """Return affine parameters of every ``BatchNorm2d`` in stable order."""

        parameters: list[nn.Parameter] = []
        names: list[str] = []
        for module_name, module in self.model.named_modules():
            if not isinstance(module, nn.BatchNorm2d) or not module.affine:
                continue
            prefix = f"{module_name}." if module_name else ""
            if module.weight is not None:
                parameters.append(module.weight)
                names.append(f"{prefix}weight")
            if module.bias is not None:
                parameters.append(module.bias)
                names.append(f"{prefix}bias")
        return parameters, names

    def set_source_eval_mode(self) -> None:
        """Configure deterministic source-only inference and freeze parameters."""

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for module in self.model.modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if module.running_mean is None or module.running_var is None:
                raise ValueError(
                    "source evaluation requires BatchNorm running statistics"
                )
            # track_running_stats is a Python attribute, not a state_dict entry.
            # Restore it explicitly after a batch-stat TENT episode.
            module.track_running_stats = True
            module.eval()

    def set_adabn_mode(self) -> None:
        """Use per-image spatial BN statistics without changing any state value.

        The model remains globally in evaluation mode and every learnable
        parameter stays frozen.  Only ``BatchNorm2d`` modules enter training
        mode, with running-stat tracking disabled.  PyTorch therefore computes
        mean/variance from the current ``[1,C,H,W]`` activation while receiving
        no running buffers to update.  Existing Source buffers are deliberately
        retained so the episodic state manager can fingerprint and restore them.
        """

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        batchnorm_count = 0
        for module in self.model.modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            batchnorm_count += 1
            if (
                module.running_mean is None
                or module.running_var is None
                or module.num_batches_tracked is None
            ):
                raise ValueError(
                    "episodic AdaBN requires intact Source running-stat buffers"
                )
            module.train()
            module.track_running_stats = False

        if batchnorm_count == 0:
            raise ValueError("episodic AdaBN requires at least one BatchNorm2d")

    def set_tent_mode(self, use_batch_stats: bool) -> None:
        """Enable only BN affine gradients under one of the frozen BN protocols.

        ``use_batch_stats=True`` puts only BatchNorm2d modules in training mode
        and disables running-stat updates. Other modules, including dropout,
        remain in evaluation mode. ``False`` keeps the stored source statistics.
        Existing running buffers are retained in both modes so a later episodic
        state manager can restore the complete source state exactly.
        """

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        for module in self.model.modules():
            if not isinstance(module, nn.BatchNorm2d):
                continue
            if not module.affine or module.weight is None or module.bias is None:
                raise ValueError("all adaptable BatchNorm2d modules must be affine")
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)
            if use_batch_stats:
                module.train()
                module.track_running_stats = False
            else:
                if module.running_mean is None or module.running_var is None:
                    raise ValueError(
                        "source-stat protocol requires BatchNorm running statistics"
                    )
                module.eval()
                module.track_running_stats = True
