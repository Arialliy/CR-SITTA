"""CPU synthetic checks; these tests never open research image/GT payloads."""
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from scripts.run_o3_spatial_residual_v1 import capture_d0, validate_sample, save_mask, write_json
from tta.model_adapter import IRSTDModelAdapter


class Host(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 16, 1)
        self.output_0 = nn.Conv2d(16, 1, 1)
        self.fail = False

    def forward(self, x, warm):
        h = self.encoder(x)
        result = self.output_0(h)
        if self.fail:
            raise RuntimeError("synthetic host failure")
        return [], result


def test_capture_identity_and_hook_removal():
    host = Host()
    adapter = IRSTDModelAdapter(host)
    adapter.set_source_eval_mode()
    x = torch.zeros(1, 3, 256, 256)
    h, logits = capture_d0(host, adapter, x)
    assert not h.requires_grad and not logits.requires_grad
    assert torch.equal(host.output_0(h), logits)
    assert not host.output_0._forward_pre_hooks
    host.fail = True
    with pytest.raises(RuntimeError, match="synthetic"):
        capture_d0(host, adapter, x)
    assert not host.output_0._forward_pre_hooks


def test_capture_rejects_trainable_host():
    host = Host().eval()
    with pytest.raises(RuntimeError, match="frozen"):
        capture_d0(host, IRSTDModelAdapter(host), torch.zeros(1, 3, 256, 256))


def test_sample_allowlist():
    sample = dict(image=torch.zeros(3, 256, 256), image_id="001", original_size=[256,256],
                  dataset="NUDT-SIRST", corruption="clean", severity=0, seed=42)
    expected = dict(image_id="001", dataset="NUDT-SIRST", corruption="clean", severity=0, seed=42)
    validate_sample(sample, **expected)
    with pytest.raises(RuntimeError, match="fields"):
        validate_sample({**sample, "target": torch.zeros(1)}, **expected)
    with pytest.raises(RuntimeError, match="metadata"):
        validate_sample({**sample, "image_id": "002"}, **expected)


def test_mask_strict_threshold_and_append_only(tmp_path):
    p = np.zeros((1, 256, 256), dtype=np.float32)
    p[0, 0, :3] = [0.49, 0.5, np.nextafter(np.float32(0.5), np.float32(1))]
    path = tmp_path / "p.png"
    save_mask(path, p)
    with Image.open(path) as handle:
        assert np.array(handle)[0, :3].tolist() == [0, 0, 255]
    with pytest.raises(FileExistsError):
        save_mask(path, p)


def test_json_no_overwrite_or_nan(tmp_path):
    path = tmp_path / "receipt.json"
    write_json(path, {"ok": True})
    with pytest.raises(FileExistsError):
        write_json(path, {})
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"bad": float("nan")})
