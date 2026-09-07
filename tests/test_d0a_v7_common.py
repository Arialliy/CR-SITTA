import json
import os
from pathlib import Path
import random
import pytest
from analysis import d0a_v7_common as common


def test_freeze_complete_and_no_overwrite(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("constant")
    output = tmp_path / "new_run"
    common.freeze_run(output, {"purpose": "test"}, [common.binding(source)])
    common.complete_run(output, {"ok": True})
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["complete"]
    assert common.sha256_file(output / "artifact_manifest.json") == complete["artifact_manifest"]["sha256"]
    with pytest.raises(FileExistsError):
        common.reserve_output(output)
    with pytest.raises(FileExistsError):
        common.freeze_run(output, {}, [])


def test_changed_input_cannot_complete(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("before")
    output = tmp_path / "run"
    common.freeze_run(output, {}, [common.binding(source)])
    source.write_text("after")
    with pytest.raises(RuntimeError):
        common.complete_run(output, {})
    assert not (output / "COMPLETE.json").exists()


@pytest.mark.skipif(os.environ.get("NS_FPN_RUN_LOCAL_ARTIFACT_TESTS") != "1",
                    reason="requires the private frozen Pilot64 manifest")
def test_actual_pilot_metadata_is_train_only():
    config = common.read_config()
    for dataset in config["datasets"]:
        records = common.load_pilot_records(dataset, config)
        assert len(records) == len({r["image_id"] for r in records}) == 64


def test_nontrain_record_rejected_without_image_open(monkeypatch, tmp_path):
    from PIL import Image
    config = common.read_config()
    ids = tmp_path / "train_ids.txt"
    ids.write_text("allowed_train_id\n")
    config["datasets"]["fixture"] = {"pilot_ids": str(ids), "train_split": str(ids)}
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("payload opened"))
    with pytest.raises(ValueError):
        common.load_sample({"dataset": "fixture", "image_id": "not_a_pilot", "image_path": "/no", "mask_path": "/no"},
                           "full_256", config, {})


def test_synthetic_crop_reproducible_and_rng_restored(tmp_path):
    import numpy as np
    import torch
    from PIL import Image
    root = tmp_path / "fake"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    rgb = np.arange(256*256*3, dtype=np.uint8).reshape(256,256,3)
    Image.fromarray(rgb).save(root / "images/a.png")
    Image.fromarray((rgb[:,:,0]>128).astype(np.uint8)*255).save(root / "masks/a.png")
    split = tmp_path / "train.txt"
    split.write_text("a\n")
    config = common.read_config()
    config["datasets"]["fixture"] = {"root": str(root), "pilot_ids": str(split), "train_split": str(split)}
    record = {"dataset":"fixture", "image_id":"a", "image_path":str(root/"images/a.png"), "mask_path":str(root/"masks/a.png")}
    before = random.getstate()
    access = {}
    x, y, m = common.load_sample(record, "train_crop_224", config, access)
    x2,y2,m2 = common.load_sample(record, "train_crop_224", config, access)
    assert random.getstate() == before
    assert torch.equal(x,x2) and torch.equal(y,y2) and m==m2
    assert x.shape == (3,224,224) and y.shape == (1,224,224)
    assert access["train_image_opens"] == access["train_mask_opens"] == 2
    assert access["test_image_opens"] == access["test_mask_opens"] == 0
