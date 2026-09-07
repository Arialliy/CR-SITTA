#!/usr/bin/env python3
"""Run the single frozen O3 spatial model candidate on source train Pilot64.

No outer-target loader is imported or called here. Each episode captures two
frozen host feature maps; only the identity-initialized spatial kernel adapts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

DEFAULT_CONFIG = REPOSITORY / "configs/cr_sitta_o3_spatial_residual_v1.yaml"


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def tensor_hash(value) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def capture_d0(model, adapter, image):
    """Capture the original head input, removing the hook even on failure."""
    import torch
    if any(child.training for child in model.modules()) or any(
        p.requires_grad or p.grad is not None for p in model.parameters()
    ):
        raise RuntimeError("host must be frozen eval with empty gradient slots")
    if image.dtype != torch.float32 or tuple(image.shape) != (1, 3, 256, 256):
        raise RuntimeError("expected one float32 256-square observation")
    captured = []

    def capture(_module, args):
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("invalid output_0 input")
        captured.append(args[0].detach().clone())

    hook = model.output_0.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            logits = adapter.forward_logits(image)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError("expected exactly one D0 feature capture")
    feature = captured[0]
    if tuple(feature.shape) != (1, 16, 256, 256) or not bool(torch.isfinite(feature).all()):
        raise RuntimeError("invalid frozen D0 features")
    with torch.no_grad():
        if not torch.equal(model.output_0(feature), logits):
            raise RuntimeError("D0/head replay does not match original host")
    return feature, logits.detach().clone()


def validate_sample(sample, *, image_id, dataset, corruption, severity, seed):
    import torch
    expected = {"image", "image_id", "original_size", "dataset", "corruption", "severity", "seed"}
    if set(sample) != expected:
        raise RuntimeError("unexpected method-facing fields")
    if (str(sample["image_id"]), sample["dataset"], sample["corruption"], sample["severity"], sample["seed"]) != (
        image_id, dataset, corruption, severity, seed
    ):
        raise RuntimeError("frozen Pilot64 sample metadata/order changed")
    if not image_id or Path(image_id).name != image_id or image_id in (".", ".."):
        raise RuntimeError("unsafe image ID")
    value = sample["image"]
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or value.device.type != "cpu" or tuple(value.shape) != (3, 256, 256)
            or value.requires_grad or not bool(torch.isfinite(value).all())):
        raise RuntimeError("invalid method observation")


def save_mask(path: Path, probability) -> None:
    import numpy as np
    from PIL import Image
    values = np.asarray(probability)
    if values.shape != (1, 256, 256) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise RuntimeError("invalid mask probability")
    if ((values < 0) | (values > 1)).any():
        raise RuntimeError("invalid mask probability range")
    if path.exists():
        raise FileExistsError(path)
    # The original unified evaluator uses strict greater-than, not >=.
    Image.fromarray(((values[0] > 0.5).astype(np.uint8) * 255), mode="L").save(path)


def run_candidate(config_path: Path, dataset: str, device_name: str = "cuda:0") -> dict:
    import numpy as np
    import torch
    from analysis import spatial_residual_contract_v1 as protocol
    from scripts import run_p3_stage_b_screen_v1 as b3
    from scripts import run_p3_stage_b4_full_pilot64_v1 as b4
    from tta.adapters.decoder_spatial_residual_v1 import DecoderSpatialResidual
    from tta.spatial_residual_episode_v1 import run_episode
    from tta.views import validated_student_perturbations

    prepared = protocol.prepare_run(config_path, dataset)
    protocol.freeze_run(prepared)
    output, parent = Path(prepared["output"]), prepared["parent_contract"]
    ids = tuple(prepared["image_ids"])
    started, completed, accepted = time.perf_counter(), 0, 0
    state_manager = None
    spatial = None
    try:
        _, model, adapter, _unused_film, state_manager, device, wrapper = b3._build_runtime(parent, dataset, device_name)
        spatial = DecoderSpatialResidual().to(device).eval()
        write_json(output / "runtime.json", {
            **b3._runtime_environment_receipt(torch, device),
            "checkpoint_wrapper": wrapper,
            "source_state_sha256": state_manager.source_fingerprint.full_sha256,
            "torch_num_threads": torch.get_num_threads(),
        })
        teacher_manifest = b4._teacher_manifest(parent, dataset)
        perturbation = validated_student_perturbations((parent.raw["view_library"]["student_perturbation"],))[0]
        for corruption, severity in prepared["new_config"]["ordered_conditions"]:
            condition = b4._condition_key(corruption, severity)
            condition_output = output / "conditions" / condition
            condition_output.mkdir(parents=True, exist_ok=False)
            (condition_output / "masks").mkdir()
            arrays = {}
            for name, shape, dtype in (
                ("source_probabilities", (64, 1, 256, 256), "<f4"),
                ("post_probabilities", (64, 1, 256, 256), "<f4"),
                ("proxy_gradients", (64, 144), "<f8"),
                ("proposal_directions", (64, 144), "<f8"),
                ("endpoint_kernels", (64, 16, 1, 3, 3), "<f4"),
            ):
                arrays[name] = np.lib.format.open_memmap(condition_output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
            inputs = b3._method_input_dataset(parent, dataset, condition)
            if len(inputs) != 64:
                raise RuntimeError("input cache must contain exactly 64 train images")
            record = b3._teacher_condition_record(teacher_manifest, condition)
            teachers = b3._verified_teacher_array(parent, dataset, teacher_manifest, record, "source_probabilities")
            uncertainties = b3._verified_teacher_array(parent, dataset, teacher_manifest, record, "view_uncertainty")
            if teachers.shape != (64, 1, 256, 256) or uncertainties.shape != (64, 2, 1, 256, 256):
                raise RuntimeError("teacher array dimensions changed")
            condition_accepted = 0
            with (condition_output / "episodes.jsonl").open("x", encoding="utf-8") as log:
                for index, image_id in enumerate(ids):
                    sample = dict(inputs[index])
                    validate_sample(sample, image_id=image_id, dataset=dataset, corruption=corruption,
                                    severity=severity, seed=42)
                    observed = sample["image"].unsqueeze(0).to(device)
                    student = perturbation.forward(observed)
                    teacher = torch.from_numpy(np.array(teachers[index], copy=True)).unsqueeze(0).to(device)
                    uncertainty = torch.from_numpy(np.array(uncertainties[index, 0], copy=True)).unsqueeze(0).to(device)
                    state_manager.reset_to_source()
                    state_manager.assert_source_state()
                    try:
                        h, logits = capture_d0(model, adapter, observed)
                        student_h, _ = capture_d0(model, adapter, student)
                        result = run_episode(contract=parent, module=spatial, head=model.output_0,
                                             observed_features=h, student_features=student_h,
                                             teacher=teacher, uncertainty=uncertainty, source_logits=logits)
                        # Assert before restoring: a reset must not hide host mutation.
                        state_manager.assert_source_state()
                    finally:
                        spatial.reset_identity_()
                        state_manager.reset_to_source()
                        state_manager.assert_source_state()
                    diagnostics = result["diagnostics"]
                    if not diagnostics["finite"] or not diagnostics["episode_reset_exact"]:
                        raise RuntimeError("invalid or unrestored episode")
                    source, post = result["source_probabilities"][0].numpy(), result["post_probabilities"][0].numpy()
                    if not np.array_equal(source, teachers[index]):
                        raise RuntimeError("Source is not bit-exact to historical teacher")
                    arrays["source_probabilities"][index] = source
                    arrays["post_probabilities"][index] = post
                    arrays["proxy_gradients"][index] = result["proxy_gradient"].numpy()
                    arrays["proposal_directions"][index] = result["direction"].numpy()
                    arrays["endpoint_kernels"][index] = result["endpoint_kernel"].numpy()
                    save_mask(condition_output / "masks" / f"{image_id}.png", post)
                    row = {
                        "dataset": dataset, "condition": condition, "image_index": index, "image_id": image_id,
                        "input_tensor_sha256": tensor_hash(observed), "student_tensor_sha256": tensor_hash(student),
                        "source_feature_sha256": tensor_hash(h), "student_feature_sha256": tensor_hash(student_h),
                        "source_probability_sha256": b3._raw_array_sha256(source),
                        "post_probability_sha256": b3._raw_array_sha256(post), **diagnostics,
                    }
                    log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    log.flush()
                    completed += 1
                    accepted += int(diagnostics["accepted_update"])
                    condition_accepted += int(diagnostics["accepted_update"])
                    if (index + 1) % 16 == 0:
                        print(json.dumps({"dataset": dataset, "condition": condition, "image": index + 1,
                                          "completed": completed, "total": 832,
                                          "accepted": accepted, "seconds": round(time.perf_counter() - started, 2)}), flush=True)
            for array in arrays.values():
                array.flush()
            del arrays, teachers, uncertainties, inputs
        state_manager.assert_source_state()
        summary = {
            "dataset": dataset, "phase": "candidate", "candidate_id": "O3_DecoderSpatialResidual",
            "episode_count": completed, "condition_count": 13, "image_count_per_condition": 64,
            "image_ids": list(ids), "accepted_update_count": accepted, "no_update_count": completed - accepted,
            "method_label_accesses": 0, "outer_target_loader_calls": 0,
            "test_payload_opens": 0, "validation_payload_opens": 0,
            "source_state_restored": True, "predictions_complete": True,
            "no_validation_split": True, "paper_result": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        protocol.complete_run(prepared, summary)
        return summary
    except BaseException as exc:
        restored = None
        if state_manager is not None:
            try:
                state_manager.reset_to_source()
                state_manager.assert_source_state()
                if spatial is not None:
                    spatial.reset_identity_()
                restored = True
            except Exception:
                restored = False
        write_json(output / "ABORTED.json", {"error_type": type(exc).__name__, "error": str(exc),
                                            "completed_episodes": completed, "source_restored": restored,
                                            "paper_result": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True, choices=("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"))
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        from analysis.spatial_residual_contract_v1 import prepare_run
        prepared = prepare_run(args.config, args.dataset)
        print(json.dumps(prepared["preflight"], sort_keys=True, indent=2))
    else:
        print(json.dumps(run_candidate(args.config, args.dataset, args.device), sort_keys=True))


if __name__ == "__main__":
    main()
