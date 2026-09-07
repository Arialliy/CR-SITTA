"""Append-only R4 IPMA engineering integration; zero outer optimizer steps.

Default execution is read-only preflight. Real execution requires both explicit
flags. All 16 label-free predictions precede the separate source-fit8 GT phase;
check8 labels, test payloads and optimizer updates have no entry point here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CONFIG = ROOT / "configs/cr_sitta_ipma_micro16_v1.yaml"
PREDICTION_NAMES = ("source", "identity", "probe", "post")


def assert_pairing(actual: Mapping[str, Any], sealed: Mapping[str, Any], fields) -> None:
    for field in fields:
        if field not in actual or field not in sealed or actual[field] != sealed[field]:
            raise RuntimeError(f"sealed R2 replay mismatch: {field}")


def ordered_lf_rows(rows, records):
    expected = [r["image_id"] for r in records]
    indexed = {}
    for row in rows:
        if (row["dataset"], row["view"], row["probe_id"]) != ("NUDT-SIRST", "train_crop_224", "L4a"):
            continue
        if row["image_id"] in indexed:
            raise ValueError("duplicate frozen LF row")
        indexed[row["image_id"]] = row
    if any(image_id not in indexed for image_id in expected):
        raise ValueError("missing frozen LF input pair")
    return [indexed[image_id] for image_id in expected]


def validate_coverage(label_free, fit, image_ids, access, *, fit_complete=True):
    from analysis.ipma_micro16_data_v1 import ACCESS_FIELDS, FORBIDDEN_ACCESS_FIELDS
    if len(image_ids) != 16 or len(set(image_ids)) != 16:
        raise ValueError("exactly 16 unique frozen image IDs required")
    if [r["image_id"] for r in label_free] != image_ids:
        raise ValueError("label-free output order/coverage mismatch")
    if [r["episode_index"] for r in label_free] != list(range(16)):
        raise ValueError("label-free episode index mismatch")
    if [r["image_id"] for r in fit] != (image_ids[:8] if fit_complete else []):
        raise ValueError("fit-only label coverage mismatch")
    if fit_complete and [r["episode_index"] for r in fit] != list(range(8)):
        raise ValueError("fit-only episode index mismatch")
    if set(access) != set(ACCESS_FIELDS) or any(type(n) is not int or n < 0 for n in access.values()):
        raise ValueError("invalid access receipt")
    if any(access[name] != 0 for name in FORBIDDEN_ACCESS_FIELDS):
        raise ValueError("forbidden check/test/validation access")
    expected_masks = 8 if fit_complete else 0
    if access["train_image_opens"] != 16 or access["train_mask_opens"] != expected_masks or access["fit_mask_opens"] != expected_masks:
        raise ValueError("unexpected source-train payload access count")


def save_tensor_artifact(path: Path, payload: dict) -> None:
    import torch
    with path.open("xb") as handle:
        torch.save(payload, handle)


def native_prediction(logits):
    import torch
    if logits.dtype != torch.float32 or logits.ndim != 4 or logits.shape[:2] != (1, 1):
        raise ValueError("native prediction requires [1,1,H,W] float32 logits")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("non-finite prediction logits")
    probabilities = logits.detach().sigmoid()
    return probabilities, (probabilities > 0.5).to(torch.uint8).mul(255)


def save_prediction_set(output: Path, image_id: str, tensors, *, probabilities=None) -> dict:
    import torch
    from PIL import Image
    from analysis.d0a_v7_common import binding
    paths = {}
    for name in PREDICTION_NAMES:
        if probabilities is None:
            _, mask = native_prediction(tensors[f"{name}_logits"])
        else:
            probability = probabilities[name]
            if probability.dtype != torch.float32 or probability.shape != tensors[f"{name}_logits"].shape:
                raise ValueError("materialized native probability precision/grid mismatch")
            if not bool(torch.isfinite(probability).all()) or bool(((probability < 0) | (probability > 1)).any()):
                raise ValueError("invalid materialized native probabilities")
            mask = (probability > 0.5).to(torch.uint8).mul(255)
        path = output / "predictions" / name / f"{image_id}.png"
        with path.open("xb") as handle:
            Image.fromarray(mask.cpu().numpy()[0, 0]).save(handle, format="PNG")
        paths[name] = binding(path)
    return paths


def require_unlabeled_phase_sealed(output, label_free, image_ids, access):
    from analysis.lf_repair_contract_v2 import assert_bindings
    validate_coverage(label_free, [], image_ids, access, fit_complete=False)
    if not (output / "LABEL_FREE_COMPLETE.json").is_file():
        raise RuntimeError("fit GT requires completed label-free phase receipt")
    for row in label_free:
        if set(row["prediction_masks"]) != set(PREDICTION_NAMES):
            raise RuntimeError("all prediction masks must exist before fit GT")
        assert_bindings([*row["prediction_masks"].values(), row["cache"], row["episode_tensors"]])


def load_frozen_host(config, device):
    import torch
    from analysis.d0a_v7_common import absolute, sha256_file
    from export_cr_sitta_d0a_safe_checkpoint import _weights_only_load, _validate_safe_payload
    from model.MSHNet_NSFPN import MSHNet_NSFPN
    host = config["host"]
    path = absolute(host["safe_checkpoint"])
    if sha256_file(path) != host["safe_checkpoint_sha256"]:
        raise RuntimeError("safe checkpoint changed after freeze")
    provenance, state = _validate_safe_payload(_weights_only_load(path))
    expected = {"architecture": "MSHNet_NSFPN", "dataset": "NUDT-SIRST", "epoch": 1000,
        "global_optimizer_step": 41000, "state_dict_keys": 505,
        "selection_rule": "fixed_final_epoch_train_only", "test_selected": False,
        "source_checkpoint_sha256": host["source_checkpoint_sha256"],
        "run_contract_sha256": host["run_contract_sha256"]}
    assert_pairing(provenance, expected, expected)
    if any(not bool(torch.isfinite(t).all()) for t in state.values()):
        raise RuntimeError("non-finite frozen host weights")
    model = MSHNet_NSFPN(input_channels=3)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    return model, provenance


def metric_protocol():
    from metrics.irstd_metrics import IRSTDEvaluationProtocol
    return IRSTDEvaluationProtocol(fixed_probability_threshold=0.5,
        froc_probability_thresholds=(0.5,), connectivity=2,
        max_centroid_distance=3.0, min_component_area=1)


def nonpromoting_summary(gate, *, access, label_free, fit, metrics, transitions, replay, host_binding):
    return {"schema_version": 1, "protocol_id": "cr-sitta-ipma-micro16-v1",
        "stage": "R4_engineering_only", "status": "engineering_passed" if gate["engineering_passed"] else "engineering_failed",
        "gate": gate, "image_count": len(label_free), "fit_image_count": len(fit),
        "access": dict(access), "host": host_binding, "A_B_A_replay": replay,
        "fit8_metrics": metrics, "fit8_target_transitions": transitions,
        "label_free_signal_count": sum(r["label_free_signal_passed"] for r in label_free),
        "fit8_empty_target_count": sum(r["target_empty"] for r in fit),
        "prediction_mask_count": len(label_free) * len(PREDICTION_NAMES),
        "native_threshold": "torch.sigmoid(float32_logits) > 0.5",
        "outer_optimizer_steps": 0, "ipma_meta_training_executed": False,
        "ipma_meta_training_allowed": False, "requires_frozen_r5_contract": True,
        "full_source_training_allowed": False, "formal_test_allowed": False,
        "new_validation_split": False, "paper_result": False,
        "task_improvement_used_for_gate": False,
        "result_scope": "untrained fixed basis / source-train engineering diagnostic; not a performance benchmark"}


def execute_prepared(prepared, device):
    import numpy as np
    from PIL import __version__ as pillow_version
    import scipy
    import torch
    from analysis.d0a_v7_common import binding, write_json_new, write_jsonl_new
    from analysis.audit_lf_repair_v2 import build_candidate_artifacts, stable_probe_seed
    from analysis.ipma_engineering_checks_v1 import (label_free_episode, supervised_fit_diagnostic,
        aggregate_engineering_gate, validate_engineering_configuration, module_signature, tensor_sha256)
    from analysis.ipma_micro16_contract_v1 import freeze_run, complete_run
    from analysis.ipma_micro16_data_v1 import (ACCESS_FIELDS, load_observation, load_fit_target,
        extract_d0_source_features, state_digest, assert_state_unchanged)
    from metrics.irstd_metrics import UnifiedResearchEvaluator
    from metrics.target_transitions import TargetTransitionEvaluator
    from model.ipma_d0_adapter_v1 import IdentityMetaAdapter
    from tta.deteriorations.image_space import imagenet_normalize

    config, lf_config = prepared["new_config"], prepared["lf_config"]
    validate_engineering_configuration(config)
    if device != config["runtime"]["device"] or device != "cuda:0":
        raise ValueError("native integration requires the frozen cuda:0 float32 path")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ValueError("set frozen CUBLAS_WORKSPACE_CONFIG and CUDA_VISIBLE_DEVICES=0 before Python")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    prepared["contract"]["runtime"] = {**config["runtime"], "python": platform.python_version(),
        "python_executable": sys.executable, "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pillow": pillow_version,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "autocast_enabled": torch.is_autocast_enabled() or torch.is_autocast_cpu_enabled()}
    if prepared["contract"]["runtime"]["autocast_enabled"]:
        raise ValueError("autocast must remain disabled")
    frozen_rows = ordered_lf_rows(prepared["legacy_lf_image_rows"], prepared["records"])
    freeze_run(prepared)  # Must precede weight loading or real image decoding.
    output = prepared["output"]
    access = {name: 0 for name in ACCESS_FIELDS}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("frozen native GPU is unavailable")
        torch.cuda.set_device(device)
        model, provenance = load_frozen_host(config, device)
        host_before = state_digest(model)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config["adapter"]["initialization_seed"])
            adapter = IdentityMetaAdapter(channels=16, rank=4)
        adapter.eval().to(device)
        adapter_before = module_signature(adapter)
        for path in (output / "cache", output / "episodes", output / "fit"):
            path.mkdir()
        (output / "predictions").mkdir()
        for name in PREDICTION_NAMES:
            (output / "predictions" / name).mkdir()
        save_tensor_artifact(output / "initial_basis.pth.tar", {"trained": False,
            "outer_optimizer_steps": 0, "initialization_seed": 42,
            "state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()}})
        save_tensor_artifact(output / "frozen_head.pth.tar", {"host": config["host"],
            "state_dict": {k: v.detach().cpu() for k, v in model.output_0.state_dict().items()}})
        write_json_new(output / "HOST_LOAD_RECEIPT.json", {"provenance": provenance,
            "strict_original_architecture_load": True, "host_state": host_before,
            "initial_adapter_state": adapter_before, "weights_only": True,
            "gpu_name": torch.cuda.get_device_name(0), "compute_device": device,
            "SFS_backward_executed": False, "native_head_dtype": "float32"})
        candidates = [c for c in lf_config["variants"] if c["probe_id"] == "L4a"]
        if len(candidates) != 1:
            raise RuntimeError("missing unique approved LF candidate")
        label_free, cache = [], []
        for index, (record, sealed) in enumerate(zip(prepared["records"], frozen_rows, strict=True)):
            normalized, metadata = load_observation(record, prepared["legacy_config"], access)
            assert_pairing(metadata, sealed["input_metadata"], ("input_tensor_sha256", "view",
                "augmentation_seed", "historical_training_random_stream_replay"))
            seed = stable_probe_seed(lf_config["probe_seed_namespace"], lf_config["global_seed"],
                "NUDT-SIRST", record["image_id"], "train_crop_224")
            with torch.no_grad():
                probe = build_candidate_artifacts(normalized, candidate=candidates[0], operator=lf_config["operator"], seed=seed)
            assert_pairing(probe["statistics"], sealed, ("probe_seed", "clean_physical_tensor_sha256",
                "preclip_physical_tensor_sha256", "postclip_physical_tensor_sha256", "random_field_sha256", "gain_sha256"))
            image = normalized.unsqueeze(0).to(device)
            probe_image = imagenet_normalize(probe["postclip"]).to(device)
            h, source_logits = extract_d0_source_features(model, image)
            hp, probe_logits = extract_d0_source_features(model, probe_image)
            hr, lr = extract_d0_source_features(model, image)
            hpr, lpr = extract_d0_source_features(model, probe_image)
            if not all(torch.equal(a, b) for a, b in ((h, hr), (hp, hpr), (source_logits, lr), (probe_logits, lpr))):
                raise RuntimeError("original host forward replay is not exact")
            teacher = source_logits.sigmoid().detach()
            receipt, tensors = label_free_episode(adapter, model.output_0, h, hp, teacher)
            if not torch.equal(tensors["source_logits"], source_logits) or not torch.equal(tensors["probe_logits"], probe_logits):
                raise RuntimeError("cached native head does not equal original host output")
            cached = {"h_observed": h.cpu(), "h_probe": hp.cpu(), "teacher": teacher.cpu(),
                "source_logits": source_logits.cpu(), "probe_logits": probe_logits.cpu()}
            cache_path = output / "cache" / f"{record['image_id']}.pth.tar"
            save_tensor_artifact(cache_path, {"image_id": record["image_id"], "metadata": metadata,
                "host_checkpoint_sha256": config["host"]["safe_checkpoint_sha256"],
                "probe_statistics": probe["statistics"], "contains_ground_truth": False, "tensors": cached})
            tensor_path = output / "episodes" / f"{record['image_id']}.pth.tar"
            cpu_tensors = {key: value.detach().cpu() for key, value in tensors.items()}
            # Materialize once on the native GPU, before any fit GT is decoded.
            # Masks and evaluators consume these SAME probabilities. Never
            # recompute sigmoid on CPU after copying the logits to the cache.
            native_probabilities = {name: native_prediction(tensors[f"{name}_logits"])[0]
                                    for name in PREDICTION_NAMES}
            cpu_tensors.update({f"{name}_probabilities": value.detach().cpu()
                                for name, value in native_probabilities.items()})
            save_tensor_artifact(tensor_path, {"image_id": record["image_id"],
                "episode_local_delta_not_model_weights": True, "tensors": cpu_tensors})
            row = {**receipt, "episode_index": index, "image_id": record["image_id"],
                "source_role": metadata["source_role"], "input_metadata": metadata,
                "sealed_R2_input_pair_exact": True, "sealed_L4a_probe_pair_exact": True,
                "host_forward_repeat_exact": True, "original_host_head_replay_exact": True,
                "cache": binding(cache_path), "episode_tensors": binding(tensor_path),
                "prediction_masks": save_prediction_set(output, record["image_id"], tensors,
                    probabilities=native_probabilities)}
            write_json_new(output / "episodes" / f"{record['image_id']}.json", row)
            label_free.append(row)
            cache.append((cached, cpu_tensors, metadata))
            print(json.dumps({"phase": "label_free", "completed": index + 1, "total": 16,
                "image_id": record["image_id"], "identity_exact": receipt["identity_exact"],
                "signal_passed": receipt["label_free_signal_passed"], "train_mask_opens": access["train_mask_opens"]}), flush=True)
        ids = config["sampling"]["ids"]
        validate_coverage(label_free, [], ids, access, fit_complete=False)
        write_jsonl_new(output / "label_free_per_image.jsonl", label_free)
        write_json_new(output / "LABEL_FREE_COMPLETE.json", {"image_ids": ids, "access": dict(access),
            "all_16_predictions_saved_before_first_fit_GT": True,
            "created_at_utc": datetime.now(timezone.utc).isoformat()})
        require_unlabeled_phase_sealed(output, label_free, ids, access)

        fit = []
        protocol = metric_protocol()
        evaluators = {name: UnifiedResearchEvaluator(protocol) for name in ("source", "post")}
        transitions = TargetTransitionEvaluator(protocol, comparison_label="D0-A fixed1000→untrained IPMA one-step")
        for index, record in enumerate(prepared["records"][:8]):
            cached, tensors, metadata = cache[index]
            target, target_metadata = load_fit_target(record, prepared["legacy_config"], access,
                native_image_size=metadata["native_image_size"])
            assert_pairing(target_metadata, frozen_rows[index]["input_metadata"],
                ("target_tensor_sha256", "augmentation_seed", "view"))
            receipt, diagnostic = supervised_fit_diagnostic(adapter, model.output_0,
                cached["h_observed"].to(device), cached["h_probe"].to(device),
                cached["teacher"].to(device), target.unsqueeze(0).to(device), config)
            if not all(torch.equal(diagnostic[a].cpu(), tensors[b]) for a, b in
                       (("pre_logits", "source_logits"), ("post_logits", "post_logits"), ("delta1", "delta1"))):
                raise RuntimeError("fit GT changed precomputed label-free outputs")
            probabilities = {name: tensors[f"{name}_probabilities"] for name in evaluators}
            image_metrics = {}
            for name, evaluator in evaluators.items():
                evaluator.update_probabilities(probabilities[name], target)
                per_image = UnifiedResearchEvaluator(protocol)
                per_image.update_probabilities(probabilities[name], target)
                image_metrics[name] = per_image.compute().to_dict()
            transition = transitions.update_probabilities(probabilities["source"], probabilities["post"], target,
                image_id=record["image_id"]).to_dict()
            row = {**receipt, "episode_index": index, "image_id": record["image_id"],
                "source_role": "meta_fit_train8", "target_metadata": target_metadata,
                "label_free_output_unchanged_after_GT": True, "metrics": image_metrics, "target_transitions": transition}
            fit.append(row)
            write_json_new(output / "fit" / f"{record['image_id']}.json", row)
            save_tensor_artifact(output / "fit" / f"{record['image_id']}.pth.tar",
                {"image_id": record["image_id"], "optimizer_steps_applied": 0,
                 "tensors": {key: value.detach().cpu() for key, value in diagnostic.items()}})
            print(json.dumps({"phase": "fit_meta_gradient", "completed": index + 1, "total": 8,
                "image_id": record["image_id"], "meta_gradient_norm": receipt["meta_gradient_norm"],
                "finite_difference_passed": receipt["finite_difference"]["all_directions_passed"]}), flush=True)
        validate_coverage(label_free, fit, ids, access)
        first, first_tensors, _ = cache[0]
        replay_receipt, replay_tensors = label_free_episode(adapter, model.output_0,
            first["h_observed"].to(device), first["h_probe"].to(device), first["teacher"].to(device))
        replay_exact = all(torch.equal(value.cpu(), first_tensors[key]) for key, value in replay_tensors.items())
        if not replay_exact:
            raise RuntimeError("A→B→A episode replay changed")
        assert_state_unchanged(model, host_before)
        if module_signature(adapter) != adapter_before:
            raise RuntimeError("adapter basis changed despite zero outer optimizer steps")
        replay = {"exact": replay_exact, "image_id": ids[0], "intervening_other_images": 15,
            "intervening_fit_diagnostics": 8, "model_parameters_and_all_buffers_unchanged": True,
            "adapter_phi_and_buffers_unchanged": True, "optimizer_created": False,
            "optimizer_state": "not_applicable_no_optimizer_in_R4",
            "delta_state": "fresh_zero_input_every_episode_not_persisted_in_model",
            "receipt": replay_receipt}
        gate = aggregate_engineering_gate(label_free, fit)
        gate.update({"exact_image_coverage": True, "data_access_passed": True,
            "sealed_input_and_probe_pairing_passed": True, "A_B_A_exact": replay_exact,
            "full_frozen_host_unchanged": True, "adapter_phi_unchanged": True})
        metrics = {name: evaluator.compute().to_dict() for name, evaluator in evaluators.items()}
        summary = nonpromoting_summary(gate, access=access, label_free=label_free, fit=fit,
            metrics=metrics, transitions=transitions.compute().to_dict(), replay=replay, host_binding=config["host"])
        write_jsonl_new(output / "fit_per_image.jsonl", fit)
        write_json_new(output / "IPMA_ENGINEERING_GATE.json", gate)
        write_json_new(output / "STATE_REPLAY.json", replay)
        write_json_new(output / "ACCESS_AUDIT.json", {"access": access,
            "all_16_predictions_saved_before_first_fit_GT": True, "check8_GT_decoded": False,
            "raw_train_payload_hash_verification_is_separate_from_decode_counts": True})
        complete_run(prepared, summary)
        return summary
    except Exception as error:
        if not (output / "ERROR.json").exists():
            write_json_new(output / "ERROR.json", {"error_type": type(error).__name__, "error": str(error),
                "access": access, "complete": False, "ipma_meta_training_allowed": False,
                "formal_test_allowed": False, "full_source_training_allowed": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def run(args):
    from analysis.ipma_micro16_contract_v1 import prepare_run
    if args.execute and not args.engineering_only:
        raise ValueError("real execution requires --execute --engineering-only")
    prepared = prepare_run(args.config)
    if not args.execute:
        return prepared["preflight"]
    return execute_prepared(prepared, args.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--engineering-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
