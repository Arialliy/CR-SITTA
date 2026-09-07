"""Train Pilot64 visibility audit of the unchanged D0-A LF/HF probes.

This is diagnostic evidence, not training, test evaluation, or proof that GT
semantics change.  In particular, an unchanged GT tensor does not establish
that its target remains visible.  No target is selected away from the report.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import __version__ as pillow_version
import scipy
from scipy import ndimage
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_cr_sitta_d0a import build_deteriorated_view, derive_probe_seed
from tta.deteriorations import (
    imagenet_denormalize,
    imagenet_normalize,
    inject_high_frequency_noise_components,
    mask_low_frequency_amplitude,
)


VIEWS = ("full_256", "train_crop_224")
PROBES = ("clean", "lf_mask", "hf_noise")
FORBIDDEN_ACCESS_FIELDS = ("test_split_reads", "test_image_opens", "test_mask_opens",
                           "validation_split_reads", "validation_image_opens", "validation_mask_opens")
DEFAULT_CONFIG = ROOT / "configs/cr_sitta_d0a_v7_diagnostics_v1.yaml"
VISIBILITY_DEFAULTS = {
    "contrast_epsilon": 1.0e-8,
    "min_clean_contrast": 1.0 / 255.0,
    "retention_threshold": 0.30,
    "failure_fraction_threshold": 0.10,
    "clipping_increase_threshold": 0.10,
    "ring_outer_radius": 5,
    "ring_inner_radius": 1,
}


def tensor_sha256(value: Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    descriptor = json.dumps(
        {"shape": list(array.shape), "dtype": str(array.dtype)}, sort_keys=True
    ).encode()
    return hashlib.sha256(descriptor + b"\n" + array.tobytes()).hexdigest()


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def distribution(values: Iterable[float | None]) -> dict[str, Any]:
    values = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "minimum": None, "p25": None,
                "median": None, "p75": None, "maximum": None}
    if not np.isfinite(values).all():
        raise ValueError("non-finite diagnostic statistic")
    quantiles = np.quantile(values, [0, 0.25, 0.5, 0.75, 1])
    return {"count": len(values), "mean": float(values.mean()),
            **dict(zip(("minimum", "p25", "median", "p75", "maximum"), map(float, quantiles)))}


def validate_diagnostic_parameters(config: Mapping[str, Any]) -> None:
    expected_probes = {"lf_mask": {"mask_ratio": 0.20, "keep_probability": 0.50},
                       "hf_noise": {"target_rms": 0.02, "low_cut_ratio": 0.20}}
    if config["probes"] != expected_probes:
        raise ValueError("visibility audit must use the unchanged frozen D0-A probes")
    if config["global_seed"] != 42 or config["human_epoch"] != 1000:
        raise ValueError("visibility audit requires the frozen seed42 epoch1000 endpoint")
    if config["probe_seed_namespace"] != "cr-sitta-d0a-supervised-lfhf-train-1000e-v2":
        raise ValueError("visibility audit must reuse the original v2 probe seed namespace")
    for name, expected in VISIBILITY_DEFAULTS.items():
        if config["visibility"].get(name) != expected:
            raise ValueError(f"visibility threshold/geometry changed: {name}")


def _spectrum(image: Tensor) -> Tensor:
    return torch.fft.fftshift(
        torch.fft.fft2(image, dim=(-2, -1), norm="ortho"), dim=(-2, -1)
    )


def lf_region(height: int, width: int, mask_ratio: float, device: torch.device) -> Tensor:
    """The exact nominal rectangle used by the frozen LF implementation."""
    half_h = max(1, int(round(height * mask_ratio / 2.0)))
    half_w = max(1, int(round(width * mask_ratio / 2.0)))
    cy, cx = height // 2, width // 2
    region = torch.zeros((height, width), dtype=torch.bool, device=device)
    region[max(0, cy - half_h):min(height, cy + half_h),
           max(0, cx - half_w):min(width, cx + half_w)] = True
    return region


def _spectral_statistics(clean: Tensor, before_clip: Tensor, after_clip: Tensor,
                         amplitude_mask: Tensor | None, mask_ratio: float) -> dict[str, Any]:
    height, width = clean.shape[-2:]
    region = lf_region(height, width, mask_ratio, clean.device)
    spectra = [_spectrum(value) for value in (clean, before_clip, after_clip)]
    energies = [float(value.abs().square().sum().item()) for value in spectra]
    lf_energies = [float(value[..., region].abs().square().sum().item()) for value in spectra]
    cy, cx = height // 2, width // 2
    dc = [value[0, :, cy, cx].real.detach().cpu().tolist() for value in spectra]
    result: dict[str, Any] = {
        "lf_nominal_region_frequency_bins": int(region.sum().item()),
        "clean_total_energy": energies[0],
        "preclip_total_energy": energies[1],
        "postclip_total_energy": energies[2],
        "preclip_total_energy_retention": _ratio(energies[1], energies[0]),
        "postclip_total_energy_retention": _ratio(energies[2], energies[0]),
        "clean_lf_energy": lf_energies[0],
        "preclip_lf_energy": lf_energies[1],
        "postclip_lf_energy": lf_energies[2],
        "preclip_lf_energy_retention": _ratio(lf_energies[1], lf_energies[0]),
        "postclip_lf_energy_retention": _ratio(lf_energies[2], lf_energies[0]),
        "clean_dc_per_channel": dc[0],
        "preclip_dc_per_channel": dc[1],
        "postclip_dc_per_channel": dc[2],
        "preclip_dc_abs_change_per_channel": [abs(b - a) for a, b in zip(dc[0], dc[1])],
        "postclip_dc_abs_change_per_channel": [abs(b - a) for a, b in zip(dc[0], dc[2])],
        "lf_nominal_region_keep_fraction": None,
        "lf_hermitian_support_keep_fraction": None,
        "lf_full_mask_keep_fraction": None,
        "lf_dc_mask_per_channel": None,
        "lf_dc_mask_changed_any_channel": None,
    }
    if amplitude_mask is not None:
        iy = torch.tensor([(2 * cy - i) % height for i in range(height)], device=clean.device)
        ix = torch.tensor([(2 * cx - i) % width for i in range(width)], device=clean.device)
        support = region | region.index_select(-2, iy).index_select(-1, ix)
        dc_mask = amplitude_mask[0, :, cy, cx]
        result.update({
            "lf_nominal_region_keep_fraction": float(amplitude_mask[..., region].mean().item()),
            "lf_hermitian_support_frequency_bins": int(support.sum().item()),
            "lf_hermitian_support_keep_fraction": float(amplitude_mask[..., support].mean().item()),
            "lf_full_mask_keep_fraction": float(amplitude_mask.mean().item()),
            "lf_dc_mask_per_channel": dc_mask.detach().cpu().tolist(),
            "lf_dc_mask_changed_any_channel": bool((dc_mask != 1).any().item()),
            "lf_amplitude_mask_sha256": tensor_sha256(amplitude_mask),
        })
    return result


def build_probe_artifacts(normalized: Tensor, *, probe_id: str, dataset: str,
                          image_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse original transform and derive its exact pre-clipping counterpart.

    Repetition and parity with the model-facing original runner are checked
    for every non-clean sample.  No target is accepted by this function.
    """
    if normalized.ndim == 3:
        normalized = normalized.unsqueeze(0)
    if normalized.ndim != 4 or normalized.shape[0] != 1 or normalized.shape[1] != 3:
        raise ValueError("expected one RGB BCHW image")
    if probe_id not in PROBES:
        raise ValueError(f"unknown probe: {probe_id}")
    clean = imagenet_denormalize(normalized)
    probes = config["probes"]
    amplitude_mask = None
    noise = None
    seed = None
    repeated_exact = True
    parity_exact = True
    if probe_id == "clean":
        before_clip = clean.clone()
        after_clip = clean.clone()
    else:
        seed = derive_probe_seed(
            str(config["probe_seed_namespace"]), int(config["global_seed"]),
            dataset, int(config["human_epoch"]), image_id, probe_id,
        )
        if probe_id == "lf_mask":
            options = {
                "mask_ratio": float(probes["lf_mask"]["mask_ratio"]),
                "keep_probability": float(probes["lf_mask"]["keep_probability"]),
            }
            output = mask_low_frequency_amplitude(clean, generator=_generator(seed), **options)
            replay = mask_low_frequency_amplitude(clean, generator=_generator(seed), **options)
            amplitude_mask = output.amplitude_mask
            before_clip = torch.fft.ifft2(
                torch.fft.ifftshift(_spectrum(clean) * amplitude_mask, dim=(-2, -1)),
                dim=(-2, -1), norm="ortho",
            ).real
            after_clip = output.image
            repeated_exact = torch.equal(after_clip, replay.image) and torch.equal(amplitude_mask, replay.amplitude_mask)
        else:
            options = {
                "target_rms": float(probes["hf_noise"]["target_rms"]),
                "low_cut_ratio": float(probes["hf_noise"]["low_cut_ratio"]),
            }
            output = inject_high_frequency_noise_components(clean, generator=_generator(seed), **options)
            replay = inject_high_frequency_noise_components(clean, generator=_generator(seed), **options)
            noise = output.noise
            before_clip = clean + noise
            after_clip = output.image
            repeated_exact = torch.equal(after_clip, replay.image) and torch.equal(noise, replay.noise)
        original = build_deteriorated_view(
            normalized, [image_id], probe_id, str(config["probe_seed_namespace"]),
            int(config["global_seed"]), dataset, int(config["human_epoch"]),
            float(probes["lf_mask"]["mask_ratio"]), float(probes["lf_mask"]["keep_probability"]),
            float(probes["hf_noise"]["target_rms"]), float(probes["hf_noise"]["low_cut_ratio"]),
        )
        parity_exact = torch.equal(imagenet_normalize(after_clip), original)
    clip_parity_exact = torch.equal(before_clip.clamp(0.0, 1.0), after_clip)
    if not (repeated_exact and parity_exact and clip_parity_exact):
        raise RuntimeError("probe repetition/original-runner/preclip parity failed")
    stats = _spectral_statistics(clean, before_clip, after_clip, amplitude_mask,
                                 float(probes["lf_mask"]["mask_ratio"]))
    stats.update({
        "probe_seed": seed,
        "deterministic_repeat_exact": repeated_exact,
        "original_runner_parity_exact": parity_exact,
        "preclip_clamp_parity_exact": clip_parity_exact,
        "clean_physical_tensor_sha256": tensor_sha256(clean),
        "preclip_physical_tensor_sha256": tensor_sha256(before_clip),
        "postclip_physical_tensor_sha256": tensor_sha256(after_clip),
        "preclip_perturbation_rms_rgb": float((before_clip - clean).square().mean().sqrt().item()),
        "postclip_perturbation_rms_rgb": float((after_clip - clean).square().mean().sqrt().item()),
        "hf_noise_rms_rgb": None if noise is None else float(noise.square().mean().sqrt().item()),
        "hf_noise_rms_gray": None if noise is None else float(noise.mean(dim=1).square().mean().sqrt().item()),
        "hf_noise_rms_per_channel": None if noise is None else noise.square().mean(dim=(-2, -1)).sqrt()[0].detach().cpu().tolist(),
    })
    return {"clean": clean, "preclip": before_clip, "postclip": after_clip,
            "noise": noise, "statistics": stats}


def target_regions(target: np.ndarray, *, outer_radius: int = 5,
                   inner_radius: int = 1) -> list[dict[str, Any]]:
    foreground = np.asarray(target) > 0
    if foreground.ndim != 2:
        raise ValueError("target must be a two-dimensional GT mask")
    if not 0 <= inner_radius < outer_radius:
        raise ValueError("ring needs 0 <= inner radius < outer radius")
    labels, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=bool))
    result = []
    for target_id in range(1, count + 1):
        core = labels == target_id
        outer = ndimage.binary_dilation(core, structure=np.ones((3, 3), dtype=bool), iterations=outer_radius)
        inner = core if inner_radius == 0 else ndimage.binary_dilation(
            core, structure=np.ones((3, 3), dtype=bool), iterations=inner_radius)
        ring = outer & ~inner & ~foreground
        yy, xx = np.nonzero(core)
        result.append({
            "target_id": target_id, "core": core, "ring": ring,
            "target_pixels": int(core.sum()), "ring_pixels": int(ring.sum()),
            "bbox_yxyx_exclusive": [int(yy.min()), int(xx.min()), int(yy.max()) + 1, int(xx.max()) + 1],
        })
    return result


def _array(image: Tensor) -> np.ndarray:
    return image.detach().cpu()[0].numpy().astype(np.float64)


def clipping_statistics(image: np.ndarray, *, core: np.ndarray | None = None) -> dict[str, float]:
    rgb = image if core is None else image[:, core]
    at_endpoint = (rgb <= 0.0) | (rgb >= 1.0)
    gray = rgb.mean(axis=0)
    return {
        "rgb_channel_endpoint_fraction": float(at_endpoint.mean()),
        "any_rgb_channel_endpoint_pixel_fraction": float(at_endpoint.any(axis=0).mean()),
        "gray_endpoint_pixel_fraction": float(((gray <= 0.0) | (gray >= 1.0)).mean()),
        "rgb_channel_out_of_range_fraction": float(((rgb < 0.0) | (rgb > 1.0)).mean()),
        "any_rgb_channel_out_of_range_pixel_fraction": float(((rgb < 0.0) | (rgb > 1.0)).any(axis=0).mean()),
    }


def audit_visibility(artifacts: Mapping[str, Any], target: Tensor | np.ndarray, *,
                     visibility: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values = dict(VISIBILITY_DEFAULTS)
    values.update(visibility)
    clean, before, after = (_array(artifacts[name]) for name in ("clean", "preclip", "postclip"))
    if isinstance(target, Tensor):
        target_array = target.detach().cpu().numpy()
    else:
        target_array = np.asarray(target)
    target_array = np.squeeze(target_array)
    if target_array.shape != clean.shape[-2:]:
        raise ValueError("GT and image grids must agree")
    gt_before = target_array.copy()
    regions = target_regions(target_array, outer_radius=int(values["ring_outer_radius"]),
                             inner_radius=int(values["ring_inner_radius"]))
    grays = [value.mean(axis=0) for value in (clean, before, after)]
    epsilon = float(values["contrast_epsilon"])
    target_rows = []
    for region in regions:
        core, ring = region["core"], region["ring"]
        valid = bool(region["ring_pixels"] > 0)
        contrasts = [float(gray[core].mean() - gray[ring].mean()) if valid else None for gray in grays]
        clean_clip = clipping_statistics(clean, core=core)
        pre_clip = clipping_statistics(before, core=core)
        post_clip = clipping_statistics(after, core=core)
        clip_delta = post_clip["rgb_channel_endpoint_fraction"] - clean_clip["rgb_channel_endpoint_fraction"]
        post_ratio = None if not valid else (abs(contrasts[2]) + epsilon) / (abs(contrasts[0]) + epsilon)
        pre_ratio = None if not valid else (abs(contrasts[1]) + epsilon) / (abs(contrasts[0]) + epsilon)
        eligible = valid and abs(contrasts[0]) >= float(values["min_clean_contrast"])
        noise = artifacts["noise"]
        noise_array = None if noise is None else _array(noise)
        target_noise_rms = None if noise_array is None else float(np.sqrt(np.square(noise_array[:, core]).mean()))
        target_noise_gray_rms = None if noise_array is None else float(np.sqrt(np.square(noise_array.mean(axis=0)[core]).mean()))
        target_rows.append({
            **{key: value for key, value in region.items() if key not in {"core", "ring"}},
            "ring_valid": valid, "invalid_reason": None if valid else "empty_background_ring",
            "clean_contrast": contrasts[0], "preclip_contrast": contrasts[1], "postclip_contrast": contrasts[2],
            "preclip_contrast_retention_ratio": pre_ratio,
            "postclip_contrast_retention_ratio": post_ratio,
            "contrast_eligible_for_e1": bool(eligible),
            "preclip_contrast_sign_flip": None if not valid else bool(contrasts[0] * contrasts[1] < 0),
            "postclip_contrast_sign_flip": None if not valid else bool(contrasts[0] * contrasts[2] < 0),
            "postclip_contrast_below_ratio_threshold": None if not valid else bool(post_ratio < float(values["retention_threshold"])),
            "clean_target_clipping": clean_clip,
            "preclip_target_clipping": pre_clip,
            "postclip_target_clipping": post_clip,
            "target_rgb_channel_endpoint_fraction_increase": clip_delta,
            "target_clipping_warning": bool(clip_delta > float(values["clipping_increase_threshold"])),
            "hf_target_noise_rms_rgb": target_noise_rms,
            "hf_target_noise_rms_gray": target_noise_gray_rms,
            "hf_target_noise_rms_rgb_relative_clean_contrast": None if not valid or target_noise_rms is None else target_noise_rms / (abs(contrasts[0]) + epsilon),
            "hf_target_noise_rms_gray_relative_clean_contrast": None if not valid or target_noise_gray_rms is None else target_noise_gray_rms / (abs(contrasts[0]) + epsilon),
            "hf_image_noise_rms_rgb_relative_clean_contrast": None if not valid or noise is None else artifacts["statistics"]["hf_noise_rms_rgb"] / (abs(contrasts[0]) + epsilon),
        })
    image_stats = {
        "target_count": len(regions), "foreground_pixels": int((target_array > 0).sum()),
        "empty_target_view": not bool(regions),
        "ring_invalid_target_count": sum(not row["ring_valid"] for row in target_rows),
        "gt_tensor_unchanged": bool(np.array_equal(target_array, gt_before)),
        "clean_image_clipping": clipping_statistics(clean),
        "preclip_image_clipping": clipping_statistics(before),
        "postclip_image_clipping": clipping_statistics(after),
        "preclip_min": float(before.min()), "preclip_max": float(before.max()),
        "postclip_min": float(after.min()), "postclip_max": float(after.max()),
    }
    return image_stats, target_rows


def summarize_targets(rows: Iterable[Mapping[str, Any]], visibility: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(VISIBILITY_DEFAULTS)
    values.update(visibility)
    rows = list(rows)
    valid = [row for row in rows if row["ring_valid"]]
    eligible = [row for row in valid if row["contrast_eligible_for_e1"]]
    failures = sum(bool(row["postclip_contrast_below_ratio_threshold"]) for row in eligible)
    raw_failures = sum(bool(row["postclip_contrast_below_ratio_threshold"]) for row in valid)
    fraction = _ratio(failures, len(eligible))
    return {
        "target_count": len(rows), "valid_ring_target_count": len(valid),
        "invalid_ring_target_count": len(rows) - len(valid),
        "eligible_target_count": len(eligible),
        "low_clean_contrast_target_count": len(valid) - len(eligible),
        "eligible_contrast_failure_count": failures,
        "eligible_contrast_failure_fraction": fraction,
        "raw_all_defined_contrast_failure_count": raw_failures,
        "raw_all_defined_contrast_failure_fraction": _ratio(raw_failures, len(valid)),
        "e1_visibility_risk_triggered": None if fraction is None else fraction > float(values["failure_fraction_threshold"]),
        "e1_insufficient_eligible_targets": not bool(eligible),
        "postclip_contrast_sign_flip_count": sum(bool(row["postclip_contrast_sign_flip"]) for row in valid),
        "target_clipping_warning_count": sum(bool(row["target_clipping_warning"]) for row in rows),
        "target_clipping_warning_fraction": _ratio(sum(bool(row["target_clipping_warning"]) for row in rows), len(rows)),
        "preclip_contrast_retention_all_defined": distribution(row["preclip_contrast_retention_ratio"] for row in valid),
        "postclip_contrast_retention_all_defined": distribution(row["postclip_contrast_retention_ratio"] for row in valid),
        "preclip_contrast_retention_eligible": distribution(row["preclip_contrast_retention_ratio"] for row in eligible),
        "postclip_contrast_retention_eligible": distribution(row["postclip_contrast_retention_ratio"] for row in eligible),
        "target_clipping_increase": distribution(row["target_rgb_channel_endpoint_fraction_increase"] for row in rows),
        "hf_target_noise_rms_rgb_relative_clean_contrast_eligible": distribution(row["hf_target_noise_rms_rgb_relative_clean_contrast"] for row in eligible),
        "hf_target_noise_rms_gray_relative_clean_contrast_eligible": distribution(row["hf_target_noise_rms_gray_relative_clean_contrast"] for row in eligible),
        "label_semantics_changed_proven": False,
    }


def summarize_images(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("lf_nominal_region_keep_fraction", "lf_hermitian_support_keep_fraction",
              "lf_full_mask_keep_fraction", "preclip_total_energy_retention",
              "postclip_total_energy_retention", "preclip_lf_energy_retention",
              "postclip_lf_energy_retention", "preclip_perturbation_rms_rgb",
              "postclip_perturbation_rms_rgb", "hf_noise_rms_rgb", "hf_noise_rms_gray")
    result = {name: distribution(row[name] for row in rows) for name in fields}
    dc_available = [row for row in rows if row["lf_dc_mask_changed_any_channel"] is not None]
    result["lf_dc_mask_changed_image_count"] = sum(row["lf_dc_mask_changed_any_channel"] for row in dc_available)
    result["lf_dc_mask_changed_image_fraction"] = _ratio(result["lf_dc_mask_changed_image_count"], len(dc_available))
    for phase in ("clean", "preclip", "postclip"):
        result[f"{phase}_image_rgb_channel_endpoint_fraction"] = distribution(
            row[f"{phase}_image_clipping"]["rgb_channel_endpoint_fraction"] for row in rows)
        result[f"{phase}_image_rgb_channel_out_of_range_fraction"] = distribution(
            row[f"{phase}_image_clipping"]["rgb_channel_out_of_range_fraction"] for row in rows)
    return result


def validate_access_counts(access: Mapping[str, int], image_count: int) -> None:
    expected = image_count * len(VIEWS)
    if access.get("train_image_opens") != expected or access.get("train_mask_opens") != expected:
        raise RuntimeError(f"train payload access count differs from two views per image: expected {expected}")
    if any(access.get(name) != 0 for name in FORBIDDEN_ACCESS_FIELDS):
        raise RuntimeError("test/validation access counter is nonzero or missing")


def run(args: argparse.Namespace) -> dict[str, Any]:
    # The shared layer is deliberately imported only by the CLI: pure numerical
    # tests do not need real dataset manifests or open any input image.
    from analysis.d0a_v7_common import (
        complete_run, freeze_run, load_pilot_records, load_sample, read_config,
        reserve_output, runtime_bindings, write_jsonl_new,
    )
    config_path = args.config.resolve()
    config = read_config(config_path)
    validate_diagnostic_parameters(config)
    records = load_pilot_records(args.dataset, config)
    result_root = Path(config["result_root"])
    if not result_root.is_absolute():
        result_root = ROOT / result_root
    output = result_root / "probe_label_preservation" / args.dataset
    dependencies = [Path(__file__), ROOT / "train_cr_sitta_d0a.py", ROOT / "train_fixed_split.py",
                    *(ROOT / "tta/deteriorations").glob("*.py")]
    bindings = runtime_bindings(config_path, dependencies, records)
    preflight = {"ready": not output.exists(), "writes": 0, "dataset": args.dataset,
                 "pilot_images": len(records), "views": list(VIEWS), "probes": list(PROBES),
                 "expected_per_image_rows": len(records) * len(VIEWS) * len(PROBES),
                 "output_dir": str(output), "device": args.device,
                 "test_payload_opens": 0, "training_enabled": False,
                 "input_bindings": len(bindings)}
    if not args.execute:
        return preflight
    if output.exists():
        raise FileExistsError(f"refusing existing diagnostic output: {output}")
    device = torch.device(args.device)
    torch.set_num_threads(int(args.threads))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    contract = {
        **preflight, "diagnostic_id": "d0a-v7-probe-label-preservation-v1",
        "role": "source_train_pilot_diagnostic_not_paper_result",
        "development_only": True, "paper_result": False, "no_validation_split": True,
        "checkpoint_used": None, "validation_split_used": False,
        "gt_definition": "GT > 0; 8-connected components independently on each transformed view",
        "ring_definition": "square 8-neighbour dilation radius5 minus radius1, exclude all GT foreground",
        "contrast_definition": "RGB arithmetic mean; target mean minus background ring mean",
        "clipping_warning_definition": "increase in target RGB channel endpoint fraction; strict > 0.10",
        "visibility": config["visibility"], "probes_parameters": config["probes"],
        "probe_seed_namespace": config["probe_seed_namespace"],
        "global_seed": config["global_seed"], "human_epoch": config["human_epoch"],
        "gate_aggregation": "separate each dataset/view/probe; no pooled gate across transforms",
        "interpretation": "E1 flags a visibility risk; mask unchanged does not imply visibility preserved; it is not proof of semantic label change",
        "target_selection": "all components retained, invalid rings and low clean contrast reported separately",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "scipy": scipy.__version__, "pillow": pillow_version,
                    "device": str(device), "threads": torch.get_num_threads(),
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cuda_runtime": torch.version.cuda,
                    "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
    }
    reserve_output(output)
    freeze_run(output, contract, bindings)
    image_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    access: dict[str, int] = {name: 0 for name in FORBIDDEN_ACCESS_FIELDS}
    access.update({"train_image_opens": 0, "train_mask_opens": 0})
    visibility = config["visibility"]
    for index, record in enumerate(records, start=1):
        for view in VIEWS:
            normalized, target, metadata = load_sample(record, view, config, access=access)
            gt_hash = tensor_sha256(target)
            normalized = normalized.to(device)
            with torch.no_grad():
                for probe_id in PROBES:
                    artifacts = build_probe_artifacts(normalized, probe_id=probe_id,
                        dataset=args.dataset, image_id=record["image_id"], config=config)
                    image_stats, targets = audit_visibility(artifacts, target, visibility=visibility)
                    if tensor_sha256(target) != gt_hash:
                        raise RuntimeError("probe diagnostics mutated GT")
                    descriptor = {"dataset": args.dataset, "image_id": record["image_id"],
                                  "view": view, "probe_id": probe_id}
                    image_rows.append({**descriptor, "input_metadata": metadata,
                                       **artifacts["statistics"], **image_stats})
                    target_rows.extend({**descriptor, **row} for row in targets)
        if index % 8 == 0 or index == len(records):
            print(json.dumps({"dataset": args.dataset, "images_completed": index,
                              "images_total": len(records), "target_rows": len(target_rows)}), flush=True)
    validate_access_counts(access, len(records))
    write_jsonl_new(output / "per_image.jsonl", image_rows)
    write_jsonl_new(output / "per_target.jsonl", target_rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in target_rows:
        grouped[(row["view"], row["probe_id"])].append(row)
    cells = []
    for view in VIEWS:
        for probe_id in PROBES:
            cell_images = [row for row in image_rows if row["view"] == view and row["probe_id"] == probe_id]
            cells.append({"view": view, "probe_id": probe_id,
                          "image_count": len(cell_images),
                          "empty_target_view_count": sum(row["empty_target_view"] for row in cell_images),
                          "image_statistics": summarize_images(cell_images),
                          **summarize_targets(grouped[(view, probe_id)], visibility)})
    summary = {"diagnostic_id": contract["diagnostic_id"], "dataset": args.dataset,
               "role": contract["role"], "pilot_images": len(records),
               "development_only": True, "paper_result": False, "no_validation_split": True,
               "per_image_rows": len(image_rows), "per_target_rows": len(target_rows),
               "cells": cells, "access": access,
               "all_probe_replays_exact": all(row["deterministic_repeat_exact"] for row in image_rows),
               "all_original_runner_parity_exact": all(row["original_runner_parity_exact"] for row in image_rows),
               "all_gt_unchanged": all(row["gt_tensor_unchanged"] for row in image_rows),
               "label_semantics_changed_proven": False,
               "training_started": False, "test_payload_opens": 0,
               "e1_visibility_risk_triggered": any(row["e1_visibility_risk_triggered"] is True for row in cells if row["probe_id"] != "clean")}
    complete_run(output, summary)
    return {"complete": True, "output_dir": str(output), **summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True, choices=("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
