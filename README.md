# CR-SITTA

**Candidate-Wise Risk-Constrained Single-Image Test-Time Adaptation for Infrared Small Target Detection and Segmentation**

CR-SITTA is an in-progress research extension of the official
[NS-FPN](https://github.com/mengduann/NS-FPN) implementation. NS-FPN remains
the host detector and Source baseline; this repository adds frozen corruption
protocols, unified IRSTD evaluation, and label-free single-image episodic
test-time adaptation. Every episode starts from the same Source checkpoint and
restores the complete Source state before the next image.

> **Development status:** this is not yet a release of the full CR-SITTA
> method. Stage B4 R0 rejected all 4 candidates, and the subsequent train-only
> ASB-SFR Stage C0 signal audit rejected all three grouped parameter spaces.
> The NUDT-SIRST D0-A fixed-endpoint clean development comparison also did not
> improve the clean Pareto front. Follow-up v7 diagnostics triggered the
> train-side LF visibility-risk screen (E1) on all three datasets, while the
> all-metric clean-harm screen (E0) and endpoint gradient-conflict screen (E2)
> did not trigger. These diagnostics do not authorize D0-B, IPMA, or formal
> test evaluation.
> The scoped decisions are summarized in [results/README.md](results/README.md)
> and preserved locally in append-only Stage B4 and Stage C0 archives. Generated
> result trees are intentionally Git-ignored and are not part of this source
> release.
> Consequently Stage C1, R1/R2, Stage B5, candidate-level risk control, and
> formal test evaluation remain blocked. AdaBN, Binary TENT, teacher, Stage-B,
> and Stage-C0 are development evidence and infrastructure, not the final
> CR-SITTA algorithm.

> **Protocol status:** the local benchmark uses only the existing `train` and
> `test` ID files under `datasets/`; no validation split is created. The current
> `best_miou` and `best_pd` checkpoints were selected by repeated fixed-test
> evaluation, so all results anchored to them are development evidence rather
> than an untouched-test main-paper benchmark. Under the frozen protocol, TTA
> parameters must be calibrated once on a fixed train-side Pilot with
> `best_miou` and then reused unchanged for `best_pd`. See
> [results/README.md](results/README.md) and the append-only Stage C0 declaration
> in [configs/artifact_eligibility_registry_v4.yaml](configs/artifact_eligibility_registry_v4.yaml),
> whose immutable parent chain is
> [v3](configs/artifact_eligibility_registry_v3.yaml) →
> [v2](configs/artifact_eligibility_registry_v2.yaml) →
> [v1](configs/artifact_eligibility_registry_v1.yaml).
> Run `./.conda/bin/python scripts/validate_result_eligibility_v2.py materialize`
> to materialize and verify the ignored local v2 parent registry; v3 and v4
> append exact full-tree negative-result declarations without rewriting their
> parents.
>
> Exact historical verification of the Stage-C/D0 frozen contracts additionally
> requires Git-ignored receipts, compiled environment files, and SHA-256-bound
> local v6/v7 decision-provenance memos. A fresh clone contains the source,
> configs, gates, and portable unit tests; opt in to local-artifact integration
> tests with `NS_FPN_RUN_LOCAL_ARTIFACT_TESTS=1` only after restoring those
> private inputs.

## Current scope

| Component | Status |
| --- | --- |
| Frozen protocol, upstream Source reproduction, unified evaluator, and deterministic corruptions | Completed and archived in [CR_SITTA_STEPS_0_3.md](CR_SITTA_STEPS_0_3.md) |
| Fixed-split NS-FPN training and clean/13-condition Source benchmark on IRSTD-1K, NUAA-SIRST, and NUDT-SIRST | Implemented |
| Single-image episodic state/reset framework and AdaBN | Implemented, including the formal 3-dataset × 13-condition runner |
| Binary Episodic TENT | Stage 1 engineering protocol and the source-train-only D0-v3 formal Stage-A diagnosis are complete. All 10 candidates failed the frozen utility gate (0/10 eligible), so R1/R2 and Stage 2/3 are hard-blocked and the all-BN entropy-update route is retained only as negative evidence |
| Non-adaptive multi-view teacher (P4) | The local source-train Pilot64 screen is complete across 3 datasets × 13 conditions. All 10 candidates failed the frozen utility gate (0/10 eligible); the best candidate reached non-clean macro ΔIoU +0.000722, below the required >+0.001. P5 remains unauthorized |
| Task-aligned Stage-B proposal generator | B1 mechanism decomposition completed on 2,496 train-side episodes; the B3 Pilot16 screen promoted 4/10 candidates. The full Pilot64 B4 gate then rejected all 4 candidates (0/4 eligible): the best P2 proposals reached non-clean macro ΔIoU +0.000357 but failed the frozen positive-family coverage requirement. The local C-0 negative archive preserves this scoped result; R1/R2, B5, and formal test remain blocked |
| Stage-C ASB-SFR signal audit | C0 completed on the fixed train Pilot64. Active support was 65.58% and candidate-proximal coverage among active episodes was 81.17%, but all spaces failed the frozen 80% finite/nonzero-gradient and >0.08 macro-cosine gates. The local Stage C0 negative archive records the independently recovered decision; C1 and formal test remain blocked |
| D0-A compatibility and v7 diagnostics | The NUDT-SIRST fixed epoch-1000 clean development comparison did not improve the clean Pareto front. Train-only Pilot64 diagnostics found an LF visibility risk on all three datasets; the E0 and E2 stop conditions did not trigger. This is diagnostic evidence only and does not authorize D0-B, IPMA, or formal test |
| Full CR-SITTA | Not implemented or evaluated yet |

The untouched historical v2 runner is preserved only as a non-authorizing
[negative-result code supplement](scripts/archive_binary_tent_ss_v2_runner_source.py),
whose ignored local artifact can be verified as documented in
[results/README.md](results/README.md).
The active v2 CLI permanently blocks every Stage-2 role before GPU or output
side effects; only a future candidate passing the reviewed v3 scientific gate
can use a separate Stage-2 authorization path.

The fixed benchmark contains clean data plus Gaussian noise, Gaussian blur,
low contrast, and stripe noise at S1, S3, and S5. Corruptions are applied
deterministically in physical RGB `[0,1]` before ImageNet normalization.
Adaptation code never receives masks; labels are exposed only to the outer
evaluator.

## Installation

The frozen research environment targets Linux, Python 3.10, PyTorch 2.1.2,
and CUDA 12.1. Model execution requires the compiled SFS CUDA extension.

```bash
git clone git@github.com:Arialliy/CR-SITTA.git
cd CR-SITTA

micromamba create -p ./.conda -f environment.cr-sitta.yml
scripts/build_sfs.sh
```

`build_sfs.sh` defaults to `/usr/local/cuda`, GCC/G++ 12, and CUDA architecture
8.6. Override `CUDA_HOME`, `CC`, `CXX`, or `TORCH_CUDA_ARCH_LIST` when needed.

## Data and checkpoints

Dataset images and masks, checkpoint binaries, compiled extensions, and
generated results are local artifacts and are not distributed in this
repository. Fixed-split experiments expect this layout:

```text
datasets/
├── IRSTD-1K/
│   ├── images/*.png
│   ├── masks/*.png
│   └── img_idx/{train_IRSTD-1K.txt,test_IRSTD-1K.txt}
├── NUAA-SIRST/
│   ├── images/*.png
│   ├── masks/*.png
│   └── img_idx/{train_NUAA-SIRST.txt,test_NUAA-SIRST.txt}
└── NUDT-SIRST/
    ├── images/*.png
    ├── masks/*.png
    └── img_idx/{train_NUDT-SIRST.txt,test_NUDT-SIRST.txt}
```

For the archived upstream reproduction, `test_source.py` also accepts dataset
roots containing `img/label`. Download the official checkpoints as documented
in [weights/README.md](weights/README.md), then configure machine-local paths:

```bash
cp configs/local_paths.example.yaml configs/local_paths.yaml
```

Formal runners use the versioned contracts under `configs/`, fail closed when
hashes or protocols drift, and refuse to overwrite completed artifacts. The
generated artifact layout is documented in [results/README.md](results/README.md).
The archived P4 and Stage-B runs also bind locally retained decision-provenance
memos and ignored predecessor receipts. Their public code and computational
contracts are inspectable here, but exact verification of those historical
artifacts is not self-contained in a fresh clone.

## Verification

With the frozen local inputs in place:

```bash
./.conda/bin/python scripts/validate_protocol.py \
  --protocol configs/protocol.yaml \
  --local-paths configs/local_paths.yaml \
  --output results/protocol_validation.json

./.conda/bin/python test_source.py \
  --dataset IRSTD-1k \
  --root /path/to/IRSTD-1K \
  --device cuda:0

# Default: portable tests that do not require ignored local research artifacts.
./.conda/bin/python -m pytest -q

# Opt in to tests that verify this machine's ignored research artifacts.
NS_FPN_RUN_LOCAL_ARTIFACT_TESTS=1 ./.conda/bin/python -m pytest -q
```

The opt-in integration tests require the ignored datasets and checkpoints,
materialized caches, engineering-smoke/parity/eligibility receipts under
`results/`, and the project-local compiled SFS extension under `.conda/`.
Populate and verify those inputs before setting
`NS_FPN_RUN_LOCAL_ARTIFACT_TESTS=1`; the default test command skips them.

## Fixed-split benchmark

The following example runs IRSTD-1K. Replace the dataset name with
`NUAA-SIRST` or `NUDT-SIRST` for the other fixed protocols.

```bash
./.conda/bin/python train_fixed_split.py \
  --dataset IRSTD-1K \
  --output-dir results/retraining_fixed_split/IRSTD-1K \
  --device cuda:0

./.conda/bin/python test_fixed_split_source.py \
  --dataset IRSTD-1K \
  --device cuda:0

./.conda/bin/python materialize_corruption_cache.py \
  --dataset IRSTD-1K

./.conda/bin/python run_source_corruption_benchmark.py \
  --dataset IRSTD-1K \
  --device cuda:0

./.conda/bin/python run_adabn_corruption_benchmark.py \
  --dataset IRSTD-1K \
  --device cuda:0
```

These are full formal runs rather than quick smoke tests. Inspect an entry
point with `--help` before launching it. Binary TENT is intentionally excluded
from the quick-start path because v2 failed the scientific gate and its
Stage 2/3 entrypoints are permanently blocked.

## CR-SITTA D0-A train-only compatibility stage

Stage C0 did not pass its frozen proxy/task-gradient gate, so C1 remains
blocked. The next registered hypothesis is D0-A supervised LF/HF degraded-view
training. It preserves the original 505-key NS-FPN checkpoint schema, but it is
not itself evidence of successful TTA or proxy alignment.

```bash
# Engineering gate: one LF and one HF train step; no test loader is built.
./.conda/bin/python train_cr_sitta_d0a.py \
  --protocol configs/cr_sitta_d0a_train_v2.yaml \
  --dataset IRSTD-1K --device cuda:0 --train-only-smoke

# Fixed 1000-epoch train-only endpoint; repeat for the other two datasets.
./.conda/bin/python train_cr_sitta_d0a.py \
  --protocol configs/cr_sitta_d0a_train_v2.yaml \
  --dataset IRSTD-1K --device cuda:0
```

The full run never opens the test split and emits
`epoch_1000_train_only.pth.tar`; it does not create a test-selected `best`
checkpoint. The next required step is to rerun the frozen train Pilot64
Stage-C0 gradient gate with the new checkpoint. Only a passing gate may unlock
episodic TTA or formal test. The frozen v2 training and D0-B configs retain the
complete boundary and dual-axis (`final_miou_axis` / `final_pd_axis`) policy;
their historical provenance is SHA-256-bound to the local v7 memo described
above.

The v2 training artifact also contains optimizer and process-RNG state, so it
is not the inference hand-off. After a complete epoch-1000 run, use
`export_cr_sitta_d0a_safe_checkpoint.py` with the independently recorded source,
run-contract, and full-freeze SHA-256 values. The exporter writes a no-replace
`epoch_1000_train_only_safe.pth.tar`; `SAFE_EXPORT.json` is its completion
sentinel. Do not export a live `last.pth.tar`.

If a worker is interrupted before epoch 1000, do not pass its CUDA-loaded
`last.pth.tar` directly to `--resume`. First use
`recover_cr_sitta_d0a_v2.py` after confirming that the worker is no longer
running. The CLI requires independently recorded SHA-256 anchors for the source,
run contract, and full freeze, plus explicit process-stopped and trusted-local
acknowledgements; inspect `--help` before use. Resume only from the resulting
receipt-bound, RNG-sanitized checkpoint. This recovery route preserves the epoch
boundary but does not claim bit-exact equivalence across CUDA processes.

The checkpoint-rebound D0-B implementation is prepared in
`run_cr_sitta_d0b_gradient_gate_v1.py`. Its read-only readiness check is:

```bash
./.conda/bin/python run_cr_sitta_d0b_gradient_gate_v1.py preflight
```

It must remain `ready=false` until all three `SAFE_EXPORT.json` receipts exist
and verify. Only then may `freeze`, per-dataset `teacher`/`candidate`/`outer`,
and `aggregate` run in that order. D0-B rebuilds every checkpoint-dependent
artifact and can authorize only D1 train-internal OOF; it cannot authorize a
formal test directly.

The reviewed hand-off is automated by
`scripts/orchestrate_cr_sitta_d0a_to_d0b_v1.py`. Its default `status` and
explicit `dry-run` modes are read-only. Only `execute` may wait for all three
1000-epoch D0-A runs, deep-validate their train-only completion, publish
hash-bound safe exports, and run D0-B with the global barrier
`all teacher -> all candidate -> all outer` on physical GPU 2:

```bash
./.conda/bin/python scripts/orchestrate_cr_sitta_d0a_to_d0b_v1.py status
./.conda/bin/python scripts/orchestrate_cr_sitta_d0a_to_d0b_v1.py dry-run
./.conda/bin/python scripts/orchestrate_cr_sitta_d0a_to_d0b_v1.py execute
```

The orchestrator terminates after D0-B `verify` and publishes
`PIPELINE_COMPLETE.json`. A negative D0-B gate is a valid scientific stop; a
positive gate authorizes, but does not launch, D1. Neither branch accesses the
formal test split.

### D0-A v7 diagnostics

After the NUDT-SIRST fixed-endpoint clean development comparison, the v7
workflow performs diagnostics only; it does not resume full training. Its
published source consists of the frozen
[configuration](configs/cr_sitta_d0a_v7_diagnostics_v1.yaml), the existing-log
[endpoint comparison](analysis/compare_d0a_fixed_endpoint.py), the train-only
[probe visibility audit](analysis/audit_d0a_probe_label_preservation.py), and
the endpoint [branch-gradient audit](analysis/audit_d0a_branch_gradients.py).
Each CLI defaults to a read-only preflight and requires `--execute` before
writing a new, no-replace local artifact. For example:

```bash
./.conda/bin/python analysis/compare_d0a_fixed_endpoint.py
./.conda/bin/python analysis/audit_d0a_probe_label_preservation.py \
  --dataset IRSTD-1K
./.conda/bin/python analysis/audit_d0a_branch_gradients.py --device cuda:0
```

The completed local P0–P3 evidence is summarized in
[results/README.md](results/README.md). Exact receipts and the SHA-bound v7
decision memo are intentionally Git-ignored; the public repository contains
the implementation, frozen configuration, and portable unit tests. The
metadata-only finalizer exists solely to recover the already computed P2
record whose final sealing step encountered a text-encoding error; it does not
recompute or alter the numerical evidence.

## Upstream and third-party attribution

This work is derived from the official NS-FPN repository at frozen commit
`b857bef068ba48f1258b62de6bf082f73dbafde4`. NS-FPN itself is based on MSHNet,
and its SFS operator vendors Multi-Scale Deformable Attention code. TENT is used
only as a pinned, read-only behavioral reference; the adaptation additions here
are local implementations. See [THIRD_PARTY.md](THIRD_PARTY.md) and
[THIRD_PARTY_COMMITS.txt](THIRD_PARTY_COMMITS.txt) for exact revisions and
provenance.

The frozen NS-FPN revision did not contain a repository-level license file.
Dataset and checkpoint redistribution terms are not supplied here; verify the
owners' terms before redistributing derived code, data, or weights.

## Upstream NS-FPN

### Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective

**CVPR 2026** — Maoxun Yuan, Duanni Meng, Ziteng Xi, Tianyi Zhao, Shiji Zhao,
Yimian Dai, and Xingxing Wei.

Paper: [arXiv:2508.06878v2](https://arxiv.org/html/2508.06878v2)

NS-FPN integrates a low-frequency guided feature purification (LFP) module and
a spiral-aware feature sampling (SFS) module into an FPN to suppress noise while
retaining target-relevant features. The original implementation reports results
on IRSTD-1K and NUAA-SIRST.

<div align="center">
  <img src="./assets/NS-FPN.png" style="width: 80%; height: auto; max-height: 70vh;" alt="NS-FPN overview" />
</div>

### Reported upstream results

| Dataset | IoU (×10⁻²) | Pd (×10⁻²) | Fa (×10⁻⁶) | Weights |
| --- | ---: | ---: | ---: | --- |
| IRSTD-1K | 69.34 | 95.58 | 8.35 | [Download](https://drive.google.com/file/d/1agnCjpJJa3J3-Aw8XqDKtpcTA6xDuHO4/view?usp=sharing) |
| NUAA-SIRST | 78.74 | 100.0 | 1.24 | [Download](https://drive.google.com/file/d/17zgfkbkPdLGyOLDz_MFNbUQI9J2bmgiI/view?usp=sharing) |

<div align="center">
  <img src="./assets/results.png" style="width: 80%; height: auto; max-height: 95vh;" alt="NS-FPN quantitative results" />
  <img src="./assets/visualization.png" style="width: 90%; height: auto; max-height: 80vh;" alt="NS-FPN visual results" />
</div>

The upstream NS-FPN implementation was developed from
[MSHNet](https://github.com/Lliu666/MSHNet).

### Citation

```bibtex
@inproceedings{yuan2026seeing,
  title={Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective},
  author={Yuan, Maoxun and Meng, Duanni and Xi, Ziteng and Zhao, Tianyi and Zhao, Shiji and Dai, Yimian and Wei, Xingxing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={27783--27792},
  year={2026}
}
```
