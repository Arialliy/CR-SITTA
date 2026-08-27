# CR-SITTA

**Candidate-Wise Risk-Constrained Single-Image Test-Time Adaptation for Infrared Small Target Detection and Segmentation**

CR-SITTA is an in-progress research extension of the official
[NS-FPN](https://github.com/mengduann/NS-FPN) implementation. NS-FPN remains
the host detector and Source baseline; this repository adds frozen corruption
protocols, unified IRSTD evaluation, and label-free single-image episodic
test-time adaptation. Every episode starts from the same Source checkpoint and
restores the complete Source state before the next image.

> **Development status:** this is not yet a release of the full CR-SITTA
> method. Candidate extraction, candidate-wise risk-constrained updates,
> backtracking, and the second-backbone study remain future work. The current
> AdaBN and Binary TENT implementations are baselines and infrastructure, not
> the final CR-SITTA algorithm.

## Current scope

| Component | Status |
| --- | --- |
| Frozen protocol, upstream Source reproduction, unified evaluator, and deterministic corruptions | Completed and archived in [CR_SITTA_STEPS_0_3.md](CR_SITTA_STEPS_0_3.md) |
| Fixed-split NS-FPN training and clean/13-condition Source benchmark on IRSTD-1K, NUAA-SIRST, and NUDT-SIRST | Implemented |
| Single-image episodic state/reset framework and AdaBN | Implemented, including the formal 3-dataset × 13-condition runner |
| Binary Episodic TENT | Smoke testing, cross-process audit, transition metrics, and train-derived calibration tooling are present; optimizer/LR calibration is in progress |
| Full CR-SITTA | Not implemented yet |

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

./.conda/bin/python -m pytest -q
```

The full suite includes fail-closed real-contract integration tests. A fresh
clone must be populated with the ignored datasets, checkpoints, and pinned
intermediate artifacts before those tests can pass; pure unit tests do not
require the local research artifacts.

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
from the quick-start path while calibration remains unfinished.

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
