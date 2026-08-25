# CR-SITTA Steps 0–3 completion record

This repository implements and validates Steps 0–3 of
`CR-SITTA_实施路线与文献代码阅读指南.md` on the frozen NS-FPN base commit
`b857bef068ba48f1258b62de6bf082f73dbafde4`.

## Status

| Step | Outcome | Primary artifacts |
|---|---|---|
| 0 | Protocol, source commit, splits, checkpoints, evaluation rules, corruption rules, and dependency versions frozen | `configs/protocol.yaml`, `BASE_COMMIT.txt`, `THIRD_PARTY_COMMITS.txt`, `THIRD_PARTY.md`, `environment.cr-sitta.yml`, `requirements.lock.txt` |
| 1 | Official Source baseline reproduced on both complete test splits | `test_source.py`, `results/source_reproduction/<dataset>/` |
| 2 | One model/probability interface and one research evaluator implemented and tested | `tta/model_adapter.py`, `dataset/research_dataset.py`, `metrics/` |
| 3 | Clean plus four deterministic physical-domain corruptions implemented; five engineering-v1 levels calibrated on source trainval data and frozen | `corruptions/`, `run_corruption_pilot.py`, `results/corruption_pilot/<dataset>/` |

Roadmap numbering starts at Step 0, so “the first four” means Steps 0, 1, 2,
and 3.

## Source reproduction

All runs use batch size 1, 256 x 256 inputs, official test splits and official
checkpoints. The runner also verifies exact first-image repeat logits, finite
outputs, matching spatial sizes, and an unchanged model-state SHA256.

| Dataset | Images | Reported IoU / Pd / Fa×10⁻⁶ | Official reproduced IoU / Pd / Fa×10⁻⁶ | Unified IoU / Pd / Fa×10⁻⁶ |
|---|---:|---:|---:|---:|
| IRSTD-1K | 201 | 0.6934 / 0.9558 / 8.35 | 0.693421 / 0.955782 / 8.350581 | 0.693873 / 0.955782 / 9.717040 |
| NUAA-SIRST | 86 | 0.7874 / 1.0000 / 1.24 | 0.787406 / 1.000000 / 1.241994 | 0.787406 / 1.000000 / 1.241994 |

Each dataset result contains `metrics.json`, one JSONL record per test image,
20 prediction visualizations, and a contact sheet used for visual inspection.

## Corruption calibration

The bounded pilot selects the same 64 source-trainval IDs per condition by
ascending SHA256 of `image_id`. It rejects `test.txt` and any renamed split
containing an official test ID. It loads the model once, applies corruption in
physical RGB `[0,1]` before normalization, and verifies the model state and
severity table remain unchanged throughout the run.

The following fixed-threshold IoU/Pd values summarize clean and S1/S3/S5:

| Dataset | Corruption | Clean | S1 | S3 | S5 |
|---|---|---:|---:|---:|---:|
| IRSTD-1K | Gaussian noise | .6716/.9677 | .6563/.9677 | .5497/.8710 | .1621/.4731 |
| IRSTD-1K | Gaussian blur | .6716/.9677 | .6716/.9677 | .5920/.8925 | .2685/.5054 |
| IRSTD-1K | Low contrast | .6716/.9677 | .6722/.9785 | .6669/.9892 | .5722/.9462 |
| IRSTD-1K | Stripe noise | .6716/.9677 | .6719/.9677 | .6483/.9140 | .5952/.8710 |
| NUAA-SIRST | Gaussian noise | .7905/.9747 | .7876/.9747 | .7200/.8861 | .2217/.3924 |
| NUAA-SIRST | Gaussian blur | .7905/.9747 | .7900/.9747 | .7676/.9241 | .6403/.8101 |
| NUAA-SIRST | Low contrast | .7905/.9747 | .7884/.9747 | .7609/.9494 | .6868/.9114 |
| NUAA-SIRST | Stripe noise | .7905/.9747 | .7877/.9747 | .7578/.9494 | .6121/.7848 |

The overall decline, distinct Pd/false-alarm behavior, progressive severity
grids, nonzero S5 performance, deterministic seeds, and unchanged masks satisfy
the Step 3 pilot criteria. Small local target-count improvements on IRSTD-1K
low-contrast levels do not change the overall clean-to-S5 trend.

There is no independent source-validation split in the upstream release, and
the official checkpoints were trained on the complete `trainval` splits. The
frozen engineering-v1 calibration therefore uses deterministic subsets of
official source `trainval` and proves zero overlap with official test IDs. It is
not a paper-grade independent-holdout calibration. A future experiment must
reserve a source holdout, retrain without it, and version both the checkpoint,
split, and severity table instead of silently changing engineering-v1.

## Reproduction commands

```bash
micromamba create -p ./.conda -f environment.cr-sitta.yml
scripts/build_sfs.sh

./.conda/bin/python scripts/validate_protocol.py \
  --protocol configs/protocol.yaml \
  --local-paths configs/local_paths.yaml \
  --output results/protocol_validation.json

./.conda/bin/python test_source.py \
  --dataset IRSTD-1k --root /path/to/IRSTD-1K --device cuda:0
./.conda/bin/python test_source.py \
  --dataset NUAA-SIRST --root /path/to/NUAA-SIRST --device cuda:0

./.conda/bin/python run_corruption_pilot.py \
  --dataset IRSTD-1k --root /path/to/IRSTD-1K --device cuda:0 \
  --output-dir results/corruption_pilot_frozen/IRSTD-1k
./.conda/bin/python run_corruption_pilot.py \
  --dataset NUAA-SIRST --root /path/to/NUAA-SIRST --device cuda:0 \
  --output-dir results/corruption_pilot_frozen/NUAA-SIRST

# After clean-commit Source and frozen-severity confirmation runs:
./.conda/bin/python scripts/build_artifact_manifest.py

./.conda/bin/python -m pytest -q
```

`results/corruption_pilot/` is the immutable provisional-table calibration
evidence whose JSON hashes are recorded in `severity_tables.yaml`; do not
overwrite it when confirming the final frozen table. The confirmation runs use
the separate `results/corruption_pilot_frozen/` paths shown above.

Local dataset paths, binary checkpoints, environments, compiled outputs, and
generated results are intentionally ignored by Git. Their hashes and expected
locations are frozen in the protocol and versioned artifact manifest. The
platform-specific Conda build/channel lock is
`environment.linux-64.explicit.txt`; pip-only packages and the post-build SFS
extension are documented by `environment.cr-sitta.yml`,
`requirements.lock.txt`, and `scripts/build_sfs.sh`.
