# Third-party source register

The CR-SITTA additions are local rewrites and do not import code from another
test-time-adaptation repository. Reference repositories under `third_party/`
are ignored by Git, pinned to exact revisions, and treated as read-only. Their
URL, full commit SHA, license, local path, and integration status are recorded
here and in `THIRD_PARTY_COMMITS.txt`.

## Host project

| Component | Upstream | Frozen revision | License status | Use |
| --- | --- | --- | --- | --- |
| NS-FPN | <https://github.com/mengduann/NS-FPN> | `b857bef068ba48f1258b62de6bf082f73dbafde4` | No repository-level license file was present at the frozen revision | Host detector |
| Multi-Scale Deformable Attention | Vendored by NS-FPN under `SFS_MSDeformAttn/ops/` | Included in the NS-FPN revision above | Source headers state Apache-2.0 and attribution to Deformable DETR | SFS CUDA operator |

The missing repository-level NS-FPN license must be clarified with the authors
before redistributing a derived release. It does not prevent local research use,
but it is a release-readiness blocker.

## External research assets

| Asset | Provenance used here | License / redistribution status | Repository policy |
| --- | --- | --- | --- |
| IRSTD-1K images and masks | Existing local research copy under the separate MSHNet workspace; corpus bytes are locked by `configs/protocol.yaml` | No dataset license or redistribution grant was present in the scoped local materials | Never copied into Git; local path/symlinks are ignored; do not redistribute until terms are confirmed |
| NUAA-SIRST images and masks | Existing local research copy under the separate MSHNet workspace; NS-FPN's own 341/86 split is used | No dataset license or redistribution grant was present in the scoped local materials | Never copied into Git; local path/symlinks are ignored; do not redistribute until terms are confirmed |
| Official NS-FPN checkpoints | Google Drive file IDs published in the frozen NS-FPN README and recorded in `weights/README.md` | No separate model-weight license or redistribution grant was supplied | Checkpoint binaries are ignored; users download from the official links and verify SHA-256 |

The corpus manifest hashes identify the exact local bytes used for experiments;
they do not grant permission to publish those bytes. A public artifact must link
to the dataset/checkpoint owners' distribution points or obtain written terms.

## Verified reference implementations

| Component | Upstream | Frozen revision | License | Local use |
| --- | --- | --- | --- | --- |
| TENT / Test-Time Norm | <https://github.com/DequanWang/tent> | `e9e926a668d85244c66a6d5c006efbd2b82e83e8` | MIT | Read-only behavioral reference at `third_party/tent`; no direct import or copied implementation |

The complete TENT file hashes and the deliberate differences in the strict
single-image episodic AdaBN rewrite are recorded in
`third_party/tent_reference.json`. In particular, CR-SITTA keeps non-BN modules
in evaluation mode, freezes all learnable parameters, disables BN running-stat
accumulation, and restores the complete Source state after every image.

## Planned reference implementations

These repositories are links only; they have not yet been cloned or copied:

- PIN-DGA: <https://github.com/jzchenriver/PIN-DGA>
- SITTA-Segmentation: <https://github.com/klarajanouskova/SITTA-Segmentation>
- MEMO: <https://github.com/zhangmarvin/memo>
- EATA: <https://github.com/mr-eggplant/EATA>
- SAR: <https://github.com/mr-eggplant/SAR>
- GraTa: <https://github.com/Chen-Ziyang/GraTa>
- InTEnt: <https://github.com/mazurowski-lab/single-image-test-time-adaptation>
- VPTTA: <https://github.com/Chen-Ziyang/VPTTA>
