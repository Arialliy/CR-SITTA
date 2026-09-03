# 实验产物保存规范

所有后续实验产物统一保存在：

```text
results
```

## 方法目录

```text
results/
├── baseline/       # NS-FPN Source / clean baseline
├── adabn/          # AdaBN / Test-Time Norm baseline
├── binary_tent/    # Binary Episodic TENT baseline
└── cr_sitta/       # 最终方法 CR-SITTA
```

`NS-FPN` 只表示 Source backbone/baseline；最终方法名固定为 `CR-SITTA`。
不同方法不得写入同一个目录，也不得覆盖已有正式实验。
历史目录迁移记录见 `relocations.json`；已经由 SHA-256 封存的历史结果只登记
迁移关系，不直接改写其内部 JSON。

## 科学资格与数据边界

官方数据入口只使用 `datasets/<dataset>/img_idx/` 中既有的 `train_*.txt`
和 `test_*.txt`，不创建验证集，也不改变这些 ID 文件。训练侧固定 64 图的
TTA Pilot 仅用于方法超参数校准；它不是新的数据划分、不是模型训练集缩减，
也不是论文性能结果。该 Pilot 与 corruption severity Pilot64 及 test ID 均
零重叠。

当前 `best_miou` 和 `best_pd` Source checkpoint 均由 500 epoch 后逐 epoch
评价固定 test 得到，因此现有 Source、corruption、AdaBN 与相关导出属于
`development_test_selected`，不能表述为未触碰 test 的无偏论文主表结果。
基础资格规则见 `configs/artifact_eligibility_registry_v1.yaml`；已完成的
best-pd development axis 由 append-only 的
`configs/artifact_eligibility_registry_v2.yaml` 登记。运行
`./.conda/bin/python scripts/validate_result_eligibility_v2.py materialize` 后会在
本目录生成被 Git 忽略的 `artifact_eligibility_registry_v2.json`，且会先严格验证
v1 父 registry，不改写任何已经封存的实验产物。

TTA 只在训练侧 Pilot 上选择一次超参数：以 `best_miou` 为校准锚点；随后
`best_pd` 必须复用完全相同的冻结参数，不允许再调一次。完整 CR-SITTA 冻结后，
才允许在完整固定 test 上运行，并为两个 checkpoint 角色分别保存指标、逐图记录
和全部预测 mask。

`binary_tent/ss_calibration_cache_v2/` 是由该训练侧 Pilot64 派生的三数据集
TENT-SS 协议/校准资产。它只把既有 test ID 用作泄漏检查，不创建 validation，
不包含主论文性能结论；eligibility 固定为 `protocol_asset_nonperformance`、
`main_paper_table: false`，仅供 development-only 方法校准使用。

`binary_tent/ss_calibration_v2/` 的 Stage 1 工程协议已经完成，但科学效用门未
通过：被排序选出的 3 个 SGD 候选在全部 39 个 cell 上均没有二值预测或指标
变化，其余 Adam 候选虽在少数 cell 有局部收益，但 39-cell macro IoU 均为负。
因此 Stage 2/3 保持禁止。v2 历史工件和封存源码快照只作为不可改写的负基线
保留；独立的
`binary_tent/ss_calibration_v2_negative_archive/` 用于保存负结果判定和哈希，
不得进入正式 paper-result 索引。后续科学门在 v3 中实现，且必须先过滤合格
候选，再进行 Top-K 排名。

历史 v2 runner 的未修改源码另存于
`binary_tent/ss_calibration_v2_negative_archive_code_supplement_v1/`，只用于负结果
复现审计，固定 `paper_result: false`、`stage2_authorization: false`；它不能恢复
Stage 2，只能在隔离的历史环境中重放。当前活动 runner 的 `worker-stage2`、
`aggregate-final`、`launch-stage2` 三个入口均永久 fail closed，在读取配置、申请
GPU、构造 worker 命令或创建输出前返回 `SCIENTIFIC_GATE_BLOCKED`（退出码 3）。
补充封存可独立验证：

```bash
./.conda/bin/python \
  scripts/archive_binary_tent_ss_v2_runner_source.py --verify-only
```

v3 的生产授权入口不再接受调用者提供的 receipt 路径或 SHA。它只信任代码内固定
的配置摘要，并沿 `formal aggregate → frozen gate manifest / Stage1 records /
diagnostics → selector-v3 重算 receipt` 验证完整证据链。当前配置仍有未冻结阈值，
所以 `run_binary_tent_ss_calibration_v3.py authorize-stage2` 必须返回
`SCIENTIFIC_GATE_BLOCKED`（退出码 3）。

后续失败机理诊断使用 `configs/tent_failure_diagnostics_v1.yaml`，输出固定在
`cr_sitta/tent_failure_diagnostics_v1/`。它只读取现有 3×64×13 source-train
cache；不创建验证集，不打开 test 图像或标签。训练 mask 只在全部 label-free
episode 完成后进入外层 oracle analyzer，方法侧标签访问恒为 0。该目录始终是
`paper_result: false`，不能用于恢复 Stage 2；正式 D0 shard 还必须先通过共享熵
梯度与冻结 v2 独立执行路径的逐候选 bit-exact equivalence。该等价门必须由
3 个独立 `exec` 新鲜进程重复，绑定 Linux PID 与 `/proc` start-time，并对 3 个
进程对的 pre/post logits、梯度、参数增量、更新张量数和 step norm 做严格复验。
父进程还会签发随机 run/child nonce，并绑定实际启动命令、PID/start-time、退出码、
stdout/stderr 与 receipt 哈希的 launch transcript。该证据说明受支持 runner 确实
执行了三次新进程；没有外部信任根时，不将其表述为对本机特权伪造的密码学证明。
本地攻击者若能同时改写 artifact 与对应 checksum/SHA ledger，属于明确不覆盖的
威胁范围；这些摘要用于发现非成对篡改、意外损坏和受支持 runner 内的协议漂移，
不是外部签名或远程证明。
单进程 receipt 写入 `equivalence/runs/<dataset>/`；只有三次均通过时才发布固定的
`equivalence/<dataset>.json`。正式 shard 同时复验完整 113 文件负结果档案、
`NEGATIVE_RESULT.json`、`SHA256SUMS` 和冻结的 Stage-1 records SHA，而不是只信任
一份 JSONL。D0 只解释失败机理；即便 D0 完成，Stage 2/3 仍保持禁止。

当前正式状态以
`cr_sitta/tent_failure_diagnostics_v3_formal_stage_a/aggregate_phase/R0/`
为准：39/39 cells 和 24,960 条外层记录均已完成并通过发布时重建，10 个候选中
0 个通过冻结科学门。该结果是完整的训练侧 development negative evidence；
R1/R2、Stage 2/3 以及旧的全-BN entropy 参数更新路线均保持禁止。

P4 非自适应多视图 teacher 的正式训练侧 Pilot64 screen 保存在
`cr_sitta/nonadaptive_teacher_screen_v1/`。三个数据集各完成 13 个条件和 64 张图，
候选生成阶段共执行 2,496 个 image-condition episode；outer evaluator 在候选
artifact 完整封存后才读取训练侧 Pilot mask，并汇总 10×3×13=390 个 candidate
cell。正式 aggregate 位于 `aggregate_phase/R0/`，协议与整数守恒均通过，但科学
状态为 `scientific_no_eligible`：10 个候选中 0 个合格。最佳候选
`flip4_source_anchor_beta_0_5` 的 36-cell non-clean macro ΔIoU 为
`+0.000721923`，低于冻结的严格门槛 `>+0.001`，因此 P5 仍未授权。
该 v1 运行还绑定本机保留的决策溯源 memo 与被忽略的 P3 前序 receipts；仓库公开
其计算配置、runner 和门函数，但 fresh clone 不包含完整的历史 artifact 复验包。

该 v1 receipt 在发布后审计中发现一个 dataset-scope 实现问题：名为
`dataset_nonclean_delta_iou/pd` 的字段实际混入了 clean，使用 13 cells；Fa 的
`each_dataset` 使用 13 cells 则是正确的。旧 v1 artifact 与哈希保持不可改写；
后续以 append-only 的 scope-correction receipt 用每数据集 12 个 non-clean cells
重算 IoU/Pd。该修正不会改变任何候选的 reason codes 或 eligibility，结论仍为
0/10、P5 禁止。不得把旧 receipt 中这两个误标字段直接解释为 non-clean 数值。

若 worker 在 receipt 构造前因工程异常退出，不会伪造单进程 receipt；这类事件
单独登记在 `cr_sitta/tent_failure_diagnostics_v1/incidents/`，固定为非论文、
非选择证据。2026-09-01 的首次 NUAA-SIRST equivalence 尝试即按此规则记录：它
暴露了 float32 存储可见步长与连续 float64 公式的参考域错误，没有产生 canonical
equivalence、shard 或 aggregate，也不代表 Adam 候选的科学失败。
同日第二次 NUAA-SIRST equivalence 尝试也在 canonical receipt 前安全停止：CUDA
同设备参考对 106/106 个参数张量及 318/318 个 optimizer-state 张量均为 bit-exact，
但 CPU 原生 float32 重放与 CUDA 端点在两个标量上相差 1 ULP。该事件记录为
`20260901_nuaa_equivalence_cross_backend_replay_failure.json`；它暴露的是把跨后端
数值可移植性诊断误作执行正确性硬门的问题，不是 Adam、D0 或科学门结果。

每个数据集先在一张隔离 GPU 上生成三进程等价 receipt，随后才允许生成对应
formal shard；三个 shard 都通过后再聚合。所有这些产物均
固定 `selection_authorized: false`、`stage2_authorized: false` 和
`stage3_authorized: false`。

当前 D0-v3 Stage-A 的正式 R0 入口如下。grid 串行执行全部 3×13 个 candidate、
candidate verify、outer 和 outer verify；aggregate 只在 39 个 cell 全部通过后
发布，并且始终不授权 R1/R2 或 Stage 2：

```bash
./.conda/bin/python scripts/run_d0_v3_formal_stage_a_r0_grid.py validate
./.conda/bin/python scripts/run_d0_v3_formal_stage_a_r0_grid.py run \
  --cuda-visible-device 0

CUDA_VISIBLE_DEVICES= ./.conda/bin/python \
  scripts/run_d0_v3_formal_stage_a_r0_aggregate.py preflight
CUDA_VISIBLE_DEVICES= ./.conda/bin/python \
  scripts/run_d0_v3_formal_stage_a_r0_aggregate.py run
CUDA_VISIBLE_DEVICES= ./.conda/bin/python \
  scripts/run_d0_v3_formal_stage_a_r0_aggregate.py verify
```

下面是历史 D0-v1 的归档命令，仅用于旧 artifact 复验，不代表当前正式入口。
示例中的 GPU 编号按实际空闲隔离卡替换；旧 `run` 不传 `--condition` 或
`--candidate`，并保持默认 `--max-images 64`；使用任何筛选或更小样本数都只能算
smoke：

```bash
for ds in IRSTD-1K NUAA-SIRST NUDT-SIRST; do
  env CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ./.conda/bin/python scripts/run_tent_failure_diagnostics.py \
    equivalence-repro \
    --dataset "$ds" --device cuda:0 \
    --output "results/cr_sitta/tent_failure_diagnostics_v1/equivalence/${ds}.json"

  env CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    ./.conda/bin/python scripts/run_tent_failure_diagnostics.py run \
    --dataset "$ds" --device cuda:0 \
    --equivalence-receipt \
    "results/cr_sitta/tent_failure_diagnostics_v1/equivalence/${ds}.json" \
    --output "results/cr_sitta/tent_failure_diagnostics_v1/shards/${ds}"
done

./.conda/bin/python scripts/run_tent_failure_diagnostics.py aggregate \
  --shard results/cr_sitta/tent_failure_diagnostics_v1/shards/IRSTD-1K \
  --shard results/cr_sitta/tent_failure_diagnostics_v1/shards/NUAA-SIRST \
  --shard results/cr_sitta/tent_failure_diagnostics_v1/shards/NUDT-SIRST \
  --output results/cr_sitta/tent_failure_diagnostics_v1/aggregate

./.conda/bin/python scripts/run_tent_failure_diagnostics.py verify \
  --path results/cr_sitta/tent_failure_diagnostics_v1/shards/IRSTD-1K

./.conda/bin/python scripts/run_tent_failure_diagnostics.py verify-aggregate \
  --path results/cr_sitta/tent_failure_diagnostics_v1/aggregate
```

AdaBN 实现门禁使用固定训练域样本，单独写入
`adabn/adabn_source_pilot_smoke_v1/`。该目录必须标记
`paper_result: false` 和 `performance_metrics_computed: false`；其中的训练域
mask 只允许用于输入完整性哈希，禁止传入适应方法或计算性能。它不是正式
benchmark，也不能与下面的正式 test 结果合并汇报。

正式 AdaBN 的固定目录为
`adabn/adabn_batch_stats_v1/`。它只读取已冻结的 full test cache，
不再调整 corruption 强度、模型、阈值或 checkpoint。

## 后续运行层级

除已经冻结的 `baseline/<dataset>/best_miou` clean 导出外，新实验使用：

```text
results/<method>/<run_id>/<dataset>/conditions/<condition>/
```

- `<method>`：`adabn`、`binary_tent` 或 `cr_sitta`；
- `<run_id>`：固定协议名或带日期的唯一运行名；
- `<dataset>`：`IRSTD-1K`、`NUAA-SIRST`、`NUDT-SIRST`；
- `<condition>`：`clean_S0` 或 `<corruption>_S<severity>`。

## 每次正式实验必须保存

```text
global_index/
├── run_config.yaml                     # 实际生效的完整配置
├── provenance.json                     # split/checkpoint/cache/code/config 哈希与 seed
├── aggregate_metrics.json              # 3×13 条件全局汇总
└── artifact_manifest.json              # 根级链式文件哈希
COMPLETE.json                               # 仅 39 个条件全部校验后写入
<dataset>/
├── dataset_index/
│   ├── dataset_run_config.yaml
│   ├── dataset_provenance.json
│   ├── dataset_summary.json        # 该数据集的 13 条件汇总
│   └── artifact_manifest.json
├── DATASET_COMPLETE.json               # 仅 13 个条件都完整时写入
└── conditions/<condition>/
    ├── metrics.json                    # legacy mean IoU/Pd/FA 与 unified IoU/Pd/FA/FROC
    ├── per_image.jsonl                 # 逐图输出哈希与 shard 索引
    ├── adaptation_diagnostics.jsonl   # 逐图 episodic 状态门禁
    ├── probabilities_256.npy          # 单个 little-endian float32 概率分片
    ├── prediction_masks_256/           # 所有二值预测 mask
    ├── artifact_manifest.json
    └── CONDITION_COMPLETE.json         # 该 full-test 条件原子完成
```

单个 full-test condition shard 是正式产物；带 `--max-images`、`--smoke`
或自定义输出目录的运行始终是 smoke，不能产生上述正式
completion sentinel。

固定工作点统一使用 `sigmoid(logit) > 0.5`；恰好等于 `0.5` 的像素属于背景。
FROC 另行扫描冻结的阈值列表。

## Episodic TTA 权重规则

AdaBN、Binary TENT 和 CR-SITTA 都是单图 episodic：每张图结束后恢复同一个
NS-FPN Source checkpoint。因此不保存或复用“适应后权重”；只记录 Source
checkpoint 路径和 SHA-256、每图更新诊断以及 reset 后状态哈希。任何正式运行
都必须证明下一张图开始前状态与 Source 完全一致。
