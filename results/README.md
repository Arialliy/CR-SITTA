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
