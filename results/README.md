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
`configs/artifact_eligibility_registry_v2.yaml` 登记；Stage B4 R0 负结果由 v3
登记，Stage C0 R0 负结果再由
`configs/artifact_eligibility_registry_v4.yaml` 以 v3 为不可变父节点追加登记，
不改写 v1/v2/v3。运行
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

P3 Stage-B 的训练侧 development 链已经完成到 B4。B1 v2 在 3×13×64=2,496
个 episode 上完成前景/背景梯度机制分解；首次 aggregate 因冻结容器的 tuple/list
适配类型不一致而在发布前失败，随后由 append-only 的
`p3_stage_b1_aggregate_recovery_v2_1/aggregate_phase_v2_1/R0/` 恢复。恢复只做
递归容器适配，canonical 语义哈希保持一致，未改变数据、覆盖、统计或阈值，也未
执行候选选择。

B3 objective/parameter-space screen 位于
`p3_stage_b3_objective_space_screen_v1/aggregate_phase/R0/`。它在 hash-selected
Pilot16 上筛选 10 个候选，`O3_P2`、`O4_P2`、`O3_DecoderFiLM` 和
`O4_DecoderFiLM` 共 4 个候选通过冻结门并进入 B4。B4 的完整 Pilot64 R0 位于
`p3_stage_b4_full_pilot64_proposal_gate_v1/aggregate_phase/R0/`，对 4 个候选各
执行 2,496 个 episode（合计 9,984）。最终 `scientific_status` 为
`scientific_no_eligible`：`O3_P2` 与 `O4_P2` 的 non-clean macro ΔIoU 均为
`+0.000357225`，但未满足正收益 corruption-family 覆盖门；两个 DecoderFiLM
候选还未满足 dataset/family 覆盖与 threshold-crossing episode fraction 门。
因此 selected-for-R1/R2 为空，B5、正式 test 和后续风险控制器均未授权。

上述终止结论已按 C-0 追加封存在
`cr_sitta/stage_b4_r0_negative_v1/`。该六文件目录用原 aggregate、candidate 和
outer 的 manifest/COMPLETE 哈希绑定 9,984 条 candidate episode、9,984 条 outer
episode、0/4 合格候选及 O4 guard 在 4,992 条 O4 episode 中零激活的事实。Stage B4
只使用 train Pilot64：方法侧标签访问为 0，train mask 仅由后置 outer evaluator
读取，validation/test 图像与 mask payload 访问均为 0；没有创建 validation split。
这不消除上游 `best_miou` checkpoint 由固定 test 反复评价选出的事实，因此该档案
仍是 `development_test_selected`、`paper_result: false`。负结论只适用于本次 Stage
B4 proposal family；完整 CR-SITTA 未被评价，R1/R2、B5 和正式 test 均保持禁止。

Stage C 的 ASB-SFR 无优化信号审计已在固定 train Pilot64 上完成，源产物位于
`cr_sitta/p3_stage_c0_signal_audit_v2/`，终止结论追加封存在
`cr_sitta/stage_c0_r0_negative_v1/`。3,022/4,608 个 non-clean probe episode
产生 active support，其中 2,453/3,022 具有候选邻近 support；这两个共享门通过。
但 R-E1、R-D0、P2 的 finite nonzero proxy gradient 覆盖率均仅为 65.58%，低于
冻结的 80% 门；宏观 outer-gradient cosine 分别为 0.04964、0.04595、0.02525，
也均未严格超过 0.08。虽然每个空间都满足 2/3 正 cosine 数据集、3/4 改善退化族
和 Source identity bit-exact，仍然没有任何合格空间，故 Stage C1、R1/R2 和正式
test 均未授权。

v2 aggregate 在完整结果落盘后的自验证因 Python `tuple()` 与 JSON `[]` 的容器
比较退出 2；该原始 aggregate 不表述为“v2 自验证通过”。独立冻结的
`cr_sitta/p3_stage_c0_aggregate_recovery_v1/RECOVERY_VERIFIED_RECEIPT.json`
先复现唯一终端错误，再只对 `StageCAuthorization.parameter_space_ids` 做内存
tuple→list 规范化，完整验证器随后通过；aggregate evidence、science decision
和 authorization 又被独立重算且完全一致。该勘误没有改变数据、覆盖、阈值、
统计或科学结论，且没有读取 image/target/validation/test payload。C0 负结果只
适用于 ASB-SFR signal gate，不表示完整 CR-SITTA 已失败或已被评价。

Stage C0 未过门后，训练兼容化分支 D0-A 已建立独立入口
`train_cr_sitta_d0a.py`。v1 train-only smoke 在 NUDT-SIRST 的第 2 个 HF
step 因未校准 BN running statistics 产生非有限 degraded loss，已原样登记在
`cr_sitta/engineering_smoke/d0a_supervised_lfhf_train_v1/`，没有启动 1000e。
v2 仅将 degraded 分支改为使用 batch statistics 且不持久更新 BN buffers；三个
数据集的两步真实 GPU smoke 均通过，证据汇总在
`cr_sitta/engineering_smoke/d0a_supervised_lfhf_train_v2/SMOKE_GATE.json`：
每个数据集 LF/HF 各一步、loss/gradient 有限、505-key 双模型 strict load 通过、
BN `num_batches_tracked` 均只增加 2，系统 `openat` 审计的 test/validation 命中为
0。全仓库 1782 项测试退出码为 0。

v2 full run 写入 `cr_sitta/d0a_supervised_lfhf_train_v2/<dataset>/`，只保存
逐 epoch `last.pth.tar` 和固定终点 `epoch_1000_train_only.pth.tar`；训练期间不
构造 test/validation loader，也不产生 `best_miou`/`best_pd`。这三个权重完成后
必须先回到冻结 train Pilot64 重跑 Stage-C0 梯度门。这里的 D0-A 只是
proxy-compatible training 假设，不能提前表述为 CR-SITTA/TTA 已成功。

不反序列化 checkpoint 的当前进度查询：

```bash
./.conda/bin/python scripts/report_cr_sitta_d0a_status.py
./.conda/bin/python scripts/report_cr_sitta_d0a_status.py --json
```

`last.pth.tar` 是含 optimizer 与进程 RNG 的训练恢复产物，不是 D0-B
直接读取的推理权重。每个 1000 epoch 运行完成后，必须用
`export_cr_sitta_d0a_safe_checkpoint.py` 校验 source checkpoint、
`run_contract.json` 和 `FULL_TRAIN_FREEZE.json` 的独立 SHA-256，再生成
no-replace 的 `epoch_1000_train_only_safe.pth.tar` 与最后发布的
`SAFE_EXPORT.json` 哨兵。不得对正在写入的 `last.pth.tar` 执行导出。

若 v2 worker 在 1000 epoch 前中断，原始 CUDA `--resume` 路径会把 CPU
RNG tensor 映射到 GPU，因此不可直接使用。只能在确认 worker
已停止后运行 `recover_cr_sitta_d0a_v2.py`；CLI 必须传入独立记录的
source/run-contract/full-freeze SHA-256，并显式确认进程已停止和本地
pickle 可信，具体参数以 `--help` 为准。工具会校验连续 JSONL
前缀、checkpoint epoch/step、训练 split 和全部冻结哈希，并只发布
绑定 receipt 的 RNG-sanitized epoch-boundary 恢复权重。该路径不声称
跨 CUDA 进程 bit-exact。

D0-B 已以独立入口 `run_cr_sitta_d0b_gradient_gate_v1.py` 实现。当前只允许
运行不写产物的 `preflight`；在三份 `SAFE_EXPORT.json` 全部存在并
校验通过之前，它必须返回 `ready=false`。准备完成后的顺序固定为
`freeze -> teacher -> candidate -> outer -> aggregate`；teacher、candidate、
proxy/outer gradient 和 aggregate 都从 D0-A 新 checkpoint 重建，不读取旧
C0 数值产物。D0-B 最多只能授权 D1 train-internal OOF，不能直接授权
formal test。

D0-A 到 D0-B 的正式交接由
`scripts/orchestrate_cr_sitta_d0a_to_d0b_v1.py` 管理。默认 `status` 和显式
`dry-run` 均为零写入；只有 `execute` 会等待三组 1000 epoch 全部完成，逐组
深验 train-only 日志、final/last checkpoint 与访问计数，再生成安全权重并按
“三个 teacher 全部完成 -> 三个 candidate 全部完成 -> 三个 outer 全部完成”
的全局屏障运行 D0-B。编排产物统一保存在
`cr_sitta/d0a_to_d0b_orchestrator_v1/`，D0-B 产物保存在
`cr_sitta/d0b_checkpoint_rebound_gradient_gate_v1/`。终点固定为 D0-B
`verify` 和 `PIPELINE_COMPLETE.json`；无论正负结果都不自动启动 D1，也不访问
formal test。

2026-09-07 的续跑与单数据集评测单独登记在
`cr_sitta/d0a_continuation_20260907/EXECUTION_PLAN.json`。NUDT-SIRST 已完成
1000 epoch 并导出 `epoch_1000_train_only_safe.pth.tar`；应用户在讨论已完成
权重的性能后提出的继续要求，先运行其 clean fixed-test **开发评测**，结果进入
`cr_sitta/d0a_development_test_v1/NUDT-SIRST/`，保存全部 664 个预测 mask 和
概率图，并同时比较 baseline 的 `best_miou`、`best_pd`。这是对原先
“三组训练与 D0-B 完成后再评价”顺序的显式调整，不能作为 D0-B/D1 的晋级
依据，也不记作正式 test 或完整 CR-SITTA TTA 的结果。训练期和安全导出期的
zero-test-access 记录仅描述其各自阶段，不能用于声称本次开发评测没有读取 test。

该评测现已完成，`COMPLETE.json` 与全量产物核验通过：664 张预测 mask、
664 张 float32 概率图及 664 条逐图记录均齐全。按与 baseline 相同的
legacy 官方口径，D0-A epoch1000 的 mIoU / PD / Fa(×10⁻⁶) 为
79.124396% / 97.248677% / 25.461955；相对 `best_miou`，mIoU 下降
1.091306 个百分点、PD 下降 0.423280 个百分点、Fa 增加 6.549330，
没有刷新 clean 性能。相对 `best_pd` 仅 mIoU 增加 0.057874 个百分点，
PD 和 Fa 均退步。双轴表见
`cr_sitta/d0a_development_test_v1/NUDT-SIRST/comparison.md`，独立产物核验见
`cr_sitta/d0a_continuation_20260907/NUDT_EVALUATION_QA.json`。
另存的 unified Fa 为 25.393015，因目标匹配算法不同，不能与本表 legacy Fa
混用；本次没有更改任一评价器或冻结协议。

IRSTD-1K 在核验 epoch-565 断点后，通过独立的 `recovery/RECOVERY.json`
恢复到原目录继续训练；NUAA-SIRST 按冻结协议从头训练。两项任务由用户
systemd 服务持久运行，日志分别为
`cr_sitta/d0a_continuation_20260907/irstd_training.log` 和
`cr_sitta/d0a_continuation_20260907/nuaa_training.log`。旧编排进程已经退出；
其原先关于安全导出顺序的收据不再描述此次单数据集提前评测，因此本次不直接
重新启动该版本。后续 D0-B 仍需使用冻结配置完成原定检查。

上述 Stage-B artifact 全部固定为 source-train-derived、development-only 和
`paper_result: false`。其冻结配置还绑定本机保留的 v5 决策溯源 memo 以及被 Git
忽略的 B1/B2/B3 前序 receipts；公开仓库包含计算配置、runner、门函数和单元测试，
但不包含可在 fresh clone 中独立复验的完整历史 artifact 包。

Stage-C/D0 冻结配置同样绑定本机保留的 v6/v7 决策溯源 memo、编译环境文件和被
Git 忽略的结果 receipts。公开仓库不发布这些私有输入或生成产物；fresh clone
默认运行可移植单元测试，只有在恢复完整本地产物后才应设置
`NS_FPN_RUN_LOCAL_ARTIFACT_TESTS=1` 执行精确历史集成复验。

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
