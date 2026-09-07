# CR-SITTA v9：LF R0–R3 实施补充协议

本文件在本轮真实图像数值诊断之前确定；最终以 `PREREGISTRATION.json` 的代码、配置、父结果和输入 hash 为准。它不是性能结果，也不修改 v9 原文或旧实验。

## 研究问题与证据

本轮只问：修复 LF 共轭采样、DC、通道共享和衰减后，在原训练 Pilot64 上能否降低预定义的目标可见性风险，同时保留非零图像扰动？不检验检测性能、TTA 收益或泛化，不启动 IPMA/全训练，不访问 test，不新增验证集。

| 主张 | 所需证据 | 本轮可支持的边界 |
| --- | --- | --- |
| 算子符合定义 | 合成张量数值测试、实际随机场与增益 hash | 数学/工程正确性，不等于目标保持 |
| 输入可与旧 LF 比较 | 同 Pilot64、原两种视图、输入/GT/目标区域 hash 与旧记录相同 | 确定性训练侧诊断，不是历史训练随机流重放 |
| 候选通过风险筛查 | 每个数据集×视图分别报告原 E1、联合风险及原始分母 | 只允许后续无标签信号工程检查，不是论文安全结论 |

## 在新结果前补齐的规则

1. 原 E1 使用旧资格：有效 ring 且 `abs(clean_contrast)>=1/255`；原风险为绝对对比保留率 `<0.30`。联合风险为原风险或符号反转，严格沿用同一个 eligible 分母。无效 ring、低对比目标、空目标裁剪均保留并单列，不能当成零风险。
2. 每个 dataset×view 的原 E1 和联合风险比例均须 `<=0.10`；任一空分母为证据不足，任一超过上限为风险失败，不能宏平均抵消。
3. 排除数值恒等：每个 dataset×view 的 64 图 post-clamp RGB 扰动 RMS 中位数必须 `>1e-6`，且至少 90%（64 图时至少 58 张）严格超过 `1e-6`。常量/无频谱能量图不删除。此固定阈值是 float32 数值级工程下限，不是噪声有效性或 target preservation 的已验证阈值。teacher–student 信号仍须下一阶段独立检查。
4. 两完整候选固定为 L4a α=0.25 和 L4b α=0.50；禁止本轮增加 seed、改强度、重选 crop。R3 只提名通过的优先候选 L4a。即使只有 L4b 通过，也只记录备用资格，不自动根据标签结果反选。L4b 的后续使用必须另有无标签 teacher 信号检查和绑定 receipt，本轮不实现该入口。
5. CI 为 source image ID cluster percentile bootstrap，固定 seed=42、2000 次、95%区间；paired delta 使用同一抽样索引。零分母重采样不补零，记录数量；区间不作为准入条件，也不声称总体风险已证明低于 10%。

## 随机场与消融

L1 每 orbit 一次抽样，不保护 DC、不共享通道、α=1；L2 只加 DC 保护；L3 再加物理通道共享；L4a/b 再固定软衰减。

新变体一律用包含 DC 的完整共轭闭包、按 canonical flattened index 排序，生成 `[B,C,O]` CPU float64 均匀场。seed 从固定 namespace、global seed、dataset、image ID、view 稳定派生，不含 variant、worker 或遍历顺序。共享版只使用 channel 0 的同一个场；DC 保护只覆盖增益，不改变 RNG 形状。逐 image/view 存一份完整随机场，逐变体记录实际 gain/hash/keep/能量，不宣称与旧 L0 逐频率随机掩码一致。

生产 LF 接口无标签和文件路径；train mask 仅由外层可见性评价读取，不控制每图扰动，不按目标失败重采样。

## 不在本轮实施的部分

IPMA 仍使用固定 NUDT D0-A epoch1000 作为未来 host，不是 NS-FPN `best_miou`。其 16 图、8 fit/8 check 都在官方 train 内。本轮不创建 adapter、不缓存 D0、不执行 inner/outer 更新；微型训练固定终点、seed、teacher 信号底噪和 IoU/PD/Fa 容差应在其执行前另行冻结。

`q` 与 `1-q` 的有效质量检查不是候选可靠性门；未来首版若不另定义 label-free 可靠性规则，应明确允许 background-only proxy，不能声称无可靠目标会自动 abstain。本轮不新增复杂候选控制算法来修饰此边界。

## 交付与授权

所有结果追加到 `results/cr_sitta/lf_repair_audit_v2`。R0 绑定旧五份 manifest、旧配置、代码提交和本轮未提交代码的独立 hash；R1 单元测试；R2 保存逐图、逐目标、随机场和按 cell 汇总；R3 从原始记录重算并核对哈希后封存 receipt。

默认 CLI 只读；初始化、R2 和 R3 的任何写入均需显式 `--execute`。完整但科学失败可以封存，不能授权训练。旧脚本、旧导出、旧结果和所有原权重保持不变。
