# v9 R4：IPMA 16 图工程检查补充协议

本协议在任何本轮真实图像/特征数值检查前冻结。阶段仅为 R4，不是 R5 meta-training，不重新训练宿主，不解禁 full/test。

## 问题与证据

| 问题 | 本轮证据 | 不支持的主张 |
| --- | --- | --- |
| δ=0 是否严格保持固定宿主输出 | 同 GPU/float32 的原 output_0 与 adapter δ=0 逐元素完全相等，重放与 A→B→A 检查 | 不等于对 NS-FPN best_miou 的恒等 |
| 已过可见性门的 LF 是否产生实际可更新信号 | 冻结 observed teacher、probe/observed logit RMS、内梯度、post logit RMS | 不等于检测收益或候选真实性 |
| 元梯度是否连接正确 | 8 fit 图原 SLS 梯度、独立 CPU double 中心有限差分 | 不等于学习到有效方向或泛化 |
| 状态和标签是否隔离 | host 参数/BN/modes、adapter 状态、每图新 δ、标签访问计数及独立接口 | 无外部签名的哈希并非抵抗本机特权造假的证明 |

## 数据与标签

只取原 NUDT Pilot64 文件顺序前 16 张，固定原 train_crop_224。原训练集/官方 train-test ID 不变，不创建验证集。

前 8 张为 `meta_fit_train8`，只在全部 label-free 预测生成之后读取其 GT，诊断原 SLS 与 meta gradient；不执行 optimizer.step。后 8 张为 `meta_check_train8`，本轮不解码 GT、不计算其有标签指标，不据其标签调整学习率、阈值或强度。继承的父证据 hash 校验可能读取已经封存的 train 文件字节，但不把 check mask 解码或传入模型。

image-only loader 通过占位 mask 复用原几何变换，占位值不是真实 GT，不能用于 outer 评价。fit mask loader 使用同随机几何且先校验 fit allowlist。输入及 LF 物理输出 hash 必须与 R2 封存对应图/视图/L4a 一致；fit GT hash 必须与原封存一致。空 crop 不删除、不替换。

## 固定算法与精度

宿主为 NUDT D0-A fixed epoch1000，载入已验证的 weights-only 副本，原权重不修改。所有宿主参数、BN running statistics 均冻结；只在 eval/no_grad 下获取 D0。LF 仍是已批准 L4a α=.25、ratio=.20、q=.50，复用 R2 同 seed/随机场。

adapter 为 16 channels、rank4、5 bases；CPU seed42 非零初始化 down/up，随后转原生 GPU float32。δ 每图从零开始、外部函数变量，不进模型状态。inner 固定一步 lr=.05、L2 半径 .01。proxy 固定权重 q/1-q 的区域归一化 soft BCE；没有额外可靠候选筛选，不能将正权重质量说成候选可靠性。background-only teacher 允许，但不伪造前景或把无效更新报作成功。

原生输出、重放、更新与预测 mask 都使用 GPU float32、autocast/TF32 关闭、严格确定性。head 参数冻结，但 adapter→head 的输入梯度开启。真正的 SFS 不进入元梯度图。

有限差分单独将同一 frozen float32 D0/head/teacher 数值转 CPU float64；teacher 在各 perturbation 间固定。此检查比较同一个 double 子图的 autograd 与数值导数，不将 double 输出冒充原生 float32 预测。SLS 输入为 raw logits（函数内部已有 sigmoid），固定 warm_epoch=5、epoch=999、with_shape=true。

## 新结果前固定的工程门

1. 所有 16 图完整、finite、δ=0 identity exact、同图重放 exact，host/BN/adapter 状态与标签边界必须通过。任何协议或非有限错误 fail closed。
2. 每图同时满足 inner gradient norm >1e-8、probe-observed logit RMS >1e-6、post-observed logit RMS >1e-6 才计为有信息；至少 12/16 图满足（75%）。其余图保留并报非信息性，不换图。这是初版工程覆盖线，不是文献安全阈值。
3. 前 8 图至少 6 图 native meta gradient norm >1e-8；finite-difference 检查均须符合预设容差。方向固定为归一化 meta gradient 和 seed20260907 的归一化随机方向；零梯度方向不可伪称非零验证，需明确记录并受上述非零数量门约束。
4. 中心差分 epsilon=1e-4；误差容差 `1e-6 + 1e-3 * max(abs(autograd), abs(FD))`。投影是否生效如实记录，不为了使真实图进入某个投影分支修改 lr/radius；两分支另有合成测试。
5. 前8图 pre/post SLS、统一 IoU/Pd/Fa、目标转移只作诊断，不用改善与否当本工程门，也不作为学习后收益；当前 basis 未训练。所有 16 图的实际预测 mask 保存，但后8图不计算有标签指标。

R4 通过仅得到进入后续有界研究的工程资格；R5 仍须单独冻结最多64次 outer 的固定终点、8fit顺序、同初始化 fixed-basis 对照、以及有标签效用/Fa 容差。本轮 `outer_optimizer_steps=0`，不启动 R5。若尺度或梯度门失败，封存负结果，不在本轮看结果改阈值、LR、α、初始化 seed 或采样。

## 产物与命令

全部新增结果位于 `results/cr_sitta/ipma_micro16_v1/engineering`：预运行合同、代码/checkpoint/父Gate/ID哈希、detached D0 cache、初始 basis、逐图 label-free 工程记录、fit-only 外层诊断、全部预测 mask、准入 receipt、manifest、执行日志及 SUMMARY。

CLI 默认只读；写入必须显式 `--execute --engineering-only`。已存在完整或部分输出拒绝覆盖。旧 LF/P0–P3 文件及任何原权重不修改。
