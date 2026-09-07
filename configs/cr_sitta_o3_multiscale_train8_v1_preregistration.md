# O3 + 可学习多尺度残差：训练侧 fit smoke v1

## 本轮问题与范围

检验保留原 O3/P2 后，源监督训练能否让轻量多尺度 D0 分支学会有用的局部修正。
本轮不是正式 test、泛化验证或新的 1000 epoch 全训练：只使用固定 NUDT-SIRST
train Pilot64 顺序前八张图及原 13 个条件，共 104 个样本。选择按原顺序，不按标签、
难度或本轮指标挑图。八张同时用于拟合和拟合后读数，没有新增验证集；任何正结果都只能
说明训练拟合信号，不能声称未见图像上的提升。

## 文献来源与独立实现边界

SCTransNet 的 CFN 提供 3×3/5×5 depthwise 多尺度局部处理与通道混合的结构依据：
[论文](https://arxiv.org/html/2401.15583v2)；
[作者代码](https://github.com/xdFai/SCTransNet/blob/main/model/SCTransNet.py)。
本实现独立编写，仅借鉴局部结构；没有复制完整 CFN/ECA、跨层 Transformer 或原训练方案，
不称为 SCTransNet 复现，也不将借鉴结构本身宣称为原创贡献。

恒等起点的设计受 NAFNet 残差缩放初始化启发：
[作者代码](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py)。
这里采用零输出投影而非零 beta/gamma，因此不是相同实现。第一步预计仅输出投影有梯度；
投影经过更新后，上游卷积应有梯度。不能把内部层第一步零梯度误判为模型无法训练。

## 模型与训练

1. 原 NS-FPN best_miou 宿主、O3 目标、P2 BN affine 参数组、半径、视图与安全规则不变。
2. 每图从 Source 开始执行原 O3/P2，在恢复 Source 前捕获其 endpoint D0 特征。
   原 O3 的 loss/safety forward 不经过新分支；捕获后立即精确恢复原模型状态。
3. 新分支输入 detached 的 D0，先做每图 RMS 缩放，再做 1×1 扩展到 32 通道、
   分成各 16 通道的 3×3/5×5 depthwise+ReLU 两支，拼接后 1×1 投影回 16 通道。
   无 bias、BN、dropout、ECA，共 1568 个参数。输出为
   `h + 0.05 * RMS_floor(h) * tanh(branch(h / RMS_floor(h)))`。
4. 输出投影全零，其他卷积正常初始化，初始预测须精确等于同次原 O3 endpoint。
   全局特征 RMS 有界不保证目标、IoU、Pd、Fa 一定不变。
5. 冻结宿主和输出头，只以 train GT 训练分支。GT 从允许 train8 的原始 mask PNG
   独立读取，复用原 nearest 256×256 预处理，核对冻结来源 hash；不绕过旧 outer cache
   的访问限制，不把 GT 传给原 O3。
6. 固定 seed=42，Adam lr=1e-3，batch=4，128 个 optimizer steps；不是 128 epochs。
   损失为普通 BCEWithLogits + 每图 soft IoU loss（平滑常数 1），不新增可调加权项。
   重复由独立 CPU torch Generator 生成的有序随机排列，预算前固定。
7. 只在初始与固定末步读取完整拟合集指标，保存初始和 step_0128 checkpoint；
   不根据训练或测试指标选 best，不自动续训、重试或更改超参数。

训练后的分支是跨图共享的 source-trained 权重，不是每张图重置为随机/零模块。
本轮分支本身没有测试时梯度更新，不是 ITTA/元学习实现；仅原 O3/P2 保持 episodic。
最初零分支与 O3 的精确一致性不等于训练后仍保持全部原收益，需要实际指标检验。

## 预定读数

报告 Source、原 O3/P2、原 O3/P2 + 训练后分支；均使用同一批 train8、同一 13 条件，
阈值严格 `probability > 0.5`，统一 IoU/nIoU/Pd/Fa 评价器。12 个非 clean 条件等权宏平均；
clean 单列。保存所有 Source/O3/训练后预测 mask、概率、逐条件指标及训练损失。
本轮 O3 对照使用同次实际执行得到的 endpoint，不声称其非确定性 CUDA 反向与历史
运行逐字节相同；Source 必须与冻结 teacher 精确相同，零分支必须与本轮 O3 精确相同。
八图拟合读数不与此前 64 图宏平均直接比较，不拼接成新的完整 Pilot64 结果。

- 学习检查：固定末步完整拟合集损失严格低于初始损失。
- 拟合性能信号：非 clean IoU 高于原 O3、Pd 不低于原 O3、Fa 不高于原 O3；
  clean 的 IoU/Pd/Fa 均不劣于原 O3（数值容差 1e-12）。
- 任一失败照实记录；不将训练损失下降或测试通过替代性能改善。
- 即使两项均通过，也不具备泛化/论文资格，不自动批准 full train 或正式 test。

## 保存与冻结

独立保存到 `results/cr_sitta/o3_multiscale_train8_v1/R0/`，不覆盖旧方案。
首次读真实图像/反序列化权重前保存 config、ID、代码和旧输入协议 hash 的 manifest；
训练前独立保存 train GT 来源 receipt。结果完整后写文件 ledger 与 COMPLETE；异常写 FAILED。
只有 README/SUMMARY 说明侧文件可在完成后补充，不改封存数据。
