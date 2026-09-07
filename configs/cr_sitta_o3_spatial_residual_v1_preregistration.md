# CR-SITTA：O3 空间残差模型候选 v1

## 目标与证据边界

用户要求从已有设计出发，针对剩余性能问题改进模型。本候选继承 v5 O3 的源锚定、前景/背景分别归一化目标及单图更新，不是重新训练 NS-FPN。此前 O3/P2 的 Gaussian noise、Gaussian blur 家族平均 IoU 仍低于 Source，整体假警也有增加。空间修正能力不足是待验证假设，并非已证明的原因。

本轮只评价一个预先确定的模型候选，不依据本轮指标搜索强度、半径、损失权重或 teacher。旧 B4 及 LF/IPMA 代码、配置、结果均不修改，不重启旧训练服务。

## 模型改动

在冻结 NS-FPN 的 decoder_0 特征 h 与 output_0 之间接入逐通道 3×3 空间残差：

`s = stopgrad(sqrt(max(mean(h²), 1e-12)))`

`h_new = h + 0.05 * s * tanh(DWConv3x3(h / s, K))`

K 为 16×1×3×3，共 144 个零初始化参数；无 bias、BN 或随机层。每张图独立提出一次绝对 L2 半径 0.25 的更新，沿用旧 Armijo 回溯和三项无标签安全检查；结束后 K 和梯度严格恢复零值。观测图和原 mild_contrast_0p95 学生图的宿主特征均冻结、detach，不经过 SFS backward。teacher 仍为冻结 Source identity，flip4 只提供原有 uncertainty，不引入 P4 融合 teacher。

旧 P2 是 416 个 BN affine 参数、相对源参数半径 0.0005；新候选是 144 个零中心空间参数、绝对半径 0.25。因此本轮比较的是参数化、插入位置和预算组成的模型改进方案，不是等容量或等函数步长的单因素因果消融，也不能把任意提升都归因于 3×3 卷积。

## 数据与执行

数据入口及固定 train/test 划分沿用仓库相对路径 `datasets/` 所绑定的原协议；没有 validation split，也不新建一个。使用原 B4 的三个数据集各固定 64 张 train pilot、相同有序 ID、已冻结污染缓存和原 best_miou 宿主权重。每数据集 13 条件：clean 与四类污染 S1/S3/S5，共 2496 个独立 episode。不是把训练集改为 64 张，也不是正式 test 实验。

运行前冻结配置、代码、原协议、ID、输入缓存、teacher、checkpoint 和历史结果 hash。三数据集全部候选预测完成并封存后，外层评价器才加载 train GT；GT 不进入候选更新函数。新 Source 概率必须逐元素精确复现已冻结 teacher；外层 Source 计数必须与历史同条件一致，否则中止比较。

预检备注（尚未运行真实图像/读取本候选性能）：当前两个包导出文件 `tta/adapters/__init__.py`、`tta/objectives/__init__.py` 与旧 B4 hash 不同。已用 Git 中 `f40d649` 的原文件验证历史 hash，并检查 `f40d649..2cb5af0` 差异：只增加后续模块的 import/export，另有 adapters 文档字符串改名，原有导入对象未替换。配置明确列出这两个文件的历史/当前精确 hash，其他旧关键代码仍须全部与 B4 hash 一致；不允许任意放宽旧代码校验。新模块的 O3 损失/梯度一致性另由合成测试验证。此兼容备注不修改方法超参数或性能门槛。

阈值及评价规则沿用旧评价器（阈值 0.5），保存所有预测 mask、source/post 概率、每图更新核/梯度和诊断，结果存放 `results/cr_sitta/o3_spatial_residual_v1/`。此轮不扩展 best_pd 宿主；只有该模型候选值得继续时，后续才在单独冻结协议下完成双权重验证，不能声称本轮代表双权重结果。

## 预先固定的性能目标

只在 36 个非 clean 条件上计算总体宏平均；每家族 9 条件，每数据集 12 条件。clean 单列。IoU、Pd 以比例计算，差值展示为百分点；Fa 单位为每百万像素。保留所有原始结果，包括失败条件。

1. Gaussian noise 和 Gaussian blur 各家族平均 IoU 均高于 Source。
2. low contrast、stripe noise 各家族平均 IoU 不低于旧 O3/P2。
3. 36 条件平均 IoU 高于旧 O3/P2，Pd 不低于旧 O3/P2，Fa 不高于 Source。
4. 每个 clean 的 IoU/Pd 绝对下降不超过 0.002，Fa 不高于 Source。

数值比较容差固定 1e-12。以上是模型是否值得继续的性能目标，不代表统计显著性、论文结论或原 B4 科学门已通过。工程正确性只决定结果能否评价，不能代替性能提升。任何目标失败都如实记录；本脚本不自动重试调参、不自动进入正式 test 或 1000 epoch 全训练。
