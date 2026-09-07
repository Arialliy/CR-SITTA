# O3 多尺度残差 v2：背景正向增量惩罚

## 依据与本轮唯一训练改动

v1 在同八张训练图拟合后，nonclean 总体有收益，但 blur 三档 FP 增加，clean 一个
原假警组件扩大两个像素。因此本轮保持现有 1568 参数多尺度结构，仅改变监督训练损失。
这是从已观察到的训练侧错误出发的改进假设，不是新文献已证明的机制或新的原创性结论。

原监督损失 `L_seg` 为 mean BCEWithLogits + mean per-image soft IoU loss，保持不变。
对第 i 张训练图，新增：

`L_bg_i = sum((1-y_i) * relu(sigmoid(z_i)-sigmoid(z_o3_i))) / max(sum(y_i), 1)`

`L_total = L_seg + mean_i(L_bg_i)`，固定系数为 1.0。

- `z_o3` 由同一批 detached O3 特征送入冻结输出头，在 no_grad 中计算并 detach；
  不用不同 batch 大小的缓存概率充当 loss 中的参考，避免初始微小数值差造成假惩罚。
- 只惩罚 **GT 背景**上概率的正向增量，允许背景降低、允许 GT 目标位置提高概率。
- 按每图目标面积（下限1）归一，防止整个背景平均稀释少量背景扩张；空目标图也有限。
- 系数1不代表与 mean BCE 同量级；很小/空目标图的惩罚可能较强，可能损失 Pd/IoU。
  必须分别报告原分割 loss、新背景项及总 loss，不用总 loss 下降代替性能结论。
- 这是训练时软惩罚，不是 Fa 硬上界，不保证所有像素均不扩张，也不能保证未见图像泛化。
- GT 只进入 source 监督 loss；不传给原 O3，不进入模型 forward 或推理筛选。
  不按已知错误图 ID、坐标、corruption 标签执行运行期特例。

## 固定比较条件

复用已封存 `o3_multiscale_train8_v1/R0` 的八图、13 条件、104 份 float32 O3 特征、
GT、Source/O3/v1 概率及 128 步样本顺序；验证全部父产物哈希和原输入来源。
父 GT 是上一轮独立 source-supervised API 保存的 train8 标签，不访问旧 outer-only cache。
这轮不重新执行 O3，不重新生成任何噪声，不打开新的训练图、GT 或 test。

从父 **initial.pth.tar** 同一初始化重新训练，使用新建空状态 Adam、同 lr=1e-3、
batch=4、128次更新；不是从 v1 末权重再多训练128次。这样在本拟合范围内，训练方法的
唯一有意改变是上述新增损失。v1 是结构与训练方案的改进基础，训练后权重不被覆盖。

训练前重放全部 O3 和 v1 endpoint，分别精确核对原保存概率；核对初始特征 identity、
原分割完整拟合集 loss、首步样本/梯度范数与父记录（父运行未保存首步完整梯度张量，
不宣称真实父梯度逐元素重放）。任何不匹配均停止并保留失败现场，
不放宽比较规则、不静默替换缓存。训练后保持相同阈值 `p > 0.5` 与统一评价器。

## 训练范围与选模限制

仅 NUDT-SIRST 原 train Pilot64 顺序前八张，拟合与读数同图，没有新验证集。
完整数据划分仍在 `datasets/NUDT-SIRST/img_idx/`，train 为 663 张，未改成八张。
只有 best_miou 宿主；不是 best_pd 对比、1000 epoch 全训练或正式 test。
只保存固定第128步权重，不根据中途指标选择 best 或搜索损失系数。

## 本轮预定判据

12 个 nonclean 条件等权宏平均、各家族三档等权平均、clean 单列，容差1e-12：

1. 完整拟合集总 loss 低于初始总 loss（学习检查）。
2. nonclean IoU 高于 v1，Pd 不低于 v1，Fa 不高于 v1。
3. low contrast、stripe 各家族 IoU 不低于 v1，检查已有收益是否丢失。
4. Gaussian blur、noise 各家族 IoU 不低于 O3、Fa 不高于 O3，检查原短板。
5. clean IoU/Pd 不低于 O3，Fa 不高于 O3。

共13个布尔判据，任何失败都照实报告。blur/noise 家族约束是在读取本轮结果前
新增的 **v2 判据**，不回写或重新解释 v1 的7项判据。
即使全部满足，也只是重复使用八张训练图上的拟合信号，不是统计显著性或泛化证明；
不自动全训练或进入正式 test。一次固定运行结束后不自动更改系数、阈值或预算。

## 结果保存

独立路径：`results/cr_sitta/o3_multiscale_guard_train8_v2/R0/`。
保存新 manifest、输入来源与复用 receipt、初始重放核对、固定训练顺序和日志、三项完整
拟合 loss、末权重、guarded 概率、四方案全部416张预测 mask、13条件全指标、汇总和文件ledger。
在真实数组/权重加载前冻结 manifest；结束复核哈希后写 COMPLETE，异常写 FAILED。
父结果、父模型代码和所有旧指标均不可修改；说明侧文件写在新 R0 外。
