# 第 11 讲：Scaling Laws 的实践：初始化、学习率与实验设计

> 配套实践：[A3 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a3.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 05-04 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf)。公式中的简化假设会单独说明。

## 1. Scaling 的真正难处在优化 recipe

上讲公式把 loss 视作 N、D 的函数，实际还依赖初始化、LR、batch、schedule、weight decay、架构与硬件。拿小模型固定 LR 的结果拟合，再用同一 LR 训练大模型，如果优化不再处于近最优区，拟合会混入优化失败。讲义比较 MiniCPM、DeepSeek、Qwen、Llama 等公开配方，重点是它们如何做实验，不是收集一张超参数表。

一个完整设计把变量拆开：决定模型家族（深宽比例、head dim、FFN ratio），先稳定训练；对每规模找 near-optimal LR/batch；再在计算预算下比较 N/D；最后检查预测点。每步都需要保存 seed、tokenizer、数据混合、schedule与最终checkpoint，而不只有一行loss。

## 2. 为什么宽度变化会改变初始化与更新

简单线性层 `h'=Wh`，输入各维方差 O(1)、fan-in n，若权重独立零均值、方差约 1/n，输出方差为 O(1)。这解释常见 fan-in 初始化。训练更新的行为不同：梯度和输入可能相关，不能用独立初始权重的平方和规则分析 `ΔW h`。

μP（maximal update parameterization）的目标是让宽度增大时，激活和更新保持非退化尺度，小模型调出的超参数能转移。它区分 input、hidden、output 参数的初始化和 LR scaling，不是给所有层统一 `lr/=width`。优化器不同、归一化 gain、残差深度、embedding和weight decay都会影响适用性。复现应使用完整参数化定义或库的 base shapes，不凭一句“lr不随宽度变”改模型。

教学检查：记录每层 activation RMS、gradient RMS、update RMS/weight RMS 随宽度的变化。若某层随着宽度趋于零更新或爆炸，平均loss看似正常也可能掩盖局部退化。

## 3. Warmup-Stable-Decay 与 Cosine

cosine schedule通常绑定预定总步数，不同预算的末期训练状态不同。WSD先warmup，较长stable阶段，再短decay；可在同一stable前缀分叉不同decay尾段，减少探索多种预算的成本。不能把stable阶段任一checkpoint的loss当成经过最优末期衰减的final loss。

末期decay可能迅速降低验证loss，因此比较两个模型必须保证都完成相似比例的decay。还应记录追加decay的计算成本，并区分从同一前缀分叉的实验与独立完整训练。

## 4. LR 与 Batch 的网格实验

先做对数网格，例如LR乘数 `[0.3,1,3]`、batch token量 `[0.5,1,2]`，查看loss曲面与稳定区。若最佳点落在边缘，下一轮应向外扩展；若大面积near-optimal，可选硬件效率更高的组合。optimal batch会随目标loss、数据量和优化器变化，不能简单随参数量线性增长。

batch变大可能减少达到目标loss所需step，但每步消耗更多token；低于critical batch时并行化较有效，过大后数据/compute效率收益递减。报告同样的loss需要多少tokens、多少FLOPs、多少seconds，三个答案可能不同。

学习率随宽度/深度变化的经验式应从同一family训练得到，不能跨tokenizer、优化器和架构移植。Muon等优化器还引入矩阵操作成本与特殊参数组：速度和loss需在受控预算上比较，不能只看小模型speedrun。

## 5. PRO 6000 上可完成的小规模路线

建立20M、50M、100M级家族，固定语料、tokenizer、上下文；每个规模先跑短稳定性测试，记录loss spikes。选择至少3个 compute budgets，每个预算用3至5个模型规模，使最低点位于采样区间内部。根据tokens/s先估总耗时，设置中断checkpoint。

这些是本地自学实验建议，不能冒充官方B200 API实测。A3正式API目标是在12 B200-hour拟合预算内预测48 B200-hour最优配置，且需要授权；服务器替代实验把预算改为你的卡时，结论限定在你的硬件。

## 6. 统计可信度与报告

拟合图应有原始点、残差图、观测区间和外推区间。对关键最低点重复seed，给误差范围；留出最高一档compute作为预测验证。不同预算共享训练前缀的结果具有相关性，bootstrap不要把它们当完全独立样本。

硬件效率是N、宽度、batch和T的函数。卡时预算模型至少使用测得的step time估算，而不是只用 `6ND`。最终选择还需满足显存、checkpoint成本和训练稳定性约束。

## 7. 面试追问与自测

“为什么小模型调参能/不能迁移？”答案要覆盖参数化与优化器、激活/更新尺度及经验验证。“为什么同预算实验不公平？”可从LR decay、超参探索额外成本、数据质量、参数计数与硬件效率解释。

练习：将一个模型width乘2，估参数约乘4（若depth固定），对比两种LR下各层update RMS；再检查loss。练习：写出共享stable前缀、三条decay尾段的总token成本，和独立训练三次的成本比较。
