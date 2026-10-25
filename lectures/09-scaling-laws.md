# 第 09 讲：Scaling Laws 的理论与计算最优配置

> 配套实践：[A3 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a3.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 04-27 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf)。公式中的简化假设会单独说明。

## 1. 幂律是什么，不是什么

经验幂律常写 `L(D)=E+A D^{-α}`，E 是不可约误差或拟合下界。只有 `L−E` 与 D 的 log-log 图近似直线，不是 raw loss 必然直线。数据越多收益越小，指数描述当前分布、架构和训练 recipe 下的趋势。幂律是可检验的经验模型；外推跨语料、tokenizer、架构或几十个数量级时会有风险。

均值估计的教学例：样本独立同分布且有限方差，则样本均值 MSE 为 `σ²/D`。复杂学习问题的 exponent 不会自动等于 1；特征分布、尾部、非参数结构、优化误差与重复数据共同作用。这个例子只解释幂律为何可能出现，不能证明 LLM 的具体指数。

## 2. 模型与数据共同决定 loss

典型参数化：

$$L(N,D)=E+A N^{-α}+B D^{-β},\qquad C≈6ND.$$

这里 N 是 dense 参数口径、D 是 token 数，不能混用 word 数；C 是理论训练 FLOPs，不是 GPU 小时。不同 tokenizer 导致每 token loss 和 D 同时变化，不能直接把另一种 tokenizer 的系数套过来。attention 长序列、MoE、重计算会使 `6ND` 更粗略。

## 3. 固定计算预算的解析最优

令 `D=C/(6N)`，则

$$L(N;C)=E+A N^{-α}+B(6N/C)^β.$$

求导为零：

$$αA N^{-α}=βB(6N/C)^β,$$
$$N^*(C)=\left(\frac{αA}{βB}\right)^{1/(α+β)}(C/6)^{β/(α+β)},$$
$$D^*(C)=\left(\frac{βB}{αA}\right)^{1/(α+β)}(C/6)^{α/(α+β)}.$$

因此 N exponent 与 D exponent 相加为 1。α≈β 时，预算扩大 k 倍，两者约扩大 √k；这不是所有模型永远固定 `D=20N` 的证明，比例系数受拟合与目标影响。

Chinchilla 与 Kaplan 的差异部分来自参数计数、学习率末期衰减、实验覆盖与拟合方式。应比较研究定义与实验 recipe，而不是只背“以前模型太大，后来数据更多”。

## 4. 三种拟合路径

**训练曲线包络：**不同模型训练到不同 D，找每 compute 最低 loss；若依赖某次 run 的中间 checkpoint，LR schedule 不一定处于最佳末期状态。

**IsoFLOPs：**固定 C，枚举 N，对应 `D=C/(6N)`，得到 U 形曲线；取最佳 N，再对 `(C,N*)`、`(C,D*)` 做幂律拟合。A3 2026 合成数据要求取**观察到的离散最低点**，不能悄悄换成二次插值最小值。

**Joint surface：**对多个 `(N,D,L)` 拟合五个正参数，常用 log-residual 或鲁棒损失；E 与小指数容易互相补偿，需要多初值、heldout 与残差诊断。参数拟合成功不等于具有唯一解释。

## 5. 数据重复与分布漂移

有限数据重复多轮通常比新独立数据价值低，`D=训练token总量` 与 `D_unique=唯一token量` 要分开。高质量小数据与低质量海量数据不一定共享一个 scaling curve。混合比例变化也可能让部分领域 loss 改善、另一部分恶化。

如果在近似平坦的最低点附近，各配置差异小于测量噪声，最佳 N 很不确定。应画全部点、重复关键配置，使用 bootstrap 或对 near-optimal region 给范围，而非给出十二位“最佳参数量”。

## 6. Train-optimal 与 Deploy-optimal

计算最优只优化训练预算。服务数亿次时，小模型多训练可能比大模型少训练更合算：总成本是训练成本加未来 serving 成本，受 latency、batch、量化、KV 和质量约束。A3 的官方目标是特定 B200 运行预算，它与 FLOPs 最优又不同：小矩阵/宽深比改变硬件效率。

你的 PRO 6000 小实验可以学习方法，但不能直接将单卡 wall-clock fit 转换成 B200 48-hour 官方最优。报告需分别写 hardware、theoretical FLOPs、measured seconds 和 tokens/s。

## 7. A3 实操与自测

`fit.py` 把合成数据按 C 分组，取离散 minimum，`log N=a log C+log k_N` 线性拟合，计算 D，再预测 1e23、1e24 FLOPs。检查 a+b≈1、系数单位与预算范围。输出 JSON 和图，图上注明 synthetic fixture，不能把它作为真实训练测量。

练习：α=β=0.3，A=B，C扩大100倍，N和D各扩大10倍。追问若每卡峰值翻倍但MFU下降，C 是否仍翻倍？需用真实有效吞吐，而非峰值乘时间。
