# 第 04 讲：注意力替代架构与 Mixture of Experts

> 配套实践：[A1 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a1.md) · [A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 04-08 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)。公式中的简化假设会单独说明。

## 1. 先区分减少哪一种成本

FlashAttention 保持 dense softmax 的数学函数，减少 IO 与中间矩阵存储；稀疏 attention 改变可见 token 集；线性 attention/SSM 改变状态表示与算子；MoE 主要稀疏化 FFN 参数激活。它们不解决同一个问题。长上下文训练、逐 token 解码、总参数存储、跨卡通信都需要分别核算。

## 2. 线性 attention 的重排与递归

普通 attention 为 `softmax(QKᵀ/√d)V`，softmax 使矩阵重排不能直接进行。若换成核特征映射 `φ`：

$$o_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t},\quad
S_t=\sum_{s\le t}\phi(k_s)v_s^\top,\ z_t=\sum_{s\le t}\phi(k_s).$$

于是 `S_t=S_{t-1}+φ(k_t)v_tᵀ`，decode 维护固定维状态，不必存所有 KV。每步通常有 `O(d_k d_v)` 运算；训练可以用 parallel scan 而非纯 Python 顺序循环。这里的核/归一化变化意味着一般不等于 softmax attention。固定容量状态会将历史压缩，精确找回罕见 key 的能力要单测。

加入忘记门 `S_t=λ_t S_{t-1}+k_t v_tᵀ`，允许逐步衰减历史。delta rule 类更新还使用当前预测残差纠正状态，例如简化表达 `S_t=S_{t-1}+β_t k_t(v_t-S_{t-1}ᵀk_t)ᵀ`，避免只累积信息。Mamba、Mamba-2 与 gated delta 架构有各自参数化、门控和实现，不能将这个教学表达当成完整模型定义。

混合模型在若干线性/状态层之间插入全局 attention 层，减少成本同时保留精确检索路径。讲义用多种公开模型作案例；具体混合比例属于该模型版本经验，并不是通用 recipe。

## 3. MoE 的计算图

输入 token x，经 router 得到专家分数 `s=W_r x`，选择 top-k 集合 `S(x)`：

$$y=\sum_{e\in S(x)}g_e(x)\,\mathrm{FFN}_e(x).$$

有 E 个专家时总参数随 E 增长，而每个 token 只激活 k 个，近似 FFN 运算随 k 增长。必须分别报告总参数、每 token 激活参数与共享模块参数。不能把“总 200B、激活 20B”的训练 FLOPs 算成 dense 200B，也不能把显存需求按 20B 算。

若每专家 FFN 参数为 P、总输入 token M，理想每专家接收 `Mk/E` 个路由，实际分布可能倾斜。capacity factor 给专家设置容量，溢出 token 可能被丢弃或延迟；dropless routing 避免丢 token，却需要可变大小调度。token-choice 和 expert-choice 改变哪个轴做 top-k，与 causal serving、负载控制之间有不同取舍。

## 4. Router 为什么难训练

top-k 选择是离散的，常见实现通过被选择专家的连续权重与其他训练技巧更新 router。某专家早期略好就获得更多 token，进一步变强，产生赢家通吃。辅助负载均衡损失常结合专家接收比例 `f_e` 与平均路由概率 `p_e`，形式如 `E Σ f_e p_e`，但系数过大又可能干扰主任务。

其他方法使用 router jitter、noise 或按负载调整专家 bias。某些 bias 只影响选择而不参与混合权重计算，必须按论文实现，不能在所有位置统一加。router 的 FP32、router z-loss 常用于稳定分数。MoE 的输出也可能依赖 batch 内竞争容量，比较推理结果要控制批次与 routing 实现。

细粒度专家增加专家数量、减小单专家宽度；共享专家为每 token 提供通用变换。两者可能提高表示能力，也增加 dispatch 和 gather 成本。是否有效要比较相同 active FLOPs、相同数据与相同训练预算。

## 5. Expert Parallelism 的实际代价

router 先产生 token→expert 的映射，再 all-to-all 把 token 发送到专家所在 GPU，执行 grouped GEMM，然后反向路由聚合。小专家批次导致 matmul 利用率低；跨节点链路、负载尾部与同步使 theoretical FLOPs savings 不一定变成 wall-clock savings。容量浪费、padding、通信、router 工作都应计入。

你的单卡 96GB 可以实现小 MoE 验证 routing 与负载统计，也能观察专家 batch size 的效率；它不能验证真实跨节点 all-to-all 带宽与吞吐扩展。与 A2 结合时，应先做单机 toy MoE 正确性，再在多卡环境测 expert parallel。

## 6. MLA 与 MTP 的位置

MLA 用低维 latent 表示压缩历史 KV，需要考虑 RoPE 相关部分如何解耦，且训练与推理投影可能重排；并非简单把 head 数改成一。MTP 给未来多个 token 加预测训练目标，可改善训练信号，也能支持 speculative decoding；最终服务仍需验证目标分布与接受率。这两项与 MoE 可以组合，但独立解决不同成本。

## 7. 自测与实践

- 为什么增加 E 不会免费提高模型？答案应含总权重存储、通信、专家小批次与路由训练。
- 为什么线性 attention 可以固定显存解码？因为把历史压进固定维状态，代价是表示/检索的变化。
- 练习：8 专家 top-2，1000 个 token。均匀负载期望每专家 250 次选择；画出实际 histogram 并记录 max/mean、丢弃比例与辅助 loss。
- 练习：构造远距 key-value 检索任务，对 dense、窗口、简单线性递归做同参数量对照，分别测准确率、峰值内存与随 T 增长的时间。

大厂面试常要求从“稀疏参数更大”继续解释到通信拓扑与服务动态 batch，学习时要把算法和系统放在同一张成本表。
