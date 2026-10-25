# 第 07 讲：Collectives 与并行训练入门

> 配套实践：[A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 04-20 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)。公式中的简化假设会单独说明。

## 1. Rank、进程组与通信语义

一个 rank 通常对应一个训练进程，一张 GPU 常由一个进程控制。初始化时需要所有 rank 使用相同 world size、后端与 rendezvous 地址。GPU 使用 NCCL，CPU 语义测试可用 Gloo。所有进程必须以一致顺序调用同一组 collective；某个 rank 先 all-reduce、另一个先 broadcast，可能无限等待。

| Collective | 输入/输出语义 | 常见用途 |
|---|---|---|
| broadcast | 一个 rank 的 tensor 复制到全部 rank | 初始化/参数同步 |
| all-reduce | 全部 tensor 按 op 归约，全部得到结果 | DDP 梯度求和 |
| reduce-scatter | 先全局归约，再各取结果的一段 | 分片梯度 |
| all-gather | 各段拼成完整 tensor，每 rank 都得到 | 参数临时聚合 |
| all-to-all | 每 rank 分别发给各 rank 并接收 | MoE dispatch |

all-reduce SUM 不自动除 world size，DDP 平均梯度需要除 p。浮点加法不满足严格结合律，不同通信拓扑的结果可能有小差异，测试使用合理容差并在相同精度下比较。

## 2. Ring all-reduce 的通信核算

对于 p 个 rank、大小 S byte 的 tensor，ring 分 reduce-scatter 与 all-gather，每阶段 p−1 步，每步传 S/p。每 rank 发送总量约 `2(p−1)S/p`，时间近似

$$t\approx2(p-1)\alpha+2\frac{p-1}{p}\frac S\beta.$$

α 是一次阶段的延迟，β 是有效带宽。小 tensor 更容易受延迟限制；大 tensor 更容易受带宽限制。实际库可选择树、分层与其他算法，公式是理解趋势的模型，不是实测替代。

同节点 GPU 互连与跨节点网络的带宽通常不同。通信 placement 的关键是把频繁同步放到快链路，把粗粒度同步放到较慢链路。单卡 CPU 多进程测试只能验证语义，不能测出 NCCL/NVLink 的扩展效率。

## 3. Data Parallelism 的梯度等价性

每 rank 处理不同数据，完整复制模型。全局 mean loss 为 `(1/p)Σ_r L_r`，所以对等大 local batch：全局梯度为 local mean gradients 的平均。若 local batch 大小不同，应按样本数加权，而不是等权平均。一个 rank 没有某个参数的梯度时，各 rank 通信序列可能不一致，动态模型需要额外处理。

朴素做法全部 backward 后通信；优化做法在某层梯度 ready 时启动异步 all-reduce，与较早层的 backward 重叠。按参数通信会有很多小 launches；bucket 把一组梯度聚合以提高带宽利用率，但太大的 bucket 延迟发起，减小 overlap。等待同步完成后才能 optimizer.step，不能对仍在写入的 gradient 更新参数。

本仓库 DDP 用 `register_post_accumulate_grad_hook`，异步发起后由 `finish_gradient_synchronization()` 等待并平均；通过官方两进程测试。它面向静态图教学，没有生产 DDP 的 unused-parameter、join、bucket/rebuild 等完整能力。

## 4. Tensor Parallelism 的两种切法

设 `Y=XW`。column-parallel 把 W 的输出列切为 `[W1,W2]`，得到 `[Y1,Y2]`；row-parallel 把 W 的输入行切开，与相应 X 子块相乘后相加。FFN 第一层 column、第二层 row，可把中间激活留在各 rank，末尾 all-reduce。attention 切 head 也类似，但 LayerNorm、残差和输出投影的分布布局必须匹配。

TP 每层都有通信，对低延迟高速网络要求高。它减少单卡参数和某些激活，不改变同一个 batch 的语义。head 数、FFN width 等需要可分或 padding；rank0 独立运行一个完整模型并不是 tensor parallel。

## 5. Pipeline Parallelism

把连续 layers 放到不同设备。若单个 batch 串行走一遍，设备轮流闲置；将 batch 切成 m 个 microbatches 后形成流水。理想平衡且忽略通信时，简单填充/排空调度的利用率约 `m/(m+p−1)`。m 越大 bubble 比例越小，但激活存储、调度与有效 batch 的约束会变化。

1F1B 等调度交错 forward/backward，降低峰值激活；interleaving 和 zero-bubble 更复杂。任何调度都要保证参数版本与梯度更新一致，不能让某些 microbatch 用更新后的参数、另一些用旧参数而不说明算法变化。

## 6. A2 验证路线与自测

先构造确定性 toy 模型与 rank0 reference：固定初始化、不同 local inputs，对拼接全局 batch 单步更新；比较各 rank 参数和 reference。随后检查梯度累积、FP16/BF16 和多个 optimizer steps。运行失败要先记录所有 rank 的错误，而非只看 rank0。

练习：4 rank 对 100 MB tensor 做 ring，忽略协议开销，每 rank 总发送约 150 MB；并解释网络双向带宽与单向吞吐的口径。练习：microbatch b=4、accum=8、DP=4，有效 batch=128；学习率 schedule 按 global tokens 记录比只按 local step 更稳妥。

面试追问从“DDP 平均梯度”继续到 uneven batches、通信 overlap、bucket 大小与 hang 排查，才说明理解落到了实现。
