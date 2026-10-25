# 第 08 讲：FSDP、ZeRO 与多维并行

> 配套实践：[A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 04-22 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)。公式中的简化假设会单独说明。

## 1. 为什么 DDP 不够

DDP 给每个 rank 复制参数、梯度和 optimizer state，增加 GPU 数只扩大 batch/吞吐，单卡静态显存不随 p 减少。对 FP32 Adam、N 参数，约 `16N` byte；6B 参数静态就约 96 GB，已没有激活空间。多卡训练首先要回答模型能否放下，其次才是扩展效率。

## 2. ZeRO 的分阶段状态分片

用 FP32 参数 4N、梯度 4N、Adam 两状态 8N 做教学核算：

| 方案 | 单 rank 静态状态 byte（忽略 padding） | 核心通信 |
|---|---:|---|
| DDP | 16N | 梯度 all-reduce |
| ZeRO-1 | 8N+8N/p | 更新后的参数同步 |
| ZeRO-2 | 4N+12N/p | 梯度 reduce-scatter + 参数同步 |
| ZeRO-3/FSDP | 16N/p | 逐模块 all-gather + reduce-scatter |

混合精度会改变各项 byte 系数，不改变分片逻辑。ZeRO-1 每 rank 只负责一部分参数的优化器状态与更新，其他参数由 owner 同步。ZeRO-2 连梯度也不持久复制。ZeRO-3 连参数也分片，需要在使用某层前临时聚合。

## 3. FSDP 的生命周期

每个被 wrap 的模块保持参数 shard。前向之前 all-gather 得完整 weight，计算后释放；反向之前再次 gather，算 input/weight gradients，然后将 full gradient reduce-scatter 成各自 shard，最后 optimizer 仅更新本地参数。长存的是 shard，完整 weight 与 full gradient 都应短命。

峰值显存仍有当前层的完整权重、激活、通信 buffer 与预取模块，因此不是纯粹 `16N/p`。wrap 一个整模型会一次 gather 全部参数，可能失去峰值收益；wrap 太细又增加通信延迟。通常按 block 或适合矩阵规模的模块分组，平衡资源与 overlap。

forward/backward prefetch 用额外临时内存换 overlap；limit-all-gathers 避免 CPU 过快发起造成过多 buffer。生产实现还需要 checkpoint state dict、mixed precision、resharding、shared weights 和异常清理。

本仓库教学 FSDP 对 A1 Linear/Embedding 分片，RMSNorm 复制，按需同步 gather；backward 的上一层 reduce-scatter 与下一层计算重叠，最多保留有限 full-gradient buffer。支持 FP32 master 与指定 compute dtype；不支持共享 weight，不实现前向 prefetch。官方测试验证数值与分片形状，不代表大集群吞吐已验证。

## 4. Activation Memory 需要另外解决

参数分片没有自动分片所有激活。普通 attention 的 `B h T²` 矩阵与各层中间状态依然可能使显存溢出。FlashAttention、activation recompute、sequence parallel、context parallel 分别减少不同部分。TP 内的 sequence parallel 可把 norm/dropout 等 pointwise 激活沿序列切分；context parallel 在更长序列中分割 Q/KV 并通信，算法必须保留跨段的全局 attention/在线 softmax。

一个常见误区是“FSDP 已经 shard everything，所以长上下文免费”。这里 everything 指主要训练状态，activation 与临时 tensor 还有各自生命周期。

## 5. 组合 DP、TP、PP、EP、CP

- DP 扩大可并行处理的数据，适合粗粒度梯度同步。
- TP 切矩阵/heads，频繁通信，通常放快互连域。
- PP 切层，适合跨节点，但受 bubble 和层间负载影响。
- EP 切专家，用 all-to-all dispatch，受负载尾部影响。
- CP 切长上下文，注意力需交换或流转 KV。

总设备数为各正交维度大小乘积，但 EP 与 DP/TP 的实际组合可能共享或重定义进程组，不能盲目把所有因子相乘。画出 rank 的坐标和每个 process group，是设计与排错的有效方法。

选择时先估参数/状态/激活能否 fit；再按拓扑放 TP/EP；选择 PP 深度与 microbatch 数；最后扩 DP。某模型的最佳配置受 batch、序列长度、硬件、专家大小影响，不能复用他人配置便声称最优。

## 6. 训练系统的可靠性

大规模作业会遇到网络抖动、设备错误、慢 rank 与数据 loader 异常。checkpoint 需要保存模型、optimizer、scheduler、RNG、数据消费位置和配置。保存 shard 时必须能重建与重新分片；只有 rank0 的局部 optimizer state 不能恢复整个 ZeRO 训练。

本仓库 ShardedOptimizer 是教学算法，默认 inherited state_dict 不能作为生产跨 world-size checkpoint 格式；指南要求分 rank 保存并使用相同 rank/world size恢复，扩展时应实现显式合并与重分片。避免把单步正确性等同于可恢复的大规模训练平台。

## 7. 练习与面试追问

8 rank、1B 参数：上述 FP32 静态 DDP 16 GB/rank，ZeRO-1 9 GB，ZeRO-2 5.5 GB，ZeRO-3 2 GB，均不含激活/临时。若临时最大层有 200M 参数，至少还要考虑它的完整 compute weight 与梯度 buffer。

问 FSDP 为什么比 DDP 多通信，应指出每层 forward/backward gather 的额外数据流；问为何仍值得，应指出原本放不下的模型、更大 batch 与更好的资源利用；问何时不值得，应指出小模型/弱链路下通信开销可能超过容量收益。
