# 第 06 讲：基准测试、性能分析与 Triton Kernels

> 配套实践：[A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 04-15 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)。公式中的简化假设会单独说明。

## 1. 先回答“慢在哪”，再写 kernel

课程通过 GELU、softmax、row sum、matmul+ReLU 展示从 PyTorch 到融合 kernel 的过程。研究对象要固定：同样的 shape、dtype、device、contiguity、数学精度和 causal 语义。PyTorch eager、内置算子、torch.compile、Triton 是四种不同路径，不能拿一条路径的首次编译时间和另一条路径的 steady state 比较。

一个可信 benchmark 包含输入生成、warmup、同步、重复计时、误差验证和内存统计。报告 median/p25/p75 或 mean/std，注明 forward 或 forward+backward。GPU 空闲情况、功耗状态和 CPU launch 开销都会影响小 kernel。profiling 看时间线和调用次数；benchmark 用低扰动方式给出总时间。profiler 本身会增加开销，所以最终速度不能只读 profiler 截图。

## 2. Triton 的执行模型

Triton program 通常处理一个 tile，而不是让 Python 写一个标量线程。`pid=tl.program_id(0)` 决定 tile，`offset=pid*BLOCK+tl.arange(0,BLOCK)` 产生一组地址；`tl.load(...,mask=offset<n,other=0)` 保护尾部；对该向量计算后 `tl.store`。`BLOCK` 等 constexpr 影响编译展开、寄存器和线程布局。

`tl.arange`、broadcast、reduction 表达逻辑 tile，编译器选择实际 GPU 执行。不同 Triton 版本的布局能力与编译效果不同，必须记录版本。tensor shape 的逻辑长度和分配的 power-of-two block 不同，mask 是正确性的一部分，不能只测整齐尺寸。

教学向量例：读取 x，计算 `x*sigmoid(x)`，写回。若 eager 产生 sigmoid 临时 tensor、乘法临时 tensor，融合减少读写与 launches。复杂数学不一定更快：若每个元素调用昂贵 transcendental、寄存器压力很高，结果需要实测。

## 3. 一行 softmax 的稳定实现

$$p_j=e^{x_j-m}/\sum_k e^{x_k-m},\quad m=\max_kx_k.$$

一个 program 读取一行，mask 后 max/sum，再写一行，避免多个 kernel 与中间 tensor。padding 的 load `other=-∞`，保证不会进入最大值或指数和。若填 0，在所有真实值为负数的行中会污染 max 与分母。

一行能否放入片内资源决定实现策略：超宽行会用很多 registers，降低 occupancy 或 spill。多 program 分段需要在线统计或二阶段 reduction；不能假设一个 row kernel 对任何宽度都最佳。softmax backward 是 `dx=p*(dp-Σp*dp)`，不需要完整 Jacobian。

## 4. Matmul tiling 与融合

program 对应输出 C 的一个 `[BM,BN]` tile；沿 K 用 BK 循环加载 A `[BM,BK]` 与 B `[BK,BN]`，`tl.dot` 累积 FP32。末尾可以直接 ReLU，再 cast 并 store，省掉 C 的额外 DRAM 往返。尾部维度要分别 mask M、N、K，不能把一个 mask 用在三个地址上。

选择 BM/BN/BK、num_warps、num_stages 是资源权衡：大 tile 重用好，寄存器/shared 多；多 stages 能重叠 load 与 compute，但增加资源；小矩阵需要足够 programs，否则很多 SM 闲置。autotune 应在合法配置集合内做，不把编译或调参时间计为最终稳态速度，也要考虑线上输入分布。

FP32 matmul 还要区分 IEEE、TF32 等内部精度模式；测试时的容差与模式应一同记录。以 float64 CPU 参考、float32 GPU 基线和 BF16 GPU kernel 做三方检查能更清楚地区分算法错误与浮点误差。

## 5. A2 FlashAttention 的工程步骤

先实现 PyTorch 分块 forward，保存 O/LSE；用小不规则尺寸比较输出、dQ/dK/dV；再把 tile 算法迁到 Triton。backward 可拆为 D 预处理、按 Q tile 计算 dQ、按 KV tile 计算 dK/dV，避免跨 program 原子加的冲突。causal mask 的 query/key 全局索引必须从各 tile 起点计算。

本仓库 `flash_triton.py` 是可读实现而非厂商最佳 kernel：采用固定 tile、没有持久化 autotuning，也未在你的服务器实测。PyTorch correctness 通过不等于 Triton correctness 通过。服务器必须先运行所有 GPU 测试，再测 naive、分块 Python、Triton、PyTorch SDPA。分块 Python 可能因大量 launches 比 naive 更慢，但用于验证算法与低内存行为。

## 6. 分析整步训练

Amdahl 定律：原运行占比 f 的模块提升 s 倍，总提速 `1/((1-f)+f/s)`。attention 占 30%，提速 3 倍，整体仅约 1.25 倍。attention 优化还可能通过更大 batch 间接改善吞吐，应单独测相同 batch 与最大可用 batch 两组。

Nsight Systems 看 CPU/GPU 间隙、同步、数据复制和通信；Nsight Compute 看单 kernel 内存吞吐、occupancy 与指令。它们是诊断工具，最终应写出假设、测量、修改、结果和回归检查。

## 7. 实践验收

用长度 127/256/1001、head dim 32/64/128、causal/non-causal、FP32/BF16 验证；再扫描 T 和 batch。保存环境 JSON、原始每次时间、误差最大值、峰值内存。面试中解释优化选择时，应能指出实际瓶颈证据，不能只说“融合更快”或“GPU 利用率高”。
