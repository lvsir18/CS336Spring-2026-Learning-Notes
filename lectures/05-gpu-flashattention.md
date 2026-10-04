# 第 05 讲：GPU/TPU 与 FlashAttention 的硬件基础

> 2026 Spring · 官网日期 04-13 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)。公式中的简化假设会单独说明。

## 1. GPU 的执行与存储层级

CPU 优化少数复杂线程的延迟，GPU 优化大量并行线程的吞吐。GPU 有多个 SM；SM 内有算术单元、warp 调度、寄存器、shared memory 等。NVIDIA 常见 warp 为 32 个线程，SIMT 以组执行指令。grid 包含 blocks，block 内线程可用 shared memory 协作；不同 block 不宜假设普通 kernel 内有全局同步。

存储由近到远包括 registers、shared/L1、L2、device DRAM。寄存器快但每 SM 数量有限；大量寄存器会降低同时驻留的 warps，溢出到 local memory 后仍走设备内存。shared memory 提供显式片内数据重用，但也有容量、bank conflict 与同步成本。TPU 同样用大矩阵阵列、片内存储与高速互连提高吞吐，其编译与网络布局不同，不能把 GPU warp 模型逐项套过去。

## 2. 为什么矩阵乘适合 GPU

`C_ij=Σ_k A_ik B_kj` 的朴素实现对每个输出反复读取同一输入。将输出分成 `Mtile×Ntile`，沿 K 分块，加载 A/B tile 后复用：每次 tile 计算 FLOPs 约 `2 Mtile Ntile Ktile`，读取约 `(Mtile Ktile+Ktile Ntile)*bytes`。tile 越大重用越多，但寄存器/shared 占用也越多，因此不是越大越好。

Tensor Cores 以规定的矩阵小块与低精度格式执行高吞吐矩阵乘。形状对齐、stride、dtype 与可用指令决定能否充分利用。大模型采用 BF16/FP8 等还需要动态范围、scale、累加精度与误差控制。FP32 accumulation 不意味着输入保留 FP32 精度；量化转置也可能需要自己的 scale。

## 3. 五类常见优化

1. **减少 divergence：**同一 warp 的不同分支可能被分别执行，尽量让工作一致。
2. **减少字节：**低精度减少读写，同时要验证 loss/梯度的稳定性。
3. **融合：**连续 pointwise 算子在片内完成，减少临时 tensor 与 launch。
4. **重计算：**有时重新算少量数值比从 DRAM 读回来更便宜。
5. **Coalescing 与 tiling：**让相邻线程访问相邻地址，加载的数据反复使用。

“沿行访问”含义容易混淆：row-major 下同一行的连续列相邻；若 warp 每个线程访问不同的行但同列，stride 很大，通常不合并。矩阵尺寸出现周期性的性能变化，还可能由 tile padding、最后一波 blocks 没填满全部 SM（wave quantization）造成。

## 4. Attention 的内存瓶颈

普通实现写出 `S=QKᵀ/√d`、`P=softmax(S)`，再算 `PV`。S/P 都是 `[B,h,T,T]`；当 T 增大，平方矩阵的写回与重读非常昂贵。FlashAttention 的关键是融合、分块、在线 softmax，只写最终 O 与少量统计量。它计算精确的 dense softmax attention，浮点求和顺序差异会造成小误差；FLOPs 仍是平方量级，不能称为线性时间 attention。

## 5. 在线 softmax 的推导

对某个 Q block，处理新 K/V block 得分 S。维护每行最大值 m、指数和 ℓ、未归一化输出 u：

$$m'=\max(m,\max_j S_j),\quad \alpha=e^{m-m'},$$
$$\ell'=\alpha\ell+\sum_j e^{S_j-m'},\quad
u'=\alpha u+\sum_j e^{S_j-m'}v_j.$$

所有 blocks 处理完得 `O=u/ℓ`，`LSE=m+log ℓ`。旧累积必须乘 α，否则换了指数基准后两部分不在同一个尺度。初始 `m=-∞,ℓ=0,u=0`；全被 mask 的行需避免 `-∞-(-∞)` 的 NaN。causal attention 应让每个 query 至少可见自己，并对 padded 行正确屏蔽。

A2 前向保存 Q/K/V、O、LSE，反向用 `P=exp(S-LSE)` 重算概率，避免保存 T² tensor。设 `dP=dO Vᵀ`、`D_i=Σ_k O_ik dO_ik`，则

$$dS_{ij}=P_{ij}(dP_{ij}-D_i),\quad
dQ=dS K/\sqrt d,\ dK=dS^\top Q/\sqrt d,\ dV=P^\top dO.$$

D 可以用 O 和上游梯度直接得到，这使反向仍能分块而不物化全矩阵。

## 6. 从课程到你的 PRO 6000

已知服务器单卡 96GB。具体产品后缀、驱动、CUDA runtime、SM architecture 都要用 `server_check.py` 实测；软件必须包含对应架构可执行代码。`torch.cuda.is_available()` 只确认设备可见，实际 matmul 和 Triton 小 kernel 才能检验运行兼容。记录 CUDA runtime 而不是只记录系统 `nvcc`。

单卡可完成 attention correctness、forward/backward benchmark、A1/OLMo-1B 训练与 RL。网络并行内容先用 CPU 两进程验证通信语义，多 GPU 真实吞吐留待有设备再测。性能不能从 A100/H100 课件直接按显存比例换算。

## 7. 练习与面试答案

手工将 `[1000,1001]` 和 `[999,1002]` 两个 logits block 做在线归一化，并与合并后减 max 的 softmax 比较；不需要真的计算 `exp(1002)`。其次用长度 256/1024/4096 比较 naive 与 Flash 的峰值内存，预期 naive 的主中间项随 T² 增长，Flash 保留量随 T 增长。

面试问“为什么多算反而更快”，应回答计算/带宽比和 HBM IO，给出在线 softmax/recompute 的实例，最后说明在小尺寸或 kernel 实现较差时未必提速。
