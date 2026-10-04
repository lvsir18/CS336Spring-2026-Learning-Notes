# 第 02 讲：PyTorch、张量与资源核算

> 2026 Spring · 官网日期 04-01 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)。公式中的简化假设会单独说明。

## 1. 张量是形状、存储、步长与设备的组合

设 batch `B`、长度 `T`、宽度 `d`、head 数 `h`、head dim `d_h=d/h`。embedding 输出 `[B,T,d]`；Q/K/V 投影后 `[B,T,h,d_h]`，换轴成 `[B,h,T,d_h]`；attention 分数为 `[B,h,T,T]`。把轴含义写出来，比只记 `view()` 更可靠。`einsum('bhtd,bhsd->bhts',q,k)` 清楚表达 Q 与 K 的特征收缩。

PyTorch tensor 可以共享底层 storage。`transpose` 一般修改 stride，不复制元素；`reshape` 在不能 view 时可能复制；`contiguous` 可能分配整块新存储。性能调试要同时看 shape、stride、dtype、device。一次 dtype cast、跨设备 copy 或意外 contiguous，都可能比一个算术操作贵。

`einops.rearrange(x,'b t (h d) -> b h t d',h=h)` 将拆 head 的约束写在代码里。不要通过形状碰巧相等判断语义正确：当 `T == h` 时错误换轴可能依然通过形状断言，测试应使用不同且非整齐的维度。

## 2. 参数与 FLOPs 核算

矩阵乘 `A[m,k] @ W[k,n]`，约 `2mkn` FLOPs（乘与加分别算一次，忽略小边界项）。线性层前向为 `2BT d_in d_out`；反向算输入梯度和参数梯度，各有一个相近规模 matmul，所以训练约为前向 3 倍。常用 dense LM 训练近似 `C≈6ND`：`N` 为参与 dense 计算的参数量，`D` 为训练 token 数。它忽略长上下文 attention、embedding 细节和重计算，MoE 不能直接代入总参数量。

一个无 bias、SwiGLU decoder block：attention 参数 `4d²`，FFN 参数 `3dd_ff`，两个 RMSNorm 参数 `2d`。若 `d_ff≈8d/3`，主参数合计约 `12d²`。独立 embedding 和 LM head 额外各 `Vd`，深度 `L` 得

$$N≈L(4d^2+3dd_{ff}+2d)+2Vd+d.$$

attention 的两个核心乘法 QKᵀ、PV 前向约 `4B T² d`，全部训练约三倍。长上下文时这项不能被 `6ND` 吸收掉。

## 3. 显存从哪里来

FP32 参数 4N byte，梯度 4N，两份 Adam 状态 8N，总计约 16N byte，尚未包含激活和 workspace。BF16 参数+梯度、FP32 master、FP32 两状态仍可能约 16N；如果参数本身一直 FP32，仅 autocast matmul，静态状态也仍约 16N。写“混合精度显存减半”必须说清减的是哪一部分。

举例：1B 参数 FP32 Adam 静态约 16 GB（十进制），96 GB 显存还要留给激活、logits、库 workspace 和缓存分配器。`[B,T,V]` logits 在 `B=8,T=2048,V=50000`、FP32 下约 3.28 GB，一旦保留多个版本会明显增加峰值。手算容量只是下界，实际跑 profiler 才能定 microbatch。

`memory_allocated` 是活跃 tensor；`memory_reserved` 是 allocator 向 CUDA 保留的内存；`nvidia-smi` 还包括上下文等。三者不同，不能把 reserved 的高值直接称为 tensor 泄漏。区分加载峰值、前向峰值、反向峰值和 optimizer step 峰值。

## 4. 算术强度与 Roofline

算术强度 `I=FLOPs/bytes_moved`，单位 FLOP/byte。理论吞吐上界

$$P\le\min(P_{peak},\ BW_{memory}\cdot I).$$

FP32 elementwise ReLU 每个元素读 4 byte 写 4 byte，只有约一次比较，I 很低。矩阵乘通过重复利用 tile 数据提高 I。大小相同的操作不一定相同速度：小 matmul 常受 kernel launch 与不足并行度限制，大 matmul 才可能接近 compute-bound。

MFU 常估算为 `6N × tokens/s / peak_training_FLOPs`。应注明是否包含 attention、GPU 数、dtype 和 dense/sparse 峰值口径。MFU 高不代表模型质量高，也不能把厂家稀疏理论峰值当 dense BF16 分母。

## 5. Autograd、梯度累积与重计算

autograd 记录构图并在 backward 沿图反向传播；`no_grad()` 用于评估；`detach()` 用于把旧策略、目标统计等视作常量。参数更新前要清梯度，否则 PyTorch 默认累加 `.grad`。

用 microbatch `b`、累积次数 `a`、数据并行规模 `p`，有效 batch 为 `bap`。每个等大 microbatch 的 mean loss 除以 `a` 后 backward，才等价于大 batch mean。最后一批不满、有效 response token 数不同、按 sequence 与按 token 求平均时，不能机械除以累积次数。

Activation checkpointing 在前向少存激活，在反向重算。收益不是参数状态变少，而是激活峰值下降；代价是多做前向计算。随机 dropout、状态性层与副作用必须保证重算语义一致。

## 6. 正确测量训练速度

CUDA 异步执行，Python wall time 不 synchronize 会低估时间。先 warmup（编译、allocator、autotune），再用 CUDA events 或前后同步测多次；记录 median、离散程度、峰值显存。测训练应包含 backward 和 optimizer，测纯算子应明确排除数据读入。`.item()` 会同步，放在每个细粒度算子后会破坏流水。

A1 `scripts/train_lm.py` 记录 step seconds、tokens/s、loss、peak allocated；A2 `benchmark_attention.py` 单测注意力。两种结果应分表，算子提速 3 倍不会必然让整步训练提速 3 倍（Amdahl 定律）。

## 7. 自测

手算 `d=768,L=12,V=10000,d_ff=2048` 的参数量与静态 Adam 状态；答案主块约 84.9M，embedding/head 约 15.36M，总约 100.3M，静态状态约 1.60 GB。再解释为什么 1.60 GB 不能据此承诺训练只占 1.60 GB。

面试中应能根据 `B,T,d,V` 快速定位参数、激活、logits 或 attention 哪一项先溢出，并提出降低 microbatch、重计算、FlashAttention、缩短上下文或分片的有针对性方案。
