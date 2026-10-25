# 第 03 讲：Transformer 架构与超参数

> 配套实践：[A1 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a1.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 04-06 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)。公式中的简化假设会单独说明。

## 1. 一个现代 Decoder Block 的完整计算

令输入 `X∈R^{B×T×d}`，课程实现使用 prenorm：

$$U=X+\mathrm{Attention}(\mathrm{RMSNorm}(X)),\quad
Y=U+\mathrm{SwiGLU}(\mathrm{RMSNorm}(U)).$$

模型末尾再做 final norm 与 vocab 投影。残差流提供接近恒等的梯度通道：对 `x+f(x)`，Jacobian 是 `I+J_f`。postnorm 则把 norm 放到残差相加后，梯度每层都穿过归一化 Jacobian；深层训练的稳定性与初始化/warmup 会更敏感。prenorm 是常见经验选择，并不意味着所有 postnorm 变体都不可训练。讲义还讨论非残差 postnorm 与 double norm，阅读时应画出 norm 具体位置。

## 2. LayerNorm、RMSNorm 与 bias

LayerNorm 用 `(x-mean)/sqrt(var+eps)`；RMSNorm 为

$$\mathrm{RMSNorm}(x)_i=g_i\frac{x_i}{\sqrt{\frac1d\sum_jx_j^2+\epsilon}}.$$

RMSNorm 不减均值，少做一组 reduction；实践常用 FP32 算平方、均值和 rsqrt，再转回 compute dtype。参数只有每特征一个 gain。eps 必须加在平方根内部；在外部加会改变小输入时行为。bias 不是数学上必须删除，现代一些模型去掉线性层 bias 是简化设计与性能经验，应以实际 config 为准。

## 3. SwiGLU 为什么有三个矩阵

$$\mathrm{SwiGLU}(x)=W_2\big[\mathrm{SiLU}(W_1x)\odot(W_3x)\big],\quad
\mathrm{SiLU}(z)=z\sigma(z).$$

`W1,W3: d→d_ff`，`W2: d_ff→d`，主参数 `3dd_ff`。普通两层 FFN 的参数 `2d(4d)=8d²`；令 gated FFN 保持相近参数，可取 `d_ff≈8d/3`，再向硬件友好的倍数对齐。因此“FFN 默认 4d”不能直接移植到 SwiGLU。

门控让一个投影决定内容，另一个决定通过程度，实际优劣来自受控实验。激活更复杂带来的 FLOPs 较少，却可能增加额外内存读写与 kernel launches；融合后效果又不同。架构比较要同时看 loss 和 wall-clock。

## 4. RoPE 的相对位置信息从哪里来

对相邻二维特征 `(x_{2i},x_{2i+1})`，位置 p 的旋转为

$$R_i(p)=\begin{bmatrix}\cos(p\omega_i)&-\sin(p\omega_i)\\
\sin(p\omega_i)&\cos(p\omega_i)\end{bmatrix},\quad
\omega_i=\theta^{-2i/d_h}.$$

Q、K 分别旋转后，点积 `qᵀR(p)ᵀR(s)k=qᵀR(s-p)k`，所以位置通过相对距离进入注意力分数。V 通常不旋转。A1 用相邻 pair；某些实现采用前后半维配对，两者有对应排列，但直接混用 checkpoint 会错误。

RoPE 保留向量范数，却不自动保证长上下文外推。训练长度、频率尺度、位置分布、长序列数据与模型容量都影响结果。调 theta、位置插值等要在长距离检索和语言建模上实际评估，不可仅看代码能接受更大 position id。

## 5. MHA、MQA、GQA 与滑窗

MHA 每个 Q head 对应独立 KV head；MQA 所有 Q heads 共享一个 KV；GQA 介于两者。其主要收益是减少生成时 KV cache 与读带宽：

$$M_{KV}=2\cdot B\cdot L\cdot T\cdot h_{kv}\cdot d_h\cdot bytes.$$

例如 `L=32,T=8192,B=1,h_kv=8,d_h=128`、BF16，KV 约 1 GiB；若 KV heads 为 32，则约 4 GiB。Q heads 不必减少，因此不能把 KV 参数减少比例直接当作全模型计算减少比例。

Sliding-window attention 将每 token 的可见历史限制到窗口 w，复杂度降为约 `O(Twd)`。局部层与全局层交错可兼顾局部效率与信息交换，但精确长距依赖仍要评测。不能只把 causal mask 换成窗口 mask 就声称长上下文能力不损失。

## 6. 超参数与稳定训练

宽度与深度共同影响参数量，但深度增加串行 critical path，宽度增加大矩阵效率与每层 KV。head dim 一般保持在适合位置编码和硬件的范围，head 数跟宽度一起调整；数值没有跨模型通用最优常数。词表大小影响语种覆盖、序列长度和 LM head 成本；预训练 dropout 与 weight decay 必须按真实 recipe，不照搬小数据分类经验。

softmax 前减 max 防指数溢出，只解决数值计算，不控制 logit 长期漂移。讲义讨论 z-loss（惩罚 logsumexp 的量级）、QK norm（限制 Q/K 幅度）、logit soft-capping `c*tanh(z/c)`。它们改变优化与模型函数；应该作为单变量消融，并记录梯度范数、attention entropy 和 loss spikes。

## 7. A1 实验与面试检查

A1 实现路径：`model.py` 的 RMSNorm、SwiGLU、RotaryPositionalEmbedding、TransformerBlock。做 prenorm/postnorm/no-norm、RoPE/no-RoPE、SwiGLU/SiLU 的对照，固定 tokenizer、token budget、随机种子与评估窗口；每次只改一个因素。观察训练是否稳定、相同 compute 的验证 loss，而非仅比较同一步数。

追问：RMSNorm 是否消除整体幅度？正的统一缩放在 eps 很小时近似不变，但可学习 gain、残差与有限 eps 仍重要。RoPE 是否加入参数？固定 sin/cos 的基本形式无可训练参数。为什么 GQA 对 decode 更关键？decode 每步读长历史 KV，通常受带宽约束。
