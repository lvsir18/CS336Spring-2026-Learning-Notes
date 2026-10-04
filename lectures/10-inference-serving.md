# 第 10 讲：推理、KV Cache 与服务系统

> 2026 Spring · 官网日期 04-29 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)。公式中的简化假设会单独说明。

## 1. Prefill 与 Decode 的不同瓶颈

prefill 同时处理输入 T 个 token，大矩阵有较高重用，通常较 compute-bound；decode 每个 request 每次只有一个新 token，需要读模型权重与历史 KV，常受 memory bandwidth 限制。attention 的 causal 结构允许缓存过去每层 K/V：新 token 的 Q 只需与过去 K 比较，再加权 V。

没有 KV cache 时，每生成一个 token 重新计算整个历史，浪费过去层计算。A1 的生成脚本刻意保持完整前向的教学实现，可验证模型与采样；生产 serving 需要额外实现 cache、position、动态 batch 和调度，不能拿该脚本速度评价成熟推理引擎。

## 2. Decode 的算术强度

单次 `X[b,d] @ W[d,d]` 前向约 `2bd²` FLOPs。若权重约 `bytes*d²`，且 b 小、输入输出忽略，强度约 `2b/bytes`。BF16 时约 b FLOP/byte；增加 batch 能重用同一权重，提升吞吐，但排队与每请求 latency 可能上升。

KV 大小为 `2BLT h_kv d_h bytes`。它随 request 长度与并发增长，而权重大小固定。GQA/MQA 减 KV heads，量化 KV 减 bytes；滑窗减少保留历史，改变模型可见上下文。量化必须分别评估权重、激活、KV，不能笼统说“INT4模型显存为原来的四分之一”。

## 3. 服务评价指标

TTFT（time to first token）包含排队、调度和 prefill；ITL/TPOT 衡量后续 token 间隔；端到端 latency 随输出长度增长；throughput 用总 tokens/s 或 requests/s，还要固定 workload。goodput 是满足 SLO 的有效吞吐。

报告平均值会掩盖 p95/p99 尾延迟。长 prompt 与短 prompt 混在一起时，prefill 大计算可能打断 decode；chunked prefill 将长输入切块，平衡吞吐与 decode 延迟。在线系统的最佳 batch 由到达分布和 SLO 决定，不是离线满负载测试的最大吞吐 batch。

## 4. Continuous Batching 与 PagedAttention

静态 batch 等所有请求完成后再开始下一批，短请求结束后的槽位空闲。continuous batching 按解码 iteration 重新组成活跃请求，完成即释放，新请求即进入。难点是 KV 存储、position 与不同长度的管理。

PagedAttention 把逻辑 KV 序列映射到物理固定大小 blocks，减少为最大长度预留导致的碎片，支持请求共享前缀与 copy-on-write。它不是让 attention 数学变稀疏；需要 block table 和相应 kernel 访问非连续存储。block 太小元数据/访问开销多，太大尾块浪费多。

前缀缓存命中还要求 tokenizer、prompt template、模型版本、adapter 与 position 一致。把不同用户敏感上下文共享到未经隔离的缓存会导致错误语义，设计时要明确缓存 key 与权限边界。

## 5. Speculative Sampling 的正确性

draft 模型快速提出 token x，target 概率 p(x)，draft 概率 q(x)，接受概率 `min(1,p(x)/q(x))`；拒绝时从归一化 `max(p−q,0)` 分布补采样。这个机制保留 target 的采样分布，不能只用“target top1同意就接受”替代随机采样算法。greedy decoding 有对应确定性验证方法，目标不同。

draft 便宜、接受率高、target 能并行验证时可获益。长 draft 被频繁拒绝、小 batch 或 target 已充分利用算力时，收益可能很低。需要报告 accepted tokens per verification、draft cost、target cost 与最终 latency。

## 6. 量化与剪枝

权重量化常按 tensor/channel/group 学 scale，例如 `w≈s(q−z)`，小 group 能减误差但增加元数据。outlier 使 uniform quantization 困难，activation 校准数据影响精度。低 bit 存储必须有高效反量化/低 bit matmul kernel，否则仅减少容量未必减少 latency。

结构化剪枝改变矩阵维度，硬件容易利用；非结构化稀疏若没有支持与达到合适模式，dense kernel 并不会跳过零。服务 benchmark 应使用真实 kernel，而不是只统计非零参数。

## 7. 单卡实践与面试

在 PRO 6000 上用同一模型扫 batch、prompt length、output length，分别记录 prefill time、decode time、peak KV。先实现正确 cache，再与无 cache 输出逐 token 比较；注意 RoPE position 不能每次从 0 重置。之后比较 HF 与专门 serving 引擎，记录版本与是否使用量化。

面试问“GPU 利用率低但 latency 很高”，应考虑内存带宽、Python调度、同步、队列与小 batch；问“如何同时提吞吐和保 latency”，应结合continuous batching、chunked prefill、cache与SLO约束给出测量方案。
