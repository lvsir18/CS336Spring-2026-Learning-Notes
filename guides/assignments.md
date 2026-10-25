# 从课程笔记跳到作业实验

[← 笔记首页](../README.md) · [作业首页](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments) · [复现与恢复](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/operations.md)

笔记解释概念，作业仓库把概念落实到张量、损失、训练和测量。每份作业教程都包含结果、执行步骤、原理与关键代码，也有返回本笔记的链接。先阅读对应笔记，再跟踪作业源码并复现一个对照，比只看最终分数更有效。

| 课程笔记 | 实现/实验入口 | 对应范围 |
|---|---|---|
| [01 · 第 01 讲：课程总览与 Tokenization](../lectures/01-overview-tokenization.md) | [A1 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a1.md) | 直接对应 |
| [02 · 第 02 讲：PyTorch、张量与资源核算](../lectures/02-pytorch-resource-accounting.md) | [A1 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a1.md) · [A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) | 直接对应 |
| [03 · 第 03 讲：Transformer 架构与超参数](../lectures/03-architecture-hyperparameters.md) | [A1 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a1.md) | 直接对应 |
| [04 · 第 04 讲：注意力替代架构与 Mixture of Experts](../lectures/04-attention-moe.md) | [A1 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a1.md) · [A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) | 仅普通attention；MoE未实现 |
| [05 · 第 05 讲：GPU/TPU 与 FlashAttention 的硬件基础](../lectures/05-gpu-flashattention.md) | [A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) | 直接对应 |
| [06 · 第 06 讲：基准测试、性能分析与 Triton Kernels](../lectures/06-kernels-triton.md) | [A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) | 直接对应 |
| [07 · 第 07 讲：Collectives 与并行训练入门](../lectures/07-collectives-parallelism.md) | [A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) | 直接对应 |
| [08 · 第 08 讲：FSDP、ZeRO 与多维并行](../lectures/08-fsdp-large-scale.md) | [A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) | 直接对应 |
| [09 · 第 09 讲：Scaling Laws 的理论与计算最优配置](../lectures/09-scaling-laws.md) | [A3 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a3.md) | 直接对应 |
| [10 · 第 10 讲：推理、KV Cache 与服务系统](../lectures/10-inference-serving.md) | [A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) · [A5 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a5.md) | 仅基础生成/性能关联，未实现完整serving |
| [11 · 第 11 讲：Scaling Laws 的实践：初始化、学习率与实验设计](../lectures/11-scaling-practice.md) | [A3 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a3.md) | 直接对应 |
| [12 · 第 12 讲：评价语言模型：指标、有效性与可信度](../lectures/12-evaluation.md) | [作业结果与完成范围](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/results/README.md) | 使用验证选择和独立test协议 |
| [13 · 第 13 讲：预训练数据来源与公开数据集](../lectures/13-data-sources.md) | [A4 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a4.md) | 直接对应 |
| [14 · 第 14 讲：过滤、去重、混合与合成数据](../lectures/14-data-filtering-dedup.md) | [A4 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a4.md) | 直接对应 |
| [15 · 第 15 讲：Midtraining、SFT、RLHF 与 DPO](../lectures/15-sft-rlhf-dpo.md) | [A5 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a5.md) | 直接对应 |
| [16 · 第 16 讲：RLVR、策略梯度、GRPO 与推理训练](../lectures/16-rlvr-grpo.md) | [A5 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a5.md) | 直接对应 |
| [17 · 第 17 讲：多模态：CLIP、SigLIP 与视觉语言模型](../lectures/17-multimodality.md) | [作业结果与完成范围](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/results/README.md) | 五份作业未单独复现多模态 |
| [18 · 第 18 讲：Daniel Selsam 嘉宾课：资料状态与相关预习](../lectures/18-guest-selsam-material-gap.md) | [作业结果与完成范围](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/results/README.md) | 保持公开材料缺口说明，无对应实测 |
| [19 · 第 19 讲：Dan Fu 嘉宾课：推理系统与全栈协同的延伸](../lectures/19-guest-dan-fu.md) | [A2 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a2.md) · [A5 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a5.md) | 仅相关系统练习，非嘉宾课完整复现 |

## 如何使用这些结果

环境为Linux共享单卡RTX PRO 6000 Blackwell 96GB，42GiB allocator限制。A1/A4是各自的预训练实验，A5从公开OLMo开始；没有把五项作业写成同一权重端到端训练。A2的分布式实现仅有双CPU rank/Gloo验证，真实多GPU/NCCL吞吐与扩展未完成。A3只有本地9点估算IsoFLOPs，不能将边界拟合外推到任意大模型。A4是缩小预算的数据对照，A5是固定单次生成协议与单训练seed。

[结果证据](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/results/README.md)保存配置、指标、验证选择、测试摘要和数据来源；[A5教程](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a5.md)解释KL、DPO/GRPO差异和不确定性。权重、语料、本地测试和重复调试文档没有上传。后续实验应新增配置与结果，保留已有基线的协议说明。
