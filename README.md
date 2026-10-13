# Stanford CS336 Spring 2026 · 系统学习笔记

为 [lvsir18](https://github.com/lvsir18) 整理的中文自学材料，面向大模型研发、训练系统和推理工程。以[2026课程官网](https://cs336.stanford.edu/)、[官方讲义/代码](https://github.com/stanford-cs336/lectures)及五份作业为主要依据，社区笔记只辅助核对覆盖范围，正文采用独立表达、推导与工程补充。

**19次课程安排各一份Markdown，正文约5.3万字符。**每篇包括核心概念、公式/张量形状、实现或实验关联、容易出错之处及自测/面试追问。第18次嘉宾课没有找到公开正式材料，独立文件清楚标记资料缺口；第19次按官网编号，公开录像的编号18与此不同。不能把这两篇补充整理当逐字课堂记录。

配套代码仓库位于同一工作区的 `cs336-2026-assignments`，所有五份作业的实现、运行步骤和测试结果在其README。笔记与代码各自是独立Git仓库，使用main分支；本笔记的GitHub仓库为 [CS336Spring-2026-Learning-Notes](https://github.com/lvsir18/CS336Spring-2026-Learning-Notes)，作业代码另行发布。

## 阅读导航

| 编号 | 官网日期 | 学习笔记 |
|---|---|---|
| 01 | 03-30 | [课程总览与 Tokenization](lectures/01-overview-tokenization.md) |
| 02 | 04-01 | [PyTorch、张量与资源核算](lectures/02-pytorch-resource-accounting.md) |
| 03 | 04-06 | [Transformer 架构与超参数](lectures/03-architecture-hyperparameters.md) |
| 04 | 04-08 | [注意力替代架构与 Mixture of Experts](lectures/04-attention-moe.md) |
| 05 | 04-13 | [GPU/TPU 与 FlashAttention 的硬件基础](lectures/05-gpu-flashattention.md) |
| 06 | 04-15 | [基准测试、性能分析与 Triton Kernels](lectures/06-kernels-triton.md) |
| 07 | 04-20 | [Collectives 与并行训练入门](lectures/07-collectives-parallelism.md) |
| 08 | 04-22 | [FSDP、ZeRO 与多维并行](lectures/08-fsdp-large-scale.md) |
| 09 | 04-27 | [Scaling Laws 的理论与计算最优配置](lectures/09-scaling-laws.md) |
| 10 | 04-29 | [推理、KV Cache 与服务系统](lectures/10-inference-serving.md) |
| 11 | 05-04 | [Scaling Laws 的实践：初始化、学习率与实验设计](lectures/11-scaling-practice.md) |
| 12 | 05-06 | [评价语言模型：指标、有效性与可信度](lectures/12-evaluation.md) |
| 13 | 05-11 | [预训练数据来源与公开数据集](lectures/13-data-sources.md) |
| 14 | 05-13 | [过滤、去重、混合与合成数据](lectures/14-data-filtering-dedup.md) |
| 15 | 05-18 | [Midtraining、SFT、RLHF 与 DPO](lectures/15-sft-rlhf-dpo.md) |
| 16 | 05-20 | [RLVR、策略梯度、GRPO 与推理训练](lectures/16-rlvr-grpo.md) |
| 17 | 05-27 | [多模态：CLIP、SigLIP 与视觉语言模型](lectures/17-multimodality.md) |
| 18 | 06-01 | [Daniel Selsam 嘉宾课：资料状态与相关预习](lectures/18-guest-selsam-material-gap.md) |
| 19 | 06-03 | [Dan Fu 嘉宾课：推理系统与全栈协同的延伸](lectures/19-guest-dan-fu.md) |

## 如何形成自己的知识体系

先读[学习路线与练习安排](guides/study-roadmap.md)，再边看原始材料边读对应笔记，闭卷写公式、手算小样本、运行作业测试，最后用实验检验理解。[数学与形状速查](guides/math-and-shapes.md)用于复习，[岗位能力与作品集](guides/career-and-interviews.md)连接实际研发工作。

课程资源、引用与版本差异见[资料来源说明](guides/sources-and-course.md)。笔记不是官方逐字翻译，未复制社区笔记全文。技术例子中的量级需结合具体模型/硬件重新计算，课程里某个模型版本的recipe不作为永久最优推荐。

## Git 与 GitHub

远端仓库：[lvsir18/CS336Spring-2026-Learning-Notes](https://github.com/lvsir18/CS336Spring-2026-Learning-Notes)。按[发布说明](guides/github.md)克隆仓库或提交后续笔记修改。大数据、权重、缓存及密钥排除在提交之外；课程官方材料保留链接，notes以原创整理为主，发布license由维护者后续选择。
