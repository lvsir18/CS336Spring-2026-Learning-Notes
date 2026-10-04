# 第 17 讲：多模态：CLIP、SigLIP 与视觉语言模型

> 2026 Spring · 官网日期 05-27 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py)。公式中的简化假设会单独说明。

## 1. 本讲材料范围

官网日程包含alignment/multimodality，公开`lecture_17.py`主要围绕CLIP、SigLIP、LLaVA及其扩展、Qwen-VL系列和Chameleon。因此本篇以公开材料中的多模态主线整理，偏好与RL内容接前两讲。文中的工程推导是补充，不声称为每一页课堂原话。

多模态系统有两个问题：如何把不同模态对齐到可比较表示；如何让语言模型条件化于图像/视频并生成答案。检索embedding模型与生成VLM的目标不同，不能用同一套score直接比较。

## 2. CLIP 的对比目标

batch中有B组匹配图文 `(I_i,T_i)`，编码后L2归一化为u_i、v_i，logits为 `s_ij=u_iᵀv_j/τ`。图到文和文到图的交叉熵平均：

$$\mathcal L=\frac12\left[-\frac1B\sum_i\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}}-\frac1B\sum_i\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}\right].$$

对角线是正样本，batch内其他pair是负样本。负样本数量/质量、重复caption、false negatives影响训练；分布式all-gather增加候选但需处理梯度与全局label index。temperature决定logit尺度，极小值可能使训练不稳定。

zero-shot分类把类别描述成text prompts，比较图像与各text embedding。prompt wording影响结果；图文检索的image encoder能力不自动等于细粒度视觉推理。分类改名、多个模板ensemble与评估协议要记录。

## 3. SigLIP 的 pairwise Sigmoid

SigLIP将每pair看成二元匹配任务，用sign label z_ij∈{−1,+1}：

$$\ell_{ij}=\log(1+e^{-z_{ij}(u_i^\top v_j\,t+b)}).$$

不用每行softmax分母，因此批次组织与负样本扩展方式不同。负样本数量远多于正样本，bias与归一化需要匹配实现；它并不意味着可以完全不考虑batch或样本不平衡。

对比预训练学习高层语义但可能忽视计数、空间关系、小文字等细节，VLM训练通常还需更细视觉token与instruction数据。

## 4. LLaVA 的连接方式

图像经过vision encoder得 `[Npatch,d_v]`，projector映射到LM宽度d，作为visual tokens插入prompt；decoder预测文本。常见阶段先训练projector对齐，再进行多模态instruction tuning，哪些模块冻结由recipe决定。

视觉embedding不是普通token ID，而是在对应位置用连续embedding替换/插入；attention mask、position IDs、labels与padding都要与展开后的长度一致。只在文字response算loss时，image/prompt positions mask掉，但仍提供条件上下文。

高分辨率带来更多patch：patch大小p、图像H×W时，patch数约 `(H/p)(W/p)`。视觉token数越多，decoder prefill、KV cache与context预算都更大。tile/crop、merge、resampler减少成本，却可能丢失空间细节。

## 5. Dynamic Resolution 与 Video

Qwen-VL系列等处理动态分辨率、多图与视频。不同图像尺寸的token数不同，需要batch中的位置与边界标识；视频还包含frame/time信息，不能简单把所有frames当无序图片。空间与时间位置设计（如多轴RoPE）把不同维度的坐标进入模型，实际规则需跟随模型代码。

长视频的成本来自frame数×每frame视觉token、LM context与视觉编码，采样帧率影响运动/事件可见性。任务要求定位短暂事件时，低帧率可能漏掉关键帧；模型回答流畅不能弥补输入丢失。

## 6. Native Multimodal 与生成

Chameleon类方法用离散image tokenizer将图像映射为token，与文本一起做autoregressive建模。相比连续vision encoder+projector，更自然地统一输入/输出序列，但图像token长度、码本质量、模态平衡和稳定性是挑战。

“统一token”不代表所有模态共用同一种语义或同样的loss权重。图像重建、视觉理解、文本生成的要求不同；总loss降低可能由较容易的模态主导，应分模态报告。

## 7. 实践与岗位检查

用公开小型vision encoder和小LM构建toy projector：验证一张图、一段prompt、response的embedding shape、mask与loss位置。对比无图、错误图、正确图三组，确认模型是否使用视觉信息；再测OCR、counting、空间关系与hallucination。

单卡96GB可学习小/中型VLM微调，但需额外数据与模块，本课程五份基础作业不要求完成完整多模态训练。把本讲扩展作为后续作品方向，不把一个图片聊天demo当多模态训练复现。

面试问CLIP为何需要归一化，应解释防止仅靠范数扩大logits、使相似度更可控；问视觉token过多如何优化，应结合分辨率、resampling、KV/GQA与任务精度取舍。
