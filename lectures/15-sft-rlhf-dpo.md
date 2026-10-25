# 第 15 讲：Midtraining、SFT、RLHF 与 DPO

> 配套实践：[A5 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a5.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 05-18 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf)。公式中的简化假设会单独说明。

## 1. Pretraining 到 Assistant 的几种变化

预训练优化通用文本next-token prediction；midtraining提高某些领域、推理或长上下文数据权重；SFT模仿期望的指令响应；偏好优化学习多个可行响应中的选择。边界在真实recipe中有重叠，但训练目标、样本格式和评估应分别写清。增加聊天模板不等于完成指令调优，SFT数据质量也不只是语句通顺。

课程强调风格、长度和chat benchmark之间的影响。一个偏好更高的模型可能只是更详细，知识与推理并未增加。训练中同时保留任务accuracy、输出长度和偏好指标，避免将“更受judge喜欢”直接解释为“更正确”。

## 2. Response-only SFT 的张量对齐

给定prompt x、response y：

$$\mathcal L_{SFT}=-\frac{\sum_t m_t\log p_\theta(z_{t+1}|z_{\le t})}{\sum_t m_t},\quad z=[x;y].$$

mask在**label位置**：输入是tokens[:-1]，labels是tokens[1:]，response mask也要shift。若prompt有3 token，首个response token位于原序列index3，那么对应label位置是index2。prompt位置不计算loss不意味着不参与forward，它提供生成响应的上下文。

不同protocol可能让prompt也参与loss，或者用packed full-token SFT；A5 optional safety题有其固定packed模板，不能把另一个response-only方案用来声称官方兼容。packing连接样本降低padding，但EOS/边界处理、是否允许跨样本attention会改变训练。记录真实策略。

数据需要覆盖多轮、拒答、工具调用、代码、长短答案等任务分布。把模型不具备的尾知识硬写进少量SFT答案，可能诱发在相似问题上的幻觉；应测知识正确性与拒绝/不确定表达，不只loss。

## 3. 偏好数据与 Reward Model

pairwise样本为 `(x,y_w,y_l)`，Bradley-Terry模型：

$$P(y_w\succ y_l|x)=\sigma(r_\phi(x,y_w)-r_\phi(x,y_l)),\quad
\mathcal L_{RM}=-\log\sigma(r_w-r_l).$$

只通过差值训练，reward的绝对零点不识别；标注者偏好、任务、位置、长度、tie、噪声都会影响模型。RM不能看成客观真理，holdout应按prompt/来源分离，避免同问题变体泄漏。

AI feedback提高扩展性，但可能继承judge盲区。要独立评估训练外分布、错误事实、长短答案和安全过度拒绝，必要时人工抽查争议样本。

## 4. KL-Regularized RLHF 与 PPO

经典目标为

$$\max_\pi\ E_{x,y\sim\pi}[r(x,y)]-\beta E_x[KL(\pi(\cdot|x)||\pi_{ref}(\cdot|x))].$$

reference限制策略过度偏离。PPO使用rollout旧策略、value function/advantage与clipped ratio更新，通常还包括KL reward shaping、value loss、entropy和训练调度。PPO中的clip不是全局KL硬约束，也不能保证reward model无法被利用。

on-policy采样与反复inner updates之间有版本管理：旧策略概率必须在采样时保存/计算并detach；reference是另一种角色，不可把old policy和reference混为一谈。

## 5. DPO 的推导与实现

KL正则目标的最优分布满足

$$\pi^*(y|x)=\frac1{Z(x)}\pi_{ref}(y|x)e^{r(x,y)/\beta},$$
$$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x).$$

把reward差值代入偏好模型，Z在同prompt两答案间抵消：

$$\mathcal L_{DPO}=-\log\sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right).$$

`log π(y|x)`是response token log-prob之和，通常不是长度平均；reference无梯度且固定。用`-logsigmoid`保证数值稳定。chosen和rejected的prompt前缀完全一致时，full-sequence log-likelihood的prompt部分相消，但response mask更直观，模板/BOS差异会破坏抵消。

初始化policy=reference时delta=0，loss=log2≈0.693，但梯度并非0：它提高chosen相对rejected的概率。β影响相对于reference的约束与loss尺度，不能简单解释为学习率。离线偏好数据不覆盖所有policy输出，DPO仍可能有分布外问题和长度偏差。

## 6. Overoptimization 与模式变化

训练reward持续上升而真实任务或人工偏好下降，可能说明reward hacking/过拟合。跟踪独立验证指标、KL、entropy、输出长度、重复模式；早停依据holdout而非训练reward。后训练还可能损害模型原来的概率校准，不能直接把最高token概率当可靠信心。

在单卡96GB上，1B模型SFT/RL十分可行；8B完整Adam更新加reference与激活需要精细显存预算，参数dtype、optimizer states、checkpointing与microbatch都要显式配置。LoRA是另一个优化选择，不等同于官方全参数更新实验。

## 7. 实践与面试

A5核心SFT/DPO函数在`cs336_alignment/core.py`，官方测试检查模板、packing、概率与loss。实验报告保存数据格式、mask样例、reference版本和独立评价。练习手算chosen/rejected两序列log-probs，检查policy=reference时loss=log2，再故意去掉detach观察reference梯度是否错误出现。

面试应能从偏好模型推到DPO，解释为什么归一化Z消掉、old/reference区别、sequence sum与mean区别，以及为什么离线偏好优化不保证所有行为都更好。
