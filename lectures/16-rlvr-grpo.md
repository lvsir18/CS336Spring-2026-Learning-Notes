# 第 16 讲：RLVR、策略梯度、GRPO 与推理训练

> 2026 Spring · 官网日期 05-20 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)。公式中的简化假设会单独说明。

## 1. 语言模型中的 RL

state是prompt与当前prefix，action是下一个token，policy是LM next-token分布，episode在EOS/stop/max tokens结束。对数学题、程序测试等，可以用verifiable reward判断最终结果；过程可很长而reward只有末尾一个，credit assignment与采样成本是核心难题。

目标 `J(θ)=E_{y~πθ}[r(y)]`，利用log derivative：

$$\nabla J=E[r(y)\nabla\log\pi_\theta(y|x)]
=E[r(y)\sum_t\nabla\log\pi_\theta(y_t|x,y_{<t})].$$

REINFORCE loss可用 `−r Σlogπ`。若baseline b(x)不依赖当前action，期望梯度不变并可减方差。batch内包含自身reward的group baseline与std normalization会引入更细的有限样本/加权效应，不能把所有GRPO归一化都称为无偏方差缩减。

## 2. GRPO 的分组优势

每prompt采G个response，reward `r_i`：

$$\bar r=\frac1G\sum_i r_i,\quad A_i=\frac{r_i-\bar r}{s_r+\epsilon}.$$

无需训练value model，降低额外状态。二元reward下，全部成功或全部失败的group在mean-baseline下advantage为0，没有相对梯度；需要跟踪zero-variance group fraction。增G可能提高发现成功答案概率，但也增加rollout成本，不能只看每步reward。

std normalization会改变不同问题权重。二元reward的std在中等成功率最大，极低/高成功率附近较小；因而某些稀有成功/失败梯度会被放大。`G=1`时样本std未定义，需明确处理；本实现给零std，mean-baseline产生零优势。

## 3. On-policy 与 Off-policy Token Ratios

on-policy采样后立即更新，可用 `−A logπ` surrogate，在采样点有正确policy gradient。反复利用同rollout时，重要性比

$$ρ_{it}=e^{\log\pi_\theta(y_{it}|prefix)-\log\pi_{old}(y_{it}|prefix)}.$$

PPO/GRPO clip surrogate：

$$\ell_{it}=-\min(ρ_{it}A_i,\ \mathrm{clip}(ρ_{it},1-ε,1+ε)A_i).$$

A为负时min方向仍不变；手工把所有超范围ratio loss直接设0会错误。old log-probs在更新前冻结；如果每次重算old=current，ratio恒1但梯度路径不一定正确，算法已不是off-policy目标。

课程/作业变体未必都含reference KL，不能在无KL要求时悄悄增加，也不能因为GRPO常见论文含KL就假定所有实现都含它。

## 4. 长度与难度归一化的变化

sequence-normalized loss是每条response token loss平均，再对sequence平均；长答案每token梯度被1/length缩小。Dr.GRPO去掉group std与response长度归一化，用固定常数做整体尺度，减少这些偏差。RFT只保留成功轨迹的正梯度；MaxRL类方法用group mean做另一种reward normalization，具体与官方A5参数匹配。

对constant normalization，常数是完整训练batch的N×max_response_length或题目指定值；microbatch的sum都除同一个总常数。再额外除accum次数会多缩小一次梯度。sequence mean则按microbatch样本数/全batch样本数加权，尤其最后microbatch不等大时。

GSPO把token log-ratios按有效response长度平均后指数化，得到sequence-levelratio，再clip；它不是把token ratios简单算术平均。mask必须只覆盖有效response，padding加入会改变ratio。

## 5. EOS、stop 与 reward 的一致性

rollout prompt可能已以`<think>`结尾，decode的response不含这段prompt。grader如何寻找`</think>`与`<answer>`必须和官方prompt一致。对`</answer>`停止的response不应添加多余文本后再评分；到max length的截断需单独记录。

本仓库采用HF rollout便于单卡学习，和官方vLLM backend不同。默认temperature=1、top-p=1、max_new_tokens=512；这让计算的model probabilities与采样policy保持一致。改变temperature/top-p后要重新定义重要性概率，不能照用原始logπ宣称完全on-policy。

## 6. 系统与评价闭环

循环为：冻结old版本→采样G条→验证reward→group advantage→保存old log-probs→inner updates→独立validation→checkpoint。训练model的dropout等随机行为要与likelihood定义一致，通常使用eval模式但保持autograd；`eval()`不会关闭梯度。

rollout常是大头。batching、KV cache、weight同步、训练/推理分离、异步rollouts都有性能收益，但引入policy staleness与版本管理。单卡HF实现先保证正确，服务器上记录rollout time/update time、GPU峰值、有效tokens/s，而不是只给optimizersteps/s。

独立GSM8K test只能做评估，不用于选prompt样例或训练。报告accuracy、format/answer reward、response长度、entropy、KL/ratio、zero-variance比例、clip fraction和样例。更长CoT不等于更正确，平均reward提升也可能只是格式改善。

## 7. 案例与自测

讲义分析DeepSeek R1/R1-Zero、Kimi K1.5、Qwen等公开工作，冷启动SFT、RLVR、拒绝采样、蒸馏与通用偏好阶段组合各异。复现时区分论文描述、开源代码和自己的推断，不能从最终模型表现反推精确训练recipe。

练习：reward `[1,0,0,1]`、G=4，均值0.5；使用sample std约0.577，优势约 `[0.866,-0.866,-0.866,0.866]`。再用population std会得到±1，解释为何测试对std口径敏感。练习：比较分成1、2、3个microbatch后的参数更新，应该在浮点容差内一致。
