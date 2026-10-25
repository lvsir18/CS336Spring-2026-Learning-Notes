# 第 12 讲：评价语言模型：指标、有效性与可信度

> 配套实践：[作业结果与完成范围](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/results/README.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 05-06 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py)。公式中的简化假设会单独说明。

## 1. “模型好”必须先定义使用场景

预训练loss衡量给定分布的预测；考试benchmarks衡量任务解答；聊天偏好衡量用户/评审偏好；agentic benchmark衡量在环境中达成目标；安全评价衡量特定失败与风险。指标之间有相关性，也可能冲突。课程把realism（是否像真实使用）和validity（是否测到声称的能力）放在核心位置。

先写任务：谁使用、输入分布、输出约束、允许工具、成本/延迟，再定指标。例如编程助手关心可执行测试通过率、修改范围、工具失败恢复与延迟；仅测单轮算法题并不能完整代理这一场景。

## 2. Cross-entropy 与 Perplexity

$$H=-\frac1T\sum_t\log p(x_t|x_{<t}),\quad PPL=e^H.$$

自然对数为nats/token；若用log2则是bits/token，perplexity为2^H。不同tokenizer、语料、上下文长度、BOS/EOS处理、packing和滑窗会影响结果。报告时写出共同评估协议，否则PPL数字不可直接比较。

长文档评估若按独立chunks重置上下文，边界token更难预测；滑窗让token得到更长历史，但不能重复计入同一token loss。训练和验证必须分离，去重/污染扫描适用于相似内容而不只文件名。

## 3. 考试与生成任务的协议

multiple-choice可比较候选答案likelihood或让模型生成字母。前者受长度/模板/归一化影响，后者受输出parser影响。few-shot样例顺序、是否有CoT、answer格式、max tokens、温度与seed全部需要固定。MMLU与数学题不能只写“用了默认设置”。

parser是测量装置：从长输出抓到第一个A-D可能抓错推理中的字母；数学答案末尾数字可能是附加说明。保存原始输出和解析结果，抽查失败。A5使用官方formatter和reward grader保证与题目协议一致；额外robust parser应作为另一个评估版本。

对于n次独立样本、其中c正确，pass@k无偏估计常用 `1−C(n−c,k)/C(n,k)`（k≤n）。它不是accuracy提高后的免费指标，计算预算随采样次数增加，比较要控制成本。

## 4. 偏好评价与 LLM-as-Judge

评审会受到答案长度、位置、风格、模型身份、语气等影响。pairwise比较需要随机交换左右顺序、隐藏身份，记录tie/uncertain。LLM judge的system-level排名与human相符，不代表每个样本准确，更不代表没有共享偏差。

length-controlled win rate可以减轻部分长度混杂，但不能替代任务实质正确性。用同一个模型生成数据、当reward model、再当最终judge，可能放大闭环偏差。至少加入独立规则/测试、人工抽查或另一类judge。

## 5. Agent 与安全的评价

agent任务需固定工具、网络、环境状态、权限、超时与恢复规则，评价最终state而不只对话文本。重复运行同一环境可能泄漏答案；环境版本变化也会破坏可比性。安全评价既要测危险请求拒绝，也要测无害请求过度拒绝，以及多轮/改写/工具路径。

不能把一次安全benchmark高分泛化为所有部署安全。面向具体产品建立失败分类：错误调用工具、越权、泄露、幻觉引用、流程中断等，再按严重程度与发生频率排优先级。

## 6. 污染、置信区间与误差分析

预训练语料可能含题干、答案、近重复和讨论，污染会让结果虚高。exact match不足以覆盖改写/翻译，需要n-gram、MinHash、语义检测与样本人工核验。仅删除评估题目字符串不能保证没有答案关联内容。

accuracy的粗略标准误差为 `sqrt(p(1-p)/n)`，样本非独立时需要按topic/problem分组bootstrap。比较同一批题两模型最好做配对统计，记录共同成功、共同失败与分歧，而非只比较两个总体百分数。

误差分析至少区分知识缺失、推理错误、格式错误、截断、工具失败和grader误判。纠正格式错误可提高表面accuracy，但不能说新增了推理能力。

## 7. 作业与岗位产出

A4对每种清洗pipeline使用同一架构/token budget，评估C4/Paloma时严格匹配官方GPT2 tokenizer和处理协议；A5同时报告answer reward、format reward、总reward、输出长度和pass@1。toy smoke训练的loss下降只是流水线验证，不能作为benchmark成绩。

自测：两个模型PPL分别10与12，但tokenizer不同，结论是什么？无法直接排序，需要共同协议。追问：聊天win rate高但code tests低如何处理？根据产品任务权重拆分指标，分析数据/长度/judge偏差，而非寻找一个覆盖全部能力的总分。
