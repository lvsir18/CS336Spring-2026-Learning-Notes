# 第 18 讲：Daniel Selsam 嘉宾课：资料状态与相关预习

> 2026 Spring · 官网日期 06-01 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://cs336.stanford.edu/)。公式中的简化假设会单独说明。

## 1. 可核实的内容与缺口

官网2026日程列出6月1日嘉宾Daniel Selsam。核验时未找到与该日期对应的公开课件或完整官方录像，官方公开播放列表还出现“18讲”与官网19次授课的编号差异。因此本篇不重建演讲观点、不编造课堂例子与结论；它保留该次课程的独立Markdown，并提供与形式化推理相关的自学预习。下列内容是补充知识，不代表已核实的嘉宾讲述主题或立场。

以后获得官方slides/video，应核对speaker、日期、内容范围，再补正文与时间戳。仅凭人物研究领域猜测演讲内容不可作为课程来源。

## 2. 补充预习：自然语言推理与形式化验证

自然语言答案可能看似合理却有步骤错误；数值verifier只能检查最终结果，无法证明全过程。形式化系统将命题、类型和证明转成可机器检查的对象，证明kernel验证对象是否满足逻辑规则。搜索证明与检查证明是两件事：模型负责提出候选，checker负责是否接受。

Lean等proof assistant在环境中定义符号、theorems、tactics与依赖。自动生成的代码需要在确切版本的library和环境中编译，错误可能来自语法、类型、缺失lemma、搜索失败，而不是命题一定不成立。

主要阅读入口：[Lean 官方文档](https://lean-lang.org/documentation/)，[Lean 语言参考](https://lean-lang.org/doc/reference/latest/)。这些是预习资料，不能用作该嘉宾课授课内容的证据。

## 3. 把证明搜索视作 RL 环境

state可以是当前proof goals、context和环境；action是tactic或proof片段；transition由执行器产生新goals或错误；终止reward来自完整proof被kernel接受。与A5数学答案奖励相比，这类verifier更严格，但reward稀疏、运行成本高、动作依赖语法和环境。

一个最小流程：固定problem corpus与library版本→模型生成action→有超时地执行→记录新state/错误→搜索或采样下一步→checker验证完整proof→保存可复现trace。成功trace可用于SFT/rejection sampling；失败trace也可用于错误分类与搜索策略分析。

成功不意味着模型发现了人类可读的最优证明，也不保证形式化命题忠实表达原始自然语言。autoformalization本身有specification gap，需要人工或独立方法验证题意。

## 4. 评价应分开计量

- formalization正确率：是否准确表达题目。
- proof validity：checker是否接受。
- pass@k：在固定次数/时间预算内找到证明的概率。
- cost：tokens、执行次数、wall-clock与内存。
- robustness：换library版本、稍改题目或移除近重复后的表现。

不能把有形式证明的错误规格称为原题解决。proof checker验证的是输入给它的命题，不验证产品需求或自然语言用户意图。

## 5. 实践准备与面试连接

先选简单代数命题，手工理解goal、context与一个合法tactic序列；把成功/类型错误/超时各记录一个trace。再尝试小规模best-of-N search，固定最大nodes与time，比较成功率。此处是自学实验建议，没有随仓库自动安装Lean或声称已运行。

它与第16讲连接在verifiable reward、稀疏反馈、采样预算与分布外泛化，与第12讲连接在specification/validity。面试追问“有checker为何还会错”，答案应含错误形式化、环境依赖、未覆盖需求与checker范围，而不是简单回答“checker完全可靠”。

## 6. 后续补全清单

取得正式材料后补充：演讲标题、原始链接、5至8个主题、核心算法/案例、与GRPO/agent的联系、公开实验和局限。每项引用真实page/time；删除与演讲无关的预习内容或移入guide。当前资料状态是**缺少正式材料**，不能标记为完整课堂复现。
