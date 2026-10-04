# 第 19 讲：Dan Fu 嘉宾课：推理系统与全栈协同的延伸

> 2026 Spring · 官网日期 06-03 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://www.youtube.com/watch?v=9EEm4iMAF5s)。公式中的简化假设会单独说明。

## 1. 来源与编号

官网6月3日列出Dan Fu嘉宾课，公开录像为[官方视频](https://www.youtube.com/watch?v=9EEm4iMAF5s)。公开播放列表称其Lecture18，本仓库按官网19次授课日程记为19。核验了录像链接与相关公开研究入口，但没有逐秒校对完整字幕；本篇是围绕公开主题和原始论文的自学补充，不提供未经核实的课堂逐字引用或时间戳。

相关入口：[Dan Fu个人主页](https://danfu.org/)、[Parcae项目](https://sandyresearch.github.io/parcae/)、[Parcae论文](https://arxiv.org/abs/2604.12946)、[代码](https://github.com/sandyresearch/parcae)。阅读论文时应进一步核对架构、训练与实验设置，不能仅从项目页宣传数字推断适合所有模型。

## 2. 推理优化需要跨层看系统

传统优化在固定Transformer结构上改善kernel与调度；全栈协同同时考虑模型结构、执行图、硬件资源与服务负载。例如减少串行layers可能降低launch与同步critical path，增大可重复利用的局部计算可能适合persistent execution，但模型质量和训练难度也会改变。

单个kernel最快不代表service最快：TTFT受prefill与queue影响，TPOT受decode与KV带宽影响，总goodput受SLO和workload影响。结构变化的收益必须在相同质量目标和服务约束下比较。

## 3. Megakernel 与 Persistent Execution 的直觉

普通执行在每层、每算子之间启动多个kernels，临时结果可能写回设备内存；megakernel将更大范围计算保留在一个长期驻留的执行中，降低launch和某些中间IO。它需要全局/阶段同步、资源分配与不同算子的布局转换。

过度融合也有代价：寄存器/shared压力大、occupancy下降、不同阶段难以各选最优tiling、全局同步受限。大型权重与KV不可能都常驻片内存储，因此“全模型一个kernel”不意味着不读DRAM。是否合算要分析实际瓶颈、batch大小与模型形状。

Triton小算子融合与全模型persistent design不是同一工程复杂度。A2可训练基本技能，扩展到这类系统前要掌握profiling、同步、layout和数值回归。

## 4. Looped Models 与 Stable Computation 的预习

looped/recurrent-depth架构在多个计算迭代复用某些参数，与普通每层独立参数相比，参数存储与compute深度不再简单线性相关。简化写作 `h_{k+1}=f_θ(h_k,x)`，迭代次数控制计算量。它可以为test-time compute提供结构路径，但需要训练稳定性、迭代状态与停止机制设计。

若局部Jacobian的有效增益持续大于1，误差可能随循环放大；稳定/收敛结构帮助控制多次迭代行为。这是理解相关研究的数学预习，不将简单收缩映射分析等同于Parcae论文全部理论。真正实现应逐项遵照原论文与代码。

## 5. 与 Scaling 和 Serving 的联系

参数复用会改变 `C≈6ND` 中N的解释：每次loop都进行计算，不能只按唯一参数总量估训练/推理FLOPs。比较需要同时报告unique parameters、active FLOPs、iterations、latency和quality。

如果可按任务难度增加iterations，服务调度需要考虑每请求不同计算预算；退出早的请求让batch动态变化。固定iterations吞吐与自适应iterations的真实负载表现应分开评估。训练最优、推理最优与质量最优仍是不同目标。

## 6. 可以做的单卡研究练习

从小decoder的一个block复用开始，比较同unique参数、同FLOPs两种约束下的loss和推理速度；记录循环中activation RMS与输出变化。先用eager正确实现，再torch.compile，再考虑融合。所有结果标成自己的实验，不声称复现Parcae性能。

benchmark至少包括batch1/8/32、不同context、迭代次数和warmup；画quality-latency Pareto曲线，而非只挑一个最快点。误差比较输出、token log-probs与最终任务准确率，速度比较steady state及compile/load成本。

## 7. 面试与作品方向

推理岗候选人应能从模型shape解释memory/compute bottleneck，从profiler解释launch与同步，再讨论改变架构是否值得训练成本。研究岗还需提出同质量/同成本的公平对照，避免系统改动和模型改动混在一个成绩中。

读论文时记录：核心假设、结构、训练recipe、kernel约束、硬件/workload、准确率基线、性能基线、未覆盖case和可复现artifact。课程结束后可将A2 FlashAttention、A3成本模型与A5推理预算联合成一个小型研究项目。
