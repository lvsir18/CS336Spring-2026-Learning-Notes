# 从 CS336 到大模型研发作品集

## 能力与证据

| 方向 | 能做的事 | 对应课程/作品 |
|---|---|---|
| 预训练研发 | 实现模型、稳定优化、分析loss | A1训练/消融/恢复，1–4讲 |
| 训练系统 | 数值正确的kernel与分片、profile | A2误差/显存/吞吐，5–8讲 |
| Scaling/研究 | 控制预算、拟合/外推/不确定性 | A3曲线与独立验证，9/11讲 |
| 推理工程 | 理解prefill/decode/KV/调度/SLO | 10/19讲服务实验 |
| 数据工程 | 可追溯获取、清洗去重、模型验收 | A4数据漏斗与PPL，13–14讲 |
| 后训练 | SFT/DPO/RLVR、reward诊断 | A5多seed与目标消融，15–16讲 |
| 多模态 | 表征对齐、visualtokens与评价 | 17讲独立扩展 |

## 推荐完成三项可展示作品

**从零小LM：**有自训练tokenizer、model/optimizer、losscurve、恢复与生成、LR和架构消融。展示你如何发现并修复一个实际问题，说明tokenbudget与验证协议。

**系统优化：**attention前后向数值对照、GPUbenchmark原始数据、peakcurve与profiler证据。对速度没有提升的case同样解释原因；比较相同dtype/shape，不用CPUtoy时间充GPU结果。

**数据+RL闭环：**固定模型下数据清洗消融，OLMo数学任务的prompt与GRPO/Dr/RFT/MaxRL对照。展示格式/答案/长度/entropy与reward区别，说明为什么选某个recipe。

这些比堆砌“复现所有SOTA”更易评审。简历只写亲自复核过的结果，报告真实设备与预算，区分借鉴、原创和待运行部分。

## 面试讲述结构

问题/约束→初始假设→实现要点→实验协议→结果→失败与局限→下一步。用具体数据说明，不只列术语。问你为什么采用Flash、GQA、FSDP或GRPO，先说瓶颈/目标，再讲机制与代价。

高频追问：6ND何时失效；大batch与microbatch是否等价；prenorm梯度路径；Flash为何是精确attention且仍平方计算；FSDP释放时机；DP/TP如何放置；tokenizerPPL可比性；MinHash概率；DPO推导；GRPO长度/难度偏差；oldpolicy和reference区别。答案分散在每讲self-test与mathguide，闭卷推导后再核对。

## 后续扩展选择

先完成已有五份实验，不同时开十个新方向。推理岗可扩展KVcache+continuousbatching；系统岗可优化Tritontiling/autotune和bucketDDP；数据岗可做外存LSH与领域mixing；后训练岗可做rolloutbackend/版本管理与可靠verifier。每次扩展设同质量或同成本对照，产物包含可运行代码与报告。
