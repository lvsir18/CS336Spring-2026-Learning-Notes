# 单卡96GB · 12周学习路线

这是一份可自行调整的学习建议，不是课程官方deadline或自动任务。每周5天、每天约2–3小时为示例；已经学过的内容可压缩，真实GPU实验时间另测。

| 周 | 理论 | 实践 | 可验收产物 |
|---|---|---|---|
| 1 | 第1讲，Unicode/BPE | 手工merges、Tokenizer训练/编码 | roundtrip、压缩/耗时统计、失败case |
| 2 | 第2–3讲 | A1模型/optimizer/tests | shape图、参数/FLOPs表、梯度检查 |
| 3 | 第4讲及A1训练 | TinyStories短跑→完整训练/消融 | loss曲线、配置、生成与resume记录 |
| 4 | 第5–6讲 | A2GPU测试、profile/Flash | forward/backward误差、速度/显存曲线 |
| 5 | 第7–8讲 | Gloo DDP/ZeRO/FSDP | globalbatch数值对照、通信推导 |
| 6 | 第9/11讲 | A3离线拟合/小网格 | IsoFLOPs、残差、holdout外推 |
| 7 | 第10/12讲 | KV/服务指标、评估协议 | TTFT/TPOT实验设计、误差分类 |
| 8 | 第13–14讲 | A4获取/抽取/过滤/去重 | 数据漏斗、人工抽查、hash manifest |
| 9 | A4模型验证 | 固定协议清洗消融 | validationPPL与成本，注明单卡适配 |
| 10 | 第15讲 | SFT/DPO | mask样例、loss推导、reference对照 |
| 11 | 第16讲 | OLMo/GSM8K基线/GRPO | reward/length/entropy、多seed分析 |
| 12 | 第17–19讲 | 变体消融、资料补充、整理 | 两repo可复现README、面试讲述 |

## 每讲的主动学习循环

第一次先看本讲目标/结构，不逐行抄写；第二次看官方原稿或录像，用笔记补公式。闭卷画计算图，说明每个轴和每步数据流，再运行一个最小实验；最后写“预测是什么、测到了什么、哪些结论尚未证实”。如果实验不符合预期，先定位数据/数值/协议问题，再修改算法。

每讲自测问题已有简短答案，应先自己答再核对。重要公式如6ND、在线softmax、ringvolume、scaling最优、DPO、GRPO可以从头推出来，不只背最终式。

## GPU实验的预算顺序

先CPU正确性→GPU小尺寸→10–30step容量/稳定性→更长预算→关键消融/重复seed。单卡没有多GPU网络，CPU模拟只验证语义，多卡吞吐行写pending。A3官方API还受凭据/预算约束，自有卡替代实验单独报告。

验证曲线不要只盯最后一个点。中间loss spikes、formatreward、zero-variancegroups、clipfraction和throughput突变通常提供更重要的诊断信息。

## 每周复盘模板

本周能够从零解释的概念；仍需查资料的步骤；亲自实现/运行的artifact；失败case与修复；下周一个可判别的实验。复盘写入你自己的学习日志，保留日期和commit，不将本仓库已有结果归为自己亲跑的服务器实验。
