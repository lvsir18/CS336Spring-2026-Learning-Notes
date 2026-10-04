# 课程与来源说明

## 官方入口

- [CS336 Spring 2026官网](https://cs336.stanford.edu/)：课程名称Language Modeling from Scratch，日程、讲师、讲义、作业与课程规则入口。
- [官方2026录像列表](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)。
- [lectures仓库](https://github.com/stanford-cs336/lectures)：Python形式可执行讲稿与Tatsu课件PDF。
- [A1 Basics](https://github.com/stanford-cs336/assignment1-basics)、[A2 Systems](https://github.com/stanford-cs336/assignment2-systems)、[A3 Scaling](https://github.com/stanford-cs336/assignment3-scaling)、[A4 Data](https://github.com/stanford-cs336/assignment4-data)、[A5 Alignment](https://github.com/stanford-cs336/assignment5-alignment)。

2026授课主线由Percy Liang、Tatsunori Hashimoto讲授，从3月30日至6月3日。按官网19个日期编号，兼顾公开播放列表少一次/不同编号情况。正式材料优先于社区标题；公开社区的7/8讲标题与某些年份/教师编号不同，不按旧年份目录盲目迁移。

## 社区材料如何使用

交叉核查入口：[narain1](https://github.com/narain1/stanford-cs336-2026-notes)、[Young-Lord](https://github.com/Young-Lord/cs336-2026-notes)、[Yulong Ge开放课程笔记](https://yulong-ge.github.io/open-course-notes/courses/stanford-cs336-2026/)、[hqhq1025](https://hqhq1025.github.io/ai-course-notes/cs336-2026/lecture18/)。其中部分由AI/transcript生成，作为导航和遗漏提示，不作为未经核验技术结论的唯一证据，不直接复用其长篇正文。

每讲开头链接对应官方文件；自行推导、单卡容量建议和面试题均为扩展。重要区分：官方授课内容、原始论文、我们的解释、我们实测、待运行实验。这五种证据级别不能相互冒用。

## 版本差异

A1 README存在旧年份字样，实际以2026题目为准；A2包含新增FSDP；A3变为B200 wall-clock API实验；A4最终固定8B200/8.59Btokens协议；A5主体为OLMo1B reasoningRL，安全SFT/DPO是补充。2025教程可用来学习基础，但不直接作为2026完整流程。

`source-manifest.json`记录2026-10-04抓取的archive/file SHA256，下载ref为main，不能声称其永不变化。配套代码仓库将A5测试资源进一步固定commit并校验每个file；原创笔记Gitcommit固定本文版本。

## 公开资料缺口

6月1日Daniel Selsam课只有官网安排得到核验，尚缺正式lecture artifact；第18篇保留自学预习且明确不重建演讲。6月3日Dan Fu有公开录像和研究链接，但未逐秒校对字幕；第19篇是谨慎延伸，后续可补真实time/page引用。当前不宣称完整记录了这两次嘉宾演讲。

## 学习准备

建议先掌握Python、NumPy/PyTorch、矩阵乘/链式法则、概率与最大似然、Transformer基本图。不会某项不必停止整门课，先对照math guide手算和小tensor检查。课程学习目的应是能解释、实现、测量、修正，而不是背术语或仅调用现成trainer。
