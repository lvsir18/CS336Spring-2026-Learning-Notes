# 第 13 讲：预训练数据来源与公开数据集

> 配套实践：[A4 实现与实验](https://github.com/lvsir18/CS336Spring-2026-Learning-Assignments/blob/main/docs/a4.md) · [课程与作业对应表](../guides/assignments.md)。具体已实现和已验证的范围以作业说明为准。

> 2026 Spring · 官网日期 05-11 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_13.py)。公式中的简化假设会单独说明。

## 1. 数据集不是一堆可互换的 token

数据来源决定语言、领域、风格、知识时效、重复、噪声与许可条件。网页量大但内容噪声多，代码有语法与可执行结构，数学/论文有专业符号，书籍提供长连贯文本，对话与问答提供交互模式。所谓“高质量”必须相对训练目标定义，不能把维基风格等同于全部用户语言。

数据账本应记录source、抓取时间、原始URI/ID、语言、许可、清洗版本、tokenizer和进入训练的比例。派生数据要能追溯回原始样本，方便排查污染、错误过滤与重复。

## 2. Common Crawl 的几个层次

WARC保存抓取的HTTP记录与payload；WAT保存一些元数据；WET保存提取文本。处理WARC时要过滤record type、响应状态、content-type与解码，再做HTML主文抽取。直接去掉HTML tags无法去除导航、广告、cookie弹窗与重复模板，也可能丢失段落结构。

crawl index用于找到分片，不要默认下载整个crawl；单分片也可能很大。先对小样本验证解码、抽取、language与质量统计，再扩大规模。保存download hash、source shard与offset，输出JSONL不要只留无来源text。

encoding猜错会产生大量乱码，后续language filter与quality classifier会被误导。主文抽取质量用人工样本核验，而非只看平均字符数。表格、代码块、数学页面尤其容易被抽取器损坏。

## 3. 结构化来源与取舍

Wikipedia较整洁但覆盖有限，模板、引用与列表需要处理；代码仓库需要按文件类型、license、生成文件和fork重复过滤，不能只统计GitHub stars；arXiv存在LaTeX宏、公式、bibliography与版本重复；问答站点有问题/答案/评论结构和信誉信号；书籍有版权及版本差异，长文本也影响packing。

技术任务还需要可执行环境与依赖版本，代码文本“像代码”不保证能运行。数学数据应保留符号结构，不把复杂公式当垃圾删除。多语数据要报告每语种token化效率与过滤通过率，英文训练出的quality classifier可能系统性削弱其他语言。

## 4. 公开数据集的演进逻辑

讲义从BERT/Wikipedia、GPT-2 WebText、CCNet/C4、GPT-3、The Pile、Gopher、Llama，延伸到RefinedWeb、Dolma、DCLM、Nemotron-CC、The Stack和Common Pile。重点在构建方法变化：网页代理质量信号、语言识别、规则清洗、模型过滤、全局去重、数据混合与更完整的数据透明度。

公开token数量不能直接排序数据价值：不同tokenizer、dedup粒度、重复epoch、语种占比与质量阈值都会改变“多少token”。有的公开的是raw corpus，有的是training mixture，有的是仅配方；复现时先确认可用artifact。

## 5. Data Quality 是可测试的假设

假设A：删除模板文本降低噪声；假设B：保留教育性内容提高某些任务；假设C：过强过滤损伤代码/少数语种。三者都应通过相同模型/token budget的消融验证。classifier只是代理，验证集中的accuracy不等于该过滤器最大化LLM训练效果。

保留raw、extracted、filtered、deduped各阶段统计，记录文档数、UTF8 bytes、tokens、length histogram、语言比例和排除原因。多个filter顺序不同会使通过率变化，统计应使用stage-by-stage漏斗，同时单独量化重叠。

## 6. 来源与使用边界

网页公开可读不自动等于可任意再分发。记录license、来源与限制，第三方课程fixtures保留上游license。此仓库不把大型原始语料或模型权重提交Git，只提供下载与处理方法；用户后续展示的是代码、实验配置、统计和报告。

许可问题不能只在最后补一句“用于学习”，数据获取阶段就应有source manifest。这里是工程流程说明，不代替具体项目法律审核。

## 7. A4 路线与实践

`process_crawl.py`读取WARC/WET/JSONL，依次抽取、规则过滤、language、NSFW/toxic/quality、PII替换，输出可追溯JSONL和统计。fastText权重必须真实下载，缺少模型明确报错；不能用几个关键词假装官方classifier。

练习：随机抽100个网页，按正文完整、模板残留、乱码、代码破坏分类，统计extractor失败比例；为20个不同语言网页比较byte/token，分析训练采样按documents、bytes、tokens的区别。

面试问“如何造一个好数据集”，应给出来源→抽取→过滤→去重→混合→污染检查→模型验证的闭环，并说明每步可能损失什么有价值的数据。
