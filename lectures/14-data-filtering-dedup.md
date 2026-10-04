# 第 14 讲：过滤、去重、混合与合成数据

> 2026 Spring · 官网日期 05-13 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py)。公式中的简化假设会单独说明。

## 1. 数据转换的每步都改变训练分布

主文提取、规范化、过滤、PII替换、exact/fuzzy dedup、领域混合和synthetic data不是互不相关的清洗工具。过强规范化可能把代码变量名、数学符号或语种差异抹掉；去重可能降低过拟合，也可能减少某些长尾领域；质量过滤优化代理指标，需验证真实模型效果。

A4规则包含长度、平均词长、ellipsis行比例、含字母词比例等。阈值来自具体作业协议，不是所有语种生产通用标准。平均词长与word definition必须一致，空文档/无word应明确定义为拒绝。

## 2. Exact Dedup 的层次

文档级hash快速找完整重复，行级消除模板，n-gram级能处理部分重叠。先明确“重复后保留一个”还是“所有重复出现都删除”。A4 exact-line题要求对全体文件出现多次的行删除**每一次出现**，不是保留首条；实现需先统计全局count，再第二遍输出唯一行。

hash collision理论可能存在，安全要求高时用强hash并对命中做内容校验。规范化需要版本化：NFD、去accent、大小写、标点、空白各改变相等关系。文件同名与输出覆盖也会破坏结果，脚本应检测。

## 3. Jaccard 与 MinHash 推导

将文档转成连续k-word shingles集合A、B：

$$J(A,B)=|A\cap B|/|A\cup B|.$$

对全集做均匀随机排列，取每个集合最小元素，两个minhash相同的概率等于Jaccard：它们相同当且仅当并集最先出现的元素属于交集。实际用多个独立hash近似permutations，signature长度m，命中比例估计J。不是把整篇文本hash一次就能估相似。

集合去掉shingle重复，因此主要比较覆盖而非词频。短于k词的文档需要定义处理策略，不能全部映射成同一个空signature，否则所有短文被判为重复。我们的实现用整段短文本作shingle，并对候选做真实Jaccard验证。

## 4. LSH 的候选概率

将m=b×r个signature分b个bands，每band r行；任一个band完全相同就成为候选。若hash独立近似，候选概率

$$P_{candidate}(s)=1-(1-s^r)^b.$$

增加bands提高召回也增候选数；增加rows降低不相似pair候选概率，也可能漏掉目标阈值附近的pair。LSH不是最终相似阈值判定，必须对候选计算真实Jaccard并筛选。调参根据目标相似度与候选规模，而不是随便写20×5。

## 5. 近重复簇的传递性

A与B相似、B与C相似，但A与C可能不达阈值。按相似边连通分量去重会把三者当一个cluster，这就是题目指定的cluster语义。union-find可以高效构造connected components；保留最低索引使结果确定，但也可能保留较差版本。生产可按来源/长度/质量选代表，需明确策略且用固定seed避免不可复现。

本仓库MinHash教学实现会把shingles/signatures放在RAM，适合小中规模实验；十亿文档需分区签名生成、外排序band keys、候选去重、分布式边处理与cluster归并。大O和输出中间存储量都要预估，不能仅靠“用了LSH”宣称能处理万亿tokens。

## 6. 模型过滤、混合与合成数据

fastText质量classifier以正/负样本学习轻量代理，train/heldout split要先完成，不能用官方测试fixtures训练。标签分布会导致偏差：Wiki为正、CC为负可能把网页排版而非内容质量学成特征。记录不同来源、语种和长度的误杀率。

数据mixing按token比例采样；温度采样可给低资源域更多权重，但有限域重复过多有收益递减。通过小模型scaling/消融估领域间transfer，再配合产品目标选择。独立领域validation比一个整体loss更有解释力。

synthetic data可用于指令、推理轨迹、改写与难例；需要生成模型版本、prompt、seed、筛选器、verifier与来源。只学习模型自己的高概率样本可能缩窄分布。对reasoning保留经验证正确的答案，不把流畅过程当正确证明。

## 7. 污染与验收

去重训练内部与排除评估污染是不同任务；需要同时比较train-train、train-val、train-benchmark。对题干、答案和改写都有检查策略，评估报告保留污染规则与命中样本数量。

A4最终验收是相同training budget下模型loss/任务指标改善，而不是删除文档最多。报告raw→clean→dedup每阶段tokens、quality/语言变化、运行时间与样本示例；对PII仅展示已替换的例子。

自测：s=0.8、r=5、b=20，候选概率约 `1−(1−0.8^5)^20≈0.99965`。如果最终threshold=0.8，候选不等于最终通过；真实Jaccard为0.75的pair即使命中band仍要排除。
