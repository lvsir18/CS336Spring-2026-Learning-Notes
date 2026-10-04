# 第 01 讲：课程总览与 Tokenization

> 2026 Spring · 官网日期 03-30 · 中文自学笔记
> 资料核验日期：2026-10-04。以下为原创整理、推导与工程补充，不是官方讲义的逐字翻译。

主要依据：[官方课程](https://cs336.stanford.edu/)；[本讲材料](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)。公式中的简化假设会单独说明。

## 1. 从“调用模型”走到“制造模型”

CS336 的主线是把文本、算力、参数与评价组成一个闭环：先把字符串变成离散 token；实现 next-token 模型；使它在设备上高效训练；用小实验预测大实验；整理数据；最后将语言建模能力转成指令遵循与推理能力。课程五份作业对应 basics、systems、scaling、data、alignment。找大模型研发岗位时，这五块的交叉处往往最有价值，例如数据清洗改变 token 分布，token 分布又改变吞吐与优化难度。

语言模型建模的是

$$p_\theta(x_{1:T})=\prod_{t=1}^T p_\theta(x_t\mid x_{<t}),\qquad
\mathcal L=-\frac1T\sum_t\log p_\theta(x_t\mid x_{<t}).$$

这是 teacher forcing：训练时已知历史真实 token，一次并行计算所有位置；生成时历史包含自己刚刚采样的结果，只能逐步推进。预训练损失降低并不直接保证聊天质量，这个区分贯穿后续评估与后训练。

## 2. Tokenizer 到底优化什么

编码器 `encode: str → list[int]` 和解码器 `decode: list[int] → str` 将非结构化文本转成有限词表。希望它同时满足：任意输入可表示、可逆、压缩率高、序列不过长、训练与服务行为一致。字符级简单但序列长；词级在中文、代码、新词上切分困难，词表增长快；UTF-8 byte 级只有 256 个基础符号，却会把常见词拆成很多 token。BPE 在 byte 完备性的基础上学习常见片段。

不要把 token 看成固定字数。`你好`、英文空格、emoji、源码缩进、罕见姓名的编码成本不同。UTF-8 一个 Unicode 字符可能占多个 byte；截断 token 后 decode 可能出现替换字符，只有完整编码的原文保证正常 round trip。文本字符数、UTF-8 byte 数、token 数应分别记录。

压缩指标可用 `UTF8_bytes / tokens`，要固定语料和特殊 token 规则。例如 1 GB 文本编码成 250 M token，压缩比约 4 byte/token。更高压缩通常减少 attention 序列成本，却扩大 embedding 和输出投影，不能单独最大化这一指标。

## 3. Byte-level BPE 逐步推导

1. 初始化 256 个单 byte token；按规定加入 `<|endoftext|>` 等特殊 token。
2. 用预分词正则切出词、标点、数字、空白片段。每个片段转成 byte 序列。
3. 统计每个片段频率；统计片段内部相邻 token pair 的加权出现次数。
4. 选择出现最多的 pair，合并成一个新 token，并更新受影响的 pair。
5. 达到目标词表大小，保存最终词表和**按学习顺序排列**的 merges。

示例片段 `low` 出现 5 次，`lower` 出现 2 次。`(l,o)`、`(o,w)` 各贡献 7，而 `(w,e)` 贡献 2。先合并哪个取决于频次与 tie-break 规则；A1 要求频次相同按 pair 的 byte 字典序选最大的，随意依赖字典插入顺序会导致训练结果不一致。统计中同一词含重复 pair 时要计每次出现，例如 `aaaa` 有三次 `(a,a)`；实际合并按非重叠的左到右扫描进行。

编码不是再运行训练。对每个预分词片段，反复选择当前相邻 pair 中 merge rank 最小的一对；训练中越早学到的 pair，rank 越小。词表“最长字符串匹配”不一定等价于按 merge rank 编码。解码则连接 token 对应的 bytes，再做 UTF-8 decode。

## 4. 特殊 token 与流式输入

特殊 token 必须作为原子符号处理，且不能让普通 BPE 跨它合并。若特殊 token 同时有 `<x>` 和 `<x>abc`，匹配顺序应优先更长的完整字符串。包含用户输入时，还应显式区分允许解释为控制符的字符串与普通字面文本，服务端不能因为内容里出现角色标记就改变权限。

`encode_iterable` 的危险点是边界。把一段文本按任意 4 KB 切块，可能切开 UTF-8 字符、一个预分词片段或特殊 token；各块独立编码不保证等价于一次编码全文。我们的编码脚本选择**逐行编码且保留换行**，会在元数据中记录该语义；它适合可复现自学实验，若要完全复现官方语料 token 序列，必须匹配官方分块方式。训练 BPE 的初版会读入整个文本，几十 GB 语料应扩展为 special-boundary 分块并行预分词与局部统计归并。

## 5. 性能优化与代码路线

最慢的朴素实现每次合并都遍历全部词和全部 pair。可保存 `word → frequency`、`pair → weighted_count`、`pair → owners`，只更新含该 pair 的词；优先队列允许过期条目，弹出时检查当前统计值。大量重复片段可以使用编码缓存，但缓存大小须有上限。

对应 A1：`src/cs336_basics/tokenizer.py` 的 `train_bpe`、`Tokenizer.encode`、`decode`、`save/load`。官方测试覆盖非英文、特殊符号、编码兼容与 streaming；不能拿测试文本训练最终实验 tokenizer。训练/验证集分离也适用于 tokenizer 训练，避免把验证数据分布提前注入学习流程。

## 6. 自测与面试追问

- **为什么不直接用 Unicode 字符作为词表？**字符集合持续扩展，稀有字符与归一化复杂；byte 基础词表能覆盖任何合法 UTF-8 字符串。
- **大词表一定更快吗？**序列变短减少 attention 和逐 token 解码步数，但 `B×T×V` logits、`V×d` 参数与 softmax 成本上升。
- **两种 tokenizer 的 perplexity 能直接比较吗？**不能；每 token 的单位不同。应使用同一 tokenizer 或换算 bits/byte 等共同单位，再评估下游任务。
- **练习：**手工对 `ab ab abab` 做两个合并；分别写出 pair 次数、tie-break 结果与 encode 输出。然后实现一次 save/load round trip，并检查 `decode(encode(text)) == text`。

岗位产出应是 tokenizer 训练配置、压缩统计、耗时/内存、特殊 token 设计及失败案例，而不是只展示一个编码样例。
