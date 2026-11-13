# 大模型「说人话」度 Benchmark 调研报告

调研日期：2026-11-13。调研问题：是否存在评测大模型「说人话」程度（或反方向「最不说人话」程度）的 benchmark，以及自建中文 benchmark 的空间。

## 结论摘要

- 「最不说人话」方向已有现成 benchmark，集中在英文世界：SlopBench（论文）和 The Slop Index（开源排行榜），两者都以可量化的表面特征给模型的 AI 味打分排名。
- 中文世界目前没有「说人话度」的模型评测基准。最接近的 HeartBench 测的是情感与社交智能（「懂人」），不是语言风格（「像正常人说话」）。
- 中文「说人话」相关项目全部是改写工具（Agent Skill），自带评测只测改写效果，不给模型排名。
- 自建中文「说人话 / AI 味」benchmark 有明确的差异化空间，方法可直接借鉴 SlopBench 与 Slop Index。

## 直接相关：英文 Slop 评测

「Slop」指语法正确、逻辑流畅但信息密度低、套话密集、一眼可辨的 AI 味文本。以下两个项目直接评测模型产出 slop 的程度，即「最不说人话」排行榜。

### SlopBench

论文《SlopBench: How Well Can We Rank Language Models by Slop? A Multi-Domain Benchmark of Repetitive AI Writing》，Dhruv Roongta、Harsha Gaddipati、An-Nghiep Huynh，2026 年 9 月 27 日发布。

| 项目 | 内容 |
| --- | --- |
| 定位 | AI 文本检测器只回答「是不是机器写的」，SlopBench 回答「哪个模型写得更僵硬、更重复」 |
| 规模 | 18 个模型 × 112 个手写任务，每任务最多采样 10 次，共 19,928 条输出 |
| 任务领域 | 邮件、社交帖、essay、工作群聊四类 |
| 评分方式 | 四个读者可手工核验的表面行为，包括输出长度对照任务规定的字数带、跨样本的开头句式重复等 |
| 特点 | 不依赖裁判模型的主观打分，指标客观、可复现 |

### The Slop Index

GitHub 开源项目（hgaddipati1118/slop-index，MIT 协议），给前沿模型的写作打 slop 分，与 pre-AI 时代人类文本基线对比。0 分代表写得像 ChatGPT 之前的人类，分数越高越 slop。

评分公式：总分 = 40% 人类偏好 + 60% 机械成分。机械成分含四个维度，各占 15%：

| 维度 | 含义 |
| --- | --- |
| Conciseness | 简洁度，是否超出必要篇幅 |
| Templating | 模板化程度，结构是否千篇一律 |
| Rhythm | 节奏，句式长短是否机械均匀 |
| Tells | AI 痕迹，标志性套话与句式 |

人类基线语料全部早于 ChatGPT：Enron 邮件、Blog Authorship 博客、ASAP 学生作文、Sentiment140 推文、DISCO 对话。任务领域与 SlopBench 相同（邮件、社交、essay、聊天），两者出自同一作者团队。

部分排名（机械成分原始分）：

| 排名 | 模型 | Slop Index |
| --- | --- | --- |
| 1（最 slop） | Mistral Large | 40.6 |
| 2 | Fable 5（Anthropic） | 35.6 |
| 3 | GPT-5.6 Terra（OpenAI） | 32.8 |
| 16 | MiniMax M3 | 23.1 |
| 17 | Muse Spark（Meta） | 23.1 |
| 18（最干净） | Kimi K2.6（Moonshot） | 21.1 |

核心发现：slop 程度与价格不相关，榜上最贵的模型接近最 slop 一端，最便宜的模型之一接近最干净一端。全部输出与评分脚本已提交到仓库，可用 `harness/score.py` 复现。

## 相邻方向

以下项目与「人味儿」相关，但评测角度与「说人话」不重合。

### HeartBench：中文拟人化评测

蚂蚁 inclusionAI 开源，国内首个面向中文心理社科领域的拟人化评测基准（arXiv:2512.21849）。

- 覆盖人格、情绪、社交等 5 大维度、15 项能力，296 题 + 2818 条评分标准。
- 专家盲测验证人机评分一致率 86%。
- 评测目标是模型是否「懂人」（情感理解、共情、社交智能），不是语言风格是否自然。与「说人话度」互补而非竞争。

### 图灵测试系：人机可分辨性

| 项目 | 内容 | 与「说人话」的区别 |
| --- | --- | --- |
| TuringBench（2021） | 20 万条人类 / 机器文本，20 个来源标签，图灵测试与作者归属两个任务 | 测检测器能否分辨人机，不给生成模型的说话自然度排名 |
| Human or Not（AI21，2023） | 大规模社会图灵测试实验，人类评判者准确率约 60% | 测「能否骗过人」，是对抗游戏而非文风质量评测 |
| Inverse Turing Bench | 给成对对话记录，判断哪段是纯人类对话。GPTZero 89.41%、Claude Opus-4.6 77.92%、GPT-5.5 75.94% | 测模型的分辨能力，方向相反 |

### SlopCodeBench：代码领域的 slop

威斯康星大学麦迪逊分校与 MIT 团队（arXiv:2603.24755）。评测编程智能体在迭代开发中的代码质量退化，指标为冗余度与结构侵蚀。与文本「说人话」无关，但「和人类基线对比 + 可量化退化指标」的方法论可参考。

### Antislop：slop 检测与消除框架

arXiv 论文（cs.LG / cs.CL），三件套：推理时用回溯抑制套话的 Antislop Sampler、对照人类基线给模型做 slop 画像并生成训练数据的流水线、在 token 级别调整 logits 的微调方法 FTPO。属于消除 slop 的训练侧方案，其「模型 slop 画像对照人类基线」的思路可用于评测。

## 中文现状：只有改写工具，没有模型评测

中文「去 AI 味」项目均为 Agent Skill 或提示词规则，评测对象是工具自身的改写效果，不是大模型。

| 项目 | 内容 | 自带评测 |
| --- | --- | --- |
| 说人话（MrGeDiao/shuorenhua） | 中文优先改写 skill：210+ 中文套话短语、96 条英文短语、19 种结构反模式、4×3 场景分档、误杀防护、标注模式 | 42 条用例（24 条该改 + 18 条不该动），测改写命中率与误杀率 |
| heige-shuorenhua | 基于歸藏 Humanizer-zh 改造，专抓破折号、「不是什么而是什么」、「首先其次」类套话，带检查脚本 | 无模型排名 |
| no-ai-slop（petergyang） | 20 多类 AI 写作模式清理，英文为主，两个月 8.2K star | 无模型排名 |
| stop-slop（hardikpandya） | AI 写作禁用词与禁用结构清单，输出前自查 | 无模型排名 |

值得注意的观察：中文 AI 味套话在快速换代。「赋能」「闭环」是旧一代痕迹，新一代模型出现「工程师表演腔」——把「兜底」「压实」「收敛」「收束」等 incident report 术语塞进日常对话。任何静态黑名单都会过时。

## 自建中文 Benchmark 的机会与设计要点

空白点：中文「说人话度」模型排行榜目前没有现成项目。英文 SlopBench / Slop Index 的任务设计与评分方法可平移，中文语料与反模式清单是核心壁垒。

设计建议：

1. **优先做反方向（AI 味指数）**。机械指标客观、可复现、不依赖裁判模型：套话词频、「不是 X 而是 Y」句式密度、破折号密度、铺垫型开场比例、超出任务字数带的程度、跨样本开头重复率。SlopBench 的四个表面行为和 Slop Index 的四维度可直接借鉴。
2. **任务设计带约束**。每个任务规定字数带与场景（日常问答、工作沟通、技术解释、社交文案），多采样测开头与结构的重复率，避免单次采样的偶然性。
3. **建中文人类基线语料**。对标 Slop Index 的 pre-AI 语料思路，收集 ChatGPT 之前的中文博客、论坛帖、邮件、聊天记录，按领域分层。
4. **反模式清单分场景、可维护**。技术文档里的「收敛」「兜底」是正当术语，评分需区分场景防止误杀。清单需要随模型换代持续更新，建议版本化管理。
5. **可选加入人类偏好成分**。参考 Slop Index 的 40% 人类偏好 + 60% 机械成分配比，用盲测 pairwise 比较校准机械分与真人感受的一致性。
6. **与 HeartBench 划清边界**。明确评测目标是语言风格自然度，不测情感理解与共情，避免定位重叠。

## 参考资料

- [SlopBench 论文（Semantic Scholar）](https://www.semanticscholar.org/paper/SlopBench:-How-Well-Can-We-Rank-Language-Models-by-Roongta-Gaddipati/bd2244670b1e51e8d46ddf3a55efb465cb5fe477)
- [The Slop Index（GitHub）](https://github.com/hgaddipati1118/slop-index)
- [HeartBench 评测体系建设实践（阿里云开发者社区）](https://developer.aliyun.com/article/1717871)
- [HeartBench（GitHub）](https://github.com/inclusionAI/HeartBench)
- [TuringBench（arXiv:2109.13296）](https://arxiv.org/pdf/2109.13296)
- [反向和错位图灵测试（澎湃）](http://m.thepaper.cn/newsDetail_forward_28679990)
- [SlopCodeBench（arXiv:2603.24755 解读）](https://www.alphaxiv.org/zh/overview/2603.24755v1)
- [说人话 skill（GitHub）](https://github.com/MrGeDiao/shuorenhua)
- [no-ai-slop 介绍](https://www.jamecling.com/archives/5263)
- [awesome-humanizer-skills（GitHub）](https://github.com/zhuyansen/awesome-humanizer-skills)