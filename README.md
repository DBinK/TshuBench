# TshuBench · 天书榜

中文工程回答的文风观察榜。基于本机 1.5 万余条真实工程对话提炼特征，结合 36 道日常工程高频题多模型采样验证，聚焦六类可观测文风病灶，分数越高越像“天书”。后训练过对齐对表达习惯的因果归因列为待验证机制假设，当前榜单直接度量给定调用条件下的表层表达风格特征。

- **零裁判模型**：不使用 LLM-as-a-Judge，基于 `markdown-it-py` AST 与 `jieba` 词法完成 100% 确定性计算
- **确定性复现**：同一份回复永远得到同一个分数，结合 1000 次 Bootstrap 95% 置信区间与观察梯队抑制小样本波动
- **完整调用溯源**：每条样本记录 `reasoning_effort`、`temperature`、`max_tokens`、`suite_hash`、`sampled_at` 与实际返回路由 `response_model`

完整文档见 [docs/README.md](docs/README.md)。

## 快速开始

```bash
uv sync
uv run python tasks/build_suite.py          # 生成 36 题日常工程题库
uv run pytest -q                            # 指标与回归单测

uv run python -m harness.cli run --demo --out-prefix runs/demo
uv run python -m harness.cli score runs/demo_tidy.jsonl runs/demo_slop.jsonl
```

跑完看 [reports/report.md](reports/report.md)，或者直接打开 [web/index.html](web/index.html)。demo 数据是合成样本，只用于验证流水线，不代表任何真实模型。

## 接真实模型

任何 OpenAI 兼容接口都能接：

```bash
export OPENAI_API_KEY=sk-xxx
uv run python -m harness.cli suite \
  -m deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --out-dir runs
```

批量跑多个模型并用 `模型:思考档位` 指定最低思考级别：

```bash
uv run python -m harness.cli run \
  -m codex/gpt-6-luna:low -m auto/gemini-4-argon:medium \
  --base-url http://127.0.0.1:3425/v1 \
  --out-dir runs/real --concurrency 6

uv run python -m harness.cli score runs/real/*.jsonl --out reports
```

`reasoning_content` 与正文里的 `<think>` 块都会在入库前剥离，思考字符数记入 `think_chars`。接口返回空正文或疑似预算截断时会自动加倍 `max_tokens` 重试，遭遇 HTTP 429 时按 `Retry-After` 或递增退避重试。

## 六大核心指标

| 代号 | 名称 | 权重 | 对应的可观测文风病灶与惩罚对象 |
| --- | --- | ---: | --- |
| M1 CVD | 首句决断延迟 | 16% | 开场太极铺垫：前 20 字内未命中结论锚点或强表态短语，支持引号归一化与前 80 字实体词兜底 |
| M2 HED | 防御性叠甲 | 22% | 免责叠甲护体：不一定、通常、更准确地说、取决于、不代表等限定词密度（不惩罚「不建议」等纯语气词），辅以伪前提直击判定 |
| M3 ENU | 报菜名堆砌 | 18% | 全面性过载与电报压缩：按篇幅归一化的顿号斜杠堆砌与连发密度、行内代码与分号高密压缩（剥离路径与比值） |
| M4 CCI | 句法负荷 | 22% | 思维链外溢与长链嵌套：剥离括号批注后的纯汉字超长分句、条件嵌套、分号链与正文外显推演痕迹 |
| M5 PNI | 欧化名物化 | 10% | 悬空裸代号与公文腔：无括号内联复述的 `R1`/`D5` 等裸代号指代，叠加 `jieba` 识别的弱动词名物化与欧化被动搭配（豁免排障单字「被」） |
| M6 VTR | 语言抽动 | 12% | 格式强迫与客服套话：黑话（自动豁免引用题干词）、伪对比、破折号、客服收尾、AST 群聊表格墙与过密大纲 |

总分 **TshuScore** 0–100，分数越低越少见这六类文风病灶，0 分表示未检出预设惩罚模式（不代表事实正确或回答完整）。按点估计划入三档娱乐观察梯队：**T1 · 人话级（< 11）**、**T2 · 黑话级（11–15）**、**T3 · 天书级（≥ 15）**，CI 明显重叠的模型之间不应解读为稳定名次差异。

**评分边界**：只读取最终正文。接口的 `reasoning_content` 与正文里的思考标签在入库前从正文剥离，思考文本写入数据集的 `thinking` 字段留档，但不参与任何指标计算，也不占用正文预算。`think_chars` 记录长度供诊断。

## 题库 Schema

`tasks/eval_suite_v2.jsonl` 每行一题，共 36 道日常工程高频题（代码排查、线上故障、技术澄清、发布交付各 9 题）：

```json
{
  "id": "code_t03",
  "scenario": "code_review_bug_triage",
  "register": "chat",
  "trap_type": "flawed_premise",
  "max_chars": 120,
  "prompt": "...",
  "verdict_anchors": ["GIL", "ProcessPoolExecutor", "多进程"],
  "actionable_entities": ["GIL", "multiprocessing", "cpu_count"],
  "fence_sitting_markers": ["各有优劣", "取决于"]
}
```

`trap_type` 均分取 `explicit_bound`、`flawed_premise`、`forced_choice`（各 12 题），配合 `verdict_anchors`、`refutation_anchors` 与 `actionable_entities` 触发不同的结论位置与伪前提直击检测。`register` 为 `chat` 时，Markdown 表格墙（`|---|`）、分割线（`---`）与多级标题（`#`）会触发 M6 跨语域惩罚，普通加粗与列表豁免为视觉扫描锚点。

## 目录

```text
tasks/eval_suite_v2.jsonl   36 题日常工程高频题集（suite_hash=02c198fe8e74）
tasks/build_suite.py        题库生成器
harness/cli.py              Typer 统一 CLI（run / score / suite）
harness/models.py           Pydantic v2 数据契约与梯队划分
harness/text.py             markdown-it-py AST 与 jieba 词法工具
harness/metrics.py          M1-M6 六大确定性指标算子
harness/score.py            聚合、Bootstrap CI、战报与静态榜单
harness/run.py              OpenAI 兼容接口采样、元数据记录与 demo 合成数据
tests/test_metrics.py       指标与误杀回归单测
web/index.html              排行榜页面
docs/                       技术方案与实证校准报告
```

## 评测边界

- 现存榜单覆盖 **17 个主流模型路由 × 36 道日常工程题（共 612 条有效非空样本）**，经奇偶分半验证总分 Spearman-Brown 信度达 **0.917**（$r = 0.847$）
- 榜单反映特定日期（2026-11-15）、特定路由配置（`temperature=0.3`、最低支持 `reasoning_effort`）下的输出风貌，不宜直接跨系统提示词或跨基线能力泛化
- 人类专家基线分位数表与未参与调参的独立盲测集正在规划建设中，当前 0 分仅代表未触发六项规则惩罚
- 榜单为静态汇总页，支持完整 36 题 JSONL 原始回复与调用元数据追溯复核
