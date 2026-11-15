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

## 六大核心指标（V3）

V3 按**体感区分度与实际方差贡献排序**，采用**「正文阅读负担 76%（`CCI 24%` + `ENU 20%` + `FMT 16%` + `TIC 16%`）+ 开口痛快度 24%（`CVD 12%` + `HED 12%`）」**拆分权重（移除了跨题信度仅 `0.458` 且语义混杂的旧 `PNI` 维度，并将原 `VTR` 拆分为纯粹的 `FMT` 排版过载与 `TIC` 套话模板两项）：

| 代号 | 名称 | 权重 | 对应的可观测文风病灶与惩罚对象 |
| --- | --- | ---: | --- |
| M1 CCI | 长链绕口与推演 | 24% | 思维链外溢与长链嵌套：剥离括号批注后的纯汉字超长分句、条件嵌套、分号链、高频「的」与正文推演痕迹 |
| M2 FMT | 排版强迫与支线 | 16% | 跨语域格式膨胀：AST 识别 `chat` 语域的表格墙、过密标题、分割线与超额代码块，以及独立「常见坑/注意事项」支线小节 |
| M3 ENU | 报菜名与堆砌 | 20% | 全面性过载与电报压缩：按篇幅归一化的顿号斜杠堆砌与连发密度、行内代码与分号高密压缩（剥离路径与比值） |
| M4 TIC | 套话抽动与模板 | 16% | 固定套话与模板收敛：黑话（自动豁免题干引词）、伪对比、破折号、客服收尾、句长均一性（CV）与跨题首句熵坍缩 |
| M5 CVD | 首句决断延迟 | 12% | 开场太极铺垫：剥离「先说结论：」等空泛元标签后检查前 70 字结论锚点或直判句式，配 30 字缓冲与前 120 字实体兜底 |
| M6 HED | 防御性叠甲 | 12% | 免责叠甲护体：不一定、通常、更准确地说、取决于、不代表等限定词密度（不惩罚「不建议」等纯语气词），辅以伪前提直击判定 |

总分 **TshuScore** 0–100，分数越低越少见这六类文风病灶，0 分表示未检出预设惩罚模式（不代表事实正确或回答完整）。按点估计划入三档娱乐观察梯队：**T1 · 人话级（< 10）**、**T2 · 黑话级（10–14）**、**T3 · 天书级（≥ 14）**，CI 明显重叠的模型之间不应解读为稳定名次差异。

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
harness/audit.py            零人工成本统计体检（基线对照、PCA 与配对置换检验）
harness/run.py              OpenAI 兼容接口采样、元数据记录与 demo 合成数据
tests/test_metrics.py       指标与误杀回归单测
tests/test_audit.py         统计体检单测
web/index.html              排行榜页面
docs/                       技术方案与实证校准报告
```

## 评测边界与方法论局限

- **定位与构念效度边界**：TshuBench 是**作者定义的启发式中文工程文风指数**，当前尚未经独立盲评读者实验校准，不能直接解释为经过验证的“认知阅读负担量表”，也不支持“20 分负担是 10 分两倍”的等比尺度推论。`0.968`（$r = 0.937$）反映的是跨 36 题的内部一致性信度，而非阅读效度，当前 0 分仅代表未触发六项预设规则。
- **特征检测与经验权重边界**：语言特征可测不等于出现就必须扣分（例如必要限定词或对照表格）。当前 24/16/20/16/12/12 权重与饱和阈值为经验设定，且部分特征跨项重叠（如分号同时影响 ENU 与 CCI）。
- **零人工成本统计体检结果（可运行 `uv run python -m harness.cli audit runs/real/*.jsonl` 复现）**：V3 总分与字数、平均句长的线性相关性较低（单题级汉字数 $r = +0.1000$，模型级总字数 $r = -0.0975$），说明分数并非简单的线性字数惩罚（但不排除非线性长度效应，且仍待人工评价对比），六项指标分半信度均达 `0.761–0.982`。按探索性 36 题配对置换检验（存在同批数据选出榜首的选择效应），Holm 校正后相邻名次无任何一对显著（`0/16`），相对第 1 名从第 4 名起达到 `p_holm < 0.05`（`p_holm = 0.0129`），接近名次间的分差不宜解读为稳定优劣。
- **采样与单轮静态边界**：榜单仅反映特定日期（2026-11-15）与特定路由配置（`temperature=0.3`、最低支持 `reasoning_effort`，含两项已标注混合路由）下的单轮静态输出风貌，未覆盖多轮追问下的条件谄媚，“后训练过对齐归因”仍属待验证假说。榜单支持完整 36 题 JSONL 原始回复与调用元数据追溯复核。
