# TshuBench · 天书榜

量化大模型后训练（SFT / RLHF / DPO / CoT）引发的中文可读性退化。基于本机 1.5 万余条真实工程对话与多模型基准采样实证，榜单聚焦六大核心过对齐病灶，分数越高越像“天书”。

- **零裁判模型**：不使用 LLM-as-a-Judge，基于 `markdown-it-py` AST 与 `jieba` 词法完成 100% 确定性计算
- **确定性复现**：同一份回复永远得到同一个分数，结合 Bootstrap 95% 置信区间与梯队划分抑制小样本噪声
- **题库锚点与六大病灶对齐**：把「结论在哪」「是否堆叠免责」「是否正文残留思维链与悬空裸代号」沉淀为确定性指标算子

完整文档见 [docs/README.md](docs/README.md)。

## 快速开始

```bash
uv sync
uv run python tasks/build_suite.py          # 生成 24 题探索性校准集
uv run pytest -q                            # 指标与回归单测

uv run python -m harness.cli run --demo --out-prefix runs/demo
uv run python -m harness.cli score runs/demo_tidy.jsonl runs/demo_slop.jsonl
```

跑完看 `reports/report.md`，或者直接打开 `web/index.html`。demo 数据是合成样本，只用于验证流水线，不代表任何真实模型。原有 `python -m harness.run` 与 `python -m harness.score` 入口继续兼容。

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
  -m auto/gpt-6-luna:low -m auto/gemini-4-argon:medium \
  --base-url http://127.0.0.1:3425/v1 \
  --out-dir runs/real --concurrency 6

uv run python -m harness.cli score runs/real/*.jsonl --out reports
```

`reasoning_content` 与正文里的 `<think>` 块都会在入库前剥离，思考字符数记入 `think_chars`。接口返回空正文时会自动加倍 `max_tokens` 重试，避免思考吃光预算。

## 六大核心指标

| 代号 | 名称 | 权重 | 对应的后训练过对齐病灶与惩罚对象 |
| --- | --- | ---: | --- |
| M1 CVD | 首句决断延迟 | 16% | 开场太极铺垫：结论藏在背景复述之后，不敢开口表态 |
| M2 HED | 防御性叠甲 | 22% | 免责叠甲护体：不一定、通常、更准确地说、取决于、不代表（豁免「非常/都不建议」等强否定），伪前提不敢直斥 |
| M3 ENU | 报菜名堆砌 | 18% | 全面性过载与电报压缩：按篇幅归一化的顿号斜杠堆砌与连发密度、行内代码与分号高密压缩（剥离路径与比值） |
| M4 CCI | 句法负荷 | 22% | 思维链外溢与长链嵌套：剥离括号批注后的纯汉字超长分句、条件嵌套、分号链与正文外显推演痕迹 |
| M5 PNI | 欧化名物化 | 10% | 悬空裸代号与公文腔：无括号内联复述的 `R1`/`D5` 等裸代号指代，叠加 `jieba` 识别的弱动词名物化与欧化被动搭配（豁免排障单字「被」） |
| M6 VTR | 语言抽动 | 12% | 格式强迫与客服套话：黑话（自动豁免引用题干词）、伪对比、破折号、客服收尾、AST 群聊表格墙与过密大纲 |

总分 **TshuScore** 0–100，越低越说人话，按分值划入三档梯队：**T1 · 人话级（< 11）**、**T2 · 黑话级（11–15）**、**T3 · 天书级（≥ 15）**。口径只测表达方式，不判答案对错。

**评分边界**：只读取最终正文。接口的 `reasoning_content` 与正文里的 ` thinking` 块在入库前从正文剥离，思考文本写入数据集的 `thinking` 字段留档，但不参与任何指标计算，也不占用正文预算。`think_chars` 记录长度供诊断。

## 题库 Schema

`tasks/eval_suite_v2.jsonl` 每行一题：

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

`trap_type` 取 `explicit_bound`、`flawed_premise`、`forced_choice`，分别配合 `verdict_anchors` 触发不同的结论位置与伪前提直击检测。`register` 为 `chat` 时，Markdown 表格墙（`|---|`）、分割线（`---`）与多级标题（`#`）会触发 M6 跨语域惩罚（普通加粗与列表豁免为视觉扫描锚点）。

## 目录

```
tasks/eval_suite_v2.jsonl   24 题探索性校准集
tasks/build_suite.py        题库生成器
harness/cli.py              Typer 统一 CLI（run / score / suite）
harness/models.py           Pydantic v2 数据契约与梯队划分
harness/text.py             markdown-it-py AST 与 jieba 词法工具
harness/metrics.py          M1-M6 六大确定性指标算子
harness/score.py            聚合、Bootstrap CI、战报与静态榜单
harness/run.py              OpenAI 兼容接口采样与 demo 合成数据
tests/test_metrics.py       指标与误杀回归单测
web/index.html              排行榜页面
docs/                       技术方案与实证校准报告
```

## MVP 边界

- 当前 `eval_suite_v2.jsonl` 定位为 **24 题探索性校准集（Calibration Set）**，规划扩展至 120 题独立盲测集，小样本下 95% 置信区间重叠的模型不宜强行拉开绝对名次
- 权重、饱和阈值与梯队分界基于首轮 8 模型实证分布标定，后续将随独立测试集扩充持续校准
- 预计算的人类专家基线分位数表尚未接入，目前仍采用固定阈值映射 0–100 分
- 榜单是静态页，没有自动更新和多题明细下钻
