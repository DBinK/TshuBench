# TshuBench · 天书榜

量化大模型后训练（SFT / RLHF / DPO / CoT）导致的中文可读性退化。跑分越高的模型，输出越像天书，这个榜单把它算成一个分数。

- **零裁判模型**：不使用 LLM-as-a-Judge，全部指标由纯 Python 标准库计算
- **确定性复现**：同一份回复永远得到同一个分数
- **题库锚点**：把「结论在哪」「有没有顺从错误前提」这类语义判断，降维成题库里预埋的字符串匹配

完整文档见 [docs/README.md](docs/README.md)。

## 快速开始

```bash
uv sync
uv run python tasks/build_suite.py          # 生成 24 题种子题库
uv run pytest -q                            # 八指标正反例单测

uv run python -m harness.run --demo --out-prefix runs/demo
uv run python -m harness.score \
  --runs runs/demo_tidy.jsonl runs/demo_slop.jsonl \
  --out reports
```

跑完看 `reports/report.md`，或者直接打开 `web/index.html`。demo 数据是合成样本，只用于验证流水线，不代表任何真实模型。

## 接真实模型

任何 OpenAI 兼容接口都能接：

```bash
export OPENAI_API_KEY=sk-xxx
uv run python -m harness.run \
  --model deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --out runs/deepseek-chat.jsonl

uv run python -m harness.score --runs runs/deepseek-chat.jsonl --out reports
```

批量跑多个模型，可用 `模型:思考档位` 指定每个模型的最低思考级别：

```bash
uv run python -m harness.run \
  --models auto/gpt-6-luna:low auto/gemini-4-argon:medium \
  --base-url http://127.0.0.1:3425/v1 \
  --out-dir runs/real --concurrency 6

uv run python -m harness.score --runs runs/real/*.jsonl --out reports
```

`reasoning_content` 与正文里的 `<think>` 块都会在入库前剥离，思考字符数记入 `think_chars`。接口返回空正文时会自动加倍 `max_tokens` 重试，避免思考吃光预算。

## 八项指标

| 代号 | 名称 | 惩罚对象 |
| --- | --- | --- |
| M1 ADR | 首结论延迟率 | 结论藏在后半段 |
| M2 EID | 有效信息密度 | 长文里只有一两个干货 |
| M3 OGR | 超额范围输出 | 越字数上限、注水段、升华尾巴 |
| M4 SSF | 奉承与骑墙 | 开场吹捧、顺从错误前提、二选一和稀泥 |
| M5 CCI | 浅层句法负荷 | 憋气长句、多重条件嵌套、的字链 |
| M6 PNI | 欧化名物化 | 进行+动名词、被动句、显性代词泛滥 |
| M7 OSC | 推理空转 | 车轱辘复读、自我怀疑兜圈 |
| M8 VTR | 语言抽动 | 黑话、伪对比、群聊滥用 Markdown、节奏呆板 |

权重：M1–M4 各 15%，M5–M8 各 10%。总分 **TshuScore** 0–100，越低越说人话。

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

`trap_type` 取 `explicit_bound`、`flawed_premise`、`forced_choice`，分别触发不同的 M4 检测分支。`register` 为 `chat` 时，Markdown 标题、加粗、列表会触发 M8 跨语域惩罚。

## 目录

```
tasks/eval_suite_v2.jsonl   24 题种子题库
tasks/build_suite.py        题库生成器
harness/metrics.py          M1-M8 指标算子
harness/score.py            聚合、战报、静态榜单
harness/run.py              OpenAI 兼容接口采样 + demo 合成数据
tests/test_metrics.py       八指标正反例
web/index.html              排行榜页面
docs/                       立项规范、技术方案、方法论、调研与题库来源
```

## MVP 边界

- 题库 24 题，规划规模 120 题
- 权重和对数饱和参数是初版标定，等真实模型数据回来再调
- 预计算的人类基线分位数表尚未接入，M1/M5 目前用固定阈值
- 榜单是静态页，没有自动更新和多题明细下钻
