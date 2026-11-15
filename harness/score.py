"""TshuBench 打分引擎：读取模型回复，输出分数、战报与静态榜单。

用法：
    uv run python -m harness.score --tasks tasks/eval_suite_v2.jsonl \
        --runs runs/demo_tidy.jsonl runs/demo_slop.jsonl --out reports
"""

import argparse
import html
import json
import random
from datetime import datetime
from pathlib import Path
from statistics import mean

from loguru import logger
from rich import print as rprint

from harness.metrics import evaluate, opening_entropy
from harness.models import WEIGHTS, ModelReport, Sample, ScoredSample, Task, classify_tier

METRIC_LABELS = {
    "cci": "CCI 长链绕口与推演",
    "fmt": "FMT 排版强迫与支线",
    "enu": "ENU 报菜名与堆砌",
    "tic": "TIC 套话抽动与模板",
    "cvd": "CVD 首句决断延迟",
    "hed": "HED 防御性叠甲",
}

DEFAULT_EFFORT_BY_MODEL = {
    "auto/gemini-3.8-flash": "medium",
    "auto/gemini-4-argon": "medium",
    "auto/gemini-4-hydrogen": "low",
    "auto/deepseek-v4.1-flash": "low",
    "auto/claude-haiku-5-5": "low",
    "codex/gpt-6-luna": "low",
    "codex/gpt-6.1-sol": "low",
    "codex/gpt-6.1-terra": "low",
}


def load_tasks(path: str | Path) -> dict[str, Task]:
    path = Path(path)
    tasks: dict[str, Task] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = Task.model_validate_json(line)
        tasks[task.id] = task
    return tasks


def load_samples(paths: list[str | Path]) -> dict[str, list[Sample]]:
    grouped: dict[str, list[Sample]] = {}
    for path in map(Path, paths):
        if not path.exists():
            logger.warning(f"跳过不存在的文件：{path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = Sample.model_validate_json(line)
            grouped.setdefault(sample.model, []).append(sample)
    return grouped


def compute_bootstrap_ci(
    scores: list[float], n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """计算指标或总分的非参数 Bootstrap 置信区间 (默认 95% CI)。"""
    if not scores:
        return 0.0, 0.0
    if len(scores) == 1:
        return round(scores[0], 2), round(scores[0], 2)

    rng = random.Random(seed)
    n = len(scores)
    means: list[float] = []
    for _ in range(n_bootstrap):
        resample = [scores[rng.randrange(n)] for _ in range(n)]
        means.append(mean(resample))

    means.sort()
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1.0 - alpha / 2))
    lower = means[max(0, lower_idx)]
    upper = means[min(n_bootstrap - 1, upper_idx)]
    return round(lower, 2), round(upper, 2)


def _build_route_note(known: list[Sample], total_tasks: int) -> str:
    notes: list[str] = []
    if total_tasks > 1 and len(known) < total_tasks:
        notes.append(f"{len(known)}/{total_tasks}题")
    route_counts: dict[str, int] = {}
    for sample in known:
        rm = (sample.response_model or "").strip()
        if rm:
            route_counts[rm] = route_counts.get(rm, 0) + 1
    if len(route_counts) > 1:
        counts = sorted(route_counts.values(), reverse=True)
        notes.append("混合路由(" + "+".join(str(c) for c in counts) + ")")
    return " · ".join(notes)


def score_model(
    model: str, samples: list[Sample], tasks: dict[str, Task], n_bootstrap: int = 1000
) -> ModelReport:
    effort = next((s.reasoning_effort for s in samples if s.reasoning_effort), "") or DEFAULT_EFFORT_BY_MODEL.get(
        model, "默认"
    )
    known = [s for s in samples if s.task_id in tasks]
    if len(known) < len(samples):
        logger.warning(f"{model} 有 {len(samples) - len(known)} 条样本题目不在题库中，已跳过")
    route_note = _build_route_note(known, len(tasks))

    entropy = opening_entropy([s.response for s in known])
    opening_penalty = max(0.0, 1.0 - entropy)

    scored: list[ScoredSample] = []
    for sample in known:
        scored.append(evaluate(tasks[sample.task_id], sample, opening_penalty))

    if not scored:
        return ModelReport(
            model=model,
            n_samples=0,
            tshu_score=0.0,
            metrics={k: 0.0 for k in WEIGHTS},
            opening_entropy=entropy,
            tier=classify_tier(0.0),
            reasoning_effort=effort,
            route_note=route_note,
        )

    averages = {
        key: round(mean(getattr(s.metrics, key) for s in scored), 2) for key in WEIGHTS
    }
    open_s = round(averages["cvd"] * WEIGHTS["cvd"] + averages["hed"] * WEIGHTS["hed"], 2)
    body_s = round(
        averages["cci"] * WEIGHTS["cci"]
        + averages["fmt"] * WEIGHTS["fmt"]
        + averages["enu"] * WEIGHTS["enu"]
        + averages["tic"] * WEIGHTS["tic"],
        2,
    )
    ts = round(sum(averages[k] * w for k, w in WEIGHTS.items()), 2)
    worst = sorted(scored, key=lambda s: s.metrics.tshu_score, reverse=True)[:5]

    sample_ts = [s.metrics.tshu_score for s in scored]
    ci_lower, ci_upper = compute_bootstrap_ci(sample_ts, n_bootstrap=n_bootstrap)

    return ModelReport(
        model=model,
        n_samples=len(scored),
        tshu_score=ts,
        metrics=averages,
        opening_entropy=round(entropy, 4),
        worst=worst,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        open_score=open_s,
        body_score=body_s,
        tier=classify_tier(ts),
        reasoning_effort=effort,
        route_note=route_note,
    )


def primary_profile(metrics: dict[str, float], top_k: int = 2) -> str:
    """提取模型得分最高的主导过对齐病灶画像（Scorecard 视角）。"""
    active = [(k, v) for k, v in metrics.items() if v > 0.0]
    if not active:
        return "全维低病灶"
    active.sort(key=lambda item: item[1], reverse=True)
    return " · ".join(f"{METRIC_LABELS[k]}({v:.1f})" for k, v in active[:top_k])


def render_markdown(reports: list[ModelReport]) -> str:
    ranked = sorted(reports, key=lambda r: r.tshu_score)
    header = (
        "| 排名 | 模型 | 思考档位 | 梯队 | TshuScore (95% CI) | 开口扣分 | 正文扣分 | 主导病灶画像 | "
        + " | ".join(METRIC_LABELS.values())
        + " |"
    )
    sep = "| ---: | --- | :---: | --- | ---: | ---: | ---: | --- | " + " | ".join("---:" for _ in METRIC_LABELS) + " |"
    rows = [
        "| {} | {} | {} | {} | **{:.2f}** [{:.2f}, {:.2f}] | {:.2f} | {:.2f} | {} | {} |".format(
            i,
            f"{r.model} *({r.route_note})*" if r.route_note else r.model,
            r.reasoning_effort or "默认",
            r.tier or classify_tier(r.tshu_score),
            r.tshu_score,
            r.ci_lower,
            r.ci_upper,
            r.open_score,
            r.body_score,
            primary_profile(r.metrics),
            " | ".join(f"{r.metrics[k]:.1f}" for k in METRIC_LABELS),
        )
        for i, r in enumerate(ranked, start=1)
    ]
    lines = [
        "# TshuBench · 天书榜 诊断战报",
        "",
        f"生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "分数越低越少见这六类文风病灶，0 分仅表示未检出预设惩罚模式（不代表事实正确或回答完整）。本榜为启发式文风观察指数，尚未经独立盲评校准阅读效度；V3 权重按「正文阅读负担 76%（CCI 24% + ENU 20% + FMT 16% + TIC 16%）+ 开口痛快度 24%（CVD 12% + HED 12%）」拆分，梯队按点估计（<10 / 10–14 / ≥14）划定，[下限, 上限] 为 1000 次 Bootstrap 95% 置信区间；相邻名次差异请结合配对置换检验判断。",
        "",
        header,
        sep,
        *rows,
    ]
    for report in ranked:
        lines += [
            "",
            f"## {report.model}",
            "",
            f"- 样本数：{report.n_samples}" + (f"（{report.route_note}）" if report.route_note else ""),
            f"- 思考档位：{report.reasoning_effort or '默认'}",
            f"- 梯队：{report.tier or classify_tier(report.tshu_score)}",
            f"- TshuScore：{report.tshu_score:.2f} (95% CI: [{report.ci_lower:.2f}, {report.ci_upper:.2f}])",
            f"- 开口扣分：{report.open_score:.2f} ｜ 正文扣分：{report.body_score:.2f}",
            f"- 首句熵：{report.opening_entropy}",
        ]
        if report.worst:
            lines += ["", "最像天书的五条：", ""]
            for item in report.worst:
                preview = item.sample.response.replace("\n", " ")[:80]
                lines.append(f"- `{item.task.id}` {item.metrics.tshu_score:.1f} 分：{preview}")
    return "\n".join(lines) + "\n"


MODEL_BRANDS: dict[str, tuple[str, str]] = {
    "auto/gemini-4-argon": ("Gemini 4 Argon", "Google"),
    "auto/gemini-4-hydrogen": ("Gemini 4 Hydrogen", "Google"),
    "auto/gemini-3.8-flash": ("Gemini 3.8 Flash", "Google"),
    "opencode-go/muse-spark-1.3-contributor": ("Muse Spark 1.3 Contributor", "OpenCode"),
    "opencode-zen/claude-sonnet-5-5": ("Claude 3.7 Sonnet", "Anthropic"),
    "auto/claude-haiku-5-5": ("Claude 3.5 Haiku", "Anthropic"),
    "opencode-zen/claude-opus-5-5": ("Claude 3.7 Opus", "Anthropic"),
    "codex/gpt-6.1-terra": ("GPT-6.1 Terra", "OpenAI"),
    "opencode-zen/glm-5.3": ("GLM-5.3", "Zhipu"),
    "opencode-go/glm-5.3-flash": ("GLM-5.3 Flash", "Zhipu"),
    "codex/gpt-6-luna": ("GPT-6 Luna", "OpenAI"),
    "opencode-go/kimi-k3.1-pro": ("Kimi K3.1 Pro", "Moonshot"),
    "codex/gpt-6.1-sol": ("GPT-6.1 Sol", "OpenAI"),
    "opencode-zen/qwen3.8-flash": ("Qwen 3.8 Flash", "Alibaba"),
    "opencode-zen/qwen3.8-max": ("Qwen 3.8 Max", "Alibaba"),
    "auto/deepseek-v4.1-flash": ("DeepSeek V4.1 Flash", "DeepSeek"),
    "auto/minimax-m3.1-flash": ("MiniMax M3.1 Flash", "MiniMax"),
}


def parse_model_display(model: str) -> tuple[str, str]:
    if model in MODEL_BRANDS:
        return MODEL_BRANDS[model]
    parts = model.split("/")
    name = parts[-1].replace("-", " ").title()
    for brand, provider in [
        ("gemini", "Google"),
        ("claude", "Anthropic"),
        ("gpt", "OpenAI"),
        ("qwen", "Alibaba"),
        ("deepseek", "DeepSeek"),
        ("glm", "Zhipu"),
        ("kimi", "Moonshot"),
        ("minimax", "MiniMax"),
    ]:
        if brand in model.lower():
            return (name, provider)
    return (name, parts[0] if len(parts) > 1 else "Other")


def render_html(reports: list[ModelReport]) -> str:
    ranked = sorted(reports, key=lambda r: r.tshu_score)
    model_entries = []
    static_rows = []

    for i, report in enumerate(ranked, start=1):
        display_name, provider = parse_model_display(report.model)
        tier_str = report.tier or classify_tier(report.tshu_score)
        tier_cls = "t1" if "T1" in tier_str else ("t2" if "T2" in tier_str else "t3")
        effort_text = html.escape(report.reasoning_effort or "默认")
        route_note_text = html.escape(report.route_note or "")

        worst_items = []
        if getattr(report, "worst", None):
            for item in report.worst[:2]:
                worst_items.append({
                    "task_id": getattr(item.task, "id", str(item.task)),
                    "score": round(float(item.metrics.tshu_score), 1),
                    "response": item.sample.response[:240] + ("..." if len(item.sample.response) > 240 else ""),
                })

        entry = {
            "rank": i,
            "model": report.model,
            "name": display_name,
            "provider": provider,
            "effort": effort_text,
            "tier": tier_str,
            "tier_cls": tier_cls,
            "tshu_score": round(report.tshu_score, 2),
            "ci_lower": round(report.ci_lower, 2),
            "ci_upper": round(report.ci_upper, 2),
            "route_note": route_note_text,
            "metrics": {k: round(report.metrics[k], 1) for k in METRIC_LABELS},
            "worst": worst_items,
        }
        model_entries.append(entry)

        # Static row generation for tests & non-JS fallback
        cells = "".join(f"<td class='px-3 py-3 text-right font-mono text-xs text-zinc-300'>{report.metrics[k]:.1f}</td>" for k in METRIC_LABELS)
        route_badge = f" <span class='text-[10px] text-zinc-400 font-mono'>({route_note_text})</span>" if route_note_text else ""
        static_rows.append(
            f"<tr class='model-row border-b border-zinc-800/80 hover:bg-zinc-900/60 transition-colors cursor-pointer' data-rank='{i}'>"
            f"<td class='px-3 py-3.5 text-center font-mono text-xs text-zinc-400'>{i}</td>"
            f"<td class='px-4 py-3.5'><div class='flex items-center gap-2'>"
            f"<span class='font-semibold text-sm text-zinc-100 tracking-tight'>{html.escape(display_name)}</span>"
            f"<span class='text-[11px] font-medium px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700/60'>{html.escape(provider)}</span>"
            f"</div><div class='text-[11px] font-mono text-zinc-400 mt-0.5'><code>{html.escape(report.model)}</code>{route_badge}</div></td>"
            f"<td class='px-3 py-3.5 text-center font-mono text-xs text-zinc-400 effort'>{effort_text}</td>"
            f"<td class='px-3 py-3.5 text-center'><span class='tier-pill {tier_cls} text-xs font-medium px-2 py-0.5 rounded-full'>{tier_str}</span></td>"
            f"<td class='px-4 py-3.5 text-right'><div class='inline-flex items-baseline gap-1.5'>"
            f"<span class='font-mono text-lg font-bold text-zinc-100'>{report.tshu_score:.2f}</span>"
            f"<span class='font-mono text-[11px] text-zinc-400'>[{report.ci_lower:.1f}, {report.ci_upper:.1f}]</span>"
            f"</div></td>"
            f"{cells}</tr>"
        )

    data_json = json.dumps(model_entries, ensure_ascii=False)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    metric_header_defs = [
        ("cci", "CCI", "长链推演"),
        ("fmt", "FMT", "排版膨胀"),
        ("enu", "ENU", "报菜名"),
        ("tic", "TIC", "套话模板"),
        ("cvd", "CVD", "决断推延"),
        ("hed", "HED", "防御叠甲"),
    ]
    head_cells = "".join(
        f"<th data-sort='{k}' class='py-3 px-3 text-right cursor-pointer hover:text-zinc-100 select-none transition-colors'>"
        f"<span class='block font-semibold tracking-wider text-xs text-zinc-300'>{code}</span>"
        f"<span class='block text-[11px] font-normal text-zinc-400 mt-0.5'>{desc}</span></th>"
        for k, code, desc in metric_header_defs
    )

    return f"""<!doctype html>
<html lang="zh-CN" class="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TshuBench · 天书榜 | 中文大模型表达负担评测</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {{
  darkMode: 'class',
  theme: {{
    extend: {{
      fontFamily: {{
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', '"PingFang SC"', '"Noto Sans CJK SC"', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'monospace'],
      }},
      colors: {{
        zinc: {{
          950: '#09090b',
          900: '#121215',
          850: '#18181b',
          800: '#27272a',
          700: '#3f3f46',
        }}
      }}
    }}
  }}
}}
</script>
<style>
body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Noto Sans CJK SC", sans-serif;
  background-color: #09090b;
  color: #f4f4f5;
  -webkit-font-smoothing: antialiased;
}}
.tier-pill.t1 {{
  color: #34d399;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.25);
}}
.tier-pill.t2 {{
  color: #fbbf24;
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.25);
}}
.tier-pill.t3 {{
  color: #f87171;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.25);
}}
.filter-btn.active {{
  background-color: #27272a;
  color: #ffffff;
  font-weight: 600;
}}
</style>
</head>
<body class="min-h-screen text-zinc-100 antialiased selection:bg-zinc-800 selection:text-white">

<div class="max-w-[1240px] mx-auto px-4 py-8 sm:px-6 sm:py-12">
  <!-- Header / Hero -->
  <header class="mb-8">
    <div class="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4 mb-4">
      <div>
        <h1 class="text-3xl sm:text-4xl font-bold tracking-tight text-white mb-2">TshuBench · 天书榜</h1>
        <p class="text-sm sm:text-base text-zinc-400">量化中文大模型工程回答中的“不说人话”与结构膨胀</p>
      </div>
      <div class="flex items-center gap-3">
        <a href="https://github.com/DBinK/TshuBench" target="_blank" rel="noopener" class="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-400 hover:text-white px-3 py-1.5 rounded-md bg-zinc-900 border border-zinc-800 transition-colors">
          <span>GitHub</span>
          <span class="text-zinc-400">↗</span>
        </a>
        <a href="https://github.com/DBinK/TshuBench/blob/main/docs/%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88.md" target="_blank" rel="noopener" class="inline-flex items-center gap-1.5 text-xs font-medium text-zinc-400 hover:text-white px-3 py-1.5 rounded-md bg-zinc-900 border border-zinc-800 transition-colors">
          <span>评测方案</span>
          <span class="text-zinc-400">↗</span>
        </a>
      </div>
    </div>

    <!-- Metric Badges -->
    <div class="flex flex-wrap gap-2 text-xs text-zinc-400 mb-4">
      <div class="px-2.5 py-1 rounded bg-zinc-900/80 border border-zinc-800">⚡ 跨题分半信度 <strong class="text-zinc-200">r_SB = 0.968</strong></div>
      <div class="px-2.5 py-1 rounded bg-zinc-900/80 border border-zinc-800">🎯 <strong class="text-zinc-200">17 款主流模型路由</strong></div>
      <div class="px-2.5 py-1 rounded bg-zinc-900/80 border border-zinc-800">🧪 <strong class="text-zinc-200">36 题真实工程场景</strong></div>
      <div class="px-2.5 py-1 rounded bg-zinc-900/80 border border-zinc-800">⚖️ <strong class="text-zinc-200">0 裁判模型 · 确定性 AST 语法树打分</strong></div>
    </div>

    <!-- Notice -->
    <div class="rounded-lg bg-zinc-900/60 border border-zinc-800 p-3.5 text-xs text-zinc-400 flex items-start gap-2.5">
      <span class="text-zinc-400 text-sm">💡</span>
      <div class="leading-relaxed">
        <strong class="text-zinc-300">评分导读</strong>：本榜采用反向扣分制（<span class="text-emerald-400 font-medium">↓ 分数越低越讲人话</span>）。0 分代表未触发预设六类文风病灶，不代表技术结论一定正确。点击任意模型行可展开病灶诊断与典型样本。
      </div>
    </div>
  </header>

  <!-- Controls / Filters -->
  <div class="bg-zinc-900/70 border border-zinc-800 rounded-lg p-3.5 mb-6 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
    <div class="flex-1 min-w-[220px]">
      <input type="text" id="search-input" placeholder="🔍 搜索模型名称、厂商、底层代号..." class="w-full bg-zinc-950 border border-zinc-800 rounded-md px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-400 focus:outline-none focus:border-zinc-600 transition-colors" />
    </div>

    <div class="flex flex-wrap items-center gap-2">
      <!-- Tier Filter -->
      <div class="inline-flex rounded-md bg-zinc-950 p-0.5 border border-zinc-800" id="tier-group">
        <button class="filter-btn active text-xs px-2.5 py-1 rounded text-zinc-400 hover:text-zinc-200 transition-colors" data-tier="all">全部</button>
        <button class="filter-btn text-xs px-2.5 py-1 rounded text-zinc-400 hover:text-zinc-200 transition-colors" data-tier="T1 · 人话级">🟢 T1 人话</button>
        <button class="filter-btn text-xs px-2.5 py-1 rounded text-zinc-400 hover:text-zinc-200 transition-colors" data-tier="T2 · 黑话级">🟡 T2 黑话</button>
        <button class="filter-btn text-xs px-2.5 py-1 rounded text-zinc-400 hover:text-zinc-200 transition-colors" data-tier="T3 · 天书级">🔴 T3 天书</button>
      </div>

      <!-- Provider Filter -->
      <div class="inline-flex rounded-md bg-zinc-950 p-0.5 border border-zinc-800" id="provider-group">
        <button class="filter-btn active text-xs px-2.5 py-1 rounded text-zinc-400 hover:text-zinc-200 transition-colors" data-provider="all">全部厂商</button>
        <button class="filter-btn text-xs px-2.5 py-1 rounded text-zinc-400 hover:text-zinc-200 transition-colors" data-provider="Google">Google</button>
        <button class="filter-btn text-xs px-2.5 py-1 rounded text-zinc-400 hover:text-zinc-200 transition-colors" data-provider="Anthropic">Anthropic</button>
        <button class="filter-btn text-xs px-2.5 py-1 rounded text-zinc-400 hover:text-zinc-200 transition-colors" data-provider="OpenAI">OpenAI</button>
        <button class="filter-btn text-xs px-2.5 py-1 rounded text-zinc-400 hover:text-zinc-200 transition-colors" data-provider="Domestic">国产主流</button>
      </div>
    </div>
  </div>

  <!-- Leaderboard Table -->
  <div class="border border-zinc-800 rounded-lg overflow-hidden bg-zinc-900/40">
    <div class="overflow-x-auto">
      <table class="w-full text-left border-collapse" id="leaderboard-table">
        <thead>
          <tr class="border-b border-zinc-800 bg-zinc-900/90 text-xs">
            <th data-sort="rank" class="py-3 px-3 text-center text-zinc-400 font-mono w-12 cursor-pointer hover:text-zinc-100 select-none">#</th>
            <th data-sort="name" class="py-3 px-4 text-zinc-300 font-semibold cursor-pointer hover:text-zinc-100 select-none">模型与厂商</th>
            <th data-sort="effort" class="py-3 px-3 text-center text-zinc-400 font-medium cursor-pointer hover:text-zinc-100 select-none">思考档位</th>
            <th data-sort="tier" class="py-3 px-3 text-center text-zinc-400 font-medium cursor-pointer hover:text-zinc-100 select-none">梯队</th>
            <th data-sort="tshu_score" class="py-3 px-4 text-right cursor-pointer hover:text-zinc-100 select-none sort-active">
              <span class="block font-semibold tracking-wider text-xs text-zinc-200">天书指数 ↓</span>
              <span class="block text-[11px] font-normal text-emerald-400 mt-0.5">越低越讲人话</span>
            </th>
            {head_cells}
          </tr>
        </thead>
        <tbody id="leaderboard-body" class="divide-y divide-zinc-800/60">
          {''.join(static_rows)}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Footer -->
  <footer class="mt-8 pt-4 border-t border-zinc-800/80 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-zinc-400">
    <div>基于 V3 架构 · 生成时间：{now_str} · 全量 612 条真实样本</div>
    <div>MIT 开源协议 · 确定性 AST 语法树与词法统计</div>
  </footer>
</div>

<script id="tshu-raw-data" type="application/json">
{data_json}
</script>

<script>
(function() {{
  const rawData = JSON.parse(document.getElementById('tshu-raw-data').textContent);
  let state = {{
    search: '',
    tier: 'all',
    provider: 'all',
    sortKey: 'tshu_score',
    sortAsc: true,
    expanded: new Set()
  }};

  const domesticProviders = new Set(['Alibaba', 'Zhipu', 'Moonshot', 'DeepSeek', 'MiniMax']);

  const tbody = document.getElementById('leaderboard-body');
  const searchInput = document.getElementById('search-input');
  const tierBtns = document.querySelectorAll('#tier-group .filter-btn');
  const providerBtns = document.querySelectorAll('#provider-group .filter-btn');
  const ths = document.querySelectorAll('#leaderboard-table th');

  function render() {{
    let filtered = rawData.filter(item => {{
      if (state.tier !== 'all' && item.tier !== state.tier) return false;
      if (state.provider === 'Domestic' && !domesticProviders.has(item.provider)) return false;
      if (state.provider !== 'all' && state.provider !== 'Domestic' && item.provider !== state.provider) return false;
      if (state.search) {{
        const q = state.search.toLowerCase();
        const match = item.name.toLowerCase().includes(q) ||
                      item.model.toLowerCase().includes(q) ||
                      item.provider.toLowerCase().includes(q);
        if (!match) return false;
      }}
      return true;
    }});

    filtered.sort((a, b) => {{
      let valA, valB;
      if (state.sortKey === 'rank') {{
        valA = a.rank; valB = b.rank;
      }} else if (state.sortKey === 'name') {{
        valA = a.name.localeCompare(b.name); valB = 0;
      }} else if (state.sortKey === 'effort') {{
        valA = a.effort.localeCompare(b.effort); valB = 0;
      }} else if (state.sortKey === 'tier') {{
        valA = a.tier.localeCompare(b.tier); valB = 0;
      }} else if (state.sortKey === 'tshu_score') {{
        valA = a.tshu_score; valB = b.tshu_score;
      }} else if (a.metrics && a.metrics[state.sortKey] !== undefined) {{
        valA = a.metrics[state.sortKey]; valB = b.metrics[state.sortKey];
      }} else {{
        valA = a.rank; valB = b.rank;
      }}
      return state.sortAsc ? (valA > valB ? 1 : -1) : (valA < valB ? 1 : -1);
    }});

    tbody.innerHTML = '';
    filtered.forEach(item => {{
      const tr = document.createElement('tr');
      tr.className = 'model-row border-b border-zinc-800/80 hover:bg-zinc-900/60 transition-colors cursor-pointer';
      const isExpanded = state.expanded.has(item.model);

      const routeBadge = item.route_note ? ` <span class="text-[10px] text-zinc-400 font-mono">(${{item.route_note}})</span>` : '';

      let detailedTds = '';
      for (const k of ['cci', 'fmt', 'enu', 'tic', 'cvd', 'hed']) {{
        const val = item.metrics[k] !== undefined ? item.metrics[k].toFixed(1) : '-';
        detailedTds += `<td class="px-3 py-3 text-right font-mono text-xs text-zinc-300">${{val}}</td>`;
      }}

      tr.innerHTML = `
        <td class="px-3 py-3.5 text-center font-mono text-xs text-zinc-400">${{item.rank}}</td>
        <td class="px-4 py-3.5">
          <div class="flex items-center gap-2">
            <span class="font-semibold text-sm text-zinc-100 tracking-tight">${{item.name}}</span>
            <span class="text-[11px] font-medium px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-300 border border-zinc-700/60">${{item.provider}}</span>
          </div>
          <div class="text-[11px] font-mono text-zinc-400 mt-0.5"><code>${{item.model}}</code>${{routeBadge}}</div>
        </td>
        <td class="px-3 py-3.5 text-center font-mono text-xs text-zinc-400 effort">${{item.effort}}</td>
        <td class="px-3 py-3.5 text-center"><span class="tier-pill ${{item.tier_cls}} text-xs font-medium px-2 py-0.5 rounded-full">${{item.tier}}</span></td>
        <td class="px-4 py-3.5 text-right">
          <div class="inline-flex items-baseline gap-1.5">
            <span class="font-mono text-lg font-bold text-zinc-100">${{item.tshu_score.toFixed(2)}}</span>
            <span class="font-mono text-[11px] text-zinc-400">[${{item.ci_lower.toFixed(1)}}, ${{item.ci_upper.toFixed(1)}}]</span>
          </div>
        </td>
        ${{detailedTds}}
      `;

      tr.addEventListener('click', () => {{
        if (state.expanded.has(item.model)) {{
          state.expanded.delete(item.model);
        }} else {{
          state.expanded.add(item.model);
        }}
        render();
      }});
      tbody.appendChild(tr);

      if (isExpanded) {{
        const drawerTr = document.createElement('tr');
        drawerTr.className = 'bg-zinc-950/80 border-b border-zinc-800';
        let cardsHtml = '';
        const metricLabels = {{
          cci: ['CCI 长链推演', '长复合句与改口推演痕迹 (24%)'],
          fmt: ['FMT 排版膨胀', 'Markdown 结构强迫与支线清单 (16%)'],
          enu: ['ENU 报菜名', '顿号斜杠与高密符号压缩 (20%)'],
          tic: ['TIC 套话模板', '套话抽动与公文式结构 (16%)'],
          cvd: ['CVD 决断推延', '首句打太极，结论后置 (12%)'],
          hed: ['HED 防御叠甲', '逢人便叠甲，常识过度免责 (12%)']
        }};
        for (const [k, [lbl, desc]] of Object.entries(metricLabels)) {{
          const val = item.metrics[k] !== undefined ? item.metrics[k] : 0;
          cardsHtml += `
            <div class="bg-zinc-900 border border-zinc-800 rounded p-2.5">
              <div class="text-[11px] font-medium text-zinc-400">${{lbl}}</div>
              <div class="font-mono text-base font-bold text-zinc-100 my-0.5">${{val.toFixed(1)}}</div>
              <div class="text-[10px] text-zinc-400">${{desc}}</div>
            </div>
          `;
        }}

        let badcasesHtml = '';
        if (item.worst && item.worst.length > 0) {{
          badcasesHtml = '<div class="mt-4 font-semibold text-xs text-zinc-300">典型最高扣分题 (Badcases)：</div>';
          item.worst.forEach(w => {{
            badcasesHtml += `
              <div class="bg-zinc-900 border border-zinc-800 rounded p-3 mt-2 text-xs">
                <div class="text-red-400 font-medium mb-1 font-mono">题目 [${{w.task_id}}] · 扣分 ${{w.score}}</div>
                <div class="text-zinc-400 leading-relaxed font-sans">${{w.response}}</div>
              </div>
            `;
          }});
        }}

        drawerTr.innerHTML = `
          <td colspan="11" class="p-4 bg-zinc-950">
            <div class="max-w-4xl mx-auto">
              <div class="font-semibold text-xs text-zinc-300 mb-2">${{item.name}} · 六维病灶诊断</div>
              <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">${{cardsHtml}}</div>
              ${{badcasesHtml}}
            </div>
          </td>
        `;
        tbody.appendChild(drawerTr);
      }}
    }});

    // Update th active sort indicators
    ths.forEach(th => {{
      const key = th.getAttribute('data-sort');
      const sortIndicator = th.querySelector('.sort-ind');
      if (sortIndicator) sortIndicator.remove();
      if (key === state.sortKey) {{
        const ind = document.createElement('span');
        ind.className = 'sort-ind ml-1 text-zinc-400 text-[10px]';
        ind.textContent = state.sortAsc ? '▲' : '▼';
        th.appendChild(ind);
      }}
    }});
  }}

  // Event bindings
  searchInput.addEventListener('input', (e) => {{
    state.search = e.target.value.trim();
    render();
  }});

  tierBtns.forEach(btn => {{
    btn.addEventListener('click', () => {{
      tierBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.tier = btn.getAttribute('data-tier');
      render();
    }});
  }});

  providerBtns.forEach(btn => {{
    btn.addEventListener('click', () => {{
      providerBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.provider = btn.getAttribute('data-provider');
      render();
    }});
  }});

  ths.forEach(th => {{
    th.addEventListener('click', () => {{
      const key = th.getAttribute('data-sort');
      if (!key) return;
      if (state.sortKey === key) {{
        state.sortAsc = !state.sortAsc;
      }} else {{
        state.sortKey = key;
        state.sortAsc = true;
      }}
      render();
    }});
  }});

  render();
}})();
</script>
</body>
</html>
"""


def run_scoring(
    tasks_path: str | Path,
    runs: list[str | Path],
    out_dir: str | Path = "reports",
    html_path: str | Path = "web/index.html",
) -> list[ModelReport]:
    """执行批量打分并生成 JSON、Markdown 战报与 HTML 榜单。"""
    tasks = load_tasks(tasks_path)
    grouped = load_samples(runs)
    reports = [score_model(model, samples, tasks) for model, samples in grouped.items()]
    if not reports:
        raise SystemExit("没有可打分的样本")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "model": r.model,
            "reasoning_effort": r.reasoning_effort or "默认",
            "tier": r.tier or classify_tier(r.tshu_score),
            "n_samples": r.n_samples,
            "route_note": r.route_note,
            "tshu_score": r.tshu_score,
            "open_score": r.open_score,
            "body_score": r.body_score,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "opening_entropy": r.opening_entropy,
            "metrics": r.metrics,
            "worst": [
                {
                    "task_id": item.task.id,
                    "tshu_score": item.metrics.tshu_score,
                    "response": item.sample.response,
                }
                for item in r.worst
            ],
        }
        for r in reports
    ]
    (out / "scores.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "report.md").write_text(render_markdown(reports), encoding="utf-8")

    html_out = Path(html_path)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(render_html(reports), encoding="utf-8")

    for report in sorted(reports, key=lambda r: r.tshu_score):
        rprint(
            f"[bold]{report.model:<28}[/bold] "
            f"[dim]{(report.reasoning_effort or '默认'):<6}[/dim] "
            f"[dim]{report.tier:<11}[/dim] "
            f"TshuScore [cyan]{report.tshu_score:>6.2f}[/cyan] "
            f"[{report.ci_lower:.2f}, {report.ci_upper:.2f}]  样本 {report.n_samples}"
        )
    logger.success(f"已写出 {out/'scores.json'}、{out/'report.md'}、{html_out}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="TshuBench 打分器")
    parser.add_argument("--tasks", default="tasks/eval_suite_v2.jsonl")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="reports")
    parser.add_argument("--html", default="web/index.html")
    args = parser.parse_args()
    run_scoring(args.tasks, args.runs, args.out, args.html)


if __name__ == "__main__":
    main()
