"""TshuBench 打分引擎：读取模型回复，输出分数、战报与静态榜单。

用法：
    uv run python -m harness.score --tasks tasks/eval_suite_v2.jsonl \
        --runs runs/demo_tidy.jsonl runs/demo_slop.jsonl --out reports
"""

import argparse
import html
import importlib.resources
import json
import random
from datetime import datetime
from pathlib import Path
from statistics import mean

from loguru import logger
from rich import print as rprint

from tshu.metrics import evaluate, opening_entropy
from tshu.models import WEIGHTS, ModelReport, Sample, ScoredSample, Task, classify_tier

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
    p = Path(path)
    if p.exists():
        content = p.read_text(encoding="utf-8")
    else:
        content = importlib.resources.files("tshu.data").joinpath("eval_suite_v2.jsonl").read_text(encoding="utf-8")
    tasks: dict[str, Task] = {}
    for line in content.splitlines():
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


def load_samples_grouped(paths: list[str | Path]) -> dict[tuple[str, str], list[Sample]]:
    """按 (model, reasoning_effort) 分组加载样本，支持同模型多思考档位独立打分。"""
    grouped: dict[tuple[str, str], list[Sample]] = {}
    for path in map(Path, paths):
        if not path.exists():
            logger.warning(f"跳过不存在的文件：{path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = Sample.model_validate_json(line)
            effort = sample.reasoning_effort or DEFAULT_EFFORT_BY_MODEL.get(sample.model, "默认")
            grouped.setdefault((sample.model, effort), []).append(sample)
    return grouped


def group_best_effort_reports(reports: list[ModelReport]) -> tuple[list[ModelReport], dict[str, list[ModelReport]]]:
    """同模型多思考档位归并：主榜取天书指数最低的档位，并返回各模型全部档位列表。"""
    by_model: dict[str, list[ModelReport]] = {}
    for r in reports:
        by_model.setdefault(r.model, []).append(r)

    primary_reports: list[ModelReport] = []
    variants_by_model: dict[str, list[ModelReport]] = {}
    for model, group in by_model.items():
        sorted_group = sorted(group, key=lambda r: r.tshu_score)
        primary_reports.append(sorted_group[0])
        variants_by_model[model] = sorted_group
    return primary_reports, variants_by_model


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
    "opencode-go/muse-spark-1.3-contributor": ("Muse Spark 1.3 Contributor", "Meta"),
    "opencode-zen/claude-sonnet-5-5": ("Claude Sonnet 5.5", "Anthropic"),
    "auto/claude-haiku-5-5": ("Claude Haiku 5.5", "Anthropic"),
    "opencode-zen/claude-opus-5-5": ("Claude Opus 5.5", "Anthropic"),
    "codex/gpt-6.1-terra": ("GPT-6.1 Terra", "OpenAI"),
    "opencode-zen/glm-5.3": ("GLM-5.3", "Zhipu"),
    "opencode-go/glm-5.3-flash": ("GLM-5.3 Flash", "Zhipu"),
    "codex/gpt-6-luna": ("GPT-6 Luna", "OpenAI"),
    "opencode-go/kimi-k3.1-pro": ("Kimi K3.1 Pro", "Moonshot"),
    "codex/gpt-6.1-sol": ("GPT-6.1 Sol", "OpenAI"),
    "opencode-zen/qwen3.8-flash": ("Qwen3.8 Flash", "Alibaba"),
    "opencode-zen/qwen3.8-max": ("Qwen3.8 Max", "Alibaba"),
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
        ("muse", "Meta"),
    ]:
        if brand in model.lower():
            return (name, provider)
    return (name, parts[0] if len(parts) > 1 else "Other")


def render_html(
    reports: list[ModelReport],
    variants_by_model: dict[str, list[ModelReport]] | None = None,
) -> str:
    ranked = sorted(reports, key=lambda r: r.tshu_score)
    model_entries = []
    static_rows = []

    for i, report in enumerate(ranked, start=1):
        display_name, provider = parse_model_display(report.model)
        tier_str = report.tier or classify_tier(report.tshu_score)
        tier_cls = "t1" if "人话" in tier_str else ("t2" if "黑话" in tier_str else "t3")
        effort_text = html.escape(report.reasoning_effort or "默认")
        route_note_text = html.escape(report.route_note or "")

        def _build_worst(rpt: ModelReport) -> list[dict]:
            items = []
            if getattr(rpt, "worst", None):
                for item in rpt.worst[:2]:
                    items.append({
                        "task_id": getattr(item.task, "id", str(item.task)),
                        "prompt": getattr(item.task, "prompt", ""),
                        "score": round(float(item.metrics.tshu_score), 1),
                        "response": item.sample.response[:240] + ("..." if len(item.sample.response) > 240 else ""),
                    })
            return items

        worst_items = _build_worst(report)

        variants_data = []
        if variants_by_model and report.model in variants_by_model:
            for v in variants_by_model[report.model]:
                v_tier = v.tier or classify_tier(v.tshu_score)
                variants_data.append({
                    "effort": v.reasoning_effort or "默认",
                    "tshu_score": round(v.tshu_score, 2),
                    "ci_lower": round(v.ci_lower, 2),
                    "ci_upper": round(v.ci_upper, 2),
                    "tier": v_tier,
                    "metrics": {k: round(v.metrics[k], 1) for k in METRIC_LABELS},
                    "worst": _build_worst(v),
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
            "effort_variants": variants_data,
        }
        model_entries.append(entry)

        # Static row generation for tests & non-JS fallback
        cells = "".join(f"<td class='px-2.5 md:px-3 py-3 md:py-3.5 text-center font-mono text-xs md:text-sm font-semibold text-zinc-200 border-b border-zinc-800 {('border-l border-zinc-800' if k == 'cci' else '')}'>{report.metrics[k]:.1f}</td>" for k in METRIC_LABELS)
        route_badge = f" <span class='text-xs text-zinc-400 font-mono'>({route_note_text})</span>" if route_note_text else ""
        medal = "🥇 " if i == 1 else ("🥈 " if i == 2 else ("🥉 " if i == 3 else ""))
        static_rows.append(
            f"<tr class='model-row hover:bg-zinc-900/70 transition-colors cursor-pointer' data-rank='{i}'>"
            f"<td class='col-rank px-1.5 md:px-3 py-3 md:py-3.5 text-center font-mono text-xs md:text-sm font-semibold text-zinc-400 border-b border-zinc-800'>{medal}{i}</td>"
            f"<td class='col-model px-2.5 md:px-4 py-3 md:py-3.5 overflow-hidden border-b border-zinc-800'><div class='flex items-center gap-1.5 md:gap-2'>"
            f"<span class='font-bold text-sm md:text-[15px] text-zinc-100 tracking-tight font-sans truncate'>{html.escape(display_name)}</span>"
            f"<span class='hidden md:inline-block text-xs font-mono px-2 py-0.5 border border-zinc-700 bg-zinc-900 text-zinc-300 shrink-0'>{html.escape(provider)}</span>"
            f"</div><div class='text-[11px] md:text-xs font-mono text-zinc-400 mt-0.5 md:mt-1 truncate'><span class='md:hidden text-zinc-300 mr-1'>{html.escape(provider)} ·</span><code class='hidden md:inline'>{html.escape(report.model)}</code>{route_badge}</div></td>"
            f"<td class='px-2 md:px-3 py-3 md:py-3.5 text-center border-b border-zinc-800 border-l border-zinc-800'>"
            f"<span class='font-mono text-base md:text-lg font-bold text-zinc-50' title='95% 置信区间: [{report.ci_lower:.1f}, {report.ci_upper:.1f}]'>{report.tshu_score:.2f}</span>"
            f"</td>"
            f"<td class='px-2 md:px-3 py-3 md:py-3.5 text-center font-mono text-xs md:text-sm text-zinc-300 effort border-b border-zinc-800'>{effort_text}</td>"
            f"<td class='px-2 md:px-3 py-3 md:py-3.5 text-center border-b border-zinc-800'><span class='tier-tag {tier_cls} text-[11px] md:text-xs font-mono font-semibold px-1.5 md:px-2 py-0.5 border'>{tier_str}</span></td>"
            f"{cells}</tr>"
        )

    data_json = json.dumps(model_entries, ensure_ascii=False)
    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M")
    date_str = now_dt.strftime("%Y-%m-%d")

    metric_header_defs = [
        ("cci", "CCI", "长链推演", "border-l border-zinc-800"),
        ("fmt", "FMT", "排版膨胀", ""),
        ("enu", "ENU", "报菜名", ""),
        ("tic", "TIC", "套话模板", ""),
        ("cvd", "CVD", "决断推延", ""),
        ("hed", "HED", "防御叠甲", ""),
    ]
    head_cells = "".join(
        f"<th scope='col' data-sort='{k}' title='{code}' class='p-0 text-center border-b border-zinc-800 {border_cls}'>"
        f"<button type='button' class='sort-btn w-full h-full py-3 md:py-3.5 px-2 inline-flex flex-col items-center justify-center cursor-pointer hover:text-white transition-colors group'>"
        f"<span class='flex items-center gap-1'>"
        f"<span class='font-semibold text-xs md:text-sm text-zinc-200 group-hover:text-white whitespace-nowrap'>{desc}</span>"
        f"<span class='sort-ind text-zinc-400 text-xs' aria-hidden='true'></span>"
        f"</span>"
        f"<span class='text-[10px] md:text-[11px] text-zinc-400 font-mono font-normal mt-0.5'>{code}</span>"
        f"</button></th>"
        for k, code, desc, border_cls in metric_header_defs
    )

    return f"""<!doctype html>
<html lang="zh-CN" class="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>TshuBench · 天书榜 | 中文大模型表达负担与结构膨胀评测</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script>
tailwind.config = {{
  darkMode: 'class',
  theme: {{
    extend: {{
      fontFamily: {{
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', '"PingFang SC"', '"Hiragino Sans GB"', '"Microsoft YaHei"', '"Noto Sans SC"', 'sans-serif'],
        mono: ['"JetBrains Mono"', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', '"PingFang SC"', '"Hiragino Sans GB"', '"Microsoft YaHei"', '"Noto Sans SC"', 'monospace'],
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
:root {{
  color-scheme: dark;
  --target-size: 24px;
}}

@media (pointer: coarse) {{
  :root {{
    --target-size: 44px;
  }}
}}

* {{
  border-radius: 0 !important;
}}

body {{
  background-color: #09090b;
  color: #f4f4f5;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  font-size: 0.9375rem;
  overflow-wrap: break-word;
}}

h1, h2, h3 {{
  text-wrap: balance;
}}

p {{
  text-wrap: pretty;
}}

:where(a, code) {{
  overflow-wrap: anywhere;
}}

:where(button, select, input[type="text"], [tabindex="0"]):focus-visible,
a:focus-visible {{
  outline: 2px solid #34d399;
  outline-offset: 2px;
}}

.touch-target {{
  min-block-size: var(--target-size);
}}

.visually-hidden {{
  position: absolute !important;
  clip-path: inset(50%) !important;
  overflow: hidden !important;
  width: 1px !important;
  height: 1px !important;
  margin: -1px !important;
  padding: 0 !important;
  border: 0 !important;
  white-space: nowrap !important;
}}

code, pre, kbd, samp, .font-mono, td.effort, .effort, [data-mono], .metric-cell {{
  font-family: 'JetBrains Mono', -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans SC", sans-serif !important;
  font-feature-settings: 'tnum' on, 'zero' on;
}}

.tier-tag.t1 {{
  color: #34d399;
  background: rgba(16, 185, 129, 0.08);
  border-color: rgba(16, 185, 129, 0.35);
}}
.tier-tag.t2 {{
  color: #fbbf24;
  background: rgba(245, 158, 11, 0.08);
  border-color: rgba(245, 158, 11, 0.35);
}}
.tier-tag.t3 {{
  color: #f87171;
  background: rgba(239, 68, 68, 0.08);
  border-color: rgba(239, 68, 68, 0.35);
}}

.filter-btn {{
  background: transparent;
  transition: all 0.12s ease;
}}
.filter-btn:hover {{
  background-color: #18181b;
  color: #ffffff;
}}
.filter-btn.active {{
  background-color: #27272a !important;
  color: #ffffff !important;
  font-weight: 700 !important;
}}

#leaderboard-scroll {{
  container-type: inline-size;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}}

#leaderboard-table {{
  border-collapse: separate;
  border-spacing: 0;
  table-layout: fixed;
  min-width: 1040px;
}}

.col-rank-width {{ width: 54px; }}
.col-score-width {{ width: 108px; }}
.col-effort-width {{ width: 90px; }}
.col-tier-width {{ width: 100px; }}
.col-metric-width {{ width: 80px; }}

.drawer-viewport {{
  position: sticky;
  left: 0;
  width: 100cqw;
  max-width: 100%;
  box-sizing: border-box;
}}

@media (max-width: 767px) {{
  #leaderboard-table {{
    min-width: 804px;
  }}
  .col-rank-width {{ width: 42px; }}
  .col-model-width {{ width: 146px; }}
  .col-score-width {{ width: 82px; }}
  .col-effort-width {{ width: 74px; }}
  .col-tier-width {{ width: 84px; }}
  .col-metric-width {{ width: 62px; }}

  #leaderboard-table th.col-rank,
  #leaderboard-table td.col-rank {{
    position: sticky;
    left: 0;
    z-index: 3;
    background-color: #09090b;
  }}
  #leaderboard-table th.col-model,
  #leaderboard-table td.col-model {{
    position: sticky;
    left: 42px;
    z-index: 3;
    background-color: #09090b;
    border-right: 1px solid #27272a;
    box-shadow: 6px 0 12px -4px rgba(0, 0, 0, 0.7);
  }}
  #leaderboard-table thead th.col-rank,
  #leaderboard-table thead th.col-model {{
    z-index: 5;
    background-color: #121215;
  }}
  #leaderboard-table tbody tr.model-row:hover td.col-rank,
  #leaderboard-table tbody tr.model-row:hover td.col-model {{
    background-color: #18181b;
  }}
}}
</style>
</head>
<body class="min-h-screen text-zinc-100 antialiased selection:bg-zinc-800 selection:text-white">

<div class="max-w-[1440px] mx-auto px-4 py-8 sm:px-6 sm:py-10">
  <!-- Header / Topbar -->
  <header class="mb-6">
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-zinc-800">
      <h1 class="flex items-center gap-3 text-2xl sm:text-3xl font-extrabold tracking-tight font-mono">
        <span class="bg-white text-zinc-950 px-3 font-mono font-black text-xl sm:text-2xl inline-flex items-center justify-center h-9 sm:h-10" style="font-weight: 900;">TshuBench</span>
        <span class="text-white text-2xl sm:text-3xl font-extrabold tracking-tight">天书榜</span>
      </h1>
      <div class="flex items-center gap-2.5 font-mono">
        <a href="https://github.com/DBinK/TshuBench" target="_blank" rel="noopener" class="border border-zinc-700 bg-zinc-900/80 hover:bg-zinc-800 hover:border-zinc-500 text-zinc-200 hover:text-white px-3.5 sm:px-4 text-xs sm:text-sm font-semibold inline-flex items-center justify-center gap-1.5 transition-colors h-9 sm:h-10">
          <span>GitHub</span>
          <span class="text-zinc-500">↗</span>
        </a>
        <a href="https://github.com/DBinK/TshuBench/blob/main/docs/methodology.md" target="_blank" rel="noopener" class="border border-zinc-700 bg-zinc-900/80 hover:bg-zinc-800 hover:border-zinc-500 text-zinc-200 hover:text-white px-3.5 sm:px-4 text-xs sm:text-sm font-semibold inline-flex items-center justify-center gap-1.5 transition-colors h-9 sm:h-10">
          <span>评测方案</span>
          <span class="text-zinc-500">↗</span>
        </a>
      </div>
    </div>
    <p class="text-sm text-zinc-400 font-sans py-3">当代大模型图一乐评测，鉴定模型还说不说人话</p>

    <!-- Analytical Terminal Spec Strip -->
    <div class="grid grid-cols-2 md:grid-cols-4 border border-zinc-800 bg-zinc-950 divide-x divide-y md:divide-y-0 divide-zinc-800">
      <div class="p-3 sm:p-4">
        <div class="text-zinc-400 text-xs font-sans tracking-wider">最近更新</div>
        <div class="text-zinc-100 font-bold text-sm sm:text-base md:text-lg font-mono mt-1">{now_str}</div>
      </div>
      <div class="p-3 sm:p-4">
        <div class="text-zinc-400 text-xs font-sans tracking-wider">参评模型</div>
        <div class="text-zinc-100 font-bold text-sm sm:text-base md:text-lg font-mono mt-1">{len(ranked)} 款主流模型</div>
      </div>
      <div class="p-3 sm:p-4">
        <div class="text-zinc-400 text-xs font-sans tracking-wider">评测样本</div>
        <div class="text-zinc-100 font-bold text-sm sm:text-base md:text-lg font-mono mt-1">36 个场景 · 612 次评测</div>
      </div>
      <div class="p-3 sm:p-4">
        <div class="text-zinc-400 text-xs font-sans tracking-wider">打分引擎</div>
        <div class="text-emerald-400 font-bold text-sm sm:text-base md:text-lg font-mono mt-1">AST 确定性规则（无 LLM 裁判）</div>
      </div>
    </div>

    <!-- Notice Strip -->
    <div class="border-x border-b border-zinc-800 bg-zinc-900/40 p-3.5 sm:p-4 text-sm text-zinc-300 font-sans leading-relaxed">
      <span class="text-emerald-400 font-bold font-mono mr-1.5">[说明]</span> 本榜为工程趣味性观测基准，不构成严谨学术结论，具体表现请以实际业务体验为准。采用反向扣分制（分数越低表达越简洁，点击模型可展开详情）。欢迎提交 PR 贡献新指标与评测算法，下一代榜单正在规划开发中。
    </div>
  </header>

  <main id="main-content">
    <div id="status-announcer" role="status" aria-live="polite" class="visually-hidden"></div>

    <!-- Leaderboard Container -->
    <div class="border border-zinc-800 bg-zinc-950">
      <!-- Controls / Filters Docked Directly Atop Table -->
      <div class="border-b border-zinc-800 flex flex-col xl:flex-row items-stretch justify-between text-sm font-mono">
        <div class="flex-1 min-w-0">
          <input type="text" id="search-input" aria-label="搜索模型名称、厂商或底层代号" placeholder="🔍 搜索模型名称、厂商、底层代号..." class="touch-target w-full bg-zinc-950 text-zinc-100 placeholder-zinc-500 px-4 py-3 text-sm font-sans border-b xl:border-b-0 xl:border-r border-zinc-800 focus:bg-zinc-900 transition-colors" />
        </div>

        <div class="flex flex-col sm:flex-row flex-wrap items-stretch sm:items-center">
          <!-- Tier Filter -->
          <div class="flex flex-wrap items-center border-b sm:border-b-0 sm:border-r border-zinc-800 divide-x divide-zinc-800" id="tier-group" role="group" aria-label="按梯队筛选">
            <button type="button" aria-pressed="true" class="filter-btn touch-target active px-3 sm:px-4 py-2.5 sm:py-3.5 text-xs sm:text-sm text-zinc-400 hover:text-white transition-colors" data-tier="all">全部梯队</button>
            <button type="button" aria-pressed="false" class="filter-btn touch-target px-3 sm:px-4 py-2.5 sm:py-3.5 text-xs sm:text-sm text-zinc-400 hover:text-white transition-colors" data-tier="人话级">🟢 人话级</button>
            <button type="button" aria-pressed="false" class="filter-btn touch-target px-3 sm:px-4 py-2.5 sm:py-3.5 text-xs sm:text-sm text-zinc-400 hover:text-white transition-colors" data-tier="黑话级">🟡 黑话级</button>
            <button type="button" aria-pressed="false" class="filter-btn touch-target px-3 sm:px-4 py-2.5 sm:py-3.5 text-xs sm:text-sm text-zinc-400 hover:text-white transition-colors" data-tier="天书级">🔴 天书级</button>
          </div>

          <!-- Provider Filter -->
          <div class="flex flex-wrap items-center divide-x divide-zinc-800" id="provider-group" role="group" aria-label="按厂商筛选">
            <button type="button" aria-pressed="true" class="filter-btn touch-target active px-3 sm:px-4 py-2.5 sm:py-3.5 text-xs sm:text-sm text-zinc-400 hover:text-white transition-colors" data-provider="all">全部厂商</button>
            <button type="button" aria-pressed="false" class="filter-btn touch-target px-3 sm:px-4 py-2.5 sm:py-3.5 text-xs sm:text-sm text-zinc-400 hover:text-white transition-colors" data-provider="Google">Google</button>
            <button type="button" aria-pressed="false" class="filter-btn touch-target px-3 sm:px-4 py-2.5 sm:py-3.5 text-xs sm:text-sm text-zinc-400 hover:text-white transition-colors" data-provider="Anthropic">Anthropic</button>
            <button type="button" aria-pressed="false" class="filter-btn touch-target px-3 sm:px-4 py-2.5 sm:py-3.5 text-xs sm:text-sm text-zinc-400 hover:text-white transition-colors" data-provider="OpenAI">OpenAI</button>
            <button type="button" aria-pressed="false" class="filter-btn touch-target px-3 sm:px-4 py-2.5 sm:py-3.5 text-xs sm:text-sm text-zinc-400 hover:text-white transition-colors" data-provider="Other">其它</button>
          </div>
        </div>
      </div>

      <!-- Mobile Table Scroll Hint (< md) -->
      <div class="md:hidden border-b border-zinc-800 px-3 py-2 bg-zinc-900/40 text-[11px] font-mono text-zinc-400 flex items-center justify-between">
        <span>← 左右滑动对比六维指标 · 点击表头排序</span>
        <span class="text-emerald-400">点击模型展开</span>
      </div>

      <!-- Unified Responsive Leaderboard Table -->
      <div id="leaderboard-scroll" tabindex="0" role="region" aria-labelledby="leaderboard-caption" class="block overflow-x-auto">
        <table class="w-full text-left" id="leaderboard-table">
          <caption id="leaderboard-caption" class="visually-hidden">TshuBench 天书榜大模型评测排行榜</caption>
          <colgroup>
            <col class="col-rank-width">   <!-- 1: # -->
            <col class="col-model-width">  <!-- 2: 模型与厂商 -->
            <col class="col-score-width">  <!-- 3: 天书指数 -->
            <col class="col-effort-width"> <!-- 4: 思考档位 -->
            <col class="col-tier-width">   <!-- 5: 梯队 -->
            <col class="col-metric-width"> <!-- 6: 长链推演 -->
            <col class="col-metric-width"> <!-- 7: 排版膨胀 -->
            <col class="col-metric-width"> <!-- 8: 报菜名 -->
            <col class="col-metric-width"> <!-- 9: 套话模板 -->
            <col class="col-metric-width"> <!-- 10: 决断推延 -->
            <col class="col-metric-width"> <!-- 11: 防御叠甲 -->
          </colgroup>
          <thead>
            <tr class="bg-zinc-900/90 text-xs md:text-sm text-zinc-300">
              <th scope="col" data-sort="rank" class="col-rank p-0 text-center text-zinc-400 font-mono border-b border-zinc-800">
                <button type="button" class="sort-btn w-full h-full py-3 md:py-3.5 px-1.5 md:px-3 inline-flex items-center justify-center gap-1 cursor-pointer hover:text-white">
                  <span class="font-semibold text-xs md:text-sm">#</span>
                  <span class="sort-ind text-zinc-400 text-xs" aria-hidden="true"></span>
                </button>
              </th>
              <th scope="col" data-sort="name" class="col-model p-0 text-left text-zinc-200 font-semibold border-b border-zinc-800">
                <button type="button" class="sort-btn w-full h-full py-3 md:py-3.5 px-2.5 md:px-4 flex items-center gap-1 cursor-pointer hover:text-white text-left">
                  <span class="font-semibold text-xs md:text-sm">模型与厂商</span>
                  <span class="sort-ind text-zinc-400 text-xs" aria-hidden="true"></span>
                </button>
              </th>
              <th scope="col" data-sort="tshu_score" aria-sort="ascending" class="p-0 text-center sort-active group border-b border-l border-zinc-800">
                <button type="button" class="sort-btn w-full h-full py-3 md:py-3.5 px-2 md:px-3 inline-flex flex-col items-center justify-center cursor-pointer hover:text-white">
                  <span class="flex items-center gap-1">
                    <span class="font-bold text-xs md:text-sm text-zinc-100 group-hover:text-white whitespace-nowrap">天书指数</span>
                    <span class="sort-ind text-zinc-400 text-xs" aria-hidden="true">▲</span>
                  </span>
                  <span class="text-[10px] md:text-[11px] text-emerald-400 font-mono font-normal mt-0.5">越低越好</span>
                </button>
              </th>
              <th scope="col" data-sort="effort" class="p-0 text-center text-zinc-300 font-medium border-b border-zinc-800">
                <button type="button" class="sort-btn w-full h-full py-3 md:py-3.5 px-2 md:px-3 inline-flex items-center justify-center gap-1 cursor-pointer hover:text-white">
                  <span class="font-semibold text-xs md:text-sm whitespace-nowrap">思考档位</span>
                  <span class="sort-ind text-zinc-400 text-xs" aria-hidden="true"></span>
                </button>
              </th>
              <th scope="col" data-sort="tier" class="p-0 text-center text-zinc-300 font-medium border-b border-zinc-800">
                <button type="button" class="sort-btn w-full h-full py-3 md:py-3.5 px-2 md:px-3 inline-flex items-center justify-center gap-1 cursor-pointer hover:text-white">
                  <span class="font-semibold text-xs md:text-sm">梯队</span>
                  <span class="sort-ind text-zinc-400 text-xs" aria-hidden="true"></span>
                </button>
              </th>
              {head_cells}
            </tr>
          </thead>
          <tbody id="leaderboard-body">
            {''.join(static_rows)}
          </tbody>
        </table>
      </div>
    </div>

    <!-- Contribute via AI Agent -->
    <section aria-labelledby="contribute-heading" class="mt-6 border border-zinc-800 bg-zinc-950 p-4">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 mb-2.5">
        <h2 id="contribute-heading" class="font-mono text-xs sm:text-sm text-zinc-200 font-bold">// 贡献评测结果：复制以下 Prompt 发给你的 AI 编程助手（Claude Code / Cursor / Antigravity 等）</h2>
        <button id="copy-agent-prompt" type="button" class="touch-target self-start sm:self-auto px-3 py-1.5 text-xs font-mono border border-emerald-500/40 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 transition-colors cursor-pointer shrink-0">复制 Prompt</button>
      </div>
      <pre id="agent-prompt-text" class="bg-black border border-zinc-800/80 p-3 text-xs sm:text-sm font-mono text-zinc-300 whitespace-pre-wrap break-words leading-relaxed">请帮我评测模型 [请在此填写模型 ID，如 deepseek/deepseek-v4.1-flash]，并准备提交到 TshuBench（天书榜）：

1. 运行 `npx skills add DBinK/TshuBench` 安装 `tshu-eval` 技能。
2. 加载 `tshu-eval` 技能并严格按照其中的规范与步骤执行评测、校验与 PR 准备。</pre>
    </section>
  </main>

  <!-- Footer -->
  <footer class="mt-8 pt-4 border-t border-zinc-800 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs font-mono text-zinc-500">
    <div>天书榜 V3 · 评测样本 N=612 · 更新时间 {now_str}</div>
    <div>无 LLM 裁判 AST 确定性引擎 · MIT 开源协议</div>
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
    expanded: new Set(),
    selectedVariant: {{}}
  }};

  const majorProviders = new Set(['Google', 'Anthropic', 'OpenAI']);
  const collator = new Intl.Collator('zh-CN', {{ numeric: true, sensitivity: 'base' }});
  const metricKeys = ['cci', 'fmt', 'enu', 'tic', 'cvd', 'hed'];
  const metricLabels = {{
    cci: ['长链推演', '长复合句与改口推演痕迹 (24%)'],
    fmt: ['排版膨胀', 'Markdown 结构强迫与支线清单 (16%)'],
    enu: ['报菜名', '顿号斜杠与高密符号压缩 (20%)'],
    tic: ['套话模板', '套话抽动与公文式结构 (16%)'],
    cvd: ['决断推延', '首句打太极，结论后置 (12%)'],
    hed: ['防御叠甲', '逢人便叠甲，常识过度免责 (12%)']
  }};

  const tbody = document.getElementById('leaderboard-body');
  const searchInput = document.getElementById('search-input');
  const tierBtns = document.querySelectorAll('#tier-group .filter-btn');
  const providerBtns = document.querySelectorAll('#provider-group .filter-btn');
  const ths = document.querySelectorAll('#leaderboard-table th[data-sort]');
  const announcer = document.getElementById('status-announcer');

  function announce(msg) {{
    if (announcer) announcer.textContent = msg;
  }}

  function esc(str) {{
    return String(str ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }}

  function buildDrawerContentHtml(item) {{
    const hasVariants = item.effort_variants && item.effort_variants.length > 0;
    const activeIdx = hasVariants ? (state.selectedVariant[item.model] ?? 0) : 0;
    const active = hasVariants ? (item.effort_variants[activeIdx] || item) : item;
    const activeMetrics = active.metrics || item.metrics;
    const activeWorst = active.worst || item.worst;
    const activeEffort = active.effort || item.effort || '默认';

    let cardsHtml = '';
    for (const [k, [lbl, desc]] of Object.entries(metricLabels)) {{
      const val = activeMetrics[k] !== undefined ? activeMetrics[k] : 0;
      cardsHtml += `
        <div class="bg-zinc-950 border border-zinc-800 p-3 sm:p-3.5">
          <div class="text-xs font-mono text-zinc-300 font-bold">${{lbl}}</div>
          <div class="font-mono text-lg sm:text-xl font-bold text-zinc-50 my-1">${{val.toFixed(1)}}</div>
          <div class="text-xs text-zinc-400 leading-normal font-sans">${{desc}}</div>
        </div>
      `;
    }}

    let variantsHtml = '';
    if (item.effort_variants && item.effort_variants.length > 1) {{
      let vCards = '';
      item.effort_variants.forEach((v, idx) => {{
        const isBest = idx === 0;
        const isActive = idx === activeIdx;
        const badge = isBest ? '<span class="text-[10px] font-mono px-1.5 py-0.5 bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 ml-2">主榜代表</span>' : '';
        const borderCls = isActive ? 'border-emerald-400 ring-1 ring-emerald-400/40' : (isBest ? 'border-emerald-500/40 hover:border-zinc-500' : 'border-zinc-800 hover:border-zinc-600');
        vCards += `
          <button type="button" data-variant-idx="${{idx}}" aria-pressed="${{isActive}}" class="variant-card touch-target text-left w-full cursor-pointer transition-colors bg-zinc-950 border ${{borderCls}} p-3">
            <div class="flex items-center justify-between text-xs font-mono text-zinc-300">
              <span>档位：${{esc(v.effort)}}</span>${{badge}}
            </div>
            <div class="font-mono text-base sm:text-lg font-bold text-zinc-50 mt-1">${{v.tshu_score.toFixed(2)}} <span class="text-xs font-normal text-zinc-400">[${{v.ci_lower.toFixed(1)}}, ${{v.ci_upper.toFixed(1)}}]</span></div>
            <div class="text-xs text-zinc-400 mt-0.5">${{esc(v.tier)}}</div>
          </button>
        `;
      }});
      variantsHtml = `
        <div class="mt-4 border border-zinc-800 bg-zinc-950 p-3 sm:p-4">
          <div class="font-mono text-xs text-zinc-300 uppercase tracking-wider mb-2.5 font-bold">// 多思考档位对比（点击卡片切换下方诊断与样本）</div>
          <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 sm:gap-3">${{vCards}}</div>
        </div>
      `;
    }}

    let badcasesHtml = '';
    if (activeWorst && activeWorst.length > 0) {{
      badcasesHtml = `
        <div class="mt-4 border border-zinc-800 bg-zinc-950 p-3 sm:p-4">
          <div class="font-mono text-xs text-zinc-300 uppercase tracking-wider mb-2.5 font-bold">// 典型高扣分样本</div>
      `;
      activeWorst.forEach(w => {{
        const promptHtml = w.prompt ? `<div class="text-zinc-200 font-sans text-xs sm:text-sm mb-2 bg-zinc-950 border border-zinc-800 px-3 py-2"><span class="text-emerald-400 font-mono font-bold mr-1.5">问：</span>${{esc(w.prompt)}}</div>` : '';
        badcasesHtml += `
          <div class="border-t border-zinc-800/80 pt-3 mt-3 first:border-0 first:pt-0 first:mt-0 text-sm">
            <div class="text-red-400 font-semibold mb-1.5 font-mono text-xs">用例：[${{esc(w.task_id)}}] · 扣分：${{w.score}}</div>
            ${{promptHtml}}
            <div class="text-zinc-300 font-mono text-xs sm:text-sm leading-relaxed bg-zinc-900/60 border border-zinc-800/60 p-3 whitespace-pre-wrap break-words"><span class="text-zinc-400 font-bold mr-1.5">答：</span>${{esc(w.response)}}</div>
          </div>
        `;
      }});
      badcasesHtml += '</div>';
    }}

    const parts = item.model.split('/');
    const prov = parts.length > 1 ? parts[0].toLowerCase() : 'unknown';
    const slug = (parts.length > 1 ? parts.slice(1).join('-') : parts[0]).replace(/[^a-zA-Z0-9._-]/g, '-');
    const effSlug = activeEffort === '默认' ? 'default' : activeEffort;
    const rawUrl = `https://github.com/DBinK/TshuBench/blob/main/runs/${{prov}}/${{slug}}_${{effSlug}}.jsonl`;

    return `
      <div class="max-w-5xl mx-auto">
        <div class="flex flex-col sm:flex-row sm:items-baseline justify-between border-b border-zinc-800 pb-3 mb-4 gap-2">
          <div class="font-bold text-sm sm:text-base text-zinc-100 font-mono tracking-tight flex flex-wrap items-center gap-2">
            <span>// ${{item.name}} (${{activeEffort}}) · 六维病灶诊断</span>
            <a href="${{rawUrl}}" target="_blank" rel="noopener" class="text-xs font-normal text-emerald-400 hover:underline">原始样本 ↗</a>
          </div>
          <div class="text-xs font-mono text-zinc-400">天书指数：<span class="text-emerald-400 font-bold text-sm">${{active.tshu_score.toFixed(2)}}</span> · 95% 置信区间 [${{active.ci_lower.toFixed(1)}}, ${{active.ci_upper.toFixed(1)}}]</div>
        </div>
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2.5 sm:gap-3">${{cardsHtml}}</div>
        ${{variantsHtml}}
        ${{badcasesHtml}}
      </div>
    `;
  }}

  function bindDrawerInteractions(container, item) {{
    container.querySelectorAll('.variant-card').forEach(card => {{
      card.addEventListener('click', (e) => {{
        e.stopPropagation();
        const idx = parseInt(card.dataset.variantIdx, 10);
        state.selectedVariant[item.model] = idx;
        render({{ focusModel: item.model, focusVariantIdx: idx }});
      }});
    }});
  }}

  function toggleExpand(model) {{
    if (state.expanded.has(model)) {{
      state.expanded.delete(model);
    }} else {{
      state.expanded.add(model);
    }}
    render({{ focusModel: model }});
  }}

  function render(focusHint) {{
    let filtered = rawData.filter(item => {{
      if (state.tier !== 'all' && item.tier !== state.tier) return false;
      if (state.provider === 'Other') {{
        if (majorProviders.has(item.provider)) return false;
      }} else if (state.provider !== 'all' && item.provider !== state.provider) {{
        return false;
      }}
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
      let cmp = 0;
      if (state.sortKey === 'rank') {{
        cmp = a.rank - b.rank;
      }} else if (state.sortKey === 'tshu_score') {{
        cmp = a.tshu_score - b.tshu_score;
      }} else if (state.sortKey === 'name' || state.sortKey === 'effort' || state.sortKey === 'tier') {{
        cmp = collator.compare(String(a[state.sortKey] || ''), String(b[state.sortKey] || ''));
      }} else {{
        cmp = (a.metrics[state.sortKey] || 0) - (b.metrics[state.sortKey] || 0);
      }}
      return state.sortAsc ? cmp : -cmp;
    }});

    tbody.innerHTML = '';

    if (filtered.length === 0) {{
      tbody.innerHTML = '<tr><td colspan="11" class="p-0 bg-zinc-950 border-b border-zinc-800"><div class="drawer-viewport py-10 px-4 text-center text-sm font-mono text-zinc-400">没有匹配的模型，请调整搜索或筛选条件</div></td></tr>';
    }} else {{
      filtered.forEach(item => {{
        const isExpanded = state.expanded.has(item.model);
        const routeBadge = item.route_note ? ` <span class="text-xs text-zinc-400 font-mono">(${{esc(item.route_note)}})</span>` : '';
        const medal = item.rank === 1 ? '🥇 ' : (item.rank === 2 ? '🥈 ' : (item.rank === 3 ? '🥉 ' : ''));

        const tr = document.createElement('tr');
        tr.className = 'model-row hover:bg-zinc-900/70 transition-colors cursor-pointer';
        let detailedTds = '';
        for (const k of metricKeys) {{
          const val = item.metrics[k] !== undefined ? item.metrics[k].toFixed(1) : '-';
          const borderCls = k === 'cci' ? 'border-l border-zinc-800' : '';
          detailedTds += `<td class="px-2 md:px-3 py-3 md:py-3.5 text-center font-mono text-xs md:text-sm font-semibold text-zinc-200 border-b border-zinc-800 ${{borderCls}}">${{val}}</td>`;
        }}

        tr.innerHTML = `
          <td class="col-rank px-1.5 md:px-3 py-3 md:py-3.5 text-center font-mono text-xs md:text-sm font-semibold text-zinc-400 border-b border-zinc-800">${{medal}}${{item.rank}}</td>
          <td class="col-model px-2.5 md:px-4 py-3 md:py-3.5 overflow-hidden border-b border-zinc-800">
            <div class="flex items-center gap-1.5 md:gap-2">
              <button type="button" data-expand-model="${{esc(item.model)}}" aria-expanded="${{isExpanded}}" class="expand-btn touch-target text-left font-bold text-xs sm:text-sm md:text-[15px] text-zinc-100 hover:text-emerald-400 tracking-tight font-sans truncate cursor-pointer">
                ${{esc(item.name)}} <span class="text-xs font-mono text-zinc-500 ml-0.5 md:ml-1" aria-hidden="true">${{isExpanded ? '▾' : '▸'}}</span>
              </button>
              <span class="hidden md:inline-block text-xs font-mono px-2 py-0.5 border border-zinc-700 bg-zinc-900 text-zinc-300 shrink-0">${{esc(item.provider)}}</span>
            </div>
            <div class="text-[11px] md:text-xs font-mono text-zinc-400 mt-0.5 md:mt-1 truncate"><span class="md:hidden text-zinc-300 mr-1">${{esc(item.provider)}}</span><code class="hidden md:inline">${{esc(item.model)}}</code>${{routeBadge}}</div>
          </td>
          <td class="px-2 md:px-3 py-3 md:py-3.5 text-center border-b border-l border-zinc-800">
            <span class="font-mono text-base md:text-lg font-bold text-zinc-50" title="95% 置信区间: [${{item.ci_lower.toFixed(1)}}, ${{item.ci_upper.toFixed(1)}}]">${{item.tshu_score.toFixed(2)}}</span>
          </td>
          <td class="px-2 md:px-3 py-3 md:py-3.5 text-center font-mono text-xs md:text-sm text-zinc-300 effort border-b border-zinc-800">${{esc(item.effort)}}</td>
          <td class="px-2 md:px-3 py-3 md:py-3.5 text-center border-b border-zinc-800"><span class="tier-tag ${{item.tier_cls}} text-[11px] md:text-xs font-mono font-semibold px-1.5 md:px-2 py-0.5 border whitespace-nowrap">${{esc(item.tier)}}</span></td>
          ${{detailedTds}}
        `;
        tr.addEventListener('click', (e) => {{
          if (e.target.closest('a, button')) return;
          toggleExpand(item.model);
        }});
        tr.querySelector('.expand-btn').addEventListener('click', (e) => {{
          e.stopPropagation();
          toggleExpand(item.model);
        }});
        tbody.appendChild(tr);

        if (isExpanded) {{
          const drawerTr = document.createElement('tr');
          drawerTr.className = 'bg-black';
          drawerTr.innerHTML = `<td colspan="11" class="p-0 bg-black border-b border-zinc-800"><div class="drawer-viewport p-4 sm:p-6">${{buildDrawerContentHtml(item)}}</div></td>`;
          bindDrawerInteractions(drawerTr, item);
          tbody.appendChild(drawerTr);
        }}
      }});
    }}

    // Sync sort headers
    ths.forEach(th => {{
      const key = th.getAttribute('data-sort');
      const ind = th.querySelector('.sort-ind');
      if (state.sortKey === key) {{
        th.setAttribute('aria-sort', state.sortAsc ? 'ascending' : 'descending');
        th.classList.add('sort-active');
        if (ind) ind.textContent = state.sortAsc ? '▲' : '▼';
      }} else {{
        th.removeAttribute('aria-sort');
        th.classList.remove('sort-active');
        if (ind) ind.textContent = '';
      }}
    }});

    // Restore focus if an expand button or variant card triggered re-render
    if (focusHint && focusHint.focusModel) {{
      let target = null;
      if (focusHint.focusVariantIdx !== undefined) {{
        target = tbody.querySelector(`.variant-card[data-variant-idx="${{focusHint.focusVariantIdx}}"]`);
      }} else {{
        const escaped = CSS.escape(focusHint.focusModel);
        target = tbody.querySelector(`[data-expand-model="${{escaped}}"]`);
      }}
      target?.focus({{ preventScroll: true }});
    }}
  }}

  searchInput.addEventListener('input', (e) => {{
    state.search = e.target.value.trim();
    render();
  }});

  tierBtns.forEach(btn => {{
    btn.addEventListener('click', () => {{
      tierBtns.forEach(b => {{
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      }});
      btn.classList.add('active');
      btn.setAttribute('aria-pressed', 'true');
      state.tier = btn.getAttribute('data-tier');
      render();
    }});
  }});

  providerBtns.forEach(btn => {{
    btn.addEventListener('click', () => {{
      providerBtns.forEach(b => {{
        b.classList.remove('active');
        b.setAttribute('aria-pressed', 'false');
      }});
      btn.classList.add('active');
      btn.setAttribute('aria-pressed', 'true');
      state.provider = btn.getAttribute('data-provider');
      render();
    }});
  }});

  ths.forEach(th => {{
    const btn = th.querySelector('button.sort-btn');
    if (!btn) return;
    btn.addEventListener('click', () => {{
      const key = th.getAttribute('data-sort');
      if (!key) return;
      if (state.sortKey === key) {{
        state.sortAsc = !state.sortAsc;
      }} else {{
        state.sortKey = key;
        state.sortAsc = true;
      }}
      render();
      announce(`已按 ${{btn.textContent.trim()}} ${{state.sortAsc ? '升序' : '降序'}} 排列`);
    }});
  }});

  render();
}})();
</script>
  <script>
  document.getElementById('copy-agent-prompt')?.addEventListener('click', async function() {{
    const btn = this;
    const text = document.getElementById('agent-prompt-text')?.innerText || '';
    const old = btn.textContent;
    try {{
      await navigator.clipboard.writeText(text);
      btn.textContent = '已复制 ✓';
    }} catch (_err) {{
      btn.textContent = '复制失败，请手动选中';
    }}
    setTimeout(() => {{ btn.textContent = old; }}, 1800);
  }});
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
    grouped = load_samples_grouped(runs)
    all_reports = [score_model(model, samples, tasks) for (model, _effort), samples in grouped.items()]
    if not all_reports:
        raise SystemExit("没有可打分的样本")

    reports, variants_by_model = group_best_effort_reports(all_reports)

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
    html_out.write_text(render_html(reports, variants_by_model=variants_by_model), encoding="utf-8")

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
