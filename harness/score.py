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

    metric_keys = list(METRIC_LABELS.keys())
    metric_names = [v.split(" ")[0] for v in METRIC_LABELS.values()]

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
            "open_score": round(report.open_score, 2),
            "body_score": round(report.body_score, 2),
            "ci_lower": round(report.ci_lower, 2),
            "ci_upper": round(report.ci_upper, 2),
            "profile": primary_profile(report.metrics),
            "route_note": route_note_text,
            "metrics": {k: round(report.metrics[k], 1) for k in METRIC_LABELS},
            "worst": worst_items,
        }
        model_entries.append(entry)

        # Static row generation for tests & non-JS fallback
        cells = "".join(f"<td class='metric-cell col-detailed'>{report.metrics[k]:.1f}</td>" for k in METRIC_LABELS)
        medal = "🥇 " if i == 1 else ("🥈 " if i == 2 else ("🥉 " if i == 3 else ""))
        route_badge = f" <span class=ci>({route_note_text})</span>" if route_note_text else ""
        static_rows.append(
            f"<tr class='model-row' data-rank='{i}'>"
            f"<td class=rank>{medal}{i}</td>"
            f"<td class=model><div class='model-title-wrap'><strong class='model-title'>{html.escape(display_name)}</strong>"
            f"<span class='provider-tag'>{html.escape(provider)}</span></div>"
            f"<div class='model-sub'><code>{html.escape(report.model)}</code>{route_badge}</div></td>"
            f"<td class=effort>{effort_text}</td>"
            f"<td class=tier><span class='tier-pill {tier_cls}'>{tier_str}</span></td>"
            f"<td class=score><div class='score-box'><b class='score-val'>{report.tshu_score:.2f}</b>"
            f"<span class=ci>[{report.ci_lower:.1f}, {report.ci_upper:.1f}]</span><span class='score-arrow' title='扣分制，分数越低越自然'>↓</span></div></td>"
            f"<td class='col-detailed'>{report.open_score:.2f}</td>"
            f"<td class='col-detailed'>{report.body_score:.2f}</td>"
            f"<td class=profile><span class='profile-badge'>{html.escape(entry['profile'])}</span></td>"
            f"{cells}</tr>"
        )

    data_json = json.dumps(model_entries, ensure_ascii=False)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    head_cells = "".join(f"<th class='col-detailed' data-sort='{k}'>{html.escape(v)}</th>" for k, v in METRIC_LABELS.items())

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TshuBench · 天书榜 | 中文大模型表达负担评测</title>
<style>
:root {{
  --bg: #0b0f19;
  --surface: #111827;
  --surface-hover: #172238;
  --surface-border: #1f293d;
  --surface-border-subtle: #2d3748;
  --text-main: #f3f4f6;
  --text-muted: #9ca3af;
  --text-dim: #64748b;
  --accent: #38bdf8;
  --t1-fg: #34d399;
  --t1-bg: rgba(16, 185, 129, 0.12);
  --t1-border: rgba(16, 185, 129, 0.35);
  --t2-fg: #fbbf24;
  --t2-bg: rgba(245, 158, 11, 0.12);
  --t2-border: rgba(245, 158, 11, 0.35);
  --t3-fg: #f87171;
  --t3-bg: rgba(239, 68, 68, 0.12);
  --t3-border: rgba(239, 68, 68, 0.35);
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 36px 20px 80px;
  background: var(--bg);
  color: var(--text-main);
  font: 14px/1.6 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Noto Sans CJK SC", sans-serif;
  -webkit-font-smoothing: antialiased;
}}
.container {{
  max-width: 1240px;
  margin: 0 auto;
}}
header {{
  margin-bottom: 28px;
}}
.header-top {{
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 14px;
  margin-bottom: 8px;
}}
h1 {{
  font-size: 30px;
  font-weight: 700;
  margin: 0;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, #ffffff 40%, #94a3b8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.sub-desc {{
  font-size: 15px;
  color: var(--text-muted);
  margin: 0 0 16px;
}}
.pills-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}}
.stat-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 9999px;
  background: var(--surface);
  border: 1px solid var(--surface-border);
  font-size: 12px;
  color: var(--text-muted);
}}
.stat-pill b {{
  color: var(--text-main);
}}
.notice-card {{
  background: rgba(56, 189, 248, 0.06);
  border: 1px solid rgba(56, 189, 248, 0.2);
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 13px;
  color: #bae6fd;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 8px;
}}
.controls-card {{
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
.search-row {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}}
.search-wrap {{
  position: relative;
  flex: 1;
  min-width: 240px;
}}
.search-input {{
  width: 100%;
  background: #090d16;
  border: 1px solid var(--surface-border);
  border-radius: 6px;
  padding: 8px 12px;
  color: var(--text-main);
  font-size: 13px;
  outline: none;
  transition: border-color 0.15s;
}}
.search-input:focus {{
  border-color: var(--accent);
}}
.filters-wrap {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}}
.btn-group {{
  display: inline-flex;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--surface-border);
  background: #090d16;
}}
.filter-btn {{
  background: transparent;
  border: none;
  color: var(--text-muted);
  padding: 6px 12px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
}}
.filter-btn:hover {{
  color: var(--text-main);
  background: rgba(255, 255, 255, 0.04);
}}
.filter-btn.active {{
  color: #fff;
  background: #1e293b;
  font-weight: 600;
}}
.toggle-label {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--text-muted);
  cursor: pointer;
  user-select: none;
}}
.toggle-label input {{
  cursor: pointer;
}}
.table-card {{
  background: var(--surface);
  border: 1px solid var(--surface-border);
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}}
.table-scroll {{
  overflow-x: auto;
  width: 100%;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  text-align: left;
  font-size: 13px;
}}
th {{
  background: #0e1422;
  color: var(--text-muted);
  font-weight: 600;
  font-size: 12px;
  padding: 12px 10px;
  border-bottom: 1px solid var(--surface-border);
  white-space: nowrap;
  user-select: none;
  cursor: pointer;
}}
th:hover {{
  color: var(--text-main);
}}
th.sort-active {{
  color: var(--accent);
}}
td {{
  padding: 12px 10px;
  border-bottom: 1px solid var(--surface-border);
  vertical-align: middle;
  white-space: nowrap;
}}
tr.model-row {{
  cursor: pointer;
  transition: background 0.12s;
}}
tr.model-row:hover {{
  background: var(--surface-hover);
}}
td.rank {{
  color: var(--text-muted);
  font-weight: 600;
  width: 44px;
  text-align: center;
}}
.model-title-wrap {{
  display: flex;
  align-items: center;
  gap: 8px;
}}
.model-title {{
  color: var(--text-main);
  font-size: 14px;
}}
.provider-tag {{
  display: inline-block;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 500;
  background: rgba(255, 255, 255, 0.08);
  color: var(--text-muted);
}}
.model-sub {{
  color: var(--text-dim);
  font-size: 11px;
  margin-top: 2px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
td.effort {{
  text-align: center;
  color: var(--text-muted);
  font-size: 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
.tier-pill {{
  display: inline-block;
  padding: 2px 8px;
  border-radius: 9999px;
  font-size: 11px;
  font-weight: 600;
}}
.tier-pill.t1 {{
  color: var(--t1-fg);
  background: var(--t1-bg);
  border: 1px solid var(--t1-border);
}}
.tier-pill.t2 {{
  color: var(--t2-fg);
  background: var(--t2-bg);
  border: 1px solid var(--t2-border);
}}
.tier-pill.t3 {{
  color: var(--t3-fg);
  background: var(--t3-bg);
  border: 1px solid var(--t3-border);
}}
.score-box {{
  display: flex;
  align-items: baseline;
  gap: 6px;
}}
.score-val {{
  font-size: 15px;
  font-weight: 700;
  color: #fff;
}}
.ci {{
  color: var(--text-dim);
  font-size: 11px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
.score-arrow {{
  color: var(--accent);
  font-size: 11px;
  font-weight: bold;
}}
.profile-badge {{
  display: inline-block;
  padding: 3px 8px;
  border-radius: 4px;
  background: #090d16;
  border: 1px solid var(--surface-border);
  color: #cbd5e1;
  font-size: 12px;
}}
.metric-cell {{
  text-align: right;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  color: #cbd5e1;
}}
.col-detailed {{
  display: none;
}}
body.show-detailed .col-detailed {{
  display: table-cell;
}}
.drawer-row {{
  display: none;
  background: #0c121e;
}}
.drawer-row.open {{
  display: table-row;
}}
.drawer-content {{
  padding: 16px 20px 20px;
  border-bottom: 1px solid var(--surface-border);
}}
.drawer-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}}
.drawer-card {{
  background: #111827;
  border: 1px solid var(--surface-border);
  border-radius: 6px;
  padding: 8px 12px;
}}
.drawer-card-label {{
  color: var(--text-dim);
  font-size: 11px;
  margin-bottom: 2px;
}}
.drawer-card-val {{
  font-size: 16px;
  font-weight: 700;
  color: var(--text-main);
}}
.badcase-card {{
  background: #111827;
  border: 1px solid var(--surface-border);
  border-radius: 6px;
  padding: 10px 14px;
  margin-top: 10px;
  font-size: 12px;
}}
.badcase-title {{
  color: #f87171;
  font-weight: 600;
  margin-bottom: 4px;
}}
.badcase-body {{
  color: var(--text-muted);
  line-height: 1.5;
  white-space: pre-wrap;
}}
footer {{
  margin-top: 32px;
  color: var(--text-dim);
  font-size: 12px;
  display: flex;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 8px;
}}
footer a {{
  color: var(--text-muted);
  text-decoration: none;
}}
footer a:hover {{
  text-decoration: underline;
}}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="header-top">
      <h1>TshuBench · 天书榜</h1>
      <span class="sub-desc">量化大模型中文工程回答中的表达负担与黑话退化</span>
    </div>
    <div class="pills-row">
      <div class="stat-pill">⚡ 跨题分半信度 <b>r_SB = 0.968</b></div>
      <div class="stat-pill">🎯 <b>17 款主流模型路由</b></div>
      <div class="stat-pill">🧪 <b>36 题真实工程高频问答</b></div>
      <div class="stat-pill">⚖️ <b>0 裁判模型 · 100% 确定性语法树与词法统计</b></div>
    </div>
    <div class="notice-card">
      <span>💡</span>
      <span><b>评分导读</b>：本榜采用反向扣分制（<b>↓ 分数越低越讲人话</b>）。0 分仅代表未触发预设六类文风病灶，不代表技术内容一定正确。点击任意模型行可展开病灶雷达与典型样本。</span>
    </div>
  </header>

  <div class="controls-card">
    <div class="search-row">
      <div class="search-wrap">
        <input type="text" id="search-input" class="search-input" placeholder="🔍 搜索模型名称、厂商、底层路由代号..." />
      </div>
      <label class="toggle-label">
        <input type="checkbox" id="toggle-detailed" />
        <span>展开 6 项病灶扣分明细</span>
      </label>
    </div>
    <div class="filters-wrap">
      <div class="btn-group" id="tier-group">
        <button class="filter-btn active" data-tier="all">全部梯队 ({len(ranked)})</button>
        <button class="filter-btn" data-tier="T1 · 人话级">🟢 T1 · 人话级</button>
        <button class="filter-btn" data-tier="T2 · 黑话级">🟡 T2 · 黑话级</button>
        <button class="filter-btn" data-tier="T3 · 天书级">🔴 T3 · 天书级</button>
      </div>
      <div class="btn-group" id="provider-group">
        <button class="filter-btn active" data-provider="all">全部厂商</button>
        <button class="filter-btn" data-provider="Google">Google</button>
        <button class="filter-btn" data-provider="Anthropic">Anthropic</button>
        <button class="filter-btn" data-provider="OpenAI">OpenAI</button>
        <button class="filter-btn" data-provider="Domestic">国产主流</button>
      </div>
    </div>
  </div>

  <div class="table-card">
    <div class="table-scroll">
      <table id="leaderboard-table">
        <thead>
          <tr>
            <th data-sort="rank" class="sort-active" style="width: 44px; text-align: center;">#</th>
            <th data-sort="name">模型与厂商</th>
            <th>思考档位</th>
            <th data-sort="tier">梯队</th>
            <th data-sort="tshu_score">天书指数 (95% CI) ↓</th>
            <th>开口扣分</th><th>正文扣分</th>
            <th data-sort="profile">主导病灶画像</th>
            {head_cells}
          </tr>
        </thead>
        <tbody id="leaderboard-body">
          {''.join(static_rows)}
        </tbody>
      </table>
    </div>
  </div>

  <footer>
    <div>基于 V3 评测架构 · 生成时间：{now_str} · 全量 612 条真实工程样本</div>
    <div><a href="https://github.com/DBinK/TshuBench" target="_blank" rel="noopener">GitHub 仓库 (DBinK/TshuBench)</a> ｜ <a href="https://github.com/DBinK/TshuBench/blob/main/docs/%E6%8A%80%E6%9C%AF%E6%96%B9%E6%A1%88.md" target="_blank">评测技术方案</a></div>
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
    detailed: false,
    sortKey: 'tshu_score',
    sortAsc: true,
    expanded: new Set()
  }};

  const domesticProviders = new Set(['Alibaba', 'Zhipu', 'Moonshot', 'DeepSeek', 'MiniMax']);

  const tbody = document.getElementById('leaderboard-body');
  const searchInput = document.getElementById('search-input');
  const toggleDetailed = document.getElementById('toggle-detailed');
  const tierBtns = document.querySelectorAll('#tier-group .filter-btn');
  const providerBtns = document.querySelectorAll('#provider-group .filter-btn');
  const ths = document.querySelectorAll('#leaderboard-table th');

  function render() {{
    document.body.classList.toggle('show-detailed', state.detailed);

    let filtered = rawData.filter(item => {{
      if (state.tier !== 'all' && item.tier !== state.tier) return false;
      if (state.provider === 'Domestic' && !domesticProviders.has(item.provider)) return false;
      if (state.provider !== 'all' && state.provider !== 'Domestic' && item.provider !== state.provider) return false;
      if (state.search) {{
        const q = state.search.toLowerCase();
        const match = item.name.toLowerCase().includes(q) ||
                      item.model.toLowerCase().includes(q) ||
                      item.provider.toLowerCase().includes(q) ||
                      item.profile.toLowerCase().includes(q);
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
      }} else if (state.sortKey === 'open_score') {{
        valA = a.open_score; valB = b.open_score;
      }} else if (state.sortKey === 'body_score') {{
        valA = a.body_score; valB = b.body_score;
      }} else if (state.sortKey === 'profile') {{
        valA = a.profile.localeCompare(b.profile); valB = 0;
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
      tr.className = 'model-row';
      const isExpanded = state.expanded.has(item.model);

      const medal = item.rank === 1 ? '🥇 ' : (item.rank === 2 ? '🥈 ' : (item.rank === 3 ? '🥉 ' : ''));
      const routeBadge = item.route_note ? ` <span class="ci">(${{item.route_note}})</span>` : '';

      let detailedTds = '';
      if (item.metrics) {{
        detailedTds += `<td class="col-detailed metric-cell">${{item.open_score.toFixed(2)}}</td>`;
        detailedTds += `<td class="col-detailed metric-cell">${{item.body_score.toFixed(2)}}</td>`;
        for (const k of ['cci', 'fmt', 'enu', 'tic', 'cvd', 'hed']) {{
          detailedTds += `<td class="metric-cell col-detailed">${{item.metrics[k] !== undefined ? item.metrics[k].toFixed(1) : '-'}}</td>`;
        }}
      }}

      tr.innerHTML = `
        <td class="rank">${{medal}}${{item.rank}}</td>
        <td class="model">
          <div class="model-title-wrap">
            <strong class="model-title">${{item.name}}</strong>
            <span class="provider-tag">${{item.provider}}</span>
          </div>
          <div class="model-sub"><code>${{item.model}}</code>${{routeBadge}}</div>
        </td>
        <td class="effort">${{item.effort}}</td>
        <td class="tier"><span class="tier-pill ${{item.tier_cls}}">${{item.tier}}</span></td>
        <td class="score">
          <div class="score-box">
            <b class="score-val">${{item.tshu_score.toFixed(2)}}</b>
            <span class="ci">[${{item.ci_lower.toFixed(1)}}, ${{item.ci_upper.toFixed(1)}}]</span>
            <span class="score-arrow" title="扣分制，越低越好">↓</span>
          </div>
        </td>
        ${{detailedTds}}
        <td class="profile"><span class="profile-badge">${{item.profile}}</span></td>
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
        drawerTr.className = 'drawer-row open';
        let cardsHtml = '';
        const metricLabels = {{
          cci: 'CCI 长链绕口',
          fmt: 'FMT 排版强迫',
          enu: 'ENU 报菜名',
          tic: 'TIC 套话模板',
          cvd: 'CVD 首句决断',
          hed: 'HED 防御叠甲'
        }};
        for (const [k, lbl] of Object.entries(metricLabels)) {{
          const val = item.metrics[k] !== undefined ? item.metrics[k] : 0;
          cardsHtml += `
            <div class="drawer-card">
              <div class="drawer-card-label">${{lbl}}</div>
              <div class="drawer-card-val">${{val.toFixed(1)}}</div>
            </div>
          `;
        }}

        let badcasesHtml = '';
        if (item.worst && item.worst.length > 0) {{
          badcasesHtml = '<div style="margin-top: 12px; font-weight: 600; color: #cbd5e1; font-size: 12px;">典型最高扣分题 (Badcases)：</div>';
          item.worst.forEach(w => {{
            badcasesHtml += `
              <div class="badcase-card">
                <div class="badcase-title">题目 [${{w.task_id}}] · 扣分 ${{w.score}}</div>
                <div class="badcase-body">${{w.response}}</div>
              </div>
            `;
          }});
        }}

        drawerTr.innerHTML = `
          <td colspan="15" style="padding: 0;">
            <div class="drawer-content">
              <div style="font-weight: 600; color: #cbd5e1; font-size: 13px; margin-bottom: 8px;">${{item.name}} · 病灶六维扣分明细：</div>
              <div class="drawer-grid">${{cardsHtml}}</div>
              ${{badcasesHtml}}
            </div>
          </td>
        `;
        tbody.appendChild(drawerTr);
      }}
    }});

    // Update th active state
    ths.forEach(th => {{
      const key = th.getAttribute('data-sort');
      if (key === state.sortKey) {{
        th.classList.add('sort-active');
        th.textContent = th.textContent.replace(/ [▲▼]/, '') + (state.sortAsc ? ' ▲' : ' ▼');
      }} else {{
        th.classList.remove('sort-active');
        th.textContent = th.textContent.replace(/ [▲▼]/, '');
      }}
    }});
  }}

  // Bind controls
  searchInput.addEventListener('input', (e) => {{
    state.search = e.target.value.trim();
    render();
  }});

  toggleDetailed.addEventListener('change', (e) => {{
    state.detailed = e.target.checked;
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
