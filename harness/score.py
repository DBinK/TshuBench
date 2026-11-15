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
    "cvd": "CVD 首句决断延迟",
    "hed": "HED 防御性叠甲",
    "enu": "ENU 报菜名堆砌",
    "cci": "CCI 句法负荷",
    "pni": "PNI 欧化名物化",
    "vtr": "VTR 语言抽动",
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
        averages["enu"] * WEIGHTS["enu"]
        + averages["cci"] * WEIGHTS["cci"]
        + averages["pni"] * WEIGHTS["pni"]
        + averages["vtr"] * WEIGHTS["vtr"],
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
        "分数越低越少见这六类文风病灶，0 分仅表示未检出预设惩罚模式（不代表事实正确或回答完整）。V2.1 权重按「开口痛快度 26%（CVD 12% + HED 14%）+ 正文阅读负担 74%（ENU 22% + CCI 22% + PNI 14% + VTR 16%）」拆分；梯队按点估计划定，[下限, 上限] 为 1000 次 Bootstrap 抽样估计的 95% 置信区间，区间重叠的模型之间不应解读为稳定名次差异。",
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


def render_html(reports: list[ModelReport]) -> str:
    ranked = sorted(reports, key=lambda r: r.tshu_score)
    head = "".join(f"<th>{html.escape(v)}</th>" for v in METRIC_LABELS.values())
    rows = []
    for i, report in enumerate(ranked, start=1):
        cells = "".join(f"<td>{report.metrics[k]:.1f}</td>" for k in METRIC_LABELS)
        width = min(100.0, report.tshu_score)
        effort_text = html.escape(report.reasoning_effort or "默认")
        tier_text = html.escape(report.tier or classify_tier(report.tshu_score))
        ci_text = f"<span class=ci>[{report.ci_lower:.1f}, {report.ci_upper:.1f}]</span>"
        profile_text = html.escape(primary_profile(report.metrics))
        model_label = html.escape(report.model)
        if report.route_note:
            model_label += f" <span class=ci>({html.escape(report.route_note)})</span>"
        rows.append(
            f"<tr><td class=rank>{i}</td><td class=model>{model_label}</td>"
            f"<td class=effort>{effort_text}</td><td class=tier>{tier_text}</td>"
            f"<td class=score><span class=bar style='width:{width:.1f}%'></span>"
            f"<b>{report.tshu_score:.1f}</b> {ci_text}</td>"
            f"<td>{report.open_score:.2f}</td><td>{report.body_score:.2f}</td>"
            f"<td class=profile>{profile_text}</td>{cells}</tr>"
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TshuBench · 天书榜</title>
<style>
:root{{--bg:#0d1117;--fg:#e6edf3;--muted:#8b949e;--line:#21262d;--hot:#f85149;--cool:#3fb950}}
*{{box-sizing:border-box}}
body{{margin:0;padding:48px 24px;background:var(--bg);color:var(--fg);
font:14px/1.6 -apple-system,"PingFang SC","Noto Sans CJK SC",sans-serif}}
main{{max-width:1180px;margin:0 auto}}
h1{{font-size:28px;margin:0 0 6px}}
p.sub{{color:var(--muted);margin:0 0 28px}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{color:var(--muted);font-weight:500;font-size:12px}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(4),td:nth-child(4){{text-align:left}}
th:nth-child(3){{text-align:center}}
td.rank{{color:var(--muted)}}
td.model{{font-weight:600}}
td.effort{{text-align:center;color:var(--muted);font-size:12px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
td.tier{{font-size:12px;color:var(--fg)}}
td.score{{position:relative;min-width:150px}}
td.score .bar{{position:absolute;left:0;top:50%;height:6px;transform:translateY(-50%);
background:linear-gradient(90deg,var(--cool),var(--hot));opacity:.35;border-radius:3px}}
td.score b{{position:relative}}
td.score span.ci{{position:relative;color:var(--muted);font-size:11px;margin-left:4px}}
td.profile{{text-align:left;color:var(--muted);font-size:12px}}
footer{{color:var(--muted);margin-top:28px;font-size:12px}}
</style>
<main>
<h1>TshuBench · 天书榜</h1>
<p class=sub>中文工程回答的文风病灶观察榜。分数越低越少见黑话、套话与结构膨胀，0 分表示未检出六类预设惩罚（不代表知识正确）。基于 36 道日常工程题（V2.1 权重：开口痛快度 26% + 正文阅读负担 74%），梯队为点估计分档，CI 重叠区间请结合病灶画像判断。</p>
<table>
<thead><tr><th>#</th><th>模型</th><th>思考档位</th><th>梯队</th><th>TshuScore</th><th>开口扣分</th><th>正文扣分</th><th>主导病灶画像</th>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<footer>生成于 {datetime.now():%Y-%m-%d %H:%M} · {len(ranked)} 个模型 · 确定性 AST + jieba 词法打分，无裁判模型</footer>
</main>
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
