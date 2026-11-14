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
from harness.models import WEIGHTS, ModelReport, Sample, ScoredSample, Task

METRIC_LABELS = {
    "cvd": "CVD 首句决断延迟",
    "hed": "HED 防御性叠甲",
    "enu": "ENU 报菜名堆砌",
    "ssf": "SSF 奉承与骑墙",
    "af": "AF 回答失效",
    "cci": "CCI 句法负荷",
    "pni": "PNI 欧化名物化",
    "vtr": "VTR 语言抽动",
}


def load_tasks(path: str | Path) -> dict[str, Task]:
    path = Path(path)
    tasks: dict[str, Task] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        task = Task(
            id=raw["id"],
            scenario=raw["scenario"],
            register=raw["register"],
            trap_type=raw["trap_type"],
            max_chars=raw["max_chars"],
            prompt=raw["prompt"],
            verdict_anchors=raw["verdict_anchors"],
            actionable_entities=raw["actionable_entities"],
            fence_sitting_markers=raw.get("fence_sitting_markers", []),
            refutation_anchors=raw.get("refutation_anchors", []),
        )
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
            raw = json.loads(line)
            sample = Sample(
                task_id=raw["task_id"],
                model=raw["model"],
                response=raw["response"],
                think_chars=raw.get("think_chars", 0),
                latency_ms=raw.get("latency_ms", 0.0),
                thinking=raw.get("thinking", ""),
            )
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


def score_model(
    model: str, samples: list[Sample], tasks: dict[str, Task], n_bootstrap: int = 1000
) -> ModelReport:
    known = [s for s in samples if s.task_id in tasks]
    if len(known) < len(samples):
        logger.warning(f"{model} 有 {len(samples) - len(known)} 条样本题目不在题库中，已跳过")

    entropy = opening_entropy([s.response for s in known])
    opening_penalty = max(0.0, 1.0 - entropy)

    scored: list[ScoredSample] = []
    for sample in known:
        scored.append(evaluate(tasks[sample.task_id], sample, opening_penalty))

    if not scored:
        return ModelReport(model, 0, 0.0, {k: 0.0 for k in WEIGHTS}, entropy)

    averages = {
        key: round(mean(getattr(s.metrics, key) for s in scored), 2) for key in WEIGHTS
    }
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
    header = "| 排名 | 模型 | TshuScore (95% CI) | 主导病灶画像 | " + " | ".join(METRIC_LABELS.values()) + " |"
    sep = "| ---: | --- | ---: | --- | " + " | ".join("---:" for _ in METRIC_LABELS) + " |"
    rows = [
        "| {} | {} | **{:.2f}** [{:.2f}, {:.2f}] | {} | {} |".format(
            i,
            r.model,
            r.tshu_score,
            r.ci_lower,
            r.ci_upper,
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
        "分数越低越说人话，0 分对齐全对齐人类专家基线。[下限, 上限] 为 1000 次 Bootstrap 抽样估计的 95% 置信区间。",
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
            f"- 样本数：{report.n_samples}",
            f"- TshuScore：{report.tshu_score:.2f} (95% CI: [{report.ci_lower:.2f}, {report.ci_upper:.2f}])",
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
        ci_text = f"<span class=ci>[{report.ci_lower:.1f}, {report.ci_upper:.1f}]</span>"
        profile_text = html.escape(primary_profile(report.metrics))
        rows.append(
            f"<tr><td class=rank>{i}</td><td class=model>{html.escape(report.model)}</td>"
            f"<td class=score><span class=bar style='width:{width:.1f}%'></span>"
            f"<b>{report.tshu_score:.1f}</b> {ci_text}</td>"
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
main{{max-width:1080px;margin:0 auto}}
h1{{font-size:28px;margin:0 0 6px}}
p.sub{{color:var(--muted);margin:0 0 28px}}
table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{color:var(--muted);font-weight:500;font-size:12px}}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
td.rank{{color:var(--muted)}}
td.model{{font-weight:600}}
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
<p class=sub>量化后训练过对齐导致的可读性退化。分数越低越说人话，0 分对齐人类专家基线。</p>
<table>
<thead><tr><th>#</th><th>模型</th><th>TshuScore</th><th>主导病灶画像</th>{head}</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<footer>生成于 {datetime.now():%Y-%m-%d %H:%M} · {len(ranked)} 个模型 · 纯 Python 确定性打分，无裁判模型</footer>
</main>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="TshuBench 打分器")
    parser.add_argument("--tasks", default="tasks/eval_suite_v2.jsonl")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--out", default="reports")
    parser.add_argument("--html", default="web/index.html")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    grouped = load_samples(args.runs)
    reports = [score_model(model, samples, tasks) for model, samples in grouped.items()]
    if not reports:
        raise SystemExit("没有可打分的样本")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "model": r.model,
            "n_samples": r.n_samples,
            "tshu_score": r.tshu_score,
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

    html_path = Path(args.html)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(reports), encoding="utf-8")

    for report in sorted(reports, key=lambda r: r.tshu_score):
        rprint(f"[bold]{report.model:<20}[/bold] TshuScore [cyan]{report.tshu_score:>6.2f}[/cyan]  样本 {report.n_samples}")
    logger.success(f"已写出 {out/'scores.json'}、{out/'report.md'}、{html_path}")


if __name__ == "__main__":
    main()
