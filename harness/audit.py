"""TshuBench 零人工成本统计体检：基线对照、方差贡献、PCA、分半信度与配对置换检验。

用法：
    uv run python -m harness.cli audit runs/real/*.jsonl
"""

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path

from rich import print as rprint
from rich.table import Table

from harness.metrics import evaluate, opening_entropy
from harness.models import WEIGHTS, ScoredSample, Task
from harness.score import load_samples, load_tasks
from harness.text import char_len, han_len, split_sentences

METRIC_KEYS = tuple(WEIGHTS.keys())


@dataclass(frozen=True)
class SampleStats:
    model: str
    task_id: str
    tshu: float
    metrics: dict[str, float]
    char_len: int
    han_len: int
    avg_sent_len: float


@dataclass(frozen=True)
class PairedResult:
    rank_a: int
    model_a: str
    rank_b: int
    model_b: str
    diff: float
    p_value: float


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """计算两组序列的 Pearson 相关系数。"""
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def jacobi_pca(corr: list[list[float]], max_iter: int = 100) -> list[tuple[float, list[float]]]:
    """用 Jacobi 特征分解计算对称相关矩阵的主成分（特征值, 特征向量）。"""
    n = len(corr)
    mat = [row[:] for row in corr]
    vecs = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_iter):
        p, q, max_val = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                val = abs(mat[i][j])
                if val > max_val:
                    max_val, p, q = val, i, j
        if max_val < 1e-10:
            break

        if abs(mat[p][p] - mat[q][q]) < 1e-12:
            theta = math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * mat[p][q], mat[q][q] - mat[p][p])
        c, s = math.cos(theta), math.sin(theta)

        for i in range(n):
            if i != p and i != q:
                aip, aiq = mat[i][p], mat[i][q]
                mat[i][p] = mat[p][i] = c * aip - s * aiq
                mat[i][q] = mat[q][i] = s * aip + c * aiq

        app, aqq, apq = mat[p][p], mat[q][q], mat[p][q]
        mat[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
        mat[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
        mat[p][q] = mat[q][p] = 0.0

        for i in range(n):
            vip, viq = vecs[i][p], vecs[i][q]
            vecs[i][p] = c * vip - s * viq
            vecs[i][q] = s * vip + c * viq

    pairs = [(mat[i][i], [vecs[r][i] for r in range(n)]) for i in range(n)]
    pairs.sort(key=lambda item: item[0], reverse=True)
    return pairs


def paired_permutation_p(
    diffs: list[float], n_perm: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """对逐题差值执行符号置换检验，返回 (均值差, 双侧 p 值)。"""
    if not diffs:
        return 0.0, 1.0
    obs = statistics.mean(diffs)
    if obs == 0.0:
        return 0.0, 1.0
    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        perm_mean = statistics.mean(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(perm_mean) >= abs(obs):
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def _collect_stats(
    tasks: dict[str, Task], runs: list[str | Path]
) -> tuple[list[SampleStats], dict[str, dict[str, SampleStats]]]:
    grouped = load_samples(runs)
    all_rows: list[SampleStats] = []
    by_model: dict[str, dict[str, SampleStats]] = {}

    for model, samples in grouped.items():
        known = [s for s in samples if s.task_id in tasks]
        if not known:
            continue
        ent = opening_entropy([s.response for s in known])
        pen = max(0.0, 1.0 - ent)
        scored: list[ScoredSample] = [evaluate(tasks[s.task_id], s, pen) for s in known]

        model_map: dict[str, SampleStats] = {}
        for sc in scored:
            sents = split_sentences(sc.sample.response)
            avg_sl = statistics.mean([han_len(x) for x in sents]) if sents else 0.0
            row = SampleStats(
                model=model,
                task_id=sc.task.id,
                tshu=sc.metrics.tshu_score,
                metrics={k: float(getattr(sc.metrics, k)) for k in METRIC_KEYS},
                char_len=char_len(sc.sample.response),
                han_len=han_len(sc.sample.response),
                avg_sent_len=avg_sl,
            )
            all_rows.append(row)
            model_map[sc.task.id] = row
        by_model[model] = model_map

    return all_rows, by_model


def compute_audit_report(
    tasks_path: str | Path, runs: list[str | Path], n_perm: int = 10000
) -> dict[str, object]:
    """计算全套零人工成本统计体检结果。"""
    tasks = load_tasks(tasks_path)
    all_rows, by_model = _collect_stats(tasks, runs)
    if len(by_model) < 2:
        raise SystemExit("至少需要 2 个模型样本才能执行统计体检")

    task_ids = list(tasks.keys())
    common_task_ids = [tid for tid in task_ids if all(tid in mp for mp in by_model.values())]
    if len(common_task_ids) < 2:
        raise SystemExit("模型间共同覆盖的题目少于 2 道，无法执行配对统计体检")
    task_ids = common_task_ids
    models_sorted = sorted(
        by_model.keys(),
        key=lambda m: statistics.mean(by_model[m][t].tshu for t in task_ids),
    )

    model_means: list[dict[str, float]] = []
    for m in models_sorted:
        recs = [by_model[m][t] for t in task_ids]
        entry: dict[str, float] = {
            "tshu": statistics.mean(r.tshu for r in recs),
            "char_len": statistics.mean(r.char_len for r in recs),
            "han_len": statistics.mean(r.han_len for r in recs),
            "avg_sent_len": statistics.mean(r.avg_sent_len for r in recs),
        }
        for k in METRIC_KEYS:
            entry[k] = statistics.mean(r.metrics[k] for r in recs)
        model_means.append(entry)

    # 1. 简单基线对照
    baseline_corr = {
        "sample_level": {
            feat: round(pearson_r([r.tshu for r in all_rows], [getattr(r, feat) for r in all_rows]), 4)
            for feat in ("char_len", "han_len", "avg_sent_len")
        },
        "model_level": {
            feat: round(
                pearson_r([r["tshu"] for r in model_means], [r[feat] for r in model_means]), 4
            )
            for feat in ("char_len", "han_len", "avg_sent_len")
        },
    }

    # 2. 实际方差贡献与相关矩阵 / PCA
    stds = {k: statistics.stdev([r[k] for r in model_means]) for k in METRIC_KEYS}
    total_wstd = sum(WEIGHTS[k] * stds[k] for k in METRIC_KEYS) or 1.0
    variance_contrib = {
        k: {
            "nominal_weight": WEIGHTS[k],
            "model_std": round(stds[k], 2),
            "weighted_std": round(WEIGHTS[k] * stds[k], 2),
            "variance_share": round(WEIGHTS[k] * stds[k] / total_wstd, 4),
        }
        for k in METRIC_KEYS
    }

    corr_matrix = [
        [
            round(pearson_r([r[k1] for r in model_means], [r[k2] for r in model_means]), 4)
            for k2 in METRIC_KEYS
        ]
        for k1 in METRIC_KEYS
    ]
    pca_raw = jacobi_pca(corr_matrix)
    total_eig = sum(max(0.0, eig) for eig, _ in pca_raw) or 1.0
    pca = [
        {
            "component": f"PC{idx + 1}",
            "explained_ratio": round(max(0.0, eig) / total_eig, 4),
            "loadings": {k: round(vec[i], 3) for i, k in enumerate(METRIC_KEYS)},
        }
        for idx, (eig, vec) in enumerate(pca_raw[:2])
    ]

    # 3. 奇偶分半信度
    odd_ids = task_ids[0::2]
    even_ids = task_ids[1::2]
    split_half: dict[str, dict[str, float]] = {}
    for key in ("tshu", *METRIC_KEYS):
        odd_vals = [
            statistics.mean(
                by_model[m][t].tshu if key == "tshu" else by_model[m][t].metrics[key] for t in odd_ids
            )
            for m in models_sorted
        ]
        even_vals = [
            statistics.mean(
                by_model[m][t].tshu if key == "tshu" else by_model[m][t].metrics[key] for t in even_ids
            )
            for m in models_sorted
        ]
        r_half = pearson_r(odd_vals, even_vals)
        r_sb = (2.0 * r_half / (1.0 + r_half)) if (1.0 + r_half) != 0 else 0.0
        split_half[key] = {"r": round(r_half, 4), "r_sb": round(r_sb, 4)}

    # 4. 配对置换检验（相邻名次 & 相对 #1）
    adjacent_pairs: list[PairedResult] = []
    for idx in range(len(models_sorted) - 1):
        ma, mb = models_sorted[idx], models_sorted[idx + 1]
        diffs = [by_model[mb][t].tshu - by_model[ma][t].tshu for t in task_ids]
        obs, p_val = paired_permutation_p(diffs, n_perm=n_perm, seed=42 + idx)
        adjacent_pairs.append(
            PairedResult(idx + 1, ma, idx + 2, mb, round(obs, 2), round(p_val, 4))
        )

    vs_top_pairs: list[PairedResult] = []
    top_model = models_sorted[0]
    for idx, mb in enumerate(models_sorted[1:], start=2):
        diffs = [by_model[mb][t].tshu - by_model[top_model][t].tshu for t in task_ids]
        obs, p_val = paired_permutation_p(diffs, n_perm=n_perm, seed=100 + idx)
        vs_top_pairs.append(PairedResult(1, top_model, idx, mb, round(obs, 2), round(p_val, 4)))

    return {
        "n_models": len(models_sorted),
        "n_tasks": len(task_ids),
        "n_samples": len(all_rows),
        "models_sorted": models_sorted,
        "baseline_corr": baseline_corr,
        "variance_contrib": variance_contrib,
        "corr_matrix": corr_matrix,
        "pca": pca,
        "split_half": split_half,
        "adjacent_pairs": adjacent_pairs,
        "vs_top_pairs": vs_top_pairs,
    }


def print_audit_report(report: dict[str, object]) -> None:
    """在终端以 Rich 表格打印统计体检结果。"""
    rprint(
        f"[bold]TshuBench 零人工成本统计体检[/bold] · "
        f"{report['n_models']} 模型 × {report['n_tasks']} 题（共 {report['n_samples']} 样本）"
    )

    bc: dict[str, dict[str, float]] = report["baseline_corr"]  # type: ignore[assignment]
    t_base = Table(title="1. 简单基线相关性对照（Pearson r）")
    t_base.add_column("特征")
    t_base.add_column("单题级 (N=612)", justify="right")
    t_base.add_column("模型级 (N=17)", justify="right")
    for feat in ("char_len", "han_len", "avg_sent_len"):
        t_base.add_row(feat, f"{bc['sample_level'][feat]:+.4f}", f"{bc['model_level'][feat]:+.4f}")
    rprint(t_base)

    vc: dict[str, dict[str, float]] = report["variance_contrib"]  # type: ignore[assignment]
    sh: dict[str, dict[str, float]] = report["split_half"]  # type: ignore[assignment]
    t_var = Table(title=f"2. 子指标信度与实际方差贡献（总分 r_SB={sh['tshu']['r_sb']:.4f}）")
    t_var.add_column("指标")
    t_var.add_column("标称权重", justify="right")
    t_var.add_column("模型间标准差", justify="right")
    t_var.add_column("实际方差贡献", justify="right")
    t_var.add_column("分半信度 r_SB", justify="right")
    for k in METRIC_KEYS:
        item = vc[k]
        t_var.add_row(
            k.upper(),
            f"{item['nominal_weight']:.0%}",
            f"{item['model_std']:.2f}",
            f"{item['variance_share']:.1%}",
            f"{sh[k]['r_sb']:.4f}",
        )
    rprint(t_var)

    t_pca = Table(title="3. 标准化子指标主成分（PCA 前两维）")
    t_pca.add_column("主成分")
    t_pca.add_column("解释方差", justify="right")
    t_pca.add_column("载荷")
    for comp in report["pca"]:  # type: ignore[attr-defined]
        loads = " ".join(f"{k.upper()}:{v:+.2f}" for k, v in comp["loadings"].items())
        t_pca.add_row(comp["component"], f"{comp['explained_ratio']:.1%}", loads)
    rprint(t_pca)

    adj: list[PairedResult] = report["adjacent_pairs"]  # type: ignore[assignment]
    sig_adj = [p for p in adj if p.p_value < 0.05]
    top_p: list[PairedResult] = report["vs_top_pairs"]  # type: ignore[assignment]
    first_sig = next((p for p in top_p if p.p_value < 0.05), None)
    rprint(
        f"[bold]4. 配对置换检验（36 题，10000 次符号置换）：[/bold]"
        f"相邻名次显著差异 {len(sig_adj)}/{len(adj)} 对；"
        + (
            f"相对 #1 首个显著拉开差距为 #{first_sig.rank_b} {first_sig.model_b} "
            f"(+{first_sig.diff:.2f}, p={first_sig.p_value:.4f})"
            if first_sig
            else "无模型显著高于 #1"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TshuBench 零人工成本统计体检")
    parser.add_argument("--tasks", default="tasks/eval_suite_v2.jsonl")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()
    report = compute_audit_report(args.tasks, args.runs, n_perm=args.n_perm)
    print_audit_report(report)


if __name__ == "__main__":
    main()
