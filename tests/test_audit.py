"""统计体检模块（harness.audit）单元测试。"""

from pathlib import Path

from harness.audit import compute_audit_report, jacobi_pca, paired_permutation_p, pearson_r


def test_pearson_r_basic() -> None:
    assert abs(pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) - 1.0) < 1e-9
    assert abs(pearson_r([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) + 1.0) < 1e-9
    assert pearson_r([1.0], [1.0]) == 0.0


def test_jacobi_pca_diagonal_and_correlated() -> None:
    corr = [
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    comps = jacobi_pca(corr)
    assert len(comps) == 3
    assert abs(comps[0][0] - 1.8) < 1e-6
    assert abs(sum(eig for eig, _ in comps) - 3.0) < 1e-6


def test_paired_permutation_p_detects_clear_difference() -> None:
    obs_sig, p_sig = paired_permutation_p([2.0] * 16, n_perm=1000, seed=42)
    assert obs_sig == 2.0
    assert p_sig < 0.01

    obs_null, p_null = paired_permutation_p([1.0, -1.0] * 8, n_perm=1000, seed=42)
    assert obs_null == 0.0
    assert p_null == 1.0


def test_compute_audit_report_on_demo_runs() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    report = compute_audit_report(
        repo_root / "tasks" / "eval_suite_v2.jsonl",
        [repo_root / "runs" / "demo_tidy.jsonl", repo_root / "runs" / "demo_slop.jsonl"],
        n_perm=500,
    )
    assert report["n_models"] == 2
    assert report["n_tasks"] >= 2  # type: ignore[operator]
    assert len(report["adjacent_pairs"]) == 1  # type: ignore[arg-type]
    assert len(report["pca"]) == 2  # type: ignore[arg-type]
