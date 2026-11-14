from harness.score import compute_bootstrap_ci, primary_profile


def test_primary_profile_picks_top_metrics():
    metrics = {"cvd": 12.0, "hed": 25.5, "enu": 30.1, "ssf": 0.0, "af": 0.0, "cci": 1.0, "pni": 0.0, "vtr": 5.0}
    profile = primary_profile(metrics)
    assert "ENU 报菜名堆砌(30.1)" in profile
    assert "HED 防御性叠甲(25.5)" in profile
    assert "CVD" not in profile


def test_bootstrap_ci_single_value():
    lower, upper = compute_bootstrap_ci([10.0])
    assert lower == 10.0
    assert upper == 10.0


def test_bootstrap_ci_identical_values():
    lower, upper = compute_bootstrap_ci([5.0, 5.0, 5.0, 5.0])
    assert lower == 5.0
    assert upper == 5.0


def test_bootstrap_ci_range():
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    lower, upper = compute_bootstrap_ci(scores, n_bootstrap=500, seed=42)
    assert 1.0 <= lower <= 5.5
    assert 5.5 <= upper <= 10.0
    assert lower <= upper
