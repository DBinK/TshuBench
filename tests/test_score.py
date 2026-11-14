from harness.score import compute_bootstrap_ci


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
