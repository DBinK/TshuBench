from harness.models import Sample, Task, classify_tier
from harness.score import compute_bootstrap_ci, primary_profile, render_html, render_markdown, score_model


def test_classify_tier():
    assert classify_tier(9.99) == "T1 · 人话级"
    assert classify_tier(10.0) == "T2 · 黑话级"
    assert classify_tier(13.99) == "T2 · 黑话级"
    assert classify_tier(14.0) == "T3 · 天书级"


def test_primary_profile_picks_top_metrics():
    metrics = {"cci": 1.0, "fmt": 0.0, "enu": 30.1, "tic": 5.0, "cvd": 12.0, "hed": 25.5}
    profile = primary_profile(metrics)
    assert "ENU 报菜名与堆砌(30.1)" in profile
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


def test_score_model_records_reasoning_effort():
    task = Task(
        id="t1",
        scenario="code_review_bug_triage",
        register="chat",
        trap_type="explicit_bound",
        max_chars=120,
        prompt="测试问题？",
        verdict_anchors=["直接改"],
        actionable_entities=["改这里"],
    )
    tasks = {"t1": task}
    sample_low = Sample(task_id="t1", model="custom/sonnet", response="直接改。", reasoning_effort="low")
    sample_legacy = Sample(task_id="t1", model="auto/gemini-4-argon", response="直接改。")
    r_low = score_model("custom/sonnet", [sample_low], tasks, n_bootstrap=50)
    r_legacy = score_model("auto/gemini-4-argon", [sample_legacy], tasks, n_bootstrap=50)
    assert r_low.reasoning_effort == "low"
    assert r_legacy.reasoning_effort == "medium"
    md = render_markdown([r_low, r_legacy])
    html_doc = render_html([r_low, r_legacy])
    assert "| 思考档位 |" in md
    assert "| 开口扣分 | 正文扣分 |" in md
    assert "- 思考档位：low" in md
    assert "<th>思考档位</th>" in html_doc
    assert "<th>开口扣分</th><th>正文扣分</th>" in html_doc
    assert "<td class=effort>medium</td>" in html_doc
