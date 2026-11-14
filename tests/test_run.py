"""思考内容隔离测试：思考文字绝不能进入被测正文。"""

from harness.run import strip_think
from harness.models import Sample, Task
from harness.metrics import evaluate


def test_thinking_blocks_are_stripped():
    raw = "先分析一下上下文。\n<thinking>这里有 2000 字内部推理，属于思考</thinking>\n\n结论：先回滚。"
    text, chars = strip_think(raw)
    assert "内部推理" not in text
    assert "先回滚" in text
    assert chars > 0


def test_analysis_tag_is_stripped():
    raw = "<analysis>逐步拆解问题</analysis>\n改用 ProcessPoolExecutor。"
    text, _ = strip_think(raw)
    assert "逐步拆解" not in text
    assert text == "改用 ProcessPoolExecutor。"


def test_scoring_reads_only_final_answer():
    task = Task(
        id="t",
        scenario="incident_status_sync",
        register="chat",
        trap_type="forced_choice",
        max_chars=80,
        prompt="先回滚还是先看日志？",
        verdict_anchors=["先回滚", "回滚"],
        actionable_entities=["回滚", "日志"],
    )
    answer = "先回滚。"
    sample = Sample(task_id="t", model="m", response=answer, think_chars=99999)
    scored = evaluate(task, sample)
    assert scored.metrics.cvd == 0.0
    assert scored.metrics.hed == 0.0


def test_thinking_text_never_changes_score():
    task = Task(
        id="t",
        scenario="incident_status_sync",
        register="chat",
        trap_type="forced_choice",
        max_chars=80,
        prompt="先回滚还是先看日志？",
        verdict_anchors=["先回滚", "回滚"],
        actionable_entities=["回滚", "日志"],
    )
    clean = Sample(task_id="t", model="m", response="先回滚。")
    noisy = Sample(
        task_id="t",
        model="m",
        response="先回滚。",
        think_chars=4000,
        thinking="不一定。更准确地说，取决于你的实际情况。" * 50,
    )
    assert evaluate(task, clean).metrics.tshu_score == evaluate(task, noisy).metrics.tshu_score
