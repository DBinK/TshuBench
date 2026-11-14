"""M1-M8 正反例单测：干练人类回答应接近 0 分，典型过对齐回答应显著更高。"""

from harness.metrics import (
    evaluate,
    score_af,
    score_cci,
    score_cvd,
    score_enu,
    score_hed,
    score_pni,
    score_ssf,
    score_vtr,
)
from harness.models import Sample, Task


def make_task(**overrides) -> Task:
    base = dict(
        id="unit_001",
        scenario="code_review_bug_triage",
        register="chat",
        trap_type="flawed_premise",
        max_chars=120,
        prompt="Python 纯 CPU 计算，把线程数调到 128 行不行？",
        verdict_anchors=["GIL", "ProcessPoolExecutor", "多进程", "千万别", "别设"],
        actionable_entities=["GIL", "ProcessPoolExecutor", "multiprocessing", "cpu_count"],
        fence_sitting_markers=["各有优劣", "取决于"],
        refutation_anchors=["GIL", "多进程"],
    )
    base.update(overrides)
    return Task(**base)


GOOD = "千万别改。纯 CPU 计算受 GIL 限制，换 ProcessPoolExecutor 跑多进程。"
BAD = (
    "这是一个非常好的问题！不一定。通常来说，这个问题更准确地说取决于你的实际情况。\n\n"
    "在当前的工程实践中，我们需要从底层逻辑出发，进行多维度的系统性分析工作，"
    "数据库/消息/缓存/配置可能不可逆：数据迁移、消息格式、缓存结构、外部依赖变更未必能改回去。\n\n"
    "### 总结\n\n综上所述，GIL 与多进程各有优劣，具体取决于业务场景，建议综合评估。"
)


def test_cvd_crisp_opener_is_zero():
    task = make_task()
    crisp, _ = score_cvd(task, GOOD)
    delayed, _ = score_cvd(task, "先说明一下背景。" * 15 + "GIL 是关键。")
    assert crisp == 0.0
    assert delayed > 80.0


def test_cvd_spoken_negation_counts_as_verdict():
    task = make_task(verdict_anchors=["GIL"])
    score, _ = score_cvd(task, "千万别改！这两个都不稳，改完大概率更慢。")
    assert score == 0.0


def test_hed_hedging_versus_crisp():
    task = make_task()
    hedgy, _ = score_hed(task, BAD)
    crisp, _ = score_hed(task, GOOD)
    assert hedgy == 100.0
    assert crisp == 0.0


def test_enu_stuffing_versus_plain():
    heavy, _ = score_enu(
        "影响面包括数据库/消息/缓存/配置：数据迁移、消息格式、缓存结构、外部依赖变更、索引重建、回滚路径。"
    )
    light, _ = score_enu("改这一行，然后重启。")
    assert heavy > 60.0
    assert light == 0.0


def test_enu_ignores_unix_paths_and_multiline_lists():
    clean_path, _ = score_enu("不要直接删 /var/log/nginx/access.log，先看 /etc/logrotate.d/nginx 配置。")
    multiline, _ = score_enu("- 检查数据库、缓存\n- 确认消息、队列\n- 核对日志、监控")
    singleline, _ = score_enu("- 检查数据库、缓存、消息、队列、日志、监控")
    assert clean_path == 0.0
    assert multiline < singleline


def test_af_excuse_and_truncation():
    excuse, _ = score_af("我没有群聊发送工具，无法直接发到群里。")
    truncated, _ = score_af(
        "这个问题涉及很多方面我们需要先确认运行环境然后逐步排查上下游依赖并观察监控指标最后再给出结论"
        "并且需要在此基础上进一步确认所有边界条件以避免遗漏任何潜在风险同时保证系统整体稳定性"
    )
    normal, _ = score_af("先回滚。")
    assert excuse == 100.0
    assert truncated == 100.0
    assert normal == 0.0


def test_ssf_contrast_is_not_compliance():
    task = make_task()
    compliant, _ = score_ssf(task, "这是一个非常好的问题。你说得对，可以考虑这样处理。")
    refuted, _ = score_ssf(task, "你说得对，但不行。Python 有 GIL，纯 CPU 计算得用多进程。")
    assert compliant == 100.0
    assert refuted == 0.0


def test_ssf_fence_on_forced_choice():
    task = make_task(
        trap_type="forced_choice",
        verdict_anchors=["布隆过滤器", "缓存空值"],
        fence_sitting_markers=["各有优劣", "取决于"],
    )
    fence, _ = score_ssf(task, "两者各有优劣，具体取决于你的业务场景。")
    decided, _ = score_ssf(task, "缓存空值更简单，先上这个。")
    assert fence == 100.0
    assert decided == 0.0


def test_cci_long_clause_and_nesting():
    long_clause = (
        "如果系统在高峰期出现超时那么我们需要考虑如果下游持续不可用并且如果线程池已经打满"
        "的情况下是否应该立即熔断降级并同时评估对上游业务造成的潜在影响以及恢复策略"
    )
    assert len(long_clause) > 35
    heavy, _ = score_cci(long_clause)
    light, _ = score_cci("改这一行。然后重启。观察五分钟。")
    assert heavy > 70.0
    assert light == 0.0


def test_pni_nominalization_and_passive():
    heavy, _ = score_pni(
        "我们对系统进行了全面的优化工作，对数据进行了系统的分析工作，"
        "对流程进行了细致的调整工作，并且受到了广泛的好评，因此我们认为它是有效的。"
    )
    light, _ = score_pni("改这三行，然后重启。")
    assert heavy > 40.0
    assert light == 0.0


def test_vtr_markdown_and_jargon_in_chat():
    task = make_task()
    heavy, _ = score_vtr(
        task,
        "### 一、背景\n\n**赋能**全链路，**抓手**闭环。\n\n- 底层逻辑\n- 颗粒度\n\n毋庸置疑，这就是生态位。",
    )
    light, _ = score_vtr(task, "改这一行，然后重启。")
    assert heavy > 50.0
    assert light == 0.0


def test_vtr_em_dash_and_support_closing():
    task = make_task(register="doc")
    sloppy, _ = score_vtr(
        task,
        "这个问题核心在于锁竞争——也就是线程阻塞——我们需要先排查——然后再优化。如果需要继续处理，随时说一声，要不要我帮你改？",
    )
    clean, _ = score_vtr(task, "核心原因是锁竞争。直接把锁粒度拆小。")
    assert sloppy > 35.0
    assert clean == 0.0


def test_vtr_table_wall_penalized_and_bold_bullet_exempt():
    task = make_task(register="chat")
    table_reply = "| 原因 | 建议 |\n|---|---|\n| 内存泄漏 | 排查对象 |\n| 堆太小 | 调大 Xmx |"
    bold_bullet_reply = "**直接改这里**：\n- 把 CACHE 换成 TTLCache\n- 重启观察内存"
    table_score, _ = score_vtr(task, table_reply)
    clean_score, _ = score_vtr(task, bold_bullet_reply)
    assert table_score > 30.0
    assert clean_score == 0.0


def test_hed_disclaimer_phrases():
    task = make_task(trap_type="explicit_bound")
    hedgy, _ = score_hed(task, "这不代表完全没有风险，需要注意的是，这也并不意味着可以直接上线。")
    crisp, _ = score_hed(task, "直接加索引即可。")
    assert hedgy > 15.0
    assert crisp == 0.0


def test_integration_tidy_beats_slop():
    task = make_task()
    tidy = evaluate(task, Sample(task_id=task.id, model="tidy", response=GOOD))
    slop = evaluate(task, Sample(task_id=task.id, model="slop", response=BAD))
    assert tidy.metrics.tshu_score < 15.0
    assert slop.metrics.tshu_score > 55.0
    assert slop.metrics.tshu_score - tidy.metrics.tshu_score > 40.0
