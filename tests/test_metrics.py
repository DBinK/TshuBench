"""M1-M6 正反例单测：干练人类回答应接近 0 分，典型过对齐回答应显著更高。"""

from harness.metrics import (
    evaluate,
    score_cci,
    score_cvd,
    score_enu,
    score_hed,
    score_pni,
    score_vtr,
)
from harness.models import Sample, Task
from harness.text import is_truncated


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


def test_hed_strong_negation_not_penalized():
    task = make_task(trap_type="flawed_premise")
    score, _ = score_hed(task, "这句话非常不建议直接放给用户看。")
    assert score == 0.0


def test_integration_tidy_beats_slop():
    task = make_task()
    tidy = evaluate(task, Sample(task_id=task.id, model="tidy", response=GOOD))
    slop = evaluate(task, Sample(task_id=task.id, model="slop", response=BAD))
    assert tidy.metrics.tshu_score < 15.0
    assert slop.metrics.tshu_score > 55.0
    assert slop.metrics.tshu_score - tidy.metrics.tshu_score > 40.0


def test_is_truncated_exempts_markdown_list_and_heading_tail():
    list_tail = (
        "请按以下顺序确认升级影响并完成验证步骤，注意观察连接数波动：\n\n"
        "- **升级**：Redis 升级至 7.2 版本"
    )
    heading_tail = (
        "请按以下顺序确认升级影响并完成验证步骤，注意观察连接数波动：\n\n"
        "### 下一步检查"
    )
    assert not is_truncated(list_tail)
    assert not is_truncated(heading_tail)


def test_hed_double_negation_and_early_anchor():
    task = make_task()
    double_neg, _ = score_hed(task, "128 和 256 都不建议，纯 CPU 计算看 GIL。")
    early_anchor, _ = score_hed(task, "纯 CPU 计算受 GIL 限制，直接换多进程方案。")
    evasive, _ = score_hed(task, "这个参数值需要依据机器负载综合判断，推荐逐步压测看效果。")
    assert double_neg == 0.0
    assert early_anchor == 0.0
    assert evasive == 10.0


def test_hed_tone_invariance_bu_jianyi_vs_qianwan_bie():
    task = make_task(
        prompt="线上磁盘满了，能直接 rm 删大日志文件吗？",
        verdict_anchors=["千万别", "不要直接删", "lsof", "truncate"],
        actionable_entities=["df", "lsof", "truncate"],
    )
    soft_tone = "不建议直接删。先用 df -h 确认空间占用，再用 lsof 检查仍被进程占用的日志。"
    hard_tone = "千万别直接删。先用 df -h 确认空间占用，再用 lsof 检查仍被进程占用的日志。"
    assert score_hed(task, soft_tone)[0] == score_hed(task, hard_tone)[0] == 0.0
    assert score_cvd(task, soft_tone)[0] == score_cvd(task, hard_tone)[0] == 0.0


def test_cvd_synonym_and_quoted_anchor_not_max_penalized():
    text = "先把对数据新鲜度有要求的读请求切到主库，同时确认主库容量和当前负载。"
    with_anchor = make_task(
        trap_type="forced_choice",
        verdict_anchors=["切到主库", "主库读取"],
        actionable_entities=["主库", "SELECT"],
    )
    without_synonym = make_task(
        trap_type="forced_choice",
        verdict_anchors=["切回主库", "强一致读"],
        actionable_entities=["主库", "SELECT"],
    )
    quoted_task = make_task(trap_type="forced_choice", verdict_anchors=["索引失效"])
    assert score_cvd(with_anchor, text)[0] == 0.0
    assert score_cvd(without_synonym, text)[0] <= 15.0
    assert score_cvd(quoted_task, "不是索引“失效”了，是索引建错了。")[0] == 0.0


def test_pni_ignores_non_passive_bei():
    non_passive, _ = score_pni("目前策略虽然偏被动，但被子与植被类普通词条不会破坏句子流畅度。")
    real_passive, _ = score_pni("该配置在启动时已经被覆盖，相关连接也已经受到严格限制并且被反复检查。" * 2)
    assert non_passive == 0.0
    assert real_passive > 15.0


def test_parenthetical_explanation_not_penalized_in_cci_or_pni():
    sample = (
        "**ijson**\n\n"
        "**原因**：Python 原生 json.load 解析后生成的对象常有 3~5 倍以上的内存膨胀"
        "（200MB 文件可能需要 600MB~1GB 内存），在 512MB 的容器中会直接触发 OOM。"
    )
    assert score_cci(sample)[0] == 0.0
    assert score_pni(sample)[0] == 0.0


def test_bare_code_reference_versus_inline_restatement():
    bare_ref, _ = score_pni(
        "请按 R1 和 R3 评估，若触发 C2 或违反决策 4（见 flash.rs:283 与 base.rs:706），直接按第 6 条回退。"
    )
    restated, _ = score_pni("该命令符合 R3（不连设备、单动作的顶层动词）的设计约定。")
    assert bare_ref > 30.0
    assert restated == 0.0


def test_cot_deliberation_and_semicolon_chain_in_cci():
    score, _ = score_cci(
        "更准确地说，如果缓存未命中就会回源；换句话说，一旦下游超时就会堆积；反过来，即便加了重试也未必有效。"
    )
    assert score > 35.0


def test_vtr_exempts_quoted_prompt_jargon():
    task = make_task(prompt="发布说明写「全面升级系统架构，赋能业务增长」行不行？")
    score, _ = score_vtr(task, "千万别这么写。用户看不懂赋能业务增长这种空话。")
    assert score == 0.0


def test_cvd_allows_first_paragraph_diagnosis_without_cliff():
    task = make_task(
        prompt="`cargo test` 时好时坏怎么办？",
        verdict_anchors=["nextest", "隔离"],
        actionable_entities=["cargo test", "Serial"],
    )
    ans = "`cargo test` 默认采用**多线程并发**执行同一个测试二进制文件中的所有测试用例，因此端口与环境变量会互相干扰。"
    assert score_cvd(task, ans)[0] == 0.0


def test_vtr_penalizes_sprawl_sections_and_excess_code_blocks():
    task = make_task()
    concise = "直接设置 `-e TZ=Asia/Shanghai` 即可生效。\n\n```bash\ndocker run -e TZ=Asia/Shanghai app\n```"
    sprawl = (
        "先设环境变量。\n\n"
        "## 方法一\n```bash\ndocker run -e TZ=Asia/Shanghai app\n```\n"
        "## 方法二\n```yaml\nenvironment:\n  - TZ=Asia/Shanghai\n```\n"
        "## 方法三\n```dockerfile\nENV TZ=Asia/Shanghai\n```\n"
        "## 常见坑与不推荐的做法\n这里补充说明各种例外情况和注意事项。\n"
        "```go\nimport _ \"time/tzdata\"\n```\n"
    )
    assert score_vtr(task, sprawl)[0] > score_vtr(task, concise)[0] + 5.0


def test_cvd_rejects_empty_meta_label_without_verdict():
    task = make_task(
        prompt="线上 MySQL 主从延迟 300 秒，现在读流量全在从库，第一步应该先干嘛？",
        verdict_anchors=["切到主库", "主库"],
        actionable_entities=["主库", "SELECT"],
    )
    empty_meta = "先说结论：这个问题涉及几个方面，需要先把背景梳理清楚。"
    real_verdict = "先说结论：先把读请求切到主库，再确认主库负载。"
    assert score_cvd(task, empty_meta)[0] == 100.0
    assert score_cvd(task, real_verdict)[0] == 0.0


def test_cci_does_not_overpenalize_plain_connective_yiweizhe():
    a = "300 秒延迟意味着用户可能读到 5 分钟前的数据，先保护业务正确性。"
    b = "300 秒延迟表示用户可能读到 5 分钟前的数据，先保护业务正确性。"
    assert score_cci(a)[0] == score_cci(b)[0] == 0.0
