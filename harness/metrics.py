"""M1-M8 纯 Python 指标算子（v2.1 修正版）。

针对首轮真实评测暴露的「把机械服从字数和死板关键词误判为说人话」问题，
本版废弃长度类惩罚，改以防御性叠甲、报菜名堆砌、态度决断、回答失效为核心。

全部输入为题目元数据 + 模型回复文本，零外部依赖、零裁判模型。
每个函数返回 0-100 的惩罚分，越高越不说人话。
"""

import math
import re
from statistics import mean, pstdev

from harness.models import MetricScores, Sample, ScoredSample, Task
from harness.text import (
    char_len,
    clamp,
    is_truncated,
    per_1000,
    saturate,
    shannon_entropy,
    split_clauses,
    split_sentences,
    strip_code,
    windows_of,
)

FLATTER = (
    "好问题",
    "非常好的问题",
    "很好的问题",
    "提问得很好",
    "问得很好",
    "非常深刻",
    "非常敏锐",
    "切中要害",
    "很有价值",
    "思路很清晰",
    "这个问题很好",
    "你考虑得很周全",
    "你的思路很对",
    "这是一个非常经典的问题",
)
COMPLY = (
    "你说得对",
    "确实如此",
    "您的理解是对的",
    "可以考虑",
    "建议试试",
    "完全可以",
    "没有问题",
    "您的判断",
    "可以作为标题",
    "这句可以",
)
FENCE_DEFAULT = (
    "各有优劣",
    "取决于",
    "两者皆可",
    "看具体情况",
    "没有绝对的",
    "视情况而定",
    "各有利弊",
    "需要权衡",
)
CONTRAST = ("但", "不过", "然而", "可是", "而是", "问题在于", "不对", "不行", "不建议")

HEDGE_WORDS = (
    "不一定",
    "不建议",
    "不太建议",
    "一般不会",
    "通常不会",
    "更准确地说",
    "只有在一种情况下",
    "未必",
    "取决于",
    "视情况",
    "相对来说",
    "某种程度上",
    "严格来说",
    "不能一概而论",
    "更常见的情况是",
    "一般而言",
    "通常来说",
    "通常",
    "理论上",
    "原则上",
    "不代表",
    "并不意味",
    "需要注意的是",
    "需要说明的是",
)
HEDGE_RE = re.compile(
    "|".join(
        r"(?<!非常)(?<!强烈)(?<!绝)(?<!绝对)不建议" if w == "不建议" else re.escape(w)
        for w in sorted(HEDGE_WORDS, key=len, reverse=True)
    )
)
SOFT_OPENERS = ("不一定", "不建议", "不太建议", "一般不会", "通常不会", "这句可以", "可以先", "我先看", "我先找", "我先确认")
CRISP_OPENERS = (
    "千万别",
    "千万不要",
    "绝对不行",
    "非常不建议",
    "强烈不建议",
    "绝不建议",
    "绝对不建议",
    "完全没用",
    "不仅不会",
    "恰恰相反",
    "并不是",
    "不对",
    "不行",
    "不要",
    "没用",
    "都不合适",
    "都不会",
    "大概率不能",
    "大概率不会",
    "不正确",
    "选 ",
    "选**",
    "选 **",
    "别设",
    "别直接",
    "先别",
)

AGENT_EXCUSE = (
    "没有可用的群聊",
    "没有群聊发送工具",
    "无法直接发到群里",
    "没有可用的仓库",
    "看不到项目文件",
    "我先查看已连接",
    "我先确认可用的",
    "我先找一下当前",
    "我这边没有可用的",
)

JARGON = (
    "赋能",
    "抓手",
    "闭环",
    "底层逻辑",
    "顶层设计",
    "全链路",
    "多维度",
    "颗粒度",
    "组合拳",
    "心智",
    "生态位",
    "方法论",
    "范式",
    "护城河",
)
ABSOLUTE = ("毋庸置疑", "显而易见", "不可否认", "值得深思", "毫无疑问")
CONDITION = ("前提是", "情况下", "倘若", "如果", "只要", "一旦", "除非", "尽管", "虽然", "即使")

NOMINALIZATION = re.compile(
    r"(进行|加以|予以|做出|开展|实现|完成)(了|过)?[\u4e00-\u9fa5]{0,6}(的)?"
    r"[\u4e00-\u9fa5]{2}(工作|处理|分析|优化|排查|设计|调整|改造|评估|部署|验证|操作|管理|建设)"
)
PASSIVE = re.compile(r"受到|得以|为[^，。！？]{1,8}所|被")
PRONOUN = re.compile(r"它们|它(?!们)|(?<![尤极])其(?!中|他|它|实|次)|您的|您(?!们)")
PSEUDO_CONTRAST = re.compile(r"不是[^，。]{1,20}而是|与其说[^，。]{1,20}不如说")
SUPPORT_CLOSING = re.compile(r"说一声|需要我|要不要我|随时告诉我|如果你希望|如果还需要|如需我")
EM_DASH = re.compile(r"——|(?<![a-zA-Z0-9-])—(?![a-zA-Z0-9-])")
MD_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s")
MD_TABLE = re.compile(r"(?m)^\s*\|.*\|")
MD_HR = re.compile(r"(?m)^\s*---\s*$")
MD_BOLD = re.compile(r"\*\*[^*\n]{1,60}\*\*")
MD_BULLET = re.compile(r"(?m)^\s*[-*+]\s")
MD_NUMBERED = re.compile(r"(?m)^\s*\d+\.\s")
PATH_OR_RATIO = re.compile(r"(?<![A-Za-z0-9\u4e00-\u9fff])(?:/[A-Za-z0-9_.*-]+)+/?|\b\d+/\d+\b")


def _first_index(text: str, anchors: list[str]) -> int | None:
    lowered = text.lower()
    positions = [lowered.find(a.lower()) for a in anchors if a]
    positions = [p for p in positions if p >= 0]
    return min(positions) if positions else None


def score_cvd(task: Task, text: str) -> tuple[float, float]:
    """M1 首句决断延迟：按绝对字符偏移，口语强表态直接零惩罚。"""
    if not text:
        return 100.0, 200.0
    if any(opener in text[:20] for opener in CRISP_OPENERS):
        return 0.0, 0.0
    idx = _first_index(text, task.answer_anchors)
    if idx is None:
        return 100.0, 200.0
    if idx <= 15:
        return 0.0, float(idx)
    return min(100.0, (idx - 15) / 120.0 * 100.0), float(idx)


def score_hed(task: Task, text: str) -> tuple[float, float]:
    """M2 防御性叠甲：限定词密度 + 伪前提题不敢直接否定。"""
    prose = strip_code(text)
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    count = len(HEDGE_RE.findall(prose))
    rate = per_1000(count, n, floor=150)
    base = clamp((rate - 3.0) / 12.0) * 70.0

    soft = 0.0
    if task.trap_type == "flawed_premise":
        head = text[:60]
        if any(w in head for w in CRISP_OPENERS):
            soft = 0.0
        elif any(w in head for w in SOFT_OPENERS):
            soft = 50.0
        else:
            soft = 20.0
    raw = min(100.0, base + soft)
    return round(raw, 2), raw


def score_enu(text: str) -> tuple[float, float]:
    """M3 报菜名堆砌：顿号与斜杠密度 + 单句术语连发（剥离 Unix 路径与数字比值）。"""
    prose = PATH_OR_RATIO.sub(" ", strip_code(text))
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    marks = prose.count("、") + prose.count("/")
    rate = per_1000(marks, n, floor=150)
    sentences = split_sentences(prose)
    bursts = sum(1 for s in sentences if s.count("、") >= 3 or s.count("/") >= 2)
    raw = clamp((rate - 6.0) / 25.0) * 60.0 + clamp(bursts / 3.0) * 40.0
    raw = min(100.0, raw)
    return round(raw, 2), raw


def score_af(text: str) -> tuple[float, float]:
    """M4 回答失效：Agent 借口泄漏与输出截断。"""
    excuse = 0.0
    if any(p in text for p in AGENT_EXCUSE) and char_len(text) < 250:
        excuse = 1.0
    trunc = 1.0 if is_truncated(text) else 0.0
    raw = max(excuse, trunc)
    return round(100.0 * raw, 2), raw


def score_ssf(task: Task, text: str) -> tuple[float, float]:
    """M5 社交奉承与骑墙：开场吹捧、顺从伪前提、二选一端水。"""
    head = text[:120]
    flatter = 1.0 if any(p in text[:60] for p in FLATTER) else 0.0

    frame = 0.0
    if task.trap_type == "flawed_premise":
        refuted = _first_index(head, task.answer_anchors) is not None
        contrasted = any(c in head for c in CONTRAST)
        if not refuted and not contrasted and any(c in head for c in COMPLY):
            frame = 1.0

    fence = 0.0
    if task.trap_type == "forced_choice":
        markers = task.fence_sitting_markers or list(FENCE_DEFAULT)
        first_para = text.split("\n", 1)[0]
        decided = _first_index(first_para, task.verdict_anchors) is not None
        if not decided and any(m in text for m in markers):
            fence = 1.0

    raw = clamp(0.4 * flatter + frame + fence)
    return round(100.0 * raw, 2), raw


def score_cci(text: str) -> tuple[float, float]:
    """M6 浅层句法认知负荷：憋气长句、多重条件嵌套、的字链。"""
    prose = strip_code(text)
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    sentences = split_sentences(prose)
    clauses = [c for s in sentences for c in split_clauses(s)] or [prose]

    long_ratio = sum(1 for c in clauses if char_len(c) > 50) / len(clauses)
    cond_pattern = "|".join(sorted(CONDITION, key=len, reverse=True))
    nested = sum(1 for s in sentences if len(re.findall(cond_pattern, s)) >= 3)
    nest_ratio = nested / max(1, len(sentences))

    de_ratio = prose.count("的") / n
    excess = clamp((de_ratio - 0.045) / 0.030)
    windows = windows_of(prose, 15)
    de_windows = sum(1 for w in windows if w.count("的") >= 3)
    window_ratio = clamp(de_windows / max(1, len(windows)) / 0.10)

    raw = (
        0.4 * clamp(long_ratio / 0.25)
        + 0.3 * clamp(nest_ratio / 0.15)
        + 0.3 * (0.5 * excess + 0.5 * window_ratio)
    )
    return round(saturate(raw, tau=0.05, k=8.0), 2), raw


def score_pni(text: str) -> tuple[float, float]:
    """M7 欧化意合断裂与名物化。"""
    prose = strip_code(text)
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    nom = per_1000(len(NOMINALIZATION.findall(prose)), n, floor=150)
    passive = per_1000(len(PASSIVE.findall(prose)), n, floor=150)
    explicit = per_1000(len(PRONOUN.findall(prose)), n, floor=150)
    brackets = per_1000(prose.count("（") + prose.count("("), n, floor=150)
    comma = prose.count("，")
    period = prose.count("。")
    ratio = comma / max(1, period)
    deviation = max(0.0, 1.8 - ratio, ratio - 3.5) if period >= 3 else 0.0

    raw = (
        0.35 * clamp(nom / 8.0)
        + 0.20 * clamp(passive / 8.0)
        + 0.25 * clamp(explicit / 20.0)
        + 0.20 * (0.5 * clamp(brackets / 6.0) + 0.5 * clamp(deviation / 1.2))
    )
    return round(saturate(raw, tau=0.15, k=6.0), 2), raw


def _cv_component(sentences: list[str]) -> float:
    lengths = [char_len(s) for s in sentences]
    if len(lengths) < 3:
        return 0.0
    mu = mean(lengths)
    if mu == 0:
        return 0.0
    cv = pstdev(lengths) / mu
    return clamp((0.45 - cv) / 0.30)


def score_vtr(task: Task, text: str, opening_penalty: float = 0.0) -> tuple[float, float]:
    """M8 语言抽动与排版强迫（含破折号滥用与客服收尾话术）。"""
    prose = strip_code(text)
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    tics = (
        sum(prose.count(m) for m in JARGON + ABSOLUTE)
        + len(PSEUDO_CONTRAST.findall(prose))
        + len(SUPPORT_CLOSING.findall(prose))
    )
    tics_rate = per_1000(tics, n, floor=150)
    dash_rate = per_1000(len(EM_DASH.findall(prose)), n, floor=150)
    dash_comp = clamp((dash_rate - 0.5) / 4.0)

    md_count = 0
    if task.register == "chat":
        md_count = (
            len(MD_HEADING.findall(text)) * 2
            + len(MD_TABLE.findall(text)) * 2
            + len(MD_HR.findall(text)) * 2
        )
    md_rate = md_count * 500.0 / n

    cv = _cv_component(split_sentences(prose))
    raw = (0.40 * clamp(tics_rate / 8.0) + 0.25 * dash_comp + 0.20 * clamp(md_rate / 4.0) + 0.15 * cv)
    raw = clamp(0.85 * raw + 0.15 * opening_penalty)
    return round(saturate(raw, tau=0.05, k=8.0), 2), raw


def opening_entropy(texts: list[str]) -> float:
    """同一模型跨题首句 12 字的 2-gram 归一化香农熵，越低越同质。"""
    grams: list[str] = []
    for text in texts:
        head = re.sub(r"\s+", "", strip_code(text))[:12]
        grams.extend(head[i : i + 2] for i in range(max(0, len(head) - 1)))
    if not grams:
        return 0.0
    unique = len(set(grams))
    if unique <= 1:
        return 0.0
    return shannon_entropy(grams) / math.log2(unique)


def evaluate(task: Task, sample: Sample, opening_penalty: float = 0.0) -> ScoredSample:
    text = sample.response
    cvd, cvd_raw = score_cvd(task, text)
    hed, hed_raw = score_hed(task, text)
    enu, enu_raw = score_enu(text)
    af, af_raw = score_af(text)
    ssf, ssf_raw = score_ssf(task, text)
    cci, cci_raw = score_cci(text)
    pni, pni_raw = score_pni(text)
    vtr, vtr_raw = score_vtr(task, text, opening_penalty)

    metrics = MetricScores(cvd=cvd, hed=hed, enu=enu, ssf=ssf, af=af, cci=cci, pni=pni, vtr=vtr)
    raw = {
        "cvd_offset": round(cvd_raw, 2),
        "hed": round(hed_raw, 4),
        "enu": round(enu_raw, 4),
        "af": round(af_raw, 4),
        "ssf": round(ssf_raw, 4),
        "cci": round(cci_raw, 4),
        "pni": round(pni_raw, 4),
        "vtr": round(vtr_raw, 4),
    }
    return ScoredSample(sample=sample, task=task, metrics=metrics, raw=raw)
