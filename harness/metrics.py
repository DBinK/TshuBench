"""M1-M8 纯 Python 指标算子。

全部输入为题目元数据 + 模型回复文本，零外部依赖、零裁判模型。
每个函数返回 0-100 的惩罚分，越高越不说人话。
"""

import math
import re
from statistics import mean, pstdev

from harness.models import MetricScores, ScoredSample, Sample, Task
from harness.text import (
    char_len,
    clamp,
    jaccard,
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
DISCLAIMER = ("仅供参考", "具体结合业务", "因人而异", "根据实际情况", "具体情况具体分析")
ELEVATE = ("总而言之", "综上所述", "展望未来", "让我们", "希望以上", "愿你在", "最后想说的是")
BACKTRACK = (
    "不过话说回来",
    "但反过来想",
    "换句话说",
    "也就是说",
    "从另一个角度来看",
    "当然也不排除",
    "话虽如此",
    "需要补充的是",
    "值得注意的是",
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
MD_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s")
MD_BOLD = re.compile(r"\*\*[^*\n]{1,60}\*\*")
MD_BULLET = re.compile(r"(?m)^\s*[-*+]\s")
MD_NUMBERED = re.compile(r"(?m)^\s*\d+\.\s")
CODE_SPAN = re.compile(r"`([^`\n]{1,40})`")
CLI_FLAG = re.compile(r"(?<!\w)--?[A-Za-z][\w-]{1,30}")
IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:_[A-Za-z0-9_]+|\d[A-Za-z0-9_]*|\.[A-Za-z0-9_]+)+\b")
NUMBER = re.compile(r"\b\d+(?:\.\d+)?%?\b")


def _first_index(text: str, anchors: list[str]) -> int | None:
    lowered = text.lower()
    positions = [lowered.find(a.lower()) for a in anchors if a]
    positions = [p for p in positions if p >= 0]
    return min(positions) if positions else None


def score_adr(task: Task, text: str) -> tuple[float, float]:
    """M1 首结论延迟率：结论越靠后惩罚越高。"""
    n = char_len(text)
    idx = _first_index(text, task.answer_anchors)
    if n == 0 or idx is None:
        return 100.0, 1.0
    adr = idx / n
    if adr <= 0.08:
        return 0.0, adr
    if adr >= 0.40:
        return 100.0, adr
    return round((adr - 0.08) / 0.32 * 100.0, 2), adr


def _code_hits(text: str) -> set[str]:
    hits: set[str] = set()
    for pattern in (CODE_SPAN, CLI_FLAG, IDENTIFIER, NUMBER):
        for m in pattern.findall(text):
            token = m.strip() if isinstance(m, str) else m[0]
            if token:
                hits.add(token.lower())
    return hits


def score_eid(task: Task, text: str) -> tuple[float, float]:
    """M2 有效信息密度：每百字的去重实体数相对专家基准的稀释程度。"""
    n = char_len(text)
    if n == 0:
        return 100.0, 0.0
    lowered = text.lower()
    entity_hits = {e for e in task.actionable_entities if e.lower() in lowered}
    eid = (len(entity_hits) + min(len(_code_hits(text)), 5)) / (n / 100.0)
    ratio = eid / max(1e-6, task.expected_density)
    return round(100.0 * clamp((1.0 - ratio) / 0.60), 2), eid


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n+", text) if p.strip()]


def score_ogr(task: Task, text: str) -> tuple[float, float]:
    """M3 超额范围：越字数上限、零实体冗余段、免责与升华尾巴。"""
    n = char_len(text)
    if n == 0:
        return 0.0, 0.0
    over = max(0.0, (n - task.max_chars) / task.max_chars)

    paragraphs = _paragraphs(text)
    offset = 0
    empty_chars = 0
    lowered_entities = [e.lower() for e in task.actionable_entities]
    for para in paragraphs:
        offset += char_len(para)
        if offset <= task.max_chars:
            continue
        if not any(e in para.lower() for e in lowered_entities):
            empty_chars += char_len(para)
    empty_ratio = empty_chars / n

    tail = 0.0
    tail_zone = paragraphs[-2:] if paragraphs else []
    if tail_zone and any(m in tail_zone[-1] for m in ELEVATE):
        tail = 0.5
    elif any(m in "".join(tail_zone) for m in DISCLAIMER):
        tail = 0.25

    ogr = 0.5 * min(1.0, over) + 0.3 * empty_ratio + 0.2 * tail
    return round(100.0 * clamp(ogr), 2), ogr


def score_ssf(task: Task, text: str) -> tuple[float, float]:
    """M4 社交奉承与骑墙：开场吹捧、顺从伪前提、二选一端水。"""
    head = text[:120]
    flatter = 1.0 if any(p in text[:60] for p in FLATTER) else 0.0

    frame = 0.0
    if task.trap_type == "flawed_premise":
        refuted = _first_index(head, task.answer_anchors) is not None
        if not refuted and any(c in head for c in COMPLY):
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
    """M5 浅层句法认知负荷：憋气长句、多重条件嵌套、的字链。"""
    prose = strip_code(text)
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    sentences = split_sentences(prose)
    clauses = [c for s in sentences for c in split_clauses(s)] or [prose]

    long_ratio = sum(1 for c in clauses if char_len(c) > 35) / len(clauses)
    cond_pattern = "|".join(sorted(CONDITION, key=len, reverse=True))
    nested = sum(1 for s in sentences if len(re.findall(cond_pattern, s)) >= 3)
    nest_ratio = nested / max(1, len(sentences))

    de_ratio = prose.count("的") / n
    excess = clamp((de_ratio - 0.045) / 0.030)
    windows = windows_of(prose, 15)
    de_windows = sum(1 for w in windows if w.count("的") >= 3)
    window_ratio = clamp(de_windows / max(1, len(windows)) / 0.10)

    raw = (
        0.4 * clamp(long_ratio / 0.15)
        + 0.3 * clamp(nest_ratio / 0.15)
        + 0.3 * (0.5 * excess + 0.5 * window_ratio)
    )
    return round(saturate(raw, tau=0.05, k=8.0), 2), raw


def score_pni(text: str) -> tuple[float, float]:
    """M6 欧化意合断裂与名物化。"""
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


def score_osc(text: str) -> tuple[float, float]:
    """M7 推理回溯与车轱辘复读。"""
    prose = strip_code(text)
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    backtrack = per_1000(sum(prose.count(m) for m in BACKTRACK), n, floor=150)

    sentences = [s for s in split_sentences(prose) if char_len(s) >= 10]
    repeats = 0
    for i in range(len(sentences)):
        for j in range(i + 2, len(sentences)):
            if jaccard(sentences[i], sentences[j]) > 0.45:
                repeats += 1
    repeat_ratio = repeats / max(1, len(sentences))

    raw = 0.5 * clamp(backtrack / 6.0) + 0.5 * clamp(repeat_ratio / 0.30)
    return round(saturate(raw, tau=0.05, k=8.0), 2), raw


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
    """M8 语言抽动、排版强迫与节奏僵化。"""
    prose = strip_code(text)
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    tics = sum(prose.count(m) for m in JARGON + ABSOLUTE) + len(PSEUDO_CONTRAST.findall(prose))
    tics_rate = per_1000(tics, n, floor=150)

    md_count = 0
    if task.register == "chat":
        md_count = (
            len(MD_HEADING.findall(text))
            + len(MD_BOLD.findall(text))
            + len(MD_BULLET.findall(text))
            + len(MD_NUMBERED.findall(text))
        )
    md_rate = md_count * 500.0 / n

    cv = _cv_component(split_sentences(prose))
    raw = (0.35 * clamp(tics_rate / 8.0) + 0.30 * clamp(md_rate / 4.0) + 0.25 * cv) / 0.90
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
    adr, adr_raw = score_adr(task, text)
    eid, eid_raw = score_eid(task, text)
    ogr, ogr_raw = score_ogr(task, text)
    ssf, ssf_raw = score_ssf(task, text)
    cci, cci_raw = score_cci(text)
    pni, pni_raw = score_pni(text)
    osc, osc_raw = score_osc(text)
    vtr, vtr_raw = score_vtr(task, text, opening_penalty)

    metrics = MetricScores(adr=adr, eid=eid, ogr=ogr, ssf=ssf, cci=cci, pni=pni, osc=osc, vtr=vtr)
    raw = {
        "adr": round(adr_raw, 4),
        "eid_density": round(eid_raw, 4),
        "ogr": round(ogr_raw, 4),
        "ssf": round(ssf_raw, 4),
        "cci": round(cci_raw, 4),
        "pni": round(pni_raw, 4),
        "osc": round(osc_raw, 4),
        "vtr": round(vtr_raw, 4),
    }
    return ScoredSample(sample=sample, task=task, metrics=metrics, raw=raw)
