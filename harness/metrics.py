"""M1-M6 六大确定性指标算子（v2.2）。

结合 markdown-it-py AST 与 jieba 词法分析，全部指标保持 100% 确定性计算、零裁判模型，
直接量化本机工程会话与基准评测实证归纳的六大后训练过对齐病灶：
- M1（CVD）：首句决断延迟，口语强表态直接零惩罚
- M2（HED）：防御性叠甲密度，支持强否定与双重否定豁免
- M3（ENU）：报菜名堆砌与行内代码/分号高密度电报压缩
- M4（CCI）：纯汉字超长分句、条件嵌套、分号链与正文外显推演思维链痕迹
- M5（PNI）：悬空裸代号/行号指代与弱动词名物化、欧化被动搭配（豁免括号内联解释与排障单字「被」）
- M6（VTR）：语言抽动（自动豁免题干引词）、破折号、客服收尾与 AST 跨语域表格墙/过密大纲

每个函数返回 0-100 的惩罚分，越高越不说人话。
"""

import math
import re
from statistics import mean, pstdev

from harness.models import MetricScores, Sample, ScoredSample, Task
from harness.text import (
    FENCED_CODE,
    analyze_markdown,
    char_len,
    clamp,
    cut_pos,
    han_len,
    per_1000,
    saturate,
    shannon_entropy,
    split_clauses,
    split_sentences,
    strip_code,
    windows_of,
)

HEDGE_WORDS = (
    "不一定",
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
    "|".join(re.escape(w) for w in sorted(HEDGE_WORDS, key=len, reverse=True))
)
SOFT_OPENERS = ("不一定", "一般不会", "通常不会", "这句可以", "可以先", "我先看", "我先找", "我先确认")
CRISP_OPENERS = (
    "千万别",
    "千万不要",
    "绝对不行",
    "非常不建议",
    "强烈不建议",
    "绝不建议",
    "绝对不建议",
    "都不建议",
    "均不建议",
    "不建议",
    "不太建议",
    "不合理",
    "不太合理",
    "没必要",
    "没有必要",
    "不需要",
    "不太行",
    "都不行",
    "都不可行",
    "都不要",
    "不能",
    "不会",
    "不是",
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
    "先止血",
    "先止损",
    "先回滚",
    "先切",
    "立刻",
    "立即",
    "拆成两个",
    "拆开",
    "选 ",
    "选**",
    "选 **",
    "别设",
    "别直接",
    "先别",
    "这不算",
    "不算",
    "不成立",
    "有几个坑",
    "存在几个",
    "有几个比较",
    "有几个地方",
    "本质原因是",
    "根本原因在于",
    "核心原因是",
    "根源一样",
    "多线程并发",
    "最省事",
    "版本一",
    "版本 1",
    "版本1",
    "简短版",
    "群发版",
)
META_LABEL_RE = re.compile(
    r"^(?:先说结论|直接说结论|结论是|结论|简短的回答是|简短回答是|简短的回答|简短回答|直接回答|一句话总结|一句话)[：:，,\s]*"
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
CONDITION_RE = re.compile("|".join(sorted(CONDITION, key=len, reverse=True)))

WEAK_VERBS = frozenset(("进行", "加以", "予以", "做出", "作出", "开展", "实现", "完成", "实施"))
NOMINAL_NOUNS = frozenset(
    (
        "工作",
        "处理",
        "分析",
        "优化",
        "排查",
        "设计",
        "调整",
        "改造",
        "评估",
        "部署",
        "验证",
        "操作",
        "管理",
        "建设",
        "决定",
        "判断",
        "调研",
        "重构",
        "讨论",
    )
)
NOMINALIZATION = re.compile(
    r"(进行|加以|予以|做出|作出|开展|实现|完成)(了|过)?[\u4e00-\u9fa5]{0,6}(的)?"
    r"[\u4e00-\u9fa5]{0,4}(工作|处理|分析|优化|排查|设计|调整|改造|评估|部署|验证|操作|管理|建设)"
)
PASSIVE_PHRASE = re.compile(r"受到|得以|为[^，。！？]{1,8}所")
PRONOUN = re.compile(r"它们|它(?!们)|(?<![尤极])其(?!中|他|它|实|次)|您的|您(?!们)")
PSEUDO_CONTRAST = re.compile(r"不是[^，。]{1,20}而是|与其说[^，。]{1,20}不如说")
SUPPORT_CLOSING = re.compile(r"说一声|需要我|要不要我|随时告诉我|如果你希望|如果还需要|如需我")
SPRAWL_SECTION_RE = re.compile(
    r"常见坑|不推荐的做法|不推荐|例外情况|什么时候不需要|何时不需要|如果不想|顺便提|注意事项|坑点|补充说明"
)
EM_DASH = re.compile(r"——|(?<![a-zA-Z0-9-])—(?![a-zA-Z0-9-])")
PATH_OR_RATIO = re.compile(
    r"(?<![A-Za-z0-9\u4e00-\u9fff])(?:/[A-Za-z0-9_.*-]+)+/?|"
    r"\b[A-Za-z0-9_.-]+(?:\s*/\s*[A-Za-z0-9_.-]+)+\b"
)
INLINE_CODE_RE = re.compile(r"`[^`\n]{1,80}`")
PAREN_ANNOTATION_RE = re.compile(r"（[^）\n]{1,80}）|\([^)\n]{1,80}\)")
COT_TRACE_RE = re.compile(
    r"更准确地说|换句话说|也就是说|反过来|退一步|归根结底|先看|再看|顺带|中途|排除|推翻|坐实"
)
BARE_REF_RE = re.compile(
    r"(?<![#\n\r\*-])\s*(?:"
    r"\b(?:R|C|S|P|I|M|D|F|A|B|H|T|G|V)\d{1,2}\b"
    r"|(?:决策|规则|方案|路径|条件|分支|阶段|步骤|假设|约束|口径)\s*(?:[0-9]{1,2}|[A-D]|[①②③④⑤])"
    r"|第\s*[0-9一二三四五六七八九十]+\s*(?:条|项|点|步|行|轮)"
    r"|(?:[A-Za-z0-9_.-]+\.[a-z]{1,4}|\s):\d{2,4}\b"
    r")(?!\s*[（(:：])"
)


_QUOTE_RE = re.compile(r"[“”\"'‘’「」『』《》*]")


def _first_index(text: str, anchors: list[str]) -> int | None:
    lowered = text.lower()
    normalized = _QUOTE_RE.sub("", lowered)
    positions: list[int] = []
    for a in anchors:
        if not a:
            continue
        la = a.lower()
        na = _QUOTE_RE.sub("", la)
        for candidate in (lowered.find(la), normalized.find(na)):
            if candidate >= 0:
                positions.append(candidate)
    return min(positions) if positions else None


def score_cvd(task: Task, text: str) -> tuple[float, float]:
    """M1 首句决断延迟：剥离空泛结论标签后，按首段字符偏移与直判句式打分。"""
    if not text:
        return 100.0, 200.0
    stripped = text.lstrip("#* \t\r\n")
    stripped = META_LABEL_RE.sub("", stripped).lstrip("#* \t\r\n")
    head = stripped[:70]
    if any(opener in head for opener in CRISP_OPENERS):
        return 0.0, 0.0
    idx = _first_index(stripped, task.answer_anchors)
    if idx is not None:
        if idx <= 30:
            return 0.0, float(idx)
        return min(100.0, (idx - 30) / 110.0 * 100.0), float(idx)
    ent_idx = _first_index(stripped[:120], task.actionable_entities)
    if ent_idx is not None:
        return min(35.0, max(0.0, (ent_idx - 15) / 110.0 * 100.0)), float(ent_idx)
    return 100.0, 200.0


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
        head = text[:80]
        if any(w in head for w in CRISP_OPENERS) or _first_index(head, task.answer_anchors) is not None:
            soft = 0.0
        elif any(w in head for w in SOFT_OPENERS) or any(w in head for w in task.fence_sitting_markers):
            soft = 30.0
        else:
            soft = 10.0
    raw = min(100.0, base + soft)
    return round(raw, 2), raw


def score_enu(text: str) -> tuple[float, float]:
    """M3 报菜名与信息过载：顿号斜杠堆砌 + 单句连发密度 + 行内代码与分号压缩。"""
    prose = PATH_OR_RATIO.sub(" ", strip_code(text))
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    marks = prose.count("、") + prose.count("/")
    rate = per_1000(marks, n, floor=250)
    sentences = split_sentences(prose)
    bursts = sum(1 for s in sentences if s.count("、") >= 4 or s.count("/") >= 3)
    burst_rate = per_1000(bursts, n, floor=250)

    raw_no_fence = FENCED_CODE.sub(" ", text)
    code_spans = len(INLINE_CODE_RE.findall(raw_no_fence))
    semis = prose.count("；") + prose.count(";")
    han = max(1, han_len(prose))
    pack_rate = (code_spans * 0.6 + semis * 2.5) * 1000.0 / max(300, han)
    pack_comp = clamp((pack_rate - 45.0) / 50.0)

    raw = clamp((rate - 8.0) / 24.0) * 55.0 + clamp(burst_rate / 4.5) * 25.0 + pack_comp * 25.0
    raw = min(100.0, raw)
    return round(raw, 2), raw


def score_cci(text: str) -> tuple[float, float]:
    """M4 句法与推演认知负荷：纯汉字超长分句、条件嵌套、分号链、外显推演词与的字链。"""
    prose = strip_code(text)
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    han = max(1, han_len(prose))
    clean_prose = PAREN_ANNOTATION_RE.sub("", prose)
    sentences = split_sentences(clean_prose)
    clauses = [c for s in sentences for c in split_clauses(s)] or [clean_prose]

    long_ratio = sum(1 for c in clauses if han_len(c) > 28) / len(clauses)
    cond_counts = [len(CONDITION_RE.findall(s)) for s in sentences]
    nest_score = sum(1.0 if c >= 3 else (0.4 if c == 2 else 0.0) for c in cond_counts)
    nest_ratio = nest_score / max(1, len(sentences))

    semis = prose.count("；") + prose.count(";")
    cot_hits = len(COT_TRACE_RE.findall(prose))
    delib_rate = per_1000(semis + cot_hits * 2, han, floor=150)
    delib_comp = clamp((delib_rate - 3.0) / 12.0)

    de_ratio = prose.count("的") / n
    excess = clamp((de_ratio - 0.045) / 0.030)
    windows = windows_of(prose, 15)
    de_windows = sum(1 for w in windows if w.count("的") >= 3)
    window_ratio = clamp(de_windows / max(1, len(windows)) / 0.10)

    raw = clamp(
        0.40 * clamp(long_ratio / 0.20)
        + 0.30 * clamp(nest_ratio / 0.15)
        + 0.25 * delib_comp
        + 0.20 * (0.5 * excess + 0.5 * window_ratio)
    )
    return round(saturate(raw, tau=0.02, k=8.0), 2), raw


def _count_nominalizations(prose: str, pos_tokens: list[tuple[str, str]]) -> int:
    clause_stops = frozenset("，。！？；：\n,.!?;:")
    pos_hits = 0
    for idx, (word, _) in enumerate(pos_tokens):
        if word not in WEAK_VERBS:
            continue
        for next_word, next_flag in pos_tokens[idx + 1 : idx + 6]:
            if any(ch in clause_stops for ch in next_word):
                break
            if next_flag == "vn" or next_word in NOMINAL_NOUNS:
                pos_hits += 1
                break
    return max(pos_hits, len(NOMINALIZATION.findall(prose)))


def score_pni(text: str) -> tuple[float, float]:
    """M5 欧化名物化与悬空指针：弱动词名物化、欧化被动、显性代词与无内联解释的裸编号。"""
    prose = strip_code(text)
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    pos_tokens = cut_pos(prose)
    nom_count = _count_nominalizations(prose, pos_tokens)
    passive_count = len(PASSIVE_PHRASE.findall(prose))
    bare_refs = len(BARE_REF_RE.findall(prose))

    nom = per_1000(nom_count, n, floor=150)
    passive = per_1000(passive_count, n, floor=150)
    explicit = per_1000(len(PRONOUN.findall(prose)), n, floor=150)
    ref_rate = per_1000(bare_refs, n, floor=150)

    raw = clamp(
        0.40 * clamp(nom / 6.0)
        + 0.20 * clamp(passive / 6.0)
        + 0.15 * clamp(explicit / 20.0)
        + 0.35 * clamp(ref_rate / 4.0)
    )
    return round(saturate(raw, tau=0.03, k=6.0), 2), raw


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
    """M6 语言抽动与排版强迫（AST 表格墙与过密大纲计数、破折号与客服收尾）。"""
    md = analyze_markdown(text)
    prose = md.prose_text
    n = char_len(prose)
    if n == 0:
        return 0.0, 0.0
    active_jargon = tuple(m for m in JARGON if m not in task.prompt)
    tics = (
        sum(prose.count(m) for m in active_jargon + ABSOLUTE)
        + len(PSEUDO_CONTRAST.findall(prose))
        + len(SUPPORT_CLOSING.findall(prose))
        + len(SPRAWL_SECTION_RE.findall(prose))
    )
    tics_rate = per_1000(tics, n, floor=150)
    dash_rate = per_1000(len(EM_DASH.findall(prose)), n, floor=150)
    dash_comp = clamp((dash_rate - 0.5) / 4.0)

    md_count = 0.0
    if task.register == "chat":
        excess_headings = max(0, md.headings - max(2, n // 240))
        excess_hr = max(0, md.hr_count - 1)
        excess_code = max(0, md.code_blocks - 2)
        md_count = md.table_rows * 2.5 + excess_headings * 1.0 + excess_hr * 0.6 + excess_code * 1.5
    md_rate = md_count * 500.0 / n

    cv = _cv_component(split_sentences(prose))
    base_raw = 0.40 * clamp(tics_rate / 8.0) + 0.25 * dash_comp + 0.25 * clamp(md_rate / 4.0) + 0.10 * cv
    raw = clamp(0.85 * base_raw + 0.15 * opening_penalty)
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
    cci, cci_raw = score_cci(text)
    pni, pni_raw = score_pni(text)
    vtr, vtr_raw = score_vtr(task, text, opening_penalty)

    metrics = MetricScores(cvd=cvd, hed=hed, enu=enu, cci=cci, pni=pni, vtr=vtr)
    raw = {
        "cvd_offset": round(cvd_raw, 2),
        "hed": round(hed_raw, 4),
        "enu": round(enu_raw, 4),
        "cci": round(cci_raw, 4),
        "pni": round(pni_raw, 4),
        "vtr": round(vtr_raw, 4),
    }
    return ScoredSample(sample=sample, task=task, metrics=metrics, raw=raw)
