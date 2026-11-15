"""中文文本切分、Markdown AST 解析与 jieba 词法工具。"""

import logging
import math
import re
import warnings
from collections import Counter
from typing import NamedTuple

from markdown_it import MarkdownIt

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import jieba
    import jieba.posseg as pseg

jieba.setLogLevel(logging.WARNING)

_MD = MarkdownIt("commonmark").enable("table")

FENCED_CODE = re.compile(r"```.*?```", re.S)
INLINE_CODE_TICKS = re.compile(r"`([^`\n]{1,80})`")
TABLE_DIVIDER_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
MD_TAIL_LINE_RE = re.compile(r"^\s*(?:#{1,6}\s|[-*+]\s|\d+\.\s|\|.*\|\s*$)")
SENTENCE_END = re.compile(r"[。！？!?…]+|\n+")
CLAUSE_SEP = re.compile(r"[，、；;：:,]")
WHITESPACE = re.compile(r"\s+")
CJK = re.compile(r"[\u4e00-\u9fa5]")
CJK_END = re.compile(r"[\u4e00-\u9fa5]$")
TERMINAL = ("。", "！", "？", "!", "?", "…", ".", "：", ":", "）", ")", "」", "》", "”", "`", "*", "|")


class MarkdownAnalysis(NamedTuple):
    """Markdown AST 解析结果。"""

    prose_text: str
    headings: int
    table_rows: int
    hr_count: int
    last_block_type: str
    code_blocks: int = 0


_BLOCK_OPEN_LABELS = {
    "heading_open": "heading",
    "list_item_open": "list_item",
    "table_open": "table",
    "paragraph_open": "paragraph",
    "fence": "fence",
    "code_block": "code_block",
}


def analyze_markdown(text: str) -> MarkdownAnalysis:
    """用 markdown-it-py 提取纯文本与结构计数，避免正则误伤代码块。"""
    if not text:
        return MarkdownAnalysis("", 0, 0, 0, "", 0)

    tokens = _MD.parse(text)
    code_ranges: list[tuple[int, int]] = []
    headings = 0
    table_rows = 0
    hr_count = 0
    last_block_type = ""

    for token in tokens:
        if token.type in ("fence", "code_block") and token.map:
            code_ranges.append((token.map[0], token.map[1]))
        elif token.type == "heading_open":
            headings += 1
        elif token.type == "tr_open":
            table_rows += 1
        elif token.type == "hr":
            hr_count += 1

        if token.type in _BLOCK_OPEN_LABELS:
            last_block_type = _BLOCK_OPEN_LABELS[token.type]

    lines = text.splitlines()
    code_line_mask = [False] * len(lines)
    for start, end in code_ranges:
        for idx in range(max(0, start), min(len(lines), end)):
            code_line_mask[idx] = True

    kept_lines = [
        line
        for idx, line in enumerate(lines)
        if not code_line_mask[idx] and not TABLE_DIVIDER_RE.match(line)
    ]
    prose = "\n".join(kept_lines)
    prose = INLINE_CODE_TICKS.sub(r"\1", prose)
    return MarkdownAnalysis(
        prose_text=prose,
        headings=headings,
        table_rows=table_rows,
        hr_count=hr_count,
        last_block_type=last_block_type,
        code_blocks=len(code_ranges),
    )


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def strip_code(text: str) -> str:
    """去掉代码块与表格分隔线，内联代码保留内容去掉反引号。"""
    return analyze_markdown(text).prose_text


def cut_pos(text: str) -> list[tuple[str, str]]:
    """使用 jieba 词性标注切分文本，返回 (词, 词性) 列表。"""
    if not text:
        return []
    return [(pair.word, pair.flag) for pair in pseg.cut(text)]


def char_len(text: str) -> int:
    """非空白字符数，作为所有密度的分母。"""
    return len(WHITESPACE.sub("", text))


def han_len(text: str) -> int:
    return len(CJK.findall(text))


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_clauses(sentence: str) -> list[str]:
    parts = CLAUSE_SEP.split(sentence)
    return [p.strip() for p in parts if p.strip()]


def ngrams(text: str, n: int = 3) -> set[str]:
    plain = WHITESPACE.sub("", text)
    if len(plain) < n:
        return {plain} if plain else set()
    return {plain[i : i + n] for i in range(len(plain) - n + 1)}


def jaccard(a: str, b: str, n: int = 3) -> float:
    ga, gb = ngrams(a, n), ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def per_1000(count: int, length: int, floor: int = 0) -> float:
    """千字频次。floor 是分母下限，避免短回答被速率放大。"""
    return count * 1000.0 / max(1, length, floor)


def shannon_entropy(items: list[str]) -> float:
    if not items:
        return 0.0
    counts = Counter(items)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def saturate(x: float, tau: float, k: float, x_max: float = 1.0) -> float:
    """对数饱和映射到 0-100，低于容忍阈值 tau 直接零惩罚。"""
    if x <= tau:
        return 0.0
    denom = math.log(1.0 + k * max(0.0, x_max - tau))
    if denom <= 0:
        return 100.0
    return min(100.0, 100.0 * math.log(1.0 + k * (x - tau)) / denom)


def windows_of(text: str, size: int) -> list[str]:
    plain = WHITESPACE.sub("", text)
    if len(plain) <= size:
        return [plain] if plain else []
    return [plain[i : i + size] for i in range(len(plain) - size + 1)]


def is_truncated(text: str) -> bool:
    """判断正文是否在句子中途被输出预算截断，豁免短列表项与标题行尾。"""
    t = text.rstrip()
    if t.count("```") % 2 == 1:
        return True
    if len(t) < 60 or t.endswith(TERMINAL):
        return False

    lines = [line for line in t.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1].strip()
        analysis = analyze_markdown(t)
        is_structural_tail = (
            analysis.last_block_type in ("heading", "list_item", "table")
            or bool(MD_TAIL_LINE_RE.match(last_line))
        )
        if is_structural_tail and len(last_line) < 90:
            return False

    return bool(CJK_END.search(t))
