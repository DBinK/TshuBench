"""中文文本切分与小工具，只用标准库。"""

import math
import re
from collections import Counter

FENCED_CODE = re.compile(r"```.*?```", re.S)
INLINE_CODE_TICKS = re.compile(r"`([^`\n]{1,80})`")
SENTENCE_END = re.compile(r"[。！？!?…]+")
CLAUSE_SEP = re.compile(r"[，、；;：:,]")
WHITESPACE = re.compile(r"\s+")
CJK = re.compile(r"[\u4e00-\u9fa5]")
CJK_END = re.compile(r"[\u4e00-\u9fa5]$")
TERMINAL = ("。", "！", "？", "!", "?", "…", ".", "：", ":", "）", ")", "」", "》", "”", "`", "*", "|")


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def strip_code(text: str) -> str:
    """去掉围栏代码块，内联代码保留内容去掉反引号。"""
    text = FENCED_CODE.sub(" ", text)
    return INLINE_CODE_TICKS.sub(r"\1", text)


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
    """判断正文是否在句子中途被输出预算截断。"""
    t = text.rstrip()
    if len(t) < 60 or t.endswith(TERMINAL):
        return False
    if CJK_END.search(t):
        return True
    return t.count("```") % 2 == 1
