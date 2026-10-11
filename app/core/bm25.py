"""BM25 关键词检索（自主实现，零外部依赖）。

为什么需要它：纯向量检索擅长"语义相近"，但对专有名词、编号、缩写
（如 P0、429、rrf_k）不敏感；BM25 按词频精确匹配，二者互补。

中文分词不引入 jieba：以"单字 + 相邻双字（bigram）"作为词元，
既能覆盖中文短语，又避免额外依赖与模型下载。
"""
import math
import re
from collections import Counter

# 连续英文/数字视为一个词；连续汉字单独处理
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """切词：英文/数字按整词，中文按单字 + 相邻双字。"""
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        s = match.group()
        if s[0].isascii():
            tokens.append(s)
        else:
            tokens.extend(s)  # 单字
            tokens.extend(s[i : i + 2] for i in range(len(s) - 1))  # 双字
    return tokens


class BM25Index:
    """标准 BM25（Okapi）实现。

    公式：score(q, d) = Σ idf(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1-b+b*|d|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[dict] = []
        self._tf: list[Counter] = []
        self._len: list[int] = []
        self._df: Counter = Counter()
        self._avgdl: float = 0.0

    def build(self, docs: list[dict]) -> None:
        """建立索引。docs 每项须含 chunk_id / doc_id / doc_name / chunk_text。"""
        self._docs = list(docs)
        self._tf = []
        self._len = []
        self._df = Counter()
        for d in self._docs:
            counts = Counter(tokenize(d["chunk_text"]))
            self._tf.append(counts)
            self._len.append(sum(counts.values()))
            self._df.update(counts.keys())
        n = len(self._docs)
        self._avgdl = (sum(self._len) / n) if n else 0.0

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """返回按 BM25 分数降序的候选（含 bm25_score 字段）。"""
        if not self._docs:
            return []
        q_terms = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self._tf):
            dl = self._len[i]
            avgdl = self._avgdl or 1.0
            score = 0.0
            for t in q_terms:
                f = counts.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                score += self._idf(t) * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, i))
        # 分数降序；分数相同时按索引升序，保证结果稳定
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [{**self._docs[i], "bm25_score": score} for score, i in scored[:top_k]]
