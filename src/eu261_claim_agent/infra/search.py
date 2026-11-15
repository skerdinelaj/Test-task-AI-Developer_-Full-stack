"""Tiny BM25 search over the local regulation and policy notes.

Small enough to need no vector database; the interface (``search(query, k)``) is what a
production version backed by embeddings + a vector store would keep.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from eu261_claim_agent.core.schemas import RegulationSnippet

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "that",
        "to",
        "was",
        "were",
        "with",
        "which",
        "who",
        "will",
        "not",
        "no",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    text: str
    tokens: tuple[str, ...]


class RegulationIndex:
    def __init__(self, docs_dir: Path, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        for path in sorted(docs_dir.glob("*.md")):
            lines = path.read_text(encoding="utf-8").splitlines()
            title = lines[0].lstrip("# ").strip() if lines else path.stem
            body = "\n".join(lines[1:])
            for para in re.split(r"\n\s*\n", body):
                para = para.strip()
                if para:
                    self.chunks.append(
                        Chunk(path.stem, title, para, tuple(tokenize(title + " " + para)))
                    )
        self.avg_len = sum(len(c.tokens) for c in self.chunks) / max(len(self.chunks), 1)
        doc_freq: Counter[str] = Counter()
        for c in self.chunks:
            doc_freq.update(set(c.tokens))
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5)) for t, df in doc_freq.items()}

    def search(self, query: str, k: int = 3) -> list[RegulationSnippet]:
        q_tokens = tokenize(query)
        scored: list[tuple[float, Chunk]] = []
        for chunk in self.chunks:
            tf = Counter(chunk.tokens)
            norm = self.k1 * (1 - self.b + self.b * len(chunk.tokens) / self.avg_len)
            score = sum(
                self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                for t in q_tokens
                if t in tf
            )
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [
            RegulationSnippet(doc_id=c.doc_id, title=c.title, text=c.text, score=round(s, 3))
            for s, c in scored[:k]
        ]
