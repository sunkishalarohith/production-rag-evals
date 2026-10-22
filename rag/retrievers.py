"""Retrievers: dense vector search, BM25, and a hybrid fused with RRF.

All retrievers return a ranked list of chunk positions (best first) so
the eval harness can compare them on equal footing.
"""

from __future__ import annotations

import re

from .chunkers import Chunk
from .index import EmbeddingIndex

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked lists of ids into one ranking.

    score(id) = sum over lists of 1 / (k + rank), rank 1-based.
    Returns [(id, fused_score)] sorted best first.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class VectorRetriever:
    name = "vector"

    def __init__(self, index: EmbeddingIndex):
        self.index = index

    def search(self, query: str, k: int) -> list[int]:
        return [pos for pos, _ in self.index.search(query, k)]


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked[:k]


class HybridRetriever:
    """BM25 + vector, fused with Reciprocal Rank Fusion."""

    name = "hybrid"

    def __init__(self, vector: VectorRetriever, bm25: BM25Retriever, pool: int = 50):
        self.vector = vector
        self.bm25 = bm25
        self.pool = pool

    def search(self, query: str, k: int) -> list[int]:
        vec_hits = self.vector.search(query, self.pool)
        bm25_hits = self.bm25.search(query, self.pool)
        fused = reciprocal_rank_fusion([vec_hits, bm25_hits])
        return [doc_id for doc_id, _ in fused[:k]]
