"""Cross-encoder reranking of a candidate pool (ms-marco MiniLM)."""

from __future__ import annotations

from .chunkers import Chunk

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[int], chunks: list[Chunk], top_n: int
    ) -> list[int]:
        """Reorder candidate chunk positions by cross-encoder score."""
        if not candidates:
            return []
        pairs = [(query, chunks[pos].text) for pos in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in order[:top_n]]
