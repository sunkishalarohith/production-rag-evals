"""RetrievalPipeline: chunk -> index -> retrieve -> (optional) rerank.

One pipeline instance = one configuration (chunker x search mode x
rerank on/off). The eval harness builds the full grid of these; the
Streamlit app builds one at a time from the user's toggles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .chunkers import CHUNKERS, Chunk
from .index import EmbeddingIndex
from .reranker import CrossEncoderReranker
from .retrievers import BM25Retriever, HybridRetriever, VectorRetriever

# How deep the base retriever looks before reranking / final cut.
POOL = 20


@dataclass
class Config:
    chunker: str  # "fixed" | "structure"
    search: str  # "vector" | "hybrid"
    rerank: bool

    @property
    def name(self) -> str:
        return f"{self.chunker}+{self.search}+{'rerank' if self.rerank else 'base'}"


class RetrievalPipeline:
    def __init__(
        self,
        documents: dict[str, str],
        config: Config,
        index: EmbeddingIndex | None = None,
        reranker: CrossEncoderReranker | None = None,
    ):
        """documents maps doc_id -> full text.

        An EmbeddingIndex may be passed in so several pipelines can share
        one build (same chunker, different search/rerank settings).
        """
        self.config = config
        chunk_fn = CHUNKERS[config.chunker]
        self.chunks: list[Chunk] = []
        for doc_id, text in documents.items():
            self.chunks.extend(chunk_fn(text, doc_id))
        self.index = index if index is not None else EmbeddingIndex(self.chunks)

        vector = VectorRetriever(self.index)
        if config.search == "vector":
            self.retriever = vector
        elif config.search == "hybrid":
            self.retriever = HybridRetriever(vector, BM25Retriever(self.chunks))
        else:
            raise ValueError(f"unknown search mode: {config.search}")
        self.reranker = reranker

    def query(self, question: str, top_k: int = 10) -> tuple[list[Chunk], float]:
        """Return (top chunks, latency_ms). Rerank uses a POOL-sized pool."""
        start = time.perf_counter()
        if self.config.rerank:
            if self.reranker is None:
                raise RuntimeError("config asks for rerank but no reranker given")
            pool = self.retriever.search(question, POOL)
            ranked = self.reranker.rerank(question, pool, self.chunks, top_k)
        else:
            ranked = self.retriever.search(question, top_k)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return [self.chunks[pos] for pos in ranked], latency_ms
