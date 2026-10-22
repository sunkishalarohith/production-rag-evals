"""Embedding index over chunks: sentence-transformers + FAISS (cosine via IP)."""

from __future__ import annotations

import numpy as np

from .chunkers import Chunk

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingIndex:
    def __init__(self, chunks: list[Chunk], model_name: str = EMBED_MODEL):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return [(chunk_position, score)] best first."""
        query_vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        query_vec = np.asarray(query_vec, dtype="float32")
        k = min(k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)
        return [(int(i), float(s)) for s, i in zip(scores[0], indices[0]) if i >= 0]

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vec, dtype="float32")
