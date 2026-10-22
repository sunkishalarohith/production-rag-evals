"""Retrieval metrics against gold anchors.

Relevance is defined by the labeled question set: each question carries
a ``gold_anchor`` — an exact, distinctive phrase from the passage that
answers it. A retrieved chunk is relevant iff it contains the anchor
(case-insensitive). That makes relevance checkable without an LLM judge
and fully reproducible.
"""

from __future__ import annotations


def is_relevant(chunk_text: str, gold_anchor: str) -> bool:
    return gold_anchor.lower() in chunk_text.lower()


def recall_at_k(retrieved_texts: list[str], gold_anchor: str, k: int) -> float:
    """1.0 if any of the top-k chunks is relevant, else 0.0.

    With a single gold passage per question, recall@k is the fraction
    of questions whose gold passage appears in the top k.
    """
    return 1.0 if any(is_relevant(t, gold_anchor) for t in retrieved_texts[:k]) else 0.0


def reciprocal_rank(retrieved_texts: list[str], gold_anchor: str, k: int = 10) -> float:
    """1/rank of the first relevant chunk within the top k, else 0."""
    for rank, text in enumerate(retrieved_texts[:k], start=1):
        if is_relevant(text, gold_anchor):
            return 1.0 / rank
    return 0.0
