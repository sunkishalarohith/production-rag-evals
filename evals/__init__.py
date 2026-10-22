"""Retrieval evaluation harness — standalone.

Import the metrics and runner without the demo app:

    from evals.metrics import recall_at_k, reciprocal_rank
    python -m evals.runner          # full configuration grid

Relevance is gold-anchor based: a chunk is relevant to a question iff it
contains the question's labeled anchor phrase. See questions.jsonl.
"""

from .metrics import is_relevant, recall_at_k, reciprocal_rank

__all__ = ["is_relevant", "recall_at_k", "reciprocal_rank"]
