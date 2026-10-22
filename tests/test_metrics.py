from evals.metrics import is_relevant, recall_at_k, reciprocal_rank

TEXTS = ["alpha beta", "gamma delta anchor phrase here", "epsilon zeta"]
ANCHOR = "anchor phrase"


def test_is_relevant_case_insensitive():
    assert is_relevant("The Anchor Phrase appears", ANCHOR)
    assert not is_relevant("nothing relevant", ANCHOR)


def test_recall_at_k():
    assert recall_at_k(TEXTS, ANCHOR, 1) == 0.0  # hit is at rank 2
    assert recall_at_k(TEXTS, ANCHOR, 2) == 1.0
    assert recall_at_k(TEXTS, "missing phrase", 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(TEXTS, ANCHOR, 10) == 0.5
    assert reciprocal_rank(TEXTS, ANCHOR, 1) == 0.0
    assert reciprocal_rank(TEXTS, "missing phrase", 10) == 0.0
