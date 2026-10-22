from rag.retrievers import reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    # id 7 is top of both lists -> must win.
    fused = reciprocal_rank_fusion([[7, 1, 2], [7, 3, 4]])
    assert fused[0][0] == 7


def test_rrf_single_list_preserves_order():
    fused = reciprocal_rank_fusion([[5, 9, 11]])
    assert [doc_id for doc_id, _ in fused] == [5, 9, 11]


def test_rrf_scores_are_sums():
    fused = dict(reciprocal_rank_fusion([[1, 2], [2, 1]], k=60))
    # rank1 in one list + rank2 in the other, symmetric -> equal scores
    assert fused[1] == fused[2]
