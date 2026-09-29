from rag.retrieve import minmax, rrf, weighted


def test_rrf_prefers_docs_found_by_both():
    fused = rrf([[(1, 0.9), (2, 0.8)], [(3, 5.0), (2, 4.0)]])
    assert fused[0][0] == 2
    assert {cid for cid, _ in fused} == {1, 2, 3}


def test_rrf_ignores_score_scale():
    a = rrf([[(1, 0.9), (2, 0.1)], [(2, 1000.0), (1, 999.0)]])
    b = rrf([[(1, 0.5), (2, 0.4)], [(2, 2.0), (1, 1.0)]])
    assert a == b


def test_minmax():
    assert minmax([]) == {}
    norm = minmax([(1, 10.0), (2, 5.0), (3, 0.0)])
    assert norm[1] > 0.99 and abs(norm[2] - 0.5) < 1e-6 and norm[3] == 0


def test_weighted_alpha_extremes():
    dense = [(1, 0.9), (2, 0.1)]
    lex = [(2, 9.0), (1, 1.0)]
    assert weighted(dense, lex, alpha=1.0)[0][0] == 1
    assert weighted(dense, lex, alpha=0.0)[0][0] == 2


def test_weighted_handles_empty_bm25():
    assert [cid for cid, _ in weighted([(1, 0.9), (2, 0.5)], [])] == [1, 2]
