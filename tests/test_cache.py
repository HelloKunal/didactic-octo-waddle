import numpy as np
import pytest
import redis

from rag.cache import SemanticCache, adds_key_terms, key_terms


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_cache(**kwargs):
    try:
        c = SemanticCache(namespace="qcache_test", **kwargs)
        c.r.ping()
    except redis.RedisError:
        pytest.skip("Redis isn't running")
    try:                                   # start from an empty index with the right dimension
        c.r.ft(c.index).dropindex(delete_documents=True)
    except redis.ResponseError:
        pass
    c.clear()
    return SemanticCache(namespace="qcache_test", **kwargs)


@pytest.fixture
def cache():
    """Vector path only (4-dim toy vectors), no guard or verifier."""
    c = make_cache(threshold=0.95, dim=4, guard=False, verifier_model=None)
    yield c
    c.clear()
    c.r.ft(c.index).dropindex()


def test_exact_hit_ignores_case_and_spacing(cache):
    cache.put("What is RAG?", unit([1, 0, 0, 0]), {"answer": "a", "sources": [], "tokens": 10})
    hit = cache.get("  what is   rag? ", unit([0, 1, 0, 0]))
    assert hit["answer"] == "a" and hit["cache"] == "exact"


def test_semantic_hit_above_threshold(cache):
    cache.put("What is RAG?", unit([1, 0, 0, 0]), {"answer": "a", "sources": [], "tokens": 10})
    hit = cache.get("Explain RAG", unit([1, 0.1, 0, 0]))       # cosine ~0.995
    assert hit["cache"] == "semantic" and hit["similarity"] >= 0.95


def test_miss_below_threshold(cache):
    cache.put("What is RAG?", unit([1, 0, 0, 0]), {"answer": "a", "sources": [], "tokens": 10})
    assert cache.get("Who won the match?", unit([0.5, 1, 0, 0])) is None    # cosine ~0.45


def test_stats_count_hits_misses_and_tokens(cache):
    cache.put("q1", unit([1, 0, 0, 0]), {"answer": "a", "sources": [], "tokens": 100})
    cache.get("q1", unit([1, 0, 0, 0]))
    cache.get("other", unit([0, 0, 1, 0]))
    s = cache.stats()
    assert s["hits_exact"] == 1 and s["misses"] == 1 and s["tokens_saved"] == 100 and s["entries"] == 1


def test_key_terms():
    assert key_terms("Who won the 2018 World Cup in Russia?") == {"2018", "world", "cup", "russia"}
    assert key_terms("What did Trump's lawyer say?") == {"trump"}


def test_guard_blocks_swapped_names_but_allows_rewording():
    cached = "When is the best time to view the partial eclipse in the Tri-Cities?"
    assert not adds_key_terms("What's the ideal time to see the partial eclipse in the Tri-Cities?", cached)
    assert adds_key_terms("When is the best time to view the partial eclipse in Seattle?", cached)
    assert adds_key_terms("When is the best time to view the partial eclipse in 2019?", cached)


def test_full_cache_with_real_models():
    from rag.embedding import embed_query

    c = make_cache()                       # default threshold, guard and verifier
    cached = "What time does the Zhuangwei Sand-Dune Line bus service operate on weekdays?"
    c.put(cached, embed_query(cached), {"answer": "a", "sources": [], "tokens": 10})
    try:
        paraphrase = "On weekdays, what are the operating hours of the Zhuangwei Sand-Dune Line bus?"
        assert c.get(paraphrase, embed_query(paraphrase))["cache"] == "semantic"
        near_miss = "What time does the Zhuangwei Sand-Dune Line bus service operate on weekends?"
        assert c.get(near_miss, embed_query(near_miss)) is None       # only the verifier catches this one
    finally:
        c.clear()
        c.r.ft(c.index).dropindex()
