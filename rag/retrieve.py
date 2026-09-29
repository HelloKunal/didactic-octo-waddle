"""Dense, lexical and hybrid retrieval. Every retriever returns [(chunk_id, score)], best first."""
from collections import Counter

import numpy as np

from rag import config
from rag.dense import dense_index
from rag.embedding import embed_query
from rag.lexical import lexical
from rag.store import store


def dense_search(q_emb, k=config.CANDIDATES, backend=None):
    return dense_index(backend).search(q_emb, k)


def lexical_search(question, k=config.CANDIDATES):
    return lexical().search(question, k)


def rrf(result_lists, k=60):
    """Reciprocal Rank Fusion: uses ranks only, so BM25 and cosine scores never need to be comparable."""
    fused = {}
    for results in result_lists:
        for rank, (cid, _) in enumerate(results, 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def minmax(results):
    if not results:
        return {}
    s = np.array([sc for _, sc in results])
    lo, hi = s.min(), s.max()
    return {cid: float((sc - lo) / (hi - lo + 1e-9)) for cid, sc in results}


def weighted(dense, lex, alpha=0.5):
    """Score fusion: alpha * normalized dense score + (1 - alpha) * normalized BM25 score."""
    d, l = minmax(dense), minmax(lex)
    fused = {cid: alpha * d.get(cid, 0) + (1 - alpha) * l.get(cid, 0) for cid in d.keys() | l.keys()}
    return sorted(fused.items(), key=lambda x: -x[1])


def cap_per_article(results, max_per=config.MAX_PER_ARTICLE):
    """Keep at most `max_per` chunks from any one article, so it can't fill every slot."""
    if not results:
        return []
    article = {c["chunk_id"]: c["article_id"] for c in store().get([cid for cid, _ in results])}
    seen, out = Counter(), []
    for cid, s in results:
        if seen[article[cid]] < max_per:
            out.append((cid, s))
            seen[article[cid]] += 1
    return out


def hybrid_search(question, q_emb=None, k=config.RERANK_INPUT, method="rrf", alpha=0.5,
                  candidates=config.CANDIDATES):
    if q_emb is None:
        q_emb = embed_query(question)
    dense = dense_search(q_emb, candidates)
    lex = lexical_search(question, candidates)
    fused = rrf([dense, lex]) if method == "rrf" else weighted(dense, lex, alpha)
    return cap_per_article(fused)[:k]
