from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag import config
from rag.embedding import passage_text
from rag.store import store


@lru_cache
def reranker(name=None):
    return CrossEncoder(name or config.RERANK_MODEL, device=config.EMBED_DEVICE, max_length=512)


def rerank(question, candidates, top_n=config.FINAL_K, min_score=None, model=None):
    """Score each (question, chunk) pair with a cross-encoder and keep the best `top_n`.

    The cross-encoder reads question and chunk together, which is far more accurate than comparing
    separately computed embeddings, but too slow for the whole corpus, so it only sees the candidates.
    """
    if not candidates:
        return []
    ids = [cid for cid, _ in candidates]
    pairs = [(question, passage_text(c)) for c in store().get(ids)]
    scores = reranker(model).predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(zip(ids, scores), key=lambda x: -x[1])
    if min_score is not None:
        ranked = [r for r in ranked if r[1] >= min_score]
    return [(int(cid), float(s)) for cid, s in ranked[:top_n]]
