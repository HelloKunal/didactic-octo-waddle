from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from rag import config


@lru_cache
def embed_model():
    return SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)


def embed_query(question):
    return embed_model().encode(
        config.QUERY_PREFIX + question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def embed_passages(texts, batch_size=64, show_progress_bar=False):
    return embed_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def passage_text(chunk):
    """What gets embedded, BM25-indexed and reranked: the title helps every stage."""
    return f"{chunk['title']}\n{chunk['text']}"
