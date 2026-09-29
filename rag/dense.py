from functools import lru_cache

import numpy as np

from rag import config


def build_faiss(embeddings, path):
    import faiss

    index = faiss.IndexHNSWFlat(embeddings.shape[1], config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
    index.add(embeddings)
    faiss.write_index(index, path)


class DenseIndex:
    """Nearest-neighbour search over unit-length chunk embeddings (inner product = cosine)."""

    def __init__(self, index_dir, backend=config.DENSE_BACKEND):
        self.backend = backend
        if backend == "faiss":
            import faiss

            self.index = faiss.read_index(f"{index_dir}/dense.faiss")
            self.index.hnsw.efSearch = config.HNSW_EF_SEARCH
        elif backend == "numpy":
            self.matrix = np.load(f"{index_dir}/embeddings.npy")
        else:
            raise ValueError(f"unknown dense backend {backend!r}")

    def search(self, q_emb, k=config.CANDIDATES):
        if self.backend == "faiss":
            scores, ids = self.index.search(q_emb[None].astype(np.float32), k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

        scores = self.matrix @ q_emb
        k = min(k, len(scores) - 1)
        top = np.argpartition(-scores, k)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]


@lru_cache
def _dense(index_dir, backend):
    return DenseIndex(index_dir, backend)


def dense_index(backend=None):
    return _dense(config.INDEX_DIR, backend or config.DENSE_BACKEND)
