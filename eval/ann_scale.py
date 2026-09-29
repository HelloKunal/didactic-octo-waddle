"""Exact vs HNSW search as the corpus grows, using synthetic vectors.

Embedding a million real chunks takes hours on a laptop CPU, so this builds bigger corpora by
jittering the real chunk embeddings with noise (keeps the real, clustered distribution) and queries
them with the real eval questions.

    python -m eval.ann_scale --sizes 100000 500000 1000000
"""
import argparse
import time

import faiss
import numpy as np

from eval.common import load_questions, save, use_index
from rag import config
from rag.embedding import embed_query

K = 50


def synthetic(base, n, noise, rng):
    out = np.empty((n, base.shape[1]), dtype=np.float32)
    for start in range(0, n, 100_000):
        m = min(100_000, n - start)
        v = base[rng.integers(0, len(base), m)] + rng.normal(0, noise, (m, base.shape[1])).astype(np.float32)
        out[start:start + m] = v / np.linalg.norm(v, axis=1, keepdims=True)
    return out


def median_ms(fn, queries):
    times = []
    for q in queries:
        start = time.perf_counter()
        fn(q)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def main(sizes, noise, ef):
    index_dir = use_index("dynamic")
    base = np.load(f"{index_dir}/embeddings.npy")
    queries = np.stack([embed_query(q["question"]) for q in load_questions(100)]).astype(np.float32)
    rng = np.random.default_rng(0)
    rows = []
    for n in sizes:
        data = synthetic(base, n, noise, rng)

        truth = [set(np.argpartition(-(data @ q), K)[:K]) for q in queries]
        exact_ms = median_ms(lambda q: np.argpartition(-(data @ q), K)[:K], queries)

        start = time.time()
        index = faiss.IndexHNSWFlat(data.shape[1], config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
        index.add(data)
        build_s = time.time() - start
        index.hnsw.efSearch = ef

        found = [set(index.search(q[None], K)[1][0]) for q in queries]
        recall = float(np.mean([len(t & f) / K for t, f in zip(truth, found)]))
        hnsw_ms = median_ms(lambda q: index.search(q[None], K), queries)
        rows.append({"vectors": n, "exact_ms_p50": round(exact_ms, 2), "hnsw_ms_p50": round(hnsw_ms, 3),
                     "speedup": round(exact_ms / hnsw_ms, 1), f"recall@{K}": round(recall, 4),
                     "hnsw_build_s": round(build_s, 1)})
        print(rows[-1], flush=True)
        del index, data

    save("ann_scale", {"noise": noise, "efSearch": ef, "queries": len(queries), "rows": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[100_000, 500_000, 1_000_000])
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--ef", type=int, default=config.HNSW_EF_SEARCH)
    args = p.parse_args()
    main(args.sizes, args.noise, args.ef)
