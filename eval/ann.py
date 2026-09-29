"""How closely FAISS HNSW matches exact search, and how much faster it is, across efSearch values.

    python -m eval.ann --index dynamic
"""
import argparse
import time

import numpy as np

from eval.common import index_meta, load_questions, save, use_index
from rag.dense import dense_index
from rag.embedding import embed_query

K = 50


def main(index, ef_values):
    use_index(index)
    q = np.stack([embed_query(x["question"]) for x in load_questions()])
    exact, hnsw = dense_index("numpy"), dense_index("faiss")

    def run(idx):
        ids, times = [], []
        for v in q:
            start = time.perf_counter()
            ids.append({cid for cid, _ in idx.search(v, K)})
            times.append((time.perf_counter() - start) * 1000)
        return ids, float(np.median(times))

    truth, exact_ms = run(exact)
    rows = []
    for ef in ef_values:
        hnsw.index.hnsw.efSearch = ef
        found, ms = run(hnsw)
        recall = float(np.mean([len(t & f) / K for t, f in zip(truth, found)]))
        rows.append({"efSearch": ef, f"recall@{K}": round(recall, 4), "latency_ms_p50": round(ms, 3)})
        print(rows[-1], flush=True)

    meta = index_meta(index)
    out = {"index": index, "vectors": meta["chunks"], "exact_latency_ms_p50": round(exact_ms, 3), "hnsw": rows}
    print(f"exact search p50 {exact_ms:.3f} ms over {meta['chunks']:,} vectors")
    save(f"ann_{index}", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="dynamic")
    p.add_argument("--ef", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    args = p.parse_args()
    main(args.index, args.ef)
