"""Retrieval quality of every retriever configuration, per index.

A question counts as found at rank r if a chunk of its source article is the r-th result.
Every configuration keeps at most 2 chunks per article, so they're compared like for like.

    python -m eval.retrieval --index dynamic
    python -m eval.retrieval --index dynamic --questions hard     # questions that avoid the article's names

bge-reranker-base is slow on CPU, so compare rerankers on a subset:

    python -m eval.retrieval --index dynamic --limit 60 --bge --tag rerankers
        --only "hybrid rrf" "hybrid rrf + rerank (MiniLM)" "hybrid rrf + rerank (bge-base)"
"""
import argparse
import math
import time

import numpy as np

from eval.common import index_meta, load_questions, save, use_index
from rag.embedding import embed_query
from rag.rerank import rerank
from rag.retrieve import cap_per_article, dense_search, hybrid_search, lexical_search
from rag.store import store

K = 20
BGE = "BAAI/bge-reranker-base"


def first_hit_rank(results, gold_article):
    articles = [c["article_id"] for c in store().get([cid for cid, _ in results])]
    return articles.index(gold_article) + 1 if gold_article in articles else None


def metrics(ranks):
    ranks = [r or math.inf for r in ranks]
    out = {f"recall@{k}": round(float(np.mean([r <= k for r in ranks])), 3) for k in (1, 5, 10, 20)}
    out["mrr"] = round(float(np.mean([1 / r for r in ranks])), 3)
    out["ndcg@10"] = round(float(np.mean([1 / math.log2(r + 1) if r <= 10 else 0 for r in ranks])), 3)
    return out


def configs(bge):
    c = {
        "dense (exact)": lambda q, e: cap_per_article(dense_search(e, 50, backend="numpy"))[:K],
        "dense (HNSW)": lambda q, e: cap_per_article(dense_search(e, 50))[:K],
        "bm25": lambda q, e: cap_per_article(lexical_search(q, 50))[:K],
        "hybrid weighted a=0.3": lambda q, e: hybrid_search(q, e, k=K, method="weighted", alpha=0.3),
        "hybrid weighted a=0.5": lambda q, e: hybrid_search(q, e, k=K, method="weighted", alpha=0.5),
        "hybrid weighted a=0.7": lambda q, e: hybrid_search(q, e, k=K, method="weighted", alpha=0.7),
        "hybrid rrf": lambda q, e: hybrid_search(q, e, k=K),
        "hybrid rrf + rerank (MiniLM)": lambda q, e: rerank(q, hybrid_search(q, e, k=30), top_n=K),
    }
    if bge:
        c["hybrid rrf + rerank (bge-base)"] = lambda q, e: rerank(q, hybrid_search(q, e, k=30), top_n=K, model=BGE)
    return c


def main(index, limit, bge, only, tag, style="normal"):
    use_index(index)
    questions = load_questions(limit, style)
    q_embs = [embed_query(q["question"]) for q in questions]
    results = {}
    for name, fn in configs(bge).items():
        if only and name not in only:
            continue
        fn(questions[0]["question"], q_embs[0])            # warm up models and indexes
        ranks, times = [], []
        for q, e in zip(questions, q_embs):
            start = time.perf_counter()
            res = fn(q["question"], e)
            times.append((time.perf_counter() - start) * 1000)
            ranks.append(first_hit_rank(res, q["article_id"]))
        results[name] = {**metrics(ranks), "latency_ms_p50": round(float(np.median(times)), 1)}
        print(f"{index:9} {name:32}", "  ".join(f"{k} {v}" for k, v in results[name].items()), flush=True)
    suffix = "".join(f"_{x}" for x in (tag, style if style != "normal" else None) if x)
    save(f"retrieval_{index}{suffix}",
         {"index": index, "meta": index_meta(index), "questions": len(questions), "style": style, "configs": results})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="dynamic")
    p.add_argument("--limit", type=int)
    p.add_argument("--bge", action="store_true", help="also run the slower bge-reranker-base")
    p.add_argument("--only", nargs="+", help="run only these configs")
    p.add_argument("--tag", help="suffix for the results file")
    p.add_argument("--questions", choices=["normal", "hard"], default="normal")
    args = p.parse_args()
    main(args.index, args.limit, args.bge, args.only, args.tag, args.questions)
