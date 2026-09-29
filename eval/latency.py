"""End-to-end latency per stage for three cases:

    cold      cache miss: embed -> hybrid search -> rerank -> LLM
    exact     the same questions again (exact-match cache hits)
    semantic  paraphrased questions (vector cache hits, if similar enough)

Run it with nothing else busy on the machine.

    python -m eval.latency --n 40
"""
import argparse
import json
import os

from eval.common import load_questions, percentiles, save, use_index
from rag import config, pipeline

PAIRS_PATH = "eval/cache_pairs.json"


def summarize(results):
    stages = sorted({k for r in results for k in r["timings"]})
    return {s: percentiles([r["timings"][s] for r in results if s in r["timings"]]) for s in stages}


def main(n, index):
    use_index(index)
    config.CACHE_NAMESPACE = "qcache_latency"         # keep real cache entries out of it
    cache = pipeline.cache()
    cache.clear()
    pipeline.warm_up()

    questions = [q["question"] for q in load_questions(n)]
    cold = [r for r in (pipeline.answer(q) for q in questions) if not r["cache_hit"]]
    exact = [pipeline.answer(q) for q in questions]

    semantic, semantic_misses = [], 0
    if os.path.exists(PAIRS_PATH):
        with open(PAIRS_PATH, encoding="utf-8") as f:
            paraphrases = {p["question"]: p["paraphrase"] for p in json.load(f)}
        for q in questions:
            if q in paraphrases:
                r = pipeline.answer(paraphrases[q])
                if r.get("cache") == "semantic":
                    semantic.append(r)
                else:
                    semantic_misses += 1

    stats = cache.stats()
    out = {
        "index": index,
        "questions": len(questions),
        "cold": summarize(cold),
        "exact_hit": summarize(exact),
        "semantic_hit": summarize(semantic) if semantic else None,
        "semantic_hit_rate": round(len(semantic) / (len(semantic) + semantic_misses), 3) if semantic or semantic_misses else None,
        "llm_tokens_per_miss": percentiles([r["tokens"] for r in cold]),
        "cache_stats": stats,
        "speedup_p50": round(summarize(cold)["total_ms"]["p50"] / summarize(exact)["total_ms"]["p50"], 1),
    }
    cache.clear()
    print(json.dumps(out, indent=2))
    save("latency", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--index", default="dynamic")
    args = p.parse_args()
    main(args.n, args.index)
