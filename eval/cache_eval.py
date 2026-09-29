"""Tune the semantic cache, then check it end to end in Redis.

For each eval question the LLM writes a paraphrase (same meaning: should hit the cache) and a
near-miss (one key detail changed: must NOT hit, or the user gets the wrong answer). Compares:
vector similarity alone, + key-term guard, + duplicate-question verifier, and all three.

    python -m eval.cache_eval --n 100
"""
import argparse
import json
import os
import random
import time

import numpy as np

from eval.common import load_questions, percentiles, save
from rag import config
from rag.cache import SemanticCache, adds_key_terms, duplicate_scores
from rag.embedding import embed_query
from rag.generate import chat

PAIRS_PATH = "eval/cache_pairs.json"
MAX_FALSE_HITS = 0.02
THRESHOLDS = np.round(np.arange(0.80, 0.975, 0.01), 2)

PROMPT = """Question: {q}

Return JSON with two fields:
"paraphrase": the same question reworded, the way a different person might ask it. Same meaning, same answer.
"different": the same question with one key detail changed (a name, date, place, number or event) so that it needs a different answer."""


def _norm(s):
    return " ".join(s.lower().split())


def make_pairs(n):
    if os.path.exists(PAIRS_PATH):
        with open(PAIRS_PATH, encoding="utf-8") as f:
            pairs = json.load(f)
    else:
        pairs = []
        for q in load_questions(n):
            try:
                out = json.loads(chat(PROMPT.format(q=q["question"]), json_mode=True)[0])
                if out.get("paraphrase") and out.get("different"):
                    pairs.append({"question": q["question"], "paraphrase": out["paraphrase"],
                                  "different": out["different"]})
            except (json.JSONDecodeError, KeyError):
                pass
        with open(PAIRS_PATH, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=1, ensure_ascii=False)
    # drop pairs where the LLM just repeated the question
    return [p for p in pairs[:n]
            if _norm(p["different"]) != _norm(p["question"]) and _norm(p["paraphrase"]) != _norm(p["question"])]


def rates(sim, passes, th):
    return round(float(((sim >= th) & passes).mean()), 3)


def main(n):
    pairs = make_pairs(n)
    orig = np.stack([embed_query(p["question"]) for p in pairs])
    para = np.stack([embed_query(p["paraphrase"]) for p in pairs])
    diff = np.stack([embed_query(p["different"]) for p in pairs])
    same_sim, diff_sim = (orig * para).sum(1), (orig * diff).sum(1)
    shuffled = random.Random(0).sample(range(len(pairs)), len(pairs))
    unrelated = np.array([orig[i] @ orig[j] for i, j in enumerate(shuffled) if i != j])

    duplicate_scores(config.CACHE_VERIFIER, [("warm", "up")])
    start = time.perf_counter()
    v_same = np.array([duplicate_scores(config.CACHE_VERIFIER, [(p["paraphrase"], p["question"])])[0] for p in pairs])
    verifier_ms = (time.perf_counter() - start) * 1000 / len(pairs)
    v_diff = duplicate_scores(config.CACHE_VERIFIER, [(p["different"], p["question"]) for p in pairs])
    g_same = np.array([not adds_key_terms(p["paraphrase"], p["question"]) for p in pairs])
    g_diff = np.array([not adds_key_terms(p["different"], p["question"]) for p in pairs])
    ok = np.ones(len(pairs), dtype=bool)

    variants = {
        "vector only": (ok, ok),
        "vector + key-term guard": (g_same, g_diff),
        "vector + verifier": (v_same >= config.CACHE_VERIFIER_MIN, v_diff >= config.CACHE_VERIFIER_MIN),
        "vector + guard + verifier": (g_same & (v_same >= config.CACHE_VERIFIER_MIN),
                                      g_diff & (v_diff >= config.CACHE_VERIFIER_MIN)),
    }
    sweep, best = {}, {}
    for name, (ps, pd) in variants.items():
        sweep[name] = [{"threshold": float(th), "paraphrase_hit_rate": rates(same_sim, ps, th),
                        "near_miss_false_hit_rate": rates(diff_sim, pd, th)} for th in THRESHOLDS]
        allowed = [s for s in sweep[name] if s["near_miss_false_hit_rate"] <= MAX_FALSE_HITS]
        best[name] = max(allowed, key=lambda s: (s["paraphrase_hit_rate"], s["threshold"])) if allowed else None
        at_default = next(s for s in sweep[name] if s["threshold"] == config.CACHE_THRESHOLD)
        print(f"{name:28} best {best[name]}   at {config.CACHE_THRESHOLD}: {at_default}")

    # end to end in Redis with the full cache, in a separate namespace so real entries are untouched
    cache = SemanticCache(namespace="qcache_eval")
    cache.clear()
    put_ms, lookup = [], {"exact": [], "paraphrase": [], "near_miss": []}
    hits = dict.fromkeys(lookup, 0)
    for p, e in zip(pairs, orig):
        start = time.perf_counter()
        cache.put(p["question"], e, {"answer": "cached", "sources": [], "tokens": 1500})
        put_ms.append((time.perf_counter() - start) * 1000)
    for kind, texts, embs in (("exact", [p["question"] for p in pairs], orig),
                              ("paraphrase", [p["paraphrase"] for p in pairs], para),
                              ("near_miss", [p["different"] for p in pairs], diff)):
        for text, e in zip(texts, embs):
            start = time.perf_counter()
            hits[kind] += cache.get(text, e) is not None
            lookup[kind].append((time.perf_counter() - start) * 1000)
    redis_check = {
        "threshold": cache.threshold, "guard": cache.guard, "verifier": cache.verifier_model,
        "entries": len(pairs),
        "hit_rate": {k: round(v / len(pairs), 3) for k, v in hits.items()},
        "lookup_ms": {k: percentiles(v, 2) for k, v in lookup.items()},
        "put_ms": percentiles(put_ms, 2),
    }
    cache.clear()
    cache.r.ft(cache.index).dropindex()
    print(json.dumps(redis_check, indent=2))

    save("cache", {
        "pairs": len(pairs),
        "similarity": {"paraphrase": percentiles(same_sim, 3), "near_miss": percentiles(diff_sim, 3),
                       "unrelated": percentiles(unrelated, 3)},
        "verifier": {"model": config.CACHE_VERIFIER, "min_score": config.CACHE_VERIFIER_MIN,
                     "ms_per_check": round(verifier_ms, 1),
                     "score_p50": {"paraphrase": round(float(np.median(v_same)), 3),
                                   "near_miss": round(float(np.median(v_diff)), 3)}},
        "max_false_hits": MAX_FALSE_HITS,
        "best": best,
        "sweep": sweep,
        "redis": redis_check,
    })


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    main(p.parse_args().n)
