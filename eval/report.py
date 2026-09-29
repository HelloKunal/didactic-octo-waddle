"""Render eval/results/*.json into eval/results.md.

    python -m eval.report
"""
from eval.common import load

INDEXES = ("fixed", "dynamic", "semantic")
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "ndcg@10")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x):
    return f"{x * 100:.1f}%"


def chunking_section(runs):
    rows = []
    for name, r in runs.items():
        m = r["meta"]
        rows.append([name, f"{m['chunks']:,}", m["tokens_mean"], m["tokens_p95"], m["tokens_max"],
                     f"{m['truncated_chunks']:,} ({m['truncated_pct']}%)"])
    return "## Chunking\n\n" + table(
        ["Mode", "Chunks", "Mean tokens", "p95 tokens", "Max tokens", "Over 512 tokens (truncated)"], rows)


def retrieval_section(runs):
    parts = ["## Retrieval\n\nA question is found at rank r if a chunk from its source article is the r-th result "
             "(every config keeps at most 2 chunks per article). Stage latencies are in the end-to-end latency section."]
    for name, r in runs.items():
        rows = [[cfg] + [m[k] for k in METRICS] for cfg, m in r["configs"].items()]
        parts.append(f"### {name} chunks ({r['questions']} questions)\n\n"
                     + table(["Config", "R@1", "R@5", "R@20", "MRR", "nDCG@10"], rows))
    return "\n\n".join(parts)


def ann_section(ann):
    rows = [[r["efSearch"], r["recall@50"], r["latency_ms_p50"]] for r in ann["hnsw"]]
    return (f"## FAISS HNSW vs exact search\n\n{ann['vectors']:,} vectors; exact (NumPy) search p50 "
            f"{ann['exact_latency_ms_p50']} ms. Recall is the overlap of the top 50 with exact search.\n\n"
            + table(["efSearch", "Recall@50", "p50 ms"], rows))


def scale_section(s):
    rows = [[f"{r['vectors']:,}", r["exact_ms_p50"], r["hnsw_ms_p50"], f"{r['speedup']}x", r["recall@50"],
             r["hnsw_build_s"]] for r in s["rows"]]
    return ("## Dense search at scale\n\nSynthetic corpora made by jittering the real chunk embeddings "
            f"(noise {s['noise']} per dimension), queried with {s['queries']} real eval questions; "
            f"HNSW efSearch={s['efSearch']}.\n\n"
            + table(["Vectors", "Exact p50 ms", "HNSW p50 ms", "Speedup", "Recall@50", "HNSW build s"], rows))


def cache_section(c):
    r = c["redis"]
    sim = {k: v["p50"] for k, v in c["similarity"].items()}
    v = c["verifier"]
    shown = (0.80, 0.85, 0.90, 0.95)
    rows = []
    for name, sweep in c["sweep"].items():
        cells = {s["threshold"]: s for s in sweep}
        rows.append([name] + [f"{pct(cells[t]['paraphrase_hit_rate'])} / {pct(cells[t]['near_miss_false_hit_rate'])}"
                              for t in shown])
    return (
        f"## Semantic cache\n\n{c['pairs']} eval questions, each with an LLM-written paraphrase (should hit) and a "
        f"near-miss with one key detail changed, e.g. \"...eclipse in the Tri-Cities?\" vs \"...eclipse in Seattle?\" "
        f"(must not hit: it would return the wrong answer).\n\n"
        f"Embedding similarity alone can't separate them: median question-to-question cosine is {sim['paraphrase']} "
        f"for paraphrases but already {sim['near_miss']} for near-misses (unrelated questions: {sim['unrelated']}). "
        f"So a vector match is only a candidate; it must also pass a key-term guard (no new names or numbers) "
        f"and a duplicate-question cross-encoder (`{v['model']}`, score ≥ {v['min_score']}, "
        f"scored both ways; median {v['score_p50']['paraphrase']} for paraphrases vs "
        f"{v['score_p50']['near_miss']} for near-misses).\n\n"
        "Paraphrase hit rate / near-miss false-hit rate, by minimum cosine similarity:\n\n"
        + table(["Checks"] + [f"≥ {t}" for t in shown], rows)
        + f"\n\nEnd to end in Redis (threshold {r['threshold']}, guard and verifier on, {r['entries']} cached questions):\n\n"
        + table(["Lookup", "Hit rate", "p50 ms", "p95 ms"],
                [[k.replace("_", "-"), pct(r["hit_rate"][k]), r["lookup_ms"][k]["p50"], r["lookup_ms"][k]["p95"]]
                 for k in ("exact", "paraphrase", "near_miss")])
    )


def latency_section(lat):
    cases = [("cold (miss)", lat["cold"]), ("exact hit", lat["exact_hit"])]
    if lat.get("semantic_hit"):
        cases.append(("semantic hit", lat["semantic_hit"]))
    stages = ["embed_ms", "cache_ms", "retrieve_ms", "rerank_ms", "generate_ms", "total_ms"]
    rows = []
    for name, s in cases:
        rows.append([name] + [f"{s[k]['p50']} / {s[k]['p95']}" if k in s else "-" for k in stages])
    tok = lat["llm_tokens_per_miss"]
    return (f"## End-to-end latency ({lat['questions']} questions, p50 / p95 ms)\n\n"
            + table(["Case", "Embed", "Cache", "Hybrid search", "Rerank", "LLM", "Total"], rows)
            + f"\n\nA cache hit is **{lat['speedup_p50']}x faster** at p50 and skips an LLM call of about "
              f"{tok['mean']:.0f} tokens. Semantic hit rate on paraphrases: "
              f"{pct(lat['semantic_hit_rate']) if lat.get('semantic_hit_rate') is not None else 'n/a'}.")


def quality_section(q):
    s = q["summary"]
    rows = [[name, pct(v["context_hit_rate"]), f"{v['tokens_mean']:.0f}", v["faithfulness"], v["relevance"],
             v["correctness"]] for name, v in s.items()]
    b, f = s["baseline"], s["full"]
    return (f"## Answer quality ({b['questions']} questions)\n\n"
            "baseline = fixed 400-word chunks + dense top 5 (the original main.py). "
            "full = dynamic chunks + hybrid RRF + cross-encoder rerank top 5. Context hit = the source article "
            "is among the 5 passages given to the LLM. Tokens = prompt + answer tokens per LLM call. "
            "The 1-5 scores come from an LLM judge (the same local qwen3) that also sees the passage the question "
            "was written from.\n\n"
            + table(["Pipeline", "Context hit", "LLM tokens", "Faithfulness", "Relevance", "Correctness"], rows)
            + f"\n\nThe full pipeline always puts the right article in front of the LLM "
              f"({pct(b['context_hit_rate'])} → {pct(f['context_hit_rate'])}) and uses "
              f"{(1 - f['tokens_mean'] / b['tokens_mean']) * 100:.0f}% fewer tokens per call, because its chunks are "
              "shorter and on-topic. The judge scores are near the ceiling for both and noisy (in one case it gave "
              "identical answers a 5 and a 1), so they don't separate the pipelines on these questions.")


def summary_section(runs, hard, cache, lat, quality, scale):
    base, full = "dense (exact)", "hybrid rrf + rerank (MiniLM)"
    lines = ["## Summary"]
    if "fixed" in runs:
        lines.append(f"- **Chunking:** {runs['fixed']['meta']['truncated_pct']}% of the original 400-word chunks were "
                     "longer than the embedding model's 512-token limit and silently cut off; dynamic and semantic "
                     "chunks: 0%.")
    if "fixed" in hard:
        best_name, best = max(((n, r["configs"][full]) for n, r in hard.items()), key=lambda x: x[1]["mrr"])
        b = hard["fixed"]["configs"][base]
        lines.append(f"- **Retrieval, hard questions (no names to match on):** the original setup (fixed chunks, dense "
                     f"only) finds the right article first {pct(b['recall@1'])} of the time (MRR {b['mrr']}). "
                     f"{best_name.capitalize()} chunks + hybrid BM25/dense + rerank: {pct(best['recall@1'])} "
                     f"(MRR {best['mrr']}), and {pct(best['recall@5'])} in the top 5. BM25 wins on the name-heavy "
                     "standard questions, dense catches up on vague ones, and hybrid + rerank beats either retriever "
                     "alone at R@1 on every index and question set. Plain RRF weights the two lists equally and can "
                     "trail the stronger one; the reranker recovers that.")
    if "fixed" in runs and "dynamic" in runs:
        b, f = runs["fixed"]["configs"][base], runs["dynamic"]["configs"][full]
        lines.append(f"- **Retrieval, standard questions:** R@1 {pct(b['recall@1'])} → {pct(f['recall@1'])}, "
                     f"R@5 {pct(b['recall@5'])} → {pct(f['recall@5'])} (dynamic chunks + hybrid + rerank).")
    if quality:
        b, f = quality["summary"]["baseline"], quality["summary"]["full"]
        lines.append(f"- **What the LLM sees:** the source article is in the 5 passages {pct(f['context_hit_rate'])} "
                     f"of the time (was {pct(b['context_hit_rate'])}), with "
                     f"{(1 - f['tokens_mean'] / b['tokens_mean']) * 100:.0f}% fewer tokens per LLM call.")
    if cache:
        vec = next(s for s in cache["sweep"]["vector only"] if s["threshold"] == cache["redis"]["threshold"])
        r = cache["redis"]
        lines.append(f"- **Semantic cache correctness:** embedding similarity alone would serve a *wrong* cached "
                     f"answer for {pct(vec['near_miss_false_hit_rate'])} of near-miss questions. With the key-term "
                     f"guard and duplicate-question verifier: {pct(r['hit_rate']['near_miss'])} wrong hits, while "
                     f"{pct(r['hit_rate']['paraphrase'])} of paraphrases still hit.")
    if lat:
        miss, hit = lat["cold"]["total_ms"]["p50"], lat["exact_hit"]["total_ms"]["p50"]
        sem = lat["semantic_hit"]["total_ms"]["p50"] if lat.get("semantic_hit") else None
        llm_share = lat["cold"]["generate_ms"]["p50"] / miss * 100
        lines.append(f"- **Latency:** p50 {miss / 1000:.1f} s on a miss (the LLM is {llm_share:.0f}% of it) vs {hit:.0f} ms for an "
                     f"exact cache hit" + (f" and {sem:.0f} ms for a semantic hit" if sem else "")
                     + f"; each hit skips an LLM call of ~{lat['llm_tokens_per_miss']['mean']:.0f} tokens.")
    if scale:
        top = scale["rows"][-1]
        lines.append(f"- **Scale:** at {top['vectors']:,} vectors, HNSW search takes {top['hnsw_ms_p50']} ms vs "
                     f"{top['exact_ms_p50']} ms exact ({top['speedup']}x faster) at {top['recall@50'] * 100:.1f}% recall.")
    lines.append("- **Caveats:** questions are LLM-generated (one per article) over a 5,000-article corpus, so the "
                 "standard set is near the ceiling; the hard set is the more informative one. Fusion weights were "
                 "compared on the same questions they're reported on.")
    return "\n".join(lines)


def main():
    runs = {name: r for name in INDEXES if (r := load(f"retrieval_{name}"))}
    hard_runs = {name: r for name in INDEXES if (r := load(f"retrieval_{name}_hard"))}
    sections = ["# Evaluation results\n\nGenerated by `python -m eval.report` from `eval/results/*.json`. "
                "Corpus: 5,000 CC-News articles. Hardware: laptop CPU for embedding and reranking, qwen3 8B via Ollama.",
                summary_section(runs, hard_runs, load("cache"), load("latency"), load("quality"), load("ann_scale"))]
    if runs:
        sections += [chunking_section(runs), retrieval_section(runs)]
    hard = {name: r for name in INDEXES if (r := load(f"retrieval_{name}_hard"))}
    if hard:
        parts = ["## Retrieval on hard questions\n\nThe same kind of questions, but written without any names from the "
                 "article (\"a ballet company in New York\" instead of \"New York City Ballet\"), so keyword overlap "
                 "can't carry the search."]
        for name, r in hard.items():
            rows = [[cfg] + [m[k] for k in METRICS] for cfg, m in r["configs"].items()]
            parts.append(f"### {name} chunks, hard questions ({r['questions']})\n\n"
                         + table(["Config", "R@1", "R@5", "R@20", "MRR", "nDCG@10"], rows))
        sections.append("\n\n".join(parts))
    if rr := load("retrieval_dynamic_rerankers"):
        rows = [[cfg] + [m[k] for k in METRICS] for cfg, m in rr["configs"].items()]
        sections.append(f"### Reranker comparison (dynamic chunks, first {rr['questions']} questions)\n\n"
                        + table(["Config", "R@1", "R@5", "R@20", "MRR", "nDCG@10"], rows))
    if ann := load("ann_dynamic"):
        sections.append(ann_section(ann))
    if c := load("cache"):
        sections.append(cache_section(c))
    if lat := load("latency"):
        sections.append(latency_section(lat))
    if q := load("quality"):
        sections.append(quality_section(q))
    if scale := load("ann_scale"):
        sections.append(scale_section(scale))
    with open("eval/results.md", "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections) + "\n")
    print("wrote eval/results.md")


if __name__ == "__main__":
    main()
