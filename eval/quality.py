"""Answer quality of the original pipeline vs the full one, graded by an LLM judge.

    baseline: fixed 400-word chunks, dense top 5 (what main.py did)
    full:     dynamic chunks, hybrid BM25 + dense (RRF), cross-encoder rerank, top 5

The judge also sees the passage each question was written from, so it can grade correctness.

    python -m eval.quality --n 40
"""
import argparse
import json

import numpy as np

from eval.common import load_questions, save, use_index
from rag import pipeline
from rag.embedding import embed_query
from rag.generate import build_prompt, chat
from rag.retrieve import dense_search
from rag.store import store

JUDGE = """You are grading an answer written by a question-answering system that searches news articles.

Question: {question}

Reference passage (the question was written from this; the system did not see it):
{reference}

Passages the system retrieved:
{passages}

System's answer:
{answer}

Grade from 1 to 5:
- "faithfulness": 5 = every claim in the answer is supported by the retrieved passages; 1 = mostly unsupported or made up.
- "relevance": 5 = directly and fully answers the question; 1 = doesn't answer it (saying the passages lack the answer counts as 1).
- "correctness": 5 = agrees with the reference passage and answers the question; 1 = wrong or missing.
Return JSON: {{"faithfulness": <1-5>, "relevance": <1-5>, "correctness": <1-5>}}"""


def baseline_passages(question):
    use_index("fixed")
    ids = [cid for cid, _ in dense_search(embed_query(question), 5, backend="numpy")]
    return store().get(ids)


def full_passages(question):
    use_index("dynamic")
    return pipeline.retrieve(question)


def judge(q, passages, answer):
    text = "\n\n".join(f"[{i}] {p['title']}\n{p['text']}" for i, p in enumerate(passages, 1))
    try:
        out = json.loads(chat(JUDGE.format(question=q["question"], reference=q["passage"],
                                           passages=text, answer=answer), json_mode=True)[0])
        return {k: int(out[k]) for k in ("faithfulness", "relevance", "correctness")}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def main(n):
    rows = {"baseline": [], "full": []}
    for i, q in enumerate(load_questions(n), 1):
        for name, get in (("baseline", baseline_passages), ("full", full_passages)):
            passages = get(q["question"])
            answer, tokens = chat(build_prompt(q["question"], passages))
            grade = judge(q, passages, answer)
            rows[name].append({
                "id": q["id"],
                "context_hit": any(p["article_id"] == q["article_id"] for p in passages),
                "prompt_chars": len(build_prompt(q["question"], passages)),
                "tokens": tokens,
                "grade": grade,
                "answer": answer,
            })
        print(f"{i}/{n}", {k: v[-1]["grade"] for k, v in rows.items()}, flush=True)

    summary = {}
    for name, rs in rows.items():
        graded = [r["grade"] for r in rs if r["grade"]]
        summary[name] = {
            "questions": len(rs),
            "graded": len(graded),
            "context_hit_rate": round(float(np.mean([r["context_hit"] for r in rs])), 3),
            "tokens_mean": round(float(np.mean([r["tokens"] for r in rs])), 1),
            **{k: round(float(np.mean([g[k] for g in graded])), 2) for k in ("faithfulness", "relevance", "correctness")},
        }
    print(json.dumps(summary, indent=2))
    save("quality", {"summary": summary, "rows": rows})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    main(p.parse_args().n)
