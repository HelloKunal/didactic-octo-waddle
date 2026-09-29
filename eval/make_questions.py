"""Build the retrieval eval set: one generated question per sampled article.

Each question is written from a random passage of the raw article, not from any chunker's output,
so no chunking mode gets an unfair advantage. Ground truth is the article id.

    python -m eval.make_questions --n 250
    python -m eval.make_questions --n 150 --style hard   # no names from the article: keyword search can't cheat
"""
import argparse
import json
import random
import re

from eval.common import questions_path
from rag.cache import key_terms
from rag.chunking import split_sentences
from rag.generate import chat
from rag.ingest import read_articles

PROMPT = """Here is a passage from a news article titled "{title}".

{passage}

Write one specific question that this passage answers. Someone who hasn't read the passage must be
able to understand the question, so name the people, places and things involved.
Use your own words: don't copy phrases from the passage, and don't refer to "the passage" or "the article".
Return only the question."""

HARD_PROMPT = """Here is a passage from a news article titled "{title}".

{passage}

Write one question that this passage answers, for testing a search engine that must find this story
from a vague description. Do NOT use any names: no people, organizations, places, teams, products or
titles from the passage. Describe them instead (e.g. "a Spanish prime minister" instead of "Mariano Rajoy",
"a ballet company in New York" instead of "New York City Ballet"). The question must still clearly be about
this specific story. Don't refer to "the passage" or "the article". Return only the question."""

PROMPTS = {"normal": PROMPT, "hard": HARD_PROMPT}

BAD = re.compile(r"\b(the|this) (passage|article|text|report|story)\b", re.I)


def random_passage(body, rng, min_words=120, max_words=250):
    sents = split_sentences(body)
    start = rng.randrange(len(sents))
    while start > 0 and sum(len(s.split()) for s in sents[start:]) < min_words:
        start -= 1
    out, n = [], 0
    for s in sents[start:]:
        if out and n + len(s.split()) > max_words:
            break
        out.append(s)
        n += len(s.split())
    return " ".join(out)


def main(n, n_articles, seed, style):
    path = questions_path(style)
    rng = random.Random(seed)
    articles = [a for a in read_articles(limit=n_articles) if len(a["body"].split()) >= 150 and a["title"]]
    rng.shuffle(articles)

    questions, rejected = [], 0
    for art in articles:
        if len(questions) == n:
            break
        passage = random_passage(art["body"], rng)
        q, _ = chat(PROMPTS[style].format(title=art["title"], passage=passage))
        q = q.strip().strip('"').splitlines()[0].strip() if q.strip() else ""
        names_used = key_terms(q) & key_terms(art["title"] + " " + passage) if style == "hard" else set()
        if not q.endswith("?") or not 6 <= len(q.split()) <= 40 or BAD.search(q) or names_used:
            rejected += 1
            continue
        questions.append({"id": len(questions), "question": q, "article_id": art["id"],
                          "title": art["title"], "passage": passage})
        if len(questions) % 25 == 0:
            print(f"{len(questions)} questions ({rejected} rejected)", flush=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(questions)} questions to {path} ({rejected} rejected)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=250)
    p.add_argument("--articles", type=int, default=5000, help="sample from the first N articles")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--style", choices=["normal", "hard"], default="normal")
    args = p.parse_args()
    main(args.n, args.articles, args.seed, args.style)
