"""Chunk the articles, embed them, and build the dense (FAISS) and lexical (BM25) indexes.

    python -m rag.index --mode fixed      # original 400-word chunks (baseline)
    python -m rag.index --mode dynamic    # sentence-aware, token-sized chunks
    python -m rag.index --mode semantic   # dynamic + breaks where the topic shifts

Each mode writes to its own folder (data/index_<mode>) so they can be compared in eval.
"""
import argparse
import json
import os
import shutil
import time
from itertools import islice

import numpy as np

from rag import config
from rag.chunking import chunk_text, chunk_words
from rag.dense import build_faiss
from rag.embedding import embed_model, embed_passages, passage_text
from rag.ingest import read_articles
from rag.lexical import build_bm25
from rag.store import ChunkStore

INDEX_FILES = ("chunks.db", "embeddings.npy", "dense.faiss", "meta.json")


def _batched(iterable, n):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


def _clear(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name in INDEX_FILES:
        if os.path.exists(f"{out_dir}/{name}"):
            os.remove(f"{out_dir}/{name}")
    shutil.rmtree(f"{out_dir}/bm25", ignore_errors=True)


def build(mode="dynamic", out_dir=None, articles_path=config.ARTICLES_PATH, limit=None, batch_articles=1000):
    out_dir = out_dir or f"{config.DATA_DIR}/index_{mode}"
    _clear(out_dir)
    model = embed_model()
    tokenizer = model.tokenizer
    store = ChunkStore(f"{out_dir}/chunks.db")

    timings = {"chunk_s": 0.0, "embed_s": 0.0}
    embeddings, token_lens, n_articles, n_chunks = [], [], 0, 0
    start = time.time()

    for batch in _batched(read_articles(articles_path, limit), batch_articles):
        t = time.time()
        rows = []
        for art in batch:
            if mode == "fixed":
                pieces = chunk_words(art["body"])
            else:
                pieces = chunk_text(art["body"], tokenizer, embed=model if mode == "semantic" else None)
            for piece in pieces:
                rows.append((n_chunks, art["id"], art["title"], art["url"], piece))
                n_chunks += 1
        store.add(rows)
        texts = [passage_text({"title": r[2], "text": r[4]}) for r in rows]
        token_lens += [len(ids) for ids in tokenizer(texts)["input_ids"]]   # includes [CLS]/[SEP]
        timings["chunk_s"] += time.time() - t

        t = time.time()
        embeddings.append(embed_passages(texts))
        timings["embed_s"] += time.time() - t

        n_articles += len(batch)
        print(f"{n_articles:,} articles -> {n_chunks:,} chunks ({time.time() - start:.0f}s)", flush=True)

    emb = np.vstack(embeddings)
    np.save(f"{out_dir}/embeddings.npy", emb)

    t = time.time()
    build_faiss(emb, f"{out_dir}/dense.faiss")
    timings["faiss_s"] = time.time() - t

    t = time.time()
    build_bm25([passage_text(c) for c in store.iter_all()], f"{out_dir}/bm25")
    timings["bm25_s"] = time.time() - t

    lens = np.array(token_lens)
    meta = {
        "mode": mode,
        "articles": n_articles,
        "chunks": n_chunks,
        "tokens_mean": round(float(lens.mean()), 1),
        "tokens_p95": int(np.percentile(lens, 95)),
        "tokens_max": int(lens.max()),
        "truncated_chunks": int((lens > model.max_seq_length).sum()),
        "truncated_pct": round(float((lens > model.max_seq_length).mean() * 100), 1),
        "embed_chunks_per_s": round(n_chunks / timings["embed_s"], 1),
        **{k: round(v, 1) for k, v in timings.items()},
        "total_s": round(time.time() - start, 1),
    }
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    store.close()
    _clear_cache()
    print(json.dumps(meta, indent=2))
    return meta


def _clear_cache():
    """Cached answers cite chunks from the old index, so drop them after a rebuild."""
    try:
        from rag.cache import SemanticCache

        SemanticCache().clear()
    except Exception:
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fixed", "dynamic", "semantic"], default="dynamic")
    p.add_argument("--out", help="output folder (default data/index_<mode>)")
    p.add_argument("--articles", default=config.ARTICLES_PATH)
    p.add_argument("--limit", type=int, help="only use the first N articles")
    args = p.parse_args()
    build(args.mode, args.out, args.articles, args.limit)
