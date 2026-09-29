"""Download CC-News articles to a JSONL file. Resumable: rerun with a bigger --n to extend the file.

    python -m rag.ingest --n 5000
"""
import argparse
import json
import os
import time

from datasets import load_dataset

from rag import config


def _read_checkpoint(path):
    ckpt = path + ".checkpoint"
    if not (os.path.exists(path) and os.path.exists(ckpt)):
        return {"stream_pos": 0, "articles": 0, "offset": 0}
    with open(ckpt) as f:
        return json.load(f)


def _write_checkpoint(path, state):
    with open(path + ".checkpoint", "w") as f:
        json.dump(state, f)


def download_articles(n, path=config.ARTICLES_PATH, log_every=1000):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = _read_checkpoint(path)
    if state["articles"] >= n:
        print(f"{path} already has {state['articles']} articles")
        return

    # drop anything written after the last checkpoint, then rebuild the duplicate filter
    seen = set()
    if state["offset"]:
        with open(path, "r+b") as f:
            f.truncate(state["offset"])
        with open(path, encoding="utf-8") as f:
            seen = {json.loads(line)["title"].lower() for line in f}

    stream = load_dataset("vblagoje/cc_news", split="train", streaming=True)
    if state["stream_pos"]:
        stream = stream.skip(state["stream_pos"])

    start, count, pos = time.time(), state["articles"], state["stream_pos"]
    with open(path, "ab" if state["offset"] else "wb") as f:
        for ex in stream:
            pos += 1
            title = (ex["title"] or "").strip()
            body = (ex["text"] or "").strip()
            if not body or (title and title.lower() in seen):   # skip empty and syndicated duplicates
                continue
            seen.add(title.lower())
            row = {"id": count, "title": title, "url": ex["url"], "date": ex.get("date"), "body": body}
            f.write((json.dumps(row) + "\n").encode("utf-8"))
            count += 1
            if count % log_every == 0 or count == n:
                f.flush()
                _write_checkpoint(path, {"stream_pos": pos, "articles": count, "offset": f.tell()})
                print(f"{count:,} articles ({pos:,} read, {time.time() - start:.0f}s)", flush=True)
            if count >= n:
                break
        f.flush()
        _write_checkpoint(path, {"stream_pos": pos, "articles": count, "offset": f.tell()})


def read_articles(path=config.ARTICLES_PATH, limit=None):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--path", default=config.ARTICLES_PATH)
    args = p.parse_args()
    download_articles(args.n, args.path)
