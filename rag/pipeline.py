"""question -> embed -> cache? -> hybrid search -> rerank -> LLM -> cache the answer"""
import json
import logging
import time
from contextlib import contextmanager
from functools import lru_cache

import redis

from rag import config
from rag.cache import SemanticCache, duplicate_scores
from rag.embedding import embed_query
from rag.generate import build_prompt, chat, chat_stream
from rag.rerank import rerank
from rag.retrieve import hybrid_search
from rag.store import store

log = logging.getLogger("rag")


@lru_cache
def cache():
    return SemanticCache()


@contextmanager
def timed(timings, name):
    start = time.perf_counter()
    yield
    timings[name] = round((time.perf_counter() - start) * 1000, 1)


def _cache_get(question, q_emb, t):
    try:
        with timed(t, "cache_ms"):
            return cache().get(question, q_emb)
    except redis.RedisError as e:           # the pipeline still works without Redis, just slower
        log.warning("cache unavailable: %s", e)
        return None


def _cache_put(question, q_emb, result):
    try:
        cache().put(question, q_emb, result)
    except redis.RedisError as e:
        log.warning("cache unavailable: %s", e)


def _finish(t):
    t["total_ms"] = round(sum(v for k, v in t.items() if k != "first_token_ms"), 1)
    return t


def _sources(passages):
    seen = {}
    for p in passages:
        seen.setdefault(p["url"], {"title": p["title"], "url": p["url"]})
    return list(seen.values())


def _log(question, result):
    log.info(json.dumps({"question": question, "cache_hit": result["cache_hit"],
                         "cache": result.get("cache"), **result["timings"]}))


def retrieve(question, q_emb=None, t=None, top_n=config.FINAL_K, rerank_model=None):
    """The top passages for a question, best first, each with its rerank score."""
    t = {} if t is None else t
    if q_emb is None:
        with timed(t, "embed_ms"):
            q_emb = embed_query(question)
    with timed(t, "retrieve_ms"):
        candidates = hybrid_search(question, q_emb)
    with timed(t, "rerank_ms"):
        top = rerank(question, candidates, top_n=top_n, model=rerank_model)
    passages = store().get([cid for cid, _ in top])
    return [{**p, "score": round(s, 4)} for p, (_, s) in zip(passages, top)]


def answer(question, use_cache=True, return_passages=False):
    t = {}
    with timed(t, "embed_ms"):
        q_emb = embed_query(question)

    if use_cache and (hit := _cache_get(question, q_emb, t)):
        result = {**hit, "cache_hit": True, "timings": _finish(t)}
        _log(question, result)
        return result

    passages = retrieve(question, q_emb, t)
    with timed(t, "generate_ms"):
        text, tokens = chat(build_prompt(question, passages))

    result = {"answer": text, "sources": _sources(passages), "tokens": tokens}
    if use_cache:
        _cache_put(question, q_emb, result)
    result = {**result, "cache_hit": False, "timings": _finish(t)}
    if return_passages:
        result["passages"] = passages
    _log(question, result)
    return result


def answer_stream(question, use_cache=True):
    """Same as answer(), as events: sources first, then answer tokens as the LLM writes them."""
    t = {}
    with timed(t, "embed_ms"):
        q_emb = embed_query(question)

    if use_cache and (hit := _cache_get(question, q_emb, t)):
        yield {"type": "sources", "sources": hit["sources"], "cache_hit": True, "cache": hit["cache"]}
        yield {"type": "token", "text": hit["answer"]}
        yield {"type": "done", "timings": _finish(t)}
        _log(question, {"cache_hit": True, "cache": hit["cache"], "timings": t})
        return

    passages = retrieve(question, q_emb, t)
    yield {"type": "sources", "sources": _sources(passages), "cache_hit": False}

    start, parts, tokens = time.perf_counter(), [], 0
    for kind, value in chat_stream(build_prompt(question, passages)):
        if kind == "token":
            if not parts:
                t["first_token_ms"] = round((time.perf_counter() - start) * 1000, 1)
            parts.append(value)
            yield {"type": "token", "text": value}
        else:
            tokens = value
    t["generate_ms"] = round((time.perf_counter() - start) * 1000, 1)

    result = {"answer": "".join(parts).strip(), "sources": _sources(passages), "tokens": tokens}
    if use_cache:
        _cache_put(question, q_emb, result)
    yield {"type": "done", "timings": _finish(t)}
    _log(question, {"cache_hit": False, "timings": t})


def warm_up():
    """Load every model and index up front, so the first request isn't slow."""
    retrieve("warm up query")
    try:
        c = cache()
        if c.verifier_model:
            duplicate_scores(c.verifier_model, [("warm up", "warm up")])
    except redis.RedisError as e:
        log.warning("cache unavailable: %s", e)
