"""Semantic cache in Redis.

A lookup tries, in order:
  1. exact match on the normalized question (a hash lookup)
  2. the most similar cached questions by embedding (Redis HNSW vector index), then two checks, because
     embeddings score "...eclipse in the Tri-Cities?" and "...eclipse in Seattle?" as near-identical:
     - key-term guard: the new question may not name anything (proper noun, number) the cached one doesn't
     - verifier: a cross-encoder trained on duplicate questions must agree they ask the same thing
"""
import hashlib
import json
import re
from functools import lru_cache

import numpy as np
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from rag import config

TOKEN = re.compile(r"[A-Za-z][A-Za-z']*|\d+(?:[.,:]\d+)*")


def key_terms(question):
    """Capitalized words and numbers, skipping the sentence-initial word."""
    return {re.sub(r"'s?$", "", t).lower() for t in TOKEN.findall(question)[1:] if t[0].isupper() or t[0].isdigit()}


def adds_key_terms(new, cached):
    return bool(key_terms(new) - key_terms(cached))


@lru_cache
def verifier(name):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device=config.EMBED_DEVICE)


def duplicate_scores(model, pairs):
    """How sure the verifier is that each (question, question) pair asks the same thing, from 0 to 1.

    The cross-encoder isn't symmetric, so score both orders and keep the lower one.
    """
    if not pairs:
        return np.array([])
    both = [p for a, b in pairs for p in ((a, b), (b, a))]
    scores = np.asarray(verifier(model).predict(both, batch_size=32, show_progress_bar=False))
    return scores.reshape(-1, 2).min(axis=1)


class SemanticCache:
    def __init__(self, url=None, threshold=None, ttl=None, namespace=None, dim=config.EMBED_DIM,
                 guard=None, verifier_model=..., verifier_min=None, candidates=3):
        self.r = redis.Redis.from_url(url or config.REDIS_URL, socket_connect_timeout=2, socket_timeout=5)
        self.threshold = config.CACHE_THRESHOLD if threshold is None else threshold
        self.ttl = ttl or config.CACHE_TTL
        self.ns = namespace or config.CACHE_NAMESPACE
        self.guard = config.CACHE_KEY_TERM_GUARD if guard is None else guard
        self.verifier_model = config.CACHE_VERIFIER if verifier_model is ... else verifier_model
        self.verifier_min = config.CACHE_VERIFIER_MIN if verifier_min is None else verifier_min
        self.candidates = candidates
        self.index = f"{self.ns}_idx"
        try:
            self.r.ft(self.index).info()
        except redis.ResponseError:          # index doesn't exist yet
            self.r.ft(self.index).create_index(
                [TextField("query"),
                 VectorField("embedding", "HNSW", {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"})],
                definition=IndexDefinition(prefix=[f"{self.ns}:entry:"], index_type=IndexType.HASH),
            )

    @staticmethod
    def _key(question):
        return hashlib.sha256(" ".join(question.lower().split()).encode()).hexdigest()

    def nearest(self, q_emb, k=None):
        """[(similarity, cached question, cached response json)] for the k most similar cached questions."""
        k = k or self.candidates
        q = (Query(f"*=>[KNN {k} @embedding $vec AS dist]")
             .sort_by("dist").return_fields("dist", "query", "response").dialect(2))
        res = self.r.ft(self.index).search(q, query_params={"vec": q_emb.astype(np.float32).tobytes()})
        return [(1 - float(d.dist), d.query, d.response) for d in res.docs]   # COSINE distance = 1 - similarity

    def match(self, question, q_emb, threshold=None):
        """The best cached entry that passes every check: (similarity, verifier score, response json) or None."""
        threshold = self.threshold if threshold is None else threshold
        found = [(sim, q, resp) for sim, q, resp in self.nearest(q_emb) if sim >= threshold]
        if self.guard:
            found = [f for f in found if not adds_key_terms(question, f[1])]
        if not found:
            return None
        if not self.verifier_model:
            sim, _, resp = found[0]
            return sim, None, resp
        scores = duplicate_scores(self.verifier_model, [(question, q) for _, q, _ in found])
        best = int(np.argmax(scores))
        if scores[best] < self.verifier_min:
            return None
        return found[best][0], float(scores[best]), found[best][2]

    def get(self, question, q_emb, threshold=None):
        raw = self.r.get(f"{self.ns}:exact:{self._key(question)}")
        kind, sim, score = "exact", 1.0, None
        if raw is None:
            kind = "semantic"
            m = self.match(question, q_emb, threshold)
            if m is None:
                self.r.incr(f"{self.ns}:stats:misses")
                return None
            sim, score, raw = m
        payload = json.loads(raw)
        self.r.incr(f"{self.ns}:stats:hits_{kind}")
        self.r.incrby(f"{self.ns}:stats:tokens_saved", payload.get("tokens", 0))
        hit = {**payload, "cache": kind, "similarity": round(sim, 4)}
        if score is not None:
            hit["verifier_score"] = round(score, 4)
        return hit

    def put(self, question, q_emb, payload):
        data = json.dumps(payload)
        key = self._key(question)
        pipe = self.r.pipeline()
        pipe.hset(f"{self.ns}:entry:{key}", mapping={
            "query": question,
            "embedding": q_emb.astype(np.float32).tobytes(),
            "response": data,
        })
        pipe.expire(f"{self.ns}:entry:{key}", self.ttl)
        pipe.set(f"{self.ns}:exact:{key}", data, ex=self.ttl)
        pipe.execute()

    def stats(self):
        names = ("hits_exact", "hits_semantic", "misses", "tokens_saved")
        values = self.r.mget([f"{self.ns}:stats:{n}" for n in names])
        s = {n: int(v or 0) for n, v in zip(names, values)}
        lookups = s["hits_exact"] + s["hits_semantic"] + s["misses"]
        s["hit_rate"] = round((s["hits_exact"] + s["hits_semantic"]) / lookups, 3) if lookups else 0.0
        s["entries"] = int(self.r.ft(self.index).info()["num_docs"])
        return s

    def clear(self):
        keys = list(self.r.scan_iter(f"{self.ns}:*", count=1000))
        for i in range(0, len(keys), 1000):
            self.r.delete(*keys[i:i + 1000])
