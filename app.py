"""HTTP API for the RAG pipeline.

    uvicorn app:app --port 8000        # docs at http://localhost:8000/docs
"""
import json
import logging
import os
from contextlib import asynccontextmanager

import redis
import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag import config, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app):
    if os.getenv("RAG_WARMUP", "1") == "1":
        pipeline.warm_up()
    yield


app = FastAPI(title="RAG at Scale", lifespan=lifespan)


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    use_cache: bool = True


@app.post("/query")
def query(q: Query):
    """Answer with sources, whether it came from the cache, and per-stage timings in ms."""
    return pipeline.answer(q.question, q.use_cache)


@app.post("/query/stream")
def query_stream(q: Query):
    """Newline-delimited JSON events: sources, then answer tokens as they're generated, then timings."""
    events = (json.dumps(e) + "\n" for e in pipeline.answer_stream(q.question, q.use_cache))
    return StreamingResponse(events, media_type="application/x-ndjson")


@app.get("/stats")
def stats():
    return pipeline.cache().stats()


@app.delete("/cache")
def clear_cache():
    pipeline.cache().clear()
    return {"cleared": True}


@app.get("/health")
def health():
    status = {"index": config.INDEX_DIR, "redis": False, "llm": False}
    try:
        status["redis"] = bool(pipeline.cache().r.ping())
    except redis.RedisError:
        pass
    try:
        status["llm"] = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=2).ok
    except requests.RequestException:
        pass
    try:
        with open(f"{config.INDEX_DIR}/meta.json") as f:
            status["index_meta"] = json.load(f)
    except OSError:
        pass
    return status
