import json
import os

os.environ["RAG_WARMUP"] = "0"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    def fake_answer(question, use_cache=True):
        return {"answer": f"echo: {question}", "sources": [], "cache_hit": False, "timings": {"total_ms": 1.0}}

    def fake_stream(question, use_cache=True):
        yield {"type": "sources", "sources": [], "cache_hit": False}
        yield {"type": "token", "text": "hel"}
        yield {"type": "token", "text": "lo"}
        yield {"type": "done", "timings": {"total_ms": 1.0}}

    monkeypatch.setattr(app_module.pipeline, "answer", fake_answer)
    monkeypatch.setattr(app_module.pipeline, "answer_stream", fake_stream)
    with TestClient(app_module.app) as c:
        yield c


def test_query(client):
    r = client.post("/query", json={"question": "what is new?"})
    assert r.status_code == 200 and r.json()["answer"] == "echo: what is new?"


def test_query_validates_input(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_stream(client):
    r = client.post("/query/stream", json={"question": "what is new?"})
    events = [json.loads(line) for line in r.text.splitlines()]
    assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "hello"
