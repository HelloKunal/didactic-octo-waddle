import json
import re

import requests

from rag import config

THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def build_prompt(question, passages):
    context = "\n\n".join(
        f"[{i}] Title: {p['title']}\n{p['text']}"
        for i, p in enumerate(passages, 1)
    )
    return (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above. Cite the article title for each claim. "
        "If the passages don't contain the answer, say so."
    )


def _request(prompt, stream=False, json_mode=False):
    body = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "think": False,                      # qwen3: skip the <think> block
        "options": {"num_ctx": 8192, "temperature": 0},
    }
    if json_mode:
        body["format"] = "json"
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=body, stream=stream, timeout=300)
    r.raise_for_status()
    return r


def _tokens(data):
    return data.get("prompt_eval_count", 0) + data.get("eval_count", 0)


def chat(prompt, json_mode=False):
    """Returns (text, tokens used)."""
    data = _request(prompt, json_mode=json_mode).json()
    return THINK_BLOCK.sub("", data["message"]["content"]).strip(), _tokens(data)


def chat_stream(prompt):
    """Yields ("token", text) pieces as they arrive, then ("done", tokens used)."""
    with _request(prompt, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if text := data.get("message", {}).get("content"):
                yield "token", text
            if data.get("done"):
                yield "done", _tokens(data)
                return


def generate(question, passages):
    return chat(build_prompt(question, passages))
