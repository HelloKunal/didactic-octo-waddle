import json
import os

import numpy as np

from rag import config

RESULTS_DIR = "eval/results"


def questions_path(style="normal"):
    return "eval/questions.json" if style == "normal" else f"eval/questions_{style}.json"


def use_index(name):
    """Point the pipeline at data/index_<name> (or a folder path)."""
    config.INDEX_DIR = name if os.path.isdir(name) else f"{config.DATA_DIR}/index_{name}"
    return config.INDEX_DIR


def index_meta(name=None):
    path = use_index(name) if name else config.INDEX_DIR
    with open(f"{path}/meta.json") as f:
        return json.load(f)


def load_questions(limit=None, style="normal"):
    with open(questions_path(style), encoding="utf-8") as f:
        questions = json.load(f)
    return questions[:limit] if limit else questions


def save(name, data):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"saved {RESULTS_DIR}/{name}.json")


def load(name):
    path = f"{RESULTS_DIR}/{name}.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def percentiles(values, digits=1):
    v = np.asarray(values, dtype=float)
    return {"p50": round(float(np.percentile(v, 50)), digits), "p95": round(float(np.percentile(v, 95)), digits),
            "mean": round(float(v.mean()), digits)}
