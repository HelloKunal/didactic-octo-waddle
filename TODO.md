# TODO: RAG at Scale

Goal (resume line):
> End-to-end RAG with **dynamic document chunking**, **hybrid search (BM25 + dense vectors)**,
> **cross-encoder reranking**, and a **Redis semantic cache** to cut latency and LLM query costs.

**Status (2026-09-22): implemented, tested and measured.** How to run it: [README.md](README.md).
The numbers: [eval/results.md](eval/results.md).

## Left for you

- [ ] **Revoke the Hugging Face token** at https://huggingface.co/settings/tokens. It was hardcoded in the old
      `embeddings.py`. That file is gone and nothing in the repo needs a token, but the token itself is still live.
- [ ] Make the first commit. Run `git status` first and check that `venv/`, `data/`, `*.npy` and `chunks.json` aren't listed.
- [ ] Optional: **use the GPU for embedding.** It runs on CPU at about 15 chunks/s, because the NVIDIA driver
      (551.61, CUDA 12.4) is older than current CUDA builds of torch need (12.8+). After a driver update:
      `pip install torch --index-url https://download.pytorch.org/whl/cu128`, then set `RAG_EMBED_DEVICE=cuda`.
- [ ] Optional: build a real 100k+ article index once embedding is on the GPU:
      `python -m rag.ingest --n 100000`, then `python -m rag.index --mode dynamic --out data/index_100k`.
      Everything is batched and SQLite-backed, so it doesn't need to fit in RAM. Scale is currently shown
      with synthetic vectors (`eval/ann_scale.py`).
- [ ] Optional: a hosted LLM backend (a second `chat()` in `rag/generate.py`), so cache savings are in dollars,
      not just tokens.
- [ ] Optional: speed up reranking on CPU (ONNX / int8 via `CrossEncoder(..., backend="onnx")`, or fewer candidates).
      It's the slowest retrieval stage at about 1 s per query.
- [ ] Optional: grade answers with a stronger judge model than the qwen3 that also writes them.

## Done

### 0. Setup
- [x] Hardcoded HF token removed; `main.py`, `embeddings.py` and the old `chunks.json` / `.npy` replaced by the `rag/` package and `data/index_*`
- [x] `.gitignore`, `requirements.txt`, `pytest.ini`
- [x] Ingest re-downloads 5,000 articles and drops 490 syndicated duplicates (`rag/ingest.py`). It's resumable:
      rerunning with a bigger `--n` continues from a checkpoint.
- [x] qwen3 "thinking" turned off, and token counts returned (`rag/generate.py`)

### 1. Dynamic chunking (`rag/chunking.py`)
- [x] Sentence-aware (nltk), sized in model tokens (target 300, cap 450), 1-sentence overlap, short tails merged
- [x] Semantic mode: also breaks where the similarity between neighbouring sentences drops (lowest 20%)
- [x] Title prepended for embedding, BM25 and reranking
- [x] The fixed 400-word baseline is kept for comparison: 25.6% of its chunks were over bge-small's 512-token
      limit and silently truncated. Dynamic and semantic chunks: 0%.

### 2. BM25 (`rag/lexical.py`)
- [x] bm25s with English stopwords and stemming, saved to disk per index

### 3. Hybrid search (`rag/retrieve.py`)
- [x] Reciprocal Rank Fusion, plus weighted min-max fusion for comparison (alpha 0.3 / 0.5 / 0.7)
- [x] At most 2 chunks per article

### 4. Cross-encoder reranking (`rag/rerank.py`)
- [x] `ms-marco-MiniLM-L-6-v2` reranks the top 30 fused candidates down to 5
- [x] Compared with `bge-reranker-base`: same R@1, slightly better R@5, 6.6x slower on CPU (8.5 s per query)

### 5. Redis semantic cache (`rag/cache.py`)
- [x] Exact-match layer, then Redis HNSW vector search over past questions (top 3 candidates)
- [x] Vector similarity alone gave wrong answers for near-miss questions ("eclipse in the Tri-Cities" vs
      "in Seattle": 76% false hits at 0.85). Added a key-term guard and a duplicate-question cross-encoder
      verifier (`quora-distilroberta-base`, scored both ways). Result: 92.7% paraphrase hit rate, 0% false hits.
- [x] TTL, hit/miss/tokens-saved stats, separate namespaces, cleared automatically when the index is rebuilt
- [x] If Redis is down, the pipeline keeps working without the cache

### 6. Evaluation (`eval/`)
- [x] 250 generated questions plus 150 hard ones (no names from the article), written from raw
      article passages, so no chunker is favoured (`make_questions.py`)
- [x] Retrieval R@1/5/10/20, MRR, nDCG@10 for every config on every index (`retrieval.py`)
- [x] HNSW vs exact search (`ann.py`), and the same at 100k-1M vectors (`ann_scale.py`)
- [x] Cache threshold, guard and verifier sweep, plus an end-to-end Redis check (`cache_eval.py`)
- [x] Answer quality with an LLM judge, original vs full pipeline (`quality.py`)
- [x] Per-stage p50/p95 latency: cold, exact hit, semantic hit (`latency.py`)
- [x] `eval/results.md` rendered from the JSON results (`report.py`)

### 7. Scale and serving
- [x] FAISS HNSW index (NumPy exact search kept as an option and as the eval reference)
- [x] Chunks in SQLite; builds run in batches
- [x] FastAPI: `/query`, `/query/stream` (NDJSON token streaming), `/stats`, `/health`, `/cache`
- [x] Structured JSON log line per request with per-stage timings
- [x] Dockerfile (CPU torch, models baked in) and docker-compose (app + Redis, optional Ollama).
      Tested: the image builds and answers queries.

### 8. Wrap-up
- [x] README with architecture diagram, setup, API and results
- [x] 22 pytest tests: chunker edge cases, fusion, cache (incl. real-model near-miss check), API
