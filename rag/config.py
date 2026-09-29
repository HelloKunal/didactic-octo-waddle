import os

DATA_DIR = os.getenv("RAG_DATA_DIR", "data")
ARTICLES_PATH = os.getenv("RAG_ARTICLES", f"{DATA_DIR}/articles.jsonl")
INDEX_DIR = os.getenv("RAG_INDEX_DIR", f"{DATA_DIR}/index_dynamic")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
EMBED_DEVICE = os.getenv("RAG_EMBED_DEVICE", "cpu")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:latest")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# chunking, in embedding-model tokens (bge-small reads at most 512)
CHUNK_TARGET = 300
CHUNK_MAX = 450
CHUNK_MIN = 50
CHUNK_OVERLAP = 1          # sentences repeated from the previous chunk

# retrieval
DENSE_BACKEND = os.getenv("RAG_DENSE_BACKEND", "faiss")   # "faiss" (HNSW) or "numpy" (exact)
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
CANDIDATES = 50            # taken from each retriever
RERANK_INPUT = 30          # fused candidates sent to the reranker
FINAL_K = 5                # passages that go into the prompt
MAX_PER_ARTICLE = 2

# semantic cache (thresholds tuned by eval/cache_eval.py)
CACHE_NAMESPACE = os.getenv("RAG_CACHE_NS", "qcache")
CACHE_THRESHOLD = float(os.getenv("RAG_CACHE_THRESHOLD", "0.85"))   # min cosine similarity for a candidate
CACHE_KEY_TERM_GUARD = True
CACHE_VERIFIER = os.getenv("RAG_CACHE_VERIFIER", "cross-encoder/quora-distilroberta-base") or None
CACHE_VERIFIER_MIN = 0.5
CACHE_TTL = 24 * 3600
