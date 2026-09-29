import random
import sqlite3
import threading
from functools import lru_cache

from rag import config

COLUMNS = ("chunk_id", "article_id", "title", "url", "text")


class ChunkStore:
    """Chunk text and metadata in SQLite, so the corpus doesn't have to fit in memory."""

    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            "chunk_id INTEGER PRIMARY KEY, article_id INTEGER, title TEXT, url TEXT, text TEXT)"
        )

    def add(self, rows):
        with self.lock:
            self.db.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)", rows)
            self.db.commit()

    def get(self, ids):
        """Chunks as dicts, in the same order as `ids`."""
        ids = [int(i) for i in ids]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        with self.lock:
            rows = self.db.execute(
                f"SELECT {', '.join(COLUMNS)} FROM chunks WHERE chunk_id IN ({placeholders})", ids
            ).fetchall()
        by_id = {r[0]: dict(zip(COLUMNS, r)) for r in rows}
        return [by_id[i] for i in ids]

    def __getitem__(self, chunk_id):
        return self.get([chunk_id])[0]

    def __len__(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def iter_all(self, batch=10_000):
        with self.lock:
            cur = self.db.execute(f"SELECT {', '.join(COLUMNS)} FROM chunks ORDER BY chunk_id")
            while rows := cur.fetchmany(batch):
                for r in rows:
                    yield dict(zip(COLUMNS, r))

    def sample(self, n, seed=0, min_words=0):
        with self.lock:
            rows = self.db.execute("SELECT chunk_id, text FROM chunks").fetchall()
        ids = [cid for cid, text in rows if len(text.split()) >= min_words]
        return self.get(random.Random(seed).sample(ids, min(n, len(ids))))

    def close(self):
        self.db.close()


@lru_cache
def _store(index_dir):
    return ChunkStore(f"{index_dir}/chunks.db")


def store():
    return _store(config.INDEX_DIR)
