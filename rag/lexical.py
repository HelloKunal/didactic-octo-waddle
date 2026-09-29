from functools import lru_cache

import bm25s
import Stemmer  # from the PyStemmer package

from rag import config

stemmer = Stemmer.Stemmer("english")


def build_bm25(texts, path):
    tokens = bm25s.tokenize(texts, stopwords="en", stemmer=stemmer, show_progress=False)
    bm25 = bm25s.BM25()   # k1=1.5, b=0.75
    bm25.index(tokens, show_progress=False)
    bm25.save(path)


class Lexical:
    def __init__(self, path):
        self.bm25 = bm25s.BM25.load(path)

    def search(self, query, k=config.CANDIDATES):
        q = bm25s.tokenize([query], stopwords="en", stemmer=stemmer, return_ids=False, show_progress=False)
        if not q[0]:                     # query was all stopwords
            return []
        ids, scores = self.bm25.retrieve(q, k=k, show_progress=False)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if s > 0]


@lru_cache
def _lexical(index_dir):
    return Lexical(f"{index_dir}/bm25")


def lexical():
    return _lexical(config.INDEX_DIR)
