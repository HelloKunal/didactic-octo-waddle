import nltk
import numpy as np

from rag import config

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize  # noqa: E402


def chunk_words(text, size=400):
    """The original fixed-size chunker, kept as the eval baseline."""
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def split_sentences(text):
    return [s.strip() for para in text.split("\n") for s in sent_tokenize(para) if s.strip()]


def count_tokens(tokenizer, texts):
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]


def chunk_text(
    text,
    tokenizer,
    target=config.CHUNK_TARGET,
    max_tokens=config.CHUNK_MAX,
    min_tokens=config.CHUNK_MIN,
    overlap=config.CHUNK_OVERLAP,
    embed=None,
):
    """Sentence-aware chunks of about `target` tokens, never more than `max_tokens`.

    Short documents stay whole, long ones are split. With `embed` (a SentenceTransformer), a new
    chunk also starts wherever the topic shifts between neighbouring sentences.
    """
    sents = split_sentences(text)
    if not sents:
        return []

    # a single sentence longer than max_tokens gets split on words
    pieces = []
    for s, n in zip(sents, count_tokens(tokenizer, sents)):
        if n <= max_tokens:
            pieces.append((s, n))
            continue
        words = s.split()
        step = max(1, int(len(words) * target / n))
        parts = [" ".join(words[i:i + step]) for i in range(0, len(words), step)]
        pieces += list(zip(parts, count_tokens(tokenizer, parts)))

    breaks = semantic_breaks([s for s, _ in pieces], embed) if embed else set()

    # pack sentences into groups of about `target` tokens
    groups, cur, cur_len = [], [], 0
    for i, (s, n) in enumerate(pieces):
        if cur and (cur_len + n > target or (i in breaks and cur_len >= min_tokens)):
            groups.append(cur)
            cur, cur_len = [], 0
        cur.append((s, n))
        cur_len += n
    groups.append(cur)

    size = lambda g: sum(n for _, n in g)  # noqa: E731
    if len(groups) > 1 and size(groups[-1]) < min_tokens and size(groups[-2]) + size(groups[-1]) <= max_tokens:
        tail = groups.pop()
        groups[-1] += tail

    # overlap: start each chunk with the last sentence(s) of the previous one
    chunks = []
    for i, g in enumerate(groups):
        prev = groups[i - 1][-overlap:] if i and overlap else []
        if size(prev) + size(g) > max_tokens:
            prev = []
        chunks.append(" ".join(s for s, _ in prev + g))
    return chunks


def semantic_breaks(sentences, model, percentile=20):
    """Sentence indexes where the topic shifts: similarity to the previous sentence is in the lowest 20%."""
    if len(sentences) < 3:
        return set()
    emb = model.encode(sentences, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    sims = (emb[:-1] * emb[1:]).sum(axis=1)
    cutoff = np.percentile(sims, percentile)
    return {i + 1 for i, s in enumerate(sims) if s < cutoff}
