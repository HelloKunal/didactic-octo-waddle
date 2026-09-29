import pytest
from transformers import AutoTokenizer

from rag import config
from rag.chunking import chunk_text, chunk_words


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(config.EMBED_MODEL)


def n_tokens(tok, text):
    return len(tok(text)["input_ids"])


def test_empty_text(tok):
    assert chunk_text("", tok) == []
    assert chunk_text("   \n  ", tok) == []


def test_short_text_is_one_chunk(tok):
    assert chunk_text("One sentence. Another sentence.", tok) == ["One sentence. Another sentence."]


def test_no_chunk_over_model_limit(tok):
    text = " ".join(f"This is sentence number {i} in a long article." for i in range(1000))
    chunks = chunk_text(text, tok)
    assert len(chunks) > 10
    assert max(n_tokens(tok, c) for c in chunks) <= 512


def test_one_huge_sentence_is_split(tok):
    text = " ".join(["word"] * 3000)            # no sentence boundary at all
    chunks = chunk_text(text, tok)
    assert len(chunks) > 1
    assert max(n_tokens(tok, c) for c in chunks) <= 512


def test_chunks_start_with_previous_sentence(tok):
    text = " ".join(f"Sentence {i} talks about topic {i}." for i in range(200))
    chunks = chunk_text(text, tok, target=60, max_tokens=120, overlap=1)
    for prev, cur in zip(chunks, chunks[1:]):
        last_sentence = prev.rsplit(". ", 1)[-1]
        assert cur.startswith(last_sentence)


def test_short_tail_is_merged(tok):
    # target=8 puts every ~8-token sentence in its own chunk; the 3-token tail joins the one before
    text = " ".join(f"Sentence {i} is here to fill space." for i in range(5)) + " Tiny end."
    chunks = chunk_text(text, tok, target=8, max_tokens=100, min_tokens=5, overlap=0)
    assert chunks[-1] == "Sentence 4 is here to fill space. Tiny end."


def test_fixed_chunker():
    assert chunk_words("a b c d e", size=2) == ["a b", "c d", "e"]
