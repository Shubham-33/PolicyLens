"""Tests for the pure retrieval core."""

import pytest

from rag import (
    MIN_SCORE,
    TfidfIndex,
    chunk_text,
    cosine,
    tokenize,
)

DOC = (
    "Minimum balance is Rs. 10,000 for metro accounts.\n\n"
    "Interest is paid quarterly at 2.70 percent per annum.\n\n"
    "The first cheque book of 20 leaves is free every year."
)


# -- tokenize ---------------------------------------------------------------


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Balance IS 500") == ["balance", "500"]


def test_tokenize_empty():
    assert tokenize("the and of") == []


# -- chunk_text -------------------------------------------------------------


def test_chunk_text_empty_returns_nothing():
    assert chunk_text("   ") == []


def test_chunk_text_single_paragraph():
    assert chunk_text("one short para") == ["one short para"]


def test_chunk_text_packs_and_overlaps():
    para = "word " * 60  # ~300 chars
    text = "\n\n".join([para.strip()] * 4)  # exceeds target -> splits
    chunks = chunk_text(text)
    assert len(chunks) > 1


# -- cosine -----------------------------------------------------------------


def test_cosine_empty_vector_is_zero():
    assert cosine({}, {"a": 1.0}) == 0.0


def test_cosine_identical_is_one():
    v = {"a": 1.0, "b": 1.0}
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_zero_norm_guard():
    # A non-empty but zero-weight vector exercises the norm==0 guard.
    assert cosine({"a": 0.0}, {"a": 0.0}) == 0.0


# -- TfidfIndex -------------------------------------------------------------


def test_index_starts_empty():
    idx = TfidfIndex()
    assert idx.is_empty()
    assert idx.chunks == []
    assert idx.doc_names == []


def test_search_on_empty_index_returns_nothing():
    assert TfidfIndex().search("anything") == []


def test_add_single_page_document_has_no_page_numbers():
    idx = TfidfIndex()
    added = idx.add_document("d1", "policy.txt", [DOC])
    assert added == len(idx.chunks) >= 1
    assert idx.doc_names == ["policy.txt"]
    assert all(chunk.page is None for chunk in idx.chunks)


def test_add_multi_page_document_tags_pages():
    idx = TfidfIndex()
    idx.add_document("d1", "policy.pdf", ["page one text here", "page two text here"])
    assert {chunk.page for chunk in idx.chunks} == {1, 2}


def test_search_returns_relevant_chunk_ranked():
    idx = TfidfIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    results = idx.search("what is the minimum balance", top_k=3)
    assert results
    assert results[0].rank == 1
    assert "minimum balance" in results[0].chunk.text.lower()


def test_search_empty_query_returns_nothing():
    idx = TfidfIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    assert idx.search("   ") == []


def test_search_all_stopword_query_returns_nothing():
    # Tokenises to nothing -> empty vector -> no matches.
    idx = TfidfIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    assert idx.search("the and of is") == []


def test_search_irrelevant_query_below_threshold():
    idx = TfidfIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    results = idx.search("photosynthesis chlorophyll organelle")
    assert results == []
    # Sanity: the threshold is what filters it.
    assert MIN_SCORE > 0


def test_clear_resets_index():
    idx = TfidfIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    idx.clear()
    assert idx.is_empty()
