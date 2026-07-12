"""Tests for the retrieval core (lexical + hybrid semantic)."""

import pytest

from rag import (
    MIN_SCORE,
    RetrievalIndex,
    chunk_text,
    cosine,
    cosine_dense,
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
    para = "word " * 60
    text = "\n\n".join([para.strip()] * 4)
    assert len(chunk_text(text)) > 1


# -- cosine (sparse + dense) ------------------------------------------------


def test_cosine_empty_vector_is_zero():
    assert cosine({}, {"a": 1.0}) == 0.0


def test_cosine_identical_is_one():
    v = {"a": 1.0, "b": 1.0}
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine({"a": 1.0}, {"b": 1.0}) == 0.0


def test_cosine_zero_norm_guard():
    assert cosine({"a": 0.0}, {"a": 0.0}) == 0.0


def test_cosine_dense_identical_is_one():
    assert cosine_dense([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_dense_orthogonal_is_zero():
    assert cosine_dense([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_dense_empty_is_zero():
    assert cosine_dense([], [1.0]) == 0.0


def test_cosine_dense_zero_norm_guard():
    assert cosine_dense([0.0, 0.0], [0.0, 0.0]) == 0.0


# -- RetrievalIndex: basics -------------------------------------------------


def test_index_starts_empty():
    idx = RetrievalIndex()
    assert idx.is_empty()
    assert idx.chunks == []
    assert idx.documents == []
    assert idx.doc_names == []
    assert idx.has_embeddings() is False


def test_search_on_empty_index_returns_nothing():
    assert RetrievalIndex().search("anything") == []


def test_add_document_returns_chunks_and_tracks_metadata():
    idx = RetrievalIndex()
    chunks = idx.add_document("d1", "policy.txt", [DOC])
    assert len(chunks) == len(idx.chunks) >= 1
    assert idx.doc_names == ["policy.txt"]
    assert idx.documents[0]["name"] == "policy.txt"
    assert idx.documents[0]["chunks"] == len(chunks)
    assert all(c.page is None for c in idx.chunks)


def test_multi_document_accumulates():
    idx = RetrievalIndex()
    idx.add_document("d1", "a.txt", ["Savings account minimum balance rules."])
    idx.add_document("d2", "b.txt", ["Home loan interest and eligibility."])
    assert len(idx.documents) == 2
    assert {d["name"] for d in idx.documents} == {"a.txt", "b.txt"}


def test_multi_page_document_tags_pages():
    idx = RetrievalIndex()
    idx.add_document("d1", "policy.pdf", ["page one text here", "page two text here"])
    assert {c.page for c in idx.chunks} == {1, 2}


def test_remove_document():
    idx = RetrievalIndex()
    idx.add_document("keep", "keep.txt", ["Savings account minimum balance."])
    idx.add_document("drop", "drop.txt", ["Home loan interest rate details."])
    assert idx.remove_document("drop") is True
    assert [d["doc_id"] for d in idx.documents] == ["keep"]
    # Search still works over the remaining doc.
    assert idx.search("minimum balance")


def test_remove_missing_document_is_false():
    idx = RetrievalIndex()
    idx.add_document("d1", "a.txt", [DOC])
    assert idx.remove_document("nope") is False


def test_clear_resets_index():
    idx = RetrievalIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    idx.clear()
    assert idx.is_empty()


# -- lexical search ---------------------------------------------------------


def test_search_returns_relevant_chunk_ranked():
    idx = RetrievalIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    results = idx.search("what is the minimum balance", top_k=3)
    assert results and results[0].rank == 1
    assert "minimum balance" in results[0].chunk.text.lower()


def test_search_empty_query_returns_nothing():
    idx = RetrievalIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    assert idx.search("   ") == []


def test_search_all_stopword_query_returns_nothing():
    idx = RetrievalIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    assert idx.search("the and of is") == []


def test_search_irrelevant_query_below_threshold():
    idx = RetrievalIndex()
    idx.add_document("d1", "policy.txt", [DOC])
    assert idx.search("photosynthesis chlorophyll organelle") == []
    assert MIN_SCORE > 0


# -- hybrid semantic search -------------------------------------------------


def _embed_index():
    """A two-chunk index with orthogonal unit embeddings attached."""
    idx = RetrievalIndex()
    # Two pages -> two separate chunks.
    chunks = idx.add_document(
        "d1", "policy.txt", ["Minimum balance rules.", "Dormant account reactivation."]
    )
    # chunk 0 -> [1,0], chunk 1 -> [0,1]
    idx.set_embeddings({chunks[0].id: [1.0, 0.0], chunks[1].id: [0.0, 1.0]})
    return idx, chunks


def test_has_embeddings_true_when_all_attached():
    idx, _ = _embed_index()
    assert idx.has_embeddings() is True


def test_hybrid_search_uses_embedding_to_rank():
    idx, chunks = _embed_index()
    # Query embedding aligned with chunk 1 pulls it to the top despite lexical.
    results = idx.search("anything", top_k=2, query_embedding=[0.0, 1.0])
    assert results[0].chunk.id == chunks[1].id


def test_hybrid_search_drops_below_floor():
    idx, _ = _embed_index()
    # Orthogonal embedding + no lexical overlap -> combined score under SEM_MIN.
    assert idx.search("zzz qqq nothingmatches", top_k=2, query_embedding=[0.0, 0.0]) == []


def test_query_embedding_ignored_without_full_embeddings():
    idx = RetrievalIndex()
    idx.add_document("d1", "policy.txt", [DOC])  # no embeddings attached
    # Falls back to lexical; still finds the balance chunk.
    results = idx.search("minimum balance", top_k=2, query_embedding=[1.0, 0.0])
    assert results and "balance" in results[0].chunk.text.lower()
