"""Tests for the Flask routes, per-session state, and middleware."""

import io
import json
from unittest.mock import MagicMock, patch

import app as app_module

# -- static-ish routes ------------------------------------------------------


def test_home_ok_and_cacheable(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "PolicyLens" in resp.get_data(as_text=True)
    assert "max-age=300" in resp.headers["Cache-Control"]


def test_favicon(client):
    resp = client.get("/favicon.ico")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "image/svg+xml"


def test_healthz(client):
    data = client.get("/healthz").get_json()
    assert data["status"] == "ok" and "llm" in data


def test_status_empty(client):
    data = client.get("/api/status").get_json()
    assert data["documents"] == [] and data["chunks"] == 0
    assert data["semantic"] is False


# -- sample + ingest --------------------------------------------------------


def test_load_sample_and_idempotent(client):
    data = client.post("/api/sample").get_json()
    assert len(data["documents"]) == 1 and data["chunks"] > 0
    # A second load does not duplicate the sample.
    again = client.post("/api/sample").get_json()
    assert len(again["documents"]) == 1


def test_ingest_text_accumulates(client):
    client.post("/api/ingest", json={"text": "Savings policy text.", "name": "A"})
    data = client.post(
        "/api/ingest", json={"text": "Loan policy text.", "name": "B"}
    ).get_json()
    assert {d["name"] for d in data["documents"]} == {"A", "B"}


def test_ingest_text_default_name(client):
    data = client.post("/api/ingest", json={"text": "Body without a name."}).get_json()
    assert data["documents"][0]["name"] == "Pasted text"


def test_ingest_file(client):
    upload = (io.BytesIO(b"A policy document uploaded as a file."), "doc.txt")
    resp = client.post("/api/ingest", data={"file": upload}, content_type="multipart/form-data")
    assert resp.get_json()["documents"][0]["name"] == "doc.txt"


def test_ingest_unsupported_file(client):
    upload = (io.BytesIO(b"data"), "archive.zip")
    resp = client.post("/api/ingest", data={"file": upload}, content_type="multipart/form-data")
    assert resp.status_code == 400 and "error" in resp.get_json()


def test_ingest_empty_text_is_error(client):
    assert client.post("/api/ingest", json={"text": "   "}).status_code == 400


# -- remove -----------------------------------------------------------------


def test_remove_document(client):
    client.post("/api/ingest", json={"text": "Doc one.", "name": "One"})
    doc_id = client.get("/api/status").get_json()["documents"][0]["doc_id"]
    resp = client.post("/api/remove", json={"doc_id": doc_id})
    assert resp.status_code == 200 and resp.get_json()["documents"] == []


def test_remove_missing_document_404(client):
    assert client.post("/api/remove", json={"doc_id": "nope"}).status_code == 404


# -- ask --------------------------------------------------------------------


def test_ask_without_question(client):
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400


def test_ask_too_long(client):
    assert client.post("/api/ask", json={"question": "x" * 501}).status_code == 400


def test_ask_without_document(client):
    assert client.post("/api/ask", json={"question": "What is the balance?"}).status_code == 400


def test_ask_returns_grounded_answer(client):
    client.post("/api/sample")
    data = client.post("/api/ask", json={"question": "What is the minimum balance?"}).get_json()
    assert data["found"] is True
    assert data["sources"] and data["sources"][0]["rank"] == 1
    assert data["semantic"] is False  # no embeddings in the default test env


def test_ask_uses_nim_when_configured(client, monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-test")
    client.post("/api/sample")
    body = json.dumps({"answer": "Rs. 10000 [1].", "citations": [1], "found": True})
    resp_mock = MagicMock()
    resp_mock.ok = True
    resp_mock.json.return_value = {"choices": [{"message": {"content": body}}]}
    with patch("nim.requests.post", return_value=resp_mock):
        data = client.post("/api/ask", json={"question": "minimum balance?"}).get_json()
    assert data["grounded"] is True and data["citations"] == [1]


def test_reset_clears_corpus(client):
    client.post("/api/sample")
    assert client.post("/api/reset").get_json()["ok"] is True
    assert client.get("/api/status").get_json()["chunks"] == 0


# -- semantic path ----------------------------------------------------------


def test_semantic_ingest_and_ask(client, monkeypatch):
    # Embeddings available: ingest attaches them and ask uses the hybrid path.
    monkeypatch.setattr(app_module.embed, "embed_passages", lambda texts: [[1.0, 0.0]] * len(texts))
    monkeypatch.setattr(app_module.embed, "embed_query", lambda text: [1.0, 0.0])
    ingest = client.post("/api/ingest", json={"text": "Minimum balance policy.", "name": "P"}).get_json()
    assert ingest["semantic"] is True
    data = client.post("/api/ask", json={"question": "balance?"}).get_json()
    assert data["semantic"] is True


# -- session isolation (R1 fix) --------------------------------------------


def test_sessions_are_isolated(app):
    a, b = app.test_client(), app.test_client()
    a.post("/api/sample")
    assert len(a.get("/api/status").get_json()["documents"]) == 1
    # A different client (session) sees an empty corpus.
    assert b.get("/api/status").get_json()["documents"] == []


def test_corpus_persists_across_requests(client):
    # Load once, then several asks on the same session all still see the doc.
    client.post("/api/sample")
    for _ in range(3):
        resp = client.post("/api/ask", json={"question": "minimum balance?"})
        assert resp.status_code == 200 and resp.get_json()["found"] is True


# -- middleware -------------------------------------------------------------


def test_security_headers_present(client):
    headers = client.get("/").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"


def test_large_response_is_gzipped(client):
    resp = client.get("/", headers={"Accept-Encoding": "gzip"})
    assert resp.headers.get("Content-Encoding") == "gzip"
    assert resp.headers["Vary"] == "Accept-Encoding"


def test_small_response_not_gzipped(client):
    resp = client.get("/healthz", headers={"Accept-Encoding": "gzip"})
    assert resp.headers.get("Content-Encoding") is None


def test_no_gzip_without_accept_encoding(client):
    resp = client.get("/", headers={"Accept-Encoding": "identity"})
    assert resp.headers.get("Content-Encoding") is None
