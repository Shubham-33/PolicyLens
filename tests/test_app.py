"""Tests for the Flask routes and middleware."""

import io
import json
from unittest.mock import MagicMock, patch

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


# -- sample + ingest --------------------------------------------------------


def test_load_sample(client):
    data = client.post("/api/sample").get_json()
    assert data["chunks"] > 0
    status = client.get("/api/status").get_json()
    assert status["chunks"] == data["chunks"]


def test_ingest_text(client):
    resp = client.post("/api/ingest", json={"text": "Some policy text here.", "name": "P"})
    data = resp.get_json()
    assert data["name"] == "P" and data["chunks"] >= 1


def test_ingest_text_default_name(client):
    data = client.post("/api/ingest", json={"text": "Policy body without a name."}).get_json()
    assert data["name"] == "Pasted text"


def test_ingest_file(client):
    upload = (io.BytesIO(b"A policy document uploaded as a file."), "doc.txt")
    resp = client.post("/api/ingest", data={"file": upload}, content_type="multipart/form-data")
    assert resp.get_json()["chunks"] >= 1


def test_ingest_unsupported_file(client):
    upload = (io.BytesIO(b"data"), "archive.zip")
    resp = client.post("/api/ingest", data={"file": upload}, content_type="multipart/form-data")
    assert resp.status_code == 400 and "error" in resp.get_json()


def test_ingest_empty_text_is_error(client):
    resp = client.post("/api/ingest", json={"text": "   "})
    assert resp.status_code == 400


# -- ask --------------------------------------------------------------------


def test_ask_without_question(client):
    resp = client.post("/api/ask", json={"question": "  "})
    assert resp.status_code == 400


def test_ask_too_long(client):
    resp = client.post("/api/ask", json={"question": "x" * 501})
    assert resp.status_code == 400


def test_ask_without_document(client):
    resp = client.post("/api/ask", json={"question": "What is the balance?"})
    assert resp.status_code == 400


def test_ask_returns_grounded_answer(client):
    client.post("/api/sample")
    resp = client.post("/api/ask", json={"question": "What is the minimum balance?"})
    data = resp.get_json()
    assert data["found"] is True
    assert data["sources"] and data["sources"][0]["rank"] == 1


def test_ask_uses_nim_when_configured(client, monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-test")
    client.post("/api/sample")
    body = json.dumps({"answer": "Minimum balance is Rs. 10000 [1].", "citations": [1], "found": True})
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
    # healthz is well under the gzip size floor.
    resp = client.get("/healthz", headers={"Accept-Encoding": "gzip"})
    assert resp.headers.get("Content-Encoding") is None


def test_no_gzip_without_accept_encoding(client):
    resp = client.get("/", headers={"Accept-Encoding": "identity"})
    assert resp.headers.get("Content-Encoding") is None
