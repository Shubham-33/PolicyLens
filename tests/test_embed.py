"""Tests for the NVIDIA embeddings layer and its fallback."""

from unittest.mock import MagicMock, patch

import requests

import embed


def _ok(vectors):
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = {"data": [{"embedding": v} for v in vectors]}
    return resp


def test_not_configured_returns_none(monkeypatch):
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    assert embed.is_configured() is False
    assert embed.embed_passages(["hi"]) is None
    assert embed.embed_query("hi") is None


def test_empty_input_returns_none(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    assert embed.embed_passages([]) is None


def test_embed_passages_success(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    with patch("embed.requests.post", return_value=_ok([[0.1, 0.2], [0.3, 0.4]])):
        vectors = embed.embed_passages(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_query_success(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    with patch("embed.requests.post", return_value=_ok([[0.5, 0.6]])):
        assert embed.embed_query("q") == [0.5, 0.6]


def test_request_exception_returns_none(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    with patch("embed.requests.post", side_effect=requests.RequestException):
        assert embed.embed_passages(["a"]) is None


def test_non_ok_returns_none(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    resp = MagicMock()
    resp.ok = False
    with patch("embed.requests.post", return_value=resp):
        assert embed.embed_passages(["a"]) is None


def test_bad_json_returns_none(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    resp = MagicMock()
    resp.ok = True
    resp.json.side_effect = ValueError("boom")
    with patch("embed.requests.post", return_value=resp):
        assert embed.embed_passages(["a"]) is None


def test_count_mismatch_returns_none(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    # Asked for two vectors, got one back.
    with patch("embed.requests.post", return_value=_ok([[0.1, 0.2]])):
        assert embed.embed_passages(["a", "b"]) is None


def test_batching_spans_multiple_requests(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    monkeypatch.setattr(embed, "MAX_BATCH", 1)
    with patch(
        "embed.requests.post", side_effect=[_ok([[1.0]]), _ok([[2.0]])]
    ) as post:
        vectors = embed.embed_passages(["a", "b"])
    assert vectors == [[1.0], [2.0]] and post.call_count == 2
