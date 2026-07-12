"""Tests for grounded answer generation and the offline fallback."""

import json
from unittest.mock import MagicMock, patch

import requests

import nim
from rag import Chunk, Retrieved


def _retrieved(*, page=None):
    """Two retrieved chunks; the first mentions interest, the second balance."""
    return [
        Retrieved(
            chunk=Chunk(0, "d", "policy.txt", 0, "Interest is paid quarterly.", page),
            score=0.5,
            rank=1,
        ),
        Retrieved(
            chunk=Chunk(1, "d", "policy.txt", 1, "Minimum balance is Rs. 10000.", page),
            score=0.3,
            rank=2,
        ),
    ]


def _nim_response(content):
    """A MagicMock shaped like a successful NIM HTTP response."""
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


# -- key handling -----------------------------------------------------------


def test_api_key_prefers_nim_var(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-a")
    monkeypatch.setenv("API_KEY", "other")
    assert nim.api_key() == "nvapi-a"


def test_api_key_falls_back_to_generic(monkeypatch):
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.setenv("API_KEY", "generic")
    assert nim.api_key() == "generic"


def test_is_configured_reflects_key(monkeypatch):
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    assert nim.is_configured() is False


# -- context + result -------------------------------------------------------


def test_build_context_includes_page_when_present():
    ctx = nim.build_context(_retrieved(page=4))
    assert "(p.4)" in ctx and "[1]" in ctx


def test_build_context_omits_page_when_absent():
    assert "(p." not in nim.build_context(_retrieved())


def test_result_to_dict_round_trips():
    result = nim.AnswerResult("hi", [1], True, True, "m")
    data = result.to_dict()
    assert data["model"] == "m" and data["answer"] == "hi" and data["found"] is True


# -- answer_question routing ------------------------------------------------


def test_no_retrieved_chunks_refuses():
    result = nim.answer_question("anything", [])
    assert result.found is False and result.citations == []


def test_uses_fallback_when_no_key(monkeypatch):
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    result = nim.answer_question("interest", _retrieved())
    assert result.grounded is False and result.found is True


def test_uses_nim_when_configured(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    payload = json.dumps({"answer": "Interest is quarterly [1].", "citations": [1], "found": True})
    with patch("nim.requests.post", return_value=_nim_response(payload)) as post:
        result = nim.answer_question("interest?", _retrieved())
    assert post.called
    assert result.grounded is True and result.citations == [1]


def test_falls_back_when_nim_fails(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    with patch("nim.requests.post", side_effect=requests.RequestException):
        result = nim.answer_question("interest?", _retrieved())
    assert result.grounded is False


# -- _nim_answer edge cases -------------------------------------------------


def test_nim_answer_non_ok_returns_none(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    resp = MagicMock()
    resp.ok = False
    with patch("nim.requests.post", return_value=resp):
        assert nim._nim_answer("q", _retrieved()) is None


def test_nim_answer_bad_json_body_returns_none(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    resp = MagicMock()
    resp.ok = True
    resp.json.side_effect = ValueError("not json")
    with patch("nim.requests.post", return_value=resp):
        assert nim._nim_answer("q", _retrieved()) is None


def test_nim_answer_missing_choices_returns_none(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "nvapi-x")
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = {"choices": []}
    with patch("nim.requests.post", return_value=resp):
        assert nim._nim_answer("q", _retrieved()) is None


# -- _parse_answer edge cases -----------------------------------------------


def test_parse_answer_no_json_returns_none():
    assert nim._parse_answer("just prose, no object", _retrieved()) is None


def test_parse_answer_invalid_json_returns_none():
    assert nim._parse_answer("{not valid json}", _retrieved()) is None


def test_parse_answer_empty_answer_returns_none():
    assert nim._parse_answer('{"answer": "", "found": true}', _retrieved()) is None


def test_parse_answer_filters_invalid_citations():
    content = '{"answer": "text [1]", "citations": [1, 9, "x"], "found": true}'
    result = nim._parse_answer(content, _retrieved())
    assert result is not None
    assert result.citations == [1]  # 9 out of range, "x" non-numeric dropped


def test_parse_answer_defaults_found_from_answer():
    # No citations field -> found defaults from the non-empty answer, citations empty.
    result = nim._parse_answer('{"answer": "has content"}', _retrieved())
    assert result is not None and result.found is True and result.citations == []


# -- extractive fallback ----------------------------------------------------


def test_extractive_picks_overlapping_sentence():
    result = nim._extractive_answer("tell me about interest", _retrieved())
    assert "Interest" in result.answer and result.citations == [1]


def test_extractive_no_overlap_uses_lead(monkeypatch):
    result = nim._extractive_answer("zzz qqq", _retrieved())
    # No lexical overlap -> falls back to the top passage's lead sentence.
    assert result.found is True and result.citations == [1]
