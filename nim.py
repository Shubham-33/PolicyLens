"""Grounded answer generation over retrieved chunks.

Primary path: NVIDIA NIM's OpenAI-compatible chat endpoint, prompted to answer
*only* from the supplied context and to return structured JSON with citations.

Fallback path: a deterministic extractive answer built from the retrieved
chunks. It runs whenever no API key is set or the call fails — so a cold start,
a flaky network, or a CI run with no secret never breaks the app.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Final

import requests

from rag import Retrieved, tokenize

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NIM_URL: Final[str] = "https://integrate.api.nvidia.com/v1/chat/completions"
# 8B is fast and reliable on NVIDIA's free tier; since answers are grounded in
# retrieved context, it is more than adequate here. Override via NIM_MODEL.
NIM_MODEL: Final[str] = os.environ.get("NIM_MODEL", "meta/llama-3.1-8b-instruct")
REQUEST_TIMEOUT_S: Final[int] = 30
MAX_ANSWER_SENTENCES: Final[int] = 3

_JSON_BLOCK_RE: Final = re.compile(r"\{.*\}", re.DOTALL)
_SENT_SPLIT_RE: Final = re.compile(r"(?<=[.!?])\s+")

_SYSTEM_PROMPT: Final[str] = (
    "You are PolicyLens, a precise assistant for banking policy documents. "
    "Answer ONLY using the numbered context passages provided. "
    "Cite every claim with its passage number in square brackets, e.g. [1]. "
    "If the answer is not contained in the context, set found to false and do "
    "not guess. Never use outside knowledge. Respond ONLY with a JSON object of "
    'the form {"answer": string, "citations": number[], "found": boolean}.'
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AnswerResult:
    """The generated answer plus provenance the UI renders."""

    answer: str
    citations: list[int]
    found: bool
    grounded: bool  # True when the LLM produced it, False for the fallback
    model: str = field(default="")

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the JSON API response."""
        return {
            "answer": self.answer,
            "citations": self.citations,
            "found": self.found,
            "grounded": self.grounded,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def api_key() -> str:
    """Return the configured NVIDIA key (env ``NIM_API_KEY`` or ``API_KEY``)."""
    return os.environ.get("NIM_API_KEY") or os.environ.get("API_KEY") or ""


def is_configured() -> bool:
    """True when an NVIDIA NIM key is available for live generation."""
    return bool(api_key())


def build_context(retrieved: list[Retrieved]) -> str:
    """Render retrieved chunks as a numbered context block for the prompt."""
    lines: list[str] = []
    for item in retrieved:
        loc = f" (p.{item.chunk.page})" if item.chunk.page else ""
        lines.append(f"[{item.rank}] {item.chunk.doc_name}{loc}: {item.chunk.text}")
    return "\n\n".join(lines)


def answer_question(question: str, retrieved: list[Retrieved]) -> AnswerResult:
    """Produce a grounded answer, using NIM when possible, else the fallback."""
    if not retrieved:
        return AnswerResult(
            answer="I couldn't find this in the provided documents.",
            citations=[],
            found=False,
            grounded=False,
        )
    if is_configured():
        result = _nim_answer(question, retrieved)
        if result is not None:
            return result
    return _extractive_answer(question, retrieved)


# ---------------------------------------------------------------------------
# NIM path
# ---------------------------------------------------------------------------


def _nim_answer(question: str, retrieved: list[Retrieved]) -> AnswerResult | None:
    """Call NIM and parse a structured answer; return None on any failure."""
    payload = {
        "model": NIM_MODEL,
        "temperature": 0.2,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context passages:\n{build_context(retrieved)}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Accept": "application/json",
    }
    try:
        resp = requests.post(
            NIM_URL, json=payload, headers=headers, timeout=REQUEST_TIMEOUT_S
        )
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return None
    return _parse_answer(content, retrieved)


def _parse_answer(content: str, retrieved: list[Retrieved]) -> AnswerResult | None:
    """Extract the JSON answer object from the model's raw text output."""
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    answer = str(data.get("answer", "")).strip()
    found = bool(data.get("found", bool(answer)))
    if not answer:
        return None
    valid_ranks = {item.rank for item in retrieved}
    citations = [
        int(c)
        for c in data.get("citations", [])
        if isinstance(c, int | float) and int(c) in valid_ranks
    ]
    return AnswerResult(
        answer=_ensure_inline_markers(answer, citations),
        citations=citations,
        found=found,
        grounded=True,
        model=NIM_MODEL,
    )


def _ensure_inline_markers(answer: str, citations: list[int]) -> str:
    """Guarantee cited passages appear as clickable [n] markers in the answer.

    Smaller models sometimes return citations in the JSON array but forget to
    embed the markers in the prose. Without an inline [n] the UI has nothing to
    make clickable, so we append any missing markers to the end of the answer.
    """
    if not citations or any(f"[{c}]" in answer for c in citations):
        return answer
    markers = "".join(f"[{c}]" for c in citations)
    return f"{answer.rstrip()} {markers}"


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------


def _extractive_answer(question: str, retrieved: list[Retrieved]) -> AnswerResult:
    """Quote the passage sentences most relevant to the question, with a citation.

    Used with no API key or on a failed call. It is honest and always grounded:
    it surfaces the source text rather than inventing prose. Sentences are ranked
    by token overlap with the question so the snippet actually answers it.
    """
    query_terms = set(tokenize(question))
    best_rank = 1
    best_sentence = ""
    best_overlap = -1
    for item in retrieved:
        for sentence in _SENT_SPLIT_RE.split(item.chunk.text.strip()):
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap > best_overlap and sentence.strip():
                best_overlap, best_sentence, best_rank = overlap, sentence, item.rank

    if best_overlap <= 0:
        # No lexical overlap in the sentences — fall back to the top passage lead.
        top = retrieved[0]
        lead = _SENT_SPLIT_RE.split(top.chunk.text.strip())[:MAX_ANSWER_SENTENCES]
        best_sentence, best_rank = " ".join(lead), top.rank
    return AnswerResult(
        answer=f"{best_sentence.strip()} [{best_rank}]",
        citations=[best_rank],
        found=True,
        grounded=False,
        model="",
    )
