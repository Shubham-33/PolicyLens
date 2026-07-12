"""Semantic embeddings via NVIDIA NIM, with a graceful no-op fallback.

Adds semantic recall on top of lexical TF-IDF: passages are embedded at ingest,
the question at query time, and the two are compared by cosine. This is what
lets "what if my account is inactive?" find a passage that says "dormant".

Every function returns ``None`` on any failure (no key, throttle, network, bad
response) so retrieval silently falls back to pure TF-IDF and the app never
breaks — the same resilience principle as the chat layer.
"""

from __future__ import annotations

import os
from typing import Final

import requests

from nim import api_key

EMBED_URL: Final[str] = "https://integrate.api.nvidia.com/v1/embeddings"
EMBED_MODEL: Final[str] = os.environ.get("EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
EMBED_TIMEOUT_S: Final[int] = 20
MAX_BATCH: Final[int] = 64


def is_configured() -> bool:
    """True when an API key is available for embedding calls."""
    return bool(api_key())


def embed_passages(texts: list[str]) -> list[list[float]] | None:
    """Embed document passages for indexing. Returns None on any failure."""
    return _embed(texts, "passage")


def embed_query(text: str) -> list[float] | None:
    """Embed a single question. Returns None on any failure."""
    vectors = _embed([text], "query")
    return vectors[0] if vectors else None


def _embed(texts: list[str], input_type: str) -> list[list[float]] | None:
    """Call the NVIDIA embeddings endpoint, batching to stay within limits."""
    if not texts or not is_configured():
        return None
    headers = {"Authorization": f"Bearer {api_key()}", "Accept": "application/json"}
    vectors: list[list[float]] = []
    for start in range(0, len(texts), MAX_BATCH):
        batch = texts[start : start + MAX_BATCH]
        payload = {
            "input": batch,
            "model": EMBED_MODEL,
            "input_type": input_type,
            "encoding_format": "float",
        }
        try:
            resp = requests.post(
                EMBED_URL, json=payload, headers=headers, timeout=EMBED_TIMEOUT_S
            )
        except requests.RequestException:
            return None
        if not resp.ok:
            return None
        try:
            vectors.extend(item["embedding"] for item in resp.json()["data"])
        except (ValueError, KeyError, TypeError):
            return None
    if len(vectors) != len(texts):
        return None
    return vectors
