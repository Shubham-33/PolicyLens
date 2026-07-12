"""Retrieval core for PolicyLens — pure Python, zero ML dependencies.

Documents are split into overlapping chunks, ranked against a question with
TF-IDF cosine similarity, and returned with the citation metadata the UI needs
to point back at the exact source passage (NotebookLM-style grounding).

Kept dependency-free on purpose: it runs anywhere, installs instantly, and the
whole thing is deterministic — which makes it trivially testable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNK_TARGET_CHARS: Final[int] = 700
CHUNK_OVERLAP_CHARS: Final[int] = 120
DEFAULT_TOP_K: Final[int] = 4
MIN_SCORE: Final[float] = 0.03  # below this a chunk is treated as irrelevant

_TOKEN_RE: Final = re.compile(r"[a-z0-9]+")
_PARA_RE: Final = re.compile(r"\n\s*\n")

# Dropping these keeps short questions from matching on filler words alone
# (e.g. "what is the weather on Mars" should not hit a chunk just for "on").
_STOPWORDS: Final[frozenset[str]] = frozenset(
    """
    a an and are as at be but by can do does for from had has have how i if in
    into is it its of on or that the their there these this to was were what
    when where which who why will with you your
    """.split()
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """A retrievable slice of a document, carrying its own source coordinates."""

    id: int
    doc_id: str
    doc_name: str
    ordinal: int  # 0-based position of this chunk within its document
    text: str
    page: int | None = None


@dataclass
class Retrieved:
    """A chunk selected for a query, with its similarity score and citation rank."""

    chunk: Chunk
    score: float
    rank: int  # 1-based index used as the [n] citation marker in the answer


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens minus stopwords; used everywhere."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def chunk_text(text: str) -> list[str]:
    """Split text into ~``CHUNK_TARGET_CHARS`` pieces on paragraph boundaries.

    Paragraphs are the natural unit for policy documents. Adjacent paragraphs
    are packed together until the target size is reached, with a small character
    overlap so a fact that straddles a boundary is not lost.
    """
    paragraphs = [p.strip() for p in _PARA_RE.split(text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if buffer and len(buffer) + len(para) + 2 > CHUNK_TARGET_CHARS:
            chunks.append(buffer)
            tail = buffer[-CHUNK_OVERLAP_CHARS:] if CHUNK_OVERLAP_CHARS else ""
            buffer = f"{tail}\n{para}" if tail else para
        else:
            buffer = f"{buffer}\n\n{para}" if buffer else para
    chunks.append(buffer)  # paragraphs is non-empty, so buffer always has content
    return chunks


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity of two sparse vectors held as term -> weight maps."""
    if not a or not b:
        return 0.0
    # Iterate the smaller vector for the dot product.
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    dot = sum(weight * large.get(term, 0.0) for term, weight in small.items())
    norm_a = math.sqrt(sum(w * w for w in a.values()))
    norm_b = math.sqrt(sum(w * w for w in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class TfidfIndex:
    """An in-memory TF-IDF index over a growing set of document chunks."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._doc_freq: dict[str, int] = {}
        self._next_id: int = 0

    @property
    def chunks(self) -> list[Chunk]:
        """All chunks currently indexed, in insertion order."""
        return list(self._chunks)

    @property
    def doc_names(self) -> list[str]:
        """Distinct source document names, in first-seen order."""
        seen: dict[str, None] = {}
        for chunk in self._chunks:
            seen.setdefault(chunk.doc_name, None)
        return list(seen)

    def is_empty(self) -> bool:
        """True when nothing has been ingested yet."""
        return not self._chunks

    def add_document(self, doc_id: str, doc_name: str, pages: list[str]) -> int:
        """Chunk and index a document supplied as a list of page texts.

        Pass a single-element list for formats without pages (txt/markdown).
        Returns the number of chunks added. Recomputes IDF over the full corpus.
        """
        added = 0
        for page_no, page_text in enumerate(pages, start=1):
            has_pages = len(pages) > 1
            for piece in chunk_text(page_text):
                chunk = Chunk(
                    id=self._next_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    ordinal=added,
                    text=piece,
                    page=page_no if has_pages else None,
                )
                self._chunks.append(chunk)
                for term in set(tokenize(piece)):
                    self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
                self._next_id += 1
                added += 1
        self._recompute()
        return added

    def clear(self) -> None:
        """Drop every document and reset the index to empty."""
        self.__init__()

    def _recompute(self) -> None:
        """Rebuild IDF weights and per-chunk vectors after the corpus changes."""
        n = len(self._chunks)
        self._idf = {
            term: math.log((n + 1) / (df + 1)) + 1.0
            for term, df in self._doc_freq.items()
        }
        self._vectors = [self._vectorize(chunk.text) for chunk in self._chunks]

    def _vectorize(self, text: str) -> dict[str, float]:
        """Turn text into an L2-normalised TF-IDF vector using the current IDF."""
        counts: dict[str, int] = {}
        for term in tokenize(text):
            counts[term] = counts.get(term, 0) + 1
        if not counts:
            return {}
        total = sum(counts.values())
        vector = {
            term: (count / total) * self._idf.get(term, 1.0)
            for term, count in counts.items()
        }
        # Non-empty counts and IDF >= 1 guarantee positive weights, so the norm
        # is always > 0 here — no zero-division guard needed.
        norm = math.sqrt(sum(w * w for w in vector.values()))
        return {term: w / norm for term, w in vector.items()}

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[Retrieved]:
        """Return the top-``top_k`` chunks for ``query``, ranked by similarity.

        Chunks scoring below ``MIN_SCORE`` are dropped, so an off-topic question
        yields an empty list — the signal the answer layer uses to refuse.
        """
        if self.is_empty() or not query.strip():
            return []
        query_vec = self._vectorize(query)
        scored = (
            (chunk, cosine(query_vec, vec))
            for chunk, vec in zip(self._chunks, self._vectors, strict=False)
        )
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        results: list[Retrieved] = []
        for chunk, score in ranked[:top_k]:
            if score < MIN_SCORE:
                break
            results.append(Retrieved(chunk=chunk, score=score, rank=len(results) + 1))
        return results
