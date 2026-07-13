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
MIN_SCORE: Final[float] = 0.03  # lexical floor: below this a chunk is irrelevant
SEM_WEIGHT: Final[float] = 0.7  # weight of semantic vs lexical in hybrid ranking
SEM_MIN: Final[float] = 0.08  # hybrid floor when embeddings are in play

_TOKEN_RE: Final = re.compile(r"[a-z0-9]+")
_PARA_RE: Final = re.compile(r"\n\s*\n")
_SENTENCE_RE: Final = re.compile(r"(?<=[.!?])\s+")

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


def best_sentence(query: str, text: str) -> str:
    """Return the sentence in ``text`` most relevant to ``query``.

    Used to show *which part* of a source passage an answer draws on: the
    sentence sharing the most significant terms with the question. Returns an
    empty string when nothing overlaps, so the UI highlights nothing rather than
    guessing.
    """
    terms = set(tokenize(query))
    if not terms:
        return ""
    best, best_overlap = "", 0
    for sentence in _SENTENCE_RE.split(text.strip()):
        overlap = len(terms & set(tokenize(sentence)))
        if overlap > best_overlap:
            best, best_overlap = sentence.strip(), overlap
    return best


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


def cosine_dense(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two dense embedding vectors."""
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class RetrievalIndex:
    """In-memory retrieval over many documents: TF-IDF by default, hybrid with
    semantic embeddings when they are supplied.

    Lexical TF-IDF is always built (instant, offline, deterministic). If per-chunk
    embeddings are attached via :meth:`set_embeddings`, :meth:`search` blends
    semantic and lexical similarity — recovering paraphrases that keyword search
    alone would miss.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[dict[str, float]] = []
        self._embeddings: dict[int, list[float]] = {}
        self._idf: dict[str, float] = {}
        self._doc_freq: dict[str, int] = {}
        self._next_id: int = 0

    @property
    def chunks(self) -> list[Chunk]:
        """All chunks currently indexed, in insertion order."""
        return list(self._chunks)

    @property
    def documents(self) -> list[dict[str, object]]:
        """One entry per source document: id, name, and chunk count."""
        order: list[str] = []
        by_id: dict[str, dict[str, object]] = {}
        for chunk in self._chunks:
            if chunk.doc_id not in by_id:
                by_id[chunk.doc_id] = {
                    "doc_id": chunk.doc_id,
                    "name": chunk.doc_name,
                    "chunks": 0,
                }
                order.append(chunk.doc_id)
            by_id[chunk.doc_id]["chunks"] = int(by_id[chunk.doc_id]["chunks"]) + 1
        return [by_id[d] for d in order]

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

    def has_embeddings(self) -> bool:
        """True when every chunk has an attached semantic embedding."""
        return bool(self._chunks) and len(self._embeddings) == len(self._chunks)

    def add_document(self, doc_id: str, doc_name: str, pages: list[str]) -> list[Chunk]:
        """Chunk and index a document supplied as a list of page texts.

        Documents accumulate — call this repeatedly to search across several at
        once. Pass a single-element list for formats without pages (txt/markdown).
        Returns the chunks added (so the caller can embed them). Recomputes IDF.
        """
        added: list[Chunk] = []
        for page_no, page_text in enumerate(pages, start=1):
            has_pages = len(pages) > 1
            for piece in chunk_text(page_text):
                chunk = Chunk(
                    id=self._next_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    ordinal=len(added),
                    text=piece,
                    page=page_no if has_pages else None,
                )
                self._chunks.append(chunk)
                for term in set(tokenize(piece)):
                    self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
                self._next_id += 1
                added.append(chunk)
        self._recompute()
        return added

    def set_embeddings(self, vectors: dict[int, list[float]]) -> None:
        """Attach semantic vectors keyed by chunk id (from :meth:`add_document`)."""
        self._embeddings.update(vectors)

    def remove_document(self, doc_id: str) -> bool:
        """Drop one document by id. Returns True if anything was removed."""
        keep = [c for c in self._chunks if c.doc_id != doc_id]
        if len(keep) == len(self._chunks):
            return False
        removed_ids = {c.id for c in self._chunks} - {c.id for c in keep}
        for cid in removed_ids:
            self._embeddings.pop(cid, None)
        self._chunks = keep
        self._doc_freq = {}
        for chunk in self._chunks:
            for term in set(tokenize(chunk.text)):
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1
        self._recompute()
        return True

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

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        query_embedding: list[float] | None = None,
    ) -> list[Retrieved]:
        """Return the top-``top_k`` chunks for ``query``, ranked by similarity.

        With a ``query_embedding`` and fully-embedded chunks, ranking is a hybrid
        of semantic and lexical similarity. Otherwise it is pure TF-IDF, where
        chunks below ``MIN_SCORE`` are dropped so an off-topic question yields an
        empty list — the signal the answer layer uses to refuse.
        """
        if self.is_empty() or not query.strip():
            return []
        query_vec = self._vectorize(query)
        if query_embedding is not None and self.has_embeddings():
            scored = self._hybrid_scores(query_vec, query_embedding)
            floor = SEM_MIN
        else:
            scored = [
                (chunk, cosine(query_vec, vec))
                for chunk, vec in zip(self._chunks, self._vectors, strict=False)
            ]
            floor = MIN_SCORE
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        results: list[Retrieved] = []
        for chunk, score in ranked[:top_k]:
            if score < floor:
                break
            results.append(Retrieved(chunk=chunk, score=score, rank=len(results) + 1))
        return results

    def _hybrid_scores(
        self, query_vec: dict[str, float], query_embedding: list[float]
    ) -> list[tuple[Chunk, float]]:
        """Blend semantic and lexical similarity for every chunk."""
        scored: list[tuple[Chunk, float]] = []
        for chunk, vec in zip(self._chunks, self._vectors, strict=False):
            lex = cosine(query_vec, vec)
            sem = cosine_dense(query_embedding, self._embeddings[chunk.id])
            scored.append((chunk, SEM_WEIGHT * sem + (1.0 - SEM_WEIGHT) * lex))
        return scored
