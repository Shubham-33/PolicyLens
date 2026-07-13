"""PolicyLens — a retrieval-grounded policy & FAQ assistant.

Upload one or more policy documents, ask questions, and get answers grounded in
those documents with clickable citations that point back to the exact source
passage. Each browser session has its own private corpus, so concurrent users
never see each other's documents.

Layers:
  * ``rag.py``    — TF-IDF + hybrid semantic retrieval (deterministic core)
  * ``embed.py``  — NVIDIA embeddings for semantic recall, with a no-op fallback
  * ``nim.py``    — NVIDIA NIM grounded generation, with an offline fallback
  * ``ingest.py`` — file -> page-text extraction
  * this module   — Flask routes, per-session state, and middleware
"""

from __future__ import annotations

import gzip
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Final

from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    jsonify,
    make_response,
    render_template,
    request,
    session,
)

import embed
import nim
import session_store
from ingest import EmptyDocumentError, UnsupportedFileError, extract_pages
from rag import DEFAULT_TOP_K, RetrievalIndex, best_sentence
from sample_data import SAMPLE_DOC_NAME, SAMPLE_POLICY

load_dotenv()

# Some minimal Linux images (e.g. the Render build image) map .js/.css to
# text/plain, which makes browsers refuse to execute the served script. Pin the
# correct types so the front-end runs everywhere, not just on macOS.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES: Final[int] = 8 * 1024 * 1024  # 8 MB
MAX_QUESTION_CHARS: Final[int] = 500
GZIP_MIN_BYTES: Final[int] = 500
INDEX_CACHE_SECONDS: Final[int] = 300
STATIC_CACHE_SECONDS: Final[int] = 60 * 60 * 24
SAMPLE_DOC_ID: Final[str] = "sample"
BUILD_ID: Final[str] = str(int(Path(__file__).stat().st_mtime))

_FAVICON: Final[str] = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#0b6b3a"/>'
    '<text x="16" y="22" font-size="18" text-anchor="middle" fill="#fff" '
    'font-family="Georgia,serif">P</text></svg>'
)

# ---------------------------------------------------------------------------
# Per-session corpus (disk-backed; see session_store)
# ---------------------------------------------------------------------------


def _current_sid() -> str:
    """Return this browser's session id, minting one on first visit."""
    sid = session.get("sid")
    if not sid:
        sid = uuid.uuid4().hex
        session["sid"] = sid
    return sid


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    """Build and configure the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = STATIC_CACHE_SECONDS
    app.config["JSON_SORT_KEYS"] = False
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "policylens-dev-secret")

    # -- Routes -------------------------------------------------------------

    @app.context_processor
    def inject_build_id() -> dict[str, str]:
        return {"build_id": BUILD_ID}

    @app.route("/")
    def home() -> Response:
        """Serve the single-page UI (cacheable for a few minutes)."""
        resp = make_response(render_template("index.html"))
        resp.headers["Cache-Control"] = f"public, max-age={INDEX_CACHE_SECONDS}"
        return resp

    @app.route("/favicon.ico")
    def favicon() -> Response:
        """Inline SVG favicon so no 404 dings the Best-Practices audit."""
        resp = make_response(_FAVICON)
        resp.headers["Content-Type"] = "image/svg+xml"
        resp.headers["Cache-Control"] = f"public, max-age={STATIC_CACHE_SECONDS}"
        return resp

    @app.route("/healthz")
    def healthz() -> Response:
        """Liveness probe."""
        return jsonify({"status": "ok", "llm": nim.is_configured()})

    @app.route("/api/status")
    def status() -> Response:
        """Report the current session's documents and available capabilities."""
        index = session_store.load(_current_sid())
        return jsonify(
            {
                "documents": index.documents,
                "chunks": len(index.chunks),
                "semantic": index.has_embeddings(),
                "llm_configured": nim.is_configured(),
                "embeddings_available": embed.is_configured(),
            }
        )

    @app.route("/api/sample", methods=["POST"])
    def load_sample() -> Response:
        """Add the bundled sample policy to this session (idempotent)."""
        sid = _current_sid()
        with session_store.locked(sid):
            index = session_store.load(sid)
            if not any(d["doc_id"] == SAMPLE_DOC_ID for d in index.documents):
                _index_document(index, SAMPLE_DOC_ID, SAMPLE_DOC_NAME, [SAMPLE_POLICY])
                session_store.save(sid, index)
        return _corpus_response(index)

    @app.route("/api/ingest", methods=["POST"])
    def ingest() -> tuple[Response, int] | Response:
        """Index an uploaded file or pasted text, adding to this session's corpus."""
        try:
            name, pages = _read_ingest_request()
        except (UnsupportedFileError, EmptyDocumentError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        sid = _current_sid()
        with session_store.locked(sid):
            index = session_store.load(sid)
            _index_document(index, uuid.uuid4().hex[:8], name, pages)
            session_store.save(sid, index)
        return _corpus_response(index)

    @app.route("/api/remove", methods=["POST"])
    def remove() -> tuple[Response, int] | Response:
        """Remove one document from this session by id."""
        sid = _current_sid()
        with session_store.locked(sid):
            index = session_store.load(sid)
            doc_id = str((request.get_json(silent=True) or {}).get("doc_id", ""))
            if not index.remove_document(doc_id):
                return jsonify({"error": "Document not found."}), 404
            session_store.save(sid, index)
        return _corpus_response(index)

    @app.route("/api/ask", methods=["POST"])
    def ask() -> tuple[Response, int] | Response:
        """Answer a question against this session's corpus, with citations."""
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Ask a question first."}), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify({"error": "Question is too long."}), 400
        index = session_store.load(_current_sid())
        if index.is_empty():
            return jsonify({"error": "Load or upload a document first."}), 400

        query_embedding = embed.embed_query(question) if index.has_embeddings() else None
        retrieved = index.search(
            question, top_k=DEFAULT_TOP_K, query_embedding=query_embedding
        )
        result = nim.answer_question(question, retrieved)
        return jsonify(
            {
                **result.to_dict(),
                "semantic": query_embedding is not None,
                "sources": [_source_dict(item, question) for item in retrieved],
            }
        )

    @app.route("/api/reset", methods=["POST"])
    def reset() -> Response:
        """Clear this session's entire corpus."""
        session_store.save(_current_sid(), RetrievalIndex())
        return jsonify({"ok": True})

    # -- Middleware ---------------------------------------------------------

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        """Conservative security headers (cheap points, real value)."""
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    @app.after_request
    def gzip_response(response: Response) -> Response:
        """Compress eligible text responses with stdlib gzip (no dependency)."""
        accepts = request.headers.get("Accept-Encoding") or ""
        if (
            response.direct_passthrough
            or not (200 <= response.status_code < 300)
            or response.headers.get("Content-Encoding")
            or "gzip" not in accepts
        ):
            return response
        if (response.content_length or 0) < GZIP_MIN_BYTES:
            return response
        compressed = gzip.compress(response.get_data(), compresslevel=6)
        response.set_data(compressed)
        response.headers["Content-Encoding"] = "gzip"
        response.headers["Content-Length"] = str(len(compressed))
        response.headers["Vary"] = "Accept-Encoding"
        return response

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _index_document(
    index: RetrievalIndex, doc_id: str, name: str, pages: list[str]
) -> None:
    """Chunk, index, and (when possible) semantically embed a document."""
    chunks = index.add_document(doc_id, name, pages)
    vectors = embed.embed_passages([c.text for c in chunks])
    if vectors:
        index.set_embeddings({c.id: v for c, v in zip(chunks, vectors, strict=False)})


def _corpus_response(index: RetrievalIndex) -> Response:
    """Standard payload describing the session's current corpus."""
    return jsonify(
        {
            "documents": index.documents,
            "chunks": len(index.chunks),
            "semantic": index.has_embeddings(),
        }
    )


def _read_ingest_request() -> tuple[str, list[str]]:
    """Pull (name, page_texts) from a file upload or a JSON text paste."""
    uploaded = request.files.get("file")
    if uploaded and uploaded.filename:
        data = uploaded.read()
        return uploaded.filename, extract_pages(uploaded.filename, data)

    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    if not text:
        raise ValueError("Provide a file or some text to index.")
    name = str(payload.get("name", "")).strip() or "Pasted text"
    return name, [text]


def _source_dict(item: Any, question: str) -> dict[str, Any]:
    """Serialise a retrieved chunk for the sources panel.

    ``highlight`` is the sentence in the passage most relevant to the question,
    so the UI can mark exactly which part the answer was drawn from.
    """
    return {
        "rank": item.rank,
        "doc_name": item.chunk.doc_name,
        "page": item.chunk.page,
        "text": item.chunk.text,
        "highlight": best_sentence(question, item.chunk.text),
        "score": round(item.score, 4),
    }


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
