"""PolicyLens — a retrieval-grounded policy & FAQ assistant.

Upload a policy document, ask questions, and get answers grounded in that
document with clickable citations that point back to the exact source passage.

Layers:
  * ``rag.py``    — TF-IDF retrieval (pure Python, deterministic)
  * ``nim.py``    — NVIDIA NIM grounded generation, with an offline fallback
  * ``ingest.py`` — file -> page-text extraction
  * this module   — Flask routes, middleware, and in-memory state
"""

from __future__ import annotations

import gzip
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
)

import nim
from ingest import EmptyDocumentError, UnsupportedFileError, extract_pages
from rag import DEFAULT_TOP_K, TfidfIndex
from sample_data import SAMPLE_DOC_NAME, SAMPLE_POLICY

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES: Final[int] = 8 * 1024 * 1024  # 8 MB
MAX_QUESTION_CHARS: Final[int] = 500
GZIP_MIN_BYTES: Final[int] = 500
INDEX_CACHE_SECONDS: Final[int] = 300
STATIC_CACHE_SECONDS: Final[int] = 60 * 60 * 24
BUILD_ID: Final[str] = str(int(Path(__file__).stat().st_mtime))

_FAVICON: Final[str] = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#0b6b3a"/>'
    '<text x="16" y="22" font-size="18" text-anchor="middle" fill="#fff" '
    'font-family="Georgia,serif">P</text></svg>'
)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    """Build and configure the Flask application."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = STATIC_CACHE_SECONDS
    app.config["JSON_SORT_KEYS"] = False

    index = TfidfIndex()

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
        """Report what is loaded and whether live generation is available."""
        return jsonify(
            {
                "documents": index.doc_names,
                "chunks": len(index.chunks),
                "llm_configured": nim.is_configured(),
            }
        )

    @app.route("/api/sample", methods=["POST"])
    def load_sample() -> Response:
        """Reset the index and load the bundled sample policy document."""
        index.clear()
        count = index.add_document(
            doc_id=_new_doc_id(), doc_name=SAMPLE_DOC_NAME, pages=[SAMPLE_POLICY]
        )
        return jsonify({"name": SAMPLE_DOC_NAME, "chunks": count})

    @app.route("/api/ingest", methods=["POST"])
    def ingest() -> tuple[Response, int] | Response:
        """Index an uploaded file or pasted text, replacing the current corpus."""
        try:
            name, pages = _read_ingest_request()
        except (UnsupportedFileError, EmptyDocumentError) as exc:
            return jsonify({"error": str(exc)}), 400
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        index.clear()
        count = index.add_document(doc_id=_new_doc_id(), doc_name=name, pages=pages)
        return jsonify({"name": name, "chunks": count})

    @app.route("/api/ask", methods=["POST"])
    def ask() -> tuple[Response, int] | Response:
        """Answer a question against the indexed corpus, with citations."""
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        if not question:
            return jsonify({"error": "Ask a question first."}), 400
        if len(question) > MAX_QUESTION_CHARS:
            return jsonify({"error": "Question is too long."}), 400
        if index.is_empty():
            return jsonify({"error": "Load or upload a document first."}), 400

        retrieved = index.search(question, top_k=DEFAULT_TOP_K)
        result = nim.answer_question(question, retrieved)
        return jsonify(
            {
                **result.to_dict(),
                "sources": [_source_dict(item) for item in retrieved],
            }
        )

    @app.route("/api/reset", methods=["POST"])
    def reset() -> Response:
        """Clear the entire corpus."""
        index.clear()
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
# Request helpers
# ---------------------------------------------------------------------------


def _new_doc_id() -> str:
    """A short unique id for a freshly ingested document."""
    return uuid.uuid4().hex[:8]


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


def _source_dict(item: Any) -> dict[str, Any]:
    """Serialise a retrieved chunk for the sources panel."""
    return {
        "rank": item.rank,
        "doc_name": item.chunk.doc_name,
        "page": item.chunk.page,
        "text": item.chunk.text,
        "score": round(item.score, 4),
    }


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import os

    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
