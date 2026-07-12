"""Turn an uploaded file into page texts the index can consume.

Supports PDF (per-page text via ``pypdf``) and plain text / markdown. Keeping
extraction in one place means the rest of the app only ever sees ``list[str]``.
"""

from __future__ import annotations

from typing import Final

TEXT_SUFFIXES: Final[tuple[str, ...]] = (".txt", ".md", ".markdown")
MIN_DOC_CHARS: Final[int] = 20


class UnsupportedFileError(ValueError):
    """Raised when a file type cannot be extracted."""


class EmptyDocumentError(ValueError):
    """Raised when a file yields no usable text (e.g. a scanned PDF)."""


def extract_pages(filename: str, data: bytes) -> list[str]:
    """Return page texts for ``data``, dispatching on the filename suffix.

    :raises UnsupportedFileError: the suffix is not a PDF or text format.
    :raises EmptyDocumentError: extraction produced no meaningful text.
    """
    lower = filename.lower()
    if lower.endswith(".pdf"):
        pages = _extract_pdf(data)
    elif lower.endswith(TEXT_SUFFIXES):
        pages = [data.decode("utf-8", errors="replace")]
    else:
        raise UnsupportedFileError(
            "Upload a PDF, .txt, or .md file (got: " + filename + ")."
        )
    if sum(len(p.strip()) for p in pages) < MIN_DOC_CHARS:
        raise EmptyDocumentError(
            "No readable text found. Scanned PDFs without a text layer are not "
            "supported."
        )
    return pages


def _extract_pdf(data: bytes) -> list[str]:
    """Extract text page by page from a PDF byte string."""
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "").strip() for page in reader.pages]
