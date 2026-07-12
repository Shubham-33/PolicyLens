"""Tests for file -> page-text extraction."""

from unittest.mock import MagicMock, patch

import pytest

from ingest import (
    EmptyDocumentError,
    UnsupportedFileError,
    extract_pages,
)

LONG_TEXT = "This is a policy document with enough text to be considered usable."


def test_extract_txt():
    pages = extract_pages("policy.txt", LONG_TEXT.encode())
    assert pages == [LONG_TEXT]


def test_extract_markdown():
    pages = extract_pages("notes.md", LONG_TEXT.encode())
    assert pages == [LONG_TEXT]


def test_unsupported_type_raises():
    with pytest.raises(UnsupportedFileError):
        extract_pages("archive.zip", b"data")


def test_empty_document_raises():
    with pytest.raises(EmptyDocumentError):
        extract_pages("blank.txt", b"   ")


def test_extract_pdf_reads_pages():
    fake_page = MagicMock()
    fake_page.extract_text.return_value = LONG_TEXT
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page, fake_page]
    with patch("pypdf.PdfReader", return_value=fake_reader):
        pages = extract_pages("policy.pdf", b"%PDF-fake")
    assert pages == [LONG_TEXT, LONG_TEXT]


def test_extract_pdf_empty_raises():
    fake_page = MagicMock()
    fake_page.extract_text.return_value = ""
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]
    with (
        patch("pypdf.PdfReader", return_value=fake_reader),
        pytest.raises(EmptyDocumentError),
    ):
        extract_pages("scanned.pdf", b"%PDF-fake")
