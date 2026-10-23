"""Unit tests for PDFLoader, page extraction, section mapping, and page number distinction."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.pdf_loader import DocumentPage, PDFLoader, compute_pdf_fingerprint


def test_missing_pdf_raises_file_not_found():
    """Verify that attempting to load a non-existent PDF raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        PDFLoader(Path("non_existent_path.pdf"))
    assert "PDF file not found" in str(exc_info.value)


def test_pdf_fingerprint_generation(tmp_path):
    """Verify SHA-256 fingerprint generation is deterministic and sensitive to content."""
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"%PDF-1.4 dummy content for hash test")

    hash1 = compute_pdf_fingerprint(dummy_pdf)
    hash2 = compute_pdf_fingerprint(dummy_pdf)
    assert hash1 == hash2
    assert len(hash1) == 64

    # Modifying content produces different hash
    dummy_pdf.write_bytes(b"%PDF-1.4 changed content")
    hash3 = compute_pdf_fingerprint(dummy_pdf)
    assert hash1 != hash3


def test_page_number_distinction():
    """Verify that 1-based PDF page and printed page numbers are correctly distinguished."""
    loader = PDFLoader.__new__(PDFLoader)
    loader.pdf_path = Path("mock.pdf")

    # PDF page 5 (0-indexed 4) has printed page '1'
    printed_p5 = loader.extract_printed_page(4, "Header\n1\nIntroduction")
    assert printed_p5 == "1"

    # PDF page 25 (0-indexed 24) has printed page '21'
    printed_p25 = loader.extract_printed_page(24, "Header\n21\nMaintenance Inspections")
    assert printed_p25 == "21"

    # PDF page 3 (0-indexed 2) has Roman numeral 'i'
    printed_p3 = loader.extract_printed_page(2, "Header\ni\nContents")
    assert printed_p3 == "i"


def test_blank_page_handling(tmp_path):
    """Verify that blank pages are identified and flagged without causing crashes."""
    dummy_pdf = tmp_path / "sample.pdf"
    import pymupdf
    doc = pymupdf.open()
    # Page 1 with text
    p1 = doc.new_page()
    p1.insert_text((50, 50), "Heavy-Duty Gas Turbine Maintenance Planning")
    # Page 2 empty
    doc.new_page()
    doc.save(dummy_pdf)
    doc.close()

    loader = PDFLoader(dummy_pdf)
    pages = loader.load()

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert not pages[0].is_blank
    assert pages[1].page_number == 2
    assert pages[1].is_blank
