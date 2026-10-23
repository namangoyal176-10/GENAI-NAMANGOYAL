"""Unit tests for CitationFormatter format compliance and strict 'Not found' fallback."""

from src.citations import CitationFormatter
from src.retriever import RetrievedPassage


def test_supported_citation_format():
    """Verify supported questions produce correct Answer, Reference, and Evidence blocks."""
    passage = RetrievedPassage(
        chunk_id="c_test",
        text="Maintenance inspection types may be broadly classified as standby, running, and disassembly inspections.",
        page=25,
        printed_page="21",
        section="Maintenance Inspections",
        source="GER-3620-P.pdf",
        distance=0.15,
        similarity=0.85,
        score=0.9,
    )

    answer = "The three main types of inspection are standby, running, and disassembly inspections."
    formatted = CitationFormatter.format_citation(answer, passage, is_supported=True)

    assert formatted.startswith("Answer: ")
    assert "Reference: GER-3620-P.pdf, PDF page 25 (printed page 21), Section Maintenance Inspections." in formatted
    assert "Evidence: " in formatted
    assert "Maintenance inspection types may be broadly classified" in formatted


def test_unsupported_question_exact_not_found():
    """Verify that unsupported questions return EXACTLY 'Not found' with zero extra text."""
    # When is_supported is False
    res1 = CitationFormatter.format_citation("Not found", None, is_supported=False)
    assert res1 == "Not found"

    # Even if answer string has punctuation
    res2 = CitationFormatter.format_citation("Not found.", None, is_supported=False)
    assert res2 == "Not found"

    # When passage is provided but answer is marked unsupported
    passage = RetrievedPassage(
        chunk_id="c_irr",
        text="The turbine cost of maintenance involves five systems.",
        page=5,
        printed_page="1",
        section="Introduction",
        source="GER-3620-P.pdf",
        distance=0.45,
        similarity=0.55,
        score=0.5,
    )
    res3 = CitationFormatter.format_citation("Not found", passage, is_supported=False)
    assert res3 == "Not found"
