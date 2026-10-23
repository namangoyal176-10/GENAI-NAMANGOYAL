"""Integration and end-to-end tests for RAGPipeline."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.answer_generator import AnswerGenerator
from src.evidence_validator import EvidenceValidator
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.retriever import HybridRetriever, RetrievedPassage
from src.vector_store import VectorStore


def test_missing_pdf_returns_setup_error(tmp_path):
    """Verify that a missing PDF returns a clear setup error, not a false 'Not found'."""
    missing_pdf = tmp_path / "missing_GER-3620-P.pdf"
    pipeline = RAGPipeline(pdf_path=missing_pdf)

    res = pipeline.query("What are the three main types of inspection?")
    assert res.setup_error is not None
    assert "SETUP ERROR" in res.formatted_response
    assert res.formatted_response != "Not found"
    assert not res.is_supported


def test_missing_index_returns_setup_error(tmp_path):
    """Verify that an unindexed vector store returns a setup error, not 'Not found'."""
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"dummy")

    mock_store = MagicMock(spec=VectorStore)
    mock_store.is_indexed.return_value = False

    pipeline = RAGPipeline(vector_store=mock_store, pdf_path=dummy_pdf)
    res = pipeline.query("What are the three main types of inspection?")

    assert res.setup_error is not None
    assert "Vector index is empty or not yet built" in res.formatted_response
    assert res.formatted_response != "Not found"


def test_pipeline_supported_answer_with_mocked_retriever(tmp_path):
    """Verify pipeline output structure on a supported question with mocked retriever."""
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"dummy")

    mock_store = MagicMock(spec=VectorStore)
    mock_store.is_indexed.return_value = True

    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = [
        RetrievedPassage(
            chunk_id="chunk_test",
            text="Maintenance inspection types may be broadly classified as standby, running, and disassembly inspections.",
            page=25,
            printed_page="21",
            section="Maintenance Inspections",
            source="GER-3620-P.pdf",
            distance=0.20,
            similarity=0.80,
            score=0.85,
        )
    ]

    pipeline = RAGPipeline(
        vector_store=mock_store,
        retriever=mock_retriever,
        validator=EvidenceValidator(),
        generator=AnswerGenerator(provider="local"),
        pdf_path=dummy_pdf,
    )

    response: RAGResponse = pipeline.query("What are the three main types of inspection?")

    assert response.is_supported
    assert "standby" in response.answer.lower()
    assert "running" in response.answer.lower()
    assert "disassembly" in response.answer.lower()
    assert response.pdf_page == 25
    assert response.printed_page == "21"
    assert response.section == "Maintenance Inspections"
    assert "Reference: GER-3620-P.pdf, PDF page 25 (printed page 21), Section Maintenance Inspections." in response.formatted_response


def test_pipeline_unsupported_question_returns_exact_not_found(tmp_path):
    """Verify pipeline returns exactly 'Not found' for unsupported questions."""
    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"dummy")

    mock_store = MagicMock(spec=VectorStore)
    mock_store.is_indexed.return_value = True

    mock_retriever = MagicMock(spec=HybridRetriever)
    # Returns passage that talks about turbine costs, but NOT purchase price
    mock_retriever.retrieve.return_value = [
        RetrievedPassage(
            chunk_id="c_cost",
            text="Maintenance costs and machine availability are two of the most important concerns to an owner.",
            page=5,
            printed_page="1",
            section="Maintenance Planning",
            source="GER-3620-P.pdf",
            distance=0.42,
            similarity=0.58,
            score=0.60,
        )
    ]

    pipeline = RAGPipeline(
        vector_store=mock_store,
        retriever=mock_retriever,
        validator=EvidenceValidator(),
        generator=AnswerGenerator(provider="local"),
        pdf_path=dummy_pdf,
    )

    response: RAGResponse = pipeline.query("What is the purchase price of this gas turbine?")

    assert not response.is_supported
    assert response.formatted_response == "Not found"
    assert response.answer == "Not found"
