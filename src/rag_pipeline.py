"""End-to-End RAG Pipeline Orchestrator for Gas Turbine Maintenance Assistant."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.answer_generator import AnswerGenerator
from src.citations import CitationFormatter
from src.config import DEFAULT_PDF_NAME, PDF_PATH
from src.evidence_validator import EvidenceValidator, ValidationResult
from src.retriever import HybridRetriever, RetrievedPassage
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Encapsulates the complete result of a RAG query execution."""
    query: str
    formatted_response: str
    answer: str
    reference: str
    evidence: str
    is_supported: bool
    pdf_page: Optional[int] = None
    printed_page: Optional[str] = None
    section: Optional[str] = None
    retrieved_passages: List[RetrievedPassage] = field(default_factory=list)
    best_distance: Optional[float] = None
    best_similarity: Optional[float] = None
    validation_reason: str = ""
    setup_error: Optional[str] = None


class RAGPipeline:
    """Coordinates retrieval, evidence validation, answer generation, and citation formatting."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[HybridRetriever] = None,
        validator: Optional[EvidenceValidator] = None,
        generator: Optional[AnswerGenerator] = None,
        pdf_path: Path = PDF_PATH,
    ):
        self.pdf_path = Path(pdf_path)
        self.vector_store = vector_store or VectorStore()
        self.retriever = retriever or HybridRetriever(self.vector_store)
        self.validator = validator or EvidenceValidator()
        self.generator = generator or AnswerGenerator()
        self.formatter = CitationFormatter()

    def check_setup(self) -> Optional[str]:
        """Checks whether the system prerequisites (PDF file and persistent index) are met."""
        if not self.pdf_path.exists():
            return (
                f"SETUP ERROR: Document '{self.pdf_path.name}' is missing from {self.pdf_path.parent.resolve()}.\n"
                f"Please place '{DEFAULT_PDF_NAME}' into the data/ folder and run 'python -m scripts.ingest_pdf'."
            )

        if not self.vector_store.is_indexed():
            return (
                f"SETUP ERROR: Vector index is empty or not yet built.\n"
                f"Please build the index first by running: 'python -m scripts.ingest_pdf'."
            )

        return None

    def query(self, question: str) -> RAGResponse:
        """Executes a question through the full RAG pipeline."""
        q = question.strip()
        if not q:
            return RAGResponse(
                query=question,
                formatted_response="Not found",
                answer="Not found",
                reference="",
                evidence="",
                is_supported=False,
                validation_reason="Empty question provided.",
            )

        # Check for setup errors first — never falsely return 'Not found' when index is unavailable
        setup_err = self.check_setup()
        if setup_err:
            return RAGResponse(
                query=q,
                formatted_response=setup_err,
                answer="",
                reference="",
                evidence="",
                is_supported=False,
                setup_error=setup_err,
                validation_reason="Setup prerequisite failed.",
            )

        # 1. Retrieve relevant passages
        passages = self.retriever.retrieve(q)

        # 2. Validate evidence outside the LLM
        validation: ValidationResult = self.validator.validate(q, passages)

        # 3. Generate grounded answer
        answer = self.generator.generate(
            question=q,
            passages=validation.validated_passages or passages,
            is_supported=validation.is_supported,
        )

        # 4. Format citations and references
        best_passage = validation.best_passage or (passages[0] if passages else None)
        formatted_text = self.formatter.format_citation(
            answer=answer,
            passage=best_passage,
            is_supported=validation.is_supported,
        )

        # Parse reference and evidence fields for UI display
        reference_str = ""
        evidence_str = ""
        if validation.is_supported and best_passage and answer != "Not found":
            printed_note = f" (printed page {best_passage.printed_page})" if best_passage.printed_page != "N/A" else ""
            reference_str = f"{best_passage.source}, PDF page {best_passage.page}{printed_note}, Section {best_passage.section}"
            evidence_str = best_passage.text[:280].replace("\n", " ").strip()

        return RAGResponse(
            query=q,
            formatted_response=formatted_text,
            answer=answer,
            reference=reference_str,
            evidence=evidence_str,
            is_supported=validation.is_supported and answer != "Not found",
            pdf_page=best_passage.page if best_passage else None,
            printed_page=best_passage.printed_page if best_passage else None,
            section=best_passage.section if best_passage else None,
            retrieved_passages=passages,
            best_distance=best_passage.distance if best_passage else None,
            best_similarity=best_passage.similarity if best_passage else None,
            validation_reason=validation.reason,
        )
