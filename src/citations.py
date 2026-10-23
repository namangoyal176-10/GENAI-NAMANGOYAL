"""Citation Formatter module enforcing strict academic and assignment citation formatting."""

import re
from typing import Optional
from src.retriever import RetrievedPassage


class CitationFormatter:
    """Formats grounded answers and citations strictly adhering to assignment rules."""

    @staticmethod
    def format_citation(
        answer: str,
        passage: Optional[RetrievedPassage] = None,
        is_supported: bool = True,
        max_evidence_len: int = 280,
    ) -> str:
        """Formats the response into the mandatory assignment structure.

        If unsupported or answer is 'Not found', returns EXACTLY 'Not found'.
        Otherwise, returns:
        Answer: [Grounded answer]
        Reference: GER-3620-P.pdf, PDF page [page], Section [section].
        Evidence: [Short supporting excerpt]
        """
        # Strict fallback rule: Unsupported questions return ONLY 'Not found'
        if not is_supported or not passage or answer.strip().lower() in ["not found", "not found.", "'not found'"]:
            return "Not found"

        # Build printed page distinction if available and different
        printed_note = ""
        if passage.printed_page and passage.printed_page != "N/A" and passage.printed_page != str(passage.page):
            printed_note = f" (printed page {passage.printed_page})"

        # Extract concise, clean evidence snippet
        evidence_text = passage.text.replace("\n", " ").strip()
        evidence_text = re.sub(r"\s+", " ", evidence_text)
        if len(evidence_text) > max_evidence_len:
            evidence_snippet = f'"{evidence_text[:max_evidence_len].rstrip()}..."'
        else:
            evidence_snippet = f'"{evidence_text}"'

        # Build standardized citation block
        return (
            f"Answer: {answer.strip()}\n\n"
            f"Reference: {passage.source}, PDF page {passage.page}{printed_note}, Section {passage.section}.\n\n"
            f"Evidence: {evidence_snippet}"
        )
