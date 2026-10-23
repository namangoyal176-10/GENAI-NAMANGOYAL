"""Text splitter module for chunking document pages with required metadata."""

from dataclasses import dataclass
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_PDF_NAME
from src.pdf_loader import DocumentPage


@dataclass
class DocumentChunk:
    """Represents a text chunk with associated document metadata."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


class DocumentSplitter:
    """Splits DocumentPages into chunks of approximately 800-1000 characters with overlap."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def split_pages(self, pages: List[DocumentPage], source_name: str = DEFAULT_PDF_NAME) -> List[DocumentChunk]:
        """Splits a list of DocumentPages into chunks with full metadata attached."""
        all_chunks: List[DocumentChunk] = []

        for page in pages:
            if page.is_blank or not page.text.strip():
                continue

            raw_chunks = self.splitter.split_text(page.text)
            for idx, chunk_text in enumerate(raw_chunks):
                chunk_id = f"{source_name}_p{page.page_number}_c{idx + 1}"
                metadata = {
                    "source": source_name,
                    "page": int(page.page_number),
                    "printed_page": str(page.printed_page),
                    "section": str(page.section),
                    "chunk_id": chunk_id,
                    "char_count": len(chunk_text),
                }

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )

        return all_chunks
