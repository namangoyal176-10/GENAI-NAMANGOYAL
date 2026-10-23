"""Script to ingest GER-3620-P.pdf, extract text/metadata, chunk, and index into ChromaDB."""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PDF_PATH
from src.pdf_loader import PDFLoader, compute_pdf_fingerprint
from src.text_splitter import DocumentSplitter
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestPDF")


def run_ingestion(pdf_path: Path = PDF_PATH, force: bool = False):
    """Executes the complete document ingestion pipeline."""
    print("=" * 70)
    print("GAS TURBINE MAINTENANCE ASSISTANT - DOCUMENT INGESTION")
    print("=" * 70)

    if not pdf_path.exists():
        print(f"\n[ERROR] PDF file not found at: {pdf_path.resolve()}")
        print("Please place 'GER-3620-P.pdf' into the data/ folder before indexing.")
        sys.exit(1)

    print(f"Loading document: {pdf_path.name}")
    start_time = time.time()

    # Step 1: Compute source fingerprint
    fingerprint = compute_pdf_fingerprint(pdf_path)
    print(f"Document SHA-256 Fingerprint: {fingerprint[:16]}...{fingerprint[-8:]}")

    # Step 2: Load PDF and extract pages
    loader = PDFLoader(pdf_path)
    pages = loader.load()
    readable_pages = [p for p in pages if not p.is_blank]
    blank_pages = [p for p in pages if p.is_blank]
    print(f"Total pages inspected: {len(pages)}")
    print(f"Readable pages: {len(readable_pages)} | Blank pages handled: {len(blank_pages)}")

    # Step 3: Split into chunks with metadata
    splitter = DocumentSplitter()
    chunks = splitter.split_pages(pages, source_name=pdf_path.name)
    print(f"Generated text chunks: {len(chunks)} (approx 800-1000 characters each)")

    if not chunks:
        print("[ERROR] Extraction produced 0 chunks. Ingestion aborted.")
        sys.exit(1)

    # Step 4: Index into persistent ChromaDB
    print("\nInitializing persistent vector store and generating embeddings...")
    vector_store = VectorStore()
    result = vector_store.build_index(
        chunks=chunks,
        fingerprint=fingerprint,
        source_name=pdf_path.name,
        total_pages=len(pages),
        force=force,
    )

    elapsed = time.time() - start_time
    print("-" * 70)
    print("INGESTION SUMMARY:")
    print(f"  Status:               {result['status'].upper()}")
    print(f"  Source Document:      {pdf_path.name}")
    print(f"  Pages Processed:      {result['pages_count']}")
    print(f"  Chunks Indexed:       {result['chunks_count']}")
    print(f"  Fingerprint:          {result['fingerprint'][:24]}...")
    print(f"  Elapsed Time:         {elapsed:.2f} seconds")
    print("=" * 70)
    print("SUCCESS: Vector database is ready for similarity retrieval and query answering.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest GER-3620-P.pdf into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force reindexing even if fingerprint matches")
    parser.add_argument("--pdf", type=str, default=str(PDF_PATH), help="Path to PDF document")
    args = parser.parse_args()

    run_ingestion(pdf_path=Path(args.pdf), force=args.force)
