"""Unit tests for VectorStore persistence, reindexing, and duplicate prevention."""

import pytest
from pathlib import Path
from src.text_splitter import DocumentChunk
from src.vector_store import VectorStore


def create_sample_chunks():
    """Helper to generate mock DocumentChunks."""
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            text="The three main types of inspection are standby, running, and disassembly inspections.",
            metadata={"source": "GER-3620-P.pdf", "page": 25, "printed_page": "21", "section": "Maintenance Inspections", "chunk_id": "chunk_1"},
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            text="A maintenance factor of two indicates a maintenance interval that is half of the baseline interval.",
            metadata={"source": "GER-3620-P.pdf", "page": 9, "printed_page": "5", "section": "Starts and Hours Criteria", "chunk_id": "chunk_2"},
        ),
    ]


def test_vector_store_indexing_and_count(tmp_path):
    """Verify that chunks are properly stored in ChromaDB and count is correct."""
    store = VectorStore(persist_dir=tmp_path / "chroma_test")
    chunks = create_sample_chunks()

    res = store.build_index(
        chunks=chunks,
        fingerprint="test_hash_123",
        source_name="GER-3620-P.pdf",
        total_pages=2,
        force=True,
    )

    assert res["status"] == "indexed"
    assert res["chunks_count"] == 2
    assert store.is_indexed()

    col = store.get_or_create_collection()
    assert col.count() == 2


def test_reindexing_prevents_duplicate_chunks(tmp_path):
    """Verify that rebuilding the index clears old entries and avoids duplicate records."""
    store = VectorStore(persist_dir=tmp_path / "chroma_dup_test")
    chunks = create_sample_chunks()

    # First index pass
    store.build_index(chunks=chunks, fingerprint="hash_v1", source_name="GER-3620-P.pdf", total_pages=2, force=True)
    assert store.get_or_create_collection().count() == 2

    # Second index pass with force=True should still result in exactly 2 chunks (no duplicates)
    store.build_index(chunks=chunks, fingerprint="hash_v2", source_name="GER-3620-P.pdf", total_pages=2, force=True)
    assert store.get_or_create_collection().count() == 2


def test_skips_reindexing_when_fingerprint_matches(tmp_path):
    """Verify that indexing is skipped when the document fingerprint has not changed."""
    store = VectorStore(persist_dir=tmp_path / "chroma_skip_test")
    chunks = create_sample_chunks()

    # First build
    res1 = store.build_index(chunks=chunks, fingerprint="same_hash", source_name="GER-3620-P.pdf", total_pages=2)
    assert res1["status"] == "indexed"

    # Second build with same fingerprint and force=False
    res2 = store.build_index(chunks=chunks, fingerprint="same_hash", source_name="GER-3620-P.pdf", total_pages=2, force=False)
    assert res2["status"] == "up_to_date"
