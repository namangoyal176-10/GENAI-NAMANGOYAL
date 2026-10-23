"""ChromaDB Vector Store management with persistent storage and fingerprint change detection."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.utils import embedding_functions

from src.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_DIR,
    DISTANCE_METRIC,
    EMBEDDING_MODEL_NAME,
)
from src.text_splitter import DocumentChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages persistent ChromaDB vector storage and embedding generation."""

    def __init__(self, persist_dir: Path | str = CHROMA_DB_DIR):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.persist_dir / "index_meta.json"

        # Initialize embedding function with sentence-transformers/all-MiniLM-L6-v2
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )

        # Initialize persistent Chroma client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = CHROMA_COLLECTION_NAME

    def get_or_create_collection(self):
        """Retrieves or creates the target ChromaDB collection with cosine distance."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )

    def is_indexed(self) -> bool:
        """Returns True if the persistent collection exists and contains indexed chunks."""
        try:
            col = self.get_or_create_collection()
            return col.count() > 0
        except Exception:
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Loads index metadata including source fingerprint, chunk count, and timestamp."""
        if not self.meta_path.exists():
            return {}
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read index metadata: {e}")
            return {}

    def save_metadata(self, meta: Dict[str, Any]) -> None:
        """Saves index metadata to disk."""
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def build_index(
        self,
        chunks: List[DocumentChunk],
        fingerprint: str,
        source_name: str,
        total_pages: int,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Indexes document chunks into persistent ChromaDB, rebuilding if fingerprint changed."""
        if not chunks:
            raise ValueError("Cannot index an empty list of document chunks.")

        current_meta = self.get_metadata()
        existing_fingerprint = current_meta.get("fingerprint")

        # Skip rebuilding if fingerprint matches, count matches, and force is False
        if not force and existing_fingerprint == fingerprint and self.is_indexed():
            logger.info("Fingerprint matches existing index. Skipping re-indexing.")
            return {
                "status": "up_to_date",
                "chunks_count": self.get_or_create_collection().count(),
                "pages_count": current_meta.get("total_pages", total_pages),
                "fingerprint": fingerprint,
            }

        logger.info(f"Building vector index for {len(chunks)} chunks...")

        # Delete existing collection to avoid duplicate chunks on reindex
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        # Create fresh collection
        col = self.get_or_create_collection()

        # Batch insert to prevent memory overload
        batch_size = 64
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            # Context-enriched embedding text with section and page prefix for high accuracy
            docs = [
                f"[Section: {c.metadata.get('section', '')} | Page: {c.metadata.get('page', '')}]\n{c.text}"
                for c in batch
            ]
            metadatas = [c.metadata for c in batch]
            col.add(ids=ids, documents=docs, metadatas=metadatas)

        final_count = col.count()
        if final_count != total_chunks:
            raise RuntimeError(
                f"Indexing verification failed: Expected {total_chunks} chunks, but ChromaDB indexed {final_count}."
            )

        # Save metadata record
        meta_record = {
            "source_name": source_name,
            "fingerprint": fingerprint,
            "total_pages": total_pages,
            "chunks_count": final_count,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "distance_metric": DISTANCE_METRIC,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save_metadata(meta_record)

        logger.info(f"Successfully indexed {final_count} chunks from {total_pages} pages.")
        return {
            "status": "indexed",
            "chunks_count": final_count,
            "pages_count": total_pages,
            "fingerprint": fingerprint,
        }
