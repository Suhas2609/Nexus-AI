from typing import Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from rag.vectorstore.embedder import get_embedder

_store = None


def get_or_create_store() -> Chroma:
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=settings.chroma_collection_name,
            persist_directory=settings.chroma_persist_dir,
            embedding_function=get_embedder(),
        )
    return _store


def add_documents_idempotent(store: Chroma, chunks: List[Document]) -> Dict[str, int]:
    existing_chunk_ids: set[str] = set()
    try:
        existing_data = store.get(include=["metadatas"])
        for metadata in existing_data.get("metadatas", []):
            if metadata and "chunk_id" in metadata:
                existing_chunk_ids.add(metadata["chunk_id"])
    except Exception:
        existing_chunk_ids = set()

    new_chunks = []
    skipped_count = 0

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.metadata.get("chunk_id")
        
        # Enforce deterministic primary key generation
        if not chunk_id:
            source = chunk.metadata.get("source", "unknown")
            chunk_id = f"{source}_chunk_{i}"
            chunk.metadata["chunk_id"] = chunk_id

        # Validate against existing store and current batch
        if chunk_id not in existing_chunk_ids:
            new_chunks.append(chunk)
            existing_chunk_ids.add(chunk_id)
        else:
            skipped_count += 1

    if new_chunks:
        store.add_documents(new_chunks)

    total_in_store = len(existing_chunk_ids)

    return {
        "chunks_added": len(new_chunks),
        "chunks_skipped": skipped_count,
        "total_in_store": total_in_store,
    }