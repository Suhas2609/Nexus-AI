from typing import Dict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from rag.vectorstore.embedder import get_embedder


def get_or_create_store() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
        embedding_function=get_embedder(),
    )


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

    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")
        if chunk_id not in existing_chunk_ids:
            new_chunks.append(chunk)
        else:
            skipped_count += 1

    if new_chunks:
        store.add_documents(new_chunks)

    total_in_store = len(existing_chunk_ids) + len(new_chunks)

    return {
        "chunks_added": len(new_chunks),
        "chunks_skipped": skipped_count,
        "total_in_store": total_in_store,
    }
