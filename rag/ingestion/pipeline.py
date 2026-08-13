from typing import Dict, Any
from rag.ingestion.loader import load_documents
from rag.ingestion.chunker import chunk_documents
from rag.vectorstore.store import get_or_create_store, add_documents_idempotent

def ingest(path: str) -> Dict[str, Any]:
    docs = load_documents(path)
    chunks = chunk_documents(docs)
    store = get_or_create_store()   
    stats = add_documents_idempotent(store, chunks)

    stats["source"] = path

    return stats