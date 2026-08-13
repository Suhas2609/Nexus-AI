from langchain_core.documents import Document

from config.settings import settings
from rag.ingestion.chunker import chunk_documents


def _make_doc(content: str, source: str = "test.pdf", page: int = 1) -> Document:
    return Document(page_content=content, metadata={"source": source, "page": page})


def test_chunk_size_within_tolerance():
    long_text = "word " * (settings.chunk_size + 100)
    chunks = chunk_documents([_make_doc(long_text)])
    tolerance = 50
    for chunk in chunks:
        assert len(chunk.page_content) <= settings.chunk_size + tolerance


def test_chunk_id_is_md5_hex():
    chunks = chunk_documents([_make_doc("Some content for hashing.")])
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert len(chunk.metadata["chunk_id"]) == 32
        assert all(c in "0123456789abcdef" for c in chunk.metadata["chunk_id"])


def test_source_metadata_preserved():
    chunks = chunk_documents([_make_doc("Content A", source="report.pdf")])
    for chunk in chunks:
        assert chunk.metadata["source"] == "report.pdf"


def test_page_metadata_preserved():
    chunks = chunk_documents([_make_doc("Content B", page=7)])
    for chunk in chunks:
        assert chunk.metadata["page"] == 7


def test_different_content_different_chunk_ids():
    chunks = chunk_documents([_make_doc("First chunk content."), _make_doc("Second chunk content.")])
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(set(ids)) == len(ids)
