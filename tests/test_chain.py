from langchain_core.documents import Document

from rag.chain.rag_chain import format_sources


def test_format_sources_deduplicates():
    docs = [
        Document(page_content="a", metadata={"source": "file.pdf", "page": 1}),
        Document(page_content="b", metadata={"source": "file.pdf", "page": 1}),
        Document(page_content="c", metadata={"source": "file.pdf", "page": 2}),
    ]
    sources = format_sources(docs)
    assert len(sources) == 2
    assert {"source": "file.pdf", "page": 1} in sources
    assert {"source": "file.pdf", "page": 2} in sources


def test_format_sources_defaults_for_missing_metadata():
    docs = [Document(page_content="no metadata")]
    sources = format_sources(docs)
    assert sources == [{"source": "unknown", "page": 0}]


def test_format_sources_empty_input():
    assert format_sources([]) == []
