from agents.state import AgentState
from rag.chain.rag_chain import format_sources
from rag.vectorstore.retriever import get_mmr_retriever
from rag.vectorstore.store import get_or_create_store

_retriever = None

def get_cached_retriever():
    global _retriever
    if _retriever is None:
        _retriever = get_mmr_retriever(get_or_create_store())
    return _retriever

def retriever_node(state: AgentState) -> dict:
    retriever = get_cached_retriever()
    docs = retriever.invoke(state["question"])
    sources = format_sources(docs)
    unique_sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    trace_msg = f"[Retriever] Fetched {len(docs)} chunks from {unique_sources}."
    return {
        "retrieved_docs": docs,
        "sources": sources,
        "agent_trace": [trace_msg],
    }
