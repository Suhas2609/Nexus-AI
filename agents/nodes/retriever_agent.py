from agents.state import AgentState
from rag.chain.rag_chain import format_sources
from rag.vectorstore.retriever import get_mmr_retriever
from rag.vectorstore.store import get_or_create_store


def retriever_node(state: AgentState) -> dict:
    store = get_or_create_store()
    retriever = get_mmr_retriever(store)
    docs = retriever.invoke(state["question"])
    sources = format_sources(docs)
    unique_sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})
    trace_msg = f"[Retriever] Fetched {len(docs)} chunks from {unique_sources}."
    return {
        "retrieved_docs": docs,
        "sources": sources,
        "agent_trace": [trace_msg],
    }
