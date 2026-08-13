from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

from config.settings import settings
from rag.chain.prompts import CONTEXTUALISE_PROMPT, RAG_PROMPT
from rag.vectorstore.retriever import get_mmr_retriever
from rag.vectorstore.store import get_or_create_store

llm = ChatGoogleGenerativeAI(
    model=settings.analyst_model,
    temperature=0.1,
    google_api_key=settings.google_api_key,
    streaming=True,
)

_chain = None


def format_sources(docs: list[Document]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    sources: list[dict] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({"source": source, "page": page})
    return sources


def build_rag_chain():
    store = get_or_create_store()
    retriever = get_mmr_retriever(store)
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALISE_PROMPT
    )
    answer_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
    return create_retrieval_chain(history_aware_retriever, answer_chain)


def get_rag_chain():
    global _chain
    if _chain is None:
        _chain = build_rag_chain()
    return _chain


async def run_rag_chain(question: str, chat_history: list) -> dict:
    chain = get_rag_chain()
    result = await chain.ainvoke({"input": question, "chat_history": chat_history})
    sources = format_sources(result.get("context", []))
    return {"answer": result["answer"], "sources": sources}
