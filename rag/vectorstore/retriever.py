from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from config.settings import settings


def get_mmr_retriever(store: Chroma) -> VectorStoreRetriever:
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": settings.retriever_fetch_k,
            "lambda_mult": settings.retriever_lambda_mult,
        },
    )
