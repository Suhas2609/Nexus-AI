from unittest.mock import MagicMock

from config.settings import settings
from rag.vectorstore.retriever import get_mmr_retriever


def test_get_mmr_retriever_config():
    mock_store = MagicMock()
    mock_retriever = MagicMock()
    mock_store.as_retriever.return_value = mock_retriever

    result = get_mmr_retriever(mock_store)

    mock_store.as_retriever.assert_called_once_with(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": settings.retriever_fetch_k,
            "lambda_mult": settings.retriever_lambda_mult,
        },
    )
    assert result is mock_retriever
