from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config.settings import settings

def get_embedder():
    return GoogleGenerativeAIEmbeddings(
        model = settings.embedding_model,
        google_api_key = settings.google_api_key,
        task_type = "retrieval_document"
    )

def get_query_embedder():
    return GoogleGenerativeAIEmbeddings(
        model = settings.embedding_model,
        google_api_key = settings.google_api_key,
        task_type = "retrieval_query"
    )