from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # LLM APIs
    google_api_key: str
    groq_api_key: str

    # LangSmith Tracing
    langchain_tracing_v2: bool = True
    langchain_api_key: Optional[str] = None
    langchain_project: str = "nexus-ai"

    # Vector Store
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "nexus_collection"

    # Model Names
    analyst_model: str = "gemini-1.5-flash"
    critic_model: str = "llama-3.1-70b-versatile"
    embedding_model: str = "gemini-embedding-001"

    # RAG Numeric Parameters
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retriever_k: int = 4
    retriever_fetch_k: int = 20
    retriever_lambda_mult: float = 0.7
    memory_window_k: int = 5

    # API Config
    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000
    fastapi_base_url: str = "http://localhost:8000"

    # Automation & Eval
    n8n_password: str = "admin"
    ragas_threshold: float = 0.75

    # Pydantic configuration to read .env
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

# Export as a singleton instance
settings = Settings()