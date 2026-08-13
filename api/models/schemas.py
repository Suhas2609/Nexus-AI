from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# --- Chat Schemas ---

class SourceDoc(BaseModel):
    source: str
    page: int

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="UUID from client for memory isolation")
    query: str = Field(..., min_length=1, max_length=2000, description="The user's input prompt")
    use_agent: bool = Field(default=False, description="Toggle between simple RAG and multi-agent graph")

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDoc]
    agent_trace: Optional[List[str]] = None
    session_id: str

# --- Document & Ingestion Schemas ---

class UploadResponse(BaseModel):
    source: str
    chunks_added: int
    chunks_skipped: int
    total_in_store: int
    message: str

class DocumentInfo(BaseModel):
    source: str
    page_count: int

class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
    total_chunks: int

# --- Evaluation & System Schemas ---

class EvaluationResponse(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    timestamp: str
    questions_evaluated: int

class MetricsResponse(BaseModel):
    # Using Dict[str, Any] is the standard Pydantic way to type an open-ended dictionary
    history: List[Dict[str, Any]]
    latest: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    chroma_chunks: int
    tracing_enabled: bool