from config.settings import settings
from api.models.schemas import HealthResponse
from fastapi import APIRouter, Request
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", response_model = HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Sub-200ms liveness probe for infrastructure monitoring."""
    # Phase 2: Database Storage Evaluation
    chunk_count = 0
    try:
        # Access the Chroma vector store bound to the FastAPI app state
        if hasattr(request.app.state, "chroma_store") and request.app.state.chroma_store:
            collection = request.app.state.chroma_store.get()
            chunk_count = len(collection.get("ids", []))
    except Exception as e:
        logger.warning(f"Vector DB unreachable during health check. Graceful degradation active. Error: {e}")

    is_tracing_active = bool(settings.langchain_api_key)

    return HealthResponse(
        status = "ok",
        model = settings.analyst_model,
        chroma_chunks = chunk_count,
        tracing_enabled = is_tracing_active
    )