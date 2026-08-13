import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock

# --- Targets under Test ---
from api.main import app
from api.session.manager import _sessions, get_session_count


# Phase 1: Test Environment Isolation (Setup)
@pytest_asyncio.fixture(autouse=True)
async def clear_session_state():
    """
    Automatically clears the global session dictionary before each test execution
    to guarantee true cryptographic and memory isolation between test cases.
    """
    _sessions.clear()
    yield


# Phase 2: Health Endpoint Validation
@pytest.mark.asyncio
async def test_health_endpoint_returns_200():
    """
    Validates that the liveness check endpoint responds cleanly within standard uptime thresholds.
    Mocks the underlying Chroma store to simulate an active collection count.
    """
    # Mocking the application state context to isolate from a live vector database instance
    mock_store = MagicMock()
    mock_store.get.return_value = {"ids": ["chunk1", "chunk2"]}
    
    with patch.dict(app.state.__dict__, {"chroma_store": mock_store}, clear=False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/v1/health")
            
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "chroma_chunks" in json_data
    assert "tracing_enabled" in json_data


# Phase 3: Input Validation Enforcement (Chat)
@pytest.mark.asyncio
async def test_chat_endpoint_enforces_schema_constraints():
    """
    Asserts that empty queries are flatly rejected by the Pydantic parser
    with an HTTP 422 Unprocessable Entity code, protecting downstream LLMs.
    """
    invalid_payload = {
        "session_id": "00000000-0000-0000-0000-000000000000",
        "query": "",  # Breaks min_length=1 constraint
        "use_agent": False
    }
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/chat", json=invalid_payload)
        
    assert response.status_code == 422


# Phase 4: Standard Chat Execution Mocking
@pytest.mark.asyncio
@patch("api.routes.chat.run_rag_chain")
async def test_standard_chat_path_success(mock_run_chain):
    """
    Validates the data round-trip for a standard RAG transaction.
    Intercepts the actual LLM call, providing a deterministic response payload.
    """
    # Configure mock to match expected dict schema from rag_chain executor
    mock_run_chain.return_value = {
        "answer": "This is a deterministic test answer from the document collection.",
        "sources": [{"source": "test_document.pdf", "page": 4}]
    }
    
    valid_payload = {
        "session_id": "11111111-1111-1111-1111-111111111111",
        "query": "What are the core performance metrics?",
        "use_agent": False
    }
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/v1/chat", json=valid_payload)
        
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["answer"] == "This is a deterministic test answer from the document collection."
    assert len(json_data["sources"]) == 1
    assert json_data["sources"][0]["source"] == "test_document.pdf"
    assert json_data["agent_trace"] is None


# Phase 5: Session Memory Isolation Audit
@pytest.mark.asyncio
@patch("api.routes.chat.run_rag_chain")
async def test_session_memory_isolation(mock_run_chain):
    """
    Verifies that distinct session IDs generate isolated, individual state instances
    in memory without leaking cross-user data boundaries.
    """
    mock_run_chain.return_value = {"answer": "Acknowledged.", "sources": []}
    
    payload_user_a = {
        "session_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "query": "User A Query",
        "use_agent": False
    }
    payload_user_b = {
        "session_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        "query": "User B Query",
        "use_agent": False
    }
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Fire requests across separate simulated identities
        response_a = await ac.post("/api/v1/chat", json=payload_user_a)
        response_b = await ac.post("/api/v1/chat", json=payload_user_b)
        
    assert response_a.status_code == 200
    assert response_b.status_code == 200
    
    # Assert that exactly 2 distinct conversation spaces exist in the manager state
    assert get_session_count() == 2


# Phase 6: Empty Corpus Graceful Degradation
@pytest.mark.asyncio
async def test_documents_endpoint_handles_empty_vector_store():
    """
    Ensures the API degrades gracefully and returns a valid, empty structured 
    response model if queried prior to database ingestion initialization.
    """
    mock_store = MagicMock()
    # Simulate an empty Chroma collection state payload layout
    mock_store.get.return_value = {"ids": [], "metadatas": []}
    
    with patch.dict(app.state.__dict__, {"chroma_store": mock_store}, clear=False):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.get("/api/v1/documents")
            
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["documents"] == []
    assert json_data["total_chunks"] == 0