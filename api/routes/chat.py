import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# --- Schemas ---
from api.models.schemas import ChatRequest, ChatResponse, SourceDoc

# --- Memory Management ---
from api.session.manager import get_or_create_session, get_chat_history

# --- Phase 2: Simple Chain ---
from rag.chain.rag_chain import run_rag_chain, get_rag_chain

# --- Phase 3: Multi-Agent Graph ---
from agents.graph import get_agent_graph, build_initial_state

logger = logging.getLogger(__name__)

# Phase 1: Router Initialization
router = APIRouter()

# Phase 2: Main Chat Endpoint (Synchronous execution with branching)
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Primary POST endpoint for processing chat queries.
    Dynamically routes between simple RAG and deep-reasoning multi-agent workflows.
    """
    # Step 1: Memory Extraction
    memory = get_or_create_session(request.session_id)
    chat_history = get_chat_history(request.session_id)

    try:
        # Step 2: Execution Branching
        if not request.use_agent:
            # --- Simple RAG Execution ---
            results = await run_rag_chain(request.query, chat_history)
            final_answer = results.get("answer", "")
            raw_sources = results.get("sources", [])
            agent_trace = None
        else:
            # --- Multi-Agent Graph Execution ---
            initial_state = build_initial_state(request.query, chat_history)
            agent_graph = get_agent_graph()
            graph_result = await agent_graph.ainvoke(initial_state)
            
            final_answer = graph_result.get("final_answer", "Graph execution failed to generate an answer.")
            raw_sources = graph_result.get("retrieved_docs", [])
            agent_trace = graph_result.get("agent_trace", [])

        # Step 3: Context Preservation (Memory Committal)
        memory.save_context({"input": request.query}, {"answer": final_answer})

        # Step 4: Source Mapping & Deduplication
        formatted_sources = []
        seen = set()
        for doc in raw_sources:
            # Handle both LangChain Document objects and raw dictionaries
            metadata = getattr(doc, "metadata", doc if isinstance(doc, dict) else {})
            
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", 1)
            
            doc_key = f"{source}_{page}"
            if doc_key not in seen:
                seen.add(doc_key)
                formatted_sources.append(SourceDoc(source=source, page=page))

        # Phase 5: Payload Response Generation
        return ChatResponse(
            answer=final_answer,
            sources=formatted_sources,
            agent_trace=agent_trace,
            session_id=request.session_id
        )

    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during AI generation.")


# Phase 3: Streaming Endpoint (Real-time read-only output)
@router.get("/chat/stream")
async def chat_stream_endpoint(session_id: str, query: str):
    """
    Lightweight GET endpoint utilizing Server-Sent Events (SSE) to stream 
    responses back to the client in real-time. Bypasses graph routing.
    """
    
    # Step 1: Generator Setup (Read-Only Memory Extraction)
    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain()

    async def stream_generator():
        try:
            # Step 2: Event Formatting 
            # Note: "input" is explicitly required by the LCEL prompt template
            async for chunk in rag_chain.astream({
                "input": query, 
                "chat_history": chat_history
            }):
                text_chunk = chunk if isinstance(chunk, str) else chunk.get("answer", "")
                
                if text_chunk:
                    yield f"data: {text_chunk}\n\n"
                    
        except Exception as e:
            logger.error(f"Streaming mechanism failed: {e}", exc_info=True)
            yield f"data: [Stream Interrupted: Internal Error]\n\n"

    # Step 3: Response Delivery
    return StreamingResponse(stream_generator(), media_type="text/event-stream")