from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from rag.chain.memory import create_memory

# Phase 1: Module-Level State Definition
# Structure: { session_id: {"memory": LangChainMemoryObject, "last_accessed": datetime} }
_sessions: Dict[str, Dict[str, Any]] = {}

def cleanup_expired_sessions() -> None:
    """
    Lazy evaluation sweep. Deletes any session older than 30 minutes 
    to prevent memory leaks and server crashes.
    """

    now = datetime.now(timezone.utc)  
    expiration_threshold = timedelta(minutes = 30)

    for session_id in list(_sessions.keys()):
        last_accessed = _sessions[session_id]["last_accessed"]
        if now - last_accessed > expiration_threshold:
            del _sessions[session_id]

def get_or_create_session(session_id: str) -> Any:
    """
    Retrieves an existing conversation buffer or builds a new one.
    Automatically handles timestamp updates and garbage collection.
    """

    cleanup_expired_sessions()
    now = datetime.now(timezone.utc)

    if session_id not in _sessions:
        _sessions[session_id] = {
            "memory": create_memory(),
            "last_accessed": now
        }
    else:
        # User is returning: Reset the expiration clock
        _sessions[session_id]["last_accessed"] = now
    
    # Step 4: Return just the LangChain memory object
    return _sessions[session_id]["memory"]

# Phase 4: Utility Extractors
def get_session_count() -> int:
    """Returns the total number of active users for the health endpoint."""

    return len(_sessions)

def get_chat_history(session_id: str) -> List[Any]:
    """
    Extracts raw message objects (HumanMessage, AIMessage) from the 
    memory buffer to be fed directly into the LangGraph state machine.
    """
    if session_id not in _sessions:
        return []

    return _sessions[session_id]["memory"].chat_memory.messages

