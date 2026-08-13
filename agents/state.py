from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # 1. Inputs
    question: str
    chat_history: List[BaseMessage]
    
    # 2. Research Data
    retrieved_docs: List[Document]
    web_results: List[str]
    
    # 3. Agent Drafts & Feedback
    analysis: str
    critique: str
    
    # 4. Routing & Control Flags
    revision_count: int
    needs_web_search: bool
    
    # 5. Final Output & Logging
    final_answer: str
    sources: List[dict]
    agent_trace: List[str]