from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.nodes.orchestrator import orchestrator_node, route_web_search
from agents.nodes.retriever_agent import retriever_node
from agents.nodes.web_search_agent import web_search_node
from agents.nodes.analyst_agent import analyst_node
from agents.nodes.critic_agent import critic_node, route_after_critic
from agents.nodes.report_writer import report_writer_node

# Phase 1: Graph Initialization
workflow = StateGraph(AgentState)

# Node Registration
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("critic", critic_node)
workflow.add_node("report_writer", report_writer_node)

# Phase 2: Edge Definition
workflow.set_entry_point("orchestrator")

# Parallel extraction split
workflow.add_edge("orchestrator", "retriever")
workflow.add_conditional_edges("orchestrator", route_web_search, {"web_search": "web_search", "analyst": "analyst"})

# Converge into Analyst
workflow.add_edge("retriever", "analyst")
workflow.add_edge("web_search", "analyst")

# The Critique loop
workflow.add_edge("analyst", "critic")
workflow.add_conditional_edges(
    "critic", 
    route_after_critic, 
    {"analyst": "analyst", "report_writer": "report_writer"}
)

# Finalization
workflow.add_edge("report_writer", END)

# Phase 3: Compilation and Singleton Export
_graph = None

def get_agent_graph():
    global _graph
    if _graph is None:
        _graph = workflow.compile()
    return _graph

# Phase 4: State Initializer Utility
def build_initial_state(question: str, chat_history: list) -> AgentState:
    return {
        "question": question,
        "chat_history": chat_history,
        "retrieved_docs": [],
        "web_results": [],
        "analysis": "",
        "critique": "",
        "revision_count": 0,
        "needs_web_search": False,
        "sources": [],
        "agent_trace": [],
        "final_answer": ""
    }