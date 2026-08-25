from agents.state import AgentState

WEB_SEARCH_TRIGGERS = [
    "latest", "recent", "current", "today", "news",
    "2024", "2025", "2026",
    "price", "stock", "update", "release", "announced",
]


def orchestrator_node(state: AgentState) -> dict:
    question = state["question"].lower()
    needs_web_search = any(trigger in question for trigger in WEB_SEARCH_TRIGGERS)
    revision_count = state.get("revision_count", 0)
    status = "ENABLED" if needs_web_search else "DISABLED"
    trace_entry = f"[Orchestrator] Web search {status} based on query heuristics."

    return {
        "needs_web_search": needs_web_search,
        "revision_count": revision_count,
        "agent_trace": [trace_entry],
    }


def route_web_search(state: AgentState) -> str:
    if state.get("needs_web_search"):
        return "web_search"
    return "analyst"