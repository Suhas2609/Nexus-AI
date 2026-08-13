from agents.state import AgentState
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

def web_search_node(state: AgentState) -> dict:
    if not state.get("needs_web_search", False):
        return {
            "web_results": [],
            "agent_trace": ["[Web Search] Skipped: Real-time data not requested."]
        }
    
    try:
        query = state["question"]
        raw_result = search_tool.run(query)

        if not raw_result or "No good search results" in raw_result:
            raise ValueError("Empty or invalid search response")

        web_results = [raw_result]
        trace_msg = f"[Web Search] Success: Fetched live data for '{query[:30]}...'"


    except Exception as e:
        web_results = []
        trace_msg = f"[Web Search] FAILED: {str(e)}"

    return {
        "web_results": web_results,
        "agent_trace": [trace_msg]
    }

        