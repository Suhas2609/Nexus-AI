from agents.state import AgentState
from agents.utils import extract_text_from_content
from rag.chain.rag_chain import format_sources

def report_writer_node(state: AgentState) -> dict:
    """
    Finalizes the research process by committing the analysis 
    to the final_answer variable.
    """
    
    # Phase 2: State Extraction
    # Use the latest analyst response
    raw_analysis = state.get("analysis", "No analysis content was generated.")
    final_text = extract_text_from_content(raw_analysis)
    
    all_docs = state.get("retrieved_docs", []) + state.get("web_results", [])
    final_sources = format_sources(all_docs)
    
    # Phase 3: Payload Return
    return {
        "final_answer": final_text,
        "sources": final_sources,
        "agent_trace": ["[Report Writer] Final response approved and formatted. Sources finalized. Graph execution complete."]
    }