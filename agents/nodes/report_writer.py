from agents.state import AgentState

def report_writer_node(state: AgentState) -> dict:
    """
    Finalizes the research process by committing the analysis 
    to the final_answer variable.
    """
    
    # Phase 2: State Extraction
    # We take the best available version of the analysis
    final_text = state.get("analysis", "No analysis content was generated.")
    
    # Phase 3: Payload Return
    return {
        "final_answer": final_text,
        "agent_trace": ["[Report Writer] Final response approved and formatted. Graph execution complete."]
    }