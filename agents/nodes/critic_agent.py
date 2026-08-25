from agents.state import AgentState
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config.settings import settings

llm = ChatGroq(
    model=settings.critic_model,
    temperature=0.1,
    groq_api_key=settings.groq_api_key,
)

def critic_node(state: AgentState) -> dict:
    evidence_parts = []
    all_docs = []

    if state.get("retrieved_docs"):
        all_docs.extend(state["retrieved_docs"])
    if state.get("web_results"):
        all_docs.extend(state["web_results"])

    for doc in all_docs:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        evidence_parts.append(f"Source: {src}, Page: {page}\nContent: {doc.page_content}")

    full_evidence = "\n".join(evidence_parts) if evidence_parts else "No evidence available."

    prompt = f"""
You are a strict QA Auditor. Your job is to verify the Analyst's response against the provided Evidence.

--- ALL EVIDENCE (DOCUMENTS + WEB) ---
{full_evidence}

--- ANALYST RESPONSE TO AUDIT ---
{state["analysis"]}

--- INSTRUCTIONS ---
1. Identify any claims in the Analyst Response that are not supported by the Evidence.
2. Identify any logical gaps in the reasoning.
3. Identify any missing critical information.
4. Provide a confidence rating as X/10 and a final VERDICT: APPROVE or REVISE.

You must strictly follow this output format:
ISSUES: [List of discrepancies or none]
MISSING: [List of missing info or none]
CONFIDENCE: [Score]/10
VERDICT: [APPROVE or REVISE]
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    critique_text = response.content

    # Confidence is diagnostic/trace-only. VERDICT drives actual control flow.
    confidence = 0
    try:
        conf_line = [line for line in critique_text.split("\n") if "CONFIDENCE:" in line][0]
        confidence_str = conf_line.split(":")[1].strip().split("/")[0]
        confidence = int(confidence_str)
    except Exception:
        confidence = 0

    verdict = "APPROVE" if "VERDICT: APPROVE" in critique_text else "REVISE"

    return {
        "critique": critique_text,
        "agent_trace": [f"[Critic] Audited response. Confidence: {confidence}/10. Verdict: {verdict}."],
    }


def route_after_critic(state: AgentState) -> str:
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    if "VERDICT: REVISE" in critique and revision_count < 2:
        return "analyst"

    return "report_writer"