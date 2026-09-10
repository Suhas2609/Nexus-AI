import re
from agents.state import AgentState
from agents.utils import extract_text_from_content
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from config.settings import settings

llm = ChatGroq(
    model=settings.critic_model,
    temperature=0.1,
    groq_api_key=settings.groq_api_key,
    max_tokens=800,
)

def parse_critic_response(text: str) -> tuple[int, str, str]:
    """
    Parses the raw text response from the Critic LLM.
    Strips any <think>...</think> blocks.
    Uses regex to extract confidence score and verdict.
    If missing/invalid, verdict MUST default to REVISE.
    Returns (confidence, verdict, normalized_critique_text).
    """
    clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Regex for confidence score: CONFIDENCE:\s*(\d+)\s*/\s*10
    conf_match = re.search(r"CONFIDENCE:\s*(\d+)\s*/\s*10", clean_text, re.IGNORECASE)
    confidence = int(conf_match.group(1)) if conf_match else 0

    # Regex for verdict: VERDICT:\s*(APPROVE|REVISE)
    verdict_match = re.search(r"VERDICT:\s*(APPROVE|REVISE)", clean_text, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    else:
        verdict = "REVISE"

    if not clean_text or not verdict_match:
        normalized_critique = (
            f"{clean_text}\n\n" if clean_text else ""
        ) + (
            "ISSUES: Critic response incomplete or invalid.\n"
            "MISSING: Explicit verdict could not be parsed.\n"
            "CONFIDENCE: 0/10\n"
            "VERDICT: REVISE"
        )
    else:
        normalized_critique = clean_text

    return confidence, verdict, normalized_critique


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

    prompt = f"""You are a strict QA auditor.
Compare the Analyst response against the supplied evidence and the original user question.
Be extremely concise. Do NOT evaluate claims one by one. Do NOT produce verbose reasoning.

--- USER QUESTION ---
{state['question']}

--- ALL EVIDENCE (DOCUMENTS + WEB) ---
{full_evidence}

--- ANALYST RESPONSE TO AUDIT ---
{state["analysis"]}

--- INSTRUCTIONS ---
1. Identify any claims in the Analyst Response that are not supported by the Evidence.
2. Identify any logical gaps in the reasoning.
3. Identify any EXPLICIT requirements in the original question that the Analyst response did not address. If the question asked for specific quantitative data, parameter counts, or numerical results, verify they appear in the response. List each unaddressed requirement under MISSING.
4. Provide a confidence rating as X/10 and a final VERDICT: APPROVE or REVISE.

Return ONLY in this format:
ISSUES: <brief summary of discrepancies or None>
MISSING: <brief summary of unaddressed requirements or None>
CONFIDENCE: <0-10>/10
VERDICT: <APPROVE or REVISE>

Keep the final response under 150 words.
Do not include markdown code fences.
Do not include <think> reasoning in the final answer.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw_text = extract_text_from_content(response.content)

    confidence, verdict, normalized_critique = parse_critic_response(raw_text)

    return {
        "critique": normalized_critique,
        "agent_trace": [f"[Critic] Audited response. Confidence: {confidence}/10. Verdict: {verdict}."],
    }


def route_after_critic(state: AgentState) -> str:
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    verdict_match = re.search(r"VERDICT:\s*(APPROVE|REVISE)", critique, re.IGNORECASE)
    verdict = verdict_match.group(1).upper() if verdict_match else "REVISE"

    if verdict == "REVISE" and revision_count < 2:
        return "analyst"

    return "report_writer"