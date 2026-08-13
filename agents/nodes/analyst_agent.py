from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from agents.state import AgentState
from config.settings import settings

llm = ChatGoogleGenerativeAI(
    model=settings.analyst_model,
    temperature=0.2,
    google_api_key=settings.google_api_key,
)


def analyst_node(state: AgentState) -> dict:
    context_parts = []

    if state.get("retrieved_docs"):
        context_parts.append("--- DOCUMENT SOURCES ---")
        for doc in state["retrieved_docs"]:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            context_parts.append(f"Source: {src}, Page: {page}\nContent: {doc.page_content}")

    if state.get("web_results"):
        context_parts.append("\n--- WEB SOURCES ---")
        context_parts.extend(state["web_results"])

    full_context = "\n".join(context_parts) if context_parts else "No context available."

    revision_instructions = ""
    payload: dict = {}

    if state.get("critique"):
        revision_instructions = (
            f"\n\n--- REVISION GUIDANCE ---\n{state['critique']}\n"
        )
        payload["revision_count"] = state.get("revision_count", 0) + 1
        trace_note = f"Revision pass {payload['revision_count']}"
    else:
        trace_note = "Initial draft pass"

    prompt = f"""You are a professional research assistant. Synthesize the context below to answer the user question.{revision_instructions}

--- CONTEXT ---
{full_context}

User Question: {state['question']}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    trace_msg = f"[Analyst] Synthesis complete ({trace_note}). Output length: {len(response.content)} chars."
    payload["analysis"] = response.content
    payload["agent_trace"] = [trace_msg]

    return payload
