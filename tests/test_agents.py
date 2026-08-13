from agents.nodes.critic_agent import route_after_critic
from agents.nodes.orchestrator import orchestrator_node
from agents.state import AgentState


def make_state(**kwargs) -> AgentState:
    base_state: AgentState = {
        "question": "",
        "chat_history": [],
        "retrieved_docs": [],
        "web_results": [],
        "analysis": "",
        "critique": "",
        "revision_count": 0,
        "needs_web_search": False,
        "sources": [],
        "agent_trace": [],
        "final_answer": "",
    }
    base_state.update(kwargs)
    return base_state


def test_orchestrator_web_search_trigger_latest():
    result = orchestrator_node(make_state(question="What is the latest news?"))
    assert result["needs_web_search"] is True


def test_orchestrator_no_web_search_trigger():
    result = orchestrator_node(make_state(question="What is the capital of France?"))
    assert result["needs_web_search"] is False


def test_route_after_critic_approve():
    state = make_state(critique="VERDICT: APPROVE", revision_count=0)
    assert route_after_critic(state) == "report_writer"


def test_route_after_critic_revise_initial():
    state = make_state(critique="VERDICT: REVISE", revision_count=0)
    assert route_after_critic(state) == "analyst"


def test_route_after_critic_revise_at_limit():
    state = make_state(critique="VERDICT: REVISE", revision_count=2)
    assert route_after_critic(state) == "report_writer"


def test_route_after_critic_revise_within_limit():
    state = make_state(critique="VERDICT: REVISE", revision_count=1)
    assert route_after_critic(state) == "analyst"
