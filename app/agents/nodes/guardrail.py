"""Input guardrail node for injection and scope checks."""

from app.agents.state import AgentState
from app.core.enums import RetrievalStrategy
from app.core.exceptions import GuardrailError
from app.guardrails.boundary import assert_in_scope
from app.guardrails.injection import detect_prompt_injection


async def guardrail_node(state: AgentState) -> dict:
    try:
        detect_prompt_injection(state["user_query"])
        assert_in_scope(state["user_query"])
    except GuardrailError as exc:
        return {
            "refused": True,
            "strategy": RetrievalStrategy.REFUSE,
            "final_answer": "请求被安全策略拦截。",
            "error": exc.error_code.value,
        }
    return {}
