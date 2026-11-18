import json

import pytest

from app.agents.nodes.generate import generate_node
from app.core.enums import RetrievalStrategy, ToolRisk
from app.providers.base import LLMResponse
from app.schemas.agent import ToolResult
from app.schemas.chat import ActionProposal
from app.schemas.common import TokenUsage


class RecordingLLM:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def generate(self, messages, **kwargs) -> LLMResponse:
        self.messages = messages
        return LLMResponse(
            content=json.dumps({"answer": "订单 ORD99999 已取消。", "citation_chunk_ids": []}),
            usage=TokenUsage(),
            model="fake",
        )

    async def ping(self) -> bool:
        return True


def _state(*, confirmed: bool) -> dict:
    state: dict = {
        "user_query": "确认执行",
        "intent": None,
        "strategy": RetrievalStrategy.AGENT,
        "retrieved_chunks": [],
        "tool_results": [
            ToolResult(
                name="cancel_order",
                success=True,
                output={"order_id": "ORD99999", "status": "cancelled"},
                risk=ToolRisk.DESTRUCTIVE,
            )
        ],
        "token_usage": TokenUsage(),
    }
    if confirmed:
        state["confirmed_action"] = ActionProposal(
            action_id="act_1",
            tool_name="cancel_order",
            arguments={"order_id": "ORD99999"},
            risk=ToolRisk.DESTRUCTIVE,
            summary="确认后将取消订单 ORD99999。",
        )
    return state


@pytest.mark.asyncio
async def test_confirmed_action_prompt_allows_stating_completion() -> None:
    llm = RecordingLLM()
    result = await generate_node(_state(confirmed=True), llm)
    system_prompt = next(
        message["content"] for message in llm.messages if message["role"] == "system"
    )
    user_prompt = next(message["content"] for message in llm.messages if message["role"] == "user")
    assert "已确认" in system_prompt
    assert "已确认执行" in user_prompt
    assert result["final_answer"] == "订单 ORD99999 已取消。"


@pytest.mark.asyncio
async def test_unconfirmed_turn_keeps_confirmation_warning() -> None:
    llm = RecordingLLM()
    await generate_node(_state(confirmed=False), llm)
    user_prompt = next(message["content"] for message in llm.messages if message["role"] == "user")
    assert "已确认执行" not in user_prompt
