import pytest

from app.agents.nodes.generate import generate_node
from app.agents.nodes.intent import IntentClassifier
from app.agents.nodes.planner import planner_node
from app.core.enums import IntentLabel, RetrievalStrategy
from app.providers.base import LLMResponse
from app.retrieval.adaptive import AdaptiveRouter
from app.schemas.agent import AgentPlan, IntentResult
from app.schemas.chat import ChatMessage
from app.schemas.common import TokenUsage
from app.tools.registry import ToolRegistry


class _RecordingLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.messages: list[dict[str, str]] = []

    async def generate(self, messages, **kwargs) -> LLMResponse:
        self.messages = messages
        return LLMResponse(content=self._content, usage=TokenUsage(), model="fake")

    async def ping(self) -> bool:
        return True


def _intent() -> IntentResult:
    return IntentResult(
        label=IntentLabel.REFUND,
        confidence=0.9,
        rationale="test",
        prompt_version="test",
    )


def _history() -> list[ChatMessage]:
    return [
        ChatMessage(role="user", content="我想退定制马克杯"),
        ChatMessage(role="assistant", content="定制商品有质量问题可以退。"),
    ]


@pytest.mark.asyncio
async def test_generation_prompt_contains_history_and_summary() -> None:
    llm = _RecordingLLM('{"answer":"ok","citation_chunk_ids":[],"refused":false}')
    await generate_node(
        {
            "user_query": "那质量问题呢",
            "strategy": RetrievalStrategy.SINGLE,
            "messages": _history(),
            "summary": "用户咨询定制杯退货",
        },
        llm,
    )
    user = next(message["content"] for message in llm.messages if message["role"] == "user")
    assert "[历史摘要]" in user
    assert "我想退定制马克杯" in user


@pytest.mark.asyncio
async def test_intent_prompt_contains_context_when_provided() -> None:
    llm = _RecordingLLM(
        '{"label":"refund","confidence":0.9,"rationale":"x","prompt_version":"intent_v3"}'
    )
    classifier = IntentClassifier(llm, apply_overlay=False)
    await classifier.classify("那质量问题呢", context="用户：我想退定制马克杯")
    user = next(message["content"] for message in llm.messages if message["role"] == "user")
    assert "对话上下文" in user
    assert "我想退定制马克杯" in user


@pytest.mark.asyncio
async def test_router_prompt_contains_context_when_provided() -> None:
    llm = _RecordingLLM(
        '{"complexity":"single_hop","strategy":"single","confidence":0.8,"rationale":"x"}'
    )
    router = AdaptiveRouter(llm)
    await router.route("那质量问题呢", _intent(), context="用户：我想退定制马克杯")
    user = next(message["content"] for message in llm.messages if message["role"] == "user")
    assert "对话上下文" in user
    assert "我想退定制马克杯" in user


class _StubPlanner:
    def __init__(self) -> None:
        self._registry = ToolRegistry()
        self.summary = ""

    async def create_plan(self, query: str, intent_label: str, summary: str) -> AgentPlan:
        self.summary = summary
        return AgentPlan(steps=[])


@pytest.mark.asyncio
async def test_planner_receives_history_and_summary() -> None:
    planner = _StubPlanner()
    await planner_node(
        {
            "user_query": "那质量问题呢",
            "intent": _intent(),
            "messages": _history(),
            "summary": "用户咨询定制杯退货",
        },
        planner,
    )
    assert "[历史摘要]" in planner.summary
    assert "我想退定制马克杯" in planner.summary
