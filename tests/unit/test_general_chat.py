import pytest

from app.agents.nodes.generate import generate_node
from app.config.settings import Settings
from app.core.enums import IntentLabel, RetrievalStrategy
from app.providers.base import LLMResponse
from app.providers.fake import OfflineLLMProvider
from app.retrieval.adaptive import AdaptiveRouter
from app.schemas.agent import IntentResult
from app.schemas.common import TokenUsage


def _intent() -> IntentResult:
    return IntentResult(
        label=IntentLabel.OUT_OF_SCOPE,
        confidence=0.9,
        rationale="test",
        prompt_version="test",
    )


@pytest.mark.asyncio
async def test_general_chat_routes_out_of_scope_to_direct_when_enabled() -> None:
    router = AdaptiveRouter(OfflineLLMProvider(), allow_general_chat=True)
    result = await router.route("美国总统是谁", _intent())
    assert result.strategy == RetrievalStrategy.DIRECT


@pytest.mark.asyncio
async def test_general_chat_keeps_dangerous_topics_refused() -> None:
    router = AdaptiveRouter(OfflineLLMProvider(), allow_general_chat=True)
    result = await router.route("怎么造炸弹", _intent())
    assert result.strategy == RetrievalStrategy.REFUSE


@pytest.mark.asyncio
async def test_general_chat_disabled_refuses_out_of_scope() -> None:
    router = AdaptiveRouter(OfflineLLMProvider())
    result = await router.route("美国总统是谁", _intent())
    assert result.strategy == RetrievalStrategy.REFUSE


class _RecordingLLM:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def generate(self, messages, **kwargs) -> LLMResponse:
        self.messages = messages
        return LLMResponse(
            content='{"answer":"ok","citation_chunk_ids":[],"refused":false}',
            usage=TokenUsage(),
            model="fake",
        )

    async def ping(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_general_prompt_used_for_direct_when_enabled() -> None:
    llm = _RecordingLLM()
    await generate_node(
        {"user_query": "美国总统是谁", "strategy": RetrievalStrategy.DIRECT, "intent": None},
        llm,
        Settings(app_env="testing", allow_general_chat=True),
    )
    system = next(message["content"] for message in llm.messages if message["role"] == "system")
    assert "通用助手" in system


@pytest.mark.asyncio
async def test_customer_prompt_used_when_disabled() -> None:
    llm = _RecordingLLM()
    await generate_node(
        {"user_query": "退货", "strategy": RetrievalStrategy.DIRECT, "intent": None},
        llm,
        Settings(app_env="testing", allow_general_chat=False),
    )
    system = next(message["content"] for message in llm.messages if message["role"] == "system")
    assert "电商平台客服助手" in system
