import asyncio

import pytest

from app.agents.nodes.retrieval import retrieval_node
from app.config.settings import Settings
from app.core.enums import RetrievalStrategy
from app.schemas.chat import ChatMessage
from app.schemas.common import LatencyBreakdown
from app.schemas.retrieval import RetrievedChunk


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        title="t",
        content="七天无理由退货",
        score=1.0,
    )


class _Rewriter:
    async def rewrite(self, query, missing_aspect=None):
        return []


class _Retriever:
    async def retrieve(self, query, *, top_k):
        return [_chunk()]


class _Reranker:
    async def rerank(self, query, chunks, *, top_n):
        await asyncio.sleep(0.01)
        return chunks[:top_n]


class _SelfRAG:
    async def retrieve_with_timing(self, query, *, top_k):
        return [_chunk()], 12.5


@pytest.mark.asyncio
async def test_single_hop_records_rerank_time() -> None:
    result = await retrieval_node(
        {
            "user_query": "退货",
            "strategy": RetrievalStrategy.SINGLE,
            "latency": LatencyBreakdown(),
        },
        retriever=_Retriever(),
        reranker=_Reranker(),
        rewriter=_Rewriter(),
        self_rag=_SelfRAG(),
        settings=Settings(app_env="testing"),
    )
    assert result["latency"].rerank_ms > 0


class _RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(self, query, *, top_k):
        self.queries.append(query)
        return []


@pytest.mark.asyncio
async def test_short_follow_up_retrieves_with_previous_turn() -> None:
    retriever = _RecordingRetriever()
    await retrieval_node(
        {
            "user_query": "那还能退吗？",
            "messages": [ChatMessage(role="user", content="定制马克杯裂纹")],
            "strategy": RetrievalStrategy.SINGLE,
            "latency": LatencyBreakdown(),
        },
        retriever=retriever,
        reranker=_Reranker(),
        rewriter=_Rewriter(),
        self_rag=_SelfRAG(),
        settings=Settings(app_env="testing"),
    )
    assert retriever.queries
    assert "定制马克杯裂纹" in retriever.queries[0]


@pytest.mark.asyncio
async def test_multi_hop_records_rerank_time_from_self_rag() -> None:
    result = await retrieval_node(
        {
            "user_query": "退货",
            "strategy": RetrievalStrategy.MULTI,
            "latency": LatencyBreakdown(),
        },
        retriever=_Retriever(),
        reranker=_Reranker(),
        rewriter=_Rewriter(),
        self_rag=_SelfRAG(),
        settings=Settings(app_env="testing"),
    )
    assert result["latency"].rerank_ms == 12.5
