"""Document retrieval node."""

from __future__ import annotations

import time

from app.agents.state import AgentState
from app.config.settings import Settings
from app.core.enums import RetrievalStrategy
from app.memory.context import retrieval_query
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query_rewrite import QueryRewriter
from app.retrieval.reranker import DocumentReranker
from app.retrieval.self_rag import SelfRAGLoop
from app.schemas.common import LatencyBreakdown
from app.schemas.retrieval import RetrievedChunk


async def retrieval_node(
    state: AgentState,
    *,
    retriever: HybridRetriever,
    reranker: DocumentReranker,
    rewriter: QueryRewriter,
    self_rag: SelfRAGLoop,
    settings: Settings,
) -> dict:
    started = time.perf_counter()
    query = retrieval_query(state["user_query"], state.get("messages") or [])
    strategy = state.get("strategy")
    top_k = (
        settings.retrieval_top_k_complex
        if strategy == RetrievalStrategy.MULTI
        else settings.retrieval_top_k_simple
    )
    rerank_ms = 0.0
    if strategy == RetrievalStrategy.MULTI:
        chunks, rerank_ms = await self_rag.retrieve_with_timing(query, top_k=top_k)
        rewritten = [query]
    else:
        suggestions = await rewriter.rewrite(query)
        rewritten = list(dict.fromkeys([query, *suggestions]))[:2]
        by_id: dict[str, RetrievedChunk] = {}
        for search_query in rewritten:
            for candidate in await retriever.retrieve(search_query, top_k=top_k):
                by_id.setdefault(candidate.chunk_id, candidate)
        rerank_started = time.perf_counter()
        chunks = await reranker.rerank(query, list(by_id.values()), top_n=settings.rerank_top_n)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000
    latency = state.get("latency") or LatencyBreakdown()
    latency.retrieval_ms = (time.perf_counter() - started) * 1000
    latency.rerank_ms = rerank_ms
    return {"retrieved_chunks": chunks, "rewritten_queries": rewritten, "latency": latency}
