"""Dependency-injection container and runtime composition."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from app.config.settings import Settings
from app.infra.rate_limit import InMemoryRateLimiter, RedisRateLimiter
from app.infra.redis_client import connect_redis
from app.mcp.adapter import merge_mcp_tools
from app.mcp.runtime import LOCAL_CATALOG
from app.providers.base import LLMProvider
from app.providers.deepseek import DeepSeekProvider
from app.providers.embeddings import build_embedding_provider
from app.providers.fake import OfflineLLMProvider
from app.providers.reranker import CrossEncoderRerankProvider
from app.repositories.db import create_engine, create_session_factory, init_schema
from app.repositories.migrations import run_migrations
from app.retrieval.cache import MemoryRetrievalCache, RedisRetrievalCache
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.ingest import ingest_knowledge
from app.retrieval.reranker import DocumentReranker
from app.retrieval.unstructured_parser import list_knowledge_files
from app.retrieval.vector import VectorRetriever
from app.services.chat_service import ChatService
from app.services.seed import seed_demo_data
from app.tools.factory import build_tool_registry

logger = logging.getLogger(__name__)


class AppContainer:
    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        chat_service: ChatService,
        engine,
        session_factory,
        retriever: HybridRetriever,
        embedder,
        redis=None,
        vector_store=None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.chat_service = chat_service
        self.engine = engine
        self.session_factory = session_factory
        self.retriever = retriever
        self.embedder = embedder
        self.redis = redis
        self.vector_store = vector_store
        self._knowledge_lock = asyncio.Lock()

    async def reload_knowledge(self) -> dict:
        """Re-ingest knowledge/ and swap the live index without restarting."""
        async with self._knowledge_lock:
            started = time.perf_counter()
            bm25, store, chunks = await ingest_knowledge(
                self.settings.knowledge_dir,
                self.embedder,
                settings=self.settings,
                qdrant_url=self.settings.qdrant_url,
                qdrant_collection=self.settings.qdrant_collection,
                embedding_dim=getattr(self.embedder, "dim", self.settings.embedding_dim),
                reset_qdrant=True,
            )
            self.retriever.replace_index(bm25, VectorRetriever(store, self.embedder))
            await self.retriever.invalidate_cache()
            self.vector_store = store
            return {
                "documents": len(list_knowledge_files(Path(self.settings.knowledge_dir))),
                "chunks": len(chunks),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            }

    async def aclose(self) -> None:
        closer = getattr(self.llm, "aclose", None)
        if closer:
            await closer()
        if self.redis is not None:
            closer = getattr(self.redis, "aclose", None)
            if closer:
                await closer()
        await self.engine.dispose()


async def build_infrastructure(settings: Settings):
    """Create the database session factory and optional Redis connection."""
    engine = create_engine(settings)
    if settings.run_migrations:
        await asyncio.to_thread(run_migrations, settings.database_url)
    else:
        await init_schema(engine)
    session_factory = create_session_factory(engine)
    redis = await connect_redis(settings)
    return engine, session_factory, redis


async def build_retrieval(settings: Settings, redis):
    """Build the knowledge ingestion and hybrid retrieval components."""
    embedder = build_embedding_provider(settings)
    cache = (
        RedisRetrievalCache(redis, ttl_seconds=settings.retrieval_cache_ttl_seconds)
        if redis is not None
        else MemoryRetrievalCache(ttl_seconds=settings.retrieval_cache_ttl_seconds)
    )
    bm25, store, _ = await ingest_knowledge(
        settings.knowledge_dir,
        embedder,
        settings=settings,
        qdrant_url=settings.qdrant_url,
        qdrant_collection=settings.qdrant_collection,
        embedding_dim=getattr(embedder, "dim", settings.embedding_dim),
    )
    retriever = HybridRetriever(
        bm25, VectorRetriever(store, embedder), rrf_k=settings.rrf_k, cache=cache
    )
    reranker = DocumentReranker(CrossEncoderRerankProvider(settings))
    return retriever, reranker, store, embedder


async def build_container(settings: Settings, seed: bool = False) -> AppContainer:
    """Compose runtime services from independently testable infrastructure builders."""
    llm: LLMProvider = (
        DeepSeekProvider(settings) if settings.llm_configured else OfflineLLMProvider()
    )
    engine, session_factory, redis = await build_infrastructure(settings)
    rate_limiter = (
        RedisRateLimiter(redis, limit=settings.rate_limit_per_minute)
        if redis is not None
        else InMemoryRateLimiter(limit=settings.rate_limit_per_minute)
    )
    retriever, reranker, store, embedder = await build_retrieval(settings, redis)
    registry = build_tool_registry(retriever)
    if settings.mcp_enabled:
        specs = list(LOCAL_CATALOG)
        if settings.mcp_remote_url:
            specs.append(
                {
                    "name": "mcp_remote_echo",
                    "description": "Remote MCP tool via HTTP (same Function Calling schema)",
                    "transport": "http",
                    "endpoint": settings.mcp_remote_url.rstrip("/") + "/mcp/call",
                }
            )
        merge_mcp_tools(registry, specs)
    chat_service = ChatService(
        settings=settings,
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        session_factory=session_factory,
        rate_limiter=rate_limiter,
        registry=registry,
    )
    if seed:
        async with session_factory() as session:
            await seed_demo_data(session)
            await session.commit()
    logger.info(
        "runtime_ready database=%s cache=%s vector_store=%s llm=%s local_embedding=%s cross_encoder=%s",
        "sqlite" if settings.use_sqlite else "postgresql",
        "redis" if redis is not None else "memory",
        type(store).__name__,
        "deepseek" if settings.llm_configured else "offline",
        settings.local_embedding_configured,
        settings.cross_encoder_configured,
    )
    return AppContainer(
        settings=settings,
        llm=llm,
        chat_service=chat_service,
        engine=engine,
        session_factory=session_factory,
        retriever=retriever,
        embedder=embedder,
        redis=redis,
        vector_store=store,
    )
