"""Liveness and readiness probes."""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_container
from app.schemas.health import (
    CapabilitiesResponse,
    DependencyStatus,
    LivenessResponse,
    ReadinessResponse,
)
from app.services.container import AppContainer

router = APIRouter(tags=["health"])


@router.get("/health", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    return LivenessResponse()


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(
    container: AppContainer = Depends(get_container),
) -> ReadinessResponse:
    """Hard dependency: database. Redis / Qdrant / LLM key are reported but optional."""
    deps: list[DependencyStatus] = []
    try:
        from sqlalchemy import text

        async with container.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        deps.append(DependencyStatus(name="database", ready=True))
    except Exception as exc:
        deps.append(DependencyStatus(name="database", ready=False, detail=type(exc).__name__))

    llm_ok = await container.llm.ping()
    deps.append(
        DependencyStatus(
            name="llm",
            ready=llm_ok,
            detail=(
                f"configured:{container.settings.deepseek_model}"
                if container.settings.llm_configured
                else "offline:no_api_key"
            ),
        )
    )
    if container.redis is not None:
        try:
            await container.redis.ping()
            deps.append(DependencyStatus(name="redis", ready=True))
        except Exception as exc:
            deps.append(DependencyStatus(name="redis", ready=False, detail=type(exc).__name__))
    else:
        deps.append(DependencyStatus(name="redis", ready=True, detail="not_configured"))

    store = container.vector_store
    if store is not None:
        try:
            vector_ok = bool(store.ping())
            deps.append(
                DependencyStatus(
                    name="vector_store",
                    ready=vector_ok,
                    detail=type(store).__name__,
                )
            )
        except Exception as exc:
            deps.append(
                DependencyStatus(name="vector_store", ready=False, detail=type(exc).__name__)
            )

    ready = all(item.ready for item in deps if item.name == "database")
    return ReadinessResponse(ready=ready, dependencies=deps)


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities(
    container: AppContainer = Depends(get_container),
) -> CapabilitiesResponse:
    """Report the concrete backends selected after startup and fallbacks."""
    reranker_provider = getattr(container.chat_service, "_reranker", None)
    provider = getattr(reranker_provider, "_provider", None)
    reranker = getattr(provider, "backend", "unknown")
    settings = container.settings
    quota = getattr(container.llm, "quota", None) or {}
    llm = "deepseek" if settings.llm_configured else "offline"
    llm_model = settings.deepseek_model if settings.llm_configured else "offline"
    vector = getattr(container.retriever, "_vector", None)
    embedder = getattr(vector, "embedder", None)
    embedding_model = getattr(embedder, "backend", "unknown")
    reranker_model = settings.cross_encoder_model if reranker == "cross-encoder" else "lexical"
    cache = "redis" if container.redis is not None else "memory"
    vector_store_type = type(container.vector_store).__name__ if container.vector_store else "none"
    vector_store = "memory" if vector_store_type == "InMemoryVectorStore" else vector_store_type
    retriever = (
        "hybrid" if settings.embedding_configured or settings.local_embedding_configured else "bm25"
    )
    reasons: list[str] = []
    if llm == "offline":
        reasons.append("llm: offline (no api key)")
    if embedding_model == "hash":
        reasons.append("embedding: hash fallback (no semantic recall)")
    if reranker == "lexical":
        error = getattr(provider, "load_error", None)
        reasons.append(
            f"reranker: lexical fallback ({error[:120]})" if error else "reranker: lexical fallback"
        )
    if retriever == "bm25":
        reasons.append("retriever: bm25 only")
    if vector_store == "memory" and settings.qdrant_configured:
        reasons.append("vector_store: in-memory (qdrant unavailable)")
    if cache == "memory" and settings.redis_configured:
        reasons.append("cache: in-memory (redis unavailable)")
    return CapabilitiesResponse(
        database="sqlite" if settings.use_sqlite else "postgresql",
        cache=cache,
        vector_store=vector_store,
        retriever=retriever,
        reranker=reranker,
        llm=llm,
        llm_model=llm_model,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        llm_quota_used_percent=quota.get("used_percent"),
        llm_quota_reset_at=quota.get("reset_at"),
        degraded=bool(reasons),
        degraded_reasons=reasons,
    )
