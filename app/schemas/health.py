"""Health and readiness response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    status: Literal["ok"] = "ok"


class DependencyStatus(BaseModel):
    name: str
    ready: bool
    detail: str = ""


class ReadinessResponse(BaseModel):
    ready: bool
    dependencies: list[DependencyStatus] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    """Resolved runtime backends and whether the process is degraded."""

    database: str
    cache: str
    vector_store: str
    retriever: str
    reranker: str
    llm: str
    llm_model: str
    embedding_model: str
    reranker_model: str
    llm_quota_used_percent: int | None = None
    llm_quota_reset_at: int | None = None
    degraded: bool
    degraded_reasons: list[str] = Field(default_factory=list)
