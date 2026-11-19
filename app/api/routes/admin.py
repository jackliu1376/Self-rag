"""Admin routes for tool/skill introspection and knowledge maintenance."""

import asyncio
import hmac
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.dependencies import get_container
from app.config.settings import Settings
from app.core.exceptions import InvalidRequestError, UnauthorizedError
from app.services.container import AppContainer
from app.skills.loader import load_skills

router = APIRouter(prefix="/admin", tags=["admin"])


async def require_admin(request: Request, container: AppContainer = Depends(get_container)) -> None:
    token = container.settings.admin_api_token.get_secret_value()
    if not token:
        if container.settings.app_env == "production":
            raise UnauthorizedError("admin API is disabled")
        return
    provided = request.headers.get("x-admin-token", "")
    if not hmac.compare_digest(provided, token):
        raise UnauthorizedError("invalid admin token")


@router.get("/tools", dependencies=[Depends(require_admin)])
async def list_tools(container: AppContainer = Depends(get_container)) -> dict:
    names = container.chat_service.registry.names()
    return {"tools": names, "count": len(names)}


@router.get("/skills", dependencies=[Depends(require_admin)])
async def list_skills() -> dict:
    return {"skills": load_skills()}


class KnowledgeDocument(BaseModel):
    content: str = Field(min_length=1, max_length=200_000)


def resolve_document_path(settings: Settings, doc_path: str) -> Path:
    """Flat knowledge dir: one visible Markdown file, no traversal."""
    root = Path(settings.knowledge_dir).resolve()
    if not doc_path or doc_path.startswith(("/", "\\")):
        raise InvalidRequestError("invalid document path")
    candidate = (root / doc_path).resolve()
    if candidate.parent != root:
        raise InvalidRequestError("document must live directly under the knowledge directory")
    if candidate.suffix.lower() != ".md" or candidate.name.startswith("."):
        raise InvalidRequestError("only visible .md documents are supported")
    return candidate


@router.post("/knowledge/reload", dependencies=[Depends(require_admin)])
async def reload_knowledge(container: AppContainer = Depends(get_container)) -> dict:
    """Re-ingest knowledge/ and swap the live index without restarting."""
    return await container.reload_knowledge()


@router.put("/knowledge/documents/{doc_path:path}", dependencies=[Depends(require_admin)])
async def upsert_document(
    doc_path: str,
    payload: KnowledgeDocument,
    container: AppContainer = Depends(get_container),
) -> dict:
    path = resolve_document_path(container.settings, doc_path)
    await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    await asyncio.to_thread(tmp.write_text, payload.content, encoding="utf-8")
    await asyncio.to_thread(tmp.replace, path)
    return {"path": doc_path, **await container.reload_knowledge()}


@router.delete("/knowledge/documents/{doc_path:path}", dependencies=[Depends(require_admin)])
async def delete_document(
    doc_path: str,
    container: AppContainer = Depends(get_container),
) -> dict:
    path = resolve_document_path(container.settings, doc_path)
    if not path.is_file():
        raise InvalidRequestError("document not found")
    await asyncio.to_thread(path.unlink)
    return {"path": doc_path, **await container.reload_knowledge()}
